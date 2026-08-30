"""
Tests for registry-driven PEFT patch dispatch (#68 PR B).

`patch_peft_model` was a four-arm ``elif model_type`` chain in which each arm
independently re-derived a layer count for its log line.  This pins the
dispatch contract as it moves behind the registry.

Note the PEFT model_types are deliberately *not* the native-load ones: the
Tiny-A2D family also answers to plain ``llama``/``qwen2``/``qwen3`` (a PEFT
model reports the base architecture), and ModernBERT is patchable but has no
native loader entry at all.
"""

import pytest


class TestPeftModelTypeCoverage:
    @pytest.mark.parametrize(
        "model_type",
        [
            "tiny-a2d-llama",
            "tiny-a2d-qwen2",
            "tiny-a2d-qwen3",
            # A PEFT-wrapped converted model reports its base architecture.
            "llama",
            "qwen2",
            "qwen3",
            "dream",
            "Dream",
            "llada",
            "modernbert-diffusion",
        ],
    )
    def test_every_previously_supported_type_still_dispatches(self, model_type):
        from unturtle.models.integrations import resolve_peft_patcher

        assert resolve_peft_patcher(model_type) is not None, (
            f"{model_type} lost PEFT support"
        )

    def test_unsupported_type_has_no_patcher(self):
        from unturtle.models.integrations import resolve_peft_patcher

        assert resolve_peft_patcher("mdlm-dit") is None
        assert resolve_peft_patcher("diffusion_gemma") is None
        assert resolve_peft_patcher("totally-unknown") is None

    def test_peft_types_are_distinct_from_native_types(self):
        """`llama` is PEFT-patchable but must not be a native load target."""
        from unturtle.models.integrations import (
            resolve_native_class,
            resolve_peft_patcher,
        )

        assert resolve_peft_patcher("llama") is not None
        assert resolve_native_class("llama") is None


class TestUnsupportedStillFailsLoudly:
    def test_raises_not_implemented_listing_supported_types(self):
        from unturtle.fast_diffusion_model import FastDiffusionModel

        class _Cfg:
            model_type = "mdlm-dit"

        class _Model:
            config = _Cfg()

        with pytest.raises(NotImplementedError) as excinfo:
            FastDiffusionModel.patch_peft_model(_Model())

        message = str(excinfo.value)
        assert "mdlm-dit" in message
        # The message must still enumerate what *is* supported.
        for expected in ("llada", "dream", "modernbert-diffusion", "tiny-a2d-llama"):
            assert expected in message, f"{expected} missing from the error message"


class TestReportedCounts:
    """Each family reported a differently-derived layer count; keep them."""

    def _peft_like(self, model_type, n_layers=3):
        import torch

        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.model_type = model_type

        layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(n_layers)])
        inner = torch.nn.Module()
        inner.layers = layers
        wrapper = torch.nn.Module()
        wrapper.model = inner
        base = torch.nn.Module()
        base.model = wrapper

        model = torch.nn.Module()
        model.base_model = base
        model.config = cfg
        return model

    @pytest.mark.parametrize(
        ("model_type", "expected_fragment", "expected_counts"),
        [
            # "3 layers" would NOT discriminate — every family renders it.
            # Each fragment below appears in exactly one report.
            ("tiny-a2d-llama", "(bidirectional, causal=False)", "1 QKV layers"),
            ("dream", "(Dream)", "4 QKV layers (bias kernel)"),
            ("modernbert-diffusion", "(ModernBERT)", "8 Wo (output proj)"),
        ],
    )
    def test_report_identifies_the_family_and_its_own_counts(
        self, model_type, expected_fragment, expected_counts, monkeypatch
    ):
        """The per-family log line survives the move behind the registry.

        Each patcher returns a *distinct* tuple: identical stubs would make
        the patcher-to-family wiring unobservable, so mis-pointing an
        integration at another family's patcher would go unnoticed.
        """
        from unturtle import fast_diffusion_model as fdm

        messages = []
        monkeypatch.setattr(fdm, "_warn_once", lambda msg: messages.append(msg))
        # Patchers touch real module internals; stub them out — this test is
        # about dispatch and reporting, not about kernel injection.
        # Tiny-A2D lives in its own provider module (#185); the façade holds no
        # `_patch_a2d_peft` any more, so the seam is the provider's attribute.
        from unturtle.models.conversion.a2d.tiny_a2d import fast_paths as a2d_fast_paths

        monkeypatch.setattr(a2d_fast_paths, "patch_peft", lambda m, d, b: (1, 2, 3))
        from unturtle.models.backbones.dream import fast_paths as dream_fast_paths

        monkeypatch.setattr(dream_fast_paths, "patch_peft", lambda m, d, b: (4, 5, 6))
        from unturtle.models.backbones.modernbert import fast_paths as mb_fast_paths

        monkeypatch.setattr(mb_fast_paths, "patch_peft", lambda m, d, b: (7, 8, 9))

        fdm.FastDiffusionModel.patch_peft_model(self._peft_like(model_type))

        assert messages, "dispatch reported nothing"
        assert expected_fragment in messages[0], messages[0]
        assert expected_counts in messages[0], (
            f"counts came from the wrong patcher: {messages[0]}"
        )

    def test_llada_counts_blocks_not_layers(self):
        """LLaDA's hierarchy is transformer.blocks, not model.layers."""
        import torch

        from unturtle import fast_diffusion_model as fdm

        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.model_type = "llada"

        transformer = torch.nn.Module()
        transformer.blocks = torch.nn.ModuleList([torch.nn.Module() for _ in range(5)])
        inner = torch.nn.Module()
        inner.transformer = transformer
        base = torch.nn.Module()
        base.model = inner

        model = torch.nn.Module()
        model.base_model = base
        model.config = cfg

        messages = []
        original_warn = fdm._warn_once
        original_patch = fdm._patch_llada_peft
        fdm._warn_once = lambda msg: messages.append(msg)
        fdm._patch_llada_peft = lambda m, d, b: (0, 0, 0)
        try:
            fdm.FastDiffusionModel.patch_peft_model(model)
        finally:
            fdm._warn_once = original_warn
            fdm._patch_llada_peft = original_patch

        assert messages
        assert "5 blocks" in messages[0], messages[0]


class TestMaxSeqLengthPropagation:
    def test_propagated_after_dispatch(self, monkeypatch):
        """The tail of patch_peft_model runs regardless of which arm matched."""
        import torch

        from unturtle import fast_diffusion_model as fdm

        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.model_type = "tiny-a2d-llama"

        inner = torch.nn.Module()
        inner.layers = torch.nn.ModuleList([torch.nn.Module()])
        wrapper = torch.nn.Module()
        wrapper.model = inner
        base = torch.nn.Module()
        base.model = wrapper
        model = torch.nn.Module()
        model.base_model = base
        model.config = cfg
        model.max_seq_length = 777

        seen = []
        monkeypatch.setattr(fdm, "_warn_once", lambda msg: None)
        from unturtle.models.conversion.a2d.tiny_a2d import fast_paths as a2d_fast_paths

        monkeypatch.setattr(a2d_fast_paths, "patch_peft", lambda m, d, b: (0, 0, 0))
        monkeypatch.setattr(
            fdm,
            "_propagate_max_seq_length",
            lambda m, length: seen.append(length),
        )

        fdm.FastDiffusionModel.patch_peft_model(model)

        assert seen == [777]
