# Copyright 2025-present nishide-dev & the Unturtle team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regressions for generation follow-ups (#48/#49).

Covers:
1. BlockDecodeMixin non-parallel cache path honors ``alg`` via
   confidence-ordered transfer (was: always origin-style random Bernoulli).
2. Dream trim-mode block decode forwards from the model's query start
   (``current_block_start - 1``) so right-shifted logits predict the block's
   first token, matching the dual-cache path.
3. ``alg='entropy'`` + threshold-based ``parallel_decode`` warns about
   degeneration (neg-entropy confidences <= 0 never reach a [0, 1] threshold).
4. BD3LM capability probes are explicit (signature-based) — genuine forward
   errors propagate instead of silently switching the numerics path.
5. ``prepare_for_sampling`` pad query rows are numerically safe (no all-False
   attention rows -> NaN softmax hazard).
"""

from types import SimpleNamespace

import pytest
import torch

from unturtle.models.generation.block_decode_mixin import BlockDecodeMixin
from unturtle.models.generation.diffusion_generation_utils import (
    MaskedDiffusionGenerationConfig,
    MaskedDiffusionGenerationMixin,
    prepare_for_sampling,
)
from unturtle.models.generation.masked_diffusion_block_mixin import (
    MaskedDiffusionBlockGenerationMixin,
    _forward_accepts_kwargs,
)


class _ScriptedBlockDecodeModel(BlockDecodeMixin):
    """Deterministic fake model for BlockDecodeMixin loop tests.

    Forwards are suffix windows of an 8-token canvas.  Logits are sharply
    peaked (high max-prob confidence) at absolute position ``sharp_pos`` and
    nearly flat elsewhere, and the favored token id changes on every forward
    call, so the final sequence records *which call* committed each position.
    """

    VOCAB = 16
    MASK_ID = 15
    MAX_LEN = 8

    def __init__(self, sharp_pos: int = 6):
        self.config = SimpleNamespace(mask_token_id=self.MASK_ID)
        self.sharp_pos = sharp_pos
        self.calls = 0

    def _model_forward_with_cache(
        self,
        input_ids,
        attention_mask,
        past_key_values,
        use_cache,
        replace_position=None,
    ):
        B, L = input_ids.shape
        favored = 2 + self.calls  # call 0 -> 2, call 1 -> 3, ...
        self.calls += 1
        logits = torch.zeros(B, L, self.VOCAB)
        for i in range(L):
            abs_pos = self.MAX_LEN - L + i  # suffix window -> absolute position
            logits[:, i, favored] = 10.0 if abs_pos == self.sharp_pos else 1.0
        cache = ((torch.zeros(B, 1, L, 1), torch.zeros(B, 1, L, 1)),)
        return SimpleNamespace(logits=logits, past_key_values=cache)


class TestBlockDecodeConfidenceOrderedTransfer:
    """#48 item 1: non-parallel cache path must honor ``alg`` by selecting the
    per-step transfer set via top confidence (Fast-dLLM non-threshold ordering,
    dev/repos/fast-dllm/v1/dream/model/generation_utils_block.py L526-559, and
    the repo's own no-cache ``_sample`` non-origin branch), keeping the
    schedule-driven per-step counts."""

    def _run(self, alg: str) -> torch.Tensor:
        model = _ScriptedBlockDecodeModel(sharp_pos=6)
        cfg = MaskedDiffusionGenerationConfig(
            max_length=8,
            steps=2,
            alg=alg,
            mask_token_id=model.MASK_ID,
            use_cache=True,
            use_replace_cache=False,
            block_length=4,
            temperature=0.0,
        )
        torch.manual_seed(0)
        input_ids = torch.full((1, 4), 9, dtype=torch.long)
        out = model._block_decode_loop(
            input_ids=input_ids, attention_mask=None, generation_config=cfg
        )
        assert out.shape == (1, 8)
        return out[0, 4:]

    def test_maskgit_plus_transfers_top_confidence_first(self):
        # Single block of 4, steps_per_block=2: step 0 transfers
        # int(4 * (1 - s/t)) = 1 token; forward calls are:
        #   call 0 = initial cache build, call 1 = step 0, call 2 = step 1.
        # Confidence ordering must commit the sharp position (abs 6) at step 0
        # (token 3) and the remaining three positions at the final step
        # (token 4).  The old code picked a random Bernoulli subset and
        # discarded the confidences entirely.
        gen = self._run("maskgit_plus")
        assert gen.tolist() == [4, 4, 3, 4], (
            f"expected the top-confidence position (abs 6) to be committed at "
            f"step 0 and the rest at the final step, got {gen.tolist()}"
        )

    def test_entropy_transfers_top_confidence_first(self):
        # Sharp logits also maximize negative entropy, so the same ordering
        # must hold for alg='entropy' (confidences <= 0 but comparable).
        gen = self._run("entropy")
        assert gen.tolist() == [4, 4, 3, 4]

    def test_origin_keeps_random_transfer(self):
        # 'origin' remains random Bernoulli: over many seeds the committed
        # pattern must NOT always equal the confidence-ordered one.
        patterns = set()
        for seed in range(8):
            model = _ScriptedBlockDecodeModel(sharp_pos=6)
            cfg = MaskedDiffusionGenerationConfig(
                max_length=8,
                steps=2,
                alg="origin",
                mask_token_id=model.MASK_ID,
                use_cache=True,
                use_replace_cache=False,
                block_length=4,
                temperature=0.0,
            )
            torch.manual_seed(seed)
            out = model._block_decode_loop(
                input_ids=torch.full((1, 4), 9, dtype=torch.long),
                attention_mask=None,
                generation_config=cfg,
            )
            patterns.add(tuple(out[0, 4:].tolist()))
        assert len(patterns) > 1, "origin transfer should stay stochastic"


class TestDreamTrimModeQueryStart:
    """#48 item 2: trim mode must forward from the model's query start.

    Dream's logits are right-shifted, so predicting the block's first token
    needs position ``current_block_start - 1`` in the query window (the same
    hook the dual-cache path already uses).  Fast-dLLM's non-dual path
    compensates differently — it commits the block's first token from the
    initial full forward (dev/repos/fast-dllm/v1/dream/model/
    generation_utils_block.py L451-456) — while unturtle aligns trim mode with
    its dual mode via ``_get_block_decode_query_start``.
    """

    @pytest.fixture
    def config(self):
        from unturtle.models.backbones.dream import DreamConfig

        return DreamConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            pad_token_id=0,
            mask_token_id=1,
            use_cache=False,
        )

    def test_trim_mode_includes_previous_token_in_query_window(self, config):
        from unturtle.models.backbones.dream import DreamGenerationConfig, DreamModel

        class SpyDreamModel(DreamModel):
            def __init__(self, cfg):
                super().__init__(cfg)
                self.denoise_calls = []

            def _model_forward_with_cache(
                self,
                input_ids,
                attention_mask,
                past_key_values,
                use_cache,
                replace_position=None,
            ):
                if past_key_values is not None:
                    cache_len = past_key_values[0][0].shape[-2]
                    self.denoise_calls.append((input_ids.shape[1], cache_len))
                return super()._model_forward_with_cache(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    replace_position=replace_position,
                )

        torch.manual_seed(0)
        model = SpyDreamModel(config).cpu().eval()
        prompt_len, max_new, block_length = 4, 4, 2
        max_length = prompt_len + max_new
        inputs = torch.tensor([[2, 3, 4, 5]])
        generation_config = DreamGenerationConfig(
            max_new_tokens=max_new,
            steps=4,
            block_length=block_length,
            use_cache=True,
            use_replace_cache=False,
            mask_token_id=config.mask_token_id,
            pad_token_id=config.pad_token_id,
        )
        with torch.no_grad():
            out = model.generate(inputs=inputs, generation_config=generation_config)

        assert out.shape == (1, max_length)
        assert not torch.any(out == config.mask_token_id)
        assert torch.equal(out[:, :prompt_len], inputs)

        assert model.denoise_calls, "expected cached denoise forwards"
        # Each denoise forward must start one position BEFORE the current
        # block (query_start = block_start - 1) so the right-shifted logits
        # cover the block's first token; the trimmed cache ends exactly at
        # query_start.  Old code forwarded from block_start with cache length
        # block_start (window one token shorter, cache one token longer).
        valid_windows = set()
        for block_idx in range(max_new // block_length):
            block_start = prompt_len + block_idx * block_length
            query_start = block_start - 1
            valid_windows.add((max_length - query_start, query_start))
        assert set(model.denoise_calls) <= valid_windows, (
            f"denoise forwards {set(model.denoise_calls)} must forward from "
            f"query_start = block_start - 1 with cache trimmed to query_start "
            f"(expected within {valid_windows})"
        )

    def test_llada_trim_mode_query_window_unchanged(self):
        """LLaDA (query_start == block_start) must keep the old windows."""
        from unturtle.models.backbones.llada import LLaDAConfig, LLaDAModelLM

        config = LLaDAConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            vocab_size=512,
            mlp_ratio=4,
            max_sequence_length=64,
            attention_dropout=0.0,
            residual_dropout=0.0,
            embedding_dropout=0.0,
            rope=True,
            block_type="llama",
            activation_type="silu",
            init_device="cpu",
            mask_token_id=511,
        )
        torch.manual_seed(42)
        model = LLaDAModelLM(config).eval()

        denoise_calls = []
        original = model._model_forward_with_cache

        def spy(*args, **kwargs):
            pkv = kwargs.get("past_key_values")
            ids = kwargs.get("input_ids")
            if pkv is not None:
                denoise_calls.append((ids.shape[1], pkv[0][0].shape[-2]))
            return original(*args, **kwargs)

        model._model_forward_with_cache = spy

        prompt_len, max_new, block_length = 4, 4, 2
        max_length = prompt_len + max_new
        gen_config = MaskedDiffusionGenerationConfig(
            max_new_tokens=max_new,
            steps=4,
            alg="origin",
            mask_token_id=config.mask_token_id,
            use_cache=True,
            use_replace_cache=False,
            block_length=block_length,
        )
        with torch.no_grad():
            out = model.generate(
                inputs=torch.randint(0, 500, (1, prompt_len)),
                generation_config=gen_config,
            )
        assert out.shape == (1, max_length)
        valid_windows = set()
        for block_idx in range(max_new // block_length):
            block_start = prompt_len + block_idx * block_length
            valid_windows.add((max_length - block_start, block_start))
        assert denoise_calls and set(denoise_calls) <= valid_windows


class _TinyA2DFactory:
    MASK_ID = 100
    PAD_ID = 0

    @classmethod
    def build(cls):
        from unturtle.models.conversion.a2d.tiny_a2d import (
            TinyA2DLlamaConfig,
            TinyA2DLlamaLMHeadModel,
        )

        config = TinyA2DLlamaConfig(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=128,
            max_position_embeddings=256,
            mask_token_id=cls.MASK_ID,
            pad_token_id=cls.PAD_ID,
        )
        model = TinyA2DLlamaLMHeadModel(config)
        model.eval()
        return model


class TestEntropyThresholdWarning:
    """#48 item 3: alg='entropy' + threshold-based parallel decode must warn.

    Neg-entropy confidences are <= 0 and never reach a threshold in [0, 1], so
    every step falls back to the single max-confidence token.  Fast-dLLM's
    threshold mode only ever uses max-probability confidence
    (dev/repos/fast-dllm/v1/dream/model/generation_utils_block.py L495-524).
    """

    def test_block_decode_loop_warns(self):
        model = _ScriptedBlockDecodeModel()
        cfg = MaskedDiffusionGenerationConfig(
            max_length=8,
            steps=2,
            alg="entropy",
            mask_token_id=model.MASK_ID,
            use_cache=True,
            use_replace_cache=False,
            parallel_decode=True,
            confidence_threshold=0.9,
            block_length=4,
            temperature=0.0,
        )
        with pytest.warns(UserWarning, match="negative-entropy"):
            model._block_decode_loop(
                input_ids=torch.full((1, 4), 9, dtype=torch.long),
                attention_mask=None,
                generation_config=cfg,
            )

    def test_shared_sample_with_cache_warns(self):
        model = _TinyA2DFactory.build()
        cfg = MaskedDiffusionGenerationConfig(
            max_new_tokens=4,
            max_length=8,
            steps=2,
            alg="entropy",
            mask_token_id=_TinyA2DFactory.MASK_ID,
            use_cache=True,
            parallel_decode=True,
            confidence_threshold=0.9,
            block_length=4,
            temperature=0.0,
        )
        input_ids = torch.randint(1, 90, (1, 4))
        with pytest.warns(UserWarning, match="negative-entropy"), torch.no_grad():
            MaskedDiffusionGenerationMixin._sample_with_cache(
                model, input_ids, None, cfg
            )

    def test_max_prob_threshold_does_not_warn(self):
        import warnings as _warnings

        model = _ScriptedBlockDecodeModel()
        cfg = MaskedDiffusionGenerationConfig(
            max_length=8,
            steps=2,
            alg="maskgit_plus",
            mask_token_id=model.MASK_ID,
            use_cache=True,
            use_replace_cache=False,
            parallel_decode=True,
            confidence_threshold=0.5,
            block_length=4,
            temperature=0.0,
        )
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", UserWarning)
            model._block_decode_loop(
                input_ids=torch.full((1, 4), 9, dtype=torch.long),
                attention_mask=None,
                generation_config=cfg,
            )


class TestBD3LMCapabilityProbe:
    """#48 item 4: capability probes must be explicit and one-time; genuine
    runtime errors in the forward must propagate instead of silently switching
    the BD3LM numerics path to the full-sequence fallback."""

    def test_forward_accepts_kwargs_signature_probe(self):
        class NoKwargs(torch.nn.Module):
            def forward(self, input_ids, attention_mask=None):
                return input_ids

        class WithPositionIds(torch.nn.Module):
            def forward(self, input_ids, attention_mask=None, position_ids=None):
                return input_ids

        class VarKwargs(torch.nn.Module):
            def forward(self, input_ids, **kwargs):
                return input_ids

        no_kwargs = NoKwargs()
        assert _forward_accepts_kwargs(no_kwargs, ("position_ids",)) is False
        assert _forward_accepts_kwargs(WithPositionIds(), ("position_ids",)) is True
        assert _forward_accepts_kwargs(VarKwargs(), ("position_ids", "use_cache"))
        # Result is cached per instance.
        assert ("position_ids",) in no_kwargs._bd3lm_forward_kwarg_support

    def test_runtime_error_in_kv_path_propagates(self, monkeypatch):
        model = _TinyA2DFactory.build()
        original_forward = model.forward

        def exploding_forward(*args, **kwargs):
            if kwargs.get("past_key_values") is not None:
                raise RuntimeError("boom: genuine failure in cached forward")
            return original_forward(*args, **kwargs)

        # Instance-level patch: the signature probe inspects
        # type(model).forward, so the KV path stays enabled.
        monkeypatch.setattr(model, "forward", exploding_forward)

        prompt = torch.randint(1, 90, (1, 4))
        # Old code swallowed this RuntimeError via `except (TypeError,
        # RuntimeError)` and silently fell back to the full-seq path.
        with pytest.raises(RuntimeError, match="boom"), torch.no_grad():
            model.generate(
                prompt,
                algorithm="bd3lm",
                bd3lm_block_size=4,
                max_new_tokens=4,
                steps=2,
                mask_token_id=_TinyA2DFactory.MASK_ID,
                pad_token_id=_TinyA2DFactory.PAD_ID,
            )

    def test_model_without_cache_kwargs_uses_full_seq_path(self):
        """Models whose forward lacks the KV-cache kwargs must keep working
        (previously via caught TypeError, now via the signature probe)."""

        class NoCacheModel(MaskedDiffusionBlockGenerationMixin, torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(
                    mask_token_id=1, pad_token_id=0, eos_token_id=None
                )
                torch.manual_seed(0)
                self.emb = torch.nn.Embedding(32, 8)
                self.head = torch.nn.Linear(8, 32)

            def forward(self, input_ids, attention_mask=None, position_ids=None):
                return SimpleNamespace(logits=self.head(self.emb(input_ids)))

        model = NoCacheModel().eval()
        cfg = MaskedDiffusionGenerationConfig(
            use_block_diffusion=True,
            bd3lm_block_size=4,
            max_new_tokens=4,
            steps=2,
            temperature=0.0,
            mask_token_id=1,
            pad_token_id=0,
        )
        with torch.no_grad():
            out = model._sample_block_diffusion(torch.tensor([[5, 6, 7]]), cfg)
        assert out.shape == (1, 7)  # 3 prompt + 4 generated (left-pad stripped)
        assert not torch.any(out[:, 3:] == 1), "all masks must be committed"


class TestBD3LMPadRowSafety:
    """#48 item 5: the BD3LM loop must never feed SDPA an attention mask with
    all-False query rows (pad positions -> NaN softmax hazard with eager/math
    SDPA).  ``prepare_for_sampling`` itself keeps pad rows all-False (its unit
    contract); the loop applies ``_pad_safe_attention_mask`` — run_attention's
    ``no_allowed`` pattern (unturtle/utils/attention_dispatch.py) — at every
    call site."""

    def test_pad_safe_attention_mask_flips_all_false_rows(self):
        from unturtle.models.generation.masked_diffusion_block_mixin import (
            _pad_safe_attention_mask,
        )

        pad_id = 0
        x = torch.tensor([[pad_id, pad_id, 5, 6, 7, 8, 9, 10]])
        attn_mask, _ = prepare_for_sampling(x, block_size=4, pad_token_id=pad_id)
        # Precondition (prepare_for_sampling contract): pad rows all-False.
        assert not attn_mask[0, 0, 0].any() and not attn_mask[0, 0, 1].any()

        safe = _pad_safe_attention_mask(attn_mask)
        # Pad query rows now attend everywhere (harmless; outputs never read).
        assert safe[0, 0, 0].all() and safe[0, 0, 1].all()
        # Every query row has at least one allowed key (no NaN softmax rows).
        assert safe.any(dim=-1).all()
        # Valid rows are bit-unchanged.
        assert torch.equal(safe[0, 0, 2:], attn_mask[0, 0, 2:])

    def test_bd3lm_non_multiple_prompt_produces_finite_logits(self, monkeypatch):
        """Non-multiple-of-block prompt forces an internal left-pad; every
        4-D mask reaching the model must be pad-row-safe and the whole bd3lm
        path must see finite logits (no NaN leak from pad rows)."""
        model = _TinyA2DFactory.build()
        original_forward = model.forward

        def asserting_forward(*args, **kwargs):
            am = kwargs.get("attention_mask")
            if am is not None and am.ndim == 4 and am.dtype == torch.bool:
                assert am.any(dim=-1).all(), (
                    "all-False attention rows reached the model (NaN hazard)"
                )
            out = original_forward(*args, **kwargs)
            assert torch.isfinite(out.logits).all(), "NaN/Inf logits in bd3lm path"
            return out

        monkeypatch.setattr(model, "forward", asserting_forward)

        prompt = torch.randint(1, 90, (2, 3))  # 3 % block_size(4) != 0 -> left pad
        with torch.no_grad():
            out = model.generate(
                prompt,
                algorithm="bd3lm",
                bd3lm_block_size=4,
                max_new_tokens=4,
                steps=2,
                temperature=0.0,
                mask_token_id=_TinyA2DFactory.MASK_ID,
                pad_token_id=_TinyA2DFactory.PAD_ID,
            )
        assert out.shape == (2, 7)
        assert not torch.any(out[:, 3:] == _TinyA2DFactory.MASK_ID)
