"""
Wiring the mask-free hybrid fast path into the Tiny-A2D forwards (#63).

#99 landed `hybrid_prefix_attention` as a standalone kernel; nothing called
it.  The Tiny-A2D forwards still build the dense eq.-(3) mask and hand it down
through every decoder layer, which forces SDPA and forfeits the ~1.4-2.1x
per-layer win the kernel measured.

The wiring is deliberately **fail-safe**: the model forward always builds the
dense mask exactly as before, and *additionally* passes
``hybrid_prompt_lengths`` down through layer kwargs when — and only when — the
mask-free split is exactly equivalent (no padding, uniform boundaries, not
packed, no prebuilt 4-D mask).  Only the patched fast forward consumes the
signal; unpatched HF attention ignores the kwarg and uses the mask it was
already given.  If the kwargs pipe is ever severed by an upstream change, the
model silently degrades to the dense path — slower, still correct.  The
failure direction is speed, never semantics.

Tests therefore pin three things: the signal is emitted exactly when the split
is equivalent, the patched attention consumes it and produces the same logits
as the unpatched dense path, and every ineligible input keeps today's
behaviour bit-for-bit.
"""

import types

import pytest
import torch


def _family(name):
    if name == "llama":
        from unturtle.models.conversion.a2d.tiny_a2d.modeling_llama import (
            TinyA2DLlamaConfig as C,
        )
        from unturtle.models.conversion.a2d.tiny_a2d.modeling_llama import (
            TinyA2DLlamaLMHeadModel as M,
        )
    elif name == "qwen2":
        from unturtle.models.conversion.a2d.tiny_a2d.modeling_qwen2 import (
            TinyA2DQwen2Config as C,
        )
        from unturtle.models.conversion.a2d.tiny_a2d.modeling_qwen2 import (
            TinyA2DQwen2LMHeadModel as M,
        )
    else:
        from unturtle.models.conversion.a2d.tiny_a2d.modeling_qwen3 import (
            TinyA2DQwen3Config as C,
        )
        from unturtle.models.conversion.a2d.tiny_a2d.modeling_qwen3 import (
            TinyA2DQwen3LMHeadModel as M,
        )
    return C, M


def _config(hybrid=True, family="llama", **kwargs):
    C, _ = _family(family)

    defaults = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        hybrid_attention=hybrid,
        # Test sequences are far below the real crossover; force the fast
        # path so the wiring is exercised.  The gate itself has its own tests.
        hybrid_fast_min_seq_len=0,
    )
    defaults.update(kwargs)
    return C(**defaults)


def _model(hybrid=True, seed=0, family="llama", **kwargs):
    _, M = _family(family)

    torch.manual_seed(seed)
    return M(_config(hybrid, family=family, **kwargs)).eval()


def _patch(model):
    """Install the fast forward the way `_patch_lora_layers` does.

    Installation in production is CUDA-only, but the function itself carries a
    CPU RoPE fallback, so the wiring is testable without a GPU.
    """
    from unturtle.fast_diffusion_model import _install_apply_stubs
    from unturtle.models.conversion.a2d.tiny_a2d._fast_forward import (
        TinyA2DAttention_fast_forward,
    )

    _install_apply_stubs(model)
    for layer in model.model.layers:
        layer.self_attn.forward = types.MethodType(
            TinyA2DAttention_fast_forward, layer.self_attn
        )
    return model


def _batch(batch=2, length=8):
    torch.manual_seed(1)
    return torch.randint(1, 64, (batch, length))


def _spy_kernel(monkeypatch):
    """Count calls to the split kernel where the fast forward imports it."""
    import unturtle.models.conversion.a2d.tiny_a2d._fast_forward as ff

    calls = []
    real = ff.hybrid_prefix_attention

    def spy(*args, **kwargs):
        calls.append(kwargs.get("prompt_lengths"))
        return real(*args, **kwargs)

    monkeypatch.setattr(ff, "hybrid_prefix_attention", spy)
    return calls


class TestThePatchedPathUsesTheKernel:
    def test_the_kernel_runs_once_per_layer(self, monkeypatch):
        calls = _spy_kernel(monkeypatch)
        model = _patch(_model())

        with torch.no_grad():
            model(
                input_ids=_batch(),
                prompt_lengths=torch.tensor([3, 3]),
            )

        assert len(calls) == 2, (
            f"kernel ran {len(calls)} times for 2 layers; the signal is not "
            "reaching the patched attention"
        )

    @pytest.mark.parametrize("family", ["llama", "qwen2"])
    def test_patched_and_unpatched_logits_agree(self, family):
        """The equivalence claim, end to end through a real model.

        Same weights, same inputs: the patched forward (split kernel, mask
        ignored) and the unpatched forward (HF attention through the dense
        mask) must produce the same logits.  This is what makes the signal
        safe to emit — either consumer yields the same answer.

        qwen3 is deliberately absent: its *patched* path diverges from the
        unpatched reference on `main` because the shared fast forward skips
        `q_norm`/`k_norm` (#102, pre-existing).  Its wiring is pinned by the
        patched-dense-vs-patched-fast test below instead, which is the claim
        this PR actually makes.
        """
        reference = _model(seed=3, family=family)
        patched = _patch(_model(seed=3, family=family))
        input_ids = _batch()
        prompt_lengths = torch.tensor([3, 3])

        with torch.no_grad():
            expected = reference(
                input_ids=input_ids, prompt_lengths=prompt_lengths
            ).logits
            got = patched(input_ids=input_ids, prompt_lengths=prompt_lengths).logits

        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("family", ["llama", "qwen2", "qwen3"])
    def test_the_fast_path_matches_the_dense_path_when_both_are_patched(
        self, family, monkeypatch
    ):
        """Signal on vs signal suppressed, same patched model, all families.

        Isolates exactly what this wiring changes — which route the patched
        attention takes — from any pre-existing patched-vs-unpatched gap
        (qwen3's #102).  The two routes must agree bit-for-bit in fp32.
        """
        import unturtle.models.conversion.a2d.tiny_a2d._fast_forward as ff

        model = _patch(_model(seed=13, family=family))
        input_ids = _batch()
        prompt_lengths = torch.tensor([3, 3])

        calls = _spy_kernel(monkeypatch)
        with torch.no_grad():
            fast = model(input_ids=input_ids, prompt_lengths=prompt_lengths).logits
        assert calls, f"{family}: the fast path did not run"

        module = __import__(
            f"unturtle.models.conversion.a2d.tiny_a2d.modeling_{family}",
            fromlist=["hybrid_fast_path_lengths"],
        )
        real = module.hybrid_fast_path_lengths
        try:
            module.hybrid_fast_path_lengths = lambda *a, **k: None
            with torch.no_grad():
                dense = model(input_ids=input_ids, prompt_lengths=prompt_lengths).logits
        finally:
            module.hybrid_fast_path_lengths = real

        torch.testing.assert_close(fast, dense, atol=1e-5, rtol=1e-5)

    def test_gqa_heads_reach_the_kernel_correctly(self):
        """Grouped KV heads must expand with HF's mapping before the split.

        The kernel takes matched head counts, so the fast forward expands
        K/V itself.  A wrong interleaving (kv head g feeding the wrong query
        group) still runs and still converges — end-to-end equality against
        the unpatched model, whose `repeat_kv` is the reference mapping, is
        what pins it.
        """
        kwargs = dict(num_attention_heads=4, num_key_value_heads=2)
        reference = _model(seed=5, **kwargs)
        patched = _patch(_model(seed=5, **kwargs))
        input_ids = _batch()
        prompt_lengths = torch.tensor([3, 3])

        with torch.no_grad():
            expected = reference(
                input_ids=input_ids, prompt_lengths=prompt_lengths
            ).logits
            got = patched(input_ids=input_ids, prompt_lengths=prompt_lengths).logits

        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


class TestTheSignalIsEmittedExactlyWhenEquivalent:
    """Model-level eligibility, observed at the attention boundary."""

    @staticmethod
    def _captured_kwargs(model, **forward_kwargs):
        seen = {}
        attn = model.model.layers[0].self_attn
        real = attn.forward

        def capture(*args, **kwargs):
            seen.update(kwargs)
            return real(*args, **kwargs)

        attn.forward = capture
        with torch.no_grad():
            model(**forward_kwargs)
        return seen

    def test_an_eligible_forward_signals_cpu_lengths(self):
        seen = self._captured_kwargs(
            _model(),
            input_ids=_batch(),
            prompt_lengths=torch.tensor([3, 3]),
        )

        signal = seen.get("hybrid_prompt_lengths")
        assert signal is not None, "eligible forward did not emit the signal"
        assert signal.device.type == "cpu", (
            "lengths must ride on CPU so the per-layer uniformity check does "
            "not force a device sync"
        )

    def test_padding_suppresses_the_signal(self):
        """A padded row breaks the pure row split, so the dense mask must win."""
        attention_mask = torch.ones(2, 8, dtype=torch.long)
        attention_mask[0, 6:] = 0

        seen = self._captured_kwargs(
            _model(),
            input_ids=_batch(),
            attention_mask=attention_mask,
            prompt_lengths=torch.tensor([3, 3]),
        )

        assert "hybrid_prompt_lengths" not in seen

    def test_ragged_boundaries_suppress_the_signal(self):
        """Per-layer fallback would rebuild the dense mask once per layer;
        suppressing at the model level keeps one build per forward."""
        seen = self._captured_kwargs(
            _model(),
            input_ids=_batch(),
            prompt_lengths=torch.tensor([3, 5]),
        )

        assert "hybrid_prompt_lengths" not in seen

    def test_no_prompt_lengths_means_no_signal(self):
        seen = self._captured_kwargs(_model(), input_ids=_batch())

        assert "hybrid_prompt_lengths" not in seen

    def test_packed_metadata_suppresses_the_signal(self):
        """Packed isolation lives in a prebuilt block mask the split cannot
        express; the packed keys must veto the signal outright."""
        seen = self._captured_kwargs(
            _model(),
            input_ids=_batch(),
            prompt_lengths=torch.tensor([3, 3]),
            packed_seq_lengths=torch.tensor([4, 4, 4, 4], dtype=torch.int32),
        )

        assert "hybrid_prompt_lengths" not in seen


class TestIneligibleInputsKeepTodaysBehaviour:
    def test_the_unpatched_model_ignores_the_signal(self):
        """The fail-safe direction: HF attention does not know the kwarg.

        With and without the signal suppressed, an unpatched forward must be
        bit-identical — the kwarg is advisory, and the dense mask it rides
        alongside is the actual contract.
        """
        import unturtle.models.conversion.a2d.tiny_a2d.modeling_llama as m

        model = _model(seed=7)
        input_ids = _batch()
        prompt_lengths = torch.tensor([3, 3])

        with torch.no_grad():
            signalled = model(input_ids=input_ids, prompt_lengths=prompt_lengths).logits

        real = m.hybrid_fast_path_lengths
        try:
            m.hybrid_fast_path_lengths = lambda *a, **k: None
            with torch.no_grad():
                suppressed = model(
                    input_ids=input_ids, prompt_lengths=prompt_lengths
                ).logits
        finally:
            m.hybrid_fast_path_lengths = real

        assert torch.equal(signalled, suppressed)

    def test_a_padded_patched_forward_matches_the_reference(self):
        """Ineligible input through the patched model still goes dense —
        and dense must agree with the unpatched reference."""
        attention_mask = torch.ones(2, 8, dtype=torch.long)
        attention_mask[0, 6:] = 0
        reference = _model(seed=9)
        patched = _patch(_model(seed=9))
        input_ids = _batch()
        prompt_lengths = torch.tensor([3, 3])

        with torch.no_grad():
            expected = reference(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_lengths=prompt_lengths,
            ).logits
            got = patched(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_lengths=prompt_lengths,
            ).logits

        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)

    def test_the_hybrid_semantics_survive_the_fast_path(self, monkeypatch):
        """The asymmetry itself, through the patched model.

        Rewriting the target region must leave prompt logits bit-identical —
        the same behavioural check slice B established for the dense path,
        now through the split kernel.  Ensures the wiring did not quietly
        route to plain bidirectional attention.
        """
        calls = _spy_kernel(monkeypatch)
        model = _patch(_model(seed=11))
        prompt_lengths = torch.tensor([4, 4])
        input_ids = _batch()

        perturbed = input_ids.clone()
        torch.manual_seed(99)
        perturbed[:, 4:] = torch.randint(1, 64, perturbed[:, 4:].shape)

        with torch.no_grad():
            before = model(input_ids=input_ids, prompt_lengths=prompt_lengths).logits
            after = model(input_ids=perturbed, prompt_lengths=prompt_lengths).logits

        assert calls, "the fast path did not run; this test would prove nothing"
        assert torch.equal(before[:, :4], after[:, :4]), (
            "rewriting the target moved prompt logits; the split kernel is "
            "not enforcing the eq.-(3) asymmetry"
        )
        assert not torch.allclose(before[:, 4:], after[:, 4:])


class TestAttentionLevelDefenceInDepth:
    """Guards on conditions the model level is supposed to prevent.

    The model forward suppresses the signal for packed and cached calls, but
    the fast forward is a public function a caller can invoke directly — and
    "the other layer checks it" is how two layers end up each assuming the
    other one did.  These exercise the attention-level guards on inputs the
    model level would never emit.
    """

    @staticmethod
    def _stub(n_heads=2, head_dim=4):
        hidden = n_heads * head_dim

        class _StubAttn:
            pass

        stub = _StubAttn()
        stub.config = type(
            "Cfg",
            (),
            {"num_attention_heads": n_heads, "num_key_value_heads": n_heads},
        )()
        stub.num_key_value_groups = 1
        stub.head_dim = head_dim
        stub.layer_idx = 0
        torch.manual_seed(0)
        w = torch.randn(hidden, 3 * hidden)
        stub.apply_qkv = lambda self, x: (x @ w).chunk(3, dim=-1)
        stub.apply_o = lambda self, x: x
        return stub, hidden

    def test_packed_metadata_beats_the_signal(self, monkeypatch):
        """Both kwargs at once: packed isolation must win.

        The split cannot express block-diagonal isolation, so taking it here
        would let packed samples attend across their boundaries — attention
        runs, loss decreases, nothing surfaces it.
        """
        import unturtle.models.conversion.a2d.tiny_a2d._fast_forward as ff

        calls = _spy_kernel(monkeypatch)
        stub, hidden = self._stub()

        out, _ = ff.TinyA2DAttention_fast_forward(
            stub,
            torch.randn(1, 8, hidden),
            position_embeddings=None,
            attention_mask=None,
            past_key_values=None,
            packed_seq_lengths=torch.tensor([4, 4], dtype=torch.int32),
            hybrid_prompt_lengths=torch.tensor([3]),
        )

        assert calls == [], "the split ran despite packed metadata"
        assert out.shape == (1, 8, hidden)

    def test_a_cache_beats_the_signal(self, monkeypatch):
        """eq. (3) has no rectangular form, so a cache must take the mask path.

        The model level raises on hybrid+cache before any signal exists, but a
        direct caller can hand both to the attention function.
        """
        from transformers import DynamicCache

        import unturtle.models.conversion.a2d.tiny_a2d._fast_forward as ff

        calls = _spy_kernel(monkeypatch)
        stub, hidden = self._stub()

        out, _ = ff.TinyA2DAttention_fast_forward(
            stub,
            torch.randn(1, 8, hidden),
            position_embeddings=None,
            attention_mask=None,
            past_key_values=DynamicCache(),
            hybrid_prompt_lengths=torch.tensor([3]),
        )

        assert calls == [], "the split ran under a KV cache"
        assert out.shape == (1, 8, hidden)


class TestPrebuiltMasksSuppressTheSignal:
    def test_a_prebuilt_4d_mask_suppresses_the_signal(self):
        """A caller-built 4-D mask carries topology the split cannot see.

        The hybrid mask is *intersected* with it (slice B), so the dense
        result respects both; the split would respect only eq. (3) and drop
        whatever the caller encoded — packed isolation being the concrete
        case.  Eligibility must therefore refuse anything but a plain 2-D
        all-ones mask.
        """
        model = _model()
        seen = {}
        attn = model.model.layers[0].self_attn
        real = attn.forward

        def capture(*args, **kwargs):
            seen.update(kwargs)
            return real(*args, **kwargs)

        attn.forward = capture
        prebuilt = torch.zeros(2, 1, 8, 8)  # additive, all-allowed

        with torch.no_grad():
            model(
                input_ids=_batch(),
                attention_mask=prebuilt,
                prompt_lengths=torch.tensor([3, 3]),
            )

        assert "hybrid_prompt_lengths" not in seen


class TestTheLengthGate:
    """The fast path is a net loss below the measured crossover.

    Full-forward measurement on an 8-layer bf16 model: 0.90x at L=1024,
    1.50x at L=2048, 1.92x at L=4096.  Emitting the signal unconditionally
    would silently slow every short-sequence forward by ~10%, so eligibility
    is gated on a declared config field — and the gate only ever trades
    speed, since the dense mask is always built.
    """

    def test_the_default_gate_suppresses_short_sequences(self):
        model = _model(hybrid_fast_min_seq_len=2048)
        seen = {}
        attn = model.model.layers[0].self_attn
        real = attn.forward

        def capture(*args, **kwargs):
            seen.update(kwargs)
            return real(*args, **kwargs)

        attn.forward = capture
        with torch.no_grad():
            model(input_ids=_batch(), prompt_lengths=torch.tensor([3, 3]))

        assert "hybrid_prompt_lengths" not in seen, (
            "an 8-token forward was signalled despite the 2048 gate; short "
            "sequences would be silently slowed"
        )

    def test_the_gate_defaults_to_the_measured_crossover(self):
        from unturtle.models.conversion.a2d.tiny_a2d.modeling_llama import (
            TinyA2DLlamaConfig,
        )

        assert TinyA2DLlamaConfig().hybrid_fast_min_seq_len == 2048
