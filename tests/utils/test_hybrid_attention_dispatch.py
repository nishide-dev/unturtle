"""
The hybrid mask flows through `run_attention` unchanged (#63).

The previous #63 note recorded, as a blocker, that `run_attention` /
`AttentionContext` "has no path for a prebuilt per-row `[B, 1, L, L]` mask
together with packed metadata".  Traced against current main: **that is not
true for the mask itself.**  `run_attention` takes `context.attention_mask`
verbatim as `local_mask` and its 4-D branch converts a float mask to keep
semantics with `.eq(0)` — which is exactly the convention
`build_hybrid_prefix_attention_mask` emits.  No plumbing change is required to
consume the reference topology on the SDPA path.

The part of the note that *is* real is narrower and concerns precedence: the
packed block-diagonal builder only runs when `attention_mask is None`, so a
caller supplying a hybrid mask overrides it and must fold packing topology in
themselves.  These tests pin both halves, so a future refactor cannot quietly
break the path or the constraint.
"""

import pytest
import torch
import torch.nn.functional as F

from unturtle.utils.attention_dispatch import (
    SDPA,
    AttentionConfig,
    AttentionContext,
    build_hybrid_prefix_attention_mask,
    run_attention,
)

B, H, L, D = 2, 2, 6, 8


def _qkv(seed=0):
    """Head-major `[B, H, L, D]`, the layout the SDPA path expects."""
    torch.manual_seed(seed)
    return (
        torch.randn(B, H, L, D),
        torch.randn(B, H, L, D),
        torch.randn(B, H, L, D),
    )


def _hybrid(prompt_lengths=(3, 2)):
    return build_hybrid_prefix_attention_mask(
        prompt_lengths=torch.tensor(prompt_lengths),
        seq_len=L,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def _context(attention_mask=None, seq_info=None, batch=B):
    return AttentionContext(
        bsz=batch,
        q_len=L,
        kv_seq_len=L,
        n_heads=H,
        head_dim=D,
        requires_grad=False,
        seq_info=seq_info,
        attention_mask=attention_mask,
        causal_mask=None,
    )


def _config():
    # causal=False: dLLM callers must never let run_attention inject causality.
    return AttentionConfig(backend=SDPA, n_kv_heads=H, n_groups=1, causal=False)


class TestHybridMaskFlowsThrough:
    def test_matches_manual_sdpa_with_the_same_mask(self):
        """The refutation of the recorded blocker, stated as an equality."""
        Q, K, V = _qkv()
        mask = _hybrid()

        got = run_attention(config=_config(), context=_context(mask), Q=Q, K=K, V=V)
        expected = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask).transpose(
            1, 2
        )

        assert torch.allclose(got, expected, atol=1e-5), (
            "run_attention altered a caller-supplied hybrid mask"
        )

    def test_the_topology_actually_takes_effect(self):
        """Guards the equality above from being satisfied by a no-op.

        If `run_attention` silently dropped the mask, the test above would
        still pass whenever the reference also dropped it.  This pins that
        masked and unmasked attention genuinely differ.
        """
        Q, K, V = _qkv()

        masked = run_attention(
            config=_config(), context=_context(_hybrid()), Q=Q, K=K, V=V
        )
        unmasked = run_attention(config=_config(), context=_context(), Q=Q, K=K, V=V)

        assert not torch.allclose(masked, unmasked, atol=1e-5), (
            "the hybrid topology had no effect on the output"
        )

    def test_prompt_rows_do_not_attend_to_the_target(self):
        """The recipe's defining asymmetry, observed through real attention.

        Equation (3) blocks prompt->target.  With V rows that differ per
        position, a prompt query's output must be reproducible from the prompt
        block alone.
        """
        Q, K, V = _qkv(seed=3)
        prompt_len = 3
        mask = _hybrid((prompt_len, prompt_len))

        full = run_attention(config=_config(), context=_context(mask), Q=Q, K=K, V=V)

        # Recompute the prompt rows using only the prompt block: identical if
        # and only if the target columns were truly blocked.
        prompt_only = F.scaled_dot_product_attention(
            Q[:, :, :prompt_len],
            K[:, :, :prompt_len],
            V[:, :, :prompt_len],
            attn_mask=mask[:, :, :prompt_len, :prompt_len],
        ).transpose(1, 2)

        assert torch.allclose(full[:, :prompt_len], prompt_only, atol=1e-5), (
            "prompt rows changed when target keys were present, so the "
            "prompt->target block is not being enforced"
        )


class TestPackedPrecedence:
    def test_a_caller_mask_overrides_the_packed_builder(self):
        """The real constraint, and the only part of the old note that holds.

        `run_attention` builds its block-diagonal packed mask *only* when
        `attention_mask is None`.  A caller supplying a hybrid mask therefore
        takes full responsibility for packing topology — silently, which is
        why this is worth a test rather than a comment.
        """
        Q, K, V = _qkv()
        seq_info = (torch.tensor([3, 3]), None, L)
        # Batch-1 mask to match the batch-1 Q/K/V: a [2, 1, L, L] mask against
        # a 1-row batch is a broadcast error, not a precedence question.
        one_row = _hybrid((3,))

        with_mask = run_attention(
            config=_config(),
            context=_context(one_row, seq_info=seq_info, batch=1),
            Q=Q[:1],
            K=K[:1],
            V=V[:1],
        )
        packed_only = run_attention(
            config=_config(),
            context=_context(None, seq_info=seq_info, batch=1),
            Q=Q[:1],
            K=K[:1],
            V=V[:1],
        )

        assert not torch.allclose(with_mask, packed_only, atol=1e-5), (
            "the caller mask did not override the packed builder; if this "
            "now passes, precedence changed and hybrid+packed composition "
            "needs revisiting"
        )

    def test_composing_hybrid_with_packing_is_the_callers_job(self):
        """Documents the composition that a future conversion recipe needs.

        Not a plumbing gap — an intersection of two boolean topologies, which
        the caller can build today with the existing helpers.
        """
        from unturtle.utils.attention_dispatch import (
            build_sdpa_packed_bidirectional_attention_mask,
        )

        seq_info = (torch.tensor([3, 3]), None, L)
        packed = build_sdpa_packed_bidirectional_attention_mask(
            seq_info, dtype=torch.float32, device=torch.device("cpu")
        )
        hybrid = _hybrid((3, 3))[:1]

        blocked = torch.finfo(torch.float32).min
        combined = torch.where(
            (packed == 0) & (hybrid == 0),
            torch.zeros_like(hybrid),
            torch.full_like(hybrid, blocked),
        )

        assert combined.shape == (1, 1, L, L)
        # Within packed sample 0 the hybrid prompt rule still applies.
        allowed = combined[0, 0] == 0
        assert bool(allowed[0, 0]), "prompt[0] must see itself"
        assert not bool(allowed[0, 1]), "prompt[0] must not see prompt[1] (causal)"
        # Across the packed boundary everything stays blocked.
        assert not bool(allowed[:3, 3:].any()), "packing boundary was not honored"


class TestTheClaimIsScopedToBinaryMasks:
    def test_a_soft_additive_bias_is_not_preserved(self):
        """Bounds the "flows through unchanged" claim, which is not general.

        The 4-D branch converts a float mask with `.eq(0)`, so any non-zero
        entry becomes a hard block regardless of magnitude.  Lossless for the
        hybrid mask — it is binary, `{0, finfo.min}` — but false for a soft
        bias such as ALiBi or a relative-position penalty, which would be
        silently promoted to full blocking.

        Worth a test rather than a caveat because #63's next slice (an
        optimized structured bias) is exactly where someone would hand this
        path a non-binary tensor.
        """
        torch.manual_seed(0)
        q, k, v = (torch.randn(1, 1, 4, D) for _ in range(3))
        soft = torch.zeros(1, 1, 4, 4)
        soft[0, 0, 0, 1] = -2.0  # a partial penalty, not a block

        config = AttentionConfig(backend=SDPA, n_kv_heads=1, n_groups=1, causal=False)
        context = AttentionContext(
            bsz=1,
            q_len=4,
            kv_seq_len=4,
            n_heads=1,
            head_dim=D,
            requires_grad=False,
            seq_info=None,
            attention_mask=soft,
            causal_mask=None,
        )

        got = run_attention(config=config, context=context, Q=q, K=k, V=v)
        as_additive = F.scaled_dot_product_attention(q, k, v, attn_mask=soft).transpose(
            1, 2
        )

        assert not torch.allclose(got, as_additive, atol=1e-5), (
            "a soft bias now survives; if the 4-D branch stopped binarising, "
            "the hybrid mask could carry graded penalties too"
        )


class TestBackendDowngrade:
    """Supplying a mask costs Flash/xFormers — the real price of this path.

    `run_attention` switches to SDPA whenever `context.attention_mask is not
    None` (`attention_dispatch.py:130-135`).  So the reference topology works
    today, but only on SDPA: handing it a `[B, 1, L, L]` mask silently gives
    up the fast backends.

    That is exactly what #63's "optimized lowering" slice exists to fix — a
    structured bias the fast paths can consume instead of a dense mask.
    Pinned so the cost is a known quantity rather than a benchmark surprise.
    """

    @pytest.mark.parametrize("requested", ["FLASH_DENSE", "FLASH_VARLEN", "XFORMERS"])
    def test_a_caller_mask_yields_the_sdpa_result(self, requested):
        from unturtle.utils import attention_dispatch as dispatch

        Q, K, V = _qkv()
        mask = _hybrid()

        def run(backend):
            config = AttentionConfig(
                backend=backend, n_kv_heads=H, n_groups=1, causal=False
            )
            return run_attention(config=config, context=_context(mask), Q=Q, K=K, V=V)

        assert torch.allclose(
            run(getattr(dispatch, requested)), run(SDPA), atol=1e-6
        ), (
            f"backend={requested} with a caller mask did not produce the SDPA "
            "result; if a fast path now consumes 4-D masks, #63's optimized "
            "lowering slice should be re-scoped"
        )


class TestConventions:
    def test_the_mask_uses_the_additive_convention_sdpa_expects(self):
        mask = _hybrid()

        assert mask.dtype == torch.float32
        assert not bool(torch.isinf(mask).any()), (
            "finfo.min, not -inf: see the #63 padded-row NaN guard"
        )
        assert bool((mask == 0).any()), "no position was left open"

    @pytest.mark.parametrize("causal", [True, False])
    def test_a_four_dim_caller_mask_is_respected_either_way(self, causal):
        """`config.causal` must not rewrite a caller-supplied 4-D mask.

        The hybrid topology is *partly* causal, so a dispatch that re-imposed
        its own causality on top would silently change the semantics.
        """
        Q, K, V = _qkv()
        mask = _hybrid()
        config = AttentionConfig(backend=SDPA, n_kv_heads=H, n_groups=1, causal=causal)

        got = run_attention(config=config, context=_context(mask), Q=Q, K=K, V=V)
        expected = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask).transpose(
            1, 2
        )

        assert torch.allclose(got, expected, atol=1e-5), (
            f"causal={causal} altered a caller-supplied 4-D mask"
        )
