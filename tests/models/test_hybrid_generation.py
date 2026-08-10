"""
Hybrid-aware masked generation (#127): the eq.-(3) prompt boundary rides
every denoise forward when — and only when — the model is hybrid.

#125/#126 measured the cost of its absence: a hybrid-trained model decoding
with bidirectional prompt attention it never trained with sat at the MAUVE
floor on both seeds.  The pinned contract (from the #67 re-ordering):

- boundary = the PHYSICAL end of the pre-generation prompt tensor
  (``input_ids.shape[1]``), never ``attention_mask.sum()`` — under left
  padding the pads sit inside the boundary on the causal side and the
  existing 2-D attention-mask intersection excludes them;
- hybrid-gated: non-hybrid MDLM generation is bitwise/API unchanged;
- the 2-D mask is preserved on the hybrid branch (`_sample`'s 4-D
  pre-broadcast would bypass the hybrid intersection by contract);
- direct forward and in-generation forward agree.
"""

import pytest
import torch

VOCAB = 16
MASK_ID = VOCAB - 1
PROMPT = 4


def _model(hybrid):
    from unturtle.models.conversion.a2d.tiny_a2d import (
        TinyA2DLlamaConfig,
        TinyA2DLlamaLMHeadModel,
    )

    torch.manual_seed(0)
    return TinyA2DLlamaLMHeadModel(
        TinyA2DLlamaConfig(
            vocab_size=VOCAB,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            mask_token_id=MASK_ID,
            hybrid_attention=hybrid,
        )
    ).eval()


def _capture_forwards(model):
    """Record (kwargs) of every denoise forward the generation loop issues."""
    seen = []
    original = model.forward

    def recording(*args, **kwargs):
        seen.append(kwargs)
        return original(*args, **kwargs)

    model.forward = recording
    return seen


class TestTheBoundaryRidesHybridGeneration:
    def test_every_denoise_forward_receives_the_physical_prompt_width(self):
        model = _model(hybrid=True)
        seen = _capture_forwards(model)
        prompt = torch.randint(0, VOCAB - 1, (2, PROMPT))

        torch.manual_seed(1)
        model.generate(prompt, algorithm="mdlm", max_new_tokens=8, steps=4)

        assert len(seen) == 4
        for kwargs in seen:
            lengths = kwargs.get("prompt_lengths")
            assert lengths is not None, "a denoise forward missed the boundary"
            assert lengths.tolist() == [PROMPT, PROMPT]

    def test_the_boundary_is_the_padded_width_not_the_mask_sum(self):
        """Left padding: rows have different TRUE prompt lengths, but the
        boundary is the physical prompt-tensor width for every row — the
        pads inside it are the attention mask's job, not the boundary's."""
        model = _model(hybrid=True)
        seen = _capture_forwards(model)
        padded_width = PROMPT + 2
        prompt = torch.randint(0, VOCAB - 1, (2, padded_width))
        attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])

        torch.manual_seed(2)
        model.generate(
            prompt,
            attention_mask=attention_mask,
            algorithm="mdlm",
            max_new_tokens=8,
            steps=2,
        )

        assert seen, "no forwards captured"
        for kwargs in seen:
            assert kwargs["prompt_lengths"].tolist() == [padded_width, padded_width]
            mask = kwargs.get("attention_mask")
            assert mask is not None and mask.dim() == 2, (
                "the hybrid branch must keep the mask 2-D: a prebuilt 4-D "
                "mask bypasses the eq.(3) intersection by contract"
            )

    def test_a_non_hybrid_model_never_sees_the_boundary(self):
        model = _model(hybrid=False)
        seen = _capture_forwards(model)
        prompt = torch.randint(0, VOCAB - 1, (2, PROMPT))

        torch.manual_seed(3)
        model.generate(prompt, algorithm="mdlm", max_new_tokens=8, steps=4)

        assert seen
        for kwargs in seen:
            assert "prompt_lengths" not in kwargs, (
                "the boundary leaked into a non-hybrid generation"
            )

    def test_non_hybrid_generation_is_seed_stable(self):
        """The bitwise-invariance pin available in-repo: the same seeded
        call yields identical ids on the non-hybrid path (a regression
        anchor for 'nothing changed there')."""
        prompt = torch.randint(0, VOCAB - 1, (2, PROMPT))

        torch.manual_seed(4)
        first = _model(hybrid=False).generate(
            prompt, algorithm="mdlm", max_new_tokens=8, steps=4
        )
        torch.manual_seed(4)
        second = _model(hybrid=False).generate(
            prompt, algorithm="mdlm", max_new_tokens=8, steps=4
        )

        assert torch.equal(first, second)

    def test_direct_and_in_generation_forwards_agree(self):
        """Parity: replay the FIRST captured in-generation forward as a
        direct call with the same tensors — logits must match exactly."""
        model = _model(hybrid=True)
        captured = []
        original = model.forward

        def recording(*args, **kwargs):
            out = original(*args, **kwargs)
            if not captured:
                # Clone: the loop mutates `x` in place after the forward, so
                # an unclosed reference would replay a LATER state.
                snapshot = {
                    key: value.clone() if torch.is_tensor(value) else value
                    for key, value in kwargs.items()
                }
                captured.append((snapshot, out.logits.detach().clone()))
            return out

        model.forward = recording
        prompt = torch.randint(0, VOCAB - 1, (2, PROMPT))
        torch.manual_seed(5)
        model.generate(prompt, algorithm="mdlm", max_new_tokens=8, steps=2)

        kwargs, in_generation_logits = captured[0]
        model.forward = original
        with torch.no_grad():
            direct = model(**kwargs).logits

        assert torch.equal(direct, in_generation_logits), (
            "the generation loop's hybrid forward diverges from a direct call"
        )

    def test_the_boundary_changes_the_denoise_logits(self):
        """The end-to-end observable #125 lacked, asserted on LOGITS: a
        sampled-ids comparison flakes (~25% of seeds let a tiny untrained
        model's argmax absorb the ~0.3 logit shift, measured in review),
        while the logits of the first in-generation forward are a stable
        observable of the same claim — the threading is not inert."""
        model = _model(hybrid=True)
        captured = []
        original = model.forward

        def recording(*args, **kwargs):
            out = original(*args, **kwargs)
            if not captured:
                snapshot = {
                    key: value.clone() if torch.is_tensor(value) else value
                    for key, value in kwargs.items()
                }
                captured.append((snapshot, out.logits.detach().clone()))
            return out

        model.forward = recording
        prompt = torch.randint(0, VOCAB - 1, (2, PROMPT))
        torch.manual_seed(6)
        model.generate(prompt, algorithm="mdlm", max_new_tokens=8, steps=2)

        kwargs, threaded_logits = captured[0]
        model.forward = original
        kwargs.pop("prompt_lengths")
        with torch.no_grad():
            unthreaded_logits = model(**kwargs).logits

        assert not torch.allclose(threaded_logits, unthreaded_logits), (
            "threading the boundary did not change the denoise logits"
        )


class TestAutoNeverPicksACachePathForHybrid:
    """The #128 review's gap: `auto` resolved hybrid models to
    `block_decode`, whose loop never threads the boundary — silently
    decoding under the exact mismatch this feature exists to close.  A
    hybrid model does not SUPPORT cache-based block decoding (the eq.(3)
    mask is square by contract), so the capability probe must say so."""

    def test_auto_resolves_a_hybrid_model_to_mdlm(self):
        from unturtle.models.generation.sampler import resolve_algorithm

        assert (
            resolve_algorithm("auto", _model(hybrid=True), bd3lm_requested=False)
            == "mdlm"
        )

    def test_explicit_block_decode_on_a_hybrid_model_is_rejected(self):
        from unturtle.models.generation.sampler import resolve_algorithm

        with pytest.raises(ValueError, match="block-decode|block_decode"):
            resolve_algorithm(
                "block_decode", _model(hybrid=True), bd3lm_requested=False
            )

    def test_non_hybrid_auto_resolution_is_unchanged(self):
        from unturtle.models.generation.sampler import resolve_algorithm

        assert (
            resolve_algorithm("auto", _model(hybrid=False), bd3lm_requested=False)
            == "block_decode"
        )
