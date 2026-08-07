"""
Tests for the sparse masked LM-head loss path (#61).

The dense path computes ``[B, L, V]`` logits and then discards every unmasked
position.  The sparse path gathers first::

    hidden [B,L,H] -> gather [M,H] -> project [M,V] -> CE on M targets

Equivalence is the whole contract: same loss, same gradients.  The subtle part
is normalization — ``n_maskable`` is ``(labels != -100).sum()`` over the full
``[B, L]``, so it must be counted *before* the gather destroys that structure.
"""

import pytest
import torch


def _tiny_a2d_model(vocab_size=64, hidden=16):
    from unturtle.models.conversion.a2d.tiny_a2d.modeling_llama import (
        TinyA2DLlamaConfig,
        TinyA2DLlamaLMHeadModel,
    )

    config = TinyA2DLlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return TinyA2DLlamaLMHeadModel(config)


def _batch(batch_size=2, seq_len=6, vocab_size=64, prompt_len=2, seed=0):
    """A masked-diffusion batch: prompt is -100, completion is supervised."""
    torch.manual_seed(seed)
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    diffusion_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    diffusion_mask[0, prompt_len] = True
    diffusion_mask[0, seq_len - 1] = True
    diffusion_mask[1, prompt_len + 1] = True
    return input_ids, labels, diffusion_mask


class TestLossEquivalence:
    @pytest.mark.parametrize("loss_norm_type", ["token", "sequence", "batch"])
    def test_matches_the_dense_loss(self, loss_norm_type):
        from unturtle.kernels.masked_diffusion_loss import fast_masked_diffusion_loss
        from unturtle.kernels.sparse_masked_loss import sparse_masked_diffusion_loss

        model = _tiny_a2d_model()
        model.eval()
        input_ids, labels, diffusion_mask = _batch()

        with torch.no_grad():
            dense = fast_masked_diffusion_loss(
                logits=model(input_ids=input_ids).logits,
                labels=labels,
                diffusion_mask=diffusion_mask,
                loss_norm_type=loss_norm_type,
            )
            sparse = sparse_masked_diffusion_loss(
                model=model,
                input_ids=input_ids,
                labels=labels,
                diffusion_mask=diffusion_mask,
                loss_norm_type=loss_norm_type,
            )

        assert torch.allclose(sparse, dense, atol=1e-6), (
            f"{loss_norm_type}: dense={dense.item()} sparse={sparse.item()}"
        )

    def test_normalization_counts_maskable_not_masked(self):
        """`n_maskable` is over `labels != -100`, not over `diffusion_mask`.

        The two differ whenever some maskable positions survived the Bernoulli
        draw — i.e. essentially always. Counting the wrong one changes the loss
        scale without changing its shape, which is exactly the kind of drift
        that goes unnoticed.
        """
        from unturtle.kernels.sparse_masked_loss import sparse_masked_diffusion_loss

        model = _tiny_a2d_model()
        model.eval()
        input_ids, labels, diffusion_mask = _batch()

        n_maskable = int((labels != -100).sum())
        n_masked = int(diffusion_mask.sum())
        assert n_maskable != n_masked, "fixture does not distinguish the two counts"

        with torch.no_grad():
            loss = sparse_masked_diffusion_loss(
                model=model,
                input_ids=input_ids,
                labels=labels,
                diffusion_mask=diffusion_mask,
            )
            summed = sparse_masked_diffusion_loss(
                model=model,
                input_ids=input_ids,
                labels=labels,
                diffusion_mask=diffusion_mask,
                loss_norm_type="batch",
            )

        # loss == sum / n_maskable, and "batch" == sum / B.
        batch_size = input_ids.shape[0]
        assert torch.allclose(loss, summed * batch_size / n_maskable, atol=1e-6)

    def test_masked_position_with_an_ignored_label_is_not_projected(self):
        """A position can be masked yet unsupervised; keep it out of the GEMM.

        `diffusion_mask` and `labels != -100` are independent, so a
        completion-only or packed batch can mark a `-100` position as masked.
        The loss is unaffected either way — `F.cross_entropy` defaults to
        `ignore_index=-100`, giving zero loss *and* zero gradient — so this
        asserts the thing that actually differs: such rows never reach the
        output projection, which is the entire point of this path.
        """
        from unturtle.kernels.masked_diffusion_loss import fast_masked_diffusion_loss
        from unturtle.kernels.sparse_masked_loss import sparse_masked_diffusion_loss

        model = _tiny_a2d_model()
        model.eval()
        input_ids, labels, diffusion_mask = _batch()
        # Mark a prompt position (label == -100) as masked.
        diffusion_mask[0, 0] = True
        assert labels[0, 0].item() == -100
        n_supervised_and_masked = int((diffusion_mask & (labels != -100)).sum())

        projected_rows = []
        head = model.get_output_embeddings()
        original = head.forward

        def record(x):
            projected_rows.append(x.shape[0])
            return original(x)

        head.forward = record
        try:
            with torch.no_grad():
                dense = fast_masked_diffusion_loss(
                    logits=model(input_ids=input_ids).logits,
                    labels=labels,
                    diffusion_mask=diffusion_mask,
                )
                projected_rows.clear()  # ignore the dense reference's own call
                sparse = sparse_masked_diffusion_loss(
                    model=model,
                    input_ids=input_ids,
                    labels=labels,
                    diffusion_mask=diffusion_mask,
                )
        finally:
            head.forward = original

        assert torch.allclose(sparse, dense, atol=1e-6)
        assert projected_rows == [n_supervised_and_masked], (
            f"projected {projected_rows} rows; the ignored-label position was "
            "pushed through the output projection for nothing"
        )

    def test_matches_dense_under_per_sequence_weights(self):
        from unturtle.kernels.masked_diffusion_loss import fast_masked_diffusion_loss
        from unturtle.kernels.sparse_masked_loss import sparse_masked_diffusion_loss

        model = _tiny_a2d_model()
        model.eval()
        input_ids, labels, diffusion_mask = _batch()
        weights = torch.tensor([2.0, 0.5])

        with torch.no_grad():
            dense = fast_masked_diffusion_loss(
                logits=model(input_ids=input_ids).logits,
                labels=labels,
                diffusion_mask=diffusion_mask,
                loss_weights=weights,
            )
            sparse = sparse_masked_diffusion_loss(
                model=model,
                input_ids=input_ids,
                labels=labels,
                diffusion_mask=diffusion_mask,
                loss_weights=weights,
            )

        assert torch.allclose(sparse, dense, atol=1e-6)

    def test_matches_dense_under_per_token_weights(self):
        """CART weights are [B, L] and must be gathered alongside the targets."""
        from unturtle.kernels.masked_diffusion_loss import fast_masked_diffusion_loss
        from unturtle.kernels.sparse_masked_loss import sparse_masked_diffusion_loss

        model = _tiny_a2d_model()
        model.eval()
        input_ids, labels, diffusion_mask = _batch()
        weights = torch.rand(input_ids.shape) + 0.5

        with torch.no_grad():
            dense = fast_masked_diffusion_loss(
                logits=model(input_ids=input_ids).logits,
                labels=labels,
                diffusion_mask=diffusion_mask,
                loss_weights=weights,
            )
            sparse = sparse_masked_diffusion_loss(
                model=model,
                input_ids=input_ids,
                labels=labels,
                diffusion_mask=diffusion_mask,
                loss_weights=weights,
            )

        assert torch.allclose(sparse, dense, atol=1e-6)

    def test_matches_dense_under_a_padding_mask(self):
        from unturtle.kernels.masked_diffusion_loss import fast_masked_diffusion_loss
        from unturtle.kernels.sparse_masked_loss import sparse_masked_diffusion_loss

        model = _tiny_a2d_model()
        model.eval()
        input_ids, labels, diffusion_mask = _batch()
        attention_mask = torch.ones_like(input_ids)
        attention_mask[0, -2:] = 0
        labels[0, -2:] = -100
        diffusion_mask[0, -1] = False

        with torch.no_grad():
            dense = fast_masked_diffusion_loss(
                logits=model(input_ids=input_ids, attention_mask=attention_mask).logits,
                labels=labels,
                diffusion_mask=diffusion_mask,
            )
            sparse = sparse_masked_diffusion_loss(
                model=model,
                input_ids=input_ids,
                labels=labels,
                diffusion_mask=diffusion_mask,
                attention_mask=attention_mask,
            )

        assert torch.allclose(sparse, dense, atol=1e-6)


class TestGradientEquivalence:
    def test_gradients_match_the_dense_path(self):
        from unturtle.kernels.masked_diffusion_loss import fast_masked_diffusion_loss
        from unturtle.kernels.sparse_masked_loss import sparse_masked_diffusion_loss

        input_ids, labels, diffusion_mask = _batch()

        torch.manual_seed(7)
        dense_model = _tiny_a2d_model()
        torch.manual_seed(7)
        sparse_model = _tiny_a2d_model()

        fast_masked_diffusion_loss(
            logits=dense_model(input_ids=input_ids).logits,
            labels=labels,
            diffusion_mask=diffusion_mask,
        ).backward()
        sparse_masked_diffusion_loss(
            model=sparse_model,
            input_ids=input_ids,
            labels=labels,
            diffusion_mask=diffusion_mask,
        ).backward()

        reference = dict(dense_model.named_parameters())
        compared = 0
        for name, param in sparse_model.named_parameters():
            expected = reference[name].grad
            if expected is None:
                assert param.grad is None, f"{name}: spurious sparse gradient"
                continue
            assert param.grad is not None, f"{name}: sparse dropped the gradient"
            assert torch.allclose(param.grad, expected, atol=1e-6), name
            compared += 1

        assert compared > 0, "no gradients compared; the test proved nothing"

    def test_lm_head_receives_gradient(self):
        """The head is only applied to M rows, but must still learn."""
        from unturtle.kernels.sparse_masked_loss import sparse_masked_diffusion_loss

        model = _tiny_a2d_model()
        input_ids, labels, diffusion_mask = _batch()

        sparse_masked_diffusion_loss(
            model=model,
            input_ids=input_ids,
            labels=labels,
            diffusion_mask=diffusion_mask,
        ).backward()

        head_grad = model.get_output_embeddings().weight.grad
        assert head_grad is not None
        assert torch.any(head_grad != 0)


class TestItStaysSparse:
    def test_no_dense_vocab_tensor_is_materialized(self):
        """#61's real requirement: the [B, L, V] GEMM must not happen."""
        from unturtle.kernels.sparse_masked_loss import sparse_masked_diffusion_loss

        model = _tiny_a2d_model()
        model.eval()
        input_ids, labels, diffusion_mask = _batch()
        n_masked = int(diffusion_mask.sum())

        seen = []
        head = model.get_output_embeddings()
        original = head.forward

        def record(x):
            seen.append(tuple(x.shape))
            return original(x)

        head.forward = record
        try:
            with torch.no_grad():
                sparse_masked_diffusion_loss(
                    model=model,
                    input_ids=input_ids,
                    labels=labels,
                    diffusion_mask=diffusion_mask,
                )
        finally:
            head.forward = original

        assert seen, "the output projection never ran"
        for shape in seen:
            assert shape[0] == n_masked, (
                f"projected {shape[0]} rows, expected {n_masked}: "
                "the head ran over the full sequence"
            )


class TestFallback:
    def test_reports_unsupported_models(self):
        """Callers need to know when to take the dense path."""
        from unturtle.kernels.sparse_masked_loss import supports_sparse_masked_loss

        class _Cfg:
            model_type = "mdlm-dit"

        class _Model:
            config = _Cfg()

        assert supports_sparse_masked_loss(_Model()) is False
        assert supports_sparse_masked_loss(_tiny_a2d_model()) is True

    def test_raises_rather_than_silently_going_dense(self):
        from unturtle.kernels.sparse_masked_loss import sparse_masked_diffusion_loss

        class _Cfg:
            model_type = "mdlm-dit"

        class _Model:
            config = _Cfg()

        input_ids, labels, diffusion_mask = _batch()
        with pytest.raises(ValueError, match="sparse"):
            sparse_masked_diffusion_loss(
                model=_Model(),
                input_ids=input_ids,
                labels=labels,
                diffusion_mask=diffusion_mask,
            )

    def test_empty_mask_does_not_divide_by_zero(self):
        from unturtle.kernels.sparse_masked_loss import sparse_masked_diffusion_loss

        model = _tiny_a2d_model()
        model.eval()
        input_ids, labels, _ = _batch()
        empty = torch.zeros_like(labels, dtype=torch.bool)

        with torch.no_grad():
            loss = sparse_masked_diffusion_loss(
                model=model,
                input_ids=input_ids,
                labels=labels,
                diffusion_mask=empty,
            )

        assert torch.isfinite(loss)
        assert loss.item() == 0.0
