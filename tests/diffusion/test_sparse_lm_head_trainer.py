"""
`DiffusionTrainer(sparse_lm_head=True)` wiring (#61).

The kernel's equivalence is covered in `test_sparse_masked_loss.py`.  What this
file pins is the *wiring*: that the trainer actually routes to the sparse path,
that routing produces the same loss and gradients as the dense path it replaces,
and that the cases where it must not route (unsupported model, `return_outputs`)
behave as documented rather than silently degrading.
"""

import pytest
import torch


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    raw = Tokenizer(models.BPE(unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="[UNK]",
        mask_token="[MASK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    tokenizer.add_special_tokens(
        {
            "unk_token": "[UNK]",
            "mask_token": "[MASK]",
            "pad_token": "[PAD]",
            "eos_token": "[EOS]",
        }
    )
    tokenizer.name_or_path = "local"
    return tokenizer


def _tiny_a2d_model(vocab_size=64, hidden=16, seed=7):
    from unturtle.models.conversion.a2d.tiny_a2d.modeling_llama import (
        TinyA2DLlamaConfig,
        TinyA2DLlamaLMHeadModel,
    )

    torch.manual_seed(seed)
    return TinyA2DLlamaLMHeadModel(
        TinyA2DLlamaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden,
            intermediate_size=hidden * 2,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    )


def _bert_model(vocab_size=64):
    """A model with no `sparse_output_projection` capability."""
    from transformers import BertConfig, BertForMaskedLM

    torch.manual_seed(7)
    return BertForMaskedLM(
        BertConfig(
            vocab_size=vocab_size,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
        )
    )


def _noised_batch(batch_size=2, seq_len=6, vocab_size=64, prompt_len=2, seed=0):
    """A pre-noised batch, so both paths see identical inputs."""
    torch.manual_seed(seed)
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    diffusion_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    diffusion_mask[0, prompt_len] = True
    diffusion_mask[0, seq_len - 1] = True
    diffusion_mask[1, prompt_len + 1] = True
    return {
        "input_ids": input_ids,
        "labels": labels,
        "diffusion_mask": diffusion_mask,
        "timesteps": torch.full((batch_size,), 0.5),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }


def _trainer(model, tmp_path, **arg_overrides):
    from unturtle.diffusion import DiffusionTrainer, DiffusionTrainingArguments

    tokenizer = _tokenizer()
    args = DiffusionTrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        max_steps=1,
        use_cpu=True,
        bf16=False,
        fp16=False,
        remove_unused_columns=False,
        report_to=[],
        **arg_overrides,
    )
    return DiffusionTrainer(
        model=model,
        args=args,
        train_dataset=[{"input_ids": [5, 6, 7]}],
        processing_class=tokenizer,
    )


WEIGHTINGS = ["uniform", "timestep", "scheduler", "cart"]


class TestLossParity:
    @pytest.mark.parametrize("weighting", WEIGHTINGS)
    @pytest.mark.parametrize("right_shift", [False, True])
    def test_sparse_trainer_loss_matches_dense(self, weighting, right_shift, tmp_path):
        """Every weighting, with and without the Dream shift.

        `right_shift` is parametrized rather than tested once because the
        sparse path implements it on hidden states while the dense path
        implements it on logits — two different tensors, so parity in one
        configuration is no evidence for the other.
        """
        batch = _noised_batch()

        dense = _trainer(
            _tiny_a2d_model(),
            tmp_path / "dense",
            loss_weight_type=weighting,
            right_shift_logits=right_shift,
            sparse_lm_head=False,
        )
        sparse = _trainer(
            _tiny_a2d_model(),
            tmp_path / "sparse",
            loss_weight_type=weighting,
            right_shift_logits=right_shift,
            sparse_lm_head=True,
        )
        dense.model.eval()
        sparse.model.eval()

        with torch.no_grad():
            dense_loss = dense.compute_loss(dense.model, dict(batch))
            sparse_loss = sparse.compute_loss(sparse.model, dict(batch))

        assert torch.allclose(sparse_loss, dense_loss, atol=1e-6), (
            f"{weighting} right_shift={right_shift}: "
            f"dense={dense_loss.item():.8f} sparse={sparse_loss.item():.8f}"
        )

    @pytest.mark.parametrize("norm", ["token", "sequence", "batch"])
    def test_parity_holds_for_every_normalization(self, norm, tmp_path):
        batch = _noised_batch()

        dense = _trainer(
            _tiny_a2d_model(), tmp_path / "d", loss_norm_type=norm, sparse_lm_head=False
        )
        sparse = _trainer(
            _tiny_a2d_model(), tmp_path / "s", loss_norm_type=norm, sparse_lm_head=True
        )
        dense.model.eval()
        sparse.model.eval()

        with torch.no_grad():
            dense_loss = dense.compute_loss(dense.model, dict(batch))
            sparse_loss = sparse.compute_loss(sparse.model, dict(batch))

        assert torch.allclose(sparse_loss, dense_loss, atol=1e-6), (
            f"{norm}: dense={dense_loss.item():.8f} sparse={sparse_loss.item():.8f}"
        )


class TestGradientParity:
    @pytest.mark.parametrize("right_shift", [False, True])
    def test_gradients_match_the_dense_trainer(self, right_shift, tmp_path):
        batch = _noised_batch()

        dense = _trainer(
            _tiny_a2d_model(),
            tmp_path / "dense",
            right_shift_logits=right_shift,
            sparse_lm_head=False,
        )
        sparse = _trainer(
            _tiny_a2d_model(),
            tmp_path / "sparse",
            right_shift_logits=right_shift,
            sparse_lm_head=True,
        )

        dense.compute_loss(dense.model, dict(batch)).backward()
        sparse.compute_loss(sparse.model, dict(batch)).backward()

        reference = dict(dense.model.named_parameters())
        compared = 0
        for name, param in sparse.model.named_parameters():
            expected = reference[name].grad
            if expected is None and param.grad is None:
                continue
            assert param.grad is not None, f"{name}: sparse grad is None"
            assert expected is not None, f"{name}: dense grad is None"
            assert torch.allclose(param.grad, expected, atol=1e-6), (
                f"{name}: max |diff| = {(param.grad - expected).abs().max().item():.3e}"
            )
            compared += 1

        assert compared > 0, "no gradients were compared"


class TestItActuallyTakesTheSparsePath:
    def test_no_full_vocab_logits_are_materialized(self, tmp_path):
        """The point of the flag. Parity alone cannot show the path was taken.

        Watches every `[B, L, V]`-shaped allocation via a `__torch_function__`
        mode: a dense run must produce at least one, a sparse run none.
        """
        from torch.overrides import TorchFunctionMode

        batch = _noised_batch()
        B, L = batch["labels"].shape
        V = 64

        class SpotFullVocabTensors(TorchFunctionMode):
            def __init__(self):
                self.seen = 0

            def __torch_function__(self, func, types, args=(), kwargs=None):
                result = func(*args, **(kwargs or {}))
                if isinstance(result, torch.Tensor) and tuple(result.shape) == (
                    B,
                    L,
                    V,
                ):
                    self.seen += 1
                return result

        for sparse_flag, expectation in ((False, "at least one"), (True, "none")):
            trainer = _trainer(
                _tiny_a2d_model(vocab_size=V),
                tmp_path / f"m{sparse_flag}",
                sparse_lm_head=sparse_flag,
            )
            trainer.model.eval()
            spotter = SpotFullVocabTensors()
            with torch.no_grad(), spotter:
                trainer.compute_loss(trainer.model, dict(batch))

            if sparse_flag:
                assert spotter.seen == 0, (
                    f"sparse path materialized {spotter.seen} [B, L, V] tensors; "
                    "expected none"
                )
            else:
                assert spotter.seen > 0, (
                    "dense path materialized no [B, L, V] tensor, so this probe "
                    f"cannot detect the difference (expected {expectation})"
                )

    def test_return_outputs_falls_back_to_dense(self, tmp_path):
        """`return_outputs` needs the model outputs, which the sparse path skips.

        Returning a loss without outputs would break the caller worse than
        being slower, so the flag yields to it.
        """
        trainer = _trainer(_tiny_a2d_model(), tmp_path, sparse_lm_head=True)
        trainer.model.eval()

        with torch.no_grad():
            result = trainer.compute_loss(
                trainer.model, dict(_noised_batch()), return_outputs=True
            )

        assert isinstance(result, tuple), "return_outputs must still yield a tuple"
        loss, outputs = result
        assert torch.isfinite(loss)
        assert hasattr(outputs, "logits"), "outputs must carry logits"


class TestPackedBatches:
    """Packed batches are this PR's real risk surface.

    The sparse path calls the backbone directly instead of going through the
    LM-head wrapper, so it is the `forward_kwargs` filter — not the loss maths
    — that decides whether `block_attention_mask` / `packed_seq_lengths` still
    reach attention.  Losing them does not raise: attention silently stops
    being blocked at sample boundaries, samples contaminate each other, and
    the loss still decreases (CLAUDE.md flags exactly this).  Parity on a
    packed batch is the cheap way to keep a future refactor of that filter
    honest.
    """

    @pytest.mark.parametrize("weighting", ["uniform", "timestep", "cart"])
    def test_packed_parity(self, weighting, tmp_path):
        """`cart` matters most here — it is the only weighting reading
        packed structure (`seq_lengths`), so it exercises the metadata path
        end to end rather than only the attention kwargs."""
        from unturtle.diffusion.packed_collator import (
            PackedMaskedDiffusionDataCollator,
        )

        tokenizer = _tokenizer()
        collator = PackedMaskedDiffusionDataCollator(
            tokenizer=tokenizer,
            max_seq_length=16,
            mask_token_id=tokenizer.mask_token_id,
            completion_only=False,
            noise=False,
        )
        clean = collator([{"input_ids": [5, 6, 7]}, {"input_ids": [8, 9]}])

        dense = _trainer(
            _tiny_a2d_model(),
            tmp_path / "dense",
            loss_weight_type=weighting,
            sparse_lm_head=False,
        )
        sparse = _trainer(
            _tiny_a2d_model(),
            tmp_path / "sparse",
            loss_weight_type=weighting,
            sparse_lm_head=True,
        )
        dense.model.eval()
        sparse.model.eval()

        # Noise ONCE and hand both trainers the same corrupted batch.  A clean
        # packed batch carries no `diffusion_mask`, so letting each trainer
        # noise it inside `compute_loss` gives them different random masks and
        # the comparison measures the RNG, not the two code paths.
        noised = dense._apply_forward_process(dict(clean))

        with torch.no_grad():
            dense_loss = dense.compute_loss(dense.model, dict(noised))
            sparse_loss = sparse.compute_loss(sparse.model, dict(noised))

        assert torch.isfinite(dense_loss), f"{weighting}: dense loss {dense_loss}"
        assert torch.allclose(sparse_loss, dense_loss, atol=1e-6), (
            f"packed {weighting}: dense={dense_loss.item():.8f} "
            f"sparse={sparse_loss.item():.8f}"
        )

    def test_packed_attention_metadata_reaches_the_backbone(self, tmp_path):
        """Pins the kwargs themselves, not just that the losses agree.

        Loss parity would survive both paths dropping the metadata together;
        this asserts the packed keys actually arrive.
        """
        from unturtle.diffusion.packed_collator import (
            PackedMaskedDiffusionDataCollator,
        )

        tokenizer = _tokenizer()
        collator = PackedMaskedDiffusionDataCollator(
            tokenizer=tokenizer,
            max_seq_length=16,
            mask_token_id=tokenizer.mask_token_id,
            completion_only=False,
            noise=False,
        )
        clean = collator([{"input_ids": [5, 6, 7]}, {"input_ids": [8, 9]}])

        trainer = _trainer(_tiny_a2d_model(), tmp_path, sparse_lm_head=True)
        trainer.model.eval()

        backbone = trainer.model.get_decoder()
        original = backbone.forward
        seen: dict[str, object] = {}

        def capture(*args, **kwargs):
            seen.update(kwargs)
            return original(*args, **kwargs)

        backbone.forward = capture
        try:
            with torch.no_grad():
                trainer.compute_loss(trainer.model, dict(clean))
        finally:
            backbone.forward = original

        for key in ("block_attention_mask", "packed_seq_lengths"):
            assert seen.get(key) is not None, (
                f"the sparse path did not pass {key!r} to the backbone; "
                "attention would not be blocked at packed-sample boundaries"
            )


class TestRejectsUnsupportedConfigurations:
    def test_unsupported_model_raises_at_construction(self, tmp_path):
        """Fail loudly rather than fall back: a silent no-op is invisible."""
        with pytest.raises(ValueError, match="sparse_output_projection"):
            _trainer(_bert_model(), tmp_path, sparse_lm_head=True)

    def test_unsupported_model_is_fine_with_the_flag_off(self, tmp_path):
        """The guard must only fire for callers who asked for the sparse path."""
        assert _trainer(_bert_model(), tmp_path, sparse_lm_head=False) is not None


class TestDefaultIsUnchanged:
    def test_flag_defaults_off(self):
        from unturtle.diffusion import DiffusionTrainingArguments

        field = DiffusionTrainingArguments.__dataclass_fields__["sparse_lm_head"]
        assert field.default is False, (
            "sparse_lm_head must default off: it costs ~10% peak memory at the "
            "~50% mask ratio MDLM training actually produces"
        )
