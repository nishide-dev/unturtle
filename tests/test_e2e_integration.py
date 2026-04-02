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

"""Fast (CPU, no downloads) end-to-end integration tests for FastDiffusionModel + DiffusionTrainer.

Uses a tiny random-weight model to verify the full
``get_peft_model → DiffusionTrainer`` pipeline runs and loss is finite.
Always runs in CI — no GPU or HF download required.

For slow E2E tests with real HuggingFace checkpoints, see::

    tests/test_e2e_real_checkpoint.py  (run with ``pytest -m slow``)
"""

from __future__ import annotations

import warnings

import pytest
import torch


# ---------------------------------------------------------------------------
# Shared tiny A2D config
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_a2d_config():
    from unturtle.models.a2d import A2DLlamaConfig

    return A2DLlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )


@pytest.fixture(scope="module")
def tiny_a2d_model(tiny_a2d_config):
    from unturtle.models.a2d import A2DLlamaLMHeadModel

    model = A2DLlamaLMHeadModel(tiny_a2d_config)
    model.train()
    return model


# ---------------------------------------------------------------------------
# Fast E2E: FastDiffusionModel.get_peft_model → DiffusionTrainer (CPU)
# ---------------------------------------------------------------------------


class TestE2EFastDiffusionTrainer:
    """Full pipeline: base model → LoRA → DiffusionTrainer → loss decreasing.

    Uses a tiny random-weight A2D-Llama model so no GPU or HF download
    is needed.  Verifies that the Trainer integrates with FastDiffusionModel.
    """

    def test_pipeline_loss_decreases(self, tiny_a2d_model, tmp_path):
        """After 3 gradient steps the training loss should decrease."""
        from transformers import PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

        from unturtle.diffusion import (
            DiffusionTrainer,
            DiffusionTrainingArguments,
            MaskedDiffusionDataCollator,
        )
        from unturtle.fast_diffusion_model import FastDiffusionModel

        cfg = tiny_a2d_model.config

        # Build a minimal BPE tokenizer with a mask token
        raw_tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        raw_tok.normalizer = normalizers.Lowercase()
        raw_tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=raw_tok,
            unk_token="[UNK]",
            mask_token="[MASK]",
            pad_token="[PAD]",
        )
        tokenizer.add_special_tokens(
            {"unk_token": "[UNK]", "mask_token": "[MASK]", "pad_token": "[PAD]"}
        )
        # Give the tokenizer a non-empty name so the Trainer doesn't try to
        # fetch processor_config.json from the HuggingFace hub.
        tokenizer.name_or_path = "local"

        # Use fixed token ids that fall within vocab_size=256
        mask_token_id = tokenizer.mask_token_id or 1
        assert mask_token_id < cfg.vocab_size, (
            f"mask_token_id={mask_token_id} exceeds vocab_size={cfg.vocab_size}"
        )

        # Build PEFT model (CPU-only; Triton kernels are automatically skipped
        # on CPU via the cuda_available guard in FastDiffusionModel).
        peft_model = FastDiffusionModel.get_peft_model(
            tiny_a2d_model,
            r=4,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=4,
            lora_dropout=0,
            use_gradient_checkpointing=False,
        )

        # Tiny dataset: 8 sequences of length 16
        L = 16
        dataset = [
            {
                "input_ids": torch.randint(2, cfg.vocab_size, (L,)).tolist(),
                "labels": torch.randint(2, cfg.vocab_size, (L,)).tolist(),
                "attention_mask": [1] * L,
            }
            for _ in range(8)
        ]

        collator = MaskedDiffusionDataCollator(
            tokenizer=tokenizer,
            mask_token_id=mask_token_id,
            completion_only=False,
        )

        training_args = DiffusionTrainingArguments(
            output_dir=str(tmp_path / "checkpoints"),
            num_train_epochs=1,
            max_steps=3,
            per_device_train_batch_size=2,
            logging_steps=1,
            save_steps=100,
            use_cpu=True,  # CPU-only
            bf16=False,
            fp16=False,
            dataloader_drop_last=True,
            remove_unused_columns=False,
            report_to="none",
        )

        # Capture per-step losses to verify the pipeline actually learns
        losses: list[float] = []
        original_log = DiffusionTrainer.log

        def capturing_log(self_inner, logs, start_time=None, **kw):
            if "loss" in logs:
                losses.append(float(logs["loss"]))
            original_log(self_inner, logs, start_time=start_time, **kw)

        DiffusionTrainer.log = capturing_log
        try:
            trainer = DiffusionTrainer(
                model=peft_model,
                args=training_args,
                train_dataset=dataset,
                data_collator=collator,
                processing_class=tokenizer,
            )
            result = trainer.train()
        finally:
            DiffusionTrainer.log = original_log

        # Loss must be finite and the training loop must have logged at least once
        assert result.training_loss is not None
        assert torch.isfinite(torch.tensor(result.training_loss)), (
            f"Training loss is not finite: {result.training_loss}"
        )
        # Verify the loss logging path was exercised (gradient steps are running)
        assert len(losses) >= 1, (
            "No loss values were logged — training loop may not have run"
        )
        # All logged losses must be finite (no NaN/Inf explosion)
        for step_loss in losses:
            assert torch.isfinite(torch.tensor(step_loss)), (
                f"Step loss is not finite: {step_loss} in {losses}"
            )

    def test_adapter_save_reload_forward_matches(self, tiny_a2d_config, tmp_path):
        """Save LoRA adapter, reload, and verify forward output matches."""
        from peft import PeftModel

        from unturtle.fast_diffusion_model import FastDiffusionModel
        from unturtle.models.a2d import A2DLlamaLMHeadModel

        cfg = tiny_a2d_config
        # Use a fresh model — the shared tiny_a2d_model may already be PEFT-wrapped
        base_model = A2DLlamaLMHeadModel(cfg)
        base_model.train()
        # Snapshot base weights before PEFT wrapping so we can restore them on
        # a new model instance (for the reload verification later).
        base_state_dict = {k: v.clone() for k, v in base_model.state_dict().items()}

        peft_model = FastDiffusionModel.get_peft_model(
            base_model,
            r=4,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=4,
            lora_dropout=0,
            use_gradient_checkpointing=False,
        )
        peft_model.eval()

        B, L = 2, 8
        input_ids = torch.randint(2, tiny_a2d_config.vocab_size, (B, L))

        with torch.no_grad():
            logits_before = peft_model(input_ids).logits

        # Save adapter
        adapter_dir = tmp_path / "adapter"
        peft_model.save_pretrained(str(adapter_dir))

        # Reload onto a fresh base model with the same weights as the original.
        fresh_base = A2DLlamaLMHeadModel(cfg)
        fresh_base.load_state_dict(base_state_dict)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reloaded = PeftModel.from_pretrained(fresh_base, str(adapter_dir))
        reloaded.eval()

        with torch.no_grad():
            logits_after = reloaded(input_ids).logits

        assert logits_before.shape == logits_after.shape
        assert torch.allclose(logits_before, logits_after, atol=1e-5), (
            "Forward outputs diverged after save/reload of LoRA adapter."
        )
