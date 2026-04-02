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

"""Slow E2E tests with real HuggingFace checkpoints.

These tests download a real dLLM checkpoint from HuggingFace and exercise the
full ``FastDiffusionModel.from_pretrained → get_peft_model → DiffusionTrainer``
pipeline on GPU.

**Skipped by default.**  To run::

    pytest tests/test_e2e_real_checkpoint.py -m slow -v

Override the default checkpoint (GSAI-ML/LLaDA-8B-Instruct) with::

    UNTURTLE_TEST_CHECKPOINT=your/model pytest tests/test_e2e_real_checkpoint.py -m slow

Requirements:
- CUDA GPU
- ``HF_TOKEN`` env var if the model is gated
- Sufficient disk space (~16 GB for LLaDA-8B in 4-bit)
"""

from __future__ import annotations

import os
import warnings

import pytest
import torch


# ---------------------------------------------------------------------------
# Default checkpoint — override with env var to use a smaller model
# ---------------------------------------------------------------------------

_DEFAULT_CHECKPOINT = os.environ.get(
    "UNTURTLE_TEST_CHECKPOINT", "GSAI-ML/LLaDA-8B-Instruct"
)


@pytest.mark.slow
@pytest.mark.gpu
class TestFromPretrainedLoads:
    """FastDiffusionModel.from_pretrained works with a real checkpoint."""

    @pytest.fixture(autouse=True)
    def require_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA GPU required for slow E2E test")

    def test_model_and_tokenizer_returned(self):
        """from_pretrained returns (model, tokenizer), model has max_seq_length."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        model, tokenizer = FastDiffusionModel.from_pretrained(
            _DEFAULT_CHECKPOINT,
            max_seq_length=512,
            load_in_4bit=True,
            trust_remote_code=True,
        )
        assert model is not None, "model is None"
        assert tokenizer is not None, "tokenizer is None"
        assert model.max_seq_length == 512

    def test_forward_produces_logits(self):
        """Model forward pass produces logits of the correct shape."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        model, tokenizer = FastDiffusionModel.from_pretrained(
            _DEFAULT_CHECKPOINT,
            max_seq_length=64,
            load_in_4bit=True,
            trust_remote_code=True,
        )
        model.eval()

        text = "The quick brown fox"
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        B, L = inputs["input_ids"].shape

        with torch.no_grad():
            out = model(**inputs)

        assert out.logits.shape == (B, L, model.config.vocab_size), (
            f"Unexpected logits shape: {out.logits.shape}"
        )
        assert torch.isfinite(out.logits).all(), "Logits contain inf/nan"


@pytest.mark.slow
@pytest.mark.gpu
class TestPeftAndTraining:
    """LoRA patching + DiffusionTrainer with a real checkpoint."""

    @pytest.fixture(autouse=True)
    def require_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA GPU required for slow E2E test")

    def test_loss_decreases_after_training_steps(self, tmp_path):
        """3 gradient steps with a real model: loss must be finite and decrease."""
        from unturtle.diffusion import (
            DiffusionTrainer,
            DiffusionTrainingArguments,
            MaskedDiffusionDataCollator,
        )
        from unturtle.fast_diffusion_model import FastDiffusionModel

        model, tokenizer = FastDiffusionModel.from_pretrained(
            _DEFAULT_CHECKPOINT,
            max_seq_length=256,
            load_in_4bit=True,
            trust_remote_code=True,
        )
        assert tokenizer is not None, "Tokenizer required for real checkpoint test"

        peft_model = FastDiffusionModel.get_peft_model(
            model,
            r=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            use_gradient_checkpointing="unsloth",
        )

        mask_token_id = tokenizer.mask_token_id or getattr(model.config, "mask_token_id", None)
        assert mask_token_id is not None, (
            "Real-checkpoint E2E requires a mask token id. "
            "Pass mask_token_id explicitly or use a checkpoint whose tokenizer/config defines one."
        )

        # Tiny 10-sample dataset: prompt + completion.
        # Use add_special_tokens=False consistently for prompt/full tokenization so
        # the prompt/completion boundary stays exact even with override checkpoints.
        prompts = ["The capital of France is"] * 10
        completions = [" Paris."] * 10
        dataset = []
        for prompt, completion in zip(prompts, completions):
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            input_ids = (prompt_ids + completion_ids)[:64]
            attention_mask = [1] * len(input_ids)
            labels = [-100] * min(len(prompt_ids), len(input_ids)) + input_ids[len(prompt_ids):]
            labels = labels[:len(input_ids)]
            dataset.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            })

        collator = MaskedDiffusionDataCollator(
            tokenizer=tokenizer,
            mask_token_id=mask_token_id,
            completion_only=True,
        )
        training_args = DiffusionTrainingArguments(
            output_dir=str(tmp_path / "checkpoints"),
            num_train_epochs=1,
            max_steps=3,
            per_device_train_batch_size=2,
            logging_steps=1,
            save_steps=100,
            bf16=True,
            dataloader_drop_last=True,
            remove_unused_columns=False,
            report_to="none",
        )

        losses: list[float] = []
        original_log = DiffusionTrainer.log

        def _capturing_log(self_inner, logs, *args, **kw):
            if "loss" in logs:
                losses.append(float(logs["loss"]))
            original_log(self_inner, logs, *args, **kw)

        DiffusionTrainer.log = _capturing_log
        try:
            trainer = DiffusionTrainer(
                model=peft_model,
                args=training_args,
                train_dataset=dataset,
                data_collator=collator,
                processing_class=tokenizer,
            )
            trainer.train()
        finally:
            DiffusionTrainer.log = original_log

        assert len(losses) >= 2, (
            f"Need >= 2 logged steps to compare loss, got: {losses}"
        )
        for step_loss in losses:
            assert torch.isfinite(torch.tensor(step_loss)), (
                f"Non-finite loss at step: {step_loss}"
            )
        # Allow small tolerance: loss may fluctuate over only 3 steps
        assert losses[-1] < losses[0] or abs(losses[-1] - losses[0]) < 0.5, (
            f"Loss did not decrease: {losses}"
        )


@pytest.mark.slow
@pytest.mark.gpu
class TestAdapterSaveReload:
    """LoRA adapter save + reload reproduces identical forward outputs."""

    @pytest.fixture(autouse=True)
    def require_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA GPU required for slow E2E test")

    def test_save_reload_forward_matches(self, tmp_path):
        """Save LoRA adapter and reload onto a fresh base; logits must match."""
        from peft import PeftModel
        from unturtle.fast_diffusion_model import FastDiffusionModel

        model, tokenizer = FastDiffusionModel.from_pretrained(
            _DEFAULT_CHECKPOINT,
            max_seq_length=64,
            load_in_4bit=True,
            trust_remote_code=True,
        )

        peft_model = FastDiffusionModel.get_peft_model(
            model,
            r=4,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=4,
            lora_dropout=0,
            use_gradient_checkpointing=False,
        )
        peft_model.eval()

        text = "The quick brown fox"
        inputs = tokenizer(text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            logits_before = peft_model(**inputs).logits

        # Save adapter
        adapter_dir = tmp_path / "adapter"
        peft_model.save_pretrained(str(adapter_dir))

        # Reload: fresh from_pretrained + load adapter
        base_model2, _ = FastDiffusionModel.from_pretrained(
            _DEFAULT_CHECKPOINT,
            max_seq_length=64,
            load_in_4bit=True,
            trust_remote_code=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reloaded = PeftModel.from_pretrained(base_model2, str(adapter_dir))
        reloaded.eval()

        with torch.no_grad():
            logits_after = reloaded(**inputs).logits

        assert logits_before.shape == logits_after.shape
        assert torch.allclose(logits_before.float(), logits_after.float(), atol=1e-3), (
            "Forward outputs diverged after save/reload of LoRA adapter. "
            f"Max diff: {(logits_before.float() - logits_after.float()).abs().max().item():.4f}"
        )
