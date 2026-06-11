"""
Headless benchmark script for ModernBERT-base dLLM SFT with dllm (reference).

REQUIRES: a separate venv with dllm installed (dllm depends on transformers 4.x).
Setup:
    uv venv .venvDllm --python 3.12
    .venvDllm/bin/python -m pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124
    .venvDllm/bin/python -m pip install -e dev/repos/dllm/

Usage (from project root):
    .venvDllm/bin/python benchmarks/modernbert/benchmark_dllm.py

Metrics saved to: outputs/benchmark_modernbert_dllm/benchmark.json
Loss history: outputs/benchmark_modernbert_dllm/loss_history.json
"""

import json
import logging
import os
import random
import time

import numpy as np
import torch
from datasets import load_dataset
from dllm.core.trainers import MDLMConfig, MDLMTrainer
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    ModernBertForMaskedLM,
)

# Silence excessive logging
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Config — matched to unturtle benchmark for apples-to-apples comparison
# ---------------------------------------------------------------------------
CFG = {
    "model_name": "answerdotai/ModernBERT-base",
    "output_dir": "outputs/benchmark_modernbert_dllm",
    "train_samples": 2000,
    "eval_samples": 200,
    "num_epochs": 2,
    "batch_size": 4,
    "lora_r": 16,
    "lora_alpha": 32,
    "lr": 1e-5,
    "max_length": 512,
    "loss_weight_type": "uniform",
    "seed": 42,
}
print(json.dumps(CFG, indent=2), flush=True)

# ---------------------------------------------------------------------------
# Seed for reproducibility
# ---------------------------------------------------------------------------
random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG["seed"])

# ---------------------------------------------------------------------------
# Model & tokenizer
# ---------------------------------------------------------------------------
print(f"\n[1/4] Loading model: {CFG['model_name']} …", flush=True)
tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])
tokenizer.padding_side = "right"  # required by MDLMTrainer

model = ModernBertForMaskedLM.from_pretrained(
    CFG["model_name"],
    torch_dtype=torch.bfloat16,
)
model.cuda()
model.train()
n_params = sum(p.numel() for p in model.parameters())
print(f"  Loaded: {n_params / 1e9:.3f}B params", flush=True)

# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
print("\n[2/4] Applying LoRA …", flush=True)
lora_config = LoraConfig(
    r=CFG["lora_r"],
    lora_alpha=CFG["lora_alpha"],
    target_modules=["Wqkv", "Wo"],
    lora_dropout=0,
    bias="none",
)
model = get_peft_model(model, lora_config)
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(
    f"  Trainable: {n_trainable / 1e6:.1f}M / {n_params / 1e9:.3f}B "
    f"({n_trainable / n_params * 100:.2f}%)",
    flush=True,
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
print("\n[3/4] Loading dataset …", flush=True)

raw = load_dataset(
    "allenai/tulu-3-sft-mixture",
    split={
        "train": f"train[:{CFG['train_samples']}]",
        "test": f"train[10000:{10000 + CFG['eval_samples']}]",
    },
)

eos = tokenizer.eos_token or ""


def preprocess(example):
    """Flatten messages to text; ModernBERT tokenizer lacks chat_template."""
    msgs = example["messages"]
    prompt_parts = []
    for m in msgs:
        if m["role"] != "assistant":
            prompt_parts.append(f"<|{m['role']}|>\n{m['content']}")
        else:
            # Last assistant message = completion
            prompt_text = "\n\n".join(prompt_parts) + "\n\n<|assistant|>\n"
            completion_text = m["content"] + eos
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(completion_text, add_special_tokens=False)[
                "input_ids"
            ]
            input_ids = (prompt_ids + completion_ids)[: CFG["max_length"]]
            labels = ([-100] * len(prompt_ids) + completion_ids)[: CFG["max_length"]]
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": [1] * len(input_ids),
            }
    # Fallback: no completion
    return {"input_ids": [], "labels": [], "attention_mask": []}


dataset = raw.map(
    preprocess, remove_columns=raw["train"].column_names, desc="Tokenising"
)
dataset = dataset.filter(lambda x: sum(1 for _l in x["labels"] if _l != -100) > 0)
print(f"  train: {len(dataset['train'])}  /  test: {len(dataset['test'])}", flush=True)


def data_collator_fn(features):
    """Simple collator for SDLM: pads and stacks input_ids/labels/attention_mask."""
    input_ids = [f["input_ids"] for f in features]
    labels = [f["labels"] for f in features]
    attention_mask = [f["attention_mask"] for f in features]

    max_len = max(len(x) for x in input_ids)
    pad_id = tokenizer.pad_token_id or 0

    batch_input_ids = []
    batch_labels = []
    batch_attention_mask = []

    for ids, lbs, mask in zip(input_ids, labels, attention_mask, strict=False):
        pad_len = max_len - len(ids)
        batch_input_ids.append(ids + [pad_id] * pad_len)
        batch_labels.append(lbs + [-100] * pad_len)
        batch_attention_mask.append(mask + [0] * pad_len)

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Trainer (dllm MDLMTrainer)
# ---------------------------------------------------------------------------
print("\n[4/4] Training …", flush=True)
os.makedirs(CFG["output_dir"], exist_ok=True)

# MDLMConfig from dllm (inherits from transformers TrainingArguments)
mdlm_args = MDLMConfig(
    output_dir=CFG["output_dir"],
    num_train_epochs=CFG["num_epochs"],
    per_device_train_batch_size=CFG["batch_size"],
    per_device_eval_batch_size=CFG["batch_size"],
    learning_rate=CFG["lr"],
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    loss_weight_type=CFG["loss_weight_type"],
    loss_norm_type="token",
    right_shift_logits=False,  # disable to match unturtle behavior
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    report_to="none",
    bf16=True,
    dataloader_num_workers=0,
)

# Collect metrics
losses = []
original_log = getattr(MDLMTrainer, "log", None)


def capturing_log(self_inner, logs, start_time=None, **kw):
    if "loss" in logs:
        losses.append(float(logs["loss"]))
    if original_log:
        original_log(self_inner, logs, start_time=start_time, **kw)


torch.cuda.reset_peak_memory_stats()

MDLMTrainer.log = capturing_log
t0 = time.time()

trainer = MDLMTrainer(
    args=mdlm_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator_fn,
    processing_class=tokenizer,
)
result = trainer.train()

elapsed = time.time() - t0
MDLMTrainer.log = original_log

peak_vram = torch.cuda.max_memory_allocated() / 1e9

print(f"\n{'=' * 60}", flush=True)
print("Training complete!", flush=True)
print(f"  train_loss : {result.training_loss:.4f}", flush=True)
print(f"  steps      : {result.global_step}", flush=True)
print(f"  elapsed    : {elapsed / 60:.1f} min", flush=True)
print(f"  peak VRAM  : {peak_vram:.2f} GB", flush=True)
print(f"  checkpoint : {CFG['output_dir']}", flush=True)

# Save loss history (normalized schema: step + loss/eval_loss)
loss_history = [{"step": (i + 1) * 10, "loss": l} for i, l in enumerate(losses)]
with open(os.path.join(CFG["output_dir"], "loss_history.json"), "w") as f:
    json.dump(loss_history, f, indent=2)
print(f"  loss log   : {CFG['output_dir']}/loss_history.json", flush=True)

# Save benchmark summary
all_losses = [e for e in losses]
benchmark = {
    "framework": "dllm",
    "model": CFG["model_name"],
    "train_loss_avg": sum(all_losses) / len(all_losses) if all_losses else None,
    "train_loss_first50": sum(all_losses[:50]) / min(50, len(all_losses))
    if all_losses
    else None,
    "train_loss_last50": sum(all_losses[-50:]) / min(50, len(all_losses))
    if all_losses
    else None,
    "elapsed_seconds": elapsed,
    "peak_vram_gb": peak_vram,
    "steps": result.global_step,
    "steps_per_second": result.global_step / elapsed if elapsed > 0 else 0,
    "n_params_total": n_params,
    "n_params_trainable": n_trainable,
    "epochs": CFG["num_epochs"],
    "batch_size": CFG["batch_size"],
    "lora_r": CFG["lora_r"],
    "loss_weight_type": CFG["loss_weight_type"],
}
with open(os.path.join(CFG["output_dir"], "benchmark.json"), "w") as f:
    json.dump(benchmark, f, indent=2)
print(f"  benchmark  : {CFG['output_dir']}/benchmark.json", flush=True)
