"""
Headless benchmark script for ModernBERT-base dLLM SFT with unturtle.

Usage:
    nohup uv run python benchmarks/modernbert/benchmark_unturtle.py > logs/benchmark_modernbert_unturtle.log 2>&1 &

Metrics saved to: outputs/benchmark_modernbert_unturtle/benchmark.json
Loss history: outputs/benchmark_modernbert_unturtle/loss_history.json
"""

import json
import os
import random
import time

import numpy as np
import peft  # noqa: F401
import torch
import transformers
import trl  # noqa: F401
from datasets import load_dataset

import unturtle  # noqa: F401
from unturtle import FastDiffusionModel
from unturtle.diffusion import DiffusionTrainer, DiffusionTrainingArguments
from unturtle.models.backbones.modernbert import DiffusionModernBertForMaskedLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CFG = {
    "model_name": "answerdotai/ModernBERT-base",
    "output_dir": "outputs/benchmark_modernbert_unturtle",
    "train_samples": 2000,
    "eval_samples": 200,
    "num_epochs": 2,
    "batch_size": 4,
    "lora_r": 16,
    "lora_alpha": 32,
    "lr": 1e-5,
    "max_length": 512,
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
print(f"\n[1/5] Loading model: {CFG['model_name']} …", flush=True)

# ModernBERT auto-detect doesn't work (model_type="modernbert"), so we pass
# the diffusion class explicitly.
model, tokenizer = FastDiffusionModel.from_pretrained(
    CFG["model_name"],
    model_class=DiffusionModernBertForMaskedLM,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Loaded: {n_params / 1e9:.3f}B params", flush=True)

mask_token_id = tokenizer.mask_token_id or getattr(model.config, "mask_token_id", None)
print(f"  mask_token_id: {mask_token_id}", flush=True)

# DiffusionModernBertForMaskedLM overrides model_type to "modernbert-diffusion",
# but the upstream config loaded from the hub retains "modernbert".
# Overwrite so patch_peft_model recognizes it.
model.config.model_type = "modernbert-diffusion"

# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
print("\n[2/5] Applying LoRA …", flush=True)

# ModernBERT uses fused Wqkv and Wo for attention (not split q/k/v/o_proj).
# MLP modules (Wi, Wii, Wo) can also be targeted but won't get Triton kernels
# yet — see issue #59 Phase 2.
model = FastDiffusionModel.get_peft_model(
    model,
    r=CFG["lora_r"],
    lora_alpha=CFG["lora_alpha"],
    target_modules=["Wqkv", "Wo"],
    lora_dropout=0,
    bias="none",
)
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(
    f"  Trainable: {n_trainable / 1e6:.1f}M / {n_params / 1e9:.3f}B "
    f"({n_trainable / n_params * 100:.2f}%)",
    flush=True,
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
print("\n[3/5] Loading dataset …", flush=True)

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
            return {"input_ids": input_ids, "labels": labels}
    # Fallback: no completion
    return {"input_ids": [], "labels": []}


dataset = raw.map(
    preprocess, remove_columns=raw["train"].column_names, desc="Tokenising"
)
dataset = dataset.filter(lambda x: sum(1 for l in x["labels"] if l != -100) > 0)
print(f"  train: {len(dataset['train'])}  /  test: {len(dataset['test'])}", flush=True)

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
print("\n[4/5] Setting up trainer …", flush=True)
os.makedirs(CFG["output_dir"], exist_ok=True)

training_args = DiffusionTrainingArguments(
    output_dir=CFG["output_dir"],
    num_train_epochs=CFG["num_epochs"],
    per_device_train_batch_size=CFG["batch_size"],
    per_device_eval_batch_size=CFG["batch_size"],
    learning_rate=CFG["lr"],
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=10,
    disable_tqdm=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    report_to="none",
    loss_weight_type="uniform",
    dataloader_num_workers=0,
)

trainer = DiffusionTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
print("\n[5/5] Training …", flush=True)

torch.cuda.reset_peak_memory_stats()

t0 = time.time()
result = trainer.train()
elapsed = time.time() - t0

peak_vram = torch.cuda.max_memory_allocated() / 1e9
print(f"\n{'=' * 60}", flush=True)
print("Training complete!", flush=True)
print(f"  train_loss : {result.training_loss:.4f}", flush=True)
print(f"  steps      : {result.global_step}", flush=True)
print(f"  elapsed    : {elapsed / 60:.1f} min", flush=True)
print(f"  peak VRAM  : {peak_vram:.2f} GB", flush=True)
print(f"  checkpoint : {CFG['output_dir']}", flush=True)

# Save loss history (normalized schema: step + loss/eval_loss)
loss_history = [
    {"step": int(e.get("step", 0)), "loss": e["loss"]}
    for e in trainer.state.log_history
    if "loss" in e and "eval_" not in e
]
loss_history.extend(
    [
        {"step": int(e.get("step", 0)), "eval_loss": e["eval_loss"]}
        for e in trainer.state.log_history
        if "eval_loss" in e
    ]
)
loss_history.sort(key=lambda x: x["step"])
with open(os.path.join(CFG["output_dir"], "loss_history.json"), "w") as f:
    json.dump(loss_history, f, indent=2)
print(f"  loss log   : {CFG['output_dir']}/loss_history.json", flush=True)

# Save benchmark summary
losses = [e["loss"] for e in loss_history if "loss" in e]
benchmark = {
    "framework": "unturtle",
    "model": CFG["model_name"],
    "train_loss_avg": sum(losses) / len(losses) if losses else None,
    "train_loss_first50": sum(losses[:50]) / min(50, len(losses)) if losses else None,
    "train_loss_last50": sum(losses[-50:]) / min(50, len(losses)) if losses else None,
    "elapsed_seconds": elapsed,
    "peak_vram_gb": peak_vram,
    "steps": result.global_step,
    "steps_per_second": result.global_step / elapsed if elapsed > 0 else 0,
    "n_params_total": n_params,
    "n_params_trainable": n_trainable,
    "epochs": CFG["num_epochs"],
    "batch_size": CFG["batch_size"],
    "lora_r": CFG["lora_r"],
    "loss_weight_type": training_args.loss_weight_type,
}
with open(os.path.join(CFG["output_dir"], "benchmark.json"), "w") as f:
    json.dump(benchmark, f, indent=2)
print(f"  benchmark  : {CFG['output_dir']}/benchmark.json", flush=True)
