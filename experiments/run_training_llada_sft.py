"""
Headless training script — mirrors the logic of llada_sft_demo.py.

Usage:
    source .venv/bin/activate
    nohup python examples/training/run_training.py > logs/llada_sft.log 2>&1 &

Checkpoints saved to: outputs/llada_sft_demo/
"""

import json
import logging
import os
import sys
import time

import peft  # noqa: F401
import torch

# transformers/trl/peft must be imported BEFORE unturtle
import transformers
import trl  # noqa: F401

# Force transformers trainer to print loss to stdout (not swallowed by tqdm)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
transformers.logging.set_verbosity_info()
transformers.logging.enable_default_handler()
transformers.logging.enable_explicit_format()

from datasets import load_dataset

import unturtle  # noqa: F401 (triggers unsloth patching)
from unturtle import FastDiffusionModel
from unturtle.diffusion import DiffusionTrainer, DiffusionTrainingArguments

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CFG = {
    "model_name": "GSAI-ML/LLaDA-8B-Instruct",
    "output_dir": "outputs/llada_sft_demo",
    "train_samples": 2000,
    "eval_samples": 200,
    "num_epochs": 2,
    "batch_size": 4,
    "lora_r": 16,
    "lora_alpha": 32,
    "lr": 1e-5,
    "max_length": 512,
}
print(json.dumps(CFG, indent=2), flush=True)

# ---------------------------------------------------------------------------
# Model & tokenizer
# ---------------------------------------------------------------------------
print(f"\n[1/5] Loading model: {CFG['model_name']} …", flush=True)
model, tokenizer = FastDiffusionModel.from_pretrained(
    CFG["model_name"],
    dtype=torch.bfloat16,
    load_in_4bit=False,
)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Loaded: {n_params / 1e9:.1f}B params", flush=True)

# mask_token_id: tokenizer may not have it, fall back to model.config
# Note: this is derived by DiffusionTrainer from processing_class=tokenizer,
# so we only print it here for debug visibility.
mask_token_id = tokenizer.mask_token_id or getattr(model.config, "mask_token_id", None)
print(f"  mask_token_id: {mask_token_id}", flush=True)

# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
print("\n[2/5] Applying LoRA …", flush=True)
model = FastDiffusionModel.get_peft_model(
    model,
    r=CFG["lora_r"],
    lora_alpha=CFG["lora_alpha"],
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0,
    bias="none",
)
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(
    f"  Trainable: {n_trainable / 1e6:.1f}M / {n_params / 1e9:.1f}B "
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

eos = tokenizer.eos_token or "<|endoftext|>"


def preprocess(example):
    msgs = example["messages"]
    prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
    completion_msgs = [m for m in msgs if m["role"] == "assistant"][-1:]
    enc = tokenizer.apply_chat_template(
        prompt_msgs,
        tokenize=True,
        add_generation_prompt=True,
    )
    prompt_ids = (
        list(enc["input_ids"])
        if hasattr(enc, "__getitem__") and not isinstance(enc, list)
        else list(enc)
    )
    completion_text = completion_msgs[0]["content"] if completion_msgs else ""
    completion_ids = tokenizer(completion_text + eos, add_special_tokens=False)[
        "input_ids"
    ]
    input_ids = (prompt_ids + completion_ids)[: CFG["max_length"]]
    labels = ([-100] * len(prompt_ids) + completion_ids)[: CFG["max_length"]]
    return {"input_ids": input_ids, "labels": labels}


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
t0 = time.time()
result = trainer.train()
elapsed = time.time() - t0

print(f"\n{'=' * 60}", flush=True)
print("Training complete!", flush=True)
print(f"  train_loss : {result.training_loss:.4f}", flush=True)
print(f"  steps      : {result.global_step}", flush=True)
print(f"  elapsed    : {elapsed / 60:.1f} min", flush=True)
print(f"  checkpoint : {CFG['output_dir']}", flush=True)

# Save loss history
history = [e for e in trainer.state.log_history if "loss" in e]
with open(os.path.join(CFG["output_dir"], "loss_history.json"), "w") as f:
    json.dump(history, f, indent=2)
print(f"  loss log   : {CFG['output_dir']}/loss_history.json", flush=True)
