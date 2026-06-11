"""
Headless benchmark script for Qwen3-0.6B-diffusion dLLM SFT with dllm (reference).

REQUIRES: a separate venv with dllm installed (dllm depends on transformers 4.x).
Setup:
    uv venv .venvDllm --python 3.12
    .venvDllm/bin/python -m pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124
    .venvDllm/bin/python -m pip install -e dev/repos/dllm/

Usage (from project root):
    .venvDllm/bin/python benchmarks/qwen3/benchmark_dllm.py

Metrics saved to: outputs/benchmark_qwen3_dllm/benchmark.json
Loss history: outputs/benchmark_qwen3_dllm/loss_history.json
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
from dllm.utils import NoAttentionMaskWrapper
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)

# Silence excessive logging
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Config — matched to ModernBERT benchmark for fair comparison
# ---------------------------------------------------------------------------
CFG = {
    "model_name": "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
    "output_dir": "outputs/benchmark_qwen3_dllm",
    "train_samples": 2000,
    "eval_samples": 200,
    "num_epochs": 2,
    "batch_size": 4,  # same as ModernBERT (not 16)
    "lora_r": 16,
    "lora_alpha": 32,
    "lr": 1e-4,  # Qwen3-specific (10x ModernBERT, per dllm examples/a2d/README.md)
    "max_length": 512,  # same as ModernBERT (not 1024)
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
tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"], trust_remote_code=True)
tokenizer.padding_side = "right"  # required by MDLMTrainer

model = AutoModelForMaskedLM.from_pretrained(
    CFG["model_name"],
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
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
    """Flatten messages to text; Qwen3 tokenizer has chat_template but we use flat for consistency."""
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
# Filter out samples with no completion (all labels are -100)
dataset = dataset.filter(
    lambda x: len(x["input_ids"]) > 0 and sum(1 for _l in x["labels"] if _l != -100) > 0
)
print(f"  train: {len(dataset['train'])}  /  test: {len(dataset['test'])}", flush=True)


# Use dllm's reference data collator (from examples/a2d/mdlm/sft.py)
data_collator = NoAttentionMaskWrapper(  # removes attention_mask so dllm uses full bidirectional attention
    DataCollatorForSeq2Seq(
        tokenizer,
        return_tensors="pt",
        padding=True,
        label_pad_token_id=-100,  # exclude padding from maskable_mask (labels != -100); loss normalization aligned with unturtle
    )
)


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
    gradient_accumulation_steps=1,  # explicitly set to 1 to match unturtle; dllm may default to 3 in some configurations
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
    data_collator=data_collator,
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

results_dir = CFG["output_dir"]
os.makedirs(results_dir, exist_ok=True)

# Save loss history (normalized schema: step + loss/eval_loss)
loss_history = [{"step": (i + 1) * 10, "loss": l} for i, l in enumerate(losses)]
loss_path = os.path.join(results_dir, "loss_history.json")
with open(loss_path, "w") as f:
    json.dump(loss_history, f, indent=2)
print(f"  loss log   : {loss_path}", flush=True)

# Save benchmark summary
all_losses = [e for e in losses]
total_samples = len(dataset["train"]) * CFG["num_epochs"]
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
    "samples_per_second": total_samples / elapsed if elapsed > 0 else 0,
    "gradient_accumulation_steps": mdlm_args.gradient_accumulation_steps,
    "n_params_total": n_params,
    "n_params_trainable": n_trainable,
    "epochs": CFG["num_epochs"],
    "batch_size": CFG["batch_size"],
    "lora_r": CFG["lora_r"],
    "loss_weight_type": CFG["loss_weight_type"],
}
benchmark_path = os.path.join(results_dir, "benchmark.json")
with open(benchmark_path, "w") as f:
    json.dump(benchmark, f, indent=2)
print(f"  benchmark  : {benchmark_path}", flush=True)
