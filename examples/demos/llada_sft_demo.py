# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "altair",
#     "polars",
#     "torch",
#     "transformers",
#     "peft",
#     "trl",
#     "datasets",
#     "bitsandbytes",
# ]
# ///
"""
LLaDA-8B SFT demo notebook using unturtle DiffusionTrainer.

Dataset : allenai/tulu-3-sft-mixture (train[:2000] / test[:200])
Model   : GSAI-ML/LLaDA-8B-Instruct
Training: DiffusionTrainer + LoRA r=16, 2 epochs, bf16

Run interactively:
    marimo edit examples/demos/llada_sft_demo.py

Run headless training:
    nohup python examples/training/run_training.py > logs/llada_sft.log 2>&1 &
"""

import marimo

__generated_with = "0.22.4"
app = marimo.App(width="full")


# ---------------------------------------------------------------------------
# Cell 1: imports & environment check
# NOTE: trl / transformers / peft must be imported BEFORE unturtle.
#       All imports in a single cell to control order.
# ---------------------------------------------------------------------------
@app.cell
def _():
    import json
    import os
    import sys
    import time
    from pathlib import Path

    import marimo as mo
    import peft
    import torch

    # Import order: trl/transformers/peft FIRST, then unturtle
    import transformers
    import trl

    import unturtle

    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
    vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        if cuda_ok
        else 0
    )

    mo.md(f"""
    ## Environment

    | | |
    |---|---|
    | CUDA | {"✅ " + gpu_name + f" ({vram_gb} GB)" if cuda_ok else "❌ CPU only"} |
    | torch | `{torch.__version__}` |
    | transformers | `{transformers.__version__}` |
    | peft | `{peft.__version__}` |
    | trl | `{trl.__version__}` |
    | unturtle | `{unturtle.__version__}` |
    """)
    return Path, cuda_ok, gpu_name, json, mo, os, peft, sys, time, torch, transformers, trl, unturtle, vram_gb


# ---------------------------------------------------------------------------
# Cell 3: hyperparameter controls (UI)
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    train_samples = mo.ui.slider(200, 4000, value=2000, step=200, label="Train samples")
    eval_samples  = mo.ui.slider(50,  500,  value=200,  step=50,  label="Eval samples")
    num_epochs    = mo.ui.slider(1,   5,    value=2,    step=1,   label="Epochs")
    batch_size    = mo.ui.slider(1,   16,   value=4,    step=1,   label="Batch size / device")
    lora_r        = mo.ui.slider(4,   64,   value=16,   step=4,   label="LoRA rank r")
    lr_exp        = mo.ui.slider(-6,  -3,   value=-5,   step=1,   label="log10(lr)")

    mo.md("## Hyperparameters"), train_samples, eval_samples, num_epochs, batch_size, lora_r, lr_exp
    return batch_size, eval_samples, lora_r, lr_exp, num_epochs, train_samples


# ---------------------------------------------------------------------------
# Cell 4: config dict (derived from UI)
# ---------------------------------------------------------------------------
@app.cell
def _(batch_size, eval_samples, lora_r, lr_exp, mo, num_epochs, train_samples):
    CFG = {
        "model_name":     "GSAI-ML/LLaDA-8B-Instruct",
        "output_dir":     "outputs/llada_sft_demo",
        "train_samples":  train_samples.value,
        "eval_samples":   eval_samples.value,
        "num_epochs":     num_epochs.value,
        "batch_size":     batch_size.value,
        "lora_r":         lora_r.value,
        "lora_alpha":     lora_r.value * 2,
        "lr":             10 ** lr_exp.value,
        "max_length":     512,
    }
    mo.md(f"```json\n{__import__('json').dumps(CFG, indent=2)}\n```")
    return (CFG,)


# ---------------------------------------------------------------------------
# Cell 5: load model & tokenizer via FastDiffusionModel
# ---------------------------------------------------------------------------
@app.cell
def _(CFG, mo, torch):
    from unturtle import FastDiffusionModel

    mo.md("## Loading model…")

    _model, _tokenizer = FastDiffusionModel.from_pretrained(
        CFG["model_name"],
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    model_base = _model
    tokenizer  = _tokenizer

    mo.md(f"Model loaded: `{CFG['model_name']}` — {sum(p.numel() for p in model_base.parameters()) / 1e9:.1f}B params")
    return FastDiffusionModel, model_base, tokenizer


# ---------------------------------------------------------------------------
# Cell 6: apply LoRA
# ---------------------------------------------------------------------------
@app.cell
def _(CFG, FastDiffusionModel, mo, model_base):
    model = FastDiffusionModel.get_peft_model(
        model_base,
        r=CFG["lora_r"],
        lora_alpha=CFG["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())

    mo.md(f"""
    ## LoRA applied

    - Rank r = {CFG['lora_r']}, alpha = {CFG['lora_alpha']}
    - Trainable: **{n_trainable / 1e6:.1f}M** / {n_total / 1e9:.1f}B ({n_trainable / n_total * 100:.2f}%)
    """)
    return (model,)


# ---------------------------------------------------------------------------
# Cell 7: load & preprocess dataset
# ---------------------------------------------------------------------------
@app.cell
def _(CFG, mo, tokenizer):
    from datasets import load_dataset

    mo.md("## Loading dataset…")

    raw = load_dataset(
        "allenai/tulu-3-sft-mixture",
        split={
            "train": f"train[:{CFG['train_samples']}]",
            "test":  f"train[10000:{10000 + CFG['eval_samples']}]",
        },
    )

    # LLaDA tokenizer: mask_token_id is in model config, not tokenizer
    _eos = tokenizer.eos_token or "<|endoftext|>"

    def _preprocess(example):
        msgs = example["messages"]
        # Split: non-assistant turns → prompt, last assistant → completion
        prompt_msgs     = [m for m in msgs if m["role"] != "assistant"]
        completion_msgs = [m for m in msgs if m["role"] == "assistant"][-1:]

        # apply_chat_template returns BatchEncoding when tokenize=True
        _enc = tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=True,
            add_generation_prompt=True,
        )
        prompt_ids = list(_enc["input_ids"]) if hasattr(_enc, "__getitem__") and not isinstance(_enc, list) else list(_enc)

        completion_text = completion_msgs[0]["content"] if completion_msgs else ""
        completion_ids  = tokenizer(
            completion_text + _eos,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = prompt_ids + completion_ids
        labels    = [-100] * len(prompt_ids) + completion_ids

        # truncate
        max_len   = CFG["max_length"]
        input_ids = input_ids[:max_len]
        labels    = labels[:max_len]

        return {"input_ids": input_ids, "labels": labels}

    dataset = raw.map(
        _preprocess,
        remove_columns=raw["train"].column_names,
        desc="Tokenising",
    )
    # filter sequences that are too short (no completion tokens)
    dataset = dataset.filter(lambda x: sum(1 for l in x["labels"] if l != -100) > 0)

    mo.md(f"""
    ## Dataset

    | split | samples |
    |-------|---------|
    | train | {len(dataset['train'])} |
    | test  | {len(dataset['test'])} |

    Sample (first 10 input token ids): `{dataset['train'][0]['input_ids'][:10]}`
    """)
    return dataset, load_dataset


# ---------------------------------------------------------------------------
# Cell 8: training arguments & trainer
# ---------------------------------------------------------------------------
@app.cell
def _(CFG, dataset, mo, model, tokenizer):
    from unturtle.diffusion import DiffusionTrainer, DiffusionTrainingArguments

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
        eval_strategy="epoch",
        save_strategy="epoch",
        disable_tqdm=True,
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

    mo.md(f"""
    ## Trainer ready

    - Output dir: `{CFG['output_dir']}`
    - Epochs: {CFG['num_epochs']} / Batch: {CFG['batch_size']} / LR: {CFG['lr']:.0e}
    - Loss weight: uniform / mask_prob: {CFG['mask_prob']}
    """)
    return DiffusionTrainer, DiffusionTrainingArguments, trainer, training_args


# ---------------------------------------------------------------------------
# Cell 9: run training (blocking — suits headless execution)
# ---------------------------------------------------------------------------
@app.cell
def _(mo, time, trainer):
    mo.md("## Training — running…")

    _t0     = time.time()
    _result = trainer.train()
    _elapsed = time.time() - _t0

    train_loss = _result.training_loss

    mo.md(f"""
    ## Training complete ✅

    | | |
    |---|---|
    | Train loss | `{train_loss:.4f}` |
    | Elapsed    | `{_elapsed/60:.1f} min` |
    | Steps      | `{_result.global_step}` |
    """)
    return (train_loss,)


# ---------------------------------------------------------------------------
# Cell 10: loss curve visualisation
# ---------------------------------------------------------------------------
@app.cell
def _(mo, trainer):
    import altair as alt
    import polars as pl

    _log = trainer.state.log_history
    _rows = [
        {"step": e["step"], "loss": e["loss"]}
        for e in _log
        if "loss" in e
    ]

    if _rows:
        _df = pl.DataFrame(_rows)
        _chart = (
            alt.Chart(_df.to_pandas())
            .mark_line(point=True)
            .encode(
                x=alt.X("step:Q", title="Step"),
                y=alt.Y("loss:Q", title="Train loss", scale=alt.Scale(zero=False)),
            )
            .properties(title="Training loss curve", width=700, height=300)
        )
        mo.md("## Loss curve"), _chart
    else:
        mo.md("No loss history recorded yet.")
    return alt, pl


# ---------------------------------------------------------------------------
# Cell 11: inference demo (before/after)
# ---------------------------------------------------------------------------
@app.cell
def _(CFG, mo, trainer):
    import torch as _torch

    mo.md("## Inference demo")

    _tok    = trainer.processing_class
    _prompt = "Please explain what a diffusion language model is in one sentence."
    _msgs   = [{"role": "user", "content": _prompt}]
    _enc    = _tok.apply_chat_template(_msgs, tokenize=True, add_generation_prompt=True)
    _input_ids = _enc["input_ids"] if hasattr(_enc, "__getitem__") and not isinstance(_enc, list) else _enc
    _ids    = _torch.tensor([_input_ids])

    _model  = trainer.model
    _device = next(_model.parameters()).device
    _ids    = _ids.to(_device)

    _model.eval()
    with _torch.no_grad():
        _out = _model.generate(
            _ids,
            max_new_tokens=128,
            do_sample=False,
        ) if hasattr(_model, "generate") else None

    if _out is not None:
        _gen_text = _tok.decode(_out[0][_ids.shape[1]:], skip_special_tokens=True)
    else:
        _gen_text = "(generate() not available on this model)"

    mo.md(f"""
    **Prompt**: {_prompt}

    **Response**: {_gen_text}
    """)
    return (torch,)


# ---------------------------------------------------------------------------
# Entry point
# - Interactive:  marimo edit examples/demos/llada_sft_demo.py
# - Headless:     python examples/training/run_training.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run()
