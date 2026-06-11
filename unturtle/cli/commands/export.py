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

"""unturtle export / list-checkpoints — checkpoint management commands."""

import json
from pathlib import Path
from typing import Optional

import typer

EXPORT_FORMATS = ["merged-16bit", "lora", "gguf"]
GGUF_QUANTS = ["q4_k_m", "q5_k_m", "q8_0", "f16"]


def list_checkpoints(
    outputs_dir: Path = typer.Option(
        Path("./outputs"),
        "--outputs-dir",
        help="Directory that holds training runs.",
    ),
):
    """List checkpoints detected in the outputs directory."""
    outputs_path = Path(outputs_dir)
    if not outputs_path.exists():
        typer.echo("No checkpoints found.")
        raise typer.Exit()

    # Group checkpoint dirs by their parent
    checkpoints: dict[Path, list[tuple[str, Path, Optional[float], str | None]]] = {}
    for ckpt_dir in sorted(outputs_path.rglob("checkpoint-*")):
        if not ckpt_dir.is_dir():
            continue
        parent = ckpt_dir.parent

        loss: Optional[float] = None
        status: str | None = "no trainer_state.json"
        state_file = ckpt_dir / "trainer_state.json"
        if state_file.exists():
            status = None
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
                log_history = state.get("log_history", [])
                # Find the last entry that has a loss or eval_loss value
                for entry in reversed(log_history):
                    if "loss" in entry:
                        loss = float(entry["loss"])
                        break
                    if "eval_loss" in entry:
                        loss = float(entry["eval_loss"])
                        break
                if loss is None:
                    status = "trainer_state.json has no loss"
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                status = "unreadable trainer_state.json"
                typer.echo(
                    f"Warning: failed to read loss from {state_file}: {exc}",
                    err=True,
                )

        checkpoints.setdefault(parent, []).append(
            (ckpt_dir.name, ckpt_dir, loss, status)
        )

    if not checkpoints:
        typer.echo("No checkpoints found.")
        raise typer.Exit()

    for parent, ckpt_list in sorted(checkpoints.items()):
        typer.echo(f"\n{parent}:")
        for name, path, loss, status in ckpt_list:
            if loss is not None:
                typer.echo(f"  {name:<30} loss={loss:.4f}  {path}")
            else:
                typer.echo(f"  {name:<30} ({status})  {path}")


def export(
    checkpoint: Path = typer.Argument(..., help="Path to checkpoint directory."),
    output_dir: Path = typer.Argument(..., help="Directory to save exported model."),
    format: str = typer.Option(
        "merged-16bit",
        "--format",
        "-f",
        help=f"Export format: {', '.join(EXPORT_FORMATS)}.",
    ),
    quantization: str = typer.Option(
        "q4_k_m",
        "--quantization",
        "-q",
        help=f"GGUF quantization method (gguf format only): {', '.join(GGUF_QUANTS)}.",
    ),
    push_to_hub: bool = typer.Option(
        False, "--push-to-hub", help="Push exported model to HuggingFace Hub."
    ),
    repo_id: Optional[str] = typer.Option(
        None, "--repo-id", help="HuggingFace repo ID (username/model-name)."
    ),
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", envvar="HF_TOKEN", help="HuggingFace token."
    ),
    private: bool = typer.Option(
        False, "--private", help="Make the HuggingFace repo private."
    ),
    max_seq_length: int = typer.Option(2048, "--max-seq-length"),
    load_in_4bit: bool = typer.Option(True, "--load-in-4bit/--no-load-in-4bit"),
):
    """Export a checkpoint to various formats (merged-16bit, lora, gguf)."""
    if format not in EXPORT_FORMATS:
        typer.echo(
            f"Error: Invalid format '{format}'. "
            f"Choose from: {', '.join(EXPORT_FORMATS)}",
            err=True,
        )
        raise typer.Exit(code=2)

    if format == "gguf" and quantization not in GGUF_QUANTS:
        typer.echo(
            f"Error: Invalid quantization '{quantization}'. "
            f"Choose from: {', '.join(GGUF_QUANTS)}",
            err=True,
        )
        raise typer.Exit(code=2)

    if push_to_hub and not repo_id:
        typer.echo("Error: --repo-id is required when using --push-to-hub", err=True)
        raise typer.Exit(code=2)

    try:
        from unturtle import FastDiffusionModel
    except ImportError as e:
        typer.echo(f"Error: failed to import unturtle — {e}", err=True)
        raise typer.Exit(code=1)

    # --- Load checkpoint ---
    typer.echo(f"Loading checkpoint: {checkpoint}")
    try:
        model, tokenizer = FastDiffusionModel.from_pretrained(
            model_name=str(checkpoint),
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            token=hf_token,
        )
    except Exception as e:
        typer.echo(f"Error: checkpoint load failed — {e}", err=True)
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Export ---
    typer.echo(f"Exporting as {format} to {output_dir} ...")
    try:
        if format == "merged-16bit":
            if push_to_hub:
                if repo_id is None:
                    raise typer.Exit(code=2)
                FastDiffusionModel.push_to_hub_merged(
                    model,
                    repo_id,
                    tokenizer,
                    token=hf_token,
                    private=private,
                )
                typer.echo(f"Pushed merged model to Hub: {repo_id}")
            else:
                FastDiffusionModel.save_pretrained_merged(
                    model, str(output_dir), tokenizer
                )
                typer.echo(f"Saved merged model to {output_dir}")

        elif format == "lora":
            FastDiffusionModel.save_lora_adapter(model, str(output_dir), tokenizer)
            if push_to_hub and repo_id:
                model.push_to_hub(repo_id, token=hf_token, private=private)
                tokenizer.push_to_hub(repo_id, token=hf_token, private=private)
                typer.echo(f"Pushed LoRA adapter to Hub: {repo_id}")
            else:
                typer.echo(f"Saved LoRA adapter to {output_dir}")

        elif format == "gguf":
            FastDiffusionModel.save_pretrained_gguf(
                model,
                str(output_dir),
                tokenizer,
                quantization_method=quantization,
            )
            if push_to_hub and repo_id:
                from huggingface_hub import HfApi

                api = HfApi(token=hf_token)
                api.create_repo(
                    repo_id=repo_id,
                    private=private,
                    exist_ok=True,
                )
                api.upload_folder(
                    folder_path=str(output_dir),
                    repo_id=repo_id,
                    token=hf_token,
                )
                typer.echo(f"Pushed GGUF to Hub: {repo_id}")
            else:
                typer.echo(f"Saved GGUF to {output_dir}")

    except Exception as e:
        typer.echo(f"Error: export failed — {e}", err=True)
        raise typer.Exit(code=1)
