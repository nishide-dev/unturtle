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

"""unturtle generate — iterative denoising inference for dLLMs.

Unlike autoregressive LMs, dLLMs denoise a fully-masked sequence over
multiple steps.  There is no token-by-token streaming; the output is the
full decoded sequence after all denoising steps complete.
"""

from typing import Optional

import typer

ALG_CHOICES = ["origin", "maskgit_plus", "topk_margin", "entropy"]


def generate(
    model: str = typer.Argument(..., help="HuggingFace model ID or local path."),
    prompt: str = typer.Argument(..., help="Text prompt to condition generation on."),
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", envvar="HF_TOKEN", help="HuggingFace token."
    ),
    max_seq_length: int = typer.Option(2048, "--max-seq-length"),
    load_in_4bit: bool = typer.Option(
        True, "--load-in-4bit/--no-load-in-4bit", help="Load model in 4-bit."
    ),
    num_steps: int = typer.Option(
        128, "--num-steps", help="Number of denoising steps."
    ),
    mask_token_id: Optional[int] = typer.Option(
        None,
        "--mask-token-id",
        help="Mask token ID override. Resolved from tokenizer if not set.",
    ),
    temperature: float = typer.Option(
        0.0, "--temperature", help="Sampling temperature (0 = greedy)."
    ),
    top_p: Optional[float] = typer.Option(
        None, "--top-p", help="Top-p (nucleus) sampling probability."
    ),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="Top-k sampling cutoff."),
    alg: str = typer.Option(
        "origin",
        "--alg",
        help=f"Denoising sampling algorithm: {', '.join(ALG_CHOICES)}.",
    ),
    alg_temp: Optional[float] = typer.Option(
        None, "--alg-temp", help="Temperature for the sampling algorithm."
    ),
    max_new_tokens: int = typer.Option(
        256, "--max-new-tokens", help="Maximum number of new tokens to generate."
    ),
    use_cache: bool = typer.Option(
        False,
        "--use-cache/--no-use-cache",
        help="Enable KV-cache (block-decode fast path).",
    ),
    block_length: Optional[int] = typer.Option(
        None, "--block-length", help="Block length for block-decode cache mode."
    ),
    output_history: bool = typer.Option(
        False,
        "--output-history/--no-output-history",
        help="Return per-step denoising history.",
    ),
    show_steps: bool = typer.Option(
        False,
        "--show-steps/--no-show-steps",
        help="Print each denoising step (requires --output-history).",
    ),
):
    """Run dLLM iterative denoising generation on a prompt."""
    if alg not in ALG_CHOICES:
        typer.echo(f"Error: --alg must be one of {ALG_CHOICES}, got '{alg}'", err=True)
        raise typer.Exit(code=2)

    try:
        from unturtle import FastDiffusionModel
        from unturtle.models.generation.diffusion_generation_utils import (
            MaskedDiffusionGenerationConfig,
        )
    except ImportError as e:
        typer.echo(f"Error: failed to import unturtle — {e}", err=True)
        raise typer.Exit(code=1)

    # --- Load model ---
    typer.echo(f"Loading model: {model}", err=True)
    try:
        loaded_model, tokenizer = FastDiffusionModel.from_pretrained(
            model_name=model,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            token=hf_token,
        )
    except Exception as e:
        typer.echo(f"Error: model load failed — {e}", err=True)
        raise typer.Exit(code=1)

    FastDiffusionModel.for_inference(loaded_model)

    # --- Resolve mask token ID ---
    resolved_mask_id = mask_token_id
    if resolved_mask_id is None:
        resolved_mask_id = getattr(tokenizer, "mask_token_id", None)
    if resolved_mask_id is None:
        resolved_mask_id = getattr(loaded_model.config, "mask_token_id", None)
    if resolved_mask_id is None:
        typer.echo(
            "Error: could not resolve mask_token_id. "
            "Provide --mask-token-id or use a tokenizer/model with mask_token_id set.",
            err=True,
        )
        raise typer.Exit(code=2)

    # --- Tokenize ---
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    model_device = next(loaded_model.parameters()).device
    input_ids = input_ids.to(model_device)

    # --- Build generation config ---
    gen_config_kwargs = dict(
        steps=num_steps,
        mask_token_id=resolved_mask_id,
        temperature=temperature,
        alg=alg,
        use_cache=use_cache,
        output_history=output_history,
        max_new_tokens=max_new_tokens,
    )
    if top_p is not None:
        gen_config_kwargs["top_p"] = top_p
    if top_k is not None:
        gen_config_kwargs["top_k"] = top_k
    if alg_temp is not None:
        gen_config_kwargs["alg_temp"] = alg_temp
    if block_length is not None:
        gen_config_kwargs["block_length"] = block_length

    try:
        gen_config = MaskedDiffusionGenerationConfig(**gen_config_kwargs)
    except Exception as e:
        typer.echo(f"Error: invalid generation config — {e}", err=True)
        raise typer.Exit(code=1)

    # --- Generate ---
    typer.echo("Generating...", err=True)
    try:
        output = loaded_model.generate(input_ids, generation_config=gen_config)
    except Exception as e:
        typer.echo(f"Error: generation failed — {e}", err=True)
        raise typer.Exit(code=1)

    # Extract token IDs: output may be a tensor or a model output object
    out_ids = output.sequences if hasattr(output, "sequences") else output

    # Decode only the generated portion (beyond the prompt)
    prompt_len = input_ids.shape[-1]
    gen_ids = out_ids[0, prompt_len:]
    result_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # --- Show per-step trace if requested ---
    if show_steps and output_history and hasattr(output, "history"):
        typer.echo("\n--- Denoising trace ---", err=True)
        for step_idx, step_ids in enumerate(output.history):
            step_tokens = step_ids[0] if step_ids.dim() > 1 else step_ids
            decoded_tokens = []
            for tok_id in step_tokens[prompt_len:].tolist():
                if tok_id == resolved_mask_id:
                    decoded_tokens.append("_")
                else:
                    decoded_tokens.append(
                        tokenizer.decode([tok_id], skip_special_tokens=True)
                    )
            typer.echo(f"Step {step_idx + 1:3d}: {''.join(decoded_tokens)}", err=True)
        typer.echo("--- End trace ---", err=True)

    typer.echo(result_text)
