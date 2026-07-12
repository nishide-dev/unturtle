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

"""
DiffuGRPOTrainer – Diffu-GRPO RL trainer for masked diffusion language models.

Extends TRL's :class:`~trl.GRPOTrainer` with diffusion-specific generation
and per-token log-probability computation under random masking.

Algorithm (d1 paper, arXiv:2502.07574):
  1. Sample prompt + generate full completion via iterative denoising.
  2. For each GRPO iteration, apply a fresh random mask to the sequence.
  3. Compute conditional log-probs p_θ(x_0 | x_t) over masked positions.
  4. Optimise the clipped policy-gradient objective (PPO-style) with
     optional KL penalty against a reference model.

Reference implementation:
  dllm-reasoning/d1  diffu-grpo/diffu_grpo_trainer.py
"""

from __future__ import annotations

import warnings
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from accelerate.utils import broadcast_object_list, gather, gather_object, set_seed
from datasets import Dataset, IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
from transformers.utils import is_peft_available
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import (
    pad,
    split_pixel_values_by_grid,
    split_tensor_dict,
    unsplit_pixel_values_by_grid,
)

# print_prompt_completions_sample was added in TRL >=0.20; provide a fallback.
try:
    from trl.trainer.utils import (
        print_prompt_completions_sample as _print_completions_sample,
    )
except ImportError:

    def _print_completions_sample(prompts, completions, rewards, step):  # type: ignore[misc]
        pass


_GRPOConfigBase = GRPOConfig

# ---------------------------------------------------------------------------
# TRL version compatibility shim
#
# TRL >=0.14 has a bug where is_vllm_available() returns a non-empty tuple
# (False, None) which evaluates to True in an if-statement, causing an
# unconditional `from vllm import ...` that fails when vllm is not installed.
# We patch trl.import_utils before importing GRPOTrainer to work around this.
# ---------------------------------------------------------------------------
import trl.import_utils as _trl_import_utils  # noqa: E402

_orig_is_vllm_available = _trl_import_utils.is_vllm_available


def _patched_is_vllm_available() -> bool:
    result = _orig_is_vllm_available()
    if isinstance(result, tuple):
        return bool(result[0])
    return bool(result)


_trl_import_utils.is_vllm_available = _patched_is_vllm_available  # type: ignore[assignment]

# Patch the copy already bound in the grpo_trainer module namespace (if loaded)
import importlib  # noqa: E402
import sys  # noqa: E402

if "trl.trainer.grpo_trainer" not in sys.modules:
    _grpo_mod_spec = importlib.util.find_spec("trl.trainer.grpo_trainer")
    if _grpo_mod_spec is not None:
        # Pre-inject the patched function so grpo_trainer.py sees the fixed version
        import trl.trainer  # noqa: F401 — ensure parent package is loaded
        # The patch on _trl_import_utils is sufficient since grpo_trainer.py
        # accesses is_vllm_available via the trl.import_utils module reference.

from trl.trainer.grpo_trainer import GRPOTrainer  # noqa: E402

if is_peft_available():
    from peft import PeftConfig

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DiffuGRPOConfig(_GRPOConfigBase):
    """Training configuration for :class:`DiffuGRPOTrainer`.

    Inherits all standard GRPO / TrainingArguments fields and adds
    diffusion-generation-specific parameters.

    **d1-style advantages:** TRL sets ``scale_rewards`` to ``\"group\"`` by default
    (per-group std scaling). The d1 / dllm references use mean-centered rewards only.
    Pass ``scale_rewards=False`` or ``scale_rewards=\"none\"`` for that behavior.
    See ``docs/diffu-grpo-d1-notes.md``.

    Args:
        block_length:      Block size for block-wise iterative denoising.
        diffusion_steps:   Total denoising steps across all blocks.
        cfg_scale:         Classifier-free guidance scale (0 = disabled).
        remasking:         Token-selection strategy during generation.
                           ``"low_confidence"`` (default) or ``"random"``.
        p_mask_prompt:     Probability of masking *prompt* tokens when
                           computing policy log-probs (default 0.3).
        mask_id:           Vocabulary index of the ``[MASK]`` token.
                           Default 126336 matches LLaDA / LLaDA-2.
        random_masking:    If True, use a fresh random seed per GRPO
                           iteration; otherwise use a fixed seed (42).
        generation_batch_size: Batch size used during generation rollouts.
                               ``None`` lets TRL derive this from ``steps_per_generation``
                               or vice versa (mutually exclusive in TRL).
    """

    # --- generation ---
    block_length: int = field(
        default=64,
        metadata={"help": "Block size for block-wise iterative denoising."},
    )
    diffusion_steps: int = field(
        default=64,
        metadata={"help": "Total denoising steps across all blocks."},
    )
    cfg_scale: float = field(
        default=0.0,
        metadata={"help": "Classifier-free guidance scale (0 = disabled)."},
    )
    remasking: str = field(
        default="low_confidence",
        metadata={"help": "Token selection strategy: 'low_confidence' or 'random'."},
    )
    generation_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Batch size for generation rollouts (None: derived by TRL)."},
    )

    # --- diffusion-GRPO specific ---
    p_mask_prompt: float = field(
        default=0.3,
        metadata={
            "help": "Probability of masking prompt tokens during log-prob computation."
        },
    )
    mask_id: int = field(
        default=126336,
        metadata={"help": "Mask token id. Default matches LLaDA / LLaDA-2."},
    )
    random_masking: bool = field(
        default=True,
        metadata={"help": "Use fresh random mask seeds per GRPO iteration."},
    )
    diffu_policy_objective: str = field(
        default="grpo",
        metadata={
            "help": "Policy objective: 'grpo' (clipped policy ratio / d1-style), "
            "'wd1' (ratio-free weighted log-likelihood, wd1 Eq. 8–9), or "
            "'wd1++' (stepwise wd1, Eq. 10 — MC log-prob at intermediate denoise states)."
        },
    )
    wd1_psi: float = field(
        default=1.0,
        metadata={
            "help": "ψ in wd1 weights exp(±ψ Â) over group-relative advantages (Eq. 9)."
        },
    )

    def __post_init__(self) -> None:
        o = self.diffu_policy_objective
        if o not in ("grpo", "wd1", "wd1++"):
            raise ValueError(
                f"diffu_policy_objective must be 'grpo', 'wd1', or 'wd1++', got {o!r}"
            )
        super().__post_init__()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class DiffuGRPOTrainer(GRPOTrainer):
    """Diffu-GRPO trainer for masked diffusion language models.

    Overrides three core methods of :class:`~trl.GRPOTrainer`:

    * :meth:`generate` – block-wise iterative denoising for dLLMs.
    * :meth:`_get_per_token_logps` – masked log-prob under forward process.
    * :meth:`compute_loss` – clipped policy-gradient + optional KL, **wd1**
      ratio-free weighted log-likelihood, or **wd1++** (Eq. 10 MC at stored trajectories).

    All other GRPO bookkeeping (reward scoring, advantage normalisation,
    buffered rollout reuse) is inherited unchanged from TRL. Use
    ``scale_rewards=False`` or ``\"none\"`` for d1-style mean-only advantages
    (see ``docs/diffu-grpo-d1-notes.md``).

    Args:
        model:                    Policy model or model-id string.
        reward_funcs:             One or more reward callables / model ids.
        args:                     :class:`DiffuGRPOConfig` instance.
        train_dataset / eval_dataset / processing_class / callbacks /
        optimizers / peft_config: Forwarded to :class:`~trl.GRPOTrainer`.

    Example::

        from unturtle.diffusion import DiffuGRPOTrainer, DiffuGRPOConfig

        def correctness_reward(prompts, completions, **kw):
            return [1.0 if "correct" in c else 0.0 for c in completions]

        cfg = DiffuGRPOConfig(
            output_dir="output",
            num_train_epochs=1,
            mask_id=126336,          # LLaDA mask token
            diffusion_steps=64,
            block_length=64,
        )
        trainer = DiffuGRPOTrainer(
            model="GSAI-ML/LLaDA-8B-Instruct",
            reward_funcs=correctness_reward,
            args=cfg,
            train_dataset=dataset,
        )
        trainer.train()
    """

    # Narrow the type of self.args so attribute access is resolved correctly.
    args: DiffuGRPOConfig  # type: ignore[assignment]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[DiffuGRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer],
            Optional[torch.optim.lr_scheduler.LambdaLR],
        ] = (None, None),
        peft_config: Optional[PeftConfig] = None,
    ) -> None:
        # TRL's GRPOTrainer sets ``model.warnings_issued["estimate_tokens"]``; some in-memory
        # :class:`~transformers.PreTrainedModel` instances (e.g. fresh configs) omit this dict.
        if not isinstance(model, str) and not hasattr(model, "warnings_issued"):
            model.warnings_issued = {}  # type: ignore[attr-defined]
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        if getattr(self.args, "diffu_policy_objective", "grpo") == "wd1++":
            warnings.warn(
                "diffu_policy_objective='wd1++' stores full denoise trajectories per rollout "
                "and runs extra forward passes; see docs/wd1plusplus-design.md.",
                UserWarning,
                stacklevel=2,
            )
        # Per-rollout mask seeds (length ``num_iterations``).  Kept on the trainer because
        # :func:`split_tensor_dict` slices list values by batch dimension, which would
        # empty ``mask_seeds`` when ``len(mask_seeds) != batch_size`` (see dllm/d1).
        self._diffu_mask_seeds: list[int] = []
        # wd1++: merged list of (x_l, x0_pred) per inner denoise step, batch dim = rollout batch.
        # Not passed through ``split_tensor_dict`` (same pattern as ``_diffu_mask_seeds``).
        self._wd1pp_trajectory: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None

    # ------------------------------------------------------------------
    # TRL input buffering (no shuffle — keep rows aligned with cached log-probs)
    # ------------------------------------------------------------------

    def _prepare_inputs(self, generation_batch: dict[str, Any]) -> dict[str, Any]:
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                generation_batch = self._generate_and_score_completions(
                    generation_batch
                )
                # ``old/ref_per_token_logps`` are ``[num_iterations, B_gen, Lc]`` —
                # iterations on dim 0, batch on dim 1. TRL's ``split_tensor_dict``
                # slices every tensor along dim 0 (the batch dim for everything
                # else), so pop them and re-attach the correct dim-1 slice per
                # micro-batch below. They are ``[]`` when unused.
                old_logps = generation_batch.pop("old_per_token_logps", [])
                ref_logps = generation_batch.pop("ref_per_token_logps", [])
                # Per-rollout artifacts that must NOT go through ``split_tensor_dict``
                # (it slices list values along the batch dim, emptying them for later
                # chunks). Attach the full objects to every buffered chunk instead, so
                # an eval rollout landing mid-cycle cannot clobber them on the trainer
                # (see d1, which stores ``mask_seeds`` in the inputs dict).
                rollout_mask_seeds = generation_batch.pop("mask_seeds", None)
                rollout_wd1pp_traj = generation_batch.pop("wd1pp_trajectory", None)
                total_rows = generation_batch["prompt_ids"].shape[0]
                chunk_size = total_rows // self.args.steps_per_generation
                generation_batch = split_pixel_values_by_grid(generation_batch)
                generation_batches = split_tensor_dict(
                    generation_batch, self.args.steps_per_generation
                )
                buffered = []
                for i, batch in enumerate(generation_batches):
                    batch = unsplit_pixel_values_by_grid(batch)
                    row_offset = i * chunk_size
                    batch["old_per_token_logps"] = (
                        old_logps[:, row_offset : row_offset + chunk_size]
                        if isinstance(old_logps, torch.Tensor)
                        else old_logps
                    )
                    batch["ref_per_token_logps"] = (
                        ref_logps[:, row_offset : row_offset + chunk_size]
                        if isinstance(ref_logps, torch.Tensor)
                        else ref_logps
                    )
                    # Row window of this micro-batch within the rollout batch, so
                    # train-time re-masking can reproduce the generation-time
                    # ``torch.rand((B_gen, L))`` draw (see ``_forward_process``).
                    batch["buffer_row_offset"] = row_offset
                    batch["buffer_total_rows"] = total_rows
                    if rollout_mask_seeds is not None:
                        batch["mask_seeds"] = rollout_mask_seeds
                    if rollout_wd1pp_traj is not None:
                        batch["wd1pp_trajectory"] = rollout_wd1pp_traj
                    buffered.append(batch)
                self._buffered_inputs = buffered
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            # Matches TRL ``GRPOTrainer._prepare_inputs`` (one increment per micro-batch).
            self._step += 1
        else:
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    # ------------------------------------------------------------------
    # Diffusion generation
    # ------------------------------------------------------------------

    @staticmethod
    def _add_gumbel_noise(
        logits: torch.Tensor,
        temperature: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Gumbel-max sampling: ``logits + temperature * gumbel`` (i.i.d. per element).

        The d1 reference ``add_gumbel_noise`` uses a different reparameterization of
        stochastic argmax; paths agree when ``temperature == 0`` (deterministic argmax).
        See ``docs/diffu-grpo-d1-notes.md``.

        Per arXiv:2409.02908, float64 noise improves perplexity but reduces
        generation quality for MDMs. We keep it in the model's native dtype.
        """
        if temperature == 0.0:
            return logits
        logits = logits.to(dtype)
        gumbel = -torch.log(-torch.log(torch.rand_like(logits, dtype=dtype)))
        return logits + temperature * gumbel

    @staticmethod
    def _get_num_transfer_tokens(
        mask_index: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        """Precompute how many tokens to unmask at each denoising step.

        Distributes the total masked token count as evenly as possible
        across ``steps``, with remainder allocated to the first steps.

        Args:
            mask_index: Bool tensor ``[B, L_block]`` — True where masked.
            steps:      Number of denoising steps for this block.

        Returns:
            Int64 tensor ``[B, steps]``.
        """
        mask_num = mask_index.sum(dim=1, keepdim=True)  # [B, 1]
        base = mask_num // steps
        remainder = mask_num % steps
        num_transfer = base.expand(-1, steps).clone()
        indices = torch.arange(steps, device=mask_index.device)
        num_transfer[indices.unsqueeze(0) < remainder] += 1
        return num_transfer.to(torch.int64)

    def _get_logits(
        self,
        model: torch.nn.Module,
        batch: torch.Tensor,
        prompt_index: torch.Tensor,
        cfg_scale: float,
        mask_id: int,
    ) -> torch.Tensor:
        """Forward pass with optional classifier-free guidance.

        When ``cfg_scale > 0``, runs the model twice: once with the
        original sequence and once with prompt tokens replaced by
        ``mask_id``.  The CFG formula is:

            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        """
        if cfg_scale > 0.0:
            prompt_idx_2d = prompt_index.unsqueeze(0).expand(batch.shape[0], -1)
            un_batch = batch.clone()
            un_batch[prompt_idx_2d] = mask_id
            combined = torch.cat([batch, un_batch], dim=0)
            logits, un_logits = torch.chunk(model(combined).logits, 2, dim=0)
            return un_logits + (cfg_scale + 1) * (logits - un_logits)
        return model(batch).logits

    def generate(
        self,
        model: torch.nn.Module,
        prompt: torch.Tensor,
        steps: int = 128,
        gen_length: int = 128,
        block_length: int = 128,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        remasking: str = "low_confidence",
        mask_id: int = 126336,
        denoise_trajectory: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """Block-wise iterative denoising generation for dLLMs.

        Generates ``gen_length`` tokens block-by-block, each block of
        ``block_length`` tokens denoised over ``steps // num_blocks``
        steps using confidence-based or random remasking.

        Args:
            model:        The policy model (unwrapped).
            prompt:       Prompt token ids, shape ``[B, P]``.
            steps:        Total denoising steps (split evenly over blocks).
            gen_length:   Number of tokens to generate.
            block_length: Tokens per block.  Must divide ``gen_length``.
            temperature:  Gumbel noise temperature (0 = argmax).
            cfg_scale:    Classifier-free guidance scale.
            remasking:    ``"low_confidence"`` or ``"random"``.
            mask_id:      Mask token id.
            denoise_trajectory: If a list is passed, appends ``(x_l, x0_pred)`` at each
                inner denoising step: ``x_l`` is the masked state **before** the forward,
                ``x0_pred`` is ``where(mask, argmax(logits + Gumbel), x)`` when
                ``temperature > 0``, else ``where(mask, argmax(logits), x)`` (matches
                the actual unmask targets). Length is ``num_blocks * steps_per_block``.

        Returns:
            Full sequence ``[B, P + gen_length]`` with generated tokens.
        """
        assert gen_length % block_length == 0, (
            f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
        )

        bs = prompt.shape[0]
        dtype: torch.dtype = model.dtype  # type: ignore[assignment]
        device = model.device

        # Initialise: prompt tokens fixed, completion = all MASK
        x = torch.full(
            (bs, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device
        )
        x[:, : prompt.shape[1]] = prompt.clone()
        prompt_index = (x != mask_id)[0]  # [P+gen_length] — True for prompt positions

        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)

        for block_idx in range(num_blocks):
            start = prompt.shape[1] + block_idx * block_length
            end = prompt.shape[1] + (block_idx + 1) * block_length

            block_mask_index = x[:, start:end] == mask_id
            num_transfer = self._get_num_transfer_tokens(
                block_mask_index, steps_per_block
            )

            for step in range(steps_per_block):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                x_l = x.clone()
                mask_index = x == mask_id

                amp_ctx = (
                    torch.amp.autocast("cuda", enabled=True)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with amp_ctx:
                    logits = self._get_logits(
                        model, x, prompt_index, cfg_scale, mask_id
                    )

                    # Sample candidates from masked positions
                    logits_noisy = self._add_gumbel_noise(logits, temperature, dtype)
                    x0 = torch.argmax(logits_noisy, dim=-1)  # [B, L]
                    del logits_noisy

                    # Confidence for remasking decision
                    if remasking == "low_confidence":
                        p = F.softmax(logits.to(dtype), dim=-1)
                        x0_p = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # [B, L]
                    elif remasking == "random":
                        x0_p = torch.rand(x0.shape, device=device)
                    else:
                        raise NotImplementedError(
                            f"Unknown remasking strategy: {remasking!r}"
                        )

                    # Prevent touching positions beyond the current block
                    x0_p[:, end:] = -np.inf

                    x0 = torch.where(mask_index, x0, x)
                    if denoise_trajectory is not None:
                        denoise_trajectory.append((x_l, x0.clone()))
                    confidence = torch.where(
                        mask_index, x0_p, torch.full_like(x0_p, -np.inf)
                    )

                    # Select top-k confident positions to unmask this step
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                    for j in range(confidence.shape[0]):
                        k = num_transfer[j, step].item()
                        if k > 0:
                            _, sel = torch.topk(confidence[j], k=k)
                            transfer_index[j, sel] = True

                    x[transfer_index] = x0[transfer_index]
                    del x0, confidence, transfer_index

        return x

    # ------------------------------------------------------------------
    # Forward process (masking for log-prob computation)
    # ------------------------------------------------------------------

    def _forward_process(
        self,
        batch: torch.Tensor,
        prompt_index: torch.Tensor,
        mask_id: int,
        seed: Optional[int] = None,
        total_rows: Optional[int] = None,
        row_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply stochastic masking to a token sequence.

        Prompt tokens are masked with probability ``p_mask_prompt``;
        completion tokens are always masked (p_mask=1.0), matching the d1
        reference implementation (diffu-grpo/diffu_grpo_trainer.py).
        The returned ``p_mask`` tensor matches that reference (per-position
        mask probability under this scheme); the trainer's cross-entropy path
        matches d1 and does not re-weight loss by ``p_mask``.

        Args:
            batch:         Token ids ``[B, L]``.
            prompt_index:  Bool mask ``[L]`` — True for prompt positions.
            mask_id:       Mask token id.
            seed:          Random seed for reproducibility across iterations.
            total_rows:    When ``batch`` is a micro-batch slice of a larger
                           rollout batch, the rollout batch size ``B_gen``. The
                           random draw is then ``torch.rand((total_rows, L))``
                           sliced at ``[row_offset:row_offset+B]`` so the mask
                           matches the one drawn at rollout time for these rows.
                           ``None`` keeps the plain ``torch.rand((B, L))`` draw.
            row_offset:    Row offset of ``batch`` within the rollout batch
                           (only meaningful with ``total_rows``).

        Returns:
            ``(noisy_batch, p_mask)`` — masked sequence and per-position
            mask probabilities ``[B, L]``.
        """
        if seed is not None:
            set_seed(seed)

        b, l = batch.shape
        t_p = torch.full((b,), self.args.p_mask_prompt, device=batch.device)
        if total_rows is None or total_rows == b:
            # Identical to torch.rand((total_rows, l)) sliced with offset 0 when
            # total_rows == b — S=1 behavior is bit-exact with the legacy draw.
            rng = torch.rand((b, l), device=batch.device)
        else:
            rng = torch.rand((total_rows, l), device=batch.device)[
                row_offset : row_offset + b
            ]

        is_mask_prompt = prompt_index.unsqueeze(0) & (rng < t_p.unsqueeze(1))
        is_mask_completion = ~prompt_index.unsqueeze(0)
        is_mask = is_mask_prompt | is_mask_completion

        noisy_batch = torch.where(
            is_mask, torch.tensor(mask_id, device=batch.device), batch
        )

        p_mask = torch.where(
            prompt_index.unsqueeze(0),
            t_p.unsqueeze(1).expand(b, l),
            torch.ones(b, l, device=batch.device),
        )
        return noisy_batch, p_mask

    def _wd1pp_noisy_for_logprob(
        self,
        x_l: torch.Tensor,
        prompt_index: torch.Tensor,
        mask_id: int,
        seed: int,
    ) -> torch.Tensor:
        """Jitter **prompt** positions with ``p_mask_prompt``; keep completion slice of ``x_l``."""
        b, llen = x_l.shape
        t_p = self.args.p_mask_prompt
        g = torch.Generator(device=x_l.device)
        g.manual_seed(int(seed))
        rng = torch.rand((b, llen), generator=g, device=x_l.device, dtype=torch.float32)
        jitter = prompt_index.unsqueeze(0) & (rng < t_p)
        noisy = x_l.clone()
        noisy[jitter] = mask_id
        return noisy

    def _wd1pp_completion_logp_sum(
        self,
        model: torch.nn.Module,
        x_l: torch.Tensor,
        x0_pred: torch.Tensor,
        logits_to_keep: int,
        seed: int,
    ) -> torch.Tensor:
        """Sum of log π(target) over completion tokens that are still ``mask_id`` in ``x_l``.

        Matches the DCE-style conditioning on ``x_l`` (wd1++ / ``docs/wd1plusplus-design.md``):
        one forward on prompt-jittered noisy input, cross-entropy targets from ``x0_pred``
        at positions where ``x_l`` is masked in the completion region.
        """
        device = x_l.device
        seq_len = x_l.size(1)
        prompt_length = seq_len - logits_to_keep
        prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=device)
        prompt_index[:prompt_length] = True
        noisy = self._wd1pp_noisy_for_logprob(
            x_l, prompt_index, self.args.mask_id, seed
        )
        logits = self._get_logits(
            model, noisy, prompt_index, self.args.cfg_scale, self.args.mask_id
        )
        comp_logits = logits[:, -logits_to_keep:, :]
        targets = x0_pred[:, -logits_to_keep:]
        mask_pos = (x_l[:, -logits_to_keep:] == self.args.mask_id).float()
        nll = F.cross_entropy(
            comp_logits.reshape(-1, comp_logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).view(x_l.size(0), logits_to_keep)
        log_probs = -nll
        return (log_probs * mask_pos).sum(dim=-1)

    # ------------------------------------------------------------------
    # Per-token log-probabilities
    # ------------------------------------------------------------------

    def _get_per_token_logps(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        logits_to_keep: int,
        mask_seeds: list[int],
        buffer_total_rows: Optional[int] = None,
        buffer_row_offset: int = 0,
    ) -> torch.Tensor:
        """Compute per-token log p_θ(x_0 | x_t) for GRPO.

        For each GRPO iteration, applies the forward-process mask with the
        corresponding seed, runs the model, and extracts log-probs over
        completion tokens.

        Args:
            model:          Policy model.
            input_ids:      ``[num_iterations, B, L]`` — repeated full sequences.
            logits_to_keep: Number of completion tokens (``L_completion``).
            mask_seeds:     One seed per GRPO iteration.
            buffer_total_rows: Rollout batch size ``B_gen`` when ``input_ids``
                            holds a micro-batch slice of a buffered rollout;
                            forwarded to :meth:`_forward_process` so the
                            per-seed mask matches the rollout-time draw.
            buffer_row_offset: Row offset of the micro-batch within the
                            rollout batch (with ``buffer_total_rows``).

        Returns:
            ``[num_iterations, B, L_completion]`` float32 log-probs.
        """
        num_iterations, batch_size, seq_len = input_ids.shape
        device = input_ids.device

        prompt_length = seq_len - logits_to_keep
        prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=device)
        prompt_index[:prompt_length] = True

        # Build masked sequences for all iterations in one pass
        all_noisy, all_original = [], []
        for i, seed in enumerate(mask_seeds):
            noisy, _ = self._forward_process(
                input_ids[i],
                prompt_index,
                self.args.mask_id,
                seed,
                total_rows=buffer_total_rows,
                row_offset=buffer_row_offset,
            )
            all_noisy.append(noisy)
            all_original.append(input_ids[i])

        noisy_batch = torch.cat(all_noisy, dim=0)  # [num_iter*B, L]
        orig_batch = torch.cat(all_original, dim=0)  # [num_iter*B, L]

        logits = self._get_logits(
            model, noisy_batch, prompt_index, self.args.cfg_scale, self.args.mask_id
        )  # [num_iter*B, L, V]

        # Cross-entropy over completion slice → log-probs
        comp_logits = logits[:, -logits_to_keep:, :]  # [num_iter*B, Lc, V]
        comp_targets = orig_batch[:, -logits_to_keep:]  # [num_iter*B, Lc]

        loss = F.cross_entropy(
            comp_logits.reshape(-1, comp_logits.size(-1)),
            comp_targets.reshape(-1),
            reduction="none",
        )  # [num_iter*B*Lc]

        log_probs = -loss.view(num_iterations * batch_size, logits_to_keep)
        del noisy_batch, logits, all_noisy, all_original
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return log_probs.view(num_iterations, batch_size, logits_to_keep).to(
            torch.float32
        )

    @staticmethod
    def _wd1_completion_coef(
        advantages: torch.Tensor,
        num_generations: int,
        psi: float,
    ) -> torch.Tensor:
        """Per-completion coefficients (w⁻ − w⁺) for wd1 (Eq. 8–9), flat ``[B]``."""
        if advantages.dim() != 1:
            raise ValueError(
                f"advantages must be 1D, got shape {tuple(advantages.shape)}"
            )
        g = num_generations
        if advantages.numel() % g != 0:
            raise ValueError(
                f"advantages length ({advantages.numel()}) must be divisible by num_generations ({g})"
            )
        adv = advantages.reshape(-1, g)
        w_plus = torch.softmax(psi * adv, dim=-1)
        w_minus = torch.softmax(-psi * adv, dim=-1)
        return (w_minus - w_plus).reshape(-1)

    def _validate_wd1_group_alignment(self, advantages: torch.Tensor) -> None:
        """wd1/wd1++ group softmax needs whole reward groups per micro-batch.

        After ``steps_per_generation`` micro-batch splitting, a chunk that does not
        contain a multiple of ``num_generations`` rows would either fail the raw
        reshape in :meth:`_wd1_completion_coef` or silently softmax over partial /
        straddled groups. Fail fast with an actionable message instead.
        """
        g = self.num_generations
        if advantages.numel() % g != 0:
            raise ValueError(
                "wd1/wd1++ objectives require each micro-batch to contain whole "
                f"reward groups: got {advantages.numel()} completions with "
                f"num_generations={g}. Configure generation_batch_size / "
                "steps_per_generation so that (rollout batch // steps_per_generation) "
                "is divisible by num_generations."
            )

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """Policy loss for diffusion RL: GRPO (default) or wd1 (ratio-free).

        **GRPO** (``args.diffu_policy_objective == "grpo"``): clipped objective

            L = -E[ min(r·A, clip(r, 1-ε, 1+ε)·A) ] + β·KL(π_θ ‖ π_ref)

        with r = exp(log π_θ - log π_old).

        **wd1** (``"wd1"``): weighted log-likelihood L = E[(w⁻ − w⁺) log π_θ(o|q)]
        with group-normalized weights (wd1 paper Eq. 8–9), using the same d1-style
        per-token log-prob sum as the sequence likelihood.

        **wd1++** (``"wd1++"``): Eq. 10-style MC objective — one uniform denoise step
        per micro-batch forward, :meth:`_wd1pp_completion_logp_sum` at stored
        ``(x_l, x_0)`` snapshots, weighted like wd1. Optional KL uses the final
        sequence (d1-style) when ``beta > 0``.

        Args:
            model:              The policy model.
            inputs:             Dict produced by :meth:`_generate_and_score_completions`.
            return_outputs:     Not supported (raises ValueError if True).
            num_items_in_batch: Unused; present for API compatibility.

        Returns:
            Scalar loss tensor.
        """
        if return_outputs:
            raise ValueError("DiffuGRPOTrainer does not support return_outputs=True")

        prompt_ids = inputs["prompt_ids"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        advantages = inputs["advantages"]

        objective = getattr(self.args, "diffu_policy_objective", "grpo")

        # ``_prepare_inputs`` increments ``_step`` once per micro-batch BEFORE
        # ``compute_loss``; the GRPO iteration index advances once per full pass
        # over the ``steps_per_generation`` buffered micro-batches.
        this_itr_idx = (
            (self._step - 1) // self.args.steps_per_generation
        ) % self.num_iterations
        # Row window of this micro-batch within the rollout batch (set by
        # ``_prepare_inputs``); ``None`` when inputs were not buffered/split.
        buffer_total_rows = inputs.get("buffer_total_rows")
        buffer_row_offset = inputs.get("buffer_row_offset", 0)
        # Mask seeds travel with the inputs (buffered chunk or eval dict) so an eval
        # rollout landing mid-generation-cycle cannot clobber them (d1 pattern);
        # fall back to the trainer attribute for back-compat.
        seeds = inputs.get("mask_seeds") or self._diffu_mask_seeds

        if objective in ("wd1", "wd1++"):
            self._validate_wd1_group_alignment(advantages)

        if objective == "wd1++":
            logits_to_keep = completion_ids.size(1)
            if not seeds or this_itr_idx >= len(seeds):
                raise RuntimeError(
                    "DiffuGRPOTrainer: missing mask seeds for this step — internal buffering error."
                )
            this_seed = seeds[this_itr_idx]
            # Same clobber-protection as ``mask_seeds``: prefer the trajectory
            # reference captured at buffering time.
            traj = inputs.get("wd1pp_trajectory") or self._wd1pp_trajectory
            if not traj:
                raise RuntimeError(
                    "DiffuGRPOTrainer: wd1++ missing denoise trajectory — internal rollout error."
                )
            B_micro = prompt_ids.size(0)
            traj_b = traj[0][0].size(0)
            # ``_prepare_inputs`` increments ``_step`` before ``compute_loss``; the buffer index
            # used to fetch this micro-batch was ``(_step - 1) % steps_per_generation``.
            buf_idx = (self._step - 1) % self.args.steps_per_generation
            if B_micro == traj_b:
                sl = slice(None)
            else:
                sl = slice(buf_idx * B_micro, (buf_idx + 1) * B_micro)

            L_tot = len(traj)
            gen = torch.Generator(device=prompt_ids.device)
            gen.manual_seed(int(this_seed) + 10_007 + buf_idx)
            lidx = int(torch.randint(0, L_tot, (1,), generator=gen).item())
            xl, x0 = traj[lidx]
            seq_logp = self._wd1pp_completion_logp_sum(
                model, xl[sl], x0[sl], logits_to_keep, int(this_seed)
            )
            coef = self._wd1_completion_coef(
                advantages, self.num_generations, float(self.args.wd1_psi)
            )
            mode = "eval" if self.control.should_evaluate else "train"
            loss = (coef * seq_logp).mean()
            if self.beta != 0.0:
                input_ids_kl = torch.cat([prompt_ids, completion_ids], dim=1)
                input_ids_3d_kl = input_ids_kl.unsqueeze(0)
                per_token_logps_kl = self._get_per_token_logps(
                    model,
                    input_ids_3d_kl,
                    logits_to_keep,
                    [this_seed],
                    buffer_total_rows=buffer_total_rows,
                    buffer_row_offset=buffer_row_offset,
                ).squeeze(0)
                ref_per_token_logps = inputs["ref_per_token_logps"][
                    this_itr_idx
                ].squeeze(0)
                per_token_kl = (
                    torch.exp(ref_per_token_logps - per_token_logps_kl)
                    - (ref_per_token_logps - per_token_logps_kl)
                    - 1
                )
                mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
                self._metrics[mode]["kl"].append(
                    self.accelerator.gather_for_metrics(mean_kl).mean().item()
                )
                loss = loss + self.beta * mean_kl
            self._metrics[mode]["clip_ratio"].append(0.0)
            return loss

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)

        if not seeds or this_itr_idx >= len(seeds):
            raise RuntimeError(
                "DiffuGRPOTrainer: missing mask seeds for this step — internal buffering error."
            )
        this_seed = [seeds[this_itr_idx]]
        input_ids_3d = input_ids.unsqueeze(0)  # [1, B, L]

        per_token_logps = self._get_per_token_logps(
            model,
            input_ids_3d,
            logits_to_keep,
            this_seed,
            buffer_total_rows=buffer_total_rows,
            buffer_row_offset=buffer_row_offset,
        )
        per_token_logps = per_token_logps.squeeze(0)  # [B, Lc]

        mode = "eval" if self.control.should_evaluate else "train"

        if objective == "wd1":
            # Minimize paper Eq. (8): E[(w− − w+) log π(o|q)]. Coefficients are signed;
            # this is not "minimize log π" and must not be negated to match the GRPO path
            # (which wraps a *maximand* in -min(...)).
            seq_logp = (per_token_logps * completion_mask).sum(dim=-1)
            coef = self._wd1_completion_coef(
                advantages, self.num_generations, float(self.args.wd1_psi)
            )
            loss = (coef * seq_logp).mean()
            if self.beta != 0.0:
                ref_per_token_logps = inputs["ref_per_token_logps"][
                    this_itr_idx
                ].squeeze(0)
                per_token_kl = (
                    torch.exp(ref_per_token_logps - per_token_logps)
                    - (ref_per_token_logps - per_token_logps)
                    - 1
                )
                mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
                self._metrics[mode]["kl"].append(
                    self.accelerator.gather_for_metrics(mean_kl).mean().item()
                )
                loss = loss + self.beta * mean_kl
            self._metrics[mode]["clip_ratio"].append(0.0)
            return loss

        # KL divergence term (GRPO path)
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"][this_itr_idx].squeeze(0)
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        # Old log-probs (for ratio computation)
        if self.num_iterations > 1:
            old_per_token_logps = inputs["old_per_token_logps"][this_itr_idx].squeeze(0)
        else:
            old_per_token_logps = per_token_logps.detach()

        # Clipped policy-gradient (TRL uses epsilon_low / epsilon_high; older TRL used epsilon)
        eps_lo = getattr(self, "epsilon_low", None)
        eps_hi = getattr(self, "epsilon_high", None)
        if eps_lo is None:
            eps_lo = getattr(self, "epsilon", 0.2)
        if eps_hi is None:
            eps_hi = getattr(self, "epsilon", 0.2)
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        ratio_clipped = torch.clamp(ratio, 1 - eps_lo, 1 + eps_hi)
        per_token_loss = -torch.min(
            ratio * advantages.unsqueeze(1),
            ratio_clipped * advantages.unsqueeze(1),
        )

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).mean().item()
            )

        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Metrics
        is_clipped = (ratio < ratio_clipped).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )

        return loss

    # ------------------------------------------------------------------
    # Rollout generation and scoring
    # ------------------------------------------------------------------

    def _compute_ref_per_token_logps(
        self,
        prompt_completion_ids: torch.Tensor,
        logits_to_keep: int,
        mask_seeds: list[int],
    ) -> torch.Tensor:
        """Reference-policy log-probs for the KL term (``beta != 0``).

        Mirrors TRL 0.24 ``GRPOTrainer``: a PEFT policy disables its adapter to
        act as its own reference (d1's assumption); otherwise TRL's
        ``self.ref_model`` — constructed in ``GRPOTrainer.__init__`` for non-PEFT
        policies when ``beta != 0`` — is used. Raises a clear error when neither
        is available instead of an ``AttributeError`` on ``disable_adapter``.
        """
        ids_expanded = prompt_completion_ids.unsqueeze(0).expand(
            self.num_iterations, -1, -1
        )
        unwrapped = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped, "disable_adapter"):
            with unwrapped.disable_adapter():
                return self._get_per_token_logps(
                    self.model, ids_expanded, logits_to_keep, mask_seeds
                )
        if getattr(self, "ref_model", None) is not None:
            return self._get_per_token_logps(
                self.ref_model, ids_expanded, logits_to_keep, mask_seeds
            )
        raise ValueError(
            "DiffuGRPOTrainer: beta != 0 requires a reference policy — either a "
            "PEFT adapter on the policy model (so it can be disabled to act as the "
            "reference) or a TRL-constructed `ref_model`. Neither is available; "
            "pass a peft_config or set beta=0."
        )

    def _compute_group_advantages(
        self, rewards: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Group-relative advantages with TRL 0.24 ``scale_rewards`` semantics.

        d1 mean-centers only; TRL additionally divides by ``std + 1e-4`` unless
        ``scale_rewards`` resolves to ``"none"`` (``"group"`` is TRL's default;
        ``"batch"`` uses the whole-batch std). Pass ``scale_rewards=False`` /
        ``"none"`` for d1-style mean-only advantages.

        Returns:
            ``(advantages, std_rewards)`` — both flat ``[B]``; ``std_rewards`` is
            the tensor used for scaling and for the std metrics.
        """
        g = self.num_generations
        mean_rewards = rewards.view(-1, g).mean(dim=1).repeat_interleave(g)
        advantages = rewards - mean_rewards
        scale = getattr(
            self, "scale_rewards", getattr(self.args, "scale_rewards", "group")
        )
        # GRPOConfig.__post_init__ already coerces bools; keep the mapping for
        # bare instances constructed without TRL's __init__.
        scale = {True: "group", False: "none"}.get(scale, scale)
        if scale in ("group", "none"):
            std_rewards = rewards.view(-1, g).std(dim=1).repeat_interleave(g)
        elif scale == "batch":
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(
                f"Invalid scale_rewards value: {scale!r}. "
                "Must be 'group', 'batch', or 'none'."
            )
        if scale != "none":
            advantages = advantages / (std_rewards + 1e-4)
        return advantages, std_rewards

    def _generate_and_score_completions(
        self,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate completions via diffusion and score with reward functions.

        Overrides the TRL base method to use block-wise iterative denoising
        instead of autoregressive sampling.  All GRPO bookkeeping (advantage
        normalisation, old log-prob caching, reward aggregation) mirrors the
        d1 reference implementation.

        Args:
            inputs: List of per-example dicts with at least a ``"prompt"`` key.

        Returns:
            Dict suitable for :meth:`compute_loss`.
        """
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(ex, self.processing_class)["prompt"]
            for ex in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        from transformers import Trainer as _Trainer

        prompt_inputs = _Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        gen_length = self.args.max_completion_length
        block_length = self.args.block_length
        steps = self.args.diffusion_steps
        temperature = getattr(self.args, "temperature", 0.0) or 0.0
        cfg_scale = self.args.cfg_scale
        gen_bs = self.args.generation_batch_size
        use_wd1pp = getattr(self.args, "diffu_policy_objective", "grpo") == "wd1++"
        self._wd1pp_trajectory = None

        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator
        ) as unwrapped:
            chunks = []
            traj_chunks: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
            for i in range(0, prompt_ids.size(0), gen_bs):
                batch_prompt = prompt_ids[i : i + gen_bs]
                if use_wd1pp:
                    step_traj: list[tuple[torch.Tensor, torch.Tensor]] = []
                    chunk = self.generate(
                        model=unwrapped,
                        prompt=batch_prompt,
                        steps=steps,
                        gen_length=gen_length,
                        block_length=block_length,
                        temperature=temperature,
                        cfg_scale=cfg_scale,
                        remasking=self.args.remasking,
                        mask_id=self.args.mask_id,
                        denoise_trajectory=step_traj,
                    )
                    traj_chunks.append(step_traj)
                else:
                    chunk = self.generate(
                        model=unwrapped,
                        prompt=batch_prompt,
                        steps=steps,
                        gen_length=gen_length,
                        block_length=block_length,
                        temperature=temperature,
                        cfg_scale=cfg_scale,
                        remasking=self.args.remasking,
                        mask_id=self.args.mask_id,
                    )
                chunks.append(chunk)
                del batch_prompt, chunk
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if use_wd1pp and traj_chunks:
                n_steps = len(traj_chunks[0])
                if not all(len(tc) == n_steps for tc in traj_chunks):
                    raise RuntimeError(
                        "DiffuGRPOTrainer: wd1++ trajectory step count mismatch across generation chunks."
                    )
                merged: list[tuple[torch.Tensor, torch.Tensor]] = []
                for s in range(n_steps):
                    xl = torch.cat(
                        [traj_chunks[c][s][0] for c in range(len(traj_chunks))], dim=0
                    )
                    x0 = torch.cat(
                        [traj_chunks[c][s][1] for c in range(len(traj_chunks))], dim=0
                    )
                    merged.append((xl, x0))
                self._wd1pp_trajectory = merged

        prompt_completion_ids = torch.cat(chunks, dim=0)
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Build completion mask: zero out positions after first EOS
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_indices = torch.arange(is_eos.size(1), device=device).expand_as(
            completion_ids
        )
        completion_mask = (seq_indices <= eos_idx.unsqueeze(1)).int()

        logits_to_keep = completion_ids.size(1)

        # Random mask seeds: one per GRPO iteration
        if self.args.random_masking:
            mask_seeds = torch.randint(
                0, 2**12, (self.num_iterations,), device=device
            ).tolist()
        else:
            mask_seeds = [42] * self.num_iterations
        self._diffu_mask_seeds = mask_seeds

        # Compute old log-probs and optionally reference log-probs
        all_old_logps: list[torch.Tensor] = []
        all_ref_logps: list[torch.Tensor] = []
        use_old_logps_for_grpo = (
            getattr(self.args, "diffu_policy_objective", "grpo") == "grpo"
            and self.num_iterations > 1
        )
        with torch.no_grad():
            if use_old_logps_for_grpo:
                ids_expanded = prompt_completion_ids.unsqueeze(0).expand(
                    self.num_iterations, -1, -1
                )
                all_old_logps = self._get_per_token_logps(
                    self.model, ids_expanded, logits_to_keep, mask_seeds
                )

            if self.beta != 0.0:
                all_ref_logps = self._compute_ref_per_token_logps(
                    prompt_completion_ids, logits_to_keep, mask_seeds
                )

        # Decode and score completions
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text, strict=True):
                prompt_copy = list(prompt)
                bootstrap = (
                    prompt_copy.pop()["content"]
                    if prompt_copy[-1]["role"] == "assistant"
                    else ""
                )
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        for i, reward_func in enumerate(self.reward_funcs):
            keys = [k for k in inputs[0] if k not in ("prompt", "completion")]
            reward_kwargs = {k: [ex[k] for ex in inputs] for k in keys}
            output = reward_func(
                prompts=prompts, completions=completions, **reward_kwargs
            )
            output = [r if r is not None else float("nan") for r in output]
            rewards_per_func[:, i] = torch.tensor(
                output, dtype=torch.float32, device=device
            )

        if torch.isnan(rewards_per_func).all(dim=1).any():
            warnings.warn(
                "All reward functions returned None for at least one sample. "
                "Ensure at least one reward function returns a valid reward."
            )

        rewards_per_func = gather(rewards_per_func)
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        # Advantage normalisation (group-relative, TRL scale_rewards semantics)
        advantages, std_rewards = self._compute_group_advantages(rewards)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Logging
        mode = "eval" if self.control.should_evaluate else "train"
        comp_len = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1))
            .float()
            .mean()
            .item()
        )
        self._metrics[mode]["completion_length"].append(comp_len)

        zero_std_ratio = (std_rewards < 1e-6).float().mean().item()
        self._metrics[mode]["zero_std_ratio"].append(zero_std_ratio)

        for i, reward_func in enumerate(self.reward_funcs):
            name = (
                reward_func.config._name_or_path.split("/")[-1]
                if isinstance(reward_func, torch.nn.Module)
                else reward_func.__name__
            )
            self._metrics[mode][f"rewards/{name}"].append(
                torch.nanmean(rewards_per_func[:, i]).item()
            )
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
        ):
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            if self.accelerator.is_main_process:
                try:
                    from trl.import_utils import is_rich_available

                    if is_rich_available():
                        _print_completions_sample(
                            prompts_to_log,
                            completions_to_log,
                            rewards.tolist(),
                            self.state.global_step,
                        )
                except ImportError:
                    pass

                try:
                    import wandb

                    if wandb.run is not None and "wandb" in (self.args.report_to or []):
                        import pandas as pd

                        wandb.log(
                            {
                                "completions": wandb.Table(
                                    dataframe=pd.DataFrame(
                                        {
                                            "step": [str(self.state.global_step)]
                                            * len(rewards),
                                            "prompt": prompts_to_log,
                                            "completion": completions_to_log,
                                            "reward": rewards.tolist(),
                                        }
                                    )
                                )
                            }
                        )
                except ImportError:
                    pass

        out = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": all_old_logps,
            "ref_per_token_logps": all_ref_logps,
            "advantages": advantages,
            # Travels with the inputs (d1 pattern); ``_prepare_inputs`` pops it
            # before ``split_tensor_dict`` and re-attaches the full list per chunk.
            "mask_seeds": mask_seeds,
        }
        if use_wd1pp:
            out["wd1pp_trajectory"] = self._wd1pp_trajectory
        return out
