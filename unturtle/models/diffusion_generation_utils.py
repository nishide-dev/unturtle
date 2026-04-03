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
#
# Shared MDLM-style generation utilities for masked diffusion LMs.
# Extracted from unturtle/models/dream/generation_utils.py (Dream-specific
# logit right-shift is intentionally excluded — it is a Dream training artefact
# and does NOT apply to A2D or LLaDA models).

"""Shared MDLM-style generation utilities for masked diffusion LMs.

This module provides :class:`MaskedDiffusionGenerationMixin` which implements
the iterative masked-token denoising loop used by LLaDA, MDLM, and A2D models.

Key difference from :class:`~unturtle.models.dream.DreamGenerationMixin`:
- **No logit right-shift** — Dream shifts logits by one position because its
  training objective is shifted; A2D / LLaDA predict token ``i`` at position
  ``i`` directly so no shift is needed.

Usage::

    from unturtle.models.diffusion_generation_utils import (
        MaskedDiffusionGenerationConfig,
        MaskedDiffusionGenerationMixin,
        MaskedDiffusionModelOutput,
    )
"""

import copy
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils import ModelOutput, is_torchdynamo_compiling, logging

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper functions (shared with Dream; kept in sync manually)
# ---------------------------------------------------------------------------

def top_p_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering of logits."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift right: keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def top_k_logits(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Top-k filtering of logits."""
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    return logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)


def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    margin_confidence: bool = False,
    neg_entropy: bool = False,
) -> Tuple[torch.Tensor, torch.LongTensor]:
    """Sample tokens from logits and return (confidence, token_ids).

    Returns
    -------
    confidence : Tensor of shape ``[N]``
    x0 : LongTensor of shape ``[N]``
    """
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        confidence = sorted_probs[:, 0] - sorted_probs[:, 1]

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class MaskedDiffusionModelOutput(ModelOutput):
    """Output of :meth:`MaskedDiffusionGenerationMixin.diffusion_generate`."""

    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.LongTensor, ...]] = None


# ---------------------------------------------------------------------------
# Generation config
# ---------------------------------------------------------------------------

class MaskedDiffusionGenerationConfig(GenerationConfig):
    """Generation configuration for MDLM-style masked diffusion models.

    Parameters
    ----------
    steps : int
        Number of denoising steps (default: 128).
    mask_token_id : int or None
        ID of the ``[MASK]`` token.  Required; must be set before generation.
    temperature : float
        Sampling temperature.  0.0 means argmax (default: 0.0).
    top_p : float or None
        Nucleus sampling probability (default: None → disabled).
    top_k : int or None
        Top-k sampling (default: None → disabled).
    alg : str
        Unmasking algorithm.  One of ``"origin"``, ``"maskgit_plus"``,
        ``"topk_margin"``, ``"entropy"`` (default: ``"origin"``).
    alg_temp : float or None
        Temperature for the confidence-based token selection step in
        non-``"origin"`` algorithms (default: None → deterministic topk).
    eps : float
        Minimum timestep (default: 1e-3).
    output_history : bool
        If True and ``return_dict=True``, include per-step token sequences in
        the output (default: False).
    return_dict : bool
        Return a :class:`MaskedDiffusionModelOutput` instead of a plain tensor
        (default: False).
    """

    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length: int = kwargs.pop("max_length", 20)
        self.max_new_tokens: Optional[int] = kwargs.pop("max_new_tokens", None)
        # diffusion-specific
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 128)
        self.alg: str = kwargs.pop("alg", "origin")
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)
        # output control
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict: bool = kwargs.pop("return_dict", False)
        self.output_history: bool = kwargs.pop("output_history", False)
        # special tokens
        self.mask_token_id: Optional[int] = kwargs.pop("mask_token_id", None)
        self.pad_token_id: Optional[int] = kwargs.pop("pad_token_id", None)
        self.bos_token_id: Optional[int] = kwargs.pop("bos_token_id", None)
        self.eos_token_id: Optional[int] = kwargs.pop("eos_token_id", None)
        # HF internals
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        if not self._from_model_config:
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        self.validate(is_init=True)

    def validate(self, is_init: bool = False, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------

class MaskedDiffusionGenerationMixin:
    """Generation mixin for MDLM-style masked diffusion LMs.

    Provides :meth:`diffusion_generate` which implements the iterative masked-
    token denoising loop.  The mixin is designed to be mixed into model classes
    as the *first* base class so that its :meth:`diffusion_generate` takes
    precedence over HuggingFace's autoregressive ``generate``.

    Subclasses must implement a HF-compatible ``forward(input_ids, ...)`` that
    returns an object with a ``.logits`` attribute of shape ``[B, L, V]``.

    Unlike :class:`~unturtle.models.dream.DreamGenerationMixin`, **no** logit
    right-shift is applied here.  A2D and LLaDA models predict token ``i``
    at output position ``i`` directly.
    """

    # ------------------------------------------------------------------
    # Internal helpers (mirror of DreamGenerationMixin)
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        if is_torchdynamo_compiling():
            return
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            raise ValueError(
                f"Input length of input_ids is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(self, generation_config, has_default_max_length, input_ids_length):
        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence."
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length
        elif has_default_max_length:
            if generation_config.max_length == MaskedDiffusionGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_pos = getattr(self.config, "max_position_embeddings", None)
                if max_pos is not None:
                    generation_config.max_length = min(generation_config.max_length, max_pos)
        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[MaskedDiffusionGenerationConfig], **kwargs
    ) -> MaskedDiffusionGenerationConfig:
        if generation_config is None:
            # Build a default config seeded from well-known HF special tokens.
            # We intentionally do NOT use from_model_config() because HF's
            # implementation compares against GenerationConfig() attributes and
            # raises AttributeError for our custom fields (eps, steps, …).
            init_kwargs = {}
            model_cfg = getattr(self, "config", None)
            if model_cfg is not None:
                for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "mask_token_id"):
                    val = getattr(model_cfg, attr, None)
                    if val is not None:
                        init_kwargs[attr] = val
            generation_config = MaskedDiffusionGenerationConfig(**init_kwargs)

        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            generation_config.update(**kwargs)

        return generation_config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[MaskedDiffusionGenerationConfig] = None,
        **kwargs,
    ) -> Union[MaskedDiffusionModelOutput, torch.LongTensor]:
        """Generate sequences via MDLM-style iterative masked-token denoising.

        Parameters
        ----------
        inputs : LongTensor of shape ``[B, L]``
            Prompt token IDs.  Completion positions should already be filled
            with ``mask_token_id``.
        generation_config : MaskedDiffusionGenerationConfig, optional
            Generation parameters.  If ``None``, model defaults are used.
        **kwargs
            Forwarded to :class:`MaskedDiffusionGenerationConfig` (e.g.
            ``steps``, ``temperature``, ``mask_token_id``, ``max_new_tokens``).

        Returns
        -------
        MaskedDiffusionModelOutput or LongTensor
            When ``generation_config.return_dict=True`` returns a
            :class:`MaskedDiffusionModelOutput`; otherwise returns the
            token-ID tensor directly.
        """
        generation_config = self._prepare_generation_config(generation_config, **kwargs)

        assert inputs is not None, "`inputs` (input_ids) must be provided"
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)

        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )
        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .diffusion_generate() with `input_ids` on a different device type than the model."
                f" `input_ids` is on {input_ids.device.type}, model is on {self.device.type}.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: MaskedDiffusionGenerationConfig,
    ) -> Union[MaskedDiffusionModelOutput, torch.LongTensor]:
        """Core MDLM denoising loop.

        Pads ``input_ids`` to ``max_length`` with ``mask_token_id``, then
        iterates over ``steps`` timesteps to progressively unmask tokens.
        """
        output_history = generation_config.output_history
        return_dict_out = generation_config.return_dict
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        if mask_token_id is None:
            # Try to get from model config as a fallback
            mask_token_id = getattr(self.config, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError(
                "`mask_token_id` must be set in `generation_config` or `model.config` before calling "
                "`diffusion_generate()`.  Pass it explicitly: "
                "`model.diffusion_generate(inputs, mask_token_id=<id>, ...)`"
            )

        histories = [] if (return_dict_out and output_history) else None

        # Pad completion region with mask tokens
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            # Broadcast to [B, 1, L, L] for SDPA
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            attention_mask = None

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        for i in range(steps):
            mask_index = x == mask_token_id

            # Forward pass — no logit shift (contrast with DreamGenerationMixin)
            logits = self(input_ids=x, attention_mask=attention_mask).logits  # [B, L, V]

            mask_logits = logits[mask_index]  # [N_masked, V]
            t = timesteps[i]
            s = timesteps[i + 1]

            if alg == "origin":
                p_transfer = 1 - s / t if i < steps - 1 else 1.0
                x0 = torch.full_like(x[mask_index], mask_token_id, dtype=torch.long)
                transfer = torch.rand(*x0.shape, device=x.device) < p_transfer
                _, sampled = sample_tokens(mask_logits[transfer], temperature=temperature, top_p=top_p, top_k=top_k)
                x0[transfer] = sampled
                x[mask_index] = x0
            else:
                if alg == "maskgit_plus":
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                elif alg == "topk_margin":
                    confidence, x0 = sample_tokens(
                        mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True
                    )
                elif alg == "entropy":
                    confidence, x0 = sample_tokens(
                        mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, neg_entropy=True
                    )
                else:
                    raise RuntimeError(f"Unknown alg: {alg!r}. Choose from 'origin', 'maskgit_plus', 'topk_margin', 'entropy'.")

                num_mask_token = mask_index.sum() / mask_index.shape[0]
                n_transfer = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
                full_confidence = torch.full_like(x, -torch.inf, dtype=logits.dtype)
                full_confidence[mask_index] = confidence

                if n_transfer > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(full_confidence, n_transfer)
                    else:
                        full_confidence = full_confidence / alg_temp
                        full_confidence = F.softmax(full_confidence, dim=-1)
                        transfer_index = torch.multinomial(full_confidence, num_samples=n_transfer)

                    x_ = torch.full_like(x, mask_token_id, dtype=torch.long)
                    x_[mask_index] = x0
                    row_idx = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand_as(transfer_index)
                    x[row_idx, transfer_index] = x_[row_idx, transfer_index]

            if histories is not None:
                histories.append(x.clone())

        if return_dict_out:
            return MaskedDiffusionModelOutput(
                sequences=x,
                history=tuple(histories) if histories is not None else None,
            )
        return x


__all__ = [
    "MaskedDiffusionGenerationConfig",
    "MaskedDiffusionGenerationMixin",
    "MaskedDiffusionModelOutput",
    "sample_tokens",
    "top_p_logits",
    "top_k_logits",
]
