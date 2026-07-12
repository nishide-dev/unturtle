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

Key difference from :class:`~unturtle.models.backbones.dream.DreamGenerationMixin`:
- **No logit right-shift** — Dream shifts logits by one position because its
  training objective is shifted; A2D / LLaDA predict token ``i`` at position
  ``i`` directly so no shift is needed.

Usage::

    from unturtle.models.generation.diffusion_generation_utils import (
        MaskedDiffusionGenerationConfig,
        MaskedDiffusionGenerationMixin,
        MaskedDiffusionModelOutput,
    )
"""

import copy
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from transformers.cache_utils import DynamicCache

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils import ModelOutput, is_torchdynamo_compiling, logging

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Cache management utilities (Phase M: Block-decode)
# ---------------------------------------------------------------------------


def _trim_kv_cache(
    past_key_values: Any,
    target_length: int,
) -> Tuple[Tuple[torch.Tensor, ...], ...]:
    """Trim KV cache to retain only the first target_length tokens.

    This follows Fast-dLLM's approach for cache trimming in non-dual mode.

    Args:
        past_key_values: Either DynamicCache or tuple of (key, value) tuples per layer.
            Shape per tensor: [batch, num_heads, seq_len, head_dim]
        target_length: Number of tokens to retain in cache.

    Returns:
        Trimmed past_key_values as tuple format with seq_len = target_length.

    Raises:
        ValueError: If target_length is invalid.
        TypeError: If cache format is unexpected.
    """
    if target_length < 0:
        raise ValueError(f"target_length must be non-negative, got {target_length}")

    if target_length == 0:
        raise ValueError("target_length=0 would result in empty cache")

    # Convert DynamicCache to tuple format if needed
    if not isinstance(past_key_values, tuple):
        if not hasattr(past_key_values, "layers"):
            raise TypeError(
                f"Expected cache object to have '.layers' attribute, but got {type(past_key_values).__name__}. "
                f"Supported cache types: DynamicCache (transformers) or raw tuple format."
            )

        new_past_key_values = []
        try:
            for layer in past_key_values.layers:
                key_trimmed = layer.keys[:, :, :target_length, :]
                value_trimmed = layer.values[:, :, :target_length, :]
                new_past_key_values.append((key_trimmed, value_trimmed))
        except AttributeError as e:
            raise RuntimeError(
                f"Failed to access cache layer attributes. Cache type: {type(past_key_values).__name__}. "
                f"Expected 'keys' and 'values' attributes on each layer. Original error: {e}"
            ) from e

        return tuple(new_past_key_values)
    else:
        # Already tuple format
        if len(past_key_values) == 0:
            raise ValueError("Received empty tuple cache")

        new_past_key_values = []
        for layer_idx in range(len(past_key_values)):
            layer_cache = past_key_values[layer_idx]

            if not isinstance(layer_cache, tuple) or len(layer_cache) != 2:
                raise TypeError(
                    f"Layer {layer_idx}: expected tuple of (key, value), "
                    f"got {type(layer_cache).__name__} with length {len(layer_cache) if hasattr(layer_cache, '__len__') else 'N/A'}"
                )

            key, value = layer_cache
            key_trimmed = key[:, :, :target_length, :]
            value_trimmed = value[:, :, :target_length, :]
            new_past_key_values.append((key_trimmed, value_trimmed))

        return tuple(new_past_key_values)


def _tuple_to_cache(
    past_key_values: Tuple[Tuple[torch.Tensor, ...], ...],
    device: torch.device,
) -> "DynamicCache":
    """Convert raw tuple cache to DynamicCache for transformers compatibility.

    This is necessary because transformers' attention layers call `.update()` on
    the cache object, which tuples don't support.

    Args:
        past_key_values: Tuple of (key, value) tuples per layer.
            Shape per tensor: [batch, num_heads, seq_len, head_dim]
        device: Device to place the cache tensors on.

    Returns:
        DynamicCache object compatible with transformers attention layers.

    Raises:
        TypeError: If cache structure is invalid.
        ValueError: If cache is empty or malformed.
    """
    from transformers.cache_utils import DynamicCache

    if not isinstance(past_key_values, tuple):
        raise TypeError(
            f"Expected past_key_values to be tuple, got {type(past_key_values).__name__}"
        )

    if len(past_key_values) == 0:
        raise ValueError("Cannot convert empty tuple to DynamicCache")

    cache = DynamicCache()
    for layer_idx, layer_cache in enumerate(past_key_values):
        if not isinstance(layer_cache, tuple):
            raise TypeError(
                f"Layer {layer_idx}: expected tuple of (key, value), got {type(layer_cache).__name__}"
            )

        if len(layer_cache) != 2:
            raise ValueError(
                f"Layer {layer_idx}: expected tuple of length 2 (key, value), got length {len(layer_cache)}"
            )

        key, value = layer_cache

        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Layer {layer_idx}: expected key and value to be torch.Tensor, "
                f"got key={type(key).__name__}, value={type(value).__name__}"
            )

        if key.shape != value.shape:
            raise ValueError(
                f"Layer {layer_idx}: key and value shape mismatch: "
                f"key.shape={key.shape}, value.shape={value.shape}"
            )

        try:
            cache.update(key, value, layer_idx)
        except Exception as e:
            raise RuntimeError(
                f"Layer {layer_idx}: DynamicCache.update() failed. "
                f"Key shape: {key.shape}, value shape: {value.shape}, device: {key.device}. "
                f"Original error: {e}"
            ) from e

    return cache


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
    confidence_type: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.LongTensor]:
    """Sample tokens from logits and return (confidence, token_ids).

    Supported confidence modes:
    - default / ``max_prob``: sampled token probability (or argmax probability)
    - ``margin``: top1 - top2 probability margin
    - ``neg_entropy``: sum(p * log p), matching Fast-dLLM / unturtle entropy mode

    ``confidence_type`` is the canonical API for block-decode codepaths.
    ``margin_confidence`` / ``neg_entropy`` are preserved for existing callers.
    """
    if confidence_type is not None:
        if confidence_type == "max_prob":
            margin_confidence = False
            neg_entropy = False
        elif confidence_type == "margin":
            margin_confidence = True
            neg_entropy = False
        elif confidence_type == "neg_entropy":
            margin_confidence = False
            neg_entropy = True
        else:
            raise ValueError(
                f"Unknown confidence_type={confidence_type!r}. "
                "Choose from 'max_prob', 'margin', 'neg_entropy'."
            )

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
        except (RuntimeError, ValueError) as e:
            error_msg = str(e).lower()
            if "probability" in error_msg or "nan" in error_msg or "inf" in error_msg:
                warnings.warn(
                    f"Categorical sampling failed due to invalid probabilities (possibly NaN/Inf). "
                    f"Falling back to argmax. Original error: {e}",
                    UserWarning,
                )
                confidence, x0 = probs.max(dim=-1)
            else:
                raise RuntimeError(
                    f"Unexpected error during token sampling with temperature={temperature}. "
                    f"Probs shape: {probs.shape}. Original error: {e}"
                ) from e
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


def select_threshold_transfer_mask(
    masked_confidence: torch.Tensor,
    mask_index_block: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Select block-local tokens to unmask under paper-style threshold semantics.

    For each batch row:
    - consider only currently masked positions in the current block
    - unmask all tokens with confidence >= threshold
    - if none meet threshold, still unmask the single max-confidence token

    `masked_confidence` is block-shaped and only meaningful at masked positions.
    Values at unmasked positions are ignored.
    """
    if masked_confidence.ndim != 2 or mask_index_block.ndim != 2:
        raise ValueError(
            "masked_confidence and mask_index_block must be 2-D tensors of shape [batch, block_length]"
        )
    if masked_confidence.shape != mask_index_block.shape:
        raise ValueError(
            f"Shape mismatch: masked_confidence.shape={masked_confidence.shape}, "
            f"mask_index_block.shape={mask_index_block.shape}"
        )

    valid_confidence = masked_confidence.masked_fill(~mask_index_block, float("-inf"))
    transfer_mask = mask_index_block & (valid_confidence >= threshold)

    has_masked = mask_index_block.any(dim=1)
    has_selected = transfer_mask.any(dim=1)
    needs_fallback = has_masked & ~has_selected
    if needs_fallback.any():
        fallback_indices = valid_confidence.argmax(dim=1)
        transfer_mask[needs_fallback, fallback_indices[needs_fallback]] = True

    return transfer_mask


# ---------------------------------------------------------------------------
# BD3LM generation helpers (shared with MaskedDiffusionBlockGenerationMixin)
# ---------------------------------------------------------------------------


def prepare_for_sampling(
    x: torch.Tensor,
    block_size: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build block-causal attention mask and logical position IDs.

    Positions are grouped into blocks of ``block_size`` by their physical index.
    Position *k* can attend to any position in blocks 0 .. block(k).  Padding
    tokens are excluded from attention: they neither query nor serve as keys.

    Args:
        x: Token ID tensor of shape ``[B, T]``.
        block_size: Number of tokens per block.
        pad_token_id: Token ID used for padding (will be masked out).

    Returns:
        attn_mask: ``[B, 1, T, T]`` bool tensor — ``True`` means the query can
            attend to the key at that position.
        position_ids: ``[B, T]`` long tensor — 0-based logical positions; padding
            positions are set to 0.
    """
    B, T = x.shape
    device = x.device

    # Per-sample valid mask
    valid = x != pad_token_id  # [B, T]

    # Logical positions for RoPE (cumsum over valid tokens, 0-based)
    pos_raw = torch.cumsum(valid.to(torch.long), dim=-1)  # [B, T] 1-based
    logical_pos = pos_raw - 1  # [B, T] 0-based

    position_ids = torch.where(
        valid,
        logical_pos,
        torch.zeros_like(logical_pos),
    ).to(device=device, dtype=torch.long)  # [B, T]

    # Block IDs in physical coordinates (shared across batch)
    pos = torch.arange(T, device=device)
    block_ids = torch.div(pos, block_size, rounding_mode="floor")  # [T]
    block_ids = block_ids.view(1, T).expand(B, -1)  # [B, T]

    # Padding positions get a sentinel block ID of -1
    block_ids = torch.where(
        valid,
        block_ids,
        torch.full_like(block_ids, -1),
    )

    # Build [B, 1, T, T] mask: key-block <= query-block AND both valid
    bid_q = block_ids.view(B, 1, T, 1)  # query
    bid_k = block_ids.view(B, 1, 1, T)  # key

    valid_q = bid_q >= 0
    valid_k = bid_k >= 0

    attn_mask = (bid_k <= bid_q) & valid_q & valid_k  # [B, 1, T, T]

    # NOTE: padding query rows are all-False here (pads neither query nor
    # serve as keys).  Consumers that feed this mask to SDPA must make those
    # rows safe first — the BD3LM loop applies
    # ``masked_diffusion_block_mixin._pad_safe_attention_mask`` (run_attention's
    # ``no_allowed`` pattern) to avoid NaN softmax rows.
    return attn_mask, position_ids


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply Gumbel-max noise to logits for temperature sampling.

    Uses float64 for numerical stability, as recommended by arXiv:2409.02908.
    """
    if temperature == 0.0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _get_num_transfer_tokens(
    mask_index: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Compute the linear unmasking schedule for one block.

    Distributes the unmasking of ``mask_num`` tokens across ``steps`` steps
    with a uniform (linear) schedule.  Any remainder is added to the first step.

    Args:
        mask_index: ``[B, L]`` bool tensor marking masked positions.
        steps: Number of inner diffusion steps.

    Returns:
        ``[B, steps]`` int64 tensor of token counts to unmask per step.
    """
    B = mask_index.shape[0]
    device = mask_index.device
    mask_num = mask_index.sum(dim=1)  # [B]

    num_transfer = torch.zeros(B, steps, dtype=torch.long, device=device)
    for b in range(B):
        n = mask_num[b].item()
        if n == 0:
            continue
        base = int(n) // steps
        remainder = int(n) % steps
        num_transfer[b] = base
        if remainder > 0:
            num_transfer[b, :remainder] += 1
    return num_transfer


def _diffusion_step_block(
    logits: torch.Tensor,  # [B, L, V]
    x_block: torch.Tensor,  # [B, L]
    mask_block: torch.Tensor,  # [B, L] bool
    num_transfer_step: torch.Tensor,  # [B]
    temperature: float,
) -> torch.Tensor:
    """One inner diffusion step over a block slice.

    Samples candidate tokens from logits, computes confidence (softmax
    probability of the predicted token), and commits the top-k most confident
    masked positions.

    Args:
        logits: Logit tensor ``[B, L, V]``.
        x_block: Current token IDs for the block ``[B, L]``.
        mask_block: Boolean mask ``[B, L]`` — ``True`` means the position is
            still masked and can be committed.
        num_transfer_step: Number of tokens to commit per sample ``[B]``.
        temperature: Gumbel noise temperature.

    Returns:
        Updated ``x_block`` tensor with committed tokens filled in.
    """
    if not mask_block.any():
        return x_block

    B, L, _ = logits.shape
    device = logits.device

    # Sample tokens via Gumbel-max
    logits_noisy = _add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_noisy, dim=-1)  # [B, L]

    # Confidence: softmax probability of the sampled token at each masked position
    p = F.softmax(logits.float(), dim=-1)
    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # [B, L]

    # Only masked positions are candidates
    x0 = torch.where(mask_block, x0, x_block)
    confidence = torch.where(mask_block, x0_p, torch.full_like(x0_p, -float("inf")))

    # Select top-k positions per sample to commit
    transfer = torch.zeros(B, L, dtype=torch.bool, device=device)
    for j in range(B):
        k = int(num_transfer_step[j].item())
        if k <= 0:
            continue
        valid_count = (confidence[j] > -float("inf")).sum().item()
        if valid_count == 0:
            continue
        k = min(k, int(valid_count))
        _, sel = torch.topk(confidence[j], k)
        transfer[j, sel] = True

    x_block_new = x_block.clone()
    x_block_new[transfer] = x0[transfer]
    return x_block_new


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class MaskedDiffusionModelOutput(ModelOutput):
    """Output of :meth:`MaskedDiffusionGenerationMixin.generate`."""

    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.LongTensor, ...]] = None
    timing: Optional[Dict[str, Any]] = None


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
    use_cache : bool
        Enable block-wise KV-cache for faster inference (default: False).
    block_length : int or None
        Number of tokens to decode in each block. Must divide ``max_new_tokens``
        evenly. If None, defaults to single-block (``max_new_tokens``).
        Only used when ``use_cache=True`` (default: None).
    use_replace_cache : bool
        Use dual-cache mode (forward only current block). If False, use trim mode
        (forward from current block start to end). Only used when ``use_cache=True``
        (default: True). See Fast-dLLM paper for details.
    parallel_decode : bool
        Enable confidence-aware parallel decoding (Phase M.2). Unmask multiple
        confident tokens per step instead of one. Only used when ``use_cache=True``
        (default: False).
    confidence_threshold : float
        Minimum confidence for parallel unmasking (0-1). Tokens with confidence
        below this threshold are NOT unmasked in the current step. Higher values
        are more conservative (fewer parallel unmasks). Only used when
        ``parallel_decode=True`` (default: 0.9).
    confidence_type : str
        Type of confidence measure for parallel decoding. One of:
        - "max_prob": Maximum probability (default)
        - "margin": Top1 - Top2 probability margin
        - "neg_entropy": Negative entropy
        Only used when ``parallel_decode=True`` (default: "max_prob").
    output_history : bool
        If True and ``return_dict=True``, include per-step token sequences in
        the output (default: False).
    return_dict : bool
        Return a :class:`MaskedDiffusionModelOutput` instead of a plain tensor
        (default: False).
    right_shift_logits : bool, optional
        If ``True``, shift logits one position right before committing tokens
        (Dream-style compatibility for right-shifted label training).
        Defaults to ``False``.
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
        # cache control (Phase M.1)
        self.use_cache: bool = kwargs.pop("use_cache", False)
        self.block_length: Optional[int] = kwargs.pop("block_length", None)
        self.use_replace_cache: bool = kwargs.pop("use_replace_cache", True)
        # parallel decode (Phase M.2)
        self.parallel_decode: bool = kwargs.pop("parallel_decode", False)
        self.confidence_threshold: float = kwargs.pop("confidence_threshold", 0.9)
        self.confidence_type: str = kwargs.pop("confidence_type", "max_prob")
        # BD3LM block-diffusion generation
        self.use_block_diffusion: bool = kwargs.pop("use_block_diffusion", False)
        self.bd3lm_block_size: int = kwargs.pop("bd3lm_block_size", 32)
        self.cfg_scale: float = kwargs.pop("cfg_scale", 0.0)
        self.right_shift_logits: bool = kwargs.pop("right_shift_logits", False)
        # output control
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict: bool = kwargs.pop("return_dict", False)
        self.output_history: bool = kwargs.pop("output_history", False)
        self.output_timing: bool = kwargs.pop("output_timing", False)
        # step progress callback: called after each denoising step with (step, total)
        # where step is 1-indexed and total is the total number of steps.
        self.step_callback: Optional[Callable[[int, int], None]] = kwargs.pop(
            "step_callback", None
        )
        # stream callback: called after each step with (step, total, x)
        # where x is a detached clone snapshot of the current token IDs (LongTensor).
        self.stream_callback: Optional[Callable[[int, int, torch.LongTensor], None]] = (
            kwargs.pop("stream_callback", None)
        )
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
        """Validate generation config parameters."""
        if self.parallel_decode and not self.use_cache:
            raise ValueError(
                "`parallel_decode=True` requires `use_cache=True`. "
                "Confidence-aware parallel decoding relies on the KV-cache infrastructure. "
                "Please set `use_cache=True` or disable `parallel_decode`."
            )

        if self.parallel_decode and self.alg == "origin":
            raise ValueError(
                "`parallel_decode=True` does not support `alg='origin'`. "
                "Use one of 'maskgit_plus', 'topk_margin', or 'entropy'."
            )

        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}"
            )

        if self.confidence_type not in {"max_prob", "margin", "neg_entropy"}:
            raise ValueError(
                f"Unknown confidence_type={self.confidence_type!r}. "
                "Choose from 'max_prob', 'margin', 'neg_entropy'."
            )

        if self.use_block_diffusion and self.use_cache:
            raise ValueError(
                "`use_block_diffusion=True` and `use_cache=True` are mutually exclusive. "
                "BD3LM generation uses its own block-causal attention mask and cannot use "
                "the Fast-dLLM KV-cache path simultaneously. "
                "Use `use_block_diffusion=True` for BD3LM (A2D models) or "
                "`use_cache=True` for Fast-dLLM block-decode (LLaDA/Dream/A2D)."
            )

        if self.use_block_diffusion and getattr(self, "return_dict", False):
            raise ValueError(
                "`use_block_diffusion=True` does not support `return_dict=True`. "
                "BD3LM generation returns a plain tensor. Use `return_dict=False`."
            )


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------


class MaskedDiffusionGenerationMixin:
    """Generation mixin for MDLM-style masked diffusion LMs.

    Provides :meth:`generate` which implements the iterative masked-token
    denoising loop with algorithm dispatch.  The mixin is designed to be mixed
    into model classes as the *first* base class so that its :meth:`generate`
    takes precedence over HuggingFace's autoregressive ``generate``.

    Subclasses must implement a HF-compatible ``forward(input_ids, ...)`` that
    returns an object with a ``.logits`` attribute of shape ``[B, L, V]``.

    Unlike :class:`~unturtle.models.backbones.dream.DreamGenerationMixin`, **no** logit
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

    def _validate_generated_length(
        self, generation_config, input_ids_length, has_default_max_length
    ):
        if is_torchdynamo_compiling():
            return
        if (
            has_default_max_length
            and generation_config.max_new_tokens is None
            and generation_config.max_length == 20
        ):
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

    def _prepare_generated_length(
        self, generation_config, has_default_max_length, input_ids_length
    ):
        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence."
                )
            generation_config.max_length = (
                generation_config.max_new_tokens + input_ids_length
            )
        elif has_default_max_length:
            if (
                generation_config.max_length
                == MaskedDiffusionGenerationConfig().max_length
            ):
                generation_config.max_length = (
                    generation_config.max_length + input_ids_length
                )
                max_pos = getattr(self.config, "max_position_embeddings", None)
                if max_pos is not None:
                    generation_config.max_length = min(
                        generation_config.max_length, max_pos
                    )
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
                for attr in (
                    "bos_token_id",
                    "eos_token_id",
                    "pad_token_id",
                    "mask_token_id",
                ):
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
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        *,
        algorithm: str = "auto",
        generation_config: Optional[MaskedDiffusionGenerationConfig] = None,
        **kwargs,
    ) -> Union[MaskedDiffusionModelOutput, torch.LongTensor]:
        """Generate sequences via masked diffusion.

        ``algorithm`` selects the decoding path (``"auto"`` | ``"mdlm"`` |
        ``"block_decode"`` | ``"bd3lm"`` | ``"block_ar"``). ``block_ar``
        is the DiffusionGemma canvas family (no mask token) and is rejected
        here; the masked paths (mdlm/block_decode/bd3lm) resolve via capability
        checks and inject their flags (``use_cache`` / ``use_block_diffusion``)
        before the denoising loop runs.

        Parameters
        ----------
        inputs : LongTensor of shape ``[B, L]``
            Prompt token IDs.  Completion positions should already be filled
            with ``mask_token_id``.
        algorithm : str, optional
            Decoding algorithm to use (see main docstring for full list).
            ``"auto"`` (default) picks the fastest discrete path the model
            supports: block-decode when available, else plain MDLM; BD3LM when
            ``use_block_diffusion=True`` is set in kwargs. ``block_ar``
            (DiffusionGemma canvas) is rejected (raises ValueError).
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
        from unturtle.models.generation.sampler import (
            algorithm_to_flags,
            resolve_algorithm,
        )

        bd3lm_requested = bool(kwargs.get("use_block_diffusion", False)) or (
            algorithm == "bd3lm"
        )
        resolved = resolve_algorithm(algorithm, self, bd3lm_requested=bd3lm_requested)
        flags = algorithm_to_flags(resolved)
        kwargs = {**kwargs, **flags}

        generation_config = self._prepare_generation_config(generation_config, **kwargs)

        assert inputs is not None, "`inputs` (input_ids) must be provided"
        input_ids = inputs
        attention_mask = kwargs.pop("attention_mask", None)

        input_ids_length = input_ids.shape[-1]
        has_default_max_length = (
            kwargs.get("max_length") is None
            and generation_config.max_length is not None
        )
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )
        self._validate_generated_length(
            generation_config, input_ids_length, has_default_max_length
        )

        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with `input_ids` on a different device type than the model."
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

        If ``generation_config.use_cache=True``, delegates to
        :meth:`_sample_with_cache` for block-wise KV-cache optimization.
        """
        if generation_config.use_cache:
            return self._sample_with_cache(input_ids, attention_mask, generation_config)

        if generation_config.use_block_diffusion:
            return self._sample_block_diffusion(input_ids, generation_config)

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
                "`generate()`.  Pass it explicitly: "
                "`model.generate(inputs, mask_token_id=<id>, ...)`"
            )

        histories = [] if (return_dict_out and output_history) else None

        # Pad completion region with mask tokens
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            attention_mask = F.pad(
                attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0
            )
            # Broadcast to [B, 1, L, L] for SDPA
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            attention_mask = None

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)
        step_callback = generation_config.step_callback
        stream_callback = generation_config.stream_callback

        for i in range(steps):
            mask_index = x == mask_token_id

            # Forward pass — no logit shift (contrast with DreamGenerationMixin)
            logits = self(
                input_ids=x, attention_mask=attention_mask
            ).logits  # [B, L, V]

            mask_logits = logits[mask_index]  # [N_masked, V]
            t = timesteps[i]
            s = timesteps[i + 1]

            if alg == "origin":
                p_transfer = 1 - s / t if i < steps - 1 else 1.0
                x0 = torch.full_like(x[mask_index], mask_token_id, dtype=torch.long)
                transfer = torch.rand(*x0.shape, device=x.device) < p_transfer
                _, sampled = sample_tokens(
                    mask_logits[transfer],
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                x0[transfer] = sampled
                x[mask_index] = x0
            else:
                if alg == "maskgit_plus":
                    confidence, x0 = sample_tokens(
                        mask_logits, temperature=temperature, top_p=top_p, top_k=top_k
                    )
                elif alg == "topk_margin":
                    confidence, x0 = sample_tokens(
                        mask_logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        margin_confidence=True,
                    )
                elif alg == "entropy":
                    confidence, x0 = sample_tokens(
                        mask_logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        neg_entropy=True,
                    )
                else:
                    raise RuntimeError(
                        f"Unknown alg: {alg!r}. Choose from 'origin', 'maskgit_plus', 'topk_margin', 'entropy'."
                    )

                num_mask_token = mask_index.sum() / mask_index.shape[0]
                n_transfer = (
                    int(num_mask_token * (1 - s / t))
                    if i < steps - 1
                    else int(num_mask_token)
                )
                full_confidence = torch.full_like(x, -torch.inf, dtype=logits.dtype)
                full_confidence[mask_index] = confidence

                if n_transfer > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(full_confidence, n_transfer)
                    else:
                        full_confidence = full_confidence / alg_temp
                        full_confidence = F.softmax(full_confidence, dim=-1)
                        transfer_index = torch.multinomial(
                            full_confidence, num_samples=n_transfer
                        )

                    x_ = torch.full_like(x, mask_token_id, dtype=torch.long)
                    x_[mask_index] = x0
                    row_idx = (
                        torch.arange(x.size(0), device=x.device)
                        .unsqueeze(1)
                        .expand_as(transfer_index)
                    )
                    x[row_idx, transfer_index] = x_[row_idx, transfer_index]

            if histories is not None:
                histories.append(x.clone())

            if stream_callback is not None:
                try:
                    stream_callback(i + 1, steps, x.detach().clone())
                except Exception as _cb_exc:
                    logger.warning(
                        "stream_callback raised at step %d: %s", i + 1, _cb_exc
                    )

            if step_callback is not None:
                try:
                    step_callback(i + 1, steps)
                except Exception as _cb_exc:
                    logger.warning(
                        "step_callback raised at step %d: %s", i + 1, _cb_exc
                    )

        if return_dict_out:
            return MaskedDiffusionModelOutput(
                sequences=x,
                history=tuple(histories) if histories is not None else None,
            )
        return x

    def _sample_with_cache(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: MaskedDiffusionGenerationConfig,
    ) -> Union[MaskedDiffusionModelOutput, torch.LongTensor]:
        """Block-wise MDLM denoising loop with tuple cache trimming (Phase M.1).

        Implements Fast-dLLM-style block-decode:
        - Divides generation into blocks of ``block_length`` tokens
        - Initial forward: full sequence → cache
        - Trim cache to previous blocks only
        - Denoising loop: forward from ``current_block_start`` with trimmed cache
        - Cache stays constant during denoising (not updated per step)

        .. note::
           Phase M.1 implementation:
           - ``output_history`` is not supported (always returns ``history=None``)
           - Only ``alg='origin'`` supported (confidence-aware algorithms in M.2+)
           - Target speedup: ≥2.0x vs Phase L baseline

        Parameters
        ----------
        input_ids : LongTensor of shape [B, L]
            Prompt token IDs.
        attention_mask : LongTensor or None
            Attention mask (currently unused in cache path).
        generation_config : MaskedDiffusionGenerationConfig
            Generation config with ``use_cache=True`` and ``block_length``.

        Returns
        -------
        MaskedDiffusionModelOutput or LongTensor
            Generated sequences (with or without history, per ``return_dict``).
        """
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
        block_length = generation_config.block_length
        parallel_decode = generation_config.parallel_decode
        confidence_threshold = generation_config.confidence_threshold
        if parallel_decode and alg == "origin":
            raise ValueError(
                "`parallel_decode=True` does not support `alg='origin'`. "
                "Use one of 'maskgit_plus', 'topk_margin', or 'entropy'."
            )
        if parallel_decode and alg == "entropy":
            # Fast-dLLM's threshold mode uses max-probability confidence only
            # (dev/repos/fast-dllm/v1/dream/model/generation_utils_block.py
            # L495-524); negative-entropy confidences are <= 0 and never reach a
            # confidence_threshold in [0, 1].
            warnings.warn(
                "alg='entropy' uses negative-entropy confidences (<= 0), which never "
                "reach a confidence_threshold in [0, 1]; threshold-based parallel "
                "decode degenerates to one token per step (the max-confidence "
                "fallback). Use alg='maskgit_plus' or 'topk_margin' with "
                "parallel_decode, or disable parallel_decode for entropy ordering.",
                UserWarning,
            )

        if mask_token_id is None:
            mask_token_id = getattr(self.config, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError(
                "`mask_token_id` must be set in `generation_config` or `model.config` before calling "
                "`generate()` with `use_cache=True`."
            )

        # Warn if output_history is requested (not yet supported in cache path)
        if return_dict_out and generation_config.output_history:
            warnings.warn(
                "`output_history=True` is not yet supported with `use_cache=True` (Phase L). "
                "History will be None. Full history tracking will be implemented in Phase M.",
                UserWarning,
            )

        # Pad completion region with mask tokens
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        prompt_len = input_ids.shape[1]
        gen_length = max_length - prompt_len

        # Note: attention_mask handling for cache path is simplified in Phase M.1.
        # Bidirectional models typically do not need explicit masks for generation.
        attention_mask = None

        # Block-decode setup
        if block_length is None:
            block_length = gen_length  # Default: single block

        if gen_length % block_length != 0:
            raise ValueError(
                f"`gen_length` ({gen_length}) must be divisible by `block_length` ({block_length}). "
                f"Adjust `max_new_tokens` or `block_length` to satisfy this constraint."
            )

        num_blocks = gen_length // block_length
        steps_per_block = steps // num_blocks
        timesteps = torch.linspace(1, eps, steps_per_block + 1, device=x.device)
        step_callback = generation_config.step_callback
        stream_callback = generation_config.stream_callback
        total_steps = num_blocks * steps_per_block
        global_step = 0

        # Block-decode loop (Phase M.1: tuple cache with trimming)
        past_key_values = None

        for num_block in range(num_blocks):
            current_block_start = prompt_len + num_block * block_length
            current_block_end = current_block_start + block_length

            # Initial forward: full sequence (including all previous blocks)
            outputs = self(input_ids=x, attention_mask=attention_mask, use_cache=True)
            past_key_values = outputs.past_key_values

            # Trim cache to previous blocks only (Fast-dLLM approach)
            if past_key_values is not None:
                past_key_values = _trim_kv_cache(past_key_values, current_block_start)

            # Denoising loop for current block
            step_idx = 0
            max_block_iterations = block_length if parallel_decode else steps_per_block
            while step_idx < max_block_iterations:
                # Mask index for current block only
                mask_index_block = (
                    x[:, current_block_start:current_block_end] == mask_token_id
                )

                # Check if all tokens in block are unmasked
                n_masked_block = mask_index_block.sum().item()
                if n_masked_block == 0:
                    break  # Block complete

                # Forward pass with cache (from current_block_start onwards)
                if isinstance(past_key_values, tuple):
                    cache_obj = _tuple_to_cache(past_key_values, x.device)
                else:
                    cache_obj = past_key_values

                x_forward = x[:, current_block_start:]
                attn_mask_forward = (
                    attention_mask[:, current_block_start:]
                    if attention_mask is not None
                    else None
                )

                # Create cache_position for transformers compatibility
                # Tells the model where these tokens sit in the full sequence
                cache_position = torch.arange(
                    current_block_start,
                    current_block_start + x_forward.shape[1],
                    device=x.device,
                    dtype=torch.long,
                )

                outputs = self(
                    input_ids=x_forward,
                    attention_mask=attn_mask_forward,
                    past_key_values=cache_obj,
                    cache_position=cache_position,
                    use_cache=True,
                )
                logits = outputs.logits  # [B, L_block+, V]

                # Extract logits for current block
                block_logits = logits[:, :block_length, :]  # [B, block_length, V]
                mask_logits = block_logits[mask_index_block]  # [N_masked, V]
                if 0 <= mask_token_id < mask_logits.shape[-1]:
                    # Masked-diffusion denoising places zero mass on the mask token
                    # (MDLM SUBS "zero masking probabilities"). Without this, a
                    # committed token can be the mask sentinel itself, so the block
                    # never completes and mask tokens leak into the returned output.
                    mask_logits = mask_logits.clone()
                    mask_logits[:, mask_token_id] = torch.finfo(mask_logits.dtype).min

                # Do NOT update past_key_values here - it stays constant for all steps in this block

                if parallel_decode:
                    if alg == "maskgit_plus":
                        confidence, sampled = sample_tokens(
                            mask_logits,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            confidence_type="max_prob",
                        )
                    elif alg == "topk_margin":
                        confidence, sampled = sample_tokens(
                            mask_logits,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            confidence_type="margin",
                        )
                    elif alg == "entropy":
                        confidence, sampled = sample_tokens(
                            mask_logits,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            confidence_type="neg_entropy",
                        )
                    else:
                        raise RuntimeError(
                            f"Unknown alg: {alg!r}. Choose from 'maskgit_plus', 'topk_margin', 'entropy'."
                        )

                    current_block = x[:, current_block_start:current_block_end]
                    masked_confidence = torch.zeros(
                        current_block.shape,
                        dtype=logits.dtype,
                        device=current_block.device,
                    )
                    masked_confidence[mask_index_block] = confidence

                    transfer_index = select_threshold_transfer_mask(
                        masked_confidence=masked_confidence,
                        mask_index_block=mask_index_block,
                        threshold=confidence_threshold,
                    )

                    selected_mask = mask_index_block & transfer_index
                    if selected_mask.any():
                        current_block[selected_mask] = sampled[
                            selected_mask[mask_index_block]
                        ]
                else:
                    t = timesteps[step_idx]
                    s = timesteps[step_idx + 1]

                    if alg == "origin":
                        p_transfer = (
                            1 - s / t if step_idx < steps_per_block - 1 else 1.0
                        )
                        # Start from current block state (not all-mask) so previously
                        # unmasked tokens are preserved across denoising steps.
                        current_block = x[
                            :, current_block_start:current_block_end
                        ].clone()
                        masked_flat = current_block[
                            mask_index_block
                        ]  # 1-D, len = n_masked
                        transfer = (
                            torch.rand(masked_flat.shape[0], device=x.device)
                            < p_transfer
                        )
                        _, sampled = sample_tokens(
                            mask_logits[transfer],
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                        )
                        masked_flat[transfer] = sampled
                        current_block[mask_index_block] = masked_flat
                        x[:, current_block_start:current_block_end] = current_block
                    else:
                        if alg == "maskgit_plus":
                            confidence, sampled = sample_tokens(
                                mask_logits,
                                temperature=temperature,
                                top_p=top_p,
                                top_k=top_k,
                                confidence_type="max_prob",
                            )
                        elif alg == "topk_margin":
                            confidence, sampled = sample_tokens(
                                mask_logits,
                                temperature=temperature,
                                top_p=top_p,
                                top_k=top_k,
                                confidence_type="margin",
                            )
                        elif alg == "entropy":
                            confidence, sampled = sample_tokens(
                                mask_logits,
                                temperature=temperature,
                                top_p=top_p,
                                top_k=top_k,
                                confidence_type="neg_entropy",
                            )
                        else:
                            raise RuntimeError(
                                f"Unknown alg: {alg!r}. Choose from 'origin', 'maskgit_plus', 'topk_margin', 'entropy'."
                            )

                        num_mask_token = (
                            mask_index_block.sum() / mask_index_block.shape[0]
                        )
                        n_transfer = (
                            int(num_mask_token * (1 - s / t))
                            if step_idx < steps_per_block - 1
                            else int(num_mask_token)
                        )
                        full_confidence = torch.full_like(
                            x[:, current_block_start:current_block_end],
                            -torch.inf,
                            dtype=logits.dtype,
                        )
                        full_confidence[mask_index_block] = confidence

                        if n_transfer > 0:
                            if alg_temp is None or alg_temp == 0:
                                _, transfer_index = torch.topk(
                                    full_confidence, n_transfer
                                )
                            else:
                                full_confidence = full_confidence / alg_temp
                                full_confidence = F.softmax(full_confidence, dim=-1)
                                transfer_index = torch.multinomial(
                                    full_confidence, num_samples=n_transfer
                                )

                            sampled_block = torch.full_like(
                                x[:, current_block_start:current_block_end],
                                mask_token_id,
                                dtype=torch.long,
                            )
                            sampled_block[mask_index_block] = sampled
                            row_idx = (
                                torch.arange(x.size(0), device=x.device)
                                .unsqueeze(1)
                                .expand_as(transfer_index)
                            )
                            x[:, current_block_start:current_block_end][
                                row_idx, transfer_index
                            ] = sampled_block[row_idx, transfer_index]

                global_step += 1
                if stream_callback is not None:
                    try:
                        stream_callback(global_step, total_steps, x.detach().clone())
                    except Exception as _cb_exc:
                        logger.warning(
                            "stream_callback raised at step %d: %s",
                            global_step,
                            _cb_exc,
                        )
                if step_callback is not None:
                    try:
                        step_callback(global_step, total_steps)
                    except Exception as _cb_exc:
                        logger.warning(
                            "step_callback raised at step %d: %s", global_step, _cb_exc
                        )
                step_idx += 1

        if return_dict_out:
            return MaskedDiffusionModelOutput(sequences=x, history=None)
        return x


__all__ = [
    "MaskedDiffusionGenerationConfig",
    "MaskedDiffusionGenerationMixin",
    "MaskedDiffusionModelOutput",
    "sample_tokens",
    "top_p_logits",
    "top_k_logits",
    "prepare_for_sampling",
]
