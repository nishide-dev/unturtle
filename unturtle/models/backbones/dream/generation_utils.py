# Copyright 2024 The Dream team, HKUNLP Group and the HuggingFace Inc. team. All rights reserved.
# Ported to Unturtle by nishide-dev & the Unturtle team (2025).
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

import copy
import warnings
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, Union, cast

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)

from unturtle.models.generation.block_decode_mixin import BlockDecodeMixin

logger = logging.get_logger(__name__)


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    if top_k is None:
        return logits
    top_k = min(cast(int, top_k), logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(
    logits,
    temperature=0.0,
    top_p=None,
    top_k=None,
    margin_confidence=False,
    neg_entropy=False,
):

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
        except (ValueError, RuntimeError):
            # probs may not sum to 1.0 exactly after softmax (ValueError from Categorical
            # validate_args), or contain NaN/inf when validate_args is disabled (RuntimeError
            # from torch.multinomial); fall back to argmax
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: Optional[torch.LongTensor] = None
    history: Optional[Tuple[torch.FloatTensor]] = None
    timing: Optional[Dict[str, Any]] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", "origin")
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)
        # cache control
        self.use_cache: bool = kwargs.pop("use_cache", False)
        self.block_length: Optional[int] = kwargs.pop("block_length", None)
        self.use_replace_cache: bool = kwargs.pop("use_replace_cache", True)
        # parallel decode
        self.parallel_decode: bool = kwargs.pop("parallel_decode", False)
        self.confidence_threshold: float = kwargs.pop("confidence_threshold", 0.9)
        self.confidence_type: str = kwargs.pop("confidence_type", "max_prob")

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict: bool = kwargs.pop("return_dict", False)
        self.output_history: bool = kwargs.pop("output_history", False)
        self.output_timing: bool = kwargs.pop("output_timing", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        self.validate(is_init=True)

    @classmethod
    def from_model_config(cls, model_config) -> "DreamGenerationConfig":
        """Build the default DreamGenerationConfig from a model config.

        transformers 5.x ``GenerationConfig.from_model_config`` hardcodes
        ``default_generation_config = GenerationConfig()`` (the BASE class) and
        then reads every attribute of the *subclass* instance off it — for any
        subclass with extra fields (``eps``, ``steps``, …) that raises
        ``AttributeError`` (#189). This override reproduces the construction
        core (``from_dict`` with the subclass) and skips the base-default
        comparison loop; every model-config attribute the loop could have
        copied is already consumed by ``from_dict``/``__init__`` (Dream pops
        ``mask/pad/bos/eos_token_id`` explicitly).
        """
        config_dict = (
            model_config.to_dict()
            if not isinstance(model_config, dict)
            else dict(model_config)
        )
        config_dict.pop("_from_model_config", None)
        # None values fall back to the generation-config defaults, as upstream.
        config_dict = {k: v for k, v in config_dict.items() if v is not None}
        generation_config = cls.from_dict(
            config_dict, return_unused_kwargs=False, _from_model_config=True
        )
        generation_config._original_object_hash = hash(generation_config)
        return generation_config

    def validate(self, is_init: bool = False, **kwargs):
        if self.parallel_decode and not self.use_cache:
            raise ValueError("`parallel_decode=True` requires `use_cache=True`.")
        if self.parallel_decode and self.alg == "origin":
            raise ValueError("`parallel_decode=True` does not support `alg='origin'`.")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}"
            )
        if self.confidence_type not in {"max_prob", "margin", "neg_entropy"}:
            raise ValueError(
                f"Unknown confidence_type={self.confidence_type!r}. Choose from 'max_prob', 'margin', 'neg_entropy'."
            )


class DreamGenerationMixin(BlockDecodeMixin):
    config: Any
    generation_config: Any

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[Optional[torch.LongTensor], Optional[torch.LongTensor]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
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
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if (
            has_default_max_length
            and generation_config.max_new_tokens is None
            and generation_config.max_length == 20
        ):
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = (
                generation_config.max_new_tokens + input_ids_length
            )

        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = (
                    generation_config.max_length + input_ids_length
                )
                max_position_embeddings = getattr(
                    self.config, "max_position_embeddings", None
                )
                if max_position_embeddings is not None:
                    generation_config.max_length = min(
                        generation_config.max_length, max_position_embeddings
                    )

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # Priority (#189, explicit contract):
        #   1. the explicit ``generation_config`` argument;
        #   2. the model-attached ``self.generation_config`` (checkpoint
        #      defaults — a Dream checkpoint attaches DreamGenerationConfig via
        #      the from_pretrained family hook; an in-memory model carries the
        #      plain GenerationConfig from the transformers 5.x postamble);
        #   3. ``DreamGenerationConfig.from_model_config(self.config)``.
        # Whatever the source, the result is a DreamGenerationConfig: a plain
        # GenerationConfig is adopted field-for-field with the Dream diffusion
        # defaults (eps/steps/alg/…) filled in — previously such a config
        # crashed the sampler with AttributeError: eps.
        using_model_generation_config = False
        if generation_config is None:
            attached = getattr(self, "generation_config", None)
            if attached is not None:
                generation_config = attached
            else:
                generation_config = cast(
                    DreamGenerationConfig,
                    DreamGenerationConfig.from_model_config(self.config),
                )
            using_model_generation_config = True
        if not isinstance(generation_config, DreamGenerationConfig):
            # None entries fall back to the Dream defaults (a plain 5.x config
            # serializes unset fields as None, e.g. num_return_sequences).
            generation_config = DreamGenerationConfig(
                **{
                    k: v
                    for k, v in generation_config.to_dict().items()
                    if v is not None
                }
            )

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = cast(
                DreamGenerationConfig, copy.deepcopy(generation_config)
            )
            _kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = (
                        self.generation_config.mask_token_id
                    )

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.

        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(
            generation_config.bos_token_id, device=device
        )
        eos_token_tensor = _tensor_or_none(
            generation_config.eos_token_id, device=device
        )
        pad_token_tensor = _tensor_or_none(
            generation_config.pad_token_id, device=device
        )
        mask_token_tensor = _tensor_or_none(
            generation_config.mask_token_id, device=device
        )

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(
                f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation."
            )

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        *,
        algorithm: str = "auto",
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        """Run masked-diffusion generation.

        ``algorithm`` selects the decode path: ``"auto"`` (default) resolves to
        ``"block_decode"`` when the model supports it, else ``"mdlm"``;
        ``"mdlm"`` and ``"block_decode"`` are also accepted explicitly.
        ``"bd3lm"`` raises ``ValueError`` — Dream does not implement
        ``_sample_block_diffusion``.  The resolved algorithm's flags are injected
        into kwargs, overriding any conflicting ``generation_config`` fields.
        """
        from unturtle.models.generation.sampler import (
            algorithm_to_flags,
            resolve_algorithm,
        )

        bd3lm_requested = bool(kwargs.get("use_block_diffusion", False)) or (
            algorithm == "bd3lm"
        )
        resolved = resolve_algorithm(algorithm, self, bd3lm_requested=bd3lm_requested)
        kwargs = {**kwargs, **algorithm_to_flags(resolved)}

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", None)
        generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", None)
        has_tokens_hook = generation_tokens_hook_func is not None
        has_logits_hook = generation_logits_hook_func is not None
        if generation_tokens_hook_func is None:
            generation_tokens_hook_func = lambda step, x, logits: x
        if generation_logits_hook_func is None:
            generation_logits_hook_func = lambda step, x, logits: logits

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
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

        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
            getattr(generation_config, "pad_token_id", None) is not None
            and torch.any(input_ids == generation_config.pad_token_id)
            and attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if generation_config.use_cache:
            if has_tokens_hook:
                raise NotImplementedError(
                    "Dream cache-enabled block decode does not support generation_tokens_hook_func."
                )
            if has_logits_hook:
                raise NotImplementedError(
                    "Dream cache-enabled block decode does not support generation_logits_hook_func."
                )
            return self._sample_with_cache(
                input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )

        result = self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            generation_tokens_hook_func=generation_tokens_hook_func,
            generation_logits_hook_func=generation_logits_hook_func,
        )
        return result

    def _sample_with_cache(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        if attention_mask is not None:
            max_length = generation_config.max_length
            prompt_len = input_ids.shape[1]
            gen_length = max_length - prompt_len
            if attention_mask.dim() == 2:
                if attention_mask.dtype == torch.bool:
                    padded_mask = F.pad(
                        attention_mask.to(input_ids.device), (0, gen_length), value=True
                    )
                    attention_mask = torch.logical_and(
                        padded_mask.unsqueeze(1).unsqueeze(-2),
                        padded_mask.unsqueeze(1).unsqueeze(-1),
                    )
                elif torch.is_floating_point(attention_mask):
                    if torch.any(attention_mask != 0.0):
                        raise ValueError(
                            "Dream cache-enabled block decode expects a boolean padding mask for 2D attention masks; "
                            "pass an expanded 4D additive mask instead."
                        )
                    attention_mask = None
                elif torch.any(attention_mask == 0):
                    padded_mask = F.pad(
                        attention_mask.to(input_ids.device), (0, gen_length), value=1
                    )
                    attention_mask = torch.logical_and(
                        padded_mask.unsqueeze(1).unsqueeze(-2),
                        padded_mask.unsqueeze(1).unsqueeze(-1),
                    )
                else:
                    attention_mask = None
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1).to(input_ids.device)
            elif attention_mask.dim() == 4:
                attention_mask = attention_mask.to(input_ids.device)
            else:
                raise ValueError(
                    f"Unexpected attention_mask shape: {attention_mask.shape}"
                )

            if attention_mask is not None and attention_mask.shape[-1] == prompt_len:
                if attention_mask.dtype == torch.bool:
                    expanded_mask = torch.ones(
                        attention_mask.shape[:2] + (max_length, max_length),
                        dtype=torch.bool,
                        device=attention_mask.device,
                    )
                    expanded_mask[:, :, :prompt_len, :prompt_len] = attention_mask
                    expanded_mask[:, :, prompt_len:, :prompt_len] = attention_mask[
                        :, :, -1:, :
                    ]
                    attention_mask = expanded_mask
                else:
                    raise ValueError(
                        "Dream cache-enabled block decode requires expanded additive masks to already cover max_length."
                    )
        else:
            attention_mask = None

        if generation_config.output_history:
            raise NotImplementedError(
                "Dream cache-enabled block decode does not support output_history."
            )

        block_decode_output = self._block_decode_loop(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )
        timing = None
        if isinstance(block_decode_output, tuple):
            x, timing = block_decode_output
        else:
            x = block_decode_output

        if generation_config.return_dict:
            return DreamModelOutput(sequences=x, history=None, timing=timing)
        return x

    def _get_block_decode_query_start(
        self,
        current_block_start: int,
        current_block_end: int,
        use_replace_cache: bool,
    ) -> int:
        del current_block_end, use_replace_cache
        return max(0, current_block_start - 1)

    def _postprocess_block_decode_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

    def _model_forward_with_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Any],
        use_cache: bool,
        replace_position: Optional[torch.Tensor] = None,
    ):
        dual_cache = replace_position is not None
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            dual_cache=dual_cache,
            replace_position=replace_position,
        )
        cache = outputs.past_key_values
        if cache is not None and not isinstance(cache, tuple):
            cache = tuple(cache)
        return SimpleNamespace(logits=outputs.logits, past_key_values=cache)

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values

        output_history = generation_config.output_history
        return_dict = generation_config.return_dict
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        histories = [] if (return_dict and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(
                attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0
            )
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        # this allows user-defined token control of the intermediate steps
        x = generation_tokens_hook_func(None, x, None)
        for i in range(steps):
            mask_index = x == mask_token_id
            logits = self(x, attention_mask, tok_idx).logits
            # Intentional right-shift from the original Dream implementation:
            # Dream's training uses a shifted-logits objective where logit[i]
            # is trained to predict the token *before* position i (see Dream
            # paper / official repo). This shift is required for correct
            # sampling even though Dream uses bidirectional attention.
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            # this allows user-defined logits control of the intermediate steps
            logits = generation_logits_hook_func(i, x, logits)

            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]

            if alg == "origin":
                p_transfer = 1 - s / t if i < steps - 1 else 1
                x0 = (
                    torch.zeros_like(
                        x[mask_index], device=self.device, dtype=torch.long
                    )
                    + mask_token_id
                )
                transfer_index_t_s = (
                    torch.rand(*x0.shape, device=self.device) < p_transfer
                )
                _, x0[transfer_index_t_s] = sample_tokens(
                    mask_logits[transfer_index_t_s],
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                x[mask_index] = x0.clone()
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
                        temperature,
                        top_p=top_p,
                        top_k=top_k,
                        neg_entropy=True,
                    )
                else:
                    raise RuntimeError(f"Unknown alg: {alg}")
                num_mask_token = mask_index.sum() / mask_index.shape[0]
                number_transfer_tokens = (
                    int(num_mask_token * (1 - s / t))
                    if i < steps - 1
                    else int(num_mask_token)
                )
                full_confidence = torch.full_like(
                    x, -torch.inf, device=self.device, dtype=logits.dtype
                )
                full_confidence[mask_index] = confidence
                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(
                            full_confidence, number_transfer_tokens
                        )
                    else:
                        full_confidence = full_confidence / alg_temp
                        full_confidence = F.softmax(full_confidence, dim=-1)
                        transfer_index = torch.multinomial(
                            full_confidence, num_samples=number_transfer_tokens
                        )
                    x_ = (
                        torch.zeros_like(x, device=self.device, dtype=torch.long)
                        + mask_token_id
                    )
                    x_[mask_index] = x0.clone()
                    row_indices = (
                        torch.arange(x.size(0), device=self.device)
                        .unsqueeze(1)
                        .expand_as(transfer_index)
                    )
                    x[row_indices, transfer_index] = x_[row_indices, transfer_index]

            # this allows user-defined token control of the intermediate steps
            x = generation_tokens_hook_func(i, x, logits)

            if histories is not None:
                histories.append(x.clone())

        if return_dict:
            return DreamModelOutput(
                sequences=x,
                history=histories,
            )
        else:
            return x
