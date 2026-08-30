"""#189 — Dream generation defaults on transformers 5.x.

Defect (frozen in the #184 artifact): a default-config ``model.generate`` on
Dream raised ``AttributeError: 'GenerationConfig' object has no attribute
'eps'`` — transformers 5.x ``from_model_config`` walks the subclass's attrs
against a hardcoded BASE-class default instance, and the None branch never
consulted the model-attached config.

Fixed contract (documented in ``_prepare_generation_config``):
  1. explicit ``generation_config`` argument
  2. model-attached ``self.generation_config``
  3. ``DreamGenerationConfig.from_model_config(self.config)``
with every source normalized to a ``DreamGenerationConfig`` (plain configs are
adopted field-for-field, Dream diffusion defaults filled, None entries falling
back to defaults).

``oracle_prepare_generation_config`` below is the *verbatim* pre-fix
implementation from ``main`` (dedented, renamed): the explicit-config route
must be unchanged against it — config dict, outputs, seed, attention mask and
length kwargs bit-for-bit.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional, cast

import pytest
import torch
from transformers import GenerationConfig
from transformers.utils import is_torchdynamo_compiling

from unturtle.models.backbones.dream.configuration_dream import DreamConfig
from unturtle.models.backbones.dream.generation_utils import (
    DreamGenerationConfig,
    DreamGenerationMixin,
)
from unturtle.models.backbones.dream.modeling_dream import DreamModel

pytestmark = [pytest.mark.gpu]  # unsloth import chain


# --- ORACLE: verbatim pre-fix implementation (main @ 0e36df7) -----------------
def oracle_prepare_generation_config(
    self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
) -> DreamGenerationConfig:
    """
    Prepares the base generation config, then applies any generation configuration options from kwargs. This
    function handles retrocompatibility with respect to configuration files.
    """
    # priority: `generation_config` argument > `model.generation_config` (the default generation config)
    using_model_generation_config = False
    if generation_config is None:
        generation_config = cast(
            DreamGenerationConfig,
            DreamGenerationConfig.from_model_config(self.config),
        )
        using_model_generation_config = True

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
                generation_config.mask_token_id = self.generation_config.mask_token_id

    return generation_config


# -----------------------------------------------------------------------------


def _tiny_dream(seed: int = 0):
    torch.manual_seed(seed)
    config = DreamConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        mask_token_id=1,
        pad_token_id=0,
    )
    return DreamModel(config).eval()


def _explicit(**overrides):
    base = dict(max_new_tokens=8, steps=4, mask_token_id=1, pad_token_id=0)
    base.update(overrides)
    return DreamGenerationConfig(**base)


# ============================ THE DEFECT, FIXED ============================


def test_default_config_generate_no_longer_raises_eps():
    model = _tiny_dream()
    assert type(model.generation_config).__name__ == "GenerationConfig"  # 5.x postamble
    ids = torch.randint(2, 500, (1, 8))
    torch.manual_seed(7)
    out = model.generate(ids, max_new_tokens=8, steps=4)
    assert tuple(out.shape) == (1, 16)
    # not silently degenerate: the mask token must have flowed (a None
    # mask_token_id pads with 0 and unmasks nothing — #189 review)
    generated = out[:, 8:]
    assert not bool((generated == 0).all())
    torch.manual_seed(7)
    out_mdlm = model.generate(ids, algorithm="mdlm", max_new_tokens=8, steps=4)
    assert not bool((out_mdlm[:, 8:] == 0).all())


def test_attached_plain_config_recovers_special_tokens_from_model_config():
    """An in-memory model's attached config is a plain 5.x GenerationConfig
    with NO mask_token_id field — the prepared config must recover it (and
    pad) from self.config, never leave it None."""
    model = _tiny_dream()
    assert not hasattr(model.generation_config, "mask_token_id")
    prepared = model._prepare_generation_config(None)
    assert prepared.mask_token_id == model.config.mask_token_id
    assert prepared.pad_token_id == model.config.pad_token_id


def test_default_route_equals_the_explicit_equivalent_bit_for_bit():
    model = _tiny_dream()
    ids = torch.randint(2, 500, (1, 8))
    torch.manual_seed(7)
    default_out = model.generate(ids, max_new_tokens=8, steps=4)
    torch.manual_seed(7)
    explicit_out = model.generate(ids, generation_config=_explicit())
    assert torch.equal(default_out, explicit_out)


def test_from_model_config_is_5x_safe_and_keeps_dream_fields():
    model = _tiny_dream()
    gc = DreamGenerationConfig.from_model_config(model.config)
    assert type(gc) is DreamGenerationConfig
    assert gc.eps == 1e-3 and gc.steps == 512
    assert gc.mask_token_id == model.config.mask_token_id
    assert gc.pad_token_id == model.config.pad_token_id
    # None entries in the model config must fall back to the generation
    # defaults (upstream contract), not overwrite them with None
    gc_none = DreamGenerationConfig.from_model_config(
        {"steps": None, "mask_token_id": 1}
    )
    assert gc_none.steps == 512 and gc_none.mask_token_id == 1
    # the 5.x base implementation crashes on exactly this call
    with pytest.raises(AttributeError, match="eps"):
        GenerationConfig.from_model_config.__func__(DreamGenerationConfig, model.config)


# ============================ PRIORITY CONTRACT ============================


def test_explicit_config_beats_the_attached_config():
    model = _tiny_dream()
    model.generation_config = _explicit(steps=5)
    seen = {}
    original = (
        DreamGenerationMixin._sample_with_cache
    )  # "auto" resolves to block decode

    def spy(self, ids, attention_mask=None, generation_config=None, **kw):
        seen["steps"] = generation_config.steps
        return original(
            self,
            ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            **kw,
        )

    DreamGenerationMixin._sample_with_cache = spy
    try:
        ids = torch.randint(2, 500, (1, 8))
        model.generate(ids, generation_config=_explicit(steps=2))
        assert seen["steps"] == 2
        model.generate(ids)  # None -> attached
        assert seen["steps"] == 5
    finally:
        DreamGenerationMixin._sample_with_cache = original


def test_attached_plain_config_is_adopted_with_dream_defaults():
    model = _tiny_dream()
    plain = GenerationConfig(max_new_tokens=8, pad_token_id=0, eos_token_id=3)
    model.generation_config = plain
    prepared = model._prepare_generation_config(None, steps=4, mask_token_id=1)
    assert type(prepared) is DreamGenerationConfig
    assert prepared.eps == 1e-3  # Dream default filled
    assert prepared.steps == 4  # kwargs applied
    assert prepared.max_new_tokens == 8 and prepared.eos_token_id == 3  # adopted
    # num_return_sequences serializes as None on a plain 5.x config — must
    # fall back to the Dream default, not None (the repeat_interleave crash)
    assert prepared.num_return_sequences == 1


def test_unattached_model_falls_back_to_from_model_config():
    model = _tiny_dream()
    model.generation_config = None
    prepared = model._prepare_generation_config(None)
    assert type(prepared) is DreamGenerationConfig
    assert prepared.mask_token_id == model.config.mask_token_id


def test_attached_config_is_never_mutated_by_generate_kwargs():
    model = _tiny_dream()
    attached = _explicit(steps=5)
    model.generation_config = attached
    ids = torch.randint(2, 500, (1, 8))
    model.generate(ids, steps=2, max_new_tokens=4)
    assert attached.steps == 5 and attached.max_new_tokens == 8  # deepcopy held


# ======================= UNCHANGED vs VERBATIM ORACLE =======================


def test_explicit_route_config_matches_the_oracle_verbatim():
    """For an explicit DreamGenerationConfig — the only route that worked
    before — the prepared config is oracle-identical across the output/seed/
    length kwargs surface."""
    model = _tiny_dream()
    for kwargs in (
        {},
        {"max_new_tokens": 4},
        {"max_length": 24},
        {"steps": 2, "temperature": 0.7, "top_p": 0.9},
        {"num_return_sequences": 2},
    ):
        explicit = _explicit()
        new = model._prepare_generation_config(explicit, **dict(kwargs))
        old = oracle_prepare_generation_config(model, explicit, **dict(kwargs))
        assert new.to_dict() == old.to_dict(), kwargs


def test_explicit_route_outputs_match_the_oracle_bit_for_bit():
    """Full generate on the explicit route, oracle-prepared vs fix-prepared:
    outputs, seed handling, attention mask and length behavior unchanged."""
    model = _tiny_dream()
    ids = torch.randint(2, 500, (2, 8))
    attention_mask = torch.ones_like(ids)
    attention_mask[1, :3] = 0

    def run(prepare):
        original = DreamGenerationMixin._prepare_generation_config
        DreamGenerationMixin._prepare_generation_config = prepare
        try:
            torch.manual_seed(11)
            return model.generate(
                ids,
                generation_config=_explicit(),
                attention_mask=attention_mask,
                max_new_tokens=6,
            )
        finally:
            DreamGenerationMixin._prepare_generation_config = original

    new_out = run(DreamGenerationMixin._prepare_generation_config)
    old_out = run(oracle_prepare_generation_config)
    assert new_out.shape == old_out.shape
    assert torch.equal(new_out, old_out)
