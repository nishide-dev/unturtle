"""#186 — the explicit generation runner contract.

``oracle_call_sampling_loop`` is the *verbatim* signature-guessing invoker
from ``main`` (renamed): every masked route must produce bit-identical outputs
under the explicit declared contract, and the contract must be immune to the
signature-hiding failure the #184 artifact froze.

Also pinned here: the canvas (block_ar) runner invokes the upstream loop
explicitly — through ``_generate_canvas`` on the wrapper class, or the
class-level ``generate`` on a plain upstream instance (instance-level shims,
e.g. unsloth's fast-generate, can never hijack it) — and the DiffusionGemma
shim routes through ``dispatch_generation`` (a replaced runner IS what runs).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from unturtle.models.generation import sampler
from unturtle.models.generation.sampler import (
    GenerationRequest,
    dispatch_generation,
)

pytestmark = [pytest.mark.gpu]  # unsloth import chain


# --- ORACLE: verbatim pre-#186 signature-guessing invoker (main) --------------
def oracle_call_sampling_loop(method: Any, request: GenerationRequest) -> Any:
    """Invoke a sampling loop, matching whatever signature it declares.

    The loops do not share one shape: ``_sample`` and ``_sample_with_cache``
    take ``attention_mask`` as a *required positional*, ``_sample_block_diffusion``
    takes it as a keyword with a default, and Dream's ``_sample`` additionally
    requires two hook callables.  Binding by inspection keeps the runners
    generic instead of encoding one backbone's argument order.

    Anything the loop does not declare stays in ``kwargs`` and is forwarded, so
    a loop with extra options still receives them.
    """
    import inspect

    parameters = inspect.signature(method).parameters
    kwargs = dict(request.kwargs)
    args: list[Any] = []

    for name, parameter in parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if name in ("input_ids", "inputs"):
            args.append(request.inputs)
        elif name == "generation_config":
            args.append(request.generation_config)
        elif name in kwargs:
            args.append(kwargs.pop(name))
        elif parameter.default is not inspect.Parameter.empty:
            args.append(parameter.default)
        else:
            # Required and unsupplied: pass None rather than TypeError-ing on
            # a positional the caller simply did not set (e.g. attention_mask,
            # or Dream's hook functions, which its loop defaults internally).
            args.append(None)

    return method(*args, **kwargs)


# -----------------------------------------------------------------------------


def _tiny_dream():
    from unturtle.models.backbones.dream.configuration_dream import DreamConfig
    from unturtle.models.backbones.dream.modeling_dream import DreamModel

    torch.manual_seed(0)
    return DreamModel(
        DreamConfig(
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
    ).eval()


def _tiny_a2d():
    from unturtle.models.conversion.a2d.tiny_a2d import (
        TinyA2DLlamaConfig,
        TinyA2DLlamaLMHeadModel,
    )

    torch.manual_seed(0)
    return TinyA2DLlamaLMHeadModel(
        TinyA2DLlamaConfig(
            vocab_size=512,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            mask_token_id=3,
            pad_token_id=0,
        )
    ).eval()


def _tiny_llada():
    from unturtle.models.backbones.llada.configuration_llada import LLaDAConfig
    from unturtle.models.backbones.llada.modeling_llada import LLaDAModelLM

    torch.manual_seed(0)
    return LLaDAModelLM(
        LLaDAConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            mlp_hidden_size=128,
            vocab_size=512,
            embedding_size=512,
            max_sequence_length=64,
            block_type="llama",
            activation_type="silu",
            rope=True,
            include_bias=False,
            include_qkv_bias=False,
            weight_tying=False,
            mask_token_id=1,
            pad_token_id=0,
        )
    ).eval()


ROUTES = [
    ("dream", _tiny_dream, "mdlm", {}),
    ("dream", _tiny_dream, "block_decode", {"block_length": 4}),
    ("tiny_a2d", _tiny_a2d, "mdlm", {}),
    ("tiny_a2d", _tiny_a2d, "block_decode", {"block_length": 4}),
    ("tiny_a2d", _tiny_a2d, "bd3lm", {"block_length": 4}),
    ("llada", _tiny_llada, "mdlm", {}),
    ("llada", _tiny_llada, "block_decode", {"block_length": 4}),
]


@pytest.mark.parametrize(
    "family,builder,algorithm,extra", ROUTES, ids=[f"{f}-{a}" for f, b, a, e in ROUTES]
)
def test_every_masked_route_matches_the_guessing_oracle_bit_for_bit(
    monkeypatch, family, builder, algorithm, extra
):
    """Per-route differential: cache / block-decode / config / kwargs
    forwarding under the explicit contract equals the old inspection-based
    binding, output bit-for-bit (same seed, batch attention mask)."""
    model = builder()
    ids = torch.randint(4, 500, (2, 8))
    attention_mask = torch.ones_like(ids)
    attention_mask[1, :2] = 0
    kwargs = dict(
        max_new_tokens=8,
        steps=4,
        attention_mask=attention_mask,
        **extra,
    )

    def run(invoker):
        monkeypatch.setattr(sampler, "_call_sampling_loop", invoker)
        torch.manual_seed(11)
        out = model.generate(ids, algorithm=algorithm, **dict(kwargs))
        monkeypatch.undo()
        return out

    new_out = run(sampler._call_sampling_loop)
    old_out = run(oracle_call_sampling_loop)
    assert new_out.shape == old_out.shape
    assert torch.equal(new_out, old_out)


# ============================== CANVAS RUNNER ==============================


class _CanvasStub:
    """Upstream-class-like: _denoising_step probe, class-level generate."""

    def __init__(self):
        self.calls = []

    def _denoising_step(self):  # capability probe
        raise NotImplementedError

    def generate(self, input_ids=None, generation_config=None, **kwargs):
        self.calls.append(("class-generate", input_ids, generation_config, kwargs))
        return "canvas-out"


class _WrappedCanvasStub(_CanvasStub):
    """Wrapper-class-like: exposes the explicit _generate_canvas target."""

    def _generate_canvas(self, inputs=None, *, generation_config=None, **kwargs):
        self.calls.append(("canvas", inputs, generation_config, kwargs))
        return "canvas-out"


def test_block_ar_runner_prefers_the_explicit_canvas_target():
    model = _WrappedCanvasStub()
    out = dispatch_generation(
        model,
        GenerationRequest(inputs="IDS", generation_config="CFG", kwargs={"k": 1}),
        algorithm="block_ar",
    )
    assert out == "canvas-out"
    assert model.calls == [("canvas", "IDS", "CFG", {"k": 1})]


def test_block_ar_runner_uses_the_class_level_generate_never_instance_shims():
    """A plain upstream instance runs its CLASS generate; an instance-level
    shim (unsloth fast-generate style) is never consulted — the behavior the
    removed runtime swap used to repair by popping the attribute."""
    model = _CanvasStub()
    hijack = []
    model.__dict__["generate"] = lambda *a, **k: hijack.append(1) or "hijacked"
    out = dispatch_generation(
        model,
        GenerationRequest(inputs="IDS", generation_config=None, kwargs={}),
        algorithm="block_ar",
    )
    assert out == "canvas-out"
    assert hijack == []
    assert model.calls and model.calls[0][0] == "class-generate"


def test_gemma_shim_routes_through_dispatch_not_super_generate(monkeypatch):
    """Runner-bypass gate: the wrapper's unified generate must execute through
    dispatch_generation — a replaced runner IS what runs. (The pre-#186 shim
    called super().generate directly, bypassing any registered runner.)"""
    from unturtle.models.backbones.diffusion_gemma.modeling import (
        UnturtleDiffusionGemmaForBlockDiffusion,
    )

    seen = {}

    def sentinel_dispatch(model, request, algorithm="auto", **kw):
        seen["algorithm"] = algorithm
        seen["inputs"] = request.inputs
        seen["kwargs"] = dict(request.kwargs)
        return "runner-output"

    monkeypatch.setattr(sampler, "dispatch_generation", sentinel_dispatch)

    class _Shell:
        pass

    stub = _Shell()
    out = UnturtleDiffusionGemmaForBlockDiffusion.generate(
        stub, "IDS", algorithm="block_ar", generation_config=None, opt=1
    )
    assert out == "runner-output"
    assert seen["algorithm"] == "block_ar"
    assert seen["inputs"] == "IDS"
    assert seen["kwargs"] == {"opt": 1}


def test_gemma_canvas_target_is_the_verbatim_upstream_delegation():
    """_generate_canvas performs NO resolution — it is the single upstream
    generate call site of the wrapper."""
    from unturtle.models.backbones.diffusion_gemma.modeling import (
        UnturtleDiffusionGemmaForBlockDiffusion,
    )

    class _Wrapper2(UnturtleDiffusionGemmaForBlockDiffusion):
        pass

    stub = object.__new__(_Wrapper2)
    monkey_calls = []

    def fake_upstream_generate(self, input_ids=None, generation_config=None, **kw):
        monkey_calls.append((input_ids, generation_config, kw))
        return "upstream-out"

    base = UnturtleDiffusionGemmaForBlockDiffusion.__bases__[0]
    original = base.generate
    base.generate = fake_upstream_generate
    try:
        out = _Wrapper2._generate_canvas(stub, "IDS", generation_config="CFG", k=2)
    finally:
        base.generate = original
    assert out == "upstream-out"
    assert monkey_calls == [("IDS", "CFG", {"k": 2})]
