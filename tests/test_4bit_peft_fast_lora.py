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

"""Execution tests for the 4-bit + PEFT fused-LoRA contract (#177).

The module docstring of ``unturtle/fast_diffusion_model.py`` documents
``from_pretrained(..., load_in_4bit=True)`` followed by ``get_peft_model(...)``.
Before #177 that flow installed the fused LoRA hooks on every layer and then
failed at the first forward: peft's ``prepare_model_for_kbit_training`` upcasts
every non-quantized bf16 parameter to fp32, so real hidden states are fp32,
while ``matmul_lora`` multiplies the activation directly against the
bf16-dequantized 4-bit weight.

These tests EXECUTE forward and backward on a real (tiny) 4-bit fixture —
installation-only assertions proved nothing here (the hooks installed fine and
then could not run). They cover the #177 acceptance matrix:

- 4-bit + PEFT: complete fused set installed AND a real forward/backward runs;
- output/gradient parity against a genuinely unfused standard-PEFT reference
  (plain ``peft.get_peft_model``, which never enters ``matmul_lora``);
- fp32-upcasted model (the pre-fix state, or a user running peft's prepare
  themselves): ALL fast paths are skipped uniformly — no partially-fast model —
  and the standard PEFT path completes forward/backward;
- repeated ``patch_peft_model`` is idempotent;
- non-quantized bf16 PEFT executes through the fused set (previously only
  installation was asserted, never execution).
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(
        importlib.util.find_spec("bitsandbytes") is None,
        reason="requires bitsandbytes",
    ),
]

DEVICE = "cuda:0"
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# bf16 has ~3 decimal digits of mantissa; the two arms round differently
# (fused bf16 GEMM vs bnb's compute-dtype matmul + fp32 adapter path).
# Declared before any parity run; do not relax after seeing a result.
LOGIT_ATOL = 3e-2
LOGIT_RTOL = 3e-2
GRAD_ATOL = 3e-2
GRAD_RTOL = 5e-2


def _has_quantized_modules(model) -> bool:
    return any(
        getattr(getattr(m, "weight", None), "quant_state", None) is not None
        for m in model.modules()
    )


@pytest.fixture(scope="module")
def tiny_dream_checkpoint(tmp_path_factory):
    """A saved tiny Dream checkpoint (bf16) loadable with load_in_4bit=True."""
    from unturtle.models.backbones.dream.configuration_dream import DreamConfig
    from unturtle.models.backbones.dream.modeling_dream import DreamModel

    torch.manual_seed(0)
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
    model = DreamModel(config)
    # _init_weights zeroes the q/k/v biases; a zero bias makes the bias-aware
    # QKV kernel's bias handling unobservable (a drop-the-bias mutant survives
    # parity). Dream's real checkpoints carry non-zero q/k/v biases.
    with torch.no_grad():
        for layer in model.model.layers:
            for proj in (
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ):
                proj.bias.normal_(std=0.2)
    model = model.to(torch.bfloat16)
    path = tmp_path_factory.mktemp("tiny_dream_4bit") / "checkpoint"
    model.save_pretrained(path)
    return path


def _load_4bit(checkpoint):
    from unturtle import FastDiffusionModel
    from unturtle.models.backbones.dream.modeling_dream import DreamModel

    # model_class is passed explicitly because a locally-saved DreamConfig has
    # no AutoConfig registration / hub auto_map; the hub checkpoint resolves
    # through the native-class path instead. The quantization, preparation and
    # patching flow under test is identical.
    model, _tokenizer = FastDiffusionModel.from_pretrained(
        str(checkpoint),
        max_seq_length=64,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map={"": DEVICE},
        model_class=DreamModel,
    )
    # from_pretrained silently falls back to full precision when 4-bit loading
    # fails — a non-quantized model here would test nothing.
    assert _has_quantized_modules(model), "fixture did not actually load in 4-bit"
    return model


def _wrap_peft(model):
    from unturtle import FastDiffusionModel

    return FastDiffusionModel.get_peft_model(
        model,
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
        target_modules=TARGET_MODULES,
        use_gradient_checkpointing=False,
    )


def _fused_hook_presence(peft_model) -> dict[str, list[bool]]:
    """Which fused hooks are installed, per layer, per hook kind."""
    from unturtle.kernels.fast_lora import (
        apply_lora_mlp_swiglu,
        apply_lora_o,
        apply_lora_qkv_with_bias,
    )

    presence: dict[str, list[bool]] = {"qkv": [], "o": [], "mlp": [], "attn_fwd": []}
    for layer in peft_model.base_model.model.model.layers:
        attn = layer.self_attn
        presence["qkv"].append(
            getattr(attn, "apply_qkv", None) is apply_lora_qkv_with_bias
        )
        presence["o"].append(getattr(attn, "apply_o", None) is apply_lora_o)
        presence["mlp"].append(
            getattr(layer.mlp.forward, "__func__", None) is apply_lora_mlp_swiglu
        )
        # instance-level forward = injected fast attention forward
        presence["attn_fwd"].append("forward" in attn.__dict__)
    return presence


def _randomize_lora_b_(peft_model) -> None:
    """Give lora_B non-zero values so parity actually exercises the adapters.

    PEFT initializes lora_B to zero; with B == 0 the LoRA contribution is
    identically zero and output parity would hold even if the adapter math
    were completely wrong. The std is large on purpose: the adapter term must
    stand clearly above the parity tolerances, or a wrong LoRA scaling slips
    under them (measured: std=0.05 let an S*2 mutant survive).
    """
    torch.manual_seed(7)
    for name, param in peft_model.named_parameters():
        if ".lora_B." in name:
            with torch.no_grad():
                param.normal_(std=0.5)


def _forward_backward(peft_model, input_ids) -> tuple[torch.Tensor, dict]:
    """Run one forward + backward; return (logits, lora grads by name)."""
    peft_model.train()
    out = peft_model(input_ids=input_ids)
    loss = out.logits.float().square().mean()
    loss.backward()
    grads = {
        name: param.grad.detach().float().cpu()
        for name, param in peft_model.named_parameters()
        if ".lora_" in name and param.grad is not None
    }
    return out.logits.detach().float().cpu(), grads


def _input_ids(vocab_size: int) -> torch.Tensor:
    torch.manual_seed(11)
    return torch.randint(2, vocab_size, (2, 16), device=DEVICE)


class Test4BitPeftFastLora:
    def test_complete_fused_set_installed(self, tiny_dream_checkpoint):
        """The documented 4-bit + PEFT call installs QKV, O and MLP fast hooks
        on every layer — no partially-fast model."""
        peft_model = _wrap_peft(_load_4bit(tiny_dream_checkpoint))
        presence = _fused_hook_presence(peft_model)
        for kind in ("qkv", "o", "mlp", "attn_fwd"):
            assert all(presence[kind]), (
                f"fused hook {kind!r} missing on some layers: {presence[kind]}"
            )

    def test_hidden_states_stay_in_compute_dtype(self, tiny_dream_checkpoint):
        """kbit preparation must not upcast frozen weights: the embedding (the
        origin of the hidden-state dtype) stays bf16, matching the dtype the
        4-bit weights dequantize to."""
        peft_model = _wrap_peft(_load_4bit(tiny_dream_checkpoint))
        embed = peft_model.get_input_embeddings()
        assert embed.weight.dtype == torch.bfloat16, (
            f"embedding upcast to {embed.weight.dtype}; hidden states will not "
            "match the bf16-dequantized 4-bit weights in matmul_lora (#177)"
        )

    def test_forward_backward_executes(self, tiny_dream_checkpoint):
        """The documented usage completes a REAL forward and backward through
        the complete fused set (this is the #177 failure reproduced as a test)."""
        peft_model = _wrap_peft(_load_4bit(tiny_dream_checkpoint))
        _randomize_lora_b_(peft_model)
        logits, grads = _forward_backward(
            peft_model, _input_ids(peft_model.config.vocab_size)
        )
        assert torch.isfinite(logits).all(), "non-finite logits"
        expected = {
            name
            for name, param in peft_model.named_parameters()
            if ".lora_" in name and param.requires_grad
        }
        assert expected, "no trainable LoRA parameters found"
        missing = expected - set(grads)
        assert not missing, f"LoRA params with no gradient: {sorted(missing)[:5]}"
        for name, grad in grads.items():
            assert torch.isfinite(grad).all(), f"non-finite grad for {name}"

    def test_parity_against_standard_peft_reference(self, tiny_dream_checkpoint):
        """Fast arm output/grads match a genuinely unfused standard-PEFT arm.

        The reference is plain ``peft.get_peft_model`` on a fresh 4-bit load
        with the same adapter weights — it never enters ``matmul_lora``
        (unturtle installs no hooks on it), unlike the pre-#177 'reference'
        which still failed through unsloth's fused MLP path.
        """
        from peft import LoraConfig, TaskType, get_peft_model

        fast = _wrap_peft(_load_4bit(tiny_dream_checkpoint))
        _randomize_lora_b_(fast)

        reference = get_peft_model(
            _load_4bit(tiny_dream_checkpoint),
            LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=4,
                lora_alpha=4,
                lora_dropout=0.0,
                bias="none",
                target_modules=TARGET_MODULES,
            ),
        )
        # reference must be genuinely unfused
        ref_presence = _fused_hook_presence(reference)
        assert not any(any(v) for v in ref_presence.values()), (
            f"reference arm is not unfused: {ref_presence}"
        )
        # same adapters on both arms
        lora_state = {k: v for k, v in fast.state_dict().items() if ".lora_" in k}
        _missing, unexpected = reference.load_state_dict(lora_state, strict=False)
        assert not unexpected, f"adapter state mismatch: {unexpected[:5]}"

        input_ids = _input_ids(fast.config.vocab_size)
        # reference arm first: it must be executable on its own
        ref_logits, ref_grads = _forward_backward(reference, input_ids)
        fast_logits, fast_grads = _forward_backward(fast, input_ids)

        delta = (fast_logits - ref_logits).abs()
        scale = ref_logits.abs()
        worst = (delta - (LOGIT_ATOL + LOGIT_RTOL * scale)).max().item()
        assert worst <= 0, (
            f"logit parity violated: max |Δ|={delta.max().item():.4e} "
            f"(atol={LOGIT_ATOL}, rtol={LOGIT_RTOL})"
        )

        assert set(fast_grads) == set(ref_grads), (
            "gradient key sets differ between arms"
        )
        worst_key, worst_excess, worst_delta = None, -float("inf"), 0.0
        for name in sorted(fast_grads):
            g_delta = (fast_grads[name] - ref_grads[name]).abs()
            g_scale = ref_grads[name].abs()
            excess = (g_delta - (GRAD_ATOL + GRAD_RTOL * g_scale)).max().item()
            if excess > worst_excess:
                worst_key, worst_excess = name, excess
                worst_delta = g_delta.max().item()
        assert worst_excess <= 0, (
            f"gradient parity violated at {worst_key}: max |Δ|={worst_delta:.4e} "
            f"(atol={GRAD_ATOL}, rtol={GRAD_RTOL})"
        )

    def test_upcasted_model_skips_all_fast_paths_uniformly(self, tiny_dream_checkpoint):
        """A model whose hidden states were upcast to fp32 (e.g. by running
        peft's own ``prepare_model_for_kbit_training``) gets NO fast hooks —
        not QKV, not O, not MLP, not the fast attention forward — and still
        completes forward/backward through the standard PEFT path.

        This is the #177 handoff contract: incompatible compute dtype must
        produce a working standard model, never a partially-fast one.
        """
        model = _load_4bit(tiny_dream_checkpoint)
        # simulate peft's upcast: every non-quantized floating parameter → fp32
        for param in model.parameters():
            if (
                param.dtype in (torch.bfloat16, torch.float16)
                and type(param).__name__ != "Params4bit"
            ):
                param.data = param.data.to(torch.float32)

        peft_model = _wrap_peft(model)
        presence = _fused_hook_presence(peft_model)
        for kind in ("qkv", "o", "mlp", "attn_fwd"):
            assert not any(presence[kind]), (
                f"fast hook {kind!r} installed on an fp32-activation model — "
                f"partially-fast state: {presence}"
            )

        _randomize_lora_b_(peft_model)
        logits, grads = _forward_backward(
            peft_model, _input_ids(peft_model.config.vocab_size)
        )
        assert torch.isfinite(logits).all()
        assert grads, "standard-path backward produced no LoRA gradients"

    def test_patch_peft_model_reentry_is_idempotent(self, tiny_dream_checkpoint):
        """Calling patch_peft_model again (documented as safe) keeps the same
        hooks and the model still executes."""
        from unturtle import FastDiffusionModel

        peft_model = _wrap_peft(_load_4bit(tiny_dream_checkpoint))
        before = _fused_hook_presence(peft_model)
        FastDiffusionModel.patch_peft_model(peft_model, lora_dropout=0.0, bias="none")
        after = _fused_hook_presence(peft_model)
        assert before == after
        _randomize_lora_b_(peft_model)
        logits, _grads = _forward_backward(
            peft_model, _input_ids(peft_model.config.vocab_size)
        )
        assert torch.isfinite(logits).all()


class TestBf16NonQuantizedExecution:
    def test_bf16_peft_forward_backward_through_fused_set(self, tiny_dream_checkpoint):
        """Non-quantized bf16 + PEFT executes forward/backward with the fused
        set installed (the regime that already worked must keep working)."""
        from unturtle import FastDiffusionModel
        from unturtle.models.backbones.dream.modeling_dream import DreamModel

        model, _ = FastDiffusionModel.from_pretrained(
            str(tiny_dream_checkpoint),
            max_seq_length=64,
            dtype=torch.bfloat16,
            load_in_4bit=False,
            device_map={"": DEVICE},
            model_class=DreamModel,
        )
        assert not _has_quantized_modules(model)
        peft_model = _wrap_peft(model)
        presence = _fused_hook_presence(peft_model)
        for kind in ("qkv", "o", "mlp", "attn_fwd"):
            assert all(presence[kind]), (
                f"fused hook {kind!r} missing on bf16 model: {presence[kind]}"
            )
        _randomize_lora_b_(peft_model)
        logits, grads = _forward_backward(
            peft_model, _input_ids(peft_model.config.vocab_size)
        )
        assert torch.isfinite(logits).all()
        assert grads
