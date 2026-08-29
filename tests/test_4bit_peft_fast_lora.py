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

# Per-operation relative-Frobenius-norm bound for the fused-vs-unfused parity
# check. The two paths round differently (fused bf16 GEMM vs bnb's
# compute-dtype matmul + fp32 adapter path), so exact equality is not
# expected. Calibration, measured on this fixture (deterministic — identical
# to the fourth decimal across repeated processes): healthy worst per-op
# divergence 0.0092 (an MLP lora_B gradient); a LoRA-scale*2 mutant reaches
# 3.0 and a dropped-bias mutant 0.25. The bound sits 5.4x above healthy and
# 5x below the weakest mutant.
PARITY_REL_NORM_BOUND = 5e-2


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


def _wrap_peft(model, use_gradient_checkpointing=False):
    from unturtle import FastDiffusionModel

    return FastDiffusionModel.get_peft_model(
        model,
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
        target_modules=TARGET_MODULES,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )


def _fused_hook_presence(peft_model) -> dict[str, list[bool]]:
    """Which fused hooks are installed, per layer, per hook kind."""
    from unturtle.kernels.fast_lora import (
        apply_lora_mlp_swiglu,
        apply_lora_o,
        apply_lora_qkv_with_bias,
    )
    from unturtle.models.backbones.dream.modeling_dream import (
        DreamAttention_fast_forward,
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
        # identity, not just presence: injecting the WRONG instance-level
        # forward must not read as the fast attention forward
        instance_forward = attn.__dict__.get("forward")
        presence["attn_fwd"].append(
            getattr(instance_forward, "__func__", None) is DreamAttention_fast_forward
        )
    return presence


def _randomize_lora_b_(peft_model) -> None:
    """Give the adapters deterministic non-zero values.

    lora_B: PEFT initializes it to zero; with B == 0 the LoRA contribution is
    identically zero and output parity would hold even if the adapter math
    were completely wrong. The std is large on purpose: the adapter term must
    stand clearly above the parity tolerances, or a wrong LoRA scaling slips
    under them (measured: std=0.05 let an S*2 mutant survive).

    lora_A must be re-drawn too: PEFT's kaiming init consumes the GLOBAL RNG,
    so its values depend on every test that ran earlier in the process. That
    made the parity comparison bimodal across test selections (an unlucky
    draw pushed the fast-vs-reference divergence from ‖Δ‖/‖ref‖≈0.01 to a
    bit-identical 0.1064 — reproducibly, since both arms share the adapters).
    """
    torch.manual_seed(7)
    for name, param in peft_model.named_parameters():
        with torch.no_grad():
            if ".lora_A." in name:
                param.normal_(std=0.1)
            elif ".lora_B." in name:
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
        # the fixture requests bf16, so bf16 is the expected concrete value …
        assert embed.weight.dtype == torch.bfloat16, (
            f"embedding upcast to {embed.weight.dtype}; hidden states will not "
            "match the bf16-dequantized 4-bit weights in matmul_lora (#177)"
        )
        # … but the CONTRACT is relational: the embedding (hidden-state origin)
        # must match what every quantized weight dequantizes to.
        quant_dtypes = {
            m.weight.quant_state.dtype
            for m in peft_model.modules()
            if getattr(getattr(m, "weight", None), "quant_state", None) is not None
        }
        assert quant_dtypes == {embed.weight.dtype}, (
            f"hidden-state dtype {embed.weight.dtype} cannot feed quantized "
            f"weights dequantizing to {quant_dtypes}"
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

    @pytest.mark.parametrize("gc_mode", ["unsloth", True])
    def test_forward_backward_executes_under_gradient_checkpointing(
        self, tiny_dream_checkpoint, gc_mode
    ):
        """get_peft_model's default is use_gradient_checkpointing="unsloth";
        the unsloth-semantics preparation must complete forward/backward under
        both GC modes, not only the GC-off path the repro used."""
        peft_model = _wrap_peft(
            _load_4bit(tiny_dream_checkpoint), use_gradient_checkpointing=gc_mode
        )
        assert peft_model.get_input_embeddings().weight.dtype == torch.bfloat16
        _randomize_lora_b_(peft_model)
        logits, grads = _forward_backward(
            peft_model, _input_ids(peft_model.config.vocab_size)
        )
        assert torch.isfinite(logits).all()
        assert grads, f"no LoRA gradients under gc_mode={gc_mode!r}"

    def test_parity_against_standard_peft_reference(self, tiny_dream_checkpoint):
        """Every fused hook matches the genuinely unfused standard-PEFT path
        on the SAME 4-bit model — outputs and backward gradients.

        The comparison is per-operation (QKV / O / MLP, every layer), not
        end-to-end logits: the units under test are the fused LoRA GEMM hooks,
        and an end-to-end comparison routes through two different attention
        implementations whose torch backend selection proved bistable across
        processes on this tiny shape (‖Δ‖/‖ref‖ flipped between 0.01 and 0.83
        on identical inputs — either arm could flip). The reference callables
        (``_original_apply_qkv`` / ``_original_apply_o`` / the unpatched class
        ``forward``) call the plain PEFT modules and never enter
        ``matmul_lora`` — unlike the pre-#177 'reference', which still failed
        through unsloth's fused MLP path. End-to-end execution is covered by
        ``test_forward_backward_executes``.
        """
        from unturtle.fast_diffusion_model import (
            _original_apply_o,
            _original_apply_qkv,
        )

        fast = _wrap_peft(_load_4bit(tiny_dream_checkpoint))
        _randomize_lora_b_(fast)
        presence = _fused_hook_presence(fast)
        for kind in ("qkv", "o", "mlp"):
            assert all(presence[kind]), f"fused hook {kind!r} not installed"

        hidden = fast.config.hidden_size
        torch.manual_seed(13)
        base_input = torch.randn(2, 16, hidden, device=DEVICE, dtype=torch.bfloat16)

        def run(op, module, X):
            """One forward + backward through `op`; returns outputs + grads."""
            X = X.clone().requires_grad_(True)
            outs = op(module, X)
            outs = outs if isinstance(outs, tuple) else (outs,)
            loss = sum(o.float().square().mean() for o in outs)
            loss.backward()
            grads = {
                name: param.grad.detach().float().cpu()
                for name, param in module.named_parameters()
                if ".lora_" in name and param.grad is not None
            }
            module.zero_grad(set_to_none=True)
            return (
                [o.detach().float().cpu() for o in outs],
                grads,
                X.grad.detach().float().cpu(),
            )

        def rel_norm(a, b):
            return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()

        worst: dict[str, tuple[float, str]] = {}

        def compare(label, fast_result, ref_result):
            (f_outs, f_grads, f_xgrad) = fast_result
            (r_outs, r_grads, r_xgrad) = ref_result
            assert set(f_grads) == set(r_grads), f"{label}: grad key mismatch"
            assert f_grads, f"{label}: no LoRA gradients"
            checks = [
                (f"{label}:out{i}", f, r)
                for i, (f, r) in enumerate(zip(f_outs, r_outs, strict=True))
            ]
            checks += [(f"{label}:X.grad", f_xgrad, r_xgrad)]
            checks += [
                (f"{label}:grad:{name}", f_grads[name], r_grads[name])
                for name in sorted(f_grads)
            ]
            for name, f, r in checks:
                value = rel_norm(f, r)
                kind = "grad" if "grad" in name else "out"
                if value > worst.get(kind, (0.0, ""))[0]:
                    worst[kind] = (value, name)
                assert value <= PARITY_REL_NORM_BOUND, (
                    f"parity violated at {name}: ‖Δ‖/‖ref‖={value:.4f} "
                    f"(bound {PARITY_REL_NORM_BOUND})"
                )

        for index, layer in enumerate(fast.base_model.model.model.layers):
            attn = layer.self_attn
            compare(
                f"layer{index}.qkv",
                run(attn.apply_qkv, attn, base_input),
                run(_original_apply_qkv, attn, base_input),
            )
            compare(
                f"layer{index}.o",
                run(attn.apply_o, attn, base_input),
                run(_original_apply_o, attn, base_input),
            )
            mlp = layer.mlp
            fused_mlp = mlp.forward.__func__
            unfused_mlp = type(mlp).forward
            assert fused_mlp is not unfused_mlp
            compare(
                f"layer{index}.mlp",
                run(fused_mlp, mlp, base_input),
                run(unfused_mlp, mlp, base_input),
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
