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

"""#174 PR 0 — four-arm RoPE load-path attribution in ONE fresh process.

    python attribution_probe.py --case <case> --device cpu|cuda --sdpa MATH|FLASH|none
                                --poison none|nan --out <file>

Arms (all forwards under the SAME pinned SDPA backend):

1. ``original``          the instance before save
2. ``direct_state_dict``  same-config reconstruction + ``load_state_dict``
                         (buffers come from ``__init__``: the analytic formula)
3. ``reload``             the test's own ``cls.from_pretrained(dir)``
4. ``reload_restored``    arm 3 with every non-persistent buffer copied from
                         arm 1 immediately before forward

``--poison nan`` allocates and frees NaN-filled tensors right before arm 3
(whether the poison lands is allocator luck); ``--poison empty_like_nan``
wraps ``torch.empty_like`` during ``from_pretrained`` so every buffer the
load path leaves uninitialized is deterministically NaN. Neither changes
production code: they make the load path's uninitialized-buffer behavior
visible on demand instead of by allocator luck.

Exit codes: 0 observed, 2 typed blocked, 3 import-root violation.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _emit(path: pathlib.Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _verify_import_root(out: pathlib.Path) -> None:
    expected = os.environ.get("UNTURTLE_EXPECTED_ROOT")
    import unturtle

    actual = pathlib.Path(unturtle.__file__).resolve().parents[1]
    if not expected or actual != pathlib.Path(expected).resolve():
        _emit(out, {"probe_error": "import_root_mismatch", "actual": str(actual)})
        raise SystemExit(3)


# ---------------------------------------------------------------------------
# cases — the two failing DiT fixtures, verbatim, plus the #184 Dream cell
# ---------------------------------------------------------------------------


def _case_mdlm_dit_plain():
    import torch

    from unturtle.models.backbones.mdlm_dit import (
        MDLMDiTConfig,
        MDLMDiTForMaskedDiffusionLM,
    )

    config = MDLMDiTConfig(
        vocab_size=512,
        hidden_size=64,
        cond_dim=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        dropout=0.0,
        max_position_embeddings=64,
        mask_token_id=511,
    )
    torch.manual_seed(0)
    model = MDLMDiTForMaskedDiffusionLM(config).eval()
    inputs = {"input_ids": torch.randint(0, config.vocab_size, (1, 8))}
    return (
        model,
        MDLMDiTForMaskedDiffusionLM,
        config,
        inputs,
        {
            "rope_dim": config.hidden_size // config.num_attention_heads,
            "rope_base": 10_000,
            "max_position_embeddings": config.max_position_embeddings,
        },
    )


def _case_mdlm_dit_latent_conditioned():
    import torch

    from unturtle.models.backbones.mdlm_dit import (
        MDLMDiTConfig,
        MDLMDiTForMaskedDiffusionLM,
    )
    from unturtle.models.latent.modeling_ladiff_dit import (
        LaDiffDiTConfig,
        LatentConditionedMDLMDiT,
    )

    vocab, hidden, layers, mask_id = 16, 32, 4, 15
    base = MDLMDiTConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        cond_dim=8,
        num_hidden_layers=layers,
        num_attention_heads=2,
        dropout=0.0,
        max_position_embeddings=32,
        mask_token_id=mask_id,
    )
    ladiff = LaDiffDiTConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        cond_dim=8,
        num_hidden_layers=layers,
        num_attention_heads=2,
        dropout=0.0,
        max_position_embeddings=32,
        mask_token_id=mask_id,
        num_latents=3,
        latent_dim=hidden,
    )
    torch.manual_seed(0)
    plain = MDLMDiTForMaskedDiffusionLM(base).eval()
    conditioned = LatentConditionedMDLMDiT(ladiff).eval()
    conditioned.model.load_state_dict(plain.model.state_dict())
    for adapter in conditioned.latent_adapters.values():
        torch.nn.init.normal_(adapter.conv_out.weight, std=0.2)
    inputs = {
        "input_ids": torch.randint(0, vocab, (2, 12)),
        "latents": torch.randn(2, 3, hidden),
    }
    return (
        conditioned,
        LatentConditionedMDLMDiT,
        ladiff,
        inputs,
        {
            "rope_dim": hidden // 2,
            "rope_base": 10_000,
            "max_position_embeddings": 32,
        },
    )


def _case_dream_native():
    import torch

    from unturtle.models.backbones.dream.configuration_dream import DreamConfig
    from unturtle.models.backbones.dream.modeling_dream import DreamModel

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
    torch.manual_seed(0)
    model = DreamModel(config).eval()
    inputs = {"input_ids": torch.randint(2, 400, (2, 12))}
    return (
        model,
        DreamModel,
        config,
        inputs,
        {
            "rope_theta": getattr(config, "rope_theta", None),
            "rope_scaling": getattr(config, "rope_scaling", None),
            "max_position_embeddings": config.max_position_embeddings,
            "head_dim": config.hidden_size // config.num_attention_heads,
        },
    )


CASES = {
    "mdlm_dit_plain": _case_mdlm_dit_plain,
    "mdlm_dit_latent_conditioned": _case_mdlm_dit_latent_conditioned,
    "dream_native": _case_dream_native,
}


def _sdpa_context(name: str):
    import contextlib

    if name == "none":
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel(
        {"MATH": SDPBackend.MATH, "FLASH": SDPBackend.FLASH_ATTENTION}[name]
    )


def _poison_allocator(mode: str) -> None:
    """Fill and free memory so freshly `torch.empty_like`d buffers are
    observably garbage. Pure observation aid; nothing persists. Whether the
    poison lands in a given buffer is allocator luck — see
    ``_empty_like_nan`` for the deterministic variant."""
    import torch

    if mode != "nan":
        return
    junk = [
        torch.full((n,), float("nan"))
        for n in (8, 8, 8, 16, 32, 64, 128, 256)
        for _ in range(200)
    ]
    del junk


class _empty_like_nan:
    """Context manager: make ``torch.empty_like`` return NaN-filled tensors.

    transformers' ``from_pretrained`` re-materializes every non-persistent
    buffer with ``torch.empty_like`` (modeling_utils, "move back
    non-persistent buffers") and relies on ``_init_weights`` to give them
    values. This makes "uninitialized" deterministic and visible instead of
    dependent on what the allocator last freed; loaded weights overwrite
    their own ``empty_like`` targets, so only genuinely uninitialized memory
    becomes NaN. Observation aid only — it never touches production code.
    """

    def __enter__(self):
        import torch

        self._original = torch.empty_like

        def empty_like_nan(*args, **kwargs):
            return self._original(*args, **kwargs).fill_(float("nan"))

        torch.empty_like = empty_like_nan
        return self

    def __exit__(self, *exc):
        import torch

        torch.empty_like = self._original
        return False


def run(args) -> dict:
    import torch

    from unturtle.diagnostics.persistence import (
        buffer_census,
        classify_rope_attribution,
        compare_tensors,
        first_state_dict_mismatch,
        instance_patches,
        process_state_snapshot,
        state_dict_digest,
    )

    device = torch.device(args.device)
    model, cls, config, inputs, rope_fields = CASES[args.case]()
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    def forward(m):
        with torch.no_grad(), _sdpa_context(args.sdpa):
            return m(**inputs).logits.detach()

    def describe(m, load_path: str, ref, ref_out):
        digest, _ = state_dict_digest(m)
        out = forward(m)
        return (
            {
                "load_path": load_path,
                "object_id": id(m),  # volatile by nature; used only for arm identity
                "class": f"{type(m).__module__}.{type(m).__qualname__}",
                "mro": [
                    f"{b.__module__}.{b.__qualname__}" for b in type(m).__mro__[:4]
                ],
                "persistent_digest": digest,
                "first_persistent_mismatch": (
                    first_state_dict_mismatch(ref, m) if ref is not None else None
                ),
                "buffers": buffer_census(m, reference=ref),
                "output_vs_original": (
                    compare_tensors(out, ref_out, label="logits")
                    if ref_out is not None
                    else None
                ),
                "instance_patches": instance_patches(m),
            },
            out,
        )

    snapshot_before = process_state_snapshot(config_type=config.model_type)

    # arm 1
    original_rec, original_out = describe(model, "original", None, None)
    original_rec["buffers"] = buffer_census(model, reference=model)
    original_rec["output_vs_original"] = compare_tensors(
        original_out, original_out, label="logits"
    )

    # arm 2: same-config reconstruction + state dict (no load path)
    direct = cls(config).to(device).eval()
    direct.load_state_dict(model.state_dict())
    direct_rec, _ = describe(direct, "direct_state_dict", model, original_out)

    # arm 3: the tests' own from_pretrained
    out_dir = tempfile.mkdtemp()
    model.save_pretrained(out_dir)
    _poison_allocator(args.poison)
    if args.poison == "empty_like_nan":
        with _empty_like_nan():
            reloaded = cls.from_pretrained(out_dir)
    else:
        reloaded = cls.from_pretrained(out_dir)
    reloaded = reloaded.to(device).eval()
    reload_rec, reload_out = describe(reloaded, "from_pretrained", model, original_out)

    # arm 4: arm 3 with every non-persistent buffer restored from arm 1
    restored = copy.deepcopy(reloaded)
    original_buffers = dict(model.named_buffers())
    restored_names = []
    with torch.no_grad():
        for name, buffer in restored.named_buffers():
            owner_name, _, attr = name.rpartition(".")
            owner = dict(restored.named_modules()).get(owner_name, restored)
            if attr in getattr(owner, "_non_persistent_buffers_set", set()):
                buffer.copy_(original_buffers[name].to(buffer.device, buffer.dtype))
                restored_names.append(name)
    restored_rec, _ = describe(
        restored, "from_pretrained+restored_buffers", model, original_out
    )
    restored_rec["restored_buffer_names"] = sorted(restored_names)

    arms = {
        "original": original_rec,
        "direct_state_dict": direct_rec,
        "reload": reload_rec,
        "reload_restored": restored_rec,
    }
    verdict = classify_rope_attribution(
        arms=arms, sdpa_backend=None if args.sdpa == "none" else args.sdpa
    )
    snapshot_after = process_state_snapshot(config_type=config.model_type)

    return {
        "status": "observed",
        "case": args.case,
        "device": args.device,
        "sdpa_backend": args.sdpa,
        "poison": args.poison,
        "model_class": original_rec["class"],
        "rope_config_fields": rope_fields,
        "init_weights_owner": (
            f"{type(model)._init_weights.__module__}."
            f"{type(model)._init_weights.__qualname__}"
        ),
        "arms": arms,
        "verdict": verdict,
        "process_state": {
            "before": snapshot_before,
            "after": snapshot_after,
            "changed_keys": sorted(
                k
                for k in snapshot_before
                if snapshot_before.get(k) != snapshot_after.get(k)
            ),
        },
        "volatile": {"tmp_dir": out_dir},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sdpa", choices=("MATH", "FLASH", "none"), default="MATH")
    parser.add_argument(
        "--poison", choices=("none", "nan", "empty_like_nan"), default="none"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    try:
        _verify_import_root(out)
        result = run(args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — typed blocked
        _emit(
            out,
            {
                "status": "blocked",
                "case": args.case,
                "reason": f"{type(exc).__name__}: {str(exc)[:300]}",
            },
        )
        raise SystemExit(2) from exc
    _emit(out, result)


if __name__ == "__main__":
    main()
