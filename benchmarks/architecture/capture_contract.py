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

"""Sole producer of the #184 architecture-contract artifact.

Runs one fresh subprocess per observation (``subprocess_probe.py``), assembles
``docs/artifacts/184-architecture-contract-v1.json`` and generates
``docs/architecture/contract-v1.md`` from it — numbers, MROs and symbol lists
are never hand-copied into the Markdown.

Reliability gates (#166/#183 lessons):
- the worktree must be clean, and the producing commit SHA is recorded;
- every probe runs with ``PYTHONPATH=<repo>`` and verifies that the imported
  ``unturtle`` resolves inside this checkout — a mismatch aborts the whole run
  (exit 3), because an artifact read from the wrong checkout must not exist;
- volatile content (versions, RNG raw states, commands) lives under
  ``producer`` / ``volatile`` keys, which are excluded from the semantic
  digest; ``--check`` regenerates everything and compares digests.

Usage::

    PYTHONPATH=$PWD .venv/bin/python benchmarks/architecture/capture_contract.py
    PYTHONPATH=$PWD .venv/bin/python benchmarks/architecture/capture_contract.py --check
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROBE = REPO_ROOT / "benchmarks" / "architecture" / "subprocess_probe.py"
ARTIFACT_PATH = REPO_ROOT / "docs" / "artifacts" / "184-architecture-contract-v1.json"
MARKDOWN_PATH = REPO_ROOT / "docs" / "architecture" / "contract-v1.md"

#: Captured BEFORE anything heavy runs in this process: the probes must see a
#: pristine environment. Importing the unturtle package here would pull the
#: unsloth chain, which mutates ~25 os.environ keys in THIS process — and
#: those would silently propagate to every probe, turning the import probes'
#: "environ_added_keys" into an empty (and wrong) observation. For the same
#: reason the diagnostics helpers are loaded from their FILE below, without
#: executing unturtle/__init__, and library versions come from
#: importlib.metadata rather than imports.
_BASE_ENV = dict(os.environ)


def _load_diagnostics():
    spec = importlib.util.spec_from_file_location(
        "_unturtle_diagnostics_architecture",
        REPO_ROOT / "unturtle" / "diagnostics" / "architecture.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_diagnostics = _load_diagnostics()
make_row = _diagnostics.make_row
semantic_digest = _diagnostics.semantic_digest

IMPORT_TARGETS = (
    "unturtle",
    "unturtle.models",
    "unturtle.registry",
    "unturtle.methods",
    "unturtle.plugins",
    "unturtle.fast_diffusion_model",
)
MODEL_FAMILIES = (
    "dream",
    "llada",
    "mdlm_dit",
    "tiny_a2d_llama",
    "tiny_a2d_qwen2",
    "tiny_a2d_qwen3",
    "modernbert_diffusion",
    "diffusion_gemma",
)
GENERATION_FAMILIES = ("dream", "tiny_a2d_llama")
PERSISTENCE_CASES = (
    "native_fp",
    "native_peft",
    "custom_adapter",
    "autoconfig_roundtrip",
    "generation_reload",
)
PROCESS_GLOBAL_CASES = ("rng_contract", "sdpa")


# ---------------------------------------------------------------------------
# Runtime mutation ledger — curated rows, each mechanically anchored to the
# production tree by a (file suffix, line substring) claim that the scanner
# must find. The scanner ALSO enumerates every mutation-shaped line and
# refuses unclaimed hits, so deleting a row (or the code drifting away from a
# row) fails the build instead of silently shrinking the ledger.
# ---------------------------------------------------------------------------

FDM = "unturtle/fast_diffusion_model.py"
LOADING = "unturtle/models/loading.py"  # #185 PR 3 loader
PREP = "unturtle/models/integrations/peft_preparation.py"  # #185 PR 2 PEFT preparation
LLADA_PROVIDER = "unturtle/models/backbones/llada/fast_paths.py"  # #185 LLaDA provider
DREAM_PROVIDER = "unturtle/models/backbones/dream/fast_paths.py"  # #185 Dream provider
MODERNBERT_PROVIDER = (
    "unturtle/models/backbones/modernbert/fast_paths.py"  # #185 ModernBERT provider
)
A2D_PROVIDER = (
    "unturtle/models/conversion/a2d/tiny_a2d/fast_paths.py"  # #185 Tiny-A2D provider
)


def _mutation(
    mutation_id: str,
    *,
    owner: str,
    target: str,
    applicability: str,
    before: str,
    after: str,
    idempotent: str,
    reversible: str,
    scope: str,
    success_signal: str,
    liveness_evidence: str,
    classification: str,
    claims: list[tuple[str, str]],
    linked_issue: int | None = None,
) -> dict:
    return {
        "mutation_id": mutation_id,
        "owner": owner,
        "target": target,
        "applicability": applicability,
        "before_identity": before,
        "after_identity": after,
        "idempotent": idempotent,
        "reversible": reversible,
        "scope": scope,
        "success_signal": success_signal,
        "liveness_evidence": liveness_evidence,
        "classification": classification,
        "linked_issue": linked_issue,
        "claims": [{"file": f, "contains": c} for f, c in claims],
    }


_WARN_ONLY = (
    "warning-only (_warn_once patch report); no structured result — "
    "REPLACE target for #185's PatchReport"
)

MUTATION_LEDGER: list[dict] = [
    _mutation(
        "a2d_attention_fast_forward",
        owner="tiny_a2d.fast_paths.patch_peft (#185 provider)",
        target="layer.self_attn.forward (instance)",
        applicability="CUDA",
        before="class-level TinyA2D attention forward",
        after="MethodType(TinyA2DAttention_fast_forward)",
        idempotent="yes (same assignment)",
        reversible="no API (instance attr could be deleted)",
        scope="object-local",
        success_signal=_WARN_ONLY,
        liveness_evidence="tests/test_4bit_peft_fast_lora.py (Dream analogue); A2D E2E tests",
        classification="EXTRACTED -> tiny_a2d.fast_paths (#185, family provider)",
        claims=[(A2D_PROVIDER, 'targets["self_attn"].forward = types.MethodType(')],
    ),
    _mutation(
        "a2d_mlp_fast_hook",
        owner="tiny_a2d.fast_paths.patch_peft (#185 provider)",
        target="layer.mlp.forward (instance)",
        applicability="CUDA + lora_dropout==0 + bias==none + LoRA + no DoRA + dtype gate (#177)",
        before="class-level MLP forward",
        after="MethodType(apply_lora_mlp_swiglu) [unsloth]",
        idempotent="yes",
        reversible="no API",
        scope="object-local",
        success_signal=_WARN_ONLY,
        liveness_evidence="fourbit-contract probe: mlp_forward_is_fast + backward",
        classification="EXTRACTED -> tiny_a2d.fast_paths (#185, family provider)",
        claims=[
            (A2D_PROVIDER, "mlp.forward = types.MethodType(apply_lora_mlp_swiglu, mlp)")
        ],
    ),
    _mutation(
        "a2d_qkv_fast_hook",
        owner="tiny_a2d.fast_paths.patch_peft (#185 provider)",
        target="layer.self_attn.apply_qkv",
        applicability="CUDA + lora_dropout==0 + bias==none + LoRA + no DoRA + dtype gate (#177)",
        before="_original_apply_qkv stub",
        after="apply_lora_qkv (unsloth)",
        idempotent="yes",
        reversible="reassign stub",
        scope="object-local",
        success_signal=_WARN_ONLY,
        liveness_evidence="A2D fast-path tests",
        classification="EXTRACTED -> tiny_a2d.fast_paths (#185, family provider)",
        claims=[(A2D_PROVIDER, 'targets["self_attn"].apply_qkv = apply_lora_qkv')],
    ),
    _mutation(
        "a2d_o_fast_hook",
        owner="tiny_a2d.fast_paths.patch_peft (#185 provider)",
        target="layer.self_attn.apply_o",
        applicability="CUDA + eligibility gates",
        before="_original_apply_o stub",
        after="apply_lora_o (unsloth)",
        idempotent="yes",
        reversible="reassign stub",
        scope="object-local",
        success_signal=_WARN_ONLY,
        liveness_evidence="A2D fast-path tests",
        classification="EXTRACTED -> tiny_a2d.fast_paths (#185, family provider)",
        claims=[(A2D_PROVIDER, 'targets["self_attn"].apply_o = apply_lora_o')],
    ),
    _mutation(
        "dream_attention_fast_forward",
        owner="dream.fast_paths.patch_peft (#185 provider)",
        target="self_attn.forward (instance)",
        applicability="CUDA + dtype gate (#177)",
        before="DreamSdpaAttention.forward (class)",
        after="MethodType(DreamAttention_fast_forward)",
        idempotent="yes",
        reversible="delete instance attr (no API)",
        scope="object-local",
        success_signal=_WARN_ONLY,
        liveness_evidence="fourbit-contract probe: instance_forward_installed + forward/backward",
        classification="EXTRACTED -> dream.fast_paths (#185, family provider)",
        claims=[
            (
                DREAM_PROVIDER,
                "self_attn.forward = types.MethodType(DreamAttention_fast_forward",
            )
        ],
        linked_issue=177,
    ),
    _mutation(
        "dream_qkv_fast_hook_bias",
        owner="dream.fast_paths.patch_peft (#185 provider)",
        target="self_attn.apply_qkv",
        applicability="CUDA + lora_dropout==0 + bias==none + LoRA + no DoRA + dtype gate (#177)",
        before="_original_apply_qkv stub",
        after="apply_lora_qkv_with_bias (unturtle.kernels.fast_lora)",
        idempotent="yes",
        reversible="reassign stub",
        scope="object-local",
        success_signal=_WARN_ONLY,
        liveness_evidence=(
            "fourbit-contract probe: before=_original_apply_qkv, "
            "after=apply_lora_qkv_with_bias, forward+backward complete"
        ),
        classification="EXTRACTED -> dream.fast_paths (#185, family provider)",
        claims=[(DREAM_PROVIDER, "self_attn.apply_qkv = apply_lora_qkv_with_bias")],
        linked_issue=177,
    ),
    _mutation(
        "dream_o_fast_hook",
        owner="dream.fast_paths.patch_peft (#185 provider)",
        target="self_attn.apply_o",
        applicability="CUDA + eligibility gates",
        before="_original_apply_o stub",
        after="apply_lora_o (unsloth)",
        idempotent="yes",
        reversible="reassign stub",
        scope="object-local",
        success_signal=_WARN_ONLY,
        liveness_evidence="fourbit-contract probe: apply_o_is_fast + backward",
        classification="EXTRACTED -> dream.fast_paths (#185, family provider)",
        claims=[(DREAM_PROVIDER, "self_attn.apply_o = apply_lora_o")],
    ),
    _mutation(
        "dream_mlp_fast_hook",
        owner="dream.fast_paths.patch_peft (#185 provider)",
        target="layer.mlp.forward (instance)",
        applicability="CUDA + eligibility gates",
        before="DreamMLP.forward (class)",
        after="MethodType(apply_lora_mlp_swiglu) [unsloth]",
        idempotent="yes",
        reversible="no API",
        scope="object-local",
        success_signal=_WARN_ONLY,
        liveness_evidence="fourbit-contract probe: mlp_forward_is_fast + backward",
        classification="EXTRACTED -> dream.fast_paths (#185, family provider)",
        claims=[
            (
                DREAM_PROVIDER,
                "mlp.forward = types.MethodType(apply_lora_mlp_swiglu, mlp)",
            )
        ],
    ),
    _mutation(
        "llada_rope_fast_forward",
        owner="llada.fast_paths.patch_peft (#185 provider)",
        target="block.rotary_emb.forward (instance)",
        applicability="CUDA",
        before="class rotary forward",
        after="MethodType(_make_llada_fast_rope_forward(...))",
        idempotent="yes — guarded by _fast_rope_patched flag",
        reversible="no API",
        scope="object-local",
        success_signal=_WARN_ONLY,
        liveness_evidence="LLaDA fast-path tests",
        classification="EXTRACTED -> llada.fast_paths (#185, family provider; hooks wired LIVE)",
        claims=[(LLADA_PROVIDER, "rotary_emb.forward = types.MethodType(")],
    ),
    _mutation(
        "llada_qkv_fast_hook",
        owner="llada.fast_paths.patch_peft (#185 provider)",
        target="block.apply_qkv",
        applicability="CUDA + eligibility gates",
        before="unset / stub",
        after="apply_lora_qkv (unsloth)",
        idempotent="yes",
        reversible="reassign",
        scope="object-local",
        success_signal=_WARN_ONLY,
        liveness_evidence="LIVE since the #185 wiring: LLaDALlamaBlock.forward dispatches through apply_qkv; probe_liveness counters positive per block, forward+backward (tests/models/test_llada_fast_paths.py)",
        classification="EXTRACTED -> llada.fast_paths (#185, family provider; hooks wired LIVE)",
        claims=[(LLADA_PROVIDER, "block.apply_qkv = apply_lora_qkv")],
    ),
    _mutation(
        "llada_o_fast_hook",
        owner="llada.fast_paths.patch_peft (#185 provider)",
        target="block.apply_o + o_proj aliasing",
        applicability="CUDA + eligibility gates",
        before="unset / stub",
        after="apply_lora_o (unsloth) + __dict__ o_proj->attn_out alias (kernel reads self.o_proj; not module-registered, so state_dict is unchanged)",
        idempotent="yes",
        reversible="reassign",
        scope="object-local",
        success_signal=_WARN_ONLY,
        liveness_evidence="LIVE since the #185 wiring: LLaDABlock.attention dispatches through apply_o (o_proj alias installed with the hook); probe_liveness counters positive per block, forward+backward (tests/models/test_llada_fast_paths.py)",
        classification="EXTRACTED -> llada.fast_paths (#185, family provider; hooks wired LIVE)",
        claims=[
            (LLADA_PROVIDER, "block.apply_o = apply_lora_o"),
            (LLADA_PROVIDER, 'block.__dict__["o_proj"] = block.attn_out'),
        ],
    ),
    _mutation(
        "llada_mlp_fast_hook",
        owner="llada.fast_paths.patch_peft (#185 provider)",
        target="block.apply_mlp + gate_proj/down_proj aliasing",
        applicability="CUDA + SiLU activation + eligibility gates",
        before="LLaDALlamaBlock._default_apply_mlp",
        after="apply_lora_mlp_swiglu (unsloth) + block.gate_proj=ff_proj alias",
        idempotent="yes",
        reversible="reassign + alias removal (no API)",
        scope="object-local",
        success_signal=_WARN_ONLY,
        liveness_evidence="LLaDA fast-path tests",
        classification="EXTRACTED -> llada.fast_paths (#185, family provider; hooks wired LIVE)",
        claims=[
            (LLADA_PROVIDER, "block.apply_mlp = apply_lora_mlp_swiglu"),
            (LLADA_PROVIDER, "block.gate_proj = block.ff_proj"),
        ],
    ),
    _mutation(
        "modernbert_attention_fast_forward",
        owner="modernbert.fast_paths.patch_peft (#185 provider)",
        target="layer.attn.forward (instance)",
        applicability="CUDA",
        before="ModernBertAttention.forward (class)",
        after="MethodType(ModernBertAttention_fast_forward)",
        idempotent="yes",
        reversible="no API",
        scope="object-local",
        success_signal=_WARN_ONLY,
        liveness_evidence="ModernBERT tests",
        classification="EXTRACTED -> modernbert.fast_paths (#185, family provider)",
        claims=[
            (
                MODERNBERT_PROVIDER,
                "attn.forward = types.MethodType(ModernBertAttention_fast_forward",
            )
        ],
    ),
    _mutation(
        "modernbert_wo_fast_hook",
        owner="modernbert.fast_paths.patch_peft (#185 provider)",
        target="attn.apply_wo + o_proj aliasing",
        applicability="CUDA + eligibility gates",
        before="_original_apply_wo stub",
        after="apply_lora_o (unsloth) + attn.o_proj=attn.Wo alias",
        idempotent="yes",
        reversible="reassign",
        scope="object-local",
        success_signal=_WARN_ONLY,
        liveness_evidence="ModernBERT tests",
        classification="EXTRACTED -> modernbert.fast_paths (#185, family provider)",
        claims=[
            (MODERNBERT_PROVIDER, "attn.apply_wo = apply_lora_o"),
            (MODERNBERT_PROVIDER, "attn.o_proj = attn.Wo"),
        ],
    ),
    _mutation(
        "post_load_class_swap",
        owner="_apply_post_load_class_swap",
        target="model.__class__",
        applicability="model_type in _POST_LOAD_CLASS_SWAPS AND isinstance(model, wrapper bases)",
        before="upstream transformers class (e.g. DiffusionGemmaForBlockDiffusion)",
        after="UnturtleDiffusionGemmaForBlockDiffusion",
        idempotent="yes (isinstance guard)",
        reversible="reassign original class (no API)",
        scope="object-local",
        success_signal="silent on success; warning when the swap is refused",
        liveness_evidence="DiffusionGemma real-checkpoint tests (slow/gpu)",
        classification="REPLACE -> #186",
        claims=[(FDM, "model.__class__ = wrapper_cls")],
    ),
    _mutation(
        "instance_generate_deletion",
        owner="_apply_post_load_class_swap",
        target="model.__dict__['generate']",
        applicability="after class swap (unsloth FastModel installs an instance-level generate)",
        before="unsloth_base_fast_generate (instance attribute)",
        after="absent -> class-level generate shim wins",
        idempotent="yes (pop with default)",
        reversible="no (original saved as _old_generate by unsloth)",
        scope="object-local",
        success_signal="silent",
        liveness_evidence="DiffusionGemma generation tests",
        classification="REPLACE -> #186",
        claims=[(FDM, 'model.__dict__.pop("generate", None)')],
    ),
    _mutation(
        "generation_config_restoration",
        owner="_apply_post_load_class_swap",
        target="model.generation_config",
        applicability="post-swap, when unset and model can_generate",
        before="absent (unsloth load path skips the __init__ postamble)",
        after="checkpoint generation config, or from_model_config fallback",
        idempotent="yes (never overwrites a populated one)",
        reversible="reassign",
        scope="object-local",
        success_signal="silent",
        liveness_evidence="#96 regression tests",
        classification="REPLACE -> #186",
        claims=[(FDM, "model.generation_config = restored")],
    ),
    _mutation(
        "apply_stubs_install",
        owner="peft_preparation.install_apply_stubs (#185 PR 2)",
        target="module.apply_qkv / module.apply_o on every q_proj+o_proj module",
        applicability="always (from_pretrained and get_peft_model)",
        before="absent",
        after="_original_apply_qkv / _original_apply_o",
        idempotent="yes (hasattr guard)",
        reversible="delete attrs (no API)",
        scope="object-local",
        success_signal="silent",
        liveness_evidence="all fast-forward tests dispatch through the stubs",
        classification="EXTRACT -> #185",
        claims=[
            (PREP, "module.apply_qkv = _original_apply_qkv"),
            (PREP, "module.apply_o = _original_apply_o"),
        ],
    ),
    _mutation(
        "modernbert_wo_stub_install",
        owner="_install_modernbert_stubs",
        target="module.apply_wo",
        applicability="ModernBERT models (CPU and CUDA)",
        before="absent",
        after="_original_apply_wo",
        idempotent="yes",
        reversible="delete attr",
        scope="object-local",
        success_signal="silent",
        liveness_evidence="ModernBERT tests",
        classification="EXTRACT -> #185",
        claims=[
            (
                "unturtle/models/backbones/modernbert/_fast_forward.py",
                "module.apply_wo = _original_apply_wo",
            )
        ],
    ),
    _mutation(
        "llada_default_apply_mlp",
        owner="LLaDALlamaBlock.__init__",
        target="self.apply_mlp",
        applicability="constructor default (not a patch)",
        before="n/a (construction)",
        after="LLaDALlamaBlock._default_apply_mlp",
        idempotent="yes",
        reversible="n/a",
        scope="object-local",
        success_signal="n/a",
        liveness_evidence="LLaDA forward tests",
        classification="KEEP",
        claims=[
            (
                "unturtle/models/backbones/llada/modeling_llada.py",
                "self.apply_mlp = LLaDALlamaBlock._default_apply_mlp",
            )
        ],
    ),
    _mutation(
        "llada_default_apply_qkv",
        owner="LLaDABlock.__init__ / LLaDALlamaBlock.__init__ (#185 wiring)",
        target="self.apply_qkv",
        applicability="constructor default (not a patch)",
        before="n/a (construction)",
        after="LLaDALlamaBlock._default_apply_qkv",
        idempotent="yes",
        reversible="n/a",
        scope="object-local",
        success_signal="n/a",
        liveness_evidence="LLaDA forward tests + tests/models/test_llada_fast_paths.py (default stub == direct projection, bit-identical)",
        classification="KEEP",
        claims=[
            (
                "unturtle/models/backbones/llada/modeling_llada.py",
                "self.apply_qkv = LLaDALlamaBlock._default_apply_qkv",
            )
        ],
    ),
    _mutation(
        "llada_default_apply_o",
        owner="LLaDABlock.__init__ / LLaDALlamaBlock.__init__ (#185 wiring)",
        target="self.apply_o",
        applicability="constructor default (not a patch)",
        before="n/a (construction)",
        after="LLaDABlock._default_apply_o",
        idempotent="yes",
        reversible="n/a",
        scope="object-local",
        success_signal="n/a",
        liveness_evidence="LLaDA forward tests + tests/models/test_llada_fast_paths.py (default stub == direct projection, bit-identical)",
        classification="KEEP",
        claims=[
            (
                "unturtle/models/backbones/llada/modeling_llada.py",
                "self.apply_o = LLaDABlock._default_apply_o",
            )
        ],
    ),
    _mutation(
        "push_to_hub_patch",
        owner="unturtle.save.patch_saving_functions",
        target="model.push_to_hub (instance)",
        applicability="every get_peft_model output",
        before="class push_to_hub",
        after="MethodType(unturtle push wrapper), original kept as original_push_to_hub",
        idempotent="unverified (re-patch wraps the wrapper unless guarded — see tests)",
        reversible="original retained on the instance",
        scope="object-local",
        success_signal="silent",
        liveness_evidence="save tests",
        classification="EXTRACT -> #185",
        claims=[("unturtle/save.py", "original_model.push_to_hub = types.MethodType(")],
    ),
    _mutation(
        "rope_extension",
        owner="loading._extend_rope_if_possible (via loading._patch_for_diffusion; #185 PR 3)",
        target="rotary_emb / rotary_embedding modules exposing extend_rope_embedding",
        applicability=(
            "every from_pretrained load — but no unturtle rotary module defines "
            "extend_rope_embedding, so the call is a no-op on every current family"
        ),
        before="constructor-initialized RoPE state",
        after="unchanged (no-op; see #174 for the actual load-path buffer defect)",
        idempotent="yes (no-op)",
        reversible="n/a",
        scope="object-local",
        success_signal="debug log only",
        liveness_evidence=(
            "#174 PR 0 attribution: the inv_freq divergence seen by the "
            "persistence.native_fp probe is NOT this mutation — transformers' "
            "from_pretrained re-materializes non-persistent buffers with "
            "torch.empty_like and Dream/MDLM-DiT _init_weights never "
            "re-initialize them (verdict ROPE LOAD-PATH CAUSAL, "
            "docs/artifacts/174-persistence-attribution-v1.json)"
        ),
        classification="linked defect -> #174 (attributed and FIXED: Rotary/DreamRotaryEmbedding reset_parameters via _init_weights; this row itself is inert)",
        linked_issue=174,
        claims=[(LOADING, "rope.extend_rope_embedding(max_seq_length)")],
    ),
    _mutation(
        "liveness_probe_counters",
        owner="probe_liveness (#185 PR 0, observation)",
        target="instance attributes of APPLIED fast targets (apply_qkv / apply_o / apply_wo / apply_mlp / forward)",
        applicability="only inside probe_liveness, on modules the PatchReport lists as applied",
        before="the installed fast callable",
        after="a counting wrapper (functools.wraps; MethodType for bound forwards) — restored in a finally block",
        idempotent="yes (installs then restores every call)",
        reversible="yes — restored before returning; identity re-verified by tests",
        scope="object-local",
        success_signal="LivenessReport counters (per module:kind), never a warning",
        liveness_evidence="tests/test_patch_report_contract.py (liveness only after forward; originals restored)",
        classification="KEEP (descriptive; #185 PR 0 contract: installed != live)",
        linked_issue=185,
        claims=[(FDM, "module.__dict__[attr] = types.MethodType(counting, module)")],
    ),
    _mutation(
        "max_seq_length_propagation",
        owner="loading._propagate_max_seq_length",
        target="module.max_seq_length on every module",
        applicability="every load and PEFT wrap",
        before="absent",
        after="int attribute on every module",
        idempotent="yes",
        reversible="delete attrs",
        scope="object-local",
        success_signal="silent",
        liveness_evidence="unsloth GC reads it",
        classification="EXTRACT -> #185",
        claims=[
            (LOADING, "module.max_seq_length = max_seq_length"),
            (LOADING, "internal.max_seq_length = max_seq_length"),
            (LOADING, "model.max_seq_length = max_seq_length"),
        ],
    ),
    _mutation(
        "dream_generation_config_from_pretrained",
        owner="DreamModel.from_pretrained (family hook)",
        target="model.generation_config",
        applicability="Dream loads with a checkpoint generation_config",
        before="plain GenerationConfig from the upstream postamble",
        after="DreamGenerationConfig.from_pretrained(...)",
        idempotent="yes",
        reversible="reassign",
        scope="object-local",
        success_signal="silent",
        liveness_evidence=(
            "persistence.generation_reload probe: reloaded model carries "
            "DreamGenerationConfig, yet the default-config generate path "
            "still crashes (it never consults self.generation_config — #189)"
        ),
        classification="linked defect -> #189",
        linked_issue=189,
        claims=[
            (
                "unturtle/models/backbones/dream/modeling_dream.py",
                "_model.generation_config = DreamGenerationConfig.from_pretrained(",
            )
        ],
    ),
    _mutation(
        "in_model_gradient_checkpointing_state",
        owner="family model constructors / GC toggles",
        target="module.gradient_checkpointing (in-model field)",
        applicability="constructor defaults and transformers GC API",
        before="n/a (construction / toggle)",
        after="bool flag consumed by each family's forward",
        idempotent="yes",
        reversible="yes",
        scope="object-local",
        success_signal="n/a (in-model state, not a patch)",
        liveness_evidence="gradient-checkpointing training tests",
        classification="KEEP",
        claims=[
            (
                "unturtle/models/backbones/dream/modeling_dream.py",
                "self.gradient_checkpointing = False",
            ),
            (
                "unturtle/models/backbones/mdlm_dit/modeling_mdlm_dit.py",
                "self.gradient_checkpointing = False",
            ),
            (
                "unturtle/models/backbones/llada/modeling_llada.py",
                "gradient_checkpointing = enable",
            ),
        ],
    ),
    _mutation(
        "gc_mode_application",
        owner="peft_preparation.apply_gradient_checkpointing_mode (#185 PR 2)",
        target="module.gradient_checkpointing + model._unturtle_gradient_checkpointing_mode",
        applicability="get_peft_model / for_inference / for_training",
        before="per-module flags",
        after="uniform bool + tracked mode attr",
        idempotent="yes",
        reversible="yes (mode round-trips)",
        scope="object-local",
        success_signal="silent",
        liveness_evidence="inference_context round-trip tests",
        classification="KEEP",
        claims=[(PREP, "module.gradient_checkpointing = bool(mode)")],
    ),
    _mutation(
        "kbit_preparation_env",
        owner="unturtle.save.prepare_model_for_kbit_training -> unsloth zoo",
        target="os.environ['UNSLOTH_MIXED_PRECISION'] + GC patch/unpatch state",
        applicability="every quantized get_peft_model",
        before="env unset",
        after="UNSLOTH_MIXED_PRECISION=float32 (measured)",
        idempotent="yes (same value)",
        reversible="no (env persists for the process)",
        scope="process-global",
        success_signal="silent",
        liveness_evidence=(
            "fourbit-contract probe: UNSLOTH_MIXED_PRECISION None->float32; "
            "a per-model API mutating process state — ledgered separately "
            "from object-local rows (#184 requirement)"
        ),
        classification="EXTRACT -> #185",
        linked_issue=177,
        claims=[(PREP, "model = prepare_model_for_kbit_training(")],
    ),
    _mutation(
        "autoclass_registration",
        owner="unturtle.models import side effects",
        target="transformers CONFIG_MAPPING / AutoModel mappings",
        applicability="once per process at import",
        before="no unturtle model_types registered",
        after=(
            "mdlm-dit, modernbert-diffusion, tiny-a2d-{llama,qwen2,qwen3} "
            "registered; Dream and LLaDA are NOT (import probe evidence)"
        ),
        idempotent="guarded (fires once per model_type)",
        reversible="no",
        scope="process-global",
        success_signal="silent",
        liveness_evidence="import probe: autoclass config_mapping_extra",
        classification="KEEP (asymmetry recorded: Dream/LLaDA unregistered)",
        claims=[
            (
                "unturtle/models/backbones/mdlm_dit/modeling_mdlm_dit.py",
                "transformers.AutoConfig.register(",
            ),
            (
                "unturtle/models/backbones/modernbert/modeling.py",
                "transformers.AutoConfig.register(",
            ),
            (
                "unturtle/models/conversion/a2d/tiny_a2d/modeling_llama.py",
                "transformers.AutoConfig.register(",
            ),
            (
                "unturtle/models/conversion/a2d/tiny_a2d/modeling_qwen2.py",
                "transformers.AutoConfig.register(",
            ),
            (
                "unturtle/models/conversion/a2d/tiny_a2d/modeling_qwen3.py",
                "transformers.AutoConfig.register(",
            ),
        ],
    ),
    _mutation(
        "default_registry_bootstrap",
        owner="unturtle import chain (ensure_default_hub)",
        target="unturtle.registry._default_hub",
        applicability="once per process, at import unturtle",
        before="None",
        after="bootstrapped RegistryHub with all builtin axes (import probe evidence)",
        idempotent="yes (memoized)",
        reversible="no API",
        scope="process-global",
        success_signal="silent",
        liveness_evidence="import probe: default_registry_hub.bootstrapped=true with axis contents",
        classification="KEEP",
        claims=[("unturtle/registry.py", "def ensure_default_hub")],
    ),
]

#: Mutation-shaped line patterns the scanner enumerates. Every hit must be
#: claimed by a ledger row.
SCAN_PATTERNS = (
    r"__class__ =",
    r"types\.MethodType\(",
    r"\.forward = ",
    r"\.apply_qkv = ",
    r"\.apply_o = ",
    r"\.apply_mlp = ",
    r"\.apply_wo = ",
    r"__dict__\[\"o_proj\"\] = ",
    r"__dict__\.pop\(\"generate\"",
    r"os\.environ\[",
    r"\.generation_config = ",
    r"extend_rope_embedding\(",
    r"\.max_seq_length = ",
    r"\.gradient_checkpointing = ",
    r"AutoConfig\.register\(",
    r"\.o_proj = attn\.Wo",
    r"\.gate_proj = ff_proj",
    r"prepare_model_for_kbit_training\(",
)

#: Production files the scanner walks. tests/, benchmarks/ and diagnostics/
#: are observation code, not production mutations.
SCAN_INCLUDE = ("unturtle",)
SCAN_EXCLUDE_PARTS = ("diagnostics",)


def scan_mutation_sites() -> list[dict]:
    hits: list[dict] = []
    pattern = re.compile("|".join(f"(?:{p})" for p in SCAN_PATTERNS))
    for base in SCAN_INCLUDE:
        for path in sorted((REPO_ROOT / base).rglob("*.py")):
            if any(part in SCAN_EXCLUDE_PARTS for part in path.parts):
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                stripped = line.strip()
                if stripped.startswith("#") or "``" in stripped:
                    continue
                # doc/comment tails: ignore matches inside trailing comments
                code = stripped.split("#", 1)[0]
                if not pattern.search(code):
                    continue
                # definitions and comparisons are not mutations
                if (
                    code.startswith(("def ", "class "))
                    and "ensure_default_hub" not in code
                ):
                    continue
                hits.append({"file": rel, "line": lineno, "code": code.strip()})
    return hits


def reconcile_ledger(hits: list[dict]) -> dict:
    # Coverage direction: every mutation-shaped scan hit must be claimed.
    claimed_flags = [False] * len(hits)
    for row in MUTATION_LEDGER:
        for claim in row["claims"]:
            for index, hit in enumerate(hits):
                if (
                    hit["file"].endswith(claim["file"])
                    and claim["contains"] in hit["code"]
                ):
                    claimed_flags[index] = True
    unclaimed = [hit for index, hit in enumerate(hits) if not claimed_flags[index]]

    # Anchoring direction: every claim must exist in the production tree.
    # Checked against file CONTENT, not only scan hits — some anchors (e.g.
    # the registry bootstrap definition) are not mutation-shaped lines.
    file_cache: dict[str, str] = {}
    rows = []
    for row in MUTATION_LEDGER:
        matched = []
        for claim in row["claims"]:
            for path in sorted((REPO_ROOT / "unturtle").rglob("*.py")):
                rel = path.relative_to(REPO_ROOT).as_posix()
                if not rel.endswith(claim["file"]):
                    continue
                text = file_cache.setdefault(rel, path.read_text(encoding="utf-8"))
                if claim["contains"] in text:
                    matched.append({"file": rel, "contains": claim["contains"]})
                    break
        status = "observed" if len(matched) == len(row["claims"]) else "unverified"
        rows.append(
            {
                **row,
                "row": make_row(
                    status,
                    reason=None
                    if status == "observed"
                    else "claim not found in production tree",
                    source="static scan + curated ledger",
                    owner=row["owner"],
                    evidence={"matched_claims": matched},
                ),
            }
        )
    return {"rows": rows, "unclaimed_hits": unclaimed, "scanned_hits": len(hits)}


# ---------------------------------------------------------------------------
# Probe orchestration
# ---------------------------------------------------------------------------


def run_probe(case: str, payload: dict | None, *, device: str) -> tuple[dict, dict]:
    """Run one probe subprocess; returns (result, provenance)."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        out_path = pathlib.Path(handle.name)
    command = [
        sys.executable,
        str(PROBE),
        case,
        "--out",
        str(out_path),
        "--json",
        json.dumps(payload or {}),
    ]
    env = dict(_BASE_ENV)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["UNTURTLE_EXPECTED_ROOT"] = str(REPO_ROOT)
    env["PYTHONHASHSEED"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = device
    proc = subprocess.run(
        command, cwd=REPO_ROOT, env=env, capture_output=True, text=True
    )
    provenance = {
        "case": case,
        "args": payload or {},
        "command": [c.replace(str(REPO_ROOT), "<repo>") for c in command],
        "cwd": "<repo>",
        "exit_code": proc.returncode,
        "cuda_visible_devices": device,
    }
    if proc.returncode == 3:
        raise SystemExit(
            f"IMPORT ROOT VIOLATION in probe {case!r}: the probe imported an "
            f"unturtle outside {REPO_ROOT}. Refusing to produce an artifact.\n"
            f"{out_path.read_text() if out_path.exists() else proc.stderr[-500:]}"
        )
    if out_path.exists() and out_path.read_text().strip():
        result = json.loads(out_path.read_text())
        out_path.unlink()
    else:
        result = {
            "status": "blocked",
            "reason": f"probe produced no output (exit {proc.returncode}); "
            f"stderr tail: {proc.stderr[-300:]}",
        }
    return result, provenance


def _normalize_strings(node):
    """Scrub volatile path fragments from every string in the artifact."""
    if isinstance(node, dict):
        return {k: _normalize_strings(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_normalize_strings(v) for v in node]
    if isinstance(node, str):
        node = node.replace(str(REPO_ROOT), "<repo>")
        node = re.sub(r"/tmp/[^\s'\"]+", "<tmp>", node)
        node = re.sub(r"/grouper/[^\s'\"]+", "<path>", node)
        return node
    return node


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def build_verdicts(sections: dict) -> dict:
    def verdict(area, value, evidence, *, status="observed", reason=None):
        return {
            "area": area,
            "verdict": value,
            "row": make_row(
                status,
                reason=reason,
                source="characterization sections",
                owner="#184",
                evidence=evidence,
            ),
        }

    fourbit = sections.get("persistence", {}).get("fourbit_contract", {})
    native_fp = sections.get("persistence", {}).get("native_fp", {})
    public_api = sections.get("public_api", {})
    unresolved_exports = [
        name
        for name, sym in public_api.get("unturtle", {}).get("symbols", {}).items()
        if not sym.get("resolved")
    ]

    return {
        "transformers_native_model_inheritance": verdict(
            "Transformers-native model inheritance",
            "KEEP",
            {"models_section": "all families subclass PreTrainedModel (MRO rows)"},
        ),
        "family_specific_configs_models": verdict(
            "Family-specific configs/models",
            "KEEP",
            {"models_section": "per-family rows"},
        ),
        "methodspec_registryhub": verdict(
            "MethodSpec / RegistryHub",
            "KEEP",
            {"import_probe": "default hub bootstrapped with 7 axes at import"},
        ),
        "explicit_plugin_loading": verdict(
            "Explicit plugin loading",
            "KEEP",
            {"import_probe": "unturtle.plugins import registers nothing by itself"},
        ),
        "fast_diffusion_model_facade": verdict(
            "FastDiffusionModel public facade", "KEEP", {"public_api": "exported"}
        ),
        "fast_diffusion_model_internal_ownership": verdict(
            "FastDiffusionModel internal ownership",
            "EXTRACT -> #185",
            {"runtime_mutations": "loader/PEFT/patch rows classified EXTRACT"},
        ),
        "integration_callbacks_into_loader": verdict(
            "Integration callbacks into loader private functions",
            "EXTRACT -> #185",
            {"integrations_section": "resolver seams"},
        ),
        "installation_only_fast_path_success": verdict(
            "Installation-only fast-path success",
            "REPLACE -> #185",
            {
                "mutation_ledger": "success_signal is warning-only on every fast hook row",
                "fourbit_contract": "liveness now proven by execution, not installation (#177)",
            },
        ),
        "signature_guessing_generation": verdict(
            "Signature-guessing generation invocation",
            "REPLACE -> #186",
            {
                "generation_probe": (
                    "sampler resolves call signatures via inspection — a "
                    "signature-hiding wrapper changed dispatch during probing"
                )
            },
        ),
        "diffusion_gemma_class_swap": verdict(
            "DiffusionGemma runtime class swap",
            "REPLACE -> #186",
            {
                "mutation_ledger": "post_load_class_swap + instance_generate_deletion rows"
            },
        ),
        "root_export_growth": verdict(
            "Root export growth",
            "DEPRECATE",
            {
                "public_api": {
                    "unturtle_all_count": public_api.get("unturtle", {}).get(
                        "all_count"
                    ),
                    "declared_but_none": unresolved_exports,
                }
            },
        ),
        "universal_hierarchy": verdict(
            "Universal model/state/solver/trainer hierarchy",
            "DO NOT CREATE",
            {"models_section": "families are peers; no shared unturtle base class"},
        ),
        "get_peft_model_random_state": verdict(
            "get_peft_model(random_state=...) contract",
            "RESOLVED -> #188 (seeded inside a forked torch RNG)",
            {
                "process_global_state": (
                    "rng_contract row: same random_state, different pre-RNG "
                    "consumption, SAME lora_A digests; caller RNG untouched"
                ),
                "rng_contract_classification": sections.get("process_global_state", {})
                .get("rng_contract", {})
                .get("classification"),
            },
        ),
        "save_reload_global_state_instability": verdict(
            "Full-suite save/reload global-state instability",
            "RESOLVED -> #174 (uninitialized non-persistent rotary buffers; fixed)",
            {
                "persistence.native_fp": {
                    "state_dict": "identical",
                    "buffer_diffs": native_fp.get("buffer_diffs"),
                    "output_bit_identical": native_fp.get("output", {}).get(
                        "bit_identical"
                    ),
                    "attribution": (
                        "#174 PR 0: uninitialized non-persistent rotary buffers "
                        "after from_pretrained (torch.empty_like + no-op "
                        "_init_weights), not _extend_rope_if_possible"
                    ),
                }
            },
        ),
        "dream_default_generation_config": verdict(
            "Dream unified generate with default config",
            "linked defect -> #189",
            {
                "generation_probe": (
                    "default-config runs raise AttributeError eps on "
                    "transformers 5.x; explicit DreamGenerationConfig executes"
                )
            },
        ),
        "dtype_gate_fail_open": verdict(
            "Fast-path dtype gate: unresolvable embedding structure",
            "unverified -> #185 SupportResult must model it",
            {
                "reason_code": "input_embedding_unresolvable",
                "note": (
                    "fail-open today (per-layer gates still apply); #185 must "
                    "represent this as unverified, never compatible"
                ),
            },
            status="unverified",
            reason="input_embedding_unresolvable",
        ),
        "registry_hub_explicit_contract": verdict(
            "RegistryHub explicit-hub contract",
            "KEEP",
            {
                "registry_hub": {
                    "fresh_empty_hub": sections.get("registry_hub", {})
                    .get("fresh_empty_hub", {})
                    .get("all_axes_empty"),
                    "bootstrap_deterministic": sections.get("registry_hub", {})
                    .get("explicit_builtin_bootstrap", {})
                    .get("deterministic_across_two_bootstraps"),
                    "repeat_bootstrap_behavior": sections.get("registry_hub", {})
                    .get("repeat_bootstrap", {})
                    .get("behavior"),
                    "hubs_isolated": not sections.get("registry_hub", {})
                    .get("hub_isolation", {})
                    .get("backing_storage_shared", True),
                    "note": (
                        "re-bootstrap of the same hub is DUPLICATE REJECTION "
                        "(ValueError, counts unchanged), not idempotent — "
                        "frozen as observed; #185/#186 and external plugins "
                        "can rely on supplied hubs being empty, deterministic "
                        "to bootstrap, and storage-isolated"
                    ),
                }
            },
        ),
        "fourbit_peft_contract": verdict(
            "4-bit + PEFT preparation/fast-path contract",
            "KEEP (fixed by #177)",
            {
                "fourbit_contract": {
                    "preparation_owner": fourbit.get("preparation_owner"),
                    "embedding_dtype": fourbit.get("embedding_dtype"),
                    "fast_path_verdict": fourbit.get("fast_path_verdict"),
                    "fallback_uniform_skip": fourbit.get("fallback_behavior", {}).get(
                        "uniform_skip"
                    ),
                }
            },
        ),
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def producer_info() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout.strip()

    dirty = git("status", "--porcelain")

    # importlib.metadata, never imports: importing torch/unsloth here would
    # mutate THIS process's environment, which the probes inherit as their
    # baseline (see _BASE_ENV).
    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "commit": git("rev-parse", "HEAD"),
        "worktree_clean": dirty == "",
        "dirty_paths": sorted(line.split(None, 1)[-1] for line in dirty.splitlines()),
        "python": platform.python_version(),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "unsloth": version("unsloth"),
        "peft": version("peft"),
        "platform": platform.platform(),
        "import_root": "<repo>",
        "probes": [],  # filled during capture
    }


def capture(device: str, *, allow_dirty: bool = False) -> dict:
    producer = producer_info()
    if not producer["worktree_clean"] and not allow_dirty:
        raise SystemExit(
            "worktree is not clean; commit producer code first "
            f"(dirty: {producer['dirty_paths'][:10]})"
        )

    def probe(case: str, payload: dict | None = None) -> dict:
        result, provenance = run_probe(case, payload, device=device)
        producer["probes"].append(provenance)
        return result

    imports = {name: probe("import", {"module": name}) for name in IMPORT_TARGETS}
    public_api = probe("public-api")
    models = {name: probe("model", {"family": name}) for name in MODEL_FAMILIES}
    integrations = probe("integrations")
    generation = {
        name: probe("generation", {"family": name}) for name in GENERATION_FAMILIES
    }
    persistence = {
        case: probe("persistence", {"case": case}) for case in PERSISTENCE_CASES
    }
    persistence["fourbit_contract"] = probe("fourbit-contract")
    process_global = {
        case: probe("process-global", {"case": case}) for case in PROCESS_GLOBAL_CASES
    }
    registry_hub = probe("registry-hub")
    process_global["unsloth_environment_mutation"] = persistence[
        "fourbit_contract"
    ].get(
        "unsloth_environment_mutation", {"status": "blocked", "reason": "probe blocked"}
    )

    hits = scan_mutation_sites()
    ledger = reconcile_ledger(hits)

    sections = {
        "imports": imports,
        "public_api": public_api,
        "models": models,
        "integrations": integrations,
        "runtime_mutations": ledger,
        "generation": generation,
        "persistence": persistence,
        "process_global_state": process_global,
        "registry_hub": registry_hub,
    }
    artifact = {
        "schema_version": 1,
        "producer": producer,
        **sections,
        "verdicts": build_verdicts(sections),
    }
    artifact = _normalize_strings(artifact)
    artifact["semantic_digest"] = semantic_digest(artifact)
    return artifact


# ---------------------------------------------------------------------------
# Markdown generation (from the JSON — never hand-copied)
# ---------------------------------------------------------------------------


def render_markdown(artifact: dict) -> str:
    lines: list[str] = []
    add = lines.append
    producer = artifact["producer"]
    add("# Unturtle architecture contract v1 (#184)")
    add("")
    add(
        "Generated by `benchmarks/architecture/capture_contract.py` from "
        "`docs/artifacts/184-architecture-contract-v1.json` — do not edit by "
        "hand; regenerate instead."
    )
    add("")
    add(f"- producer commit: `{producer['commit']}`")
    add(f"- semantic digest: `{artifact['semantic_digest']}`")
    add(f"- worktree clean at capture: `{producer['worktree_clean']}`")
    add("")

    add("## Import side effects")
    add("")
    add(
        "| module | CUDA initialized | env keys added | hub bootstrapped | AutoConfig extras |"
    )
    add("|---|---|---|---|---|")
    for name, row in artifact["imports"].items():
        torch_state = row.get("torch", {})
        hub = row.get("default_registry_hub", {})
        add(
            f"| `{name}` | {torch_state.get('cuda_initialized')} | "
            f"{len(row.get('environ_added_keys', []))} | "
            f"{hub.get('bootstrapped', hub.get('default_hub_created'))} | "
            f"{len(row.get('autoclass', {}).get('config_mapping_extra', []))} |"
        )
    add("")

    add("## Public API")
    add("")
    api = artifact["public_api"]
    for module_name in ("unturtle", "unturtle.models"):
        described = api.get(module_name, {})
        unresolved = [
            k for k, v in described.get("symbols", {}).items() if not v.get("resolved")
        ]
        add(
            f"- `{module_name}`: {described.get('all_count')} symbols in "
            f"`__all__`; declared-but-None: {sorted(unresolved) or 'none'}"
        )
    add("")

    add("## Model contracts")
    add("")
    add("| family | class | model_type | generate owner | AutoConfig registered |")
    add("|---|---|---|---|---|")
    for family, row in artifact["models"].items():
        if row.get("status") != "observed":
            add(f"| {family} | blocked: {row.get('reason')} | — | — | — |")
            continue
        gen = row["method_owners"].get("generate", {})
        add(
            f"| {family} | `{row['model_class']}` | `{row['model_type']}` | "
            f"`{gen.get('defined_in')}` | {row['autoclass_config_registered']} |"
        )
    add("")

    add("## Runtime mutation ledger")
    add("")
    ledger = artifact["runtime_mutations"]
    add(
        f"{len(ledger['rows'])} curated rows over {ledger['scanned_hits']} "
        f"scanned mutation-shaped sites; unclaimed: {len(ledger['unclaimed_hits'])}"
    )
    add("")
    add("| mutation | owner | scope | success signal | classification |")
    add("|---|---|---|---|---|")
    for row in ledger["rows"]:
        add(
            f"| {row['mutation_id']} | `{row['owner']}` | {row['scope']} | "
            f"{row['success_signal'][:60]} | {row['classification']} |"
        )
    add("")

    add("## Generation execution map")
    add("")
    for family, row in artifact["generation"].items():
        add(f"### {family}")
        add("")
        add("| algorithm | default-config result | invoked | NFE |")
        add("|---|---|---|---|")
        for algorithm, result in sorted(row.get("per_algorithm", {}).items()):
            default = result.get("default_config_run", {})
            explicit = result.get("explicit_config_run")
            active = explicit if explicit and not explicit.get("raised") else default
            outcome = (
                "ok"
                if not default.get("raised")
                else f"raised `{str(default.get('raised'))[:60]}`"
            )
            add(
                f"| {algorithm} | {outcome} | "
                f"{', '.join(active.get('invoked_methods', [])) or '—'} | "
                f"{active.get('nfe')} |"
            )
        add("")

    add("## Persistence matrix")
    add("")
    for case, row in artifact["persistence"].items():
        add(f"- **{case}**: status={row.get('status')}")
        output = row.get("output")
        if output:
            volatile = output.get("volatile", {})
            add(
                f"  - output bit_identical={output['bit_identical']}, "
                f"within_relnorm_0.05={output.get('within_rel_norm_0p05')}"
                + (
                    f" (this capture: max|Δ|={volatile['max_abs_delta']:.3e}, "
                    f"relnorm={volatile['relative_norm']:.3e} — volatile)"
                    if volatile
                    else ""
                )
            )
        if row.get("buffer_diffs"):
            add(f"  - buffer diffs: {row['buffer_diffs']}")
    add("")

    add("## RegistryHub explicit-hub contract")
    add("")
    hub_section = artifact.get("registry_hub", {})
    add(
        f"- fresh `RegistryHub()`: all axes empty = "
        f"{hub_section.get('fresh_empty_hub', {}).get('all_axes_empty')}; "
        f"side effects: {hub_section.get('fresh_empty_hub', {}).get('surroundings')}"
    )
    bootstrap_cell = hub_section.get("explicit_builtin_bootstrap", {})
    add(
        f"- explicit builtin bootstrap deterministic across two hubs: "
        f"{bootstrap_cell.get('deterministic_across_two_bootstraps')}"
    )
    for axis, names in sorted(bootstrap_cell.get("ordered_axis_names", {}).items()):
        add(f"  - {axis}: {', '.join(names)}")
    add(
        f"- repeat bootstrap: {hub_section.get('repeat_bootstrap', {}).get('behavior')} "
        f"(`{hub_section.get('repeat_bootstrap', {}).get('raised')}`)"
    )
    isolation_cell = hub_section.get("hub_isolation", {})
    add(
        f"- isolation: leaked_to_other_hub="
        f"{isolation_cell.get('sentinel_leaked_to_other_hub')}, "
        f"leaked_to_default={isolation_cell.get('sentinel_leaked_to_default_hub')}, "
        f"backing_storage_shared={isolation_cell.get('backing_storage_shared')}"
    )
    add("")

    add("## Verdicts")
    add("")
    add("| area | verdict | status |")
    add("|---|---|---|")
    for entry in artifact["verdicts"].values():
        add(f"| {entry['area']} | {entry['verdict']} | {entry['row']['status']} |")
    add("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0", help="CUDA_VISIBLE_DEVICES for probes")
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate and compare semantic digests against the committed artifact",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="development only: capture from a dirty worktree (never for the committed artifact)",
    )
    args = parser.parse_args()

    artifact = capture(args.device, allow_dirty=args.allow_dirty)

    if args.check:
        committed = json.loads(ARTIFACT_PATH.read_text())
        fresh, existing = artifact["semantic_digest"], committed["semantic_digest"]
        recomputed = semantic_digest(committed)
        print(f"committed digest : {existing}")
        print(f"recomputed       : {recomputed}")
        print(f"fresh capture    : {fresh}")
        if existing != recomputed:
            raise SystemExit("committed artifact digest does not match its content")
        if fresh != existing:
            raise SystemExit("fresh capture diverges from the committed artifact")
        print("deterministic: fresh capture matches the committed artifact")
        return

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    MARKDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
    MARKDOWN_PATH.write_text(render_markdown(artifact) + "\n")
    print(f"wrote {ARTIFACT_PATH.relative_to(REPO_ROOT)}")
    print(f"wrote {MARKDOWN_PATH.relative_to(REPO_ROOT)}")
    print(f"semantic digest: {artifact['semantic_digest']}")
    unclaimed = artifact["runtime_mutations"]["unclaimed_hits"]
    if unclaimed:
        print(f"WARNING: {len(unclaimed)} unclaimed mutation sites:")
        for hit in unclaimed[:20]:
            print(f"  {hit['file']}:{hit['line']}: {hit['code'][:80]}")
        raise SystemExit(
            "mutation ledger incomplete — claim or exclude the sites above"
        )


if __name__ == "__main__":
    main()
