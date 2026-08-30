# unturtle-flm — FLM/FMLM reference method pack (EXPERIMENTAL)

FLM/FMLM: Flow Map Language Models (arXiv:2602.16813) as an Unturtle method
pack (#155). Reference/checkpoint **parity + generation/evaluation only** —
no training or distillation reproduction.

- Implementation oracle: https://github.com/david3684/flm (Apache-2.0)
  @ `a1918d51`.
- Reference checkpoints: `david3684/FLM-B-OWT` @ `624471b9` and
  `david3684/FMLM-B-OWT` @ `483ea1b3` (gpt2 tokenizer, length 1024,
  one-hot Euclidean state).
- Two methods, registered separately because their semantics differ
  structurally (two-time flow map vs one-time Euler flow):
  - `flm` — multi-step continuous flow (Euler ODE over the tau grid);
  - `fmlm` — distilled flow map, one/few-step (NOT `flm` with steps=1).
- Names deliberately do not collide with Unturtle's historical `flowlm`
  prototype, which is untouched.

Install (editable, into the host Unturtle env; requires flash-attn for the
GPU path — the oracle's DiT hard-imports it):

```bash
uv pip install -e packs/unturtle-flm
```

Loading is explicit — importing this package registers nothing:

```python
from unturtle.plugins import load_plugins
from unturtle.registry import RegistryHub, bootstrap_builtin_hub

hub = bootstrap_builtin_hub(RegistryHub())
load_plugins(hub, names=["flm"])
```

Parity evidence is two-tier (disclosed): CPU-tier parity is EXACT
(`torch.equal` against the oracle through its own `use_jvp_attn=True`
pure-torch path — a path the official sampler itself never takes); the
GPU/flash tier has no bitwise oracle test and is validated statistically
by the official 3-cell band reproduction (all within ~1% of published
values). Every frontier record carries `use_jvp_attn` so artifacts are
self-describing.

No stable ABI, no capability promotion, no cross-family claims: this pack
fills the `flow_map` Tier-A role (and provides the one-hot Euclidean flow
control) for the #152 frontier protocol.


## Lifecycle (see packs/README.md)

**Stage: reference** (since 2026-08; #155 freeze). Parity + generation/
evaluation against the pinned oracle only.

- **Compatibility scope**: the host Unturtle at this repo revision;
  oracle `david3684/flm@a1918d51`; checkpoints `david3684/FLM-B-OWT@624471b9`
  and `david3684/FMLM-B-OWT@483ea1b3` (gpt2 tokenizer, length 1024, one-hot
  Euclidean state). flash-attn required for the GPU path (oracle DiT
  hard-imports it).
- **Entry points**: `unturtle_flm.plugin` (explicit `load_plugins` target);
  registered algorithms `flm` (multi-step Euler) and `fmlm` (distilled flow
  map — NOT flm with steps=1).
- **Test command**: `uv run python -m pytest tests/test_flm_pack_parity.py -q`
  (parity tiers marked by resource requirement; GPU tiers need flash-attn).
- **Training**: NOT provided; no training or distillation reproduction is
  promised at this stage.
- **Retirement condition → archived**: when the FMLM posterior-refinement
  line (#162) concludes without adopting this pack as its base, or when
  parity against the frozen oracle can no longer be kept green without core
  changes, or when the pinned checkpoints become unavailable.
