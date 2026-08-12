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

No stable ABI, no capability promotion, no cross-family claims: this pack
fills the `flow_map` Tier-A role (and provides the one-hot Euclidean flow
control) for the #152 frontier protocol.
