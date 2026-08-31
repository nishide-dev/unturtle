# unturtle-elf — ELF reference method pack (EXPERIMENTAL)

ELF: Embedded Language Flows (arXiv:2605.10938) as an Unturtle method pack
(#153). Reference/checkpoint **parity + generation/evaluation only** — no
training reproduction (that is #154).

- Implementation oracle: official `pytorch_elf` branch of
  https://github.com/lillian039/ELF (MIT) @ `b29d8833`.
- Reference checkpoint: `embedded-language-flows/ELF-B-owt-torch`
  @ `146f8413` (EMA weights, T5-small embedding space, max_length 1024).
- The denoiser/sampler ports in this pack follow the official PyTorch
  reference operation-for-operation; divergences are documented inline and
  in the #153 Stage-0 freeze.

Install (editable, into the host Unturtle env):

```bash
uv pip install -e packs/unturtle-elf
```

Loading is explicit — importing this package registers nothing:

```python
from unturtle.plugins import load_plugins
from unturtle.registry import RegistryHub, bootstrap_builtin_hub

hub = bootstrap_builtin_hub(RegistryHub())
load_plugins(hub, names=["elf"])
```

No stable ABI, no capability promotion, no cross-family claims: this pack
fills the `embedding_flow` Tier-A role of the #152 frontier protocol only.


## Lifecycle (see packs/README.md)

**Stage: reference** (since 2026-08; #153 Stage-0 freeze). Parity + generation/
evaluation against the pinned oracle only.

- **Compatibility scope**: the host Unturtle at this repo revision;
  oracle `lillian039/ELF@b29d8833`; checkpoint
  `embedded-language-flows/ELF-B-owt-torch@146f8413` (T5-small embedding
  space, max_length 1024). Other revisions are out of scope.
- **Entry points**: `unturtle_elf.plugin` (explicit `load_plugins` target);
  `unturtle_elf.training.elf_training_loss` (training-mechanics oracle
  differential only — not a training product); generation/eval via the
  registered hub algorithms.
- **Test command**: `uv run python -m pytest tests/test_elf_pack_parity.py
  tests/test_elf_training_mechanics.py -q` (CPU; oracle-differential tests
  additionally need the machine-local `dev/repos/elf` clone and skip without
  it).
- **Training**: NOT provided. Training reproduction is #154; until it lands,
  this pack trains nothing and promises nothing about training.
- **Retirement condition → archived**: when #154 concludes without adopting
  ELF, or when keeping parity green against the frozen oracle requires
  changes to core seams, or when the pinned checkpoint becomes unavailable
  and no re-pin is warranted.
