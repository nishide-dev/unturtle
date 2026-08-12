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
