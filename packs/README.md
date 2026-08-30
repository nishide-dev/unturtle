# packs/ — separate-distribution method packs

Each directory here is its **own distribution** (own `pyproject.toml`,
installed explicitly, e.g. `uv pip install -e packs/unturtle-elf`). The design
invariants are frozen (#205; they predate it and do not change here):

- packs are **not** part of the root wheel;
- loading is **explicit** (`unturtle.plugins.load_plugins`) — importing a pack
  registers nothing and never mutates the default hub;
- packs work against **isolated hubs** (`RegistryHub`) as well as an
  explicitly-bootstrapped default hub;
- **no paper-specific branches in core**: a pack adapts to core seams, never
  the reverse;
- a pack's existence is **not a capability promotion** — core docs do not
  advertise pack methods as supported Unturtle capabilities;
- **core migration is not the default goal** of any pack;
- splitting packs into separate repositories is out of scope for #205 —
  in-tree separate distribution stays until a concrete benefit is demonstrated
  (and not before #151 completes).

## Lifecycle stages

Every pack declares exactly one stage in its README:

| Stage | Meaning | Maintenance promise |
|---|---|---|
| **reference** | Faithful port of a published method against a frozen upstream oracle + pinned checkpoints; parity/eval only | Kept importable and parity-green against the pinned oracle revision; no feature work; breaking-core changes fix the pack or trigger *archived* |
| **research** | Active experimentation on top of core seams; interfaces may change without notice | Best effort; may break between commits |
| **maintained integration** | A method the project actively supports and tests as part of its own surface | Fast-suite coverage; API stability within the repo's normal deprecation practice |
| **archived** | Retired; kept for provenance only | None — not installed by any documented flow; parity tests skipped/removed |

Stage transitions are recorded in the pack README (date + reason). The
**retirement condition** below in each pack README states exactly when the
pack moves to *archived*.

## Current packs

| Pack | Stage | Method | Scope |
|---|---|---|---|
| `unturtle-elf` | **reference** | ELF — Embedded Language Flows (arXiv:2605.10938) | checkpoint parity + generation/evaluation; training reproduction is #154, not promised here |
| `unturtle-flm` | **reference** | FLM/FMLM — Flow Map LMs (arXiv:2602.16813) | checkpoint parity + generation/evaluation; no training/distillation reproduction |
