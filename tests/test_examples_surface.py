"""#205 PR 3 — the examples/ surface and its README stay bidirectionally
consistent: every shipped file is documented, every documented path exists."""

from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"
README = EXAMPLES / "README.md"


def _shipped_files() -> set[str]:
    return {
        str(p.relative_to(EXAMPLES))
        for p in EXAMPLES.rglob("*")
        if p.is_file() and p.suffix in {".py", ".yaml"} and "__pycache__" not in p.parts
    }


def _readme_paths() -> set[str]:
    text = README.read_text(encoding="utf-8")
    # `configs/x.yaml` / `grpo_diffu_train_smoke.py` style backticked paths,
    # plus explicit examples/… command lines.
    hits = set(re.findall(r"`((?:[\w./-]+/)?[\w-]+\.(?:py|yaml))`", text))
    hits |= {
        m.removeprefix("examples/")
        for m in re.findall(r"examples/([\w./-]+\.(?:py|yaml))", text)
    }
    return {h.removeprefix("examples/") for h in hits}


def test_every_shipped_example_is_documented():
    shipped = _shipped_files()
    documented = _readme_paths()
    undocumented = {f for f in shipped if f not in documented}
    assert not undocumented, (
        f"examples/ ships files the README does not mention: {sorted(undocumented)}"
    )


def test_every_documented_example_exists():
    shipped = _shipped_files()
    missing = {
        d
        for d in _readme_paths()
        if "/" not in d or d.startswith("configs/")
        if d not in shipped
    }
    assert not missing, (
        f"examples/README.md references files that do not exist: {sorted(missing)}"
    )
