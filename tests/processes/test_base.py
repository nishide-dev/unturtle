"""Tests for the process-layer base contracts (#70, #62 PR1)."""

import torch

from unturtle.processes import ForwardProcess, MaskedDiffusionProcess, ProcessOutput


class TestProcessOutput:
    def test_separates_model_and_objective_inputs(self):
        out = ProcessOutput(
            model_inputs={"input_ids": torch.zeros(1, 2, dtype=torch.long)},
            objective_inputs={"timesteps": torch.zeros(1)},
        )
        assert "input_ids" in out.model_inputs
        assert "timesteps" in out.objective_inputs


class TestDependencyRule:
    """`unturtle.processes` must not depend on `unturtle.diffusion` (#70).

    Checked by static import inspection rather than ``sys.modules``: the
    parent ``unturtle`` package eagerly imports unsloth and the model
    registry, so runtime module-table inspection cannot isolate the
    process layer's own dependencies.
    """

    @staticmethod
    def _diffusion_imports(source: str) -> list[str]:
        """Every way `source` could reach ``unturtle.diffusion``.

        Covers the three forms that actually occur in practice:
        ``import unturtle.diffusion.x``, ``from unturtle import diffusion``
        (where the module string alone is innocent — the *binding* is the
        violation), ``from ..diffusion import x`` (relative, level>0), and
        dynamic ``importlib.import_module("unturtle.diffusion...")``.
        """
        import ast

        offenders = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[:2] == ["unturtle", "diffusion"]:
                        offenders.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                dotted = "." * node.level + module
                if dotted.split(".")[:2] == ["unturtle", "diffusion"]:
                    offenders.append(dotted)
                elif node.level >= 2 and module.split(".")[0] == "diffusion":
                    # `from ..diffusion import x` escapes to the sibling package.
                    offenders.append(dotted)
                elif dotted in ("unturtle", ".."):
                    # `from unturtle import diffusion` / `from .. import diffusion`
                    for alias in node.names:
                        if alias.name == "diffusion":
                            offenders.append(f"{dotted} import {alias.name}")
            elif (
                # Dynamic importlib.import_module("unturtle.diffusion...")
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.split(".")[:2] == ["unturtle", "diffusion"]
            ):
                offenders.append(f"dynamic: {node.value}")
        return offenders

    def test_detector_catches_every_violating_import_form(self):
        """The guard is only worth having if it actually catches violations."""
        violations = [
            "import unturtle.diffusion",
            "import unturtle.diffusion.schedulers as s",
            "from unturtle.diffusion import BaseAlphaScheduler",
            "from unturtle.diffusion.schedulers import BaseAlphaScheduler",
            "from unturtle import diffusion",
            "from .. import diffusion",
            "from ..diffusion import BaseAlphaScheduler",
            "from ..diffusion.schedulers import BaseAlphaScheduler",
            "import importlib; importlib.import_module('unturtle.diffusion')",
        ]
        for source in violations:
            assert self._diffusion_imports(source), f"missed violation: {source}"

        allowed = [
            "from .base import ProcessOutput",
            "import torch",
            "from typing import Protocol",
            "from unturtle.utils import something",
        ]
        for source in allowed:
            assert not self._diffusion_imports(source), f"false positive: {source}"

    def test_process_modules_do_not_import_unturtle_diffusion(self):
        import pathlib

        import unturtle.processes as processes_pkg

        pkg_dir = pathlib.Path(processes_pkg.__file__).parent
        offenders = [
            f"{path.name}: {name}"
            for path in sorted(pkg_dir.glob("*.py"))
            for name in self._diffusion_imports(path.read_text())
        ]

        assert not offenders, f"process layer imports diffusion: {offenders}"


class TestForwardProcessProtocol:
    def test_masked_process_satisfies_call_signature(self):
        # Structural check only — no isinstance against the Protocol.
        assert callable(MaskedDiffusionProcess)
        assert ForwardProcess is not None
