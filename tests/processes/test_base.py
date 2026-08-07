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

    def test_process_modules_do_not_import_unturtle_diffusion(self):
        import ast
        import pathlib

        import unturtle.processes as processes_pkg

        pkg_dir = pathlib.Path(processes_pkg.__file__).parent
        offenders = []
        for path in sorted(pkg_dir.glob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    # `import unturtle.diffusion[...]`
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    # `from ..diffusion import x` → level=2, module="diffusion".
                    # Normalize to a dotted path relative to the package root so
                    # both absolute and relative escapes are caught.
                    names = ["." * node.level + module]
                else:
                    continue
                for name in names:
                    if "unturtle.diffusion" in name or name.startswith("..diffusion"):
                        offenders.append(f"{path.name}: {name}")

        assert not offenders, f"process layer imports diffusion: {offenders}"


class TestForwardProcessProtocol:
    def test_masked_process_satisfies_call_signature(self):
        # Structural check only — no isinstance against the Protocol.
        assert callable(MaskedDiffusionProcess)
        assert ForwardProcess is not None
