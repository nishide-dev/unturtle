"""Guards that the optional lm_eval dependency is not required to import eval modules."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# Run the "import without lm_eval" check in a SUBPROCESS. Blocking lm_eval and
# re-importing unturtle.eval in-process would mutate global sys.modules state and leak
# into other tests (e.g. a freshly re-imported unturtle.eval drops its `experimental`
# subpackage attribute, breaking dotted monkeypatch targets elsewhere). A subprocess
# isolates that completely and faithfully simulates lm_eval being absent.
_BOUNDARY_SCRIPT = textwrap.dedent(
    """
    import sys

    # Make any import of the optional deps fail, as if not installed.
    # lm_eval and mauve share the same boundary rule (#123): the eval
    # package must import without either.
    class _BlockOptionalDeps:
        _BLOCKED = ("lm_eval", "mauve")

        def find_spec(self, name, path=None, target=None):
            if any(
                name == blocked or name.startswith(blocked + ".")
                for blocked in self._BLOCKED
            ):
                raise ImportError(f"simulated: {name} not installed")
            return None

    sys.meta_path.insert(0, _BlockOptionalDeps())
    sys.modules.pop("lm_eval", None)
    sys.modules.pop("mauve", None)

    # These must all import WITHOUT lm_eval present.
    import unturtle.eval  # noqa: F401
    import unturtle.eval.harness  # noqa: F401
    import unturtle.eval.harness.configs  # noqa: F401
    import unturtle.eval.harness.model_adapter  # noqa: F401
    import unturtle.eval.harness.runner  # noqa: F401

    # Sanity: lm_eval really was blocked.
    try:
        import lm_eval  # noqa: F401
    except ImportError:
        pass
    else:
        raise AssertionError("lm_eval import was not blocked in the subprocess")

    print("OK")
    """
)


def test_import_eval_without_lm_eval(tmp_path) -> None:  # noqa: ANN001
    script = tmp_path / "boundary_check.py"
    script.write_text(_BOUNDARY_SCRIPT)
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"import boundary failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "OK" in proc.stdout


def test_harness_call_without_lm_eval_raises_importerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "lm_eval", None)
    from unturtle.eval.harness.model_adapter import build_harness_lm

    class _M:
        def parameters(self):
            import torch

            yield torch.zeros(1)

    with pytest.raises(ImportError):
        build_harness_lm(
            model=_M(),
            tokenizer=object(),
            num_steps=1,
            max_new_tokens=1,
            temperature=0.0,
            use_chat_template=False,
        )
