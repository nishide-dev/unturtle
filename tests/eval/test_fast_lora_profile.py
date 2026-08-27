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

"""#166 row 5 — the fast-LoRA preflight, recorded as `unsupported`.

The cell produces no timing: the fast path installs on all 28 layers and then
fails its first real forward (#177). These tests pin the disposition so a future
run cannot quietly turn a non-execution into a measurement.
"""

from __future__ import annotations

import importlib.util
import inspect
import pathlib

import pytest


def _producer():
    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "kernels"
        / "fast_lora_profile.py"
    )
    spec = importlib.util.spec_from_file_location("_fast_lora_profile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _RaisingTrainer:
    """Fails the preflight forward with a message the classifier does not
    recognize, so the disposition path can be exercised without a 7B model."""

    def compute_loss(self, model, batch):
        raise RuntimeError("synthetic preflight failure for test purposes")


class _DtypeRaisingTrainer:
    """Fails with the real #177 signature."""

    def compute_loss(self, model, batch):
        raise RuntimeError(
            "expected mat1 and mat2 to have the same dtype, but got: float != "
            "c10::BFloat16"
        )


class _RaisingModel:
    def zero_grad(self, set_to_none=True):
        pass

    def get_decoder(self):
        return self

    @property
    def layers(self):
        return []


class _NoopOptimizer:
    def zero_grad(self, set_to_none=True):
        pass


def _fake_attention_holder(layer_count: int):
    """A minimal stand-in for the decoder: `attention_modules` only needs
    `get_decoder().layers[i].self_attn`."""
    import torch

    class Holder:
        def __init__(self, layers):
            self.layers = layers

        def get_decoder(self):
            return self

    layers = []
    modules = []
    for _ in range(layer_count):
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layers.append(layer)
        modules.append(layer.self_attn)
    return Holder(layers), modules


class TestFailureClassification:
    def test_the_known_dtype_failure_maps_to_a_stable_code(self):
        producer = _producer()
        error = RuntimeError(
            "expected mat1 and mat2 to have the same dtype, but got: float != "
            "c10::BFloat16"
        )
        assert producer.classify_failure(error) == producer.DTYPE_MISMATCH_REASON

    def test_an_unrelated_error_is_not_mislabelled(self):
        """A new defect must not be filed under the known limitation."""
        producer = _producer()
        for error in (
            RuntimeError("CUDA out of memory"),
            RuntimeError("shape mismatch: expected [4, 8] got [4, 9]"),
            RuntimeError("expected mat1 and mat2 to be 2-D"),
        ):
            assert producer.classify_failure(error) is None

    def test_an_error_that_merely_mentions_a_dtype_is_not_the_known_one(self):
        """Both clauses have to bind. A message can name `float` for a reason
        that has nothing to do with the fused matmul's operand dtypes."""
        producer = _producer()
        for message in (
            "CUDA out of memory. Tried to allocate 2.00 GiB (float32 buffer)",
            "value cannot be converted to type float without overflow",
            "\"round_cpu\" not implemented for 'BFloat16'",
        ):
            assert producer.classify_failure(RuntimeError(message)) is None

    def test_the_operand_phrasing_alone_is_not_enough(self):
        """The dtype vocabulary has to appear too, or any operand-shaped
        complaint would be filed under #177."""
        producer = _producer()
        assert (
            producer.classify_failure(
                RuntimeError("expected mat1 and mat2 to be 2-D tensors")
            )
            is None
        )

    def test_a_shape_error_is_not_the_dtype_limitation(self):
        """ATen's SHAPE error shares the `self and mat2` prefix with the dtype
        error and can name `float`. Matching the prefix filed a head-dim
        regression under #177, where it would be closed as known and never
        investigated."""
        producer = _producer()
        assert (
            producer.classify_failure(
                RuntimeError(
                    "self and mat2 shapes cannot be multiplied (4x8 and 9x2) "
                    "for float tensors"
                )
            )
            is None
        )

    def test_a_device_error_is_not_the_dtype_limitation(self):
        producer = _producer()
        assert (
            producer.classify_failure(
                RuntimeError(
                    "self and mat2 must have the same device, got cuda:0 and "
                    "cuda:1 (float32)"
                )
            )
            is None
        )

    def test_an_oom_is_not_the_dtype_limitation(self):
        """ "The machine was busy" must never be published as "row 5 is not
        performance-measurable in its frozen configuration"."""
        producer = _producer()
        assert (
            producer.classify_failure(
                RuntimeError(
                    "CUDA out of memory. Tried to allocate 2.00 GiB; self and "
                    "mat2 are float"
                )
            )
            is None
        )

    def test_a_dtype_phrase_alongside_an_unrelated_cause_is_refused(self):
        """Requiring "same dtype" already excludes a bare shape or OOM error, so
        the exclusion list only decides messages carrying BOTH phrasings. Those
        are the ambiguous ones, and they must not be filed under #177."""
        producer = _producer()
        for message in (
            "expected mat1 and mat2 to have the same dtype; also shapes cannot "
            "be multiplied",
            "CUDA out of memory while checking that self and mat2 have the same "
            "dtype (float)",
            "tensors must have the same dtype and the same device, got cuda:0 "
            "and cuda:1",
        ):
            assert producer.classify_failure(RuntimeError(message)) is None, message

    def test_the_plain_dtype_error_still_classifies(self):
        """The exclusions must not swallow the real thing."""
        producer = _producer()
        assert (
            producer.classify_failure(
                RuntimeError(
                    "expected mat1 and mat2 to have the same dtype, but got: "
                    "float != c10::BFloat16"
                )
            )
            == producer.DTYPE_MISMATCH_REASON
        )

    def test_a_non_runtime_error_carrying_the_same_text_is_not_the_known_one(self):
        """The dtype limitation surfaces as a RuntimeError from ATen. The same
        words in a ValueError are a different bug — the type is part of the
        signature, not decoration."""
        producer = _producer()
        message = (
            "expected mat1 and mat2 to have the same dtype, but got: float != "
            "c10::BFloat16"
        )
        assert producer.classify_failure(RuntimeError(message)) is not None
        for error in (
            ValueError(message),
            TypeError(message),
            AssertionError(message),
        ):
            assert producer.classify_failure(error) is None

    def test_the_arms_are_separated_by_an_unturtle_specific_field(self):
        """Both arms raise at the same frame, so `raised_in` is identical for
        both and cannot separate them. A reader diffing only that field would
        conclude the two failures share a code path."""
        producer = _producer()
        source = inspect.getsource(producer.execution_preflight)
        assert '"raised_in_unturtle_kernels"' in source
        assert "unturtle/kernels" in source

    def test_the_frame_label_says_which_file_not_just_the_basename(self):
        """A basename made BOTH arms read `utils.py:1099`, which named neither
        the package nor the distinction the field exists to record."""
        producer = _producer()
        record = producer.execution_preflight(
            _RaisingModel(), _RaisingTrainer(), _NoopOptimizer(), {}, "fast"
        )
        assert record["executable"] is False
        assert record["raised_in"] is not None
        # `<dir>/<file>.py:<line>` — the parent directory must be present.
        assert "/" in record["raised_in"], record["raised_in"]
        assert record["raised_in"].split(":")[0].endswith(".py")

    def test_an_unsloth_kernel_frame_is_not_counted_as_ours(self):
        """`unsloth/kernels/utils.py` contains "kernels". Matching that bare
        word is precisely what made the field unable to separate the arms."""
        producer = _producer()
        source = inspect.getsource(producer.execution_preflight)
        assert '"unturtle/kernels" in f.filename' in source
        # And the discriminator must be false for a purely-Unsloth traceback.
        record = producer.execution_preflight(
            _RaisingModel(), _RaisingTrainer(), _NoopOptimizer(), {}, "reference"
        )
        assert record["raised_in_unturtle_qkv_kernel"] is False
        assert record["raised_in_unturtle_kernels"] is None

    def test_the_origin_frame_is_recorded_not_just_the_message(self):
        """The reference arm fails with the SAME message from Unsloth's MLP
        path, so message matching alone credits it to the QKV kernel."""
        source = inspect.getsource(_producer().execution_preflight)
        assert "raised_in" in source
        assert "raised_in_unturtle_qkv_kernel" in source
        assert "unturtle/kernels/fast_lora" in source


class TestUnsupportedDisposition:
    """Asserted by CALLING `disposition`, not by grepping `main`'s source."""

    @staticmethod
    def _blocked(reason="fast_path_execution_dtype_mismatch", kind=None):
        producer = _producer()
        if kind is None:
            kind = "product_limitation" if reason else "unclassified"
        return producer.disposition(
            {
                "arm": "fast",
                "executable": False,
                "reason_code": reason,
                "failure_kind": kind,
            },
            28,
        )

    def test_no_timing_fields_are_populated_when_the_arm_cannot_execute(self):
        """A zero latency or 0% share reads as measured; these must be null."""
        record = self._blocked()
        for field in ("speed_verdict", "operation_profile", "peak_memory"):
            assert field in record
            assert record[field] is None, f"{field} must be null, not {record[field]!r}"
        assert record["timing_attempted"] is False
        assert record["warmup_attempted"] is False
        assert record["measurement_valid"] is False

    def test_the_cell_is_ineligible_for_target_selection(self):
        record = self._blocked()
        assert record["target_selection_eligible"] is False
        assert record["blocked_by"] == "#177"

    def test_an_unclassified_failure_is_not_published_as_a_product_limitation(self):
        """OOM from a co-resident job, or a trainer signature change, must not
        become "row 5 is not performance-measurable in its frozen documented
        configuration" — that is a claim about the product, not the machine."""
        record = self._blocked(reason=None)
        assert record["status"] == "measurement_invalid"
        assert record["status"] != "unsupported"
        assert record["blocked_by"] is None
        assert record["failure_kind"] == "unclassified"

    def test_the_failure_kind_is_derived_from_the_classification(self):
        """A constant `product_limitation` would make every infrastructure
        failure a claim about the product."""
        producer = _producer()
        classified = producer.execution_preflight(
            _RaisingModel(), _DtypeRaisingTrainer(), _NoopOptimizer(), {}, "fast"
        )
        unclassified = producer.execution_preflight(
            _RaisingModel(), _RaisingTrainer(), _NoopOptimizer(), {}, "fast"
        )
        assert classified["failure_kind"] == "product_limitation"
        assert unclassified["failure_kind"] == "unclassified"

    def test_a_classified_failure_is_typed_unsupported(self):
        record = self._blocked()
        assert record["status"] == "unsupported"
        assert record["failure_kind"] == "product_limitation"

    def test_an_executable_arm_does_not_produce_a_failure_record(self):
        producer = _producer()
        record = producer.disposition(
            {"arm": "fast", "executable": True, "reason_code": None}, 28
        )
        assert record["status"] == "preflight_passed"
        assert "speed_verdict" not in record

    def test_a_working_reference_arm_cannot_upgrade_the_cell(self):
        """A two-arm comparison needs BOTH arms under the same fixture, so the
        status is decided by the fast arm alone."""
        producer = _producer()
        source = inspect.getsource(producer.main)
        assert '"diagnostic_only": True' in source
        assert "disposition(fast_check, layers)" in source
        assert "reference_check" not in inspect.getsource(producer.disposition)

    def test_no_parity_result_is_claimed(self):
        """An earlier run reported 784/784 bit-identical while BOTH arms were
        silently on the reference callable, so the record must not restate that
        as fast-vs-reference parity."""
        record = self._blocked()
        assert record["parity"] is None
        assert "reference against itself" in record["parity_note"]
        assert "parity_preflight(" not in inspect.getsource(_producer().main)

    def test_the_unreachable_parity_check_carries_its_erratum(self):
        doc = inspect.getdoc(_producer().parity_preflight)
        assert "NOT REACHABLE" in doc
        assert "5f141cb" in doc

    def test_the_record_states_the_blocker_is_broader_than_this_kernel(self):
        """Reading the record as a Dream-QKV-kernel defect would scope #177
        wrongly and invite a fix that cannot unblock the cell."""
        note = self._blocked()["scope_note"]
        assert "BROADER" in note
        assert "MLP fast_lora" in note

    def test_the_record_does_not_claim_raised_in_separates_the_arms(self):
        assert "SAME `raised_in`" in self._blocked()["scope_note"]

    def test_the_declared_timing_window_is_marked_unexercised(self):
        """`frozen_constants: {TRIALS: 3, STEPS: 8}` beside
        `warmup_attempted: false` is unreconcilable unless the record says the
        window never ran."""
        import argparse

        producer = _producer()
        run = producer.provenance(argparse.Namespace(device="cuda:0", out="/tmp/x"))
        assert run["frozen_constants"]["exercised"] is False
        assert "zero times" in run["frozen_constants"]["note"]

    def test_provenance_refuses_to_publish_an_unknown_measuring_commit(self):
        """`head_sha: "unknown"` silently discards the provenance the record
        exists to carry."""
        import argparse

        producer = _producer()
        run = producer.provenance(argparse.Namespace(device="cuda:0", out="/tmp/x"))
        # A real run in a real repo: the SHA must be a SHA, never a placeholder.
        assert run["head_sha"] != "unknown"
        assert len(run["head_sha"]) == 40
        assert isinstance(run["worktree_clean"], bool)
        # And the failure path must raise rather than emit a placeholder.
        assert "refusing to write an artifact" in inspect.getsource(producer.provenance)

    def test_every_unreachable_function_says_so(self):
        """Roughly 40% of this module is the timing machinery the cell never
        reaches. Unmarked, a reader takes the design docstrings as a
        description of the artifact."""
        producer = _producer()
        for name in (
            "rng_fingerprint",
            "state_fingerprint",
            "run_condition",
            "StateSnapshot",
            "oom_phase",
        ):
            doc = inspect.getdoc(getattr(producer, name)) or ""
            assert "NOT REACHED" in doc, f"{name} does not declare it is unreachable"

    def test_the_module_docstring_describes_what_the_run_produces(self):
        producer = _producer()
        doc = producer.__doc__ or ""
        assert "NO TIMING" in doc
        assert "NOT EXERCISED" in doc

    def test_provenance_raises_when_git_is_unavailable(self):
        """Rather than publishing `head_sha: "unknown"`, which is an artifact
        that cannot be traced to code."""
        import argparse
        import subprocess

        producer = _producer()
        original = subprocess.run

        def failing(*args, **kwargs):
            raise OSError("git not found")

        subprocess.run = failing
        try:
            with pytest.raises(SystemExit, match="refusing to write an artifact"):
                producer.provenance(argparse.Namespace(device="cuda:0", out="/tmp/x"))
        finally:
            subprocess.run = original

    def test_timing_never_runs_after_a_failed_preflight(self):
        source = inspect.getsource(_producer().main)
        after = source.split("fast_check = execution_preflight")[1]
        for forbidden in ("run_condition(", "timer.collect", "CONDITION_ORDERS"):
            assert forbidden not in after

    def test_the_reproduction_executes_a_forward(self):
        """The pre-existing repo test asserted installation only, which is why
        the defect survived."""
        path = (
            pathlib.Path(__file__).resolve().parents[2]
            / "benchmarks"
            / "kernels"
            / "repro_dream_4bit_lora_dtype.py"
        )
        source = path.read_text()
        assert "attn.apply_qkv(attn, probe)" in source
        assert "get_input_embeddings" in source


class TestInstallationGates:
    def test_a_mixed_installation_is_refused(self):
        """A partial patch makes a speed difference unattributable."""
        source = inspect.getsource(_producer().assert_arm_installed)
        assert "mixed or unexpected installation" in source
        assert 'counts["other"]' in source

    def test_a_build_that_installed_no_fast_layers_is_refused(self):
        """peft's own `get_peft_model` leaves every layer on the reference
        callable, because the Dream patch keys on `hasattr(q_proj, "lora_A")`
        and runs during `from_pretrained`. Both arms then ran the reference and
        the cell would have reported ~0% as a kernel finding."""
        producer = _producer()
        producer.require_fast_baseline({"fast": 28, "reference": 0, "other": 0}, 28)
        for counts in (
            {"fast": 0, "reference": 28, "other": 0},
            {"fast": 27, "reference": 1, "other": 0},
            {"fast": 28, "reference": 0, "other": 1},
        ):
            with pytest.raises(SystemExit, match="fast QKV path was not installed"):
                producer.require_fast_baseline(counts, 28)

    def test_a_reference_layer_alone_is_refused(self):
        """Redundant with the `other` check while the counts sum to the layer
        total, but a partial install must be refused on its own terms rather
        than by arithmetic that a future counting change could void."""
        producer = _producer()
        with pytest.raises(SystemExit, match="was not installed"):
            producer.require_fast_baseline({"fast": 28, "reference": 4, "other": 0}, 28)

    def test_a_model_with_no_attention_layers_is_refused(self):
        """`fast == expected_layers` holds vacuously at zero: a build that found
        no layers to patch must not read as fully patched."""
        producer = _producer()
        with pytest.raises(SystemExit, match="was not installed"):
            producer.require_fast_baseline({"fast": 0, "reference": 0, "other": 0}, 0)

    def test_the_supported_adapter_entry_point_is_used(self):
        source = inspect.getsource(_producer().build)
        assert "FastDiffusionModel.get_peft_model" in source
        assert "require_fast_baseline(" in source

    def test_the_fingerprint_unwraps_instrumentation(self):
        source = inspect.getsource(_producer().callable_identities)
        assert "inspect.unwrap" in source
        assert "__func__" in source

    def test_a_wraps_wrapper_is_not_treated_as_a_change(self):
        """`functools.wraps` copies `__qualname__` but the wrapper is a
        different object, so an id-based fingerprint moves. The instrumentation
        wrapper and the identity gate would then read each other as changes."""
        import functools

        import torch

        producer = _producer()

        def original(self, X):
            return X

        @functools.wraps(original)
        def instrumented(self, X):
            return original(self, X)

        class Attn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.o_proj = torch.nn.Linear(2, 2)

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Attn()

        class Holder:
            def __init__(self, layers):
                self.layers = layers

            def get_decoder(self):
                return self

        layer = Layer()
        holder = Holder([layer])
        layer.self_attn.apply_qkv = original
        before = producer.callable_identities(holder)["attn0.apply_qkv"]
        layer.self_attn.apply_qkv = instrumented
        after = producer.callable_identities(holder)["attn0.apply_qkv"]
        assert before == after, "instrumentation must not read as a swap"

    def test_a_wrapper_around_a_bound_method_is_unwrapped_to_the_function(self):
        """`inspect.unwrap` of a wraps-wrapper over a bound method yields the
        BOUND METHOD, not the function, so a second unbind is required —
        otherwise per-access bound-method identity leaks back in."""
        import functools

        import torch

        producer = _producer()

        class Attn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.o_proj = torch.nn.Linear(2, 2)

            def apply_qkv(self, X):
                return X

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Attn()

        class Holder:
            def __init__(self, layers):
                self.layers = layers

            def get_decoder(self):
                return self

        layer = Layer()
        holder = Holder([layer])
        bound = layer.self_attn.apply_qkv
        before = producer.callable_identities(holder)["attn0.apply_qkv"]
        layer.self_attn.apply_qkv = functools.wraps(bound)(lambda X: bound(X))
        after = producer.callable_identities(holder)["attn0.apply_qkv"]
        assert before == after

    def test_bound_method_access_is_not_treated_as_a_change(self):
        """`id(module.forward)` differs per access; the fingerprint must not."""
        import torch

        producer = _producer()

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = torch.nn.Module()
                self.self_attn.o_proj = torch.nn.Linear(2, 2)

        class Holder:
            def __init__(self, layer):
                self.layers = [layer]

            def get_decoder(self):
                return self

        layer = Tiny()
        holder = Holder(layer)
        first = producer.callable_identities(holder)
        second = producer.callable_identities(holder)
        assert first == second, "a no-op must not read as a callable change"


class TestArmSwap:
    def test_each_arm_installs_its_own_callable(self):
        """A swap that ignores `arm` makes both arms the same implementation,
        so any measured difference would be noise reported as a speedup."""
        producer = _producer()
        from unturtle.fast_diffusion_model import _original_apply_qkv
        from unturtle.kernels.fast_lora import apply_lora_qkv_with_bias

        holder, modules = _fake_attention_holder(3)
        installed = {}
        for arm, expected in (
            ("fast", apply_lora_qkv_with_bias),
            ("reference", _original_apply_qkv),
        ):
            with producer.qkv_arm(holder, arm) as count:
                assert count == 3
                installed[arm] = {m.apply_qkv for m in modules}
                assert installed[arm] == {expected}
        assert installed["fast"] != installed["reference"]

    def test_the_previous_callable_is_restored_even_on_failure(self):
        """A leaked patch would instrument the NEXT arm, which is supposed to be
        the other implementation."""
        producer = _producer()
        holder, modules = _fake_attention_holder(3)
        sentinel = object()
        for module in modules:
            module.apply_qkv = sentinel

        with pytest.raises(ZeroDivisionError), producer.qkv_arm(holder, "fast"):
            assert all(m.apply_qkv is not sentinel for m in modules)
            raise ZeroDivisionError
        assert all(m.apply_qkv is sentinel for m in modules)

    def test_a_module_without_the_attribute_ends_without_it(self):
        producer = _producer()
        holder, modules = _fake_attention_holder(2)
        with producer.qkv_arm(holder, "fast"):
            pass
        assert not any(hasattr(m, "apply_qkv") for m in modules)


class TestSingleDeviceGate:
    def test_non_default_cuda_is_refused(self):
        producer = _producer()
        producer.require_supported_device("cuda:0")
        for device in ("cpu", "cuda:1", "cuda"):
            with pytest.raises(SystemExit, match="cuda:0 only"):
                producer.require_supported_device(device)

    def test_a_sharded_model_is_refused(self):
        """`device_map="auto"` spread this 7B checkpoint over cuda:1/2/3 with
        nothing on cuda:0, so every timing assumption was already void."""
        producer = _producer()
        producer.require_single_device({"cuda:0"}, "cuda:0")
        for shards in ({"cuda:1", "cuda:2"}, {"cuda:0", "cuda:1"}, {"cpu"}, set()):
            with pytest.raises(SystemExit, match="spread over"):
                producer.require_single_device(shards, "cuda:0")

    def test_the_loader_pins_every_shard_to_the_requested_device(self):
        source = inspect.getsource(_producer().build)
        assert 'device_map={"": args.device}' in source
        assert "require_single_device(" in source
