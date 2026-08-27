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

    def test_the_origin_frame_is_recorded_not_just_the_message(self):
        """The reference arm fails with the SAME message from Unsloth's MLP
        path, so message matching alone credits it to the QKV kernel."""
        source = inspect.getsource(_producer().execution_preflight)
        assert "raised_in" in source
        assert "raised_in_unturtle_qkv_kernel" in source
        assert "unturtle/kernels/fast_lora" in source


class TestUnsupportedDisposition:
    def test_no_timing_fields_are_populated_when_the_arm_cannot_execute(self):
        """Zero looks measurable; these must be absent or null."""
        source = inspect.getsource(_producer().main)
        unsupported = source.split('"status": "unsupported"')[1]
        for field in ("speed_verdict", "operation_profile", "peak_memory"):
            assert f'"{field}": None' in unsupported
        assert '"timing_attempted": False' in unsupported
        assert '"warmup_attempted": False' in unsupported

    def test_the_cell_is_ineligible_for_target_selection(self):
        source = inspect.getsource(_producer().main)
        assert '"target_selection_eligible": False' in source
        assert '"blocked_by": "#177"' in source

    def test_a_working_reference_arm_cannot_upgrade_the_cell(self):
        """A two-arm comparison needs BOTH arms under the same fixture."""
        source = inspect.getsource(_producer().main)
        assert '"diagnostic_only": True' in source
        # Status is decided by the FAST arm alone.
        assert 'if fast_check["executable"]:' in source
        assert "reference_check[" not in source.split("if fast_check")[1][:400]

    def test_timing_never_runs_after_a_failed_preflight(self):
        source = inspect.getsource(_producer().main)
        after = source.split("fast_check = execution_preflight")[1]
        for forbidden in ("run_condition(", "timer.collect", "CONDITION_ORDERS"):
            assert forbidden not in after

    def test_the_reproduction_executes_a_forward(self):
        """The existing repo test asserted installation only, which is why the
        defect survived."""
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
