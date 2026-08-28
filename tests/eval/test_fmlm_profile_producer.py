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

"""#166 Stage 1 — the FMLM profile producer's measurement discipline.

The producer itself needs a GPU and the real checkpoint; these tests pin the
discipline that decides whether its numbers mean anything: the verdict basis,
the share denominator, per-trial gating, the structural zero, typed OOM, and the
separation of diagnostics from timing.
"""

from __future__ import annotations

import importlib.util
import inspect
import pathlib

import pytest

pytest.importorskip("unturtle_flm", reason="FLM pack not installed")

import torch  # noqa: E402


def _estimate(producer, off_walls, on_walls):
    """Build the paired-trial shape `overhead_estimate` now consumes."""
    return producer.overhead_estimate(
        [
            {
                "off_wall_seconds": off,
                "on_wall_seconds": on,
                "paired_overhead_seconds": on - off,
            }
            for off, on in zip(off_walls, on_walls, strict=True)
        ]
    )


def _code_only(source: str) -> str:
    """Source with comments and string literals removed.

    Asserting "`.item()` does not appear" against raw text fails on the very
    docstring that forbids it, so the checks below run against executable
    tokens only.
    """
    import io
    import tokenize

    kept = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        kept.append(token.string)
    return " ".join(kept)


def _producer():
    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "flm"
        / "fmlm_profile.py"
    )
    spec = importlib.util.spec_from_file_location("_fmlm_profile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestFrozenConfiguration:
    def test_the_measurement_window_is_not_cli_settable(self):
        """A verdict that moves with a command-line flag is not a verdict."""
        producer = _producer()
        source = inspect.getsource(producer.parse_args)
        for forbidden in ("--trials", "--warmup", "--steps", "--gamma", "--seed"):
            assert forbidden not in source, forbidden
        assert isinstance(producer.TRIALS, int)
        assert isinstance(producer.WARMUP, int)

    def test_the_official_cell_configuration_is_pinned(self):
        producer = _producer()
        assert producer.GAMMA == 1.0
        assert producer.SEED == 100
        assert producer.MAX_LENGTH == 1024
        assert producer.STEPS_CELLS == (1, 32)
        assert producer.BATCH_SIZES == (1, 8, 32)

    def test_the_window_values_are_pinned_not_merely_typed(self):
        """`TRIALS = 1` passes an isinstance check while making every median a
        single sample and the residual arithmetic untestable, with provenance
        faithfully reporting the degraded value."""
        producer = _producer()
        assert producer.TRIALS == 3
        assert producer.WARMUP == 2
        assert producer.SHARE_TOLERANCE == 0.02

    def test_the_device_gate_requires_a_single_cuda_device(self):
        """Any single CUDA device is allowed, but it must be MADE CURRENT: the
        peak-stat calls default to the current device, so measuring cuda:N while
        device 0 is current would record device 0's numbers."""
        import torch

        producer = _producer()
        producer.require_supported_device("cuda:0")
        assert torch.cuda.current_device() == 0
        for device in ("cpu", "cuda", "cuda:x", ""):
            with pytest.raises(SystemExit, match="not supported"):
                producer.require_supported_device(device)

    def test_a_nonexistent_device_index_is_refused(self):
        producer = _producer()
        with pytest.raises(SystemExit, match="does not exist"):
            producer.require_supported_device("cuda:99")

    def test_the_device_is_made_current_so_peak_stats_are_right(self):
        producer = _producer()
        source = inspect.getsource(producer.require_supported_device)
        assert "torch.cuda.set_device(index)" in source

    def test_the_gpu_name_comes_from_the_device_under_test(self):
        """`get_device_name(0)` was hardcoded, so a run on another device would
        have published device 0's name."""
        producer = _producer()
        source = inspect.getsource(producer.environment)
        assert "get_device_name(torch.cuda.current_device())" in source
        assert "get_device_name(0)" not in source
        assert '"device_index"' in source

    def test_the_diagnostic_flags_stay_out_of_the_public_kwargs(self):
        """`request.kwargs` is the documented surface; a diagnostic flag there
        would change the public contract."""
        producer = _producer()
        plain = producer.Request(steps=1, num_samples=1)
        flagged = producer.Request(
            steps=1, num_samples=1, diagnostics=("terminal_rng", "final_latent")
        )
        expected = {"steps", "num_samples", "seed", "gamma"}
        assert set(plain.kwargs) == expected
        assert set(flagged.kwargs) == expected
        assert not hasattr(plain, "_unturtle_profile_diagnostics")
        assert flagged._unturtle_profile_diagnostics == frozenset(
            {"terminal_rng", "final_latent"}
        )

    def test_only_device_and_out_are_configurable(self):
        producer = _producer()
        source = inspect.getsource(producer.parse_args)
        assert source.count("add_argument") == 2


class TestExpectedCounts:
    @pytest.mark.parametrize(
        ("steps", "expected"),
        [
            (
                1,
                {
                    "grid_init": 1,
                    "time_schedule": 1,
                    "flow_map_forward": 1,
                    "state_update": 0,
                    "endpoint_decode": 1,
                },
            ),
            (
                32,
                {
                    "grid_init": 1,
                    "time_schedule": 32,
                    "flow_map_forward": 32,
                    "state_update": 31,
                    "endpoint_decode": 1,
                },
            ),
        ],
    )
    def test_the_frozen_per_trial_counts(self, steps, expected):
        assert _producer().expected_counts(steps) == expected

    @pytest.mark.parametrize(
        ("steps", "expected"),
        [(1, {"randn": 1, "randn_like": 0}), (32, {"randn": 1, "randn_like": 31})],
    )
    def test_the_frozen_random_call_counts(self, steps, expected):
        """gamma=1.0, so every non-final step draws churn noise."""
        assert _producer().expected_random_calls(steps) == expected

    def test_the_expected_order_ends_without_a_state_update(self):
        order = _producer().expected_event_order(32)
        assert order[0] == "grid_init"
        assert order[-1] == "endpoint_decode"
        assert order[-3:] == ["time_schedule", "flow_map_forward", "endpoint_decode"]
        assert order.count("state_update") == 31

    def test_the_one_step_order_has_no_state_update(self):
        assert _producer().expected_event_order(1) == [
            "grid_init",
            "time_schedule",
            "flow_map_forward",
            "endpoint_decode",
        ]


class TestPerTrialGate:
    def test_a_clean_trial_passes(self):
        producer = _producer()
        assert producer.gate_trial(32, producer.expected_counts(32)) == []

    def test_a_single_missing_call_fails(self):
        producer = _producer()
        calls = producer.expected_counts(32) | {"flow_map_forward": 31}
        assert producer.gate_trial(32, calls)

    def test_an_extra_state_update_at_one_step_fails(self):
        """The final step must not emit one."""
        producer = _producer()
        calls = producer.expected_counts(1) | {"state_update": 1}
        assert producer.gate_trial(1, calls)

    def test_an_unknown_event_fails(self):
        producer = _producer()
        calls = producer.expected_counts(1) | {"latent_update": 1}
        problems = producer.gate_trial(1, calls)
        assert any("frozen taxonomy" in p for p in problems)

    def test_gate_trials_checks_every_trial(self):
        """Called, not grepped: a loop truncated to zero iterations passes any
        source-text assertion."""
        producer = _producer()
        good = {"calls": producer.expected_counts(32)}
        bad = {"calls": producer.expected_counts(32) | {"flow_map_forward": 31}}
        assert producer.gate_trials(32, [good, good, good]) == []
        # A bad trial in ANY position must be caught.
        for position in range(3):
            trials = [good, good, good]
            trials[position] = bad
            problems = producer.gate_trials(32, trials)
            assert problems, f"a bad trial at index {position} passed"
            assert f"on_trial[{position}]" in problems[0]

    def test_gate_trials_refuses_an_empty_trial_list(self):
        """Zero trials vacuously satisfies "every trial is clean"."""
        producer = _producer()
        problems = producer.gate_trials(32, [])
        assert problems
        assert "no on-trials" in problems[0]

    def test_an_aggregate_match_does_not_excuse_a_bad_trial(self):
        """Two trials can sum to the right total while both are wrong."""
        producer = _producer()
        low = producer.expected_counts(32) | {"time_schedule": 31}
        high = producer.expected_counts(32) | {"time_schedule": 33}
        assert producer.gate_trial(32, low)
        assert producer.gate_trial(32, high)


class TestStructuralZero:
    @staticmethod
    def _trials(steps, seconds):
        producer = _producer()
        # The walls VARY across trials, and not by 1.0. Three identical walls
        # make median == min == max == mean, so a denominator swapped to `min`
        # or `max` is unobservable; a unit wall makes a DROPPED denominator
        # unobservable. Median stays 4.0.
        walls = (3.0, 4.0, 5.0)
        return producer, [
            {
                "trial": index,
                "order": ["off", "on"] if index % 2 == 0 else ["on", "off"],
                "off_wall_seconds": wall,
                "on_wall_seconds": wall,
                "paired_overhead_seconds": 0.0,
                "peak_allocated_bytes": 100,
                "peak_reserved_bytes": 200,
                "event_seconds": dict(seconds),
                "calls": producer.expected_counts(steps),
            }
            for index, wall in enumerate(walls)
        ]

    def test_state_update_is_a_flagged_zero_at_one_step(self):
        producer, trials = self._trials(
            1,
            {
                "grid_init": 0.1,
                "time_schedule": 0.05,
                "flow_map_forward": 0.5,
                "endpoint_decode": 0.02,
            },
        )
        rows = {row["name"]: row for row in producer.assemble_events(1, trials)}
        row = rows["state_update"]
        assert row["calls"] == 0
        assert row["seconds"] == 0.0
        assert row["structural_zero"] is True
        assert "final-step branch exits" in row["reason"]

    def test_a_structural_zero_carries_no_share(self):
        """A 0% share would read as measured-and-negligible."""
        producer, trials = self._trials(
            1,
            {
                "grid_init": 0.1,
                "time_schedule": 0.05,
                "flow_map_forward": 0.5,
                "endpoint_decode": 0.02,
            },
        )
        rows = {row["name"]: row for row in producer.assemble_events(1, trials)}
        assert "share_of_on_wall" not in rows["state_update"]
        assert rows["flow_map_forward"]["share_of_on_wall"] is not None

    def test_state_update_is_measured_at_thirty_two_steps(self):
        producer, trials = self._trials(
            32,
            {
                "grid_init": 0.1,
                "time_schedule": 0.05,
                "flow_map_forward": 0.5,
                "state_update": 0.3,
                "endpoint_decode": 0.02,
            },
        )
        rows = {row["name"]: row for row in producer.assemble_events(32, trials)}
        assert "structural_zero" not in rows["state_update"]
        assert rows["state_update"]["seconds"] == 0.3
        assert rows["state_update"]["calls"] == 31
        # Median of per-trial shares: 0.3/3, 0.3/4, 0.3/5 -> median 0.3/4.
        assert rows["state_update"]["share_of_on_wall"] == pytest.approx(0.075)

    def test_the_denominator_is_a_central_tendency_not_a_best_case(self):
        """A denominator swapped to `min(on_walls)` inflates every published
        share by the ratio of median to fastest trial."""
        producer, trials = self._trials(
            32,
            {
                "grid_init": 0.1,
                "time_schedule": 0.05,
                "flow_map_forward": 0.5,
                "state_update": 0.3,
                "endpoint_decode": 0.02,
            },
        )
        walls = [trial["on_wall_seconds"] for trial in trials]
        assert min(walls) != max(walls), "the fixture must vary the wall"
        rows = {row["name"]: row for row in producer.assemble_events(32, trials)}
        # 0.5/4 = 0.125 for the median wall; 0.5/3 = 0.167 for the fastest.
        assert rows["flow_map_forward"]["share_of_on_wall"] == pytest.approx(0.125)

    def test_shares_are_per_trial_medians_not_a_ratio_of_medians(self):
        """Per-event medians can come from DIFFERENT trials, so dividing a
        median event time by a median wall let the shares sum over 100% — it did,
        in 3 of 5 published cells."""
        producer = _producer()
        # Event maxima deliberately land in different trials.
        trials = [
            {
                "on_wall_seconds": 1.0,
                "off_wall_seconds": 1.0,
                "event_seconds": {
                    "grid_init": 0.5,
                    "time_schedule": 0.1,
                    "flow_map_forward": 0.1,
                    "state_update": 0.1,
                    "endpoint_decode": 0.1,
                },
                "calls": producer.expected_counts(32),
            },
            {
                "on_wall_seconds": 1.0,
                "off_wall_seconds": 1.0,
                "event_seconds": {
                    "grid_init": 0.1,
                    "time_schedule": 0.5,
                    "flow_map_forward": 0.1,
                    "state_update": 0.1,
                    "endpoint_decode": 0.1,
                },
                "calls": producer.expected_counts(32),
            },
            {
                "on_wall_seconds": 1.0,
                "off_wall_seconds": 1.0,
                "event_seconds": {
                    "grid_init": 0.1,
                    "time_schedule": 0.1,
                    "flow_map_forward": 0.5,
                    "state_update": 0.1,
                    "endpoint_decode": 0.1,
                },
                "calls": producer.expected_counts(32),
            },
        ]
        rows = {row["name"]: row for row in producer.assemble_events(32, trials)}
        # Each event's per-trial shares are [0.5, 0.1, 0.1] in some order, so
        # every median is 0.1 — the sum (0.5) is BELOW each trial's true 0.9
        # coverage. That is why the summed per-event shares are not a coverage
        # figure and the artifact says so; coverage is gated per trial instead.
        for name in ("grid_init", "time_schedule", "flow_map_forward"):
            assert rows[name]["share_of_on_wall"] == pytest.approx(0.1), name
        by_trial = [
            sum(trial["event_seconds"].values()) / trial["on_wall_seconds"]
            for trial in trials
        ]
        assert all(ratio == pytest.approx(0.9) for ratio in by_trial)

    def test_the_artifact_warns_that_shares_do_not_sum_to_coverage(self):
        """A reader adding up the per-event shares gets a number that is not
        coverage — each event's median can come from a different trial — so the
        record must say so next to the figure."""
        producer = _producer()
        source = inspect.getsource(producer.profile_cell)
        note = source.split('"coverage_note": (')[1].split("),")[0]
        for phrase in ("per-trial medians", "do NOT sum", "different trial"):
            assert phrase in note, phrase
        assert '"coverage_basis"' in source
        assert '"coverage_ratio_trials": coverage_per_trial' in source

    def test_the_over_coverage_gate_is_a_live_branch(self):
        """Disabling the comparison leaves every string intact, so the branch is
        checked structurally: the guard must compare against the tolerance and
        return `profile_invalid`."""
        producer = _producer()
        source = inspect.getsource(producer.profile_cell)
        assert "if coverage_per_trial[index] > 1.0 + SHARE_TOLERANCE:" in source
        guard = source.split("if trial_problems:")[1]
        head = guard[: guard.index("problems=")]
        assert 'status="profile_invalid"' in head
        assert 'reason_code="per_trial_coverage_or_residual_invalid"' in head
        assert "cleanup()" in head

    def test_over_coverage_and_negative_residual_are_separate_gates(self):
        """The residual comes from summed seconds, so it can look healthy while
        the per-event coverage still exceeds the wall."""
        producer = _producer()
        source = inspect.getsource(producer.profile_cell)
        # Coverage, residual, wall positivity and finiteness are each checked
        # per trial in the same block; the resolvable-negative-overhead gate is
        # separate.
        assert "coverage_per_trial[index] > 1.0 + SHARE_TOLERANCE" in source
        assert "residuals[index] < 0" in source
        assert "resolvable_negative_overhead" in source
        assert source.count('status="profile_invalid"') == 2

    def test_coverage_is_gated_per_trial(self):
        """The gate must not use the sum of per-event medians, which is not a
        coverage figure."""
        producer = _producer()
        source = inspect.getsource(producer.profile_cell)
        assert "coverage_per_trial" in source
        assert "per_trial_coverage_or_residual_invalid" in source
        assert (
            'sum(trial["event_seconds"].values()) / trial["on_wall_seconds"]' in source
        )
        # EVERY trial, not just the median.
        assert "for index, trial in enumerate(trials):" in source

    def test_no_artificial_event_is_emitted_for_the_structural_zero(self):
        """The row is supplied at ASSEMBLY time; the sampler must not be made to
        fire a boundary that does not exist."""
        producer = _producer()
        sampler_source = pathlib.Path(
            __import__("unturtle_flm.sampler", fromlist=["x"]).__file__
        ).read_text()
        # Exactly one state_update scope, inside the loop after the final-step
        # break.
        assert sampler_source.count('_scope(observer, "state_update")') == 1
        assert "structural_zero" not in sampler_source
        assert "structural_zero" in inspect.getsource(producer.assemble_events)


class TestVerdictBasis:
    def test_the_verdict_is_the_instrumentation_off_wall(self):
        producer = _producer()
        source = inspect.getsource(producer.profile_cell)
        assert '"verdict_basis": "instrumentation_off_outer_wall_clock"' in source
        assert '"verdict_seconds": statistics.median(off_walls)' in source

    def test_the_off_pass_carries_no_instrumentation(self):
        producer = _producer()
        code = _code_only(inspect.getsource(producer._run_off_trial))
        for forbidden in ("Observer", "observer", "stable_hash", "get_rng_state"):
            assert forbidden not in code, forbidden

    def test_shares_use_the_on_pass_wall_not_the_off_wall(self):
        """Mixing the OFF wall into a share denominator would credit the
        instrumentation overhead to the operations."""
        producer = _producer()
        source = inspect.getsource(producer.assemble_events)
        # The divisor is the trial's OWN wall; no OFF-pass value is reachable
        # here at all, since `assemble_events` only receives ON trials.
        assert 'trial["on_wall_seconds"]' in source
        assert "off" not in _code_only(source)
        cell = inspect.getsource(producer.profile_cell)
        assert '"denominator": "per_trial_on_wall_seconds"' in cell
        assert '"aggregation": "median_of_per_trial_ratios"' in cell

    def test_peak_memory_comes_from_the_off_pass(self):
        producer = _producer()
        source = inspect.getsource(producer.profile_cell)
        assert '"basis": "instrumentation_off_trials"' in source
        # Per-trial arrays plus the aggregate max, not one number for all trials.
        assert '"allocated_bytes_trials": allocated' in source
        assert '"max_allocated_bytes": max(allocated)' in source
        assert "reset_peak_memory_stats()" in inspect.getsource(producer._run_off_trial)

    def test_the_overhead_is_recorded_with_its_noise_floor(self):
        """A signed point estimate is not publishable at TRIALS=3: the OFF
        trials alone spread by up to 16% while the difference is -1.4% to
        +10.1%, so the SIGN is an artifact of which trial was the median."""
        producer = _producer()
        assert "instrumentation_overhead" in inspect.getsource(producer.profile_cell)

    def test_an_overhead_inside_the_noise_is_not_resolvable(self):
        producer = _producer()
        # OFF spreads by 20 ms; the difference is 5 ms.
        estimate = _estimate(producer, [1.00, 1.01, 1.02], [1.015, 1.02, 1.025])
        assert estimate["resolvable"] is False
        assert estimate["off_trial_spread"] == pytest.approx(0.02)

    def test_an_overhead_larger_than_the_noise_is_resolvable(self):
        producer = _producer()
        estimate = _estimate(producer, [1.00, 1.001, 1.002], [1.50, 1.51, 1.52])
        assert estimate["resolvable"] is True
        assert estimate["median_paired_delta"] > 0

    def test_a_negative_difference_inside_the_noise_is_not_a_speedup(self):
        """Instrumentation cannot make the code faster; a negative difference
        inside the spread must not be published as one."""
        producer = _producer()
        estimate = _estimate(producer, [1.00, 1.05, 1.10], [1.02, 1.03, 1.04])
        assert estimate["median_paired_delta"] < 0
        assert estimate["resolvable"] is False

    def test_a_large_negative_difference_is_still_flagged_resolvable(self):
        """The magnitude decides resolvability, not the direction: an
        unsigned comparison is required, since a big negative difference means
        the measurement is broken and must not be silently unresolvable."""
        producer = _producer()
        estimate = _estimate(producer, [2.00, 2.001, 2.002], [1.00, 1.01, 1.02])
        assert estimate["median_paired_delta"] < 0
        assert abs(estimate["median_paired_delta"]) > estimate["off_trial_spread"]
        assert estimate["resolvable"] is True

    def test_the_overhead_is_a_structured_estimate_not_a_bare_number(self):
        """A bare `instrumentation_overhead_seconds` float is what published the
        unresolvable signs in the first place."""
        producer = _producer()
        source = inspect.getsource(producer.profile_cell)
        assert '"instrumentation_overhead": overhead' in source
        assert '"instrumentation_overhead_seconds"' not in source
        estimate = _estimate(producer, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
        for key in (
            "median_paired_delta",
            "paired_delta_trials",
            "off_trial_spread",
            "resolvable",
            "basis",
        ):
            assert key in estimate, key

    def test_the_overhead_basis_explains_the_sign_caveat(self):
        doc = _estimate(_producer(), [1.0], [1.0])["basis"]
        assert "spread" in doc
        assert "sign carries no information" in doc


class TestDiagnosticsAreSeparateFromTiming:
    @staticmethod
    def _observer_source():
        """`inspect.getsource` cannot reach a class from a module loaded via
        `exec_module`, so the class body is sliced out of the file text."""
        producer = _producer()
        whole = pathlib.Path(producer.__file__).read_text()
        start = whole.index("class CudaEventObserver")
        return whole[start : whole.index("\ndef ", start)]

    def test_the_timed_observer_only_records_cuda_events(self):
        """A hash, `.item()`, `.cpu()` or a clone inside the window would change
        what is being measured."""
        source = _code_only(self._observer_source())
        for forbidden in (
            "stable_hash",
            "get_rng_state",
            ".item()",
            ".cpu()",
            ".clone()",
            "sha256",
        ):
            assert forbidden not in source, forbidden

    def test_the_observer_never_synchronizes(self):
        """`collect` must not sync: the caller performs the ONE window-closing
        synchronize INSIDE its timed span, so the wall contains the queue
        drain. A sync inside `collect` lands outside the span."""
        assert "synchronize" not in _code_only(self._observer_source())

    def test_the_on_wall_includes_the_queue_drain(self):
        """Generation is async, so `run_once` returns with kernels in flight.
        Reading the wall before the drain made the ON wall SHORTER than the
        CUDA-event total it contains: negative instrumentation overhead in 3 of
        5 cells and event shares over 100%."""
        producer = _producer()
        source = inspect.getsource(producer._run_on_trial)
        drain = source.split("run_once(model, request, observer)")[1]
        sync_index = drain.index("torch.cuda.synchronize()")
        wall_index = drain.index("wall = time.perf_counter() - begin")
        assert sync_index < wall_index, (
            "the wall is read before the queue drains, so it excludes work the "
            "events measure"
        )

    def test_the_observer_does_not_block_per_event(self):
        """Counting occurrences cannot see RELOCATION, so the counter is driven
        through a fake device module instead."""
        producer = _producer()

        class FakeEvent:
            def record(self):
                pass

            def elapsed_time(self, _other):
                return 1.0

        calls = {"sync": 0}

        class FakeCuda:
            @staticmethod
            def Event(enable_timing=False):
                return FakeEvent()

            @staticmethod
            def synchronize():
                calls["sync"] += 1

        class FakeTorch:
            cuda = FakeCuda

        observer = producer.CudaEventObserver(FakeTorch)
        for name in ("grid_init", "time_schedule", "flow_map_forward"):
            observer(name, "enter")
            observer(name, "exit")
        assert calls["sync"] == 0, "the observer synchronized while recording"
        observer.collect()
        assert calls["sync"] == 0, "`collect` synchronized; the caller must"

    def test_an_unclosed_window_is_refused(self):
        producer = _producer()
        observer = producer.CudaEventObserver(torch)
        observer._open.append(("grid_init", None, None))
        with pytest.raises(RuntimeError, match="never closed"):
            observer.collect()

    def test_the_timed_passes_request_no_diagnostics(self):
        producer = _producer()
        for function in (
            producer._run_off_trial,
            producer._run_on_trial,
            producer.paired_trials,
            producer.warmup_arms,
        ):
            code = _code_only(inspect.getsource(function))
            assert "diagnostics" not in code, function.__name__

    def test_only_the_preflight_captures_hashes(self):
        producer = _producer()
        source = inspect.getsource(producer.non_interference_preflight)
        assert "terminal_rng" in source
        assert "final_latent" in source
        assert "stable_hash" in source

    def test_the_random_call_spy_is_restored_immediately(self):
        """The patch is process-global, so it must not outlive one request."""
        producer = _producer()
        source = inspect.getsource(producer.random_call_preflight)
        assert "finally:" in source
        assert "torch.randn, torch.randn_like = original_randn, original_like" in source
        assert source.count("run_once") == 1

    def test_the_spy_never_runs_during_timing(self):
        producer = _producer()
        for function in (
            producer._run_off_trial,
            producer._run_on_trial,
            producer.paired_trials,
        ):
            assert "randn" not in _code_only(inspect.getsource(function))


class TestExecutionPath:
    def test_both_arms_use_the_public_entry_point(self):
        producer = _producer()
        source = inspect.getsource(producer.run_once)
        assert "sampler.run_fmlm_request(model, request)" in source
        assert source.count("run_fmlm_request") == 1

    def test_no_private_loop_shortcut_exists(self):
        """The producer must never re-implement the sampling loop: the ON arm
        has to differ from OFF by the observer alone."""
        producer = _producer()
        code = _code_only(pathlib.Path(producer.__file__).read_text())
        for forbidden in (
            "_scope",
            "_FMLM_EVENTS",
            "tau_vals",
            "argmax",
            "linspace",
            "_tau_to_t",
        ):
            assert forbidden not in code, forbidden

    def test_every_run_goes_through_run_once(self):
        """Exactly one CALL site of the sampler exists, and it is in
        `run_once`; other mentions are prose."""
        producer = _producer()
        whole = pathlib.Path(producer.__file__).read_text()
        assert whole.count("sampler.run_fmlm_request(") == 1
        assert "sampler.run_fmlm_request(" in inspect.getsource(producer.run_once)

    def test_the_observer_is_none_before_and_after_each_run(self):
        producer = _producer()
        source = inspect.getsource(producer.run_once)
        assert source.count("_OBSERVER_CONTEXT.get() is None") == 2
        assert "finally:" in source

    def test_the_seam_is_restored_even_when_the_run_raises(self):
        from unturtle_flm import sampler

        producer = _producer()

        class Boom:
            kwargs = {"steps": 1, "num_samples": 1, "seed": 100, "gamma": 1.0}

        # `object()` is not a flow map, so the sampler's own guard raises.
        with pytest.raises(ValueError, match="not a pack-loaded FMLM flow map"):
            producer.run_once(object(), Boom(), lambda _n, _p: None)
        assert sampler._OBSERVER_CONTEXT.get() is None

    def test_the_producer_does_not_parallelise_cells(self):
        """Sequential by construction — asserted structurally rather than by
        counting process threads, which would also forbid CUDA's own."""
        producer = _producer()
        whole = pathlib.Path(producer.__file__).read_text()
        for forbidden in (
            "ThreadPool",
            "ProcessPool",
            "concurrent.futures",
            "multiprocessing",
            "asyncio",
            "threading",
        ):
            assert forbidden not in whole, forbidden

    def test_the_cell_order_is_deterministic(self):
        producer = _producer()
        source = inspect.getsource(producer.main)
        assert "for steps in STEPS_CELLS:" in source
        assert "for batch in BATCH_SIZES:" in source


class TestTypedFailures:
    def test_an_oom_is_classified(self):
        producer = _producer()
        assert (
            producer.classify_failure(torch.cuda.OutOfMemoryError("CUDA out of memory"))
            == "cuda_out_of_memory"
        )
        assert (
            producer.classify_failure(RuntimeError("CUDA out of memory. Tried..."))
            == "cuda_out_of_memory"
        )

    def test_a_shape_error_is_not_an_oom(self):
        """The #166 row-5 lesson: a different defect must not be filed as a
        capacity limit."""
        producer = _producer()
        for error in (
            RuntimeError("shape mismatch: expected [4, 8] got [4, 9]"),
            RuntimeError("expected mat1 and mat2 to have the same dtype"),
            ValueError("out of memory"),
        ):
            assert producer.classify_failure(error) is None

    def test_one_oom_does_not_abort_the_producer(self):
        """The latent is [B, 1024, V], so a large batch is genuinely heavy; one
        capacity limit must not cost the other cells."""
        producer = _producer()
        source = inspect.getsource(producer.main)
        assert "cells.append(cell)" in source
        # No bare raise/exit inside the cell loop.
        loop = source.split("for steps in STEPS_CELLS:")[1]
        assert "raise" not in loop
        assert "SystemExit" not in loop

    def test_a_failed_cell_emits_nulls_not_zeros(self):
        """Asserted by CALLING the builder: a 0.0 latency or an empty event list
        reads as "measured, nothing there"."""
        producer = _producer()
        record = producer.failure_record(
            stage="off_trial", reason_code="cuda_out_of_memory", timing_attempted=True
        )
        for field in ("latency", "events", "peak_memory", "attribution"):
            assert field in record
            assert record[field] is None, f"{field} = {record[field]!r}"

    def test_a_classified_capacity_failure_is_typed_oom(self):
        record = _producer().failure_record(
            stage="off_trial", reason_code="cuda_out_of_memory", timing_attempted=True
        )
        assert record["status"] == "oom"

    def test_an_unclassified_failure_is_not_typed_oom(self):
        """`reason_code=None` means nobody diagnosed it; calling that a capacity
        limit would close a real defect as a known one."""
        record = _producer().failure_record(
            stage="off_trial", reason_code=None, timing_attempted=True
        )
        assert record["status"] == "measurement_invalid"

    def test_a_negative_residual_is_typed_profile_invalid(self):
        """Over-coverage means the spans overlap or the clock is wrong. Never
        clamped to zero — a clamped residual hides a broken measurement."""
        producer = _producer()
        record = producer.failure_record(
            stage="on_trial",
            reason_code="negative_unattributed_seconds",
            timing_attempted=True,
            status="profile_invalid",
        )
        assert record["status"] == "profile_invalid"
        assert record["latency"] is None
        code = _code_only(inspect.getsource(producer.profile_cell))
        for clamping in ("max(0", "max(0.0", "abs(unattributed"):
            assert clamping not in code, clamping

    def test_a_negative_residual_is_never_clamped(self):
        """Verified by computing the residual the way the producer does and
        checking a negative value survives to the gate rather than becoming
        zero."""
        import statistics

        producer = _producer()
        # attributed exceeds the wall: over-coverage.
        walls = [1.0, 1.0, 1.0]
        attributed_per_trial = [1.2, 1.1, 1.3]
        residuals = [
            wall - attributed
            for wall, attributed in zip(walls, attributed_per_trial, strict=True)
        ]
        assert statistics.median(residuals) < 0
        source = inspect.getsource(producer.profile_cell)
        assert "statistics.median(residuals)" in source
        assert "residuals[index] < 0" in source
        # No clamping of the residual, in executable code.
        code = _code_only(source)
        for clamping in ("max ( 0", "abs ("):
            assert clamping not in code, clamping

    def test_the_negative_residual_branch_returns_profile_invalid(self):
        """The branch is exercised through `failure_record`, since driving
        `profile_cell` needs a GPU and the real checkpoint."""
        producer = _producer()
        source = inspect.getsource(producer.profile_cell)
        block = source.split("if trial_problems:")[1].split("return cell |")[1]
        assert 'status="profile_invalid"' in block
        assert 'reason_code="per_trial_coverage_or_residual_invalid"' in block
        assert "cleanup()" in source.split("if trial_problems:")[1][:200]

    def test_the_residual_is_a_median_of_per_trial_residuals(self):
        """Summing medians and subtracting the median wall is the "residual as
        a difference of medians" error: median-of-sums != sum-of-medians, so the
        two can cross and yield a NEGATIVE unattributed time. Three of five
        cells did exactly that before this was fixed."""
        producer = _producer()
        source = inspect.getsource(producer.profile_cell)
        assert "median of per-trial (wall - attributed)" in source
        assert "statistics.median(residuals)" in source
        assert '"unattributed_seconds_trials": residuals' in source

    def test_the_failure_stage_is_recorded(self):
        producer = _producer()
        source = inspect.getsource(producer.profile_cell)
        for stage in ("preflight", "warmup", "paired_trials"):
            assert f'"{stage}"' in source or f'= "{stage}"' in source
        # `timing_attempted` is true ONLY once a trial clock has started, so a
        # warmup OOM is not recorded as a timed failure.
        assert 'timing_attempted=stage == "paired_trials"' in source

    def test_cuda_state_is_cleaned_after_a_failure(self):
        producer = _producer()
        source = inspect.getsource(producer.profile_cell)
        failure_block = source.split("except Exception")[1]
        assert "cleanup()" in failure_block


class TestPairedInterleavedOrder:
    """The producer first ran OFF,OFF,OFF then ON,ON,ON. That loads thermal
    state, clock drift and allocator growth onto whichever arm always runs
    second — in the same direction as the effect being measured."""

    def test_the_arm_order_reverses_every_trial(self):
        producer = _producer()
        source = inspect.getsource(producer.paired_trials)
        assert 'order = ("off", "on") if index % 2 == 0 else ("on", "off")' in source

    def test_both_arms_run_inside_one_trial(self):
        """Not all of one arm and then all of the other."""
        producer = _producer()
        source = inspect.getsource(producer.paired_trials)
        loop = source.split("for index in range(TRIALS):")[1]
        assert "_run_off_trial" in loop
        assert "_run_on_trial" in loop

    def test_the_paired_delta_is_computed_from_the_two_walls(self):
        """A hardcoded 0.0 delta leaves every field present and every count
        right, while making the overhead permanently unresolvable — so it must
        be checked as arithmetic, not as a key."""
        producer = _producer()
        source = inspect.getsource(producer.paired_trials)
        assert (
            'measured["on"]["wall_seconds"] - measured["off"]["wall_seconds"]' in source
        )
        # And no constant is assigned to the field.
        delta_line = source.split('"paired_overhead_seconds":')[1].split("\n")[0]
        assert "0.0" not in delta_line
        assert "0," not in delta_line

    def test_the_recorded_delta_matches_the_recorded_walls(self):
        """Consistency of the trial record itself: whatever the walls say, the
        delta must be their difference."""
        producer = _producer()
        for off, on in ((1.0, 1.25), (2.5, 2.5), (3.0, 2.75)):
            trial = {
                "off_wall_seconds": off,
                "on_wall_seconds": on,
                "paired_overhead_seconds": on - off,
            }
            estimate = producer.overhead_estimate([trial])
            assert estimate["median_paired_delta"] == pytest.approx(on - off)

    def test_each_trial_records_its_order_and_paired_delta(self):
        producer = _producer()
        source = inspect.getsource(producer.paired_trials)
        for field in (
            '"order": list(order)',
            '"off_wall_seconds"',
            '"on_wall_seconds"',
            '"paired_overhead_seconds"',
        ):
            assert field in source, field

    def test_the_orders_alternate_across_the_frozen_window(self):
        producer = _producer()
        orders = [
            ("off", "on") if index % 2 == 0 else ("on", "off")
            for index in range(producer.TRIALS)
        ]
        assert orders == [("off", "on"), ("on", "off"), ("off", "on")]

    def test_warmup_is_a_separate_stage_from_the_trials(self):
        """A warmup OOM must not be recorded as a timed failure."""
        producer = _producer()
        assert "run_once" in inspect.getsource(producer.warmup_arms)
        assert "perf_counter" not in inspect.getsource(producer.warmup_arms)
        cell = inspect.getsource(producer.profile_cell)
        assert 'stage = "warmup"' in cell
        assert 'stage = "paired_trials"' in cell
        assert cell.index('stage = "warmup"') < cell.index('stage = "paired_trials"')

    def test_the_trials_are_not_run_inside_the_warmup_helper(self):
        producer = _producer()
        assert "TRIALS" not in inspect.getsource(producer.warmup_arms)


class TestPairedOverhead:
    def test_the_overhead_is_a_median_of_paired_deltas(self):
        """A difference of medians pairs an OFF trial with an unrelated ON
        trial — the error this cell already made twice."""
        producer = _producer()
        # Deltas are +0.10, +0.10, +0.10 within pairs, but the medians of the
        # two arms differ by much less because the walls drift together.
        trials = [
            {
                "off_wall_seconds": 1.0,
                "on_wall_seconds": 1.1,
                "paired_overhead_seconds": 0.1,
            },
            {
                "off_wall_seconds": 2.0,
                "on_wall_seconds": 2.1,
                "paired_overhead_seconds": 0.1,
            },
            {
                "off_wall_seconds": 3.0,
                "on_wall_seconds": 3.1,
                "paired_overhead_seconds": 0.1,
            },
        ]
        estimate = producer.overhead_estimate(trials)
        assert estimate["median_paired_delta"] == pytest.approx(0.1)
        assert estimate["paired_delta_trials"] == [0.1, 0.1, 0.1]

    def test_a_drifting_baseline_does_not_swamp_the_paired_signal(self):
        """The pairing is what makes a small overhead visible under drift: each
        delta is measured against its OWN adjacent baseline."""
        producer = _producer()
        trials = [
            {
                "off_wall_seconds": 1.0,
                "on_wall_seconds": 1.05,
                "paired_overhead_seconds": 0.05,
            },
            {
                "off_wall_seconds": 5.0,
                "on_wall_seconds": 5.05,
                "paired_overhead_seconds": 0.05,
            },
            {
                "off_wall_seconds": 9.0,
                "on_wall_seconds": 9.05,
                "paired_overhead_seconds": 0.05,
            },
        ]
        estimate = producer.overhead_estimate(trials)
        assert estimate["median_paired_delta"] == pytest.approx(0.05)
        # The OFF spread is huge, so it is honestly reported as unresolvable
        # rather than the delta being inflated by the drift.
        assert estimate["off_trial_spread"] == pytest.approx(8.0)
        assert estimate["resolvable"] is False

    def test_a_resolvable_negative_overhead_invalidates_the_cell(self):
        """Instrumentation cannot accelerate the measured code, so this is a
        broken clock, not a result."""
        producer = _producer()
        source = inspect.getsource(producer.profile_cell)
        assert (
            'overhead["resolvable"] and overhead["median_paired_delta"] < 0' in source
        )
        guard = source.split(
            'if overhead["resolvable"] and overhead["median_paired_delta"] < 0:'
        )[1]
        head = guard[: guard.index("problems=")]
        assert 'status="profile_invalid"' in head
        assert 'reason_code="resolvable_negative_overhead"' in head
        assert "cannot accelerate" in guard


class TestPerTrialGates:
    """A median-only gate hides one broken trial: coverage [1.05, 0.99, 0.98]
    and residuals [-3ms, +1ms, +2ms] both pass while a trial is plainly wrong."""

    def test_every_trial_is_checked_for_coverage(self):
        producer = _producer()
        source = inspect.getsource(producer.profile_cell)
        block = source.split("trial_problems: list[str] = []")[1]
        assert "for index, trial in enumerate(trials):" in block
        assert "coverage_per_trial[index] > 1.0 + SHARE_TOLERANCE" in block

    def test_every_trial_is_checked_for_a_negative_residual(self):
        producer = _producer()
        block = inspect.getsource(producer.profile_cell).split(
            "trial_problems: list[str] = []"
        )[1]
        assert "residuals[index] < 0" in block

    def test_every_trial_is_checked_for_a_positive_wall(self):
        producer = _producer()
        block = inspect.getsource(producer.profile_cell).split(
            "trial_problems: list[str] = []"
        )[1]
        assert 'trial["on_wall_seconds"] <= 0' in block
        assert 'trial["off_wall_seconds"] <= 0' in block

    def test_every_event_value_is_checked_for_finiteness(self):
        producer = _producer()
        block = inspect.getsource(producer.profile_cell).split(
            "trial_problems: list[str] = []"
        )[1]
        assert "math.isfinite(value)" in block
        assert "value < 0" in block

    def test_both_per_trial_arrays_reach_the_artifact(self):
        """The earlier record kept the residual trials and discarded the
        coverage trials."""
        producer = _producer()
        source = inspect.getsource(producer.profile_cell)
        assert '"coverage_ratio_trials": coverage_per_trial' in source
        assert '"unattributed_seconds_trials": residuals' in source


class TestPerTrialPeakMemory:
    def test_peak_stats_are_reset_inside_each_trial(self):
        """One reset before all trials reports the maximum over the whole run as
        if it were a per-trial figure."""
        producer = _producer()
        source = inspect.getsource(producer._run_off_trial)
        assert "reset_peak_memory_stats()" in source
        reset = source.index("reset_peak_memory_stats()")
        clock = source.index("begin = time.perf_counter()")
        assert reset < clock, "the reset must precede the clock"

    def test_the_arrays_and_the_aggregate_max_are_both_recorded(self):
        producer = _producer()
        source = inspect.getsource(producer.profile_cell)
        for field in (
            '"allocated_bytes_trials": allocated',
            '"reserved_bytes_trials": reserved',
            '"max_allocated_bytes": max(allocated)',
            '"max_reserved_bytes": max(reserved)',
        ):
            assert field in source, field

    def test_the_peak_comes_from_the_off_arm(self):
        """The ON arm carries the observer's own allocations."""
        producer = _producer()
        source = inspect.getsource(producer.paired_trials)
        assert (
            '"peak_allocated_bytes": measured["off"]["peak_allocated_bytes"]' in source
        )
        assert "peak" not in _code_only(inspect.getsource(producer._run_on_trial))


class TestDeviceExclusivity:
    """A typed `oom` is DATA in this protocol, so it must be attributable to the
    cell. The previous 32x32 OOM was recorded while three foreign processes held
    ~19.8 GiB of 47.37 GiB, leaving 1.77 GiB free — that says nothing about
    whether the cell intrinsically exceeds the device."""

    def test_a_shared_device_is_refused(self):
        producer = _producer()
        import os

        foreign = {
            "device": "cuda:0",
            "compute_processes": [
                {"pid": os.getpid(), "used_mib": 100},
                {"pid": os.getpid() + 99999, "used_mib": 14096},
            ],
        }
        producer.device_occupancy = lambda _device: foreign
        with pytest.raises(SystemExit, match="is shared"):
            producer.require_idle_device("cuda:0")

    def test_an_idle_device_passes_and_returns_the_occupancy(self):
        """The producer's OWN process holds memory on the device it measures, so
        counting self as foreign would refuse every legitimate run."""
        producer = _producer()
        import os

        idle = {
            "device": "cuda:0",
            "compute_processes": [{"pid": os.getpid(), "used_mib": 4096}],
        }
        producer.device_occupancy = lambda _device: idle
        assert producer.require_idle_device("cuda:0") is idle

    def test_the_producers_own_process_is_not_counted_as_foreign(self):
        producer = _producer()
        import os

        # Self only, with a large allocation: must still pass.
        producer.device_occupancy = lambda _device: {
            "device": "cuda:0",
            "compute_processes": [{"pid": os.getpid(), "used_mib": 40000}],
        }
        producer.require_idle_device("cuda:0")
        # Empty is also fine (measured before any allocation).
        producer.device_occupancy = lambda _device: {
            "device": "cuda:0",
            "compute_processes": [],
        }
        producer.require_idle_device("cuda:0")

    def test_the_foreign_count_excludes_self(self):
        producer = _producer()
        occupancy = producer.device_occupancy("cuda:0")
        own = [
            entry
            for entry in occupancy["compute_processes"]
            if entry["pid"] == __import__("os").getpid()
        ]
        assert occupancy["foreign_process_count"] == len(
            occupancy["compute_processes"]
        ) - len(own)

    def test_occupancy_uses_nvml_process_data_not_process_names(self):
        """`pgrep` cannot tell whether a process holds GPU memory, nor on which
        device."""
        producer = _producer()
        source = inspect.getsource(producer.device_occupancy)
        assert "compute-apps" in source
        assert "gpu_uuid" in source
        # The docstring explains why `pgrep` is unsuitable, so the check has to
        # look at executable code rather than prose.
        code = _code_only(source)
        assert "pgrep" not in code
        assert "process_name" not in code
        assert "mem_get_info" in code

    def test_occupancy_is_scoped_to_the_target_device(self):
        """`nvidia-smi` lists compute apps for EVERY device, so unscoped rows
        would fail a run because another GPU is busy."""
        producer = _producer()
        source = inspect.getsource(producer.device_occupancy)
        assert "row_uuid == uuid" in source

    def test_occupancy_records_free_and_total_memory(self):
        producer = _producer()
        occupancy = producer.device_occupancy("cuda:0")
        assert occupancy["total_bytes"] > 0
        assert 0 <= occupancy["free_bytes"] <= occupancy["total_bytes"]
        assert occupancy["source"].startswith("nvidia-smi")

    def test_occupancy_is_rechecked_before_every_cell(self):
        """A foreign process can arrive mid-run; a startup-only check would
        leave every later cell silently contaminated."""
        producer = _producer()
        source = inspect.getsource(producer.profile_cell)
        assert 'cell["device_occupancy_before"] = require_idle_device(' in source

    def test_the_artifact_records_the_exclusivity_contract(self):
        producer = _producer()
        source = inspect.getsource(producer.provenance)
        assert '"device_occupancy_at_start"' in source
        assert '"exclusivity_contract"' in source
        assert "attributable to the cell" in source


class TestOomClassification:
    def test_a_cuda_specific_message_classifies(self):
        producer = _producer()
        for message in (
            "CUDA out of memory. Tried to allocate 2.00 GiB",
            "CUDA error: out of memory",
        ):
            assert (
                producer.classify_failure(RuntimeError(message)) == "cuda_out_of_memory"
            )

    def test_a_host_side_out_of_memory_is_not_a_cuda_oom(self):
        """`time_schedule` runs a scipy spline on the HOST, so a bare
        "out of memory" here can be a host or SciPy allocation failure — a
        different capacity story that must not be filed as a CUDA OOM."""
        producer = _producer()
        for message in (
            "out of memory",
            "Unable to allocate 4.00 GiB for an array with shape (10000,)",
            "std::bad_alloc: out of memory",
        ):
            assert producer.classify_failure(RuntimeError(message)) is None, message

    def test_a_cpu_allocator_message_is_not_a_cuda_oom(self):
        """ "tried to allocate" is NOT CUDA-specific: the host allocator uses the
        same phrasing, and this cell deliberately enters NumPy/SciPy host code
        every step."""
        producer = _producer()
        for message in (
            "CPU out of memory. Tried to allocate 6.14 GiB",
            "DefaultCPUAllocator: not enough memory: you tried to allocate "
            "6144000000 bytes. Out of memory",
            "out of memory. Tried to allocate 2.00 GiB on host",
        ):
            assert producer.classify_failure(RuntimeError(message)) is None, message

    def test_the_classifier_requires_the_word_cuda(self):
        producer = _producer()
        source = inspect.getsource(producer.classify_failure)
        assert '"cuda" in text' in source
        # The over-broad phrase list is gone from executable code (the docstring
        # still names it as the rejected approach).
        code = _code_only(source)
        assert "cuda_specific" not in code
        # And behaviorally: "cuda" present classifies, absent does not.
        assert (
            producer.classify_failure(
                RuntimeError("CUDA out of memory. Tried to allocate 1 GiB")
            )
            == "cuda_out_of_memory"
        )
        assert (
            producer.classify_failure(
                RuntimeError("out of memory. Tried to allocate 1 GiB")
            )
            is None
        )

    def test_the_torch_oom_type_always_classifies(self):
        producer = _producer()
        assert (
            producer.classify_failure(torch.cuda.OutOfMemoryError("anything"))
            == "cuda_out_of_memory"
        )


class TestStableHash:
    def test_it_detects_a_tiny_perturbation(self):
        producer = _producer()
        base = torch.randn(8, 8)
        perturbed = base.clone()
        perturbed[0, 0] += 1e-7
        assert producer.stable_hash(base) != producer.stable_hash(perturbed)

    def test_it_is_deterministic(self):
        producer = _producer()
        tensor = torch.randn(8, 8)
        assert producer.stable_hash(tensor) == producer.stable_hash(tensor)

    @pytest.mark.parametrize(
        "tensor",
        [
            torch.randn(2, 3),
            torch.randn(2, 3).bfloat16(),
            torch.randint(0, 5, (2, 3)),
            torch.get_rng_state(),
        ],
    )
    def test_it_handles_every_dtype_the_producer_hashes(self, tensor):
        record = _producer().stable_hash(tensor)
        assert len(record["sha256"]) == 64
        assert record["dtype"] and record["shape"]

    def test_it_records_shape_dtype_and_device(self):
        """A bare hash with no shape is not auditable."""
        record = _producer().stable_hash(torch.randn(4, 5))
        assert record["shape"] == [4, 5]
        assert record["dtype"] == "torch.float32"
        assert "device" in record


class TestArtifactSemantics:
    def test_the_execution_scope_is_recorded(self):
        producer = _producer()
        source = inspect.getsource(producer.provenance)
        for key in (
            "request_concurrency",
            "execution_mode",
            "observer_isolation",
            "observer_concurrency_contract",
            "rng_concurrency_contract",
        ):
            assert key in source, key

    def test_the_two_concurrency_contracts_are_stated_separately(self):
        """Event isolation and RNG determinism are different claims; conflating
        them would overstate what the ContextVar buys."""
        producer = _producer()
        source = inspect.getsource(producer.provenance)
        assert "event isolation only" in source
        assert "not deterministic across concurrent" in source
        assert "not a defect of this profile" in source

    def test_the_gamma_reduction_is_algebraic_not_asserted_bit_identity(self):
        producer = _producer()
        source = inspect.getsource(producer.provenance)
        assert "ALGEBRAICALLY reduces" in source
        assert "not an asserted" in source

    def test_the_state_update_is_not_described_as_free(self):
        producer = _producer()
        source = inspect.getsource(producer.provenance)
        assert "degenerate bookkeeping" in source
        assert "randn_like draws" in source
        assert "measured rather than assumed free" in source

    def test_every_event_description_states_its_boundaries(self):
        producer = _producer()
        assert set(producer.EVENT_TAXONOMY) == {
            "grid_init",
            "time_schedule",
            "flow_map_forward",
            "state_update",
            "endpoint_decode",
        }
        assert "exp()" in producer.EVENT_TAXONOMY["flow_map_forward"]
        assert "randn_like" in producer.EVENT_TAXONOMY["state_update"]
        assert (
            "NOT reached on the final step" in producer.EVENT_TAXONOMY["state_update"]
        )

    def test_the_time_schedule_description_names_the_host_round_trip(self):
        """A reader seeing 33.7% on "scalar arithmetic" would suspect the
        instrumentation. The description has to name the real mechanism."""
        text = _producer().EVENT_TAXONOMY["time_schedule"]
        assert "cpu()" in text
        assert "SYNCHRONIZES" in text
        assert "CubicSpline" in text
        assert "Three such round trips per step" in text

    def test_no_taxonomy_description_quotes_a_measurement(self):
        """A description must not carry numbers from a run other than the one
        that produced the artifact around it. The `time_schedule` entry did
        exactly that: it quoted "~515 ms attributed" from the invalidated
        unpaired run, while the paired run attributes ~354 ms."""
        import re

        producer = _producer()
        # Any millisecond/second/percent figure is a measurement, and
        # measurements belong in the cells, not the taxonomy.
        pattern = re.compile(r"\d+(?:\.\d+)?\s*(?:ms|milliseconds|s\b|%)")
        for name, description in producer.EVENT_TAXONOMY.items():
            found = pattern.findall(description)
            assert not found, f"{name} quotes a measurement: {found}"

    def test_the_time_schedule_description_stays_qualitative(self):
        """The mechanism is named without a figure pinned to it."""
        text = _producer().EVENT_TAXONOMY["time_schedule"]
        assert "cpu()" in text
        assert "CubicSpline" in text
        assert "run-specific figure" in text

    def test_the_description_records_that_the_dominance_claim_was_refuted(self):
        """An earlier run on a SHARED GPU attributed ~354 ms (28.7%) to this
        event; on an idle device the same cell attributes ~8 ms. The mechanism
        is real, the dominance was contention. A future reader must not
        resurrect the retracted claim from the mechanism's presence."""
        text = _producer().EVENT_TAXONOMY["time_schedule"]
        assert "SHARED GPU" in text
        assert "did not survive re-measurement" in text

    def test_grid_init_excludes_manual_seed(self):
        """The description must match the real boundary, not absorb adjacent
        work."""
        assert "NOT include manual_seed" in _producer().EVENT_TAXONOMY["grid_init"]

    def test_unattributed_time_is_reported_rather_than_folded_away(self):
        producer = _producer()
        source = inspect.getsource(producer.profile_cell)
        assert "unattributed_seconds" in source
        assert "deliberately left unattributed" in source

    def test_provenance_refuses_an_unknown_measuring_commit(self):
        producer = _producer()
        source = inspect.getsource(producer.provenance)
        assert "refusing to write an artifact" in source
        assert '"unknown"' not in source
