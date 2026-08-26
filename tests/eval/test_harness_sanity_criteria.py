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

"""#166 sanity-gate criteria — the pure decision functions, no GPU needed."""

from __future__ import annotations

import importlib.util
import pathlib

import pytest


def _harness():
    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "kernels"
        / "harness_sanity.py"
    )
    spec = importlib.util.spec_from_file_location("_harness_sanity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestNoisingEquivalence:
    """The noising finding is an ABSENCE of effect, so the gate is an
    equivalence test — a dispersion test would accept measurement chaos."""

    def test_wild_symmetric_noise_is_not_equivalence(self):
        """+50/-50/+50 has its median inside its own spread with mixed signs,
        so a dispersion criterion would pass it. It is chaos, not agreement."""
        harness = _harness()
        result = harness.equivalence(
            [0.5, -0.5, 0.5], margin=harness.NOISING_EQUIVALENCE_MARGIN
        )
        assert result["equivalent"] is False

    def test_small_scattered_deltas_are_equivalent(self):
        harness = _harness()
        result = harness.equivalence(
            [0.004, -0.006, 0.011, -0.002, 0.008],
            margin=harness.NOISING_EQUIVALENCE_MARGIN,
        )
        assert result["equivalent"] is True

    def test_a_small_but_consistent_bias_is_not_equivalence(self):
        """3% every trial is material even though it is 'small'."""
        harness = _harness()
        result = harness.equivalence(
            [0.03, 0.031, 0.029, 0.03, 0.032],
            margin=harness.NOISING_EQUIVALENCE_MARGIN,
        )
        assert result["equivalent"] is False

    def test_one_excursion_fails_even_with_a_tiny_median(self):
        """Every trial must sit inside the margin, not just the median."""
        harness = _harness()
        result = harness.equivalence(
            [0.001, 0.002, 0.9, -0.001, 0.0],
            margin=harness.NOISING_EQUIVALENCE_MARGIN,
        )
        assert result["median_within_margin"] is True
        assert result["all_trials_within_margin"] is False
        assert result["equivalent"] is False

    def test_the_margin_and_trial_count_are_frozen(self):
        harness = _harness()
        assert harness.NOISING_EQUIVALENCE_MARGIN == 0.02
        assert harness.NOISING_TRIALS == 5

    def test_empty_deltas_are_refused(self):
        harness = _harness()
        with pytest.raises(ValueError, match="nothing to check"):
            harness.equivalence([], margin=0.02)


class TestSignConsistency:
    def test_sign_alone_is_not_enough_without_clearing_spread(self):
        harness = _harness()
        result = harness.sign_consistency([-0.001, -0.002, -0.5], expect_negative=True)
        assert result["majority_agrees"] is True
        assert result["exceeds_spread"] is False
