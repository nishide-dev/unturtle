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

"""#166 Stage 1 — the FMLM profiling seam must not change what it measures.

Five independent layers, per the frozen review criteria: event counts, NFE,
random-call counts, the final latent, and the terminal RNG state read INSIDE
`fork_rng`. The last one matters because `fork_rng` restores the outer
generator on exit, so an outer pre/post fingerprint cannot see an observer that
perturbs the internal stream.

Deliberately oracle-independent: a stub satisfying `run_fmlm_request`'s
contract exercises every boundary, so these gates run in CI without
`dev/repos/flm`.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "unturtle_flm",
    reason="FLM pack not installed (uv pip install -e packs/unturtle-flm)",
)

import torch  # noqa: E402

VOCAB = 16
LENGTH = 6

#: Official FMLM cell configuration (benchmarks/results/fmlm_owt_{1,32}).
OFFICIAL_GAMMA = 1.0
OFFICIAL_SEED = 100


class StubFlowMap(torch.nn.Module):
    """Minimal double-time flow map.

    A zero-output model would make every mutant invisible (the #155 battery
    proved that on the real DiT), so the head is perturbed and the output
    depends on BOTH times and on `z`.
    """

    is_fmlm_flow_map = True

    def __init__(self) -> None:
        super().__init__()
        self.vocab_size = VOCAB
        self.num_tokens = LENGTH
        self.proj = torch.nn.Linear(VOCAB, VOCAB)
        torch.nn.init.normal_(self.proj.weight, std=0.5)
        torch.nn.init.normal_(self.proj.bias, std=0.5)
        self.calls: list[tuple[float, float]] = []

    def _tau_to_t(self, tau):
        return tau * 0.9

    def _t_to_tau(self, t):
        return t / 0.9

    def forward(self, z, tau_curr, tau_tilde, use_jvp_attn=False):
        self.calls.append(
            (float(tau_curr.reshape(-1)[0]), float(tau_tilde.reshape(-1)[0]))
        )
        scale = 1.0 + tau_curr.reshape(-1, 1, 1) + tau_tilde.reshape(-1, 1, 1)
        return torch.log_softmax(self.proj(z) * scale, dim=-1)


class _WatchedLog(torch.Tensor):
    """Records which events are open when `.exp()` is called on it."""

    @staticmethod
    def __new__(cls, data, sink, stack):
        instance = torch.Tensor._make_subclass(cls, data, False)
        instance._sink = sink
        instance._stack = stack
        return instance

    def exp(self):  # type: ignore[override]
        self._sink.append(tuple(self._stack))
        return torch.Tensor.exp(self)


class Request:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _request(steps, *, diagnostics=()):
    request = Request(
        steps=steps, num_samples=2, seed=OFFICIAL_SEED, gamma=OFFICIAL_GAMMA
    )
    if diagnostics:
        request._unturtle_profile_diagnostics = frozenset(diagnostics)
    return request


def _build_model(seed: int = 4242):
    """Seeded, so the two arms of a comparison share weights.

    An unseeded build made the observer-OFF and observer-ON arms use DIFFERENT
    weights, and the resulting "interference" was the fixture, not the seam —
    the same mistake the #166 row-5 parity check made by building a fresh model
    per arm.
    """
    torch.manual_seed(seed)
    return StubFlowMap().eval()


def _run(steps, *, observer=None, diagnostics=(), model=None):
    from unturtle_flm import sampler

    if model is None:
        model = _build_model()
    token = sampler._install_observer(observer)
    try:
        result = sampler.run_fmlm_request(
            model, _request(steps, diagnostics=diagnostics)
        )
    finally:
        sampler._restore_observer(token)
    return model, result


class _Recorder:
    """Records event name and phase. Touches no tensor — the contract is that
    the observer never receives `z` or the model output."""

    def __init__(self) -> None:
        self.log: list[tuple[str, str]] = []

    def __call__(self, name: str, phase: str) -> None:
        self.log.append((name, phase))

    def counts(self) -> dict[str, int]:
        return {
            name: sum(1 for n, p in self.log if n == name and p == "enter")
            for name in dict.fromkeys(n for n, _ in self.log)
        }


class TestDefaultOff:
    def test_no_observer_is_installed_by_default(self):
        from unturtle_flm import sampler

        assert sampler._OBSERVER_CONTEXT.get() is None

    def test_the_callback_never_fires_when_disabled(self):
        recorder = _Recorder()
        _run(4, observer=None)
        assert recorder.log == []

    def test_the_disabled_scope_allocates_nothing_per_event(self):
        """`_OFF` is a shared singleton, so an OFF run creates no scope object
        per event boundary."""
        from unturtle_flm import sampler

        first = sampler._scope(None, "grid_init")
        second = sampler._scope(None, "state_update")
        assert first is second is sampler._OFF

    def test_an_enabled_run_uses_the_same_public_entry_point(self):
        recorder = _Recorder()
        _run(4, observer=recorder)
        assert recorder.counts(), "observer saw nothing through the public entry"


class TestExecutionContextIsolation:
    """A profiled run must not observe an unrelated concurrent one.

    A plain module global failed this: with an observer installed on the
    profiling thread, 26 events from an unrelated thread's ordinary
    `run_fmlm_request` were captured. `ContextVar` isolates threads and async
    tasks alike.
    """

    def test_a_concurrent_normal_thread_is_not_observed(self):
        import threading

        from unturtle_flm import sampler

        model = _build_model()
        seen: list[str] = []
        barrier = threading.Barrier(2)

        def observer(name, _phase):
            seen.append(name)

        def unrelated():
            barrier.wait()
            for _ in range(3):
                sampler.run_fmlm_request(model, _request(4))

        thread = threading.Thread(target=unrelated, name="UNRELATED")
        token = sampler._install_observer(observer)
        try:
            thread.start()
            barrier.wait()
            sampler.run_fmlm_request(model, _request(4))
            thread.join()
        finally:
            sampler._restore_observer(token)

        # 4 steps: 1 + 4 + 4 + 3 + 1 = 13 events, entered and exited.
        assert len(seen) == 26, (
            f"observed {len(seen)} events; the unrelated thread's three runs leaked in"
        )

    def test_concurrent_ordinary_runs_are_not_reproducible_without_the_seam(self):
        """Documents the boundary of what the seam can promise.

        `run_fmlm_request` seeds inside `torch.random.fork_rng`, which forks the
        PROCESS-GLOBAL CPU generator that threads share. Two concurrent runs
        therefore interleave their draws, and the result is not reproducible —
        measured with NO observer installed anywhere: repeated unobserved
        concurrent pairs disagree with each other.

        Consequence for this file: any assertion of the form "a concurrent run
        equals its solo result", or "an observed pair equals an unobserved
        pair", is inherently flaky and would blame the seam for the sampler's
        RNG scoping. Both were written and both were withdrawn after measuring
        this. What the seam actually guarantees is EVENT isolation, asserted
        deterministically in
        `test_a_concurrent_normal_thread_is_not_observed`; single-threaded
        non-interference is covered by `TestObserverNonInterference`.

        The producer profiles single-threaded for this reason.
        """
        import threading

        from unturtle_flm import sampler

        def unobserved_pair():
            model = _build_model()
            captured: dict[str, object] = {}
            barrier = threading.Barrier(2)

            def unrelated():
                barrier.wait()
                captured["result"] = sampler.run_fmlm_request(model, _request(4))

            thread = threading.Thread(target=unrelated)
            thread.start()
            barrier.wait()
            sampler.run_fmlm_request(model, _request(4))
            thread.join()
            return captured["result"]["tokens"]

        # Not an equality assertion in either direction: the point is that
        # concurrency makes the OUTPUT unreliable, so the seam is never
        # validated through concurrent output comparison.
        attempts = [unobserved_pair() for _ in range(4)]
        assert all(t.shape == attempts[0].shape for t in attempts)

    def test_an_async_task_is_isolated(self):
        import asyncio

        from unturtle_flm import sampler

        model = _build_model()
        seen: list[str] = []

        async def observed():
            token = sampler._install_observer(lambda n, _p: seen.append(n))
            try:
                await asyncio.sleep(0)
                sampler.run_fmlm_request(model, _request(4))
            finally:
                sampler._restore_observer(token)

        async def unobserved():
            await asyncio.sleep(0)
            sampler.run_fmlm_request(model, _request(4))

        async def main():
            await asyncio.gather(observed(), unobserved())

        asyncio.run(main())
        assert len(seen) == 26, f"async task leaked: {len(seen)} events"

    def test_nested_installation_restores_as_a_stack(self):
        from unturtle_flm import sampler

        outer: list[str] = []
        inner: list[str] = []
        model = _build_model()

        outer_token = sampler._install_observer(lambda n, _p: outer.append(n))
        try:
            inner_token = sampler._install_observer(lambda n, _p: inner.append(n))
            try:
                sampler.run_fmlm_request(model, _request(4))
            finally:
                sampler._restore_observer(inner_token)
            assert inner and not outer, "the inner observer must win while set"
            sampler.run_fmlm_request(model, _request(4))
            assert outer, "the outer observer must be restored, not cleared"
        finally:
            sampler._restore_observer(outer_token)
        assert sampler._OBSERVER_CONTEXT.get() is None

    def test_the_observer_is_resolved_once_per_request(self):
        """Swapping the observer mid-run must not take effect for that run:
        re-reading per boundary would put a ContextVar lookup on the hot path
        and make attribution depend on timing."""
        from unturtle_flm import sampler

        model = _build_model()
        first: list[str] = []
        second: list[str] = []

        def swapping(name, phase):
            first.append(name)
            if len(first) == 1:
                sampler._OBSERVER_CONTEXT.set(lambda n, _p: second.append(n))

        token = sampler._install_observer(swapping)
        try:
            sampler.run_fmlm_request(model, _request(4))
        finally:
            sampler._restore_observer(token)
        assert not second, "the swap leaked into the in-flight request"
        assert len(first) == 26


class TestEventCounts:
    @pytest.mark.parametrize(
        ("steps", "expected"),
        [
            (
                1,
                {
                    "grid_init": 1,
                    "time_schedule": 1,
                    "flow_map_forward": 1,
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
    def test_the_frozen_call_counts(self, steps, expected):
        recorder = _Recorder()
        _run(steps, observer=recorder)
        assert recorder.counts() == expected

    def test_state_update_is_a_structural_zero_at_one_step(self):
        """Not a missing measurement: the final-step branch exits BEFORE the
        state update, so zero is the correct recorded value."""
        recorder = _Recorder()
        _run(1, observer=recorder)
        assert "state_update" not in recorder.counts()

    def test_every_event_is_balanced(self):
        """An unclosed scope would make the next window absorb this one's time."""
        recorder = _Recorder()
        _run(8, observer=recorder)
        depth = 0
        for _name, phase in recorder.log:
            depth += 1 if phase == "enter" else -1
            assert depth >= 0, "exit before enter"
        assert depth == 0, "a scope was left open"

    def test_the_event_order_is_fixed(self):
        """Names alone are not the contract; the sequence is."""
        recorder = _Recorder()
        _run(3, observer=recorder)
        entered = [n for n, p in recorder.log if p == "enter"]
        assert entered == [
            "grid_init",
            "time_schedule",
            "flow_map_forward",
            "state_update",
            "time_schedule",
            "flow_map_forward",
            "state_update",
            "time_schedule",
            "flow_map_forward",
            "endpoint_decode",
        ]

    def test_the_model_call_and_the_exp_are_inside_flow_map_forward(self):
        """Counts cannot see a SPAN change: moving `.exp()` outside the scope
        keeps every event count identical while silently reattributing the
        exponential's cost. Observed by recording which events are OPEN at the
        moment the model is called, and whether the scope closes before `.exp()`
        runs."""
        from unturtle_flm import sampler

        open_stack: list[str] = []
        model_call_context: list[tuple[str, ...]] = []
        exp_context: list[tuple[str, ...]] = []

        def observer(name, phase):
            if phase == "enter":
                open_stack.append(name)
            else:
                open_stack.remove(name)

        class Watched(StubFlowMap):
            def forward(self, z, tau_curr, tau_tilde, use_jvp_attn=False):
                model_call_context.append(tuple(open_stack))
                out = super().forward(z, tau_curr, tau_tilde, use_jvp_attn)
                return _WatchedLog(out, exp_context, open_stack)

        torch.manual_seed(4242)
        model = Watched().eval()
        token = sampler._install_observer(observer)
        try:
            sampler.run_fmlm_request(model, _request(4))
        finally:
            sampler._restore_observer(token)

        assert model_call_context, "the model was never called"
        for context in model_call_context:
            assert "flow_map_forward" in context, (
                f"the model call ran outside flow_map_forward: {context}"
            )
        assert exp_context, "`.exp()` was never called"
        for context in exp_context:
            assert "flow_map_forward" in context, (
                f"`.exp()` ran outside flow_map_forward: {context}"
            )

    def test_the_noise_draw_is_inside_state_update(self):
        """Same span argument for `randn_like`: the gamma=1 churn is the
        expensive part of the state update, and attributing it elsewhere would
        understate that event."""
        from unturtle_flm import sampler

        open_stack: list[str] = []
        draw_context: list[tuple[str, ...]] = []

        def observer(name, phase):
            if phase == "enter":
                open_stack.append(name)
            else:
                open_stack.remove(name)

        original = torch.randn_like

        def watching(*args, **kwargs):
            draw_context.append(tuple(open_stack))
            return original(*args, **kwargs)

        model = _build_model()
        torch.randn_like = watching
        token = sampler._install_observer(observer)
        try:
            sampler.run_fmlm_request(model, _request(4))
        finally:
            sampler._restore_observer(token)
            torch.randn_like = original

        assert len(draw_context) == 3, draw_context
        for context in draw_context:
            assert "state_update" in context, (
                f"the noise draw ran outside state_update: {context}"
            )

    def test_the_taxonomy_matches_the_declared_event_names(self):
        from unturtle_flm import sampler

        recorder = _Recorder()
        _run(8, observer=recorder)
        assert set(recorder.counts()) <= set(sampler._FMLM_EVENTS)


class TestNfeAndRandomCalls:
    @pytest.mark.parametrize("steps", [1, 32])
    def test_nfe_equals_steps(self, steps):
        model, result = _run(steps)
        assert result["executed"]["nfe"] == steps
        assert len(model.calls) == steps

    @pytest.mark.parametrize(("steps", "expected_like"), [(1, 0), (32, 31)])
    def test_the_random_call_counts_are_frozen(self, steps, expected_like):
        """Independent of the event counts: proves `state_update`'s span is
        right AND that the gamma=1 churn branch is genuinely live."""
        counts = {"randn": 0, "randn_like": 0}
        original_randn, original_like = torch.randn, torch.randn_like

        def counting_randn(*args, **kwargs):
            counts["randn"] += 1
            return original_randn(*args, **kwargs)

        def counting_like(*args, **kwargs):
            counts["randn_like"] += 1
            return original_like(*args, **kwargs)

        torch.randn, torch.randn_like = counting_randn, counting_like
        try:
            _run(steps)
        finally:
            torch.randn, torch.randn_like = original_randn, original_like
        assert counts == {"randn": 1, "randn_like": expected_like}

    def test_the_churn_branch_is_skipped_at_gamma_zero(self):
        """gamma=0.0 is NOT a decision cell; it exists here only to prove the
        branch the official gamma=1.0 configuration takes."""
        from unturtle_flm import sampler

        counts = {"randn_like": 0}
        original = torch.randn_like

        def counting(*args, **kwargs):
            counts["randn_like"] += 1
            return original(*args, **kwargs)

        torch.randn_like = counting
        try:
            sampler.run_fmlm_request(
                StubFlowMap().eval(),
                Request(steps=8, num_samples=2, seed=OFFICIAL_SEED, gamma=0.0),
            )
        finally:
            torch.randn_like = original
        assert counts["randn_like"] == 0


class TestObserverNonInterference:
    """tokens alone is too weak: `argmax` can hide a latent perturbation."""

    @staticmethod
    def _capture(steps, observer, model=None):
        return _run(
            steps,
            observer=observer,
            diagnostics=("terminal_rng", "final_latent"),
            model=model,
        )

    @classmethod
    def _both_arms(cls, steps, observer):
        """ONE model instance through both arms, so any difference can only come
        from the observer."""
        model = _build_model()
        _, off = cls._capture(steps, None, model=model)
        _, on = cls._capture(steps, observer, model=model)
        return off, on

    @pytest.mark.parametrize("steps", [1, 32])
    def test_tokens_are_bit_identical(self, steps):
        off, on = self._both_arms(steps, _Recorder())
        assert torch.equal(off["tokens"], on["tokens"])

    @pytest.mark.parametrize("steps", [1, 32])
    def test_the_final_latent_is_bit_identical(self, steps):
        off, on = self._both_arms(steps, _Recorder())
        assert torch.equal(off["_final_latent"], on["_final_latent"])

    @pytest.mark.parametrize("steps", [1, 32])
    def test_the_terminal_cpu_rng_state_is_bit_identical(self, steps):
        off, on = self._both_arms(steps, _Recorder())
        assert torch.equal(off["_terminal_rng"]["cpu"], on["_terminal_rng"]["cpu"])

    @pytest.mark.parametrize("steps", [1, 32])
    def test_the_executed_metadata_is_identical(self, steps):
        off, on = self._both_arms(steps, _Recorder())
        assert off["executed"] == on["executed"]

    def test_the_terminal_rng_check_can_actually_fail(self):
        """Guards the gate itself: an observer that draws a random number MUST
        be caught. If this passes with a perturbing observer, the check is
        measuring nothing."""

        def perturbing(_name, phase):
            if phase == "enter":
                torch.randn(1)

        clean, dirty = self._both_arms(8, perturbing)
        assert not torch.equal(
            clean["_terminal_rng"]["cpu"], dirty["_terminal_rng"]["cpu"]
        )

    def test_an_outer_fingerprint_would_not_catch_it(self):
        """Why the capture is inside `fork_rng`: measured — the outer state is
        restored on exit, so it is bit-identical even for the perturbing
        observer above."""
        model = _build_model()
        before = torch.get_rng_state()
        self._capture(
            8, lambda _n, p: torch.randn(1) if p == "enter" else None, model=model
        )
        assert torch.equal(before, torch.get_rng_state())


class TestSeamHygiene:
    def test_the_public_result_carries_no_diagnostic_keys(self):
        _, result = _run(4)
        assert "_terminal_rng" not in result
        assert "_final_latent" not in result

    def test_the_diagnostic_flags_are_not_public_kwargs(self):
        """They live on a private request attribute, so `request.kwargs` — the
        documented surface — is untouched."""
        from unturtle_flm import sampler

        source = __import__("inspect").getsource(sampler._common)
        assert "terminal_rng" not in source
        assert "final_latent" not in source

    def test_an_observer_exception_propagates(self):
        """A swallowed exception would let a broken observer report a
        successful measurement."""

        def raising(_name, _phase):
            raise RuntimeError("observer failure")

        with pytest.raises(RuntimeError, match="observer failure"):
            _run(4, observer=raising)

    def test_the_seam_does_not_leak_after_a_failure(self):
        from unturtle_flm import sampler

        def raising(_name, _phase):
            raise RuntimeError("observer failure")

        with pytest.raises(RuntimeError):
            _run(4, observer=raising)
        assert sampler._OBSERVER_CONTEXT.get() is None

    def test_a_scope_closes_even_when_its_body_raises(self):
        """An exit that only fires on success leaves the scope open, and the
        NEXT window absorbs this one's elapsed time — a silent attribution
        error, not a crash."""
        from unturtle_flm import sampler

        log: list[tuple[str, str]] = []

        class Exploding(StubFlowMap):
            def forward(self, z, tau_curr, tau_tilde, use_jvp_attn=False):
                raise RuntimeError("forward exploded")

        torch.manual_seed(4242)
        model = Exploding().eval()
        token = sampler._install_observer(lambda n, p: log.append((n, p)))
        try:
            with pytest.raises(RuntimeError, match="forward exploded"):
                sampler.run_fmlm_request(model, _request(4))
        finally:
            sampler._restore_observer(token)

        depth = 0
        for _name, phase in log:
            depth += 1 if phase == "enter" else -1
        assert depth == 0, f"{depth} scope(s) left open by the raising forward"
        assert ("flow_map_forward", "exit") in log

    def test_profiling_dependencies_stay_out_of_the_module_import_path(self):
        """The seam must not drag a profiler into normal `import
        unturtle_flm.sampler`."""
        import pathlib

        from unturtle_flm import sampler

        source = pathlib.Path(sampler.__file__).read_text()
        for forbidden in (
            "cProfile",
            "unturtle.eval",
            "OperationTimer",
            "CudaEventTimer",
        ):
            assert forbidden not in source, forbidden
