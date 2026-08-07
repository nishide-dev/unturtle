"""
Discrete flow-matching forward process (#65, FS-DFM).

Implements the conditional probability path of arXiv:2509.20624 §3::

    p_t(x^i | x_0, x_1) = (1 - kappa_t) * delta_{x_0}(x^i) + kappa_t * delta_{x_1}(x^i)

Each position independently holds either its source token (`x_0`) or its target
token (`x_1`), mixed by `kappa_t`.  The paper uses the linear scheduler
`kappa(t) = t`.

Reimplemented from the paper.  The official Apple repository is under the Apple
Sample Code License and was deliberately not read or ported.

The property that separates this from masked diffusion, and that most of these
tests exist to pin: **a uniform-source flow is not absorbing.**  Masked
diffusion's corruption is one-way, so `diffusion_mask` fully describes the
state.  Here a position can hold a plausible-but-wrong real token that is
indistinguishable from a correct one, so the process must emit the source state
itself.
"""

import pytest
import torch

MASK_ID = 99
VOCAB = 50


def _process(source="mask", **kwargs):
    from unturtle.processes import DiscreteFlowProcess

    return DiscreteFlowProcess(
        vocab_size=VOCAB,
        mask_token_id=MASK_ID,
        source=source,
        **kwargs,
    )


def _batch(batch_size=2, seq_len=8):
    torch.manual_seed(0)
    return {
        "input_ids": torch.randint(1, VOCAB, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }


class TestConditionalPath:
    def test_every_position_is_either_source_or_target(self):
        """`p_t` mixes two point masses, so no third value can appear."""
        batch = _batch()
        out = _process()(batch)

        x_t = out.model_inputs["input_ids"]
        x_0 = out.objective_inputs["source_ids"]
        x_1 = out.objective_inputs["labels"]

        is_source = x_t == x_0
        is_target = x_t == x_1
        assert bool((is_source | is_target).all()), (
            "a position held neither its source nor its target token"
        )

    def test_t_near_zero_is_almost_all_source(self):
        """`kappa(0) = 0`, so the state starts at the source distribution."""
        batch = _batch(batch_size=8, seq_len=64)
        out = _process()(batch, timesteps=torch.full((8,), 0.01))

        x_t = out.model_inputs["input_ids"]
        x_0 = out.objective_inputs["source_ids"]

        fraction_source = (x_t == x_0).float().mean().item()
        assert fraction_source > 0.9, f"only {fraction_source:.2f} held the source"

    def test_t_near_one_is_almost_all_target(self):
        """`kappa(1) = 1`, so the state ends at the data distribution."""
        batch = _batch(batch_size=8, seq_len=64)
        out = _process()(batch, timesteps=torch.full((8,), 0.99))

        x_t = out.model_inputs["input_ids"]
        x_1 = out.objective_inputs["labels"]

        fraction_target = (x_t == x_1).float().mean().item()
        assert fraction_target > 0.9, f"only {fraction_target:.2f} held the target"

    def test_mixing_rate_tracks_kappa(self):
        """The target fraction should track `kappa(t) = t`, not merely increase.

        A process that interpolated on some other monotone curve would pass a
        "more target at higher t" check while training against a different
        path than the objective assumes.
        """
        batch = _batch(batch_size=16, seq_len=128)

        for t in (0.25, 0.5, 0.75):
            out = _process()(batch, timesteps=torch.full((16,), t))
            x_t = out.model_inputs["input_ids"]
            x_1 = out.objective_inputs["labels"]
            observed = (x_t == x_1).float().mean().item()
            assert abs(observed - t) < 0.05, (
                f"kappa(t)={t} but the observed target fraction was {observed:.3f}"
            )

    def test_positions_are_interpolated_independently(self):
        """The path factorizes over positions; it is not a whole-sequence coin.

        A per-sequence implementation would make every position agree, which
        at t=0.5 still gives the right *average* over a batch.
        """
        batch = _batch(batch_size=4, seq_len=64)
        out = _process()(batch, timesteps=torch.full((4,), 0.5))

        x_t = out.model_inputs["input_ids"]
        x_1 = out.objective_inputs["labels"]
        per_row = (x_t == x_1).float().mean(dim=-1)

        assert bool(((per_row > 0.05) & (per_row < 0.95)).all()), (
            f"rows were all-or-nothing ({per_row.tolist()}), so positions are "
            "not being drawn independently"
        )


class TestSourceDistributions:
    def test_mask_source_uses_the_mask_token(self):
        batch = _batch(batch_size=4, seq_len=32)
        out = _process(source="mask")(batch, timesteps=torch.full((4,), 0.5))

        assert bool((out.objective_inputs["source_ids"] == MASK_ID).all()), (
            "a mask source must be the mask token at every position"
        )

    def test_uniform_source_draws_real_tokens(self):
        """The paper's released checkpoints use the uniform source."""
        batch = _batch(batch_size=8, seq_len=64)
        out = _process(source="uniform")(batch, timesteps=torch.full((8,), 0.5))

        source = out.objective_inputs["source_ids"]
        assert not bool((source == MASK_ID).all()), "uniform source collapsed to mask"
        assert int(source.unique().numel()) > 10, (
            f"uniform source drew only {source.unique().numel()} distinct tokens"
        )
        assert bool(((source >= 0) & (source < VOCAB)).all()), "source out of vocab"

    def test_uniform_source_is_not_absorbing(self):
        """The property that forces `source_ids` to exist at all.

        Under a uniform source a corrupted position holds a real token, so it
        is indistinguishable from an uncorrupted one by inspection — unlike
        masked diffusion, where the mask token marks corruption.  Without the
        source state the objective cannot tell which positions moved.
        """
        batch = _batch(batch_size=8, seq_len=64)
        out = _process(source="uniform")(batch, timesteps=torch.full((8,), 0.5))

        x_t = out.model_inputs["input_ids"]
        assert not bool((x_t == MASK_ID).any()), (
            "a uniform-source state should contain no mask tokens at all"
        )
        assert "source_ids" in out.objective_inputs, (
            "the objective cannot recover which positions were corrupted "
            "without the source state"
        )

    def test_uniform_source_may_coincide_with_the_target(self):
        """The source is plain uniform — it does NOT exclude the target token.

        With probability ~1/V a "corrupted" position draws exactly its own
        clean token, so `x_t == x_0 == x_1`.  That is the paper's forward
        process: §B.3 describes "a uniform source over tokens", full stop.
        Exclusion appears only in the *sampler*, where a jump resamples "from
        the off-diagonals ... renormalized to exclude the current token"
        (Appendix B.1) — a different mechanism at a different stage.

        Pinned because de-duplicating the source looks like an obvious
        improvement and would change the process the objective trains against.
        """
        from unturtle.processes import DiscreteFlowProcess

        small_vocab = 4
        process = DiscreteFlowProcess(
            vocab_size=small_vocab, mask_token_id=MASK_ID, source="uniform"
        )
        batch = {
            "input_ids": torch.randint(0, small_vocab, (16, 64)),
            "attention_mask": torch.ones(16, 64, dtype=torch.long),
        }

        out = process(batch, timesteps=torch.zeros(16))  # t=0: pure source

        collided = out.objective_inputs["source_ids"] == out.objective_inputs["labels"]
        assert bool(collided.any()), (
            "with V=4 over 1024 positions the source should coincide with the "
            "target somewhere; if it never does, the source is excluding the "
            "target and no longer matches the paper"
        )

    def test_unknown_source_is_rejected(self):
        with pytest.raises(ValueError, match="source"):
            _process(source="gaussian")

    def test_mask_source_requires_a_mask_token(self):
        from unturtle.processes import DiscreteFlowProcess

        with pytest.raises(ValueError, match="mask_token_id"):
            DiscreteFlowProcess(vocab_size=VOCAB, mask_token_id=None, source="mask")


class TestScheduler:
    def test_linear_kappa_is_the_default(self):
        from unturtle.processes import DiscreteFlowProcess

        process = DiscreteFlowProcess(vocab_size=VOCAB, mask_token_id=MASK_ID)
        t = torch.tensor([0.0, 0.25, 0.5, 1.0])

        assert torch.allclose(process.kappa(t), t, atol=1e-6), (
            "the paper uses kappa(t) = t; a different default would silently "
            "train against another path"
        )

    def test_a_custom_kappa_is_honored(self):
        """Scheduler choice is open in DFM; the paper just picks linear."""
        batch = _batch(batch_size=16, seq_len=128)

        class Quadratic:
            def kappa(self, t):
                return t**2

        out = _process(scheduler=Quadratic())(batch, timesteps=torch.full((16,), 0.5))
        x_t = out.model_inputs["input_ids"]
        x_1 = out.objective_inputs["labels"]

        observed = (x_t == x_1).float().mean().item()
        assert abs(observed - 0.25) < 0.05, (
            f"kappa(0.5)=0.25 for t^2, but observed {observed:.3f}"
        )


class TestProcessContract:
    def test_it_is_callable_the_way_the_protocol_specifies(self):
        """`ForwardProcess` is not `@runtime_checkable`, so `isinstance` raises.

        Checked structurally instead, and more than the existing
        "assert callable" convention: the protocol fixes `(batch, *,
        generator)`, so a process must accept exactly that call. Extra
        keyword-only parameters (this one adds `timesteps`) are a compatible
        superset; a missing `generator`, or `batch` being keyword-only, is not.
        """
        import inspect

        from unturtle.processes import ForwardProcess

        signature = inspect.signature(_process().__call__)
        protocol = inspect.signature(ForwardProcess.__call__)

        for name, expected in protocol.parameters.items():
            if name == "self":
                continue
            assert name in signature.parameters, f"missing parameter {name!r}"
            assert signature.parameters[name].kind == expected.kind, (
                f"{name!r} is {signature.parameters[name].kind}, "
                f"protocol requires {expected.kind}"
            )

        # And it actually runs under that exact call shape.
        out = _process()(_batch(), generator=torch.Generator().manual_seed(0))
        assert out.model_inputs and out.objective_inputs

    def test_the_input_batch_is_not_mutated(self):
        batch = _batch()
        before = batch["input_ids"].clone()

        _process()(batch)

        assert torch.equal(batch["input_ids"], before), (
            "the process mutated its caller's batch"
        )

    def test_same_generator_seed_reproduces_the_state(self):
        batch = _batch()

        first = _process()(batch, generator=torch.Generator().manual_seed(3))
        second = _process()(batch, generator=torch.Generator().manual_seed(3))

        assert torch.equal(
            first.model_inputs["input_ids"], second.model_inputs["input_ids"]
        )
        assert torch.equal(
            first.objective_inputs["source_ids"], second.objective_inputs["source_ids"]
        )

    def test_timesteps_are_sampled_when_not_supplied(self):
        batch = _batch(batch_size=16)

        out = _process()(batch)
        timesteps = out.objective_inputs["timesteps"]

        assert timesteps.shape == (16,)
        assert bool(((timesteps >= 0.0) & (timesteps <= 1.0)).all())
        assert int(timesteps.unique().numel()) > 1, "every row drew the same t"

    def test_explicit_per_position_timesteps_are_honored(self):
        """The `[B, L]` branch, which packed batches take.

        Only reached implicitly via `segment_ids` otherwise, so a broadcasting
        bug there would show up as a subtly wrong path rather than an error.
        Left half at t=0 (all source), right half at t=1 (all target).
        """
        batch = _batch(batch_size=2, seq_len=16)
        t = torch.zeros(2, 16)
        t[:, 8:] = 1.0

        out = _process()(batch, timesteps=t)

        x_t = out.model_inputs["input_ids"]
        x_0 = out.objective_inputs["source_ids"]
        x_1 = out.objective_inputs["labels"]

        assert torch.equal(x_t[:, :8], x_0[:, :8]), "t=0 half should be all source"
        assert torch.equal(x_t[:, 8:], x_1[:, 8:]), "t=1 half should be all target"

    def test_mismatched_timestep_shape_is_rejected(self):
        batch = _batch(batch_size=2, seq_len=8)

        with pytest.raises(ValueError, match="timesteps"):
            _process()(batch, timesteps=torch.zeros(2, 7))

    def test_pass_through_fields_reach_the_model(self):
        batch = _batch()
        batch["position_ids"] = torch.arange(8).expand(2, 8)

        out = _process()(batch)

        assert "position_ids" in out.model_inputs
        assert "attention_mask" in out.model_inputs

    def test_supervision_is_not_forwarded_to_the_model(self):
        """`labels` / `source_ids` are objective inputs, not forward kwargs.

        The batch is seeded with all three keys deliberately.  Passing a batch
        that lacks them makes this vacuous — the filter has nothing to strip,
        and removing it entirely still passes.  Mutation-verified.
        """
        batch = _batch()
        batch["labels"] = batch["input_ids"].clone()
        batch["source_ids"] = torch.zeros_like(batch["input_ids"])
        batch["timesteps"] = torch.zeros(batch["input_ids"].shape[0])

        out = _process()(batch)

        for key in ("labels", "source_ids", "timesteps"):
            assert key not in out.model_inputs, f"{key} leaked into model_inputs"
            assert key in out.objective_inputs, f"{key} missing from objective_inputs"

    def test_supplied_supervision_is_rebuilt_not_passed_through(self):
        """A stale `source_ids` from a previous step must not survive.

        The process owns these fields; carrying a caller's copy forward would
        silently supervise against the wrong source state.
        """
        batch = _batch()
        sentinel = torch.full_like(batch["input_ids"], 7)
        batch["source_ids"] = sentinel

        out = _process(source="mask")(batch)

        assert not torch.equal(out.objective_inputs["source_ids"], sentinel), (
            "the caller's source_ids was passed through instead of resampled"
        )


class TestPadding:
    def test_padding_is_never_corrupted(self):
        batch = {
            "input_ids": torch.randint(1, VOCAB, (1, 8)),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]]),
        }
        clean = batch["input_ids"].clone()

        out = _process()(batch, timesteps=torch.tensor([0.5]))

        x_t = out.model_inputs["input_ids"]
        assert torch.equal(x_t[0, 4:], clean[0, 4:]), (
            "padding positions were corrupted; they carry no supervision and "
            "should pass through untouched"
        )

    def test_padding_is_excluded_from_the_labels(self):
        batch = {
            "input_ids": torch.randint(1, VOCAB, (1, 8)),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]]),
        }

        out = _process()(batch, timesteps=torch.tensor([0.5]))

        labels = out.objective_inputs["labels"]
        assert bool((labels[0, 4:] == -100).all()), "padding must not be supervised"
        assert not bool((labels[0, :4] == -100).any()), "real tokens lost supervision"


class TestPackedSegments:
    def test_each_segment_draws_its_own_timestep(self):
        """Mirrors `MaskedDiffusionProcess`: a packed row holds several samples.

        One `t` per row is the wrong `t` for every sample in it — the exact
        defect #62 removed for masked diffusion.
        """
        batch = {
            "input_ids": torch.randint(1, VOCAB, (1, 12)),
            "attention_mask": torch.ones(1, 12, dtype=torch.long),
            "segment_ids": torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]]),
        }

        out = _process()(batch)
        timesteps = out.objective_inputs["timesteps"]

        assert timesteps.shape == (1, 12), (
            f"expected per-position [B, L] timesteps, got {tuple(timesteps.shape)}"
        )
        assert timesteps[0, 0] == timesteps[0, 3], "a segment must share one t"
        assert int(timesteps[0].unique().numel()) > 1, "segments drew the same t"

    def test_padding_segments_do_not_break_the_gather(self):
        """`segment_ids = -1` marks positions no sample owns.

        A raw gather on -1 raises on CPU and fires an async device-side assert
        on CUDA that poisons the context — the bug #62 hit.
        """
        batch = {
            "input_ids": torch.randint(1, VOCAB, (1, 8)),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]]),
            "segment_ids": torch.tensor([[0, 0, 1, 1, 1, -1, -1, -1]]),
        }

        out = _process()(batch)

        assert torch.isfinite(out.objective_inputs["timesteps"]).all()

    def test_segment_ids_are_not_forwarded_to_the_model(self):
        batch = {
            "input_ids": torch.randint(1, VOCAB, (1, 4)),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "segment_ids": torch.tensor([[0, 0, 1, 1]]),
        }

        out = _process()(batch)

        assert "segment_ids" not in out.model_inputs, (
            "packing topology is not a model input"
        )
