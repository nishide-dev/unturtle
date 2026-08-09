"""
Continuous flow-matching forward process (#66, FlowLM prototype).

Third implementation of the ``ForwardProcess`` contract (RFC
``docs/rfcs/continuous-latent.md``): the state is a continuous latent, the
supervision is a target tensor.  The FlowLM specifics pinned here come from
the paper's Algorithm 1 verbatim (arXiv:2605.20199), not its abstract:

- linear interpolation path ``z_t = (1 - t) * z_0 + t * eps`` with
  ``eps ~ N(0, I)`` — **t = 1 is noise, t = 0 is data** (the reverse of the
  masked convention, where corruption grows with t);
- time sampled uniformly from a **discrete grid** ``t_step/T`` with
  ``t_step ~ Uniform({1, .., T})`` and T small (paper: 20, cut down from
  2000 — aligning the training grid with the few-step sampling target is
  reported as a significant win, and loss-aware sampling as a loss);
- the model-facing time is rescaled by ``time_scale`` (paper: x1000, to
  preserve DiffuSeq's pretrained [0, 1000] conditioning range) while the
  objective keeps the unscaled ``t`` (the 1/t^2 regularizer and the solver
  both need it).

The collator never sees any of this: the process consumes ``latents`` the
caller already encoded (acceptance criterion — no continuous tensors in
``MaskedDiffusionDataCollator``).
"""

import pytest
import torch


def _batch(rows=4, length=6, hidden=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return {
        "latents": torch.randn(rows, length, hidden, generator=generator),
        "attention_mask": torch.ones(rows, length, dtype=torch.long),
    }


class TestThePathMatchesAlgorithmOne:
    def test_the_state_is_the_linear_interpolation_of_data_and_noise(self):
        """`z_t = (1-t) z_0 + t eps`, reconstructed exactly from the returned
        noise and timesteps — the one equation everything else builds on."""
        from unturtle.processes.continuous_flow import ContinuousFlowProcess

        process = ContinuousFlowProcess(num_timesteps=20)
        batch = _batch()

        out = process(batch, generator=torch.Generator().manual_seed(1))

        z_t = out.model_inputs["latents"]
        z_0 = out.objective_inputs["target_latents"]
        noise = out.objective_inputs["noise"]
        t = out.objective_inputs["timesteps"].view(-1, 1, 1)
        assert torch.allclose(z_t, (1 - t) * z_0 + t * noise, atol=1e-6)
        assert torch.equal(z_0, batch["latents"])

    def test_time_lives_on_the_discrete_grid_and_excludes_zero(self):
        """`t_step ~ Uniform({1..T})`: every t is k/T for integer k in [1, T].
        t = 0 never occurs (the model is never asked to denoise clean data,
        and the sampler's average velocity divides by t)."""
        from unturtle.processes.continuous_flow import ContinuousFlowProcess

        process = ContinuousFlowProcess(num_timesteps=5)
        generator = torch.Generator().manual_seed(2)

        seen = set()
        for _ in range(200):
            out = process(_batch(rows=8), generator=generator)
            t = out.objective_inputs["timesteps"]
            steps = t * 5
            assert torch.allclose(steps, steps.round(), atol=1e-5)
            assert bool((steps >= 1).all()) and bool((steps <= 5).all())
            seen.update(int(s) for s in steps.round().tolist())
        assert seen == {1, 2, 3, 4, 5}, f"grid not fully sampled: {seen}"

    def test_the_model_facing_time_is_rescaled(self):
        """Paper §3.3: `t_input = t * 1000` preserves the pretrained
        conditioning range; the objective keeps the unscaled t."""
        from unturtle.processes.continuous_flow import ContinuousFlowProcess

        process = ContinuousFlowProcess(num_timesteps=4, time_scale=1000.0)

        out = process(_batch(), generator=torch.Generator().manual_seed(3))

        assert torch.allclose(
            out.model_inputs["timesteps"],
            out.objective_inputs["timesteps"] * 1000.0,
        )

    def test_the_input_batch_is_not_mutated(self):
        """The ForwardProcess mutability contract, verbatim."""
        from unturtle.processes.continuous_flow import ContinuousFlowProcess

        batch = _batch()
        reference = {k: v.clone() for k, v in batch.items()}

        ContinuousFlowProcess()(batch, generator=torch.Generator().manual_seed(4))

        for key, tensor in reference.items():
            assert torch.equal(batch[key], tensor), f"{key} was mutated"

    def test_passthrough_fields_ride_on_model_inputs(self):
        from unturtle.processes.continuous_flow import ContinuousFlowProcess

        batch = _batch()

        out = ContinuousFlowProcess()(batch, generator=torch.Generator().manual_seed(5))

        assert torch.equal(out.model_inputs["attention_mask"], batch["attention_mask"])

    def test_a_seeded_generator_makes_the_process_reproducible(self):
        """The research-benchmark reproducibility rule, at the source."""
        from unturtle.processes.continuous_flow import ContinuousFlowProcess

        process = ContinuousFlowProcess()
        batch = _batch()

        first = process(batch, generator=torch.Generator().manual_seed(6))
        second = process(batch, generator=torch.Generator().manual_seed(6))

        assert torch.equal(
            first.model_inputs["latents"], second.model_inputs["latents"]
        )
        assert torch.equal(
            first.objective_inputs["timesteps"], second.objective_inputs["timesteps"]
        )


class TestItRejectsWhatItCannotNoise:
    def test_a_batch_without_latents_is_rejected(self):
        from unturtle.processes.continuous_flow import ContinuousFlowProcess

        with pytest.raises(ValueError, match="latents"):
            ContinuousFlowProcess()({"input_ids": torch.ones(2, 4, dtype=torch.long)})

    def test_integer_latents_are_rejected(self):
        """Passing token ids where latents belong is THE likely mistake, and
        Gaussian noise on integer ids is finite, plausible and wrong."""
        from unturtle.processes.continuous_flow import ContinuousFlowProcess

        with pytest.raises(ValueError, match="floating"):
            ContinuousFlowProcess()({"latents": torch.ones(2, 4, 8, dtype=torch.long)})

    @pytest.mark.parametrize("steps", [0, -3])
    def test_a_non_positive_grid_is_rejected(self, steps):
        from unturtle.processes.continuous_flow import ContinuousFlowProcess

        with pytest.raises(ValueError, match="num_timesteps"):
            ContinuousFlowProcess(num_timesteps=steps)
