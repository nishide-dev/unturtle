"""
End-to-end discrete flow matching (#65 Phase A).

The three Phase A components landed separately — the process (#86), the
objective (#94), and the solver (#95) — each with its own unit tests.  Nothing
until now ran them *together*, and the interesting failures live exactly in the
seams: the objective reading a different ``t`` convention than the process
samples, the solver assuming a source the process never produces, a sign that
is self-consistent within one module and inverted relative to the next.

So this file trains a real (tiny) model with the real objective on the real
process and samples it with the real solver, then asserts the only thing that
matters end to end: **the sampler reproduces the distribution the data came
from**.

That property is what a per-module test cannot reach.  The objective in this
same issue shipped with an inverted sign that thirteen unit tests missed, and
the solver shipped forbidding the model's own argmax while all twenty of its
tests passed.  Both would fail here in one line, because a model trained
backwards or sampled with a biased terminal step cannot reproduce its training
distribution.

Kept CPU-only and small enough for the default suite; nothing here needs a GPU
or a real checkpoint.
"""

import math

import pytest
import torch
import torch.nn as nn

from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss
from unturtle.models.generation.dfm_solver import solve_discrete_flow
from unturtle.processes.discrete_flow import DiscreteFlowProcess, LinearKappa

VOCAB = 6
LENGTH = 8
MASK_ID = VOCAB - 1


class TinyDenoiser(nn.Module):
    """A minimal time-conditioned token denoiser.

    Deliberately not a transformer: Phase A needs to show the *pipeline*
    composes, and a per-position MLP with a time embedding is enough to learn
    a position-independent target distribution while keeping the test fast.
    Attention would add capacity this assertion does not use.
    """

    def __init__(self, vocab: int = VOCAB, width: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.time = nn.Linear(1, width)
        self.body = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, vocab),
        )

    def forward(self, input_ids: torch.Tensor, timesteps: torch.Tensor):
        hidden = self.embed(input_ids)
        t = timesteps.reshape(-1, 1, 1).to(hidden.dtype).expand(-1, hidden.shape[1], 1)
        return self.body(hidden + self.time(t))


def _target_distribution() -> torch.Tensor:
    """A non-uniform distribution over the non-mask tokens.

    Non-uniform on purpose: a uniform target is reproduced by a broken sampler
    that merely emits noise, so it could not distinguish success from failure.
    The mask id carries zero mass — it is the source token, never a target.
    """
    weights = torch.tensor([0.40, 0.25, 0.20, 0.10, 0.05, 0.00])
    assert len(weights) == VOCAB
    assert float(weights[MASK_ID]) == 0.0
    return weights


def _sample_corpus(n: int, generator: torch.Generator) -> torch.Tensor:
    probs = _target_distribution().expand(n * LENGTH, VOCAB)
    drawn = torch.multinomial(probs, num_samples=1, generator=generator)
    return drawn.reshape(n, LENGTH)


def _train(steps: int = 400, seed: int = 0) -> TinyDenoiser:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    model = TinyDenoiser()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    process = DiscreteFlowProcess(
        vocab_size=VOCAB, mask_token_id=MASK_ID, source="mask"
    )
    scheduler = LinearKappa()

    for _ in range(steps):
        clean = _sample_corpus(64, generator)
        out = process({"input_ids": clean}, generator=generator)

        logits = model(out.model_inputs["input_ids"], out.objective_inputs["timesteps"])
        loss = discrete_flow_matching_loss(
            logits,
            clean,
            out.model_inputs["input_ids"],
            out.objective_inputs["timesteps"],
            scheduler=scheduler,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def _sample(model: TinyDenoiser, *, steps: int, rows: int = 256, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    x_0 = torch.full((rows, LENGTH), MASK_ID, dtype=torch.long)

    def denoise(x_t, t, h):
        with torch.no_grad():
            return model(x_t, t)

    return solve_discrete_flow(denoise, x_0, steps=steps, generator=generator)


def _total_variation(samples: torch.Tensor) -> float:
    empirical = torch.bincount(samples.reshape(-1), minlength=VOCAB).float()
    empirical /= empirical.sum()
    return 0.5 * float((empirical - _target_distribution()).abs().sum())


@pytest.mark.slow
class TestDiscreteFlowEndToEnd:
    """Process -> objective -> solver, composed."""

    def test_a_trained_model_reproduces_its_training_distribution(self):
        """The single assertion Phase A exists to support.

        A model trained with the DFM objective on the DFM process, then sampled
        with the DFM solver, must emit tokens distributed like its training
        corpus.  An inverted objective drives the model away from the data; a
        biased terminal draw distorts the marginals; a source mismatch leaves
        mask tokens in the output.  All three land on this number.

        The 0.10 bound sits above a measured worst case of 0.083 across 5
        training seeds x 4 sampling seeds — a 1.2x margin, not a generous one.
        Everything here is seeded, so CI cannot flake; but anyone retuning the
        model, the corpus size, or the training budget should re-measure that
        spread rather than trusting the headroom. The bound is still far below
        the failures it guards against: the objective bug in this issue
        produced 0.31 and the solver bug 0.43.
        """
        model = _train()
        samples = _sample(model, steps=16)

        distance = _total_variation(samples)

        assert distance < 0.10, (
            f"sampled marginals sit {distance:.4f} in total variation from the "
            f"training distribution {_target_distribution().tolist()}"
        )

    def test_the_source_token_does_not_survive_sampling(self):
        """`[MASK]` is the source, never a target: it must be gone at `t = 1`.

        A separate failure from a distribution mismatch — a solver that leaves
        source tokens in place still produces plausible marginals over the
        remaining vocabulary, so the previous test alone would not catch it.
        """
        model = _train()
        samples = _sample(model, steps=16)

        share_of_mask = float((samples == MASK_ID).float().mean())

        assert share_of_mask < 0.02, (
            f"{share_of_mask:.4f} of sampled positions still hold the source "
            "token; the process's source is leaking through the solver"
        )

    def test_quality_does_not_collapse_at_small_step_budgets(self):
        """Phase A's baseline claim, and the reference Phase B improves on.

        FS-DFM exists because few-step sampling of an ordinary DFM model
        degrades.  Phase A only has to show the degradation is *bounded* — that
        the baseline is a usable reference rather than noise at low NFE.  How
        much Phase B's step-aware objective recovers is measured against these
        numbers, so they are recorded here rather than merely bounded:

            steps    1      2      4      8     16     32     64    128
            TV     .056   .077   .064   .040   .057   .058   .069   .062
            mask   .032   .011   .004   .003   .001   .000   .000   .000

        The ~0.05 floor is this tiny model's fit, not sampler bias: the model's
        own prediction on an all-mask input sits 0.076 from the target at 400
        training steps and 0.035 at 1500, tracking the sampled numbers.  The
        sampler reproduces whatever the model learned.

        Note this baseline is flat rather than degrading at low NFE, because
        the target here is position-independent — one step suffices in
        principle.  A real backbone with inter-token structure is where the
        few-step gap Phase B closes actually appears; this test guards the
        pipeline, not the difficulty of the task.
        """
        model = _train()

        measured = {
            steps: _total_variation(_sample(model, steps=steps))
            for steps in (1, 2, 4, 16, 64)
        }

        for steps, distance in measured.items():
            assert distance < 0.20, (
                f"at {steps} steps the total variation is {distance:.4f}; the "
                f"baseline degrades past usefulness (full curve: "
                f"{ {k: round(v, 4) for k, v in measured.items()} })"
            )

        assert all(math.isfinite(v) for v in measured.values())
