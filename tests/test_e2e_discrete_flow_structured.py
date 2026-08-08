"""
Structured DFM baseline: NFE must be load-bearing (#65, Phase B prerequisite).

The first end-to-end DFM test (`test_e2e_discrete_flow.py`) trains on a
position-independent target, where one denoising call suffices in principle
and the measured quality-vs-NFE curve is flat.  That validates the *pipeline*
but cannot serve as a few-step control: an FS-DFM-style method would have no
degradation to recover, so "Phase B improves low-NFE quality" would be
unfalsifiable against it.

This baseline uses a task where the clean target is inferable **only from
other positions**: sequences follow ``x_i = (s + i) mod V`` for a random start
``s``.  Every position's marginal is uniform over the data vocabulary, so a
single denoising call from the all-mask state — which samples positions
independently from their marginals — cannot do better than chance on the
chain rule.  Consistency requires *committing* tokens across rounds so later
denoiser calls are conditioned on earlier commitments; that is exactly the
mechanism whose step count NFE measures.

Measured control curve (frozen; CUDA seed 0, the Phase B reference —
regenerate with ``benchmarks/dfm_structured_baseline.py``):

    steps            1      2      4      8     16     64
    adjacent-ok   .124   .443   .706   .851   .944   .978
    full-seq ok   .000   .000   .070   .316   .660   .855

The 1-step value sits at the independent-sampling floor (``1/V = 0.125``;
the terminal draw can also emit the mask token, which nudges it slightly
below an exact ``1/V``), and the curve is strongly monotone — the property
this file pins.  Across 8 training seeds with the fully seeded pipeline:
1-step 0.125–0.130, 4-step 0.617–0.705, 64-step 0.953–0.977, gap
0.823–0.850; thresholds below sit well outside those ranges.  The scores are
*not* bit-stable across thread counts (measured 0.962 vs 0.978 at
OMP_NUM_THREADS=1 vs 8), which the margins absorb.

Every random draw — data, noising, sampling — goes through seeded local
generators.  The first draft passed a generator only to the corpus and let
the process consume the global RNG; a single unrelated global draw then
shifted the 64-step score by 0.01, and under that contamination one training
seed dipped to 0.816.  Self-contained RNG removed the outlier entirely.

A trained model is built once per module (scope="module" fixture): training
takes ~10 s on CPU and the assertions here are all reads of the same model.
"""

import pytest
import torch

from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss
from unturtle.models.generation.dfm_solver import solve_discrete_flow
from unturtle.processes.discrete_flow import DiscreteFlowProcess, LinearKappa

DATA_VOCAB = 8
MASK_ID = DATA_VOCAB  # vocab_size = DATA_VOCAB + 1
LENGTH = 16


def _corpus(n, generator):
    start = torch.randint(0, DATA_VOCAB, (n, 1), generator=generator)
    return (start + torch.arange(LENGTH)) % DATA_VOCAB


def _adjacent_consistency(samples):
    return float(((samples[:, 1:] - samples[:, :-1]) % DATA_VOCAB == 1).float().mean())


@pytest.fixture(scope="module")
def trained_model():
    from unturtle.models.conversion.a2d.tiny_a2d.modeling_llama import (
        TinyA2DLlamaConfig,
        TinyA2DLlamaLMHeadModel,
    )

    config = TinyA2DLlamaConfig(
        vocab_size=DATA_VOCAB + 1,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=LENGTH,
    )
    torch.manual_seed(0)
    model = TinyA2DLlamaLMHeadModel(config).train()
    process = DiscreteFlowProcess(
        vocab_size=DATA_VOCAB + 1, mask_token_id=MASK_ID, source="mask"
    )
    scheduler = LinearKappa()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(0)

    for _ in range(1200):
        clean = _corpus(64, generator)
        # The process draws through the same seeded generator -- nothing in
        # this pipeline rides on the global RNG, so an unrelated import or
        # model-init change cannot shift the trained trajectory.
        out = process({"input_ids": clean}, generator=generator)
        logits = model(input_ids=out.model_inputs["input_ids"]).logits
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

    return model.eval()


def _sample(model, *, steps, rows=256, seed=0):
    def denoise(x_t, t, h):
        with torch.no_grad():
            return model(input_ids=x_t).logits

    generator = torch.Generator().manual_seed(seed)
    x_0 = torch.full((rows, LENGTH), MASK_ID, dtype=torch.long)
    return solve_discrete_flow(denoise, x_0, steps=steps, generator=generator)


@pytest.mark.slow
class TestNFEIsLoadBearing:
    """The property the Phase B control exists to have."""

    def test_the_model_learned_the_task_at_high_nfe(self, trained_model):
        """Ceiling first: without this, a low 1-step score means nothing.

        A model that never learned the chain scores ~1/V at every budget, and
        the "degradation" below would be vacuous.
        """
        consistency = _adjacent_consistency(_sample(trained_model, steps=64))

        assert consistency > 0.85, (
            f"64-step adjacent consistency is {consistency:.3f}; the model did "
            "not learn the chain, so no NFE comparison on it is meaningful"
        )

    def test_one_step_sampling_sits_at_the_independence_floor(self, trained_model):
        """A single call samples positions independently from their marginals.

        Every position's marginal is uniform over the data vocabulary by
        construction, so one-step consistency cannot beat ~1/V no matter how
        well the model fits — the failure is structural, not a fitting gap.
        This is what makes the task a real few-step control: the low-NFE
        deficit is information the sampler must recover across rounds.
        """
        consistency = _adjacent_consistency(_sample(trained_model, steps=1))

        assert consistency < 0.25, (
            f"1-step consistency is {consistency:.3f}, well above the "
            f"independence floor of {1 / DATA_VOCAB:.3f}; either the task "
            "leaks positional information or the sampler is not sampling"
        )

    def test_quality_rises_strongly_and_monotonically_with_nfe(self, trained_model):
        """The control curve itself: the gap Phase B must close.

        Ordinary DFM recovers the chain only by spending steps.  The 1->64
        gap is the headroom a step-aware objective is measured against;
        thresholds sit outside the 4-seed ranges (gap 0.799-0.867,
        4-step 0.640-0.706) so a real regression fails and seed luck does not.
        """
        curve = {
            steps: _adjacent_consistency(_sample(trained_model, steps=steps))
            for steps in (1, 4, 64)
        }

        # `<=` at the top on purpose: a model good enough to saturate by 4
        # steps ties 64 rather than losing to it, and a successful Phase B
        # method is *supposed* to close that gap.  Strict `<` there would
        # fail precisely when few-step training succeeds.
        assert curve[1] < curve[4] <= curve[64], f"curve is not monotone: {curve}"
        assert curve[4] > 0.5, (
            f"4-step consistency {curve[4]:.3f}; intermediate budgets should "
            "already recover much of the chain"
        )
        assert curve[64] - curve[1] > 0.5, (
            f"NFE gap is only {curve[64] - curve[1]:.3f} "
            f"(curve: { {k: round(v, 3) for k, v in curve.items()} }); "
            "the task is not making step count load-bearing"
        )

    def test_the_samples_are_not_mode_collapsed(self, trained_model):
        """Collapse must fail deliberately, not incidentally.

        A model that always emits one memorized chain scores ~1.0 consistency
        at every budget.  The 1-step floor happens to catch that (1.0 > 0.25),
        but only as a side effect of collapse *raising* a score the floor
        expects low — nothing above measures diversity itself.  Measured
        healthy runs produce 43-75 distinct rows out of 256 at 64 steps.
        """
        samples = _sample(trained_model, steps=64)

        distinct = len({tuple(row.tolist()) for row in samples})

        assert distinct > 5, (
            f"only {distinct} distinct sequences in 256 samples; the sampler "
            "has collapsed onto a handful of memorized chains"
        )
