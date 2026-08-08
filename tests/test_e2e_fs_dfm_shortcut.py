"""
FS-DFM shortcut fine-tuning lifts few-step quality (#65 Phase B, end to end).

The Phase B claim, run on the structured chain task from
``test_e2e_discrete_flow_structured.py``: fine-tuning a pretrained DFM model
with the step-aware objective (path loss below tau, RK-4 distillation above,
EMA teacher) must improve its own few-step sampling.

**Two claims, two homes.**  This test pins the *self-paired* claim — the same
model, before vs after shortcut fine-tuning, deterministic and seeded — which
is what a CPU-sized run can referee.  The stronger cross-arm claim lives in
``benchmarks/fs_dfm_shortcut.py`` with two frozen CUDA control arms —
step-matched (2700 steps) AND FLOP-matched (4414 steps; the distillation
branch's RK-4 teacher makes equal steps ~1.64x cheaper for the control) —
and the shortcut wins all 12 paired seed x budget comparisons against both
(FLOP-matched means: 2-step +0.064, 4-step +0.079, 8-step +0.053).

**What no method can lift here:** 1-step quality.  At one call the terminal
draw samples positions independently from their marginals, which are uniform
by construction of the task, so ~1/V adjacent consistency is the *correct*
1-step ceiling — beating it requires distorting the marginals.  The paper's
1-step claims live on real text, where marginals are far from uniform.  The
measurable win on this control is at 2-8 steps, and that is what is asserted.

Self-paired CPU measurements (seeds 0/1, 900 + 900 steps): 4-step
0.428 -> 0.719 and 0.607 -> 0.773 (deltas +0.292 / +0.166); 2-step
0.135 -> 0.446 and 0.129 -> 0.480.  Thresholds sit under those margins.
Note the pre-fine-tune model samples with out-of-grid ``h`` conditioning
(it was pretrained at ``h = 2^-6`` only), so part of the pre-vs-post gap is
the conditioning becoming meaningful at all — the matched-compute benchmark
is the arm that isolates the objective itself.
"""

import copy

import pytest
import torch

from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss
from unturtle.diffusion.fs_dfm import (
    StepAwareWrapper,
    clip_step_to_path,
    few_step_distillation_loss,
    rk_teacher_logits,
)
from unturtle.models.generation.dfm_solver import solve_discrete_flow
from unturtle.processes.discrete_flow import DiscreteFlowProcess, LinearKappa

DATA_VOCAB = 8
MASK_ID = DATA_VOCAB
LENGTH = 16
H_PRETRAIN = 2.0**-6
TAU = 2.0**-5
GRID = [2.0**k for k in range(-6, 1)]


def _corpus(n, generator):
    start = torch.randint(0, DATA_VOCAB, (n, 1), generator=generator)
    return (start + torch.arange(LENGTH)) % DATA_VOCAB


def _adjacent_consistency(samples):
    return float(((samples[:, 1:] - samples[:, :-1]) % DATA_VOCAB == 1).float().mean())


def _sample(model, *, steps, seed=0):
    def denoise(x_t, t, h):
        with torch.no_grad():
            return model(x_t, t, h)

    generator = torch.Generator().manual_seed(seed)
    x_0 = torch.full((256, LENGTH), MASK_ID, dtype=torch.long)
    return solve_discrete_flow(denoise, x_0, steps=steps, generator=generator)


@pytest.fixture(scope="module")
def before_and_after():
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
    model = StepAwareWrapper(TinyA2DLlamaLMHeadModel(config)).train()
    process = DiscreteFlowProcess(
        vocab_size=DATA_VOCAB + 1, mask_token_id=MASK_ID, source="mask"
    )
    scheduler = LinearKappa()
    generator = torch.Generator().manual_seed(0)

    # Stage 1: plain DFM pretraining (h conditioning pinned to one small h).
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(900):
        clean = _corpus(64, generator)
        out = process({"input_ids": clean}, generator=generator)
        x_t = out.model_inputs["input_ids"]
        timesteps = out.objective_inputs["timesteps"]
        loss = discrete_flow_matching_loss(
            model(x_t, timesteps, H_PRETRAIN),
            clean,
            x_t,
            timesteps,
            scheduler=scheduler,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    pretrained = copy.deepcopy(model).eval()

    # Stage 2: shortcut fine-tune — eq. (4.5) branch per sampled h, RK-4
    # teacher over EMA weights (eq. 4.1), everything through seeded local
    # generators per the research-benchmark reproducibility rule.
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    beta = 0.99
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    for _ in range(900):
        clean = _corpus(64, generator)
        out = process({"input_ids": clean}, generator=generator)
        x_t = out.model_inputs["input_ids"]
        timesteps = out.objective_inputs["timesteps"]
        h = GRID[int(torch.randint(0, len(GRID), (1,), generator=generator))]
        scaled_t, h_eff = clip_step_to_path(timesteps, h)

        if h < TAU:
            loss = discrete_flow_matching_loss(
                model(x_t, scaled_t, h),
                clean,
                x_t,
                scaled_t,
                scheduler=scheduler,
                step_size=h_eff,
            )
        else:
            teacher = rk_teacher_logits(
                lambda x, t, hh: ema(x, t, float(hh)),
                x_t,
                scaled_t,
                h_eff,
                generator=generator,
            )
            loss = few_step_distillation_loss(model(x_t, scaled_t, h), teacher)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for ema_param, param in zip(
                ema.parameters(), model.parameters(), strict=True
            ):
                ema_param.mul_(beta).add_(param, alpha=1 - beta)

    return pretrained, model.eval()


@pytest.mark.slow
class TestShortcutFineTuningLiftsFewStepQuality:
    def test_four_step_sampling_improves(self, before_and_after):
        """The headline: the same model, better at 4 steps after fine-tuning.

        Measured deltas +0.292 / +0.166 over two seeds; the threshold sits
        well under both.
        """
        pretrained, tuned = before_and_after

        before = _adjacent_consistency(_sample(pretrained, steps=4))
        after = _adjacent_consistency(_sample(tuned, steps=4))

        assert after - before > 0.08, (
            f"4-step consistency moved {before:.3f} -> {after:.3f} "
            f"(delta {after - before:+.3f}); shortcut fine-tuning did not "
            "teach the model to use its step budget"
        )

    def test_two_step_sampling_improves(self, before_and_after):
        """Two steps is the hardest budget any method can influence here
        (1-step is capped at the independence floor by the task itself)."""
        pretrained, tuned = before_and_after

        before = _adjacent_consistency(_sample(pretrained, steps=2))
        after = _adjacent_consistency(_sample(tuned, steps=2))

        assert after - before > 0.15, (
            f"2-step consistency moved {before:.3f} -> {after:.3f}"
        )

    def test_the_tuned_model_is_absolutely_usable_at_four_steps(self, before_and_after):
        """A relative lift over a broken baseline would be vacuous."""
        _, tuned = before_and_after

        assert _adjacent_consistency(_sample(tuned, steps=4)) > 0.65

    def test_many_step_quality_is_not_sacrificed(self, before_and_after):
        """The failure mode of aggressive distillation: better at 4 steps,
        broken at 64.  The tuned model must stay strong at large budgets."""
        _, tuned = before_and_after

        assert _adjacent_consistency(_sample(tuned, steps=64)) > 0.9
