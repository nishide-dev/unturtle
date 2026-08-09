"""
DFM generation as a registry family (#65 public-API slice).

`solve_discrete_flow` (#95/#110) was reachable only by hand-wiring a denoiser
callable; this slice registers it as the `discrete_flow` family the sampler
registry reserved from day one.  Design decisions under test:

- **opt-in capability** (`supports_dfm_generation is True`): the DFM
  sampling quality of real backbones is only tiny-control-validated (#65),
  so no masked family is silently claimed — a model must declare itself;
- **`dfm_denoiser(x_t, t, h)` is the overridable seam**: the default is the
  time-agnostic forward (logits from ids, t/h unused); FS-DFM's step-aware
  models override it.  No `step_aware` flag;
- the public path is **bitwise-identical** to calling the solver directly —
  the API adds routing, never behavior;
- prompts are rejected loudly (DFM conditioning is outside this issue's
  validated scope), as is `source="mask"` without a mask token.
"""

import pytest
import torch

VOCAB = 12
MASK_ID = VOCAB - 1
LENGTH = 6


def _tiny_dfm_model(mask_token_id=MASK_ID):
    from transformers import PretrainedConfig, PreTrainedModel
    from transformers.modeling_outputs import MaskedLMOutput

    from unturtle.models.generation.dfm_mixin import DiscreteFlowGenerationMixin

    class _Config(PretrainedConfig):
        model_type = "tiny-dfm-test"

        def __init__(self, **kwargs):
            self.vocab_size = VOCAB
            self.hidden_size = 16
            self.max_position_embeddings = LENGTH
            super().__init__(mask_token_id=kwargs.pop("mask_token_id", None), **kwargs)

    class _Model(DiscreteFlowGenerationMixin, PreTrainedModel):
        config_class = _Config

        def __init__(self, config):
            super().__init__(config)
            self.embedding = torch.nn.Embedding(VOCAB, 16)
            self.head = torch.nn.Linear(16, VOCAB)
            self.post_init()

        def forward(self, input_ids, **_):
            return MaskedLMOutput(logits=self.head(self.embedding(input_ids)))

    torch.manual_seed(0)
    return _Model(_Config(mask_token_id=mask_token_id)).eval()


class TestTheRegistryEntry:
    def test_dfm_registers_as_the_discrete_flow_family(self):
        from unturtle.models.generation.sampler import find_algorithm

        entry = find_algorithm("dfm")

        assert entry is not None
        assert entry.family == "discrete_flow"
        assert entry.flags == {}

    def test_a_masked_model_cannot_be_asked_for_dfm(self):
        """Opt-in means opt-in: TinyA2D is a masked model that COULD run the
        jump process mechanically, but its DFM quality is unvalidated — it
        must not be silently claimed."""
        from unturtle.models.conversion.a2d.tiny_a2d import (
            TinyA2DLlamaConfig,
            TinyA2DLlamaLMHeadModel,
        )
        from unturtle.models.generation.sampler import resolve_algorithm

        masked = TinyA2DLlamaLMHeadModel(
            TinyA2DLlamaConfig(
                vocab_size=VOCAB,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=LENGTH,
            )
        )

        with pytest.raises(ValueError):
            resolve_algorithm("dfm", masked, bd3lm_requested=False)

    def test_auto_resolves_to_dfm_on_an_opted_in_model(self):
        from unturtle.models.generation.sampler import resolve_algorithm

        assert (
            resolve_algorithm("auto", _tiny_dfm_model(), bd3lm_requested=False) == "dfm"
        )


class TestTheMixin:
    def test_the_public_path_is_bitwise_identical_to_the_solver(self):
        """The API adds routing, never behavior: same seeds, same ids as
        hand-wiring `solve_discrete_flow` — the strongest possible API
        regression test, and it needs no training."""
        from unturtle.models.generation.dfm_solver import solve_discrete_flow

        model = _tiny_dfm_model()

        via_api = model.generate(
            algorithm="dfm",
            batch_size=4,
            steps=4,
            generator=torch.Generator().manual_seed(7),
        )

        def denoise(x_t, t, h):
            with torch.no_grad():
                return model(input_ids=x_t).logits

        x_0 = torch.full((4, LENGTH), MASK_ID, dtype=torch.long)
        direct = solve_discrete_flow(
            denoise, x_0, steps=4, generator=torch.Generator().manual_seed(7)
        )

        assert torch.equal(via_api, direct), (
            "the registry path diverged from the direct solver call"
        )

    def test_the_denoiser_seam_receives_the_solver_grid(self):
        """`dfm_denoiser(x_t, t, h)` is the overridable seam FS-DFM's
        step-aware models plug into; the grid it receives must be the
        solver's own (t = k/S, h = 1/S)."""
        from unturtle.models.generation.dfm_mixin import DiscreteFlowGenerationMixin

        model = _tiny_dfm_model()
        seen = []
        original = DiscreteFlowGenerationMixin.dfm_denoiser

        def recording(self, x_t, t, h):
            seen.append((t.clone(), h))
            return original(self, x_t, t, h)

        model.dfm_denoiser = recording.__get__(model)
        model.generate(
            algorithm="dfm",
            batch_size=2,
            steps=4,
            generator=torch.Generator().manual_seed(1),
        )

        assert len(seen) == 4
        for k, (t, h) in enumerate(seen):
            assert h == pytest.approx(0.25)
            assert torch.allclose(t, torch.full_like(t, k * 0.25))

    def test_uniform_source_draws_within_the_vocabulary(self):
        model = _tiny_dfm_model(mask_token_id=None)

        ids = model.generate(
            algorithm="dfm",
            batch_size=4,
            steps=2,
            source="uniform",
            generator=torch.Generator().manual_seed(3),
        )

        assert ids.shape == (4, LENGTH)
        assert bool((ids >= 0).all()) and bool((ids < VOCAB).all())

    def test_mask_source_without_a_mask_token_is_rejected(self):
        """Silently falling back to uniform would sample from a different
        process than the one the model was trained on."""
        model = _tiny_dfm_model(mask_token_id=None)

        with pytest.raises(ValueError, match="mask_token_id"):
            model.generate(
                algorithm="dfm",
                batch_size=2,
                steps=2,
                generator=torch.Generator().manual_seed(4),
            )

    def test_a_prompt_is_rejected(self):
        model = _tiny_dfm_model()

        with pytest.raises(ValueError, match="prompt|unconditional"):
            model.generate(
                torch.randint(0, VOCAB - 1, (2, LENGTH)),
                algorithm="dfm",
                steps=2,
            )

    def test_solver_validation_reaches_the_caller(self):
        """steps/temperature ride through to the solver's own guards —
        the mixin must not swallow or default them."""
        model = _tiny_dfm_model()

        with pytest.raises(ValueError, match="steps"):
            model.generate(algorithm="dfm", batch_size=1, steps=0)
        with pytest.raises(ValueError, match="temperature"):
            model.generate(algorithm="dfm", batch_size=1, steps=2, temperature=0.0)


@pytest.mark.slow
def test_quality_vs_nfe_through_the_public_api():
    """Quality at fixed NFE, measured through `generate(algorithm="dfm")`:
    a trained chain-task model's adjacent-token consistency must rise
    monotonically with the step budget and clear a floor at 8 steps — the
    #104 control's shape, reproduced via the public path."""
    from transformers import PretrainedConfig, PreTrainedModel
    from transformers.modeling_outputs import MaskedLMOutput

    from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss
    from unturtle.models.generation.dfm_mixin import DiscreteFlowGenerationMixin
    from unturtle.processes.discrete_flow import DiscreteFlowProcess, LinearKappa

    # The chain task needs context: the API-test stub (embedding -> linear,
    # position-independent) cannot learn it, so the e2e gets a small
    # bidirectional transformer carrying the same mixin.
    class _Config(PretrainedConfig):
        model_type = "tiny-dfm-e2e"

        def __init__(self, **kwargs):
            self.vocab_size = VOCAB
            self.hidden_size = 64
            self.max_position_embeddings = LENGTH
            super().__init__(mask_token_id=kwargs.pop("mask_token_id", None), **kwargs)

    class _Model(DiscreteFlowGenerationMixin, PreTrainedModel):
        config_class = _Config

        def __init__(self, config):
            super().__init__(config)
            self.embedding = torch.nn.Embedding(VOCAB, 64)
            self.position_embedding = torch.nn.Embedding(LENGTH, 64)
            layer = torch.nn.TransformerEncoderLayer(
                d_model=64,
                nhead=4,
                dim_feedforward=128,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            self.blocks = torch.nn.TransformerEncoder(layer, num_layers=2)
            self.head = torch.nn.Linear(64, VOCAB)
            self.post_init()

        def forward(self, input_ids, **_):
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            hidden = self.embedding(input_ids) + self.position_embedding(positions)
            return MaskedLMOutput(logits=self.head(self.blocks(hidden)))

    torch.manual_seed(0)
    model = _Model(_Config(mask_token_id=MASK_ID))
    model.train()
    generator = torch.Generator().manual_seed(0)
    process = DiscreteFlowProcess(
        vocab_size=VOCAB, mask_token_id=MASK_ID, source="mask"
    )
    scheduler = LinearKappa()

    def corpus(n):
        start = torch.randint(0, VOCAB - 1, (n, 1), generator=generator)
        return (start + torch.arange(LENGTH)) % (VOCAB - 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    for _ in range(600):
        clean = corpus(64)
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
    model.eval()

    def consistency(steps):
        ids = model.generate(
            algorithm="dfm",
            batch_size=256,
            steps=steps,
            generator=torch.Generator().manual_seed(11),
        )
        follows = (ids[:, 1:] - ids[:, :-1]) % (VOCAB - 1) == 1
        return float(follows.float().mean())

    curve = {steps: consistency(steps) for steps in (1, 2, 4, 8)}
    print(f"\nquality vs NFE via public API: {curve}")

    assert curve[8] > 0.75, f"8-step consistency too low: {curve[8]:.3f}"
    assert curve[1] < curve[4] < curve[8], (
        f"consistency is not rising with NFE: {curve}"
    )
