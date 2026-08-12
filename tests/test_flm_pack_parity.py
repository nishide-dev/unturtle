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

"""#155 Stage 1/2/3: FLM/FMLM pack differential parity against the oracle.

Oracle = the official checkout at dev/repos/flm (Stage-0: commit a1918d51),
instantiated as REAL `algo.FLM` / `algo.FMLM` Lightning modules on tiny
configs.  CPU runs force the oracle's own `use_jvp_attn=True` pure-torch
attention (the flash kernels are CUDA-only — Stage-0 two-tier plan); the
pack routes the same path on CPU by construction.

The #155 headline separations pinned here:

- fixed-state forward parity for BOTH contracts (one-time FLM, two-time
  FMLM), weights copied strict both directions;
- end-to-end decoded-token parity for the FLM Euler loop and the FMLM
  flow-map composition (deterministic seeds);
- FMLM is NOT FLM-with-steps=1: the two runners refuse each other's
  models, and the flow-map call carries the (tau, tau_tilde) pair;
- NFE = actual model evaluations; gamma/steps knobs perturb execution.
"""

import pathlib
import sys

import pytest
import torch

pytest.importorskip(
    "unturtle_flm",
    reason="FLM pack not installed (uv pip install -e packs/unturtle-flm)",
)
pytest.importorskip(
    "lightning",
    reason="oracle deps missing (uv pip install lightning hydra-core timm wandb)",
)

ORACLE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "dev" / "repos" / "flm"

TINY = dict(length=8, hidden_size=32, cond_dim=16, n_blocks=2, n_heads=2)
VOCAB = 64


class TinyTokenizer:
    vocab_size = VOCAB
    pad_token_id = 0
    eos_token_id = 1

    def __len__(self):
        return VOCAB


def _tiny_config(oracle_algo: str):
    from omegaconf import OmegaConf

    config = OmegaConf.load(ORACLE_ROOT / "configs" / "config.yaml")
    OmegaConf.set_struct(config, False)
    config.algo = OmegaConf.load(
        ORACLE_ROOT / "configs" / "algo" / f"{oracle_algo}.yaml"
    )
    config.model = OmegaConf.load(ORACLE_ROOT / "configs" / "model" / "small.yaml")
    for key, value in TINY.items():
        setattr(config.model, key, value)
    config.loader.global_batch_size = 2
    config.loader.batch_size = 2
    config.loader.eval_batch_size = 2
    config.trainer.devices = 1
    config.trainer.accumulate_grad_batches = 1
    config.prior = OmegaConf.load(ORACLE_ROOT / "configs" / "prior" / "none.yaml")
    config.algo.teacher_path = ""
    return config


@pytest.fixture(scope="module")
def oracle():
    """Import the official checkout; force its forward through the
    pure-torch jvp attention path (CPU has no flash kernels — the shim
    selects between two ORACLE-provided paths, it changes no math)."""
    if not ORACLE_ROOT.exists():
        pytest.skip("official FLM checkout missing (dev/repos/flm)")
    sys.path.insert(0, str(ORACLE_ROOT))
    try:
        import algo
        import trainer_base

        original_forward = trainer_base.TrainerBase.forward

        def cpu_forward(self, xt, sigma, sigma_prime=None, use_jvp_attn=False):
            return original_forward(self, xt, sigma, sigma_prime, use_jvp_attn=True)

        trainer_base.TrainerBase.forward = cpu_forward
        try:
            yield {"algo": algo}
        finally:
            trainer_base.TrainerBase.forward = original_forward
    finally:
        sys.path.remove(str(ORACLE_ROOT))
        for name in list(sys.modules):
            if name in (
                "algo",
                "trainer_base",
                "dataloader",
                "metrics",
                "utils",
                "models",
                "models.dit",
                "models.ema",
                "utils.jvp",
            ) and not name.startswith("unturtle"):
                sys.modules.pop(name, None)


def _oracle_model(oracle, algo_name):
    torch.manual_seed(123)
    cls = oracle["algo"].FLM if algo_name == "flm" else oracle["algo"].FMLM
    return cls(_tiny_config(algo_name), TinyTokenizer()).eval()


def _pack_model(oracle_model, kind):
    from unturtle_flm._reference.dit import DIT
    from unturtle_flm.model import FlmInferenceModel

    config = oracle_model.config
    backbone = DIT(config, vocab_size=VOCAB)
    backbone.load_state_dict(oracle_model.backbone.state_dict(), strict=True)
    backbone.eval()
    model = FlmInferenceModel(backbone, vocab_size=VOCAB, length=TINY["length"])
    model.eval()
    model.flm_config = config
    if kind == "flm":
        model.is_flm_denoiser = True
    else:
        model.is_fmlm_flow_map = True
    return model


class Request:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class TestFixedStateForwardParity:
    def test_flm_forward_is_identical(self, oracle):
        oracle_model = _oracle_model(oracle, "flm")
        pack_model = _pack_model(oracle_model, "flm")

        torch.manual_seed(7)
        z = torch.randn(2, TINY["length"], VOCAB)
        tau = torch.full((2,), 0.4)
        pack_out = pack_model(z, tau, use_jvp_attn=True)
        oracle_out = oracle_model.forward(z, tau)
        assert torch.equal(pack_out, oracle_out)

    def test_fmlm_two_time_forward_is_identical(self, oracle):
        oracle_model = _oracle_model(oracle, "fmlm")
        pack_model = _pack_model(oracle_model, "fmlm")

        torch.manual_seed(9)
        z = torch.randn(2, TINY["length"], VOCAB)
        tau_s = torch.full((2,), 0.3)
        tau_t = torch.full((2,), 0.8)
        pack_out = pack_model(z, tau_s, tau_t, use_jvp_attn=True)
        oracle_out = oracle_model.forward(z, tau_s, tau_t)
        assert torch.equal(pack_out, oracle_out)

    def test_time_reparameterization_matches_the_oracle_luts(self, oracle):
        """Stage-1 invariant: tau<->t via the SAME LUT construction — spot
        values compared against the oracle model's own LUTs."""
        oracle_model = _oracle_model(oracle, "flm")
        pack_model = _pack_model(oracle_model, "flm")
        tau = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        assert torch.equal(pack_model._tau_to_t(tau), oracle_model._tau_to_t(tau))
        t = torch.tensor([0.0, 0.1, 0.6, 1.0])
        assert torch.equal(pack_model._t_to_tau(t), oracle_model._t_to_tau(t))


class TestTrajectoryParity:
    def test_flm_decoded_tokens_match_the_oracle(self, oracle):
        from unturtle_flm.sampler import run_flm_request

        oracle_model = _oracle_model(oracle, "flm")
        pack_model = _pack_model(oracle_model, "flm")

        result = run_flm_request(pack_model, Request(num_samples=2, steps=6, seed=5))
        torch.manual_seed(5)
        oracle_tokens = oracle_model.generate_samples(num_samples=2, num_steps=6)
        assert torch.equal(result["tokens"], oracle_tokens)
        assert result["executed"]["nfe"] == 6
        assert result["executed"]["solver"] == "euler"

    @pytest.mark.parametrize(
        "steps,gamma", [(1, 0.0), (4, 1.0)], ids=["one-step", "few-step-churn"]
    )
    def test_fmlm_decoded_tokens_match_the_oracle(self, oracle, steps, gamma):
        from unturtle_flm.sampler import run_fmlm_request

        oracle_model = _oracle_model(oracle, "fmlm")
        pack_model = _pack_model(oracle_model, "fmlm")
        oracle_model.config.sampling.gamma = gamma

        result = run_fmlm_request(
            pack_model, Request(num_samples=2, steps=steps, seed=5, gamma=gamma)
        )
        torch.manual_seed(5)
        oracle_tokens = oracle_model.generate_samples(num_samples=2, num_steps=steps)
        assert torch.equal(result["tokens"], oracle_tokens)
        assert result["executed"]["nfe"] == steps
        assert result["executed"]["gamma"] == gamma
        assert result["executed"]["solver"] == "flow_map"


class TestSemanticSeparation:
    def test_the_runners_refuse_each_others_models(self, oracle):
        from unturtle_flm.sampler import run_flm_request, run_fmlm_request

        flm_oracle = _oracle_model(oracle, "flm")
        flm_model = _pack_model(flm_oracle, "flm")
        fmlm_oracle = _oracle_model(oracle, "fmlm")
        fmlm_model = _pack_model(fmlm_oracle, "fmlm")

        with pytest.raises(ValueError, match="flow map"):
            run_fmlm_request(flm_model, Request(steps=1))
        with pytest.raises(ValueError, match="FLM denoiser"):
            run_flm_request(fmlm_model, Request(steps=2))

    def test_fmlm_one_step_uses_exactly_one_two_time_call(self, oracle):
        """NFE honesty + the two-time contract: ONE forward, carrying a
        (tau_curr, tau_tilde) PAIR — never the one-time FLM signature."""
        oracle_model = _oracle_model(oracle, "fmlm")
        pack_model = _pack_model(oracle_model, "fmlm")

        calls = []
        original = pack_model.forward

        def spy(xt, sigma, sigma_prime=None, use_jvp_attn=False):
            calls.append(sigma_prime is not None)
            return original(xt, sigma, sigma_prime, use_jvp_attn=use_jvp_attn)

        pack_model.forward = spy
        from unturtle_flm.sampler import run_fmlm_request

        result = run_fmlm_request(pack_model, Request(num_samples=2, steps=1, seed=3))
        assert calls == [True]  # one call, two-time pair present
        assert result["executed"]["nfe"] == 1

    def test_step_count_perturbs_execution(self, oracle):
        from unturtle_flm.sampler import run_fmlm_request

        oracle_model = _oracle_model(oracle, "fmlm")
        pack_model = _pack_model(oracle_model, "fmlm")
        one = run_fmlm_request(pack_model, Request(num_samples=2, steps=1, seed=3))
        four = run_fmlm_request(pack_model, Request(num_samples=2, steps=4, seed=3))
        assert one["executed"]["nfe"] == 1
        assert four["executed"]["nfe"] == 4
        assert len(one["executed"]["tau_grid"]) == 2
        assert len(four["executed"]["tau_grid"]) == 5

    def test_flm_endpoint_jump_and_time_direction(self, oracle):
        """Stage-2 invariants: tau runs 0 -> 1 (noise -> data) and the final
        step jumps to the prediction (the grid's last point is exactly 1)."""
        from unturtle_flm.sampler import run_flm_request

        oracle_model = _oracle_model(oracle, "flm")
        pack_model = _pack_model(oracle_model, "flm")
        result = run_flm_request(pack_model, Request(num_samples=1, steps=3, seed=0))
        grid = result["executed"]["tau_grid"]
        assert grid[0] == 0.0 and grid[-1] == 1.0
        assert grid == sorted(grid)

    def test_rng_scoping_does_not_pollute_the_caller(self, oracle):
        from unturtle_flm.sampler import run_flm_request

        oracle_model = _oracle_model(oracle, "flm")
        pack_model = _pack_model(oracle_model, "flm")
        torch.manual_seed(4242)
        state_before = torch.random.get_rng_state()
        run_flm_request(pack_model, Request(num_samples=1, steps=2, seed=7))
        assert torch.equal(torch.random.get_rng_state(), state_before)


class TestLoaderCrossGuards:
    """Stage-1 'wrong checkpoint accepted' mutation target, fast tier: the
    loader refuses the wrong contract regardless of what HF returns."""

    def _fake_load(self, monkeypatch, *, has_prime, algo_name):
        import unturtle_flm.loader as loader

        class FakeBackbone:
            sigma_map_prime = object() if has_prime else None

        class FakeModel:
            backbone = FakeBackbone()

        monkeypatch.setattr(
            loader, "_load", lambda repo, rev, dev: (FakeModel(), algo_name)
        )
        return loader

    def test_flow_map_checkpoint_refused_by_the_flm_loader(self, monkeypatch):
        loader = self._fake_load(monkeypatch, has_prime=True, algo_name="fmlm")
        with pytest.raises(ValueError, match="load_fmlm_model"):
            loader.load_flm_model()

    def test_plain_flm_checkpoint_refused_by_the_fmlm_loader(self, monkeypatch):
        loader = self._fake_load(monkeypatch, has_prime=False, algo_name="flm")
        with pytest.raises(ValueError, match="load_flm_model"):
            loader.load_fmlm_model()


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
