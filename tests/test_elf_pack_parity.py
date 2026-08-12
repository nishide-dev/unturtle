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

"""#153 Stage 1/2: ELF pack differential parity against the official oracle.

Oracle = the official pytorch_elf checkout at dev/repos/elf (Stage-0 freeze:
commit b29d8833).  The pack's `_reference` modules are verbatim ports; these
tests prove the DIFFERENTIAL — same inputs, same weights → same outputs —
and the ADAPTER's semantics (solver-as-requested, executed steps, endpoint-
only discretization, one-generator ownership) on a tiny CPU model.

The real-checkpoint (ELF-B download) audits are @slow.  A missing oracle
checkout skips (the pack's own suite still runs); the fast structural
parity here needs no checkpoint and no network.
"""

import pathlib
import sys

import pytest
import torch

pytest.importorskip(
    "unturtle_elf",
    reason="ELF pack not installed (uv pip install -e packs/unturtle-elf)",
)

ORACLE_SRC = (
    pathlib.Path(__file__).resolve().parent.parent / "dev" / "repos" / "elf" / "src"
)

TINY = dict(
    text_encoder_dim=16,
    max_length=8,
    bottleneck_dim=8,
    num_time_tokens=2,
    num_self_cond_cfg_tokens=2,
    num_model_mode_tokens=2,
    vocab_size=64,
)


@pytest.fixture
def oracle(monkeypatch):
    """Import the official checkout under its own bare module names, and
    sweep those generic names (modules/utils/configs) out of sys.modules
    afterwards so nothing else in the process can accidentally see them."""
    if not ORACLE_SRC.exists():
        pytest.skip("official ELF checkout missing (dev/repos/elf)")
    monkeypatch.syspath_prepend(str(ORACLE_SRC))
    injected = []
    try:
        import importlib

        for name in (
            "modules.model",
            "utils.sampling_utils",
            "utils.generation_utils",
            "configs.config",
        ):
            importlib.import_module(name)
            injected.extend([name, name.split(".")[0]])
        yield {
            "model": sys.modules["modules.model"],
            "sampling": sys.modules["utils.sampling_utils"],
            "generation": sys.modules["utils.generation_utils"],
            "config": sys.modules["configs.config"],
        }
    finally:
        for name in list(sys.modules):
            # unturtle.utils etc. are untouched: only BARE top-levels go.
            if name.split(".")[0] in (
                "modules",
                "utils",
                "configs",
            ) and not name.startswith("unturtle"):
                sys.modules.pop(name, None)


def _tiny_pack_model(seed=0):
    from unturtle_elf._reference.model import ELF

    torch.manual_seed(seed)
    model = ELF(depth=2, hidden_size=32, num_heads=2, **TINY)
    # The reference zero-inits final_layer.linear (DiT convention), so an
    # UNTRAINED model's flow output is identically zero and would hide any
    # conditioning difference from the semantic tests below.  Perturb the
    # head; the same state dict is copied into the oracle, so differential
    # parity is untouched.
    torch.nn.init.normal_(model.final_layer.linear.weight, std=0.02)
    torch.nn.init.normal_(model.final_layer.linear.bias, std=0.02)
    model.eval()
    model.is_elf_denoiser = True
    model.elf_config = {
        "model": "ELF-B",
        "max_length": TINY["max_length"],
        "encoder_model_name": "t5-small",
        "denoiser_noise_scale": 2.0,
        "t_eps": 0.05,
        "self_cond_prob": 0.5,
        "num_self_cond_cfg_tokens": TINY["num_self_cond_cfg_tokens"],
        "num_model_mode_tokens": TINY["num_model_mode_tokens"],
        "num_time_tokens": TINY["num_time_tokens"],
        "bottleneck_dim": TINY["bottleneck_dim"],
    }
    return model


def _tiny_oracle_model(oracle, pack_model):
    torch.manual_seed(123)  # independent init: weights come from the pack
    model = oracle["model"].ELF(depth=2, hidden_size=32, num_heads=2, **TINY)
    model.load_state_dict(pack_model.state_dict(), strict=True)
    model.eval()
    return model


def _reference_config(oracle, pack_model):
    config = oracle["config"].Config()
    for key, value in pack_model.elf_config.items():
        setattr(config, key, value)
    return config


class TestFixedStateForwardParity:
    def test_denoiser_forward_is_identical(self, oracle):
        """Same weights, same (x, t, SC-CFG) → identical outputs, fp32 CPU.
        The state dict loads strict=True in BOTH directions, which is also
        the key-coverage audit for the ported architecture."""
        pack_model = _tiny_pack_model()
        oracle_model = _tiny_oracle_model(oracle, pack_model)

        torch.manual_seed(7)
        z = torch.randn(2, TINY["max_length"], TINY["text_encoder_dim"])
        x = torch.cat([z, torch.zeros_like(z)], dim=-1)
        t = torch.full((2,), 0.37)
        scale = torch.full((2,), 3.0)

        pack_out, pack_logits = pack_model(
            x,
            t,
            deterministic=True,
            self_cond_cfg_scale=scale,
            decoder_step_active=True,
        )
        oracle_out, oracle_logits = oracle_model(
            x,
            t,
            deterministic=True,
            self_cond_cfg_scale=scale,
            decoder_step_active=True,
        )
        assert torch.equal(pack_out, oracle_out)
        assert torch.equal(pack_logits, oracle_logits)

    def test_endpoint_argmax_parity_on_a_fixed_state(self, oracle):
        """Stage-1 pin 8: final token argmax parity for a fixed latent."""
        from unturtle_elf._reference.generation_utils import _dlm_decode_batch

        pack_model = _tiny_pack_model()
        oracle_model = _tiny_oracle_model(oracle, pack_model)
        config = _reference_config(oracle, pack_model)

        torch.manual_seed(11)
        z = torch.randn(3, TINY["max_length"], TINY["text_encoder_dim"])
        pack_tokens = _dlm_decode_batch(z, pack_model, 1.0, config, 3.0)
        oracle_tokens = oracle["generation"]._dlm_decode_batch(
            z, oracle_model, 1.0, config, 3.0
        )
        assert torch.equal(pack_tokens, oracle_tokens)


class TestAdapterTrajectoryParity:
    def _run_pack(self, pack_model, **kwargs):
        from unturtle_elf.sampler import run_generation_request

        from unturtle.models.generation.sampler import GenerationRequest

        request = GenerationRequest(
            inputs=None,
            generation_config=None,
            kwargs={"num_samples": 2, "seed": 5, **kwargs},
        )
        return run_generation_request(pack_model, request)

    def _run_oracle(
        self,
        oracle,
        oracle_model,
        pack_model,
        *,
        solver,
        steps,
        gamma,
        time_schedule="uniform",
    ):
        config = _reference_config(oracle, pack_model)
        sampling = oracle["config"].SamplingConfig()
        sampling.sampling_method = solver
        sampling.num_sampling_steps = [steps]
        sampling.cfgs = [1.0]
        sampling.self_cond_cfg_scales = [3.0]
        sampling.time_schedule = time_schedule
        sampling.sde_gamma = gamma

        generator = torch.Generator().manual_seed(5)
        torch.manual_seed(5)
        t_steps = oracle["sampling"].get_sampling_steps(
            steps, time_schedule=time_schedule, dtype=torch.float32
        )
        z = (
            torch.randn(
                (2, TINY["max_length"], TINY["text_encoder_dim"]),
                generator=generator,
                dtype=torch.float32,
            )
            * config.denoiser_noise_scale
        )
        latent = oracle["generation"]._generate_samples_single_batch(
            model=oracle_model,
            generator=generator,
            z=z,
            t_steps=t_steps,
            cond_seq=None,
            cond_seq_mask=None,
            config=config,
            sampling_config=sampling,
            cfg_scale=1.0,
            self_cond_cfg_scale=3.0,
        )
        tokens = oracle["generation"]._dlm_decode_batch(
            latent, oracle_model, t_steps[-1], config, 3.0
        )
        # The oracle pipeline masks after the first EOS before decoding
        # (generation.py:184) — part of the end-to-end contract.
        return oracle["generation"].mask_after_eos(
            tokens, eos_token_id=1, pad_token_id=0
        )

    @pytest.mark.parametrize(
        ("solver", "gamma"), [("sde", 1.5), ("ode", 0.0)], ids=["sde", "ode"]
    )
    def test_end_to_end_decoded_tokens_match_the_oracle(self, oracle, solver, gamma):
        """The ADAPTER (unturtle_elf.sampler) against the oracle's own
        rollout: same seed, uniform grid (deterministic), CPU fp32 → the
        decoded token ids must be IDENTICAL.  This is the strongest rung of
        the parity ladder available without the real checkpoint."""
        pack_model = _tiny_pack_model()
        oracle_model = _tiny_oracle_model(oracle, pack_model)

        result = self._run_pack(
            pack_model,
            solver=solver,
            steps=6,
            sde_gamma=gamma,
            time_schedule="uniform",
            self_cond_cfg_scale=3.0,
        )
        oracle_tokens = self._run_oracle(
            oracle,
            oracle_model,
            pack_model,
            solver=solver,
            steps=6,
            gamma=gamma,
        )
        assert torch.equal(result["tokens"], oracle_tokens)
        assert result["executed"]["solver"] == solver
        assert result["executed"]["steps_executed"] == 6
        assert result["executed"]["nfe"] == 6


class TestAdapterSemantics:
    def test_executed_steps_come_from_the_grid_not_the_request(self):
        pack_model = _tiny_pack_model()
        result = TestAdapterTrajectoryParity()._run_pack(
            pack_model,
            solver="ode",
            steps=4,
            sde_gamma=0.0,
            time_schedule="uniform",
        )
        assert result["executed"]["steps_requested"] == 4
        assert result["executed"]["steps_executed"] == 4

    def test_requested_solver_is_the_solver_executed(self, monkeypatch):
        """Parameter-default trap (issue mutation target: 'requested SDE
        executes ODE or vice versa') — pinned by counting the ACTUAL step
        functions the reference rollout dispatches to."""
        from unturtle_elf._reference import generation_utils

        calls = {"sde": 0, "ode": 0}
        real_sde, real_ode = generation_utils._sde_step, generation_utils._ode_step

        def spy_sde(*args, **kwargs):
            calls["sde"] += 1
            return real_sde(*args, **kwargs)

        def spy_ode(*args, **kwargs):
            calls["ode"] += 1
            return real_ode(*args, **kwargs)

        monkeypatch.setattr(generation_utils, "_sde_step", spy_sde)
        monkeypatch.setattr(generation_utils, "_ode_step", spy_ode)

        maker = TestAdapterTrajectoryParity()
        pack_model = _tiny_pack_model()
        maker._run_pack(
            pack_model,
            solver="sde",
            steps=6,
            sde_gamma=1.5,
            time_schedule="uniform",
        )
        # The reference reserves the FINAL interval for a deterministic ODE
        # step even under SDE (generation_utils.py:110) — executed shape is
        # 5 SDE + 1 ODE, and that split is itself frozen reference behavior.
        assert calls == {"sde": 5, "ode": 1}

        calls.update(sde=0, ode=0)
        maker._run_pack(pack_model, solver="ode", steps=6, time_schedule="uniform")
        assert calls == {"sde": 0, "ode": 6}

    def test_self_conditioning_scale_changes_the_forward(self):
        """SC-CFG is threaded, not decorative: scale 0 vs 3 on the same
        fixed state produce different denoiser outputs."""
        pack_model = _tiny_pack_model()
        z = torch.randn(2, TINY["max_length"], TINY["text_encoder_dim"])
        x = torch.cat([z, torch.zeros_like(z)], dim=-1)
        t = torch.full((2,), 0.4)
        out_low, _ = pack_model(
            x,
            t,
            deterministic=True,
            self_cond_cfg_scale=torch.full((2,), 0.0),
        )
        out_high, _ = pack_model(
            x,
            t,
            deterministic=True,
            self_cond_cfg_scale=torch.full((2,), 3.0),
        )
        assert not torch.equal(out_low, out_high)

    def test_non_elf_model_is_refused(self):
        from unturtle_elf.sampler import run_generation_request

        from unturtle.models.generation.sampler import GenerationRequest

        class NotElf:
            pass

        with pytest.raises(ValueError, match="ELF denoiser"):
            run_generation_request(
                NotElf(),
                GenerationRequest(inputs=None, generation_config=None, kwargs={}),
            )

    def test_unknown_solver_is_loud(self):
        pack_model = _tiny_pack_model()
        with pytest.raises(ValueError, match="solver"):
            TestAdapterTrajectoryParity()._run_pack(pack_model, solver="dpm")


class TestReviewPins160:
    """Pins for the #160 review's CRITICAL findings, RED-first."""

    def test_logit_normal_grid_uses_the_checkpoint_schedule(self, oracle):
        """Review F1 (CRITICAL): the oracle passes the CHECKPOINT's
        denoiser_p_mean/p_std into get_sampling_steps (generation.py:151);
        ELF-B's config.yml has p_mean=-1.5, and the function's -0.8 default
        allocates ~5x fewer points to the high-noise regime.  The Stage-0
        freeze flagged exactly this trap.  Pinned by comparing the grid the
        ADAPTER records against the oracle's grid under the same seed."""
        pack_model = _tiny_pack_model()
        pack_model.elf_config["denoiser_p_mean"] = -1.5
        pack_model.elf_config["denoiser_p_std"] = 0.8

        torch.manual_seed(31)
        result = TestAdapterTrajectoryParity()._run_pack(
            pack_model,
            solver="ode",
            steps=6,
            time_schedule="logit_normal",
            seed=31,
        )
        torch.manual_seed(31)
        oracle_grid = oracle["sampling"].get_sampling_steps(
            6,
            time_schedule="logit_normal",
            P_mean=-1.5,
            P_std=0.8,
            dtype=torch.float32,
        )
        assert result["executed"]["t_grid"] == pytest.approx(oracle_grid.tolist())

    def test_post_eos_content_is_masked_like_the_oracle(self, oracle):
        """Review F2 (CRITICAL): the oracle masks everything after the
        first EOS on T5 ids BEFORE decoding (generation.py:184); unmasked
        argmax ids leak unbounded post-EOS junk into every evaluator
        column.  The adapter must return oracle-masked tokens."""
        pack_model = _tiny_pack_model()
        result = TestAdapterTrajectoryParity()._run_pack(
            pack_model, solver="ode", steps=4, time_schedule="uniform"
        )
        tokens = result["tokens"]
        eos_id = result["executed"]["eos_token_id"]
        pad_id = result["executed"]["pad_token_id"]
        oracle_masked = oracle["generation"].mask_after_eos(
            tokens.clone(), eos_token_id=eos_id, pad_token_id=pad_id
        )
        assert torch.equal(tokens, oracle_masked)  # already masked
        # And the mask is not vacuous on this batch by construction: force
        # an EOS mid-row and re-mask to prove the semantics are the
        # reference's (first EOS kept, tail padded).
        forced = tokens.clone()
        forced[:, 1] = eos_id
        forced[:, 2:] = eos_id + 1  # junk that must vanish
        remasked = oracle["generation"].mask_after_eos(
            forced.clone(), eos_token_id=eos_id, pad_token_id=pad_id
        )
        assert (remasked[:, 2:] == pad_id).all()


class TestLoaderKeyPolicy:
    """Stage-1 pin 1 (fast tier): checkpoint key coverage is LOUD — the
    issue's 'wrong checkpoint key silently dropped' mutation target."""

    def _fake_checkpoint(self, tmp_path, monkeypatch, mutate_state):
        from unturtle_elf._reference import model as reference_model
        from unturtle_elf.loader import load_elf_model_from_files

        tiny = _tiny_pack_model()
        monkeypatch.setattr(reference_model, "ELF_B", lambda **kwargs: tiny)
        state = dict(tiny.state_dict())
        mutate_state(state)
        path = tmp_path / "checkpoint_fake"
        torch.save({"params": state, "ema_params1": state}, path)
        raw_config = {
            "model": "ELF-B",
            "max_length": TINY["max_length"],
            "encoder_model_name": "t5-small",
        }
        return lambda: load_elf_model_from_files(str(path), raw_config)

    def test_unexpected_keys_raise(self, tmp_path, monkeypatch):
        load = self._fake_checkpoint(
            tmp_path,
            monkeypatch,
            lambda state: state.update(bogus_extra_key=torch.zeros(1)),
        )
        with pytest.raises(RuntimeError, match="bogus_extra_key"):
            load()

    def test_missing_keys_raise(self, tmp_path, monkeypatch):
        load = self._fake_checkpoint(
            tmp_path,
            monkeypatch,
            lambda state: state.pop("final_layer.linear.weight"),
        )
        with pytest.raises(RuntimeError, match="final_layer.linear.weight"):
            load()

    def test_missing_ema_falls_back_loudly(self, tmp_path, monkeypatch):
        from unturtle_elf._reference import model as reference_model
        from unturtle_elf.loader import load_elf_model_from_files

        tiny = _tiny_pack_model()
        monkeypatch.setattr(reference_model, "ELF_B", lambda **kwargs: tiny)
        path = tmp_path / "checkpoint_no_ema"
        torch.save({"params": dict(tiny.state_dict())}, path)
        raw_config = {
            "model": "ELF-B",
            "max_length": TINY["max_length"],
            "encoder_model_name": "t5-small",
        }
        with pytest.warns(UserWarning, match="no EMA"):
            model = load_elf_model_from_files(str(path), raw_config)
        assert model.is_elf_denoiser is True


@pytest.mark.slow
class TestRealCheckpointAudit:
    """Stage-1 audits on the real ELF-B checkpoint (downloads ~840MB)."""

    @pytest.fixture(scope="class")
    def elf_b(self):
        from unturtle_elf.loader import load_elf_model

        return load_elf_model()

    def test_parameter_count_matches_the_paper_scale(self, elf_b):
        total = sum(p.numel() for p in elf_b.parameters())
        assert 100_000_000 < total < 110_000_000  # "105M" reference scale

    def test_provenance_rides_with_the_model(self, elf_b):
        from unturtle_elf.loader import DEFAULT_CHECKPOINT, DEFAULT_REVISION

        assert elf_b.elf_checkpoint.repo_id == DEFAULT_CHECKPOINT
        assert elf_b.elf_checkpoint.revision == DEFAULT_REVISION
        assert elf_b.elf_checkpoint.used_ema is True
        assert elf_b.elf_config["max_length"] == 1024

    def test_short_real_generation_smoke(self, elf_b):
        result = TestAdapterTrajectoryParity()._run_pack(
            elf_b,
            solver="sde",
            steps=4,
            sde_gamma=1.5,
        )
        assert result["tokens"].shape == (2, 1024)
        assert result["executed"]["nfe"] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
