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

"""#154 Stage 2: ELF trainability gate — entry guards + the five conditions.

Entry guards (user-mandated, BEFORE the first optimizer-backed step):

- the Muon/aux-Adam partition (`ndim == 2` -> Muon) pinned from an
  IMMUTABLE parameter snapshot, by name, and cross-checked against the
  optimizer the pack actually builds;
- fp32 master params / EMA state survive bf16 autocast and a save/load
  round trip without quantization.

The five conditions (Stage-0 freeze):

1. loss decreases substantially on a tiny overfit;
2. endpoint token reconstruction improves from initialization;
3. the trained model responds to time / noise / self-conditioning
   perturbations (liveness);
4. save/load round trip preserves outputs;
5. gradient / precision ownership: only intended components train.
"""

import pytest
import torch

pytest.importorskip(
    "unturtle_elf",
    reason="ELF pack not installed (uv pip install -e packs/unturtle-elf)",
)
pytest.importorskip(
    "muon", reason="training dep missing (uv pip install muon-optimizer)"
)

from tests.test_elf_training_mechanics import (  # noqa: E402
    TINY,
    FrozenEncoder,
    TrainConfig,
    _batch,
    _tiny_model,
)


def _overfit(steps=200, lr=1e-2, seed=5, batch=None, config=None):
    """Run the real objective through the real optimizer on one tiny batch.

    Budget note (measured, not tuned to pass a threshold): the objective
    redraws t, the CE/L2 branch, and the SC-CFG guidance target EVERY step,
    so a single-batch loss curve is intrinsically noisy.  Measured on this
    fixture: 40 steps @2e-3 is simply too little optimization (loss still
    descending, ratio ~1.4); 200 steps @1e-2 reaches ratio ~0.27 with BOTH
    branches learning (CE 4.13->0.44, L2 44.8->11.9).  Freezing t
    (p_std=0) gives ~0.15, which localizes the residual noise to the time
    draw rather than to the objective.
    """
    from unturtle_elf.training import (
        build_muon_optimizer,
        elf_training_loss,
        ema_update,
        init_ema,
    )

    config = config or TrainConfig()
    batch = batch or _batch(batch_size=2)
    model = _tiny_model(seed=seed)
    encoder = FrozenEncoder()
    optimizer = build_muon_optimizer(model, lr=lr)
    ema = init_ema(model)
    generator = torch.Generator().manual_seed(seed)

    losses = []
    torch.manual_seed(seed)
    for _ in range(steps):
        loss, _, _ = elf_training_loss(
            model, encoder, batch, config, dropout_generator=generator
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()
        ema_update(ema, model, config.ema_decay1)
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
    return model, encoder, ema, losses, batch, config


class TestEntryGuards:
    def test_muon_adam_partition_from_an_immutable_snapshot(self):
        """ndim == 2 -> Muon, everything else -> aux Adam, checked by NAME
        from a snapshot and matched against the built optimizer."""
        from unturtle_elf.training import (
            build_muon_optimizer,
            muon_parameter_partition,
        )

        model = _tiny_model(seed=1)
        snapshot = {
            name: (param.ndim, param.requires_grad)
            for name, param in model.named_parameters()
        }
        partition = muon_parameter_partition(model)

        assert set(partition["muon"]) == {
            name for name, (ndim, req) in snapshot.items() if req and ndim == 2
        }
        assert set(partition["adam"]) == {
            name for name, (ndim, req) in snapshot.items() if req and ndim != 2
        }
        assert partition["muon"] and partition["adam"]  # both non-empty
        assert not (set(partition["muon"]) & set(partition["adam"]))

        optimizer = build_muon_optimizer(model, lr=1e-3)
        groups = {bool(group["use_muon"]): group for group in optimizer.param_groups}
        assert len(groups[True]["params"]) == len(partition["muon"])
        assert len(groups[False]["params"]) == len(partition["adam"])
        # Every Muon-group tensor is 2D and every aux tensor is not.
        assert all(p.ndim == 2 for p in groups[True]["params"])
        assert all(p.ndim != 2 for p in groups[False]["params"])
        # The oracle's optax-matched hyperparameters (muon_utils.py:166-171).
        assert groups[True]["momentum"] == 0.95
        assert groups[False]["betas"] == (0.9, 0.999)
        assert groups[True]["weight_decay"] == 0.0

    def test_fp32_master_params_and_ema_survive_autocast_and_roundtrip(self, tmp_path):
        """bf16 autocast must not quantize the master params or the EMA
        shadow, and neither may a save/load round trip."""
        from unturtle_elf.training import elf_training_loss, ema_update, init_ema

        config = TrainConfig()
        model = _tiny_model(seed=2)
        ema = init_ema(model)
        assert all(param.dtype == torch.float32 for param in model.parameters())
        assert all(tensor.dtype == torch.float32 for tensor in ema.values())

        # A bf16 autocast region around the objective (CPU bf16 autocast is
        # supported, unlike fp32) must leave master dtypes alone.
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss, _, _ = elf_training_loss(
                model,
                FrozenEncoder(),
                _batch(batch_size=2),
                config,
                dropout_generator=torch.Generator().manual_seed(1),
            )
            loss.backward()
        assert all(param.dtype == torch.float32 for param in model.parameters())
        assert all(
            param.grad is None or param.grad.dtype == torch.float32
            for param in model.parameters()
        )

        with torch.no_grad():
            for param in model.parameters():
                param.add_(0.01)
        ema_update(ema, model, decay=0.9)

        path = tmp_path / "state.pt"
        torch.save({"model": model.state_dict(), "ema": ema}, path)
        payload = torch.load(path, weights_only=True)
        for name, tensor in payload["ema"].items():
            assert tensor.dtype == torch.float32
            assert torch.equal(tensor, ema[name])
        for tensor in payload["model"].values():
            assert tensor.dtype == torch.float32


class TestTinyOverfitGate:
    @pytest.fixture(scope="class")
    def overfit(self):
        return _overfit()

    def test_condition_1_loss_decreases_substantially(self, overfit):
        """Stage-0 condition 1.  Judged on the combined loss AND on each
        branch separately — a single combined number can hide one dead
        branch, which is exactly what this gate exists to catch."""
        model, encoder, _, losses, batch, config = overfit
        head = sum(losses[:10]) / 10
        tail = sum(losses[-10:]) / 10
        assert tail < 0.5 * head, f"loss did not halve: {head:.4f} -> {tail:.4f}"

        # Both branches must be alive: re-measure per-branch metrics on the
        # trained model with the branch forced each way.
        from unturtle_elf.training import elf_training_loss

        def branch_loss(decoder_prob):
            local = TrainConfig()
            local.decoder_prob = decoder_prob
            torch.manual_seed(99)
            _, metrics, _ = elf_training_loss(
                model,
                encoder,
                batch,
                local,
                dropout_generator=torch.Generator().manual_seed(99),
            )
            return metrics

        trained_ce = float(branch_loss(1.0)["ce_loss"])
        trained_l2 = float(branch_loss(0.0)["l2_loss"])
        fresh = _tiny_model(seed=5)

        def fresh_branch(decoder_prob, key):
            local = TrainConfig()
            local.decoder_prob = decoder_prob
            torch.manual_seed(99)
            _, metrics, _ = elf_training_loss(
                fresh,
                encoder,
                batch,
                local,
                dropout_generator=torch.Generator().manual_seed(99),
            )
            return float(metrics[key])

        assert trained_ce < 0.5 * fresh_branch(1.0, "ce_loss")
        assert trained_l2 < 0.5 * fresh_branch(0.0, "l2_loss")

    def test_condition_2_endpoint_reconstruction_improves(self, overfit):
        """The decoder head must reconstruct the batch's own tokens better
        than at initialization (t=1, the reference's decode-time state)."""
        trained, encoder, _, _, batch, config = overfit
        fresh = _tiny_model(seed=5)

        def accuracy(model):
            x0 = (encoder(batch["input_ids"]) - config.latent_mean) / config.latent_std
            z_input = torch.cat([x0, torch.zeros_like(x0)], dim=-1)
            t = torch.ones(x0.shape[0])
            with torch.no_grad():
                _, logits = model(
                    z_input,
                    t,
                    deterministic=True,
                    self_cond_cfg_scale=torch.full((x0.shape[0],), 3.0),
                    decoder_step_active=True,
                )
            return (logits.argmax(-1) == batch["input_ids"]).float().mean().item()

        before, after = accuracy(fresh), accuracy(trained)
        assert after > before, f"reconstruction did not improve: {before} -> {after}"
        assert after > 0.5  # a tiny batch should be largely memorized

    def test_condition_3_liveness_to_time_noise_and_self_conditioning(self, overfit):
        """The trained model must RESPOND to each conditioning axis: two
        different times, two different noise states, and SC on vs off must
        each change the output."""
        trained, encoder, _, _, batch, config = overfit
        x0 = (encoder(batch["input_ids"]) - config.latent_mean) / config.latent_std
        generator = torch.Generator().manual_seed(0)
        z = torch.randn(x0.shape, generator=generator)
        scale = torch.full((x0.shape[0],), 3.0)

        def forward(state, t_value, sc_half):
            with torch.no_grad():
                out, _ = trained(
                    torch.cat([state, sc_half], dim=-1),
                    torch.full((state.shape[0],), t_value),
                    deterministic=True,
                    self_cond_cfg_scale=scale,
                    decoder_step_active=None,
                )
            return out

        zeros = torch.zeros_like(z)
        early, late = forward(z, 0.1, zeros), forward(z, 0.9, zeros)
        assert not torch.allclose(early, late), "no response to time"

        other = torch.randn(x0.shape, generator=generator)
        assert not torch.allclose(forward(z, 0.5, zeros), forward(other, 0.5, zeros)), (
            "no response to the noise state"
        )

        assert not torch.allclose(forward(z, 0.5, zeros), forward(z, 0.5, x0)), (
            "no response to the self-conditioning input"
        )

    def test_condition_4_save_load_roundtrip_preserves_outputs(self, overfit, tmp_path):
        trained, encoder, ema, _, batch, config = overfit
        x0 = (encoder(batch["input_ids"]) - config.latent_mean) / config.latent_std
        z_input = torch.cat([x0, torch.zeros_like(x0)], dim=-1)
        t = torch.full((x0.shape[0],), 0.4)
        scale = torch.full((x0.shape[0],), 3.0)

        with torch.no_grad():
            before, before_logits = trained(
                z_input,
                t,
                deterministic=True,
                self_cond_cfg_scale=scale,
                decoder_step_active=True,
            )

        path = tmp_path / "roundtrip.pt"
        torch.save({"model": trained.state_dict(), "ema": ema}, path)
        payload = torch.load(path, weights_only=True)
        restored = _tiny_model(seed=123)
        restored.load_state_dict(payload["model"], strict=True)
        restored.eval()
        with torch.no_grad():
            after, after_logits = restored(
                z_input,
                t,
                deterministic=True,
                self_cond_cfg_scale=scale,
                decoder_step_active=True,
            )
        assert torch.equal(before, after)
        assert torch.equal(before_logits, after_logits)

    def test_condition_5_only_intended_components_trained(self, overfit):
        """Gradient/precision ownership: the frozen encoder never moved, the
        denoiser did, and the EMA shadow tracked (but is not identical to)
        the live weights."""
        trained, encoder, ema, _, _, _ = overfit
        fresh_encoder = FrozenEncoder()
        assert torch.equal(encoder.table, fresh_encoder.table)
        assert encoder.table.grad is None

        fresh = _tiny_model(seed=5)
        moved = sum(
            1
            for (_, trained_param), (_, fresh_param) in zip(
                trained.named_parameters(), fresh.named_parameters(), strict=True
            )
            if not torch.equal(trained_param, fresh_param)
        )
        total = sum(1 for _ in trained.named_parameters())
        assert moved / total > 0.9, f"only {moved}/{total} parameters moved"

        for name, param in trained.named_parameters():
            assert not torch.equal(ema[name], param), name  # EMA lags
            assert ema[name].dtype == torch.float32


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
