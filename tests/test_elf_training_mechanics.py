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

"""#154 Stage 1: ELF training objective/mechanics — RED-first.

Oracle = official `train_step.py` at dev/repos/elf (Stage-0 training freeze:
commit b29d8833, config from the ELF-B checkpoint).  Pinned here, per the
freeze and the user's Stage-1 list:

- ORACLE DIFFERENTIAL: identical loss AND identical gradients on a tiny
  model under frozen RNG (grad_accum=2 so the oracle takes no optimizer
  step);
- denoiser minimizer = the clean latent (analytic: a cheating model that
  outputs x0 achieves ~zero L2; perturbations strictly increase it);
- decoder CE minimizer = the target tokens;
- Bernoulli mixed branch + SINGLE denominator (hand-computed);
- padding/cond positions contribute nothing;
- the self-conditioning target is a detached no-grad seam;
- gradients reach the denoiser, never the frozen encoder;
- the training time distribution uses the CHECKPOINT's p_mean=-1.5
  (the #153 default-trap, now on the training side);
- determinism given seeds; EMA lerp semantics.
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
    vocab_size=32,
)


class TrainConfig:
    """The training-relevant slice of the frozen ELF-B config (Stage-0),
    scaled to tiny dims where size-only."""

    t_eps = 0.05
    self_cond_prob = 0.5
    latent_mean = 0.0
    latent_std = 0.2
    decoder_prob = 0.2
    decoder_noise_scale = 5.0
    decoder_p_mean = 0.8
    decoder_p_std = 0.8
    denoiser_p_mean = -1.5
    denoiser_p_std = 0.8
    denoiser_noise_scale = 2.0
    time_schedule = "logit_normal"
    num_self_cond_cfg_tokens = TINY["num_self_cond_cfg_tokens"]
    self_cond_cfg_min = 0.5
    self_cond_cfg_max = 5.0
    label_drop_prob = 0.0
    pad_token = "pad"
    use_bf16 = False  # CPU fp32 tier
    grad_accum_steps = 2  # oracle takes no optimizer step on the first call
    ema_decay1 = 0.9999


class FrozenEncoder(torch.nn.Module):
    """Stands in for the frozen T5 encoder: deterministic embedding lookup
    with REAL (frozen) parameters so gradient-reach tests are meaningful."""

    def __init__(self):
        super().__init__()
        # Local generator: reseeding the GLOBAL stream in __init__ would
        # silently desync every RNG-order-sensitive test that constructs an
        # encoder after its manual_seed (it did — caught at first run).
        self.table = torch.nn.Parameter(
            torch.randn(
                TINY["vocab_size"],
                TINY["text_encoder_dim"],
                generator=torch.Generator().manual_seed(77),
            )
        )
        self.requires_grad_(False)

    def forward(self, input_ids, attention_mask=None, deterministic=True):
        del attention_mask, deterministic
        return self.table[input_ids]


def _tiny_model(seed=0):
    from unturtle_elf._reference.model import ELF

    torch.manual_seed(seed)
    model = ELF(depth=2, hidden_size=32, num_heads=2, **TINY)
    # Zero-init DiT head hides everything on untrained models (standing
    # lesson from #153/#155): perturb so objectives are observable.
    torch.nn.init.normal_(model.final_layer.linear.weight, std=0.5)
    torch.nn.init.normal_(model.final_layer.linear.bias, std=0.5)
    model.train()
    return model


def _batch(batch_size=4, seed=3):
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(
        0, TINY["vocab_size"], (batch_size, TINY["max_length"]), generator=generator
    )
    ones = torch.ones(batch_size, TINY["max_length"])
    return {
        "input_ids": input_ids,
        "attention_mask": ones.clone(),
        "encoder_attention_mask": ones.clone(),
        "cond_seq_mask": torch.zeros(batch_size, TINY["max_length"]),
    }


class TestOracleDifferential:
    def test_loss_and_gradients_match_the_oracle(self):
        """Same tiny model/weights, same frozen RNG → identical loss and
        identical parameter gradients (the oracle runs backward but no
        optimizer step at grad_accum=2)."""
        sys.path.insert(0, str(ORACLE_SRC))
        try:
            import train_step as oracle_train_step
            from unturtle_elf.training import elf_training_loss
            from utils.train_utils import TrainState

            config = TrainConfig()
            encoder = FrozenEncoder()
            batch = _batch()

            model_oracle = _tiny_model(seed=11)
            model_pack = _tiny_model(seed=999)
            model_pack.load_state_dict(model_oracle.state_dict(), strict=True)

            state = TrainState(
                model=model_oracle,
                optimizer=torch.optim.SGD(model_oracle.parameters(), lr=0.0),
                dropout_generator=torch.Generator().manual_seed(5),
                step=0,
            )
            torch.manual_seed(21)
            _, oracle_metrics = oracle_train_step.train_step(
                state, encoder, batch, config
            )

            torch.manual_seed(21)
            loss, pack_metrics, _ = elf_training_loss(
                model_pack,
                encoder,
                batch,
                config,
                dropout_generator=torch.Generator().manual_seed(5),
            )
            # The oracle backpropagates loss/accum_steps (train_step.py:266)
            # — mirror it, or every gradient differs by exactly that factor.
            (loss / config.grad_accum_steps).backward()

            assert torch.equal(loss.detach(), oracle_metrics["loss"])
            assert torch.equal(pack_metrics["l2_loss"], oracle_metrics["l2_loss"])
            assert torch.equal(pack_metrics["ce_loss"], oracle_metrics["ce_loss"])
            for (name_o, param_o), (name_p, param_p) in zip(
                model_oracle.named_parameters(),
                model_pack.named_parameters(),
                strict=True,
            ):
                assert name_o == name_p
                if param_o.grad is None:
                    assert param_p.grad is None, name_p
                else:
                    assert torch.equal(param_o.grad, param_p.grad), name_o
        finally:
            sys.path.remove(str(ORACLE_SRC))
            for name in list(sys.modules):
                if name.split(".")[0] in (
                    "train_step",
                    "utils",
                    "modules",
                    "configs",
                ) and not name.startswith("unturtle"):
                    sys.modules.pop(name, None)


class _CheatingDenoiser(torch.nn.Module):
    """Returns exactly the clean latent (plus optional perturbation) as the
    x-prediction, and one-hot logits for the decoder head."""

    def __init__(self, x0, input_ids, x_offset=0.0, logit_target=None):
        super().__init__()
        self.x0 = x0
        self.x_offset = x_offset
        self.logits = (
            torch.nn.functional.one_hot(
                input_ids if logit_target is None else logit_target,
                TINY["vocab_size"],
            ).float()
            * 50.0
        )
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        z_input,
        t,
        deterministic=True,
        self_cond_cfg_scale=None,
        decoder_step_active=None,
    ):
        out = self.x0 + self.x_offset + 0.0 * self.dummy
        if decoder_step_active is not None:
            return out, self.logits
        return out


class TestAnalyticMinimizers:
    def _loss(self, model, batch, config, seed=9):
        from unturtle_elf.training import elf_training_loss

        torch.manual_seed(seed)
        loss, metrics, _ = elf_training_loss(
            model,
            FrozenEncoder(),
            batch,
            config,
            dropout_generator=torch.Generator().manual_seed(5),
        )
        return loss, metrics

    def _x0(self, batch):
        return (FrozenEncoder()(batch["input_ids"]) - 0.0) / 0.2

    def test_denoiser_minimizer_is_the_clean_latent(self):
        """SC off, denoiser-only: predicting exactly x0 gives ~zero L2
        (v_pred == v_target identically); any offset strictly increases it,
        and a target-sign flip would make the 'perfect' model score WORSE
        than the offset one — pinned via ordering."""
        config = TrainConfig()
        config.self_cond_prob = 0.0
        config.num_self_cond_cfg_tokens = 0
        config.decoder_prob = 0.0
        batch = _batch()
        x0 = self._x0(batch)

        perfect, metrics = self._loss(
            _CheatingDenoiser(x0, batch["input_ids"]), batch, config
        )
        off, _ = self._loss(
            _CheatingDenoiser(x0, batch["input_ids"], x_offset=0.5), batch, config
        )
        assert perfect.item() == pytest.approx(0.0, abs=1e-8)
        assert off.item() > 1e-3
        assert metrics["ce_loss"].item() == 0.0  # no decoder rows drawn

    def test_decoder_minimizer_is_the_target_tokens(self):
        config = TrainConfig()
        config.self_cond_prob = 0.0
        config.num_self_cond_cfg_tokens = 0
        config.decoder_prob = 1.0
        batch = _batch()
        x0 = self._x0(batch)

        right, _ = self._loss(_CheatingDenoiser(x0, batch["input_ids"]), batch, config)
        wrong_targets = (batch["input_ids"] + 1) % TINY["vocab_size"]
        wrong, _ = self._loss(
            _CheatingDenoiser(x0, batch["input_ids"], logit_target=wrong_targets),
            batch,
            config,
        )
        assert right.item() == pytest.approx(0.0, abs=1e-6)
        assert wrong.item() > 1.0

    def test_single_denominator_hand_computed(self):
        """Mixed batch: loss == (Σce·mask + Σl2·mask) / Σloss_mask exactly,
        recomputed by hand from the aux tensors."""
        from unturtle_elf.training import elf_training_loss

        config = TrainConfig()
        config.self_cond_prob = 0.0
        config.num_self_cond_cfg_tokens = 0
        config.decoder_prob = 0.5
        batch = _batch(batch_size=6)
        x0 = self._x0(batch)
        model = _CheatingDenoiser(x0, batch["input_ids"], x_offset=0.3)

        torch.manual_seed(9)
        loss, _, aux = elf_training_loss(
            model,
            FrozenEncoder(),
            batch,
            config,
            dropout_generator=torch.Generator().manual_seed(5),
        )
        hand = (
            (aux["ce_per_token"] * aux["ce_mask"]).sum()
            + (aux["l2_per_token"] * aux["l2_mask"]).sum()
        ) / aux["loss_mask"].sum()
        assert torch.equal(loss.detach(), hand.detach())
        # Both branches actually drawn in this batch (else the test is weak).
        assert aux["ce_mask"].sum() > 0 and aux["l2_mask"].sum() > 0

    def test_padding_positions_contribute_nothing(self):
        config = TrainConfig()
        config.self_cond_prob = 0.0
        config.num_self_cond_cfg_tokens = 0
        config.decoder_prob = 0.5
        batch = _batch()
        batch["attention_mask"][:, -3:] = 0.0
        x0 = self._x0(batch)
        model = _CheatingDenoiser(x0, batch["input_ids"], x_offset=0.3)

        base, _ = self._loss(model, batch, config)

        corrupted = {k: v.clone() for k, v in batch.items()}
        corrupted["input_ids"] = batch["input_ids"].clone()
        corrupted["input_ids"][:, -3:] = 0  # garbage targets under the pad
        model2 = _CheatingDenoiser(x0, batch["input_ids"], x_offset=0.3)
        second, _ = self._loss(model2, corrupted, config)
        assert torch.equal(base.detach(), second.detach())


class TestSeams:
    def test_self_conditioning_target_is_detached(self):
        """The SC-guided v target comes from two no-grad forwards and is
        detached — gradients flow ONLY through the single training forward."""
        from unturtle_elf.training import elf_training_loss

        config = TrainConfig()
        config.decoder_prob = 0.0
        batch = _batch()
        model = _tiny_model(seed=1)

        torch.manual_seed(13)
        loss, _, aux = elf_training_loss(
            model,
            FrozenEncoder(),
            batch,
            config,
            dropout_generator=torch.Generator().manual_seed(5),
        )
        assert aux["v_final_target"].requires_grad is False
        loss.backward()  # must not error (no second-graph retain needed)

    def test_gradients_reach_the_model_but_not_the_encoder(self):
        from unturtle_elf.training import elf_training_loss

        config = TrainConfig()
        batch = _batch()
        model = _tiny_model(seed=2)
        encoder = FrozenEncoder()

        torch.manual_seed(17)
        loss, _, _ = elf_training_loss(
            model,
            encoder,
            batch,
            config,
            dropout_generator=torch.Generator().manual_seed(4),
        )
        loss.backward()
        assert encoder.table.grad is None
        with_grads = sum(
            1
            for p in model.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        total = sum(1 for _ in model.parameters())
        assert with_grads / total > 0.9  # both heads exercised by the mixed batch

    def test_training_time_distribution_uses_the_checkpoint_p_mean(self):
        """The #153 default-trap, training side: the draw must be seeded-
        equal to the oracle's sample_timesteps with p_mean=-1.5 and differ
        from the -0.8 default."""
        sys.path.insert(0, str(ORACLE_SRC))
        try:
            from unturtle_elf.training import elf_training_loss
            from utils.sampling_utils import sample_timesteps

            config = TrainConfig()
            config.self_cond_prob = 0.0
            config.num_self_cond_cfg_tokens = 0
            config.decoder_prob = 0.0
            batch = _batch()
            x0 = (FrozenEncoder()(batch["input_ids"]) - 0.0) / 0.2

            torch.manual_seed(31)
            _, _, aux = elf_training_loss(
                _CheatingDenoiser(x0, batch["input_ids"]),
                FrozenEncoder(),
                batch,
                config,
                dropout_generator=torch.Generator().manual_seed(5),
            )
            torch.manual_seed(31)
            expected = sample_timesteps(
                batch["input_ids"].shape[0],
                P_mean=-1.5,
                P_std=0.8,
                time_schedule="logit_normal",
                dtype=torch.float32,
            )
            assert torch.equal(aux["t"], expected)
            torch.manual_seed(31)
            default_draw = sample_timesteps(
                batch["input_ids"].shape[0],
                time_schedule="logit_normal",
                dtype=torch.float32,
            )
            assert not torch.equal(aux["t"], default_draw)
        finally:
            sys.path.remove(str(ORACLE_SRC))
            for name in list(sys.modules):
                if name.split(".")[0] in ("utils", "modules", "configs") and not (
                    name.startswith("unturtle")
                ):
                    sys.modules.pop(name, None)

    def test_deterministic_given_seeds(self):
        from unturtle_elf.training import elf_training_loss

        config = TrainConfig()
        batch = _batch()

        losses = []
        for _ in range(2):
            model = _tiny_model(seed=6)
            torch.manual_seed(23)
            loss, _, _ = elf_training_loss(
                model,
                FrozenEncoder(),
                batch,
                config,
                dropout_generator=torch.Generator().manual_seed(8),
            )
            losses.append(loss.detach())
        assert torch.equal(losses[0], losses[1])


class TestEma:
    def test_ema_lerp_semantics(self):
        from unturtle_elf.training import ema_update, init_ema

        model = _tiny_model(seed=4)
        ema = init_ema(model)
        before = {name: tensor.clone() for name, tensor in ema.items()}
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)
        ema_update(ema, model, decay=0.9)
        for name, param in model.named_parameters():
            expected = 0.9 * before[name] + 0.1 * param.detach()
            assert torch.allclose(ema[name], expected)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
