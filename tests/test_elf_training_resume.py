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

"""#154 / #164-review: training-resume identity and mask-reduction property.

Two claims the earlier work asserted without a regression test:

1. **Training resume identity** — N steps run straight through must equal
   N steps + full save/restore + 1 more step.  The previous check only
   reloaded model weights and compared a forward pass, which cannot see a
   lost optimizer moment, RNG stream, or data cursor.
2. **Encoder-mask reduction** — the 3D self-attention mask collapses to the
   2D validity mask for the UNCONDITIONAL scope only; a conditional mask
   must not be reduced silently.
"""

import numpy as np
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
    FrozenEncoder,
    TrainConfig,
    _batch,
    _tiny_model,
)


def _training_state(model, optimizer, ema, generator, cursor):
    return {
        "model": {k: v.clone() for k, v in model.state_dict().items()},
        "ema": {k: v.clone() for k, v in ema.items()},
        "optimizer": optimizer.state_dict(),
        "cpu_rng": torch.get_rng_state(),
        "dropout_generator": generator.get_state(),
        "row_cursor": cursor,
    }


def _step(model, encoder, batch, config, optimizer, ema, generator):
    from unturtle_elf.training import elf_training_loss, ema_update

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
    return float(loss.detach())


def _fresh(seed=5, lr=1e-2):
    from unturtle_elf.training import build_muon_optimizer, init_ema

    model = _tiny_model(seed=seed)
    return (
        model,
        FrozenEncoder(),
        build_muon_optimizer(model, lr=lr),
        init_ema(model),
        torch.Generator().manual_seed(seed),
    )


class TestTrainingResumeIdentity:
    def test_n_steps_equals_n_plus_restore_plus_one(self):
        """The real resume claim: a run interrupted after N steps and fully
        restored must produce the SAME step N+1 as an uninterrupted run —
        model, EMA and the loss value."""
        config = TrainConfig()
        batches = [_batch(batch_size=2, seed=s) for s in (3, 4, 5)]

        # Arm A: 3 steps straight through.
        torch.manual_seed(101)
        model_a, enc_a, opt_a, ema_a, gen_a = _fresh()
        losses_a = [
            _step(model_a, enc_a, batches[i], config, opt_a, ema_a, gen_a)
            for i in range(3)
        ]

        # Arm B: 2 steps, snapshot EVERYTHING, restore into fresh objects,
        # then take the third step.
        torch.manual_seed(101)
        model_b, enc_b, opt_b, ema_b, gen_b = _fresh()
        for i in range(2):
            _step(model_b, enc_b, batches[i], config, opt_b, ema_b, gen_b)
        snapshot = _training_state(model_b, opt_b, ema_b, gen_b, cursor=2)

        model_c, enc_c, opt_c, ema_c, gen_c = _fresh(seed=999)  # different init
        model_c.load_state_dict(snapshot["model"], strict=True)
        opt_c = __import__(
            "unturtle_elf.training", fromlist=["build_muon_optimizer"]
        ).build_muon_optimizer(model_c, lr=1e-2)
        opt_c.load_state_dict(snapshot["optimizer"])
        ema_c = {k: v.clone() for k, v in snapshot["ema"].items()}
        gen_c.set_state(snapshot["dropout_generator"])
        torch.set_rng_state(snapshot["cpu_rng"])
        cursor = snapshot["row_cursor"]

        loss_c = _step(model_c, enc_c, batches[cursor], config, opt_c, ema_c, gen_c)

        assert loss_c == pytest.approx(losses_a[2], rel=0, abs=0), (
            f"resumed step-3 loss {loss_c} != uninterrupted {losses_a[2]}"
        )
        for (name, param_a), (_, param_c) in zip(
            model_a.named_parameters(), model_c.named_parameters(), strict=True
        ):
            assert torch.equal(param_a, param_c), name
        for name in ema_a:
            assert torch.equal(ema_a[name], ema_c[name]), name

    def test_dropping_optimizer_state_breaks_the_identity(self):
        """The test above must actually depend on the optimizer state —
        otherwise it would pass even for the old weights-only resume."""
        config = TrainConfig()
        batches = [_batch(batch_size=2, seed=s) for s in (3, 4, 5)]

        torch.manual_seed(101)
        model_a, enc_a, opt_a, ema_a, gen_a = _fresh()
        losses_a = [
            _step(model_a, enc_a, batches[i], config, opt_a, ema_a, gen_a)
            for i in range(3)
        ]

        torch.manual_seed(101)
        model_b, enc_b, opt_b, ema_b, gen_b = _fresh()
        for i in range(2):
            _step(model_b, enc_b, batches[i], config, opt_b, ema_b, gen_b)
        snapshot = _training_state(model_b, opt_b, ema_b, gen_b, cursor=2)

        # Weights + RNG restored, optimizer moments DISCARDED (the old
        # "resume" semantics).
        model_d, enc_d, opt_d, ema_d, gen_d = _fresh(seed=999)
        model_d.load_state_dict(snapshot["model"], strict=True)
        gen_d.set_state(snapshot["dropout_generator"])
        torch.set_rng_state(snapshot["cpu_rng"])
        _step(model_d, enc_d, batches[2], config, opt_d, ema_d, gen_d)

        differs = any(
            not torch.equal(pa, pd)
            for (_, pa), (_, pd) in zip(
                model_a.named_parameters(), model_d.named_parameters(), strict=True
            )
        )
        assert differs, (
            "discarding optimizer state changed nothing — the resume test "
            "above would be vacuous"
        )
        _ = losses_a


class TestEncoderMaskReduction:
    """Regression for the #154 correction #4 equivalence argument, which had
    lived only in a docstring."""

    def _masks(self, lengths, cond_lengths):
        from unturtle_elf._reference.encoder_utils import (
            build_self_attn_cond_masks,
        )

        length = 8
        positions = np.arange(length)[None, :]
        is_cond = positions < np.asarray(cond_lengths)[:, None]
        is_valid = positions < np.asarray(lengths)[:, None]
        return build_self_attn_cond_masks(is_cond, is_valid, xp=np)

    def test_unconditional_3d_mask_reduces_to_the_2d_validity_mask(self):
        encoder_attn, attn, cond = self._masks([5, 8, 3], [0, 0, 0])
        assert cond.sum() == 0
        for row in range(encoder_attn.shape[0]):
            for query in range(encoder_attn.shape[1]):
                assert (encoder_attn[row, query] == attn[row]).all(), (row, query)

    def test_conditional_3d_mask_is_NOT_row_constant(self):
        """The reduction must not be applied when conditioning exists: the
        cond rows attend only to cond columns, so query rows differ."""
        encoder_attn, attn, cond = self._masks([8, 8], [3, 2])
        assert cond.sum() > 0
        row_constant = all(
            (encoder_attn[b] == encoder_attn[b][0]).all()
            for b in range(encoder_attn.shape[0])
        )
        assert not row_constant, (
            "conditional mask looked row-constant — the 2D reduction would "
            "silently change semantics"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
