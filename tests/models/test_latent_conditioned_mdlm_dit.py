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

"""LatentConditionedMDLMDiT (#130 codec slice, PR-codec-1).

The paper's decoder-side conditioning, eq. (32):

    h <- h + ZeroConv(CrossAttention(ZeroConv(h); z))

cross-attention layers inserted BETWEEN the MDLM decoder's self-attention
blocks (main config: the first and the last inter-block gap), extracting
information from the latent channel only, wrapped in zero-initialized
pointwise convolutions so the pretrained decoder is bitwise-unchanged at
init.

Zero-init discipline (memory: unkillable-mutant / vacuous-gradient traps):
liveness and gradient-flow tests OPEN the channel first — with both convs
zero-initialized, gradients reach conv_out first and only then the interior,
so a test asserting interior gradients at init would be vacuous.
"""

import pytest
import torch

from unturtle.models.backbones.mdlm_dit import (
    MDLMDiTConfig,
    MDLMDiTForMaskedDiffusionLM,
)
from unturtle.models.latent.modeling_ladiff_dit import (
    LaDiffDiTConfig,
    LatentConditionedMDLMDiT,
)

VOCAB = 16
MASK_ID = VOCAB - 1
HIDDEN = 32
LAYERS = 4  # gaps at (0, LAYERS-2) = (0, 2)


def base_config() -> MDLMDiTConfig:
    return MDLMDiTConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        cond_dim=8,
        num_hidden_layers=LAYERS,
        num_attention_heads=2,
        dropout=0.0,
        max_position_embeddings=32,
        mask_token_id=MASK_ID,
    )


def ladiff_config() -> LaDiffDiTConfig:
    return LaDiffDiTConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        cond_dim=8,
        num_hidden_layers=LAYERS,
        num_attention_heads=2,
        dropout=0.0,
        max_position_embeddings=32,
        mask_token_id=MASK_ID,
        num_latents=3,
        latent_dim=HIDDEN,
    )


def models():
    """A plain MDLM-DiT and a latent-conditioned one carrying the SAME
    pretrained trunk weights (the tiny stand-in for the converted
    checkpoint)."""
    torch.manual_seed(0)
    plain = MDLMDiTForMaskedDiffusionLM(base_config()).eval()
    conditioned = LatentConditionedMDLMDiT(ladiff_config()).eval()
    conditioned.model.load_state_dict(plain.model.state_dict())
    return plain, conditioned


def open_channel(conditioned, gap=None):
    """Make the adapter output non-zero (zero-init makes it inert by
    contract): give every requested adapter's OUTER conv real weights."""
    for key, adapter in conditioned.latent_adapters.items():
        if gap is None or key == str(gap):
            torch.nn.init.normal_(adapter.conv_out.weight, std=0.2)
    return conditioned


class TestInitIsBitwiseThePretrainedDecoder:
    def test_latents_change_nothing_at_init(self):
        plain, conditioned = models()
        ids = torch.randint(0, VOCAB, (2, 12))
        latents = torch.randn(2, 3, HIDDEN)
        with torch.no_grad():
            reference = plain(input_ids=ids).logits
            with_latents = conditioned(input_ids=ids, latents=latents).logits
            without = conditioned(input_ids=ids).logits
        assert torch.equal(with_latents, reference), (
            "zero-init adapters must leave the pretrained decoder bitwise intact"
        )
        assert torch.equal(without, reference)

    def test_latents_none_skips_the_adapters_entirely(self):
        _, conditioned = models()
        open_channel(conditioned)
        calls = []
        for key, adapter in conditioned.latent_adapters.items():
            adapter.register_forward_hook(lambda *a, key=key, **k: calls.append(key))
        with torch.no_grad():
            conditioned(input_ids=torch.randint(0, VOCAB, (1, 8)))
        assert calls == [], (
            "latents=None is the plain MDLM path; the unconditional MODE "
            "is latents=noise (p_zdropout), not a skipped channel"
        )


class TestTheChannelIsLive:
    def test_opened_adapter_makes_latents_matter(self):
        _, conditioned = models()
        open_channel(conditioned)
        ids = torch.randint(0, VOCAB, (2, 12))
        a = torch.randn(2, 3, HIDDEN, generator=torch.Generator().manual_seed(1))
        b = torch.randn(2, 3, HIDDEN, generator=torch.Generator().manual_seed(2))
        with torch.no_grad():
            la = conditioned(input_ids=ids, latents=a).logits
            lb = conditioned(input_ids=ids, latents=b).logits
        assert not torch.allclose(la, lb), "open channel ignores the latent"

    def test_each_declared_gap_hosts_a_live_adapter(self):
        """Open ONE gap at a time: each must independently move the logits —
        an adapter parked on a dead gap would be silent forever."""
        for gap in (0, LAYERS - 2):
            _, conditioned = models()
            open_channel(conditioned, gap=gap)
            ids = torch.randint(0, VOCAB, (1, 10))
            latents = torch.randn(1, 3, HIDDEN)
            with torch.no_grad():
                with_l = conditioned(input_ids=ids, latents=latents).logits
                without = conditioned(input_ids=ids).logits
            assert not torch.allclose(with_l, without), f"gap {gap} adapter inert"

    def test_gradients_reach_the_interior_once_the_outer_conv_opens(self):
        """Two-step opening dynamics: at init only conv_out can receive
        gradient; once it is non-zero, the interior (conv_in, cross-attn)
        and the LATENT INPUT itself must all receive gradient."""
        _, conditioned = models()
        open_channel(conditioned)
        conditioned.train()
        ids = torch.randint(0, VOCAB, (2, 12))
        latents = torch.randn(2, 3, HIDDEN, requires_grad=True)
        conditioned(input_ids=ids, latents=latents).logits.square().mean().backward()

        assert latents.grad is not None and latents.grad.abs().sum() > 0
        for key, adapter in conditioned.latent_adapters.items():
            assert adapter.conv_in.weight.grad is not None, f"gap {key} conv_in"
            assert adapter.conv_in.weight.grad.abs().sum() > 0, f"gap {key} conv_in"
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in adapter.cross_attn.parameters()
            ), f"gap {key} cross-attn got no gradient"

    def test_conv_out_receives_gradient_at_exact_init(self):
        """The opening mechanism itself: even with everything zero-init the
        outer conv's gradient is non-zero (queries are zero -> uniform
        attention over V(z) -> non-zero CA output), so training can open
        the channel without any manual kick."""
        _, conditioned = models()
        conditioned.train()
        ids = torch.randint(0, VOCAB, (2, 12))
        latents = torch.randn(2, 3, HIDDEN)
        conditioned(input_ids=ids, latents=latents).logits.square().mean().backward()
        for key, adapter in conditioned.latent_adapters.items():
            grad = adapter.conv_out.weight.grad
            assert grad is not None and grad.abs().sum() > 0, (
                f"gap {key}: channel can never open"
            )


class TestAdapterPlacement:
    def test_adapters_run_between_the_declared_blocks(self):
        """Order pin: block0 -> adapter[0] -> block1 ... block(L-2) ->
        adapter[L-2] -> block(L-1)."""
        _, conditioned = models()
        open_channel(conditioned)
        order = []
        for i, block in enumerate(conditioned.model.blocks):
            block.register_forward_hook(lambda *a, i=i, **k: order.append(f"block{i}"))
        for key, adapter in conditioned.latent_adapters.items():
            adapter.register_forward_hook(
                lambda *a, key=key, **k: order.append(f"adapter{key}")
            )
        with torch.no_grad():
            conditioned(
                input_ids=torch.randint(0, VOCAB, (1, 8)),
                latents=torch.randn(1, 3, HIDDEN),
            )
        assert order == [
            "block0",
            "adapter0",
            "block1",
            "block2",
            f"adapter{LAYERS - 2}",
            "block3",
        ]


class TestAutoencoderFreeze:
    def test_freeze_helper_freezes_exactly_the_embedding_table(self):
        """Paper C.1: the pretrained embedding table is frozen during AE
        training; everything else (trunk, adapters) stays trainable — the
        warmup asymmetry is the training script's job, not the model's."""
        _, conditioned = models()
        conditioned.freeze_for_autoencoder_training()
        assert not conditioned.model.vocab_embed.embedding.requires_grad
        trainable = [n for n, p in conditioned.named_parameters() if p.requires_grad]
        assert any("blocks" in n for n in trainable)
        assert any("latent_adapters" in n for n in trainable)
        assert all("vocab_embed" not in n for n in trainable)


class TestGenerationStillWorks:
    def test_mdlm_generation_runs_unconditionally(self):
        _, conditioned = models()
        prompt = torch.randint(0, VOCAB - 1, (1, 4))
        torch.manual_seed(3)
        out = conditioned.generate(prompt, algorithm="mdlm", max_new_tokens=8, steps=2)
        assert out.shape[1] >= 8


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
