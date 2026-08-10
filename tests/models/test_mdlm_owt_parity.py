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

"""Real-checkpoint parity: kuleshov-group/mdlm-owt vs native MDLM-DiT (#130 PR0).

The published forward runs ONLY as CUDA + flash-attn varlen + hard-coded bf16
autocast — a pure-fp32 execution of upstream's own code does not exist.  The
evidence chain used here:

1. shim validation (CUDA, bf16): an SDPA attention shim and a rotate-half
   rotary shim are differentially validated against the real flash kernels on
   random tensors, so neither is an unvalidated transcription;
2. fp32 reference (CPU): upstream's own module code runs with only those two
   kernel calls shimmed (patched around each call, never persistently) — on
   CPU the hard-coded ``torch.cuda.amp.autocast`` region is inert, so this is
   upstream math in fp32.  Native fp32 must match it tightly;
3. canonical differential (CUDA, bf16): unpatched upstream (flash varlen)
   vs native under bf16 autocast — kernel-noise-scale disagreement only;
4. seeded sampling swap (CPU, fp32): the same seeded MDLM generation, with the
   native model's forward swapped for the shimmed upstream, must reproduce the
   identical token trajectory.

Stop/go (#130): if this module fails, latent training does not start.

Marker layout: the whole module is ``slow`` (real-checkpoint downloads);
only the two flash-kernel classes are ``gpu`` — the fp32 reference and the
sampling swap are pure CPU and must not be lost to a gpu deselection on a
CPU-only runner.  Maintenance tripwire: the upstream remote code calls the
deprecated ``torch.cuda.amp.autocast``; if a future torch removes it, the
CPU legs die with AttributeError (an environment failure, not a parity
failure).
"""

import contextlib

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.slow

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

REPO = "kuleshov-group/mdlm-owt"
MASK_ID = 50257


# ---------------------------------------------------------------------------
# Shims (test-only): the two flash-attn calls upstream's DDiTBlock makes.
# Both compute in fp32 internally and cast back, so their validation against
# the fused kernels compares at single-rounding scale.
# ---------------------------------------------------------------------------


def sdpa_varlen_qkvpacked(qkv, cu_seqlens, max_seqlen, dropout_p, causal):
    """SDPA stand-in for flash_attn_varlen_qkvpacked_func on UNIFORM seqlens
    (upstream's seqlens=None path builds uniform cu_seqlens)."""
    assert dropout_p == 0.0 and not causal
    batch = cu_seqlens.numel() - 1
    q, k, v = qkv.view(batch, max_seqlen, 3, *qkv.shape[-2:]).float().unbind(2)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
    )
    out = out.transpose(1, 2).reshape(batch * max_seqlen, *qkv.shape[-2:])
    return out.to(qkv.dtype)


def rotary_torch_qkv(qkv, cos, sin):
    """Rotate-half stand-in for flash_attn.layers.rotary.apply_rotary_emb_qkv_
    (non-interleaved / GPT-NeoX convention; q and k rotated, v untouched)."""
    q, k, v = qkv.float().unbind(2)
    half = cos.shape[-1]
    c = cos[None, :, None, :].float()
    s = sin[None, :, None, :].float()

    def rot(x):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)

    return torch.stack([rot(q), rot(k), v], dim=2).to(qkv.dtype)


@contextlib.contextmanager
def flash_kernels_shimmed():
    """Swap the two flash kernels for the shims, strictly for the duration of
    the block — module-level patches must never leak into the canonical leg."""
    import flash_attn.flash_attn_interface as fai
    import flash_attn.layers.rotary as rot

    originals = (fai.flash_attn_varlen_qkvpacked_func, rot.apply_rotary_emb_qkv_)
    fai.flash_attn_varlen_qkvpacked_func = sdpa_varlen_qkvpacked
    rot.apply_rotary_emb_qkv_ = rotary_torch_qkv
    try:
        yield
    finally:
        fai.flash_attn_varlen_qkvpacked_func, rot.apply_rotary_emb_qkv_ = originals


@pytest.fixture(scope="module")
def flash():
    return pytest.importorskip("flash_attn")


@pytest.fixture(scope="module")
def upstream_cpu(flash):
    """Upstream remote code, fp32 weights, CPU. Unpatched — shims are applied
    around individual calls via flash_kernels_shimmed().

    Built by instantiating the remote-code class directly and loading the
    safetensors state dict strict — NOT via from_pretrained: the remote code
    predates transformers 5.x, whose loading machinery requires post_init
    state (``all_tied_weights_keys``) the upstream ``MDLM.__init__`` never
    sets.  The manual path also keeps the reference free of any HF loading
    transformations."""
    import json

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    model_cls = get_class_from_dynamic_module("modeling_mdlm.MDLM", REPO)
    config_cls = get_class_from_dynamic_module("configuration_mdlm.MDLMConfig", REPO)
    with open(hf_hub_download(REPO, "config.json")) as f:
        config = config_cls(**json.load(f))
    model = model_cls(config)
    model.load_state_dict(load_file(hf_hub_download(REPO, "model.safetensors")))
    return model.eval()


@pytest.fixture(scope="module")
def native_cpu():
    from unturtle.models.backbones.mdlm_dit.convert_mdlm_owt import load_mdlm_owt

    return load_mdlm_owt(REPO).eval()


def upstream_logits(model, input_ids):
    timesteps = torch.zeros(
        input_ids.shape[0], device=input_ids.device, dtype=torch.float32
    )
    out = model(input_ids=input_ids, timesteps=timesteps)
    return out if torch.is_tensor(out) else out.logits


def shimmed_upstream_logits(model, input_ids):
    with flash_kernels_shimmed(), torch.no_grad():
        return upstream_logits(model, input_ids)


def parity_batch(device, seq_len=128, seed=7):
    """Real text + random ids + mask corruption, equal lengths (no padding:
    the upstream forward accepts none)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    text = (
        "The history of natural language processing generally started in the "
        "1950s, although work can be found from earlier periods. In 1950, Alan "
        "Turing published an article titled Computing Machinery and Intelligence "
        "which proposed what is now called the Turing test as a criterion of "
        "intelligence. The Georgetown experiment in 1954 involved fully automatic "
        "translation of more than sixty Russian sentences into English. "
    )
    real = tok(text * 3, return_tensors="pt").input_ids[:, :seq_len]
    assert real.shape[1] == seq_len, "prompt shorter than seq_len"
    g = torch.Generator().manual_seed(seed)
    rand = torch.randint(0, 50257, (2, seq_len), generator=g)
    batch = torch.cat([real, real.clone(), rand], dim=0)
    # Mask 50% of the second row and 15% of the third (mask id 50257).
    batch[1, torch.rand(seq_len, generator=g) < 0.5] = MASK_ID
    batch[2, torch.rand(seq_len, generator=g) < 0.15] = MASK_ID
    return batch.to(device)


# ---------------------------------------------------------------------------
# 1. Shim validation — the shims are equivalent to the real flash kernels.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@requires_cuda
class TestShimsMatchFlashKernels:
    def test_sdpa_shim_matches_flash_varlen(self, flash):
        import flash_attn.flash_attn_interface as fai

        torch.manual_seed(0)
        B, S, H, D = 3, 64, 12, 64
        qkv = torch.randn(B * S, 3, H, D, device="cuda", dtype=torch.bfloat16)
        cu = torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device="cuda")
        ref = fai.flash_attn_varlen_qkvpacked_func(qkv, cu, S, 0.0, causal=False)
        got = sdpa_varlen_qkvpacked(qkv, cu, S, 0.0, causal=False)
        assert (ref.float() - got.float()).abs().max().item() < 5e-2

    def test_rotary_shim_matches_flash_fused(self, flash):
        import flash_attn.layers.rotary as rot

        torch.manual_seed(1)
        B, S, H, D = 2, 64, 12, 64
        qkv = torch.randn(B, S, 3, H, D, device="cuda", dtype=torch.bfloat16)
        inv = 1.0 / (10_000 ** (torch.arange(0, D, 2).float() / D))
        freqs = torch.outer(torch.arange(S).float(), inv).cuda()
        cos, sin = freqs.cos(), freqs.sin()
        got = rotary_torch_qkv(qkv.clone(), cos, sin)
        ref = rot.apply_rotary_emb_qkv_(
            qkv.clone(), cos.to(qkv.dtype), sin.to(qkv.dtype)
        )
        assert (ref.float() - got.float()).abs().max().item() < 5e-2


# ---------------------------------------------------------------------------
# 2. fp32 reference: shimmed upstream (CPU) vs native (CPU).
# ---------------------------------------------------------------------------


class TestFp32LogitsParity:
    def test_native_fp32_matches_shimmed_upstream_fp32(self, upstream_cpu, native_cpu):
        batch = parity_batch("cpu")
        ref = shimmed_upstream_logits(upstream_cpu, batch).float()
        with torch.no_grad():
            got = native_cpu(input_ids=batch).logits.float()
        diff = (ref - got).abs().max().item()
        assert diff < 1e-3, f"fp32 logits diverge: max abs diff {diff}"
        assert (ref.argmax(-1) == got.argmax(-1)).float().mean().item() == 1.0

    def test_full_context_length(self, upstream_cpu, native_cpu):
        """The 1024-token anchor: rotary tables and long-range attention at
        the checkpoint's training context."""
        g = torch.Generator().manual_seed(11)
        batch = torch.randint(0, 50257, (1, 1024), generator=g)
        batch[0, torch.rand(1024, generator=g) < 0.3] = MASK_ID
        ref = shimmed_upstream_logits(upstream_cpu, batch).float()
        with torch.no_grad():
            got = native_cpu(input_ids=batch).logits.float()
        diff = (ref - got).abs().max().item()
        assert diff < 1e-3, f"fp32 logits diverge at L=1024: max abs diff {diff}"


# ---------------------------------------------------------------------------
# 3. Canonical differential: unpatched upstream (CUDA bf16 flash) vs native.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@requires_cuda
class TestCanonicalBf16Parity:
    def test_native_bf16_within_the_selfnoise_of_canonical_upstream(
        self, flash, upstream_cpu, native_cpu
    ):
        """v2 rule (#130, declared after the v1 FAIL was recorded on the
        issue): the two bf16 paths are DIFFERENT legitimate executions
        (flash varlen + internal autocast vs flash_attn_func + external
        autocast), so a fixed agreement constant is uncalibratable — v1's
        99% was above what upstream's own bf16 achieves against its own
        fp32 math (98.6%).  Calibrate on self-noise instead:

          flips(cross)   <= flips(up-self) + flips(nat-self)
          maxdiff(cross) <= maxdiff(up-self) + maxdiff(nat-self)
          every cross-disagreeing position is a genuine near-tie in fp32
          (margin < 0.5, vs an all-position median of ~2.5)

        A conversion bug flips confidently-decided positions and shifts
        logits beyond precision noise; self-noise cannot hide it."""
        import flash_attn.flash_attn_interface as fai

        assert fai.flash_attn_varlen_qkvpacked_func is not sdpa_varlen_qkvpacked, (
            "canonical leg must run the real flash kernels"
        )
        batch_cpu = parity_batch("cpu")
        up_fp32 = shimmed_upstream_logits(upstream_cpu, batch_cpu).float()
        with torch.no_grad():
            nat_fp32 = native_cpu(input_ids=batch_cpu).logits.float()

        upstream = upstream_cpu.cuda()
        # Upstream's Rotary caches cos/sin as plain attributes keyed only on
        # seq_len — a cache built during the CPU legs would be served to the
        # CUDA triton kernel (crash) whenever the lengths coincide. Reset it.
        upstream.backbone.rotary_emb.seq_len_cached = None
        native = native_cpu.cuda()
        batch = parity_batch("cuda")
        try:
            with torch.no_grad():
                up_bf16 = upstream_logits(upstream, batch).float().cpu()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    nat_bf16 = native(input_ids=batch).logits.float().cpu()
        finally:
            upstream_cpu.cpu()
            native_cpu.cpu()

        def flips(a, b):
            return int((a.argmax(-1) != b.argmax(-1)).sum())

        def maxdiff(a, b):
            return (a - b).abs().max().item()

        up_self, nat_self = flips(up_bf16, up_fp32), flips(nat_bf16, nat_fp32)
        cross = flips(up_bf16, nat_bf16)
        assert cross <= up_self + nat_self, (
            f"cross argmax flips {cross} exceed combined self-noise "
            f"{up_self}+{nat_self}"
        )
        assert maxdiff(up_bf16, nat_bf16) <= (
            maxdiff(up_bf16, up_fp32) + maxdiff(nat_bf16, nat_fp32)
        ), "cross logit diff exceeds combined self-noise"

        disagree = up_bf16.argmax(-1) != nat_bf16.argmax(-1)
        if bool(disagree.any()):
            top2 = up_fp32[disagree].topk(2, dim=-1).values
            worst_margin = (top2[:, 0] - top2[:, 1]).max().item()
            assert worst_margin < 0.5, (
                f"a confidently-decided position flipped (fp32 margin "
                f"{worst_margin}); that is not precision noise"
            )


# ---------------------------------------------------------------------------
# 4. Seeded sampling parity: forward-swap inside the SAME generation loop.
# ---------------------------------------------------------------------------


class TestSeededSamplingParity:
    def test_forward_swap_reproduces_the_token_trajectory(
        self, upstream_cpu, native_cpu
    ):
        from transformers import AutoTokenizer
        from transformers.modeling_outputs import CausalLMOutputWithPast

        tok = AutoTokenizer.from_pretrained("gpt2")
        prompt = tok("The meaning of life is", return_tensors="pt").input_ids

        torch.manual_seed(1234)
        ours = native_cpu.generate(prompt, algorithm="mdlm", max_new_tokens=48, steps=8)

        original = native_cpu.forward

        def swapped(input_ids=None, attention_mask=None, **kwargs):
            assert attention_mask is None or bool(attention_mask.all())
            return CausalLMOutputWithPast(
                logits=shimmed_upstream_logits(upstream_cpu, input_ids)
            )

        native_cpu.forward = swapped
        try:
            torch.manual_seed(1234)
            theirs = native_cpu.generate(
                prompt, algorithm="mdlm", max_new_tokens=48, steps=8
            )
        finally:
            native_cpu.forward = original

        assert torch.equal(ours, theirs), (
            "the same seeded MDLM sampling diverged when the converted model "
            "was swapped for the (shimmed) upstream reference"
        )

    def test_seeded_generation_is_deterministic(self, native_cpu):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("gpt2")
        prompt = tok("In a shocking finding,", return_tensors="pt").input_ids
        torch.manual_seed(99)
        first = native_cpu.generate(
            prompt, algorithm="mdlm", max_new_tokens=32, steps=8
        )
        torch.manual_seed(99)
        second = native_cpu.generate(
            prompt, algorithm="mdlm", max_new_tokens=32, steps=8
        )
        assert torch.equal(first, second)
