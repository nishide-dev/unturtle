# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Unit tests for packed-attention mask helpers with sliding-window logic."""

import math

import pytest
import torch

from unturtle.utils import attention_dispatch
from unturtle.utils import packing as packing_utils


def _make_seq_info(lengths):
    lengths = torch.tensor(lengths, dtype=torch.int32)
    cu = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(lengths, dim=0, dtype=torch.int32),
        ]
    )
    max_len = int(lengths.max().item())
    return lengths, cu, max_len


def test_sdpa_packed_attention_mask_sliding_window():
    seq_info = _make_seq_info([5, 3])
    mask = packing_utils.build_sdpa_packed_attention_mask(
        seq_info,
        dtype=torch.float32,
        device=torch.device("cpu"),
        sliding_window=3,
    )

    assert mask.shape == (1, 1, 8, 8)

    block_first = mask[0, 0, :5, :5]
    upper = torch.triu(torch.ones_like(block_first), diagonal=1).bool()
    assert torch.all(block_first[upper] == float("-inf"))
    assert block_first[3, 0].item() == float("-inf")
    assert block_first[4, 1].item() == float("-inf")
    assert block_first[4, 2].item() > -math.inf
    assert mask[0, 0, 0, 6].item() == float("-inf")


def test_xformers_block_mask_sliding_window(monkeypatch):
    class _FakeMask:
        def __init__(self, lengths, window=None):
            self.lengths = lengths
            self.window = window

        @classmethod
        def from_seqlens(cls, lengths):
            return cls(tuple(lengths))

        def make_local_attention(self, window_size):
            return _FakeMask(self.lengths, window=window_size)

    monkeypatch.setattr(packing_utils, "_XFormersBlockMask", _FakeMask, raising=False)

    seq_info = _make_seq_info([4, 4])
    mask = packing_utils.build_xformers_block_causal_mask(
        seq_info,
        sliding_window=2,
    )

    assert isinstance(mask, _FakeMask)
    assert mask.window == 2


def test_run_attention_sdpa_passes_sliding_window(monkeypatch):
    seq_info = _make_seq_info([3, 2])
    sliding_window = 2

    original_builder = attention_dispatch.build_sdpa_packed_attention_mask
    captured = {}

    def _capture_builder(seq_info_arg, *, dtype, device, sliding_window=None):
        captured["window"] = sliding_window
        return original_builder(
            seq_info_arg,
            dtype=dtype,
            device=device,
            sliding_window=sliding_window,
        )

    monkeypatch.setattr(
        attention_dispatch,
        "build_sdpa_packed_attention_mask",
        _capture_builder,
    )

    def _fake_sdpa(Q, K, V, **kwargs):
        captured["mask"] = kwargs.get("attn_mask")
        return torch.zeros_like(Q)

    monkeypatch.setattr(attention_dispatch, "scaled_dot_product_attention", _fake_sdpa)

    config = attention_dispatch.AttentionConfig(
        backend=attention_dispatch.SDPA,
        n_kv_heads=1,
        n_groups=1,
    )

    context = attention_dispatch.AttentionContext(
        bsz=1,
        q_len=5,
        kv_seq_len=5,
        n_heads=1,
        head_dim=1,
        requires_grad=False,
        seq_info=seq_info,
        attention_mask=None,
        causal_mask=None,
        sliding_window=sliding_window,
    )

    Q = torch.zeros(1, 1, 5, 1)
    K = torch.zeros_like(Q)
    V = torch.zeros_like(Q)

    attention_dispatch.run_attention(
        config=config,
        context=context,
        Q=Q,
        K=K,
        V=V,
    )

    assert captured["window"] == sliding_window
    mask = captured["mask"]
    assert mask is not None and mask.shape == (1, 1, 5, 5)
    assert mask[0, 0, 4, 1].item() == float("-inf")


def test_run_attention_xformers_passes_sliding_window(monkeypatch):
    seq_info = _make_seq_info([4])
    sliding_window = 3

    class _FakeBias:
        pass

    captured = {}

    def _fake_builder(seq_info_arg, *, sliding_window=None, base_mask=None):
        captured["window"] = sliding_window
        captured["base"] = base_mask
        return _FakeBias()

    def _fake_attention(Q, K, V, attn_bias=None, **_):
        captured["bias"] = attn_bias
        return torch.zeros_like(Q)

    monkeypatch.setattr(
        attention_dispatch, "build_xformers_block_causal_mask", _fake_builder
    )
    monkeypatch.setattr(
        attention_dispatch, "xformers_attention", _fake_attention, raising=False
    )
    monkeypatch.setattr(
        attention_dispatch, "XFORMERS_BLOCK_DIAG_CLS", _FakeBias, raising=False
    )

    config = attention_dispatch.AttentionConfig(
        backend=attention_dispatch.XFORMERS,
        n_kv_heads=1,
        n_groups=1,
    )

    context = attention_dispatch.AttentionContext(
        bsz=1,
        q_len=4,
        kv_seq_len=4,
        n_heads=1,
        head_dim=1,
        requires_grad=False,
        seq_info=seq_info,
        attention_mask=None,
        causal_mask=None,
        sliding_window=sliding_window,
    )

    Q = torch.zeros(1, 1, 4, 1)
    K = torch.zeros_like(Q)
    V = torch.zeros_like(Q)

    attention_dispatch.run_attention(
        config=config,
        context=context,
        Q=Q,
        K=K,
        V=V,
    )

    assert captured["window"] == sliding_window
    assert isinstance(captured["bias"], _FakeBias)


def test_sdpa_packed_bidirectional_mask_has_no_causal_blocks():
    seq_info = _make_seq_info([5, 3])
    mask = packing_utils.build_sdpa_packed_bidirectional_attention_mask(
        seq_info,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert mask.shape == (1, 1, 8, 8)
    # Full bidirectional visibility inside each block (incl. upper triangle).
    assert torch.all(mask[0, 0, :5, :5] == 0.0)
    assert torch.all(mask[0, 0, 5:, 5:] == 0.0)
    # No cross-sample visibility.
    assert torch.all(mask[0, 0, :5, 5:] == float("-inf"))
    assert torch.all(mask[0, 0, 5:, :5] == float("-inf"))


def test_sdpa_packed_bidirectional_mask_symmetric_sliding_window():
    packing_utils.clear_packed_caches()
    seq_info = _make_seq_info([5])
    mask = packing_utils.build_sdpa_packed_bidirectional_attention_mask(
        seq_info,
        dtype=torch.float32,
        device=torch.device("cpu"),
        sliding_window=2,
    )

    block = mask[0, 0]
    # |q - k| < 2 kept, both directions.
    assert block[2, 1].item() == 0.0
    assert block[2, 3].item() == 0.0
    assert block[2, 0].item() == float("-inf")
    assert block[2, 4].item() == float("-inf")


def _bidirectional_config():
    return attention_dispatch.AttentionConfig(
        backend=attention_dispatch.SDPA,
        n_kv_heads=1,
        n_groups=1,
        sdpa_kwargs={"is_causal": False},
        causal=False,
    )


def test_run_attention_bidirectional_packed_mask_is_not_causal(monkeypatch):
    """causal=False + seq_info + no mask must NOT fall back to the causal packed mask."""
    packing_utils.clear_packed_caches()
    seq_info = _make_seq_info([3, 2])
    captured = {}

    def _fake_sdpa(Q, K, V, **kwargs):
        captured["mask"] = kwargs.get("attn_mask")
        captured["is_causal"] = kwargs.get("is_causal")
        return torch.zeros_like(Q)

    monkeypatch.setattr(attention_dispatch, "scaled_dot_product_attention", _fake_sdpa)

    context = attention_dispatch.AttentionContext(
        bsz=1,
        q_len=5,
        kv_seq_len=5,
        n_heads=1,
        head_dim=1,
        requires_grad=False,
        seq_info=seq_info,
        attention_mask=None,
        causal_mask=None,
    )

    Q = torch.zeros(1, 1, 5, 1)
    attention_dispatch.run_attention(
        config=_bidirectional_config(), context=context, Q=Q, K=Q.clone(), V=Q.clone()
    )

    mask = captured["mask"]
    assert mask is not None and mask.shape == (1, 1, 5, 5)
    assert captured["is_causal"] is False
    # Upper triangle within each block must be visible (bidirectional).
    assert mask[0, 0, 0, 2].item() == 0.0
    assert mask[0, 0, 3, 4].item() == 0.0
    # Cross-sample must stay blocked.
    assert mask[0, 0, 0, 3].item() == float("-inf")


def test_run_attention_bidirectional_2d_mask_has_no_causal_and(monkeypatch):
    """causal=False + 2-D padding mask must not AND a causal keep-mask."""
    captured = {}

    def _fake_sdpa(Q, K, V, **kwargs):
        captured["mask"] = kwargs.get("attn_mask")
        return torch.zeros_like(Q)

    monkeypatch.setattr(attention_dispatch, "scaled_dot_product_attention", _fake_sdpa)

    attention_mask = torch.tensor([[1, 1, 1, 0]])  # [B=1, L=4], last is padding
    context = attention_dispatch.AttentionContext(
        bsz=1,
        q_len=4,
        kv_seq_len=4,
        n_heads=1,
        head_dim=1,
        requires_grad=False,
        seq_info=None,
        attention_mask=attention_mask,
        causal_mask=None,
    )

    Q = torch.zeros(1, 1, 4, 1)
    attention_dispatch.run_attention(
        config=_bidirectional_config(), context=context, Q=Q, K=Q.clone(), V=Q.clone()
    )

    mask = captured["mask"]
    assert mask is not None and mask.dtype == torch.bool
    # All real keys visible from every query — including future positions.
    assert torch.all(mask[0, 0, :, :3])
    # Padding key stays hidden.
    assert not mask[0, 0, 0, 3]


def test_run_attention_causal_2d_mask_still_ands_causal(monkeypatch):
    """Default (causal=True) config keeps the legacy causal AND on 2-D masks."""
    captured = {}

    def _fake_sdpa(Q, K, V, **kwargs):
        captured["mask"] = kwargs.get("attn_mask")
        return torch.zeros_like(Q)

    monkeypatch.setattr(attention_dispatch, "scaled_dot_product_attention", _fake_sdpa)

    config = attention_dispatch.AttentionConfig(
        backend=attention_dispatch.SDPA,
        n_kv_heads=1,
        n_groups=1,
    )
    context = attention_dispatch.AttentionContext(
        bsz=1,
        q_len=4,
        kv_seq_len=4,
        n_heads=1,
        head_dim=1,
        requires_grad=False,
        seq_info=None,
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        causal_mask=None,
    )

    Q = torch.zeros(1, 1, 4, 1)
    attention_dispatch.run_attention(
        config=config, context=context, Q=Q, K=Q.clone(), V=Q.clone()
    )

    mask = captured["mask"]
    assert not mask[0, 0, 0, 1], "future position must be masked for causal config"
    assert mask[0, 0, 1, 0]


def test_run_attention_bidirectional_no_mask_disables_is_causal(monkeypatch):
    """causal=False, mask None, q_len == k_len must not default is_causal=True."""
    captured = {}

    def _fake_sdpa(Q, K, V, **kwargs):
        captured["is_causal"] = kwargs.get("is_causal")
        return torch.zeros_like(Q)

    monkeypatch.setattr(attention_dispatch, "scaled_dot_product_attention", _fake_sdpa)

    config = attention_dispatch.AttentionConfig(
        backend=attention_dispatch.SDPA,
        n_kv_heads=1,
        n_groups=1,
        causal=False,  # no sdpa_kwargs — gate must come from the causal field
    )
    context = attention_dispatch.AttentionContext(
        bsz=1,
        q_len=4,
        kv_seq_len=4,
        n_heads=1,
        head_dim=1,
        requires_grad=False,
        seq_info=None,
        attention_mask=None,
        causal_mask=None,
    )

    Q = torch.zeros(1, 1, 4, 1)
    attention_dispatch.run_attention(
        config=config, context=context, Q=Q, K=Q.clone(), V=Q.clone()
    )

    assert captured["is_causal"] is False


def test_run_attention_bidirectional_xformers_packed_falls_back_to_sdpa(monkeypatch):
    """causal=False must never route packed input through the causal xformers bias."""
    packing_utils.clear_packed_caches()
    captured = {}

    def _fake_sdpa(Q, K, V, **kwargs):
        captured["mask"] = kwargs.get("attn_mask")
        return torch.zeros_like(Q)

    def _fail_xformers(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("xformers path must not be used for bidirectional packed")

    monkeypatch.setattr(attention_dispatch, "scaled_dot_product_attention", _fake_sdpa)
    monkeypatch.setattr(
        attention_dispatch, "xformers_attention", _fail_xformers, raising=False
    )

    config = attention_dispatch.AttentionConfig(
        backend=attention_dispatch.XFORMERS,
        n_kv_heads=1,
        n_groups=1,
        causal=False,
    )
    seq_info = _make_seq_info([2, 2])
    context = attention_dispatch.AttentionContext(
        bsz=1,
        q_len=4,
        kv_seq_len=4,
        n_heads=1,
        head_dim=1,
        requires_grad=False,
        seq_info=seq_info,
        attention_mask=None,
        causal_mask=None,
    )

    Q = torch.zeros(1, 1, 4, 1)
    attention_dispatch.run_attention(
        config=config, context=context, Q=Q, K=Q.clone(), V=Q.clone()
    )

    mask = captured["mask"]
    assert mask is not None and mask.shape == (1, 1, 4, 4)
    assert mask[0, 0, 0, 1].item() == 0.0  # bidirectional inside block
    assert mask[0, 0, 0, 2].item() == float("-inf")  # blocked across samples


def test_run_attention_packed_lengths_exceeding_seq_len_raises(monkeypatch):
    """#49: packed lengths summing past the actual sequence length must not pass silently."""
    packing_utils.clear_packed_caches()
    monkeypatch.setattr(
        attention_dispatch,
        "scaled_dot_product_attention",
        lambda Q, K, V, **kwargs: torch.zeros_like(Q),
    )

    seq_info = _make_seq_info([4, 3])  # sums to 7 > q_len 5
    context = attention_dispatch.AttentionContext(
        bsz=1,
        q_len=5,
        kv_seq_len=5,
        n_heads=1,
        head_dim=1,
        requires_grad=False,
        seq_info=seq_info,
        attention_mask=None,
        causal_mask=None,
    )

    Q = torch.zeros(1, 1, 5, 1)
    with pytest.raises(ValueError, match="Packed seq_info lengths"):
        attention_dispatch.run_attention(
            config=_bidirectional_config(),
            context=context,
            Q=Q,
            K=Q.clone(),
            V=Q.clone(),
        )


def test_select_attention_backend_uses_sdpa_on_cpu_even_when_fast_paths_installed(
    monkeypatch,
):
    monkeypatch.setattr(attention_dispatch, "HAS_FLASH_ATTENTION", True)
    monkeypatch.setattr(attention_dispatch, "HAS_XFORMERS", True)

    backend = attention_dispatch.select_attention_backend(
        use_varlen=False,
        device_type="cpu",
    )

    assert backend == attention_dispatch.SDPA


def test_run_attention_flash_varlen_receives_window_and_softcap(monkeypatch):
    seq_info = _make_seq_info([4])
    sliding_window = 3
    softcap = 0.5
    window_tuple = (sliding_window, sliding_window)

    captured = {}

    def _fake_flash_varlen(Q, K, V, cu_q, cu_k, max_q, max_k, **kwargs):
        captured["kwargs"] = kwargs
        return torch.zeros_like(Q)

    monkeypatch.setattr(
        attention_dispatch,
        "flash_attn_varlen_func",
        _fake_flash_varlen,
    )
    monkeypatch.setattr(attention_dispatch, "HAS_FLASH_ATTENTION", True)

    config = attention_dispatch.AttentionConfig(
        backend=attention_dispatch.FLASH_VARLEN,
        n_kv_heads=1,
        n_groups=1,
        flash_varlen_kwargs={
            "dropout_p": 0.0,
            "softmax_scale": 1.0,
            "causal": True,
            "softcap": softcap,
            "window_size": window_tuple,
        },
    )

    context = attention_dispatch.AttentionContext(
        bsz=1,
        q_len=4,
        kv_seq_len=4,
        n_heads=1,
        head_dim=2,
        requires_grad=False,
        seq_info=seq_info,
        attention_mask=None,
        causal_mask=None,
        sliding_window=sliding_window,
    )

    Q = torch.zeros(1, 1, 4, 2)
    K = torch.zeros_like(Q)
    V = torch.zeros_like(Q)

    attention_dispatch.run_attention(
        config=config,
        context=context,
        Q=Q,
        K=K,
        V=V,
    )

    assert captured["kwargs"]["softcap"] == softcap
    assert captured["kwargs"]["window_size"] == window_tuple


"""Unit tests for packed-attention mask helpers with sliding-window logic."""
