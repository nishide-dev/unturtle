"""
Hybrid-attention conversion on the Tiny-A2D families (#63 slice B).

The conversion is a config flag, not a new model family.  `TinyA2D*Model.forward`
already replaces the causal mask with a bidirectional one and already passes a
caller-supplied 4-D mask through untouched, and slice A established that
`run_attention` consumes such a mask verbatim.  So hybrid attention is one mask
substitution — no new class, no fast-forward patch, no monkeypatching, which
satisfies #63's "no model-private monkeypatch branches" criterion structurally.

The property tested hardest here is that the **default path is unchanged**.
This touches a shipped forward, and a regression there would be far more
expensive than the feature is worth.
"""

import pytest
import torch


def _config(hybrid=False, **kwargs):
    from unturtle.models.conversion.a2d.tiny_a2d.modeling_llama import (
        TinyA2DLlamaConfig,
    )

    return TinyA2DLlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        hybrid_attention=hybrid,
        **kwargs,
    )


def _model(hybrid=False, seed=0, **kwargs):
    from unturtle.models.conversion.a2d.tiny_a2d.modeling_llama import (
        TinyA2DLlamaLMHeadModel,
    )

    torch.manual_seed(seed)
    return TinyA2DLlamaLMHeadModel(_config(hybrid, **kwargs)).eval()


def _batch(batch_size=2, seq_len=8):
    torch.manual_seed(0)
    return {
        "input_ids": torch.randint(1, 64, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }


class TestDefaultBehaviourIsUnchanged:
    """The regression surface. Everything else here is new capability."""

    def test_the_flag_defaults_off(self):
        assert _config().hybrid_attention is False

    def test_output_is_identical_with_the_flag_off(self):
        """Byte-identical, not merely close.

        Same seed, same weights, same batch: enabling the *option* without
        using it must not perturb a single logit.
        """
        batch = _batch()

        with torch.no_grad():
            baseline = _model(hybrid=False, seed=7)(**batch).logits
            flagged = _model(hybrid=True, seed=7)(**batch).logits

        assert torch.equal(baseline, flagged), (
            "enabling hybrid_attention changed the output without "
            "prompt_lengths; the flag must be inert until used"
        )

    def test_prompt_lengths_without_the_flag_are_ignored(self):
        """A caller passing prompt_lengths on an unconverted model.

        Silently applying hybrid attention would change training semantics
        for someone who never opted in.
        """
        batch = _batch()

        with torch.no_grad():
            plain = _model(hybrid=False, seed=7)(**batch).logits
            with_lengths = _model(hybrid=False, seed=7)(
                **batch, prompt_lengths=torch.tensor([3, 3])
            ).logits

        assert torch.equal(plain, with_lengths)


class TestHybridChangesAttention:
    def test_it_differs_from_bidirectional(self):
        """The recipe's whole claim is that this is not uniform bidirectional."""
        batch = _batch()
        prompt_lengths = torch.tensor([3, 3])

        with torch.no_grad():
            bidirectional = _model(hybrid=False, seed=7)(**batch).logits
            hybrid = _model(hybrid=True, seed=7)(
                **batch, prompt_lengths=prompt_lengths
            ).logits

        assert not torch.allclose(bidirectional, hybrid, atol=1e-5), (
            "hybrid attention produced the same output as uniform "
            "bidirectional; the mask is not reaching attention"
        )

    def test_prompt_logits_do_not_depend_on_the_target(self):
        """Equation (3)'s asymmetry, observed end to end through a real model.

        This is the mechanism the paper credits: the corrupted target must not
        perturb prompt representations.  Changing target tokens must leave
        prompt-position logits bit-identical.
        """
        prompt_len = 3
        batch = _batch(batch_size=1, seq_len=8)
        model = _model(hybrid=True, seed=7)

        altered = dict(batch)
        altered["input_ids"] = batch["input_ids"].clone()
        altered["input_ids"][0, prompt_len:] = 5  # rewrite the whole target

        with torch.no_grad():
            first = model(**batch, prompt_lengths=torch.tensor([prompt_len])).logits
            second = model(**altered, prompt_lengths=torch.tensor([prompt_len])).logits

        assert torch.equal(first[:, :prompt_len], second[:, :prompt_len]), (
            "prompt logits changed when the target changed, so the "
            "prompt->target block is not being enforced"
        )

    def test_target_logits_do_depend_on_the_prompt(self):
        """The converse: targets see the whole prompt (eq. 3 case 2)."""
        prompt_len = 3
        batch = _batch(batch_size=1, seq_len=8)
        model = _model(hybrid=True, seed=7)

        altered = dict(batch)
        altered["input_ids"] = batch["input_ids"].clone()
        altered["input_ids"][0, :prompt_len] = 5

        with torch.no_grad():
            first = model(**batch, prompt_lengths=torch.tensor([prompt_len])).logits
            second = model(**altered, prompt_lengths=torch.tensor([prompt_len])).logits

        assert not torch.allclose(
            first[:, prompt_len:], second[:, prompt_len:], atol=1e-5
        ), "target logits ignored a prompt change"

    def test_each_row_uses_its_own_prompt_length(self):
        batch = _batch(batch_size=2, seq_len=8)
        model = _model(hybrid=True, seed=7)

        with torch.no_grad():
            same = model(**batch, prompt_lengths=torch.tensor([3, 3])).logits
            different = model(**batch, prompt_lengths=torch.tensor([3, 5])).logits

        assert torch.equal(same[0], different[0]), "row 0 should be unaffected"
        assert not torch.allclose(same[1], different[1], atol=1e-5), (
            "row 1's prompt length was ignored"
        )


class TestPaddingComposes:
    def test_padding_is_still_excluded(self):
        """Hybrid topology must intersect with padding, not replace it.

        Real tokens attending to padding is the classic silent bug: the loss
        still decreases while embeddings of nothing mix in.
        """
        batch = {
            "input_ids": torch.randint(1, 64, (1, 8)),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]]),
        }
        model = _model(hybrid=True, seed=7)

        altered = dict(batch)
        altered["input_ids"] = batch["input_ids"].clone()
        altered["input_ids"][0, 5:] = 63  # rewrite the padded region

        with torch.no_grad():
            first = model(**batch, prompt_lengths=torch.tensor([2])).logits
            second = model(**altered, prompt_lengths=torch.tensor([2])).logits

        assert torch.equal(first[:, :5], second[:, :5]), (
            "real-token logits changed when padding changed"
        )


class TestPrebuilt4DMasksAreIntersected:
    """A 4-D caller mask carries topology the hybrid mask does not know about.

    Most importantly packed block-diagonal isolation.  Replacing it with the
    hybrid mask would let packed samples attend across their boundaries —
    attention still runs, the loss still decreases, and the contamination is
    invisible.  This is the failure mode CLAUDE.md flags for packed metadata.
    """

    def _packed_mask(self, seq_len=8, split=4):
        from unturtle.models.conversion.a2d.tiny_a2d._hybrid import (
            maybe_build_hybrid_mask,
        )

        blocked = torch.finfo(torch.float32).min
        packed = torch.zeros(1, 1, seq_len, seq_len)
        packed[0, 0, :split, split:] = blocked
        packed[0, 0, split:, :split] = blocked

        class _Config:
            hybrid_attention = True

        return packed, maybe_build_hybrid_mask(
            _Config(),
            torch.tensor([2]),
            packed,
            batch_size=1,
            seq_len=seq_len,
            key_value_length=seq_len,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

    def test_packed_boundaries_survive(self):
        _, combined = self._packed_mask()
        allowed = combined[0, 0] == 0

        assert not bool(allowed[:4, 4:].any()), (
            "the hybrid mask replaced the packed mask; samples would attend "
            "across their boundaries"
        )
        assert not bool(allowed[4:, :4].any())

    def test_the_hybrid_topology_still_applies_inside_a_sample(self):
        """Intersection, not replacement in the other direction either."""
        _, combined = self._packed_mask()
        allowed = combined[0, 0] == 0

        # prompt_lengths=2, so position 0 is prompt and 2..3 are target.
        assert not bool(allowed[0, 2:4].any()), (
            "prompt->target was reopened when intersecting with the packed mask"
        )
        assert bool(allowed[0, 0]), "prompt lost sight of itself"


class TestTheLMHeadPathIsExplicit:
    def test_prompt_lengths_reaches_the_inner_model_through_the_lm_head(self):
        """Pins the path a `transformers` kwarg-filtering change could break.

        `prompt_lengths` is declared on `TinyA2D*Model.forward`, and reaches it
        through the LM head's generic `**kwargs` forwarding.  If upstream ever
        filters unknown kwargs, hybrid attention would silently stop applying
        with no error at all — so assert the effect, not just the call.
        """
        prompt_len = 3
        batch = _batch(batch_size=1, seq_len=8)
        model = _model(hybrid=True, seed=7)

        altered = dict(batch)
        altered["input_ids"] = batch["input_ids"].clone()
        altered["input_ids"][0, prompt_len:] = 5

        with torch.no_grad():
            first = model(**batch, prompt_lengths=torch.tensor([prompt_len])).logits
            second = model(**altered, prompt_lengths=torch.tensor([prompt_len])).logits

        assert torch.equal(first[:, :prompt_len], second[:, :prompt_len]), (
            "prompt_lengths did not reach the inner model through the LM head"
        )


class TestValidation:
    def test_a_prompt_longer_than_the_sequence_is_rejected(self):
        model = _model(hybrid=True)

        with pytest.raises(ValueError, match="prompt_lengths"):
            model(**_batch(batch_size=1, seq_len=8), prompt_lengths=torch.tensor([99]))

    def test_a_batch_size_mismatch_is_rejected(self):
        """Broadcasting a 1-row mask over a 2-row batch would silently apply
        the wrong prompt boundary to every other row."""
        model = _model(hybrid=True)

        with pytest.raises(ValueError, match="prompt_lengths"):
            model(**_batch(batch_size=2, seq_len=8), prompt_lengths=torch.tensor([3]))


class TestKVCache:
    """Hybrid attention and a KV cache are incompatible, loudly.

    A cache makes attention rectangular (`[q_len, kv_len]`) while eq. (3) is
    defined over one sequence.  Building the square mask anyway would silently
    mis-align every row — so the combination raises rather than approximating.

    This is only reachable deliberately: the masked-diffusion generation path
    never supplies `prompt_lengths`, which is why generation is unaffected and
    is asserted below.
    """

    def _cached(self):
        model = _model(hybrid=True, seed=7)
        with torch.no_grad():
            first = model(**_batch(batch_size=1, seq_len=8), use_cache=True)
        return model, first.past_key_values

    def test_incremental_decoding_with_prompt_lengths_raises(self):
        model, cache = self._cached()

        with pytest.raises(ValueError, match="KV cache"), torch.no_grad():
            model(
                input_ids=torch.randint(1, 64, (1, 1)),
                attention_mask=torch.ones(1, 9, dtype=torch.long),
                past_key_values=cache,
                prompt_lengths=torch.tensor([3]),
                use_cache=True,
            )

    def test_generation_without_prompt_lengths_is_unaffected(self):
        """The path that actually runs: no prompt_lengths, so no hybrid mask."""
        model, cache = self._cached()

        with torch.no_grad():
            out = model(
                input_ids=torch.randint(1, 64, (1, 1)),
                attention_mask=torch.ones(1, 9, dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
            )

        assert out.logits.shape == (1, 1, 64)


class TestConfigRoundTrip:
    def test_the_flag_survives_save_and_reload(self):
        """A flag that vanished on reload would silently revert a converted
        model to uniform bidirectional — the failure #63 warns about, in the
        other direction."""
        import tempfile

        from unturtle.models.conversion.a2d.tiny_a2d.modeling_llama import (
            TinyA2DLlamaConfig,
        )

        with tempfile.TemporaryDirectory() as directory:
            _config(hybrid=True).save_pretrained(directory)
            reloaded = TinyA2DLlamaConfig.from_pretrained(directory)

        assert reloaded.hybrid_attention is True


class TestLoRA:
    def test_the_topology_survives_lora(self):
        """#63's explicit criterion: LoRA must not silently restore causality.

        A PEFT wrapper interposes on the forward, so the prompt_lengths kwarg
        and the mask construction both have to survive the indirection.
        """
        pytest.importorskip("peft")
        from peft import LoraConfig, get_peft_model

        prompt_len = 3
        batch = _batch(batch_size=1, seq_len=8)
        base = _model(hybrid=True, seed=7)
        model = get_peft_model(
            base,
            LoraConfig(
                r=4,
                lora_alpha=8,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.0,
            ),
        ).eval()

        altered = dict(batch)
        altered["input_ids"] = batch["input_ids"].clone()
        altered["input_ids"][0, prompt_len:] = 5

        with torch.no_grad():
            first = model(**batch, prompt_lengths=torch.tensor([prompt_len])).logits
            second = model(**altered, prompt_lengths=torch.tensor([prompt_len])).logits

        assert torch.equal(first[:, :prompt_len], second[:, :prompt_len]), (
            "under LoRA the prompt->target block stopped being enforced"
        )


class TestOtherFamilies:
    @pytest.mark.parametrize("family", ["qwen2", "qwen3"])
    def test_qwen_families_support_the_flag_too(self, family):
        """#63 says Qwen/Llama first; all three share the same forward shape."""
        import importlib

        module = importlib.import_module(
            f"unturtle.models.conversion.a2d.tiny_a2d.modeling_{family}"
        )
        config_cls = getattr(module, f"TinyA2D{family.capitalize()}Config")
        model_cls = getattr(module, f"TinyA2D{family.capitalize()}LMHeadModel")

        torch.manual_seed(7)
        model = model_cls(
            config_cls(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=64,
                hybrid_attention=True,
            )
        ).eval()

        batch = _batch(batch_size=1, seq_len=8)
        altered = dict(batch)
        altered["input_ids"] = batch["input_ids"].clone()
        altered["input_ids"][0, 3:] = 5

        with torch.no_grad():
            first = model(**batch, prompt_lengths=torch.tensor([3])).logits
            second = model(**altered, prompt_lengths=torch.tensor([3])).logits

        assert torch.equal(first[:, :3], second[:, :3])
