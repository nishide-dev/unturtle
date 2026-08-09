"""
AR-checkpoint loading for the Tiny-A2D conversion (#63).

Turning a pretrained AR checkpoint into a Tiny-A2D diffusion model is the
conversion recipe's loading half: same architecture, same tensors, different
*behaviour* (bidirectional attention, masked-diffusion objective).  The
regression properties here are the ones #63 fixed from #107's loader
post-mortem:

- the intended upstream head is what actually gets loaded — asserted against
  the checkpoint's ``architectures``, not merely a matching ``model_type``
  (a spoofed model_type produced a chimera in #107);
- converted initialization preserves the AR checkpoint tensors bit-for-bit,
  BEFORE the intentional adaptation (which is behavioural, not structural);
- an incompatible or unmapped architecture is rejected loudly, never
  class-stamped;
- head resolution goes through an Unturtle-owned seam
  (``ar_head_classes()``), the #107 pattern — never a transformers
  monkeypatch, and never a generic Auto* loader.
"""

import pytest
import torch


def _tiny_ar(vocab=32, seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=vocab,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    )


class TestConvertPreservesTheCheckpoint:
    def test_every_ar_tensor_survives_bit_for_bit(self):
        """The conversion is behavioural; the tensors must be untouched.

        A silent re-init anywhere (a post_init re-run, a tied weight rebuilt)
        yields a model that trains fine and benchmarks as if conversion from
        pretrained weights buys nothing — the matched benchmark this issue
        plans would be comparing noise.
        """
        from unturtle.models.conversion.a2d.tiny_a2d.loading import convert_ar_model

        ar = _tiny_ar()
        reference = {k: v.clone() for k, v in ar.state_dict().items()}

        converted = convert_ar_model(ar, mask_token_id=31)

        converted_state = converted.state_dict()
        assert set(converted_state) == set(reference)
        for name, tensor in reference.items():
            assert torch.equal(converted_state[name], tensor), (
                f"{name} changed during conversion"
            )

    def test_the_converted_model_is_the_tiny_a2d_class(self):
        from unturtle.models.conversion.a2d.tiny_a2d.loading import convert_ar_model
        from unturtle.models.conversion.a2d.tiny_a2d.modeling_llama import (
            TinyA2DLlamaLMHeadModel,
        )

        converted = convert_ar_model(_tiny_ar(), mask_token_id=31)

        assert type(converted) is TinyA2DLlamaLMHeadModel
        assert converted.config.model_type == "tiny-a2d-llama"
        assert converted.config.mask_token_id == 31

    def test_attention_became_bidirectional(self):
        """The one *intentional* behavioural change, observed not assumed.

        In the AR model a suffix edit cannot move a prefix position's logits;
        in the converted model it must.  This is what separates "loaded the
        weights" from "converted the model".
        """
        from unturtle.models.conversion.a2d.tiny_a2d.loading import convert_ar_model

        ar = _tiny_ar().eval()
        converted = convert_ar_model(_tiny_ar(), mask_token_id=31).eval()

        ids = torch.randint(0, 31, (1, 8))
        edited = ids.clone()
        edited[0, -1] = (ids[0, -1] + 1) % 31

        with torch.no_grad():
            ar_moved = not torch.allclose(
                ar(input_ids=ids).logits[0, 0], ar(input_ids=edited).logits[0, 0]
            )
            converted_moved = not torch.allclose(
                converted(input_ids=ids).logits[0, 0],
                converted(input_ids=edited).logits[0, 0],
            )

        assert not ar_moved, "the AR reference is not causal; fixture broken"
        assert converted_moved, (
            "a suffix edit did not reach a prefix position; the converted "
            "model is still causal"
        )

    def test_hybrid_flag_is_carried_onto_the_config(self):
        from unturtle.models.conversion.a2d.tiny_a2d.loading import convert_ar_model

        converted = convert_ar_model(
            _tiny_ar(), mask_token_id=31, hybrid_attention=True
        )

        assert converted.config.hybrid_attention is True


class TestMaskTokenEstablishment:
    def test_omitting_the_id_extends_the_vocabulary_by_one(self):
        """No AR checkpoint carries a mask token; the default mints one.

        The new row must not disturb any original row — the preservation
        property extends to the resize.
        """
        from unturtle.models.conversion.a2d.tiny_a2d.loading import convert_ar_model

        ar = _tiny_ar(vocab=32)
        original_embed = ar.get_input_embeddings().weight.detach().clone()

        converted = convert_ar_model(ar)

        assert converted.config.vocab_size == 33
        assert converted.config.mask_token_id == 32
        grown = converted.get_input_embeddings().weight.detach()
        assert torch.equal(grown[:32], original_embed), (
            "extending the vocabulary disturbed the original embedding rows"
        )

    def test_an_explicit_id_reuses_the_vocabulary(self):
        from unturtle.models.conversion.a2d.tiny_a2d.loading import convert_ar_model

        converted = convert_ar_model(_tiny_ar(vocab=32), mask_token_id=5)

        assert converted.config.vocab_size == 32
        assert converted.config.mask_token_id == 5

    def test_an_out_of_range_id_is_rejected(self):
        from unturtle.models.conversion.a2d.tiny_a2d.loading import convert_ar_model

        with pytest.raises(ValueError, match="mask_token_id"):
            convert_ar_model(_tiny_ar(vocab=32), mask_token_id=32)


class TestCheckpointResolution:
    def test_a_saved_checkpoint_loads_through_the_concrete_head(self, tmp_path):
        from unturtle.models.conversion.a2d.tiny_a2d.loading import (
            load_tiny_a2d_from_ar,
        )

        ar = _tiny_ar()
        ar.save_pretrained(tmp_path / "ar")
        reference = {k: v.clone() for k, v in ar.state_dict().items()}

        converted = load_tiny_a2d_from_ar(str(tmp_path / "ar"), mask_token_id=31)

        state = converted.state_dict()
        for name, tensor in reference.items():
            assert torch.equal(state[name], tensor)

    def test_a_spoofed_model_type_is_rejected_via_architectures(self, tmp_path):
        """`model_type` alone proved nothing in #107; `architectures` is the
        contract that names the concrete head.  A checkpoint whose config
        claims `llama` but whose architectures field names a different head
        must be rejected BEFORE any weights load."""
        import json

        from unturtle.models.conversion.a2d.tiny_a2d.loading import (
            load_tiny_a2d_from_ar,
        )

        ar = _tiny_ar()
        ar.save_pretrained(tmp_path / "spoof")
        config_path = tmp_path / "spoof" / "config.json"
        payload = json.loads(config_path.read_text())
        payload["architectures"] = ["SomethingElseForCausalLM"]
        config_path.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="architectures"):
            load_tiny_a2d_from_ar(str(tmp_path / "spoof"), mask_token_id=31)

    def test_an_unmapped_model_type_is_rejected_not_auto_loaded(self, tmp_path):
        """No Tiny-A2D recipe exists for gpt2; falling through to a generic
        Auto* loader would produce exactly the #107 chimera."""
        import json

        from unturtle.models.conversion.a2d.tiny_a2d.loading import (
            load_tiny_a2d_from_ar,
        )

        ar = _tiny_ar()
        ar.save_pretrained(tmp_path / "odd")
        config_path = tmp_path / "odd" / "config.json"
        payload = json.loads(config_path.read_text())
        payload["model_type"] = "gpt2"
        config_path.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="model_type"):
            load_tiny_a2d_from_ar(str(tmp_path / "odd"), mask_token_id=31)

    def test_resolution_goes_through_the_unturtle_seam(self, tmp_path, monkeypatch):
        """The #107 pattern: tests patch `ar_head_classes` on the unturtle
        module, never the transformers module (unsloth replaces
        sys.modules['transformers'], making such patches silently inert)."""
        from unturtle.models.conversion.a2d.tiny_a2d import loading

        ar = _tiny_ar()
        ar.save_pretrained(tmp_path / "seam")
        seen = []
        real = loading.ar_head_classes

        def spy():
            mapping = real()
            seen.append(sorted(mapping))
            return mapping

        monkeypatch.setattr(loading, "ar_head_classes", spy)

        loading.load_tiny_a2d_from_ar(str(tmp_path / "seam"), mask_token_id=31)

        assert seen, "resolution bypassed the seam"
        assert "llama" in seen[0]
