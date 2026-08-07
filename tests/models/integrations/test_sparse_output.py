"""
Tests for the sparse-output capability (#68 PR C).

#61 wants to skip the ``[B, L, V]`` LM-head GEMM when the masked-diffusion
objective only consumes masked positions:

    hidden [B,L,H] -> gather [M,H] -> project [M,V] -> CE on M targets

That needs two model-specific things — final hidden states *without* the LM
head, and the output projection itself — which must not become another
``model_type`` branch inside ``DiffusionTrainer``.  This PR provides the
access; it does not implement the sparse loss.
"""

import pytest
import torch


def _tiny_a2d_model(vocab_size=64, hidden=16):
    from unturtle.models.conversion.a2d.tiny_a2d.modeling_llama import (
        TinyA2DLlamaConfig,
        TinyA2DLlamaLMHeadModel,
    )

    config = TinyA2DLlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return TinyA2DLlamaLMHeadModel(config)


class TestCapabilityDeclaration:
    def test_tiny_a2d_declares_sparse_output(self):
        from unturtle.models.integrations import find_integration

        for model_type in ("tiny-a2d-llama", "tiny-a2d-qwen2", "tiny-a2d-qwen3"):
            integration = find_integration(model_type)
            assert integration is not None
            assert integration.has_capability("sparse_output_projection"), (
                f"{model_type} should be the first sparse-output target"
            )

    def test_untried_backbones_do_not_claim_it(self):
        """#68 scopes PR C to Tiny-A2D; claiming more would be untested."""
        from unturtle.models.integrations import find_integration

        for model_type in ("llada", "mdlm-dit", "dream", "diffusion_gemma"):
            integration = find_integration(model_type)
            assert integration is not None
            assert not integration.has_capability("sparse_output_projection")


class TestSparseOutputAccess:
    def test_returns_none_for_a_model_without_the_capability(self):
        from unturtle.models.integrations import resolve_sparse_output

        class _Cfg:
            model_type = "mdlm-dit"

        class _Model:
            config = _Cfg()

        assert resolve_sparse_output(_Model()) is None

    def test_tiny_a2d_exposes_hidden_states_and_projection(self):
        from unturtle.models.integrations import resolve_sparse_output

        model = _tiny_a2d_model()
        access = resolve_sparse_output(model)

        assert access is not None
        assert callable(access.hidden_states)
        assert callable(access.project)

    def test_hidden_states_skip_the_lm_head(self):
        """The whole point: no [B, L, V] tensor is materialized."""
        from unturtle.models.integrations import resolve_sparse_output

        model = _tiny_a2d_model()
        model.eval()
        access = resolve_sparse_output(model)

        input_ids = torch.randint(0, 64, (2, 5))
        with torch.no_grad():
            hidden = access.hidden_states(model, input_ids=input_ids)

        assert hidden.shape == (2, 5, model.config.hidden_size)
        assert hidden.shape[-1] != model.config.vocab_size

    def test_lm_head_is_never_called_for_hidden_states(self):
        """Asserted on the head itself, not just on the output shape."""
        from unturtle.models.integrations import resolve_sparse_output

        model = _tiny_a2d_model()
        model.eval()
        access = resolve_sparse_output(model)

        calls = []
        original = model.lm_head.forward
        model.lm_head.forward = lambda *a, **k: calls.append(1) or original(*a, **k)
        try:
            with torch.no_grad():
                access.hidden_states(model, input_ids=torch.randint(0, 64, (1, 4)))
        finally:
            model.lm_head.forward = original

        assert calls == [], "the LM head ran; the sparse path saved nothing"

    def test_projection_matches_the_dense_path(self):
        """Gather-then-project must equal project-then-gather, exactly."""
        from unturtle.models.integrations import resolve_sparse_output

        model = _tiny_a2d_model()
        model.eval()
        access = resolve_sparse_output(model)

        input_ids = torch.randint(0, 64, (2, 6))
        mask = torch.zeros(2, 6, dtype=torch.bool)
        mask[0, 1] = mask[0, 4] = mask[1, 3] = True

        with torch.no_grad():
            dense = model(input_ids=input_ids).logits[mask]

            hidden = access.hidden_states(model, input_ids=input_ids)
            sparse = access.project(model, hidden[mask])

        assert sparse.shape == (3, model.config.vocab_size)
        assert torch.allclose(sparse, dense, atol=1e-5), (
            "sparse projection diverged from the dense LM head"
        )

    def test_projection_is_the_models_own_output_embedding(self):
        """Must use the real head, or tied weights would silently diverge."""
        from unturtle.models.integrations import resolve_sparse_output

        model = _tiny_a2d_model()
        access = resolve_sparse_output(model)

        with torch.no_grad():
            hidden = torch.randn(3, model.config.hidden_size)
            projected = access.project(model, hidden)
            expected = model.get_output_embeddings()(hidden)

        assert torch.equal(projected, expected)


class TestNoTrainerSideModelInspection:
    def test_access_is_reachable_without_naming_a_model_type(self):
        """#61 must not re-introduce model hierarchy knowledge in the trainer."""
        import inspect

        from unturtle.models.integrations import resolve_sparse_output

        # A caller passes the model and gets access or None — no config
        # sniffing, no isinstance ladder, no attribute spelunking.
        signature = inspect.signature(resolve_sparse_output)
        assert list(signature.parameters) == ["model"]

        assert resolve_sparse_output(_tiny_a2d_model()) is not None


class TestGracefulDegradation:
    def test_model_without_output_embeddings_has_no_access(self):
        from unturtle.models.integrations import resolve_sparse_output

        class _Cfg:
            model_type = "tiny-a2d-llama"

        class _Model:
            config = _Cfg()

            def get_output_embeddings(self):
                return None

        assert resolve_sparse_output(_Model()) is None

    def test_model_missing_the_backbone_has_no_access(self):
        from unturtle.models.integrations import resolve_sparse_output

        class _Cfg:
            model_type = "tiny-a2d-llama"

        class _Model:
            config = _Cfg()

            def get_output_embeddings(self):
                return torch.nn.Linear(4, 8)

        # No `.model` backbone to run without the head.
        assert resolve_sparse_output(_Model()) is None


@pytest.mark.parametrize("wrapped", [False, True])
def test_works_through_a_peft_style_wrapper(wrapped):
    """Real training runs are PEFT-wrapped; access must survive that."""
    from unturtle.models.integrations import resolve_sparse_output

    model = _tiny_a2d_model()
    if not wrapped:
        assert resolve_sparse_output(model) is not None
        return

    class _Wrapper:
        def __init__(self, inner):
            self.base_model = inner
            self.config = inner.config

        def get_output_embeddings(self):
            return self.base_model.get_output_embeddings()

        @property
        def model(self):
            return self.base_model.model

    assert resolve_sparse_output(_Wrapper(model)) is not None
