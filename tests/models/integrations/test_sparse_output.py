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

    def test_projection_tracks_updates_to_the_real_head(self):
        """A private copy of the head would drift once weights change.

        Comparing against `get_output_embeddings()(hidden)` alone would be
        tautological — that is literally what the implementation calls. So
        mutate the head's weight and require the projection to follow.
        """
        from unturtle.models.integrations import resolve_sparse_output

        model = _tiny_a2d_model()
        access = resolve_sparse_output(model)
        hidden = torch.randn(3, model.config.hidden_size)

        with torch.no_grad():
            before = access.project(model, hidden).clone()
            model.get_output_embeddings().weight.mul_(2.0)
            after = access.project(model, hidden)

        assert not torch.allclose(before, after), (
            "projection ignored a weight update; it is not using the live head"
        )
        assert torch.allclose(after, before * 2.0, atol=1e-5)


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

        # No decoder backbone to run without the head.
        assert resolve_sparse_output(_Model()) is None

    def test_self_returning_decoder_has_no_access(self):
        """A `get_decoder()` returning the model itself would re-run the head."""
        from unturtle.models.integrations import resolve_sparse_output

        class _Cfg:
            model_type = "tiny-a2d-llama"

        class _Model:
            config = _Cfg()

            def get_output_embeddings(self):
                return torch.nn.Linear(4, 8)

            def get_decoder(self):
                return self

        assert resolve_sparse_output(_Model()) is None


def _peft_wrapped(model):
    import peft

    return peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
        ),
    )


class TestThroughRealPeft:
    """Training runs are PEFT-wrapped, so this is the primary use case.

    A hand-rolled wrapper is not good enough here: PEFT nests as
    ``PeftModel.model -> LM-head model`` (one level shallower than intuition
    suggests), so a stand-in that exposes the backbone at ``.model`` would
    model a hierarchy PEFT never produces and hide a real bug.
    """

    def test_access_resolves(self):
        from unturtle.models.integrations import resolve_sparse_output

        assert resolve_sparse_output(_peft_wrapped(_tiny_a2d_model())) is not None

    def test_hidden_states_are_hidden_sized_not_vocab_sized(self):
        from unturtle.models.integrations import resolve_sparse_output

        model = _peft_wrapped(_tiny_a2d_model())
        model.eval()
        access = resolve_sparse_output(model)

        with torch.no_grad():
            hidden = access.hidden_states(model, input_ids=torch.randint(0, 64, (2, 5)))

        assert hidden.shape == (2, 5, 16), (
            f"got {tuple(hidden.shape)}; vocab-sized output means the LM head ran"
        )

    def test_lm_head_never_runs(self):
        from unturtle.models.integrations import resolve_sparse_output

        model = _peft_wrapped(_tiny_a2d_model())
        model.eval()
        access = resolve_sparse_output(model)

        head = model.get_output_embeddings()
        calls = []
        original = head.forward
        head.forward = lambda *a, **k: calls.append(1) or original(*a, **k)
        try:
            with torch.no_grad():
                access.hidden_states(model, input_ids=torch.randint(0, 64, (1, 4)))
        finally:
            head.forward = original

        assert calls == [], "the LM head ran under PEFT; the sparse path saved nothing"

    def test_projection_matches_the_dense_path(self):
        from unturtle.models.integrations import resolve_sparse_output

        model = _peft_wrapped(_tiny_a2d_model())
        model.eval()
        access = resolve_sparse_output(model)

        input_ids = torch.randint(0, 64, (2, 6))
        mask = torch.zeros(2, 6, dtype=torch.bool)
        mask[0, 2] = mask[1, 5] = True

        with torch.no_grad():
            dense = model(input_ids=input_ids).logits[mask]
            hidden = access.hidden_states(model, input_ids=input_ids)
            sparse = access.project(model, hidden[mask])

        assert torch.allclose(sparse, dense, atol=1e-5)


class TestGradientsFlow:
    """#61 is a *training* path; forward equivalence alone proves nothing."""

    def test_sparse_gradients_match_the_dense_path(self):
        from unturtle.models.integrations import resolve_sparse_output

        input_ids = torch.randint(0, 64, (2, 6))
        mask = torch.zeros(2, 6, dtype=torch.bool)
        mask[0, 1] = mask[0, 3] = mask[1, 4] = True
        targets = torch.randint(0, 64, (int(mask.sum()),))

        torch.manual_seed(0)
        dense_model = _tiny_a2d_model()
        torch.manual_seed(0)
        sparse_model = _tiny_a2d_model()

        dense_logits = dense_model(input_ids=input_ids).logits[mask]
        torch.nn.functional.cross_entropy(dense_logits, targets).backward()

        access = resolve_sparse_output(sparse_model)
        hidden = access.hidden_states(sparse_model, input_ids=input_ids)
        sparse_logits = access.project(sparse_model, hidden[mask])
        torch.nn.functional.cross_entropy(sparse_logits, targets).backward()

        dense_grads = dict(dense_model.named_parameters())
        checked = 0
        for name, param in sparse_model.named_parameters():
            reference = dense_grads[name]
            if reference.grad is None:
                assert param.grad is None, f"{name}: sparse produced a spurious grad"
                continue
            assert param.grad is not None, f"{name}: sparse path dropped the gradient"
            assert torch.allclose(param.grad, reference.grad, atol=1e-6), name
            checked += 1

        assert checked > 0, "no gradients compared; the test proved nothing"

    def test_hidden_states_stay_attached_to_the_graph(self):
        """A stray `.detach()` would silently make training a no-op."""
        from unturtle.models.integrations import resolve_sparse_output

        model = _tiny_a2d_model()
        access = resolve_sparse_output(model)

        hidden = access.hidden_states(model, input_ids=torch.randint(0, 64, (1, 4)))

        assert hidden.requires_grad
        assert hidden.grad_fn is not None


class TestForwardKwargsReachTheBackbone:
    """A dropped attention mask would corrupt every padded batch, silently."""

    @pytest.mark.parametrize("extra_key", ["attention_mask", "position_ids"])
    def test_kwarg_changes_the_result(self, extra_key):
        from unturtle.models.integrations import resolve_sparse_output

        model = _tiny_a2d_model()
        model.eval()
        access = resolve_sparse_output(model)

        input_ids = torch.randint(1, 64, (1, 6))
        if extra_key == "attention_mask":
            extra = torch.tensor([[1, 1, 1, 1, 0, 0]])
        else:
            extra = torch.tensor([[5, 4, 3, 2, 1, 0]])

        with torch.no_grad():
            plain = access.hidden_states(model, input_ids=input_ids)
            with_extra = access.hidden_states(
                model, input_ids=input_ids, **{extra_key: extra}
            )

        assert not torch.allclose(plain, with_extra), (
            f"{extra_key} did not reach the backbone; it is being dropped"
        )

    def test_matches_dense_under_a_padding_mask(self):
        from unturtle.models.integrations import resolve_sparse_output

        model = _tiny_a2d_model()
        model.eval()
        access = resolve_sparse_output(model)

        input_ids = torch.randint(1, 64, (2, 6))
        attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
        mask = torch.zeros(2, 6, dtype=torch.bool)
        mask[0, 2] = mask[1, 4] = True

        with torch.no_grad():
            dense = model(input_ids=input_ids, attention_mask=attention_mask).logits[
                mask
            ]
            hidden = access.hidden_states(
                model, input_ids=input_ids, attention_mask=attention_mask
            )
            sparse = access.project(model, hidden[mask])

        assert torch.allclose(sparse, dense, atol=1e-5)


class TestLoudFailureOnLogitReturningBackbone:
    def test_raises_rather_than_treating_logits_as_hidden_states(self):
        """The failure mode that shipped green before review.

        A backbone returning logits looks exactly like hidden states, so
        accepting it would double-apply the output head.
        """
        from unturtle.models.integrations.sparse_output import (
            _standard_hidden_states,
        )

        class _LogitBackbone(torch.nn.Module):
            def forward(self, **kwargs):
                # A plain tuple, as some upstream backbones return.
                return (torch.randn(1, 4, 64),)

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = _LogitBackbone()

            def get_decoder(self):
                return self.backbone

        with pytest.raises(TypeError, match="last_hidden_state"):
            _standard_hidden_states(_Model(), input_ids=torch.zeros(1, 4).long())
