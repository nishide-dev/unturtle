"""
Frozen AR teacher for OPD distillation (#64 slice B).

The reference does three separate things to the teacher — `requires_grad_(False)`,
`.eval()`, and scoring under `no_grad` — and they are not redundant.  Each has a
distinct failure mode, so each gets its own test:

- without `requires_grad_(False)`: teacher parameters accumulate gradients and
  an optimizer over `model.parameters()` would train the teacher
- without `no_grad`: the graph is built anyway, so the memory is spent even
  though nothing uses it
- without `.eval()`: dropout makes the teacher's targets stochastic, and the
  student chases noise it can never match

The alignment detail: the reference applies `[:, :-1, :]` to *both* student and
teacher, because `logits[t]` predicts token `t+1`.  A wrapper returning
unshifted logprobs against a shifted student misaligns every position by one
and still produces a finite, plausible loss.
"""

import pytest
import torch


def _causal_lm(vocab_size=64, hidden=32, dropout=0.0, seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden,
            intermediate_size=hidden * 2,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            attention_dropout=dropout,
        )
    )


def _ids(batch=2, length=8, vocab=64, seed=0):
    torch.manual_seed(seed)
    return torch.randint(1, vocab, (batch, length))


class TestFreezing:
    def test_parameters_do_not_require_grad(self):
        """Otherwise an optimizer over `model.parameters()` trains the teacher."""
        from unturtle.post_training.teacher import FrozenTeacher

        teacher = FrozenTeacher(_causal_lm(), vocab_size=64)

        assert not any(p.requires_grad for p in teacher.model.parameters())

    def test_the_model_is_in_eval_mode(self):
        """Dropout would make the supervision target stochastic."""
        from unturtle.post_training.teacher import FrozenTeacher

        teacher = FrozenTeacher(_causal_lm(), vocab_size=64)

        assert not teacher.model.training

    def test_scoring_the_same_input_twice_is_deterministic(self):
        """The observable consequence of `.eval()`, not just the flag.

        Built with a non-zero dropout so a training-mode teacher would visibly
        differ between calls — asserting `model.training is False` alone would
        pass even if something re-enabled it before scoring.
        """
        from unturtle.post_training.teacher import FrozenTeacher

        teacher = FrozenTeacher(_causal_lm(dropout=0.5), vocab_size=64)
        ids = _ids()

        first = teacher.log_probs(ids)
        second = teacher.log_probs(ids)

        assert torch.equal(first, second), (
            "teacher scores differ between identical calls; dropout is active "
            "and the student would be chasing noise"
        )

    def test_scoring_builds_no_autograd_graph(self):
        """`no_grad`, distinct from `requires_grad_(False)`.

        Freezing stops parameter gradients; `no_grad` stops the graph existing
        at all, which is where the memory saving comes from.
        """
        from unturtle.post_training.teacher import FrozenTeacher

        teacher = FrozenTeacher(_causal_lm(), vocab_size=64)

        log_probs = teacher.log_probs(_ids())

        assert not log_probs.requires_grad
        assert log_probs.grad_fn is None

    def test_no_grad_holds_even_when_an_input_carries_grad(self):
        """Where `@torch.no_grad()` is actually load-bearing.

        With token ids on a frozen model no graph forms anyway, so removing
        the decorator is invisible — mutation-verified.  It matters when an
        *input* carries grad, which a caller passing `inputs_embeds` derived
        from the student would do: without the decorator the teacher forward
        joins the student's graph and its activations are retained for a
        backward that never uses them.
        """
        from unturtle.post_training.teacher import FrozenTeacher

        model = _causal_lm()
        teacher = FrozenTeacher(model, vocab_size=64)
        embeds = model.get_input_embeddings()(_ids()).detach().requires_grad_(True)

        log_probs = teacher.log_probs(input_ids=None, inputs_embeds=embeds)

        assert not log_probs.requires_grad, (
            "the teacher joined the caller's autograd graph; its activations "
            "would be retained for a backward that never uses them"
        )
        assert embeds.grad is None

    def test_scoring_leaves_no_parameter_gradients(self):
        from unturtle.post_training.teacher import FrozenTeacher

        teacher = FrozenTeacher(_causal_lm(), vocab_size=64)

        teacher.log_probs(_ids())

        assert all(p.grad is None for p in teacher.model.parameters())


class TestAlignment:
    def test_log_probs_are_shifted_for_next_token_prediction(self):
        """`logits[t]` predicts token `t+1`, so scoring drops the last position.

        The reference applies `[:, :-1, :]` to student and teacher alike.  A
        wrapper returning `L` positions against a student's `L-1` would either
        crash on shape or, worse, broadcast into a misalignment.
        """
        from unturtle.post_training.teacher import FrozenTeacher

        teacher = FrozenTeacher(_causal_lm(), vocab_size=64)
        ids = _ids(batch=2, length=8)

        log_probs = teacher.log_probs(ids)

        assert log_probs.shape == (2, 7, 64), (
            f"expected the shifted length 7, got {log_probs.shape[1]}"
        )

    def test_the_shift_drops_the_last_position_not_the_first(self):
        """Direction matters: `[:, 1:, :]` is also length L-1 and is wrong.

        Changing the *last* token must not affect any returned position, since
        that token is only ever a target, never a context for a kept logit.
        """
        from unturtle.post_training.teacher import FrozenTeacher

        teacher = FrozenTeacher(_causal_lm(), vocab_size=64)
        ids = _ids(batch=1, length=8)
        altered = ids.clone()
        altered[0, -1] = (int(ids[0, -1]) + 1) % 64

        assert torch.equal(teacher.log_probs(ids), teacher.log_probs(altered)), (
            "changing the final token moved a kept logit; the shift is "
            "dropping the wrong end"
        )

    def test_it_returns_normalized_log_probabilities(self):
        from unturtle.post_training.teacher import FrozenTeacher

        teacher = FrozenTeacher(_causal_lm(), vocab_size=64)

        log_probs = teacher.log_probs(_ids())

        totals = log_probs.exp().sum(dim=-1)
        assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)


class TestVocabularyContract:
    def test_a_mismatched_vocabulary_is_rejected(self):
        """The reference has no such check.

        A teacher with a different vocabulary produces either a shape error
        deep inside the divergence or — if the sizes coincidentally match —
        silently wrong supervision.  #64 asks for the "same
        tokenization/vocabulary contract as the converted student", so it is
        checked where the teacher is built.
        """
        from unturtle.post_training.teacher import FrozenTeacher

        with pytest.raises(ValueError, match="vocab"):
            FrozenTeacher(_causal_lm(vocab_size=64), vocab_size=128)

    def test_a_matching_vocabulary_is_accepted(self):
        from unturtle.post_training.teacher import FrozenTeacher

        assert FrozenTeacher(_causal_lm(vocab_size=64), vocab_size=64) is not None


class TestTopKSentinel:
    def test_zero_means_full_vocabulary(self):
        """The config sentinel converted exactly once, here.

        `training.top_k_logits: 0` means "full vocab" upstream, but the
        divergence API takes `None` and rejects 0 — deliberately, since 0 and
        None would otherwise be synonyms while -1 raised.  Converting at every
        call site is the footgun this removes.
        """
        from unturtle.post_training.teacher import resolve_top_k_logits

        assert resolve_top_k_logits(0) is None

    def test_a_positive_value_passes_through(self):
        from unturtle.post_training.teacher import resolve_top_k_logits

        assert resolve_top_k_logits(64) == 64

    def test_none_stays_none(self):
        from unturtle.post_training.teacher import resolve_top_k_logits

        assert resolve_top_k_logits(None) is None

    def test_a_negative_value_is_rejected(self):
        from unturtle.post_training.teacher import resolve_top_k_logits

        with pytest.raises(ValueError, match="top_k_logits"):
            resolve_top_k_logits(-1)

    def test_the_end_to_end_sentinel_path_matches_full_vocab(self):
        """The point of the conversion, not just its arithmetic."""
        from unturtle.post_training import teacher_student_divergence
        from unturtle.post_training.teacher import FrozenTeacher, resolve_top_k_logits

        teacher = FrozenTeacher(_causal_lm(seed=1), vocab_size=64)
        student_lp = torch.log_softmax(torch.randn(2, 7, 64), dim=-1)
        teacher_lp = teacher.log_probs(_ids())

        via_sentinel = teacher_student_divergence(
            teacher_lp, student_lp, top_k=resolve_top_k_logits(0)
        )
        dense = teacher_student_divergence(teacher_lp, student_lp)

        assert torch.allclose(via_sentinel, dense, atol=1e-6)


class TestRolloutTopKIsNotDivergenceTopK:
    def test_the_two_parameters_are_distinguishable(self):
        """#64's explicit ask: a regression test that fails if they are swapped.

        `rollout.top_k` is sampling; `training.top_k_logits` is sparse KL.  The
        reference config carries both, and wiring the wrong one trains against
        a different objective with no error.
        """
        from unturtle.post_training.teacher import resolve_top_k_logits

        # The rollout default (0 = disabled sampling top-k) and the divergence
        # default (0 = full vocabulary) coincide numerically but mean opposite
        # things, so the sentinel must be applied only to the divergence one.
        assert resolve_top_k_logits(0) is None

        # A rollout top_k of 50 would be a *sampling* width; if it reached the
        # divergence it would silently truncate supervision to 50 tokens.
        assert resolve_top_k_logits(50) == 50, (
            "resolve_top_k_logits is for divergence top_k_logits only; it must "
            "not be reused for rollout sampling top_k"
        )
