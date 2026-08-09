"""
Qwen3 conversion as the OPD student — the reference's shipped pairing (#64).

#111 closed the OPD pipeline on a tiny Llama with a random-init student that
needed 200 DFM warm-start steps to escape bistability.  The reference
(OPDLM `convert_qwen_to_bd3lm.py`, `divelab/Qwen3-0.6B-a2d-init`) never
trains from random init: **the student is the frozen teacher's own
checkpoint, converted** — weights bit-preserved, attention bidirectional,
vocabulary untouched, and the mask token reusing an unused slot in the
padded vocab region (`<|MASK|>` = id 151669 for Qwen3-0.6B; the config vocab
151936 exceeds the BPE vocab, so no resize).  This file pins that protocol
on Unturtle's machinery:

- ``convert_ar_model`` on a tiny Qwen3 reproduces the reference recipe's
  observable properties (class, bit-for-bit weights including the
  qwen3-specific ``q_norm``/``k_norm``, unchanged vocab, recorded mask id,
  behavioural bidirectionality);
- the converted-teacher student runs the full OPD cycle (#111's loop) with
  NO warm start — the AR initialization *is* the warm start, which is the
  reference's answer to the bistability #111 measured — and divergence on
  fresh rollouts falls.
"""

import pytest
import torch
import torch.nn.functional as F

DATA_VOCAB = 8
# Reference shape: config vocab exceeds the used vocab (GPU-aligned padding)
# and the mask id reuses an unused padded slot rather than minting a row.
PADDED_VOCAB = 12
MASK_ID = 9
PROMPT = 4
RESPONSE = 8
LENGTH = PROMPT + RESPONSE
BLOCK = 4


def _tiny_qwen3_ar(seed=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=PADDED_VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            max_position_embeddings=LENGTH,
        )
    )


def _corpus(n, generator):
    start = torch.randint(0, DATA_VOCAB, (n, 1), generator=generator)
    return (start + torch.arange(LENGTH)) % DATA_VOCAB


class TestQwen3ConversionMatchesTheReferenceRecipe:
    def test_the_reference_recipe_properties_hold(self):
        """One conversion, every observable the reference recipe promises:
        the TinyA2D class, every tensor bit-for-bit (q_norm/k_norm included —
        the qwen3-specific parameters a llama-shaped conversion would drop),
        the vocabulary untouched, and the mask id recorded on the config."""
        from unturtle.models.conversion.a2d.tiny_a2d import convert_ar_model
        from unturtle.models.conversion.a2d.tiny_a2d.modeling_qwen3 import (
            TinyA2DQwen3LMHeadModel,
        )

        ar = _tiny_qwen3_ar()
        # RMSNorm weights initialize to ones, so a conversion that DROPPED
        # q_norm/k_norm and let the fresh model's own ones-init stand would
        # still compare equal (the params-at-default trap).  Perturb them so
        # bit-for-bit equality actually pins their survival.
        with torch.no_grad():
            for name, parameter in ar.named_parameters():
                if "q_norm" in name or "k_norm" in name:
                    parameter.add_(torch.randn_like(parameter) * 0.1)
        reference = {k: v.clone() for k, v in ar.state_dict().items()}
        assert any("q_norm" in k for k in reference), "fixture lost q_norm"

        student = convert_ar_model(ar, mask_token_id=MASK_ID)

        assert type(student) is TinyA2DQwen3LMHeadModel
        assert student.config.model_type == "tiny-a2d-qwen3"
        assert student.config.vocab_size == PADDED_VOCAB, (
            "explicit mask-id reuse must not resize the padded vocabulary"
        )
        assert student.config.mask_token_id == MASK_ID
        state = student.state_dict()
        assert set(state) == set(reference)
        for name, tensor in reference.items():
            assert torch.equal(state[name], tensor), f"{name} changed"

    def test_the_converted_student_is_bidirectional(self):
        from unturtle.models.conversion.a2d.tiny_a2d import convert_ar_model

        ar = _tiny_qwen3_ar().eval()
        student = convert_ar_model(_tiny_qwen3_ar(), mask_token_id=MASK_ID).eval()

        ids = torch.randint(0, DATA_VOCAB, (1, LENGTH))
        edited = ids.clone()
        edited[0, -1] = (edited[0, -1] + 1) % DATA_VOCAB

        with torch.no_grad():
            ar_moved = not torch.allclose(
                ar(input_ids=ids).logits[0, 0], ar(input_ids=edited).logits[0, 0]
            )
            student_moved = not torch.allclose(
                student(input_ids=ids).logits[0, 0],
                student(input_ids=edited).logits[0, 0],
            )

        assert not ar_moved, "the AR reference is not causal; fixture broken"
        assert student_moved, "the converted qwen3 student is still causal"


def _monotone_block_decode(model, prompts):
    """Copied from tests/test_e2e_opd_cycle.py (#111): greedy unmask, one
    position per step, block-sequential — monotone by construction, so
    `replay_rounds` reconstructs exactly the states this decode visited."""
    batch = prompts.shape[0]
    ids = torch.full((batch, LENGTH), MASK_ID, dtype=torch.long)
    ids[:, :PROMPT] = prompts
    step_map = torch.zeros(batch, RESPONSE, dtype=torch.long)
    step = 0
    with torch.no_grad():
        for block_start in range(PROMPT, LENGTH, BLOCK):
            block_end = min(block_start + BLOCK, LENGTH)
            for _ in range(block_end - block_start):
                logits = model(input_ids=ids).logits
                probs = F.softmax(logits[:, block_start:block_end, :DATA_VOCAB], -1)
                confidence, tokens = probs.max(dim=-1)
                masked = ids[:, block_start:block_end] == MASK_ID
                confidence = confidence.masked_fill(~masked, -1.0)
                position = confidence.argmax(dim=-1)
                for row in range(batch):
                    ids[row, block_start + position[row]] = tokens[row, position[row]]
                    step_map[row, block_start + position[row] - PROMPT] = step
                step += 1
    return ids, step_map


@pytest.mark.slow
def test_the_converted_teacher_runs_the_opd_cycle_without_a_warm_start():
    """The reference protocol end-to-end: student = converted teacher.

    #111 measured random-init bistability (one of three seeds collapsed and
    diverged without 200 DFM warm-start steps).  The reference sidesteps it
    by construction — the student starts as the teacher, so its very first
    rollouts are already coherent.  Assert the same closed loop as #111
    (decode → replay → one-state-per-block → buffer → frozen causal teacher
    with alignment="roll" → divergence) with ZERO warm-start steps.
    """
    from unturtle.models.conversion.a2d.tiny_a2d import convert_ar_model
    from unturtle.post_training import (
        FrozenTeacher,
        SupervisionBuffer,
        combine_rounds_one_state_per_block,
        replay_rounds,
    )
    from unturtle.post_training.divergence import teacher_student_divergence

    generator = torch.Generator().manual_seed(0)

    # Teacher: tiny qwen3 causal LM taught the chain task.
    teacher = _tiny_qwen3_ar(seed=0).train()
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-3)
    for _ in range(300):
        ids = _corpus(32, generator)
        out = teacher(input_ids=ids, labels=ids)
        optimizer.zero_grad()
        out.loss.backward()
        optimizer.step()
    teacher = teacher.eval()
    frozen = FrozenTeacher(teacher, vocab_size=PADDED_VOCAB)

    # Student: the teacher's own checkpoint, converted.  No DFM warm start.
    student = convert_ar_model(teacher, mask_token_id=MASK_ID).train()

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    buffer = SupervisionBuffer(batch_size=8)
    unique = 0
    initial_divergence = None
    final_divergence = None
    batches_trained = 0

    for _ in range(100):
        prompts = _corpus(8, generator)[:, :PROMPT]
        ids, step_map = _monotone_block_decode(student, prompts)
        states = []
        for row in range(8):
            round_states, round_masks = replay_rounds(
                ids[row],
                step_map[row],
                prompt_length=PROMPT,
                block_size=BLOCK,
                mask_token_id=MASK_ID,
            )
            states.append(
                combine_rounds_one_state_per_block(
                    round_states,
                    round_masks,
                    ids[row],
                    sample_id=f"rollout-{unique}",
                    prompt_length=PROMPT,
                    block_size=BLOCK,
                    generator=torch.Generator().manual_seed(unique),
                )
            )
            unique += 1

        for batch in buffer.extend(states):
            teacher_logprobs = frozen.log_probs(batch.input_ids)
            student_logprobs = F.log_softmax(
                student(input_ids=batch.input_ids).logits, dim=-1
            )
            per_position = teacher_student_divergence(
                teacher_logprobs=teacher_logprobs,
                student_logprobs=student_logprobs,
            )
            loss = per_position[batch.supervision_mask].mean()
            if initial_divergence is None:
                initial_divergence = float(loss.detach())
            final_divergence = float(loss.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batches_trained += 1

    assert batches_trained == 100, batches_trained
    assert len(buffer) == 0
    assert initial_divergence is not None and final_divergence is not None
    assert final_divergence < 0.3, (
        f"divergence on fresh rollouts ended at {final_divergence:.3f}"
    )
    assert final_divergence < initial_divergence / 2, (
        f"divergence did not fall: {initial_divergence:.3f} -> {final_divergence:.3f}"
    )
