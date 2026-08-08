"""
Tiny end-to-end OPD cycle (#64): every merged piece, one training loop.

The chain closed by this test::

    monotone block decode (commit steps recorded)
      -> replay_rounds -> combine_rounds_one_state_per_block   (#110, #109)
      -> SupervisionState -> SupervisionBuffer                 (#91, #100)
      -> FrozenTeacher(alignment="roll") -> teacher_student_divergence  (#93, #85)
      -> optimizer step on the student

The decode is **monotone by construction** (greedy unmask, one position per
step, block-sequential), so the replay is exact rather than the idealized
reconstruction the #110 docstring warns about for jump-process samplers —
this is the reference's own regime.

**What is asserted, and what deliberately is not.**  The direct objective —
teacher-student divergence measured on the student's own fresh rollout
states — must fall dramatically; measured across 3 seeds it drops 1.37→0.06,
0.80→0.02 and 2.47→0.01.  Decode *quality* (chain consistency) is
deliberately NOT asserted: the warm-started student already saturates it
before OPD (greedy decode of a decent model on a deterministic task), and
distillation toward an imperfect teacher can even reduce it (measured
1.000→0.837 on one seed).  The tiny cycle referees the pipeline and the
objective, not the downstream metric.

**Why the student is warm-started.**  The reference's student is a converted
AR model, never a random init — and the distinction is load-bearing:
measured without the warm start, one of three seeds never escaped a
collapsed decode (constant-token rollouts, consistency 0.000) and its
divergence *increased* (1.24→1.50).  On-policy distillation is bistable from
a random init at this scale, because a collapsed decode yields degenerate
rollout states the teacher cannot rescue.  200 plain DFM steps reproduce the
reference's protocol shape and stabilize every seed.
"""

import pytest
import torch
import torch.nn.functional as F

DATA_VOCAB = 8
MASK_ID = DATA_VOCAB
PROMPT = 4
RESPONSE = 8
LENGTH = PROMPT + RESPONSE
BLOCK = 4


def _corpus(n, generator):
    start = torch.randint(0, DATA_VOCAB, (n, 1), generator=generator)
    return (start + torch.arange(LENGTH)) % DATA_VOCAB


def _monotone_block_decode(model, prompts):
    """Greedy unmask, one position per step, block-sequential.

    Returns ``(ids, step_map)`` — the commit trace is recorded directly,
    since we choose when each position unmasks.  Monotone by construction:
    a committed token is never revisited, so `replay_rounds` reconstructs
    exactly the states this decode visited.
    """
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
def test_the_full_opd_cycle_reduces_divergence_on_fresh_rollouts():
    from transformers import LlamaConfig, LlamaForCausalLM

    from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss
    from unturtle.models.conversion.a2d.tiny_a2d.modeling_llama import (
        TinyA2DLlamaConfig,
        TinyA2DLlamaLMHeadModel,
    )
    from unturtle.post_training import (
        FrozenTeacher,
        SupervisionBuffer,
        combine_rounds_one_state_per_block,
        replay_rounds,
    )
    from unturtle.post_training.divergence import teacher_student_divergence
    from unturtle.processes.discrete_flow import DiscreteFlowProcess, LinearKappa

    generator = torch.Generator().manual_seed(0)

    # --- Teacher: a causal LM that knows the chain task.  Causal on purpose:
    # FrozenTeacher's alignment="roll" exists for exactly this pairing
    # (causal teacher, diffusion student, block_size > 1 — the reference's
    # shipped configuration).
    teacher_config = LlamaConfig(
        vocab_size=DATA_VOCAB + 1,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=LENGTH,
    )
    torch.manual_seed(0)
    teacher = LlamaForCausalLM(teacher_config).train()
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-3)
    for _ in range(300):
        ids = _corpus(32, generator)
        out = teacher(input_ids=ids, labels=ids)
        optimizer.zero_grad()
        out.loss.backward()
        optimizer.step()
    frozen = FrozenTeacher(teacher.eval(), vocab_size=DATA_VOCAB + 1)

    # --- Student: diffusion model, warm-started (see module docstring).
    student_config = TinyA2DLlamaConfig(
        vocab_size=DATA_VOCAB + 1,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=LENGTH,
    )
    torch.manual_seed(1)
    student = TinyA2DLlamaLMHeadModel(student_config).train()
    process = DiscreteFlowProcess(
        vocab_size=DATA_VOCAB + 1, mask_token_id=MASK_ID, source="mask"
    )
    scheduler = LinearKappa()
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    for _ in range(200):
        clean = _corpus(32, generator)
        out = process({"input_ids": clean}, generator=generator)
        x_t = out.model_inputs["input_ids"]
        timesteps = out.objective_inputs["timesteps"]
        loss = discrete_flow_matching_loss(
            student(input_ids=x_t).logits, clean, x_t, timesteps, scheduler=scheduler
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # --- The OPD cycle.
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

    # Every rollout state flowed through the whole chain exactly once.
    assert batches_trained == 100, batches_trained
    assert len(buffer) == 0

    # The direct objective: divergence measured on the student's own fresh
    # rollout states.  Measured 1.37 -> 0.06 / 0.80 -> 0.02 / 2.47 -> 0.01
    # across 3 seeds; both thresholds sit far outside those.
    assert initial_divergence is not None and final_divergence is not None
    assert final_divergence < 0.3, (
        f"final on-rollout divergence {final_divergence:.3f}; the cycle is "
        "not distilling"
    )
    assert final_divergence < initial_divergence / 2, (
        f"divergence moved {initial_divergence:.3f} -> {final_divergence:.3f}"
    )
