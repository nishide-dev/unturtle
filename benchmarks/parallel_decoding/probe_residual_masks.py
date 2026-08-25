"""#157 — does the shared block-decode loop leave residual mask tokens?

Written to settle a diagnosis during step 2. The step-2 quality run produced
output dominated by one high token id, and the first reading was that the
decoder was failing to unmask positions. That reading was wrong: the dominant
id was ``<|endoftext|>`` (126081), not the mask sentinel (``<|mdm_mask|>``,
126336). This probe pins the correct fact so the mistake is not repeated —
block-decode completes every block, leaving zero residual masks, including at
the exact configuration the step-2 run used (gen 1024 / steps 128 / block 128).

What remains true and is NOT a decoder defect: the cached arms emit EOS almost
immediately under unconditional sampling, so little content survives
``skip_special_tokens=True``.
"""

import json

import torch

from unturtle import FastDiffusionModel

CHECKPOINT = "GSAI-ML/LLaDA-8B-Instruct"
REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"

# (gen_length, steps, block_length) — the last row is the step-2 run's config.
CONFIGS = [
    (128, 128, 128),
    (128, 16, 128),
    (256, 32, 128),
    (1024, 128, 128),
]


def main() -> None:
    model, _tokenizer = FastDiffusionModel.from_pretrained(
        CHECKPOINT,
        revision=REVISION,
        dtype=torch.bfloat16,
        device_map=None,
    )
    model.to("cuda:0").eval()
    mask_id = int(model.config.mask_token_id)

    rows = []
    for gen_length, steps, block_length in CONFIGS:
        prompt = torch.full((2, 1), mask_id, dtype=torch.long, device="cuda:0")
        with torch.no_grad():
            out = model.generate(
                prompt,
                max_length=gen_length + 1,
                steps=steps,
                mask_token_id=mask_id,
                alg="origin",
                temperature=1.0,
                algorithm="block_decode",
                block_length=block_length,
                return_dict=False,
            )
        ids = out[:, 1:]
        num_blocks = gen_length // block_length
        rows.append(
            {
                "gen_length": gen_length,
                "steps": steps,
                "block_length": block_length,
                "steps_per_block": steps // num_blocks,
                "residual_mask_fraction": (ids == mask_id).float().mean().item(),
                "per_block_residual": [
                    (ids[:, b * block_length : (b + 1) * block_length] == mask_id)
                    .float()
                    .mean()
                    .item()
                    for b in range(num_blocks)
                ],
            }
        )
        print(json.dumps(rows[-1]), flush=True)

    worst = max(r["residual_mask_fraction"] for r in rows)
    print(f"mask_token_id={mask_id}  worst residual fraction={worst:.6f}")


if __name__ == "__main__":
    main()
