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

"""Reproduction: Dream bias-aware fast LoRA rejects the dtype its own model produces.

Found while building the #166 row-5 profile. Under the usage the module
docstring documents — `from_pretrained(load_in_4bit=True)` then
`get_peft_model(...)` — the fast QKV path is installed on all 28 layers and then
fails at the first forward:

    RuntimeError: expected mat1 and mat2 to have the same dtype,
                  but got: float != c10::BFloat16

The trigger is the INPUT dtype, not quantization as such:

- the 4-bit weight dequantizes to bf16, and `matmul_lora` multiplies it against
  the activation directly;
- `get_peft_model` runs `prepare_model_for_kbit_training`, which upcasts
  embeddings and norms to fp32 — measured, 143 bf16 parameters before, 535 fp32
  after;
- so the model's real hidden states are fp32, which is precisely what the path
  refuses.

Without 4-bit the same call succeeds, because nothing upcasts and the
activations stay bf16.

The existing test only asserts the callable is INSTALLED
(`tests/test_fast_diffusion_model.py::test_dream_peft_qkv_uses_bias_kernel`); it
never executes a forward, and it uses a tiny non-quantized fixture, so neither
the 4-bit path nor any execution is covered.

RESOLVED by #177: ``unturtle.save.prepare_model_for_kbit_training`` now uses
unsloth semantics (frozen parameters keep their loaded dtype), so the model's
real hidden states stay bf16 and this script's final line reports the mismatch
does not arise. The fp32 probe still fails by design — that is the kernel's
dtype contract, enforced model-wide by ``patch_peft_model`` (an fp32-upcasted
model gets no fast hooks at all, reason ``incompatible_compute_dtype``).

Run::

    .venv/bin/python benchmarks/kernels/repro_dream_4bit_lora_dtype.py
"""

from __future__ import annotations

import torch

from unturtle import FastDiffusionModel
from unturtle.kernels.fast_lora import apply_lora_qkv_with_bias

CHECKPOINT = "Dream-org/Dream-v0-Instruct-7B"
REVISION = "05334cb9faaf763692dcf9d8737c642be2b2a6ae"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def main() -> None:
    model, _tokenizer = FastDiffusionModel.from_pretrained(
        CHECKPOINT,
        revision=REVISION,
        load_in_4bit=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    )
    model = FastDiffusionModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=TARGETS,
        use_gradient_checkpointing=False,
    )
    attn = model.get_decoder().layers[0].self_attn
    print("fast path installed:", attn.apply_qkv is apply_lora_qkv_with_bias)

    hidden = model.config.hidden_size
    for dtype in (torch.bfloat16, torch.float32):
        probe = torch.randn(1, 16, hidden, device="cuda:0", dtype=dtype)
        try:
            query, _key, _value = attn.apply_qkv(attn, probe)
            print(f"  input {str(dtype):17s} -> OK, Q {query.dtype}")
        except RuntimeError as error:
            print(f"  input {str(dtype):17s} -> FAILS: {str(error)[:70]}")

    embeddings = model.get_input_embeddings()
    ids = torch.randint(1, 1000, (1, 16), device="cuda:0")
    produced = embeddings(ids).dtype
    print("model's real hidden-state dtype:", produced)
    print(
        "the documented 4-bit + PEFT setup therefore feeds the path a dtype it rejects"
        if produced is torch.float32
        else "hidden states are bf16 here; the mismatch does not arise"
    )


if __name__ == "__main__":
    main()
