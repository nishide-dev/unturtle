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

"""Non-Lightning wrapper over the verbatim DiT backbone (#155 Stage 1) —
Unturtle ADAPTATION.

The oracle's `FLMBase` is a Lightning module entangled with trainer/data
machinery; this wrapper reproduces ONLY its inference chain, each method
citing the oracle lines it mirrors:

- `_process_sigma`        — algo.py:805-815 (FLMBase)
- `_process_model_output` — algo.py:817-820 (30*tanh(out/30) -> log_softmax)
- `forward`               — trainer_base.py:311-321 (fp32 autocast around
                            the backbone; sigma_prime for the flow map)
- `_tau_to_t`/`_t_to_tau` — algo.py:883-889 (LUT time reparameterization)

`is_flm_denoiser` / `is_fmlm_flow_map` markers drive the pack's supports
probes; a wrapper never carries both (the flow map is a different model
contract — double time conditioning)."""

from __future__ import annotations

from typing import Any

import torch


class FlmInferenceModel(torch.nn.Module):
    """Shared inference chain; concrete marker set by the loader."""

    def __init__(self, backbone: Any, *, vocab_size: int, length: int) -> None:
        from unturtle_flm._reference.flow_utils import build_luts

        super().__init__()
        self.backbone = backbone
        self.vocab_size = vocab_size
        self.num_tokens = length
        self.lut_a2g, self.lut_g2a = build_luts(K=vocab_size)

    # --- oracle FLMBase._process_sigma (algo.py:805-815), verbatim logic ---
    def _process_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)
        assert sigma.ndim == 2
        sigma = sigma.mean(-1).squeeze()
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        # time_conditioning is True for both frozen configs (flm.yaml /
        # fmlm.yaml), so the zeroing branch never fires for this pack.
        assert sigma.ndim == 1, sigma.shape
        return sigma

    # --- oracle FLMBase._process_model_output (algo.py:817-820) ---
    def _process_model_output(
        self, model_output: torch.Tensor, cap_value: float = 30.0
    ) -> torch.Tensor:
        model_output = cap_value * torch.tanh(model_output / cap_value)
        return model_output.log_softmax(dim=-1)

    # --- oracle trainer_base.forward (trainer_base.py:311-321) ---
    def forward(
        self,
        xt: torch.Tensor,
        sigma: torch.Tensor,
        sigma_prime: torch.Tensor | None = None,
        use_jvp_attn: bool = False,
    ) -> torch.Tensor:
        sigma = self._process_sigma(sigma)
        if sigma_prime is not None:
            sigma_prime = self._process_sigma(sigma_prime)
        device_type = xt.device.type
        autocast_enabled = device_type == "cuda"  # CPU fp32 autocast is a no-op
        with torch.amp.autocast(
            device_type=device_type,
            dtype=torch.float32,
            enabled=autocast_enabled,
        ):
            model_output = self.backbone(
                xt, sigma, sigma_prime, use_jvp_attn=use_jvp_attn
            )
        return self._process_model_output(model_output=model_output)

    # --- oracle FLMBase time reparameterization (algo.py:883-889) ---
    def _tau_to_t(self, tau: torch.Tensor) -> torch.Tensor:
        from unturtle_flm._reference.flow_utils import alpha_to_gamma

        return alpha_to_gamma(tau, self.lut_a2g)

    def _t_to_tau(self, t: torch.Tensor) -> torch.Tensor:
        from unturtle_flm._reference.flow_utils import gamma_to_alpha

        return gamma_to_alpha(t, self.lut_g2a)


__all__ = ["FlmInferenceModel"]
