# compatibility shim — canonical implementation is in unturtle.kernels
from unturtle.kernels.masked_diffusion_loss import *  # noqa: F401, F403
from unturtle.kernels.masked_diffusion_loss import (  # noqa: F401
    fast_masked_diffusion_loss,
    masked_diffusion_loss_from_timesteps,
)
