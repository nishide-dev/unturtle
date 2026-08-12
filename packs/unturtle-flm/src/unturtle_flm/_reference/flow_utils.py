# Ported VERBATIM from the official FLM/FMLM reference utils.py —
# https://github.com/david3684/flm @ a1918d5164e5038e37d0b7a4fb2010ce75b863b3
# (Apache-2.0).  FILE SUBSET, functions unmodified: the oracle's utils.py
# hard-imports training-only dependencies (timm scheduler etc.) at module
# level, so only the flow time-reparameterization functions are extracted.
# DO NOT edit computation here; adaptations live outside _reference/.

from typing import Union

import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import CubicSpline
from scipy.special import log_ndtr

def compute_alpha_exact(gamma: np.ndarray, K: int, n_gh: int = 100, sigma_floor: float = 1e-12, is_diffusion=False) -> np.ndarray:
    """
    Computes q_c (Alpha) from Gamma using Gauss-Hermite integration.
    This is the ground-truth function mapping Gamma -> Alpha.
    """
    gamma = np.asarray(gamma)

    # 1. Standardized means (assuming tau=0, b=1.0 for this conversion)
    sigma = 1.0 - gamma
    if is_diffusion:
        sigma = np.sqrt(sigma)
    sigma = np.maximum(sigma, sigma_floor)
    
    m_c = gamma / sigma
    
    # 2. GH nodes/weights
    x, w = hermgauss(n_gh)
    w = w / np.sqrt(np.pi)
    z_nodes = np.sqrt(2.0) * x

    # 3. Broadcasting
    m_c_expanded = m_c[:, None]   # (B, 1)
    z_expanded = z_nodes[None, :] # (1, n_gh)

    # 4. Compute Log-CDFs
    # L_cu = log(Phi(z + m_c))
    L_cu = log_ndtr(z_expanded + m_c_expanded)

    # 5. Weighted sum
    # log_prod_c = (K - 1) * L_cu
    log_prod_c = (K - 1) * L_cu
    q_c = np.sum(w * np.exp(log_prod_c), axis=-1)
    
    # Debugged. should consider prob. from uniform noise.
    alpha = K/(K-1.) * (q_c - 1./K)

    alpha += (gamma-1) * 1e-10 # minor trick to ensure monotonicity

    alpha = np.clip(alpha, 0.0, 1.0)

    return alpha

def build_luts(K: int, n_points: int = 10000, is_diffusion=False) -> tuple[CubicSpline, CubicSpline]:
    """
    Builds two lookup tables (Splines):
    1. Alpha -> Gamma (Forward)
    2. Gamma -> Alpha (Inverse)
    
    Reverted to Linear (Uniform) spacing.
    Chebyshev nodes concentrate points at 0 and 1, but for large K, the curve 
    is often sigmoid-like (flat at ends, steep in middle). 
    Uniform spacing captures the transition region better.
    """
    # 1. Create Alpha grid using Uniform Spacing
    # Simple linspace covers the whole range evenly.
    gamma_vals = np.linspace(0.0, 1.0, n_points) # cont.
    
    # 2. Compute corresponding Gamma grid (Exact)
    alpha_vals = compute_alpha_exact(gamma_vals, K=K, is_diffusion=is_diffusion) # disc.
    
    # 3. Build Forward Spline (Alpha -> Gamma)
    # Alpha is strictly increasing. Safe.
    lut_g2a = CubicSpline(gamma_vals, alpha_vals)
    
    # 4. Build Inverse Spline (Gamma -> Alpha)
    # Gamma values must be strictly increasing to be 'x' in CubicSpline.
    
    # Sort just in case (though usually monotonic)
    sorted_indices = np.argsort(alpha_vals)
    gamma_sorted = gamma_vals[sorted_indices]
    alpha_sorted = alpha_vals[sorted_indices]
    
    # Remove duplicates in Gamma
    # Duplicates often happen at very low alpha (gamma ~ 1/K) or very high alpha (gamma ~ 1.0)
    unique_alpha, unique_indices = np.unique(alpha_sorted, return_index=True)
    unique_gamma = gamma_sorted[unique_indices]

    # Create Spline
    lut_a2g = CubicSpline(unique_alpha, unique_gamma)
    
    return lut_a2g, lut_g2a

# Initialize LUTs globally (lazy loading or explicit init recommended in real apps, 
# but running here for immediate use)
# Using a default K=50000 as per previous context.

# LUT_A2G, LUT_G2A = build_luts(K=50000)

def alpha_to_gamma(alpha: Union[np.ndarray, torch.tensor], lut: CubicSpline) -> Union[np.ndarray, torch.tensor]:
    """
    Maps Alpha -> Gamma using the LUT.
    """
    if isinstance(alpha, torch.Tensor):
        dtype = alpha.dtype
        gamma = np.clip(lut(alpha.cpu().numpy()), 0.0, 1.0)
        return torch.from_numpy(gamma).to(alpha.device, dtype=dtype)
    else:
        return np.clip(lut(alpha), 0.0, 1.0)

def gamma_to_alpha(gamma: Union[np.ndarray, torch.tensor], lut: CubicSpline) -> Union[np.ndarray, torch.tensor]:
    """
    Maps Gamma -> Alpha using the LUT.
    """
    # Clip result to [0, 1] to avoid spline overshoot
    if isinstance(gamma, torch.Tensor):
        dtype = gamma.dtype
        alpha = np.clip(lut(gamma.cpu().numpy()), 0.0, 1.0)
        return torch.from_numpy(alpha).to(gamma.device, dtype=dtype)
    else:
        return np.clip(lut(gamma), 0.0, 1.0)

