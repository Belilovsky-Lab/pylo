"""Newton-Schulz orthogonalization for PyTorch tensors.

Ported from the JAX implementation in celo2_optax.py.
"""

import torch


def orthogonalize_newton_schulz(
    x: torch.Tensor,
    ns_coeffs: torch.Tensor,
    ns_steps: int = 5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Newton-Schulz orthogonalization.

    Args:
        x: Input tensor with at least 2 dimensions.
        ns_coeffs: Tensor of shape (3,) with Newton-Schulz coefficients.
        ns_steps: Number of Newton-Schulz iterations.
        eps: Epsilon for numerical stability.

    Returns:
        Orthogonalized tensor with same shape as input.
    """
    if x.ndim < 2:
        raise ValueError(f"Input must have >= 2 dims, got {x.shape}")

    transposed = False
    if x.shape[-2] > x.shape[-1]:
        x = x.transpose(-2, -1)
        transposed = True

    x = x / (torch.linalg.norm(x, dim=(-2, -1), keepdim=True) + eps)
    ns_coeffs = ns_coeffs.to(x.dtype)

    for _ in range(ns_steps):
        x_t = x.transpose(-2, -1)
        a = x @ x_t
        b = ns_coeffs[1] * a + ns_coeffs[2] * (a @ a)
        x = ns_coeffs[0] * x + b @ x

    if transposed:
        x = x.transpose(-2, -1)

    return x
