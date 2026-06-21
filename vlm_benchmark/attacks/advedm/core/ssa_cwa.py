"""SSA-CWA optimizer for black-box transfer attacks."""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import torch


def dct(x: torch.Tensor, norm: Optional[str] = None) -> torch.Tensor:
    """1D Discrete Cosine Transform, Type II."""
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.fft.fft(v)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc.real * W_r - Vc.imag * W_i
    if norm == "ortho":
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)
    return V


def idct(X: torch.Tensor, norm: Optional[str] = None) -> torch.Tensor:
    """1D Inverse DCT (Type III), such that idct(dct(x)) == x."""
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.complex(real=V_r, imag=V_i)
    v = torch.fft.ifft(V)

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape).real


def dct_2d(x: torch.Tensor, norm: Optional[str] = None) -> torch.Tensor:
    """2D DCT (separable)."""
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X: torch.Tensor, norm: Optional[str] = None) -> torch.Tensor:
    """2D Inverse DCT, such that idct_2d(dct_2d(x)) == x."""
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def ssa_gradient(
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    N: int = 20,
    sigma: float = 16.0 / 255.0,
    rho: float = 0.5,
) -> torch.Tensor:
    """Estimate gradient via N DCT-augmented samples (SSA)."""
    noise_grad = torch.zeros_like(x)

    for _ in range(N):
        gauss = torch.randn_like(x) * sigma
        x_noisy = x + gauss

        x_dct = dct_2d(x_noisy)
        mask = (torch.rand_like(x) * 2 * rho + 1 - rho)
        x_aug = idct_2d(x_dct * mask)

        x_aug = x_aug.detach().requires_grad_(True)
        loss = loss_fn(x_aug)
        loss.backward()
        noise_grad = noise_grad + x_aug.grad.data

    return noise_grad / N


def ssa_cwa_attack(
    x_orig: torch.Tensor,                  # [B, C, H, W] in [0,1]
    loss_fns: List[Callable[[torch.Tensor], torch.Tensor]],
    num_iters: int = 30,
    epsilon: float = 16.0 / 255.0,
    inner_step_size: float = 250.0,
    ksi: float = 16.0 / 255.0 / 5.0,
    mu: float = 1.0,
    N: int = 20,
    sigma: float = 16.0 / 255.0,
    rho: float = 0.5,
    verbose: bool = True,
) -> torch.Tensor:
    """SSA-CWA optimizer: Common Weakness Attack with Spectrum Simulation over surrogate ensemble."""
    B = x_orig.shape[0]

    x_adv = x_orig.clone()
    # One shared inner momentum across all surrogates, accumulated across outer iterations.
    inner_momentum = torch.zeros_like(x_orig)
    outer_momentum = torch.zeros_like(x_orig)

    for t in range(num_iters):
        x_start = x_adv.clone().detach()

        x_new = x_adv.clone().detach()
        for loss_fn in loss_fns:
            g = ssa_gradient(loss_fn, x_new, N=N, sigma=sigma, rho=rho)

            g_norm = torch.norm(
                g.reshape(B, -1), p=2, dim=1
            ).view(B, 1, 1, 1).clamp_min(1e-5)
            inner_momentum = mu * inner_momentum + g / g_norm

            x_new = x_new + inner_step_size * inner_momentum

            x_new = torch.clamp(x_new, 0.0, 1.0)
            x_new = torch.clamp(x_new, x_orig - epsilon, x_orig + epsilon)

        fake_grad = x_new - x_start
        fg_norm = torch.norm(fake_grad, p=1).clamp_min(1e-8)
        outer_momentum = mu * outer_momentum + fake_grad / fg_norm

        x_adv = x_start + ksi * outer_momentum.sign()

        x_adv = torch.clamp(x_adv, x_orig - epsilon, x_orig + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        if verbose and (t % 5 == 0 or t == num_iters - 1):
            pert_linf = (x_adv - x_orig).abs().max().item()
            print(f"  SSA-CWA iter {t:2d}/{num_iters}: "
                  f"L∞={pert_linf:.5f} (budget={epsilon:.5f})")

    return x_adv.detach()
