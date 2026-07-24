"""Gram tensors for vectors of 1D basis functions.

Each basis family gets a ``*Gram`` class that owns its product-integral logic and
exposes a shared n-way ``log_gram`` / ``gram`` API. Add a new family by subclassing
``Basis1DGram`` and implementing ``log_product_integral``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


class Basis1DGram:
    """
    n-way Gram tensors for vectors of 1D basis functions.

    Parameter convention for ``log_gram`` / ``gram``:
        Each positional argument is a sequence of tensors, one per basis vector,
        with shape ``(*batch, n_k)``. Shared leading dims are parameter/batch axes;
        the last dim of each tensor is that vector's basis index. Optional
        ``lows`` / ``highs`` must broadcast to ``(*batch,)``.

    Returns:
        ``(*batch, n_0, ..., n_{m-1})``.
    """

    @staticmethod
    def expand_basis_axes(tensor: torch.Tensor, axis: int, n_axes: int) -> torch.Tensor:
        """Place ``tensor``'s last dim on gram axis ``axis`` among ``n_axes`` basis axes."""
        return tensor[(...,) + tuple(slice(None) if i == axis else None for i in range(n_axes))]

    @classmethod
    def _validate_param_seqs(cls, param_seqs: tuple[Sequence[torch.Tensor], ...]) -> int:
        if len(param_seqs) == 0:
            raise ValueError("expected at least one parameter sequence")
        n_axes = len(param_seqs[0])
        if n_axes == 0:
            raise ValueError("each parameter sequence must contain at least one tensor")
        for seq in param_seqs:
            if len(seq) != n_axes:
                raise ValueError("all parameter sequences must have the same length")

        batch_ndim = param_seqs[0][0].dim() - 1
        batch_shape = param_seqs[0][0].shape[:batch_ndim]
        for p, seq in enumerate(param_seqs):
            for k, tensor in enumerate(seq):
                if tensor.dim() - 1 != batch_ndim:
                    raise ValueError("all parameter tensors must share the same number of leading batch dims")
                if tensor.shape[:batch_ndim] != batch_shape:
                    raise ValueError("all parameter tensors must share the same leading batch shape")
                if p > 0 and tensor.shape != param_seqs[0][k].shape:
                    raise ValueError(
                        f"parameter tensors for basis vector {k} must share the same shape "
                        f"(got {tuple(tensor.shape)} vs {tuple(param_seqs[0][k].shape)})"
                    )
        return n_axes

    @classmethod
    def expand_param_seqs(
        cls,
        *param_seqs: Sequence[torch.Tensor],
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        """Expand each basis vector's last dim onto a distinct gram axis."""
        n_axes = cls._validate_param_seqs(param_seqs)
        return tuple(
            tuple(cls.expand_basis_axes(tensor, k, n_axes) for k, tensor in enumerate(seq))
            for seq in param_seqs
        )

    @classmethod
    def log_product_integral(
        cls,
        *param_seqs: Sequence[torch.Tensor],
        lows: torch.Tensor | None = None,
        highs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Log integral of prod_k basis_k(x; params_k) dx for already-broadcast parameter tensors.

        Subclasses implement the family-specific closed form. All tensors in
        ``param_seqs`` must be mutually broadcastable.
        """
        raise NotImplementedError

    @classmethod
    def log_gram(
        cls,
        *param_seqs: Sequence[torch.Tensor],
        lows: torch.Tensor | None = None,
        highs: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Log of the n-way Gram tensor.

        ``out[..., i0, ..., i_{m-1}] = log integral prod_k basis(x; params[k][..., ik]) dx``.
        """
        n_axes = cls._validate_param_seqs(param_seqs)
        expanded = cls.expand_param_seqs(*param_seqs)
        if lows is not None:
            lows = lows[(...,) + (None,) * n_axes]
        if highs is not None:
            highs = highs[(...,) + (None,) * n_axes]
        return cls.log_product_integral(*expanded, lows=lows, highs=highs, **kwargs)

    @classmethod
    def gram(
        cls,
        *param_seqs: Sequence[torch.Tensor],
        lows: torch.Tensor | None = None,
        highs: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """n-way Gram tensor (see ``log_gram``)."""
        return torch.exp(cls.log_gram(*param_seqs, lows=lows, highs=highs, **kwargs))


class GaussianGram(Basis1DGram):
    """Gram tensors for 1D normal PDF factors N(x | mean, std^2)."""

    @staticmethod
    def dim_integral(
        mu: torch.Tensor,
        std: torch.Tensor,
        lows: torch.Tensor | None = None,
        highs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integral of N(x | mu, std^2) over [lows, highs] (broadcastable)."""
        if lows is None and highs is None:
            return torch.ones_like(mu)
        std = std.clamp_min(torch.finfo(mu.dtype).eps)
        cdf_low = 0.0 if lows is None else torch.special.ndtr((lows - mu) / std)
        cdf_high = 1.0 if highs is None else torch.special.ndtr((highs - mu) / std)
        return cdf_high - cdf_low

    @classmethod
    def log_product_integral(
        cls,
        means: Sequence[torch.Tensor],
        stds: Sequence[torch.Tensor],
        lows: torch.Tensor | None = None,
        highs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Log integral of product_k N(x | means[k], stds[k]^2) over [lows, highs]."""
        if len(means) != len(stds):
            raise ValueError("means and stds must have the same length")
        inv_var = sum(std.reciprocal().square() for std in stds)
        mu_over_var = sum(mu / std.square() for mu, std in zip(means, stds))
        mu_sq_over_var = sum(mu.square() / std.square() for mu, std in zip(means, stds))
        m = mu_over_var / inv_var
        std_m = inv_var.rsqrt()
        log_two_pi = means[0].new_tensor(2.0 * math.pi)
        surplus = mu_sq_over_var - mu_over_var.square() / inv_var
        log_norm = -0.5 * (
            (len(means) - 1) * log_two_pi
            + sum(torch.log(std.square()) for std in stds)
            + torch.log(inv_var)
            + surplus
        )
        domain = cls.dim_integral(m, std_m, lows, highs)
        return log_norm + domain.clamp_min(torch.finfo(means[0].dtype).tiny).log()


class BetaGram(Basis1DGram):
    """Gram tensors for 1D Beta PDF factors on (0, 1)."""

    normalized: bool = True

    @staticmethod
    def log_beta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)

    @staticmethod
    def regularized_incomplete_beta(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Regularized incomplete beta I_x(a, b)."""
        betainc = getattr(torch.special, "betainc", None)
        if betainc is not None:
            return betainc(x, a, b)
        # Torch builds without torch.special.betainc: fall back to SciPy (no autograd).
        import numpy as np
        import scipy.special as sp

        x_np = np.asarray(x.detach().cpu().numpy(), dtype=np.float64)
        a_np = np.asarray(a.detach().cpu().numpy(), dtype=np.float64)
        b_np = np.asarray(b.detach().cpu().numpy(), dtype=np.float64)
        out = sp.betainc(a_np, b_np, x_np)
        return torch.as_tensor(out, dtype=a.dtype, device=a.device)

    @classmethod
    def log_product_integral(
        cls,
        alphas: Sequence[torch.Tensor],
        betas: Sequence[torch.Tensor],
        lows: torch.Tensor | None = None,
        highs: torch.Tensor | None = None,
        normalized: bool | None = None,
    ) -> torch.Tensor:
        """
        Log integral of product_k Beta(x | alphas[k], betas[k]) over [lows, highs].

        For m factors the unnormalized product is x^{A-1}(1-x)^{B-1} with
        ``A = sum(alphas) - (m - 1)`` and ``B = sum(betas) - (m - 1)``.
        """
        if len(alphas) != len(betas):
            raise ValueError("alphas and betas must have the same length")
        if normalized is None:
            normalized = cls.normalized

        m = len(alphas)
        a = sum(alphas) - float(m - 1)
        b = sum(betas) - float(m - 1)
        log_ip = cls.log_beta(a, b)
        if normalized:
            for alpha_k, beta_k in zip(alphas, betas):
                log_ip = log_ip - cls.log_beta(alpha_k, beta_k)

        if lows is None and highs is None:
            return log_ip

        low = a.new_zeros(a.shape) if lows is None else lows
        high = a.new_ones(a.shape) if highs is None else highs
        inc = cls.regularized_incomplete_beta(high, a, b) - cls.regularized_incomplete_beta(low, a, b)
        return log_ip + inc.clamp_min(torch.finfo(a.dtype).tiny).log()
