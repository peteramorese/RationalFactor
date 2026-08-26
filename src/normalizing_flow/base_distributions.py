"""Parametric separable base densities on the unit cube [0, 1]^d."""

from __future__ import annotations

import math

import torch

from rational_factor.models.density_model import DensityModel
from rational_factor.models.gram import BetaGram


class SeparableBeta(DensityModel):
    """Product of independent Beta(α_i, β_i) densities on [0, 1]^d.

    Concentrations are ``min_concentration * exp(raw)``, so they stay at least
    ``min_concentration`` while remaining trainable. The default floor of 1
    keeps the density bounded, so ``supremum_bound()`` is finite and
    differentiable in the parameters.
    """

    def __init__(
        self,
        dim: int,
        alpha: torch.Tensor | float | None = None,
        beta: torch.Tensor | float | None = None,
        min_concentration: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__(dim=dim)
        if min_concentration < 0:
            raise ValueError("min_concentration must be nonnegative")
        self.min_concentration = min_concentration
        self.eps = eps
        alpha = self._positive_vector(alpha, dim, "alpha")
        beta = self._positive_vector(beta, dim, "beta")
        self.raw_alpha = torch.nn.Parameter(self._to_raw(alpha, "alpha"))
        self.raw_beta = torch.nn.Parameter(self._to_raw(beta, "beta"))

    def _to_raw(self, value: torch.Tensor, name: str) -> torch.Tensor:
        floor = self.min_concentration
        if floor > 0:
            if torch.any(value < floor):
                raise ValueError(f"{name} must be at least min_concentration={floor}")
            return (value / floor).log()
        return value.log()

    @staticmethod
    def _positive_vector(value: torch.Tensor | float | None, dim: int, name: str) -> torch.Tensor:
        if value is None:
            return torch.ones(dim)
        t = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if t.numel() == 1:
            t = t.expand(dim).clone()
        if t.numel() != dim:
            raise ValueError(f"{name} must have 1 or {dim} entries, got {t.numel()}")
        if not torch.all(t > 0):
            raise ValueError(f"{name} must be positive")
        return t

    def concentrations(self) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self.raw_alpha.exp()
        beta = self.raw_beta.exp()
        if self.min_concentration > 0:
            alpha = self.min_concentration * alpha
            beta = self.min_concentration * beta
        return alpha, beta

    def dtype_device(self):
        return self.raw_alpha.dtype, self.raw_alpha.device

    def log_density(self, x: torch.Tensor, **contexts: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.dim, "x must have shape (n_data, dim)"
        alpha, beta = self.concentrations()
        x_c = x.clamp(self.eps, 1.0 - self.eps)
        log_p = torch.distributions.Beta(alpha, beta).log_prob(x_c).sum(dim=-1)
        return self._clip_log_density(log_p)

    def sample(self, n_samples: int, **contexts: torch.Tensor) -> torch.Tensor:
        alpha, beta = self.concentrations()
        return torch.distributions.Beta(alpha, beta).sample((n_samples,))

    @staticmethod
    def _log_mode_1d(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Exact log-supremum of each 1D Beta(α, β) density on [0, 1].

        For α ≥ 1 and β ≥ 1 the density is bounded and the maximum is

            ((α-1)^{α-1} (β-1)^{β-1} / (α+β-2)^{α+β-2}) / B(α, β)

        with the convention 0 log 0 = 0, which also covers the uniform, strictly
        decreasing, and strictly increasing endpoint cases. If α < 1 or β < 1 the
        density is unbounded and the result is +∞.
        """
        a1 = (alpha - 1.0).clamp(min=0.0)
        b1 = (beta - 1.0).clamp(min=0.0)
        s = a1 + b1
        log_sup = (
            torch.xlogy(a1, a1) + torch.xlogy(b1, b1) - torch.xlogy(s, s)
            - BetaGram.log_beta(alpha, beta)
        )
        finite = (alpha >= 1.0) & (beta >= 1.0)
        return torch.where(finite, log_sup, torch.full_like(log_sup, math.inf))

    def supremum_bound(self) -> torch.Tensor:
        alpha, beta = self.concentrations()
        return self._log_mode_1d(alpha, beta).sum().exp()

    def marginal(self, marginal_dims: tuple[int, ...]) -> "SeparableBeta":
        dims = tuple(marginal_dims)
        assert all(0 <= i < self.dim for i in dims), "marginal_dims must be in [0, dim)"
        alpha, beta = self.concentrations()
        return SeparableBeta(
            dim=len(dims),
            alpha=alpha[list(dims)].detach().clone(),
            beta=beta[list(dims)].detach().clone(),
            min_concentration=self.min_concentration,
            eps=self.eps,
        )


class SeparableBernstein(DensityModel):
    """Product of independent degree-n Bernstein polynomial densities on [0, 1]^d.

    Each coordinate has nonnegative Bernstein coefficients ``c_{i,0}, …, c_{i,n}``
    summing to ``n + 1``, so

        p_i(x) = Σ_{k=0}^n c_{i,k} binom(n, k) x^k (1-x)^{n-k}

    integrates to 1. Equivalently, ``p_i`` is a mixture of Beta(k+1, n-k+1)
    with weights ``c_{i,k} / (n + 1)``. The coefficient bound ``max_k c_k`` is
    a differentiable function of the logits (subgradient through the argmax).
    """

    def __init__(
        self,
        dim: int,
        degree: int,
        logits: torch.Tensor | None = None,
        eps: float = 1e-6,
    ):
        super().__init__(dim=dim)
        if degree < 0:
            raise ValueError("degree must be nonnegative")
        self.degree = degree
        self.eps = eps
        n_coeff = degree + 1

        if logits is None:
            logits = torch.zeros(dim, n_coeff)
        else:
            logits = torch.as_tensor(logits, dtype=torch.float32)
            if logits.shape != (dim, n_coeff):
                raise ValueError(f"logits must have shape ({dim}, {n_coeff}), got {tuple(logits.shape)}")

        self.logits = torch.nn.Parameter(logits)
        k = torch.arange(n_coeff, dtype=torch.float32)
        n = torch.tensor(float(degree))
        self.register_buffer(
            "log_binom",
            torch.lgamma(n + 1.0) - torch.lgamma(k + 1.0) - torch.lgamma(n - k + 1.0),
        )

    def coefficients(self) -> torch.Tensor:
        """Bernstein PDF coefficients, shape ``(dim, degree + 1)``, each row sums to ``degree + 1``."""
        weights = torch.nn.functional.softmax(self.logits, dim=-1)
        return (self.degree + 1.0) * weights

    def dtype_device(self):
        return self.logits.dtype, self.logits.device

    def log_density(self, x: torch.Tensor, **contexts: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.dim, "x must have shape (n_data, dim)"
        x_c = x.clamp(self.eps, 1.0 - self.eps)
        k = torch.arange(self.degree + 1, device=x.device, dtype=x.dtype)
        log_b = (
            self.log_binom.to(dtype=x.dtype)
            + k * torch.log(x_c).unsqueeze(-1)
            + (self.degree - k) * torch.log1p(-x_c).unsqueeze(-1)
        )
        log_c = self.coefficients().clamp_min(self.eps).log()
        log_p = torch.logsumexp(log_c.unsqueeze(0) + log_b, dim=-1).sum(dim=-1)
        return self._clip_log_density(log_p)

    def sample(self, n_samples: int, **contexts: torch.Tensor) -> torch.Tensor:
        weights = torch.nn.functional.softmax(self.logits, dim=-1)
        idx = torch.multinomial(weights, n_samples, replacement=True)
        alpha = (idx + 1).to(dtype=self.logits.dtype)
        beta = (self.degree - idx + 1).to(dtype=self.logits.dtype)
        samples = torch.distributions.Beta(alpha, beta).sample()
        return samples.transpose(0, 1)

    def supremum_bound(self) -> torch.Tensor:
        """Tight coefficient bound: min_k c_k ≤ B(x) ≤ max_k c_k on [0, 1].

        Sharp at the endpoints, since B(0) = c_0 and B(1) = c_n. The joint bound
        is the product of the per-coordinate maxima.
        """
        return self.coefficients().max(dim=-1).values.prod()

    def marginal(self, marginal_dims: tuple[int, ...]) -> "SeparableBernstein":
        dims = tuple(marginal_dims)
        assert all(0 <= i < self.dim for i in dims), "marginal_dims must be in [0, dim)"
        return SeparableBernstein(
            dim=len(dims),
            degree=self.degree,
            logits=self.logits.detach().clone()[list(dims), :],
            eps=self.eps,
        )
