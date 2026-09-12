"""MLP-conditional versions of the parametric base densities."""

from __future__ import annotations

import math

import torch

from normalizing_flow.base_distributions import SeparableBeta
from rational_factor.models.density_model import ConditionalDensityModel


def _require_conditioner(conditioner: torch.Tensor | None) -> torch.Tensor:
    if conditioner is None:
        raise ValueError("conditioner is required; got None")
    return conditioner


class ConditionalStandardNormalDensity(ConditionalDensityModel):
    """Conditional isotropic Gaussian ``N(μ(c), I)`` on ``R^d``.

    The MLP maps the conditioner to a mean ``μ`` of shape ``(batch, dim)``.
    """

    def __init__(self, dim: int, conditioner_dim: int, mlp: torch.nn.Module):
        super().__init__(dim=dim, conditioner_dim=conditioner_dim)
        self.mlp = mlp

    def _mean(self, conditioner: torch.Tensor) -> torch.Tensor:
        mean = self.mlp(conditioner)
        if mean.shape != (conditioner.shape[0], self.dim):
            raise ValueError(
                f"mlp output must have shape (batch, {self.dim}), got {tuple(mean.shape)}"
            )
        return mean

    def dtype_device(self):
        p = next(self.mlp.parameters())
        return p.dtype, p.device

    def log_density(self, x: torch.Tensor, *, conditioner: torch.Tensor, **contexts) -> torch.Tensor:
        mean = self._mean(conditioner)
        log_z = -0.5 * self.dim * math.log(2.0 * math.pi)
        return self._clip_log_density(log_z - 0.5 * ((x - mean) ** 2).sum(dim=-1))

    def sample(self, conditioner: torch.Tensor, **contexts) -> torch.Tensor:
        if conditioner.ndim == 1:
            conditioner = conditioner.unsqueeze(0)
        mean = self._mean(conditioner)
        return mean + torch.randn_like(mean)

    def supremum_bound(self, conditioner: torch.Tensor | None = None) -> torch.Tensor:
        conditioner = _require_conditioner(conditioner)
        if conditioner.ndim == 1:
            conditioner = conditioner.unsqueeze(0)
        bound = conditioner.new_tensor((2.0 * math.pi) ** (-0.5 * self.dim))
        return bound.expand(conditioner.shape[0])


class ConditionalSeparableBeta(ConditionalDensityModel):
    """Conditional product of independent Beta densities on ``[0, 1]^d``.

    The MLP maps the conditioner to raw concentration logits of shape
    ``(batch, 2 * n_basis * dim)``, split into ``raw_alpha`` and ``raw_beta``.
    Concentrations are ``min_concentration + softplus(raw)``, so they stay at
    least ``min_concentration`` (default 1 keeps the density bounded).
    """

    def __init__(
        self,
        dim: int,
        conditioner_dim: int,
        mlp: torch.nn.Module,
        n_basis: int = 1,
        min_concentration: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__(dim=dim, conditioner_dim=conditioner_dim)
        if n_basis < 1:
            raise ValueError("n_basis must be at least 1")
        if min_concentration < 0:
            raise ValueError("min_concentration must be nonnegative")
        self.n_basis = n_basis
        self.min_concentration = min_concentration
        self.eps = eps
        self.param_dim = 2 * n_basis * dim
        self.mlp = mlp

    def _raw_params(self, conditioner: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.mlp(conditioner)
        if raw.shape != (conditioner.shape[0], self.param_dim):
            raise ValueError(
                f"mlp output must have shape (batch, {self.param_dim}), got {tuple(raw.shape)}"
            )
        half = self.n_basis * self.dim
        raw_alpha = raw[:, :half]
        raw_beta = raw[:, half:]
        if self.n_basis == 1:
            return raw_alpha, raw_beta
        return (
            raw_alpha.view(-1, self.n_basis, self.dim),
            raw_beta.view(-1, self.n_basis, self.dim),
        )

    def concentrations(self, conditioner: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_alpha, raw_beta = self._raw_params(conditioner)
        if self.min_concentration > 0:
            alpha = self.min_concentration + torch.nn.functional.softplus(raw_alpha)
            beta = self.min_concentration + torch.nn.functional.softplus(raw_beta)
        else:
            alpha = raw_alpha.exp()
            beta = raw_beta.exp()
        return alpha, beta

    def dtype_device(self):
        p = next(self.mlp.parameters())
        return p.dtype, p.device

    def log_density(self, x: torch.Tensor, *, conditioner: torch.Tensor, **contexts) -> torch.Tensor:
        alpha, beta = self.concentrations(conditioner)
        x_c = x.clamp(self.eps, 1.0 - self.eps)
        if x_c.ndim != 2 or x_c.shape[1] != self.dim:
            raise ValueError(f"x must have shape (n_data, {self.dim}), got {tuple(x.shape)}")

        if self.n_basis == 1:
            log_p = torch.distributions.Beta(alpha, beta).log_prob(x_c).sum(dim=-1)
        else:
            # alpha, beta: (batch, n_basis, dim); x: (batch, dim)
            log_p = (
                torch.distributions.Beta(alpha, beta)
                .log_prob(x_c.unsqueeze(1))
                .sum(dim=-1)
            )
        return self._clip_log_density(log_p)

    def sample(self, conditioner: torch.Tensor, **contexts) -> torch.Tensor:
        if conditioner.ndim == 1:
            conditioner = conditioner.unsqueeze(0)
        alpha, beta = self.concentrations(conditioner)
        if self.n_basis == 1:
            return torch.distributions.Beta(alpha, beta).sample()
        # Sample the first basis product when a family is parameterized.
        return torch.distributions.Beta(alpha[:, 0], beta[:, 0]).sample()

    def supremum_bound(self, conditioner: torch.Tensor | None = None) -> torch.Tensor:
        conditioner = _require_conditioner(conditioner)
        if conditioner.ndim == 1:
            conditioner = conditioner.unsqueeze(0)
        alpha, beta = self.concentrations(conditioner)
        # alpha/beta: (batch, dim) or (batch, n_basis, dim)
        log_sup = SeparableBeta._log_mode_1d(alpha, beta).sum(dim=-1).exp()
        return log_sup


class ConditionalBernstein1D(ConditionalDensityModel):
    """Conditional degree-n Bernstein density on a sacrificial coordinate.

    The MLP maps the conditioner to Bernstein logits of shape
    ``(batch, degree + 1)``. The joint conditional density is ``p(x | c) = B(x_s | c)``
    with other coordinates independent Uniform[0, 1].
    """

    def __init__(
        self,
        dim: int,
        conditioner_dim: int,
        degree: int,
        mlp: torch.nn.Module,
        sacrificial_index: int = 0,
        eps: float = 1e-6,
    ):
        super().__init__(dim=dim, conditioner_dim=conditioner_dim)
        if degree < 0:
            raise ValueError("degree must be nonnegative")
        if not (0 <= sacrificial_index < dim):
            raise ValueError(f"sacrificial_index must be in [0, {dim}), got {sacrificial_index}")
        self.degree = degree
        self.sacrificial_index = sacrificial_index
        self.eps = eps
        self.n_coeff = degree + 1
        self.mlp = mlp

        k = torch.arange(self.n_coeff, dtype=torch.float32)
        n = torch.tensor(float(degree))
        self.register_buffer(
            "log_binom",
            torch.lgamma(n + 1.0) - torch.lgamma(k + 1.0) - torch.lgamma(n - k + 1.0),
        )

    def logits(self, conditioner: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(conditioner)
        if logits.shape != (conditioner.shape[0], self.n_coeff):
            raise ValueError(
                f"mlp output must have shape (batch, {self.n_coeff}), got {tuple(logits.shape)}"
            )
        return logits

    def coefficients(self, conditioner: torch.Tensor) -> torch.Tensor:
        """Bernstein PDF coefficients, shape ``(batch, degree + 1)``."""
        weights = torch.nn.functional.softmax(self.logits(conditioner), dim=-1)
        return (self.degree + 1.0) * weights

    def dtype_device(self):
        p = next(self.mlp.parameters())
        return p.dtype, p.device

    def log_density(self, x: torch.Tensor, *, conditioner: torch.Tensor, **contexts) -> torch.Tensor:
        assert x.shape[1] == self.dim, "x must have shape (n_data, dim)"
        x_s = x[:, self.sacrificial_index].clamp(self.eps, 1.0 - self.eps)
        k = torch.arange(self.degree + 1, device=x.device, dtype=x.dtype)
        log_b = (
            self.log_binom.to(dtype=x.dtype)
            + k * torch.log(x_s).unsqueeze(-1)
            + (self.degree - k) * torch.log1p(-x_s).unsqueeze(-1)
        )
        log_c = self.coefficients(conditioner).clamp_min(self.eps).log()
        log_p = torch.logsumexp(log_c + log_b, dim=-1)
        return self._clip_log_density(log_p)

    def sample(self, conditioner: torch.Tensor, **contexts) -> torch.Tensor:
        if conditioner.ndim == 1:
            conditioner = conditioner.unsqueeze(0)
        weights = torch.nn.functional.softmax(self.logits(conditioner), dim=-1)
        idx = torch.multinomial(weights, 1).squeeze(-1)
        alpha = (idx + 1).to(dtype=weights.dtype)
        beta = (self.degree - idx + 1).to(dtype=weights.dtype)
        x_s = torch.distributions.Beta(alpha, beta).sample()
        samples = torch.rand(
            conditioner.shape[0], self.dim, device=conditioner.device, dtype=weights.dtype
        )
        samples[:, self.sacrificial_index] = x_s
        return samples

    def supremum_bound(self, conditioner: torch.Tensor | None = None) -> torch.Tensor:
        conditioner = _require_conditioner(conditioner)
        if conditioner.ndim == 1:
            conditioner = conditioner.unsqueeze(0)
        return self.coefficients(conditioner).max(dim=-1).values
