from __future__ import annotations

import math

import torch

from rational_factor.models.basis_functions import BSpline1DBasis
from rational_factor.models.density_model import DensityModel
from rational_factor.models.gram import BetaGram


class StandardNormalDensity(DensityModel):
    """Isotropic standard normal on ``R^d``."""

    def __init__(self, features: int, batch_shape: tuple[int, ...] | int = (1,)):
        super().__init__(features=features, batch_shape=batch_shape)
        self.register_buffer("_anchor", torch.zeros(batch_shape))

    def log_density(self, x: torch.Tensor, **contexts: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.features, "x must have shape (n_data, features)"
        log_z = -0.5 * self.features * math.log(2.0 * math.pi)
        log_p = log_z - 0.5 * (x * x).sum(dim=-1)
        # (n_data,) → (n_data, *batch_shape)
        for _ in self.batch_shape:
            log_p = log_p.unsqueeze(-1)
        return self._clip_log_density(log_p + self._anchor)

    def sample(self, n_samples: int, **contexts: torch.Tensor) -> torch.Tensor:
        return torch.randn(
            n_samples,
            *self.batch_shape,
            self.features,
            device=self._anchor.device,
            dtype=self._anchor.dtype,
        )

    def supremum_bound(self) -> torch.Tensor:
        return self._anchor.new_full(
            self.batch_shape, (2.0 * math.pi) ** (-0.5 * self.features)
        )

    def dtype_device(self):
        return self._anchor.dtype, self._anchor.device


class SeparableBeta(DensityModel):
    """Product of independent Beta(α_i, β_i) densities on [0, 1]^d.

    Concentrations are ``min_concentration * exp(raw)``, so they stay at least
    ``min_concentration`` while remaining trainable. The default floor of 1
    keeps the density bounded, so ``supremum_bound()`` is finite and
    differentiable in the parameters.

    Parameters have shape ``(*batch_shape, features)``. Then ``log_density`` has
    shape ``(n_data, *batch_shape)`` and ``supremum_bound`` has shape
    ``batch_shape``.
    """

    def __init__(
        self,
        features: int,
        alpha: torch.Tensor | float | None = None,
        beta: torch.Tensor | float | None = None,
        batch_shape: tuple[int, ...] | int = (1,),
        min_concentration: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__(features=features, batch_shape=batch_shape)
        if min_concentration < 0:
            raise ValueError("min_concentration must be nonnegative")
        self.min_concentration = min_concentration
        self.eps = eps
        self.raw_alpha = torch.nn.Parameter(self._to_raw(alpha, "alpha"))
        self.raw_beta = torch.nn.Parameter(self._to_raw(beta, "beta"))

    def _to_raw(self, value: torch.Tensor, name: str) -> torch.Tensor:
        floor = self.min_concentration
        if floor > 0:
            if torch.any(value < floor):
                raise ValueError(f"{name} must be at least min_concentration={floor}")
            return (value / floor).log()
        return value.log()

    def concentrations(self) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self.raw_alpha.exp()
        beta = self.raw_beta.exp()
        assert (alpha >= 0).all(), "alpha must be nonnegative"
        assert (beta >= 0).all(), "beta must be nonnegative"
        if self.min_concentration > 0:
            alpha = self.min_concentration * alpha
            beta = self.min_concentration * beta
        return alpha, beta

    def dtype_device(self):
        return self.raw_alpha.dtype, self.raw_alpha.device

    def log_density(self, x: torch.Tensor, **contexts: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.features:
            raise ValueError(f"x must have shape (n_data, {self.features}), got {tuple(x.shape)}")
        alpha, beta = self.concentrations()
        x_c = self._expand_data(x.clamp(self.eps, 1.0 - self.eps))
        log_p = torch.distributions.Beta(alpha, beta).log_prob(x_c).sum(dim=-1)
        return self._clip_log_density(log_p)

    def sample(self, n_samples: int, **contexts: torch.Tensor) -> torch.Tensor:
        alpha, beta = self.concentrations()
        return torch.distributions.Beta(alpha, beta).sample((n_samples,))

    @staticmethod
    def _xlogx(x: torch.Tensor) -> torch.Tensor:
        """``x log x`` with the convention ``0 log 0 = 0`` and a zero subgradient at 0.

        ``torch.xlogy(x, x)`` is NaN in the backward pass at ``x = 0``, which
        breaks ``supremum_bound`` at the Uniform ``Beta(1, 1)`` point used at init.
        """
        safe = torch.where(x > 0, x, torch.ones_like(x))
        return torch.where(x > 0, x * torch.log(safe), torch.zeros_like(x))

    @staticmethod
    def _log_mode_1d(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Exact log-supremum of each 1D Beta(α, β) density on [0, 1].

        For α ≥ 1 and β ≥ 1 the density is bounded and the maximum is

            ((α-1)^{α-1} (β-1)^{β-1} / (α+β-2)^{α+β-2}) / B(α, β)

        with the convention 0 log 0 = 0. If α < 1 or β < 1 the density is
        unbounded and the result is +∞.
        """
        a1 = (alpha - 1.0).clamp(min=0.0)
        b1 = (beta - 1.0).clamp(min=0.0)
        s = a1 + b1
        log_sup = (
            SeparableBeta._xlogx(a1)
            + SeparableBeta._xlogx(b1)
            - SeparableBeta._xlogx(s)
            - BetaGram.log_beta(alpha, beta)
        )
        finite = (alpha >= 1.0) & (beta >= 1.0)
        return torch.where(finite, log_sup, torch.full_like(log_sup, math.inf))

    def supremum_bound(self) -> torch.Tensor:
        alpha, beta = self.concentrations()
        return self._log_mode_1d(alpha, beta).sum(dim=-1).exp()

    def marginal(self, marginal_dims: tuple[int, ...]) -> "SeparableBeta":
        dims = tuple(marginal_dims)
        assert all(0 <= i < self.features for i in dims), "marginal_dims must be in [0, dim)"
        alpha, beta = self.concentrations()
        return SeparableBeta(
            features=len(dims),
            alpha=alpha[..., list(dims)].detach().clone(),
            beta=beta[..., list(dims)].detach().clone(),
            batch_shape=self.batch_shape,
            min_concentration=self.min_concentration,
            eps=self.eps,
        )


class Bernstein1D(DensityModel):
    """Degree-n Bernstein density on [0, 1].

    The joint density is

        p(x) = B(x_s)

    where ``x_s = x[sacrificial_index]`` and ``B`` is the 1D Bernstein PDF

        B(t) = Σ_{k=0}^n c_k binom(n, k) t^k (1-t)^{n-k}

    with nonnegative coefficients ``c_0, …, c_n`` summing to ``n + 1``. Other
    coordinates are independent Uniform[0, 1]. Equivalently, ``B`` is a mixture
    of Beta(k+1, n-k+1) with weights ``c_k / (n + 1)``. The bound ``max_k c_k``
    is a differentiable function of the logits (subgradient through the argmax).

    Logits have shape ``(*batch_shape, degree + 1)``.
    """

    def __init__(
        self,
        features: int,
        degree: int,
        logits: torch.Tensor | None = None,
        batch_shape: tuple[int, ...] | int = (1,),
        sacrificial_index: int = 0,
        eps: float = 1e-6,
    ):
        super().__init__(features=features, batch_shape=batch_shape)
        if degree < 0:
            raise ValueError("degree must be nonnegative")
        if not (0 <= sacrificial_index < features):
            raise ValueError(f"sacrificial_index must be in [0, {features}), got {sacrificial_index}")
        self.degree = degree
        self.sacrificial_index = sacrificial_index
        self.eps = eps
        n_coeff = degree + 1
        param_shape = batch_shape + (n_coeff,)

        if logits is None:
            logits = torch.zeros(param_shape)
        else:
            logits = torch.as_tensor(logits, dtype=torch.float32)
            if tuple(logits.shape) == (n_coeff,):
                logits = logits.reshape((1,) * len(batch_shape) + (n_coeff,)).expand(param_shape).clone()
            elif tuple(logits.shape) != param_shape:
                raise ValueError(
                    f"logits must have shape ({n_coeff},) or {param_shape}, got {tuple(logits.shape)}"
                )

        self.logits = torch.nn.Parameter(logits)
        k = torch.arange(n_coeff, dtype=torch.float32)
        n = torch.tensor(float(degree))
        self.register_buffer(
            "log_binom",
            torch.lgamma(n + 1.0) - torch.lgamma(k + 1.0) - torch.lgamma(n - k + 1.0),
        )

    def coefficients(self) -> torch.Tensor:
        """Bernstein PDF coefficients, shape ``(*batch_shape, degree + 1)``, summing to ``degree + 1``."""
        weights = torch.nn.functional.softmax(self.logits, dim=-1)
        return (self.degree + 1.0) * weights

    def dtype_device(self):
        return self.logits.dtype, self.logits.device

    def log_density(self, x: torch.Tensor, **contexts: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.features, "x must have shape (n_data, features)"
        x_s = x[:, self.sacrificial_index].clamp(self.eps, 1.0 - self.eps)
        k = torch.arange(self.degree + 1, device=x.device, dtype=x.dtype)
        log_b = (
            self.log_binom.to(dtype=x.dtype)
            + k * torch.log(x_s).unsqueeze(-1)
            + (self.degree - k) * torch.log1p(-x_s).unsqueeze(-1)
        )
        # log_b: (n_data, n_coeff); log_c: (*batch_shape, n_coeff)
        log_c = self.coefficients().clamp_min(self.eps).log()
        log_b = log_b.reshape(x.shape[0], *([1] * len(self.batch_shape)), -1)
        log_p = torch.logsumexp(log_c + log_b, dim=-1)
        return self._clip_log_density(log_p)

    def sample(self, n_samples: int, **contexts: torch.Tensor) -> torch.Tensor:
        weights = torch.nn.functional.softmax(self.logits, dim=-1)
        flat_w = weights.reshape(-1, self.degree + 1)
        idx = torch.multinomial(flat_w, n_samples, replacement=True).T
        idx = idx.reshape(n_samples, *self.batch_shape)
        alpha = (idx + 1).to(dtype=self.logits.dtype)
        beta = (self.degree - idx + 1).to(dtype=self.logits.dtype)
        x_s = torch.distributions.Beta(alpha, beta).sample()
        samples = torch.rand(
            n_samples,
            *self.batch_shape,
            self.features,
            device=self.logits.device,
            dtype=self.logits.dtype,
        )
        samples[..., self.sacrificial_index] = x_s
        return samples

    def supremum_bound(self) -> torch.Tensor:
        """Tight coefficient bound: min_k c_k ≤ B(t) ≤ max_k c_k on [0, 1].

        Sharp at the endpoints, since B(0) = c_0 and B(1) = c_n. Other
        coordinates are Uniform[0, 1], so the joint supremum equals that of ``B``.
        """
        return self.coefficients().max(dim=-1).values

    def marginal(self, marginal_dims: tuple[int, ...]) -> "Bernstein1D":
        dims = tuple(marginal_dims)
        assert all(0 <= i < self.features for i in dims), "marginal_dims must be in [0, dim)"
        if self.sacrificial_index not in dims:
            # Uniform on the remaining coordinates: degree-0 Bernstein (constant 1).
            return Bernstein1D(
                features=len(dims),
                degree=0,
                logits=torch.zeros(self.batch_shape + (1,)),
                batch_shape=self.batch_shape,
                sacrificial_index=0,
                eps=self.eps,
            )
        new_s = dims.index(self.sacrificial_index)
        return Bernstein1D(
            features=len(dims),
            degree=self.degree,
            logits=self.logits.detach().clone(),
            batch_shape=self.batch_shape,
            sacrificial_index=new_s,
            eps=self.eps,
        )


class BSpline1D(DensityModel):
    """Open-uniform B-spline density on a sacrificial coordinate.

    The joint density is

        p(x) = S(x_s)

    where ``x_s = x[sacrificial_index]`` and ``S`` is the 1D spline PDF

        S(t) = Σ_{i=0}^{m-1} α_i N_{i,p}(t)

    with open-uniform degree-``p`` B-splines ``N_{i,p}`` on ``[0, 1]`` (a
    partition of unity) and nonnegative coefficients satisfying
    ``Σ_i α_i ∫ N_{i,p} = 1``. Other coordinates are independent Uniform[0, 1].

    Logits parameterize

        α_i = exp(ℓ_i) / Σ_j exp(ℓ_j) μ_j,    μ_j = ∫ N_{j,p},

    so zero logits give the Uniform density (``α_i ≡ 1``). When
    ``n_basis == degree + 1`` this reduces to :class:`Bernstein1D`. Local
    support (``n_basis > degree + 1``) lets a single large ``α_i`` produce a
    sharp bump. The bound ``max_i α_i`` is a differentiable function of the
    logits (subgradient through the argmax).

    Logits have shape ``(*batch_shape, n_basis)``. Knot construction and
    Cox–de Boor evaluation are delegated to
    :class:`~rational_factor.models.basis_functions.BSpline1DBasis`.
    """

    def __init__(
        self,
        features: int,
        n_basis: int,
        degree: int = 3,
        logits: torch.Tensor | None = None,
        batch_shape: tuple[int, ...] | int = (1,),
        sacrificial_index: int = 0,
        eps: float = 1e-6,
    ):
        super().__init__(features=features, batch_shape=batch_shape)
        if degree < 0:
            raise ValueError("degree must be nonnegative")
        if n_basis < degree + 1:
            raise ValueError("n_basis must be at least degree + 1")
        if not (0 <= sacrificial_index < features):
            raise ValueError(f"sacrificial_index must be in [0, {features}), got {sacrificial_index}")
        self.n_basis = n_basis
        self.degree = degree
        self.sacrificial_index = sacrificial_index
        self.eps = eps
        param_shape = batch_shape + (n_basis,)

        if logits is None:
            logits = torch.zeros(param_shape)
        else:
            logits = torch.as_tensor(logits, dtype=torch.float32)
            if tuple(logits.shape) == (n_basis,):
                logits = logits.reshape((1,) * len(batch_shape) + (n_basis,)).expand(param_shape).clone()
            elif tuple(logits.shape) != param_shape:
                raise ValueError(
                    f"logits must have shape ({n_basis},) or {param_shape}, got {tuple(logits.shape)}"
                )

        self.logits = torch.nn.Parameter(logits)
        knots = BSpline1DBasis.open_uniform_knots(n_basis, degree)
        mass = BSpline1DBasis.basis_mass(knots, degree)
        self.register_buffer("knots", knots)
        self.register_buffer("mass", mass)
        self.register_buffer("log_mass", mass.clamp_min(torch.finfo(mass.dtype).tiny).log())

    def coefficients(self) -> torch.Tensor:
        """Spline PDF coefficients ``α``, shape ``(*batch_shape, n_basis)``, with ``α · mass = 1``."""
        log_z = torch.logsumexp(self.logits + self.log_mass, dim=-1, keepdim=True)
        return (self.logits - log_z).exp()

    def mixture_weights(self) -> torch.Tensor:
        """Mixture weights ``w_i = α_i μ_i`` for the normalized bases ``N_i / μ_i``."""
        return torch.nn.functional.softmax(self.logits + self.log_mass, dim=-1)

    def eval_basis(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate ``N_{i,p}(t)``, shape ``(n, n_basis)``."""
        return BSpline1DBasis.eval_basis(
            t.to(dtype=self.knots.dtype, device=self.knots.device),
            self.knots,
            self.degree,
            self.n_basis,
        )

    def dtype_device(self):
        return self.logits.dtype, self.logits.device

    def log_density(self, x: torch.Tensor, **contexts: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.features, "x must have shape (n_data, features)"
        x_s = x[:, self.sacrificial_index].clamp(self.eps, 1.0 - self.eps)
        N = self.eval_basis(x_s)
        # N: (n_data, n_basis); log_alpha: (*batch_shape, n_basis)
        log_alpha = self.coefficients().clamp_min(self.eps).log()
        log_N = N.clamp_min(self.eps).log().reshape(
            x.shape[0], *([1] * len(self.batch_shape)), -1
        )
        log_p = torch.logsumexp(log_alpha + log_N, dim=-1)
        return self._clip_log_density(log_p)

    def sample_normalized_bsplines(
        idx: torch.Tensor,
    ) -> torch.Tensor:
        """Draw from ``N_{idx,p} / μ_{idx}`` by rejection (accept with probability ``N``)."""

    def sample(self, n_samples: int, **contexts: torch.Tensor) -> torch.Tensor:
        """Mixture of normalized B-splines; each component via rejection on its support."""
        
        def _sample_normalized_bsplines(idx: torch.Tensor) -> torch.Tensor:
            p = self.degree
            lo = self.knots[idx]
            hi = self.knots[idx + p + 1]
            n = idx.shape[0]
            out = torch.empty(n, device=idx.device, dtype=self.logits.dtype)
            pending = torch.ones(n, dtype=torch.bool, device=idx.device)
            for _ in range(64):
                if not pending.any():
                    break
                n_pend = int(pending.sum().item())
                u = lo[pending] + (hi[pending] - lo[pending]) * torch.rand(
                    n_pend, device=idx.device, dtype=self.logits.dtype
                )
                N = self.eval_basis(u)
                n_i = N.gather(1, idx[pending].unsqueeze(1)).squeeze(1)
                ok = torch.rand(n_pend, device=idx.device, dtype=self.logits.dtype) <= n_i
                filled_idx = pending.nonzero(as_tuple=False).squeeze(-1)
                out[filled_idx[ok]] = u[ok]
                pending[filled_idx[ok]] = False
            if pending.any():
                u = lo[pending] + (hi[pending] - lo[pending]) * torch.rand(
                    int(pending.sum().item()), device=idx.device, dtype=self.logits.dtype
                )
                out[pending] = u
            return out

        weights = self.mixture_weights()
        flat_w = weights.reshape(-1, self.n_basis)
        idx = torch.multinomial(flat_w, n_samples, replacement=True).T
        idx_flat = idx.reshape(-1)
        x_s_flat = _sample_normalized_bsplines(idx_flat)
        x_s = x_s_flat.reshape(n_samples, *self.batch_shape)
        samples = torch.rand(
            n_samples,
            *self.batch_shape,
            self.features,
            device=self.logits.device,
            dtype=self.logits.dtype,
        )
        samples[..., self.sacrificial_index] = x_s
        return samples

    def supremum_bound(self) -> torch.Tensor:
        """Coefficient bound: ``min α ≤ S(t) ≤ max α`` by partition of unity.

        Sharp when a single basis dominates near a point where that basis is 1
        (e.g. endpoints for the first/last open-uniform spline).
        """
        return self.coefficients().max(dim=-1).values

    def marginal(self, marginal_dims: tuple[int, ...]) -> "BSpline1D":
        dims = tuple(marginal_dims)
        assert all(0 <= i < self.features for i in dims), "marginal_dims must be in [0, dim)"
        if self.sacrificial_index not in dims:
            return BSpline1D(
                features=len(dims),
                n_basis=1,
                degree=0,
                logits=torch.zeros(self.batch_shape + (1,)),
                batch_shape=self.batch_shape,
                sacrificial_index=0,
                eps=self.eps,
            )
        new_s = dims.index(self.sacrificial_index)
        return BSpline1D(
            features=len(dims),
            n_basis=self.n_basis,
            degree=self.degree,
            logits=self.logits.detach().clone(),
            batch_shape=self.batch_shape,
            sacrificial_index=new_s,
            eps=self.eps,
        )

