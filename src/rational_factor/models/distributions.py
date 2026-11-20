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



class SmoothBox(DensityModel):
    """Gaussian-smoothed uniform density on a trainable axis-aligned box.

    A draw is ``U + sigma * epsilon``, where ``U`` is uniform on
    ``[lower, upper]`` and ``epsilon`` is standard normal. This is a properly
    normalized density on R^d, with strictly positive tails. As ``sigma``
    approaches zero it converges to the box-uniform density.

    ``lower``, ``upper``, and ``sigma`` may be scalars, ``(features,)`` tensors,
    or tensors broadcastable to ``(*batch_shape, features)``. Every box has
    separately trainable center, side lengths and per-coordinate sigmas.
    ``bounds()`` and ``sigmas()`` retain the autograd computation graph.
    """

    def __init__(
        self,
        features: int,
        lower: torch.Tensor | float | None = None,
        upper: torch.Tensor | float | None = None,
        sigma: torch.Tensor | float = 0.1,
        batch_shape: tuple[int, ...] | int = (1,),
        min_width: float = 1e-5,
        min_sigma: float = 1e-6,
    ):
        super().__init__(features=features, batch_shape=batch_shape)
        if min_width < 0 or min_sigma < 0:
            raise ValueError("min_width and min_sigma must be nonnegative")
        self.min_width = float(min_width)
        self.min_sigma = float(min_sigma)

        values = (lower, upper, sigma)
        reference = next((v for v in values if isinstance(v, torch.Tensor)), None)
        dtype = reference.dtype if reference is not None else torch.get_default_dtype()
        device = reference.device if reference is not None else torch.device("cpu")
        shape = self.batch_shape + (features,)

        def broadcast(value, name):
            t = torch.as_tensor(value, dtype=dtype, device=device)
            try:
                return torch.broadcast_to(t, shape).clone()
            except RuntimeError as exc:
                raise ValueError(f"{name} must broadcast to shape {shape}") from exc

        lower = broadcast(-1.0 if lower is None else lower, "lower")
        upper = broadcast(1.0 if upper is None else upper, "upper")
        sigma = broadcast(sigma, "sigma")
        width = upper - lower
        if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
            raise ValueError("box bounds must be finite")
        if not torch.isfinite(sigma).all():
            raise ValueError("sigma must be finite")
        if torch.any(width <= self.min_width):
            raise ValueError("each box width must exceed min_width")
        if torch.any(sigma <= self.min_sigma):
            raise ValueError("sigma must exceed min_sigma")

        # Inverse softplus: numerically stable even for large initialization.
        def inverse_softplus(y):
            return y + torch.log(-torch.expm1(-y))

        self.center = torch.nn.Parameter((lower + upper) / 2)
        self.raw_width = torch.nn.Parameter(inverse_softplus(width - self.min_width))
        self.raw_sigma = torch.nn.Parameter(inverse_softplus(sigma - self.min_sigma))

    def widths(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_width) + self.min_width

    def bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        half_width = self.widths() / 2
        return self.center - half_width, self.center + half_width

    def sigmas(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_sigma) + self.min_sigma

    @staticmethod
    def _log_ndtr_stable(z: torch.Tensor) -> torch.Tensor:
        """Log Gaussian CDF with stable gradients far into the left tail.

        PyTorch's log_ndtr backward can lose accuracy for very large negative
        arguments in float32. Use the standard Mills-ratio asymptotic expansion
        below -10, where the relative error is negligible.
        """
        tail = z < -10.0
        z_regular = torch.where(tail, -10.0, z)
        z_tail = torch.where(tail, z, -10.0)
        inv_sq = z_tail.reciprocal().square()
        correction = 1 - inv_sq + 3 * inv_sq.square() - 15 * inv_sq.pow(3) + 105 * inv_sq.pow(4)
        log_tail = (
            -0.5 * z_tail.square()
            - (-z_tail).log()
            - 0.5 * math.log(2 * math.pi)
            + correction.log()
        )
        return torch.where(tail, log_tail, torch.special.log_ndtr(z_regular))

    @staticmethod
    def _log_cdf_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Stable log(Phi(a) - Phi(b)) for a > b, including distant tails.

        For arguments in the positive tail, use Gaussian survival functions
        to avoid subtracting CDF values that both round to one.
        """
        # Switch *arguments* before evaluating the logarithms. Computing both
        # branches and selecting afterward would create -inf in the inactive
        # branch for distant tails, which can cause NaN gradients.
        upper_tail = b > 0
        hi = torch.where(upper_tail, -b, a)
        lo = torch.where(upper_tail, -a, b)
        log_hi = SmoothBox._log_ndtr_stable(hi)
        log_lo = SmoothBox._log_ndtr_stable(lo)
        return log_hi + torch.log(-torch.expm1(log_lo - log_hi))

    def log_density(self, x: torch.Tensor, **contexts: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.features:
            raise ValueError(f"x must have shape (n_data, {self.features})")
        lower, upper = self.bounds()
        sigma = self.sigmas()
        # Data: (n_data, 1, ..., 1, features). Params: (*batch_shape, features).
        x = x.reshape(x.shape[0], *((1,) * len(self.batch_shape)), self.features)
        a = (x - lower) / sigma
        b = (x - upper) / sigma
        log_p = (self._log_cdf_difference(a, b) - self.widths().log()).sum(dim=-1)
        # Intentionally do not clip: tail penalties must keep growing when
        # sigma decreases, so particles outside the box remain detectable.
        return log_p

    def sample(self, n_samples: int, **contexts: torch.Tensor) -> torch.Tensor:
        lower, upper = self.bounds()
        shape = (n_samples, *self.batch_shape, self.features)
        uniform = torch.rand(shape, device=self.center.device, dtype=self.center.dtype)
        noise = torch.randn(shape, device=self.center.device, dtype=self.center.dtype)
        return lower + self.widths() * uniform + self.sigmas() * noise

    def supremum_bound(self) -> torch.Tensor:
        """Exact maximum, attained at the box center."""
        t = self.widths() / (2 * self.sigmas())
        log_peak = (self._log_cdf_difference(t, -t) - self.widths().log()).sum(-1)
        return log_peak.exp()

    def dtype_device(self):
        return self.center.dtype, self.center.device


class SmoothBoxMixture(DensityModel):
    """Trainable mixture of Gaussian-smoothed rectangular uniform densities.

    ``lower``, ``upper``, and ``sigma`` broadcast to
    ``(*batch_shape, n_components, features)``. With unspecified bounds, boxes
    initialize as nonoverlapping slabs of [-1, 1]^d. Component logits broadcast to
    ``(*batch_shape, n_components)`` and are converted to weights by softmax.

    The geometry is stored in ``components`` (a :class:`SmoothBox`). The
    convenience methods ``bounds()`` and ``sigmas()`` expose its parameters.
    """

    def __init__(
        self,
        features: int,
        n_components: int,
        lower: torch.Tensor | float | None = None,
        upper: torch.Tensor | float | None = None,
        sigma: torch.Tensor | float = 0.1,
        logits: torch.Tensor | None = None,
        batch_shape: tuple[int, ...] | int = (1,),
        min_width: float = 1e-5,
        min_sigma: float = 1e-6,
    ):
        super().__init__(features=features, batch_shape=batch_shape)
        if n_components < 1:
            raise ValueError("n_components must be positive")
        self.n_components = n_components
        if lower is None and upper is None:
            # Initialize disjoint slabs of [-1, 1]^d rather than identical
            # overlapping boxes (a symmetric configuration can be hard to
            # escape using gradient-based overlap penalties).
            ref = sigma if isinstance(sigma, torch.Tensor) else None
            opts = ({"dtype": ref.dtype, "device": ref.device} if ref is not None else {})
            lower = torch.full((n_components, features), -1.0, **opts)
            upper = torch.full((n_components, features), 1.0, **opts)
            edges = torch.linspace(-1.0, 1.0, n_components + 1, **opts)
            lower[:, 0] = edges[:-1]
            upper[:, 0] = edges[1:]
        self.components = SmoothBox(
            features=features,
            lower=lower,
            upper=upper,
            sigma=sigma,
            batch_shape=self.batch_shape + (n_components,),
            min_width=min_width,
            min_sigma=min_sigma,
        )
        shape = self.batch_shape + (n_components,)
        if logits is None:
            initial_logits = torch.zeros(
                shape, device=self.components.center.device, dtype=self.components.center.dtype
            )
        else:
            initial_logits = torch.as_tensor(
                logits, device=self.components.center.device, dtype=self.components.center.dtype
            )
            try:
                initial_logits = torch.broadcast_to(initial_logits, shape).clone()
            except RuntimeError as exc:
                raise ValueError(f"logits must broadcast to shape {shape}") from exc
        self.logits = torch.nn.Parameter(initial_logits)

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=-1)

    def bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.components.bounds()

    def sigmas(self) -> torch.Tensor:
        return self.components.sigmas()

    def log_density(self, x: torch.Tensor, **contexts: torch.Tensor) -> torch.Tensor:
        log_components = self.components.log_density(x, **contexts)
        return torch.logsumexp(log_components + torch.log_softmax(self.logits, dim=-1), dim=-1)

    def sample(self, n_samples: int, **contexts: torch.Tensor) -> torch.Tensor:
        # Sample one categorical component per sample and batch entry.
        idx = torch.distributions.Categorical(logits=self.logits).sample((n_samples,))
        lower, upper = self.bounds()
        sigma = self.sigmas()
        shape = (n_samples, *self.batch_shape, self.n_components, self.features)
        gather_index = idx[..., None, None].expand(*idx.shape, 1, self.features)
        lower = lower.expand(shape).gather(-2, gather_index).squeeze(-2)
        upper = upper.expand(shape).gather(-2, gather_index).squeeze(-2)
        sigma = sigma.expand(shape).gather(-2, gather_index).squeeze(-2)
        uniform = torch.rand_like(lower)
        noise = torch.randn_like(lower)
        return lower + (upper - lower) * uniform + sigma * noise

    def supremum_bound(self) -> torch.Tensor:
        """Differentiable upper bound: weighted sum of component maxima."""
        return (self.weights() * self.components.supremum_bound()).sum(dim=-1)

    def dtype_device(self):
        return self.logits.dtype, self.logits.device
