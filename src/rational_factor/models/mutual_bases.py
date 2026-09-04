from __future__ import annotations

import math

import torch

import numpy as np
from numpy.polynomial.legendre import leggauss


from normalizing_flow.vp_flow import VolumePreservingFlow
from rational_factor.models.basis_functions import Basis, BetaBasis
from rational_factor.models.parameters import Parameters, Order1QuasiseparableFactorization
from rational_factor.models.structured_matrices import (
    DenseMatrix,
    Matrix,
    Order1QSGenerators,
    Order1Quasiseparable,
    Rank1PlusDiagonal,
)


class MutualPairBasis:
    def __init__(
        self,
        dim: int,
        batch_size: int,
        n_basis: int,
        params: tuple[Parameters, ...],
        coeffs: tuple[Parameters | None, Parameters | None] | None = None,
    ):
        self._dim = dim
        self._batch_size = batch_size
        self._n_basis = n_basis
        self._params = params
        self._coeffs = coeffs if coeffs is not None else (None, None)

    def get_basis(self, index: int, coeffs: Parameters | None = None) -> MutualPairMemberBasis:
        basis = MutualPairMemberBasis(self, index)
        if coeffs is not None:
            basis.set_coeffs(coeffs)
        return basis

    def Omega1(self, index: int, lows: torch.Tensor = None, highs: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError("Omega1 is not implemented for this mutual basis")

    def Omega2(self, lows: torch.Tensor = None, highs: torch.Tensor = None) -> Matrix:
        raise NotImplementedError("Omega2 is not implemented for this mutual basis")

    def eval(self, y: torch.Tensor, index: int = None):
        raise NotImplementedError("eval is not implemented for this mutual basis")

    def dim(self) -> int:
        return self._dim

    def batch_size(self) -> int:
        return self._batch_size

    def n_basis_functions(self) -> int:
        return self._n_basis

    def dtype_device(self):
        return self._params[0]().dtype, self._params[0]().device


class MutualPairMemberBasis(Basis):
    def __init__(self, owner: MutualPairBasis, index: int):
        assert index in (0, 1), "index must be 0 or 1"
        self.owner = owner
        self.index = index
        super().__init__(
            owner._dim,
            owner._batch_size,
            owner._n_basis,
            owner._params,
            owner._coeffs[index],
        )

    def dtype_device(self):
        return self.owner.dtype_device()

    def _is_mate(self, other: MutualPairMemberBasis | list[MutualPairMemberBasis]) -> bool:
        if isinstance(other, MutualPairMemberBasis):
            return self.owner is other.owner and self.index != other.index
        same_owner = all(self.owner is basis.owner for basis in other)
        unique_indices = len({basis.index for basis in other}) == len(other)
        return same_owner and unique_indices

    def Omega1(self, lows: torch.Tensor = None, highs: torch.Tensor = None):
        return self.owner.Omega1(self.index, lows, highs) * self.coeffs()

    def Omega2(self, other: MutualPairMemberBasis, lows: torch.Tensor = None, highs: torch.Tensor = None) -> Matrix:
        if not self._is_mate(other):
            raise ValueError("Omega2 is not defined for non-mate bases")
        G = self.owner.Omega2(lows, highs)
        if self.index == 1:
            G = G.T
        return G.mul_diag_left(self.coeffs()).mul_diag_right(other.coeffs())

    def __call__(self, y: torch.Tensor):
        values = self.owner.eval(y, self.index)
        coeffs = self.coeffs().to(dtype=values.dtype, device=values.device)
        return values * coeffs


class FeasibleZeroMeanPWC:
    """Fixed feasible ``m x (m+1)`` equal-width PWC basis matrix.

        F = [ sqrt(m+1) I - eta 11^T  |  -1 ]

    with ``eta = (sqrt(m+1) - 1) / m``. Rows have zero integral and
    ``(1/(m+1)) F F^T = I`` on the equal partition of ``[0, 1]``.
    """

    def __init__(self, n_basis: int):
        self.n_basis = n_basis
        self.n_cells = n_basis + 1
        self.sqrt_p = math.sqrt(self.n_cells)
        self.eta = (self.sqrt_p - 1.0) / n_basis

    def columns(self, cell: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """``F[:, cell]^T`` for each requested cell. ``cell`` is ``(N,)`` → ``(N, m)``."""
        cell = cell.reshape(-1)
        N, m = cell.shape[0], self.n_basis
        out = torch.full((N, m), -self.eta, dtype=dtype, device=device)
        head = cell < m
        if head.any():
            rows = torch.arange(N, device=device)[head]
            out[rows, cell[head]] += self.sqrt_p
        if (~head).any():
            out[~head] = -1.0
        return out

    def to_dense(self, dtype=torch.float32, device=None) -> torch.Tensor:
        m = self.n_basis
        head = self.sqrt_p * torch.eye(m, dtype=dtype, device=device) - self.eta
        tail = -torch.ones(m, 1, dtype=dtype, device=device)
        return torch.cat([head, tail], dim=-1)
        

class Orthogonal1DPWCBasis(MutualPairBasis):
    """Piecewise-constant mutual pair ``alpha = P F``, ``beta = Lambda P^{-T} F``.

    ``F`` is the fixed feasible zero-mean orthogonal PWC matrix; ``P`` is an
    invertible order-1 quasiseparable factor. Then

        integral alpha_i = integral beta_i = 0,
        <alpha_i, beta_j> = Lambda_i delta_ij.
    """

    def __init__(
        self,
        qs_factorization: Order1QuasiseparableFactorization,
        gram_diag_params: Parameters = None,
        coeffs: tuple[Parameters | None, Parameters | None] | None = None,
    ):
        d = qs_factorization.diag()
        batch_size, n_basis = d.shape[0], d.shape[-1]
        if gram_diag_params is not None and gram_diag_params().shape != (batch_size, n_basis):
            raise ValueError("gram_diag_params must have shape (batch_size, n_basis)")

        params = qs_factorization.parameters
        if gram_diag_params is not None:
            params = params + (gram_diag_params,)

        super().__init__(1, batch_size, n_basis, params, coeffs)
        self._qs_factorization = qs_factorization
        self._gram_diag_params = gram_diag_params
        self._F = FeasibleZeroMeanPWC(n_basis)
        self._n_cells = n_basis + 1

    def get_feasible_matrix(self) -> FeasibleZeroMeanPWC:
        return self._F

    @property
    def n_cells(self) -> int:
        return self._n_cells

    def _get_P(self) -> Order1Quasiseparable:
        return self._qs_factorization()

    def _get_gram_diag(self) -> torch.Tensor:
        if self._gram_diag_params is None:
            return torch.ones_like(self._qs_factorization.diag())
        return self._gram_diag_params()

    def cell_edges(self, batch_index: int = 0) -> torch.Tensor:
        """Equal partition of ``[0, 1]``, shape ``(n_cells + 1,)``."""
        d = self._qs_factorization.diag()
        return torch.linspace(0.0, 1.0, self._n_cells + 1, dtype=d.dtype, device=d.device)

    def _cell_index(self, y: torch.Tensor) -> torch.Tensor:
        d = self._qs_factorization.diag()
        y = torch.as_tensor(y, dtype=d.dtype, device=d.device).reshape(-1)
        return torch.floor(y * self._n_cells).long().clamp(0, self._n_cells - 1)

    def _cell_overlaps(self, lows: torch.Tensor | None, highs: torch.Tensor | None) -> torch.Tensor:
        d = self._qs_factorization.diag()
        dtype, device = d.dtype, d.device
        lo = torch.zeros((), dtype=dtype, device=device) if lows is None else torch.as_tensor(lows, dtype=dtype, device=device).reshape(-1)[0]
        hi = torch.ones((), dtype=dtype, device=device) if highs is None else torch.as_tensor(highs, dtype=dtype, device=device).reshape(-1)[0]
        edges = self.cell_edges()
        return (edges[1:].clamp(max=hi) - edges[:-1].clamp(min=lo)).clamp(min=0)

    def _apply_F(self, raw: torch.Tensor, index: int, keep_batch: bool = False) -> torch.Tensor:
        """Map F-columns ``raw`` of shape ``(N, m)`` through ``P`` or ``Lambda P^{-T}``.

        Returns ``(N, m)`` when ``batch_size == 1`` and ``keep_batch`` is false,
        otherwise ``(batch, N, m)``.
        """
        P = self._get_P()
        B = P.d.shape[0]
        raw = raw.unsqueeze(1).expand(-1, B, -1)
        out = P.matvec(raw) if index == 0 else P.invT_matvec(raw) * self._get_gram_diag()
        if keep_batch:
            return out.permute(1, 0, 2)
        if B != 1:
            raise ValueError("eval currently requires parameter batch_size == 1")
        return out.squeeze(1)

    def eval(self, y: torch.Tensor, index: int | None = None) -> torch.Tensor:
        """Evaluate alpha / beta at ``y ∈ [0, 1]``.

        Returns ``(N, m)`` for ``index`` in ``{0, 1}`` and ``(N, 2, m)`` if
        ``index is None``. Requires parameter ``batch_size == 1``.
        """
        d = self._qs_factorization.diag()
        y = torch.as_tensor(y, dtype=d.dtype, device=d.device).reshape(-1)
        raw = self._F.columns(self._cell_index(y), dtype=d.dtype, device=d.device)
        if index in (0, 1):
            return self._apply_F(raw, index)
        if index is None:
            return torch.stack([self._apply_F(raw, 0), self._apply_F(raw, 1)], dim=1)
        raise ValueError("index must be 0, 1, or None")

    def _cell_values(self, index: int) -> torch.Tensor:
        """Basis values on every cell, shape ``(batch, n_cells, m)``."""
        d = self._qs_factorization.diag()
        cells = torch.arange(self._n_cells, device=d.device)
        raw = self._F.columns(cells, dtype=d.dtype, device=d.device)
        return self._apply_F(raw, index, keep_batch=True)

    def Omega1(self, index: int, lows: torch.Tensor = None, highs: torch.Tensor = None) -> torch.Tensor:
        if index not in (0, 1):
            raise ValueError("index must be 0 or 1")
        if lows is None and highs is None:
            return torch.zeros_like(self._qs_factorization.diag())
        w = self._cell_overlaps(lows, highs)
        return torch.einsum("k,bkm->bm", w, self._cell_values(index))

    def Omega2(self, lows: torch.Tensor = None, highs: torch.Tensor = None) -> Matrix:
        """Cross Gram ``<alpha_i, beta_j>``, shape ``(batch, m, m)``."""
        if lows is None and highs is None:
            d = self._get_gram_diag()
            z = torch.zeros_like(d)
            return Rank1PlusDiagonal(z, z, d)
        w = self._cell_overlaps(lows, highs)
        a, b = self._cell_values(0), self._cell_values(1)
        return DenseMatrix(torch.einsum("k,bki,bkj->bij", w, a, b))

    def _PF_bounds(
        self,
        generators: Order1QSGenerators,
        row_sum: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_min, q_max = generators.row_minmax()
        first_min = self._F.sqrt_p * q_min - self._F.eta * row_sum
        first_max = self._F.sqrt_p * q_max - self._F.eta * row_sum
        last = -row_sum
        return torch.minimum(first_min, last), torch.maximum(first_max, last)

    @staticmethod
    def _scale_interval(lower: torch.Tensor, upper: torch.Tensor, scale: torch.Tensor):
        a, b = scale * lower, scale * upper
        return torch.minimum(a, b), torch.maximum(a, b)

    def bounds(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact inf/sup of every basis function, each shape ``(batch, m)``.

        ``α = P F`` and ``β = Λ P^{-T} F``. Row extrema of ``P`` / ``P^{-T}``
        come from the order-1 QS generators; F's two column types then give
        the extrema of the PWC pair without enumerating cells.
        """
        P = self._get_P()
        ones = torch.ones_like(P.d)
        if index == 0:
            return self._PF_bounds(P.direct_generators(), P.matvec(ones))
        if index == 1:
            lower, upper = self._PF_bounds(P.inverse_transpose_generators(), P.invT_matvec(ones))
            return self._scale_interval(lower, upper, self._get_gram_diag())
        raise ValueError("index must be 0 or 1")

    def infimum(self, index: int) -> torch.Tensor:
        return self.bounds(index)[0]

    def supremum(self, index: int) -> torch.Tensor:
        return self.bounds(index)[1]


class FixedDegreeBSplineMutualBasis(MutualPairBasis, torch.nn.Module):
    r"""Fixed-degree open-uniform B-spline basis and its L2 dual.

    Let

        alpha(x) = N(x)

    where N is the vector of degree-p open-uniform B-splines on [0, 1].
    Define

        G = integral N(x) N(x)^T dx

    and

        beta(x) = Lambda G^{-1} N(x).

    Then

        <alpha_i, beta_j> = Lambda_j delta_ij.

    If ``gram_diag_params is None``, Lambda = I.

    Important properties
    --------------------
    * alpha_i >= 0.
    * alpha_i has compact support spanning at most p + 1 knot intervals.
    * G has half-bandwidth p.
    * G^{-1} is dense, but its entries decay away from the diagonal.
    * beta_i is therefore globally supported but typically strongly localized.
    * alpha and beta are piecewise polynomials of degree p.
    * Their extrema can be obtained span-by-span by solving degree-(p - 1)
      derivative polynomials.

    Notes
    -----
    The number of basis functions grows by adding knots while keeping p fixed.
    This distinction is important. If n_basis == degree + 1, these reduce to
    the Bernstein polynomials and the basis is not spatially local.
    """

    def __init__(self, n_basis: int, degree: int = 3):
        if degree < 0:
            raise ValueError("degree must be >= 0")
        if n_basis < degree + 1:
            raise ValueError(
                "n_basis must be at least degree + 1 "
                "for an open B-spline basis"
            )

        MutualPairBasis.__init__(self, 1, 1, n_basis, ())
        torch.nn.Module.__init__(self)

        self._degree = degree

        self._n_spans = n_basis - degree

        self.register_buffer("_breaks", torch.linspace(0.0, 1.0, self._n_spans + 1))

        self.register_buffer("_knots", torch.cat([torch.zeros(degree + 1), self._breaks[1:-1], torch.ones(degree + 1)]))

        # ---------------------------------------------------------------
        # Gauss-Legendre quadrature.
        #
        # Product of two degree-p splines has degree 2p on each span.
        # p + 1 Gauss points integrate degree <= 2p + 1 exactly.
        # ---------------------------------------------------------------
        qx, qw = leggauss(degree + 1)

        self.register_buffer("_quad_x", torch.as_tensor(qx))
        self.register_buffer("_quad_w", torch.as_tensor(qw))

        # ---------------------------------------------------------------
        # Construct exact spline Gram matrix up to floating point.
        # ---------------------------------------------------------------
        mass, G = self._interval_moments_raw(
            torch.zeros(()),
            torch.ones(()),
        )

        # Force exact symmetry and exact band structure.
        G = 0.5 * (G + G.T)

        ids = torch.arange(n_basis)
        band_mask = torch.abs(ids[:, None] - ids[None, :]) <= degree
        G = torch.where(band_mask, G, torch.zeros_like(G))

        self.register_buffer("_alpha_mass", mass)
        self.register_buffer("_gram_matrix", G)

        # Banded Cholesky factor.
        #
        # G has half-bandwidth degree, and its Cholesky factor has the
        # same half-bandwidth.
        self._L = self._banded_cholesky(G, degree)

        # Bounds/polynomial caches.
        self._alpha_power_coeffs = None
        self._beta_power_coeffs = None
        self._alpha_bounds = None
        self._beta_bounds = None

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def knots(self) -> torch.Tensor:
        return self._knots

    @property
    def breakpoints(self) -> torch.Tensor:
        return self._breaks

    @property
    def gram_matrix(self) -> torch.Tensor:
        return self._gram_matrix

    def _eval_alpha(self, y: torch.Tensor) -> torch.Tensor:
        """Evaluate all alpha B-splines.

        Parameters
        ----------
        y:
            Tensor of arbitrary shape.

        Returns
        -------
        Tensor
            Shape ``(num_points, n_basis)`` after flattening y.
        """

        if torch.any(y < 0) or torch.any(y > 1):
            raise ValueError("B-spline evaluation requires y in [0, 1]")

        t = self._knots

        # Degree-zero B-splines.
        #
        # There are len(knots)-1 of these. Recursion reduces this to
        # n_basis functions after p steps.
        N = ((y[:, None] >= t[:-1]) & (y[:, None] < t[1:]))

        # Cox-de Boor recursion.
        for k in range(1, self._degree + 1):
            n_out = N.shape[-1] - 1

            left_den = t[k:k + n_out] - t[:n_out]
            right_den = (t[k + 1:k + 1 + n_out] - t[1:1 + n_out])

            left_num = y[:, None] - t[:n_out]
            right_num = (t[k + 1:k + 1 + n_out] - y[:, None])

            left = torch.where(
                left_den[None, :] != 0,
                left_num / torch.where(
                    left_den == 0,
                    torch.ones_like(left_den),
                    left_den,
                )[None, :] * N[:, :n_out],
                torch.zeros_like(N[:, :n_out]),
            )

            right = torch.where(
                right_den[None, :] != 0,
                right_num / torch.where(
                    right_den == 0,
                    torch.ones_like(right_den),
                    right_den,
                )[None, :] * N[:, 1:n_out + 1],
                torch.zeros_like(N[:, :n_out]),
            )

            N = left + right

        # Open B-spline convention at x = 1:
        #
        #     N_{m-1}(1) = 1.
        #
        at_right = y == 1
        if torch.any(at_right):
            N = N.clone()
            N[at_right] = 0
            N[at_right, -1] = 1

        return N

    @staticmethod
    def _banded_cholesky(
        G: torch.Tensor,
        half_bandwidth: int,
    ) -> torch.Tensor:
        """Cholesky factorization exploiting fixed bandwidth.

        Returns lower-triangular L satisfying

            G = L L^T.

        Complexity is O(m p^2), where p is the half-bandwidth.
        """
        m = G.shape[-1]
        p = half_bandwidth

        L = torch.zeros_like(G)

        for i in range(m):
            j0 = max(0, i - p)

            for j in range(j0, i + 1):
                k0 = max(0, i - p, j - p)

                if j > k0:
                    correction = (L[i, k0:j] * L[j, k0:j]).sum()
                else:
                    correction = G.new_zeros(())

                value = G[i, j] - correction

                if i == j:
                    if value <= 0:
                        raise RuntimeError(
                            "B-spline Gram matrix is not "
                            "numerically positive definite"
                        )
                    L[i, j] = torch.sqrt(value)
                else:
                    L[i, j] = value / L[j, j]

        return L

    def _solve_gram(
        self,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        """Solve G x = rhs along the final dimension.

        ``rhs`` has shape ``(..., m)``.

        Because G is symmetric,

            row @ G^{-1}

        is numerically identical to solving

            G x = row^T.

        Complexity: O(p m) per right-hand side for fixed degree p.
        """
        rhs = torch.as_tensor(rhs)

        if rhs.shape[-1] != self._n_basis:
            raise ValueError(
                "Last rhs dimension must equal n_basis"
            )

        L = self._L
        m = self._n_basis
        p = self._degree

        # Forward solve: L z = rhs.
        z_values = []

        for i in range(m):
            j0 = max(0, i - p)

            if i > j0:
                previous = torch.stack(z_values[j0:i], dim=-1)
                correction = (previous * L[i, j0:i]).sum(dim=-1)
            else:
                correction = torch.zeros_like(rhs[..., i])

            zi = (rhs[..., i] - correction) / L[i, i]

            z_values.append(zi)

        z = torch.stack(z_values, dim=-1)

        # Backward solve: L^T x = z.
        x_values = [None] * m

        for i in range(m - 1, -1, -1):
            j1 = min(m, i + p + 1)

            if i + 1 < j1:
                following = torch.stack(x_values[i + 1:j1], dim=-1)
                correction = (following * L[i + 1:j1, i]).sum(dim=-1)
            else:
                correction = torch.zeros_like(z[..., i])

            xi = (z[..., i] - correction) / L[i, i]

            x_values[i] = xi

        return torch.stack(x_values, dim=-1)

    # ===================================================================
    # Public evaluation
    # ===================================================================

    def eval(self, y: torch.Tensor, index: int | None = None) -> torch.Tensor:
        """Evaluate alpha and/or beta.

        Returns
        -------
        index == 0:
            alpha, shape ``(N, m)``

        index == 1:
            beta, shape ``(N, m)``

        index is None:
            shape ``(N, 2, m)``

        As in ``Orthogonal1DPWCBasis``, evaluation currently requires
        parameter batch_size == 1.
        """
        if self._batch_size != 1:
            raise ValueError(
                "eval currently requires parameter batch_size == 1"
            )

        alpha = self._eval_alpha(y)

        if index == 0:
            return alpha

        beta = self._solve_gram(alpha)

        if index == 1:
            return beta

        if index is None:
            return torch.stack([alpha, beta], dim=1)

        raise ValueError("index must be 0, 1, or None")

    def _parse_interval(self, lows: torch.Tensor | None, highs: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        lo = (torch.zeros(()) if lows is None else torch.as_tensor(lows).reshape(-1)[0])

        hi = (torch.ones(()) if highs is None else torch.as_tensor(highs).reshape(-1)[0])

        lo = lo.clamp(0.0, 1.0)
        hi = hi.clamp(0.0, 1.0)

        return lo, hi

    def _interval_moments_raw(self, lo: torch.Tensor, hi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return

            s_i = integral alpha_i
            M_ij = integral alpha_i alpha_j

        over [lo, hi].

        Gauss-Legendre quadrature is exact on every knot span because
        alpha_i alpha_j has degree <= 2p.
        """
        left = torch.maximum(self._breaks[:-1], lo)
        right = torch.minimum(self._breaks[1:], hi)

        half = 0.5 * (right - left).clamp(min=0)

        mid = 0.5 * (right + left)

        # Shape: (n_spans, n_quad)
        x = (mid[:, None] + half[:, None] * self._quad_x[None, :])

        n_spans, n_quad = x.shape

        values = self._eval_alpha(x.reshape(-1)).reshape(n_spans, n_quad, self._n_basis)

        weights = (half[:, None] * self._quad_w[None, :])

        mass = torch.einsum("sq,sqi->i", weights, values)

        gram = torch.einsum("sq,sqi,sqj->ij", weights, values, values)

        return mass, gram

    def _interval_moments(self, lows: torch.Tensor | None, highs: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        lo, hi = self._parse_interval(lows, highs)
        return self._interval_moments_raw(lo, hi)

    def Omega1(self, index: int, lows: torch.Tensor = None, highs: torch.Tensor = None) -> torch.Tensor:
        """Integral of each alpha_i or beta_i.

        Shape: ``(batch, m)``.
        """
        if index not in (0, 1):
            raise ValueError("index must be 0 or 1")


        # Full-domain identities.
        if lows is None and highs is None:
            if index == 0:
                return self._alpha_mass.unsqueeze(0).expand(self._batch_size, -1)

            return torch.ones_like(self._alpha_mass).unsqueeze(0).expand(self._batch_size, -1)

        mass, _ = self._interval_moments(lows, highs)

        if index == 0:
            return mass.unsqueeze(0).expand(self._batch_size, -1)

        beta_mass = self._solve_gram(mass)

        return (beta_mass.unsqueeze(0))

    def Omega2(self, lows: torch.Tensor = None, highs: torch.Tensor = None) -> Matrix:
        """Cross Gram

            Omega2_ij = integral alpha_i(x) beta_j(x) dx.

        Full domain:
            diag(Lambda)

        Restricted domain:
            M_[lo,hi] G^{-1} Lambda
        """
        lam = self._get_gram_diag()

        if lows is None and highs is None:
            z = torch.zeros_like(lam)
            return Rank1PlusDiagonal(z, z, lam)

        _, local_gram = self._interval_moments(lows, highs)

        cross_unscaled = self._solve_gram(local_gram)

        # Scale beta index, i.e. matrix columns.
        cross = (cross_unscaled.unsqueeze(0) * lam[:, None, :])

        return DenseMatrix(cross)

    def _get_alpha_power_coeffs(self,) -> torch.Tensor:
        """Polynomial coefficients of alpha on every knot span.

        Returns
        -------
        coeffs:
            Shape ``(n_spans, m, degree + 1)``.

        For span s, with

            u = (x - break_s) / (break_{s+1} - break_s),

        we have

            alpha_i(x)
              = sum_r coeffs[s, i, r] u^r.
        """
        if self._alpha_power_coeffs is not None:
            return self._alpha_power_coeffs

        p = self._degree
        q = p + 1

        # Interior interpolation nodes avoid ambiguity at knots.
        nodes, _ = leggauss(q)
        u = 0.5 * (torch.as_tensor(nodes) + 1.0)

        # Vandermonde:
        #
        # V[k, r] = u_k^r.
        #
        V = torch.stack([ u ** r for r in range(q) ], dim=-1)

        V_inv = torch.linalg.inv(V)

        left = self._breaks[:-1]
        right = self._breaks[1:]

        x = (left[:, None] + (right - left)[:, None] * u[None, :])

        values = self._eval_alpha(x.reshape(-1)).reshape(self._n_spans, q, self._n_basis)

        # (span, power, basis)
        coeff = torch.einsum("rq,sqm->srm", V_inv, values)

        # (span, basis, power)
        coeff = coeff.permute(0, 2, 1)

        self._alpha_power_coeffs = coeff
        return coeff

    def _get_beta_power_coeffs(
        self,
    ) -> torch.Tensor:
        """Power coefficients of the unscaled dual basis G^{-1} alpha."""
        if self._beta_power_coeffs is not None:
            return self._beta_power_coeffs

        alpha_coeff = self._get_alpha_power_coeffs()

        # For every span and every polynomial power, solve
        #
        #     G c_beta = c_alpha.
        #
        rhs = alpha_coeff.permute(0, 2, 1)

        beta_coeff = self._solve_gram(rhs).permute(0, 2, 1)

        self._beta_power_coeffs = beta_coeff
        return beta_coeff

    @staticmethod
    def _polyval_ascending(
        coeff: np.ndarray,
        x: float,
    ) -> float:
        """Evaluate c0 + c1*x + ... using Horner."""
        out = 0.0
        for c in coeff[::-1]:
            out = out * x + c
        return float(out)

    @classmethod
    def _exact_piecewise_bounds(cls, coeffs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Numerical global extrema of piecewise polynomials.

        For each span, solve the derivative polynomial exactly as a
        finite root problem and evaluate endpoints + all real roots in
        [0, 1].

        ``coeffs`` has shape

            (n_spans, n_basis, degree + 1).

        Returns shape ``(n_basis,)`` for lower and upper.

        This is mathematically exact modulo floating-point polynomial
        reconstruction and root solving.
        """
        device = coeffs.device
        dtype = coeffs.dtype

        c_np = coeffs.detach().cpu().double().numpy()

        n_spans, n_basis, q = c_np.shape
        degree = q - 1

        lower = np.full(n_basis, np.inf, dtype=np.float64)
        upper = np.full(n_basis, -np.inf, dtype=np.float64)

        for s in range(n_spans):
            for i in range(n_basis):
                c = c_np[s, i]

                candidates = [0.0, 1.0]

                if degree >= 1:
                    # Ascending derivative coefficients.
                    d = np.array([ r * c[r] for r in range(1, q) ], dtype=np.float64)

                    scale = max(1.0, float(np.max(np.abs(d))) if d.size > 0 else 1.0)
                    tol = 1e-12 * scale

                    # Trim numerically-zero leading terms.
                    while (d.size > 0 and abs(d[-1]) <= tol):
                        d = d[:-1]

                    # Constant derivative has no interior root.
                    if d.size >= 2:
                        roots = np.roots(d[::-1])

                        for root in roots:
                            if abs(root.imag) <= 1e-10:
                                r = float(root.real)

                                if r >= -1e-10 and r <= 1.0 + 1e-10:
                                    candidates.append(min(1.0, max(0.0, r)))

                vals = [cls._polyval_ascending(c, u) for u in candidates]

                lower[i] = min(lower[i], min(vals))
                upper[i] = max(upper[i], max(vals))

        return (torch.as_tensor(lower, dtype=dtype, device=device), torch.as_tensor(upper, dtype=dtype, device=device))

    # ===================================================================
    # Bounds
    # ===================================================================

    @staticmethod
    def _scale_interval(lower: torch.Tensor, upper: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = scale * lower
        b = scale * upper

        return (torch.minimum(a, b), torch.maximum(a, b))

    def bounds(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Global lower/upper extrema of every basis function.

        Returns
        -------
        lower, upper:
            Each has shape ``(batch, m)``.

        For alpha and beta, extrema are found span-by-span from their
        exact degree-p polynomial representation.

        For cubic B-splines this means solving only quadratic derivative
        equations on each knot span.
        """
        if index == 0:
            if self._alpha_bounds is None:
                self._alpha_bounds = (
                    self._exact_piecewise_bounds(
                        self._get_alpha_power_coeffs()
                    )
                )

            lower, upper = self._alpha_bounds

            return (
                lower.unsqueeze(0).expand(self._batch_size, -1),
                upper.unsqueeze(0).expand(self._batch_size, -1),
            )

        if index == 1:
            if self._beta_bounds is None:
                self._beta_bounds = (
                    self._exact_piecewise_bounds(
                        self._get_beta_power_coeffs()
                    )
                )

            lower, upper = self._beta_bounds

            lower = lower.unsqueeze(0).expand(self._batch_size, -1)
            upper = upper.unsqueeze(0).expand(self._batch_size, -1)

            return self._scale_interval(lower, upper, self._get_gram_diag())

        raise ValueError("index must be 0 or 1")

    def infimum(self, index: int) -> torch.Tensor:
        return self.bounds(index)[0]

    def supremum(self, index: int,) -> torch.Tensor:
        return self.bounds(index)[1]

class VolumePreservingPairBasis(torch.nn.Module, MutualPairBasis):
    """Mutual pair whose matched products are conditional VP-flow densities.

    A single volume-preserving flow ``T(· | e_i)`` is shared across the ``m``
    basis functions and conditioned on a learned index embedding ``e_i``.
    With per-index base density ``p_{0,i}`` and ``N_i = sup p_{0,i}``,

        n_i(x) = p_{0,i}(T(x | e_i)),

    since ``log |det DT| = 0``. If ``flow`` is ``None`` (or the map is 1D
    identity), ``n_i(x) = p_{0,i}(x)``. A single shared base is broadcast
    across indices. The split

        α_i(x) = n_i(x)/N + (1 - n_i(x)/N) s_i(x)
        β_i(x) = n_i(x) / α_i(x)

    then satisfies ``α_i β_i = n_i``. The splitter is a shared scalar net
    ``σ(x, e_i)``: concatenate each ``x`` with every index embedding and
    evaluate once to get ``s ∈ (0, 1)^m``. Because ``s ∈ (0, 1)`` and
    ``n ≤ N``,

        n/N ≤ α ≤ 1,    β ≤ N.

    A 0-d rest space (empty product over no coordinates) is the identity
    pair ``α = β = n = 1``.
    """

    def __init__(
        self,
        base: BetaBasis,
        splitter: torch.nn.Module,
        embedding: torch.nn.Embedding,
        flow: VolumePreservingFlow | None = None,
        eps: float = 1e-6,
        coeffs: tuple[Parameters | None, Parameters | None] | None = None,
    ):
        torch.nn.Module.__init__(self)
        n_basis = embedding.num_embeddings
        if n_basis < 1:
            raise ValueError("n_basis must be at least 1")
        base_n = base.n_basis_functions()
        if base_n not in (1, n_basis):
            raise ValueError(
                f"base n_basis {base_n} must be 1 or match embedding n_basis {n_basis}"
            )
        if base.dim() == 0 and flow is not None:
            raise ValueError("flow is not defined for 0-dimensional rest coordinates")
        if flow is not None:
            if flow.dim != base.dim():
                raise ValueError("flow and base must have the same dimension")
            if embedding.embedding_dim != flow.conditioner_dim:
                raise ValueError(
                    f"embedding dim {embedding.embedding_dim} must match "
                    f"flow conditioner_dim {flow.conditioner_dim}"
                )

        MutualPairBasis.__init__(self, base.dim(), 1, n_basis, (), coeffs)

        self.flow = flow
        self.base = base
        self._base_param_modules = torch.nn.ModuleList(
            [p for p in base._params if p.is_module()]
        )
        self.splitter = splitter
        self.index_embedding = embedding
        self.eps = eps

    def dtype_device(self):
        p = next(self.parameters())
        return p.dtype, p.device

    def _index_conditioners(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        idx = torch.arange(self._n_basis, device=device)
        return self.index_embedding(idx).to(dtype=dtype)

    def _as_data(self, y: torch.Tensor) -> torch.Tensor:
        y = torch.as_tensor(y)
        if self._dim == 0:
            if y.ndim == 1:
                y = y.unsqueeze(-1)[:, :0]
            if y.ndim != 2 or y.shape[1] != 0:
                raise ValueError(f"y must have shape (n_data, 0) for a 0-d pair, got {tuple(y.shape)}")
            return y
        if y.ndim == 1:
            y = y.unsqueeze(-1) if self._dim == 1 else y.unsqueeze(0)
        if y.ndim != 2 or y.shape[1] != self._dim:
            raise ValueError(f"y must have shape (n_data, {self._dim}), got {tuple(y.shape)}")
        return y

    def _eval_base(self, y: torch.Tensor) -> torch.Tensor:
        """``p_{0,i}(y)``, shape ``(n_data, n_basis)``."""
        n = self.base(y)
        if n.shape[-1] == 1:
            return n.expand(-1, self._n_basis)
        return n

    def flow_density(self, y: torch.Tensor) -> torch.Tensor:
        """Base-flow densities ``n_i(y)``, shape ``(n_data, n_basis)``."""
        y = self._as_data(y)
        n_data, m = y.shape[0], self._n_basis
        if self._dim == 0:
            return y.new_ones(n_data, m)
        if self.flow is None or self.flow.dim == 1:
            return self._eval_base(y)
        cond = self._index_conditioners(y.dtype, y.device)
        y_rep = y.unsqueeze(1).expand(-1, m, -1).reshape(n_data * m, self._dim)
        c_rep = cond.unsqueeze(0).expand(n_data, -1, -1).reshape(n_data * m, -1)
        z, ladj = self.flow.forward(y_rep, conditioner=c_rep)
        n = self.base(z)
        if n.shape[-1] == 1:
            n = n.reshape(n_data, m)
        else:
            n = n.view(n_data, m, m).diagonal(dim1=-2, dim2=-1)
        return n * torch.exp(ladj).view(n_data, m)

    def _split(self, y: torch.Tensor, n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bound = self.base.supremum_bound().to(dtype=n.dtype, device=n.device)
        r = n / bound.clamp_min(self.eps)
        n_data, m = y.shape[0], self._n_basis
        e = self._index_conditioners(y.dtype, y.device)
        y_rep = y.unsqueeze(1).expand(-1, m, -1)
        e_rep = e.unsqueeze(0).expand(n_data, -1, -1)
        inp = torch.cat([y_rep, e_rep], dim=-1).reshape(n_data * m, -1)
        s = torch.sigmoid(self.splitter(inp)).reshape(n_data, m)
        alpha = r + (1.0 - r) * s
        beta = n / alpha.clamp_min(self.eps)
        return alpha, beta

    def eval(self, y: torch.Tensor | None = None, index: int | None = None):
        if y is None:
            return torch.nn.Module.eval(self)
        y = self._as_data(y)
        if self._dim == 0:
            ones = y.new_ones(y.shape[0], self._n_basis)
            if index in (0, 1):
                return ones
            if index is None:
                return torch.stack([ones, ones], dim=1)
            raise ValueError("index must be 0, 1, or None")
        n = self.flow_density(y)
        alpha, beta = self._split(y, n)
        if index == 0:
            return alpha
        if index == 1:
            return beta
        if index is None:
            return torch.stack([alpha, beta], dim=1)
        raise ValueError("index must be 0, 1, or None")

    def supremum(self, index: int) -> torch.Tensor:
        """Per-function supremum bounds, shape ``(batch, n_basis)``.

        ``α ≤ 1`` and ``β ≤ N_i``. ``N_i = sup p_{0,i}`` is recomputed
        from the current base parameters, so it tracks training.
        """
        dtype, device = self.dtype_device()
        ones = torch.ones(self._batch_size, self._n_basis, dtype=dtype, device=device)
        if index not in (0, 1):
            raise ValueError("index must be 0 or 1")
        if self._dim == 0:
            return ones
        if index == 0:
            return ones
        return ones * self.base.supremum_bound()


class MaskedGramMutualBasis(torch.nn.Module, MutualPairBasis):
    """Product of a 1D PWC pair on a sacrificial coordinate and a VP pair on the rest.

    For sacrificial index ``l`` in ``{0, …, d-1}``,

        α_i(x) = α^{pwc}_i(x_l) α^{vp}_i(x_{≠l})
        β_i(x) = β^{pwc}_i(x_l) β^{vp}_i(x_{≠l})

    If ``d = 1``, the rest space is empty and the VP pair is identically 1,
    so the masked pair reduces to the sacrificial PWC pair.

    The PWC Gram is diagonal, so it zeros VP off-diagonal couplings. On the
    full unit cube the VP paired products integrate to 1, and

        Ω2_ij = Ω2^{pwc}_ij * δ_ij = Λ_i δ_ij.
    """

    def __init__(
        self,
        pwc: Orthogonal1DPWCBasis,
        sacrificial_index: int,
        vp: VolumePreservingPairBasis,
        coeffs: tuple[Parameters | None, Parameters | None] | None = None,
        domain_lows: torch.Tensor | None = None,
        domain_highs: torch.Tensor | None = None,
    ):
        torch.nn.Module.__init__(self)
        if pwc.dim() != 1:
            raise ValueError("pwc must be 1-dimensional")
        dim = vp.dim() + 1
        if not (0 <= sacrificial_index < dim):
            raise ValueError(f"sacrificial_index must be in [0, {dim}), got {sacrificial_index}")
        if pwc.n_basis_functions() != vp.n_basis_functions():
            raise ValueError(
                f"pwc n_basis {pwc.n_basis_functions()} must match "
                f"vp n_basis {vp.n_basis_functions()}"
            )
        if (domain_lows is None) != (domain_highs is None):
            raise ValueError("domain_lows and domain_highs must be provided together")

        MutualPairBasis.__init__(
            self,
            dim,
            pwc.batch_size(),
            pwc.n_basis_functions(),
            pwc._params,
            coeffs,
        )
        self.pwc = pwc
        self.vp = vp
        self.sacrificial_index = sacrificial_index
        self._pwc_param_modules = torch.nn.ModuleList(
            [p for p in pwc._params if p.is_module()]
        )
        if domain_lows is None:
            self.domain_lows = None
            self.domain_highs = None
        else:
            self.register_buffer("domain_lows", torch.as_tensor(domain_lows, dtype=torch.float32).reshape(-1))
            self.register_buffer("domain_highs", torch.as_tensor(domain_highs, dtype=torch.float32).reshape(-1))
            if self.domain_lows.numel() != dim or self.domain_highs.numel() != dim:
                raise ValueError(f"domain bounds must have length {dim}")

    def dtype_device(self):
        return self.pwc.dtype_device()

    def _domain_volume(self) -> torch.Tensor | float:
        if self.domain_lows is None:
            return 1.0
        return (self.domain_highs - self.domain_lows).prod()

    def _to_unit(self, y: torch.Tensor) -> torch.Tensor:
        if self.domain_lows is None:
            return y
        lo = self.domain_lows.to(dtype=y.dtype, device=y.device)
        hi = self.domain_highs.to(dtype=y.dtype, device=y.device)
        u = (y - lo) / (hi - lo).clamp_min(1e-12)
        return u.clamp(1e-4, 1.0 - 1e-4)

    def _split_coords(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = torch.as_tensor(y)
        if y.ndim == 1:
            y = y.unsqueeze(0)
        if y.ndim != 2 or y.shape[1] != self._dim:
            raise ValueError(f"y must have shape (n_data, {self._dim}), got {tuple(y.shape)}")
        y = self._to_unit(y)
        l = self.sacrificial_index
        rest = [i for i in range(self._dim) if i != l]
        return y[:, l], y[:, rest]

    def eval(self, y: torch.Tensor | None = None, index: int | None = None):
        if y is None:
            return torch.nn.Module.eval(self)
        x_l, x_rest = self._split_coords(y)
        return self.pwc.eval(x_l, index) * self.vp.eval(x_rest, index)

    def Omega1(self, index: int, lows: torch.Tensor = None, highs: torch.Tensor = None) -> torch.Tensor:
        if lows is not None or highs is not None:
            raise ValueError("MaskedGramMutualBasis.Omega1 is only defined on the full domain")
        dtype, device = self.pwc.dtype_device()
        return torch.zeros(self._batch_size, self._n_basis, dtype=dtype, device=device)

    def Omega2(self, lows: torch.Tensor = None, highs: torch.Tensor = None) -> Matrix:
        if lows is not None or highs is not None:
            raise ValueError("MaskedGramMutualBasis.Omega2 is only defined on the full domain")
        return self.pwc.Omega2().scale(self._domain_volume())

    def supremum(self, index: int) -> torch.Tensor:
        return self.pwc.supremum(index) * self.vp.supremum(index)


class PositiveMaskedGramMutualBasis(MaskedGramMutualBasis):
    """Shifted masked pair ``α = α_u + a``, ``β = β_u + b``.

    ``α_u``, ``β_u`` are the unsigned ``MaskedGramMutualBasis`` products. Because
    the VP factors are positive, the most negative unsigned values are bounded
    by the PWC infima times the VP suprema, so the constant shifts

        a = -inf(α^{pwc}) ⊙ sup(α^{vp})
        b = -inf(β^{pwc}) ⊙ sup(β^{vp})

    make ``α, β ≥ 0``. On the unit cube the unsigned pair is zero-mean, hence

        Ω1(α) = V a,    Ω1(β) = V b,
        Ω2 = V (diag(Λ) + a bᵀ),

    where ``V`` is the volume of the affine domain box (1 on the unit cube).
    """

    def _shift_cache_key(self) -> tuple:
        key = []
        for param in self.pwc._params:
            leaves = list(param.parameters()) + list(param.buffers()) if param.is_module() else [param()]
            for t in leaves:
                key.append((t.data_ptr(), t._version, bool(t.requires_grad)))
        return tuple(key)

    def constant_shifts(self) -> tuple[torch.Tensor, torch.Tensor]:
        key = self._shift_cache_key()
        cached = getattr(self, "_shift_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        a = -self.pwc.infimum(0) * self.vp.supremum(0)
        b = -self.pwc.infimum(1) * self.vp.supremum(1)
        self._shift_cache = (key, a, b)
        def _invalidate(_grad):
            self._shift_cache = None
        if a.requires_grad:
            a.register_hook(_invalidate)
        if b.requires_grad:
            b.register_hook(_invalidate)
        return a, b

    def eval(self, y: torch.Tensor | None = None, index: int | None = None):
        if y is None:
            return torch.nn.Module.eval(self)
        unsigned = MaskedGramMutualBasis.eval(self, y, index)
        a, b = self.constant_shifts()
        if index == 0:
            return unsigned + a
        if index == 1:
            return unsigned + b
        if index is None:
            return unsigned + torch.stack([a, b], dim=-2)
        raise ValueError("index must be 0, 1, or None")

    def Omega1(self, index: int, lows: torch.Tensor = None, highs: torch.Tensor = None) -> torch.Tensor:
        if lows is not None or highs is not None:
            raise ValueError("PositiveMaskedGramMutualBasis.Omega1 is only defined on the full domain")
        if index not in (0, 1):
            raise ValueError("index must be 0 or 1")
        a, b = self.constant_shifts()
        shift = a if index == 0 else b
        return shift * self._domain_volume()

    def Omega2(self, lows: torch.Tensor = None, highs: torch.Tensor = None) -> Matrix:
        unsigned = MaskedGramMutualBasis.Omega2(self, lows, highs)
        a, b = self.constant_shifts()
        vol = self._domain_volume()
        return Rank1PlusDiagonal(a * vol, b, unsigned.diag())

    def supremum(self, index: int) -> torch.Tensor:
        a, b = self.constant_shifts()
        shift = a if index == 0 else b
        return super().supremum(index) + shift
