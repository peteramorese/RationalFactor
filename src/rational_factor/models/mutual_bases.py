from __future__ import annotations

import math

import torch

import numpy as np
from numpy.polynomial.legendre import leggauss

from normalizing_flow.normalizing_flow import ConditionalNormalizingFlow
from normalizing_flow.vp_flow import VolumePreservingFlow
from rational_factor.models.basis_functions import Basis, BetaBasis
from rational_factor.models.composite_model import CompositeConditionalModel
from rational_factor.models.parameters import Parameters, Order1QuasiseparableFactorization
from rational_factor.models.structured_matrices import (
    DenseMatrix,
    Identity,
    Diagonal,
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


class DisjointSupport1DPWCBasis(MutualPairBasis):
    """Disjoint support piecewise-constant mutual pair on variable-width cells.
    
    Divides [0, 1] into n_basis cells with normalized widths. On cell i, only
    alpha_i and beta_i are active (constant), and zero elsewhere. With
    alpha_i * beta_i = 1 on cell i,

        <alpha_i, beta_j> = delta_ij * width_i,

    so the Gram matrix is ``diag(widths)``, not the identity.
    
    Parameters
    ----------
    cell_widths_params : Parameters
        Normalized positive parameters summing to 1, defining relative cell widths.
    alpha_params : Parameters
        Positive parameters for alpha values on each cell.
    coeffs : tuple, optional
        Optional coefficients for alpha and beta bases.
    """
    
    def __init__(
        self,
        cell_widths_params: Parameters,
        alpha_params: Parameters,
        coeffs: tuple[Parameters | None, Parameters | None] | None = None,
    ):
        widths = cell_widths_params()
        alphas = alpha_params()
        
        if widths.ndim != 2:
            raise ValueError("cell_widths_params must have shape (batch_size, n_basis)")
        if alphas.ndim != 2:
            raise ValueError("alpha_params must have shape (batch_size, n_basis)")
        
        batch_size, n_basis = widths.shape
        if alphas.shape != (batch_size, n_basis):
            raise ValueError("alpha_params and cell_widths_params must have the same shape")
        
        # Ensure widths are normalized
        if not torch.allclose(widths.sum(dim=-1), torch.ones(batch_size, dtype=widths.dtype, device=widths.device)):
            raise ValueError("cell widths must sum to 1 (use normalized PositiveParameters)")
        
        super().__init__(1, batch_size, n_basis, (cell_widths_params, alpha_params), coeffs)
        self._cell_widths_params = cell_widths_params
        self._alpha_params = alpha_params
        self._n_cells = n_basis
    
    @property
    def n_cells(self) -> int:
        return self._n_cells
    
    def cell_edges(self, batch_index: int = 0) -> torch.Tensor:
        widths = self._cell_widths_params()[batch_index]
        edges = torch.cat([
            torch.zeros(1, dtype=widths.dtype, device=widths.device),
            torch.cumsum(widths, dim=0)
        ])
        return edges
    
    def _cell_index(self, y: torch.Tensor, batch_index: int = 0) -> torch.Tensor:
        widths = self._cell_widths_params()
        y = torch.as_tensor(y, dtype=widths.dtype, device=widths.device).reshape(-1)
        y = y.clamp(0.0, 1.0)
        
        edges = self.cell_edges(batch_index)
        # searchsorted returns the insertion index; subtract 1 to get cell index
        cell = torch.searchsorted(edges, y, right=False) - 1
        # Clamp to valid range [0, n_cells-1]
        cell = cell.clamp(0, self._n_cells - 1)
        
        # Handle edge case: y = 1.0 should be in the last cell
        cell = torch.where(y >= edges[-1], torch.full_like(cell, self._n_cells - 1), cell)
        
        return cell
    
    def _cell_overlaps(self, lows: torch.Tensor | None, highs: torch.Tensor | None, batch_index: int = 0) -> torch.Tensor:
        widths = self._cell_widths_params()
        dtype, device = widths.dtype, widths.device
        
        lo = torch.zeros((), dtype=dtype, device=device) if lows is None else torch.as_tensor(lows, dtype=dtype, device=device).reshape(-1)[0]
        hi = torch.ones((), dtype=dtype, device=device) if highs is None else torch.as_tensor(highs, dtype=dtype, device=device).reshape(-1)[0]
        
        edges = self.cell_edges(batch_index)
        # For each cell [edges[i], edges[i+1]], compute overlap with [lo, hi]
        cell_starts = edges[:-1]
        cell_ends = edges[1:]
        
        overlap_starts = torch.maximum(cell_starts, lo)
        overlap_ends = torch.minimum(cell_ends, hi)
        overlaps = (overlap_ends - overlap_starts).clamp(min=0)
        
        return overlaps
    
    def eval(self, y: torch.Tensor, index: int | None = None) -> torch.Tensor:
        alphas = self._alpha_params()
        y = torch.as_tensor(y, dtype=alphas.dtype, device=alphas.device).reshape(-1)
        
        if self._batch_size != 1:
            raise ValueError("eval currently requires parameter batch_size == 1")
        
        # Get cell indices for each y value
        cells = self._cell_index(y, batch_index=0)
        
        # Vectorized: create one-hot encoding of cell indices and multiply by values
        N = y.shape[0]
        # Create one-hot encoding: shape (N, n_basis)
        one_hot = torch.zeros(N, self._n_basis, dtype=alphas.dtype, device=alphas.device)
        one_hot.scatter_(1, cells.unsqueeze(1), 1.0)
        
        # Multiply by alpha or beta values
        result_alpha = one_hot * alphas[0]
        result_beta = one_hot / alphas[0]
        
        if index == 0:
            return result_alpha
        if index == 1:
            return result_beta
        if index is None:
            return torch.stack([result_alpha, result_beta], dim=1)
        raise ValueError("index must be 0, 1, or None")
    
    def Omega1(self, index: int, lows: torch.Tensor = None, highs: torch.Tensor = None) -> torch.Tensor:
        if index not in (0, 1):
            raise ValueError("index must be 0 or 1")
        
        alphas = self._alpha_params()
        batch_size = self._batch_size
        
        result = torch.zeros(batch_size, self._n_basis, dtype=alphas.dtype, device=alphas.device)
        
        for b in range(batch_size):
            overlaps = self._cell_overlaps(lows, highs, batch_index=b)
            if index == 0:
                result[b] = alphas[b] * overlaps
            else:  # index == 1
                result[b] = (1.0 / alphas[b]) * overlaps
        
        return result
    
    def Omega2(self, lows: torch.Tensor = None, highs: torch.Tensor = None) -> Matrix:
        """Cross Gram ``<alpha_i, beta_j>`` over ``[lows, highs]``.

        Because ``alpha_i beta_i = 1`` on cell ``i`` and both vanish elsewhere,

            <alpha_i, beta_j> = delta_ij * overlap(cell_i, [lows, highs]).

        On the full domain this is ``diag(widths)``.
        """
        alphas = self._alpha_params()
        dtype, device = alphas.dtype, alphas.device
        diag_values = torch.zeros(self._batch_size, self._n_basis, dtype=dtype, device=device)
        for b in range(self._batch_size):
            diag_values[b] = self._cell_overlaps(lows, highs, batch_index=b)
        return Diagonal(diag_values)
    
    def bounds(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        alphas = self._alpha_params()
        zeros = torch.zeros_like(alphas)
        
        if index == 0:
            # alpha: min is 0, max is alpha_i
            return zeros, alphas
        if index == 1:
            # beta: min is 0, max is 1/alpha_i
            return zeros, 1.0 / alphas
        raise ValueError("index must be 0 or 1")
    
    def infimum(self, index: int) -> torch.Tensor:
        return self.bounds(index)[0]
    
    def supremum(self, index: int) -> torch.Tensor:
        return self.bounds(index)[1]
        

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

    Lambda = I, so ``<alpha_i, beta_j> = delta_ij``. None of the knot,
    Gram, or dual maps are trainable.

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

    def __init__(
        self,
        n_basis: int,
        degree: int = 3,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        if degree < 0:
            raise ValueError("degree must be >= 0")
        if n_basis < degree + 1:
            raise ValueError(
                "n_basis must be at least degree + 1 "
                "for an open B-spline basis"
            )

        MutualPairBasis.__init__(self, 1, 1, n_basis, ())
        torch.nn.Module.__init__(self)

        if dtype is None:
            dtype = torch.float32

        self._degree = degree
        self._n_spans = n_basis - degree

        self.register_buffer("_breaks", torch.linspace(0.0, 1.0, self._n_spans + 1, dtype=dtype, device=device))
        self.register_buffer(
            "_knots",
            torch.cat(
                [
                    torch.zeros(degree + 1, dtype=dtype, device=device),
                    self._breaks[1:-1],
                    torch.ones(degree + 1, dtype=dtype, device=device),
                ]
            ),
        )

        # Product of two degree-p splines has degree 2p on each span.
        # p + 1 Gauss points integrate degree <= 2p + 1 exactly.
        qx, qw = leggauss(degree + 1)
        self.register_buffer("_quad_x", torch.as_tensor(qx, dtype=dtype, device=device))
        self.register_buffer("_quad_w", torch.as_tensor(qw, dtype=dtype, device=device))

        mass, G = self._interval_moments_raw(
            torch.zeros((), dtype=dtype, device=device),
            torch.ones((), dtype=dtype, device=device),
        )
        G = 0.5 * (G + G.T)
        ids = torch.arange(n_basis, device=G.device)
        band_mask = torch.abs(ids[:, None] - ids[None, :]) <= degree
        G = torch.where(band_mask, G, torch.zeros_like(G))

        self.register_buffer("_alpha_mass", mass)
        self.register_buffer("_gram_matrix", G)
        self.register_buffer("_gram_inv", torch.linalg.inv(G))

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

    def dtype_device(self):
        return self._knots.dtype, self._knots.device

    def _get_gram_diag(self) -> torch.Tensor:
        return torch.ones(self._batch_size, self._n_basis, dtype=self._knots.dtype, device=self._knots.device)

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
        y = torch.as_tensor(y, dtype=self._knots.dtype, device=self._knots.device).reshape(-1).clamp(0.0, 1.0)

        t = self._knots
        p = self._degree
        m = self._n_basis
        n = m - 1

        # Last nonempty span is [t[n], t[n+1]]; map the right endpoint there.
        span = torch.searchsorted(t, y, right=True) - 1
        span = torch.where(y >= t[n + 1], torch.full_like(span, n), span)
        span = span.clamp(p, n)

        # Compact Cox–de Boor: Nloc[:, j] = N_{span-p+j, p}(y).
        Nloc = y.new_zeros(y.shape[0], p + 1)
        Nloc[:, 0] = 1.0
        left = y.new_zeros(y.shape[0], p + 1)
        right = y.new_zeros(y.shape[0], p + 1)
        for j in range(1, p + 1):
            left[:, j] = y - t[span + 1 - j]
            right[:, j] = t[span + j] - y
            saved = torch.zeros_like(y)
            for r in range(j):
                tmp = Nloc[:, r] / (right[:, r + 1] + left[:, j - r])
                Nloc[:, r] = saved + right[:, r + 1] * tmp
                saved = left[:, j - r] * tmp
            Nloc[:, j] = saved

        idx = span.unsqueeze(1) - p + torch.arange(p + 1, device=y.device)
        out = y.new_zeros(y.shape[0], m)
        return out.scatter(1, idx, Nloc)

    def _solve_gram(self, rhs: torch.Tensor) -> torch.Tensor:
        """Solve ``G x = rhs`` along the last dimension via ``x = rhs @ G^{-1}``."""
        rhs = torch.as_tensor(rhs, dtype=self._gram_inv.dtype, device=self._gram_inv.device)
        if rhs.shape[-1] != self._n_basis:
            raise ValueError("Last rhs dimension must equal n_basis")
        return rhs @ self._gram_inv

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

        weighted = values * (half[:, None] * self._quad_w[None, :]).unsqueeze(-1)
        mass = weighted.sum(dim=(0, 1))
        gram = weighted.reshape(-1, self._n_basis).T @ values.reshape(-1, self._n_basis)
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
        return beta_mass.unsqueeze(0).expand(self._batch_size, -1)

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
        if self._alpha_power_coeffs is not None:
            return self._alpha_power_coeffs

        p = self._degree
        q = p + 1
        dtype, device = self.dtype_device()

        # Interior interpolation nodes avoid ambiguity at knots.
        nodes, _ = leggauss(q)
        u = 0.5 * (torch.as_tensor(nodes, dtype=dtype, device=device) + 1.0)
        V = torch.stack([u ** r for r in range(q)], dim=-1)

        left = self._breaks[:-1]
        right = self._breaks[1:]
        x = left[:, None] + (right - left)[:, None] * u[None, :]
        values = self._eval_alpha(x.reshape(-1)).reshape(self._n_spans, q, self._n_basis)

        # (span, basis, power)
        coeff = torch.linalg.solve(V, values).permute(0, 2, 1)

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
    
    def Omega2_diag(self):
        return torch.eye(self._n_basis)

class NFPairBasis(torch.nn.Module, MutualPairBasis):
    """Mutual pair split from index-conditioned normalizing-flow densities.

    For index embedding ``e_i`` and conditional density ``n_i(x)``:

        alpha_i(x) = s(x, e_i)
        beta_i(x) = n_i(x) / alpha_i(x)

    The splitter is evaluated on the concatenated vector ``[x, e_i]`` and is
    expected to be nonnegative. Its output is clamped below by ``eps`` to keep
    the quotient finite. All point/index pairs are evaluated in one splitter
    call and one conditional-flow call.
    """

    def __init__(
        self,
        nf: ConditionalNormalizingFlow | CompositeConditionalModel,
        splitter: torch.nn.Module,
        embedding: torch.nn.Embedding,
        eps: float = 1e-6,
        coeffs: tuple[Parameters | None, Parameters | None] | None = None,
    ):
        torch.nn.Module.__init__(self)
        if nf.dim < 1:
            raise ValueError("nf.dim must be at least 1")
        if embedding.num_embeddings < 1:
            raise ValueError("embedding must contain at least one index")
        if embedding.embedding_dim != nf.conditioner_dim:
            raise ValueError(
                f"embedding dim {embedding.embedding_dim} must match "
                f"nf conditioner_dim {nf.conditioner_dim}"
            )
        if eps <= 0:
            raise ValueError("eps must be positive")

        MutualPairBasis.__init__(
            self,
            nf.dim,
            1,
            embedding.num_embeddings,
            (),
            coeffs,
        )
        self.nf = nf
        self.splitter = splitter
        self.index_embedding = embedding
        self.eps = eps

    def dtype_device(self):
        weight = self.index_embedding.weight
        return weight.dtype, weight.device

    def _as_data(self, y: torch.Tensor) -> torch.Tensor:
        dtype, device = self.dtype_device()
        y = torch.as_tensor(y, dtype=dtype, device=device)
        if y.ndim == 1:
            y = y.unsqueeze(-1) if self._dim == 1 else y.unsqueeze(0)
        if y.ndim != 2 or y.shape[1] != self._dim:
            raise ValueError(
                f"y must have shape (n_data, {self._dim}), got {tuple(y.shape)}"
            )
        return y

    def _expanded_inputs(
        self,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flatten the Cartesian product of data points and basis indices."""
        n_data, m = y.shape[0], self._n_basis
        indices = torch.arange(m, device=y.device)
        conditioners = self.index_embedding(indices).to(dtype=y.dtype)
        y_pairs = y[:, None, :].expand(-1, m, -1)
        c_pairs = conditioners[None, :, :].expand(n_data, -1, -1)
        flat_y = y_pairs.reshape(n_data * m, self._dim)
        flat_c = c_pairs.reshape(n_data * m, conditioners.shape[-1])
        splitter_inputs = torch.cat((flat_y, flat_c), dim=-1)
        return flat_y, flat_c, splitter_inputs

    def _alpha_from_inputs(
        self,
        splitter_inputs: torch.Tensor,
        n_data: int,
    ) -> torch.Tensor:
        raw = self.splitter(splitter_inputs)
        expected = n_data * self._n_basis
        if raw.numel() != expected:
            raise ValueError(
                "splitter must return one value per point/index pair; "
                f"expected {expected} values, got shape {tuple(raw.shape)}"
            )
        return raw.reshape(n_data, self._n_basis).clamp_min(self.eps)

    def flow_density(self, y: torch.Tensor) -> torch.Tensor:
        """Evaluate every ``n_i(x)`` in one conditional-flow call."""
        y = self._as_data(y)
        flat_y, flat_c, _ = self._expanded_inputs(y)
        return self.nf(flat_y, conditioner=flat_c).reshape(
            y.shape[0], self._n_basis
        )

    def eval(self, y: torch.Tensor | None = None, index: int | None = None):
        if y is None:
            return torch.nn.Module.eval(self)
        if index not in (0, 1, None):
            raise ValueError("index must be 0, 1, or None")

        y = self._as_data(y)
        flat_y, flat_c, splitter_inputs = self._expanded_inputs(y)
        alpha = self._alpha_from_inputs(splitter_inputs, y.shape[0])
        if index == 0:
            return alpha

        density = self.nf(flat_y, conditioner=flat_c).reshape(
            y.shape[0], self._n_basis
        )
        beta = density / alpha
        if index == 1:
            return beta
        return torch.stack((alpha, beta), dim=1)

    def Omega2_diag(self) -> torch.Tensor:
        """Known matched Gram entries ``integral alpha_i beta_i = 1``."""
        dtype, device = self.dtype_device()
        return torch.ones(
            self._batch_size,
            self._n_basis,
            dtype=dtype,
            device=device,
        )


class MaskedGramMutualBasis(torch.nn.Module, MutualPairBasis):
    """
    Element wise product of a masking basis and a free basis to achieve a diagonal Gram matrix.
    """

    def __init__(
        self,
        masking_basis: DisjointSupport1DPWCBasis | Orthogonal1DPWCBasis,
        sacrificial_index: int,
        free_basis: VolumePreservingPairBasis,
        coeffs: tuple[Parameters | None, Parameters | None] | None = None,
    ):
        torch.nn.Module.__init__(self)
        if masking_basis.dim() != 1:
            raise ValueError("masking_basis must be 1-dimensional")
        dim = free_basis.dim() + 1
        if not (0 <= sacrificial_index < dim):
            raise ValueError(f"sacrificial_index must be in [0, {dim}), got {sacrificial_index}")
        if masking_basis.n_basis_functions() != free_basis.n_basis_functions():
            raise ValueError(
                f"masking_basis n_basis {masking_basis.n_basis_functions()} must match "
                f"free_basis n_basis {free_basis.n_basis_functions()}"
            )

        MutualPairBasis.__init__(
            self,
            dim,
            masking_basis.batch_size(),
            masking_basis.n_basis_functions(),
            masking_basis._params,
            coeffs,
        )
        self.masking_basis = masking_basis
        self.free_basis = free_basis
        self.sacrificial_index = sacrificial_index
        self._pwc_param_modules = torch.nn.ModuleList(
            [p for p in masking_basis._params if p.is_module()]
        )

    def dtype_device(self):
        return self.masking_basis.dtype_device()

    def _split_coords(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = torch.as_tensor(y)
        if y.ndim == 1:
            y = y.unsqueeze(0)
        if y.ndim != 2 or y.shape[1] != self._dim:
            raise ValueError(f"y must have shape (n_data, {self._dim}), got {tuple(y.shape)}")
        l = self.sacrificial_index
        rest = [i for i in range(self._dim) if i != l]
        return y[:, l], y[:, rest]

    def eval(self, y: torch.Tensor | None = None, index: int | None = None):
        if y is None:
            return torch.nn.Module.eval(self)
        x_l, x_rest = self._split_coords(y)
        return self.masking_basis.eval(x_l, index) * self.free_basis.eval(x_rest, index)

    def Omega1(self, index: int, lows: torch.Tensor = None, highs: torch.Tensor = None) -> torch.Tensor:
        #TODO
        pass

    def Omega2(self, lows: torch.Tensor = None, highs: torch.Tensor = None) -> Matrix:
        if lows is not None or highs is not None:
            raise ValueError("MaskedGramMutualBasis.Omega2 is only defined on the full domain")
        return self.masking_basis.Omega2().mul_diag_right(self.free_basis.Omega2_diag())

    def supremum(self, index: int) -> torch.Tensor:
        return self.masking_basis.supremum(index) * self.free_basis.supremum(index)


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
        for param in self.masking_basis._params:
            leaves = list(param.parameters()) + list(param.buffers()) if param.is_module() else [param()]
            for t in leaves:
                key.append((t.data_ptr(), t._version, bool(t.requires_grad)))
        return tuple(key)

    def constant_shifts(self) -> tuple[torch.Tensor, torch.Tensor]:
        key = self._shift_cache_key()
        cached = getattr(self, "_shift_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        a = -self.masking_basis.infimum(0) * self.free_basis.supremum(0)
        b = -self.masking_basis.infimum(1) * self.free_basis.supremum(1)
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
