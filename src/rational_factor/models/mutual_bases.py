from __future__ import annotations

import torch
import math
from rational_factor.models.basis_functions import Basis
from rational_factor.models.parameters import Parameters
from rational_factor.models.qs_matrix import (
    Order1QSGenerators,
    Order1Quasiseparable,
    Order1QuasiseparableFactorization,
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

    def get_basis(self, index: int) -> Basis:
        raise NotImplementedError("get_basis is not implemented for this mutual basis")

    def Omega1(self, index: int, lows: torch.Tensor = None, highs: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError("Omega1 is not implemented for this mutual basis")

    def Omega2(self, lows: torch.Tensor = None, highs: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError("Omega2 is not implemented for this mutual basis")

    def eval(self, y: torch.Tensor, index: int = None):
        raise NotImplementedError("eval is not implemented for this mutual basis")

    def dim(self) -> int:
        return self._dim

    def batch_size(self) -> int:
        return self._batch_size

    def n_basis_functions(self) -> int:
        return self._n_basis


class MutualPairMemberBasis(Basis):
    def __init__(self, owner: MutualPairBasis, index: int):
        assert index in (0, 1), "index must be 0 or 1"
        super().__init__(
            owner._dim,
            owner._batch_size,
            owner._n_basis,
            owner._params,
            owner._coeffs[index],
        )
        self.owner = owner
        self.index = index

    def _is_mate(self, other: MutualPairMemberBasis | list[MutualPairMemberBasis]) -> bool:
        if isinstance(other, MutualPairMemberBasis):
            return self.owner is other.owner and self.index != other.index
        same_owner = all(self.owner is basis.owner for basis in other)
        unique_indices = len({basis.index for basis in other}) == len(other)
        return same_owner and unique_indices

    def Omega1(self, lows: torch.Tensor = None, highs: torch.Tensor = None):
        return self.owner.Omega1(self.index, lows, highs) * self.coeffs()

    def Omega2(self, other: MutualPairMemberBasis, lows: torch.Tensor = None, highs: torch.Tensor = None):
        if not self._is_mate(other):
            raise ValueError("Omega2 is not defined for non-mate bases")
        G = self.owner.Omega2(lows, highs)
        if self.index == 1:
            G = torch.transpose(G, -2, -1)
        return G * self.coeffs()[:, :, None] * other.coeffs()[:, None, :]

    def __call__(self, y: torch.Tensor):
        return self.owner.eval(y, self.index) * self.coeffs()


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

    def get_basis(self, index: int) -> MutualPairMemberBasis:
        return MutualPairMemberBasis(self, index)

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

    def Omega2(self, lows: torch.Tensor = None, highs: torch.Tensor = None) -> torch.Tensor:
        """Cross Gram ``<alpha_i, beta_j>``, shape ``(batch, m, m)``."""
        if lows is None and highs is None:
            return torch.diag_embed(self._get_gram_diag())
        w = self._cell_overlaps(lows, highs)
        a, b = self._cell_values(0), self._cell_values(1)
        return torch.einsum("k,bki,bkj->bij", w, a, b)

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
        """Exact inf/sup of every basis function, each shape ``(batch, m)``."""
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
