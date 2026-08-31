from __future__ import annotations

import math

import torch

from normalizing_flow.vp_flow import VolumePreservingFlow
from rational_factor.models.basis_functions import Basis, BetaBasis
from rational_factor.models.gram import Omega2Gram
from rational_factor.models.lrpd import Rank1PlusDiagonal
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

    def get_basis(self, index: int, coeffs: Parameters | None = None) -> MutualPairMemberBasis:
        basis = MutualPairMemberBasis(self, index)
        if coeffs is not None:
            basis.set_coeffs(coeffs)
        return basis

    def Omega1(self, index: int, lows: torch.Tensor = None, highs: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError("Omega1 is not implemented for this mutual basis")

    def Omega2(self, lows: torch.Tensor = None, highs: torch.Tensor = None) -> Omega2Gram:
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

    def Omega2(self, other: MutualPairMemberBasis, lows: torch.Tensor = None, highs: torch.Tensor = None) -> Omega2Gram:
        if not self._is_mate(other):
            raise ValueError("Omega2 is not defined for non-mate bases")
        G = Omega2Gram(self.owner.Omega2(lows, highs))
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

    def Omega2(self, lows: torch.Tensor = None, highs: torch.Tensor = None) -> Omega2Gram:
        """Cross Gram ``<alpha_i, beta_j>``, shape ``(batch, m, m)``."""
        if lows is None and highs is None:
            return Omega2Gram(torch.diag_embed(self._get_gram_diag()))
        w = self._cell_overlaps(lows, highs)
        a, b = self._cell_values(0), self._cell_values(1)
        return Omega2Gram(torch.einsum("k,bki,bkj->bij", w, a, b))

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
        if index == 0:
            return ones
        if index == 1:
            return ones * self.base.supremum_bound()
        raise ValueError("index must be 0 or 1")


class MaskedGramMutualBasis(torch.nn.Module, MutualPairBasis):
    """Product of a 1D PWC pair on a sacrificial coordinate and a VP pair on the rest.

    For sacrificial index ``l`` in ``{0, …, d-1}``,

        α_i(x) = α^{pwc}_i(x_l) α^{vp}_i(x_{≠l})
        β_i(x) = β^{pwc}_i(x_l) β^{vp}_i(x_{≠l})

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

    def Omega2(self, lows: torch.Tensor = None, highs: torch.Tensor = None) -> Omega2Gram:
        if lows is not None or highs is not None:
            raise ValueError("MaskedGramMutualBasis.Omega2 is only defined on the full domain")
        return Omega2Gram(self.pwc.Omega2()).scale(self._domain_volume())

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

    def Omega2(self, lows: torch.Tensor = None, highs: torch.Tensor = None) -> Omega2Gram:
        unsigned = MaskedGramMutualBasis.Omega2(self, lows, highs)
        a, b = self.constant_shifts()
        vol = self._domain_volume()
        return Omega2Gram(Rank1PlusDiagonal(unsigned.diag(), a * vol, b))

    def supremum(self, index: int) -> torch.Tensor:
        a, b = self.constant_shifts()
        shift = a if index == 0 else b
        return super().supremum(index) + shift
