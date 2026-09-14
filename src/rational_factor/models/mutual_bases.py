from __future__ import annotations

import math
from math import ceil

import torch

import numpy as np
from numpy.polynomial.legendre import leggauss

from normalizing_flow.vp_flow import ConditionalUnitBoxVolumePreservingFlow
from rational_factor.models.basis_functions import Basis, BetaBasis, GaussianBasis
from rational_factor.models.density_model import ConditionalDensityModel
from rational_factor.models.domain_transformation import DomainTF
from rational_factor.models.parameters import Parameters, Order1QuasiseparableFactorization
from rational_factor.models.structured_matrices import (
    Banded,
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
    
    def supremum_bound(self) -> torch.Tensor:
        raise NotImplementedError("supremum_bound is not implemented for this mutual basis")
    
    def infemum_bound(self) -> torch.Tensor:
        raise NotImplementedError("infemum_bound is not implemented for this mutual basis")


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
        base: BetaBasis | GaussianBasis,
        splitter: torch.nn.Module,
        embedding: torch.nn.Embedding,
        flow: ConditionalUnitBoxVolumePreservingFlow | None = None,
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
        z, ladj = self.flow.transform(y_rep, conditioner=c_rep)
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
    
    def Omega1(self, index: int, lows: torch.Tensor = None, highs: torch.Tensor = None) -> torch.Tensor:
        if lows is not None or highs is not None:
            raise NotImplementedError("Restricted-domain moments are not implemented")
        if index not in (0, 1):
            raise ValueError("index must be 0 or 1")
        dtype, device = self.dtype_device()
        if self._dim == 0:
            return torch.ones(self._batch_size, self._n_basis, dtype=dtype, device=device)
        raise NotImplementedError(
            "VolumePreservingPairBasis.Omega1 is only closed-form for 0-d rest space"
        )

    def Omega2_diag(self) -> torch.Tensor:
        """Matched Gram diagonal ``∫ α_i β_i = 1``."""
        dtype, device = self.dtype_device()
        return torch.ones(self._batch_size, self._n_basis, dtype=dtype, device=device)


class NormalizedProductPairBasis(torch.nn.Module, MutualPairBasis):
    """Mutual pair from conditional base density ``q(·|e)`` and map ``T(·|e)``.

    Each basis index ``i`` has embedding ``e_i``. With ``u = T(x | e_i)``:

        α_i(x) = |det J_{T(·|e_i)}(x)|
        β_i(x) = q(T(x | e_i) | e_i)

    so ``α_i β_i`` is the conditional change-of-variables density. Matched
    products integrate to one (``Omega2_diag`` is the identity). ``Omega1(0)``
    returns ones. ``supremum_bound(1)`` is ``q.supremum_bound(e_i)`` for each
    index embedding.
    """

    def __init__(
        self,
        base_box_distribution: ConditionalDensityModel,
        domain_transformation: DomainTF,
        embedding: torch.nn.Embedding,
        coeffs: tuple[Parameters | None, Parameters | None] | None = None,
    ):
        torch.nn.Module.__init__(self)
        n_basis = embedding.num_embeddings
        if n_basis < 1:
            raise ValueError("embedding must contain at least one index")
        #if base_box_distribution.dim != domain_transformation.dim:
        #    raise ValueError(
        #        f"base dim {base_box_distribution.dim} must match "
        #        f"domain_transformation dim {domain_transformation.dim}"
        #    )
        if embedding.embedding_dim != base_box_distribution.conditioner_dim:
            raise ValueError(
                f"embedding dim {embedding.embedding_dim} must match "
                f"base conditioner_dim {base_box_distribution.conditioner_dim}"
            )
        #if domain_transformation.context_features is None:
        #    raise ValueError(
        #        "domain_transformation must be conditional "
        #        "(set context_features)"
        #    )
        #if embedding.embedding_dim != domain_transformation.context_features:
        #    raise ValueError(
        #        f"embedding dim {embedding.embedding_dim} must match "
        #        f"domain_transformation context_features "
        #        f"{domain_transformation.context_features}"
        #    )

        MutualPairBasis.__init__(
            self,
            base_box_distribution.dim,
            1,
            n_basis,
            (),
            coeffs,
        )
        self.base_box_distribution = base_box_distribution
        self.domain_transformation = domain_transformation
        self.index_embedding = embedding

    def dtype_device(self):
        weight = self.index_embedding.weight
        return weight.dtype, weight.device

    def _index_conditioners(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        idx = torch.arange(self._n_basis, device=device)
        return self.index_embedding(idx).to(dtype=dtype)

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

    def _expanded_inputs(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Flatten the Cartesian product of data points and basis indices."""
        n_data, m = y.shape[0], self._n_basis
        cond = self._index_conditioners(y.dtype, y.device)
        y_rep = y.unsqueeze(1).expand(-1, m, -1).reshape(n_data * m, self._dim)
        c_rep = cond.unsqueeze(0).expand(n_data, -1, -1).reshape(n_data * m, -1)
        return y_rep, c_rep

    def eval(self, y: torch.Tensor | None = None, index: int | None = None):
        if y is None:
            return torch.nn.Module.eval(self)
        if index not in (0, 1, None):
            raise ValueError("index must be 0, 1, or None")

        y = self._as_data(y)
        n_data, m = y.shape[0], self._n_basis
        y_rep, c_rep = self._expanded_inputs(y)
        u, ladj = self.domain_transformation.forward(y_rep, context=c_rep)
        alpha = torch.exp(ladj).reshape(n_data, m)
        if index == 0:
            return alpha

        beta = self.base_box_distribution(u, conditioner=c_rep).reshape(n_data, m)
        if index == 1:
            return beta
        return torch.stack((alpha, beta), dim=1)

    def Omega2_diag(self) -> torch.Tensor:
        """Matched Gram diagonal ``∫ α_i β_i = 1``."""
        dtype, device = self.dtype_device()
        return torch.ones(self._batch_size, self._n_basis, dtype=dtype, device=device)

    def Omega1(self, index: int, lows: torch.Tensor = None, highs: torch.Tensor = None) -> torch.Tensor:
        if lows is not None or highs is not None:
            raise NotImplementedError("Restricted-domain moments are not implemented")
        if index == 0:
            dtype, device = self.dtype_device()
            return torch.ones(self._batch_size, self._n_basis, dtype=dtype, device=device)
        if index == 1:
            raise NotImplementedError("Omega1 is only defined for alpha (index=0)")
        raise ValueError("index must be 0 or 1")

    def supremum_bound(self, index: int) -> torch.Tensor:
        if index == 1:
            dtype, device = self.dtype_device()
            cond = self._index_conditioners(dtype, device)
            bound = self.base_box_distribution.supremum_bound(cond).to(
                dtype=dtype, device=device
            )
            ones = torch.ones(self._batch_size, self._n_basis, dtype=dtype, device=device)
            return ones * bound.reshape(-1)
        raise NotImplementedError("supremum_bound is only defined for beta (index=1)")

    def supremum(self, index: int) -> torch.Tensor:
        return self.supremum_bound(index)


# Backward-compatible alias.
NFPairBasis = NormalizedProductPairBasis


class MaskedGramMutualBasis(torch.nn.Module, MutualPairBasis):
    """
    Elementwise product of a masking basis and a free basis to achieve a
    diagonal matched Gram matrix.

    If ``free_basis`` is ``None``, the rest space is 0-dimensional and the free
    pair is the identity ``α = β = 1``.

    If ``swap_alpha_beta`` is True, logical index 0 is the underlying beta and
    index 1 is the underlying alpha; ``Omega2`` is transposed accordingly so
    it remains ``∫ logical_α_i logical_β_j``.
    """

    def __init__(
        self,
        masking_basis: LocalBSplineMutualBasis,
        sacrificial_index: int,
        free_basis: VolumePreservingPairBasis | NormalizedProductPairBasis | None = None,
        coeffs: tuple[Parameters | None, Parameters | None] | None = None,
        swap_alpha_beta: bool = False,
    ):
        torch.nn.Module.__init__(self)
        if masking_basis.dim() != 1:
            raise ValueError("masking_basis must be 1-dimensional")

        if free_basis is None:
            dim = 1
            if sacrificial_index != 0:
                raise ValueError(
                    "sacrificial_index must be 0 when free_basis is None (0-d rest space)"
                )
        else:
            dim = free_basis.dim() + 1
            if not (0 <= sacrificial_index < dim):
                raise ValueError(
                    f"sacrificial_index must be in [0, {dim}), got {sacrificial_index}"
                )
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
        self.swap_alpha_beta = bool(swap_alpha_beta)

    def dtype_device(self):
        return self.masking_basis.dtype_device()

    def _pair_index(self, index: int | None) -> int | None:
        """Map logical (α, β) index to underlying (α, β) index."""
        if self.swap_alpha_beta and index in (0, 1):
            return 1 - index
        return index

    def _ones(self, n_data: int | None = None) -> torch.Tensor:
        dtype, device = self.dtype_device()
        if n_data is None:
            return torch.ones(self._batch_size, self._n_basis, dtype=dtype, device=device)
        return torch.ones(n_data, self._n_basis, dtype=dtype, device=device)

    def _free_eval(self, x_rest: torch.Tensor, index: int | None):
        if self.free_basis is None:
            ones = self._ones(x_rest.shape[0])
            if index in (0, 1):
                return ones
            if index is None:
                return torch.stack([ones, ones], dim=1)
            raise ValueError("index must be 0, 1, or None")
        return self.free_basis.eval(x_rest, index)

    def _free_Omega1(self, index: int) -> torch.Tensor:
        if self.free_basis is None:
            if index not in (0, 1):
                raise ValueError("index must be 0 or 1")
            # Empty product over no free coordinates is the constant 1.
            return self._ones()
        return self.free_basis.Omega1(index)

    def _free_Omega2_diag(self) -> torch.Tensor:
        if self.free_basis is None:
            return self._ones()
        return self.free_basis.Omega2_diag()

    def _free_supremum(self, index: int) -> torch.Tensor:
        if self.free_basis is None:
            if index not in (0, 1):
                raise ValueError("index must be 0 or 1")
            return self._ones()
        return self.free_basis.supremum(index)

    def _split_coords(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = torch.as_tensor(y)
        if y.ndim == 1:
            y = y.unsqueeze(0)
        if y.ndim != 2 or y.shape[1] != self._dim:
            raise ValueError(f"y must have shape (n_data, {self._dim}), got {tuple(y.shape)}")
        l = self.sacrificial_index
        rest = [i for i in range(self._dim) if i != l]
        return y[:, l], y[:, rest]

    def _omega2_unswapped(self) -> Matrix:
        return self.masking_basis.Omega2().mul_diag_right(self._free_Omega2_diag())

    def eval(self, y: torch.Tensor | None = None, index: int | None = None):
        if y is None:
            return torch.nn.Module.eval(self)
        x_l, x_rest = self._split_coords(y)
        raw = self._pair_index(index)
        if raw is None:
            out = self.masking_basis.eval(x_l, None) * self._free_eval(x_rest, None)
            return out.flip(1) if self.swap_alpha_beta else out
        return self.masking_basis.eval(x_l, raw) * self._free_eval(x_rest, raw)

    def Omega1(self, index: int, lows: torch.Tensor = None, highs: torch.Tensor = None) -> torch.Tensor:
        if lows is not None or highs is not None:
            raise ValueError("MaskedGramMutualBasis.Omega1 is only defined on the full domain")
        raw = self._pair_index(index)
        return self.masking_basis.Omega1(raw) * self._free_Omega1(raw)

    def Omega2(self, lows: torch.Tensor = None, highs: torch.Tensor = None) -> Matrix:
        if lows is not None or highs is not None:
            raise ValueError("MaskedGramMutualBasis.Omega2 is only defined on the full domain")
        G = self._omega2_unswapped()
        return G.T if self.swap_alpha_beta else G

    def supremum(self, index: int) -> torch.Tensor:
        # Local B-spline alpha is a partition of unity (≤ 1), so
        # α = α_mask α_free ≤ α_free ≤ free.supremum(0).
        if self._pair_index(index) == 0:
            return self._free_supremum(0)
        raise NotImplementedError("MaskedGramMutualBasis.supremum is only implemented for alpha")


class PositiveMaskedGramMutualBasis(MaskedGramMutualBasis):
    r"""Masked pair with a positivity correction on beta only.

    Alpha is the unsigned product (assumed nonnegative):

        α_i(y) = α^m_i(x_s) α^f_i(x_rest)

    Beta corrects the possibly-negative masking dual with
    ``b_j = relu(-β^m_j)`` and a free-beta bound ``u_b ≥ β^f``:

        β_j(y) = β^m_j(x_s) β^f_j(x_rest) + u_{b,j} \, b_j(x_s)

    When ``β^m_j ≥ 0`` the correction vanishes. When ``β^m_j < 0``,
    ``β_j = β^m_j (β^f_j - u_{b,j}) ≥ 0`` provided ``β^f_j ≤ u_{b,j}``.
    If ``free_basis is None`` then ``α^f = β^f = u_b = 1`` and
    ``β = relu(β^m)``.

    The matched Gram gains one extra term from the correction:

        ∫ α_i β_j = (Ω²_mask ∘ Ω²_free)_{ij}
            + G^{αb}_{ij} (Ω¹_free_α)_i u_{b,j}

    i.e. ``Omega2_alpha_b().mul_diag_left(free.Omega1(0)).mul_diag_right(u_b)``.

    With ``swap_alpha_beta=True``, logical index 0 is the corrected beta and
    index 1 is unsigned alpha; ``Omega2`` is transposed to match.
    """

    def eval(self, y: torch.Tensor | None = None, index: int | None = None):
        if y is None:
            return torch.nn.Module.eval(self)
        if index not in (0, 1, None):
            raise ValueError("index must be 0, 1, or None")

        x_l, x_rest = self._split_coords(y)
        u_b = self._free_supremum(1)
        alpha = self.masking_basis.eval(x_l, 0) * self._free_eval(x_rest, 0)
        beta = self.masking_basis.eval(x_l, 1) * self._free_eval(x_rest, 1) + self.masking_basis.eval_b(x_l) * u_b
        if self.swap_alpha_beta:
            alpha, beta = beta, alpha

        if index == 0:
            return alpha
        if index == 1:
            return beta
        return torch.stack([alpha, beta], dim=1)

    def Omega1(self, index: int, lows: torch.Tensor = None, highs: torch.Tensor = None) -> torch.Tensor:
        if lows is not None or highs is not None:
            raise ValueError("PositiveMaskedGramMutualBasis.Omega1 is only defined on the full domain")
        # Integral of unsigned alpha only; corrected beta needs ∫b.
        if self._pair_index(index) == 0:
            return self.masking_basis.Omega1(0) * self._free_Omega1(0)
        raise NotImplementedError(
            "PositiveMaskedGramMutualBasis.Omega1 for beta requires ∫b, not yet implemented"
        )

    def Omega2(self, lows: torch.Tensor = None, highs: torch.Tensor = None) -> Matrix:
        if lows is not None or highs is not None:
            raise ValueError("PositiveMaskedGramMutualBasis.Omega2 is only defined on the full domain")
        unsigned = self._omega2_unswapped()
        gab = self.masking_basis.Omega2_alpha_b()
        omega1 = self._free_Omega1(0)
        u_b = self._free_supremum(1)
        # extra_ij = Gab_ij * omega1_i * u_b_j  (underlying α row, β column)
        if omega1.shape[0] == 1 and u_b.shape[0] == 1:
            extra = gab.mul_diag_left(omega1[0]).mul_diag_right(u_b[0])
            G = DenseMatrix(unsigned.to_dense() + extra.to_dense())
        else:
            extra_dense = omega1.unsqueeze(-1) * gab.to_dense() * u_b.unsqueeze(-2)
            G = DenseMatrix(unsigned.to_dense() + extra_dense)
        return G.T if self.swap_alpha_beta else G

    def supremum(self, index: int) -> torch.Tensor:
        if self._pair_index(index) == 0:
            return self._free_supremum(0)
        raise NotImplementedError("PositiveMaskedGramMutualBasis.supremum is only implemented for alpha")
