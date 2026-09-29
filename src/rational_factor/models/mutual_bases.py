from __future__ import annotations

import math
from math import ceil


from collections.abc import Sequence

import torch
import torch.nn.functional as F

from normalizing_flow.vp_flow import ConditionalUnitBoxVolumePreservingFlow
from rational_factor.models.basis_functions import Basis, SeparableBasis, BetaBasis, GaussianBasis
from rational_factor.models.density_model import ConditionalDensityModel
from nflows.transforms.base import Transform
from rational_factor.models.parameters import Parameters
from rational_factor.models.structured_matrices import (
    Banded,
    TTMatrix,
    DenseMatrix,
    Identity,
    Diagonal,
    Matrix,
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
            if flow.features != base.dim():
                raise ValueError("flow and base must have the same dimension")
            if embedding.embedding_dim != flow.context_features:
                raise ValueError(
                    f"embedding dim {embedding.embedding_dim} must match "
                    f"flow context_features {flow.context_features}"
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
        if self.flow is None or self.flow.features == 1:
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
        domain_transformation: Transform,
        embedding: torch.nn.Embedding,
        coeffs: tuple[Parameters | None, Parameters | None] | None = None,
    ):
        torch.nn.Module.__init__(self)
        n_basis = embedding.num_embeddings
        if n_basis < 1:
            raise ValueError("embedding must contain at least one index")
        #if base_box_distribution.features != domain_transformation.dim:
        #    raise ValueError(
        #        f"base dim {base_box_distribution.features} must match "
        #        f"domain_transformation dim {domain_transformation.dim}"
        #    )
        if embedding.embedding_dim != base_box_distribution.context_features:
            raise ValueError(
                f"embedding dim {embedding.embedding_dim} must match "
                f"base context_features {base_box_distribution.context_features}"
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
            base_box_distribution.features,
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


class TTMutualBasis(torch.nn.Module, MutualPairBasis):
    r"""Mutual basis whose m = p**d functions are represented by TT cores.

    Let ``phi_primitive`` and ``psi_primitive`` be SeparableBasis objects
    containing p primitive 1-D basis functions in each of d dimensions.

    A basis-function index is tensorized as

        i = (i_1, ..., i_d),    i_k = 0, ..., p-1,

    so the total number of multivariate basis functions is

        m = p**d.

    At dimension k, phi has a nonnegative coefficient core

        A_phi[k] : (r_{k-1}, p, p, r_k)

    whose entries are indexed as

        A_phi[k][a, i_k, alpha_k, b].

    The resulting basis function is

        phi_i(x)
          = sum_{alpha_1,...,alpha_d}
              A_phi[0][i_1,alpha_1]
              ...
              A_phi[d-1][i_d,alpha_d]
              prod_k primitive_phi[alpha_k](x_k),

    with matrix multiplication/contraction over the TT rank indices.

    psi is represented analogously.

    Because primitive functions and coefficient cores are nonnegative,
    all resulting phi_i and psi_j are nonnegative.

    The cross Gram is returned directly as a TTMatrix/MPO.  Its k-th core is

        W_k[(a,c), i, j, (b,d)]
          = sum_{alpha,beta}
              A_phi[k][a,i,alpha,b]
              H_k[alpha,beta]
              A_psi[k][c,j,beta,d],

    where

        H_k[alpha,beta]
          = <primitive_phi_alpha, primitive_psi_beta>_k.

    Thus if phi and psi have TT rank r, the Gram MPO has ranks at most r**2.

    Notes
    -----
    The current TTMatrix implementation is unbatched. Therefore Omega2()
    currently requires the primitive 1-D Gram to have batch size 1.
    eval(), however, supports ordinary data batches.
    """

    def __init__(
        self,
        phi_primitive: SeparableBasis,
        psi_primitive: SeparableBasis,
        rank: int | Sequence[int] = 1,
        *,
        init_std: float = 0.1,
    ):
        torch.nn.Module.__init__(self)

        # --------------------------------------------------------------
        # Validate primitive bases.
        # --------------------------------------------------------------
        if phi_primitive.dim() != psi_primitive.dim():
            raise ValueError(
                "phi_primitive and psi_primitive must have the same dimension, "
                f"got {phi_primitive.dim()} and {psi_primitive.dim()}"
            )

        if phi_primitive.batch_size() != psi_primitive.batch_size():
            raise ValueError(
                "phi_primitive and psi_primitive must have the same batch size, "
                f"got {phi_primitive.batch_size()} and "
                f"{psi_primitive.batch_size()}"
            )

        p = phi_primitive.n_basis_functions()

        if psi_primitive.n_basis_functions() != p:
            raise ValueError(
                "For this implementation, phi_primitive and psi_primitive "
                "must have the same number p of primitive 1-D basis functions. "
                f"Got {p} and {psi_primitive.n_basis_functions()}."
            )

        d = phi_primitive.dim()
        m = p**d

        phi_dtype, phi_device = phi_primitive.dtype_device()
        psi_dtype, psi_device = psi_primitive.dtype_device()

        if phi_dtype != psi_dtype:
            raise ValueError(
                "phi_primitive and psi_primitive must have the same dtype, "
                f"got {phi_dtype} and {psi_dtype}"
            )
        if phi_device != psi_device:
            raise ValueError(
                "phi_primitive and psi_primitive must be on the same device, "
                f"got {phi_device} and {psi_device}"
            )

        MutualPairBasis.__init__(
            self,
            dim=d,
            batch_size=phi_primitive.batch_size(),
            n_basis=m,
            params=(),
        )

        self.phi_primitive = phi_primitive
        self.psi_primitive = psi_primitive

        self._p = p
        self._ranks = self._normalize_ranks(d, rank)

        # --------------------------------------------------------------
        # Raw trainable cores.
        #
        # softplus(raw_core) is used everywhere below, guaranteeing
        # nonnegative TT coefficients.
        # --------------------------------------------------------------
        self.phi_raw_cores = torch.nn.ParameterList()
        self.psi_raw_cores = torch.nn.ParameterList()

        for k in range(d):
            r_left = self._ranks[k]
            r_right = self._ranks[k + 1]

            shape = (r_left, p, p, r_right)

            phi_raw = self._initial_raw_core(
                shape,
                p=p,
                r_right=r_right,
                dtype=phi_dtype,
                device=phi_device,
                std=init_std,
            )

            psi_raw = self._initial_raw_core(
                shape,
                p=p,
                r_right=r_right,
                dtype=psi_dtype,
                device=psi_device,
                std=init_std,
            )

            self.phi_raw_cores.append(torch.nn.Parameter(phi_raw))
            self.psi_raw_cores.append(torch.nn.Parameter(psi_raw))

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_ranks(
        d: int,
        rank: int | Sequence[int],
    ) -> tuple[int, ...]:
        """Return TT rank tuple ``(1, r_1, ..., r_{d-1}, 1)``."""

        if isinstance(rank, int):
            if rank < 1:
                raise ValueError(f"rank must be >= 1, got {rank}")
            return (1,) + (rank,) * max(d - 1, 0) + (1,)

        ranks = tuple(int(r) for r in rank)

        # Conveniently allow just the internal ranks.
        if len(ranks) == d - 1:
            ranks = (1,) + ranks + (1,)

        if len(ranks) != d + 1:
            raise ValueError(
                f"rank sequence must have length d-1={d - 1} "
                f"or d+1={d + 1}, got {len(ranks)}"
            )

        if ranks[0] != 1 or ranks[-1] != 1:
            raise ValueError(
                "TT boundary ranks must both equal 1, "
                f"got {ranks[0]} and {ranks[-1]}"
            )

        if any(r < 1 for r in ranks):
            raise ValueError(f"all TT ranks must be >= 1, got {ranks}")

        return ranks

    @staticmethod
    def _initial_raw_core(
        shape: tuple[int, int, int, int],
        *,
        p: int,
        r_right: int,
        dtype: torch.dtype,
        device: torch.device,
        std: float,
    ) -> torch.Tensor:
        """Initialize raw parameters at a modest positive softplus value."""

        # Avoid huge initial products/sums across dimensions.
        target = 1.0 / max(p * r_right, 1)

        target_t = torch.tensor(target, dtype=dtype, device=device)
        raw_mean = torch.log(torch.expm1(target_t))

        return raw_mean + std * torch.randn(
            shape,
            dtype=dtype,
            device=device,
        )

    def _phi_core(self, k: int) -> torch.Tensor:
        return F.softplus(self.phi_raw_cores[k])

    def _psi_core(self, k: int) -> torch.Tensor:
        return F.softplus(self.psi_raw_cores[k])

    # ------------------------------------------------------------------
    # Useful metadata
    # ------------------------------------------------------------------

    @property
    def mode_size(self) -> int:
        """Number p of function indices per TT site."""
        return self._p

    @property
    def tt_ranks(self) -> tuple[int, ...]:
        return self._ranks

    def dtype_device(self):
        core = self.phi_raw_cores[0]
        return core.dtype, core.device

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _eval_side(
        self,
        y: torch.Tensor,
        primitive: SeparableBasis,
        *,
        side: int,
    ) -> torch.Tensor:
        """Evaluate all p**d functions for one side.

        Returns
        -------
        Tensor
            Shape ``(batch, p**d)``.

        Flattening convention
        ---------------------
        The function grid

            (i_1, ..., i_d)

        is flattened in standard PyTorch row-major order, so ``i_d`` varies
        fastest. This is the same ordering used by TTMatrix when its row modes
        are ``(p, ..., p)``.
        """

        dtype, device = self.dtype_device()

        y = torch.as_tensor(
            y,
            dtype=dtype,
            device=device,
        )

        if y.ndim == 1:
            y = y.unsqueeze(0)

        if y.ndim != 2 or y.shape[1] != self._dim:
            raise ValueError(
                f"y must have shape (batch, {self._dim}), "
                f"got {tuple(y.shape)}"
            )

        # (batch, d, p)
        primitive_values = primitive.eval_dim(y)

        if primitive_values.shape != (
            y.shape[0],
            self._dim,
            self._p,
        ):
            raise ValueError(
                "primitive.eval_dim returned an unexpected shape: "
                f"expected {(y.shape[0], self._dim, self._p)}, "
                f"got {tuple(primitive_values.shape)}"
            )

        cores = (
            [self._phi_core(k) for k in range(self._dim)]
            if side == 0
            else [self._psi_core(k) for k in range(self._dim)]
        )

        batch = y.shape[0]

        # Start with the left TT boundary rank r_0 = 1.
        #
        # During the sweep:
        #
        #   out.shape =
        #       (batch, i_1, ..., i_k, r_k)
        #
        out = torch.ones(
            batch,
            1,
            dtype=dtype,
            device=device,
        )

        for k, core in enumerate(cores):
            # core:
            #   (r_{k-1}, i_k, alpha_k, r_k)
            #
            # primitive_values[:, k]:
            #   (batch, alpha_k)
            #
            # local:
            #   (batch, r_{k-1}, i_k, r_k)
            local = torch.einsum(
                "nu,aiur->nair",
                primitive_values[:, k, :],
                core,
            )

            # Contract the previous TT rank while appending the new
            # function-grid index i_k.
            #
            # before:
            #   out   : (batch, i_1, ..., i_{k-1}, r_{k-1})
            #   local : (batch, r_{k-1}, i_k, r_k)
            #
            # after:
            #   out   : (batch, i_1, ..., i_k, r_k)
            out = torch.einsum(
                "n...a,nair->n...ir",
                out,
                local,
            )

        # Final TT boundary rank is 1.
        out = out.squeeze(-1)

        return out.reshape(batch, self._n_basis)

    def eval(
        self,
        y: torch.Tensor | None = None,
        index: int | None = None,
    ):
        """Evaluate phi, psi, or both.

        Parameters
        ----------
        y:
            Input with shape ``(batch, d)``.
        index:
            ``0`` -> phi
            ``1`` -> psi
            ``None`` -> both

        Returns
        -------
        index == 0 or 1:
            ``(batch, m)``
        index is None:
            ``(batch, 2, m)``
        """

        # Preserve torch.nn.Module.eval() behavior.
        if y is None:
            return torch.nn.Module.eval(self)

        if index not in (0, 1, None):
            raise ValueError(
                f"index must be 0, 1, or None, got {index!r}"
            )

        if index == 0:
            return self._eval_side(
                y,
                self.phi_primitive,
                side=0,
            )

        if index == 1:
            return self._eval_side(
                y,
                self.psi_primitive,
                side=1,
            )

        phi = self._eval_side(
            y,
            self.phi_primitive,
            side=0,
        )
        psi = self._eval_side(
            y,
            self.psi_primitive,
            side=1,
        )

        return torch.stack((phi, psi), dim=1)

    # ------------------------------------------------------------------
    # Gram / Omega2
    # ------------------------------------------------------------------

    def Omega2(
        self,
        lows: torch.Tensor = None,
        highs: torch.Tensor = None,
    ) -> Matrix:
        r"""Return the exact phi/psi cross Gram as a TTMatrix.

        For dimension k, let

            H_k[alpha,beta]
              = integral primitive_phi_alpha(x_k)
                         primitive_psi_beta(x_k) dx_k.

        The Gram MPO core is

            W_k[(a,c), i, j, (b,d)]
              =
                sum_{alpha,beta}
                    A_phi[k][a,i,alpha,b]
                    H_k[alpha,beta]
                    A_psi[k][c,j,beta,d].

        Hence the Gram's TT/MPO ranks are the products of the phi and psi
        ranks. With equal rank r, its internal MPO rank is at most r**2.
        """

        # (primitive_batch, d, p, p)
        log_H = self.phi_primitive.log_Omega2_dim(
            self.psi_primitive,
            lows=lows,
            highs=highs,
        )

        expected_tail = (
            self._dim,
            self._p,
            self._p,
        )

        if log_H.ndim != 4 or tuple(log_H.shape[1:]) != expected_tail:
            raise ValueError(
                "phi_primitive.log_Omega2_dim(psi_primitive) returned "
                "an unexpected shape: expected "
                f"(batch, {self._dim}, {self._p}, {self._p}), "
                f"got {tuple(log_H.shape)}"
            )

        # The TTMatrix implementation written previously represents one
        # operator, not a leading batch of operators.
        if log_H.shape[0] != 1:
            raise NotImplementedError(
                "TTMutualBasis.Omega2 currently requires primitive Gram "
                "batch size 1 because TTMatrix is currently unbatched. "
                f"Got batch size {log_H.shape[0]}. "
                "eval() still supports ordinary data batches."
            )

        # (d, p, p)
        H = torch.exp(log_H[0])

        gram_cores: list[torch.Tensor] = []

        for k in range(self._dim):
            A = self._phi_core(k)
            B = self._psi_core(k)
            Hk = H[k]

            # Shapes:
            #
            #   A  : (ra0, i, alpha, ra1)
            #   Hk : (alpha, beta)
            #   B  : (rb0, j, beta, rb1)
            #
            # First contract the primitive phi index against H.
            #
            #   AH : (ra0, i, beta, ra1)
            AH = torch.einsum(
                "aiub,uv->aivb",
                A,
                Hk,
            )

            # Contract beta with the psi core.
            #
            # Result:
            #
            #   W6 :
            #       (ra0, rb0, i, j, ra1, rb1)
            W6 = torch.einsum(
                "aivb,cjvd->acijbd",
                AH,
                B,
            )

            ra0, _, _, ra1 = A.shape
            rb0, _, _, rb1 = B.shape

            # Pair phi/psi hidden states:
            #
            #   (ra0, rb0) -> Gram left rank
            #   (ra1, rb1) -> Gram right rank
            #
            # torchTT MPO core convention:
            #
            #   (R_left, row_mode, col_mode, R_right)
            W = W6.reshape(
                ra0 * rb0,
                self._p,
                self._p,
                ra1 * rb1,
            )

            gram_cores.append(W)

        return TTMatrix.from_cores(gram_cores)