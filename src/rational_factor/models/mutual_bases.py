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
    r"""TT mutual basis with independently tensorized output index.

    There are:

        d = number of input dimensions
        q = number of output-index TT cores
        p_s = size of output mode s

    so the total number of basis functions is

        m = prod_s p_s,

    independently of d.

    The TT chain contains two kinds of cores.

    Input/dimension core:
        A_k[alpha_k] : (r_left, r_right)

    Output core:
        O_s[i_s] : (r_left, r_right)

    For example,

        M_1(x_1) ... M_l(x_l)
        O_1[i_1]
        M_{l+1}(x_{l+1}) ...
        O_2[i_2]
        ...

    where

        M_k(x_k)
          = sum_alpha A_k[alpha] primitive_alpha(x_k).

    All raw cores are passed through softplus. Therefore, assuming the
    primitive 1-D bases are nonnegative, the resulting phi/psi functions
    are nonnegative.

    ``output_positions[s]`` is the number of input dimensions appearing
    BEFORE output core s.

    Example with d=12 and output_positions=(3, 6, 9):

        dims 0:3
        O_0
        dims 3:6
        O_1
        dims 6:9
        O_2
        dims 9:12

    If every output mode has size p=2, this gives m=2**3=8.
    """

    def __init__(
        self,
        phi_primitive: SeparableBasis,
        psi_primitive: SeparableBasis,
        *,
        rank: int | Sequence[int] = 4,
        n_output_modes: int = 1,
        output_mode_size: int = 2,
        output_mode_sizes: Sequence[int] | None = None,
        output_positions: Sequence[int] | None = None,
        init_std: float = 0.05,
    ):
        torch.nn.Module.__init__(self)

        # --------------------------------------------------------------
        # Primitive bases
        # --------------------------------------------------------------

        if phi_primitive.dim() != psi_primitive.dim():
            raise ValueError(
                "phi_primitive and psi_primitive must have the same dimension"
            )

        if phi_primitive.batch_size() != psi_primitive.batch_size():
            raise ValueError(
                "phi_primitive and psi_primitive must have the same batch size"
            )

        d = phi_primitive.dim()

        if d < 1:
            raise ValueError("TTMutualBasis requires d >= 1")

        phi_dtype, phi_device = phi_primitive.dtype_device()
        psi_dtype, psi_device = psi_primitive.dtype_device()

        if phi_dtype != psi_dtype:
            raise ValueError("phi/psi primitive bases must have the same dtype")

        if phi_device != psi_device:
            raise ValueError("phi/psi primitive bases must be on the same device")

        self.phi_primitive = phi_primitive
        self.psi_primitive = psi_primitive

        self._n_phi_primitive = phi_primitive.n_basis_functions()
        self._n_psi_primitive = psi_primitive.n_basis_functions()

        # --------------------------------------------------------------
        # Output tensorization
        # --------------------------------------------------------------

        if output_mode_sizes is None:
            if n_output_modes < 1:
                raise ValueError("n_output_modes must be >= 1")

            output_mode_sizes = (
                int(output_mode_size),
            ) * int(n_output_modes)
        else:
            output_mode_sizes = tuple(int(p) for p in output_mode_sizes)
            n_output_modes = len(output_mode_sizes)

        if any(p < 1 for p in output_mode_sizes):
            raise ValueError(
                f"all output mode sizes must be >= 1, got {output_mode_sizes}"
            )

        q = n_output_modes
        self._output_mode_sizes = tuple(output_mode_sizes)
        self._n_output_modes = q

        # --------------------------------------------------------------
        # Choose where output cores are placed.
        #
        # Position s means: insert O_s after `position` input dimensions.
        #
        # Default spreads them approximately evenly.
        # --------------------------------------------------------------

        if output_positions is None:
            if q > d:
                raise ValueError(
                    "default output placement currently requires "
                    f"n_output_modes <= d, got q={q}, d={d}"
                )

            # Examples:
            #
            # d=12, q=1 -> (6,)
            # d=12, q=3 -> (3, 6, 9)
            # d=12, q=12 -> (1, ..., 12)
            output_positions = tuple(
                ((s + 1) * (d + 1)) // (q + 1)
                for s in range(q)
            )
        else:
            output_positions = tuple(int(v) for v in output_positions)

            if len(output_positions) != q:
                raise ValueError(
                    f"expected {q} output positions, "
                    f"got {len(output_positions)}"
                )

        if any(pos < 0 or pos > d for pos in output_positions):
            raise ValueError(
                f"output positions must lie in [0, {d}], "
                f"got {output_positions}"
            )

        if any(
            output_positions[s] >= output_positions[s + 1]
            for s in range(q - 1)
        ):
            raise ValueError(
                "output_positions must be strictly increasing"
            )

        self._output_positions = output_positions

        # --------------------------------------------------------------
        # Build the actual TT chain.
        #
        # Example:
        #
        #   ("dim", 0)
        #   ("dim", 1)
        #   ("out", 0)
        #   ("dim", 2)
        #   ...
        # --------------------------------------------------------------

        pos_to_output = {
            pos: s for s, pos in enumerate(output_positions)
        }

        chain: list[tuple[str, int]] = []

        for pos in range(d + 1):
            if pos in pos_to_output:
                chain.append(("out", pos_to_output[pos]))

            if pos < d:
                chain.append(("dim", pos))

        self._chain = tuple(chain)

        n_chain_cores = len(chain)

        self._chain_ranks = self._normalize_ranks(
            n_chain_cores,
            rank,
        )

        # Work out left/right rank for every input/output core.
        dim_shapes = [None] * d
        out_shapes = [None] * q

        for t, (kind, idx) in enumerate(chain):
            r_left = self._chain_ranks[t]
            r_right = self._chain_ranks[t + 1]

            if kind == "dim":
                dim_shapes[idx] = (r_left, r_right)
            else:
                out_shapes[idx] = (r_left, r_right)

        # --------------------------------------------------------------
        # Parameter cores
        # --------------------------------------------------------------

        self.phi_dim_raw_cores = torch.nn.ParameterList()
        self.psi_dim_raw_cores = torch.nn.ParameterList()

        for k in range(d):
            r_left, r_right = dim_shapes[k]

            self.phi_dim_raw_cores.append(
                torch.nn.Parameter(
                    self._init_raw(
                        (
                            r_left,
                            self._n_phi_primitive,
                            r_right,
                        ),
                        fan=self._n_phi_primitive * r_right,
                        dtype=phi_dtype,
                        device=phi_device,
                        std=init_std,
                    )
                )
            )

            self.psi_dim_raw_cores.append(
                torch.nn.Parameter(
                    self._init_raw(
                        (
                            r_left,
                            self._n_psi_primitive,
                            r_right,
                        ),
                        fan=self._n_psi_primitive * r_right,
                        dtype=psi_dtype,
                        device=psi_device,
                        std=init_std,
                    )
                )
            )

        self.phi_output_raw_cores = torch.nn.ParameterList()
        self.psi_output_raw_cores = torch.nn.ParameterList()

        for s in range(q):
            r_left, r_right = out_shapes[s]
            p = self._output_mode_sizes[s]

            self.phi_output_raw_cores.append(
                torch.nn.Parameter(
                    self._init_raw(
                        (r_left, p, r_right),
                        fan=p * r_right,
                        dtype=phi_dtype,
                        device=phi_device,
                        std=init_std,
                    )
                )
            )

            self.psi_output_raw_cores.append(
                torch.nn.Parameter(
                    self._init_raw(
                        (r_left, p, r_right),
                        fan=p * r_right,
                        dtype=psi_dtype,
                        device=psi_device,
                        std=init_std,
                    )
                )
            )

        m = math.prod(self._output_mode_sizes)

        MutualPairBasis.__init__(
            self,
            dim=d,
            batch_size=phi_primitive.batch_size(),
            n_basis=m,
            params=(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_ranks(
        n_cores: int,
        rank: int | Sequence[int],
    ) -> tuple[int, ...]:

        if isinstance(rank, int):
            if rank < 1:
                raise ValueError("rank must be >= 1")

            return (
                (1,)
                + (rank,) * max(n_cores - 1, 0)
                + (1,)
            )

        ranks = tuple(int(r) for r in rank)

        if len(ranks) == n_cores - 1:
            ranks = (1,) + ranks + (1,)

        if len(ranks) != n_cores + 1:
            raise ValueError(
                f"rank sequence must have length {n_cores - 1} "
                f"or {n_cores + 1}, got {len(ranks)}"
            )

        if ranks[0] != 1 or ranks[-1] != 1:
            raise ValueError("boundary TT ranks must equal 1")

        if any(r < 1 for r in ranks):
            raise ValueError("all TT ranks must be >= 1")

        return ranks

    @staticmethod
    def _init_raw(
        shape,
        *,
        fan: int,
        dtype,
        device,
        std: float,
    ):
        target = 1.0 / max(fan, 1)

        target = torch.tensor(
            target,
            dtype=dtype,
            device=device,
        )

        mean = torch.log(torch.expm1(target))

        return mean + std * torch.randn(
            shape,
            dtype=dtype,
            device=device,
        )

    def _phi_dim_core(self, k):
        return F.softplus(self.phi_dim_raw_cores[k])

    def _psi_dim_core(self, k):
        return F.softplus(self.psi_dim_raw_cores[k])

    def _phi_output_core(self, s):
        return F.softplus(self.phi_output_raw_cores[s])

    def _psi_output_core(self, s):
        return F.softplus(self.psi_output_raw_cores[s])

    @property
    def n_output_modes(self):
        return self._n_output_modes

    @property
    def output_mode_sizes(self):
        return self._output_mode_sizes

    @property
    def output_positions(self):
        return self._output_positions

    @property
    def tt_ranks(self):
        return self._chain_ranks

    def dtype_device(self):
        p = self.phi_dim_raw_cores[0]
        return p.dtype, p.device

    # ------------------------------------------------------------------
    # Dense evaluation
    # ------------------------------------------------------------------

    def _eval_side(
        self,
        y: torch.Tensor,
        primitive: SeparableBasis,
        *,
        side: int,
    ):
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

        primitive_values = primitive.eval_dim(y)

        n_primitive = (
            self._n_phi_primitive
            if side == 0
            else self._n_psi_primitive
        )

        expected = (
            y.shape[0],
            self._dim,
            n_primitive,
        )

        if tuple(primitive_values.shape) != expected:
            raise ValueError(
                f"primitive.eval_dim expected {expected}, "
                f"got {tuple(primitive_values.shape)}"
            )

        batch = y.shape[0]

        # Shape:
        #
        #   (batch, existing output modes..., current TT rank)
        out = torch.ones(
            batch,
            1,
            dtype=dtype,
            device=device,
        )

        for kind, idx in self._chain:

            if kind == "dim":
                core = (
                    self._phi_dim_core(idx)
                    if side == 0
                    else self._psi_dim_core(idx)
                )

                # primitive_values:
                #   (batch, alpha)
                #
                # core:
                #   (r_left, alpha, r_right)
                #
                # local:
                #   (batch, r_left, r_right)
                local = torch.einsum(
                    "nu,aur->nar",
                    primitive_values[:, idx, :],
                    core,
                )

                out = torch.einsum(
                    "n...a,nar->n...r",
                    out,
                    local,
                )

            else:
                core = (
                    self._phi_output_core(idx)
                    if side == 0
                    else self._psi_output_core(idx)
                )

                # core:
                #   (r_left, i_s, r_right)
                #
                # Append a new output/function index.
                out = torch.einsum(
                    "n...a,air->n...ir",
                    out,
                    core,
                )

        out = out.squeeze(-1)

        return out.reshape(
            batch,
            self._n_basis,
        )

    def eval(
        self,
        y: torch.Tensor | None = None,
        index: int | None = None,
    ):
        if y is None:
            return torch.nn.Module.eval(self)

        if index not in (0, 1, None):
            raise ValueError("index must be 0, 1, or None")

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

        return torch.stack(
            (phi, psi),
            dim=1,
        )

    # ------------------------------------------------------------------
    # Gram construction
    # ------------------------------------------------------------------

    def Omega2(
        self,
        lows: torch.Tensor = None,
        highs: torch.Tensor = None,
    ) -> Matrix:

        # --------------------------------------------------------------
        # Primitive 1-D cross Grams:
        #
        # H:
        #   (batch, d, n_phi, n_psi)
        # --------------------------------------------------------------

        log_H = self.phi_primitive.log_Omega2_dim(
            self.psi_primitive,
            lows=lows,
            highs=highs,
        )

        expected_tail = (
            self._dim,
            self._n_phi_primitive,
            self._n_psi_primitive,
        )

        if (
            log_H.ndim != 4
            or tuple(log_H.shape[1:]) != expected_tail
        ):
            raise ValueError(
                "unexpected primitive Gram shape: "
                f"expected (batch, {expected_tail}), "
                f"got {tuple(log_H.shape)}"
            )

        # Current TTMatrix stores one MPO, not a batch of MPOs.
        if log_H.shape[0] != 1:
            raise NotImplementedError(
                "TTMutualBasis.Omega2 currently requires primitive "
                "Gram batch size 1 because TTMatrix is unbatched."
            )

        H = torch.exp(log_H[0])

        # --------------------------------------------------------------
        # We remove all dimension-only nodes from the final MPO by
        # contracting them into the q output MPO cores.
        #
        # pending has shape
        #
        #   (right rank of previous output MPO core,
        #    current paired TT rank)
        # --------------------------------------------------------------

        dtype, device = self.dtype_device()

        pending = torch.ones(
            1,
            1,
            dtype=dtype,
            device=device,
        )

        mpo_cores: list[torch.Tensor] = []

        for kind, idx in self._chain:

            if kind == "dim":
                A = self._phi_dim_core(idx)
                B = self._psi_dim_core(idx)
                Hk = H[idx]

                # A: (ra0, alpha, ra1)
                # B: (rb0, beta,  rb1)
                #
                # T:
                #   (ra0*rb0, ra1*rb1)
                T4 = torch.einsum(
                    "aub,uv,cvd->acbd",
                    A,
                    Hk,
                    B,
                )

                T = T4.reshape(
                    A.shape[0] * B.shape[0],
                    A.shape[2] * B.shape[2],
                )

                pending = pending @ T

            else:
                Oa = self._phi_output_core(idx)
                Ob = self._psi_output_core(idx)

                # Oa: (ra0, i, ra1)
                # Ob: (rb0, j, rb1)
                #
                # Pair phi/psi hidden states.
                W6 = torch.einsum(
                    "aib,cjd->acijbd",
                    Oa,
                    Ob,
                )

                p = self._output_mode_sizes[idx]

                W = W6.reshape(
                    Oa.shape[0] * Ob.shape[0],
                    p,
                    p,
                    Oa.shape[2] * Ob.shape[2],
                )

                # Absorb all dimension-only contractions since the
                # previous output core.
                #
                # pending:
                #   (R_prev, W_left)
                #
                # W:
                #   (W_left, p, p, W_right)
                W = torch.einsum(
                    "la,aijb->lijb",
                    pending,
                    W,
                )

                mpo_cores.append(W)

                # Start a fresh transfer from this output site's right bond.
                R_right = W.shape[-1]

                pending = torch.eye(
                    R_right,
                    dtype=dtype,
                    device=device,
                )

        # Remaining input dimensions after the final output core are
        # absorbed into that core's right boundary.
        if not mpo_cores:
            raise RuntimeError(
                "TTMutualBasis requires at least one output core"
            )

        mpo_cores[-1] = torch.einsum(
            "aijb,bc->aijc",
            mpo_cores[-1],
            pending,
        )

        return TTMatrix.from_cores(mpo_cores)