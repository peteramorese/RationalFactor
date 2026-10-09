from __future__ import annotations

import torch

from rational_factor.models.basis_functions import Basis, SeparableBasis
from rational_factor.models.parameters import FixedParameters, TTVectorParameters
from rational_factor.models.structured_matrices import TTMatrix
from rational_factor.models.structured_vectors import TTVector
from rational_factor.models.tt.nested_tt import (
    NestedTTMatrix,
    NestedTTVector,
    nested_tt_matrix_from_separable_cores,
    ones_nested_tt_vector,
)
from rational_factor.models.tt.nested_tt_parameters import (
    FixedNestedTTVectorParameters,
    NestedTTVectorParameters,
)


class TTBasis(Basis):
    r"""Tensor-product basis with optional TT/NestedTT coefficients.

    For per-dimension primitives ``phi_k[i](x_k)``, ``forward`` returns the
    structured tensor

        A[i_1,...,i_d] * prod_k phi_k[i_k](x_k).

    When no coefficient tensor is supplied, ``A`` is the all-ones rank-one
    tensor.  In that common case ``Omega2`` is simply the Kronecker product of
    the one-dimensional Gram matrices, represented as a rank-one TT/MPO (or
    NestedTTMatrix), never as a dense matrix.
    """

    def __init__(
        self,
        primitives: SeparableBasis,
        coeffs: TTVectorParameters | NestedTTVectorParameters | None = None,
        *,
        nested_depth: int | None = None,
    ):
        if not isinstance(primitives, SeparableBasis):
            raise TypeError(
                "primitives must be a SeparableBasis, got "
                f"{type(primitives).__name__}"
            )
        if primitives.batch_size() != 1:
            raise ValueError("TTBasis primitives must have parameter batch size 1")

        self._primitives = primitives
        self._modes = (primitives.n_basis_functions(),) * primitives.dim()
        self._nested_depth = None if nested_depth is None else int(nested_depth)
        default_coeffs = coeffs is None

        if coeffs is None:
            coeffs = self._default_ones_coeffs()
        else:
            self._validate_coeffs(coeffs)

        # Basis.__init__ routes through set_coeffs, which clears _default_coeffs.
        # Restore the flag so coefficient-free Omega2 stays available.
        self._default_coeffs = default_coeffs
        super().__init__(
            dim=primitives.dim(),
            batch_size=1,
            n_basis=primitives.n_basis_functions() ** primitives.dim(),
            params=primitives.params(),
            coeffs=coeffs,
        )
        self._default_coeffs = default_coeffs

    # ------------------------------------------------------------------
    # Basic metadata
    # ------------------------------------------------------------------

    @property
    def primitives(self) -> SeparableBasis:
        return self._primitives

    @property
    def modes(self) -> tuple[int, ...]:
        return self._modes

    @property
    def nested_depth(self) -> int | None:
        return self._nested_depth

    @property
    def ranks(self) -> tuple[int, ...]:
        return self.coeffs().ranks

    def n_primitives_per_dim(self) -> int:
        return self._primitives.n_basis_functions()

    def dtype_device(self):
        return self._primitives.dtype_device()

    # ------------------------------------------------------------------
    # Coefficients
    # ------------------------------------------------------------------

    def _default_ones_coeffs(self):
        dtype, device = self.dtype_device()
        if self._nested_depth is not None:
            return FixedNestedTTVectorParameters(
                ones_nested_tt_vector(
                    self._modes,
                    depth=self._nested_depth,
                    dtype=dtype,
                    device=device,
                )
            )
        return TTVectorParameters.from_cores(
            [
                FixedParameters(torch.ones(1, n, 1, dtype=dtype, device=device))
                for n in self._modes
            ]
        )

    def _validate_coeffs(self, coeffs) -> None:
        if isinstance(coeffs, NestedTTVectorParameters):
            modes = coeffs.modes
            depth = coeffs.spec.depth
        elif isinstance(coeffs, FixedNestedTTVectorParameters):
            modes = coeffs.modes
            depth = coeffs.depth
        elif isinstance(coeffs, TTVectorParameters):
            modes = coeffs.modes
            depth = None
        else:
            raise TypeError(
                "coeffs must be TTVectorParameters, NestedTTVectorParameters, "
                f"FixedNestedTTVectorParameters, or None; got {type(coeffs).__name__}"
            )

        if tuple(modes) != self._modes:
            raise ValueError(
                f"coefficient modes must equal {self._modes}, got {tuple(modes)}"
            )

        if depth is not None:
            if self._nested_depth is None:
                self._nested_depth = int(depth)
            elif int(depth) != self._nested_depth:
                raise ValueError(
                    f"nested_depth={self._nested_depth} does not match coeff depth={depth}"
                )

    def set_coeffs(
        self,
        coeffs: TTVectorParameters
        | NestedTTVectorParameters
        | FixedNestedTTVectorParameters,
    ) -> None:
        self._validate_coeffs(coeffs)
        self.coeffs = coeffs
        self._default_coeffs = False

    def set_coeffs_to_one(self) -> None:
        self.coeffs = self._default_ones_coeffs()
        self._default_coeffs = True

    # ------------------------------------------------------------------
    # Physical rank-one factors
    # ------------------------------------------------------------------

    def _primitive_factors(self, y: torch.Tensor) -> tuple[torch.Tensor, ...]:
        values = self._primitives.eval_dim(y)
        if values.ndim != 3 or int(values.shape[1]) != self._dim:
            raise ValueError(
                "primitives.eval_dim(y) must have shape (batch, dim, n_basis); "
                f"got {tuple(values.shape)}"
            )
        if int(values.shape[2]) != self._modes[0]:
            raise ValueError("primitive basis size does not match TTBasis modes")

        if int(values.shape[0]) == 1:
            return tuple(values[0, k] for k in range(self._dim))
        return tuple(values[:, k] for k in range(self._dim))

    @staticmethod
    def _scale_tt(coeffs: TTVector, factors: tuple[torch.Tensor, ...]) -> TTVector:
        cores = []
        for core, factor in zip(coeffs.cores, factors):
            if factor.ndim == 1:
                cores.append(core * factor[None, :, None])
            else:
                cores.append(core * factor[:, None, :, None])
        tol = float(getattr(coeffs, "numerical_tolerance", 1e-20))
        return TTVector(cores, numerical_tolerance=tol)

    # ------------------------------------------------------------------
    # Evaluation / integrals
    # ------------------------------------------------------------------

    def forward(self, y: torch.Tensor) -> TTVector | NestedTTVector:
        factors = self._primitive_factors(y)
        coeffs = self.coeffs()
        if isinstance(coeffs, NestedTTVector):
            return coeffs.elementwise_multiply(factors)
        return self._scale_tt(coeffs, factors)

    __call__ = forward

    def Omega1(
        self,
        lows: torch.Tensor = None,
        highs: torch.Tensor = None,
    ) -> TTVector | NestedTTVector:
        log_omega = self._primitives.log_Omega1_dim(lows, highs)
        if log_omega.ndim != 3 or int(log_omega.shape[1]) != self._dim:
            raise ValueError(
                "log_Omega1_dim must have shape (batch, dim, n_basis); "
                f"got {tuple(log_omega.shape)}"
            )
        omega = torch.exp(log_omega)
        factors = (
            tuple(omega[0, k] for k in range(self._dim))
            if int(omega.shape[0]) == 1
            else tuple(omega[:, k] for k in range(self._dim))
        )
        coeffs = self.coeffs()
        if isinstance(coeffs, NestedTTVector):
            return coeffs.elementwise_multiply(factors)
        return self._scale_tt(coeffs, factors)

    def Omega2(
        self,
        other: "TTBasis",
        lows: torch.Tensor = None,
        highs: torch.Tensor = None,
    ) -> TTMatrix | NestedTTMatrix:
        """Return the tensor-product Gram matrix as a rank-one MPO.

        General coefficient-weighted Gram matrices are intentionally not
        handled here: recovering separable factors from an arbitrary NestedTT
        coefficient tensor would defeat the no-materialization invariant.
        """
        if not isinstance(other, TTBasis):
            raise TypeError("TTBasis.Omega2 requires another TTBasis")
        if self.dim() != other.dim():
            raise ValueError("TTBasis dimensions must match")
        if not self._default_coeffs or not other._default_coeffs:
            raise NotImplementedError(
                "Omega2 with nontrivial coefficient tensors is not implemented; "
                "the coefficient-free Gram matrix is represented exactly as a rank-one MPO"
            )

        log_gram = self._primitives.log_Omega2_dim(
            other._primitives, lows, highs
        )
        if log_gram.ndim != 4 or int(log_gram.shape[1]) != self._dim:
            raise ValueError(
                "log_Omega2_dim must have shape "
                "(batch, dim, n_self, n_other)"
            )
        if int(log_gram.shape[0]) != 1:
            raise ValueError("structured Omega2 currently requires Gram batch size 1")

        gram = torch.exp(log_gram[0])
        factors = tuple(gram[k] for k in range(self._dim))

        depths = {d for d in (self._nested_depth, other._nested_depth) if d is not None}
        if len(depths) > 1:
            raise ValueError("nested TTBasis objects must use the same nested_depth")
        if depths:
            return nested_tt_matrix_from_separable_cores(
                factors, depth=depths.pop()
            )

        return TTMatrix([G[None, :, :, None] for G in factors])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(dim={self._dim}, modes={self.modes}, "
            f"n_basis={self._n_basis}, ranks={self.ranks}, "
            f"nested_depth={self._nested_depth}, dtype={self.dtype_device()[0]}, "
            f"device={self.dtype_device()[1]})"
        )
