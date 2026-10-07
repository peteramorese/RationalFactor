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
    r"""Full tensor-product basis with optional TT-structured coefficients.

    Given a :class:`SeparableBasis` providing per-dimension primitives

        phi^{(l)}_i(x_l),

    this class represents the full tensor-product basis

        Phi_{i_1,...,i_d}(x)
            = prod_l phi^{(l)}_{i_l}(x_l),

    optionally scaled by a TT / NestedTT coefficient tensor

        A_{i_1,...,i_d}.

    Hence evaluation produces

        v_{i_1,...,i_d}(x)
            = A_{i_1,...,i_d}
              prod_l phi^{(l)}_{i_l}(x_l),

    represented as a TTVector or NestedTTVector.

    If ``coeffs`` is ``None``, ``A`` is the all-ones rank-1 TT (no trainable
    coefficients).

    Notes
    -----
    ``forward(y)`` returns a (possibly batched) structured vector. ``Omega2``
    still requires the primitive Gram matrices to have batch size 1 because
    structured matrices have no batch axis.
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
        assert primitives.batch_size() == 1, "TTBasis currently only supports batch size 1"

        n_local = primitives.n_basis_functions()
        dim = primitives.dim()
        expected_modes = (n_local,) * dim

        self._primitives = primitives
        self._modes = expected_modes
        self._nested_depth = None if nested_depth is None else int(nested_depth)

        n = primitives.n_basis_functions() ** dim

        if coeffs is None:
            coeffs = self._default_ones_coeffs()
        else:
            self._validate_coeffs(coeffs, expected_modes)

        super().__init__(
            dim=dim,
            batch_size=1,
            n_basis=n,
            params=primitives.params(),
            coeffs=coeffs,
        )

    def _default_ones_coeffs(self):
        dtype, device = self._primitives.dtype_device()
        if self._nested_depth is not None:
            ones = ones_nested_tt_vector(
                self._modes,
                depth=self._nested_depth,
                dtype=dtype,
                device=device,
            )
            return FixedNestedTTVectorParameters(ones)
        cores = [
            FixedParameters(torch.ones(1, n, 1, dtype=dtype, device=device))
            for n in self._modes
        ]
        return TTVectorParameters.from_cores(cores)

    def _validate_coeffs(self, coeffs, expected_modes):
        if isinstance(coeffs, NestedTTVectorParameters):
            if coeffs.modes != expected_modes:
                raise ValueError(
                    "NestedTT coefficient modes must match the primitive basis "
                    f"size in every dimension. Expected {expected_modes}, "
                    f"got {coeffs.modes}"
                )
            if self._nested_depth is None:
                self._nested_depth = coeffs.depth
            elif coeffs.depth != self._nested_depth:
                raise ValueError(
                    f"nested_depth={self._nested_depth} does not match "
                    f"coeffs.depth={coeffs.depth}"
                )
        elif isinstance(coeffs, FixedNestedTTVectorParameters):
            vec = coeffs()
            if vec.modes != expected_modes:
                raise ValueError(
                    "NestedTT coefficient modes must match the primitive "
                    f"basis size. Expected {expected_modes}, got {vec.modes}"
                )
            if self._nested_depth is None:
                self._nested_depth = vec.depth
        elif isinstance(coeffs, TTVectorParameters):
            if coeffs.modes != expected_modes:
                raise ValueError(
                    "TT coefficient modes must match the primitive basis size "
                    "in every dimension. Expected "
                    f"{expected_modes}, got {coeffs.modes}"
                )
        else:
            raise TypeError(
                "coeffs must be TTVectorParameters, NestedTTVectorParameters, "
                f"FixedNestedTTVectorParameters, or None; got "
                f"{type(coeffs).__name__}"
            )

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
        coeffs = self.coeffs
        if isinstance(coeffs, NestedTTVectorParameters):
            return coeffs().ranks
        if isinstance(coeffs, FixedNestedTTVectorParameters):
            return coeffs().ranks
        return self.coeffs.ranks

    def n_primitives_per_dim(self) -> int:
        return self._primitives.n_basis_functions()

    def dtype_device(self):
        return self._primitives.dtype_device()

    def set_coeffs(
        self,
        coeffs: TTVectorParameters
        | NestedTTVectorParameters
        | FixedNestedTTVectorParameters,
    ):
        self._validate_coeffs(coeffs, self._modes)
        if hasattr(self, "_n_basis"):
            vals = coeffs()
            n = int(vals.shape[0]) if hasattr(vals, "shape") else int(vals.n)
            if n != self._n_basis:
                raise ValueError(
                    f"coeffs.n must equal n_basis={self._n_basis}, got {n}"
                )
        self.coeffs = coeffs

    def set_coeffs_to_one(self):
        self.set_coeffs(self._default_ones_coeffs())

    def _primitive_factors(self, y: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Per-dimension primitive values, shape ``(batch, n)`` each."""
        values = self._primitives.eval_dim(y)
        if values.dim() != 3:
            raise ValueError(
                "primitives.eval_dim(y) must return shape "
                "(batch, dim, n_basis), got "
                f"{tuple(values.shape)}"
            )
        batch, dim, _n_local = values.shape
        if dim != self._dim:
            raise ValueError(f"Expected input dimension {self._dim}, got {dim}")
        if batch == 1:
            return tuple(values[0, ell, :] for ell in range(dim))
        return tuple(values[:, ell, :] for ell in range(dim))

    def forward(self, y: torch.Tensor) -> TTVector | NestedTTVector:
        r"""Evaluate the TT-weighted tensor-product basis at a batch of points.

        Supports arbitrary leading batch size for both TT and NestedTT
        coefficients.  NestedTT results carry the batch on physical-scale
        factors (see :class:`NestedTTVector`).
        """
        coeff_vals = self.coeffs()

        if isinstance(coeff_vals, NestedTTVector):
            return coeff_vals.elementwise_multiply(self._primitive_factors(y))

        # Standard TT path: scale each core's physical mode by batched primitives.
        values = self._primitives.eval_dim(y)
        batch, dim, n_local = values.shape
        if dim != self._dim:
            raise ValueError(f"Expected input dimension {self._dim}, got {dim}")
        coeff_cores = coeff_vals.cores
        if len(coeff_cores) != dim:
            raise ValueError(
                "Number of TT cores must equal the basis dimension: "
                f"{len(coeff_cores)} != {dim}"
            )

        out_cores = []
        for ell, core in enumerate(coeff_cores):
            if core.shape[-2] != n_local:
                raise ValueError(
                    f"TT mode mismatch at dimension {ell}: "
                    f"core mode is {core.shape[-2]}, "
                    f"primitive basis has {n_local} functions"
                )
            local_values = values[:, ell, :][:, None, :, None]
            out_cores.append(core * local_values)
        tol = float(getattr(coeff_vals, "numerical_tolerance", 1e-20))
        return TTVector(out_cores, numerical_tolerance=tol)

    def __call__(
        self,
        y: torch.Tensor,
    ) -> TTVector | NestedTTVector:
        return self.forward(y)

    def Omega1(
        self,
        lows: torch.Tensor = None,
        highs: torch.Tensor = None,
    ) -> TTVector | NestedTTVector | tuple:
        r"""Integral of each weighted tensor-product basis function."""
        log_omega = self._primitives.log_Omega1_dim(lows, highs)

        if log_omega.dim() != 3:
            raise ValueError(
                "log_Omega1_dim must return shape "
                "(batch, dim, n_basis), got "
                f"{tuple(log_omega.shape)}"
            )

        omega = torch.exp(log_omega)
        coeff_vals = self.coeffs()

        if isinstance(coeff_vals, NestedTTVector):
            if omega.shape[0] != 1:
                raise ValueError(
                    "NestedTT Omega1 currently requires batch size 1, "
                    f"got {omega.shape[0]}"
                )
            factors = tuple(omega[0, ell] for ell in range(self._dim))
            return coeff_vals.elementwise_multiply(factors)

        coeff_cores = coeff_vals.cores
        outputs = []
        for b in range(omega.shape[0]):
            cores = []
            for ell, core in enumerate(coeff_cores):
                factor = omega[b, ell].reshape(1, -1, 1)
                cores.append(core * factor)
            outputs.append(TTVector(cores))

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def Omega2(
        self,
        other: "TTBasis",
        lows: torch.Tensor = None,
        highs: torch.Tensor = None,
    ) -> TTMatrix | NestedTTMatrix:
        r"""Return the Gram matrix as a TTMatrix / NestedTTMatrix / MPO."""
        if not isinstance(other, TTBasis):
            raise TypeError(
                "TTBasis.Omega2 currently requires another TTBasis, got "
                f"{type(other).__name__}"
            )

        if self.dim() != other.dim():
            raise ValueError(
                "TTBasis dimensions must match, got "
                f"{self.dim()} and {other.dim()}"
            )

        log_gram = self._primitives.log_Omega2_dim(
            other._primitives,
            lows,
            highs,
        )

        if log_gram.dim() != 4:
            raise ValueError(
                "log_Omega2_dim must return shape "
                "(batch, dim, n_self, n_other), got "
                f"{tuple(log_gram.shape)}"
            )

        if log_gram.shape[0] != 1:
            raise ValueError(
                "Structured Omega2 requires primitive Gram batch size 1. "
                f"Got batch size {log_gram.shape[0]}."
            )

        if log_gram.shape[1] != self._dim:
            raise ValueError(
                f"Expected {self._dim} Gram dimensions, got "
                f"{log_gram.shape[1]}"
            )

        # (dim, n_self, n_other)
        gram = torch.exp(log_gram[0])
        
        assert self.coeffs() is None, "TTBasis.Omega2 currently does not support trainable coefficients"

        self_vals = self.coeffs()
        other_vals = other.coeffs()
        use_nested = isinstance(self_vals, NestedTTVector) or isinstance(
            other_vals, NestedTTVector
        )

        if use_nested:
            depth = self._nested_depth or other._nested_depth
            if depth is None:
                if isinstance(self_vals, NestedTTVector):
                    depth = self_vals.depth
                else:
                    depth = other_vals.depth

            # Weighted separable cores: a_k[:,None] * G_k * b_k[None,:]
            # for rank-one coefficient tensors; otherwise densify factors.
            if isinstance(self_vals, NestedTTVector):
                from rational_factor.models.tt.nested_tt import (
                    rank_one_factors_from_nested,
                )
                a_factors = rank_one_factors_from_nested(self_vals)
            else:
                a_factors = tuple(
                    c.reshape(-1) for c in self_vals.cores
                )
            if isinstance(other_vals, NestedTTVector):
                from rational_factor.models.tt.nested_tt import (
                    rank_one_factors_from_nested,
                )
                b_factors = rank_one_factors_from_nested(other_vals)
            else:
                b_factors = tuple(
                    c.reshape(-1) for c in other_vals.cores
                )

            weighted = []
            for ell in range(self._dim):
                G = gram[ell]
                weighted.append(
                    a_factors[ell][:, None] * G * b_factors[ell][None, :]
                )
            return nested_tt_matrix_from_separable_cores(weighted, depth=depth)

        self_cores = self_vals.cores
        other_cores = other_vals.cores
        mpo_cores = []

        for ell, (a_core, b_core) in enumerate(zip(self_cores, other_cores)):
            G = gram[ell]

            if G.shape != (a_core.shape[1], b_core.shape[1]):
                raise ValueError(
                    f"Gram mode mismatch at dimension {ell}: "
                    f"Gram has shape {tuple(G.shape)}, "
                    f"coefficient modes are "
                    f"({a_core.shape[1]}, {b_core.shape[1]})"
                )

            core = torch.einsum(
                "aic,ij,bjd->abijcd",
                a_core,
                G,
                b_core,
            )

            core = core.reshape(
                a_core.shape[0] * b_core.shape[0],
                a_core.shape[1],
                b_core.shape[1],
                a_core.shape[2] * b_core.shape[2],
            )

            mpo_cores.append(core)

        return TTMatrix(mpo_cores)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dim={self._dim}, "
            f"modes={self.modes}, "
            f"n_basis={self._n_basis}, "
            f"ranks={self.ranks}, "
            f"nested_depth={self._nested_depth}, "
            f"dtype={self.dtype_device()[0]}, "
            f"device={self.dtype_device()[1]})"
        )
