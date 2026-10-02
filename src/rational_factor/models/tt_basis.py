from __future__ import annotations

import torch

from rational_factor.models.basis_functions import Basis, SeparableBasis
from rational_factor.models.parameters import FixedParameters, TTVectorParameters
from rational_factor.models.structured_matrices import TTMatrix
from rational_factor.models.structured_vectors import TTVector


class TTBasis(Basis):
    r"""Full tensor-product basis with optional TT-structured coefficients.

    Given a :class:`SeparableBasis` providing per-dimension primitives

        phi^{(l)}_i(x_l),

    this class represents the full tensor-product basis

        Phi_{i_1,...,i_d}(x)
            = prod_l phi^{(l)}_{i_l}(x_l),

    optionally scaled by a TT coefficient tensor

        A_{i_1,...,i_d}.

    Hence evaluation produces

        v_{i_1,...,i_d}(x)
            = A_{i_1,...,i_d}
              prod_l phi^{(l)}_{i_l}(x_l),

    represented as a TTVector.

    If ``coeffs`` is ``None``, ``A`` is the all-ones rank-1 TT (no trainable
    coefficients).

    Notes
    -----
    ``forward(y)`` returns a (possibly batched) :class:`TTVector`. ``Omega2``
    still requires the primitive Gram matrices to have batch size 1 because
    ``TTMatrix`` has no batch axis.
    """

    def __init__(
        self,
        primitives: SeparableBasis,
        coeffs: TTVectorParameters | None = None,
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
        dtype, device = primitives.dtype_device()

        if coeffs is None:
            coeffs = self._ones_rank1_tt_coeffs()
        elif not isinstance(coeffs, TTVectorParameters):
            raise TypeError(
                "coeffs must be a TTVectorParameters or None, got "
                f"{type(coeffs).__name__}"
            )
        elif coeffs.modes != expected_modes:
            raise ValueError(
                "TT coefficient modes must match the primitive basis size "
                "in every dimension. Expected "
                f"{expected_modes}, got {coeffs.modes}"
            )

        self._primitives = primitives
        self._modes = expected_modes

        super().__init__(
            dim=dim,
            batch_size=1,
            n_basis=coeffs.n,
            params=primitives.params(),
            coeffs=coeffs,
        )

    @property
    def primitives(self) -> SeparableBasis:
        return self._primitives

    @property
    def modes(self) -> tuple[int, ...]:
        return self._modes

    @property
    def ranks(self) -> tuple[int, ...]:
        return self.coeffs.ranks

    def n_primitives_per_dim(self) -> int:
        return self._primitives.n_basis_functions()

    def dtype_device(self):
        return self._primitives.dtype_device()

    def set_coeffs(self, coeffs: TTVectorParameters):
        if not isinstance(coeffs, TTVectorParameters):
            raise TypeError(
                "TTBasis coeffs must be TTVectorParameters, got "
                f"{type(coeffs).__name__}"
            )
        if coeffs.modes != self._modes:
            raise ValueError(
                "TT coefficient modes must match existing modes "
                f"{self._modes}, got {coeffs.modes}"
            )
        if hasattr(self, "_n_basis") and coeffs.n != self._n_basis:
            raise ValueError(
                f"coeffs.n must equal n_basis={self._n_basis}, got {coeffs.n}"
            )
        self.coeffs = coeffs

    def set_coeffs_to_one(self):
        dtype, device = self.dtype_device()

        cores = [
            FixedParameters(torch.ones(1, n, 1, dtype=dtype, device=device))
            for n in self._modes
        ]
        ones = TTVectorParameters.from_cores(cores)
        self.set_coeffs(ones)

    def forward(self, y: torch.Tensor) -> TTVector:
        r"""Evaluate the TT-weighted tensor-product basis at a batch of points.
        The TT ranks are unchanged. The operation only scales the physical
        dimension of each TT core by the corresponding 1D primitive values.
        """
        values = self._primitives.eval_dim(y)

        if values.dim() != 3:
            raise ValueError(
                "primitives.eval_dim(y) must return shape "
                "(batch, dim, n_basis), got "
                f"{tuple(values.shape)}"
            )

        batch, dim, n_local = values.shape

        if dim != self._dim:
            raise ValueError(
                f"Expected input dimension {self._dim}, got {dim}"
            )

        coeff_cores = self.coeffs().cores
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

            local_values = values[:, ell, :]
            local_values = local_values[:, None, :, None]
            out_cores.append(core * local_values)

        return TTVector(out_cores)

    def __call__(
        self,
        y: torch.Tensor,
    ) -> TTVector:
        return self.forward(y)

    def Omega1(
        self,
        lows: torch.Tensor = None,
        highs: torch.Tensor = None,
    ) -> TTVector | tuple[TTVector, ...]:
        r"""Integral of each weighted tensor-product basis function.

        Returns the structured vector

            coeffs[i_1,...,i_d]
            * prod_l <phi^{(l)}_{i_l}, 1>.

        The TT ranks are unchanged.
        """
        log_omega = self._primitives.log_Omega1_dim(lows, highs)

        if log_omega.dim() != 3:
            raise ValueError(
                "log_Omega1_dim must return shape "
                "(batch, dim, n_basis), got "
                f"{tuple(log_omega.shape)}"
            )

        omega = torch.exp(log_omega)
        coeff_cores = self.coeffs().cores

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
    ) -> TTMatrix:
        r"""Return the Gram matrix as a TTMatrix / MPO.

        The matrix entries are

            Omega[i, j]
              = A[i] B[j]
                prod_l <phi^{(l)}_{i_l}, psi^{(l)}_{j_l}>,

        where ``A`` and ``B`` are the TT coefficient tensors of ``self`` and
        ``other``.

        If the coefficient TT ranks are ``r_l`` and ``s_l``, respectively,
        the resulting MPO has ranks

            r_l * s_l.

        The primitive tensor-product Gram itself has MPO rank 1.
        """
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
                "TTMatrix currently has no batch dimension, so "
                "TTBasis.Omega2 requires primitive Gram batch size 1. "
                f"Got batch size {log_gram.shape[0]}."
            )

        if log_gram.shape[1] != self._dim:
            raise ValueError(
                f"Expected {self._dim} Gram dimensions, got "
                f"{log_gram.shape[1]}"
            )

        # (dim, n_self, n_other)
        gram = torch.exp(log_gram[0])
        self_cores = self.coeffs().cores
        other_cores = other.coeffs().cores

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
            f"dtype={self.dtype_device()[0]}, "
            f"device={self.dtype_device()[1]})"
        )
