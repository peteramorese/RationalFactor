from __future__ import annotations

import torch

from rational_factor.models.basis_functions import Basis, SeparableBasis
from rational_factor.models.tt.tt import MPO, TTVector
from rational_factor.models.parameters import OneVectorParameters


class TTBasis(Basis):
    """Tensor-product basis with rank-one TT evaluations and MPO Grams.

    The wrapped SeparableBasis supplies local factors only; no coefficient
    tensor is applied. Its parameter batch size must be 1, but evaluations
    may contain any number of samples.
    """

    def __init__(self, primitives: SeparableBasis):
        if not isinstance(primitives, SeparableBasis):
            raise TypeError("primitives must be a SeparableBasis")
        if primitives.batch_size() != 1:
            raise ValueError("TTBasis requires primitives with batch_size=1")
        self._primitives = primitives
        self._modes = (primitives.n_basis_functions(),) * primitives.dim()
        super().__init__(
            dim=primitives.dim(), batch_size=1,
            n_basis=primitives.n_basis_functions() ** primitives.dim(),
            params=primitives.params(), coeffs=None,
        )


    def set_coeffs_to_one(self) -> None:
        """Use implicit all-ones tensor until coefficients are actually supplied."""
        dtype, device = self.dtype_device()
        self.coeffs = OneVectorParameters(self._n_basis, (1,), dtype, device)
        self._tt_has_coeffs = False

    def set_coeffs(self, coeffs) -> None:
        """Enable TT coefficient weighting (needed for RFF g/h belief views)."""
        super().set_coeffs(coeffs)
        values = coeffs()
        if not isinstance(values, TTVector) or values.modes != self._modes:
            raise ValueError("TTBasis coefficients must be a TTVector with matching modes")
        self._tt_has_coeffs = True

    def _coeff_tt(self) -> TTVector:
        if self._tt_has_coeffs:
            return self.coeffs()
        dtype, device = self.dtype_device()
        return TTVector([torch.ones(1, n, 1, dtype=dtype, device=device)
                         for n in self.modes])

    @property
    def primitives(self) -> SeparableBasis:
        return self._primitives

    @property
    def modes(self) -> tuple[int, ...]:
        return self._modes

    @property
    def ranks(self) -> tuple[int, ...]:
        return self._coeff_tt().ranks if self._tt_has_coeffs else (1,) * (self.dim() + 1)

    def n_primitives_per_dim(self) -> int:
        return self._primitives.n_basis_functions()

    def dtype_device(self):
        return self._primitives.dtype_device()

    @staticmethod
    def _tt_factors(values: torch.Tensor, batched: bool) -> TTVector:
        if batched:
            return TTVector([v[:, None, :, None] for v in values.unbind(dim=1)])
        return TTVector([v[None, :, None] for v in values[0].unbind(dim=0)])

    def forward(self, y: torch.Tensor) -> TTVector:
        """Return a rank-one TT of shape (batch, n_basis) (unbatched when batch=1)."""
        values = self._primitives.eval_dim(y)  # (batch, dim, n_local)
        out = self._tt_factors(values, batched=values.shape[0] != 1)
        return out * self._coeff_tt() if self._tt_has_coeffs else out

    __call__ = forward

    def Omega1(self, lows: torch.Tensor = None, highs: torch.Tensor = None) -> TTVector:
        """Tensor-product basis integrals as an unbatched rank-one TT."""
        values = self._primitives.Omega1_dim(lows, highs, return_log=False)
        out = self._tt_factors(values, batched=False)
        return out * self._coeff_tt() if self._tt_has_coeffs else out

    def Omega2(
        self, other: "TTBasis", lows: torch.Tensor = None, highs: torch.Tensor = None,
    ) -> MPO:
        """Tensor-product Gram as an exact bond-rank-one MPO, including signed entries."""
        if not isinstance(other, TTBasis) or self.dim() != other.dim():
            raise ValueError("other must be a TTBasis of the same dimension")
        grams = self._primitives.Omega2_dim(
            other._primitives, lows, highs, return_log=False,
        )  # (1, dim, n_self, n_other)
        if not self._tt_has_coeffs and not other._tt_has_coeffs:
            return MPO([g[None, :, :, None] for g in grams[0].unbind(dim=0)])
        left, right = self._coeff_tt(), other._coeff_tt()
        cores = []
        for a, g, b in zip(left.cores, grams[0].unbind(dim=0), right.cores):
            # diag(left) @ gram @ diag(right), with TT bond products.
            c = torch.einsum('aib,ij,cjd->acijbd', a, g, b)
            cores.append(c.reshape(a.shape[0] * b.shape[0], g.shape[0],
                                   g.shape[1], a.shape[-1] * b.shape[-1]))
        return MPO(cores)

    def __repr__(self) -> str:
        return f"TTBasis(dim={self.dim()}, modes={self.modes}, ranks={self.ranks})"
