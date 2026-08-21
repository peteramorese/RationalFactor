import torch


class LowRankPlusDiagonal:
    """Batched low-rank-plus-diagonal matrix ``M = diag(D) + Uᵀ V``.

    Factors are stored with the thin axis first:

    - ``D``: ``(..., n)`` diagonal entries
    - ``U``, ``V``: ``(..., k, n)`` with ``k << n``

    Matrix–vector products use ``Mx = D ⊙ x + Uᵀ (V x)`` and cost ``O(n k)``
    per batch element (linear in ``n`` for fixed rank ``k``), never forming the
    dense ``n × n`` matrix.
    """

    def __init__(self, D: torch.Tensor, U: torch.Tensor, V: torch.Tensor):
        D = torch.as_tensor(D)
        U = torch.as_tensor(U)
        V = torch.as_tensor(V)

        if D.dim() < 1:
            raise ValueError("D must have shape (..., n)")
        if U.dim() < 2 or V.dim() < 2:
            raise ValueError("U and V must have shape (..., k, n)")
        if U.shape != V.shape:
            raise ValueError(f"U and V must have the same shape, got {tuple(U.shape)} and {tuple(V.shape)}")
        if U.shape[:-2] != D.shape[:-1]:
            raise ValueError(
                f"batch shapes of D and U/V must match, got D[..., n]={tuple(D.shape)} "
                f"and U/V[..., k, n]={tuple(U.shape)}"
            )
        if U.shape[-1] != D.shape[-1]:
            raise ValueError(
                f"trailing size of D and U/V must match (n), got n={D.shape[-1]} and U/V n={U.shape[-1]}"
            )
        if D.dtype != U.dtype or D.dtype != V.dtype:
            raise ValueError("D, U, and V must share the same dtype")
        if D.device != U.device or D.device != V.device:
            raise ValueError("D, U, and V must share the same device")

        self.D = D
        self.U = U
        self.V = V

    @property
    def n(self) -> int:
        return self.D.shape[-1]

    @property
    def k(self) -> int:
        return self.U.shape[-2]

    @property
    def batch_shape(self) -> torch.Size:
        return self.D.shape[:-1]

    @property
    def shape(self) -> torch.Size:
        return self.batch_shape + torch.Size([self.n, self.n])

    @property
    def dtype(self) -> torch.dtype:
        return self.D.dtype

    @property
    def device(self) -> torch.device:
        return self.D.device

    @property
    def T(self) -> "LowRankPlusDiagonal":
        return LowRankPlusDiagonal(self.D, self.V, self.U)

    def to(self, *args, **kwargs) -> "LowRankPlusDiagonal":
        return LowRankPlusDiagonal(
            self.D.to(*args, **kwargs),
            self.U.to(*args, **kwargs),
            self.V.to(*args, **kwargs),
        )

    def to_dense(self) -> torch.Tensor:
        return torch.diag_embed(self.D) + self.U.transpose(-2, -1) @ self.V

    def inverse(self) -> "LowRankPlusDiagonal":
        """Invser using woodbury formula."""
        D_inv = self.D.reciprocal()
        U_scaled = self.U * D_inv.unsqueeze(-2)
        V_scaled = self.V * D_inv.unsqueeze(-2)
        # S = I + V D^{-1} Uᵀ = I + V @ U_scaledᵀ
        eye = torch.eye(self.k, dtype=self.dtype, device=self.device)
        S = eye + self.V @ U_scaled.transpose(-2, -1)
        # M^{-1} = diag(D_inv) + U_invᵀ V_inv with
        # U_inv = -S^{-ᵀ} U_scaled, V_inv = V_scaled
        U_inv = -torch.linalg.solve(S.transpose(-2, -1), U_scaled)
        return LowRankPlusDiagonal(D_inv, U_inv, V_scaled)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``M @ x`` in ``O(n k)`` time (per batch / RHS).

        Args:
            x: ``(..., n)`` or ``(..., n, m)``. Leading dims broadcast with
                ``batch_shape``.
        """
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        n = self.n
        batch_ndim = self.D.dim() - 1
        # Multi-RHS: (..., n, m) — one more axis than a batched vector.
        if x.dim() >= batch_ndim + 2 and x.shape[-2] == n:
            Vx = torch.einsum("...ki,...im->...km", self.V, x)
            return self.D.unsqueeze(-1) * x + torch.einsum("...ki,...km->...im", self.U, Vx)
        if x.shape[-1] == n:
            Vx = torch.einsum("...ki,...i->...k", self.V, x)
            return self.D * x + torch.einsum("...ki,...k->...i", self.U, Vx)
        raise ValueError(
            f"x must have shape (..., n) or (..., n, m) with n={n}, got {tuple(x.shape)}"
        )

    def __matmul__(self, other: torch.Tensor) -> torch.Tensor:
        return self.matvec(other)

    def __rmatmul__(self, other: torch.Tensor) -> torch.Tensor:
        """Row-vector / left multiply: ``x @ M == (Mᵀ @ xᵀ)ᵀ``."""
        other = torch.as_tensor(other, dtype=self.dtype, device=self.device)
        if other.shape[-1] == self.n and other.dim() == self.D.dim():
            return self.T.matvec(other)
        if other.shape[-1] == self.n:
            # other: (..., m, n) → treat rows as vectors
            return self.T.matvec(other.transpose(-2, -1)).transpose(-2, -1)
        raise ValueError(
            f"left operand must have shape (..., n) or (..., m, n) with n={self.n}, got {tuple(other.shape)}"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(batch_shape={tuple(self.batch_shape)}, "
            f"n={self.n}, k={self.k}, dtype={self.dtype}, device={self.device})"
        )
