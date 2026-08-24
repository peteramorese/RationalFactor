import torch
from rational_factor.models.parameters import Parameters


class Rank1PlusDiagonal:
    """Batched rank-1-plus-diagonal matrix ``M = diag(d) + u vᵀ``.

    - ``d``: ``(..., n)`` diagonal entries
    - ``u``, ``v``: ``(..., n)`` vectors

    Matrix–vector products use ``Mx = d ⊙ x + u (vᵀ x)`` and cost ``O(n)``
    per batch element, never forming the dense ``n × n`` matrix.
    """

    def __init__(self, d: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
        d = torch.as_tensor(d)
        u = torch.as_tensor(u)
        v = torch.as_tensor(v)

        if d.dim() < 1:
            raise ValueError("d must have shape (..., n)")
        if u.dim() < 1 or v.dim() < 1:
            raise ValueError("u and v must have shape (..., n)")
        if u.shape != v.shape:
            raise ValueError(f"u and v must have the same shape, got {tuple(u.shape)} and {tuple(v.shape)}")
        if u.shape != d.shape:
            raise ValueError(
                f"d, u, and v must have the same shape (..., n), got "
                f"d={tuple(d.shape)}, u={tuple(u.shape)}, v={tuple(v.shape)}"
            )
        if d.dtype != u.dtype or d.dtype != v.dtype:
            raise ValueError("d, u, and v must share the same dtype")
        if d.device != u.device or d.device != v.device:
            raise ValueError("d, u, and v must share the same device")

        self.d = d
        self.u = u
        self.v = v

    @property
    def n(self) -> int:
        return self.d.shape[-1]

    @property
    def batch_shape(self) -> torch.Size:
        return self.d.shape[:-1]

    @property
    def shape(self) -> torch.Size:
        return self.batch_shape + torch.Size([self.n, self.n])

    @property
    def dtype(self) -> torch.dtype:
        return self.d.dtype

    @property
    def device(self) -> torch.device:
        return self.d.device

    @property
    def T(self) -> "Rank1PlusDiagonal":
        """Transpose: ``(diag(d) + u vᵀ)ᵀ = diag(d) + v uᵀ``."""
        return Rank1PlusDiagonal(self.d, self.v, self.u)

    def to(self, *args, **kwargs) -> "Rank1PlusDiagonal":
        return Rank1PlusDiagonal(
            self.d.to(*args, **kwargs),
            self.u.to(*args, **kwargs),
            self.v.to(*args, **kwargs),
        )

    def to_dense(self) -> torch.Tensor:
        """Materialize the full ``(..., n, n)`` matrix (``O(n^2)``)."""
        return torch.diag_embed(self.d) + self.u.unsqueeze(-1) * self.v.unsqueeze(-2)

    def _prepare_diag(self, a: torch.Tensor) -> torch.Tensor:
        a = torch.as_tensor(a, dtype=self.dtype, device=self.device)
        if a.shape[-1] != self.n:
            raise ValueError(
                f"diagonal must have trailing size n={self.n}, got shape {tuple(a.shape)}"
            )
        try:
            torch.broadcast_shapes(a.shape, self.d.shape)
        except RuntimeError as e:
            raise ValueError(
                f"diagonal shape {tuple(a.shape)} is not broadcastable with "
                f"factor shape {tuple(self.d.shape)}"
            ) from e
        return a

    def mul_diag_left(self, a: torch.Tensor) -> "Rank1PlusDiagonal":
        """Left-multiply by ``diag(a)``: ``diag(a) M = diag(a ⊙ d) + (a ⊙ u) vᵀ``.

        ``a`` has shape ``(..., n)``, broadcastable with ``d``.
        """
        a = self._prepare_diag(a)
        d, u, v = torch.broadcast_tensors(a * self.d, a * self.u, self.v)
        return Rank1PlusDiagonal(d.clone(), u.clone(), v.clone())

    def mul_diag_right(self, a: torch.Tensor) -> "Rank1PlusDiagonal":
        """Right-multiply by ``diag(a)``: ``M diag(a) = diag(d ⊙ a) + u (a ⊙ v)ᵀ``.

        ``a`` has shape ``(..., n)``, broadcastable with ``d``.
        """
        a = self._prepare_diag(a)
        d, u, v = torch.broadcast_tensors(self.d * a, self.u, self.v * a)
        return Rank1PlusDiagonal(d.clone(), u.clone(), v.clone())

    def mul_diag(self, a: torch.Tensor, *, side: str = "left") -> "Rank1PlusDiagonal":
        """Multiply by ``diag(a)`` on ``side`` ``\"left\"`` or ``\"right\"``."""
        if side == "left":
            return self.mul_diag_left(a)
        if side == "right":
            return self.mul_diag_right(a)
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    def inverse(self) -> "Rank1PlusDiagonal":
        """Inverse via Sherman–Morrison in ``O(n)`` time (batched over leading dims)."""
        d_inv = self.d.reciprocal()
        u_scaled = d_inv * self.u
        v_scaled = d_inv * self.v
        alpha = 1.0 + (self.v * u_scaled).sum(dim=-1)
        u_inv = -u_scaled / alpha.unsqueeze(-1)
        return Rank1PlusDiagonal(d_inv, u_inv, v_scaled)

    def flip(self, seq_dim: int = 0) -> "Rank1PlusDiagonal":
        """Reverse factor order along ``seq_dim`` (e.g. for ``(M₀⋯M_{T-1})⁻¹``)."""
        return Rank1PlusDiagonal(
            self.d.flip(seq_dim),
            self.u.flip(seq_dim),
            self.v.flip(seq_dim),
        )

    def sequential_matvec(
        self,
        x0: torch.Tensor,
        *,
        seq_dim: int = 0,
        reverse: bool = False,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """Apply ``M_t`` sequentially: ``x_{t+1} = M_t x_t``.

        One batch axis (``seq_dim``) indexes the time-ordered factors; all other
        batch axes are independent sequences updated in parallel.

        With ``reverse=False`` (default), applies ``M_0``, then ``M_1``, … so the
        product is ``M_{T-1} ⋯ M_0``. With ``reverse=True``, applies ``M_{T-1}``,
        then ``M_{T-2}``, … (same as ``self.flip(seq_dim).sequential_matvec(...)``).

        Args:
            x0: ``(..., n)`` where ``...`` is ``batch_shape`` with ``seq_dim``
                removed (e.g. factors ``(B, T, n)``, ``seq_dim=1`` → ``x0`` is
                ``(B, n)``).
            seq_dim: batch axis along which factors are ordered in time.
            reverse: if True, traverse the sequence axis backward.
            return_trajectory: if True, return all states with shape
                ``batch_shape + (n,)`` (same as factors but ``seq_dim`` length
                is ``T + 1``). Otherwise return the final state only.

        Returns:
            Final state ``(..., n)``, or full trajectory if
            ``return_trajectory=True``.
        """
        if reverse:
            return self.flip(seq_dim).sequential_matvec(
                x0, seq_dim=seq_dim, reverse=False, return_trajectory=return_trajectory
            )

        batch_ndim = self.d.dim() - 1
        if batch_ndim == 0:
            raise ValueError(
                "sequential_matvec requires at least one batch dim (sequence length); "
                f"got factor shape {tuple(self.d.shape)}"
            )
        if not (-batch_ndim <= seq_dim < batch_ndim):
            raise ValueError(
                f"seq_dim must index a batch axis of d/u/v, got seq_dim={seq_dim} "
                f"for batch_shape={tuple(self.batch_shape)}"
            )
        if seq_dim < 0:
            seq_dim += batch_ndim

        d = self.d.movedim(seq_dim, 0)
        u = self.u.movedim(seq_dim, 0)
        v = self.v.movedim(seq_dim, 0)
        T_seq = d.shape[0]
        state_shape = d.shape[1:]

        x = torch.as_tensor(x0, dtype=self.dtype, device=self.device)
        if x.shape != state_shape:
            raise ValueError(
                f"x0 must have shape {tuple(state_shape)} (batch without seq_dim + (n,)), "
                f"got {tuple(x.shape)}"
            )

        if return_trajectory:
            traj = x.new_empty((T_seq + 1,) + state_shape)
            traj[0] = x
            for t in range(T_seq):
                vx = (v[t] * x).sum(dim=-1)
                x = d[t] * x + u[t] * vx.unsqueeze(-1)
                traj[t + 1] = x
            return traj.movedim(0, seq_dim)

        for t in range(T_seq):
            vx = (v[t] * x).sum(dim=-1)
            x = d[t] * x + u[t] * vx.unsqueeze(-1)
        return x

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``M @ x`` in ``O(n)`` time (per batch / RHS).

        Args:
            x: ``(..., n)`` or ``(..., n, m)``. Leading dims broadcast with
                ``batch_shape``.
        """
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        n = self.n
        batch_ndim = self.d.dim() - 1
        if x.dim() >= batch_ndim + 2 and x.shape[-2] == n:
            vx = torch.einsum("...i,...im->...m", self.v, x)
            return self.d.unsqueeze(-1) * x + self.u.unsqueeze(-1) * vx.unsqueeze(-2)
        if x.shape[-1] == n:
            vx = torch.einsum("...i,...i->...", self.v, x)
            return self.d * x + self.u * vx.unsqueeze(-1)
        raise ValueError(
            f"x must have shape (..., n) or (..., n, m) with n={n}, got {tuple(x.shape)}"
        )

    def __matmul__(self, other: torch.Tensor) -> torch.Tensor:
        return self.matvec(other)

    def __rmatmul__(self, other: torch.Tensor) -> torch.Tensor:
        """Row-vector / left multiply: ``x @ M == (Mᵀ @ xᵀ)ᵀ``."""
        other = torch.as_tensor(other, dtype=self.dtype, device=self.device)
        if other.shape[-1] == self.n and other.dim() == self.d.dim():
            return self.T.matvec(other)
        if other.shape[-1] == self.n:
            return self.T.matvec(other.transpose(-2, -1)).transpose(-2, -1)
        raise ValueError(
            f"left operand must have shape (..., n) or (..., m, n) with n={self.n}, got {tuple(other.shape)}"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(batch_shape={tuple(self.batch_shape)}, "
            f"n={self.n}, dtype={self.dtype}, device={self.device})"
        )


class Rank1PlusDiagonalFactorization:
    """Stores R1PD factor tensors ``d, u, v`` with shape ``(T, n)``.

    Call ``()`` to get a ``Rank1PlusDiagonal``; apply products with
    ``sequential_matvec`` / ``inverse`` / ``T`` / ``flip`` on that object.
    """

    def __init__(self, d: Parameters, u: Parameters, v: Parameters):
        assert d.size() == u.size() == v.size(), "d, u, and v must have the same shape"
        assert d.size().__len__() == 2, "d, u, and v must have shape (n_factors, n)"
        self.d = d
        self.u = u
        self.v = v

    def __call__(self) -> Rank1PlusDiagonal:
        return Rank1PlusDiagonal(self.d(), self.u(), self.v())
