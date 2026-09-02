"""Structured matrix approximators: rank-1-plus-diagonal and order-1 quasiseparable."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


class Matrix(ABC):
    """Batched matrix-like linear map of shape ``(..., n, m)``.

    Subclasses keep a dense tensor or a structured factorization. Callers use
    ``matvec`` / ``rev_matvec`` and do not branch on the representation.
    """

    @property
    @abstractmethod
    def shape(self) -> torch.Size: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def T(self) -> Matrix: ...

    @abstractmethod
    def to_dense(self) -> torch.Tensor: ...

    @abstractmethod
    def matvec(self, x: torch.Tensor) -> torch.Tensor: ...

    def rev_matvec(self, x: torch.Tensor) -> torch.Tensor:
        """``Mᵀ @ x``."""
        return self.T.matvec(x)

    def diag(self) -> torch.Tensor:
        """Main-diagonal vector ``(..., min(n, m))``."""
        return self.to_dense().diagonal(dim1=-2, dim2=-1)

    def sum(self) -> torch.Tensor:
        """Sum of all matrix entries."""
        return self.to_dense().sum()

    def mul_diag_left(self, a: torch.Tensor) -> Matrix:
        """Left-multiply by ``diag(a)``."""
        a = torch.as_tensor(a, dtype=self.dtype, device=self.device)
        return DenseMatrix(a.unsqueeze(-1) * self.to_dense())

    def mul_diag_right(self, a: torch.Tensor) -> Matrix:
        """Right-multiply by ``diag(a)``."""
        a = torch.as_tensor(a, dtype=self.dtype, device=self.device)
        return DenseMatrix(self.to_dense() * a.unsqueeze(-2))

    def mul_diag(self, a: torch.Tensor, *, side: str = "left") -> Matrix:
        if side == "left":
            return self.mul_diag_left(a)
        if side == "right":
            return self.mul_diag_right(a)
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    def scale(self, s: torch.Tensor | float) -> Matrix:
        """Multiply every entry by scalar ``s`` (broadcasts over batch)."""
        s = torch.as_tensor(s, dtype=self.dtype, device=self.device)
        return DenseMatrix(self.to_dense() * s)

    def __matmul__(self, other: torch.Tensor) -> torch.Tensor:
        return self.matvec(other)

    def __rmatmul__(self, other: torch.Tensor) -> torch.Tensor:
        """Row-vector / left multiply: ``x @ M == (Mᵀ @ xᵀ)ᵀ``."""
        other = torch.as_tensor(other, dtype=self.dtype, device=self.device)
        n_out = self.shape[-2]
        if other.shape[-1] != n_out:
            raise ValueError(
                f"left operand must have shape (..., {n_out}) or (..., k, {n_out}), got {tuple(other.shape)}"
            )
        batch_ndim = len(self.shape) - 2
        if other.dim() == batch_ndim + 1:
            return self.T.matvec(other)
        return self.T.matvec(other.transpose(-2, -1)).transpose(-2, -1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(shape={tuple(self.shape)}, "
            f"dtype={self.dtype}, device={self.device})"
        )


class DenseMatrix(Matrix):
    """Arbitrary dense matrix stored as a ``(..., n, m)`` tensor."""

    def __init__(self, values: torch.Tensor):
        values = torch.as_tensor(values)
        if values.dim() < 2:
            raise ValueError(f"dense matrix must have shape (..., n, m), got {tuple(values.shape)}")
        self._values = values

    @property
    def shape(self) -> torch.Size:
        return self._values.shape

    @property
    def dtype(self) -> torch.dtype:
        return self._values.dtype

    @property
    def device(self) -> torch.device:
        return self._values.device

    @property
    def T(self) -> DenseMatrix:
        return DenseMatrix(self._values.transpose(-2, -1))

    def to_dense(self) -> torch.Tensor:
        return self._values

    def diag(self) -> torch.Tensor:
        return self._values.diagonal(dim1=-2, dim2=-1)

    def sum(self) -> torch.Tensor:
        return self._values.sum()

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        n_in = self._values.shape[-1]
        batch_ndim = self._values.dim() - 2
        if x.dim() >= batch_ndim + 2 and x.shape[-2] == n_in:
            return torch.einsum("...ij,...jk->...ik", self._values, x)
        if x.shape[-1] == n_in:
            return torch.einsum("...ij,...j->...i", self._values, x)
        raise ValueError(
            f"x must have shape (..., {n_in}) or (..., {n_in}, k), got {tuple(x.shape)}"
        )

    def mul_diag_left(self, a: torch.Tensor) -> DenseMatrix:
        a = torch.as_tensor(a, dtype=self.dtype, device=self.device)
        return DenseMatrix(a.unsqueeze(-1) * self._values)

    def mul_diag_right(self, a: torch.Tensor) -> DenseMatrix:
        a = torch.as_tensor(a, dtype=self.dtype, device=self.device)
        return DenseMatrix(self._values * a.unsqueeze(-2))

    def scale(self, s: torch.Tensor | float) -> DenseMatrix:
        s = torch.as_tensor(s, dtype=self.dtype, device=self.device)
        return DenseMatrix(self._values * s)


def as_matrix(obj: torch.Tensor | Matrix) -> Matrix:
    """Return ``obj`` if it is already a ``Matrix``, otherwise wrap a dense tensor."""
    if isinstance(obj, Matrix):
        return obj
    return DenseMatrix(obj)


class Rank1PlusDiagonal(Matrix):
    """Batched rank-1-plus-diagonal matrix ``M = diag(d) + u vᵀ``.

    - ``u``, ``v``: ``(..., n)`` vectors
    - ``d``: ``(..., n)`` diagonal entries. Optional: if omitted and
      ``normalization_dim`` is ``None``, the diagonal is ones (``M = I + u vᵀ``).
    - ``normalization_dim``: ``None``, ``0``, or ``1``. If set, ``d`` is ignored
      and ``u``, ``v`` are unconstrained parameters of a stochastic matrix
      ``diag(1 - u) + u vᵀ`` (row-stochastic) or its transpose (column-stochastic).
      Every matrix in the batch is normalized independently.

    Matrix–vector products use ``Mx = d ⊙ x + u (vᵀ x)`` and cost ``O(n)``
    per batch element, never forming the dense ``n × n`` matrix.
    """

    def __init__(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        d: torch.Tensor | None = None,
        normalization_dim: int | None = None,
    ):
        u = torch.as_tensor(u)
        v = torch.as_tensor(v)

        if u.dim() < 1 or v.dim() < 1:
            raise ValueError("u and v must have shape (..., n)")
        if u.shape != v.shape:
            raise ValueError(f"u and v must have the same shape, got {tuple(u.shape)} and {tuple(v.shape)}")
        if u.dtype != v.dtype:
            raise ValueError("u and v must share the same dtype")
        if u.device != v.device:
            raise ValueError("u and v must share the same device")

        if normalization_dim is not None:
            if normalization_dim not in (0, 1):
                raise ValueError(f"normalization_dim must be None, 0, or 1, got {normalization_dim!r}")
            if normalization_dim == 1:
                # Row-stochastic: M = diag(1 - u) + u vᵀ with u ∈ [0, 1], v a distribution.
                u = torch.sigmoid(u)
                v = torch.softmax(v, dim=-1)
                d = 1.0 - u
            else:
                # Column-stochastic: M = diag(1 - v) + u vᵀ with v ∈ [0, 1], u a distribution.
                u = torch.softmax(u, dim=-1)
                v = torch.sigmoid(v)
                d = 1.0 - v
        elif d is None:
            d = torch.ones_like(u)
        else:
            d = torch.as_tensor(d)
            if d.dim() < 1:
                raise ValueError("d must have shape (..., n)")
            if u.shape != d.shape:
                raise ValueError(
                    f"d, u, and v must have the same shape (..., n), got "
                    f"d={tuple(d.shape)}, u={tuple(u.shape)}, v={tuple(v.shape)}"
                )
            if d.dtype != u.dtype:
                raise ValueError("d, u, and v must share the same dtype")
            if d.device != u.device:
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
        return Rank1PlusDiagonal(self.v, self.u, self.d)

    def to(self, *args, **kwargs) -> "Rank1PlusDiagonal":
        return Rank1PlusDiagonal(
            self.u.to(*args, **kwargs),
            self.v.to(*args, **kwargs),
            self.d.to(*args, **kwargs),
        )

    def to_dense(self) -> torch.Tensor:
        """Materialize the full ``(..., n, n)`` matrix (``O(n^2)``)."""
        return torch.diag_embed(self.d) + self.u.unsqueeze(-1) * self.v.unsqueeze(-2)

    def diag(self) -> torch.Tensor:
        return self.d + self.u * self.v

    def sum(self) -> torch.Tensor:
        return self.d.sum() + (self.u.sum(dim=-1) * self.v.sum(dim=-1)).sum()

    def scale(self, s: torch.Tensor | float) -> Rank1PlusDiagonal:
        s = torch.as_tensor(s, dtype=self.dtype, device=self.device)
        return Rank1PlusDiagonal(self.u * s, self.v, self.d * s)

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
        return Rank1PlusDiagonal(u.clone(), v.clone(), d.clone())

    def mul_diag_right(self, a: torch.Tensor) -> "Rank1PlusDiagonal":
        """Right-multiply by ``diag(a)``: ``M diag(a) = diag(d ⊙ a) + u (a ⊙ v)ᵀ``.

        ``a`` has shape ``(..., n)``, broadcastable with ``d``.
        """
        a = self._prepare_diag(a)
        d, u, v = torch.broadcast_tensors(self.d * a, self.u, self.v * a)
        return Rank1PlusDiagonal(u.clone(), v.clone(), d.clone())

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
        return Rank1PlusDiagonal(u_inv, v_scaled, d_inv)

    def flip(self, seq_dim: int = 0) -> "Rank1PlusDiagonal":
        """Reverse factor order along ``seq_dim`` (e.g. for ``(M₀⋯M_{T-1})⁻¹``)."""
        return Rank1PlusDiagonal(
            self.u.flip(seq_dim),
            self.v.flip(seq_dim),
            self.d.flip(seq_dim),
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


class Semiseparable(Matrix):
    """Unit order-1 semiseparable matrix with generators ``p, a, q`` (shape ``(..., n)``).

    Lower (``upper=False``):

        M[i, j] = p[i] (prod_{k=j+1}^{i-1} a[k]) q[j]   i > j
        M[i, i] = 1
        M[i, j] = 0                                      i < j

    Upper is the same recurrence on reversed indices. Matvec and triangular
    solve are ``O(n)`` via an affine scan; ``to_dense`` is ``O(n^2)``.
    """

    def __init__(
        self,
        p: torch.Tensor,
        a: torch.Tensor,
        q: torch.Tensor,
        *,
        upper: bool = False,
    ):
        p, a, q = torch.as_tensor(p), torch.as_tensor(a), torch.as_tensor(q)
        if p.dim() < 1:
            raise ValueError("p, a, and q must have shape (..., n)")
        if p.shape != a.shape or p.shape != q.shape:
            raise ValueError(
                f"p, a, and q must have the same shape, got "
                f"p={tuple(p.shape)}, a={tuple(a.shape)}, q={tuple(q.shape)}"
            )
        self.p, self.a, self.q = p, a, q
        self.upper = upper

    @property
    def n(self) -> int:
        return self.p.shape[-1]

    @property
    def shape(self) -> torch.Size:
        return self.p.shape[:-1] + torch.Size([self.n, self.n])

    @property
    def dtype(self) -> torch.dtype:
        return self.p.dtype

    @property
    def device(self) -> torch.device:
        return self.p.device

    @property
    def T(self) -> Semiseparable:
        return Semiseparable(self.q, self.a, self.p, upper=not self.upper)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply(x, solve=False)

    def solve(self, x: torch.Tensor) -> torch.Tensor:
        """Solve ``M y = x`` (unit triangular, ``O(n)``)."""
        return self._apply(x, solve=True)

    def _apply(self, x: torch.Tensor, *, solve: bool) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=self.p.dtype, device=self.p.device)
        p, a, q = self.p, self.a, self.q
        if self.upper:
            x, p, a, q = x.flip(-1), p.flip(-1), a.flip(-1), q.flip(-1)
        if x.shape[-1] == 1:
            y = x
        else:
            trans_a = a[..., :-1] - q[..., :-1] * p[..., :-1] if solve else a[..., :-1]
            s = self.scan_from_zero(trans_a, q[..., :-1] * x[..., :-1])
            y = x + (-p if solve else p) * s
        return y.flip(-1) if self.upper else y

    @staticmethod
    def scan_from_zero(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """``s[..., 0] = 0``, ``s[..., i+1] = a[..., i] s[..., i] + b[..., i]``.

        ``a, b`` have shape ``(..., n)`` (``n`` transitions). Result is ``(..., n+1)``.
        """
        a, b = torch.broadcast_tensors(a, b)
        zero = b.new_zeros(b.shape[:-1] + (1,))
        if a.shape[-1] == 0:
            return zero
        n, k = a.shape[-1], 1
        while k < n:
            a_l, b_l = a[..., :-k], b[..., :-k]
            a_r, b_r = a[..., k:], b[..., k:]
            b = torch.cat((b[..., :k], a_r * b_l + b_r), dim=-1)
            a = torch.cat((a[..., :k], a_r * a_l), dim=-1)
            k *= 2
        return torch.cat((zero, b), dim=-1)

    def strict_triangle(self) -> torch.Tensor:
        """Off-diagonal triangle (zeros on and above/below the diagonal)."""
        p, a, q = self.p, self.a, self.q
        if self.upper:
            p, a, q = p.flip(-1), a.flip(-1), q.flip(-1)
        m = a.shape[-1]
        ones = torch.ones_like(a)
        idx_i = torch.arange(m, device=a.device).view(*([1] * (a.ndim - 1)), m, 1)
        idx_j = torch.arange(m, device=a.device).view(*([1] * (a.ndim - 1)), 1, m)
        factors = torch.where(idx_i > idx_j, a.unsqueeze(-1), ones.unsqueeze(-1))
        scale = a.new_zeros(a.shape + (m,))
        scale[..., 1:, :] = torch.cumprod(factors, dim=-2)[..., :-1, :]
        tri = torch.tril(p.unsqueeze(-1) * scale * q.unsqueeze(-2), diagonal=-1)
        return tri.flip(-1).flip(-2) if self.upper else tri

    def to_dense(self) -> torch.Tensor:
        return torch.diag_embed(torch.ones_like(self.p)) + self.strict_triangle()


@dataclass
class Order1QSGenerators:
    """Direct scalar generators of an order-1 quasiseparable matrix

        Q[i,j] =
            p[i] * prod_{k=j+1}^{i-1} a[k] * q[j],   i > j
            d[i],                                     i = j
            g[i] * prod_{k=i+1}^{j-1} b[k] * h[j],   i < j

    Every tensor has shape ``(..., m)``. Used for row extrema and debug densify.
    """

    p: torch.Tensor
    a: torch.Tensor
    q: torch.Tensor
    d: torch.Tensor
    g: torch.Tensor
    b: torch.Tensor
    h: torch.Tensor

    @property
    def n(self) -> int:
        return self.d.shape[-1]

    def row_minmax(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact per-row min/max over columns."""
        M = self.to_dense()
        return M.amin(dim=-1), M.amax(dim=-1)

    def to_dense(self) -> torch.Tensor:
        """Materialize the full matrix (``O(m^2)``, debug / row extrema)."""
        lower = Semiseparable(self.p, self.a, self.q)
        upper = Semiseparable(self.g, self.b, self.h, upper=True)
        return torch.diag_embed(self.d) + lower.strict_triangle() + upper.strict_triangle()


class Order1Quasiseparable(Matrix):
    """Invertible order-1 quasiseparable ``P = L D U``.

    ``L`` / ``U`` are unit lower / upper ``Semiseparable`` factors; ``D`` is a
    nonzero diagonal. All generators have shape ``(..., m)``.

    Matvecs ``P x``, ``P^{-1} x``, ``P^T x``, ``P^{-T} x`` are ``O(m)``.
    """

    def __init__(
        self,
        lower_p: torch.Tensor,
        lower_a: torch.Tensor,
        lower_q: torch.Tensor,
        diag: torch.Tensor,
        upper_g: torch.Tensor,
        upper_b: torch.Tensor,
        upper_h: torch.Tensor,
    ):
        tensors = (lower_p, lower_a, lower_q, diag, upper_g, upper_b, upper_h)
        if any(x.shape != diag.shape for x in tensors):
            raise ValueError("all generators must share the same shape")
        self.L = Semiseparable(lower_p, lower_a, lower_q)
        self.d = diag
        self.U = Semiseparable(upper_g, upper_b, upper_h, upper=True)

    @property
    def n(self) -> int:
        return self.d.shape[-1]

    @property
    def shape(self) -> torch.Size:
        return self.d.shape[:-1] + torch.Size([self.n, self.n])

    @property
    def dtype(self) -> torch.dtype:
        return self.d.dtype

    @property
    def device(self) -> torch.device:
        return self.d.device

    @property
    def T(self) -> Order1Quasiseparable:
        """``Pᵀ = Uᵀ D Lᵀ``."""
        return Order1Quasiseparable(self.uh, self.ub, self.ug, self.d, self.lq, self.la, self.lp)

    @property
    def lp(self) -> torch.Tensor:
        return self.L.p

    @property
    def la(self) -> torch.Tensor:
        return self.L.a

    @property
    def lq(self) -> torch.Tensor:
        return self.L.q

    @property
    def ug(self) -> torch.Tensor:
        return self.U.p

    @property
    def ub(self) -> torch.Tensor:
        return self.U.a

    @property
    def uh(self) -> torch.Tensor:
        return self.U.q

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """``P x = L D U x``."""
        return self.L.matvec(self.d * self.U.matvec(x))

    def inverse_matvec(self, x: torch.Tensor) -> torch.Tensor:
        """``P^{-1} x = U^{-1} D^{-1} L^{-1} x``."""
        return self.U.solve(self.L.solve(x) / self.d)

    def T_matvec(self, x: torch.Tensor) -> torch.Tensor:
        """``P^T x = U^T D L^T x``."""
        return self.T.matvec(x)

    def invT_matvec(self, x: torch.Tensor) -> torch.Tensor:
        """``P^{-T} x = L^{-T} D^{-1} U^{-T} x``."""
        return self.T.inverse_matvec(x)

    def direct_generators(self) -> Order1QSGenerators:
        """Direct QS generators of ``P`` itself (``O(m)``)."""
        trans_a = self.la[..., :-1] * self.ub[..., :-1]
        trans_b = self.lq[..., :-1] * self.d[..., :-1] * self.ug[..., :-1]
        S = Semiseparable.scan_from_zero(trans_a, trans_b)
        return Order1QSGenerators(
            p=self.lp,
            a=self.la,
            q=self.lq * self.d + self.la * self.uh * S,
            d=self.d + self.lp * self.uh * S,
            g=self.d * self.ug + self.lp * self.ub * S,
            b=self.ub,
            h=self.uh,
        )

    def inverse_transpose_generators(self) -> Order1QSGenerators:
        """Direct QS generators of ``P^{-T}`` (``O(m)``)."""
        A_g, A_b, A_h = self.lq, self.la - self.lq * self.lp, -self.lp
        B_p, B_a, B_q = self.uh, self.ub - self.uh * self.ug, -self.ug
        Dinv = self.d.reciprocal()
        trans_a = (A_b * B_a).flip(-1)[..., :-1]
        trans_b = (A_h * Dinv * B_p).flip(-1)[..., :-1]
        T = Semiseparable.scan_from_zero(trans_a, trans_b).flip(-1)
        return Order1QSGenerators(
            p=Dinv * B_p + A_g * B_a * T,
            a=B_a,
            q=B_q,
            d=Dinv + A_g * B_q * T,
            g=A_g,
            b=A_b,
            h=A_h * Dinv + A_b * B_q * T,
        )

    def to_dense(self) -> torch.Tensor:
        """Materialize ``P`` (``O(m^2)``, debug only)."""
        return self.direct_generators().to_dense()

