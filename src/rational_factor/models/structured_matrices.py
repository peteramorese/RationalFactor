"""Structured matrix approximators: banded, low-rank, rank-1-plus-diagonal, and quasiseparable."""

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

    def inverse(self) -> Matrix:
        """Return ``M^{-1}`` as a :class:`Matrix`.

        Default densifies and inverts. Prefer :meth:`inverse_matvec` when only
        applying the inverse to vectors; structured subclasses may keep a
        compact representation of the inverse.
        """
        n, m = self.shape[-2], self.shape[-1]
        if n != m:
            raise ValueError(
                f"inverse requires a square matrix, got shape {tuple(self.shape)}"
            )
        return DenseMatrix(torch.linalg.inv(self.to_dense()))

    def inverse_matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Solve ``A y = x`` for ``y`` (same ``x`` shapes as :meth:`matvec`).

        Default uses a dense linear solve and avoids forming ``A^{-1}``
        explicitly. Structured subclasses override this for cheaper solves.
        """
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        n = self.shape[-1]
        if self.shape[-2] != n:
            raise ValueError(
                f"inverse_matvec requires a square matrix, got shape {tuple(self.shape)}"
            )
        A = self.to_dense()
        batch_ndim = A.dim() - 2
        if x.dim() >= batch_ndim + 2 and x.shape[-2] == n:
            return torch.linalg.solve(A, x)
        if x.shape[-1] == n:
            return torch.linalg.solve(A, x)
        raise ValueError(
            f"x must have shape (..., {n}) or (..., {n}, k), got {tuple(x.shape)}"
        )

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

    def batch_linear_combine(
        self, weights: torch.Tensor, *, batch_dim: int = 0
    ) -> Matrix:
        """Weighted sum along one batch axis.

        ``self`` has shape ``(..., k, ..., n, m)`` with ``k`` at ``batch_dim``
        (any batch axis; the last two dims are the matrix). ``weights`` has shape
        ``(..., k)``. Returns a matrix of shape
        ``(weights_batch..., remaining_batch..., n, m)``, always with at least
        one leading batch axis (``weights`` of shape ``(k,)`` and no remaining
        batch axes yields batch size 1).

        The default path densifies; structured subclasses override this to keep
        their storage format and use batched torch ops on the factors only.
        """
        weights = torch.as_tensor(weights, dtype=self.dtype, device=self.device)
        dim = _normalize_matrix_batch_dim(self.shape, batch_dim)
        k = self.shape[dim]
        if weights.shape[-1] != k:
            raise ValueError(
                f"weights trailing size must be k={k}, got {tuple(weights.shape)}"
            )
        values = self.to_dense().movedim(dim, 0)
        out = torch.tensordot(weights, values, dims=1)
        return DenseMatrix(_ensure_leading_batch(out, trailing=2))


    def expm(self) -> Matrix:
        return DenseMatrix(torch.matrix_exp(self.to_dense()))

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

    def inverse(self) -> DenseMatrix:
        n, m = self._values.shape[-2], self._values.shape[-1]
        if n != m:
            raise ValueError(
                f"inverse requires a square matrix, got shape {tuple(self.shape)}"
            )
        return DenseMatrix(torch.linalg.inv(self._values))

    def inverse_matvec(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        n = self._values.shape[-1]
        if self._values.shape[-2] != n:
            raise ValueError(
                f"inverse_matvec requires a square matrix, got shape {tuple(self.shape)}"
            )
        batch_ndim = self._values.dim() - 2
        if x.dim() >= batch_ndim + 2 and x.shape[-2] == n:
            return torch.linalg.solve(self._values, x)
        if x.shape[-1] == n:
            return torch.linalg.solve(self._values, x)
        raise ValueError(
            f"x must have shape (..., {n}) or (..., {n}, k), got {tuple(x.shape)}"
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

    def batch_linear_combine(
        self, weights: torch.Tensor, *, batch_dim: int = 0
    ) -> DenseMatrix:
        weights = torch.as_tensor(weights, dtype=self.dtype, device=self.device)
        dim = _normalize_matrix_batch_dim(self.shape, batch_dim)
        k = self._values.shape[dim]
        if weights.shape[-1] != k:
            raise ValueError(
                f"weights trailing size must be k={k}, got {tuple(weights.shape)}"
            )
        out = torch.tensordot(weights, self._values.movedim(dim, 0), dims=1)
        return DenseMatrix(_ensure_leading_batch(out, trailing=2))



def as_matrix(obj: torch.Tensor | Matrix) -> Matrix:
    """Return ``obj`` if it is already a ``Matrix``, otherwise wrap a dense tensor."""
    if isinstance(obj, Matrix):
        return obj
    return DenseMatrix(obj)


def _normalize_matrix_batch_dim(shape: torch.Size, batch_dim: int) -> int:
    """Resolve ``batch_dim`` against a matrix shape ``(..., n, m)``."""
    ndim = len(shape)
    if ndim < 3:
        raise ValueError(
            f"batch_linear_combine requires at least one batch axis, got shape {tuple(shape)}"
        )
    dim = batch_dim + ndim if batch_dim < 0 else batch_dim
    if not (0 <= dim < ndim - 2):
        raise ValueError(
            f"batch_dim must index a batch axis of shape {tuple(shape)}, got {batch_dim}"
        )
    return dim


def _ensure_leading_batch(out: torch.Tensor, *, trailing: int) -> torch.Tensor:
    """Keep at least one leading batch axis when a reduction removes all of them."""
    if out.dim() == trailing:
        out = out.unsqueeze(0)
    return out



class Identity(Matrix):
    """Identity matrix of size n × n.
    
    Optionally batched with shape (..., n, n). The batch shape is inferred
    from operations that provide tensor inputs.
    """
    
    def __init__(self, n: int, batch_shape: tuple[int, ...] = (), dtype: torch.dtype | None = None, device: torch.device | None = None):
        self.n = n
        self._batch_shape = batch_shape
        self._dtype = dtype if dtype is not None else torch.float32
        self._device = device if device is not None else torch.device("cpu")

    @property
    def shape(self) -> torch.Size:
        return torch.Size(self._batch_shape + (self.n, self.n))

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def T(self) -> "Identity":
        return Identity(self.n, self._batch_shape, self._dtype, self._device)

    def to_dense(self) -> torch.Tensor:
        eye = torch.eye(self.n, dtype=self._dtype, device=self._device)
        if self._batch_shape:
            eye = eye.expand(*self._batch_shape, self.n, self.n)
        return eye

    def diag(self) -> torch.Tensor:
        ones = torch.ones(*self._batch_shape, self.n, dtype=self._dtype, device=self._device)
        return ones

    def sum(self) -> torch.Tensor:
        return torch.full(self._batch_shape, self.n, dtype=self._dtype, device=self._device)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=self._dtype, device=self._device)

    def mul_diag_left(self, a: torch.Tensor) -> "Diagonal":
        a = torch.as_tensor(a, dtype=self._dtype, device=self._device)
        return Diagonal(a)

    def mul_diag_right(self, a: torch.Tensor) -> "Diagonal":
        a = torch.as_tensor(a, dtype=self._dtype, device=self._device)
        return Diagonal(a)

    def mul_diag(self, a: torch.Tensor, *, side: str = "left") -> "Diagonal":
        if side == "left":
            return self.mul_diag_left(a)
        if side == "right":
            return self.mul_diag_right(a)
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    def scale(self, s: torch.Tensor | float) -> "Diagonal":
        s = torch.as_tensor(s, dtype=self._dtype, device=self._device)
        d = torch.full((*self._batch_shape, self.n), s.item() if s.numel() == 1 else s, dtype=self._dtype, device=self._device)
        return Diagonal(d)

    def inverse(self) -> "Identity":
        return Identity(self.n, self._batch_shape, self._dtype, self._device)

    def inverse_matvec(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=self._dtype, device=self._device)


class Diagonal(Matrix):
    def __init__(self, d: torch.Tensor):
        self.d = torch.as_tensor(d)
        if self.d.dim() < 1:
            raise ValueError(f"diagonal must have shape (..., n), got {tuple(self.d.shape)}")

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
    def T(self) -> Diagonal:
        return Diagonal(self.d)

    def to_dense(self) -> torch.Tensor:
        return torch.diag_embed(self.d)

    def diag(self) -> torch.Tensor:
        return self.d

    def sum(self) -> torch.Tensor:
        return self.d.sum()

    def scale(self, s: torch.Tensor | float) -> Diagonal:
        s = torch.as_tensor(s, dtype=self.dtype, device=self.device)
        return Diagonal(self.d * s)

    def batch_linear_combine(
        self, weights: torch.Tensor, *, batch_dim: int = 0
    ) -> Diagonal:
        weights = torch.as_tensor(weights, dtype=self.dtype, device=self.device)
        dim = _normalize_matrix_batch_dim(self.shape, batch_dim)
        k = self.d.shape[dim]
        if weights.shape[-1] != k:
            raise ValueError(
                f"weights trailing size must be k={k}, got {tuple(weights.shape)}"
            )
        out = torch.tensordot(weights, self.d.movedim(dim, 0), dims=1)
        return Diagonal(_ensure_leading_batch(out, trailing=1))


    def mul_diag_left(self, a: torch.Tensor) -> Diagonal:
        a = torch.as_tensor(a, dtype=self.dtype, device=self.device)
        return Diagonal(a * self.d)

    def mul_diag_right(self, a: torch.Tensor) -> Diagonal:
        a = torch.as_tensor(a, dtype=self.dtype, device=self.device)
        return Diagonal(self.d * a)

    def mul_diag(self, a: torch.Tensor, *, side: str = "left") -> Diagonal:
        if side == "left":
            return self.mul_diag_left(a)
        if side == "right":
            return self.mul_diag_right(a)
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    def inverse(self) -> Diagonal:
        return Diagonal(self.d.reciprocal())

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.d * x

    def inverse_matvec(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        n = self.n
        batch_ndim = self.d.dim() - 1
        if x.dim() >= batch_ndim + 2 and x.shape[-2] == n:
            return x / self.d.unsqueeze(-1)
        if x.shape[-1] == n:
            return x / self.d
        raise ValueError(
            f"x must have shape (..., {n}) or (..., {n}, k), got {tuple(x.shape)}"
        )


def _banded_half_bandwidths(offsets: torch.Tensor) -> tuple[int, int]:
    """LAPACK ``(kl, ku)``: subdiagonals (``row-col > 0``), superdiagonals (``row-col < 0``)."""
    offs = offsets.tolist()
    if not offs:
        return 0, 0
    return max(0, max(offs)), max(0, -min(offs))


def _pack_banded_ab(offsets: torch.Tensor, data: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Pack diagonal storage into LAPACK-style band form ``(..., kl+ku+1, n)``.

    Entry ``A[..., i, j]`` with ``i - j = off`` lives at ``ab[..., ku + off, j]``.
    """
    kl, ku = _banded_half_bandwidths(offsets)
    n = data.shape[-1]
    ab = data.new_zeros(data.shape[:-2] + (kl + ku + 1, n))
    for r, off in enumerate(offsets.tolist()):
        off = int(off)
        if off > kl or off < -ku:
            continue
        if off >= 0:
            cols = torch.arange(0, n - off, device=data.device)
        else:
            cols = torch.arange(-off, n, device=data.device)
        ab[..., ku + off, cols] = data[..., r, cols]
    return ab, kl, ku


def _solve_banded_ab(
    ab: torch.Tensor,
    kl: int,
    ku: int,
    b: torch.Tensor,
) -> torch.Tensor:
    """Solve a banded system stored in ``ab`` via Gaussian elimination (no pivoting).

    Complexity is ``O(n · kl · ku)`` per batch element. Suitable when ``A`` admits
    an LU factorization without pivoting (e.g. diagonally dominant / SPD).
    """
    ab = ab.clone()
    x = b.clone()
    n = ab.shape[-1]
    multi_rhs = x.dim() == ab.dim() and x.shape[-2] == n

    for j in range(n - 1):
        piv = ab[..., ku, j]
        for i in range(1, min(kl, n - 1 - j) + 1):
            mult = ab[..., ku + i, j] / piv
            ab[..., ku + i, j] = mult
            for c in range(1, min(ku, n - 1 - j) + 1):
                ab[..., ku + i - c, j + c] = (
                    ab[..., ku + i - c, j + c] - mult * ab[..., ku - c, j + c]
                )
            if multi_rhs:
                x[..., j + i, :] = x[..., j + i, :] - mult.unsqueeze(-1) * x[..., j, :]
            else:
                x[..., j + i] = x[..., j + i] - mult * x[..., j]

    for j in range(n - 1, -1, -1):
        if multi_rhs:
            s = x[..., j, :].clone()
        else:
            s = x[..., j].clone()
        for c in range(1, min(ku, n - 1 - j) + 1):
            if multi_rhs:
                s = s - ab[..., ku - c, j + c].unsqueeze(-1) * x[..., j + c, :]
            else:
                s = s - ab[..., ku - c, j + c] * x[..., j + c]
        diag = ab[..., ku, j]
        if multi_rhs:
            x[..., j, :] = s / diag.unsqueeze(-1)
        else:
            x[..., j] = s / diag
    return x


class Banded(Matrix):
    """Square banded matrix in column-oriented diagonal storage.

    ``data[..., r, j]`` stores ``A[..., j + offsets[r], j]`` when that row is
    in range. Storage is ``O(n * number_of_diagonals)``, never ``O(n^2)``.

    - ``offsets``: ``(n_diag,)`` integer row−column offsets
    - ``data``: ``(..., n_diag, n)`` diagonal values
    """

    def __init__(self, offsets: torch.Tensor, data: torch.Tensor):
        offsets = torch.as_tensor(offsets)
        data = torch.as_tensor(data)
        if offsets.dim() != 1:
            raise ValueError(f"offsets must have shape (n_diag,), got {tuple(offsets.shape)}")
        if data.dim() < 2:
            raise ValueError(f"data must have shape (..., n_diag, n), got {tuple(data.shape)}")
        if data.shape[-2] != offsets.shape[0]:
            raise ValueError(
                f"data n_diag={data.shape[-2]} must match offsets length {offsets.shape[0]}"
            )
        self.offsets = offsets.to(dtype=torch.long, device=data.device)
        self.data = data

    @property
    def n(self) -> int:
        return self.data.shape[-1]

    @property
    def n_diag(self) -> int:
        return self.offsets.shape[0]

    @property
    def batch_shape(self) -> torch.Size:
        return self.data.shape[:-2]

    @property
    def shape(self) -> torch.Size:
        return self.batch_shape + torch.Size([self.n, self.n])

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def T(self) -> Banded:
        """Transpose: negate offsets and shift each stored diagonal."""
        n = self.n
        new_data = torch.zeros_like(self.data)
        for r, off in enumerate(self.offsets.tolist()):
            off = int(off)
            if off >= 0:
                if off < n:
                    new_data[..., r, off:] = self.data[..., r, : n - off]
            elif -off < n:
                new_data[..., r, : n + off] = self.data[..., r, -off:]
        return Banded(-self.offsets, new_data)

    def to_dense(self) -> torch.Tensor:
        n = self.n
        out = self.data.new_zeros(self.batch_shape + (n, n))
        for r, off in enumerate(self.offsets.tolist()):
            off = int(off)
            if off >= 0:
                cols = torch.arange(0, n - off, device=self.device)
            else:
                cols = torch.arange(-off, n, device=self.device)
            rows = cols + off
            out[..., rows, cols] = self.data[..., r, cols]
        return out

    def diag(self) -> torch.Tensor:
        zero = (self.offsets == 0).nonzero(as_tuple=False)
        if zero.numel() == 0:
            return self.data.new_zeros(self.batch_shape + (self.n,))
        return self.data[..., int(zero[0]), :]

    def sum(self) -> torch.Tensor:
        total = None
        n = self.n
        for r, off in enumerate(self.offsets.tolist()):
            off = int(off)
            if off >= 0:
                part = self.data[..., r, : n - off]
            else:
                part = self.data[..., r, -off:]
            total = part.sum() if total is None else total + part.sum()
        if total is None:
            return self.data.new_zeros(())
        return total

    def scale(self, s: torch.Tensor | float) -> Banded:
        s = torch.as_tensor(s, dtype=self.dtype, device=self.device)
        return Banded(self.offsets, self.data * s)

    def batch_linear_combine(
        self, weights: torch.Tensor, *, batch_dim: int = 0
    ) -> Banded:
        weights = torch.as_tensor(weights, dtype=self.dtype, device=self.device)
        dim = _normalize_matrix_batch_dim(self.shape, batch_dim)
        k = self.data.shape[dim]
        if weights.shape[-1] != k:
            raise ValueError(
                f"weights trailing size must be k={k}, got {tuple(weights.shape)}"
            )
        out = torch.tensordot(weights, self.data.movedim(dim, 0), dims=1)
        return Banded(self.offsets, _ensure_leading_batch(out, trailing=2))


    def mul_diag_left(self, a: torch.Tensor) -> Banded:
        """Left-multiply by ``diag(a)``: ``(diag(a) A)[i, j] = a[i] A[i, j]``."""
        a = torch.as_tensor(a, dtype=self.dtype, device=self.device)
        if a.shape[-1] != self.n:
            raise ValueError(
                f"diagonal must have trailing size n={self.n}, got shape {tuple(a.shape)}"
            )
        n = self.n
        new_data = torch.zeros_like(self.data)
        for r, off in enumerate(self.offsets.tolist()):
            off = int(off)
            if off >= 0:
                cols = torch.arange(0, n - off, device=self.device)
            else:
                cols = torch.arange(-off, n, device=self.device)
            rows = cols + off
            new_data[..., r, cols] = a[..., rows] * self.data[..., r, cols]
        return Banded(self.offsets, new_data)

    def mul_diag_right(self, a: torch.Tensor) -> Banded:
        """Right-multiply by ``diag(a)``: ``(A diag(a))[i, j] = A[i, j] a[j]``."""
        a = torch.as_tensor(a, dtype=self.dtype, device=self.device)
        if a.shape[-1] != self.n:
            raise ValueError(
                f"diagonal must have trailing size n={self.n}, got shape {tuple(a.shape)}"
            )
        return Banded(self.offsets, self.data * a.unsqueeze(-2))

    def mul_diag(self, a: torch.Tensor, *, side: str = "left") -> Banded:
        if side == "left":
            return self.mul_diag_left(a)
        if side == "right":
            return self.mul_diag_right(a)
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``A @ x`` in ``O(n * n_diag)`` time.

        Args:
            x: ``(..., n)`` or ``(..., n, k)``. Leading dims broadcast with
                ``batch_shape``.
        """
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        n = self.n
        batch_ndim = len(self.batch_shape)
        multi_rhs = x.dim() >= batch_ndim + 2 and x.shape[-2] == n
        if not multi_rhs and x.shape[-1] != n:
            raise ValueError(
                f"x must have shape (..., {n}) or (..., {n}, k), got {tuple(x.shape)}"
            )

        out = torch.zeros_like(x)
        for r, off in enumerate(self.offsets.tolist()):
            off = int(off)
            if off >= 0:
                cols = torch.arange(0, n - off, device=self.device)
            else:
                cols = torch.arange(-off, n, device=self.device)
            rows = cols + off
            diag = self.data[..., r, cols]
            if multi_rhs:
                src = x[..., cols, :] * diag.unsqueeze(-1)
                out.index_add_(-2, rows, src)
            else:
                src = x[..., cols] * diag
                out.index_add_(-1, rows, src)
        return out

    def inverse(self) -> DenseMatrix:
        """Dense inverse; inversion does not preserve band structure."""
        return DenseMatrix(torch.linalg.inv(self.to_dense()))

    def inverse_matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Solve ``A y = x`` in ``O(n · kl · ku)`` via banded LU (no pivoting).

        ``x`` has shape ``(..., n)`` or ``(..., n, k)``. Prefer this over
        :meth:`inverse` when only matrix–vector products with ``A^{-1}`` are needed.
        """
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        n = self.n
        batch_ndim = len(self.batch_shape)
        multi_rhs = x.dim() >= batch_ndim + 2 and x.shape[-2] == n
        if not multi_rhs and x.shape[-1] != n:
            raise ValueError(
                f"x must have shape (..., {n}) or (..., {n}, k), got {tuple(x.shape)}"
            )
        ab, kl, ku = _pack_banded_ab(self.offsets, self.data)
        return _solve_banded_ab(ab, kl, ku, x)


class LowRankFactorization(Matrix):
    """Batched low-rank matrix ``M = U Vᵀ``.

    - ``U``: ``(..., n, r)`` left factor
    - ``V``: ``(..., m, r)`` right factor

    Storage is ``O((n + m) r)``. Matrix–vector products use
    ``M x = U (Vᵀ x)`` and cost ``O((n + m) r)`` per RHS, never forming the
    dense ``n × m`` matrix.
    """

    def __init__(self, U: torch.Tensor, V: torch.Tensor):
        U = torch.as_tensor(U)
        V = torch.as_tensor(V)
        if U.dim() < 2 or V.dim() < 2:
            raise ValueError(
                f"U and V must have shape (..., n, r) and (..., m, r), "
                f"got {tuple(U.shape)} and {tuple(V.shape)}"
            )
        if U.shape[-1] != V.shape[-1]:
            raise ValueError(
                f"U and V must share rank r as trailing size, got "
                f"U.shape[-1]={U.shape[-1]} and V.shape[-1]={V.shape[-1]}"
            )
        if U.shape[:-2] != V.shape[:-2]:
            raise ValueError(
                f"U and V must share batch shape, got {tuple(U.shape[:-2])} and "
                f"{tuple(V.shape[:-2])}"
            )
        if U.dtype != V.dtype:
            raise ValueError("U and V must share the same dtype")
        if U.device != V.device:
            raise ValueError("U and V must share the same device")
        self.U = U
        self.V = V

    @property
    def n(self) -> int:
        """Number of rows."""
        return self.U.shape[-2]

    @property
    def m(self) -> int:
        """Number of columns."""
        return self.V.shape[-2]

    @property
    def r(self) -> int:
        """Factor rank (inner dimension)."""
        return self.U.shape[-1]

    @property
    def batch_shape(self) -> torch.Size:
        return self.U.shape[:-2]

    @property
    def shape(self) -> torch.Size:
        return self.batch_shape + torch.Size([self.n, self.m])

    @property
    def dtype(self) -> torch.dtype:
        return self.U.dtype

    @property
    def device(self) -> torch.device:
        return self.U.device

    @property
    def T(self) -> "LowRankFactorization":
        """Transpose: ``(U Vᵀ)ᵀ = V Uᵀ``."""
        return LowRankFactorization(self.V, self.U)

    def to(self, *args, **kwargs) -> "LowRankFactorization":
        return LowRankFactorization(
            self.U.to(*args, **kwargs),
            self.V.to(*args, **kwargs),
        )

    def to_dense(self) -> torch.Tensor:
        """Materialize ``U Vᵀ`` as a dense ``(..., n, m)`` tensor."""
        return self.U @ self.V.transpose(-2, -1)

    def diag(self) -> torch.Tensor:
        """Main diagonal ``(..., min(n, m))`` without forming the dense matrix."""
        k = min(self.n, self.m)
        return (self.U[..., :k, :] * self.V[..., :k, :]).sum(dim=-1)

    def sum(self) -> torch.Tensor:
        """Sum of all entries: ``1ᵀ U Vᵀ 1 = (Uᵀ 1) · (Vᵀ 1)``."""
        return (self.U.sum(dim=-2) * self.V.sum(dim=-2)).sum()

    def scale(self, s: torch.Tensor | float) -> "LowRankFactorization":
        s = torch.as_tensor(s, dtype=self.dtype, device=self.device)
        return LowRankFactorization(self.U * s, self.V)

    def _prepare_row_diag(self, a: torch.Tensor, size: int, name: str) -> torch.Tensor:
        a = torch.as_tensor(a, dtype=self.dtype, device=self.device)
        if a.shape[-1] != size:
            raise ValueError(
                f"{name} must have trailing size {size}, got shape {tuple(a.shape)}"
            )
        try:
            torch.broadcast_shapes(a.shape[:-1], self.batch_shape)
        except RuntimeError as e:
            raise ValueError(
                f"{name} shape {tuple(a.shape)} is not broadcastable with "
                f"batch shape {tuple(self.batch_shape)}"
            ) from e
        return a

    def mul_diag_left(self, a: torch.Tensor) -> "LowRankFactorization":
        """Left-multiply by ``diag(a)``: ``diag(a) U Vᵀ = (a ⊙ rows of U) Vᵀ``."""
        a = self._prepare_row_diag(a, self.n, "left diagonal")
        return LowRankFactorization(a.unsqueeze(-1) * self.U, self.V)

    def mul_diag_right(self, a: torch.Tensor) -> "LowRankFactorization":
        """Right-multiply by ``diag(a)``: ``U Vᵀ diag(a) = U (a ⊙ rows of V)ᵀ``."""
        a = self._prepare_row_diag(a, self.m, "right diagonal")
        return LowRankFactorization(self.U, a.unsqueeze(-1) * self.V)

    def mul_diag(self, a: torch.Tensor, *, side: str = "left") -> "LowRankFactorization":
        if side == "left":
            return self.mul_diag_left(a)
        if side == "right":
            return self.mul_diag_right(a)
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    def batch_linear_combine(
        self, weights: torch.Tensor, *, batch_dim: int = 0
    ) -> "LowRankFactorization":
        """Weighted sum along one batch axis, keeping low-rank form.

        Combines factors at ``batch_dim`` into a single factorization of rank
        ``k r`` by concatenating the scaled factors.
        """
        weights = torch.as_tensor(weights, dtype=self.dtype, device=self.device)
        dim = _normalize_matrix_batch_dim(self.shape, batch_dim)
        k = self.U.shape[dim]
        if weights.shape[-1] != k:
            raise ValueError(
                f"weights trailing size must be k={k}, got {tuple(weights.shape)}"
            )

        U_m = self.U.movedim(dim, 0)  # (k, *batch_rest, n, r)
        V_m = self.V.movedim(dim, 0)  # (k, *batch_rest, m, r)
        # Broadcast weights over the moved factors: (..., k, *batch_rest, n, r)
        expand = (1,) * (U_m.dim() - 1)
        U_scaled = U_m * weights.view(weights.shape + expand)
        V_exp = V_m.expand(weights.shape[:-1] + V_m.shape)
        # (..., k, *batch_rest, n, r) -> (..., *batch_rest, n, k, r) -> flatten rank
        k_axis = weights.dim() - 1
        U_out = U_scaled.moveaxis(k_axis, -2).contiguous().flatten(-2, -1)
        V_out = V_exp.moveaxis(k_axis, -2).contiguous().flatten(-2, -1)
        U_out = _ensure_leading_batch(U_out, trailing=2)
        V_out = _ensure_leading_batch(V_out, trailing=2)
        return LowRankFactorization(U_out, V_out)


    def inverse(self) -> "LowRankFactorization":
        """Inverse when square and full factor rank: ``(U Vᵀ)^{-1} = V^{-T} U^{-T}``.

        Requires ``n = m = r``. Rank-deficient factorizations are singular.
        """
        n, m, r = self.n, self.m, self.r
        if n != m:
            raise ValueError(
                f"inverse requires a square matrix, got shape {tuple(self.shape)}"
            )
        if r != n:
            raise ValueError(
                f"low-rank inverse requires full factor rank r=n={n}, got r={r}"
            )
        U_inv = torch.linalg.inv(self.U)
        V_inv = torch.linalg.inv(self.V)
        return LowRankFactorization(
            V_inv.transpose(-2, -1),
            U_inv.transpose(-2, -1),
        )

    def inverse_matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Solve ``U Vᵀ y = x`` when square and ``r = n``.

        Uses ``y = V^{-T} U^{-1} x`` without forming the inverse explicitly.
        """
        n, m, r = self.n, self.m, self.r
        if n != m:
            raise ValueError(
                f"inverse_matvec requires a square matrix, got shape {tuple(self.shape)}"
            )
        if r != n:
            raise ValueError(
                f"low-rank inverse_matvec requires full factor rank r=n={n}, got r={r}"
            )
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        batch_ndim = self.U.dim() - 2
        VT = self.V.transpose(-2, -1)
        if x.dim() >= batch_ndim + 2 and x.shape[-2] == n:
            z = torch.linalg.solve(self.U, x)
            return torch.linalg.solve(VT, z)
        if x.shape[-1] == n:
            z = torch.linalg.solve(self.U, x.unsqueeze(-1)).squeeze(-1)
            return torch.linalg.solve(VT, z.unsqueeze(-1)).squeeze(-1)
        raise ValueError(
            f"x must have shape (..., {n}) or (..., {n}, k), got {tuple(x.shape)}"
        )

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``M @ x = U (Vᵀ x)`` in ``O((n + m) r)`` time.

        Args:
            x: ``(..., m)`` or ``(..., m, k)``. Leading dims broadcast with
                ``batch_shape``.
        """
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        m = self.m
        batch_ndim = self.U.dim() - 2
        if x.dim() >= batch_ndim + 2 and x.shape[-2] == m:
            # Vᵀ x: (..., r, k), then U @ that: (..., n, k)
            return self.U @ (self.V.transpose(-2, -1) @ x)
        if x.shape[-1] == m:
            # Vᵀ x: (..., r), then U @ that: (..., n)
            return (self.U @ (self.V.transpose(-2, -1) @ x.unsqueeze(-1))).squeeze(-1)
        raise ValueError(
            f"x must have shape (..., {m}) or (..., {m}, k), got {tuple(x.shape)}"
        )


class Rank1PlusDiagonal(Matrix):
    """Batched rank-1-plus-diagonal matrix ``M = diag(d) + u vᵀ``.

    - ``u``, ``v``: ``(..., n)`` vectors
    - ``d``: ``(..., n)`` diagonal entries. Optional: if omitted and
      ``normalization`` is ``None``, the diagonal is ones (``M = I + u vᵀ``).
    - ``normalization``: ``None``, ``'r'``, or ``'c'``. If set, ``d`` is ignored
      and ``u``, ``v`` are unconstrained parameters of a stochastic matrix
      ``diag(1 - u) + u vᵀ`` (row-stochastic) or its transpose (column-stochastic).
      Every matrix in the batch is normalized independently.

    This class represents a single R1PD (with optional outer batch axes). Products
    of several factors along a sequence axis live in :class:`R1PDFactorization`.

    Matrix–vector products use ``Mx = d ⊙ x + u (vᵀ x)`` and cost ``O(n)``
    per batch element, never forming the dense ``n × n`` matrix.
    """

    def __init__(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        d: torch.Tensor | None = None,
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

        if d is None:
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
    def normalization(self) -> str | None:
        return self._normalization

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

    def inverse_matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Solve ``M y = x`` in ``O(n)`` via Sherman–Morrison (no dense inverse)."""
        return self.inverse().matvec(x)

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


class R1PDFactorization(Rank1PlusDiagonal):
    """Product of rank-1-plus-diagonal factors along one tensor axis.

    Stores the same ``d, u, v`` tensors as :class:`Rank1PlusDiagonal`, but with
    an extra sequence axis ``seq_dim`` indexing the factors ``M_0, ..., M_{T-1}``.
    This object represents the product ``M_{T-1} ⋯ M_0``; the sequence axis is
    consumed and therefore does not appear in :attr:`shape`.

    ``matvec`` applies the factors sequentially in ``O(T n)`` without forming
    the dense product.
    """

    def __init__(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        d: torch.Tensor | None = None,
        *,
        seq_dim: int = -2,
    ):
        super().__init__(u, v, d)
        batch_ndim = self.d.dim() - 1
        if batch_ndim < 1:
            raise ValueError(
                "R1PDFactorization requires a sequence axis; "
                f"got factor shape {tuple(self.d.shape)}"
            )
        if not (-batch_ndim <= seq_dim < batch_ndim):
            raise ValueError(
                f"seq_dim must index a batch axis of d/u/v, got seq_dim={seq_dim} "
                f"for shape {tuple(self.d.shape)}"
            )
        self.seq_dim = seq_dim % batch_ndim

    @property
    def n_factors(self) -> int:
        return self.d.shape[self.seq_dim]

    @property
    def batch_shape(self) -> torch.Size:
        shape = list(self.d.shape[:-1])
        del shape[self.seq_dim]
        return torch.Size(shape)

    @property
    def shape(self) -> torch.Size:
        return self.batch_shape + torch.Size([self.n, self.n])

    def _factor(self, index: int) -> Rank1PlusDiagonal:
        return Rank1PlusDiagonal(
            self.u.select(self.seq_dim, index),
            self.v.select(self.seq_dim, index),
            self.d.select(self.seq_dim, index),
        )

    def to(self, *args, **kwargs) -> "R1PDFactorization":
        return R1PDFactorization(
            self.u.to(*args, **kwargs),
            self.v.to(*args, **kwargs),
            self.d.to(*args, **kwargs),
            seq_dim=self.seq_dim,
        )

    def flip(self) -> "R1PDFactorization":
        """Reverse factor order along the sequence axis."""
        return R1PDFactorization(
            self.u.flip(self.seq_dim),
            self.v.flip(self.seq_dim),
            self.d.flip(self.seq_dim),
            seq_dim=self.seq_dim,
        )

    @property
    def T(self) -> "R1PDFactorization":
        """Transpose: reverse factor order and transpose each factor."""
        return R1PDFactorization(
            self.v.flip(self.seq_dim),
            self.u.flip(self.seq_dim),
            self.d.flip(self.seq_dim),
            seq_dim=self.seq_dim,
        )

    def matvec(
        self,
        x: torch.Tensor,
        *,
        reverse: bool = False,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """Apply ``M_t`` sequentially: ``x_{t+1} = M_t x_t``.

        With ``reverse=False`` (default), applies ``M_0``, then ``M_1``, … so the
        product is ``M_{T-1} ⋯ M_0``. With ``reverse=True``, applies ``M_{T-1}``,
        then ``M_{T-2}``, … (same as ``self.flip().matvec(...)``).

        Args:
            x: ``(..., n)`` or ``(..., n, m)``. Leading dims broadcast with
                :attr:`batch_shape` (the factor batch with ``seq_dim`` removed).
            reverse: if True, traverse the sequence axis backward.
            return_trajectory: if True, return all states with the sequence axis
                length ``T + 1`` inserted at ``seq_dim``. Requires ``x`` to match
                :attr:`batch_shape` exactly (no extra data-batch dims).

        Returns:
            Final state, or full trajectory if ``return_trajectory=True``.
        """
        if reverse:
            return self.flip().matvec(x, reverse=False, return_trajectory=return_trajectory)

        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        d = self.d.movedim(self.seq_dim, 0)
        u = self.u.movedim(self.seq_dim, 0)
        v = self.v.movedim(self.seq_dim, 0)
        T_seq = d.shape[0]
        factor_batch = d.shape[1:]  # batch_shape + (n,)

        if return_trajectory:
            if x.shape != factor_batch:
                raise ValueError(
                    f"return_trajectory requires x shape {tuple(factor_batch)}, got {tuple(x.shape)}"
                )
            traj = x.new_empty((T_seq + 1,) + factor_batch)
            traj[0] = x
            for t in range(T_seq):
                vx = (v[t] * x).sum(dim=-1)
                x = d[t] * x + u[t] * vx.unsqueeze(-1)
                traj[t + 1] = x
            return traj.movedim(0, self.seq_dim)

        # Multi-RHS: (..., n, m), possibly with a data batch in front of factor batch.
        if x.dim() >= len(factor_batch) + 1 and x.shape[-2] == self.n:
            for t in range(T_seq):
                vx = (v[t].unsqueeze(-1) * x).sum(dim=-2)
                x = d[t].unsqueeze(-1) * x + u[t].unsqueeze(-1) * vx.unsqueeze(-2)
            return x

        if x.shape[-1] != self.n:
            raise ValueError(
                f"x must have shape (..., {self.n}) or (..., {self.n}, m), got {tuple(x.shape)}"
            )

        for t in range(T_seq):
            vx = (v[t] * x).sum(dim=-1)
            x = d[t] * x + u[t] * vx.unsqueeze(-1)
        return x

    def to_dense(self) -> torch.Tensor:
        eye = torch.eye(self.n, dtype=self.dtype, device=self.device)
        eye = eye.expand(self.batch_shape + (self.n, self.n)).clone()
        return self.matvec(eye)

    def diag(self) -> torch.Tensor:
        return self.to_dense().diagonal(dim1=-2, dim2=-1)

    def sum(self) -> torch.Tensor:
        return self.to_dense().sum()

    def inverse(self) -> "R1PDFactorization":
        """Inverse of the product: reverse order and invert each factor."""
        inv_factors = [
            self._factor(index).inverse()
            for index in range(self.n_factors - 1, -1, -1)
        ]
        return R1PDFactorization(
            torch.stack([f.u for f in inv_factors], dim=self.seq_dim),
            torch.stack([f.v for f in inv_factors], dim=self.seq_dim),
            torch.stack([f.d for f in inv_factors], dim=self.seq_dim),
            seq_dim=self.seq_dim,
        )

    def _replace_factor(self, index: int, factor: Rank1PlusDiagonal) -> "R1PDFactorization":
        u = self.u.clone()
        v = self.v.clone()
        d = self.d.clone()
        selector = [slice(None)] * u.dim()
        selector[self.seq_dim] = index
        selector = tuple(selector)
        u[selector], v[selector], d[selector] = factor.u, factor.v, factor.d
        return R1PDFactorization(u, v, d, seq_dim=self.seq_dim)

    def mul_diag_left(self, a: torch.Tensor) -> "R1PDFactorization":
        last = self.n_factors - 1
        return self._replace_factor(last, self._factor(last).mul_diag_left(a))

    def mul_diag_right(self, a: torch.Tensor) -> "R1PDFactorization":
        return self._replace_factor(0, self._factor(0).mul_diag_right(a))

    def scale(self, s: torch.Tensor | float) -> "R1PDFactorization":
        last = self.n_factors - 1
        return self._replace_factor(last, self._factor(last).scale(s))


class Semiseparable(Matrix):
    """Unit semiseparable matrix of order ``k`` with diagonal transitions.

    Generators ``p, a, q`` always have shape ``(..., n, k)``. Lower
    (``upper=False``):

        M[i, j] = sum_r p[i,r] (prod_{t=j+1}^{i-1} a[t,r]) q[j,r]   i > j
        M[i, i] = 1
        M[i, j] = 0                                                    i < j

    Upper is the same recurrence on reversed indices. Matvec and triangular
    solve cost ``O(k n)`` arithmetic via a length-``n`` affine scan, implemented
    with batched parallel prefix (``O(k n log n)`` torch ops). ``to_dense`` is
    ``O(k n^2)``.
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
        if p.dim() < 2:
            raise ValueError(
                f"p, a, and q must have shape (..., n, k), got {tuple(p.shape)}"
            )
        if p.shape != a.shape or p.shape != q.shape:
            raise ValueError(
                f"p, a, and q must have the same shape, got "
                f"p={tuple(p.shape)}, a={tuple(a.shape)}, q={tuple(q.shape)}"
            )
        self.p, self.a, self.q = p, a, q
        self.upper = upper

    @property
    def n(self) -> int:
        return self.p.shape[-2]

    @property
    def order(self) -> int:
        return self.p.shape[-1]

    @property
    def shape(self) -> torch.Size:
        return self.p.shape[:-2] + torch.Size([self.n, self.n])

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
        """Solve ``M y = x`` (unit triangular, ``O(k n)``)."""
        return self._apply(x, solve=True)

    def inverse_matvec(self, x: torch.Tensor) -> torch.Tensor:
        """``M^{-1} x``; alias of :meth:`solve` for the unit-triangular structure."""
        return self.solve(x)

    def _apply(self, x: torch.Tensor, *, solve: bool) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=self.p.dtype, device=self.p.device)
        p, a, q = self.p, self.a, self.q
        if self.upper:
            x = x.flip(-1)
            p, a, q = p.flip(-2), a.flip(-2), q.flip(-2)
        if x.shape[-1] == 1:
            y = x
        elif solve and self.order > 1:
            # Inverse couples modes: s ← (diag(a) - q pᵀ) s + q x, so the
            # transition is no longer diagonal. A short sequential scan is
            # still O(k n) and avoids a dense k×k prefix scan.
            y = self._apply_solve_sequential(x, p, a, q)
        else:
            a_t = a[..., :-1, :]
            q_t = q[..., :-1, :]
            p_t = p[..., :-1, :]
            b = q_t * x[..., :-1].unsqueeze(-1)
            if solve:
                # Order-1: q (p s) = (q p) s, so the transition stays diagonal.
                a_t = a_t - q_t * p_t
            s = self.scan_modes_from_zero(a_t, b)
            sign = -1.0 if solve else 1.0
            y = x + sign * (p * s).sum(dim=-1)
        return y.flip(-1) if self.upper else y

    @staticmethod
    def _apply_solve_sequential(
        x: torch.Tensor,
        p: torch.Tensor,
        a: torch.Tensor,
        q: torch.Tensor,
    ) -> torch.Tensor:
        """Unit-triangular solve with diagonal generators, ``O(k n)`` sequential."""
        m = x.shape[-1]
        state = x.new_zeros(x.shape[:-1] + (p.shape[-1],))
        cols = []
        for i in range(m):
            yi = x[..., i] - (p[..., i, :] * state).sum(dim=-1)
            cols.append(yi)
            if i < m - 1:
                state = a[..., i, :] * state + q[..., i, :] * yi.unsqueeze(-1)
        return torch.stack(cols, dim=-1)

    @staticmethod
    def scan_from_zero(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """``s[..., 0] = 0``, ``s[..., i+1] = a[..., i] s[..., i] + b[..., i]``.

        ``a, b`` have shape ``(..., n)`` (``n`` transitions). Result is ``(..., n+1)``.
        """
        a, b = torch.broadcast_tensors(a, b)
        zero = b.new_zeros(b.shape[:-1] + (1,))
        if a.shape[-1] == 0:
            return zero
        n, step = a.shape[-1], 1
        while step < n:
            a_l, b_l = a[..., :-step], b[..., :-step]
            a_r, b_r = a[..., step:], b[..., step:]
            b = torch.cat((b[..., :step], a_r * b_l + b_r), dim=-1)
            a = torch.cat((a[..., :step], a_r * a_l), dim=-1)
            step *= 2
        return torch.cat((zero, b), dim=-1)

    @classmethod
    def scan_modes_from_zero(cls, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Mode-wise scan: ``a, b`` are ``(..., n, k)`` → state ``(..., n+1, k)``."""
        a, b = torch.broadcast_tensors(a, b)
        s = cls.scan_from_zero(a.transpose(-1, -2), b.transpose(-1, -2))
        return s.transpose(-1, -2)

    def strict_triangle(self) -> torch.Tensor:
        """Off-diagonal triangle (zeros on and above/below the diagonal)."""
        p, a, q = self.p, self.a, self.q
        if self.upper:
            p, a, q = p.flip(-2), a.flip(-2), q.flip(-2)
        m, k = self.n, self.order
        ones = torch.ones_like(a)
        idx_i = torch.arange(m, device=a.device).view(*([1] * (a.ndim - 2)), m, 1, 1)
        idx_j = torch.arange(m, device=a.device).view(*([1] * (a.ndim - 2)), 1, m, 1)
        factors = torch.where(idx_i > idx_j, a.unsqueeze(-2), ones.unsqueeze(-2))
        scale = a.new_zeros(a.shape[:-2] + (m, m, k))
        scale[..., 1:, :, :] = torch.cumprod(factors, dim=-3)[..., :-1, :, :]
        tri = torch.tril((p.unsqueeze(-2) * scale * q.unsqueeze(-3)).sum(-1), diagonal=-1)
        return tri.flip(-1).flip(-2) if self.upper else tri

    def to_dense(self) -> torch.Tensor:
        ones = torch.ones(self.p.shape[:-1], dtype=self.dtype, device=self.device)
        return torch.diag_embed(ones) + self.strict_triangle()


@dataclass
class QSGenerators:
    """Direct generators of an order-``k`` quasiseparable matrix with diagonal transitions.

        Q[i,j] =
            sum_r p[i,r] * prod_{t=j+1}^{i-1} a[t,r] * q[j,r],   i > j
            d[i],                                                 i = j
            sum_r g[i,r] * prod_{t=i+1}^{j-1} b[t,r] * h[j,r],   i < j

    ``p, a, q, g, b, h`` have shape ``(..., m, k)``; ``d`` has shape ``(..., m)``.
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

    @property
    def order(self) -> int:
        if self.p.shape == self.d.shape:
            return 1
        return self.p.shape[-1]

    def row_minmax(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact per-row min/max over columns."""
        M = self.to_dense()
        return M.amin(dim=-1), M.amax(dim=-1)

    def _mode_gens(self) -> tuple[torch.Tensor, ...]:
        """Return generators with an explicit mode axis ``(..., m, k)``."""
        if self.p.shape == self.d.shape:
            return (
                self.p.unsqueeze(-1),
                self.a.unsqueeze(-1),
                self.q.unsqueeze(-1),
                self.g.unsqueeze(-1),
                self.b.unsqueeze(-1),
                self.h.unsqueeze(-1),
            )
        return self.p, self.a, self.q, self.g, self.b, self.h

    def to_dense(self) -> torch.Tensor:
        """Materialize the full matrix (``O(k m^2)``, debug / row extrema)."""
        p, a, q, g, b, h = self._mode_gens()
        lower = Semiseparable(p, a, q)
        upper = Semiseparable(g, b, h, upper=True)
        return torch.diag_embed(self.d) + lower.strict_triangle() + upper.strict_triangle()


# Backward-compatible alias.
Order1QSGenerators = QSGenerators


class Quasiseparable(Matrix):
    """Invertible order-``k`` quasiseparable ``P = L D U`` with diagonal transitions.

    ``L`` / ``U`` are unit lower / upper :class:`Semiseparable` factors of order
    ``k``; ``D`` is a nonzero diagonal. Generator shapes:

    - ``lower_p, lower_a, lower_q, upper_g, upper_b, upper_h``: ``(..., m, k)``
    - ``diag``: ``(..., m)``

    Matvecs ``P x``, ``P^{-1} x``, ``P^T x``, ``P^{-T} x`` are ``O(k m)``
    (``O(k m log m)`` torch work via parallel scans).
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
        diag = torch.as_tensor(diag)
        gens = [torch.as_tensor(x) for x in (lower_p, lower_a, lower_q, upper_g, upper_b, upper_h)]
        if gens[0].dim() != diag.dim() + 1:
            raise ValueError(
                f"generators must have shape diag.shape + (k,), got diag={tuple(diag.shape)}, "
                f"gen={tuple(gens[0].shape)}"
            )
        expected = diag.shape + (gens[0].shape[-1],)
        if any(x.shape != expected for x in gens):
            raise ValueError(
                "all generators must share shape (..., m, k) matching diag's (..., m)"
            )
        self.L = Semiseparable(gens[0], gens[1], gens[2])
        self.d = diag
        self.U = Semiseparable(gens[3], gens[4], gens[5], upper=True)

    @property
    def n(self) -> int:
        return self.d.shape[-1]

    @property
    def order(self) -> int:
        return self.L.order

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
    def T(self) -> Quasiseparable:
        """``Pᵀ = Uᵀ D Lᵀ``."""
        return Quasiseparable(self.uh, self.ub, self.ug, self.d, self.lq, self.la, self.lp)

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

    def direct_generators(self) -> QSGenerators:
        """Direct QS generators of ``P`` itself (``O(k^2 m)``; ``O(m log m)`` when ``k=1``)."""
        lp, la, lq = self.L.p, self.L.a, self.L.q
        d, ug, ub, uh = self.d, self.U.p, self.U.a, self.U.q
        if self.order == 1:
            # Scalar coupling scan (matches the classical order-1 formulas).
            trans_a = (la * ub)[..., :-1, 0]
            trans_b = (lq * d.unsqueeze(-1) * ug)[..., :-1, 0]
            S = Semiseparable.scan_from_zero(trans_a, trans_b).unsqueeze(-1)
            return QSGenerators(
                p=lp,
                a=la,
                q=lq * d.unsqueeze(-1) + la * uh * S,
                d=d + (lp * uh * S).squeeze(-1),
                g=d.unsqueeze(-1) * ug + lp * ub * S,
                b=ub,
                h=uh,
            )
        k = self.order
        batch = self.d.shape[:-1]
        S = self.d.new_zeros(batch + (k, k))
        qP = torch.empty_like(lq)
        dP = torch.empty_like(d)
        gP = torch.empty_like(ug)
        for i in range(self.n):
            uh_i = uh[..., i, :]
            lp_i = lp[..., i, :]
            dP[..., i] = d[..., i] + torch.einsum("...r,...rs,...s->...", lp_i, S, uh_i)
            qP[..., i, :] = d[..., i].unsqueeze(-1) * lq[..., i, :] + la[..., i, :] * torch.einsum(
                "...rs,...s->...r", S, uh_i
            )
            gP[..., i, :] = d[..., i].unsqueeze(-1) * ug[..., i, :] + ub[..., i, :] * torch.einsum(
                "...rs,...r->...s", S, lp_i
            )
            outer = torch.einsum(
                "...,...r,...s->...rs",
                d[..., i],
                lq[..., i, :],
                ug[..., i, :],
            )
            S = outer + la[..., i, :, None] * S * ub[..., i, None, :]
        return QSGenerators(p=lp, a=la, q=qP, d=dP, g=gP, b=ub, h=uh)

    def inverse_transpose_generators(self) -> QSGenerators:
        """Direct QS generators of ``P^{-T}``.

        For order 1 this is ``O(m log m)``. For ``k > 1``, diagonal-transition
        generators cannot represent ``P^{-T}`` in general (mode coupling in the
        triangular solves), so this raises; use :meth:`invT_matvec` instead.
        """
        if self.order > 1:
            raise NotImplementedError(
                "inverse_transpose_generators requires order 1; "
                "for k>1 use invT_matvec or to_dense()/linalg.inv"
            )
        lp, la, lq = self.L.p, self.L.a, self.L.q
        ug, ub, uh = self.U.p, self.U.a, self.U.q
        A_g, A_b, A_h = lq, la - lq * lp, -lp
        B_p, B_a, B_q = uh, ub - uh * ug, -ug
        Dinv = self.d.reciprocal()
        trans_a = (A_b * B_a).squeeze(-1).flip(-1)[..., :-1]
        trans_b = (A_h * Dinv.unsqueeze(-1) * B_p).squeeze(-1).flip(-1)[..., :-1]
        T = Semiseparable.scan_from_zero(trans_a, trans_b).flip(-1).unsqueeze(-1)
        return QSGenerators(
            p=Dinv.unsqueeze(-1) * B_p + A_g * B_a * T,
            a=B_a,
            q=B_q,
            d=Dinv + (A_g * B_q * T).squeeze(-1),
            g=A_g,
            b=A_b,
            h=A_h * Dinv.unsqueeze(-1) + A_b * B_q * T,
        )

    def to_dense(self) -> torch.Tensor:
        """Materialize ``P`` (``O(k m^2)``, debug only)."""
        return self.direct_generators().to_dense()


class Order1Quasiseparable(Quasiseparable):
    """Invertible order-1 quasiseparable ``P = L D U``.

    Convenience wrapper around :class:`Quasiseparable` with scalar generators of
    shape ``(..., m)``. Matvecs are ``O(m)``.
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
        tensors = tuple(
            torch.as_tensor(x)
            for x in (lower_p, lower_a, lower_q, diag, upper_g, upper_b, upper_h)
        )
        diag = tensors[3]
        if any(x.shape != diag.shape for x in tensors):
            raise ValueError("all generators must share the same shape")
        super().__init__(
            tensors[0].unsqueeze(-1),
            tensors[1].unsqueeze(-1),
            tensors[2].unsqueeze(-1),
            diag,
            tensors[4].unsqueeze(-1),
            tensors[5].unsqueeze(-1),
            tensors[6].unsqueeze(-1),
        )

    @property
    def T(self) -> Order1Quasiseparable:
        """``Pᵀ = Uᵀ D Lᵀ``."""
        return Order1Quasiseparable(
            self.uh.squeeze(-1),
            self.ub.squeeze(-1),
            self.ug.squeeze(-1),
            self.d,
            self.lq.squeeze(-1),
            self.la.squeeze(-1),
            self.lp.squeeze(-1),
        )

    @property
    def lp(self) -> torch.Tensor:
        return self.L.p.squeeze(-1)

    @property
    def la(self) -> torch.Tensor:
        return self.L.a.squeeze(-1)

    @property
    def lq(self) -> torch.Tensor:
        return self.L.q.squeeze(-1)

    @property
    def ug(self) -> torch.Tensor:
        return self.U.p.squeeze(-1)

    @property
    def ub(self) -> torch.Tensor:
        return self.U.a.squeeze(-1)

    @property
    def uh(self) -> torch.Tensor:
        return self.U.q.squeeze(-1)

    def direct_generators(self) -> QSGenerators:
        gen = Quasiseparable.direct_generators(self)
        return QSGenerators(
            p=gen.p.squeeze(-1),
            a=gen.a.squeeze(-1),
            q=gen.q.squeeze(-1),
            d=gen.d,
            g=gen.g.squeeze(-1),
            b=gen.b.squeeze(-1),
            h=gen.h.squeeze(-1),
        )

    def inverse_transpose_generators(self) -> QSGenerators:
        gen = Quasiseparable.inverse_transpose_generators(self)
        return QSGenerators(
            p=gen.p.squeeze(-1),
            a=gen.a.squeeze(-1),
            q=gen.q.squeeze(-1),
            d=gen.d,
            g=gen.g.squeeze(-1),
            b=gen.b.squeeze(-1),
            h=gen.h.squeeze(-1),
        )
