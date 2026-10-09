from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch


def _split2(n: int) -> tuple[int, int]:
    """Deterministically split n=a*b with factors as balanced as possible."""
    n = int(n)
    if n < 1:
        raise ValueError("dimension must be positive")
    a = int(math.isqrt(n))
    while a > 1 and n % a:
        a -= 1
    return a, n // a


def _joint_softmax(x: torch.Tensor, axes: Sequence[int]) -> torch.Tensor:
    axes = tuple(sorted(set(int(a) for a in axes)))
    if not axes:
        return x
    keep = tuple(i for i in range(x.ndim) if i not in axes)
    perm = keep + axes
    inv = [0] * x.ndim
    for p, a in enumerate(perm):
        inv[a] = p
    xp = x.permute(perm)
    keep_shape = xp.shape[: len(keep)]
    norm_shape = xp.shape[len(keep) :]
    xp = torch.softmax(xp.reshape(*keep_shape, -1), dim=-1)
    return xp.reshape(*keep_shape, *norm_shape).permute(inv)


@dataclass(frozen=True)
class NestedTTVectorLeaf:
    r"""Terminal vector-core bank with fixed separation rank.

    The represented bank is

        g[i] = sum_p X[p, i] kron C[p].

    The latent ``p`` index is never flattened.  Under repeated matrix-vector
    products it becomes an MPS bond connecting successive terminal transfer
    blocks, so the number of represented paths may grow exponentially while
    storage/contraction width remains controlled by ``separation_rank``.
    """

    X: torch.Tensor
    C: torch.Tensor

    def __post_init__(self) -> None:
        X = torch.as_tensor(self.X)
        C = torch.as_tensor(self.C)
        if X.ndim < 3:
            raise ValueError("X must have separation and matrix axes")
        if C.ndim != 3:
            raise ValueError("C must have shape (separation_rank,row,col)")
        if int(X.shape[0]) != int(C.shape[0]):
            raise ValueError("X and C separation ranks must match")
        if int(X.shape[0]) < 1:
            raise ValueError("separation rank must be positive")
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "C", C)

    @property
    def separation_rank(self) -> int:
        return int(self.X.shape[0])

    @property
    def bank_shape(self) -> tuple[int, ...]:
        return tuple(int(n) for n in self.X.shape[1:-2])

    @property
    def row_modes(self) -> tuple[int, ...]:
        return (int(self.X.shape[-2]), int(self.C.shape[-2]))

    @property
    def col_modes(self) -> tuple[int, ...]:
        return (int(self.X.shape[-1]), int(self.C.shape[-1]))

    @property
    def row_dim(self) -> int:
        return math.prod(self.row_modes)

    @property
    def col_dim(self) -> int:
        return math.prod(self.col_modes)

    @staticmethod
    def shapes(
        bank_shape: Sequence[int],
        row_dim: int,
        col_dim: int,
        separation_rank: int = 1,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        q = int(separation_rank)
        if q < 1:
            raise ValueError("separation_rank must be positive")
        xr, cr = _split2(int(row_dim))
        xc, cc = _split2(int(col_dim))
        return (
            (q, *tuple(int(n) for n in bank_shape), xr, xc),
            (q, cr, cc),
        )

    @classmethod
    def from_tensors(
        cls,
        bank_shape: Sequence[int],
        row_dim: int,
        col_dim: int,
        X: torch.Tensor,
        C: torch.Tensor,
        separation_rank: int = 1,
    ) -> "NestedTTVectorLeaf":
        sx, sc = cls.shapes(bank_shape, row_dim, col_dim, separation_rank)
        if tuple(X.shape) != sx or tuple(C.shape) != sc:
            raise ValueError(
                f"vector leaf shapes must be X={sx}, C={sc}; got "
                f"{tuple(X.shape)}, {tuple(C.shape)}"
            )
        return cls(X, C)

    def emit(self, builder, *, bank_labels, row_labels, col_labels) -> None:
        if len(row_labels) != 2 or len(col_labels) != 2:
            raise ValueError("vector leaf expects exactly two tensorized modes")
        p = builder.new_label(self.separation_rank)
        builder.add(
            self.X,
            (p, *tuple(bank_labels), int(row_labels[0]), int(col_labels[0])),
        )
        builder.add(self.C, (p, int(row_labels[1]), int(col_labels[1])))

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.X, self.C)


@dataclass(frozen=True)
class NestedTTMatrixLeaf:
    r"""Terminal matrix-core bank with fixed separation rank.

    The represented bank is

        A[j,i] = sum_q L[q,j] kron R[q,i].

    ``q`` is an explicit latent/MPS bond.  Repeated matvec therefore produces
    transfer blocks ``K[q_t,q_{t-1}]`` rather than expanding ``Q**t`` paths.
    """

    L: torch.Tensor
    R: torch.Tensor
    row_bank_shape: tuple[int, ...]
    col_bank_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        L = torch.as_tensor(self.L)
        R = torch.as_tensor(self.R)
        rb = tuple(int(n) for n in self.row_bank_shape)
        cb = tuple(int(n) for n in self.col_bank_shape)
        if L.ndim != len(rb) + 3 or tuple(L.shape[1:-2]) != rb:
            raise ValueError("L bank axes do not match row_bank_shape")
        if R.ndim != len(cb) + 3 or tuple(R.shape[1:-2]) != cb:
            raise ValueError("R bank axes do not match col_bank_shape")
        if int(L.shape[0]) != int(R.shape[0]):
            raise ValueError("L and R separation ranks must match")
        if int(L.shape[0]) < 1:
            raise ValueError("separation rank must be positive")
        object.__setattr__(self, "L", L)
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "row_bank_shape", rb)
        object.__setattr__(self, "col_bank_shape", cb)

    @property
    def separation_rank(self) -> int:
        return int(self.L.shape[0])

    @property
    def bank_shape(self) -> tuple[int, ...]:
        return self.row_bank_shape + self.col_bank_shape

    @property
    def row_modes(self) -> tuple[int, ...]:
        return (int(self.L.shape[-2]), int(self.R.shape[-2]))

    @property
    def col_modes(self) -> tuple[int, ...]:
        return (int(self.L.shape[-1]), int(self.R.shape[-1]))

    @property
    def row_dim(self) -> int:
        return math.prod(self.row_modes)

    @property
    def col_dim(self) -> int:
        return math.prod(self.col_modes)

    @staticmethod
    def shapes(
        row_bank_shape: Sequence[int],
        col_bank_shape: Sequence[int],
        row_dim: int,
        col_dim: int,
        separation_rank: int = 1,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        q = int(separation_rank)
        if q < 1:
            raise ValueError("separation_rank must be positive")
        lr, rr = _split2(int(row_dim))
        lc, rc = _split2(int(col_dim))
        return (
            (q, *tuple(int(n) for n in row_bank_shape), lr, lc),
            (q, *tuple(int(n) for n in col_bank_shape), rr, rc),
        )

    @classmethod
    def from_tensors(
        cls,
        row_bank_shape: Sequence[int],
        col_bank_shape: Sequence[int],
        row_dim: int,
        col_dim: int,
        L: torch.Tensor,
        R: torch.Tensor,
        separation_rank: int = 1,
    ) -> "NestedTTMatrixLeaf":
        sl, sr = cls.shapes(
            row_bank_shape, col_bank_shape, row_dim, col_dim, separation_rank
        )
        if tuple(L.shape) != sl or tuple(R.shape) != sr:
            raise ValueError(
                f"matrix leaf shapes must be L={sl}, R={sr}; got "
                f"{tuple(L.shape)}, {tuple(R.shape)}"
            )
        return cls(
            L=L,
            R=R,
            row_bank_shape=tuple(int(n) for n in row_bank_shape),
            col_bank_shape=tuple(int(n) for n in col_bank_shape),
        )

    def emit(
        self,
        builder,
        *,
        row_bank_labels: Sequence[int],
        col_bank_labels: Sequence[int],
        row_labels: Sequence[int],
        col_labels: Sequence[int],
    ) -> None:
        if len(row_labels) != 2 or len(col_labels) != 2:
            raise ValueError("matrix leaf expects exactly two tensorized modes")
        q = builder.new_label(self.separation_rank)
        builder.add(
            self.L,
            (q, *tuple(row_bank_labels), int(row_labels[0]), int(col_labels[0])),
        )
        builder.add(
            self.R,
            (q, *tuple(col_bank_labels), int(row_labels[1]), int(col_labels[1])),
        )

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.L, self.R)

    def normalized(
        self,
        *,
        sum_row_bank_axes: Sequence[int] = (),
        sum_col_bank_axes: Sequence[int] = (),
    ) -> "NestedTTMatrixLeaf":
        r"""Return a nonnegative row-stochastic terminal transfer.

        For each fixed ``q``, ``L[q]`` is normalized over the output-side axes
        owned by this leaf.  ``R`` is normalized jointly over ``q`` and the
        input-side axes owned by this leaf.  Hence

            sum_q sum_i (L[q,j] kron R[q,i]) 1 = 1,

        while preserving a nontrivial mixture over the ``q`` components.
        """
        # Axis 0 is q; row/col bank axes begin at 1; matrix row-output axis is
        # immediately after the bank axes.
        l_axes = tuple(1 + int(a) for a in sum_row_bank_axes) + (
            1 + len(self.row_bank_shape),
        )
        r_axes = (0,) + tuple(1 + int(a) for a in sum_col_bank_axes) + (
            1 + len(self.col_bank_shape),
        )
        L = _joint_softmax(self.L, l_axes)
        R = _joint_softmax(self.R, r_axes)
        return NestedTTMatrixLeaf(L, R, self.row_bank_shape, self.col_bank_shape)
