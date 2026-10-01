from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Sequence

import torch


class Vector(ABC):
    @property
    @abstractmethod
    def shape(self) -> torch.Size: ...

    @property
    @abstractmethod
    def n(self) -> int: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @abstractmethod
    def to_dense(self) -> torch.Tensor: ...

    def __mul__(self, other: torch.Tensor) -> torch.Tensor:
        return self.to_dense() * other

    def __rmul__(self, other: torch.Tensor) -> torch.Tensor:
        return other * self.to_dense()

    def __truediv__(self, other: torch.Tensor) -> torch.Tensor:
        return self.to_dense() / other

    def __rtruediv__(self, other: torch.Tensor) -> torch.Tensor:
        return other / self.to_dense()

    def sum(self) -> torch.Tensor:
        return self.to_dense().sum(dim=-1)

class OneVector(Vector):
    def __init__(
        self,
        n: int,
        batch_shape: tuple[int, ...] = (),
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = torch.device("cpu"),
    ):
        self._n = int(n)
        self._batch_shape = tuple(batch_shape)
        self._dtype = dtype
        self._device = torch.device(device)

    @property
    def shape(self) -> torch.Size:
        return torch.Size(self._batch_shape + (self._n,))

    @property
    def n(self) -> int:
        return self._n

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    def to_dense(self) -> torch.Tensor:
        return torch.ones(
            self._batch_shape + (self._n,),
            dtype=self._dtype,
            device=self._device,
        )

    def __mul__(self, other: torch.Tensor) -> torch.Tensor:
        try:
            torch.broadcast_shapes(other.shape, self.shape)
        except RuntimeError as e:
            raise ValueError(f"Shape mismatch: {other.shape} vs {self.shape}") from e
        return other

    def __rmul__(self, other: torch.Tensor) -> torch.Tensor:
        try:
            torch.broadcast_shapes(other.shape, self.shape)
        except RuntimeError as e:
            raise ValueError(f"Shape mismatch: {other.shape} vs {self.shape}") from e
        return other
    
    def sum(self) -> torch.Tensor:
        return self._n


class DenseVector(Vector):
    def __init__(self, values: torch.Tensor):
        values = torch.as_tensor(values)
        if values.dim() < 1:
            raise ValueError(
                f"dense vector must have shape (..., n), got {tuple(values.shape)}"
            )
        self._values = values

    @property
    def shape(self) -> torch.Size:
        return self._values.shape

    @property
    def n(self) -> int:
        return self._values.shape[-1]

    @property
    def dtype(self) -> torch.dtype:
        return self._values.dtype

    @property
    def device(self) -> torch.device:
        return self._values.device

    def to_dense(self) -> torch.Tensor:
        return self._values

    def __mul__(self, other: torch.Tensor) -> torch.Tensor:
        if other.shape != self.shape:
            raise ValueError(f"Shape mismatch: {other.shape} != {self.shape}")
        return other * self._values

    def __rmul__(self, other: torch.Tensor) -> torch.Tensor:
        if other.shape != self.shape:
            raise ValueError(f"Shape mismatch: {other.shape} != {self.shape}")
        return other * self._values

    def sum(self) -> torch.Tensor:
        return self._values.sum(dim=-1)


class TTVector(Vector):
    r"""Vector stored as a Tensor Train, optionally with leading batch dims.

    Cores have shape (*batch_shape, r_{k-1}, n_k, r_k) with boundary ranks ``r_0 = r_d = 1``.

    The flattened vector dimension is n = prod(n_k) and the public dense shape is (*batch_shape, n).
    """

    def __init__(self, cores: Sequence[torch.Tensor]):
        if len(cores) == 0:
            raise ValueError("TTVector requires at least one core")

        cores_t = tuple(torch.as_tensor(c) for c in cores)

        for i, c in enumerate(cores_t):
            if c.dim() < 3:
                raise ValueError(
                    f"TTVector core {i} must have shape "
                    f"(*batch, r_left, n, r_right), got {tuple(c.shape)}"
                )

        # All leading batch dimensions are allowed to broadcast.
        try:
            batch_shape = torch.broadcast_shapes(
                *(c.shape[:-3] for c in cores_t)
            )
        except RuntimeError as exc:
            raise ValueError(
                "TTVector core batch dimensions must be broadcast-compatible, "
                f"got {[tuple(c.shape[:-3]) for c in cores_t]}"
            ) from exc

        cores_t = tuple(
            c.expand(*batch_shape, *c.shape[-3:])
            for c in cores_t
        )

        if cores_t[0].shape[-3] != 1 or cores_t[-1].shape[-1] != 1:
            raise ValueError(
                "TTVector boundary ranks must be 1, got "
                f"r0={cores_t[0].shape[-3]}, "
                f"rd={cores_t[-1].shape[-1]}"
            )

        for i in range(len(cores_t) - 1):
            if cores_t[i].shape[-1] != cores_t[i + 1].shape[-3]:
                raise ValueError(
                    f"TTVector rank mismatch at bond {i}: "
                    f"{cores_t[i].shape[-1]} != "
                    f"{cores_t[i + 1].shape[-3]}"
                )

        self._cores = cores_t
        self._batch_shape = torch.Size(batch_shape)
        self._modes = tuple(int(c.shape[-2]) for c in cores_t)
        self._n = int(math.prod(self._modes))

    @classmethod
    def from_cores(cls, cores: Sequence[torch.Tensor]) -> "TTVector":
        return cls(cores)

    @property
    def cores(self) -> tuple[torch.Tensor, ...]:
        return self._cores

    @property
    def batch_shape(self) -> torch.Size:
        return self._batch_shape

    @property
    def is_batched(self) -> bool:
        return len(self._batch_shape) > 0

    @property
    def modes(self) -> tuple[int, ...]:
        return self._modes

    @property
    def ranks(self) -> tuple[int, ...]:
        return (1,) + tuple(
            int(c.shape[-1])
            for c in self._cores
        )

    @property
    def shape(self) -> torch.Size:
        return self._batch_shape + (self._n,)

    @property
    def n(self) -> int:
        return self._n

    @property
    def dtype(self) -> torch.dtype:
        return self._cores[0].dtype

    @property
    def device(self) -> torch.device:
        return self._cores[0].device

    @property
    def is_rank_one(self) -> bool:
        """Whether all TT ranks are one."""
        return all(r == 1 for r in self.ranks)

    def to_dense(self) -> torch.Tensor:
        r"""Materialize the flattened dense vectors.
        """

        t = self._cores[0].squeeze(-3)

        for core in self._cores[1:]:
            t = torch.einsum(
                "...ir,...rjs->...ijs",
                t,
                core,
            )

            t = t.reshape(
                *self._batch_shape,
                -1,
                core.shape[-1],
            )

        # Final boundary rank is one.
        return t.squeeze(-1)

    def scale(
        self,
        s: torch.Tensor | float,
    ) -> "TTVector":
        r"""Scale the TT.

        ``s`` may be either:

        - a scalar, or
        - a tensor broadcast-compatible with ``batch_shape``.

        A batched scale is absorbed into the first core.
        """
        s = torch.as_tensor(
            s,
            dtype=self.dtype,
            device=self.device,
        )

        try:
            torch.broadcast_shapes(
                self._batch_shape,
                s.shape,
            )
        except RuntimeError as exc:
            raise ValueError(
                f"Scale shape {tuple(s.shape)} is not broadcast-compatible "
                f"with TT batch shape {tuple(self._batch_shape)}"
            ) from exc

        s_core = s.reshape(*s.shape, 1, 1, 1)

        cores = list(self._cores)
        cores[0] = cores[0] * s_core

        return TTVector(cores)

    def batch_select(self, *index) -> "TTVector":
        r"""Index only the leading batch dimensions.
        """
        if len(index) > len(self._batch_shape):
            raise IndexError(
                f"Received {len(index)} batch indices for TT with "
                f"batch shape {tuple(self._batch_shape)}"
            )

        idx = tuple(index) + (
            slice(None),
        ) * (len(self._batch_shape) - len(index))

        return TTVector([
            c[idx]
            for c in self._cores
        ])

    def elementwise_divide(
        self,
        other: "TTVector",
    ) -> "TTVector":
        r"""Return the elementwise quotient ``self / other``.

        Only implemented when both TT vectors have rank one.
        """
        if not isinstance(other, TTVector):
            raise TypeError(
                "TTVector.elementwise_divide requires another TTVector"
            )

        if self.modes != other.modes:
            raise ValueError(
                "TTVector elementwise division requires matching modes, got "
                f"{self.modes} and {other.modes}"
            )

        if not self.is_rank_one or not other.is_rank_one:
            raise ValueError(
                "TTVector elementwise division is only implemented for "
                "rank-1 TT vectors, got "
                f"ranks={self.ranks} and ranks={other.ranks}"
            )

        try:
            torch.broadcast_shapes(
                self.batch_shape,
                other.batch_shape,
            )
        except RuntimeError as exc:
            raise ValueError(
                "TTVector batch shapes are not broadcast-compatible: "
                f"{tuple(self.batch_shape)} and "
                f"{tuple(other.batch_shape)}"
            ) from exc

        return TTVector([
            a / b
            for a, b in zip(self._cores, other._cores)
        ])

    def __truediv__(
        self,
        other: "TTVector",
    ) -> "TTVector":
        return self.elementwise_divide(other)

    def sum(self) -> torch.Tensor:
        r"""Return the sum of all vector entries using TT contraction.
        This never materializes the dense vector.
        """

        acc = self._cores[0].sum(dim=-2)

        for core in self._cores[1:]:
            core_sum = core.sum(dim=-2)
            acc = torch.matmul(acc, core_sum)
        return acc.squeeze(-1).squeeze(-1)

    def clone(self) -> "TTVector":
        return TTVector([
            c.clone()
            for c in self._cores
        ])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"shape={tuple(self.shape)}, "
            f"batch_shape={tuple(self._batch_shape)}, "
            f"modes={self._modes}, "
            f"ranks={self.ranks}, "
            f"dtype={self.dtype}, "
            f"device={self.device})"
        )