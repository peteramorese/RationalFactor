from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Sequence

import torch


class Vector(ABC):
    """Abstract structured vector.

    Subclasses that support elementwise division should expose
    ``numerical_tolerance`` and apply it to the denominator inside division
    (never by forming ``q + tol`` at call sites, which can raise TT rank).
    """

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

    @property
    def numerical_tolerance(self) -> float:
        return getattr(self, "_numerical_tolerance", 1e-20)

    @abstractmethod
    def to_dense(self) -> torch.Tensor: ...

    def with_numerical_tolerance(self, numerical_tolerance: float) -> "Vector":
        """Return a view/copy of this vector using ``numerical_tolerance`` for division."""
        raise NotImplementedError

    def __mul__(self, other: torch.Tensor) -> torch.Tensor:
        return self.to_dense() * other

    def __rmul__(self, other: torch.Tensor) -> torch.Tensor:
        return other * self.to_dense()

    def __truediv__(self, other: torch.Tensor) -> torch.Tensor:
        other_vals = other.to_dense() if hasattr(other, "to_dense") else torch.as_tensor(other)
        tol = float(
            getattr(other, "numerical_tolerance", self.numerical_tolerance)
        )
        return self.to_dense() / (other_vals + tol)

    def __rtruediv__(self, other: torch.Tensor) -> torch.Tensor:
        other_vals = torch.as_tensor(other)
        return other_vals / (self.to_dense() + self.numerical_tolerance)

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
    def __init__(
        self,
        values: torch.Tensor,
        numerical_tolerance: float = 1e-20,
    ):
        values = torch.as_tensor(values)
        if values.dim() < 1:
            raise ValueError(
                f"dense vector must have shape (..., n), got {tuple(values.shape)}"
            )
        self._values = values
        self._numerical_tolerance = float(numerical_tolerance)

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

    @property
    def numerical_tolerance(self) -> float:
        return self._numerical_tolerance

    def with_numerical_tolerance(self, numerical_tolerance: float) -> "DenseVector":
        return DenseVector(self._values, numerical_tolerance=numerical_tolerance)

    def to_dense(self) -> torch.Tensor:
        return self._values

    def elementwise_divide(self, other) -> "DenseVector":
        other_vals = other.to_dense() if hasattr(other, "to_dense") else torch.as_tensor(other)
        tol = float(getattr(other, "numerical_tolerance", self.numerical_tolerance))
        return DenseVector(
            self._values / (other_vals + tol),
            numerical_tolerance=self.numerical_tolerance,
        )

    def __truediv__(self, other) -> "DenseVector":
        return self.elementwise_divide(other)

    def __rtruediv__(self, other) -> "DenseVector":
        other_vals = (
            other.to_dense() if hasattr(other, "to_dense") else torch.as_tensor(other)
        )
        return DenseVector(
            other_vals / (self._values + self.numerical_tolerance),
            numerical_tolerance=self.numerical_tolerance,
        )

    def __mul__(self, other: torch.Tensor) -> torch.Tensor:
        if hasattr(other, "to_dense"):
            other = other.to_dense()
        if other.shape != self.shape:
            raise ValueError(f"Shape mismatch: {other.shape} != {self.shape}")
        return other * self._values

    def __rmul__(self, other: torch.Tensor) -> torch.Tensor:
        if hasattr(other, "to_dense"):
            other = other.to_dense()
        if other.shape != self.shape:
            raise ValueError(f"Shape mismatch: {other.shape} != {self.shape}")
        return other * self._values

    def sum(self) -> torch.Tensor:
        return self._values.sum(dim=-1)


def as_vector(obj: torch.Tensor | Vector, numerical_tolerance: float | None = None):
    """Return ``obj`` if it is already a structured vector, otherwise wrap a dense tensor.

    When ``numerical_tolerance`` is given, structured vectors are re-wrapped so
    that subsequent elementwise divisions use that floor on the denominator.
    """
    if isinstance(obj, Vector):
        if numerical_tolerance is not None:
            return obj.with_numerical_tolerance(numerical_tolerance)
        return obj
    # NestedTTVector (and similar) are duck-typed structured vectors.
    if type(obj).__name__ == "NestedTTVector":
        if numerical_tolerance is not None and hasattr(obj, "with_numerical_tolerance"):
            return obj.with_numerical_tolerance(numerical_tolerance)
        return obj
    tol = 1e-20 if numerical_tolerance is None else float(numerical_tolerance)
    return DenseVector(torch.as_tensor(obj), numerical_tolerance=tol)