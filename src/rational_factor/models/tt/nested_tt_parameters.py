from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

try:
    from .nested_tt import (
        NestedTTMatrix,
        NestedTTMatrixSpec,
        NestedTTVector,
        NestedTTVectorSpec,
    )
except ImportError:  # pragma: no cover - allows standalone tests
    from rational_factor.models.tt.nested_tt import (
        NestedTTMatrix,
        NestedTTMatrixSpec,
        NestedTTVector,
        NestedTTVectorSpec,
    )


LeafFactory = Callable[[tuple[int, ...]], Any]


def _resolve_hierarchy_ranks(depth: int, ranks: int | Sequence[int]) -> tuple[int, ...]:
    depth = int(depth)
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if isinstance(ranks, int):
        if ranks < 1:
            raise ValueError("rank must be positive")
        return (int(ranks),) * depth
    values = tuple(int(r) for r in ranks)
    if any(r < 1 for r in values):
        raise ValueError("all ranks must be positive")
    if len(values) == depth:
        return values
    # Backward-compatible shorthand from the previous implementation.
    if len(values) == depth - 1:
        return (1,) + values
    raise ValueError(f"expected {depth} hierarchy ranks, got {values}")


def _default_leaf_factory(shape: tuple[int, ...]) -> torch.nn.Parameter:
    value = 0.05 * torch.randn(*shape)
    return torch.nn.Parameter(value)


def _value(p: Any) -> torch.Tensor:
    if isinstance(p, torch.Tensor):
        return p
    if callable(p):
        return torch.as_tensor(p())
    return torch.as_tensor(p)


def _is_trainable(p: Any) -> bool:
    if hasattr(p, "is_trainable"):
        return bool(p.is_trainable())
    if isinstance(p, torch.Tensor):
        return bool(p.requires_grad)
    if isinstance(p, torch.nn.Module):
        return any(q.requires_grad for q in p.parameters())
    return bool(_value(p).requires_grad)


def _parameter_modules(p: Any) -> list[torch.nn.Module]:
    if hasattr(p, "parameter_modules"):
        return list(p.parameter_modules())
    if isinstance(p, torch.nn.Module):
        return [p]
    return []




class FixedNestedTTVectorParameters:
    """Parameter-style wrapper around an already constructed NestedTTVector."""

    def __init__(self, value: NestedTTVector) -> None:
        if not isinstance(value, NestedTTVector):
            raise TypeError("value must be a NestedTTVector")
        self._value = value

    @property
    def modes(self) -> tuple[int, ...]:
        return self._value.modes

    @property
    def depth(self) -> int:
        return self._value.depth

    @property
    def ranks(self) -> tuple[int, ...]:
        return self._value.ranks

    @property
    def separation_rank(self) -> int:
        return int(self._value.spec.separation_rank) if self._value.spec is not None else 1

    def __call__(self) -> NestedTTVector:
        return self._value

    def is_trainable(self) -> bool:
        return False

    def parameter_modules(self) -> list[torch.nn.Module]:
        return []

    def parameters(self) -> tuple[torch.Tensor, ...]:
        return ()

    def is_module(self) -> bool:
        return False


class NestedTTVectorParameters:
    """Parameters for :class:`NestedTTVector` terminal ``X/C`` factors."""

    def __init__(self, spec: NestedTTVectorSpec, leaves: Sequence[Any]) -> None:
        self._spec = spec
        self._leaves = tuple(leaves)
        expected = NestedTTVector.leaf_shapes(spec)
        if len(self._leaves) != len(expected):
            raise ValueError(f"expected {len(expected)} leaf parameters, got {len(self._leaves)}")
        for k, (p, shape) in enumerate(zip(self._leaves, expected)):
            if tuple(_value(p).shape) != tuple(shape):
                raise ValueError(
                    f"leaf parameter {k} has shape {tuple(_value(p).shape)}, expected {shape}"
                )

    @classmethod
    def from_leaves(
        cls, spec: NestedTTVectorSpec, leaves: Sequence[Any]
    ) -> "NestedTTVectorParameters":
        return cls(spec, leaves)

    @classmethod
    def from_core_spec(
        cls,
        modes: Sequence[int],
        *,
        depth: int,
        ranks: int | Sequence[int] = 1,
        separation_rank: int = 1,
        leaf_factory: LeafFactory | None = None,
    ) -> "NestedTTVectorParameters":
        modes = tuple(int(n) for n in modes)
        spec = NestedTTVectorSpec(
            modes=modes,
            depth=int(depth),
            ranks=_resolve_hierarchy_ranks(depth, ranks),
            separation_rank=int(separation_rank),
        )
        factory = _default_leaf_factory if leaf_factory is None else leaf_factory
        return cls(spec, [factory(shape) for shape in NestedTTVector.leaf_shapes(spec)])

    @property
    def spec(self) -> NestedTTVectorSpec:
        return self._spec

    @property
    def modes(self) -> tuple[int, ...]:
        return self._spec.modes

    @property
    def depth(self) -> int:
        return self._spec.depth

    @property
    def ranks(self) -> tuple[int, ...]:
        return self._spec.ranks

    @property
    def separation_rank(self) -> int:
        return self._spec.separation_rank

    @property
    def leaves(self) -> tuple[Any, ...]:
        return self._leaves

    @property
    def leaf_shapes(self) -> tuple[tuple[int, ...], ...]:
        return NestedTTVector.leaf_shapes(self._spec)

    def _leaf_values(self) -> tuple[torch.Tensor, ...]:
        return tuple(_value(p) for p in self._leaves)

    def __call__(self) -> NestedTTVector:
        return NestedTTVector(self._spec, self._leaf_values())

    def is_trainable(self) -> bool:
        return any(_is_trainable(p) for p in self._leaves)

    def parameter_modules(self) -> list[torch.nn.Module]:
        return [m for p in self._leaves for m in _parameter_modules(p)]

    def parameters(self) -> tuple[torch.Tensor, ...]:
        return tuple(x for x in self._leaf_values() if x.requires_grad)

    def is_module(self) -> bool:
        return False


class NestedTTMatrixParameters:
    """Parameters for :class:`NestedTTMatrix` terminal ``L/R`` factors."""

    def __init__(self, spec: NestedTTMatrixSpec, leaves: Sequence[Any]) -> None:
        self._spec = spec
        self._leaves = tuple(leaves)
        expected = NestedTTMatrix.leaf_shapes(spec)
        if len(self._leaves) != len(expected):
            raise ValueError(f"expected {len(expected)} leaf parameters, got {len(self._leaves)}")
        for k, (p, shape) in enumerate(zip(self._leaves, expected)):
            value = _value(p)
            if tuple(value.shape) != tuple(shape):
                raise ValueError(
                    f"leaf parameter {k} has shape {tuple(value.shape)}, expected {shape}"
                )

    @classmethod
    def from_leaves(
        cls, spec: NestedTTMatrixSpec, leaves: Sequence[Any]
    ) -> "NestedTTMatrixParameters":
        return cls(spec, leaves)

    @classmethod
    def from_core_spec(
        cls,
        row_modes: Sequence[int],
        *,
        depth: int,
        ranks: int | Sequence[int] = 1,
        separation_rank: int = 1,
        col_modes: Sequence[int] | None = None,
        leaf_factory: LeafFactory | None = None,
    ) -> "NestedTTMatrixParameters":
        row_modes = tuple(int(n) for n in row_modes)
        col_modes = row_modes if col_modes is None else tuple(int(n) for n in col_modes)
        spec = NestedTTMatrixSpec(
            row_modes=row_modes,
            col_modes=col_modes,
            depth=int(depth),
            ranks=_resolve_hierarchy_ranks(depth, ranks),
            separation_rank=int(separation_rank),
        )
        factory = _default_leaf_factory if leaf_factory is None else leaf_factory
        return cls(spec, [factory(shape) for shape in NestedTTMatrix.leaf_shapes(spec)])

    @property
    def spec(self) -> NestedTTMatrixSpec:
        return self._spec

    @property
    def row_modes(self) -> tuple[int, ...]:
        return self._spec.row_modes

    @property
    def col_modes(self) -> tuple[int, ...]:
        return self._spec.col_modes

    @property
    def depth(self) -> int:
        return self._spec.depth

    @property
    def ranks(self) -> tuple[int, ...]:
        return self._spec.ranks

    @property
    def separation_rank(self) -> int:
        return self._spec.separation_rank

    @property
    def leaves(self) -> tuple[Any, ...]:
        return self._leaves

    @property
    def leaf_shapes(self) -> tuple[tuple[int, ...], ...]:
        return NestedTTMatrix.leaf_shapes(self._spec)

    def _leaf_values(self) -> tuple[torch.Tensor, ...]:
        return tuple(_value(p) for p in self._leaves)

    def __call__(self) -> NestedTTMatrix:
        return NestedTTMatrix(self._spec, self._leaf_values())

    def is_trainable(self) -> bool:
        return any(_is_trainable(p) for p in self._leaves)

    def parameter_modules(self) -> list[torch.nn.Module]:
        return [m for p in self._leaves for m in _parameter_modules(p)]

    def parameters(self) -> tuple[torch.Tensor, ...]:
        return tuple(x for x in self._leaf_values() if x.requires_grad)

    def is_module(self) -> bool:
        return False


class RowStochasticNestedTTMatrixParameters(NestedTTMatrixParameters):
    """Nonnegative row-stochastic nested MPO parameters.

    Raw ``L/R`` leaf parameters are logits.  Normalization is applied with the
    recursive telescoping ownership rule: the first child at a nested MPO level
    owns inherited input-sum axes; every child owns its newly introduced output
    mode and outgoing virtual state.  The resulting dense matrix therefore
    satisfies ``B >= 0`` and ``B @ 1 = 1`` without evaluating its entries in
    normal use.
    """

    def __call__(self) -> NestedTTMatrix:
        return NestedTTMatrix(
            self._spec,
            self._leaf_values(),
            normalize_leaves=True,
        )
