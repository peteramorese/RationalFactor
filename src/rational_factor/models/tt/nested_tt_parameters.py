from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from rational_factor.models.parameters import (
    Parameters,
    PositiveParameters,
    RowStochasticMatrixParameters,
    TrainableParameters,
)
from rational_factor.models.tt.nested_tt import (
    NestedTTMatrix,
    NestedTTMatrixSpec,
    NestedTTVector,
    NestedTTVectorSpec,
)


LeafFactory = Callable[[tuple[int, ...]], Any]


def _resolve_hierarchy_ranks(depth: int, ranks: int | Sequence[int]) -> tuple[int, ...]:
    """Resolve bottom-to-top hierarchy ranks.

    The bottom rank is always one.  An integer ``q`` means
    ``(1, q, ..., q)``.  A sequence may contain either all ``depth`` ranks or
    only the non-bottom ``depth - 1`` ranks.
    """
    depth = int(depth)
    if depth < 1:
        raise ValueError("depth must be >= 1")

    if isinstance(ranks, int):
        if ranks < 1:
            raise ValueError(f"rank must be >= 1, got {ranks}")
        return (1,) if depth == 1 else (1,) + (int(ranks),) * (depth - 1)

    values = tuple(int(r) for r in ranks)
    if any(r < 1 for r in values):
        raise ValueError(f"all ranks must be >= 1, got {values}")

    if len(values) == depth:
        if values[0] != 1:
            raise ValueError(
                "full hierarchy rank sequence must start with the terminating "
                f"rank one, got {values}"
            )
        return values

    if len(values) == depth - 1:
        return (1,) + values

    raise ValueError(
        f"ranks must be an int, a length-{depth - 1} non-bottom sequence, "
        f"or a length-{depth} full sequence; got {values}"
    )


def _parameter_modules(parameter: Any) -> list[torch.nn.Module]:
    if hasattr(parameter, "parameter_modules"):
        return list(parameter.parameter_modules())
    if isinstance(parameter, torch.nn.Module):
        return [parameter]
    return []


def _is_trainable(parameter: Any) -> bool:
    if hasattr(parameter, "is_trainable"):
        return bool(parameter.is_trainable())
    if isinstance(parameter, torch.nn.Module):
        return any(p.requires_grad for p in parameter.parameters())
    value = parameter() if callable(parameter) else parameter
    return isinstance(value, torch.Tensor) and bool(value.requires_grad)


def _joint_softmax(x: torch.Tensor, axes: Sequence[int]) -> torch.Tensor:
    """Softmax jointly over an arbitrary collection of tensor axes."""
    axes = tuple(sorted(set(int(a) for a in axes)))
    if not axes:
        raise ValueError("joint softmax requires at least one axis")
    ndim = x.ndim
    if any(a < 0 or a >= ndim for a in axes):
        raise ValueError(f"invalid softmax axes {axes} for shape {tuple(x.shape)}")

    cond_axes = tuple(i for i in range(ndim) if i not in axes)
    perm = cond_axes + axes
    inverse = [0] * ndim
    for new_pos, old_pos in enumerate(perm):
        inverse[old_pos] = new_pos

    xp = x.permute(perm)
    cond_shape = xp.shape[: len(cond_axes)]
    output_shape = xp.shape[len(cond_axes) :]
    xp = xp.reshape(*cond_shape, -1)
    yp = torch.softmax(xp, dim=-1).reshape(*cond_shape, *output_shape)
    return yp.permute(inverse).contiguous()


def _default_positive_leaf_factory(
    *,
    mean: float = 1.0,
    std: float = 1.0,
    epsilon: float = 0.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> LeafFactory:
    def factory(shape: tuple[int, ...]):
        leaf = PositiveParameters.random_init(
            shape=tuple(shape),
            mean=mean,
            std=std,
            epsilon=epsilon,
        )
        if device is not None or dtype is not None:
            leaf = leaf.to(
                device=device if device is not None else leaf._p.device,
                dtype=dtype if dtype is not None else leaf._p.dtype,
            )
        return leaf

    return factory


def _default_trainable_leaf_factory(
    *,
    mean: float = 0.0,
    std: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> LeafFactory:
    def factory(shape: tuple[int, ...]):
        leaf = TrainableParameters.random_init(
            shape=tuple(shape),
            mean=mean,
            std=std,
        )
        if device is not None or dtype is not None:
            leaf = leaf.to(
                device=device if device is not None else leaf._p.device,
                dtype=dtype if dtype is not None else leaf._p.dtype,
            )
        return leaf

    return factory


class NestedTTVectorParameters(Parameters):
    """Leaf parameters for a :class:`NestedTTVector`."""

    def __init__(
        self,
        spec: NestedTTVectorSpec,
        leaves: Sequence[Any],
    ) -> None:
        self._spec = spec
        self._leaves = tuple(leaves)

        expected = NestedTTVector.leaf_shapes(spec)
        if len(self._leaves) != len(expected):
            raise ValueError(
                f"expected {len(expected)} leaf Parameters objects, "
                f"got {len(self._leaves)}"
            )

        values = self._leaf_values()
        for k, (value, shape) in enumerate(zip(values, expected)):
            if tuple(value.shape) != tuple(shape):
                raise ValueError(
                    f"leaf parameter {k} has shape {tuple(value.shape)}, "
                    f"expected {tuple(shape)}"
                )

    @classmethod
    def from_leaves(
        cls,
        spec: NestedTTVectorSpec,
        leaves: Sequence[Any],
    ) -> "NestedTTVectorParameters":
        return cls(spec, leaves)

    @classmethod
    def from_core_spec(
        cls,
        modes: Sequence[int],
        *,
        depth: int,
        ranks: int | Sequence[int] = 1,
        leaf_factory: LeafFactory | None = None,
        mean: float = 1.0,
        std: float = 1.0,
        epsilon: float = 0.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "NestedTTVectorParameters":
        modes = tuple(int(m) for m in modes)
        hierarchy_ranks = _resolve_hierarchy_ranks(depth, ranks)
        spec = NestedTTVectorSpec(
            modes=modes,
            depth=int(depth),
            ranks=hierarchy_ranks,
        )
        if leaf_factory is None:
            leaf_factory = _default_positive_leaf_factory(
                mean=mean,
                std=std,
                epsilon=epsilon,
                device=device,
                dtype=dtype,
            )
        leaves = [leaf_factory(shape) for shape in NestedTTVector.leaf_shapes(spec)]
        return cls(spec, leaves)

    @property
    def spec(self) -> NestedTTVectorSpec:
        return self._spec

    @property
    def leaves(self) -> tuple[Any, ...]:
        return self._leaves

    @property
    def modes(self) -> tuple[int, ...]:
        return self._spec.modes

    @property
    def depth(self) -> int:
        return self._spec.depth

    @property
    def n(self) -> int:
        n = 1
        for mode in self.modes:
            n *= mode
        return n

    @property
    def leaf_shapes(self) -> tuple[tuple[int, ...], ...]:
        return NestedTTVector.leaf_shapes(self._spec)

    def _leaf_values(self) -> tuple[torch.Tensor, ...]:
        return tuple(torch.as_tensor(p()) for p in self._leaves)

    def __call__(self) -> NestedTTVector:
        return NestedTTVector(self._spec, self._leaf_values())

    def is_trainable(self) -> bool:
        return any(_is_trainable(p) for p in self._leaves)

    def parameter_modules(self) -> list[torch.nn.Module]:
        return [m for p in self._leaves for m in _parameter_modules(p)]

    def is_module(self) -> bool:
        return False


class FixedNestedTTVectorParameters(Parameters):
    """Wrap an existing (possibly composed) NestedTTVector as coefficients."""

    def __init__(self, vector: NestedTTVector) -> None:
        if not isinstance(vector, NestedTTVector):
            raise TypeError(
                "FixedNestedTTVectorParameters requires a NestedTTVector, got "
                f"{type(vector).__name__}"
            )
        self._vector = vector

    def __call__(self) -> NestedTTVector:
        return self._vector

    def is_trainable(self) -> bool:
        return False

    def parameter_modules(self) -> list[torch.nn.Module]:
        return []

    def is_module(self) -> bool:
        return False


class NestedTTMatrixParameters(Parameters):
    """Leaf parameters for a :class:`NestedTTMatrix`.

    The class is deliberately duck-typed against the project's ``Parameters``
    interface: each leaf object only needs to be callable and return a tensor.
    """

    def __init__(
        self,
        spec: NestedTTMatrixSpec,
        leaves: Sequence[Any],
    ) -> None:
        self._spec = spec
        self._leaves = tuple(leaves)

        expected = NestedTTMatrix.leaf_shapes(spec)
        if len(self._leaves) != len(expected):
            raise ValueError(
                f"expected {len(expected)} leaf Parameters objects, "
                f"got {len(self._leaves)}"
            )

        # Validate immediately, matching TTMatrixParameters behavior.
        values = self._leaf_values()
        for k, (value, shape) in enumerate(zip(values, expected)):
            if tuple(value.shape) != tuple(shape):
                raise ValueError(
                    f"leaf parameter {k} has shape {tuple(value.shape)}, "
                    f"expected {tuple(shape)}"
                )

    @classmethod
    def from_leaves(
        cls,
        spec: NestedTTMatrixSpec,
        leaves: Sequence[Any],
    ) -> "NestedTTMatrixParameters":
        return cls(spec, leaves)

    @classmethod
    def from_core_spec(
        cls,
        row_modes: Sequence[int],
        *,
        depth: int,
        ranks: int | Sequence[int] = 1,
        col_modes: Sequence[int] | None = None,
        leaf_factory: LeafFactory | None = None,
        mean: float = 1.0,
        std: float = 1.0,
        epsilon: float = 0.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "NestedTTMatrixParameters":
        """Construct leaf Parameters from the nested matrix specification.

        ``leaf_factory(shape)`` should return one project-specific Parameters
        object producing a tensor with exactly ``shape``.  If omitted, positive
        random leaves are used.
        """
        row_modes = tuple(int(m) for m in row_modes)
        col_modes = row_modes if col_modes is None else tuple(int(n) for n in col_modes)
        hierarchy_ranks = _resolve_hierarchy_ranks(depth, ranks)
        spec = NestedTTMatrixSpec(
            row_modes=row_modes,
            col_modes=col_modes,
            depth=int(depth),
            ranks=hierarchy_ranks,
        )
        if leaf_factory is None:
            leaf_factory = _default_positive_leaf_factory(
                mean=mean,
                std=std,
                epsilon=epsilon,
                device=device,
                dtype=dtype,
            )
        leaves = [leaf_factory(shape) for shape in NestedTTMatrix.leaf_shapes(spec)]
        return cls(spec, leaves)

    @property
    def spec(self) -> NestedTTMatrixSpec:
        return self._spec

    @property
    def leaves(self) -> tuple[Any, ...]:
        return self._leaves

    @property
    def leaf_shapes(self) -> tuple[tuple[int, ...], ...]:
        return NestedTTMatrix.leaf_shapes(self._spec)

    def _leaf_values(self) -> tuple[torch.Tensor, ...]:
        return tuple(torch.as_tensor(p()) for p in self._leaves)

    def __call__(self) -> NestedTTMatrix:
        return NestedTTMatrix(self._spec, self._leaf_values())

    def is_trainable(self) -> bool:
        return any(_is_trainable(p) for p in self._leaves)

    def parameter_modules(self) -> list[torch.nn.Module]:
        return [m for p in self._leaves for m in _parameter_modules(p)]

    def is_module(self) -> bool:
        return False


class RowStochasticNestedTTMatrixParameters(
    NestedTTMatrixParameters, RowStochasticMatrixParameters
):
    """Nested MPO parameters that are row stochastic by construction.

    The raw leaf tensors are logits.  Each leaf receives a joint softmax over
    the exact output axes assigned by the recursive stochastic-MPO
    construction.  This guarantees, for every outer core,

        sum_{physical column, outgoing outer bond} core = 1

    for each fixed physical row and incoming outer bond.  Consequently the
    full :class:`NestedTTMatrix` is nonnegative and row stochastic.

    Importantly, not every leaf is normalized over the same axes.  At each
    nested MPO level the first child is responsible for the inherited output
    bank axes, while every child normalizes its own outgoing local bond mode.
    This is what makes the normalization telescope through the hierarchy.
    """

    @classmethod
    def from_leaves(
        cls,
        spec: NestedTTMatrixSpec,
        leaves: Sequence[Any],
    ) -> "RowStochasticNestedTTMatrixParameters":
        return cls(spec, leaves)

    @classmethod
    def from_core_spec(
        cls,
        row_modes: Sequence[int],
        *,
        depth: int,
        ranks: int | Sequence[int] = 1,
        col_modes: Sequence[int] | None = None,
        leaf_factory: LeafFactory | None = None,
        mean: float = 0.0,
        std: float = 1.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        **_kwargs,
    ) -> "RowStochasticNestedTTMatrixParameters":
        if leaf_factory is None:
            leaf_factory = _default_trainable_leaf_factory(
                mean=mean,
                std=std,
                device=device,
                dtype=dtype,
            )
        return super().from_core_spec(
            row_modes,
            depth=depth,
            ranks=ranks,
            col_modes=col_modes,
            leaf_factory=leaf_factory,
        )

    @property
    def stochastic_leaf_specs(
        self,
    ) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        return NestedTTMatrix.row_stochastic_leaf_specs(self._spec)

    def __call__(self) -> NestedTTMatrix:
        raw = self._leaf_values()
        specs = self.stochastic_leaf_specs
        if len(raw) != len(specs):
            raise RuntimeError("internal stochastic leaf-spec mismatch")

        leaves: list[torch.Tensor] = []
        for k, (value, (shape, axes)) in enumerate(zip(raw, specs)):
            if tuple(value.shape) != tuple(shape):
                raise ValueError(
                    f"leaf parameter {k} has shape {tuple(value.shape)}, "
                    f"expected {tuple(shape)}"
                )
            leaves.append(_joint_softmax(value, axes))

        return NestedTTMatrix(self._spec, leaves)
