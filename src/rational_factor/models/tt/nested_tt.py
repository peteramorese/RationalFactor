from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import opt_einsum as oe
import torch


# =============================================================================
# Specifications
# =============================================================================


@dataclass(frozen=True)
class NestedTTVectorSpec:
    """Specification of a recursively nested TT vector.

    ``depth`` counts the outer TT itself.  Thus a depth-L vector has ``d**L``
    bottom leaf cores when ``d = len(modes)`` and no sharing is used.

    ``ranks[l]`` is the uniform TT/MPO rank at hierarchy level ``l`` counted
    from the bottom.  The lowest rank must be one.  ``ranks[-1]`` is the rank
    of the public / outer TT.
    """

    modes: tuple[int, ...]
    depth: int
    ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("depth must be >= 1")
        if not self.modes or any(int(n) <= 0 for n in self.modes):
            raise ValueError(f"modes must be positive and non-empty, got {self.modes}")
        if len(self.ranks) != self.depth:
            raise ValueError(
                f"ranks must have length depth={self.depth}, got {self.ranks}"
            )
        if any(int(r) <= 0 for r in self.ranks):
            raise ValueError(f"all ranks must be positive, got {self.ranks}")
        if int(self.ranks[0]) != 1:
            raise ValueError("lowest hierarchy rank must be one: ranks[0] == 1")

    @property
    def d(self) -> int:
        return len(self.modes)


@dataclass(frozen=True)
class NestedTTMatrixSpec:
    """Specification of a recursively nested TT matrix / MPO."""

    row_modes: tuple[int, ...]
    col_modes: tuple[int, ...]
    depth: int
    ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("depth must be >= 1")
        if not self.row_modes:
            raise ValueError("row_modes must be non-empty")
        if len(self.row_modes) != len(self.col_modes):
            raise ValueError("row_modes and col_modes must have equal length")
        if any(int(n) <= 0 for n in self.row_modes + self.col_modes):
            raise ValueError("all physical modes must be positive")
        if len(self.ranks) != self.depth:
            raise ValueError(
                f"ranks must have length depth={self.depth}, got {self.ranks}"
            )
        if any(int(r) <= 0 for r in self.ranks):
            raise ValueError(f"all ranks must be positive, got {self.ranks}")
        if int(self.ranks[0]) != 1:
            raise ValueError("lowest hierarchy rank must be one: ranks[0] == 1")

    @property
    def d(self) -> int:
        return len(self.row_modes)


# =============================================================================
# Helpers
# =============================================================================


def _prime_factors(n: int) -> list[int]:
    out: list[int] = []
    p = 2
    while p * p <= n:
        while n % p == 0:
            out.append(p)
            n //= p
        p += 1
    if n > 1:
        out.append(n)
    return out


def _balanced_factor_modes(n: int, d: int) -> tuple[int, ...]:
    """Factor ``n`` into ``d`` integer modes, approximately balanced."""
    n = int(n)
    d = int(d)
    if n <= 0 or d <= 0:
        raise ValueError("n and d must be positive")
    modes = [1] * d
    for p in sorted(_prime_factors(n), reverse=True):
        j = min(range(d), key=modes.__getitem__)
        modes[j] *= p
    return tuple(modes)


def _uniform_bonds(d: int, rank: int) -> tuple[int, ...]:
    if d == 1:
        return (1, 1)
    return (1,) + (int(rank),) * (d - 1) + (1,)


def _flatten_index(index: Sequence[int], modes: Sequence[int]) -> int:
    out = 0
    for i, n in zip(index, modes):
        out = out * int(n) + int(i)
    return out


# =============================================================================
# Tensor-network builder
# =============================================================================


# torch.einsum only accepts subscripts in [a-zA-Z] (52 letters).  opt_einsum maps
# integer labels beyond that to unicode, which torch then rejects.
_TORCH_EINSUM_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


class _NetworkBuilder:
    """Collect tensors with integer-labelled axes for opt_einsum."""

    def __init__(self) -> None:
        self.operands: list[tuple[torch.Tensor, tuple[int, ...]]] = []
        self._next_label = 0
        self._sizes: dict[int, int] = {}

    def new_label(self, size: int) -> int:
        label = self._next_label
        self._next_label += 1
        self._sizes[label] = int(size)
        return label

    def new_labels(self, sizes: Sequence[int]) -> tuple[int, ...]:
        return tuple(self.new_label(int(s)) for s in sizes)

    def check_labels(self, labels: Sequence[int], sizes: Sequence[int]) -> None:
        labels = tuple(labels)
        sizes = tuple(int(s) for s in sizes)
        if len(labels) != len(sizes):
            raise ValueError(f"label/mode mismatch: {len(labels)} != {len(sizes)}")
        for label, size in zip(labels, sizes):
            known = self._sizes.get(int(label))
            if known is None:
                self._sizes[int(label)] = size
            elif known != size:
                raise ValueError(
                    f"tensor-network edge size mismatch for label {label}: "
                    f"{known} != {size}"
                )

    def add(self, tensor: torch.Tensor, labels: Sequence[int]) -> None:
        labels = tuple(int(x) for x in labels)
        if tensor.ndim != len(labels):
            raise ValueError(
                f"tensor rank {tensor.ndim} does not match {len(labels)} labels"
            )
        self.check_labels(labels, tensor.shape)
        self.operands.append((tensor, labels))

    def contract(
        self,
        output_labels: Sequence[int] = (),
        *,
        optimize: str = "greedy",
    ) -> torch.Tensor:
        output_labels = tuple(int(x) for x in output_labels)
        if not self.operands:
            raise ValueError("cannot contract an empty tensor network")

        unique_labels = {lab for _, labs in self.operands for lab in labs}
        unique_labels.update(output_labels)
        if len(unique_labels) <= len(_TORCH_EINSUM_CHARS):
            args: list[object] = []
            for tensor, labels in self.operands:
                args.extend((tensor, list(labels)))
            args.append(list(output_labels))
            return oe.contract(*args, backend="torch", optimize=optimize)

        # More than 52 distinct edges: run the opt_einsum path one step at a
        # time, remapping each step's labels onto ASCII letters torch accepts.
        remaining: list[tuple[torch.Tensor, tuple[int, ...]]] = [
            (tensor, tuple(labels)) for tensor, labels in self.operands
        ]
        path_args: list[object] = []
        for tensor, labels in remaining:
            path_args.extend((tensor, list(labels)))
        path_args.append(list(output_labels))
        path, _info = oe.contract_path(*path_args, optimize=optimize)

        final_needed = set(output_labels)
        for indices in path:
            idxs = tuple(int(i) for i in indices)
            chosen = [remaining[i] for i in idxs]
            for i in sorted(idxs, reverse=True):
                remaining.pop(i)

            needed = set(final_needed)
            for _, labs in remaining:
                needed.update(labs)

            appearance: list[int] = []
            for _, labs in chosen:
                for lab in labs:
                    if lab not in appearance:
                        appearance.append(lab)
            step_out = tuple(lab for lab in appearance if lab in needed)

            remap: dict[int, str] = {}

            def _sym(lab: int) -> str:
                ch = remap.get(lab)
                if ch is None:
                    if len(remap) >= len(_TORCH_EINSUM_CHARS):
                        raise RuntimeError(
                            "tensor-network contraction step needs more than "
                            f"{len(_TORCH_EINSUM_CHARS)} indices; torch.einsum "
                            "cannot express it"
                        )
                    ch = _TORCH_EINSUM_CHARS[len(remap)]
                    remap[lab] = ch
                return ch

            terms = ["".join(_sym(lab) for lab in labs) for _, labs in chosen]
            rhs = "".join(_sym(lab) for lab in step_out)
            equation = ",".join(terms) + "->" + rhs
            remaining.append(
                (torch.einsum(equation, *[tensor for tensor, _ in chosen]), step_out)
            )

        if len(remaining) != 1:
            raise RuntimeError(
                f"contraction path left {len(remaining)} tensors; expected 1"
            )
        tensor, labels = remaining[0]
        if tuple(labels) == output_labels:
            return tensor
        if set(labels) != set(output_labels) or len(labels) != len(output_labels):
            raise RuntimeError(
                f"contraction output labels {labels} != requested {output_labels}"
            )
        if not output_labels:
            return tensor
        return tensor.permute(*[labels.index(lab) for lab in output_labels])


# =============================================================================
# A bank of recursively represented MPOs
# =============================================================================


class _MPOBank:
    """Abstract bank of matrices whose matrix dimensions stay tensorized.

    A bank has logical shape

        (*bank_shape, row_dim, col_dim)

    but ``row_dim`` and ``col_dim`` are represented by ``row_modes`` and
    ``col_modes`` rather than flattened.  ``emit`` adds the exact tensor
    network for this bank without materializing any bank matrix.
    """

    bank_shape: tuple[int, ...]
    row_dim: int
    col_dim: int
    row_modes: tuple[int, ...]
    col_modes: tuple[int, ...]

    @property
    def leaf_count(self) -> int:
        raise NotImplementedError

    def leaves(self) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def emit(
        self,
        builder: _NetworkBuilder,
        *,
        bank_labels: Sequence[int],
        row_labels: Sequence[int],
        col_labels: Sequence[int],
        batch_label: int | None = None,
    ) -> None:
        raise NotImplementedError

    def materialize(self) -> torch.Tensor:
        """Reference/debug path.  ``sum`` never calls this method."""
        builder = _NetworkBuilder()
        bank_labels = builder.new_labels(self.bank_shape)
        row_labels = builder.new_labels(self.row_modes)
        col_labels = builder.new_labels(self.col_modes)
        self.emit(
            builder,
            bank_labels=bank_labels,
            row_labels=row_labels,
            col_labels=col_labels,
            batch_label=None,
        )
        dense_modes = builder.contract((*bank_labels, *row_labels, *col_labels))
        return dense_modes.reshape(*self.bank_shape, self.row_dim, self.col_dim)


class _StaticMPOBank(_MPOBank):
    """Static recursive bank-of-MPOs generated by rank-one leaves.

    At recursion depth zero the represented matrix is 1x1, so the leaf is
    simply a tensor over the accumulated bank indices.  At positive depth the
    matrix is represented by a d-core MPO.  Each MPO core is itself a bank of
    lower-level MPOs representing its matrix-valued virtual slice.
    """

    def __init__(
        self,
        *,
        depth: int,
        d: int,
        ranks: Sequence[int],
        bank_shape: Sequence[int],
        row_dim: int,
        col_dim: int,
        children: Sequence[_MPOBank] | None = None,
        leaf: torch.Tensor | None = None,
    ) -> None:
        self.depth = int(depth)
        self.d = int(d)
        self.ranks = tuple(int(r) for r in ranks)
        self.bank_shape = tuple(int(s) for s in bank_shape)
        self.row_dim = int(row_dim)
        self.col_dim = int(col_dim)

        if self.depth == 0:
            if self.row_dim != 1 or self.col_dim != 1:
                raise ValueError(
                    "depth-zero bank must represent a scalar map, got "
                    f"{self.row_dim}x{self.col_dim}"
                )
            if leaf is None:
                raise ValueError("depth-zero bank requires a leaf tensor")
            leaf = torch.as_tensor(leaf)
            if tuple(leaf.shape) != self.bank_shape:
                raise ValueError(
                    f"leaf shape mismatch: expected {self.bank_shape}, "
                    f"got {tuple(leaf.shape)}"
                )
            self._leaf = leaf
            self._children = ()
            self.row_modes = ()
            self.col_modes = ()
            return

        if len(self.ranks) < self.depth:
            raise ValueError("not enough hierarchy ranks for static MPO bank")

        self.row_modes = _balanced_factor_modes(self.row_dim, self.d)
        self.col_modes = _balanced_factor_modes(self.col_dim, self.d)
        q = self.ranks[self.depth - 1]
        bonds = _uniform_bonds(self.d, q)

        if children is None or len(children) != self.d:
            raise ValueError(f"depth-{self.depth} bank requires {self.d} children")
        self._children = tuple(children)
        self._leaf = None

        for k, child in enumerate(self._children):
            expected_bank = self.bank_shape + (
                self.row_modes[k],
                self.col_modes[k],
            )
            if child.bank_shape != expected_bank:
                raise ValueError(
                    f"child {k} bank shape mismatch: "
                    f"{child.bank_shape} != {expected_bank}"
                )
            if child.row_dim != bonds[k] or child.col_dim != bonds[k + 1]:
                raise ValueError(
                    f"child {k} virtual shape mismatch: "
                    f"{child.row_dim}x{child.col_dim} != "
                    f"{bonds[k]}x{bonds[k + 1]}"
                )

    @property
    def leaf_count(self) -> int:
        if self.depth == 0:
            return 1
        return sum(c.leaf_count for c in self._children)

    def leaves(self) -> tuple[torch.Tensor, ...]:
        if self.depth == 0:
            return (self._leaf,)  # type: ignore[arg-type]
        return tuple(x for c in self._children for x in c.leaves())

    def emit(
        self,
        builder: _NetworkBuilder,
        *,
        bank_labels: Sequence[int],
        row_labels: Sequence[int],
        col_labels: Sequence[int],
        batch_label: int | None = None,
    ) -> None:
        bank_labels = tuple(bank_labels)
        row_labels = tuple(row_labels)
        col_labels = tuple(col_labels)
        builder.check_labels(bank_labels, self.bank_shape)
        builder.check_labels(row_labels, self.row_modes)
        builder.check_labels(col_labels, self.col_modes)

        if self.depth == 0:
            builder.add(self._leaf, bank_labels)  # type: ignore[arg-type]
            return

        # Virtual bonds between the d MPO cores.  These virtual spaces are
        # themselves tensorized according to the child banks.
        left_virtual: tuple[int, ...] = builder.new_labels(
            self._children[0].row_modes
        )
        for k, child in enumerate(self._children):
            if k == self.d - 1:
                right_virtual = builder.new_labels(child.col_modes)
            else:
                right_virtual = builder.new_labels(child.col_modes)
                next_modes = self._children[k + 1].row_modes
                if tuple(child.col_modes) != tuple(next_modes):
                    raise ValueError(
                        "recursive MPO virtual tensorizations do not match: "
                        f"{child.col_modes} != {next_modes}"
                    )
            child.emit(
                builder,
                bank_labels=(*bank_labels, row_labels[k], col_labels[k]),
                row_labels=left_virtual,
                col_labels=right_virtual,
                batch_label=batch_label,
            )
            left_virtual = right_virtual

        # The first/last virtual dimensions are always one.  Their exposed
        # labels therefore have size one and are harmless scalar boundaries.


class _PhysicalScaleBank(_MPOBank):
    """Scale a one-axis bank by a rank-one physical basis factor.

    ``weights`` may be shape ``(n,)`` or batched ``(B, n)``.
    """

    def __init__(self, base: _MPOBank, weights: torch.Tensor) -> None:
        if len(base.bank_shape) != 1:
            raise ValueError("physical scaling expects a one-axis vector-core bank")
        weights = torch.as_tensor(weights)
        n = base.bank_shape[0]
        if weights.ndim == 1:
            if int(weights.shape[0]) != n:
                raise ValueError(
                    f"basis factor must have shape ({n},), got {tuple(weights.shape)}"
                )
            self._batch_size: int | None = None
        elif weights.ndim == 2:
            if int(weights.shape[-1]) != n:
                raise ValueError(
                    f"batched basis factor must have shape (B, {n}), "
                    f"got {tuple(weights.shape)}"
                )
            self._batch_size = int(weights.shape[0])
        else:
            raise ValueError(
                f"basis factor must have shape ({n},) or (B, {n}), "
                f"got {tuple(weights.shape)}"
            )
        self.base = base
        self.weights = weights
        self.bank_shape = base.bank_shape
        self.row_dim = base.row_dim
        self.col_dim = base.col_dim
        self.row_modes = base.row_modes
        self.col_modes = base.col_modes

    @property
    def batch_size(self) -> int | None:
        return self._batch_size

    @property
    def leaf_count(self) -> int:
        return self.base.leaf_count

    def leaves(self) -> tuple[torch.Tensor, ...]:
        return self.base.leaves()

    def emit(
        self,
        builder: _NetworkBuilder,
        *,
        bank_labels: Sequence[int],
        row_labels: Sequence[int],
        col_labels: Sequence[int],
        batch_label: int | None = None,
    ) -> None:
        bank_labels = tuple(bank_labels)
        self.base.emit(
            builder,
            bank_labels=bank_labels,
            row_labels=row_labels,
            col_labels=col_labels,
            batch_label=batch_label,
        )
        w = self.weights
        # Match model dtype/device lazily to the first model leaf.
        leaves = self.base.leaves()
        if leaves:
            w = w.to(dtype=leaves[0].dtype, device=leaves[0].device)
        if w.ndim == 1:
            builder.add(w, (bank_labels[0],))
            return
        if batch_label is None:
            raise ValueError(
                "batched PhysicalScaleBank requires a batch_label in emit()"
            )
        builder.add(w, (batch_label, bank_labels[0]))


class _MatvecMPOBank(_MPOBank):
    """Symbolic H[j] = sum_i A[j,i] kron G[i].

    Crucially, the Kronecker-product row/column spaces are represented by
    concatenating the tensorized modes of A and G; they are never flattened.
    """

    def __init__(self, matrix_bank: _MPOBank, vector_bank: _MPOBank) -> None:
        if len(matrix_bank.bank_shape) != 2:
            raise ValueError("matrix core bank must have bank shape (M, N)")
        if len(vector_bank.bank_shape) != 1:
            raise ValueError("vector core bank must have bank shape (N,)")
        M, N = matrix_bank.bank_shape
        if vector_bank.bank_shape[0] != N:
            raise ValueError(
                f"physical mode mismatch: matrix N={N}, "
                f"vector N={vector_bank.bank_shape[0]}"
            )
        self.matrix_bank = matrix_bank
        self.vector_bank = vector_bank
        self.bank_shape = (M,)
        self.row_dim = matrix_bank.row_dim * vector_bank.row_dim
        self.col_dim = matrix_bank.col_dim * vector_bank.col_dim
        self.row_modes = matrix_bank.row_modes + vector_bank.row_modes
        self.col_modes = matrix_bank.col_modes + vector_bank.col_modes

    @property
    def leaf_count(self) -> int:
        return self.matrix_bank.leaf_count + self.vector_bank.leaf_count

    def leaves(self) -> tuple[torch.Tensor, ...]:
        return self.matrix_bank.leaves() + self.vector_bank.leaves()

    def emit(
        self,
        builder: _NetworkBuilder,
        *,
        bank_labels: Sequence[int],
        row_labels: Sequence[int],
        col_labels: Sequence[int],
        batch_label: int | None = None,
    ) -> None:
        bank_labels = tuple(bank_labels)
        row_labels = tuple(row_labels)
        col_labels = tuple(col_labels)
        builder.check_labels(bank_labels, self.bank_shape)
        builder.check_labels(row_labels, self.row_modes)
        builder.check_labels(col_labels, self.col_modes)

        nr = len(self.matrix_bank.row_modes)
        nc = len(self.matrix_bank.col_modes)
        matrix_row = row_labels[:nr]
        vector_row = row_labels[nr:]
        matrix_col = col_labels[:nc]
        vector_col = col_labels[nc:]

        i_label = builder.new_label(self.matrix_bank.bank_shape[1])
        self.matrix_bank.emit(
            builder,
            bank_labels=(bank_labels[0], i_label),
            row_labels=matrix_row,
            col_labels=matrix_col,
            batch_label=batch_label,
        )
        self.vector_bank.emit(
            builder,
            bank_labels=(i_label,),
            row_labels=vector_row,
            col_labels=vector_col,
            batch_label=batch_label,
        )


# =============================================================================
# Recursive construction from leaf tensors
# =============================================================================


class _LeafCursor:
    def __init__(self, leaves: Sequence[torch.Tensor]) -> None:
        self.leaves = tuple(torch.as_tensor(x) for x in leaves)
        self.i = 0

    def take(self, shape: Sequence[int]) -> torch.Tensor:
        if self.i >= len(self.leaves):
            raise ValueError("not enough leaf tensors")
        value = self.leaves[self.i]
        expected = tuple(int(s) for s in shape)
        if tuple(value.shape) != expected:
            raise ValueError(
                f"leaf {self.i} shape mismatch: expected {expected}, "
                f"got {tuple(value.shape)}"
            )
        self.i += 1
        return value


def _bank_leaf_shapes(
    *,
    depth: int,
    d: int,
    ranks: Sequence[int],
    bank_shape: Sequence[int],
    row_dim: int,
    col_dim: int,
) -> list[tuple[int, ...]]:
    bank_shape = tuple(int(s) for s in bank_shape)
    if depth == 0:
        if row_dim != 1 or col_dim != 1:
            raise ValueError(
                "hierarchy does not terminate at rank-one maps; "
                f"bottom map is {row_dim}x{col_dim}"
            )
        return [bank_shape]

    row_modes = _balanced_factor_modes(row_dim, d)
    col_modes = _balanced_factor_modes(col_dim, d)
    q = int(ranks[depth - 1])
    bonds = _uniform_bonds(d, q)
    out: list[tuple[int, ...]] = []
    for k in range(d):
        out.extend(
            _bank_leaf_shapes(
                depth=depth - 1,
                d=d,
                ranks=ranks,
                bank_shape=bank_shape + (row_modes[k], col_modes[k]),
                row_dim=bonds[k],
                col_dim=bonds[k + 1],
            )
        )
    return out




def _stochastic_bank_leaf_specs(
    *,
    depth: int,
    d: int,
    ranks: Sequence[int],
    bank_shape: Sequence[int],
    row_dim: int,
    col_dim: int,
    output_bank_axes: Sequence[int],
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Return ``(leaf_shape, softmax_axes)`` for a stochastic MPO bank.

    The represented bank ``B`` is required to satisfy

        sum_{output bank axes, col state} B[..., row, col] = 1

    for every fixed value of the remaining bank axes and row state.  The
    normalization is pushed recursively to the rank-one leaves.  At each MPO
    level the first child is responsible for consuming the inherited output
    bank axes; every child also consumes its own local column physical mode.
    This is the recursive analogue of normalizing an ordinary MPO core over
    ``(physical column, right bond)`` for each fixed ``(physical row, left
    bond)``.
    """
    bank_shape = tuple(int(s) for s in bank_shape)
    output_bank_axes = tuple(int(a) for a in output_bank_axes)

    if any(a < 0 or a >= len(bank_shape) for a in output_bank_axes):
        raise ValueError(
            f"output bank axes {output_bank_axes} are invalid for shape "
            f"{bank_shape}"
        )

    if depth == 0:
        if row_dim != 1 or col_dim != 1:
            raise ValueError(
                "hierarchy does not terminate at rank-one maps; "
                f"bottom map is {row_dim}x{col_dim}"
            )
        if not output_bank_axes:
            raise ValueError(
                "a stochastic rank-one leaf must have at least one output axis"
            )
        return [(bank_shape, tuple(sorted(set(output_bank_axes))))]

    row_modes = _balanced_factor_modes(row_dim, d)
    col_modes = _balanced_factor_modes(col_dim, d)
    q = int(ranks[depth - 1])
    bonds = _uniform_bonds(d, q)

    out: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for k in range(d):
        child_bank_shape = bank_shape + (row_modes[k], col_modes[k])
        local_col_axis = len(bank_shape) + 1
        child_output_axes = (
            (*output_bank_axes, local_col_axis)
            if k == 0
            else (local_col_axis,)
        )
        out.extend(
            _stochastic_bank_leaf_specs(
                depth=depth - 1,
                d=d,
                ranks=ranks,
                bank_shape=child_bank_shape,
                row_dim=bonds[k],
                col_dim=bonds[k + 1],
                output_bank_axes=child_output_axes,
            )
        )
    return out

def _build_static_bank(
    *,
    depth: int,
    d: int,
    ranks: Sequence[int],
    bank_shape: Sequence[int],
    row_dim: int,
    col_dim: int,
    cursor: _LeafCursor,
) -> _StaticMPOBank:
    bank_shape = tuple(int(s) for s in bank_shape)
    if depth == 0:
        leaf = cursor.take(bank_shape)
        return _StaticMPOBank(
            depth=0,
            d=d,
            ranks=ranks,
            bank_shape=bank_shape,
            row_dim=row_dim,
            col_dim=col_dim,
            leaf=leaf,
        )

    row_modes = _balanced_factor_modes(row_dim, d)
    col_modes = _balanced_factor_modes(col_dim, d)
    q = int(ranks[depth - 1])
    bonds = _uniform_bonds(d, q)
    children = []
    for k in range(d):
        children.append(
            _build_static_bank(
                depth=depth - 1,
                d=d,
                ranks=ranks,
                bank_shape=bank_shape + (row_modes[k], col_modes[k]),
                row_dim=bonds[k],
                col_dim=bonds[k + 1],
                cursor=cursor,
            )
        )
    return _StaticMPOBank(
        depth=depth,
        d=d,
        ranks=ranks,
        bank_shape=bank_shape,
        row_dim=row_dim,
        col_dim=col_dim,
        children=children,
    )


# =============================================================================
# Rank-one basis parsing
# =============================================================================


def _rank_one_factors(other, modes: Sequence[int]) -> tuple[torch.Tensor, ...]:
    """Parse rank-one factors with shape ``(n,)`` or batched ``(B, n)``."""
    modes = tuple(int(n) for n in modes)

    # TTVector-like objects expose tensor cores; NestedTTVector.cores are banks.
    cores = getattr(other, "cores", None)
    other_modes = getattr(other, "modes", None)
    if (
        cores is not None
        and other_modes is not None
        and len(cores) > 0
        and torch.is_tensor(cores[0])
    ):
        other_modes = tuple(int(n) for n in other_modes)
        if other_modes != modes:
            raise ValueError(
                f"rank-one tensor modes do not match: {other_modes} != {modes}"
            )
        factors = []
        batch_shape: tuple[int, ...] | None = None
        for k, (core, n) in enumerate(zip(cores, modes)):
            core = torch.as_tensor(core)
            # Accept (..., 1, n, 1) with optional leading batch dims.
            if core.ndim >= 3 and tuple(core.shape[-3:]) == (1, n, 1):
                leading = core.shape[:-3]
                if batch_shape is None:
                    batch_shape = tuple(int(s) for s in leading)
                elif batch_shape != tuple(int(s) for s in leading):
                    raise ValueError(
                        "inconsistent TT batch shapes across cores: "
                        f"{batch_shape} vs {tuple(leading)}"
                    )
                if len(leading) == 0:
                    factors.append(core.reshape(n))
                elif len(leading) == 1:
                    factors.append(core.reshape(leading[0], n))
                else:
                    raise ValueError(
                        "NestedTT rank-one factors currently support at most "
                        f"one batch dim; core {k} has shape {tuple(core.shape)}"
                    )
                continue
            raise ValueError(
                "elementwise_multiply only supports a rank-one TT with cores "
                f"shape (..., 1, n, 1); core {k} has shape {tuple(core.shape)}"
            )
        return tuple(factors)

    try:
        factors = tuple(torch.as_tensor(x) for x in other)
    except TypeError as exc:
        raise TypeError(
            "expected a sequence of factors or a rank-one TT-like object"
        ) from exc

    if len(factors) != len(modes):
        raise ValueError(f"expected {len(modes)} factors, got {len(factors)}")

    batch_size: int | None = None
    for k, (factor, n) in enumerate(zip(factors, modes)):
        if factor.ndim == 1:
            if int(factor.shape[0]) != n:
                raise ValueError(
                    f"factor {k} must have shape ({n},), got {tuple(factor.shape)}"
                )
            if batch_size is not None:
                raise ValueError(
                    "mixed batched / unbatched rank-one factors are not supported"
                )
        elif factor.ndim == 2:
            if int(factor.shape[-1]) != n:
                raise ValueError(
                    f"factor {k} must have shape (B, {n}), got {tuple(factor.shape)}"
                )
            if batch_size is None:
                batch_size = int(factor.shape[0])
            elif batch_size != int(factor.shape[0]):
                raise ValueError(
                    "inconsistent batch sizes across rank-one factors: "
                    f"{batch_size} vs {int(factor.shape[0])}"
                )
        else:
            raise ValueError(
                f"factor {k} must have shape ({n},) or (B, {n}), "
                f"got {tuple(factor.shape)}"
            )
    return factors


def _batch_shape_from_factors(
    factors: Sequence[torch.Tensor],
) -> tuple[int, ...]:
    for factor in factors:
        if torch.as_tensor(factor).ndim == 2:
            return (int(factor.shape[0]),)
    return ()


# =============================================================================
# Outer tensor-network emission helpers
# =============================================================================


def _emit_vector_network(
    core_banks: Sequence[_MPOBank],
    modes: Sequence[int],
    *,
    output_physical: bool,
    batch_size: int | None = None,
) -> tuple[_NetworkBuilder, tuple[int, ...]]:
    modes = tuple(int(n) for n in modes)
    if len(core_banks) != len(modes):
        raise ValueError("core count/mode count mismatch")

    builder = _NetworkBuilder()
    batch_label = (
        builder.new_label(int(batch_size)) if batch_size is not None else None
    )
    physical_labels = tuple(builder.new_label(n) for n in modes)

    # Left boundary of the first outer core.
    left_labels = builder.new_labels(core_banks[0].row_modes)

    for k, bank in enumerate(core_banks):
        if k > 0:
            if tuple(bank.row_modes) != tuple(core_banks[k - 1].col_modes):
                raise ValueError(
                    "adjacent nested TT bond tensorizations do not match: "
                    f"{bank.row_modes} != {core_banks[k - 1].col_modes}"
                )
        if k == len(core_banks) - 1:
            right_labels = builder.new_labels(bank.col_modes)
        else:
            next_modes = core_banks[k + 1].row_modes
            if tuple(bank.col_modes) != tuple(next_modes):
                raise ValueError(
                    "adjacent nested TT bond tensorizations do not match: "
                    f"{bank.col_modes} != {next_modes}"
                )
            right_labels = builder.new_labels(bank.col_modes)

        bank.emit(
            builder,
            bank_labels=(physical_labels[k],),
            row_labels=left_labels,
            col_labels=right_labels,
            batch_label=batch_label,
        )
        left_labels = right_labels

    if output_physical:
        output = (
            (batch_label, *physical_labels)
            if batch_label is not None
            else physical_labels
        )
    else:
        output = (batch_label,) if batch_label is not None else ()
    return builder, output


def _emit_matrix_network(
    core_banks: Sequence[_MPOBank],
    row_modes: Sequence[int],
    col_modes: Sequence[int],
) -> tuple[_NetworkBuilder, tuple[int, ...]]:
    row_modes = tuple(int(n) for n in row_modes)
    col_modes = tuple(int(n) for n in col_modes)
    builder = _NetworkBuilder()
    row_labels = tuple(builder.new_label(n) for n in row_modes)
    col_labels = tuple(builder.new_label(n) for n in col_modes)

    left_labels = builder.new_labels(core_banks[0].row_modes)
    for k, bank in enumerate(core_banks):
        if k > 0 and tuple(bank.row_modes) != tuple(core_banks[k - 1].col_modes):
            raise ValueError("adjacent nested MPO bond tensorizations do not match")
        if k == len(core_banks) - 1:
            right_labels = builder.new_labels(bank.col_modes)
        else:
            if tuple(bank.col_modes) != tuple(core_banks[k + 1].row_modes):
                raise ValueError("adjacent nested MPO bond tensorizations do not match")
            right_labels = builder.new_labels(bank.col_modes)
        bank.emit(
            builder,
            bank_labels=(row_labels[k], col_labels[k]),
            row_labels=left_labels,
            col_labels=right_labels,
            batch_label=None,
        )
        left_labels = right_labels

    return builder, (*row_labels, *col_labels)


# =============================================================================
# Public classes
# =============================================================================


class NestedTTVector:
    """TT vector whose matrix-valued cores remain recursively MPO-structured."""

    def __init__(
        self,
        spec: NestedTTVectorSpec,
        leaves: Sequence[torch.Tensor],
        numerical_tolerance: float = 1e-20,
    ) -> None:
        expected = self.leaf_shapes(spec)
        if len(leaves) != len(expected):
            raise ValueError(
                f"expected {len(expected)} vector leaf cores, got {len(leaves)}"
            )
        cursor = _LeafCursor(leaves)
        top_bonds = _uniform_bonds(spec.d, spec.ranks[-1])
        banks: list[_MPOBank] = []
        for k, n in enumerate(spec.modes):
            banks.append(
                _build_static_bank(
                    depth=spec.depth - 1,
                    d=spec.d,
                    ranks=spec.ranks,
                    bank_shape=(n,),
                    row_dim=top_bonds[k],
                    col_dim=top_bonds[k + 1],
                    cursor=cursor,
                )
            )
        if cursor.i != len(leaves):
            raise ValueError("too many vector leaves")
        self.spec = spec
        self._core_banks = tuple(banks)
        self._modes = tuple(spec.modes)
        self._depth = spec.depth
        self._history_length = 1
        self._batch_shape: tuple[int, ...] = ()
        self._numerical_tolerance = float(numerical_tolerance)

    @classmethod
    def _from_banks(
        cls,
        *,
        modes: Sequence[int],
        depth: int,
        core_banks: Sequence[_MPOBank],
        history_length: int,
        numerical_tolerance: float = 1e-20,
        batch_shape: tuple[int, ...] = (),
    ) -> "NestedTTVector":
        obj = cls.__new__(cls)
        obj.spec = None
        obj._core_banks = tuple(core_banks)
        obj._modes = tuple(int(n) for n in modes)
        obj._depth = int(depth)
        obj._history_length = int(history_length)
        obj._batch_shape = tuple(int(s) for s in batch_shape)
        obj._numerical_tolerance = float(numerical_tolerance)
        return obj

    @property
    def numerical_tolerance(self) -> float:
        return self._numerical_tolerance

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._batch_shape

    @property
    def is_batched(self) -> bool:
        return len(self._batch_shape) > 0

    def with_numerical_tolerance(self, numerical_tolerance: float) -> "NestedTTVector":
        return NestedTTVector._from_banks(
            modes=self.modes,
            depth=self.depth,
            core_banks=self._core_banks,
            history_length=self.history_length,
            numerical_tolerance=numerical_tolerance,
            batch_shape=self.batch_shape,
        )

    @staticmethod
    def leaf_shapes(spec: NestedTTVectorSpec) -> tuple[tuple[int, ...], ...]:
        top_bonds = _uniform_bonds(spec.d, spec.ranks[-1])
        out: list[tuple[int, ...]] = []
        for k, n in enumerate(spec.modes):
            out.extend(
                _bank_leaf_shapes(
                    depth=spec.depth - 1,
                    d=spec.d,
                    ranks=spec.ranks,
                    bank_shape=(n,),
                    row_dim=top_bonds[k],
                    col_dim=top_bonds[k + 1],
                )
            )
        return tuple(out)

    @property
    def cores(self) -> tuple[_MPOBank, ...]:
        """Outer TT cores, each represented as a bank of nested MPOs."""
        return self._core_banks

    @property
    def modes(self) -> tuple[int, ...]:
        return self._modes

    @property
    def d(self) -> int:
        return len(self._modes)

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def history_length(self) -> int:
        return self._history_length

    @property
    def ranks(self) -> tuple[int, ...]:
        return (self._core_banks[0].row_dim,) + tuple(
            b.col_dim for b in self._core_banks
        )

    @property
    def leaf_count(self) -> int:
        return sum(b.leaf_count for b in self._core_banks)

    @property
    def leaves(self) -> tuple[torch.Tensor, ...]:
        return tuple(x for b in self._core_banks for x in b.leaves())

    @property
    def shape(self) -> torch.Size:
        return torch.Size(self._batch_shape + (math.prod(self._modes),))

    def materialize_cores(self) -> tuple[torch.Tensor, ...]:
        """Reference/debug operation; not used by ``sum``."""
        if self.is_batched:
            raise NotImplementedError(
                "materialize_cores is not supported for batched NestedTTVector"
            )
        out = []
        for bank in self._core_banks:
            dense = bank.materialize()  # (n, r_left, r_right)
            out.append(dense.permute(1, 0, 2).contiguous())
        return tuple(out)

    def to_dense(self) -> torch.Tensor:
        # Contract the leaf tensor network directly and leave only physical
        # coefficient indices open.  Parent core matrices are never built.
        batch_size = self._batch_shape[0] if self.is_batched else None
        builder, output = _emit_vector_network(
            self._core_banks,
            self._modes,
            output_physical=True,
            batch_size=batch_size,
        )
        tensor = builder.contract(output)
        if self.is_batched:
            return tensor.reshape(*self._batch_shape, self.n)
        return tensor.reshape(self.n)

    @property
    def n(self) -> int:
        return int(math.prod(self._modes))

    @property
    def dtype(self) -> torch.dtype:
        leaves = self.leaves
        if not leaves:
            return torch.float32
        return leaves[0].dtype

    @property
    def device(self) -> torch.device:
        leaves = self.leaves
        if not leaves:
            return torch.device("cpu")
        return leaves[0].device

    def elementwise_multiply(self, other) -> "NestedTTVector":
        if isinstance(other, (int, float)) or (
            torch.is_tensor(other) and other.ndim == 0
        ):
            scale = torch.as_tensor(other, dtype=self.dtype, device=self.device)
            # Put the full scale on the first mode only.
            factors = (
                scale
                * torch.ones(self.modes[0], dtype=self.dtype, device=self.device),
            ) + tuple(
                torch.ones(n, dtype=self.dtype, device=self.device)
                for n in self.modes[1:]
            )
            return self.elementwise_multiply(factors)

        # Hadamard product against another NestedTTVector: require this vector
        # to be (numerically) rank-one and scale ``other`` by those factors.
        if isinstance(other, NestedTTVector):
            if other.modes != self.modes:
                raise ValueError(
                    "NestedTTVector elementwise product requires matching modes, "
                    f"got {self.modes} and {other.modes}"
                )
            if other.depth != self.depth:
                raise ValueError(
                    "NestedTTVector elementwise product requires matching depth, "
                    f"got {self.depth} and {other.depth}"
                )
            if self.is_batched:
                raise ValueError(
                    "NestedTTVector * NestedTTVector currently requires the "
                    "left operand to be unbatched when factorizing it"
                )
            factors = rank_one_factors_from_nested(self)
            return other.elementwise_multiply(factors)

        factors = _rank_one_factors(other, self.modes)
        factor_batch = _batch_shape_from_factors(factors)
        if self.is_batched and factor_batch and factor_batch != self.batch_shape:
            raise ValueError(
                "incompatible batch shapes for NestedTT elementwise product: "
                f"{self.batch_shape} vs {factor_batch}"
            )
        batch_shape = factor_batch or self.batch_shape
        banks = [
            _PhysicalScaleBank(bank, factor)
            for bank, factor in zip(self._core_banks, factors)
        ]
        return NestedTTVector._from_banks(
            modes=self.modes,
            depth=self.depth,
            core_banks=banks,
            history_length=self.history_length,
            numerical_tolerance=self.numerical_tolerance,
            batch_shape=batch_shape,
        )

    def elementwise_divide(self, other) -> "NestedTTVector":
        """Hadamard division by a rank-one tensor (same mode shape).

        Applies the denominator's ``numerical_tolerance`` (falling back to
        ``self.numerical_tolerance``) factor-wise so callers never need
        ``q + tol``.
        """
        if isinstance(other, NestedTTVector):
            if other.is_batched:
                raise ValueError(
                    "NestedTTVector division currently requires an unbatched "
                    "rank-one denominator"
                )
            factors = rank_one_factors_from_nested(other)
            tol = float(other.numerical_tolerance)
        else:
            factors = _rank_one_factors(other, self.modes)
            if _batch_shape_from_factors(factors):
                raise ValueError(
                    "NestedTTVector division currently requires an unbatched "
                    "rank-one denominator"
                )
            tol = float(getattr(other, "numerical_tolerance", self.numerical_tolerance))
        inv = tuple(1.0 / (torch.as_tensor(f) + tol) for f in factors)
        return self.elementwise_multiply(inv)

    def __mul__(self, other) -> "NestedTTVector":
        return self.elementwise_multiply(other)

    def __rmul__(self, other) -> "NestedTTVector":
        return self.elementwise_multiply(other)

    def __truediv__(self, other) -> "NestedTTVector":
        return self.elementwise_divide(other)

    def __rtruediv__(self, other) -> "NestedTTVector":
        # other / self with ``other`` a rank-one TT-like object or factor seq.
        return ones_like(self).elementwise_multiply(other).elementwise_divide(self)

    def sum(self) -> torch.Tensor:
        """Sum all physical entries without materializing any outer core.

        The contraction is emitted directly from the recursively nested MPO
        leaves.  In particular, after recurrent matvecs no matrix of size
        ``R**t * r`` by ``R**t * r`` is constructed.

        Returns a scalar for unbatched vectors, or shape ``batch_shape`` when
        the vector carries a leading batch from basis evaluation.
        """
        batch_size = self._batch_shape[0] if self.is_batched else None
        builder, output = _emit_vector_network(
            self._core_banks,
            self._modes,
            output_physical=False,
            batch_size=batch_size,
        )
        tensor = builder.contract(output)
        if self.is_batched:
            return tensor.reshape(*self._batch_shape)
        return tensor.reshape(())

    def __repr__(self) -> str:
        return (
            f"NestedTTVector(shape={tuple(self.shape)}, modes={self.modes}, "
            f"ranks={self.ranks}, depth={self.depth}, "
            f"batch_shape={self.batch_shape}, "
            f"history_length={self.history_length}, leaves={self.leaf_count})"
        )


class NestedTTMatrix:
    """TT matrix whose matrix-valued cores remain recursively MPO-structured."""

    def __init__(
        self,
        spec: NestedTTMatrixSpec,
        leaves: Sequence[torch.Tensor],
    ) -> None:
        expected = self.leaf_shapes(spec)
        if len(leaves) != len(expected):
            raise ValueError(
                f"expected {len(expected)} matrix leaf cores, got {len(leaves)}"
            )
        cursor = _LeafCursor(leaves)
        top_bonds = _uniform_bonds(spec.d, spec.ranks[-1])
        banks: list[_MPOBank] = []
        for k, (m, n) in enumerate(zip(spec.row_modes, spec.col_modes)):
            banks.append(
                _build_static_bank(
                    depth=spec.depth - 1,
                    d=spec.d,
                    ranks=spec.ranks,
                    bank_shape=(m, n),
                    row_dim=top_bonds[k],
                    col_dim=top_bonds[k + 1],
                    cursor=cursor,
                )
            )
        if cursor.i != len(leaves):
            raise ValueError("too many matrix leaves")
        self.spec = spec
        self._core_banks = tuple(banks)

    @staticmethod
    def leaf_shapes(spec: NestedTTMatrixSpec) -> tuple[tuple[int, ...], ...]:
        top_bonds = _uniform_bonds(spec.d, spec.ranks[-1])
        out: list[tuple[int, ...]] = []
        for k, (m, n) in enumerate(zip(spec.row_modes, spec.col_modes)):
            out.extend(
                _bank_leaf_shapes(
                    depth=spec.depth - 1,
                    d=spec.d,
                    ranks=spec.ranks,
                    bank_shape=(m, n),
                    row_dim=top_bonds[k],
                    col_dim=top_bonds[k + 1],
                )
            )
        return tuple(out)

    @staticmethod
    def row_stochastic_leaf_specs(
        spec: NestedTTMatrixSpec,
    ) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        """Leaf shapes and softmax axes guaranteeing row stochasticity.

        Each outer MPO core is normalized over its physical column index and
        outgoing TT bond for every fixed physical row and incoming TT bond.
        The returned axis sets push that normalization all the way to the
        rank-one leaves without materializing any parent core.
        """
        top_bonds = _uniform_bonds(spec.d, spec.ranks[-1])
        out: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for k, (m, n) in enumerate(zip(spec.row_modes, spec.col_modes)):
            out.extend(
                _stochastic_bank_leaf_specs(
                    depth=spec.depth - 1,
                    d=spec.d,
                    ranks=spec.ranks,
                    bank_shape=(m, n),
                    row_dim=top_bonds[k],
                    col_dim=top_bonds[k + 1],
                    output_bank_axes=(1,),  # physical column N
                )
            )
        return tuple(out)

    @property
    def cores(self) -> tuple[_MPOBank, ...]:
        return self._core_banks

    @property
    def row_modes(self) -> tuple[int, ...]:
        return self.spec.row_modes

    @property
    def col_modes(self) -> tuple[int, ...]:
        return self.spec.col_modes

    @property
    def d(self) -> int:
        return self.spec.d

    @property
    def depth(self) -> int:
        return self.spec.depth

    @property
    def ranks(self) -> tuple[int, ...]:
        return (self._core_banks[0].row_dim,) + tuple(
            b.col_dim for b in self._core_banks
        )

    @property
    def leaf_count(self) -> int:
        return sum(b.leaf_count for b in self._core_banks)

    @property
    def leaves(self) -> tuple[torch.Tensor, ...]:
        return tuple(x for b in self._core_banks for x in b.leaves())

    @property
    def shape(self) -> torch.Size:
        return torch.Size((math.prod(self.row_modes), math.prod(self.col_modes)))

    @property
    def dtype(self) -> torch.dtype:
        leaves = self.leaves
        if not leaves:
            return torch.float32
        return leaves[0].dtype

    @property
    def device(self) -> torch.device:
        leaves = self.leaves
        if not leaves:
            return torch.device("cpu")
        return leaves[0].device

    def materialize_cores(self) -> tuple[torch.Tensor, ...]:
        out = []
        for bank in self._core_banks:
            dense = bank.materialize()  # (M, N, R0, R1)
            out.append(dense.permute(2, 0, 1, 3).contiguous())
        return tuple(out)

    def to_dense(self) -> torch.Tensor:
        builder, output = _emit_matrix_network(
            self._core_banks, self.row_modes, self.col_modes
        )
        tensor = builder.contract(output)
        return tensor.reshape(math.prod(self.row_modes), math.prod(self.col_modes))

    @property
    def T(self) -> "NestedTTMatrix":
        """Transpose a static nested MPO by swapping physical / bond pairs."""
        if self.spec is None:
            raise TypeError("transpose is only supported for static NestedTTMatrix")
        new_spec = NestedTTMatrixSpec(
            row_modes=self.col_modes,
            col_modes=self.row_modes,
            depth=self.depth,
            ranks=self.spec.ranks,
        )
        new_leaves = [_transpose_nested_matrix_leaf(leaf) for leaf in self.leaves]
        return NestedTTMatrix(new_spec, new_leaves)

    def matvec(self, x) -> NestedTTVector:
        # Rank-one TTVectors (e.g. coefficient-free TTBasis evals) convert in.
        if not isinstance(x, NestedTTVector):
            if (
                hasattr(x, "cores")
                and hasattr(x, "modes")
                and len(getattr(x, "cores", ())) > 0
                and torch.is_tensor(x.cores[0])
            ):
                factors = _rank_one_factors(x, self.col_modes)
                tol = float(getattr(x, "numerical_tolerance", 1e-20))
                x = nested_tt_vector_from_factors(
                    factors, depth=self.depth, numerical_tolerance=tol
                )
            else:
                raise TypeError(
                    "NestedTTMatrix.matvec requires NestedTTVector or rank-one "
                    f"TT-like vector, got {type(x).__name__}"
                )
        if x.modes != self.col_modes:
            raise ValueError(
                f"vector modes {x.modes} do not match matrix columns {self.col_modes}"
            )
        if x.depth != self.depth:
            raise ValueError(
                f"nested depth mismatch: matrix={self.depth}, vector={x.depth}"
            )
        if x.d != self.d:
            raise ValueError(
                f"outer core-count mismatch: matrix={self.d}, vector={x.d}"
            )

        banks = [
            _MatvecMPOBank(Mb, vb)
            for Mb, vb in zip(self._core_banks, x._core_banks)
        ]
        return NestedTTVector._from_banks(
            modes=self.row_modes,
            depth=self.depth,
            core_banks=banks,
            history_length=x.history_length + 1,
            numerical_tolerance=x.numerical_tolerance,
            batch_shape=x.batch_shape,
        )

    def rev_matvec(self, x) -> NestedTTVector:
        """``Mᵀ @ x``."""
        return self.T.matvec(x)

    def sum(self) -> torch.Tensor:
        """Sum of all matrix entries without densifying the full matrix."""
        ones = ones_nested_tt_vector(
            self.col_modes,
            depth=self.depth,
            dtype=self.dtype,
            device=self.device,
        )
        return self.matvec(ones).sum()

    def __matmul__(self, x: NestedTTVector) -> NestedTTVector:
        return self.matvec(x)

    def __repr__(self) -> str:
        return (
            f"NestedTTMatrix(shape={tuple(self.shape)}, "
            f"modes={list(zip(self.row_modes, self.col_modes))}, "
            f"ranks={self.ranks}, depth={self.depth}, leaves={self.leaf_count})"
        )


# =============================================================================
# Construction helpers used by RFF / TTBasis integration
# =============================================================================


def _transpose_nested_matrix_leaf(leaf: torch.Tensor) -> torch.Tensor:
    """Swap physical (M, N) axes of a nested MPO leaf.

    Nested bond-factor axes are left unchanged: for the static NestedTT
    construction this yields a matrix whose dense form is the transpose.
    """
    leaf = torch.as_tensor(leaf)
    if leaf.ndim < 2:
        raise ValueError(f"matrix leaf must have at least 2 dims, got {tuple(leaf.shape)}")
    return leaf.transpose(0, 1).contiguous()


def _embed_factor_into_leaf(
    factor: torch.Tensor,
    shape: Sequence[int],
    *,
    leading: int,
) -> torch.Tensor:
    """Reshape a leading-axis factor into a nested leaf padded with ones-axes."""
    factor = torch.as_tensor(factor)
    shape = tuple(int(s) for s in shape)
    if tuple(factor.shape) != shape[:leading]:
        raise ValueError(
            f"factor shape {tuple(factor.shape)} incompatible with leaf {shape}"
        )
    if any(s != 1 for s in shape[leading:]):
        raise ValueError(
            f"cannot embed into non-trailing-ones leaf shape {shape}"
        )
    return factor.reshape(shape)


def ones_nested_tt_vector(
    modes: Sequence[int],
    *,
    depth: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    numerical_tolerance: float = 1e-20,
) -> NestedTTVector:
    """Rank-one nested TT vector of all ones (coefficient-free basis partner)."""
    modes = tuple(int(m) for m in modes)
    depth = int(depth)
    ranks = (1,) * depth
    spec = NestedTTVectorSpec(modes=modes, depth=depth, ranks=ranks)
    device = torch.device("cpu" if device is None else device)
    leaves = [
        torch.ones(shape, dtype=dtype, device=device)
        for shape in NestedTTVector.leaf_shapes(spec)
    ]
    return NestedTTVector(
        spec, leaves, numerical_tolerance=numerical_tolerance
    )


def ones_like(v: NestedTTVector) -> NestedTTVector:
    return ones_nested_tt_vector(
        v.modes,
        depth=v.depth,
        dtype=v.dtype,
        device=v.device,
        numerical_tolerance=v.numerical_tolerance,
    )


def nested_tt_vector_from_factors(
    factors: Sequence[torch.Tensor],
    *,
    depth: int,
    numerical_tolerance: float = 1e-20,
) -> NestedTTVector:
    """Build a rank-one NestedTTVector whose dense form is ``kron(*factors)``.

    Factors may be ``(n,)`` or batched ``(B, n)``.  Batched factors are applied
    as physical scales on an all-ones nested vector (no densification).
    """
    factors = tuple(torch.as_tensor(f) for f in factors)
    batch_shape = _batch_shape_from_factors(factors)
    if batch_shape:
        # Validate shapes then scale an unbatched ones vector.
        modes = []
        for f in factors:
            if f.ndim == 1:
                modes.append(int(f.shape[0]))
            else:
                modes.append(int(f.shape[-1]))
        ones = ones_nested_tt_vector(
            modes,
            depth=depth,
            dtype=factors[0].dtype,
            device=factors[0].device,
            numerical_tolerance=numerical_tolerance,
        )
        return ones.elementwise_multiply(factors)

    if any(f.ndim != 1 for f in factors):
        raise ValueError("all unbatched factors must be 1-D")
    modes = tuple(int(f.shape[0]) for f in factors)
    depth = int(depth)
    ranks = (1,) * depth
    spec = NestedTTVectorSpec(modes=modes, depth=depth, ranks=ranks)
    shapes = NestedTTVector.leaf_shapes(spec)
    d = len(modes)
    if len(shapes) % d != 0:
        raise RuntimeError("internal leaf layout mismatch")
    leaves_per = len(shapes) // d
    leaves: list[torch.Tensor] = []
    for k, factor in enumerate(factors):
        block = shapes[k * leaves_per : (k + 1) * leaves_per]
        for j, shape in enumerate(block):
            if j == 0:
                leaves.append(
                    _embed_factor_into_leaf(factor, shape, leading=1)
                )
            else:
                leaves.append(
                    torch.ones(
                        shape, dtype=factor.dtype, device=factor.device
                    )
                )
    return NestedTTVector(
        spec, leaves, numerical_tolerance=numerical_tolerance
    )


def nested_tt_matrix_from_separable_cores(
    cores: Sequence[torch.Tensor],
    *,
    depth: int,
) -> NestedTTMatrix:
    """Embed a rank-one / Kronecker MPO into a NestedTTMatrix of given depth.

    ``cores[k]`` has shape ``(M_k, N_k)``.  Hierarchy ranks are all one, so the
    dense matrix equals ``kron(cores[0], ..., cores[d-1])``.
    """
    cores = tuple(torch.as_tensor(c) for c in cores)
    if any(c.ndim != 2 for c in cores):
        raise ValueError("separable MPO cores must be 2-D")
    row_modes = tuple(int(c.shape[0]) for c in cores)
    col_modes = tuple(int(c.shape[1]) for c in cores)
    depth = int(depth)
    ranks = (1,) * depth
    spec = NestedTTMatrixSpec(
        row_modes=row_modes,
        col_modes=col_modes,
        depth=depth,
        ranks=ranks,
    )
    shapes = NestedTTMatrix.leaf_shapes(spec)
    d = len(row_modes)
    leaves_per = len(shapes) // d
    leaves: list[torch.Tensor] = []
    for k, core in enumerate(cores):
        block = shapes[k * leaves_per : (k + 1) * leaves_per]
        for j, shape in enumerate(block):
            if j == 0:
                leaves.append(
                    _embed_factor_into_leaf(core, shape, leading=2)
                )
            else:
                leaves.append(
                    torch.ones(shape, dtype=core.dtype, device=core.device)
                )
    return NestedTTMatrix(spec, leaves)


def nested_tt_matrix_from_tt_matrix(matrix, depth: int) -> NestedTTMatrix:
    """Convert a boundary-rank-1 :class:`TTMatrix` into a NestedTTMatrix."""
    cores = getattr(matrix, "cores", None)
    if cores is None:
        raise TypeError("expected a TTMatrix-like object with .cores")
    cores_t = tuple(torch.as_tensor(c) for c in cores)
    if any(c.shape[0] != 1 or c.shape[-1] != 1 for c in cores_t):
        raise ValueError(
            "nested_tt_matrix_from_tt_matrix requires a rank-1 TTMatrix "
            f"(boundary bonds 1), got ranks={[c.shape[0] for c in cores_t] + [cores_t[-1].shape[-1]]}"
        )
    slices = tuple(c.reshape(c.shape[1], c.shape[2]) for c in cores_t)
    return nested_tt_matrix_from_separable_cores(slices, depth=depth)


def rank_one_factors_from_dense(
    values: torch.Tensor, modes: Sequence[int]
) -> tuple[torch.Tensor, ...]:
    """Exact CP/TT rank-one factorization of a tensorized vector via successive SVDs."""
    modes = tuple(int(m) for m in modes)
    values = torch.as_tensor(values).reshape(-1)
    if int(values.numel()) != int(math.prod(modes)):
        raise ValueError(
            f"values has {values.numel()} entries, modes {modes} need {math.prod(modes)}"
        )
    rest = values.reshape(modes)
    factors: list[torch.Tensor] = []
    for _ in range(len(modes) - 1):
        n_k = rest.shape[0]
        mat = rest.reshape(n_k, -1)
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        scale = torch.sqrt(s[0].clamp_min(0))
        factors.append(u[:, 0] * scale)
        rest = (scale * vh[0, :]).reshape(rest.shape[1:])
    factors.append(rest.reshape(-1))
    return tuple(factors)


def rank_one_factors_from_nested(v):
    if v.is_batched:
        raise ValueError("expected unbatched vector")

    if any(r != 1 for r in v.ranks):
        raise ValueError(
            f"NestedTTVector is not outer-rank-one: ranks={v.ranks}"
        )

    factors = []

    for n, bank in zip(v.modes, v.cores):
        builder = _NetworkBuilder()

        physical = builder.new_label(n)
        rows = builder.new_labels(bank.row_modes)
        cols = builder.new_labels(bank.col_modes)

        bank.emit(
            builder,
            bank_labels=(physical,),
            row_labels=rows,
            col_labels=cols,
            batch_label=None,
        )

        # All row/col dimensions have total size one because the
        # outer TT rank is one.
        factor = builder.contract((physical,))
        factors.append(factor.reshape(n))

    return tuple(factors)