from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

try:
    from .nested_tt_leaf import NestedTTMatrixLeaf, NestedTTVectorLeaf
except ImportError:  # pragma: no cover - allows standalone tests
    from rational_factor.models.tt.nested_tt_leaf import NestedTTMatrixLeaf, NestedTTVectorLeaf


# =============================================================================
# Public specifications
# =============================================================================


@dataclass(frozen=True)
class NestedTTVectorSpec:
    modes: tuple[int, ...]
    depth: int
    ranks: tuple[int, ...]
    separation_rank: int = 1

    def __post_init__(self) -> None:
        if int(self.depth) < 1:
            raise ValueError("depth must be >= 1")
        if not self.modes or any(int(n) < 1 for n in self.modes):
            raise ValueError("modes must be non-empty and positive")
        if len(self.ranks) != int(self.depth):
            raise ValueError("ranks must have one entry per hierarchy level")
        if any(int(r) < 1 for r in self.ranks):
            raise ValueError("all hierarchy ranks must be positive")
        if int(self.separation_rank) < 1:
            raise ValueError("separation_rank must be positive")

    @property
    def d(self) -> int:
        return len(self.modes)


@dataclass(frozen=True)
class NestedTTMatrixSpec:
    row_modes: tuple[int, ...]
    col_modes: tuple[int, ...]
    depth: int
    ranks: tuple[int, ...]
    separation_rank: int = 1

    def __post_init__(self) -> None:
        if int(self.depth) < 1:
            raise ValueError("depth must be >= 1")
        if not self.row_modes or len(self.row_modes) != len(self.col_modes):
            raise ValueError("row_modes and col_modes must be non-empty and equal length")
        if any(int(n) < 1 for n in self.row_modes + self.col_modes):
            raise ValueError("all physical modes must be positive")
        if len(self.ranks) != int(self.depth):
            raise ValueError("ranks must have one entry per hierarchy level")
        if any(int(r) < 1 for r in self.ranks):
            raise ValueError("all hierarchy ranks must be positive")
        if int(self.separation_rank) < 1:
            raise ValueError("separation_rank must be positive")

    @property
    def d(self) -> int:
        return len(self.row_modes)


# =============================================================================
# Small helpers
# =============================================================================


def _prime_factors(n: int) -> list[int]:
    ans: list[int] = []
    p = 2
    while p * p <= n:
        while n % p == 0:
            ans.append(p)
            n //= p
        p += 1
    if n > 1:
        ans.append(n)
    return ans


def _balanced_factor_modes(n: int, d: int) -> tuple[int, ...]:
    n, d = int(n), int(d)
    if n < 1 or d < 1:
        raise ValueError("n and d must be positive")
    out = [1] * d
    for p in sorted(_prime_factors(n), reverse=True):
        j = min(range(d), key=out.__getitem__)
        out[j] *= p
    return tuple(out)


def _uniform_bonds(d: int, rank: int) -> tuple[int, ...]:
    if d == 1:
        return (1, 1)
    return (1,) + (int(rank),) * (d - 1) + (1,)




def _broadcast_batch_sizes(*sizes: int) -> int:
    out = 1
    for size in sizes:
        size = int(size)
        if size < 1:
            raise ValueError("batch size must be positive")
        if size != 1:
            if out not in (1, size):
                raise ValueError(f"incompatible batch sizes: {sizes}")
            out = size
    return out

def _numel_from_labels(labels: Sequence[int], sizes: dict[int, int]) -> int:
    out = 1
    for label in labels:
        out *= int(sizes[int(label)])
    return out


# =============================================================================
# Local/streaming tensor-network contraction
# =============================================================================


@dataclass(frozen=True)
class ContractionStats:
    operands: int
    pair_contractions: int
    max_intermediate_numel: int
    max_intermediate_ndim: int


class _NetworkBuilder:
    """Integer-labelled local factor graph.

    Unlike the previous implementation, ``contract`` never sends the whole
    graph to one global einsum.  It greedily combines two factors at a time.
    Shared labels that still occur elsewhere are retained (Hadamard-style);
    labels whose final two occurrences meet are summed immediately.
    """

    def __init__(self) -> None:
        self.operands: list[tuple[torch.Tensor, tuple[int, ...]]] = []
        self._next_label = 0
        self._sizes: dict[int, int] = {}
        self.last_stats: ContractionStats | None = None

    def new_label(self, size: int) -> int:
        label = self._next_label
        self._next_label += 1
        self._sizes[label] = int(size)
        return label

    def new_labels(self, sizes: Sequence[int]) -> tuple[int, ...]:
        return tuple(self.new_label(int(n)) for n in sizes)

    def check_labels(self, labels: Sequence[int], sizes: Sequence[int]) -> None:
        if len(labels) != len(sizes):
            raise ValueError("label/mode mismatch")
        for label, size in zip(labels, sizes):
            label, size = int(label), int(size)
            old = self._sizes.get(label)
            if old is None:
                self._sizes[label] = size
            elif old != size:
                raise ValueError(f"edge {label} has incompatible sizes {old} and {size}")

    def add(self, tensor: torch.Tensor, labels: Sequence[int]) -> None:
        tensor = torch.as_tensor(tensor)
        labels = tuple(int(x) for x in labels)
        if tensor.ndim != len(labels):
            raise ValueError(
                f"tensor rank {tensor.ndim} does not match {len(labels)} labels"
            )
        self.check_labels(labels, tensor.shape)
        # Size-one virtual modes are algebraically trivial.  Removing them
        # eagerly is essential for long histories: otherwise a perfectly
        # small rank-one network can exceed PyTorch's 64-axis tensor limit
        # purely because of bookkeeping axes.
        squeeze_axes = tuple(i for i, n in enumerate(tensor.shape) if int(n) == 1)
        if squeeze_axes:
            tensor = tensor.squeeze(dim=squeeze_axes)
            labels = tuple(x for i, x in enumerate(labels) if i not in squeeze_axes)
        self.operands.append((tensor, labels))

    @staticmethod
    def _ordered_union(a: Sequence[int], b: Sequence[int]) -> tuple[int, ...]:
        seen: set[int] = set()
        out: list[int] = []
        for x in (*a, *b):
            if x not in seen:
                seen.add(x)
                out.append(int(x))
        return tuple(out)

    @staticmethod
    def _align_tensor(
        tensor: torch.Tensor,
        labels: Sequence[int],
        target_labels: Sequence[int],
    ) -> torch.Tensor:
        """Permute/unsqueeze ``tensor`` onto ``target_labels`` axes."""
        labels = tuple(int(x) for x in labels)
        target_labels = tuple(int(x) for x in target_labels)
        if len(set(labels)) != len(labels):
            raise ValueError("repeated labels inside one factor are unsupported")
        present = [x for x in target_labels if x in set(labels)]
        perm = [labels.index(x) for x in present]
        if perm != list(range(len(labels))):
            tensor = tensor.permute(perm)
        shape = []
        j = 0
        present_set = set(present)
        for x in target_labels:
            if x in present_set:
                shape.append(int(tensor.shape[j]))
                j += 1
            else:
                shape.append(1)
        return tensor.reshape(shape)

    def contract(self, output_labels: Sequence[int] = ()) -> torch.Tensor:
        """Contract by eliminating one index at a time.

        This is a local variable-elimination scheduler, not a global einsum.
        For each non-output label it gathers only the factors incident to that
        label, multiplies those small factors with broadcasting, and immediately
        sums the label.  Choosing the smallest resulting bucket avoids the bad
        contraction orders that can otherwise turn a bounded-width nested
        network into exponentially large intermediates (notably for reverse
        matvec histories).
        """
        output_labels = tuple(int(x) for x in output_labels)
        if not self.operands:
            raise ValueError("cannot contract an empty graph")

        ops = list(self.operands)
        initial_operands = len(ops)
        max_numel = max(int(t.numel()) for t, _ in ops)
        max_ndim = max(int(t.ndim) for t, _ in ops)
        pair_steps = 0

        while True:
            incident: dict[int, list[int]] = {}
            for idx, (_, labels) in enumerate(ops):
                for label in labels:
                    if label not in output_labels:
                        incident.setdefault(label, []).append(idx)
            if not incident:
                break

            best = None
            for label, idxs in incident.items():
                union: list[int] = []
                seen: set[int] = set()
                for idx in idxs:
                    for x in ops[idx][1]:
                        if x not in seen:
                            seen.add(x)
                            union.append(int(x))
                union_t = tuple(union)
                out_labels = tuple(x for x in union_t if x != label)
                out_numel = _numel_from_labels(out_labels, self._sizes)
                joint_numel = _numel_from_labels(union_t, self._sizes)
                # Minimize the post-elimination factor first.  The remaining
                # tie breakers favor small temporary products and eliminate
                # larger buckets when the widths are otherwise equal.
                key = (out_numel, joint_numel, -len(idxs), len(out_labels))
                if best is None or key < best[0]:
                    best = (key, label, tuple(idxs), union_t, out_labels)

            assert best is not None
            _, label, idxs, union_labels, new_labels = best
            idx_set = set(idxs)

            product = None
            for idx in idxs:
                tensor, labels = ops[idx]
                aligned = self._align_tensor(tensor, labels, union_labels)
                if product is None:
                    product = aligned
                else:
                    product = product * aligned
                    pair_steps += 1
                max_numel = max(max_numel, int(product.numel()))
                max_ndim = max(max_ndim, int(product.ndim))

            assert product is not None
            product = product.sum(dim=union_labels.index(label))
            max_numel = max(max_numel, int(product.numel()))
            max_ndim = max(max_ndim, int(product.ndim))

            ops = [op for idx, op in enumerate(ops) if idx not in idx_set]
            ops.append((product, new_labels))

        # Only requested output labels remain.  Disconnected factors are
        # multiplied by broadcasting onto the common output axes.
        result = None
        for tensor, labels in ops:
            aligned = self._align_tensor(tensor, labels, output_labels)
            if result is None:
                result = aligned
            else:
                result = result * aligned
                pair_steps += 1
            max_numel = max(max_numel, int(result.numel()))
            max_ndim = max(max_ndim, int(result.ndim))

        assert result is not None
        self.last_stats = ContractionStats(
            operands=initial_operands,
            pair_contractions=pair_steps,
            max_intermediate_numel=max_numel,
            max_intermediate_ndim=max_ndim,
        )
        return result


# =============================================================================
# Recursive bank of operators
# =============================================================================


class _MPOBank:
    bank_shape: tuple[int, ...]
    row_dim: int
    col_dim: int
    row_modes: tuple[int, ...]
    col_modes: tuple[int, ...]
    batch_size: int = 1

    @property
    def leaf_count(self) -> int:
        raise NotImplementedError

    def leaves(self) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def emit(
        self,
        builder: _NetworkBuilder,
        *,
        bank_labels,
        row_labels,
        col_labels,
        batch_label: int | None = None,
    ) -> None:
        raise NotImplementedError

    def materialize(self) -> torch.Tensor:
        builder = _NetworkBuilder()
        batch = builder.new_label(self.batch_size) if self.batch_size > 1 else None
        b = builder.new_labels(self.bank_shape)
        r = builder.new_labels(self.row_modes)
        c = builder.new_labels(self.col_modes)
        self.emit(
            builder,
            bank_labels=b,
            row_labels=r,
            col_labels=c,
            batch_label=batch,
        )
        output = ((batch,) if batch is not None else ()) + tuple(b) + tuple(r) + tuple(c)
        value = builder.contract(output)
        shape = (*((self.batch_size,) if self.batch_size > 1 else ()), *self.bank_shape, self.row_dim, self.col_dim)
        return value.reshape(shape)


class _LeafCursor:
    def __init__(self, tensors: Sequence[torch.Tensor]) -> None:
        self.tensors = tuple(torch.as_tensor(x) for x in tensors)
        self.i = 0

    def take(self, shape: Sequence[int]) -> torch.Tensor:
        if self.i >= len(self.tensors):
            raise ValueError("not enough leaf tensors")
        x = self.tensors[self.i]
        expected = tuple(int(n) for n in shape)
        if tuple(x.shape) != expected:
            raise ValueError(
                f"leaf tensor {self.i} has shape {tuple(x.shape)}, expected {expected}"
            )
        self.i += 1
        return x


class _StaticMPOBank(_MPOBank):
    batch_size = 1

    def __init__(
        self,
        *,
        kind: str,
        depth: int,
        d: int,
        ranks: Sequence[int],
        bank_shape: Sequence[int],
        bank_roles: Sequence[str],
        row_dim: int,
        col_dim: int,
        children: Sequence[_MPOBank] | None = None,
        leaf: NestedTTVectorLeaf | NestedTTMatrixLeaf | None = None,
    ) -> None:
        self.kind = str(kind)
        self.depth = int(depth)
        self.d = int(d)
        self.ranks = tuple(int(r) for r in ranks)
        self.bank_shape = tuple(int(n) for n in bank_shape)
        self.bank_roles = tuple(str(x) for x in bank_roles)
        self.row_dim = int(row_dim)
        self.col_dim = int(col_dim)
        if len(self.bank_shape) != len(self.bank_roles):
            raise ValueError("bank_shape/bank_roles mismatch")

        if self.depth == 0:
            if leaf is None:
                raise ValueError("terminal bank requires a leaf")
            self._leaf = leaf
            self._children = ()
            self.row_modes = leaf.row_modes
            self.col_modes = leaf.col_modes
            if leaf.row_dim != self.row_dim or leaf.col_dim != self.col_dim:
                raise ValueError("leaf matrix dimensions do not match bank")
            return

        self._leaf = None
        self.row_modes = _balanced_factor_modes(self.row_dim, self.d)
        self.col_modes = _balanced_factor_modes(self.col_dim, self.d)
        q = self.ranks[self.depth - 1]
        bonds = _uniform_bonds(self.d, q)
        if children is None or len(children) != self.d:
            raise ValueError("recursive bank requires d children")
        self._children = tuple(children)
        for k, child in enumerate(self._children):
            if child.row_dim != bonds[k + 1] or child.col_dim != bonds[k]:
                raise ValueError("recursive virtual dimensions are inconsistent")

    @property
    def leaf_count(self) -> int:
        if self.depth == 0:
            return len(self._leaf.tensors())  # type: ignore[union-attr]
        return sum(c.leaf_count for c in self._children)

    def leaves(self) -> tuple[torch.Tensor, ...]:
        if self.depth == 0:
            return self._leaf.tensors()  # type: ignore[union-attr]
        return tuple(x for child in self._children for x in child.leaves())

    def emit(
        self,
        builder: _NetworkBuilder,
        *,
        bank_labels,
        row_labels,
        col_labels,
        batch_label: int | None = None,
    ) -> None:
        # Static trainable parameters have no batch axis.  ``batch_label`` is
        # intentionally ignored; batched wrappers attach the shared label.
        bank_labels = tuple(bank_labels)
        row_labels = tuple(row_labels)
        col_labels = tuple(col_labels)
        builder.check_labels(bank_labels, self.bank_shape)
        builder.check_labels(row_labels, self.row_modes)
        builder.check_labels(col_labels, self.col_modes)

        if self.depth == 0:
            if self.kind == "vector":
                assert isinstance(self._leaf, NestedTTVectorLeaf)
                self._leaf.emit(
                    builder,
                    bank_labels=bank_labels,
                    row_labels=row_labels,
                    col_labels=col_labels,
                )
            else:
                assert isinstance(self._leaf, NestedTTMatrixLeaf)
                row_bank_labels = tuple(
                    label
                    for label, role in zip(bank_labels, self.bank_roles)
                    if role == "out"
                )
                col_bank_labels = tuple(
                    label
                    for label, role in zip(bank_labels, self.bank_roles)
                    if role == "in"
                )
                self._leaf.emit(
                    builder,
                    row_bank_labels=row_bank_labels,
                    col_bank_labels=col_bank_labels,
                    row_labels=row_labels,
                    col_labels=col_labels,
                )
            return

        left_virtual = builder.new_labels(self._children[0].col_modes)
        for k, child in enumerate(self._children):
            right_virtual = builder.new_labels(child.row_modes)
            if k < self.d - 1 and tuple(child.row_modes) != tuple(self._children[k + 1].col_modes):
                raise ValueError("adjacent recursive virtual tensorizations differ")
            child.emit(
                builder,
                bank_labels=(*bank_labels, row_labels[k], col_labels[k]),
                row_labels=right_virtual,
                col_labels=left_virtual,
                batch_label=None,
            )
            left_virtual = right_virtual


class _PhysicalScaleBank(_MPOBank):
    """Multiply an outer vector-core bank by unbatched or batched weights."""

    def __init__(self, base: _MPOBank, weights: torch.Tensor) -> None:
        if len(base.bank_shape) != 1:
            raise ValueError("physical scaling is only valid on an outer vector core")
        weights = torch.as_tensor(weights)
        if weights.ndim == 2 and int(weights.shape[0]) == 1:
            weights = weights[0]
        if weights.ndim not in (1, 2) or int(weights.shape[-1]) != base.bank_shape[0]:
            raise ValueError(
                f"basis factor must have shape ({base.bank_shape[0]},) or "
                f"(batch,{base.bank_shape[0]}), got {tuple(weights.shape)}"
            )
        weight_batch = 1 if weights.ndim == 1 else int(weights.shape[0])
        self.base = base
        self.weights = weights
        self.batch_size = _broadcast_batch_sizes(base.batch_size, weight_batch)
        self.bank_shape = base.bank_shape
        self.row_dim, self.col_dim = base.row_dim, base.col_dim
        self.row_modes, self.col_modes = base.row_modes, base.col_modes

    @property
    def leaf_count(self) -> int:
        return self.base.leaf_count

    def leaves(self) -> tuple[torch.Tensor, ...]:
        return self.base.leaves()

    def emit(
        self,
        builder: _NetworkBuilder,
        *,
        bank_labels,
        row_labels,
        col_labels,
        batch_label: int | None = None,
    ) -> None:
        if self.batch_size > 1 and batch_label is None:
            raise ValueError("batched vector bank requires a batch label")
        self.base.emit(
            builder,
            bank_labels=bank_labels,
            row_labels=row_labels,
            col_labels=col_labels,
            batch_label=batch_label if self.base.batch_size > 1 else None,
        )
        leaves = self.base.leaves()
        w = self.weights
        if leaves:
            w = w.to(dtype=leaves[0].dtype, device=leaves[0].device)
        physical = int(tuple(bank_labels)[0])
        if w.ndim == 1:
            builder.add(w, (physical,))
        else:
            assert batch_label is not None
            builder.add(w, (batch_label, physical))


class _PhysicalMatrixScaleBank(_MPOBank):
    """Multiply an outer matrix-core bank by separable matrix weights."""

    def __init__(self, base: _MPOBank, weights: torch.Tensor) -> None:
        if len(base.bank_shape) != 2:
            raise ValueError("matrix physical scaling requires a bank (M,N)")
        weights = torch.as_tensor(weights)
        if weights.ndim == 3 and int(weights.shape[0]) == 1:
            weights = weights[0]
        if weights.ndim not in (2, 3) or tuple(weights.shape[-2:]) != base.bank_shape:
            raise ValueError(
                f"matrix factor must have shape {base.bank_shape} or "
                f"(batch,{base.bank_shape[0]},{base.bank_shape[1]}), got {tuple(weights.shape)}"
            )
        weight_batch = 1 if weights.ndim == 2 else int(weights.shape[0])
        self.base = base
        self.weights = weights
        self.batch_size = _broadcast_batch_sizes(base.batch_size, weight_batch)
        self.bank_shape = base.bank_shape
        self.row_dim, self.col_dim = base.row_dim, base.col_dim
        self.row_modes, self.col_modes = base.row_modes, base.col_modes

    @property
    def leaf_count(self) -> int:
        return self.base.leaf_count

    def leaves(self) -> tuple[torch.Tensor, ...]:
        return self.base.leaves()

    def emit(
        self,
        builder: _NetworkBuilder,
        *,
        bank_labels,
        row_labels,
        col_labels,
        batch_label: int | None = None,
    ) -> None:
        if self.batch_size > 1 and batch_label is None:
            raise ValueError("batched matrix bank requires a batch label")
        self.base.emit(
            builder,
            bank_labels=bank_labels,
            row_labels=row_labels,
            col_labels=col_labels,
            batch_label=batch_label if self.base.batch_size > 1 else None,
        )
        leaves = self.base.leaves()
        w = self.weights
        if leaves:
            w = w.to(dtype=leaves[0].dtype, device=leaves[0].device)
        j, i = (int(x) for x in tuple(bank_labels))
        if w.ndim == 2:
            builder.add(w, (j, i))
        else:
            assert batch_label is not None
            builder.add(w, (batch_label, j, i))


class _PhysicalLinearMapBank(_MPOBank):
    """Apply a rank-one outer MPO core directly to a vector-core bank.

    ``H[j] = sum_i weights[j,i] * base[i]``.  Because there is no nontrivial
    MPO virtual bond, the nested hidden-state modes and ranks are unchanged.
    """

    def __init__(self, base: _MPOBank, weights: torch.Tensor) -> None:
        if len(base.bank_shape) != 1:
            raise ValueError("physical linear map requires a vector bank")
        weights = torch.as_tensor(weights)
        if weights.ndim == 3 and int(weights.shape[0]) == 1:
            weights = weights[0]
        if weights.ndim not in (2, 3) or int(weights.shape[-1]) != base.bank_shape[0]:
            raise ValueError("linear-map factor has incompatible shape")
        weight_batch = 1 if weights.ndim == 2 else int(weights.shape[0])
        self.base = base
        self.weights = weights
        self.batch_size = _broadcast_batch_sizes(base.batch_size, weight_batch)
        self.bank_shape = (int(weights.shape[-2]),)
        self.row_dim, self.col_dim = base.row_dim, base.col_dim
        self.row_modes, self.col_modes = base.row_modes, base.col_modes

    @property
    def leaf_count(self) -> int:
        return self.base.leaf_count

    def leaves(self) -> tuple[torch.Tensor, ...]:
        return self.base.leaves()

    def emit(
        self,
        builder: _NetworkBuilder,
        *,
        bank_labels,
        row_labels,
        col_labels,
        batch_label: int | None = None,
    ) -> None:
        if self.batch_size > 1 and batch_label is None:
            raise ValueError("batched physical map requires a batch label")
        j = int(tuple(bank_labels)[0])
        i = builder.new_label(self.base.bank_shape[0])
        self.base.emit(
            builder,
            bank_labels=(i,),
            row_labels=row_labels,
            col_labels=col_labels,
            batch_label=batch_label if self.base.batch_size > 1 else None,
        )
        leaves = self.base.leaves()
        w = self.weights
        if leaves:
            w = w.to(dtype=leaves[0].dtype, device=leaves[0].device)
        if w.ndim == 2:
            builder.add(w, (j, i))
        else:
            assert batch_label is not None
            builder.add(w, (batch_label, j, i))


class _MatvecMPOBank(_MPOBank):
    """Exact symbolic ``H[j] = sum_i A[j,i] kron G[i]``."""

    def __init__(self, matrix_bank: _MPOBank, vector_bank: _MPOBank) -> None:
        if len(matrix_bank.bank_shape) != 2 or len(vector_bank.bank_shape) != 1:
            raise ValueError("matvec expects matrix bank (M,N) and vector bank (N,)")
        M, N = matrix_bank.bank_shape
        if vector_bank.bank_shape[0] != N:
            raise ValueError("physical input mode mismatch")
        self.matrix_bank = matrix_bank
        self.vector_bank = vector_bank
        self.batch_size = _broadcast_batch_sizes(matrix_bank.batch_size, vector_bank.batch_size)
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
        bank_labels,
        row_labels,
        col_labels,
        batch_label: int | None = None,
    ) -> None:
        if self.batch_size > 1 and batch_label is None:
            raise ValueError("batched matvec bank requires a batch label")
        bank_labels = tuple(bank_labels)
        row_labels = tuple(row_labels)
        col_labels = tuple(col_labels)
        nr = len(self.matrix_bank.row_modes)
        nc = len(self.matrix_bank.col_modes)
        i = builder.new_label(self.matrix_bank.bank_shape[1])
        self.matrix_bank.emit(
            builder,
            bank_labels=(bank_labels[0], i),
            row_labels=row_labels[:nr],
            col_labels=col_labels[:nc],
            batch_label=batch_label if self.matrix_bank.batch_size > 1 else None,
        )
        self.vector_bank.emit(
            builder,
            bank_labels=(i,),
            row_labels=row_labels[nr:],
            col_labels=col_labels[nc:],
            batch_label=batch_label if self.vector_bank.batch_size > 1 else None,
        )


class _RevMatvecMPOBank(_MPOBank):
    """Exact symbolic ``H[i] = sum_j A[j,i] kron G[j]``.

    This is the physical transpose action ``A.T @ G``.  The MPO virtual
    directions are not transposed: only the physical output/input bank indices
    are exchanged.  Consequently the tensorized hidden-state modes grow in the
    same way as forward matvec.
    """

    def __init__(self, matrix_bank: _MPOBank, vector_bank: _MPOBank) -> None:
        if len(matrix_bank.bank_shape) != 2 or len(vector_bank.bank_shape) != 1:
            raise ValueError("reverse matvec expects matrix bank (M,N) and vector bank (M,)")
        M, N = matrix_bank.bank_shape
        if vector_bank.bank_shape[0] != M:
            raise ValueError("physical output mode mismatch")
        self.matrix_bank = matrix_bank
        self.vector_bank = vector_bank
        self.batch_size = _broadcast_batch_sizes(matrix_bank.batch_size, vector_bank.batch_size)
        self.bank_shape = (N,)
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
        bank_labels,
        row_labels,
        col_labels,
        batch_label: int | None = None,
    ) -> None:
        if self.batch_size > 1 and batch_label is None:
            raise ValueError("batched reverse-matvec bank requires a batch label")
        bank_labels = tuple(bank_labels)
        row_labels = tuple(row_labels)
        col_labels = tuple(col_labels)
        nr = len(self.matrix_bank.row_modes)
        nc = len(self.matrix_bank.col_modes)
        j = builder.new_label(self.matrix_bank.bank_shape[0])
        self.matrix_bank.emit(
            builder,
            bank_labels=(j, bank_labels[0]),
            row_labels=row_labels[:nr],
            col_labels=col_labels[:nc],
            batch_label=batch_label if self.matrix_bank.batch_size > 1 else None,
        )
        self.vector_bank.emit(
            builder,
            bank_labels=(j,),
            row_labels=row_labels[nr:],
            col_labels=col_labels[nc:],
            batch_label=batch_label if self.vector_bank.batch_size > 1 else None,
        )


# =============================================================================
# Static recursive construction
# =============================================================================


def _leaf_shapes(
    *,
    kind: str,
    depth: int,
    d: int,
    ranks: Sequence[int],
    bank_shape: Sequence[int],
    bank_roles: Sequence[str],
    row_dim: int,
    col_dim: int,
    separation_rank: int,
) -> list[tuple[int, ...]]:
    if depth == 0:
        if kind == "vector":
            return list(NestedTTVectorLeaf.shapes(bank_shape, row_dim, col_dim, separation_rank))
        row_bank = tuple(n for n, r in zip(bank_shape, bank_roles) if r == "out")
        col_bank = tuple(n for n, r in zip(bank_shape, bank_roles) if r == "in")
        return list(NestedTTMatrixLeaf.shapes(row_bank, col_bank, row_dim, col_dim, separation_rank))

    row_modes = _balanced_factor_modes(row_dim, d)
    col_modes = _balanced_factor_modes(col_dim, d)
    q = int(ranks[depth - 1])
    bonds = _uniform_bonds(d, q)
    out: list[tuple[int, ...]] = []
    for k in range(d):
        out.extend(
            _leaf_shapes(
                kind=kind,
                depth=depth - 1,
                d=d,
                ranks=ranks,
                bank_shape=(*bank_shape, row_modes[k], col_modes[k]),
                bank_roles=(*bank_roles, "out", "in"),
                row_dim=bonds[k + 1],
                col_dim=bonds[k],
                separation_rank=separation_rank,
            )
        )
    return out


def _build_static_bank(
    *,
    kind: str,
    depth: int,
    d: int,
    ranks: Sequence[int],
    bank_shape: Sequence[int],
    bank_roles: Sequence[str],
    row_dim: int,
    col_dim: int,
    cursor: _LeafCursor,
    separation_rank: int,
    normalize_matrix_leaves: bool = False,
    stochastic_sum_axes: Sequence[int] = (),
) -> _StaticMPOBank:
    if depth == 0:
        if kind == "vector":
            sx, sc = NestedTTVectorLeaf.shapes(bank_shape, row_dim, col_dim, separation_rank)
            leaf = NestedTTVectorLeaf.from_tensors(
                bank_shape,
                row_dim,
                col_dim,
                cursor.take(sx),
                cursor.take(sc),
                separation_rank=separation_rank,
            )
        else:
            row_bank = tuple(n for n, r in zip(bank_shape, bank_roles) if r == "out")
            col_bank = tuple(n for n, r in zip(bank_shape, bank_roles) if r == "in")
            sl, sr = NestedTTMatrixLeaf.shapes(row_bank, col_bank, row_dim, col_dim, separation_rank)
            leaf = NestedTTMatrixLeaf.from_tensors(
                row_bank,
                col_bank,
                row_dim,
                col_dim,
                cursor.take(sl),
                cursor.take(sr),
                separation_rank=separation_rank,
            )
            if normalize_matrix_leaves:
                sum_set = set(int(a) for a in stochastic_sum_axes)
                out_globals = [i for i, role in enumerate(bank_roles) if role == "out"]
                in_globals = [i for i, role in enumerate(bank_roles) if role == "in"]
                sum_row = tuple(k for k, g in enumerate(out_globals) if g in sum_set)
                sum_col = tuple(k for k, g in enumerate(in_globals) if g in sum_set)
                leaf = leaf.normalized(
                    sum_row_bank_axes=sum_row,
                    sum_col_bank_axes=sum_col,
                )
        return _StaticMPOBank(
            kind=kind,
            depth=0,
            d=d,
            ranks=ranks,
            bank_shape=bank_shape,
            bank_roles=bank_roles,
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
                kind=kind,
                depth=depth - 1,
                d=d,
                ranks=ranks,
                bank_shape=(*bank_shape, row_modes[k], col_modes[k]),
                bank_roles=(*bank_roles, "out", "in"),
                row_dim=bonds[k + 1],
                col_dim=bonds[k],
                cursor=cursor,
                separation_rank=separation_rank,
                normalize_matrix_leaves=normalize_matrix_leaves,
                stochastic_sum_axes=(
                    (*tuple(int(a) for a in stochastic_sum_axes), len(tuple(bank_shape)))
                    if k == 0
                    else (len(tuple(bank_shape)),)
                ) if normalize_matrix_leaves else (),
            )
        )
    return _StaticMPOBank(
        kind=kind,
        depth=depth,
        d=d,
        ranks=ranks,
        bank_shape=bank_shape,
        bank_roles=bank_roles,
        row_dim=row_dim,
        col_dim=col_dim,
        children=children,
    )


# =============================================================================
# Whole-vector / whole-matrix emission
# =============================================================================


def _emit_vector_network(core_banks, modes, *, output_physical: bool):
    builder = _NetworkBuilder()
    batch_size = _broadcast_batch_sizes(*(bank.batch_size for bank in core_banks))
    batch = builder.new_label(batch_size) if batch_size > 1 else None
    physical = builder.new_labels(modes)
    output = (
        (*((batch,) if batch is not None else ()), *physical)
        if output_physical
        else ((batch,) if batch is not None else ())
    )

    left = builder.new_labels(core_banks[0].col_modes)
    for k, bank in enumerate(core_banks):
        right = builder.new_labels(bank.row_modes)
        bank.emit(
            builder,
            bank_labels=(physical[k],),
            row_labels=right,
            col_labels=left,
            batch_label=batch if bank.batch_size > 1 else None,
        )
        if k < len(core_banks) - 1:
            if tuple(bank.row_modes) != tuple(core_banks[k + 1].col_modes):
                raise ValueError("adjacent vector bond tensorizations do not match")
        left = right
    return builder, output, batch_size


def _emit_matrix_network(core_banks, row_modes, col_modes):
    builder = _NetworkBuilder()
    batch_size = _broadcast_batch_sizes(*(bank.batch_size for bank in core_banks))
    batch = builder.new_label(batch_size) if batch_size > 1 else None
    rows = builder.new_labels(row_modes)
    cols = builder.new_labels(col_modes)
    left = builder.new_labels(core_banks[0].col_modes)
    for k, bank in enumerate(core_banks):
        right = builder.new_labels(bank.row_modes)
        bank.emit(
            builder,
            bank_labels=(rows[k], cols[k]),
            row_labels=right,
            col_labels=left,
            batch_label=batch if bank.batch_size > 1 else None,
        )
        if k < len(core_banks) - 1:
            if tuple(bank.row_modes) != tuple(core_banks[k + 1].col_modes):
                raise ValueError("adjacent matrix bond tensorizations do not match")
        left = right
    output = (*((batch,) if batch is not None else ()), *rows, *cols)
    return builder, output, batch_size


# =============================================================================
# Rank-one physical factors
# =============================================================================


def _rank_one_factors(other, modes: Sequence[int]) -> tuple[torch.Tensor, ...]:
    """Return one physical factor per TT mode.

    Each factor may be ``(n,)`` or ``(batch,n)``.  Batched factors share one
    leading batch size and are broadcast against unbatched factors.
    Known rank-one :class:`NestedTTVector` objects expose these factors
    directly, so no dense recovery/factorization is ever attempted.
    """
    modes = tuple(int(n) for n in modes)
    known = getattr(other, "rank_one_factors", None)
    if known is not None:
        factors = tuple(torch.as_tensor(x) for x in known)
        if tuple(int(x.shape[-1]) for x in factors) != modes:
            raise ValueError("rank-one NestedTTVector modes do not match")
        _broadcast_batch_sizes(*(1 if x.ndim == 1 else int(x.shape[0]) for x in factors))
        return factors
    if isinstance(other, (int, float)):
        factors = [torch.ones(n) for n in modes]
        factors[0] = factors[0] * float(other)
        return tuple(factors)
    if isinstance(other, torch.Tensor):
        if other.ndim == 0:
            return _rank_one_factors(float(other), modes)
        if len(modes) == 1 and other.ndim in (1, 2) and int(other.shape[-1]) == modes[0]:
            return (other,)
        raise ValueError(
            "a dense multi-dimensional divisor/multiplier is intentionally not "
            "factorized here; pass one factor of shape (n,) or (batch,n) per TT mode"
        )
    if isinstance(other, Sequence):
        factors = tuple(torch.as_tensor(x) for x in other)
        if len(factors) != len(modes):
            raise ValueError("expected one rank-one factor per physical mode")
        batches = []
        for x, n in zip(factors, modes):
            if x.ndim not in (1, 2) or int(x.shape[-1]) != n:
                raise ValueError(
                    f"factor must have shape ({n},) or (batch,{n}), got {tuple(x.shape)}"
                )
            batches.append(1 if x.ndim == 1 else int(x.shape[0]))
        _broadcast_batch_sizes(*batches)
        return factors
    raise TypeError("expected scalar or a sequence of rank-one physical factors")


def _safe_reciprocal(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = torch.as_tensor(x)
    eps_t = torch.as_tensor(float(eps), dtype=x.dtype, device=x.device)
    sign = torch.where(x < 0, -torch.ones_like(x), torch.ones_like(x))
    safe = torch.where(x.abs() < eps_t, sign * eps_t, x)
    return safe.reciprocal()


# =============================================================================
# Public NestedTTVector
# =============================================================================


class NestedTTVector:
    def __init__(self, spec: NestedTTVectorSpec, leaves: Sequence[torch.Tensor]) -> None:
        expected = self.leaf_shapes(spec)
        if len(leaves) != len(expected):
            raise ValueError(f"expected {len(expected)} leaf tensors, got {len(leaves)}")
        cursor = _LeafCursor(leaves)
        bonds = _uniform_bonds(spec.d, spec.ranks[-1])
        banks = []
        for k, n in enumerate(spec.modes):
            banks.append(
                _build_static_bank(
                    kind="vector",
                    depth=spec.depth - 1,
                    d=spec.d,
                    ranks=spec.ranks,
                    bank_shape=(n,),
                    bank_roles=("vector",),
                    row_dim=bonds[k + 1],
                    col_dim=bonds[k],
                    cursor=cursor,
                    separation_rank=spec.separation_rank,
                )
            )
        if cursor.i != len(leaves):
            raise ValueError("too many vector leaf tensors")
        self.spec: NestedTTVectorSpec | None = spec
        self._modes = spec.modes
        self._depth = spec.depth
        self._core_banks = tuple(banks)
        self._history_length = 0
        self._last_contract_stats: ContractionStats | None = None
        self._physical_factors: tuple[torch.Tensor, ...] | None = None

    @classmethod
    def _from_banks(
        cls,
        *,
        modes,
        depth,
        core_banks,
        history_length,
        physical_factors: Sequence[torch.Tensor] | None = None,
    ):
        obj = cls.__new__(cls)
        obj.spec = None
        obj._modes = tuple(int(n) for n in modes)
        obj._depth = int(depth)
        obj._core_banks = tuple(core_banks)
        _broadcast_batch_sizes(*(b.batch_size for b in obj._core_banks))
        obj._history_length = int(history_length)
        obj._last_contract_stats = None
        obj._physical_factors = (
            None if physical_factors is None else tuple(torch.as_tensor(x) for x in physical_factors)
        )
        return obj

    @staticmethod
    def leaf_shapes(spec: NestedTTVectorSpec) -> tuple[tuple[int, ...], ...]:
        bonds = _uniform_bonds(spec.d, spec.ranks[-1])
        out: list[tuple[int, ...]] = []
        for k, n in enumerate(spec.modes):
            out.extend(
                _leaf_shapes(
                    kind="vector",
                    depth=spec.depth - 1,
                    d=spec.d,
                    ranks=spec.ranks,
                    bank_shape=(n,),
                    bank_roles=("vector",),
                    row_dim=bonds[k + 1],
                    col_dim=bonds[k],
                    separation_rank=spec.separation_rank,
                )
            )
        return tuple(out)

    @property
    def cores(self):
        return self._core_banks

    @property
    def modes(self) -> tuple[int, ...]:
        return self._modes

    @property
    def d(self) -> int:
        return len(self._modes)

    @property
    def n(self) -> int:
        return math.prod(self.modes)

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def batch_size(self) -> int:
        return _broadcast_batch_sizes(*(b.batch_size for b in self._core_banks))

    @property
    def history_length(self) -> int:
        return self._history_length

    @property
    def ranks(self) -> tuple[int, ...]:
        return (self._core_banks[0].col_dim,) + tuple(b.row_dim for b in self._core_banks)

    @property
    def shape(self) -> torch.Size:
        return torch.Size((self.n,)) if self.batch_size == 1 else torch.Size((self.batch_size, self.n))

    @property
    def leaf_count(self) -> int:
        return sum(b.leaf_count for b in self._core_banks)

    @property
    def leaves(self) -> tuple[torch.Tensor, ...]:
        return tuple(x for b in self._core_banks for x in b.leaves())

    @property
    def rank_one_factors(self) -> tuple[torch.Tensor, ...] | None:
        return self._physical_factors

    @property
    def last_contract_stats(self) -> ContractionStats | None:
        return self._last_contract_stats

    def materialize_cores(self) -> tuple[torch.Tensor, ...]:
        out = []
        for bank in self._core_banks:
            dense = bank.materialize()
            if bank.batch_size == 1:  # (n,out,in) -> (in,n,out)
                out.append(dense.permute(2, 0, 1).contiguous())
            else:  # (B,n,out,in) -> (B,in,n,out)
                out.append(dense.permute(0, 3, 1, 2).contiguous())
        return tuple(out)

    def to_dense(self) -> torch.Tensor:
        builder, output, batch_size = _emit_vector_network(
            self._core_banks, self.modes, output_physical=True
        )
        value = builder.contract(output)
        self._last_contract_stats = builder.last_stats
        return value.reshape(self.n) if batch_size == 1 else value.reshape(batch_size, self.n)

    def elementwise_multiply(self, other) -> "NestedTTVector":
        factors = _rank_one_factors(other, self.modes)
        banks = [
            _PhysicalScaleBank(bank, factor)
            for bank, factor in zip(self._core_banks, factors)
        ]
        physical_factors = None
        if self._physical_factors is not None:
            physical_factors = tuple(a * b for a, b in zip(self._physical_factors, factors))
        return NestedTTVector._from_banks(
            modes=self.modes,
            depth=self.depth,
            core_banks=banks,
            history_length=self.history_length,
            physical_factors=physical_factors,
        )

    def elementwise_divide(self, other, *, eps: float = 1e-12) -> "NestedTTVector":
        """Divide by a rank-one physical tensor without dense materialization.

        Only the supplied one-dimensional/batched physical factors are
        inverted.  Denominators with ``abs(x) < eps`` are replaced by a
        sign-preserving ``eps`` before the reciprocal is taken.
        """
        if eps <= 0:
            raise ValueError("eps must be positive")
        factors = _rank_one_factors(other, self.modes)
        inv = tuple(_safe_reciprocal(x, eps) for x in factors)
        return self.elementwise_multiply(inv)

    def __mul__(self, other) -> "NestedTTVector":
        return self.elementwise_multiply(other)

    def __rmul__(self, other) -> "NestedTTVector":
        return self.elementwise_multiply(other)

    def sum(self) -> torch.Tensor:
        builder, output, batch_size = _emit_vector_network(
            self._core_banks, self.modes, output_physical=False
        )
        value = builder.contract(output)
        self._last_contract_stats = builder.last_stats
        return value.reshape(()) if batch_size == 1 else value.reshape(batch_size)

    def __repr__(self) -> str:
        return (
            f"NestedTTVector(shape={tuple(self.shape)}, modes={self.modes}, "
            f"batch_size={self.batch_size}, ranks={self.ranks}, depth={self.depth}, "
            f"history_length={self.history_length}, leaves={self.leaf_count})"
        )


# =============================================================================
# Public NestedTTMatrix
# =============================================================================


class NestedTTMatrix:
    def __init__(
        self,
        spec: NestedTTMatrixSpec,
        leaves: Sequence[torch.Tensor],
        *,
        normalize_leaves: bool = False,
    ) -> None:
        expected = self.leaf_shapes(spec)
        if len(leaves) != len(expected):
            raise ValueError(f"expected {len(expected)} leaf tensors, got {len(leaves)}")
        cursor = _LeafCursor(leaves)
        bonds = _uniform_bonds(spec.d, spec.ranks[-1])
        banks = []
        for k, (m, n) in enumerate(zip(spec.row_modes, spec.col_modes)):
            banks.append(
                _build_static_bank(
                    kind="matrix",
                    depth=spec.depth - 1,
                    d=spec.d,
                    ranks=spec.ranks,
                    bank_shape=(m, n),
                    bank_roles=("out", "in"),
                    row_dim=bonds[k + 1],
                    col_dim=bonds[k],
                    cursor=cursor,
                    separation_rank=spec.separation_rank,
                    normalize_matrix_leaves=normalize_leaves,
                    stochastic_sum_axes=(1,) if normalize_leaves else (),
                )
            )
        if cursor.i != len(leaves):
            raise ValueError("too many matrix leaf tensors")
        self.spec = spec
        self._core_banks = tuple(banks)
        self._last_contract_stats: ContractionStats | None = None
        self._separable_cores: tuple[torch.Tensor, ...] | None = None

    @classmethod
    def _from_banks(
        cls,
        *,
        spec: NestedTTMatrixSpec,
        core_banks,
        separable_cores: Sequence[torch.Tensor] | None = None,
    ) -> "NestedTTMatrix":
        obj = cls.__new__(cls)
        obj.spec = spec
        obj._core_banks = tuple(core_banks)
        _broadcast_batch_sizes(*(b.batch_size for b in obj._core_banks))
        obj._last_contract_stats = None
        obj._separable_cores = (
            None if separable_cores is None else tuple(torch.as_tensor(x) for x in separable_cores)
        )
        return obj

    @staticmethod
    def leaf_shapes(spec: NestedTTMatrixSpec) -> tuple[tuple[int, ...], ...]:
        bonds = _uniform_bonds(spec.d, spec.ranks[-1])
        out: list[tuple[int, ...]] = []
        for k, (m, n) in enumerate(zip(spec.row_modes, spec.col_modes)):
            out.extend(
                _leaf_shapes(
                    kind="matrix",
                    depth=spec.depth - 1,
                    d=spec.d,
                    ranks=spec.ranks,
                    bank_shape=(m, n),
                    bank_roles=("out", "in"),
                    row_dim=bonds[k + 1],
                    col_dim=bonds[k],
                    separation_rank=spec.separation_rank,
                )
            )
        return tuple(out)

    @property
    def cores(self):
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
    def batch_size(self) -> int:
        return _broadcast_batch_sizes(*(b.batch_size for b in self._core_banks))

    @property
    def ranks(self) -> tuple[int, ...]:
        return (self._core_banks[0].col_dim,) + tuple(b.row_dim for b in self._core_banks)

    @property
    def shape(self) -> torch.Size:
        mn = (math.prod(self.row_modes), math.prod(self.col_modes))
        return torch.Size(mn) if self.batch_size == 1 else torch.Size((self.batch_size, *mn))

    @property
    def leaf_count(self) -> int:
        return sum(b.leaf_count for b in self._core_banks)

    @property
    def leaves(self) -> tuple[torch.Tensor, ...]:
        return tuple(x for b in self._core_banks for x in b.leaves())

    @property
    def separable_cores(self) -> tuple[torch.Tensor, ...] | None:
        return self._separable_cores

    @property
    def last_contract_stats(self) -> ContractionStats | None:
        return self._last_contract_stats

    def materialize_cores(self) -> tuple[torch.Tensor, ...]:
        out = []
        for bank in self._core_banks:
            dense = bank.materialize()
            if bank.batch_size == 1:  # (M,N,out,in) -> (in,M,N,out)
                out.append(dense.permute(3, 0, 1, 2).contiguous())
            else:  # (B,M,N,out,in) -> (B,in,M,N,out)
                out.append(dense.permute(0, 4, 1, 2, 3).contiguous())
        return tuple(out)

    def to_dense(self) -> torch.Tensor:
        builder, output, batch_size = _emit_matrix_network(
            self._core_banks, self.row_modes, self.col_modes
        )
        value = builder.contract(output)
        self._last_contract_stats = builder.last_stats
        m, n = math.prod(self.row_modes), math.prod(self.col_modes)
        return value.reshape(m, n) if batch_size == 1 else value.reshape(batch_size, m, n)

    def elementwise_multiply(self, factors: Sequence[torch.Tensor]) -> "NestedTTMatrix":
        factors = tuple(torch.as_tensor(x) for x in factors)
        if len(factors) != self.d:
            raise ValueError("expected one separable matrix factor per physical mode")
        batches = []
        for k, (factor, m, n) in enumerate(zip(factors, self.row_modes, self.col_modes)):
            if factor.ndim not in (2, 3) or tuple(factor.shape[-2:]) != (m, n):
                raise ValueError(
                    f"matrix factor {k} must have shape ({m},{n}) or (batch,{m},{n}), "
                    f"got {tuple(factor.shape)}"
                )
            batches.append(1 if factor.ndim == 2 else int(factor.shape[0]))
        _broadcast_batch_sizes(self.batch_size, *batches)
        banks = [
            _PhysicalMatrixScaleBank(bank, factor)
            for bank, factor in zip(self._core_banks, factors)
        ]
        sep = None
        if self._separable_cores is not None:
            sep = tuple(a * b for a, b in zip(self._separable_cores, factors))
        return NestedTTMatrix._from_banks(spec=self.spec, core_banks=banks, separable_cores=sep)

    def matvec(self, x: NestedTTVector) -> NestedTTVector:
        if not isinstance(x, NestedTTVector):
            raise TypeError("matvec requires NestedTTVector")
        if x.modes != self.col_modes:
            raise ValueError("matrix column modes do not match vector modes")
        if x.depth != self.depth or x.d != self.d:
            raise ValueError("matrix/vector nested structures are incompatible")
        _broadcast_batch_sizes(self.batch_size, x.batch_size)

        # A separable outer MPO has bond rank one, so it acts only on each
        # physical mode and must not create another nested history/rank mode.
        if self._separable_cores is not None:
            banks = [
                _PhysicalLinearMapBank(g, W)
                for g, W in zip(x._core_banks, self._separable_cores)
            ]
            return NestedTTVector._from_banks(
                modes=self.row_modes,
                depth=self.depth,
                core_banks=banks,
                history_length=x.history_length,
            )

        banks = [_MatvecMPOBank(A, g) for A, g in zip(self._core_banks, x._core_banks)]
        return NestedTTVector._from_banks(
            modes=self.row_modes,
            depth=self.depth,
            core_banks=banks,
            history_length=x.history_length + 1,
        )

    def rev_matvec(self, x: NestedTTVector) -> NestedTTVector:
        """Apply the physical transpose without materializing ``self.T``.

        Computes ``self.T @ x`` directly.  For a general nested matrix this
        contracts the matrix output bank index with the vector physical index
        and leaves the matrix input index open.  For a separable/rank-one outer
        MPO, each local physical map is simply transposed, so no nested history
        mode is introduced.
        """
        if not isinstance(x, NestedTTVector):
            raise TypeError("rev_matvec requires NestedTTVector")
        if x.modes != self.row_modes:
            raise ValueError("matrix row modes do not match vector modes")
        if x.depth != self.depth or x.d != self.d:
            raise ValueError("matrix/vector nested structures are incompatible")
        _broadcast_batch_sizes(self.batch_size, x.batch_size)

        # Physical transpose of a separable outer MPO remains separable:
        # H[i] = sum_j W[j,i] G[j] = (W.T G)[i].
        if self._separable_cores is not None:
            banks = [
                _PhysicalLinearMapBank(g, W.transpose(-2, -1))
                for g, W in zip(x._core_banks, self._separable_cores)
            ]
            return NestedTTVector._from_banks(
                modes=self.col_modes,
                depth=self.depth,
                core_banks=banks,
                history_length=x.history_length,
            )

        banks = [
            _RevMatvecMPOBank(A, g)
            for A, g in zip(self._core_banks, x._core_banks)
        ]
        return NestedTTVector._from_banks(
            modes=self.col_modes,
            depth=self.depth,
            core_banks=banks,
            history_length=x.history_length + 1,
        )

    def __matmul__(self, x: NestedTTVector) -> NestedTTVector:
        return self.matvec(x)

    def __mul__(self, other) -> "NestedTTMatrix":
        return self.elementwise_multiply(other)

    def __rmul__(self, other) -> "NestedTTMatrix":
        return self.elementwise_multiply(other)

    def sum(self) -> torch.Tensor:
        builder, output, batch_size = _emit_matrix_network(
            self._core_banks, self.row_modes, self.col_modes
        )
        value = builder.contract(output[:1] if batch_size > 1 else ())
        self._last_contract_stats = builder.last_stats
        return value.reshape(()) if batch_size == 1 else value.reshape(batch_size)

    def __repr__(self) -> str:
        return (
            f"NestedTTMatrix(shape={tuple(self.shape)}, "
            f"modes={list(zip(self.row_modes, self.col_modes))}, "
            f"batch_size={self.batch_size}, ranks={self.ranks}, "
            f"depth={self.depth}, leaves={self.leaf_count})"
        )


# =============================================================================
# Convenience constructors used by model code
# =============================================================================


def _hierarchy(depth: int, ranks: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(ranks, int):
        return (int(ranks),) * int(depth)
    values = tuple(int(r) for r in ranks)
    if len(values) != int(depth):
        raise ValueError(f"expected {depth} hierarchy ranks, got {values}")
    return values


def ones_nested_tt_vector(
    modes: Sequence[int],
    *,
    depth: int = 1,
    ranks: int | Sequence[int] = 1,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> NestedTTVector:
    modes = tuple(int(n) for n in modes)
    hierarchy = _hierarchy(depth, ranks)
    if any(r != 1 for r in hierarchy):
        raise ValueError("ones_nested_tt_vector requires hierarchy ranks equal to one")
    spec = NestedTTVectorSpec(modes=modes, depth=int(depth), ranks=hierarchy, separation_rank=1)
    leaves = [
        torch.ones(shape, dtype=dtype, device=device)
        for shape in NestedTTVector.leaf_shapes(spec)
    ]
    v = NestedTTVector(spec, leaves)
    v._physical_factors = tuple(
        torch.ones(n, dtype=dtype, device=device) for n in modes
    )
    return v


def nested_tt_vector_from_factors(
    factors: Sequence[torch.Tensor], *, depth: int = 1
) -> NestedTTVector:
    raw = tuple(torch.as_tensor(x) for x in factors)
    if not raw:
        raise ValueError("at least one rank-one factor is required")
    modes = tuple(int(x.shape[-1]) for x in raw)
    factors = _rank_one_factors(raw, modes)
    first = factors[0]
    v = ones_nested_tt_vector(
        modes,
        depth=depth,
        ranks=1,
        dtype=first.dtype,
        device=first.device,
    )
    return v.elementwise_multiply(factors)


def rank_one_factors_from_nested(v: NestedTTVector) -> tuple[torch.Tensor, ...]:
    """Return known separable physical factors without dense factorization."""
    if not isinstance(v, NestedTTVector):
        raise TypeError("expected NestedTTVector")
    if v.rank_one_factors is None:
        raise ValueError(
            "this NestedTTVector is not known to be a rank-one physical tensor; "
            "refusing to densify/factorize it"
        )
    return v.rank_one_factors


def nested_tt_matrix_from_separable_cores(
    cores: Sequence[torch.Tensor], *, depth: int = 1
) -> NestedTTMatrix:
    """Create a rank-one outer MPO from local dense matrix factors.

    Each core may be ``(m,n)`` or ``(batch,m,n)``.  The returned nested matrix
    keeps those factors symbolic; no global Kronecker matrix is materialized.
    """
    cores = tuple(torch.as_tensor(x) for x in cores)
    if not cores:
        raise ValueError("at least one separable matrix core is required")
    row_modes = tuple(int(x.shape[-2]) for x in cores)
    col_modes = tuple(int(x.shape[-1]) for x in cores)
    batches = []
    for x in cores:
        if x.ndim not in (2, 3):
            raise ValueError("separable matrix cores must have shape (m,n) or (batch,m,n)")
        batches.append(1 if x.ndim == 2 else int(x.shape[0]))
    _broadcast_batch_sizes(*batches)
    spec = NestedTTMatrixSpec(
        row_modes=row_modes,
        col_modes=col_modes,
        depth=int(depth),
        ranks=(1,) * int(depth),
        separation_rank=1,
    )
    first = cores[0]
    leaves = [
        torch.ones(shape, dtype=first.dtype, device=first.device)
        for shape in NestedTTMatrix.leaf_shapes(spec)
    ]
    base = NestedTTMatrix(spec, leaves)
    banks = tuple(
        _PhysicalMatrixScaleBank(bank, core)
        for bank, core in zip(base._core_banks, cores)
    )
    return NestedTTMatrix._from_banks(spec=spec, core_banks=banks, separable_cores=cores)

