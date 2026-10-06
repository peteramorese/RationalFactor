import torch

from rational_factor.models.tt.nested_tt import (
    NestedTTVectorSpec,
    NestedTTMatrixSpec,
    NestedTTVector,
    NestedTTMatrix,
)


def make_leaves(shapes, *, seed, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randn(*shape, generator=g, dtype=torch.float64).requires_grad_(requires_grad)
        for shape in shapes
    ]


def rank_one_dense(factors):
    out = factors[0]
    for factor in factors[1:]:
        out = torch.kron(out, factor)
    return out


def check_case(depth: int, d: int = 2, steps: int = 2):
    modes = (2,) * d
    ranks = tuple([1] + [2] * (depth - 1))
    vspec = NestedTTVectorSpec(modes=modes, depth=depth, ranks=ranks)
    mspec = NestedTTMatrixSpec(
        row_modes=modes,
        col_modes=modes,
        depth=depth,
        ranks=ranks,
    )

    v = NestedTTVector(
        vspec,
        make_leaves(NestedTTVector.leaf_shapes(vspec), seed=1000 + depth),
    )
    M = NestedTTMatrix(
        mspec,
        make_leaves(NestedTTMatrix.leaf_shapes(mspec), seed=2000 + depth),
    )

    assert v.leaf_count == d ** depth
    assert M.leaf_count == d ** depth

    Md = M.to_dense()
    ref = v.to_dense()
    cur = v

    errors = []
    for step in range(1, steps + 1):
        cur = M.matvec(cur)
        ref = Md @ ref
        got = cur.to_dense()
        torch.testing.assert_close(got, ref, rtol=1e-10, atol=1e-10)
        errors.append(float((got - ref).abs().max()))
        assert cur.leaf_count == v.leaf_count + step * M.leaf_count

    return {
        "depth": depth,
        "d": d,
        "initial_leaves": v.leaf_count,
        "after_steps_leaves": cur.leaf_count,
        "flattened_ranks": cur.ranks,
        "errors": errors,
    }


def sum_without_materialization_check(depth: int = 3, d: int = 2):
    modes = (2,) * d
    ranks = tuple([1] + [2] * (depth - 1))
    vspec = NestedTTVectorSpec(modes=modes, depth=depth, ranks=ranks)
    mspec = NestedTTMatrixSpec(
        row_modes=modes,
        col_modes=modes,
        depth=depth,
        ranks=ranks,
    )

    v = NestedTTVector(
        vspec,
        make_leaves(NestedTTVector.leaf_shapes(vspec), seed=3001),
    )
    M = NestedTTMatrix(
        mspec,
        make_leaves(NestedTTMatrix.leaf_shapes(mspec), seed=3002),
    )

    # Two recurrent steps: flattened outer rank is already 8 for rank-2 M/v.
    vt = M.matvec(M.matvec(v))
    dense = vt.to_dense()

    # If sum() accidentally tries to materialize a parent core, this guard
    # makes the test fail immediately.
    originals = []
    try:
        for bank in vt.cores:
            originals.append((bank, bank.materialize))

            def fail_materialize(*args, **kwargs):
                raise AssertionError("sum() attempted to materialize a nested core")

            bank.materialize = fail_materialize

        got = vt.sum()
    finally:
        for bank, original in originals:
            bank.materialize = original

    torch.testing.assert_close(got, dense.sum(), rtol=1e-10, atol=1e-10)

    # Rank-one basis weighting must also remain fully symbolic during sum().
    factors = [
        torch.randn(n, generator=torch.Generator().manual_seed(4000 + k), dtype=torch.float64)
        for k, n in enumerate(modes)
    ]
    weighted = vt.elementwise_multiply(factors)
    basis = rank_one_dense(factors)

    originals = []
    try:
        for bank in weighted.cores:
            originals.append((bank, bank.materialize))

            def fail_materialize(*args, **kwargs):
                raise AssertionError("weighted sum() materialized a nested core")

            bank.materialize = fail_materialize
        weighted_sum = weighted.sum()
    finally:
        for bank, original in originals:
            bank.materialize = original

    torch.testing.assert_close(
        weighted_sum,
        (dense * basis).sum(),
        rtol=1e-10,
        atol=1e-10,
    )

    return {
        "depth": depth,
        "d": d,
        "history_length": vt.history_length,
        "leaves": vt.leaf_count,
        "flattened_ranks": vt.ranks,
        "sum": float(got),
        "weighted_sum": float(weighted_sum),
    }


def gradient_check(depth: int = 3, d: int = 2):
    modes = (2,) * d
    ranks = tuple([1] + [2] * (depth - 1))
    vspec = NestedTTVectorSpec(modes=modes, depth=depth, ranks=ranks)
    mspec = NestedTTMatrixSpec(
        row_modes=modes,
        col_modes=modes,
        depth=depth,
        ranks=ranks,
    )

    vleaves = make_leaves(
        NestedTTVector.leaf_shapes(vspec), seed=5001, requires_grad=True
    )
    mleaves = make_leaves(
        NestedTTMatrix.leaf_shapes(mspec), seed=5002, requires_grad=True
    )
    v = NestedTTVector(vspec, vleaves)
    M = NestedTTMatrix(mspec, mleaves)

    factors = [
        torch.randn(n, dtype=torch.float64, requires_grad=True)
        for n in modes
    ]
    value = M.matvec(v).elementwise_multiply(factors).sum()
    value.backward()

    assert all(x.grad is not None for x in vleaves)
    assert all(x.grad is not None for x in mleaves)
    assert all(x.grad is not None for x in factors)


if __name__ == "__main__":
    for depth in (1, 2, 3):
        print("matvec:", check_case(depth=depth, d=2, steps=2))
    print("sum:", sum_without_materialization_check(depth=3, d=2))
    gradient_check(depth=3, d=2)
    print("gradient check: ok")
