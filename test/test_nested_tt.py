import math

import torch

from rational_factor.models.tt.nested_tt import (
    NestedTTMatrix,
    NestedTTMatrixSpec,
    NestedTTVector,
    NestedTTVectorSpec,
    nested_tt_matrix_from_separable_cores,
    nested_tt_vector_from_factors,
)


def make_leaves(shapes, *, seed, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randn(*shape, generator=g, dtype=torch.float64).requires_grad_(requires_grad)
        for shape in shapes
    ]


def kron_factors(factors):
    out = factors[0]
    for factor in factors[1:]:
        out = torch.kron(out, factor)
    return out


def make_case(depth: int, d: int = 2, rank: int = 2):
    modes = (2,) * d
    ranks = (rank,) * depth
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
    return v, M


def test_dense_equivalence_depths_1_to_3():
    for depth in (1, 2, 3):
        v, M = make_case(depth)
        vd = v.to_dense()
        Md = M.to_dense()

        y = M.matvec(v)
        torch.testing.assert_close(y.to_dense(), Md @ vd, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(y.sum(), (Md @ vd).sum(), rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(v.sum(), vd.sum(), rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(M.sum(), Md.sum(), rtol=1e-10, atol=1e-10)


def test_repeated_matvec_dense_equivalence_and_linear_symbolic_growth():
    v, M = make_case(depth=3)
    Md = M.to_dense()
    ref = v.to_dense()
    cur = v
    initial_matrix_leaves = M.leaf_count
    initial_vector_leaves = v.leaf_count

    max_intermediates = []
    operand_counts = []
    for t in range(1, 9):
        cur = M.matvec(cur)
        ref = Md @ ref
        got = cur.sum()
        torch.testing.assert_close(got, ref.sum(), rtol=1e-9, atol=1e-11)

        assert cur.history_length == t
        assert cur.leaf_count == initial_vector_leaves + t * initial_matrix_leaves
        stats = cur.last_contract_stats
        assert stats is not None
        max_intermediates.append(stats.max_intermediate_numel)
        operand_counts.append(stats.operands)

    # The factor graph grows linearly with time, while the largest production
    # contraction intermediate stays bounded in this fixed-depth case.
    assert operand_counts == sorted(operand_counts)
    assert operand_counts[-1] <= operand_counts[0] * 8
    assert max(max_intermediates) <= 2 * max_intermediates[0]


def test_sum_does_not_materialize_parent_cores():
    v, M = make_case(depth=3)
    cur = M.matvec(M.matvec(M.matvec(v)))

    originals = []
    try:
        for bank in cur.cores:
            originals.append((bank, bank.materialize))

            def fail_materialize(*args, **kwargs):
                raise AssertionError("production sum() attempted dense core materialization")

            bank.materialize = fail_materialize
        _ = cur.sum()
    finally:
        for bank, original in originals:
            bank.materialize = original


def test_rank_one_elementwise_multiply_then_sum():
    v, M = make_case(depth=3)
    cur = M.matvec(M.matvec(v))
    dense = cur.to_dense()
    factors = [
        torch.randn(n, generator=torch.Generator().manual_seed(3000 + k), dtype=torch.float64)
        for k, n in enumerate(cur.modes)
    ]
    weighted = cur.elementwise_multiply(factors)
    expected = (dense * kron_factors(factors)).sum()
    torch.testing.assert_close(weighted.sum(), expected, rtol=1e-10, atol=1e-10)


def test_elementwise_divide_is_factorwise_and_tolerant():
    v, M = make_case(depth=2)
    cur = M.matvec(v)
    dense = cur.to_dense()
    eps = 1e-6
    factors = (
        torch.tensor([2.0, 1e-20], dtype=torch.float64),
        torch.tensor([-4.0, 0.5], dtype=torch.float64),
    )

    # Guard the operation itself against accidental dense conversion.
    original = NestedTTVector.to_dense
    try:
        def fail_dense(self):
            raise AssertionError("elementwise_divide attempted dense materialization")
        NestedTTVector.to_dense = fail_dense
        divided = cur.elementwise_divide(factors, eps=eps)
    finally:
        NestedTTVector.to_dense = original

    safe = []
    for f in factors:
        sign = torch.where(f < 0, -torch.ones_like(f), torch.ones_like(f))
        safe.append(torch.where(f.abs() < eps, sign * eps, f))
    denom = kron_factors(safe)
    torch.testing.assert_close(divided.to_dense(), dense / denom, rtol=1e-10, atol=1e-10)


def test_gradients_through_matvec_weighted_sum():
    depth = 3
    modes = (2, 2)
    ranks = (2,) * depth
    vspec = NestedTTVectorSpec(modes, depth, ranks)
    mspec = NestedTTMatrixSpec(modes, modes, depth, ranks)
    vleaves = make_leaves(
        NestedTTVector.leaf_shapes(vspec), seed=5001, requires_grad=True
    )
    mleaves = make_leaves(
        NestedTTMatrix.leaf_shapes(mspec), seed=5002, requires_grad=True
    )
    v = NestedTTVector(vspec, vleaves)
    M = NestedTTMatrix(mspec, mleaves)
    factors = tuple(
        torch.randn(n, dtype=torch.float64, requires_grad=True) for n in modes
    )

    loss = M.matvec(v).elementwise_multiply(factors).sum()
    loss.backward()

    assert all(x.grad is not None for x in vleaves)
    assert all(x.grad is not None for x in mleaves)
    assert all(x.grad is not None for x in factors)


if __name__ == "__main__":
    test_dense_equivalence_depths_1_to_3()
    test_repeated_matvec_dense_equivalence_and_linear_symbolic_growth()
    test_sum_does_not_materialize_parent_cores()
    test_rank_one_elementwise_multiply_then_sum()
    test_elementwise_divide_is_factorwise_and_tolerant()
    test_gradients_through_matvec_weighted_sum()
    print("test_nested_tt.py: all tests passed")



def batched_kron_factors(factors):
    batch = max(1 if x.ndim == 1 else int(x.shape[0]) for x in factors)
    rows = []
    for b in range(batch):
        local = [x if x.ndim == 1 else x[b] for x in factors]
        rows.append(kron_factors(local))
    return torch.stack(rows)


def batched_kron_matrices(cores):
    batch = max(1 if x.ndim == 2 else int(x.shape[0]) for x in cores)
    rows = []
    for b in range(batch):
        local = [x if x.ndim == 2 else x[b] for x in cores]
        rows.append(kron_factors(local))
    return torch.stack(rows)


def test_batched_rank_one_weighting_and_sum():
    v, _ = make_case(depth=3)
    dense = v.to_dense()
    B = 4
    factors = tuple(
        torch.randn(B, n, generator=torch.Generator().manual_seed(7100 + k), dtype=torch.float64)
        for k, n in enumerate(v.modes)
    )
    weighted = v.elementwise_multiply(factors)
    expected = dense.unsqueeze(0) * batched_kron_factors(factors)
    assert weighted.batch_size == B
    assert weighted.shape == (B, dense.numel())
    torch.testing.assert_close(weighted.to_dense(), expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(weighted.sum(), expected.sum(dim=1), rtol=1e-10, atol=1e-10)


def test_rank_one_nested_vector_multiplier_never_densifies():
    v, _ = make_case(depth=3)
    B = 3
    factors = tuple(
        torch.randn(B, n, generator=torch.Generator().manual_seed(7200 + k), dtype=torch.float64)
        for k, n in enumerate(v.modes)
    )
    phi = nested_tt_vector_from_factors(factors, depth=v.depth)
    expected = v.to_dense().unsqueeze(0) * batched_kron_factors(factors)

    original = NestedTTVector.to_dense
    try:
        def fail_dense(self):
            raise AssertionError("rank-one NestedTT multiplier attempted dense materialization")
        NestedTTVector.to_dense = fail_dense
        weighted = v.elementwise_multiply(phi)
        got_sum = weighted.sum()
    finally:
        NestedTTVector.to_dense = original

    torch.testing.assert_close(got_sum, expected.sum(dim=1), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(weighted.to_dense(), expected, rtol=1e-10, atol=1e-10)


def test_unbatched_matrix_times_batched_vector():
    v, M = make_case(depth=3)
    B = 3
    factors = tuple(
        torch.randn(B, n, generator=torch.Generator().manual_seed(7300 + k), dtype=torch.float64)
        for k, n in enumerate(v.modes)
    )
    vb = v.elementwise_multiply(factors)
    got = M.matvec(vb)
    expected = torch.einsum("mn,bn->bm", M.to_dense(), vb.to_dense())
    assert got.batch_size == B
    torch.testing.assert_close(got.to_dense(), expected, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(got.sum(), expected.sum(dim=1), rtol=1e-9, atol=1e-10)


def test_batched_separable_matrix_and_matvec_broadcast():
    v, _ = make_case(depth=3)
    B = 3
    cores = tuple(
        torch.randn(B, 2, 2, generator=torch.Generator().manual_seed(7400 + k), dtype=torch.float64)
        for k in range(v.d)
    )
    Omega = nested_tt_matrix_from_separable_cores(cores, depth=v.depth)
    dense_Omega = batched_kron_matrices(cores)
    torch.testing.assert_close(Omega.to_dense(), dense_Omega, rtol=1e-10, atol=1e-10)

    got = Omega.matvec(v)
    expected = torch.einsum("bmn,n->bm", dense_Omega, v.to_dense())
    assert got.batch_size == B
    torch.testing.assert_close(got.to_dense(), expected, rtol=1e-9, atol=1e-10)


def test_basis_weight_then_gram_then_model_matvec_chain():
    v, M = make_case(depth=3)
    B = 4
    basis_factors = tuple(
        torch.randn(B, n, generator=torch.Generator().manual_seed(7500 + k), dtype=torch.float64)
        for k, n in enumerate(v.modes)
    )
    basis_eval = nested_tt_vector_from_factors(basis_factors, depth=v.depth)
    gram_cores = tuple(
        torch.randn(2, 2, generator=torch.Generator().manual_seed(7600 + k), dtype=torch.float64)
        for k in range(v.d)
    )
    Omega = nested_tt_matrix_from_separable_cores(gram_cores, depth=v.depth)

    weighted = v.elementwise_multiply(basis_eval)
    got = M.matvec(Omega.matvec(weighted))

    dense_weighted = v.to_dense().unsqueeze(0) * batched_kron_factors(basis_factors)
    dense_omega = batched_kron_matrices(gram_cores)[0]
    expected = torch.einsum("mn,nk,bk->bm", M.to_dense(), dense_omega, dense_weighted)
    torch.testing.assert_close(got.to_dense(), expected, rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(got.sum(), expected.sum(dim=1), rtol=1e-8, atol=1e-9)


def test_separable_gram_matvec_preserves_nested_history_and_leaf_count():
    v, _ = make_case(depth=3)
    gram_cores = tuple(
        torch.randn(2, 2, generator=torch.Generator().manual_seed(7700 + k), dtype=torch.float64)
        for k in range(v.d)
    )
    Omega = nested_tt_matrix_from_separable_cores(gram_cores, depth=v.depth)
    out = Omega.matvec(v)
    assert out.history_length == v.history_length
    assert out.leaf_count == v.leaf_count
    assert out.ranks == v.ranks
    expected = batched_kron_matrices(gram_cores)[0] @ v.to_dense()
    torch.testing.assert_close(out.to_dense(), expected, rtol=1e-10, atol=1e-10)


def test_batched_separable_matrix_times_batched_vector_same_batch():
    v, _ = make_case(depth=3)
    B = 3
    vf = tuple(
        torch.randn(B, n, generator=torch.Generator().manual_seed(7800 + k), dtype=torch.float64)
        for k, n in enumerate(v.modes)
    )
    vb = v.elementwise_multiply(vf)
    cores = tuple(
        torch.randn(B, 2, 2, generator=torch.Generator().manual_seed(7900 + k), dtype=torch.float64)
        for k in range(v.d)
    )
    Omega = nested_tt_matrix_from_separable_cores(cores, depth=v.depth)
    got = Omega.matvec(vb)
    expected = torch.einsum("bmn,bn->bm", Omega.to_dense(), vb.to_dense())
    torch.testing.assert_close(got.to_dense(), expected, rtol=1e-9, atol=1e-10)


def test_general_nested_matrix_supports_leading_batch_via_separable_scaling():
    v, M = make_case(depth=3)
    B = 3
    factors = tuple(
        torch.randn(B, m, n, generator=torch.Generator().manual_seed(8000 + k), dtype=torch.float64)
        for k, (m, n) in enumerate(zip(M.row_modes, M.col_modes))
    )
    Mb = M.elementwise_multiply(factors)
    scale = batched_kron_matrices(factors)
    expected_matrix = M.to_dense().unsqueeze(0) * scale
    assert Mb.batch_size == B
    assert Mb.shape == (B, *M.shape)
    torch.testing.assert_close(Mb.to_dense(), expected_matrix, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(Mb.sum(), expected_matrix.sum(dim=(1, 2)), rtol=1e-10, atol=1e-10)

    got = Mb.matvec(v)
    expected_vec = torch.einsum("bmn,n->bm", expected_matrix, v.to_dense())
    torch.testing.assert_close(got.to_dense(), expected_vec, rtol=1e-9, atol=1e-10)


def test_batched_elementwise_divide_tolerance():
    v, _ = make_case(depth=2)
    B = 2
    factors = (
        torch.tensor([[2.0, 1e-30], [-3.0, -1e-30]], dtype=torch.float64),
        torch.tensor([[0.5, -4.0], [1e-30, 2.0]], dtype=torch.float64),
    )
    eps = 1e-6
    got = v.elementwise_divide(factors, eps=eps)
    safe = []
    for f in factors:
        sign = torch.where(f < 0, -torch.ones_like(f), torch.ones_like(f))
        safe.append(torch.where(f.abs() < eps, sign * eps, f))
    denom = batched_kron_factors(tuple(safe))
    expected = v.to_dense().unsqueeze(0) / denom
    torch.testing.assert_close(got.to_dense(), expected, rtol=1e-10, atol=1e-10)


def test_reverse_matvec_dense_equivalence_depths_1_to_3():
    for depth in (1, 2, 3):
        v, M = make_case(depth)
        vd = v.to_dense()
        Md = M.to_dense()

        y = M.rev_matvec(v)
        expected = Md.transpose(-2, -1) @ vd
        torch.testing.assert_close(y.to_dense(), expected, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(y.sum(), expected.sum(), rtol=1e-10, atol=1e-10)
        assert y.modes == M.col_modes
        assert y.history_length == v.history_length + 1
        assert y.leaf_count == v.leaf_count + M.leaf_count


def test_repeated_reverse_matvec_dense_equivalence_and_linear_symbolic_growth():
    v, M = make_case(depth=3)
    MdT = M.to_dense().transpose(-2, -1)
    ref = v.to_dense()
    cur = v
    initial_matrix_leaves = M.leaf_count
    initial_vector_leaves = v.leaf_count

    max_intermediates = []
    operand_counts = []
    for t in range(1, 9):
        cur = M.rev_matvec(cur)
        ref = MdT @ ref
        got = cur.sum()
        torch.testing.assert_close(got, ref.sum(), rtol=1e-9, atol=1e-11)

        assert cur.history_length == t
        assert cur.leaf_count == initial_vector_leaves + t * initial_matrix_leaves
        stats = cur.last_contract_stats
        assert stats is not None
        max_intermediates.append(stats.max_intermediate_numel)
        operand_counts.append(stats.operands)

    assert operand_counts == sorted(operand_counts)
    assert operand_counts[-1] <= operand_counts[0] * 8
    assert max(max_intermediates) <= 2 * max_intermediates[0]


def test_reverse_matvec_rectangular_modes():
    depth = 2
    row_modes = (2, 3)
    col_modes = (3, 2)
    ranks = (2, 2)
    mspec = NestedTTMatrixSpec(
        row_modes=row_modes,
        col_modes=col_modes,
        depth=depth,
        ranks=ranks,
    )
    vspec = NestedTTVectorSpec(modes=row_modes, depth=depth, ranks=ranks)
    M = NestedTTMatrix(
        mspec,
        make_leaves(NestedTTMatrix.leaf_shapes(mspec), seed=8201),
    )
    x = NestedTTVector(
        vspec,
        make_leaves(NestedTTVector.leaf_shapes(vspec), seed=8202),
    )

    got = M.rev_matvec(x)
    expected = M.to_dense().transpose(-2, -1) @ x.to_dense()
    assert got.modes == col_modes
    assert got.shape == torch.Size((math.prod(col_modes),))
    torch.testing.assert_close(got.to_dense(), expected, rtol=1e-10, atol=1e-10)


def test_separable_reverse_matvec_preserves_history_and_supports_batch():
    v, _ = make_case(depth=3)
    B = 3
    cores = tuple(
        torch.randn(B, 2, 2, generator=torch.Generator().manual_seed(8300 + k), dtype=torch.float64)
        for k in range(v.d)
    )
    Omega = nested_tt_matrix_from_separable_cores(cores, depth=v.depth)
    dense_Omega = batched_kron_matrices(cores)

    got = Omega.rev_matvec(v)
    expected = torch.einsum("bmn,m->bn", dense_Omega, v.to_dense())
    assert got.batch_size == B
    assert got.history_length == v.history_length
    assert got.leaf_count == v.leaf_count
    assert got.ranks == v.ranks
    torch.testing.assert_close(got.to_dense(), expected, rtol=1e-9, atol=1e-10)


def test_general_reverse_matvec_batch_broadcast_and_no_dense_fallback():
    v, M = make_case(depth=3)
    B = 3
    vf = tuple(
        torch.randn(B, n, generator=torch.Generator().manual_seed(8400 + k), dtype=torch.float64)
        for k, n in enumerate(v.modes)
    )
    vb = v.elementwise_multiply(vf)
    expected = torch.einsum("mn,bm->bn", M.to_dense(), vb.to_dense())

    matrix_to_dense = NestedTTMatrix.to_dense
    vector_to_dense = NestedTTVector.to_dense
    try:
        def fail_matrix_dense(self):
            raise AssertionError("rev_matvec attempted matrix dense materialization")

        def fail_vector_dense(self):
            raise AssertionError("rev_matvec attempted vector dense materialization")

        NestedTTMatrix.to_dense = fail_matrix_dense
        NestedTTVector.to_dense = fail_vector_dense
        got = M.rev_matvec(vb)
        got_sum = got.sum()
    finally:
        NestedTTMatrix.to_dense = matrix_to_dense
        NestedTTVector.to_dense = vector_to_dense

    assert got.batch_size == B
    torch.testing.assert_close(got_sum, expected.sum(dim=1), rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(got.to_dense(), expected, rtol=1e-9, atol=1e-10)


def test_separation_rank_increases_expressivity_and_depth_amplifies_it():
    """Q is local; depth can compose it into much larger effective rank."""
    modes = (8, 8)
    depth = 3
    hierarchy_rank = 8
    separation_rank = 2
    mspec = NestedTTMatrixSpec(
        row_modes=modes,
        col_modes=modes,
        depth=depth,
        ranks=(hierarchy_rank,) * depth,
        separation_rank=separation_rank,
    )
    M = NestedTTMatrix(
        mspec,
        make_leaves(NestedTTMatrix.leaf_shapes(mspec), seed=10),
    )
    factors = (
        torch.randn(8, generator=torch.Generator().manual_seed(1), dtype=torch.float64),
        torch.randn(8, generator=torch.Generator().manual_seed(2), dtype=torch.float64),
    )
    v = nested_tt_vector_from_factors(factors, depth=depth)
    y = M.matvec(v).to_dense().reshape(*modes)
    s = torch.linalg.svdvals(y)
    numerical_rank = int((s > s[0] * 1e-10).sum())

    # The global output rank can exceed the local leaf separation rank because
    # the nested hierarchy composes many q-indices implicitly.
    assert numerical_rank > separation_rank
    assert numerical_rank == 8


def test_separation_rank_repeated_matvec_uses_mps_history_without_blowup():
    modes = (2, 2)
    depth = 3
    ranks = (2,) * depth
    vspec = NestedTTVectorSpec(modes, depth, ranks, separation_rank=1)
    mspec = NestedTTMatrixSpec(modes, modes, depth, ranks, separation_rank=2)
    v = NestedTTVector(
        vspec,
        make_leaves(NestedTTVector.leaf_shapes(vspec), seed=8101),
    )
    M = NestedTTMatrix(
        mspec,
        [0.1 * x for x in make_leaves(NestedTTMatrix.leaf_shapes(mspec), seed=8102)],
    )

    Md = M.to_dense()
    ref = v.to_dense()
    cur = v
    max_intermediates = []
    operands = []
    for t in range(1, 9):
        cur = M.matvec(cur)
        ref = Md @ ref
        got = cur.sum()
        torch.testing.assert_close(got, ref.sum(), rtol=1e-8, atol=1e-10)
        stats = cur.last_contract_stats
        assert stats is not None
        max_intermediates.append(stats.max_intermediate_numel)
        operands.append(stats.operands)
        assert cur.history_length == t

    # The symbolic graph grows linearly in time, while q_t is retained as an
    # MPS bond.  In this fixed-depth Q=2 case, contraction width is constant.
    assert operands == sorted(operands)
    assert all(b - a == operands[1] - operands[0] for a, b in zip(operands, operands[1:]))
    assert max(max_intermediates) == max_intermediates[0]
    assert max_intermediates[0] <= 32


def test_separation_rank_reverse_matvec_dense_and_bounded():
    modes = (2, 2)
    depth = 3
    ranks = (2,) * depth
    vspec = NestedTTVectorSpec(modes, depth, ranks, separation_rank=1)
    mspec = NestedTTMatrixSpec(modes, modes, depth, ranks, separation_rank=2)
    v = NestedTTVector(
        vspec,
        make_leaves(NestedTTVector.leaf_shapes(vspec), seed=8201),
    )
    M = NestedTTMatrix(
        mspec,
        [0.1 * x for x in make_leaves(NestedTTMatrix.leaf_shapes(mspec), seed=8202)],
    )
    Md = M.to_dense()
    ref = v.to_dense()
    cur = v
    widths = []
    for _ in range(6):
        cur = M.rev_matvec(cur)
        ref = Md.T @ ref
        torch.testing.assert_close(cur.sum(), ref.sum(), rtol=1e-8, atol=1e-10)
        widths.append(cur.last_contract_stats.max_intermediate_numel)
    assert max(widths) <= 2 * widths[0]
