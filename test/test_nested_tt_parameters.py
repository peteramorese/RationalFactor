import torch

from rational_factor.models.tt.nested_tt import NestedTTMatrix, NestedTTVector, ones_nested_tt_vector
from rational_factor.models.tt.nested_tt_parameters import (
    NestedTTMatrixParameters,
    NestedTTVectorParameters,
    RowStochasticNestedTTMatrixParameters,
)


def tensor_factory(shape):
    return torch.nn.Parameter(torch.randn(*shape, dtype=torch.float64))


def test_parameter_construction_and_gradients():
    vp = NestedTTVectorParameters.from_core_spec(
        (2, 2), depth=3, ranks=2, leaf_factory=tensor_factory
    )
    mp = NestedTTMatrixParameters.from_core_spec(
        (2, 2), depth=3, ranks=2, leaf_factory=tensor_factory
    )
    v = vp()
    M = mp()
    assert isinstance(v, NestedTTVector)
    assert isinstance(M, NestedTTMatrix)

    loss = M.matvec(v).sum()
    loss.backward()
    assert all(p.grad is not None for p in vp.parameters())
    assert all(p.grad is not None for p in mp.parameters())


def test_row_stochastic_depths_1_to_3():
    for depth in (1, 2, 3):
        p = RowStochasticNestedTTMatrixParameters.from_core_spec(
            (2, 2), depth=depth, ranks=2, leaf_factory=tensor_factory
        )
        M = p()
        dense = M.to_dense()
        assert float(dense.detach().min()) >= 0.0
        torch.testing.assert_close(
            dense.sum(dim=1),
            torch.ones(dense.shape[0], dtype=dense.dtype),
            rtol=1e-10,
            atol=1e-10,
        )

        # Verify the defining action B 1 = 1 as well.
        ones = torch.ones(dense.shape[1], dtype=dense.dtype)
        torch.testing.assert_close(dense @ ones, torch.ones_like(ones), rtol=1e-10, atol=1e-10)

        nested_ones = ones_nested_tt_vector(
            (2, 2), depth=depth, ranks=1, dtype=dense.dtype
        )
        nested_result = M.matvec(nested_ones)
        torch.testing.assert_close(
            nested_result.to_dense(),
            torch.ones(4, dtype=dense.dtype),
            rtol=1e-10,
            atol=1e-10,
        )


def test_row_stochastic_gradients():
    p = RowStochasticNestedTTMatrixParameters.from_core_spec(
        (2, 2), depth=3, ranks=2, leaf_factory=tensor_factory
    )
    M = p()
    # A nonconstant objective so softmax logits receive gradients.
    dense = M.to_dense()
    weights = torch.arange(dense.numel(), dtype=dense.dtype).reshape_as(dense)
    loss = (dense * weights).sum()
    loss.backward()
    assert all(q.grad is not None for q in p.parameters())


def test_parameter_separation_rank_and_row_stochasticity():
    q = 3
    vp = NestedTTVectorParameters.from_core_spec(
        (2, 2), depth=3, ranks=2, separation_rank=q, leaf_factory=tensor_factory
    )
    mp = NestedTTMatrixParameters.from_core_spec(
        (2, 2), depth=3, ranks=2, separation_rank=q, leaf_factory=tensor_factory
    )
    assert vp.separation_rank == q
    assert mp.separation_rank == q
    assert vp.spec.separation_rank == q
    assert mp.spec.separation_rank == q

    for depth in (1, 2, 3):
        sp = RowStochasticNestedTTMatrixParameters.from_core_spec(
            (2, 2), depth=depth, ranks=2, separation_rank=q, leaf_factory=tensor_factory
        )
        M = sp()
        dense = M.to_dense()
        assert float(dense.detach().min()) >= 0.0
        torch.testing.assert_close(
            dense.sum(dim=1),
            torch.ones(dense.shape[0], dtype=dense.dtype),
            rtol=1e-10,
            atol=1e-10,
        )

        # Nonconstant objective exercises gradients through the q-mixture softmax.
        weights = torch.arange(dense.numel(), dtype=dense.dtype).reshape_as(dense)
        (dense * weights).sum().backward()
        assert all(p.grad is not None for p in sp.parameters())


if __name__ == "__main__":
    test_parameter_construction_and_gradients()
    test_row_stochastic_depths_1_to_3()
    test_row_stochastic_gradients()
    test_parameter_separation_rank_and_row_stochasticity()
    print("test_nested_tt_parameters.py: all tests passed")

