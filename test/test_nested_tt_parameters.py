import torch

from rational_factor.models.tt.nested_tt import NestedTTMatrix
from rational_factor.models.tt.nested_tt_parameters import (
    NestedTTMatrixParameters,
    RowStochasticNestedTTMatrixParameters,
)


class DummyTensorParameters(torch.nn.Module):
    def __init__(self, shape, *, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.value = torch.nn.Parameter(
            torch.randn(*shape, generator=g, dtype=torch.float64)
        )

    def forward(self):
        return self.value

    def is_trainable(self):
        return self.value.requires_grad

    def parameter_modules(self):
        return [self]


def make_factory(seed0):
    counter = {"i": 0}

    def factory(shape):
        i = counter["i"]
        counter["i"] += 1
        return DummyTensorParameters(shape, seed=seed0 + i)

    return factory


def check_plain(depth):
    params = NestedTTMatrixParameters.from_core_spec(
        row_modes=(2, 3),
        col_modes=(3, 2),
        depth=depth,
        ranks=2,
        leaf_factory=make_factory(1000 + 100 * depth),
    )
    M = params()
    direct = NestedTTMatrix(M.spec, [p() for p in params.leaves])
    torch.testing.assert_close(M.to_dense(), direct.to_dense())
    assert params.is_trainable()
    assert len(params.parameter_modules()) == M.leaf_count
    return M.leaf_count


def check_row_stochastic(depth):
    params = RowStochasticNestedTTMatrixParameters.from_core_spec(
        row_modes=(2, 3),
        col_modes=(3, 2),
        depth=depth,
        ranks=2,
        leaf_factory=make_factory(2000 + 100 * depth),
    )
    M = params()
    dense = M.to_dense()

    assert torch.all(dense >= 0)
    torch.testing.assert_close(
        dense.sum(dim=1),
        torch.ones(dense.shape[0], dtype=dense.dtype),
        rtol=1e-11,
        atol=1e-11,
    )

    # Stronger local condition: every outer MPO core is normalized over
    # (physical column, outgoing bond) for fixed (incoming bond, row mode).
    for core in M.materialize_cores():
        local_rows = core.sum(dim=(2, 3))
        torch.testing.assert_close(
            local_rows,
            torch.ones_like(local_rows),
            rtol=1e-11,
            atol=1e-11,
        )

    # Equivalent row-stochastic witness M 1 = 1.
    ones = torch.ones(dense.shape[1], dtype=dense.dtype)
    torch.testing.assert_close(
        dense @ ones,
        torch.ones(dense.shape[0], dtype=dense.dtype),
        rtol=1e-11,
        atol=1e-11,
    )

    # Gradients must flow from the normalized NestedTTMatrix to the raw logits.
    loss = (dense.square()).sum()
    loss.backward()
    assert all(p.value.grad is not None for p in params.leaves)

    max_row_error = float((dense.sum(dim=1) - 1).abs().max().detach())
    return {
        "depth": depth,
        "leaves": M.leaf_count,
        "shape": tuple(dense.shape),
        "min_entry": float(dense.min().detach()),
        "max_row_error": max_row_error,
    }


if __name__ == "__main__":
    for depth in (1, 2, 3):
        print("plain leaves:", depth, check_plain(depth))
        print("stochastic:", check_row_stochastic(depth))
