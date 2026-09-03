"""
Checks for Rank1PlusDiagonal construction, optional diagonal, and stochastic parameterization.

Run:
  PYTHONPATH=src python test/test_lrpd.py
"""

from __future__ import annotations

import torch

from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.factor_forms import SumProdRFF
from rational_factor.models.parameters import (
    PositiveParameters,
    Rank1PlusDiagonalFactorization,
    SequentialRank1PlusDiagonalFactorization,
    TrainableParameters,
)
from rational_factor.models.structured_matrices import Rank1PlusDiagonal, SequentialRank1PlusDiagonal


SEED = 0
N = 5
BATCH = 3


def main() -> None:
    torch.manual_seed(SEED)
    d = 0.5 + torch.rand(BATCH, N)
    u = torch.randn(BATCH, N)
    v = torch.randn(BATCH, N)

    M = Rank1PlusDiagonal(u, v, d)
    dense = torch.diag_embed(d) + u.unsqueeze(-1) * v.unsqueeze(-2)
    assert torch.allclose(M.to_dense(), dense)
    assert torch.allclose(M.T.to_dense(), dense.transpose(-2, -1))

    ones = torch.ones(BATCH, N)
    M_I = Rank1PlusDiagonal(u, v)
    dense_I = torch.diag_embed(ones) + u.unsqueeze(-1) * v.unsqueeze(-2)
    assert torch.allclose(M_I.d, ones)
    assert torch.allclose(M_I.to_dense(), dense_I)

    u_raw = torch.randn(BATCH, N)
    v_raw = torch.randn(BATCH, N)

    M_row = Rank1PlusDiagonal(u_raw, v_raw, normalization_dim=1)
    assert torch.allclose(M_row.u, torch.sigmoid(u_raw))
    assert torch.allclose(M_row.v, torch.softmax(v_raw, dim=-1))
    assert torch.allclose(M_row.d, 1.0 - M_row.u)
    row_sums = M_row.to_dense().sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums))
    assert (M_row.u >= 0).all() and (M_row.u <= 1).all()
    assert torch.allclose(M_row.v.sum(dim=-1), torch.ones(BATCH))

    M_col = Rank1PlusDiagonal(u_raw, v_raw, normalization_dim=0)
    assert torch.allclose(M_col.u, torch.softmax(u_raw, dim=-1))
    assert torch.allclose(M_col.v, torch.sigmoid(v_raw))
    assert torch.allclose(M_col.d, 1.0 - M_col.v)
    col_sums = M_col.to_dense().sum(dim=-2)
    assert torch.allclose(col_sums, torch.ones_like(col_sums))
    assert (M_col.v >= 0).all() and (M_col.v <= 1).all()
    assert torch.allclose(M_col.u.sum(dim=-1), torch.ones(BATCH))

    # Transpose of a row-stochastic batch is column-stochastic, without re-normalizing.
    assert torch.allclose(M_row.T.to_dense().sum(dim=-2), torch.ones(BATCH, N))

    # A sequence axis is consumed into one matrix product per outer batch.
    n_factors = 4
    seq_factors = Rank1PlusDiagonal(
        torch.randn(BATCH, n_factors, N),
        torch.randn(BATCH, n_factors, N),
        0.5 + torch.rand(BATCH, n_factors, N),
    )
    product = SequentialRank1PlusDiagonal(seq_factors, seq_dim=1)
    dense_factors = seq_factors.to_dense()
    expected = torch.eye(N).expand(BATCH, N, N)
    for index in range(n_factors):
        expected = dense_factors[:, index] @ expected
    assert product.shape == (BATCH, N, N)
    assert torch.allclose(product.to_dense(), expected, atol=1e-5)
    x = torch.randn(BATCH, N)
    assert torch.allclose(product.matvec(x), torch.einsum("bij,bj->bi", expected, x), atol=1e-5)
    assert torch.allclose(product.T.to_dense(), expected.transpose(-2, -1), atol=1e-5)

    left_diag = torch.rand(BATCH, N)
    right_diag = torch.rand(BATCH, N)
    assert torch.allclose(
        product.mul_diag_left(left_diag).to_dense(),
        torch.diag_embed(left_diag) @ expected,
        atol=1e-5,
    )
    assert torch.allclose(
        product.mul_diag_right(right_diag).to_dense(),
        expected @ torch.diag_embed(right_diag),
        atol=1e-5,
    )

    # SumProdRFF sees only the outer batch axis, and owns the factor modules.
    basis_shape = (1, 1, N)
    means = TrainableParameters.random_init(basis_shape)
    stds = PositiveParameters.set_init(basis_shape, 1.0)
    coeffs = PositiveParameters.set_init((1, N), 1.0)
    g = GaussianBasis(means, stds, coeffs=coeffs)
    psi = GaussianBasis(TrainableParameters.random_init(basis_shape), PositiveParameters.set_init(basis_shape, 1.0))

    factor_shape = (1, n_factors, N)
    b_params = tuple(TrainableParameters.random_init(factor_shape, std=0.1) for _ in range(3))
    p_params = tuple(TrainableParameters.random_init(factor_shape, std=0.1) for _ in range(2))
    B = SequentialRank1PlusDiagonalFactorization(
        Rank1PlusDiagonalFactorization(*b_params), seq_dim=1
    )
    P = SequentialRank1PlusDiagonalFactorization(
        Rank1PlusDiagonalFactorization(*p_params, normalization_dim=0), seq_dim=1
    )
    model = SumProdRFF(g, psi, B, P)
    assert B().shape == P().shape == (1, N, N)
    model_param_ids = {id(param) for param in model.parameters()}
    assert all(id(param._p) in model_param_ids for param in (*b_params, *p_params))

    data = torch.randn(7, 1)
    model.log_density(data, conditioner=data).sum().backward()
    assert all(param._p.grad is not None for param in (*b_params, *p_params))
    model.cpu()

    try:
        Rank1PlusDiagonal(u_raw, v_raw, normalization_dim=2)
        raise AssertionError("expected ValueError for invalid normalization_dim")
    except ValueError:
        pass

    print("ok")


if __name__ == "__main__":
    main()
