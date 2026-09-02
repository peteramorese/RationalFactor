"""
Checks for Rank1PlusDiagonal construction, optional diagonal, and stochastic parameterization.

Run:
  PYTHONPATH=src python test/test_lrpd.py
"""

from __future__ import annotations

import torch

from rational_factor.models.structured_matrices import Rank1PlusDiagonal


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

    try:
        Rank1PlusDiagonal(u_raw, v_raw, normalization_dim=2)
        raise AssertionError("expected ValueError for invalid normalization_dim")
    except ValueError:
        pass

    print("ok")


if __name__ == "__main__":
    main()
