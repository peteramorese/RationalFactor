"""
Checks for PositiveMaskedGramMutualBasis with LocalBSplineMutualBasis masking.

Run:
  PYTHONPATH=src python test/test_positive_masked_gram_basis.py
"""

from __future__ import annotations

import torch

from rational_factor.models.basis_functions import BetaBasis
from rational_factor.models.domain_transformation import MLP
from rational_factor.models.mutual_bases import (
    LocalBSplineMutualBasis,
    MaskedGramMutualBasis,
    PositiveMaskedGramMutualBasis,
    VolumePreservingPairBasis,
)
from rational_factor.models.parameters import PositiveParameters
from rational_factor.models.structured_matrices import DenseMatrix


SEED = 0
N_BASIS = 8
K_ALPHA = 3
K_BETA = 2
CONDITIONER_DIM = 4
N_POINTS = 64
N_GRID = 20_001
N_MC = 40_000
GRAM_ATOL = 0.08


def _make_masking() -> LocalBSplineMutualBasis:
    return LocalBSplineMutualBasis(
        n_basis=N_BASIS,
        k_alpha=K_ALPHA,
        k_beta=K_BETA,
        dtype=torch.float64,
    )


def _make_vp(n_basis: int, rest_dim: int) -> VolumePreservingPairBasis:
    shape = (1, rest_dim, 1)
    base = BetaBasis(
        PositiveParameters.set_init(shape, 0.0, epsilon=1.0),
        PositiveParameters.set_init(shape, 0.0, epsilon=1.0),
    )
    embedding = torch.nn.Embedding(n_basis, CONDITIONER_DIM)
    splitter = MLP(
        in_features=max(rest_dim, 0) + CONDITIONER_DIM,
        out_features=1,
        hidden_features=16,
        num_hidden_layers=2,
        zero_init_last=True,
    )
    return VolumePreservingPairBasis(base, splitter, embedding, flow=None)


def _numerical_gram(y: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.trapezoid(a[:, :, None] * b[:, None, :], y, dim=0)


def main() -> None:
    torch.manual_seed(SEED)
    masking = _make_masking()
    vp = _make_vp(N_BASIS, rest_dim=0)
    unsigned = MaskedGramMutualBasis(masking, 0, vp)
    pos = PositiveMaskedGramMutualBasis(masking, 0, vp)
    pos.eval()
    unsigned.eval()

    assert pos.dim() == 1
    u_b = vp.supremum(1)
    assert torch.allclose(u_b, torch.ones(1, N_BASIS, dtype=u_b.dtype))

    y = torch.linspace(0, 1, N_POINTS, dtype=torch.float64).unsqueeze(-1)
    alpha_u = unsigned.eval(y, index=0)
    beta_u = unsigned.eval(y, index=1)
    alpha = pos.eval(y, index=0)
    beta = pos.eval(y, index=1)
    b = masking.eval_b(y.squeeze(-1))

    assert torch.allclose(alpha, alpha_u)
    assert torch.allclose(beta, beta_u + b * u_b)
    # With identity free pair, correction is exactly relu(beta_mask).
    assert torch.allclose(beta, torch.relu(beta_u), atol=1e-12)
    assert beta.min().item() >= -1e-12

    stacked = pos.eval(y, index=None)
    assert torch.allclose(stacked[:, 0, :], alpha)
    assert torch.allclose(stacked[:, 1, :], beta)

    gram = pos.Omega2()
    assert isinstance(gram, DenseMatrix)
    dense = gram.to_dense()
    unsigned_gram = unsigned.Omega2().to_dense()
    gab = masking.Omega2_alpha_b().to_dense()
    omega1 = vp.Omega1(0)
    expected = unsigned_gram + omega1.unsqueeze(-1) * gab * u_b.unsqueeze(-2)
    assert torch.allclose(dense, expected)
    print(f"Omega2 diag: {dense[0].diag().tolist()}")

    # Dense quadrature reference on the sacrificial line.
    y_grid = torch.linspace(0, 1, N_GRID, dtype=torch.float64)
    a_grid = pos.eval(y_grid.unsqueeze(-1), index=0)
    b_grid = pos.eval(y_grid.unsqueeze(-1), index=1)
    G_num = _numerical_gram(y_grid, a_grid, b_grid)
    print(f"max quadrature Gram error = {(G_num - dense[0]).abs().max().item():.3e}")
    assert torch.allclose(G_num, dense[0], atol=5e-5, rtol=5e-5)

    try:
        pos.Omega2(lows=torch.zeros(1), highs=torch.ones(1))
        raise AssertionError("Omega2 should reject partial domains")
    except ValueError:
        pass

    omega1_a = pos.Omega1(0)
    assert torch.allclose(omega1_a, masking.Omega1(0) * vp.Omega1(0))
    print(f"Omega1 alpha: {omega1_a[0].tolist()}")

    y_mc = torch.rand(N_MC, 1, dtype=torch.float64)
    a_mc = pos.eval(y_mc, index=0)
    b_mc = pos.eval(y_mc, index=1)
    assert a_mc.min().item() >= -1e-12
    assert b_mc.min().item() >= -1e-12
    gram_mc = (a_mc.T @ b_mc) / N_MC
    print(f"MC gram diag: {gram_mc.diag().tolist()}")
    assert torch.allclose(gram_mc, dense[0], atol=GRAM_ATOL, rtol=0.15)

    alpha_b = pos.get_basis(0)
    beta_b = pos.get_basis(1)
    assert torch.allclose(alpha_b(y), alpha)
    assert torch.allclose(beta_b(y), beta)
    G_ab = alpha_b.Omega2(beta_b)
    G_ba = beta_b.Omega2(alpha_b)
    assert torch.allclose(G_ab.to_dense(), dense)
    assert torch.allclose(G_ba.to_dense(), dense.transpose(-2, -1))

    print("ok")


if __name__ == "__main__":
    main()
