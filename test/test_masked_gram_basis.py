"""
Checks for MaskedGramMutualBasis with LocalBSplineMutualBasis masking.

Run:
  PYTHONPATH=src python test/test_masked_gram_basis.py
"""

from __future__ import annotations

import torch

from rational_factor.models.basis_functions import BetaBasis
from rational_factor.models.domain_transformation import MLP
from rational_factor.models.mutual_bases import (
    LocalBSplineMutualBasis,
    MaskedGramMutualBasis,
    VolumePreservingPairBasis,
)
from rational_factor.models.parameters import PositiveParameters


SEED = 0
N_BASIS = 8
K_ALPHA = 3
K_BETA = 2
CONDITIONER_DIM = 4
N_POINTS = 32
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


def main() -> None:
    torch.manual_seed(SEED)
    masking = _make_masking()
    vp = _make_vp(N_BASIS, rest_dim=0)
    masked = MaskedGramMutualBasis(masking, 0, vp)
    masked.eval()

    assert masked.dim() == 1
    y = torch.linspace(0.05, 0.95, N_POINTS, dtype=torch.float64).unsqueeze(-1)
    x_l, x_rest = masked._split_coords(y)
    alpha = masked.eval(y, index=0)
    beta = masked.eval(y, index=1)
    assert torch.allclose(alpha, masking.eval(x_l, index=0) * vp.eval(x_rest, index=0))
    assert torch.allclose(beta, masking.eval(x_l, index=1) * vp.eval(x_rest, index=1))
    stacked = masked.eval(y, index=None)
    assert torch.allclose(stacked[:, 0, :], alpha)
    assert torch.allclose(stacked[:, 1, :], beta)

    gram = masked.Omega2().to_dense()
    masking_gram = masking.Omega2().to_dense()
    print(f"Omega2 diagonal: {gram[0].diag().tolist()}")
    assert torch.allclose(gram, masking_gram)
    assert torch.allclose(gram[0], torch.eye(N_BASIS, dtype=gram.dtype), atol=1e-12)

    try:
        masked.Omega2(lows=torch.zeros(1), highs=torch.ones(1))
        raise AssertionError("Omega2 should reject partial domains")
    except ValueError:
        pass
    try:
        masked.Omega1(0, lows=torch.zeros(1))
        raise AssertionError("Omega1 should reject partial domains")
    except ValueError:
        pass

    omega1 = masked.Omega1(0)
    assert torch.allclose(omega1, masking.Omega1(0) * vp.Omega1(0))
    print(f"Omega1 alpha: {omega1[0].tolist()}")

    y_mc = torch.rand(N_MC, 1, dtype=torch.float64)
    a_mc = masked.eval(y_mc, index=0)
    b_mc = masked.eval(y_mc, index=1)
    gram_mc = (a_mc.T @ b_mc) / N_MC
    print(f"MC gram diag: {gram_mc.diag().tolist()}")
    assert torch.allclose(gram_mc, gram[0], atol=GRAM_ATOL, rtol=0.15)

    assert torch.allclose(masked.get_basis(0)(y), alpha)
    assert torch.allclose(masked.get_basis(1)(y), beta)
    print("ok")


if __name__ == "__main__":
    main()
