"""
Checks for MaskedGramMutualBasis.

Run:
  PYTHONPATH=src python test/test_masked_gram_basis.py
"""

from __future__ import annotations

import torch

from normalizing_flow.vp_flow import VolumePreservingFlow
from rational_factor.models.basis_functions import BetaBasis
from rational_factor.models.domain_transformation import MLP
from rational_factor.models.mutual_bases import (
    MaskedGramMutualBasis,
    Orthogonal1DPWCBasis,
    VolumePreservingPairBasis,
)
from rational_factor.models.parameters import Parameters, PositiveParameters
from rational_factor.models.qs_matrix import Order1QuasiseparableFactorization


SEED = 0
DIM = 2
N_BASIS = 3
SACRIFICIAL = 0
CONDITIONER_DIM = 4
N_POINTS = 32
N_MC = 40_000
GRAM_ATOL = 0.08


def _randn(g: torch.Generator, n_basis: int) -> Parameters:
    return Parameters(torch.randn(1, n_basis, generator=g))


def _make_pwc(n_basis: int, seed: int) -> Orthogonal1DPWCBasis:
    g = torch.Generator().manual_seed(seed)
    fac = Order1QuasiseparableFactorization(
        _randn(g, n_basis),
        _randn(g, n_basis),
        _randn(g, n_basis),
        _randn(g, n_basis),
        _randn(g, n_basis),
        _randn(g, n_basis),
        _randn(g, n_basis),
        transition_bound=0.99,
    )
    lam = Parameters(0.5 + torch.rand(1, n_basis, generator=g))
    return Orthogonal1DPWCBasis(fac, gram_diag_params=lam)


def main() -> None:
    torch.manual_seed(SEED)
    pwc = _make_pwc(N_BASIS, SEED)
    rest_dim = DIM - 1
    flow = VolumePreservingFlow(
        dim=rest_dim,
        conditioner_dim=CONDITIONER_DIM,
        n_steps=4,
        hidden_features=16,
        num_hidden_layers=2,
        zero_init=False,
    )
    shape = (1, rest_dim, 1)
    base = BetaBasis(
        PositiveParameters.set_init(shape, 0.0, epsilon=1.0),
        PositiveParameters.set_init(shape, 0.0, epsilon=1.0),
    )
    embedding = torch.nn.Embedding(N_BASIS, CONDITIONER_DIM)
    splitter = MLP(
        in_features=rest_dim + CONDITIONER_DIM,
        out_features=1,
        hidden_features=16,
        num_hidden_layers=2,
        zero_init_last=True,
    )
    vp = VolumePreservingPairBasis(base, splitter, embedding, flow)
    masked = MaskedGramMutualBasis(pwc, SACRIFICIAL, vp)
    masked.eval()

    y = 0.05 + 0.9 * torch.rand(N_POINTS, DIM)
    x_l, x_rest = masked._split_coords(y)
    alpha = masked.eval(y, index=0)
    beta = masked.eval(y, index=1)
    alpha_pwc, beta_pwc = pwc.eval(x_l, index=0), pwc.eval(x_l, index=1)
    alpha_vp, beta_vp = vp.eval(x_rest, index=0), vp.eval(x_rest, index=1)

    assert torch.allclose(alpha, alpha_pwc * alpha_vp)
    assert torch.allclose(beta, beta_pwc * beta_vp)
    stacked = masked.eval(y, index=None)
    assert torch.allclose(stacked[:, 0, :], alpha)
    assert torch.allclose(stacked[:, 1, :], beta)

    gram = masked.Omega2().to_dense()
    pwc_gram = pwc.Omega2().to_dense()
    print(f"Omega2 diagonal: {gram[0].diag().tolist()}")
    print(f"PWC  diagonal:   {pwc_gram[0].diag().tolist()}")
    assert torch.allclose(gram, pwc_gram)
    off = gram[0] - torch.diag(gram[0].diag())
    assert off.abs().max().item() == 0.0

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
    assert torch.allclose(omega1, torch.zeros_like(omega1))

    y_mc = torch.rand(N_MC, DIM)
    a_mc = masked.eval(y_mc, index=0)
    b_mc = masked.eval(y_mc, index=1)
    gram_mc = (a_mc.T @ b_mc) / N_MC
    print(f"MC gram diag:    {gram_mc.diag().tolist()}")
    assert torch.allclose(gram_mc, gram[0], atol=GRAM_ATOL, rtol=0.15)

    alpha_b = masked.get_basis(0)(y)
    beta_b = masked.get_basis(1)(y)
    assert torch.allclose(alpha_b, alpha)
    assert torch.allclose(beta_b, beta)

    print("ok")


if __name__ == "__main__":
    main()
