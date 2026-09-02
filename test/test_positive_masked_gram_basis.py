"""
Checks for PositiveMaskedGramMutualBasis.

Run:
  PYTHONPATH=src python test/test_positive_masked_gram_basis.py
"""

from __future__ import annotations

import torch

from normalizing_flow.vp_flow import VolumePreservingFlow
from rational_factor.models.basis_functions import BetaBasis
from rational_factor.models.domain_transformation import MLP
from rational_factor.models.structured_matrices import Rank1PlusDiagonal
from rational_factor.models.mutual_bases import (
    MaskedGramMutualBasis,
    Orthogonal1DPWCBasis,
    PositiveMaskedGramMutualBasis,
    VolumePreservingPairBasis,
)
from rational_factor.models.parameters import FixedParameters, PositiveParameters, Order1QuasiseparableFactorization


SEED = 0
DIM = 2
N_BASIS = 3
SACRIFICIAL = 0
CONDITIONER_DIM = 4
N_POINTS = 32
N_MC = 40_000
GRAM_ATOL = 0.08
MEAN_ATOL = 0.03


def _randn(g: torch.Generator, n_basis: int) -> FixedParameters:
    return FixedParameters(torch.randn(1, n_basis, generator=g))


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
    lam = FixedParameters(0.5 + torch.rand(1, n_basis, generator=g))
    return Orthogonal1DPWCBasis(fac, gram_diag_params=lam)


def _make_vp(n_basis: int, rest_dim: int) -> VolumePreservingPairBasis:
    flow = None
    if rest_dim > 0:
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
    embedding = torch.nn.Embedding(n_basis, CONDITIONER_DIM)
    splitter = MLP(
        in_features=rest_dim + CONDITIONER_DIM,
        out_features=1,
        hidden_features=16,
        num_hidden_layers=2,
        zero_init_last=True,
    )
    return VolumePreservingPairBasis(base, splitter, embedding, flow)


def main() -> None:
    torch.manual_seed(SEED)
    pwc = _make_pwc(N_BASIS, SEED)
    vp = _make_vp(N_BASIS, DIM - 1)
    unsigned = MaskedGramMutualBasis(pwc, SACRIFICIAL, vp)
    pos = PositiveMaskedGramMutualBasis(pwc, SACRIFICIAL, vp)
    pos.eval()
    unsigned.eval()

    a, b = pos.constant_shifts()
    a_expected = -pwc.infimum(0) * vp.supremum(0)
    b_expected = -pwc.infimum(1) * vp.supremum(1)
    assert torch.allclose(a, a_expected)
    assert torch.allclose(b, b_expected)
    assert (a >= 0).all()
    assert (b >= 0).all()
    print(f"a: {a[0].tolist()}")
    print(f"b: {b[0].tolist()}")

    y = 0.05 + 0.9 * torch.rand(N_POINTS, DIM)
    alpha_u, beta_u = unsigned.eval(y, index=0), unsigned.eval(y, index=1)
    alpha, beta = pos.eval(y, index=0), pos.eval(y, index=1)
    assert torch.allclose(alpha, alpha_u + a)
    assert torch.allclose(beta, beta_u + b)
    stacked = pos.eval(y, index=None)
    assert torch.allclose(stacked[:, 0, :], alpha)
    assert torch.allclose(stacked[:, 1, :], beta)

    y_mc = torch.rand(N_MC, DIM)
    a_mc = pos.eval(y_mc, index=0)
    b_mc = pos.eval(y_mc, index=1)
    assert a_mc.min().item() >= -1e-5
    assert b_mc.min().item() >= -1e-5

    gram = pos.Omega2()
    assert isinstance(gram, Rank1PlusDiagonal)
    dense = gram.to_dense()
    unsigned_gram = unsigned.Omega2().to_dense()
    expected = unsigned_gram + a.unsqueeze(-1) * b.unsqueeze(-2)
    assert torch.allclose(dense, expected)
    print(f"Omega2 diag: {dense[0].diag().tolist()}")

    try:
        pos.Omega2(lows=torch.zeros(1), highs=torch.ones(1))
        raise AssertionError("Omega2 should reject partial domains")
    except ValueError:
        pass

    omega1_a = pos.Omega1(0)
    omega1_b = pos.Omega1(1)
    assert torch.allclose(omega1_a, a)
    assert torch.allclose(omega1_b, b)
    print(f"MC mean alpha: {a_mc.mean(0).tolist()}")
    print(f"Omega1 alpha:  {omega1_a[0].tolist()}")
    assert torch.allclose(a_mc.mean(0), omega1_a[0], atol=MEAN_ATOL, rtol=0.15)
    assert torch.allclose(b_mc.mean(0), omega1_b[0], atol=MEAN_ATOL, rtol=0.15)

    gram_mc = (a_mc.T @ b_mc) / N_MC
    print(f"MC gram diag:  {gram_mc.diag().tolist()}")
    assert torch.allclose(gram_mc, dense[0], atol=GRAM_ATOL, rtol=0.15)

    alpha_b = pos.get_basis(0)
    beta_b = pos.get_basis(1)
    assert torch.allclose(alpha_b(y), alpha)
    assert torch.allclose(beta_b(y), beta)
    G_ab = alpha_b.Omega2(beta_b)
    G_ba = beta_b.Omega2(alpha_b)
    assert isinstance(G_ab, Rank1PlusDiagonal)
    assert torch.allclose(G_ab.to_dense(), dense)
    assert torch.allclose(G_ba.to_dense(), dense.transpose(-2, -1))

    _check_sacrificial_only()
    print("ok")


def _check_sacrificial_only() -> None:
    pwc = _make_pwc(N_BASIS, SEED + 1)
    vp = _make_vp(N_BASIS, rest_dim=0)
    unsigned = MaskedGramMutualBasis(pwc, 0, vp)
    pos = PositiveMaskedGramMutualBasis(pwc, 0, vp)
    assert unsigned.dim() == 1
    assert torch.allclose(vp.supremum(0), torch.ones(1, N_BASIS))
    assert torch.allclose(vp.supremum(1), torch.ones(1, N_BASIS))

    y = 0.05 + 0.9 * torch.rand(N_POINTS, 1)
    alpha_u, beta_u = unsigned.eval(y, index=0), unsigned.eval(y, index=1)
    assert torch.allclose(alpha_u, pwc.eval(y, index=0))
    assert torch.allclose(beta_u, pwc.eval(y, index=1))

    a, b = pos.constant_shifts()
    assert torch.allclose(a, -pwc.infimum(0))
    assert torch.allclose(b, -pwc.infimum(1))
    alpha, beta = pos.eval(y, index=0), pos.eval(y, index=1)
    assert torch.allclose(alpha, alpha_u + a)
    assert torch.allclose(beta, beta_u + b)
    assert alpha.min().item() >= -1e-5
    assert beta.min().item() >= -1e-5
    print("sacrificial-only ok")


if __name__ == "__main__":
    main()
