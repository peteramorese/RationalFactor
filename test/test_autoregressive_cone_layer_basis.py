"""
Checks for AutoregressiveConeLayerBasis.

Verifies positivity of (alpha, beta) and that a Monte-Carlo empirical Gram
on the unit cube matches ``Omega2``.

Run:
  PYTHONPATH=src python test/test_autoregressive_cone_layer_basis.py
"""

from __future__ import annotations

import math

import torch

from rational_factor.models.autoregressive_basis import AutoregressiveConeLayerBasis
from rational_factor.models.basis_functions import BSpline1DBasis
from rational_factor.models.parameters import LowRankFactorizationParameters, PositiveParameters


SEED = 0
DIM = 2
N_BASIS = 6
N_LAYERS = 2
BSPLINE_DEGREE = 2
N_MC = 80_000
N_PROBE = 4_000
GRAM_ATOL = 0.02
POS_ATOL = 1e-7


def _positive_low_rank(d: int, m: int, r: int) -> LowRankFactorizationParameters:
    return LowRankFactorizationParameters(
        PositiveParameters.random_init((d, m, r), mean=0.0, std=0.5, epsilon=1e-6),
        PositiveParameters.random_init((d, m, r), mean=0.0, std=0.5, epsilon=1e-6),
    )


def _make_basis() -> AutoregressiveConeLayerBasis:
    n_cells = N_BASIS - BSPLINE_DEGREE
    nom_alpha = BSpline1DBasis(n_cells=n_cells, degree=BSPLINE_DEGREE)
    nom_beta = BSpline1DBasis(n_cells=n_cells, degree=BSPLINE_DEGREE)
    assert nom_alpha.n_basis_functions() == N_BASIS

    r = math.ceil(N_BASIS ** (1.0 / DIM))
    A0 = _positive_low_rank(DIM, N_BASIS, r)
    B0 = _positive_low_rank(DIM, N_BASIS, r)
    return AutoregressiveConeLayerBasis(
        nom_alpha,
        nom_beta,
        A0,
        B0,
        n_layers=N_LAYERS,
        hidden_features=16,
    )


def _uniform_gram(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Estimate ``∫_{[0,1]^D} α_i β_j`` with uniform Monte Carlo."""
    n = alpha.shape[0]
    return torch.einsum("ni,nj->ij", alpha, beta) / n


def main() -> None:
    torch.manual_seed(SEED)
    basis = _make_basis()

    y_probe = torch.rand(N_PROBE, DIM)
    alpha = basis.eval(y_probe, index=0)
    beta = basis.eval(y_probe, index=1)
    stacked = basis.eval(y_probe, index=None)

    assert alpha.shape == (N_PROBE, N_BASIS)
    assert beta.shape == (N_PROBE, N_BASIS)
    assert stacked.shape == (N_PROBE, 2, N_BASIS)
    assert torch.allclose(stacked[:, 0], alpha)
    assert torch.allclose(stacked[:, 1], beta)

    print(f"alpha range: {alpha.min().item():.4e} .. {alpha.max().item():.4e}")
    print(f"beta range:  {beta.min().item():.4e} .. {beta.max().item():.4e}")
    assert (alpha >= -POS_ATOL).all(), "alpha has negative values"
    assert (beta >= -POS_ATOL).all(), "beta has negative values"

    y_mc = torch.rand(N_MC, DIM)
    alpha_mc = basis.eval(y_mc, index=0)
    beta_mc = basis.eval(y_mc, index=1)
    assert (alpha_mc >= -POS_ATOL).all()
    assert (beta_mc >= -POS_ATOL).all()

    G_emp = _uniform_gram(alpha_mc, beta_mc)
    G = basis.Omega2().to_dense().detach()
    if G.dim() == 3:
        assert G.shape[0] == 1
        G = G.squeeze(0)

    err = (G_emp - G).abs().max().item()
    print(f"Omega2:\n{G}")
    print(f"empirical Gram:\n{G_emp}")
    print(f"max |G_emp - Omega2|: {err:.4e}")
    assert err < GRAM_ATOL, f"empirical Gram mismatch: {err} >= {GRAM_ATOL}"

    assert torch.allclose(basis.get_basis(0)(y_probe), alpha)
    assert torch.allclose(basis.get_basis(1)(y_probe), beta)

    print("ok")


if __name__ == "__main__":
    main()
