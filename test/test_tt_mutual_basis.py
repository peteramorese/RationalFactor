"""
Checks for TTMutualBasis.

Verifies positivity of (phi, psi) and that a Monte-Carlo empirical Gram
on the unit cube matches ``Omega2``.

Run:
  PYTHONPATH=src python test/test_tt_mutual_basis.py
"""

from __future__ import annotations

import torch

from rational_factor.models.basis_functions import BetaBasis
from rational_factor.models.mutual_bases import TTMutualBasis
from rational_factor.models.parameters import PositiveParameters


SEED = 0
DIM = 3
N_PRIMITIVE = 3
N_OUTPUT_MODES = 2
OUTPUT_MODE_SIZE = 2
RANK = 3
N_MC = 80_000
N_PROBE = 4_000
GRAM_ATOL = 0.02
POS_ATOL = 1e-7

N_BASIS = OUTPUT_MODE_SIZE ** N_OUTPUT_MODES


def _make_primitive() -> BetaBasis:
    shape = (1, DIM, N_PRIMITIVE)
    return BetaBasis(
        PositiveParameters.random_init(shape, mean=0.5, std=0.3, epsilon=0.5),
        PositiveParameters.random_init(shape, mean=0.5, std=0.3, epsilon=0.5),
    )


def _make_basis() -> TTMutualBasis:
    return TTMutualBasis(
        _make_primitive(),
        _make_primitive(),
        rank=RANK,
        n_output_modes=N_OUTPUT_MODES,
        output_mode_size=OUTPUT_MODE_SIZE,
    )


def _uniform_gram(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Estimate ``∫_{[0,1]^D} α_i β_j`` with uniform Monte Carlo."""
    n = alpha.shape[0]
    return torch.einsum("ni,nj->ij", alpha, beta) / n


def main() -> None:
    torch.manual_seed(SEED)
    basis = _make_basis()

    assert basis.dim() == DIM
    assert basis.n_basis_functions() == N_BASIS

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
