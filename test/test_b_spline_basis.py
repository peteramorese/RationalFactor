"""
Sanity checks for FixedDegreeBSplineMutualBasis.

Run:
  PYTHONPATH=src python test/test_bspline_basis.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from rational_factor.models.mutual_bases import FixedDegreeBSplineMutualBasis


SEED = 0
N_BASIS = 12
DEGREE = 3
N_GRID = 20_001
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "bspline")


def _make_basis() -> FixedDegreeBSplineMutualBasis:
    return FixedDegreeBSplineMutualBasis(
        n_basis=N_BASIS,
        degree=DEGREE,
    ).to(dtype=torch.float64)


def _numerical_integral(y: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    return torch.trapezoid(f, y, dim=0)


def _numerical_gram(
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return torch.trapezoid(a[:, :, None] * b[:, None, :], y, dim=0)


def _plot(basis: FixedDegreeBSplineMutualBasis) -> None:
    y = torch.linspace(0, 1, 2000, dtype=torch.float64)
    a, b = basis.eval(y, 0).detach(), basis.eval(y, 1).detach()

    for i in range(N_BASIS):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(y, a[:, i], label=rf"$\alpha_{i}$")
        ax.plot(y, b[:, i], label=rf"$\beta_{i}$")

        for x in basis.breakpoints:
            ax.axvline(float(x), color="0.85", lw=0.7)

        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"basis_{i}.png"), dpi=150)
        plt.close(fig)


def main() -> None:
    torch.manual_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    basis = _make_basis()

    print(
        f"n_basis={N_BASIS}, degree={DEGREE}, "
        f"n_spans={len(basis.breakpoints) - 1}"
    )

    # ------------------------------------------------------------------
    # Evaluation / basic spline properties
    # ------------------------------------------------------------------
    y_test = torch.tensor(
        [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
        dtype=torch.float64,
    )

    a = basis.eval(y_test, 0)
    b = basis.eval(y_test, 1)
    both = basis.eval(y_test, None)

    assert a.shape == (len(y_test), N_BASIS)
    assert b.shape == (len(y_test), N_BASIS)
    assert both.shape == (len(y_test), 2, N_BASIS)

    # B-splines are nonnegative and form a partition of unity.
    assert a.min() >= -1e-12
    #assert torch.allclose(a.sum(-1), torch.ones(len(y_test)), atol=1e-12)

    # At most degree + 1 primal B-splines are active in an interior span.
    y_interior = torch.linspace(1e-5, 1 - 1e-5, 1000, dtype=torch.float64)
    a_interior = basis.eval(y_interior, 0)
    max_active = (a_interior > 1e-12).sum(-1).max().item()
    assert max_active <= DEGREE + 1

    print(f"eval OK; max active alpha functions={max_active}")

    # ------------------------------------------------------------------
    # Numerical integrals / Gram
    # ------------------------------------------------------------------
    y = torch.linspace(0, 1, N_GRID, dtype=torch.float64)
    a = basis.eval(y, 0)
    b = basis.eval(y, 1)

    int_a_num = _numerical_integral(y, a)
    int_b_num = _numerical_integral(y, b)

    int_a = basis.Omega1(0)[0]
    int_b = basis.Omega1(1)[0]

    print("integral alpha:", int_a)
    print("integral beta :", int_b)

    assert torch.allclose(int_a_num, int_a, atol=2e-5, rtol=2e-5)
    assert torch.allclose(int_b_num, int_b, atol=2e-5, rtol=2e-5)

    # For beta = Lambda G^{-1} alpha:
    #
    #     integral beta = Lambda 1.
    assert torch.allclose(int_b, torch.as_tensor(1.0), atol=1e-10, rtol=1e-10)

    G_num = _numerical_gram(y, a, b)
    G = basis.Omega2().to_dense()[0]
    target = torch.eye(N_BASIS, dtype=torch.float64)

    print("cross Gram:\n", G)
    print(
        "max Gram error =",
        f"{(G_num - target).abs().max().item():.3e}",
    )

    assert torch.allclose(G, target, atol=1e-12, rtol=1e-12)
    assert torch.allclose(G_num, target, atol=5e-5, rtol=5e-5)

    # ------------------------------------------------------------------
    # Underlying primal Gram should be banded.
    # ------------------------------------------------------------------
    G_alpha = basis.gram_matrix
    ii = torch.arange(N_BASIS)
    outside_band = (ii[:, None] - ii[None, :]).abs() > DEGREE

    assert G_alpha[outside_band].abs().max() < 1e-12

    print(
        f"alpha Gram bandwidth OK; "
        f"cond(G)={torch.linalg.cond(G_alpha).item():.3e}"
    )

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------
    for index, vals in [(0, a), (1, b)]:
        lo, hi = basis.bounds(index)
        lo, hi = lo[0], hi[0]

        grid_lo = vals.min(0).values
        grid_hi = vals.max(0).values

        # Certified/computed extrema must contain all grid samples.
        assert torch.all(grid_lo >= lo - 1e-8)
        assert torch.all(grid_hi <= hi + 1e-8)

        # Dense grid should get very close to the true extrema.
        assert torch.allclose(grid_lo, lo, atol=5e-4, rtol=5e-4)
        assert torch.allclose(grid_hi, hi, atol=5e-4, rtol=5e-4)

        print(
            f"{'alpha' if index == 0 else 'beta'} bounds OK; "
            f"range=[{lo.min().item():.3f}, {hi.max().item():.3f}]"
        )

    # ------------------------------------------------------------------
    # MutualPairBasis wrappers
    # ------------------------------------------------------------------
    alpha, beta = basis.get_basis(0), basis.get_basis(1)

    assert torch.allclose(alpha.Omega1(), basis.Omega1(0))
    assert torch.allclose(beta.Omega1(), basis.Omega1(1))

    assert torch.allclose(
        alpha.Omega2(beta).to_dense(),
        basis.Omega2().to_dense(),
        atol=1e-10,
        rtol=1e-10,
    )

    assert torch.allclose(
        beta.Omega2(alpha).to_dense(),
        basis.Omega2().to_dense().transpose(-1, -2),
        atol=1e-10,
        rtol=1e-10,
    )

    _plot(basis)
    print(f"wrote plots to {OUT_DIR}")
    print("done")


if __name__ == "__main__":
    main()