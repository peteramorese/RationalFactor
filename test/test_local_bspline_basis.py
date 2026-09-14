"""
Sanity checks for LocalBSplineMutualBasis.

Verifies empirical biorthogonality of (alpha, beta) and that the offline
alpha–b Gram (b = relu(-beta)) matches a dense quadrature reference.
Also plots the bases over [0, 1].

Run:
  PYTHONPATH=src python test/test_local_bspline_basis.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from rational_factor.models.masking_bases import LocalBSplineMutualBasis
from rational_factor.models.structured_matrices import Banded


SEED = 0
N_BASIS = 50
K_ALPHA = 8
K_BETA = 16
N_GRID = 20_001
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "local_bspline")


def _make_basis(*, trainable_beta: bool = False) -> LocalBSplineMutualBasis:
    return LocalBSplineMutualBasis(
        n_basis=N_BASIS,
        k_alpha=K_ALPHA,
        k_beta=K_BETA,
        trainable_beta=trainable_beta,
        dtype=torch.float64,
    )


def _numerical_gram(
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return torch.trapezoid(a[:, :, None] * b[:, None, :], y, dim=0)


def _plot(basis: LocalBSplineMutualBasis) -> None:
    y = torch.linspace(0, 1, 2000, dtype=torch.float64)
    a = basis.eval(y, 0).detach()
    b = basis.eval(y, 1).detach()
    bb = basis.eval_b(y).detach()
    y_np = y.numpy()

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(
        rf"LocalBSplineMutualBasis  ($n={N_BASIS}$, "
        rf"$k_\alpha={K_ALPHA}$, $k_\beta={K_BETA}$)"
    )

    cmap = plt.cm.viridis
    for i in range(N_BASIS):
        color = cmap(i / max(N_BASIS - 1, 1))
        axes[0].plot(y_np, a[:, i].numpy(), color=color, lw=1.4, label=rf"$\alpha_{{{i}}}$")
        axes[1].plot(y_np, b[:, i].numpy(), color=color, lw=1.4, label=rf"$\beta_{{{i}}}$")
        axes[2].plot(y_np, bb[:, i].numpy(), color=color, lw=1.4, label=rf"$b_{{{i}}}$")

    titles = (r"primal $\alpha$", r"dual $\beta$", r"$b=\mathrm{relu}(-\beta)$")
    for ax, title in zip(axes, titles):
        for x in basis.breakpoints:
            ax.axvline(float(x), color="0.85", lw=0.7)
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend(ncol=min(N_BASIS, 6), fontsize=8, loc="best")

    axes[2].set_xlabel("y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "alpha_beta_basis.png"), dpi=150)
    plt.close(fig)

    # One subplot per index: alpha_i and beta_i together (twin y-axes).
    n_cols = 5
    n_rows = (N_BASIS + n_cols - 1) // n_cols
    fig2, axes2 = plt.subplots(
        n_rows, n_cols, figsize=(3.2 * n_cols, 2.4 * n_rows), sharex=True
    )
    fig2.suptitle(
        rf"Paired $\alpha_i$, $\beta_i$  ($n={N_BASIS}$, "
        rf"$k_\alpha={K_ALPHA}$, $k_\beta={K_BETA}$)"
    )
    axes2_flat = axes2.ravel()
    for i in range(N_BASIS):
        ax = axes2_flat[i]
        ax_b = ax.twinx()
        ax.plot(y_np, a[:, i].numpy(), color="C0", lw=1.3, label=rf"$\alpha_{{{i}}}$")
        ax_b.plot(y_np, b[:, i].numpy(), color="C1", lw=1.3, label=rf"$\beta_{{{i}}}$")
        for x in basis.breakpoints:
            ax.axvline(float(x), color="0.88", lw=0.6)
        ax.set_title(rf"$i={i}$", fontsize=9)
        ax.grid(alpha=0.25)
        if i // n_cols == n_rows - 1:
            ax.set_xlabel("y")
        lines = ax.get_lines() + ax_b.get_lines()
        #ax.legend(lines, [ln.get_label() for ln in lines], fontsize=7, loc="best")
        ax.tick_params(axis="y", labelcolor="C0", labelsize=7)
        ax_b.tick_params(axis="y", labelcolor="C1", labelsize=7)
    for j in range(N_BASIS, len(axes2_flat)):
        axes2_flat[j].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "alpha_beta_pairs.png"), dpi=300)
    plt.close(fig2)


def _check_biorthogonality(basis: LocalBSplineMutualBasis, y: torch.Tensor) -> None:
    a = basis.eval(y, 0)
    b = basis.eval(y, 1)
    G_num = _numerical_gram(y, a, b)
    target = torch.eye(N_BASIS, dtype=torch.float64)
    G = basis.Omega2().to_dense()[0]

    print("cross Gram (structured):\n", G)
    print(
        "max empirical Gram error =",
        f"{(G_num - target).abs().max().item():.3e}",
    )

    assert torch.allclose(G, target, atol=1e-12, rtol=1e-12)
    assert torch.allclose(G_num, target, atol=5e-5, rtol=5e-5)


def _check_alpha_b_gram(basis: LocalBSplineMutualBasis, y: torch.Tensor) -> None:
    a = basis.eval(y, 0)
    bb = basis.eval_b(y)
    Gab_num = _numerical_gram(y, a, bb)
    Gab = basis.Omega2_alpha_b(recompute=True)

    assert isinstance(Gab, Banded)
    Gab_dense = Gab.to_dense()

    print(
        "max alpha-b Gram error =",
        f"{(Gab_num - Gab_dense).abs().max().item():.3e}",
    )
    print(f"alpha-b Frobenius norm = {Gab_dense.norm().item():.3e}")

    assert torch.allclose(Gab_num, Gab_dense, atol=5e-5, rtol=5e-5)

    # Cached path should match a fresh rebuild.
    Gab_cached = basis.Omega2_alpha_b(recompute=False)
    assert torch.allclose(Gab_cached.to_dense(), Gab_dense, atol=0.0, rtol=0.0)


def main() -> None:
    torch.manual_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    basis = _make_basis()
    print(
        f"n_basis={N_BASIS}, k_alpha={K_ALPHA}, k_beta={K_BETA}, "
        f"degree={basis.degree}, n_cells={len(basis.breakpoints) - 1}, "
        f"n_beta_trainable={basis.n_beta_trainable}"
    )

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
    assert a.min() >= -1e-12
    print("eval OK")

    y = torch.linspace(0, 1, N_GRID, dtype=torch.float64)

    _plot(basis)

    print("--- default beta_theta = 0 ---")
    _check_biorthogonality(basis, y)
    _check_alpha_b_gram(basis, y)

    # Nullspace DOFs must preserve biorthogonality exactly (up to quadrature).
    print("--- perturbed beta_theta ---")
    basis_t = _make_basis(trainable_beta=True)
    with torch.no_grad():
        basis_t.beta_theta.copy_(0.25 * torch.randn_like(basis_t.beta_theta))
    basis_t.rebuild_alpha_b_gram()

    _check_biorthogonality(basis_t, y)
    _check_alpha_b_gram(basis_t, y)

    print(f"wrote plots to {OUT_DIR}")
    print("done")


if __name__ == "__main__":
    main()
