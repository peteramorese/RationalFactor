"""
Sanity checks for DisjointSupport1DPWCBasis: identity Gram, disjoint support.

Run:
  PYTHONPATH=src python test/test_disjoint_pwc.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from rational_factor.models.mutual_bases import DisjointSupport1DPWCBasis
from rational_factor.models.parameters import PositiveParameters

SEED = 0
N_BASIS = 5
N_PLOT = 400
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "disjoint_pwc")


def _make_basis(n_basis: int, seed: int) -> DisjointSupport1DPWCBasis:
    """Create a DisjointSupport1DPWCBasis with random cell widths and alphas."""
    g = torch.Generator().manual_seed(seed)
    
    # Normalized cell widths (sum to 1)
    cell_widths_params = PositiveParameters(
        trainable_init_values=torch.randn(1, n_basis, generator=g),
        normalization_dim=-1,  # Normalize along the n_basis dimension
        epsilon=0.0,
    )
    
    # Alpha values (positive)
    alpha_params = PositiveParameters(
        trainable_init_values=torch.randn(1, n_basis, generator=g),
        epsilon=1e-6,
    )
    
    return DisjointSupport1DPWCBasis(cell_widths_params, alpha_params)


def _plot_bases(basis: DisjointSupport1DPWCBasis, out_dir: str) -> list[str]:
    """Plot alpha and beta for each basis function."""
    edges = basis.cell_edges(batch_index=0).detach().cpu()
    y = torch.linspace(0.0, 1.0, N_PLOT).unsqueeze(-1)
    
    alpha = basis.eval(y, index=0).detach().cpu()
    beta = basis.eval(y, index=1).detach().cpu()
    x = y.squeeze(-1).cpu()
    paths = []

    # Plot all alpha functions
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(alpha.shape[1]):
        ax.plot(x, alpha[:, i], label=rf"$\alpha_{{{i}}}$")
    for e in edges:
        ax.axvline(float(e), color="0.8", lw=0.8, zorder=0)
    ax.set_xlabel("y")
    ax.set_ylabel("value")
    ax.set_title(r"$\alpha$ basis functions (disjoint support)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "alpha_basis_functions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # Plot all beta functions
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(beta.shape[1]):
        ax.plot(x, beta[:, i], label=rf"$\beta_{{{i}}}$")
    for e in edges:
        ax.axvline(float(e), color="0.8", lw=0.8, zorder=0)
    ax.set_xlabel("y")
    ax.set_ylabel("value")
    ax.set_title(r"$\beta$ basis functions (disjoint support)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "beta_basis_functions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    return paths


def _check_disjoint_support(basis: DisjointSupport1DPWCBasis) -> None:
    """Verify that only one basis function is active at each point."""
    y = torch.linspace(0.0, 0.999, 100)
    alpha = basis.eval(y, index=0)
    beta = basis.eval(y, index=1)
    
    # Count nonzero entries per row
    alpha_active = (alpha.abs() > 1e-8).sum(dim=-1)
    beta_active = (beta.abs() > 1e-8).sum(dim=-1)
    
    assert torch.all(alpha_active <= 1), "More than one alpha active at some point"
    assert torch.all(beta_active <= 1), "More than one beta active at some point"
    print(f"✓ Disjoint support verified: at most 1 alpha and 1 beta active per point")


def _check_gram_identity(basis: DisjointSupport1DPWCBasis) -> None:
    """Verify that Omega2 is the identity matrix."""
    G = basis.Omega2().to_dense()
    
    if G.ndim == 3:
        G = G[0]  # Take first batch
    
    I = torch.eye(basis.n_basis_functions(), dtype=G.dtype, device=G.device)
    
    assert torch.allclose(G, I, atol=1e-5, rtol=1e-4), f"Gram is not identity:\n{G}"
    print(f"✓ Gram matrix is identity (max |G - I| = {(G - I).abs().max().item():.2e})")


def _check_alpha_beta_product(basis: DisjointSupport1DPWCBasis) -> None:
    """Verify that alpha_i * beta_i = 1 on each cell."""
    y = torch.linspace(0.0, 0.999, 100)
    alpha = basis.eval(y, index=0)
    beta = basis.eval(y, index=1)
    
    # Pointwise product
    product = alpha * beta
    
    # Where alpha and beta are both active, product should be 1
    both_active = (alpha.abs() > 1e-8) & (beta.abs() > 1e-8)
    
    if both_active.any():
        products_active = product[both_active]
        expected = torch.ones_like(products_active)
        assert torch.allclose(products_active, expected, atol=1e-5), \
            f"alpha * beta != 1 on active cells: {products_active}"
        print(f"✓ alpha_i * beta_i = 1 on active cells (max |product - 1| = {(products_active - expected).abs().max().item():.2e})")


def _check_integrals(basis: DisjointSupport1DPWCBasis) -> None:
    """Check that integrals match cell overlaps."""
    alphas = basis._alpha_params()[0]
    widths = basis._cell_widths_params()[0]
    
    # Omega1 for alpha
    int_alpha = basis.Omega1(index=0, lows=None, highs=None)[0]
    expected_alpha = alphas * widths
    
    assert torch.allclose(int_alpha, expected_alpha, atol=1e-5), \
        f"alpha integrals mismatch: got {int_alpha}, expected {expected_alpha}"
    
    # Omega1 for beta
    int_beta = basis.Omega1(index=1, lows=None, highs=None)[0]
    expected_beta = (1.0 / alphas) * widths
    
    assert torch.allclose(int_beta, expected_beta, atol=1e-5), \
        f"beta integrals mismatch: got {int_beta}, expected {expected_beta}"
    
    print(f"✓ Integrals match expected values")


def main() -> None:
    torch.manual_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    basis = _make_basis(N_BASIS, SEED)
    
    print(f"n_basis={N_BASIS}, n_cells={basis.n_cells}")
    print(f"cell widths: {basis._cell_widths_params()[0].tolist()}")
    print(f"cell edges: {basis.cell_edges(batch_index=0).tolist()}")
    print(f"alpha values: {basis._alpha_params()[0].tolist()}")
    print(f"beta values (1/alpha): {(1.0 / basis._alpha_params()[0]).tolist()}")

    _check_disjoint_support(basis)
    _check_gram_identity(basis)
    _check_alpha_beta_product(basis)
    _check_integrals(basis)

    for path in _plot_bases(basis, OUT_DIR):
        print(f"wrote {path}")

    # Test eval with different indices
    y = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    a = basis.eval(y, index=0)
    b = basis.eval(y, index=1)
    both = basis.eval(y, index=None)
    
    assert a.shape == (y.numel(), N_BASIS), f"alpha shape: {a.shape}"
    assert b.shape == (y.numel(), N_BASIS), f"beta shape: {b.shape}"
    assert both.shape == (y.numel(), 2, N_BASIS), f"both shape: {both.shape}"
    print(f"✓ eval shapes OK")

    # Test bounds
    alpha_inf, alpha_sup = basis.bounds(0)
    beta_inf, beta_sup = basis.bounds(1)
    
    assert torch.allclose(alpha_inf, torch.zeros_like(alpha_inf)), "alpha infimum should be 0"
    assert torch.allclose(alpha_sup, basis._alpha_params()), "alpha supremum should be alpha values"
    assert torch.allclose(beta_inf, torch.zeros_like(beta_inf)), "beta infimum should be 0"
    assert torch.allclose(beta_sup, 1.0 / basis._alpha_params()), "beta supremum should be 1/alpha"
    print(f"✓ bounds correct")

    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    main()
