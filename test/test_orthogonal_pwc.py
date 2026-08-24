"""
Sanity checks for Orthogonal1DPWCBasis: plots, MC integrals, empirical Gram.

Run:
  PYTHONPATH=src python test/test_orthogonal_pwc.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from rational_factor.models.lrpd import Rank1PlusDiagonalFactorization
from rational_factor.models.mutual_bases import Orthogonal1DPWCBasis
from rational_factor.models.parameters import Parameters


SEED = 0
N_BASIS = 20
N_PLOT = 400
N_MC = 200_000
EXACT_MEAN_TOL = 5e-5
EXACT_GRAM_REL_TOL = 2e-3
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "orthogonal_pwc")


def _make_basis(
    n_basis: int,
    seed: int,
    *,
    zero_mean: bool = True,
    diagonal_L: bool = False,
    n_l_factors: int = 5,
    rank1_scale: float = 0.75,
) -> Orthogonal1DPWCBasis:
    g = torch.Generator().manual_seed(seed)

    alpha = 0.5 + torch.rand(1, n_basis, generator=g)
    beta = 0.5 + torch.rand(1, n_basis, generator=g)
    # unconstrained logits → softmax to a simplex of cell widths
    width_logits = torch.randn(1, n_basis, generator=g)

    # Stronger mixing than a near-diagonal L, still float32-stable.
    d = 0.5 + 1.5 * torch.rand(n_l_factors, n_basis - 1, generator=g)
    if diagonal_L:
        u = torch.zeros(n_l_factors, n_basis - 1)
        v = torch.zeros(n_l_factors, n_basis - 1)
    else:
        u = rank1_scale * torch.randn(n_l_factors, n_basis - 1, generator=g)
        v = rank1_scale * torch.randn(n_l_factors, n_basis - 1, generator=g)

    r1pd = Rank1PlusDiagonalFactorization(Parameters(d), Parameters(u), Parameters(v))
    last_alpha = torch.randn(1, n_basis, generator=g)
    last_beta = torch.randn(1, n_basis, generator=g)
    last_alpha[0, -1] = 1.0
    last_beta[0, -1] = 1.0
    return Orthogonal1DPWCBasis(
        value_params_1=Parameters(alpha),
        value_params_2=Parameters(beta),
        cell_width_params=Parameters(width_logits),
        r1pd_factorization=r1pd,
        zero_mean=zero_mean,
        last_alpha_cell_vector=Parameters(last_alpha),
        last_beta_cell_vector=Parameters(last_beta),
    )


def _mc_basis_integrals(
    basis: Orthogonal1DPWCBasis,
    index: int,
    n_samples: int,
) -> torch.Tensor:
    y = torch.rand(n_samples, 1)
    vals = basis.eval(y, index=index)
    return vals.mean(dim=0)  # domain volume = 1


def _mc_gram(basis: Orthogonal1DPWCBasis, n_samples: int) -> torch.Tensor:
    y = torch.rand(n_samples, 1)
    a = basis.eval(y, index=0)
    b = basis.eval(y, index=1)
    return torch.einsum("ni,nj->ij", a, b) / a.shape[0]


def _cell_midpoints(basis: Orthogonal1DPWCBasis) -> torch.Tensor:
    edges = basis.cell_edges()
    return 0.5 * (edges[:-1] + edges[1:])


def _exact_basis_integrals(basis: Orthogonal1DPWCBasis, index: int) -> torch.Tensor:
    """Exact ∫ on [0, 1] for PWC: Σ_cells width_k * value_k."""
    widths = basis._get_normalized_cell_widths()[0]
    y = _cell_midpoints(basis).unsqueeze(-1)
    return (widths.unsqueeze(-1) * basis.eval(y, index=index)).sum(dim=0)


def _exact_gram(basis: Orthogonal1DPWCBasis) -> torch.Tensor:
    """Exact Gram Σ_k width_k α(y_k) β(y_k)^T."""
    widths = basis._get_normalized_cell_widths()[0]
    y = _cell_midpoints(basis).unsqueeze(-1)
    a = basis.eval(y, index=0)
    b = basis.eval(y, index=1)
    return torch.einsum("k,ki,kj->ij", widths, a, b)


def _plot_bases(basis: Orthogonal1DPWCBasis, out_dir: str) -> list[str]:
    """Write one figure per basis index with that α_i and β_i."""
    edges = basis.cell_edges().detach().cpu()
    y = torch.linspace(0.0, 1.0, N_PLOT).unsqueeze(-1)
    y[-1, 0] = min(1.0 - 1e-6, float(y[-1, 0]))

    alpha = basis.eval(y, index=0).detach().cpu()
    beta = basis.eval(y, index=1).detach().cpu()
    x = y.squeeze(-1).cpu()
    n = alpha.shape[1]
    paths = []

    for i in range(n):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, alpha[:, i], label=rf"$\alpha_{{{i}}}$")
        ax.plot(x, beta[:, i], label=rf"$\beta_{{{i}}}$")
        for e in edges:
            ax.axvline(float(e), color="0.8", lw=0.8, zorder=0)
        ax.set_xlabel("y")
        ax.set_ylabel("value")
        ax.set_title(rf"Basis ${i}$ on $[0, 1]$")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"basis_{i}_alpha_beta.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths


def main() -> None:
    torch.manual_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    basis = _make_basis(N_BASIS, SEED, zero_mean=True, diagonal_L=False)
    widths = basis._get_normalized_cell_widths()[0]
    Lfac = basis._r1pd_factorization()
    print(f"widths={widths.tolist()}, sum={widths.sum().item():.6f}")
    print(f"M factors T={Lfac.d.shape[0]}, k={Lfac.d.shape[1]}, ||u||={Lfac.u.norm().item():.3f}")

    for path in _plot_bases(basis, OUT_DIR):
        print(f"wrote {path}")

    # Smoke: cell lookup + shape; non-diagonal L should mix support across cells
    y = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.999])
    a = basis.eval(y, index=0)
    b = basis.eval(y, index=1)
    assert a.shape == (y.numel(), N_BASIS) and b.shape == (y.numel(), N_BASIS)
    print(
        f"eval shapes OK; max nonzero α-comps/row={(a.abs() > 1e-8).sum(-1).max().item()}, "
        f"β={(b.abs() > 1e-8).sum(-1).max().item()}"
    )

    int_alpha_mc = _mc_basis_integrals(basis, index=0, n_samples=N_MC)
    int_beta_mc = _mc_basis_integrals(basis, index=1, n_samples=N_MC)
    print("MC  ∫ α:", int_alpha_mc.tolist())
    print("MC  ∫ β:", int_beta_mc.tolist())
    print(
        f"MC  max |∫α|={int_alpha_mc.abs().max().item():.3e}, "
        f"max |∫β|={int_beta_mc.abs().max().item():.3e}  "
        f"(~1/sqrt(N)={N_MC ** -0.5:.1e} sampling noise)"
    )

    int_alpha = _exact_basis_integrals(basis, index=0)
    int_beta = _exact_basis_integrals(basis, index=1)
    print("exact ∫ α:", int_alpha.tolist())
    print("exact ∫ β:", int_beta.tolist())
    print(
        f"exact max |∫α|={int_alpha.abs().max().item():.3e}, "
        f"max |∫β|={int_beta.abs().max().item():.3e}"
    )
    assert int_alpha.abs().max().item() < EXACT_MEAN_TOL, f"α integrals not ~0: {int_alpha}"
    assert int_beta.abs().max().item() < EXACT_MEAN_TOL, f"β integrals not ~0: {int_beta}"

    G_mc = _mc_gram(basis, n_samples=N_MC)
    off_mc = (G_mc - torch.diag(torch.diag(G_mc))).abs().max().item()
    print("MC Gram max |off-diag|=", f"{off_mc:.3e}")

    G = _exact_gram(basis)
    off = G - torch.diag(torch.diag(G))
    print("exact Gram:\n", G)
    print(f"exact max |off-diag|={off.abs().max().item():.3e}")
    rel = off.abs().max() / G.diag().abs().max().clamp_min(1e-12)
    print(f"exact relative off-diag={rel.item():.3e}")
    assert rel.item() < EXACT_GRAM_REL_TOL, f"Gram not diagonal: rel={rel.item()}"

    print("done")


if __name__ == "__main__":
    main()
