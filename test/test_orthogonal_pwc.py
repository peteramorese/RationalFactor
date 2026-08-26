"""
Sanity checks for Orthogonal1DPWCBasis: plots, exact integrals, Gram, bounds.

Run:
  PYTHONPATH=src python test/test_orthogonal_pwc.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from rational_factor.models.mutual_bases import Orthogonal1DPWCBasis
from rational_factor.models.parameters import Parameters
from rational_factor.models.qs_matrix import Order1QuasiseparableFactorization


SEED = 0
N_BASIS = 5
N_PLOT = 400
N_MC = 200_000
EXACT_MEAN_TOL = 5e-5
EXACT_GRAM_REL_TOL = 2e-3
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "orthogonal_pwc")


def _randn(g: torch.Generator, n_basis: int) -> Parameters:
    return Parameters(torch.randn(1, n_basis, generator=g))


def _make_basis(
    n_basis: int,
    seed: int,
    *,
    gram_diag: bool = True,
    transition_bound: float | None = 0.99,
) -> Orthogonal1DPWCBasis:
    g = torch.Generator().manual_seed(seed)
    fac = Order1QuasiseparableFactorization(
        _randn(g, n_basis),
        _randn(g, n_basis),
        _randn(g, n_basis),
        _randn(g, n_basis),
        _randn(g, n_basis),
        _randn(g, n_basis),
        _randn(g, n_basis),
        transition_bound=transition_bound,
    )
    lam = None
    if gram_diag:
        lam = Parameters(0.5 + torch.rand(1, n_basis, generator=g))
    return Orthogonal1DPWCBasis(fac, gram_diag_params=lam)


def _cell_widths(basis: Orthogonal1DPWCBasis) -> torch.Tensor:
    return torch.full((basis.n_cells,), 1.0 / basis.n_cells)


def _mc_basis_integrals(basis: Orthogonal1DPWCBasis, index: int, n_samples: int) -> torch.Tensor:
    y = torch.rand(n_samples, 1)
    return basis.eval(y, index=index).mean(dim=0)


def _mc_gram(basis: Orthogonal1DPWCBasis, n_samples: int) -> torch.Tensor:
    y = torch.rand(n_samples, 1)
    a, b = basis.eval(y, index=0), basis.eval(y, index=1)
    return torch.einsum("ni,nj->ij", a, b) / a.shape[0]


def _cell_midpoints(basis: Orthogonal1DPWCBasis) -> torch.Tensor:
    edges = basis.cell_edges()
    return 0.5 * (edges[:-1] + edges[1:])


def _exact_basis_integrals(basis: Orthogonal1DPWCBasis, index: int) -> torch.Tensor:
    widths = _cell_widths(basis)
    y = _cell_midpoints(basis).unsqueeze(-1)
    return (widths.unsqueeze(-1) * basis.eval(y, index=index)).sum(dim=0)


def _exact_gram(basis: Orthogonal1DPWCBasis) -> torch.Tensor:
    widths = _cell_widths(basis)
    y = _cell_midpoints(basis).unsqueeze(-1)
    a, b = basis.eval(y, index=0), basis.eval(y, index=1)
    return torch.einsum("k,ki,kj->ij", widths, a, b)


def _plot_bases(basis: Orthogonal1DPWCBasis, out_dir: str) -> list[str]:
    edges = basis.cell_edges().detach().cpu()
    y = torch.linspace(0.0, 1.0, N_PLOT).unsqueeze(-1)
    y[-1, 0] = min(1.0 - 1e-6, float(y[-1, 0]))

    alpha = basis.eval(y, index=0).detach().cpu()
    beta = basis.eval(y, index=1).detach().cpu()
    x = y.squeeze(-1).cpu()
    paths = []

    for i in range(alpha.shape[1]):
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


def _check_factorization(basis: Orthogonal1DPWCBasis) -> None:
    P = basis._get_P()
    F = basis.get_feasible_matrix().to_dense(dtype=P.d.dtype, device=P.d.device)
    Pd = P.to_dense()[0]
    gen = P.direct_generators().to_dense()[0]
    assert torch.allclose(Pd, gen, atol=1e-5, rtol=1e-4), "direct generators != LDU dense"

    x = torch.randn(P.n)
    Px = P.matvec(x.unsqueeze(0)).squeeze(0)
    assert torch.allclose(Px, Pd @ x, atol=1e-5, rtol=1e-4)
    assert torch.allclose(P.inverse_matvec(P.matvec(x.unsqueeze(0))), x.unsqueeze(0), atol=1e-4, rtol=1e-4)
    assert torch.allclose(P.invT_matvec(P.T_matvec(x.unsqueeze(0))), x.unsqueeze(0), atol=1e-4, rtol=1e-4)

    p = F.shape[1]
    gram_F = (F @ F.T) / p
    assert torch.allclose(gram_F, torch.eye(P.n), atol=1e-5, rtol=1e-4), "F is not orthonormal"
    assert torch.allclose(F.sum(-1), torch.zeros(P.n), atol=1e-5), "F 1 != 0"

    invT_gen = P.inverse_transpose_generators().to_dense()[0]
    PinvT = torch.linalg.solve(Pd.T, torch.eye(P.n, dtype=Pd.dtype))
    assert torch.allclose(invT_gen, PinvT, atol=1e-4, rtol=1e-4), "P^{-T} generators mismatch"


def main() -> None:
    torch.manual_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    basis = _make_basis(N_BASIS, SEED)
    _check_factorization(basis)

    P = basis._get_P()
    print(f"n_basis={N_BASIS}, n_cells={basis.n_cells}, ||d||={P.d.norm().item():.3f}")

    for path in _plot_bases(basis, OUT_DIR):
        print(f"wrote {path}")

    y = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.999])
    a, b = basis.eval(y, index=0), basis.eval(y, index=1)
    both = basis.eval(y, index=None)
    assert a.shape == (y.numel(), N_BASIS) and b.shape == (y.numel(), N_BASIS)
    assert both.shape == (y.numel(), 2, N_BASIS)
    print(
        f"eval shapes OK; max nonzero α-comps/row={(a.abs() > 1e-8).sum(-1).max().item()}, "
        f"β={(b.abs() > 1e-8).sum(-1).max().item()}"
    )

    int_alpha_mc = _mc_basis_integrals(basis, index=0, n_samples=N_MC)
    int_beta_mc = _mc_basis_integrals(basis, index=1, n_samples=N_MC)
    print(
        f"MC  max |∫α|={int_alpha_mc.abs().max().item():.3e}, "
        f"max |∫β|={int_beta_mc.abs().max().item():.3e}  "
        f"(~1/sqrt(N)={N_MC ** -0.5:.1e} sampling noise)"
    )

    int_alpha = _exact_basis_integrals(basis, index=0)
    int_beta = _exact_basis_integrals(basis, index=1)
    print(
        f"exact max |∫α|={int_alpha.abs().max().item():.3e}, "
        f"max |∫β|={int_beta.abs().max().item():.3e}"
    )
    assert int_alpha.abs().max().item() < EXACT_MEAN_TOL, f"α integrals not ~0: {int_alpha}"
    assert int_beta.abs().max().item() < EXACT_MEAN_TOL, f"β integrals not ~0: {int_beta}"

    Omega1_a = basis.Omega1(0)
    Omega1_b = basis.Omega1(1)
    assert torch.allclose(Omega1_a, torch.zeros_like(Omega1_a))
    assert torch.allclose(Omega1_b, torch.zeros_like(Omega1_b))

    G_mc = _mc_gram(basis, n_samples=N_MC)
    print("MC Gram max |off-diag|=", f"{(G_mc - torch.diag(torch.diag(G_mc))).abs().max().item():.3e}")

    G = _exact_gram(basis)
    off = G - torch.diag(torch.diag(G))
    print("exact Gram:\n", G)
    print(f"exact max |off-diag|={off.abs().max().item():.3e}")
    rel = off.abs().max() / G.diag().abs().max().clamp_min(1e-12)
    print(f"exact relative off-diag={rel.item():.3e}")
    assert rel.item() < EXACT_GRAM_REL_TOL, f"Gram not diagonal: rel={rel.item()}"

    G_omega = basis.Omega2()[0]
    assert torch.allclose(G, G_omega, atol=1e-4, rtol=1e-4), "Omega2 != exact Gram"

    alpha, beta = basis.get_basis(0), basis.get_basis(1)
    assert torch.allclose(alpha.Omega1(), Omega1_a)
    assert torch.allclose(alpha.Omega2(beta)[0], G_omega, atol=1e-4, rtol=1e-4)
    assert torch.allclose(beta.Omega2(alpha)[0], G_omega.T, atol=1e-4, rtol=1e-4)

    mids = _cell_midpoints(basis)
    for index in (0, 1):
        vals = basis.eval(mids, index=index)
        lo, hi = basis.bounds(index)
        assert torch.allclose(vals.min(0).values, lo[0], atol=1e-4, rtol=1e-4)
        assert torch.allclose(vals.max(0).values, hi[0], atol=1e-4, rtol=1e-4)
    print("bounds match cell extrema")

    print("done")


if __name__ == "__main__":
    main()
