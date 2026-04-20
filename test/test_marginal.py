"""
Randomized checks that 2D marginals of FF density models integrate to ~1.

Basis marginal conventions in this test:
  - QuadraticExpBasis, UnnormalizedBetaBasis, GaussianBasis, BetaBasis: marginal_dims = coordinate indices to *keep* (always two).

Models are built like scripts/composite_experiments/vdp_nfdf_gaussian.py: bases from
`random_init`, then `LinearFF.from_rff` / `Linear2FF.from_r2ff` / `QuadraticFF.from_rff`.
Trainable parameters on the resulting density model are then re-randomized (in place) to
mimic post-optimization values.

Edit the CONFIG block below, then run:
  PYTHONPATH=src python test/test_marginal.py
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Sequence, Type

import matplotlib.pyplot as plt
import torch

from rational_factor.models.basis_functions import (
    BetaBasis,
    GaussianBasis,
    QuadraticExpBasis,
    UnnormalizedBetaBasis,
)
from rational_factor.models.composite_model import CompositeDensityModel
from rational_factor.models.domain_transformation import ErfSeparableTF
from rational_factor.models.factor_forms import Linear2FF, LinearFF, QuadraticFF
from rational_factor.tools.analysis import check_pdf_valid
from rational_factor.tools.visualization import plot_belief

# --- edit these -------------------------------------------------------------

# "quadratic_exp" | "unnormalized_beta" | "gaussian" | "beta"
BASIS = "quadratic_exp"

# Subset of: "linear_ff", "linear2ff", "quadratic_ff", "composite_erf_linear_ff"
MODELS = ("linear_ff", "linear2ff", "composite_erf_linear_ff")

DIM = 6  # full dimension d, must be >= 2
N_BASIS = 12
N_MARGINAL_SAMPLES = 4  # random 2D marginals per model (pairs of coordinate indices)
N_MC = 10000
SEED = 2

USE_CPU = False

# If True, saves one figure with a contour plot for each (model, marginal) trial.
PLOT_MARGINALS = True
PLOT_PATH = Path("figures/test_marginal__2d_marginals.png")
N_PLOT_POINTS = 80

# `random_init` hyperparameters (same spirit as vdp_nfdf_gaussian.py for GaussianBasis)
VARIANCE = 20.0
# QuadraticExpBasis: small variance + centered offsets so modes sit near 0 and most mass
# lies inside the quadratic_exp MC box (-10, 10) (large VARIANCE spreads exp(a x^2+b x) tails).
QUADRATIC_EXP_INIT_VARIANCE = 1.35
QUADRATIC_EXP_OFFSETS = (0.0, 0.0)  # len must match n_params_per_basis == 2 (raw_a, b)
QUADRATIC_EXP_EPS = 1e-6
# UnnormalizedBetaBasis.random_init
UNNORMALIZED_BETA_OFFSETS = (0.0, 0.0)
UNNORMALIZED_BETA_VARIANCE = 1.0
UNNORMALIZED_BETA_MIN_CONCENTRATION = 1.0
UNNORMALIZED_BETA_EPS = 1e-6
# GaussianBasis.random_init (cf. scripts using offsets [0, 30], variance 20)
GAUSSIAN_OFFSETS = (0.0, 30.0)
GAUSSIAN_MIN_STD = 1e-3
# BetaBasis.random_init
BETA_OFFSETS = (0.0, 0.0)
BETA_MIN_CONCENTRATION = 1.0
BETA_EPS = 1e-6

# Standard deviation for Gaussian noise written into each *trainable* parameter after build
PARAM_RAND_STD = 1.0

# CompositeDensityModel(ErfSeparableTF, base density) checks happen in x-space over a finite
# window that captures essentially all Gaussian-CDF tail mass.
ERF_LOC = 0.0
ERF_SCALE = 1.0
ERF_BOX = (-6.0, 6.0)

# ---------------------------------------------------------------------------

SeparableBasisType = (
    Type[QuadraticExpBasis] | Type[UnnormalizedBetaBasis] | Type[GaussianBasis] | Type[BetaBasis]
)

# (basis class, how marginal_dims is interpreted, per-dim domain (low, high) for MC)
_BASIS_REGISTRY: dict[str, tuple[SeparableBasisType, str, tuple[float, float]]] = {
    "quadratic_exp": (QuadraticExpBasis, "keep", (-10.0, 10.0)),
    # Use near-full support to match analytic Beta integrals in basis marginal/Omega ops.
    "unnormalized_beta": (UnnormalizedBetaBasis, "keep", (1e-6, 1.0 - 1e-6)),
    "gaussian": (GaussianBasis, "keep", (-12.0, 12.0)),
    "beta": (BetaBasis, "keep", (1e-6, 1.0 - 1e-6)),
}


def _randomize_trainable_params(module: torch.nn.Module, std: float, seed: int) -> None:
    """In-place Gaussian reinitialization of every parameter with requires_grad=True."""
    gens: dict[tuple[str, int | None], torch.Generator] = {}

    def _generator_for(device: torch.device) -> torch.Generator:
        key = (device.type, device.index)
        if key not in gens:
            gen = torch.Generator(device=device.type)
            # Keep deterministic-but-distinct streams per device.
            gen.manual_seed(seed + len(gens))
            gens[key] = gen
        return gens[key]

    with torch.no_grad():
        for p in module.parameters():
            if p.requires_grad:
                gen = _generator_for(p.device)
                noise = torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=gen)
                p.copy_(noise * std)


def _random_basis(
    basis_name: str,
    d: int,
    n_basis: int,
    device: torch.device,
    dtype: torch.dtype,
    freeze_params: bool = False,
):
    """Single `random_init` basis (per BASIS kind). QuadraticExp uses its own init variance."""
    cls = _BASIS_REGISTRY[basis_name][0]

    if basis_name == "quadratic_exp":
        off = torch.tensor(QUADRATIC_EXP_OFFSETS, device=device, dtype=dtype)
        basis = cls.random_init(
            d,
            n_basis,
            offsets=off,
            variance=QUADRATIC_EXP_INIT_VARIANCE,
            eps=QUADRATIC_EXP_EPS,
        )
        if freeze_params:
            basis = basis.freeze_params()
        return basis
    if basis_name == "unnormalized_beta":
        off = torch.tensor(UNNORMALIZED_BETA_OFFSETS, dtype=dtype)
        basis = cls.random_init(
            d,
            n_basis,
            offsets=off,
            variance=VARIANCE,
            min_concentration=UNNORMALIZED_BETA_MIN_CONCENTRATION,
            eps=UNNORMALIZED_BETA_EPS,
        ).to(device=device, dtype=dtype)
        if freeze_params:
            basis = basis.freeze_params()
        return basis
    if basis_name == "gaussian":
        off = torch.tensor(GAUSSIAN_OFFSETS, dtype=dtype)
        basis = cls.random_init(
            d,
            n_basis,
            offsets=off,
            variance=VARIANCE,
            min_std=GAUSSIAN_MIN_STD,
            device=device,
        ).to(device=device, dtype=dtype)
        if freeze_params:
            basis = basis.freeze_params()
        return basis
    if basis_name == "beta":
        off = torch.tensor(BETA_OFFSETS, dtype=dtype)
        basis = cls.random_init(
            d,
            n_basis,
            offsets=off,
            variance=VARIANCE,
            min_concentration=BETA_MIN_CONCENTRATION,
            eps=BETA_EPS,
            device=device,
        ).to(device=device, dtype=dtype)
        if freeze_params:
            basis = basis.freeze_params()
        return basis

    raise ValueError(f"Unknown basis name: {basis_name}")


def make_linear_ff(
    basis_name: str, d: int, n_basis: int, device: torch.device, dtype: torch.dtype
) -> LinearFF:
    phi = _random_basis(basis_name, d, n_basis, device, dtype, freeze_params=True)
    psi0 = _random_basis(basis_name, d, n_basis, device, dtype)
    a = torch.randn(n_basis, device=device, dtype=dtype).softmax(0)
    return LinearFF(a, phi, psi0)


def make_linear_2ff(
    basis_name: str, d: int, n_basis: int, device: torch.device, dtype: torch.dtype
) -> Linear2FF:
    xi = _random_basis(basis_name, d, n_basis, device, dtype, freeze_params=True)
    phi = _random_basis(basis_name, d, n_basis, device, dtype, freeze_params=True)
    psi0 = _random_basis(basis_name, d, n_basis, device, dtype)

    d = torch.randn(n_basis, device=device, dtype=dtype).softmax(0)
    a = torch.randn(n_basis, device=device, dtype=dtype).softmax(0)
    return Linear2FF(d, xi, a, phi, psi0)


def make_quadratic_ff(
    basis_name: str, d: int, n_basis: int, device: torch.device, dtype: torch.dtype
) -> QuadraticFF:
    phi = _random_basis(basis_name, d, n_basis, device, dtype, freeze_params=True)
    psi0 = _random_basis(basis_name, d, n_basis, device, dtype)
    Au = torch.randn(n_basis, n_basis, device=device, dtype=dtype).softmax(0)
    A = Au @ Au.T
    return QuadraticFF(A, phi, psi0)


def make_composite_erf_linear_ff(
    basis_name: str, d: int, n_basis: int, device: torch.device, dtype: torch.dtype
) -> CompositeDensityModel:
    del basis_name
    base_density = make_linear_ff("beta", d, n_basis, device, dtype)
    loc = torch.full((d,), ERF_LOC, device=device, dtype=dtype)
    scale = torch.full((d,), ERF_SCALE, device=device, dtype=dtype)
    domain_tf = ErfSeparableTF(d, loc=loc, scale=scale, trainable=False)
    return CompositeDensityModel(domain_tf, base_density)


_MODEL_BUILDERS: dict[
    str,
    Callable[
        [str, int, int, torch.device, torch.dtype],
        LinearFF | Linear2FF | QuadraticFF | CompositeDensityModel,
    ],
] = {
    "linear_ff": make_linear_ff,
    "linear2ff": make_linear_2ff,
    "quadratic_ff": make_quadratic_ff,
    "composite_erf_linear_ff": make_composite_erf_linear_ff,
}


def full_box_bounds(d: int, per_dim: tuple[float, float]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    lo, hi = per_dim
    return (tuple(lo for _ in range(d)), tuple(hi for _ in range(d)))


def marginal_box_bounds(
    d: int,
    marginal_dims: tuple[int, ...],
    convention: str,
    per_dim: tuple[float, float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    full_lo, full_hi = full_box_bounds(d, per_dim)
    if convention == "keep":
        keep = marginal_dims
    elif convention == "integrate_out":
        keep = tuple(i for i in range(d) if i not in set(marginal_dims))
    else:
        raise ValueError(convention)
    return (
        tuple(full_lo[i] for i in keep),
        tuple(full_hi[i] for i in keep),
    )


def model_per_dim_bounds(model_name: str, basis_name: str) -> tuple[float, float]:
    if model_name == "composite_erf_linear_ff":
        return ERF_BOX
    return _BASIS_REGISTRY[basis_name][2]


def random_marginal_arg(d: int, convention: str, rng: random.Random) -> tuple[int, ...]:
    """Return marginal_dims suitable for DensityModel.marginal (per basis convention).

    For convention ``keep``, always returns exactly two distinct coordinate indices so the
    marginal density is 2D.
    """
    if d < 2:
        raise ValueError("Need dim >= 2 to form a 2D marginal.")

    if convention == "keep":
        return tuple(sorted(rng.sample(range(d), 2)))

    # integrate_out: remove a non-empty proper subset so the kept marginal is 2D when d > 2
    if d == 2:
        raise ValueError("integrate_out convention needs dim > 2 for a 2D marginal.")
    m = d - 2  # number of coordinates to integrate out
    return tuple(sorted(rng.sample(range(d), m)))


def run_trial(
    *,
    basis_name: str,
    model_names: Sequence[str],
    dim: int,
    n_basis: int,
    n_marginal_samples: int,
    n_mc: int,
    seed: int,
    device: torch.device,
    param_rand_std: float,
    plot_marginals: bool = False,
    plot_path: Path | None = None,
    n_plot_points: int = 80,
) -> None:
    _, convention, _ = _BASIS_REGISTRY[basis_name]
    rng = random.Random(seed)
    torch.manual_seed(seed)
    dtype = torch.float32

    if plot_marginals:
        nrows, ncols = len(model_names), n_marginal_samples
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False)

    for mi, mname in enumerate(model_names):
        builder = _MODEL_BUILDERS[mname]
        model = builder(basis_name, dim, n_basis, device, dtype)
        model = model.to(device=device)
        model.eval()
        per_dim = model_per_dim_bounds(mname, basis_name)

        _randomize_trainable_params(model, param_rand_std, seed)

        for t in range(n_marginal_samples):
            arg = random_marginal_arg(dim, convention, rng)
            m_model = model.marginal(arg)
            m_d = m_model.dim
            if m_d != 2:
                raise RuntimeError(f"Expected 2D marginal, got dim {m_d} for marginal_dims {arg}.")

            bounds = marginal_box_bounds(dim, arg, convention, per_dim)
            print(f"\n{basis_name} / {mname} / marginal {arg} -> dim {m_d} (trial {t + 1}/{n_marginal_samples})")
            check_pdf_valid(m_model, bounds, n_samples=n_mc, device=device)

            if plot_marginals:
                assert plot_path is not None
                lo, hi = bounds
                x_range = (float(lo[0]), float(hi[0]))
                y_range = (float(lo[1]), float(hi[1]))
                ax = axes[mi, t]
                plot_belief(ax, m_model, x_range=x_range, y_range=y_range, n_points=n_plot_points)
                ax.set_title(f"{mname}\n{arg}", fontsize=9)

    if plot_marginals:
        assert plot_path is not None
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.suptitle(f"{basis_name} basis — 2D marginals", fontsize=11, y=1.02)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved marginal figure to {plot_path.resolve()}")


def main() -> None:
    if BASIS not in _BASIS_REGISTRY:
        raise ValueError(f"BASIS must be one of {sorted(_BASIS_REGISTRY.keys())}, got {BASIS!r}")

    if DIM < 2:
        raise ValueError("DIM must be >= 2")

    unknown = [m for m in MODELS if m not in _MODEL_BUILDERS]
    if unknown:
        raise ValueError(f"Unknown MODELS entries {unknown}; choose from {sorted(_MODEL_BUILDERS.keys())}")

    device = torch.device("cpu") if USE_CPU else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_trial(
        basis_name=BASIS,
        model_names=MODELS,
        dim=DIM,
        n_basis=N_BASIS,
        n_marginal_samples=N_MARGINAL_SAMPLES,
        n_mc=N_MC,
        seed=SEED,
        device=device,
        param_rand_std=PARAM_RAND_STD,
        plot_marginals=PLOT_MARGINALS,
        plot_path=PLOT_PATH if PLOT_MARGINALS else None,
        n_plot_points=N_PLOT_POINTS,
    )
    print("\nAll marginal PDF checks passed.")


if __name__ == "__main__":
    main()
