"""
Randomized checks that lower-dimensional marginals of FF density models integrate to ~1.

Basis marginal conventions in this codebase differ by class:
  - GaussianBasis, BetaBasis: marginal_dims = coordinate indices to *keep*.
  - QuadraticExpBasis: marginal_dims = coordinate indices to *integrate out*.

Models are built like scripts/composite_experiments/vdp_nfdf_gaussian.py: bases from
`random_init`, then `LinearFF.from_rff` / `Linear2FF.from_r2ff` / `QuadraticFF.from_rff`.
Trainable parameters on the resulting density model are then re-randomized (in place) to
mimic post-optimization values.

Edit the CONFIG block below, then run:
  PYTHONPATH=src python test/test_marginal.py
"""

from __future__ import annotations

import random
from typing import Callable, Sequence, Type

import torch

from rational_factor.models.basis_functions import BetaBasis, GaussianBasis, QuadraticExpBasis
from rational_factor.models.factor_forms import Linear2FF, LinearFF, LinearR2FF, LinearRFF, QuadraticFF, QuadraticRFF
from rational_factor.tools.analysis import check_pdf_valid

# --- edit these -------------------------------------------------------------

# "gaussian" | "beta" | "quadratic_exp"
BASIS = "gaussian"

# Subset of: "linear_ff", "linear2ff", "quadratic_ff"
MODELS = ("linear_ff", "linear2ff", "quadratic_ff")

DIM = 6  # full dimension d, must be >= 2
N_BASIS = 12
N_MARGINAL_SAMPLES = 4  # random marginals per model
N_MC = 100000  # MC samples for ∫ p(x) dx
SEED = 0

# If True, use CPU; if False, use CUDA when available.
USE_CPU = False

# `random_init` hyperparameters (same spirit as vdp_nfdf_gaussian.py for GaussianBasis)
VARIANCE = 20.0
# GaussianBasis.random_init
GAUSSIAN_OFFSETS = (0.0, 30.0)  # length-2 tensor passed as offsets=
GAUSSIAN_MIN_STD = 1e-3
# BetaBasis.random_init (offsets length 2; created on CPU then moved — Beta uses CPU randn)
BETA_OFFSETS = (0.0, 0.0)
BETA_MIN_CONCENTRATION = 1.0
BETA_EPS = 1e-6
# QuadraticExpBasis.random_init
QUADRATIC_EXP_OFFSETS = (0.0, 0.0, 0.0)
QUADRATIC_EXP_EPS = 1e-6

# Standard deviation for Gaussian noise written into each *trainable* parameter after build
PARAM_RAND_STD = 1.0

# ---------------------------------------------------------------------------

SeparableBasisType = Type[GaussianBasis] | Type[BetaBasis] | Type[QuadraticExpBasis]

# (basis class, how marginal_dims is interpreted, per-dim domain (low, high) for MC)
_BASIS_REGISTRY: dict[str, tuple[SeparableBasisType, str, tuple[float, float]]] = {
    "gaussian": (GaussianBasis, "keep", (-10.0, 10.0)),
    "beta": (BetaBasis, "keep", (0.01, 0.99)),
    "quadratic_exp": (QuadraticExpBasis, "integrate_out", (-10.0, 10.0)),
}


def _randomize_trainable_params(module: torch.nn.Module, std: float, generator: torch.Generator) -> None:
    """In-place Gaussian reinitialization of every parameter with requires_grad=True."""
    with torch.no_grad():
        for p in module.parameters():
            if p.requires_grad:
                noise = torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=generator)
                p.copy_(noise * std)


def _random_basis(
    basis_name: str,
    d: int,
    n_basis: int,
    device: torch.device,
    dtype: torch.dtype,
    freeze_params: bool = False,
):
    """Single `random_init` basis, same hyperparameters as vdp_nfdf_gaussian.py (per BASIS kind)."""
    cls = _BASIS_REGISTRY[basis_name][0]

    if basis_name == "gaussian":
        off = torch.tensor(GAUSSIAN_OFFSETS, device=device, dtype=dtype)
        basis = cls.random_init(d, n_basis, offsets=off, variance=VARIANCE, min_std=GAUSSIAN_MIN_STD)
        if freeze_params:
            basis = cls.freeze_params(basis)
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
        ).to(device=device, dtype=dtype)
        if freeze_params:
            basis = cls.freeze_params(basis)
        return basis

    if basis_name == "quadratic_exp":
        off = torch.tensor(QUADRATIC_EXP_OFFSETS, device=device, dtype=dtype)
        basis = cls.random_init(d, n_basis, offsets=off, variance=VARIANCE, eps=QUADRATIC_EXP_EPS)
        if freeze_params:
            basis = cls.freeze_params(basis)
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


_MODEL_BUILDERS: dict[str, Callable[[str, int, int, torch.device, torch.dtype], LinearFF | Linear2FF | QuadraticFF]] = {
    "linear_ff": make_linear_ff,
    "linear2ff": make_linear_2ff,
    "quadratic_ff": make_quadratic_ff,
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


def random_marginal_arg(d: int, convention: str, rng: random.Random) -> tuple[int, ...]:
    """Return marginal_dims suitable for DensityModel.marginal (per basis convention)."""
    if d < 2:
        raise ValueError("Need dim >= 2 to form a proper lower-dimensional marginal.")

    if convention == "keep":
        k = rng.randint(1, d - 1)
        return tuple(sorted(rng.sample(range(d), k)))

    # integrate_out: non-empty proper subset to remove
    m = rng.randint(1, d - 1)
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
) -> None:
    _, convention, per_dim = _BASIS_REGISTRY[basis_name]
    rng = random.Random(seed)
    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    dtype = torch.float32

    for mname in model_names:
        builder = _MODEL_BUILDERS[mname]
        model = builder(basis_name, dim, n_basis, device, dtype)
        model = model.to(device=device)
        model.eval()

        _randomize_trainable_params(model, param_rand_std, gen)

        for t in range(n_marginal_samples):
            arg = random_marginal_arg(dim, convention, rng)
            m_model = model.marginal(arg)
            m_d = m_model.dim
            if m_d < 1:
                raise RuntimeError("Marginal has dim < 1 (basis / marginal_dims bug).")

            bounds = marginal_box_bounds(dim, arg, convention, per_dim)
            print(f"\n{basis_name} / {mname} / marginal {arg} -> dim {m_d} (trial {t + 1}/{n_marginal_samples})")
            check_pdf_valid(m_model, bounds, n_samples=n_mc)


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
    )
    print("\nAll marginal PDF checks passed.")


if __name__ == "__main__":
    main()
