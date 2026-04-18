"""
Generate a random 2D FF density and visualize:
1) full 2D belief
2) marginal over x0
3) marginal over x1

Run:
  PYTHONPATH=src python test/test_marginal_2D.py
"""

from __future__ import annotations

import random
from typing import Callable, Type

import matplotlib.pyplot as plt
import torch

from rational_factor.models.basis_functions import (
    BetaBasis,
    GaussianBasis,
    QuadraticExpBasis,
    UnnormalizedBetaBasis,
)
from rational_factor.models.factor_forms import Linear2FF, LinearFF, QuadraticFF
from rational_factor.tools.analysis import check_pdf_valid
from rational_factor.tools.visualization import plot_belief


# --- edit these -------------------------------------------------------------

# "quadratic_exp" | "unnormalized_beta" | "gaussian" | "beta"
BASIS = "quadratic_exp"

# None => random choice from ("linear_ff", "linear2ff", "quadratic_ff")
#MODEL: str | None = None
MODEL: str = "linear_ff"

SEED = 0
N_BASIS = 12
USE_CPU = False
PARAM_RAND_STD = 1.0

# Plot controls
N_POINTS_2D = 120
N_POINTS_1D = 400
FIGSIZE = (14, 4)

# Basis init controls
VARIANCE = 20.0
# QuadraticExpBasis: keep mass concentrated for MC/plots on (-10, 10); see test_marginal.py.
QUADRATIC_EXP_INIT_VARIANCE = 1.0
QUADRATIC_EXP_OFFSETS = (0.0, 0.0)  # len must match n_params_per_basis == 2 (raw_a, b)
QUADRATIC_EXP_EPS = 1e-6
UNNORMALIZED_BETA_OFFSETS = (0.0, 0.0)
UNNORMALIZED_BETA_MIN_CONCENTRATION = 1.0
UNNORMALIZED_BETA_EPS = 1e-6
GAUSSIAN_OFFSETS = (0.0, 30.0)
GAUSSIAN_MIN_STD = 1e-3
BETA_OFFSETS = (0.0, 0.0)
BETA_MIN_CONCENTRATION = 1.0
BETA_EPS = 1e-6

# ---------------------------------------------------------------------------

SeparableBasisType = (
    Type[QuadraticExpBasis] | Type[UnnormalizedBetaBasis] | Type[GaussianBasis] | Type[BetaBasis]
)
ModelType = LinearFF | Linear2FF | QuadraticFF

_BASIS_REGISTRY: dict[str, tuple[SeparableBasisType, tuple[float, float]]] = {
    "quadratic_exp": (QuadraticExpBasis, (-10.0, 10.0)),
    # Use near-full support to match analytic Beta integrals in basis marginal/Omega ops.
    "unnormalized_beta": (UnnormalizedBetaBasis, (1e-6, 1.0 - 1e-6)),
    "gaussian": (GaussianBasis, (-12.0, 12.0)),
    "beta": (BetaBasis, (1e-6, 1.0 - 1e-6)),
}


def _module_device(module: torch.nn.Module) -> torch.device:
    p = next(module.parameters(), None)
    if p is not None:
        return p.device
    b = next(module.buffers(), None)
    if b is not None:
        return b.device
    return torch.device("cpu")


def _randomize_trainable_params(module: torch.nn.Module, std: float, seed: int) -> None:
    gens: dict[tuple[str, int | None], torch.Generator] = {}

    def _generator_for(device: torch.device) -> torch.Generator:
        key = (device.type, device.index)
        if key not in gens:
            gen = torch.Generator(device=device.type)
            gen.manual_seed(seed + len(gens))
            gens[key] = gen
        return gens[key]

    with torch.no_grad():
        for p in module.parameters():
            if p.requires_grad:
                gen = _generator_for(p.device)
                p.copy_(torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=gen) * std)


def _random_basis(
    basis_name: str,
    d: int,
    n_basis: int,
    device: torch.device,
    dtype: torch.dtype,
    freeze_params: bool = False,
):
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
    elif basis_name == "unnormalized_beta":
        off = torch.tensor(UNNORMALIZED_BETA_OFFSETS, dtype=dtype)
        basis = cls.random_init(
            d,
            n_basis,
            offsets=off,
            variance=VARIANCE,
            min_concentration=UNNORMALIZED_BETA_MIN_CONCENTRATION,
            eps=UNNORMALIZED_BETA_EPS,
        ).to(device=device, dtype=dtype)
    elif basis_name == "gaussian":
        off = torch.tensor(GAUSSIAN_OFFSETS, dtype=dtype)
        basis = cls.random_init(
            d,
            n_basis,
            offsets=off,
            variance=VARIANCE,
            min_std=GAUSSIAN_MIN_STD,
            device=device,
        ).to(device=device, dtype=dtype)
    elif basis_name == "beta":
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
    else:
        raise ValueError(f"Unknown basis name: {basis_name}")

    return basis.freeze_params() if freeze_params else basis


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
    dvec = torch.randn(n_basis, device=device, dtype=dtype).softmax(0)
    a = torch.randn(n_basis, device=device, dtype=dtype).softmax(0)
    return Linear2FF(dvec, xi, a, phi, psi0)


def make_quadratic_ff(
    basis_name: str, d: int, n_basis: int, device: torch.device, dtype: torch.dtype
) -> QuadraticFF:
    phi = _random_basis(basis_name, d, n_basis, device, dtype, freeze_params=True)
    psi0 = _random_basis(basis_name, d, n_basis, device, dtype)
    Au = torch.randn(n_basis, n_basis, device=device, dtype=dtype).softmax(0)
    A = Au @ Au.T
    return QuadraticFF(A, phi, psi0)


_MODEL_BUILDERS: dict[str, Callable[[str, int, int, torch.device, torch.dtype], ModelType]] = {
    "linear_ff": make_linear_ff,
    "linear2ff": make_linear_2ff,
    "quadratic_ff": make_quadratic_ff,
}


def _plot_1d_marginal(ax: plt.Axes, model: ModelType, x_low: float, x_high: float, n_points: int, title: str) -> None:
    device = _module_device(model)
    x = torch.linspace(x_low, x_high, n_points, device=device, dtype=torch.float32)
    x_in = x[:, None]
    with torch.no_grad():
        model.eval()
        y = model(x_in).squeeze(-1)
    ax.plot(x.detach().cpu().numpy(), y.detach().cpu().numpy(), linewidth=2.0)
    ax.set_xlim(x_low, x_high)
    ax.set_xlabel(title.split()[0])
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.grid(alpha=0.25)


def main() -> None:
    if BASIS not in _BASIS_REGISTRY:
        raise ValueError(f"BASIS must be one of {sorted(_BASIS_REGISTRY.keys())}, got {BASIS!r}")
    if MODEL is not None and MODEL not in _MODEL_BUILDERS:
        raise ValueError(f"MODEL must be one of {sorted(_MODEL_BUILDERS.keys())} or None, got {MODEL!r}")

    #random.seed(SEED)
    #torch.manual_seed(SEED)
    device = torch.device("cpu") if USE_CPU else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    model_name = MODEL if MODEL is not None else random.choice(tuple(_MODEL_BUILDERS.keys()))
    builder = _MODEL_BUILDERS[model_name]
    belief = builder(BASIS, 2, N_BASIS, device, dtype).to(device=device)
    _randomize_trainable_params(belief, PARAM_RAND_STD, SEED)
    belief.eval()

    # In this codebase, marginal(arg) keeps the dims in arg.
    marg_x0 = belief.marginal((0,))
    marg_x1 = belief.marginal((1,))

    x_low, x_high = _BASIS_REGISTRY[BASIS][1]
    bounds_2d = ((x_low, x_low), (x_high, x_high))
    bounds_1d = ((x_low,), (x_high,))

    print(f"Model: {model_name} | Basis: {BASIS} | Device: {device}")
    print("Checking full 2D belief normalization...")
    check_pdf_valid(belief, bounds_2d, n_samples=100000, device=device)
    print("Checking x0 marginal normalization...")
    check_pdf_valid(marg_x0, bounds_1d, n_samples=100000, device=device)
    print("Checking x1 marginal normalization...")
    check_pdf_valid(marg_x1, bounds_1d, n_samples=100000, device=device)

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)

    plot_belief(
        axes[0],
        belief,
        x_range=(x_low, x_high),
        y_range=(x_low, x_high),
        n_points=N_POINTS_2D,
    )
    axes[0].set_title(f"2D belief ({model_name}, {BASIS})")
    axes[0].set_xlabel("x0")
    axes[0].set_ylabel("x1")

    _plot_1d_marginal(axes[1], marg_x0, x_low, x_high, N_POINTS_1D, "x0 marginal")
    _plot_1d_marginal(axes[2], marg_x1, x_low, x_high, N_POINTS_1D, "x1 marginal")

    fig.tight_layout()
    plt.savefig(f"figures/test_marginal_2D.png")


if __name__ == "__main__":
    main()
