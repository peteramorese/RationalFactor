"""
Train CBBNormalizingFlow on two separate 2D datasets (alpha / beta branches),
visualize learned densities, and verify normalization + bilinear inner product via MC.

Checks:
  - ∫ p_alpha = 1 and ∫ p_beta = 1 on the unit box
  - prod_i Z_i(x) has near-zero variance over the unit box (constant bilinear inner product)
  - mean prod_i Z_i(x) matches flow.inner_product()

Run:
  PYTHONPATH=src python test/test_cbb_normalizing_flow.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from normalizing_flow.bernstein_normalizing_flow import (
    CBBNormalizingFlow,
    CoupledMABernsteinConditioner,
)
from rational_factor.tools.analysis import mc_integral_box

SEED = 0
N_SAMPLES = 1500
BATCH_SIZE = 128
EPOCHS = 400
LR = 2e-3
DEGREE = 10
NUM_BLOCKS = 4
HIDDEN_FEATURES = 64
MC_SAMPLES = 200_000
INTEGRAL_TOL = 0.12
INNER_PRODUCT_TOL = 0.15
BILINEAR_VAR_TOL = 1e-2
FIG_PATH = Path("figures/test_cbb_normalizing_flow.png")
GRID_POINTS = 120
DOMAIN_BOUNDS = ((0.0, 0.0), (1.0, 1.0))


def make_blob_dataset(
    n_samples: int,
    mean: torch.Tensor,
    cov: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    chol = torch.linalg.cholesky(cov)
    noise = torch.randn(n_samples, mean.numel(), device=device, dtype=dtype)
    return noise @ chol.T + mean


def fit_unit_box_map(
    *datasets: torch.Tensor, pad: float = 0.08
) -> tuple[torch.Tensor, torch.Tensor]:
    stacked = torch.cat(datasets, dim=0)
    low = stacked.min(dim=0).values
    high = stacked.max(dim=0).values
    span = (high - low).clamp_min(1e-4)
    low = low - pad * span
    high = high + pad * span
    return low, high


def to_unit_box(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return ((x - low) / (high - low)).clamp(1e-5, 1.0 - 1e-5)


def evaluate_density_grid(
    density_fn,
    n_points: int,
    device: torch.device,
    dtype: torch.dtype,
):
    x_lin = torch.linspace(0.0, 1.0, n_points, device=device, dtype=dtype)
    y_lin = torch.linspace(0.0, 1.0, n_points, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x_lin, y_lin, indexing="xy")
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    with torch.no_grad():
        z = density_fn(grid).reshape(n_points, n_points)
    return xx.detach().cpu(), yy.detach().cpu(), z.detach().cpu()


LOG_INTERVAL = 20


def train_flow(
    flow: CBBNormalizingFlow,
    alpha_data: torch.Tensor,
    beta_data: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
) -> list[float]:
    alpha_loader = DataLoader(
        TensorDataset(alpha_data), batch_size=batch_size, shuffle=True
    )
    beta_loader = DataLoader(
        TensorDataset(beta_data), batch_size=batch_size, shuffle=True
    )
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    losses: list[float] = []

    for epoch in range(epochs):
        flow.train()
        epoch_loss = 0.0
        n_batches = 0
        for (alpha_batch,), (beta_batch,) in zip(alpha_loader, beta_loader):
            optimizer.zero_grad()
            log_alpha, _ = flow(alpha_batch)
            _, log_beta = flow(beta_batch)
            loss = -(log_alpha.mean() + log_beta.mean())
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        mean_loss = epoch_loss / max(n_batches, 1)
        losses.append(mean_loss)
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % LOG_INTERVAL == 0:
            print(f"epoch {epoch + 1:4d}/{epochs}: loss = {mean_loss:.6f}")

    return losses


def bilinear_inner_product_integrand(
    flow: CBBNormalizingFlow, x: torch.Tensor
) -> torch.Tensor:
    """Monte Carlo integrand for prod_i Z_i(x) from the coupled conditioner."""
    conditioner = flow.transform.autoregressive_net
    x = x[:, flow.permutation]
    weights = CoupledMABernsteinConditioner.forward(
        conditioner, x, None, return_simplex_weights=True
    )
    alpha = weights[..., 0, :]
    operation_matrix = conditioner.bernstein_operation_matrix.to(dtype=alpha.dtype)
    a = torch.einsum("ij,...j->...i", operation_matrix, alpha)
    inner_prod_values = conditioner.inner_product_values(a)
    return inner_prod_values.prod(dim=-1, keepdim=True)


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    alpha_raw = make_blob_dataset(
        N_SAMPLES,
        mean=torch.tensor([1.5, 1.0], device=device, dtype=dtype),
        cov=torch.tensor([[0.20, 0.05], [0.05, 0.15]], device=device, dtype=dtype),
        device=device,
        dtype=dtype,
    )
    beta_raw = make_blob_dataset(
        N_SAMPLES,
        mean=torch.tensor([-1.2, 1.8], device=device, dtype=dtype),
        cov=torch.tensor([[0.35, -0.12], [-0.12, 0.25]], device=device, dtype=dtype),
        device=device,
        dtype=dtype,
    )

    low, high = fit_unit_box_map(alpha_raw, beta_raw)
    alpha_data = to_unit_box(alpha_raw, low, high)
    beta_data = to_unit_box(beta_raw, low, high)

    flow = CBBNormalizingFlow(
        dim=2,
        degree=DEGREE,
        num_blocks=NUM_BLOCKS,
        hidden_features=HIDDEN_FEATURES,
    ).to(device=device, dtype=dtype)

    losses = train_flow(flow, alpha_data, beta_data, EPOCHS, BATCH_SIZE, LR)
    flow.eval()

    def alpha_density(x: torch.Tensor) -> torch.Tensor:
        log_alpha, _ = flow(x)
        return torch.exp(log_alpha)

    def beta_density(x: torch.Tensor) -> torch.Tensor:
        _, log_beta = flow(x)
        return torch.exp(log_beta)

    def product_density(x: torch.Tensor) -> torch.Tensor:
        log_alpha, log_beta = flow(x)
        return torch.exp(log_alpha + log_beta)

    integral_alpha = mc_integral_box(
        alpha_density, DOMAIN_BOUNDS, n_samples=MC_SAMPLES, device=device
    )
    integral_beta = mc_integral_box(
        beta_density, DOMAIN_BOUNDS, n_samples=MC_SAMPLES, device=device
    )
    integral_product_density = mc_integral_box(
        product_density, DOMAIN_BOUNDS, n_samples=MC_SAMPLES, device=device
    )

    lows = torch.tensor(DOMAIN_BOUNDS[0], device=device, dtype=dtype)
    highs = torch.tensor(DOMAIN_BOUNDS[1], device=device, dtype=dtype)
    x_samples = torch.rand(MC_SAMPLES, flow.dim, device=device, dtype=dtype)
    x_samples = x_samples * (highs - lows) + lows
    with torch.no_grad():
        bilinear_samples = bilinear_inner_product_integrand(flow, x_samples).squeeze(-1)
        bilinear_mean = bilinear_samples.mean().item()
        bilinear_variance = bilinear_samples.var(unbiased=False).item()
        predicted_inner_product = flow.inner_product().reshape(()).item()

    print(f"Final training loss: {losses[-1]:.4f}")
    print(f"MC integral alpha:              {integral_alpha.item():.4f}")
    print(f"MC integral beta:               {integral_beta.item():.4f}")
    print(f"MC integral p_alpha * p_beta:   {integral_product_density.item():.4f}")
    print(f"bilinear integrand mean:        {bilinear_mean:.6f}")
    print(f"bilinear integrand variance:    {bilinear_variance:.6e}")
    print(f"inner_product():                {predicted_inner_product:.6f}")

    assert abs(integral_alpha.item() - 1.0) < INTEGRAL_TOL, (
        f"alpha integral {integral_alpha.item():.4f} not near 1"
    )
    assert abs(integral_beta.item() - 1.0) < INTEGRAL_TOL, (
        f"beta integral {integral_beta.item():.4f} not near 1"
    )
    assert bilinear_variance < BILINEAR_VAR_TOL, (
        f"bilinear integrand variance {bilinear_variance:.6e} is not near zero"
    )
    assert abs(bilinear_mean - predicted_inner_product) < INNER_PRODUCT_TOL, (
        "mean bilinear integrand does not match inner_product()"
    )

    xx_a, yy_a, zz_a = evaluate_density_grid(alpha_density, GRID_POINTS, device, dtype)
    xx_b, yy_b, zz_b = evaluate_density_grid(beta_density, GRID_POINTS, device, dtype)
    xx_p, yy_p, zz_p = evaluate_density_grid(product_density, GRID_POINTS, device, dtype)

    alpha_cpu = alpha_data.detach().cpu()
    beta_cpu = beta_data.detach().cpu()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    axes[0].scatter(alpha_cpu[:, 0], alpha_cpu[:, 1], s=6, alpha=0.35, label="alpha data")
    axes[0].scatter(beta_cpu[:, 0], beta_cpu[:, 1], s=6, alpha=0.35, label="beta data")
    axes[0].set_title("Training data (unit box)")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(fontsize=8)

    for ax, xx, yy, zz, title in zip(
        axes[1:],
        (xx_a, xx_b, xx_p),
        (yy_a, yy_b, yy_p),
        (zz_a, zz_b, zz_p),
        ("Alpha density", "Beta density", "Product density"),
    ):
        cf = ax.contourf(xx.numpy(), yy.numpy(), zz.numpy(), levels=24)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("x0")
        ax.set_ylabel("x1")

    fig.suptitle(
        "CBBNormalizingFlow: "
        f"∫pα={integral_alpha.item():.3f}, ∫pβ={integral_beta.item():.3f}, "
        f"Var(∏Z)={bilinear_variance:.2e}, inner_product={predicted_inner_product:.3f}"
    )
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=160, bbox_inches="tight")
    print(f"Saved figure to {FIG_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
