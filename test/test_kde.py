"""
Build a complex 2D dataset, fit GaussianKDE, optimize bandwidth via LOO-MLE,
and visualize the estimated density before/after training.

Run:
  PYTHONPATH=src python test/test_kde.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from rational_factor.models.kde import AdaptiveGaussianKDE, GaussianKDE


SEED = 7
N_SAMPLES = 1800
TRAIN_EPOCHS = 500
TRAIN_LR = 0.5
VAL_FRACTION = 0.25
VAL_GAP_THRESHOLD = 0.02

FIG_PATH = Path("figures/test_kde.png")
GRID_POINTS = 220


def make_complex_2d_dataset(n_samples: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Mixture of ring + tilted Gaussian blobs."""
    if n_samples < 12:
        raise ValueError("n_samples should be >= 12")

    n_ring = int(0.45 * n_samples)
    n_blob_a = int(0.25 * n_samples)
    n_blob_b = n_samples - n_ring - n_blob_a

    theta = 2.0 * torch.pi * torch.rand(n_ring, device=device, dtype=dtype)
    ring_radius = 2.2 + 0.20 * torch.randn(n_ring, device=device, dtype=dtype)
    ring_x = ring_radius * torch.cos(theta)
    ring_y = 0.7 * ring_radius * torch.sin(theta)
    ring = torch.stack([ring_x, ring_y], dim=1)

    mean_a = torch.tensor([2.5, 1.8], device=device, dtype=dtype)
    cov_a = torch.tensor([[0.30, 0.22], [0.22, 0.45]], device=device, dtype=dtype)
    la = torch.linalg.cholesky(cov_a)
    blob_a = torch.randn(n_blob_a, 2, device=device, dtype=dtype) @ la.T + mean_a

    mean_b = torch.tensor([-2.8, -1.2], device=device, dtype=dtype)
    cov_b = torch.tensor([[0.45, -0.28], [-0.28, 0.35]], device=device, dtype=dtype)
    lb = torch.linalg.cholesky(cov_b)
    blob_b = torch.randn(n_blob_b, 2, device=device, dtype=dtype) @ lb.T + mean_b

    x = torch.cat([ring, blob_a, blob_b], dim=0)
    x = x[torch.randperm(x.shape[0], device=device)]
    return x


def evaluate_density_grid(kde: GaussianKDE | AdaptiveGaussianKDE, xlim: tuple[float, float], ylim: tuple[float, float], n_points: int):
    x_lin = torch.linspace(xlim[0], xlim[1], n_points, device=kde.data.device, dtype=kde.data.dtype)
    y_lin = torch.linspace(ylim[0], ylim[1], n_points, device=kde.data.device, dtype=kde.data.dtype)
    xx, yy = torch.meshgrid(x_lin, y_lin, indexing="xy")
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    with torch.no_grad():
        z = torch.exp(kde.log_density(grid)).reshape(n_points, n_points)
    return xx.detach().cpu(), yy.detach().cpu(), z.detach().cpu()


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    x = make_complex_2d_dataset(N_SAMPLES, device=device, dtype=dtype)
    n_val = max(1, int(VAL_FRACTION * x.shape[0]))
    x_val = x[:n_val]
    x_train = x[n_val:]
    kde = GaussianKDE(x_train)
    initial_bw = float(kde.bandwidth.item())

    kde_before = GaussianKDE(x_train, bandwidth=kde.bandwidth.detach().clone())
    kde_val = GaussianKDE(x_train, bandwidth=kde.bandwidth.detach().clone())
    x_cpu = x.detach().cpu()
    x_val_cpu = x_val.detach().cpu()
    x_min, _ = x_cpu.min(dim=0)
    x_max, _ = x_cpu.max(dim=0)
    pad = 0.8
    xlim = (float(x_min[0] - pad), float(x_max[0] + pad))
    ylim = (float(x_min[1] - pad), float(x_max[1] + pad))

    xx0, yy0, zz0 = evaluate_density_grid(kde_before, xlim, ylim, GRID_POINTS)
    best_bw, best_obj = kde.fit_bandwidth_loo_mle(epochs=TRAIN_EPOCHS, lr=TRAIN_LR, verbose=True)
    best_bw_val, best_val_obj = kde_val.fit_bandwidth_validation_mle(
        validation_data=x_val,
        epochs=TRAIN_EPOCHS,
        lr=TRAIN_LR,
        threshold=VAL_GAP_THRESHOLD,
        verbose=True,
    )
    xx1, yy1, zz1 = evaluate_density_grid(kde, xlim, ylim, GRID_POINTS)
    xxv, yyv, zzv = evaluate_density_grid(kde_val, xlim, ylim, GRID_POINTS)
    adaptive_kde = AdaptiveGaussianKDE.from_gaussian_kde(kde)
    adaptive_loo = float(adaptive_kde.loo_log_likelihood().item())
    xx2, yy2, zz2 = evaluate_density_grid(adaptive_kde, xlim, ylim, GRID_POINTS)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax = axes.ravel()
    ax[0].scatter(x_cpu[:, 0], x_cpu[:, 1], s=4, alpha=0.45)
    ax[0].scatter(x_val_cpu[:, 0], x_val_cpu[:, 1], s=6, alpha=0.7, c="tab:red", label="validation")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title("Complex 2D dataset")
    ax[0].set_xlabel("x0")
    ax[0].set_ylabel("x1")
    ax[0].set_xlim(*xlim)
    ax[0].set_ylim(*ylim)

    c1 = ax[1].contourf(xx0.numpy(), yy0.numpy(), zz0.numpy(), levels=24)
    ax[1].scatter(x_cpu[:, 0], x_cpu[:, 1], s=2, c="white", alpha=0.15)
    ax[1].set_title(f"KDE before training\nh={initial_bw:.4f}")
    ax[1].set_xlabel("x0")
    ax[1].set_ylabel("x1")
    ax[1].set_xlim(*xlim)
    ax[1].set_ylim(*ylim)
    fig.colorbar(c1, ax=ax[1], fraction=0.046, pad=0.04)

    c2 = ax[2].contourf(xx1.numpy(), yy1.numpy(), zz1.numpy(), levels=24)
    ax[2].scatter(x_cpu[:, 0], x_cpu[:, 1], s=2, c="white", alpha=0.15)
    ax[2].set_title(
        "KDE after LOO-MLE training\n"
        f"h={float(best_bw.item()):.4f}, LOO loglik={best_obj:.4f}"
    )
    ax[2].set_xlabel("x0")
    ax[2].set_ylabel("x1")
    ax[2].set_xlim(*xlim)
    ax[2].set_ylim(*ylim)
    fig.colorbar(c2, ax=ax[2], fraction=0.046, pad=0.04)

    c3 = ax[3].contourf(xx2.numpy(), yy2.numpy(), zz2.numpy(), levels=24)
    ax[3].scatter(x_cpu[:, 0], x_cpu[:, 1], s=2, c="white", alpha=0.15)
    ax[3].set_title(
        "Adaptive KDE (Abramson)\n"
        f"LOO loglik={adaptive_loo:.4f}"
    )
    ax[3].set_xlabel("x0")
    ax[3].set_ylabel("x1")
    ax[3].set_xlim(*xlim)
    ax[3].set_ylim(*ylim)
    fig.colorbar(c3, ax=ax[3], fraction=0.046, pad=0.04)

    c4 = ax[4].contourf(xxv.numpy(), yyv.numpy(), zzv.numpy(), levels=24)
    ax[4].scatter(x_cpu[:, 0], x_cpu[:, 1], s=2, c="white", alpha=0.15)
    ax[4].set_title(
        "KDE after validation-gap fitting\n"
        f"h={float(best_bw_val.item()):.4f}, val loglik={best_val_obj:.4f}"
    )
    ax[4].set_xlabel("x0")
    ax[4].set_ylabel("x1")
    ax[4].set_xlim(*xlim)
    ax[4].set_ylim(*ylim)
    fig.colorbar(c4, ax=ax[4], fraction=0.046, pad=0.04)

    ax[5].axis("off")

    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Device: {device}")
    print(f"Initial bandwidth: {initial_bw:.6f}")
    print(f"Trained bandwidth: {float(best_bw.item()):.6f}")
    print(f"Validation-fit bandwidth: {float(best_bw_val.item()):.6f}")
    print(f"Global KDE mean LOO log-likelihood: {best_obj:.6f}")
    print(f"Validation-fit best validation log-likelihood: {best_val_obj:.6f}")
    print(f"Adaptive KDE mean LOO log-likelihood: {adaptive_loo:.6f}")
    if adaptive_loo > best_obj:
        print("Adaptive KDE improves the LOO objective over global KDE.")
    else:
        print("Adaptive KDE does not improve the LOO objective over global KDE on this run.")
    print(f"Saved figure to: {FIG_PATH.resolve()}")


if __name__ == "__main__":
    main()
