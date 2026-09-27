from pathlib import Path
from math import ceil, sqrt

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.factor_forms import SumProdRFF, LinearFF
from rational_factor.models.kde import GaussianKDE
from rational_factor.models.parameters import (
    PositiveParameters,
    TrainableParameters,
    param_group_iter,
    DenseMatrixFactorization,
)
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood, check_pdf_valid
from rational_factor.tools.visualization import plot_belief
import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.tools.propagate as propagate

from rational_factor.models.gram_preserving_basis import DeepGramPreservingBasis
from rational_factor.models.index_embedding_model import IndexEmbeddingTransform
from normalizing_flow.transforms import Transforms
from rational_factor.models.space_splitter import LatentReflectionSpaceSplitter
from rational_factor.models.mlp import MLP


def _plot_conditional_slices_model_vs_data(
    tran_model,
    x_k_data: torch.Tensor,
    x_kp1_data: torch.Tensor,
    out_path: Path,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    n_random_points: int = 10,
    n_grid: int = 120,
    min_points_per_slice: int = 40,
    bin_width: float = 0.4,
    random_seed: int = 0,
    title: str = "",
) -> None:
    """Compare trained p(x'|x) to binned empirical samples and a conditional KDE.

    Conditioners share one randomly chosen x1 and span different x2 values so
    rows can be compared for conditional dependence on x2.
    """
    if x_k_data.shape[1] != 2 or x_kp1_data.shape[1] != 2:
        raise ValueError("_plot_conditional_slices_model_vs_data expects 2D state data")

    param = next(iter(tran_model.parameters()), None)
    buffer = next(iter(tran_model.buffers()), None)
    dev = (
        param.device
        if param is not None
        else (buffer.device if buffer is not None else torch.device("cpu"))
    )
    dt = (
        param.dtype
        if param is not None
        else (buffer.dtype if buffer is not None else torch.float32)
    )

    xk_cpu = x_k_data.detach().cpu()
    xkp1_cpu = x_kp1_data.detach().cpu()
    lo = torch.tensor([x_range[0], y_range[0]], dtype=xk_cpu.dtype)
    hi = torch.tensor([x_range[1], y_range[1]], dtype=xk_cpu.dtype)

    x_lin = torch.linspace(float(lo[0]), float(hi[0]), n_grid)
    y_lin = torch.linspace(float(lo[1]), float(hi[1]), n_grid)
    X, Y = torch.meshgrid(x_lin, y_lin, indexing="xy")
    xp_grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).to(device=dev, dtype=dt)

    n_data = xk_cpu.shape[0]
    if n_data == 0:
        raise ValueError("x_k_data is empty")
    n_random_points = max(1, min(n_random_points, n_data))
    rng = torch.Generator(device="cpu")
    rng.manual_seed(random_seed)
    half_width = torch.full((2,), 0.5 * float(bin_width), dtype=xk_cpu.dtype)

    ref_idx = int(torch.randint(0, n_data, (1,), generator=rng).item())
    x1_fixed = 0.0
    near_x1 = (xk_cpu[:, 0] - x1_fixed).abs() <= half_width[0]
    near_pts = xk_cpu[near_x1]
    if near_pts.shape[0] < n_random_points:
        near_pts = xk_cpu
        x1_fixed = float(near_pts[ref_idx, 0])
    order = torch.argsort(near_pts[:, 1])
    near_sorted = near_pts[order]
    if near_sorted.shape[0] == n_random_points:
        pick = torch.arange(n_random_points)
    else:
        pick = torch.linspace(
            0, near_sorted.shape[0] - 1, n_random_points
        ).round().long()
        pick = torch.unique(pick)
        if pick.numel() < n_random_points:
            need = n_random_points - pick.numel()
            all_idx = torch.arange(near_sorted.shape[0])
            mask = torch.ones(near_sorted.shape[0], dtype=torch.bool)
            mask[pick] = False
            extra = all_idx[mask][:need]
            pick = torch.sort(torch.cat([pick, extra]))[0]
    centers = near_sorted[pick]
    centers = centers.clone()
    centers[:, 0] = x1_fixed

    fig, axes = plt.subplots(
        n_random_points, 3, figsize=(15, 3.2 * n_random_points), squeeze=False
    )
    cmap = "viridis"
    eps = torch.finfo(dt).eps

    xk_dev = xk_cpu.to(device=dev, dtype=dt)
    xkp1_dev = xkp1_cpu.to(device=dev, dtype=dt)
    bw_x = GaussianKDE.scott_bandwidth(xk_dev).clamp_min(eps)
    bw_xp = GaussianKDE.scott_bandwidth(xkp1_dev).clamp_min(eps)

    with torch.no_grad():
        tran_model.eval()
        for row, center in enumerate(centers):
            lo_box = center - half_width
            hi_box = center + half_width
            mask = (
                (xk_cpu[:, 0] >= lo_box[0])
                & (xk_cpu[:, 0] < hi_box[0])
                & (xk_cpu[:, 1] >= lo_box[1])
                & (xk_cpu[:, 1] < hi_box[1])
            )
            xp_slice = xkp1_cpu[mask]

            ax_emp = axes[row, 0]
            if xp_slice.shape[0] >= min_points_per_slice:
                ax_emp.scatter(
                    xp_slice[:, 0].numpy(),
                    xp_slice[:, 1].numpy(),
                    s=5,
                    alpha=1.0,
                    c="orange",
                    edgecolors="none",
                    rasterized=True,
                )
            else:
                ax_emp.text(
                    0.5,
                    0.5,
                    f"Too few samples\nn={xp_slice.shape[0]}",
                    ha="center",
                    va="center",
                    transform=ax_emp.transAxes,
                )
            ax_emp.set_title(
                "empirical samples x' | x in bin\n"
                f"n={xp_slice.shape[0]}, "
                f"x1={float(center[0]):.3f}, x2={float(center[1]):.3f}"
            )
            ax_emp.set_xlim(float(lo[0]), float(hi[0]))
            ax_emp.set_ylim(float(lo[1]), float(hi[1]))
            ax_emp.set_aspect("equal")
            ax_emp.set_ylabel("x'_2")

            center_dev = center.to(device=dev, dtype=dt)
            cond = center_dev.unsqueeze(0).expand(xp_grid.shape[0], -1)
            model_pdf = (
                tran_model.log_density(xp_grid, conditioner=cond)
                .exp()
                .reshape(n_grid, n_grid)
                .detach()
                .cpu()
                .numpy()
            )
            ax_model = axes[row, 1]
            cf = ax_model.contourf(X.numpy(), Y.numpy(), model_pdf, levels=40, cmap=cmap)
            fig.colorbar(cf, ax=ax_model, fraction=0.046, pad=0.04)
            ax_model.set_title(
                "model p(x'|x1 fixed, x2)\n"
                f"x2={float(center[1]):.3f}, "
                f"bin_w=({float(2 * half_width[0]):.3f}, {float(2 * half_width[1]):.3f})"
            )
            ax_model.set_xlim(float(lo[0]), float(hi[0]))
            ax_model.set_ylim(float(lo[1]), float(hi[1]))
            ax_model.set_aspect("equal")
            if xp_slice.shape[0] > 0:
                ax_model.scatter(
                    xp_slice[:, 0].numpy(),
                    xp_slice[:, 1].numpy(),
                    s=5,
                    alpha=1.0,
                    c="orange",
                    edgecolors="none",
                    rasterized=True,
                )

            diff_x = xk_dev - center_dev.unsqueeze(0)
            w_x = torch.exp(-0.5 * diff_x.square().sum(dim=1) / (bw_x * bw_x))
            w_sum = w_x.sum().clamp_min(eps)
            kde_vals = []
            block = 1024
            for start in range(0, xp_grid.shape[0], block):
                end = min(start + block, xp_grid.shape[0])
                xp_blk = xp_grid[start:end]
                diff_xp = xp_blk[:, None, :] - xkp1_dev[None, :, :]
                k_xp = torch.exp(-0.5 * diff_xp.square().sum(dim=2) / (bw_xp * bw_xp))
                kde_vals.append((k_xp * w_x.unsqueeze(0)).sum(dim=1) / w_sum)
            kde_pdf = (
                torch.cat(kde_vals, dim=0).reshape(n_grid, n_grid).detach().cpu().numpy()
            )

            ax_kde = axes[row, 2]
            cf_kde = ax_kde.contourf(X.numpy(), Y.numpy(), kde_pdf, levels=40, cmap=cmap)
            fig.colorbar(cf_kde, ax=ax_kde, fraction=0.046, pad=0.04)
            ax_kde.set_title(
                f"conditional KDE p(x'|x)\n"
                f"x1={float(center[0]):.3f}, x2={float(center[1]):.3f}"
            )
            ax_kde.set_xlim(float(lo[0]), float(hi[0]))
            ax_kde.set_ylim(float(lo[1]), float(hi[1]))
            ax_kde.set_aspect("equal")
            if xp_slice.shape[0] > 0:
                ax_kde.scatter(
                    xp_slice[:, 0].numpy(),
                    xp_slice[:, 1].numpy(),
                    s=5,
                    alpha=1.0,
                    c="orange",
                    edgecolors="none",
                    rasterized=True,
                )

    for col in range(3):
        axes[-1, col].set_xlabel("x'_1")
    if title:
        fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_2d_values_grid(
    vals: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
    out_path: Path,
    *,
    title: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> None:
    """Plot (n_pts, n_basis) values on a 2D mesh as a subplot grid."""
    while vals.ndim > 2 and vals.shape[0] == 1:
        vals = vals.squeeze(0)
    if vals.ndim != 2:
        raise ValueError(
            f"Expected values shape (n_grid*n_grid, n_basis), got {tuple(vals.shape)}"
        )

    n_grid = int(X.shape[0])
    if X.shape != (n_grid, n_grid) or Y.shape != (n_grid, n_grid):
        raise ValueError("X and Y must be square meshes of equal shape")
    if vals.shape[0] != n_grid * n_grid:
        raise ValueError(
            f"values leading dim {vals.shape[0]} != n_grid*n_grid={n_grid * n_grid}"
        )

    vals_np = vals.detach().cpu().numpy().reshape(n_grid, n_grid, -1)
    n_basis = vals_np.shape[-1]
    x_np = X.detach().cpu().numpy()
    y_np = Y.detach().cpu().numpy()

    n_cols = max(1, ceil(sqrt(n_basis)))
    n_rows = max(1, ceil(n_basis / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.6 * n_cols, 1.45 * n_rows),
        squeeze=False,
    )
    if title:
        fig.suptitle(title, y=1.01)

    cmap = "viridis"
    for i in range(n_basis):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        ax.contourf(x_np, y_np, vals_np[:, :, i], levels=40, cmap=cmap)
        ax.set_title(str(i), fontsize=7, pad=1)
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        ax.set_aspect("equal")
        ax.tick_params(labelsize=5)
        if r < n_rows - 1:
            ax.set_xticklabels([])
        if c > 0:
            ax.set_yticklabels([])

    for j in range(n_basis, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _state_space_mesh(
    pair_basis,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    *,
    n_grid: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(X, Y, y_flat)`` on a 2D box in ``R^2`` (not the unit square)."""
    dtype, device = pair_basis.dtype_device()
    x_lin = torch.linspace(x_range[0], x_range[1], n_grid, device=device, dtype=dtype)
    y_lin = torch.linspace(y_range[0], y_range[1], n_grid, device=device, dtype=dtype)
    X, Y = torch.meshgrid(x_lin, y_lin, indexing="xy")
    y = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    return X, Y, y


def _plot_2d_pair_member_grid(
    pair_basis,
    out_path: Path,
    *,
    index: int,
    title: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    n_grid: int = 64,
) -> None:
    """Plot every 2D phi (index=0) or psi (index=1) basis on a square subplot grid."""
    if index not in (0, 1):
        raise ValueError(f"index must be 0 (phi) or 1 (psi), got {index}")

    dim = pair_basis.dim()
    if dim != 2:
        raise ValueError(
            f"_plot_2d_pair_member_grid expects a 2D pair basis, got dim={dim}"
        )

    X, Y, y = _state_space_mesh(
        pair_basis, x_range=x_range, y_range=y_range, n_grid=n_grid
    )
    with torch.no_grad():
        if isinstance(pair_basis, torch.nn.Module):
            torch.nn.Module.eval(pair_basis)
        vals = pair_basis.eval(y, index).detach().cpu()

    _plot_2d_values_grid(
        vals, X, Y, out_path, title=title, x_range=x_range, y_range=y_range
    )


def _plot_sigmoid_deformer(
    gp_basis: DeepGramPreservingBasis,
    out_path: Path,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    n_grid: int = 80,
    title: str = "sigmoid(deformer(x)) — unconstrained σ per layer",
) -> None:
    """Heatmaps of each deformer channel after sigmoid (before L/U constraints)."""
    X, Y, y = _state_space_mesh(
        gp_basis, x_range=x_range, y_range=y_range, n_grid=n_grid
    )
    with torch.no_grad():
        torch.nn.Module.eval(gp_basis)
        sigma = torch.sigmoid(gp_basis._deformer(y)).detach().cpu()

    n_layers = sigma.shape[-1]
    x_np = X.detach().cpu().numpy()
    y_np = Y.detach().cpu().numpy()
    fig, axes = plt.subplots(
        1, n_layers, figsize=(4.0 * n_layers, 3.6), squeeze=False
    )
    fig.suptitle(title, y=1.02)
    for l in range(n_layers):
        ax = axes[0, l]
        z = sigma[:, l].reshape(n_grid, n_grid).numpy()
        cf = ax.contourf(x_np, y_np, z, levels=40, cmap="coolwarm", vmin=0.0, vmax=1.0)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"σ_{l}(x)  [{z.min():.3f}, {z.max():.3f}]")
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        ax.set_aspect("equal")
        ax.set_xlabel("x1")
        if l == 0:
            ax.set_ylabel("x2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _lu_bounds(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    alpha_p: torch.Tensor,
    beta_p: torch.Tensor,
    J: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same L, U construction as ``DeepGramPreservingBasis._constrained_s``."""
    J = J.unsqueeze(-1)
    L = (-beta / (J * beta_p).clamp_min(eps)).amax(dim=-1)
    ratios_u = torch.where(
        alpha > eps,
        alpha_p.clamp_min(0.0) / alpha.clamp_min(eps),
        torch.full_like(alpha, float("inf")),
    )
    U = ratios_u.amin(dim=-1)
    U = torch.where(torch.isfinite(U), torch.maximum(U, L), L)
    return L, U


def _layerwise_LU(
    gp_basis: DeepGramPreservingBasis, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(L, U)`` of shape ``(n_data, n_layers)`` at each reflection layer.

    Uses the same branch as the applied ``s`` (set-0 vs set-1 / partner).
    """
    n_layers = gp_basis._n_layers
    eps = gp_basis._eps
    n = y.shape[0]
    device, dtype = y.device, y.dtype
    L_all = torch.empty(n, n_layers, device=device, dtype=dtype)
    U_all = torch.empty(n, n_layers, device=device, dtype=dtype)

    alpha = gp_basis._eval_base(gp_basis._base_phi, y)
    beta = gp_basis._eval_base(gp_basis._base_psi, y)
    y_partner, set_index, ladj, _ = gp_basis._space_splitter.partner(y)

    for l in range(n_layers):
        yp = y_partner[:, l, :]
        alpha_p, beta_p = gp_basis._apply_layers(yp, l)
        J = torch.exp(ladj[:, l])
        si = set_index[:, l]

        L_y, U_y = _lu_bounds(alpha, beta, alpha_p, beta_p, J, eps)
        L_p, U_p = _lu_bounds(
            alpha_p, beta_p, alpha, beta, torch.exp(-ladj[:, l]), eps
        )
        L_all[:, l] = torch.where(si == 0, L_y, L_p)
        U_all[:, l] = torch.where(si == 0, U_y, U_p)

        # Advance (α, β) like ``_apply_layers`` so later layers see the same state.
        sig = torch.sigmoid(gp_basis._deformer(y))[:, l].clamp(eps, 1.0 - eps)
        sig_p = torch.sigmoid(gp_basis._deformer(yp))[:, l].clamp(eps, 1.0 - eps)
        s_y = (U_y - L_y) * sig + L_y
        s_p = (U_p - L_p) * sig_p + L_p
        s = torch.where(si == 0, s_y, s_p).unsqueeze(-1)
        J2 = J.unsqueeze(-1)
        set0 = (si == 0).unsqueeze(-1)
        alpha = torch.where(set0, alpha, alpha - s * alpha_p).clamp_min(0.0)
        beta = torch.where(set0, beta + J2 * s * beta_p, beta).clamp_min(0.0)

    return L_all, U_all


def _plot_deformer_interval_width(
    gp_basis: DeepGramPreservingBasis,
    out_path: Path,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    n_grid: int = 80,
    title: str = "deformer interval (L − U) per layer",
) -> None:
    """Heatmaps of ``L(x) − U(x)`` from the positivity bounds in ``_constrained_s``."""
    X, Y, y = _state_space_mesh(
        gp_basis, x_range=x_range, y_range=y_range, n_grid=n_grid
    )
    with torch.no_grad():
        torch.nn.Module.eval(gp_basis)
        L, U = _layerwise_LU(gp_basis, y)
        width = (L - U).detach().cpu()

    n_layers = width.shape[-1]
    x_np = X.detach().cpu().numpy()
    y_np = Y.detach().cpu().numpy()
    fig, axes = plt.subplots(
        1, n_layers, figsize=(4.0 * n_layers, 3.6), squeeze=False
    )
    fig.suptitle(title, y=1.02)
    for l in range(n_layers):
        ax = axes[0, l]
        z = width[:, l].reshape(n_grid, n_grid).numpy()
        # L≤U after the max(U,L) clamp, so L−U ≤ 0; center colormap at 0.
        vmin = float(z.min())
        vmax = float(z.max())
        vabs = max(abs(vmin), abs(vmax), 1e-12)
        cf = ax.contourf(
            x_np, y_np, z, levels=40, cmap="coolwarm", vmin=-vabs, vmax=vabs
        )
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"(L−U)_{l}  [{z.min():.3g}, {z.max():.3g}]")
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        ax.set_aspect("equal")
        ax.set_xlabel("x1")
        if l == 0:
            ax.set_ylabel("x2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_set_indices(
    gp_basis: DeepGramPreservingBasis,
    out_path: Path,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    n_grid: int = 120,
    title: str = "space-splitter set index per layer",
) -> None:
    """Two-color map of set_index(y) plus continuous d_to_boundary with zero contour.

    Also writes ``{out_path.stem}_d_to_boundary{out_path.suffix}`` so a collapsed
    partition (boundary pushed outside the plot box) is visible.
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    X, Y, y = _state_space_mesh(
        gp_basis, x_range=x_range, y_range=y_range, n_grid=n_grid
    )
    with torch.no_grad():
        torch.nn.Module.eval(gp_basis)
        _, set_index, _, d_to_boundary = gp_basis._space_splitter.partner(y)
        set_index = set_index.detach().cpu()
        d_to_boundary = d_to_boundary.detach().cpu()

    n_layers = set_index.shape[1]
    x_np = X.detach().cpu().numpy()
    y_np = Y.detach().cpu().numpy()
    cmap = ListedColormap(["#4C78A8", "#F58518"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    fig, axes = plt.subplots(
        1, n_layers, figsize=(4.0 * n_layers, 3.6), squeeze=False
    )
    fig.suptitle(title, y=1.02)
    legend_handles = [
        Patch(facecolor="#4C78A8", edgecolor="none", label="set 0 (z≥0)"),
        Patch(facecolor="#F58518", edgecolor="none", label="set 1 (z<0)"),
    ]
    for l in range(n_layers):
        ax = axes[0, l]
        z = set_index[:, l].reshape(n_grid, n_grid).numpy()
        d = d_to_boundary[:, l].reshape(n_grid, n_grid).numpy()
        ax.pcolormesh(x_np, y_np, z, cmap=cmap, norm=norm, shading="auto")
        # Zero contour of latent coordinate — empty if boundary left the box.
        try:
            ax.contour(
                x_np, y_np, d, levels=[0.0], colors="k", linewidths=1.5
            )
        except Exception:
            pass
        frac1 = float((z == 1).mean())
        ax.set_title(
            f"layer {l}  (set1={frac1:.1%})\n"
            f"d∈[{d.min():.2f},{d.max():.2f}]"
        )
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        ax.set_aspect("equal")
        ax.set_xlabel("x1")
        if l == 0:
            ax.set_ylabel("x2")
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Continuous signed distance so a one-sided box is obvious.
    d_path = out_path.with_name(f"{out_path.stem}_d_to_boundary{out_path.suffix}")
    fig, axes = plt.subplots(
        1, n_layers, figsize=(4.0 * n_layers, 3.6), squeeze=False
    )
    fig.suptitle("latent d_to_boundary = z[reflection_axis] (0 = split)", y=1.02)
    for l in range(n_layers):
        ax = axes[0, l]
        d = d_to_boundary[:, l].reshape(n_grid, n_grid).numpy()
        vmax = max(abs(float(d.min())), abs(float(d.max())), 1e-6)
        cf = ax.contourf(
            x_np, y_np, d, levels=40, cmap="coolwarm", vmin=-vmax, vmax=vmax
        )
        ax.contour(x_np, y_np, d, levels=[0.0], colors="k", linewidths=1.5)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"layer {l}  [{d.min():.2f}, {d.max():.2f}]")
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        ax.set_aspect("equal")
        ax.set_xlabel("x1")
        if l == 0:
            ax.set_ylabel("x2")
    fig.tight_layout()
    fig.savefig(d_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved d_to_boundary map to {d_path}")


def _plot_layerwise_alpha_beta(
    gp_basis: DeepGramPreservingBasis,
    out_dir: Path,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    n_grid: int = 64,
) -> None:
    """Plot all α / β basis functions after 0, 1, …, n_layers reflections.

    Uses ``_apply_layers(y, ℓ)`` only — no changes to the basis class.
    """
    X, Y, y = _state_space_mesh(
        gp_basis, x_range=x_range, y_range=y_range, n_grid=n_grid
    )
    n_layers = gp_basis._n_layers
    with torch.no_grad():
        torch.nn.Module.eval(gp_basis)
        for layer in range(n_layers + 1):
            alpha, beta = gp_basis._apply_layers(y, layer)
            alpha = alpha.detach().cpu()
            beta = beta.detach().cpu()
            tag = "base" if layer == 0 else f"after_layer_{layer}"
            alpha_path = out_dir / f"alpha_{tag}.png"
            beta_path = out_dir / f"beta_{tag}.png"
            _plot_2d_values_grid(
                alpha,
                X,
                Y,
                alpha_path,
                title=f"α (phi) — {tag}",
                x_range=x_range,
                y_range=y_range,
            )
            _plot_2d_values_grid(
                beta,
                X,
                Y,
                beta_path,
                title=f"β (psi) — {tag}",
                x_range=x_range,
                y_range=y_range,
            )
            print(f"Saved layerwise α/β ({tag}) to {alpha_path.name}, {beta_path.name}")


if __name__ == "__main__":
    problem = FULLY_OBSERVABLE_PROBLEMS["van_der_pol"]

    ###
    use_gpu = torch.cuda.is_available()
    n_basis = 30
    n_hidden_features = 32
    n_hidden_layers = 3
    n_gp_layers = 3
    embedding_dim = 2
    tran_params = {
        "n_epochs_per_group": [3, 3],
        "iterations": 4,
        "lr_basis": 3e-3,
        "lr_weights": 5e-2,
    }
    init_params = {
        "n_epochs_per_group": [10],  # h0 coeffs only
        "iterations": 20,
        "lr_weights": 1e-2,
    }

    batch_size = 256
    n_timesteps_prop = problem.n_timesteps

    ###


    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using GPU: ", use_gpu)
    print("Device: ", device)

    system = problem.system
    dim = system.dim()
    if dim != 2:
        raise ValueError(f"VDP script expects a 2D system, got dim={dim}")

    m = n_basis
    d = dim
    rank = max(1, ceil(m ** (1.0 / d)))

    x0 = problem.train_initial_state_data()
    x_k, x_kp1 = problem.train_state_transition_data()
    traj_data = problem.test_data()

    x0_dataloader = DataLoader(
        TensorDataset(x0), batch_size=batch_size, shuffle=True, pin_memory=use_gpu
    )
    xp_dataloader = DataLoader(
        TensorDataset(x_kp1, x_k), batch_size=batch_size, shuffle=True, pin_memory=use_gpu
    )

    box_lows = problem.plot_bounds_low.to(device=device)
    box_highs = problem.plot_bounds_high.to(device=device)
    plot_x_range = (float(box_lows[0]), float(box_highs[0]))
    plot_y_range = (float(box_lows[1]), float(box_highs[1]))

    t = torch.linspace(0.0, 1.0, n_basis, device=device)
    means = box_lows.unsqueeze(-1) + (box_highs - box_lows).unsqueeze(-1) * t
    means = means.unsqueeze(0)  # (1, dim, n_basis)
    mean_jitter = 0.35
    std_init = 0.35
    phi_means = TrainableParameters.from_values(
        means + mean_jitter * torch.randn_like(means)
    ).to(device)
    phi_stds = PositiveParameters.from_values(
        torch.full((1, dim, n_basis), std_init, device=device)
    ).to(device)
    psi_means = TrainableParameters.from_values(
        means
        + torch.tensor([0.3, -0.2], device=device).view(1, dim, 1)
        + mean_jitter * torch.randn_like(means)
    ).to(device)
    psi_stds = PositiveParameters.from_values(
        torch.full((1, dim, n_basis), std_init + 0.1, device=device)
    ).to(device)
    phi_basis = GaussianBasis(phi_means, phi_stds, coeffs=None)
    psi_basis = GaussianBasis(psi_means, psi_stds, coeffs=None)

    embedding = torch.nn.Embedding(n_gp_layers, embedding_dim).to(device)
    tf = Transforms.make_transform(
        "maf",
        features=dim,
        context_features=embedding_dim,
        init_identity=True,
    ).to(device)
    idx_tf = IndexEmbeddingTransform(tf, embedding)
    space_splitter = LatentReflectionSpaceSplitter(idx_tf, reflection_axis=0).to(device)
    deformer = MLP(
        in_features=dim,
        out_features=n_gp_layers,
        hidden_features=n_hidden_features,
        num_hidden_layers=n_hidden_layers,
        zero_init_last=True,
    ).to(device)

    phi_psi_mutual = DeepGramPreservingBasis(
        phi_basis,
        psi_basis,
        space_splitter,
        deformer,
        fixed_base_basis=False,
    ).to(device)

    g_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0]), epsilon=1e-3
    ).to(device)
    h0_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0])
    ).to(device)

    B_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis, n_basis),
        mean=torch.tensor([1.0]),
        std=torch.tensor([1.0]),
        epsilon=0.0,
    ).to(device)
    B = DenseMatrixFactorization(B_coeffs)

    g_basis = phi_psi_mutual.get_basis(0, coeffs=g_coeffs)
    psi_basis = phi_psi_mutual.get_basis(1)

    tran_model = SumProdRFF(g_basis, psi_basis, B, numerical_tolerance=problem.numerical_tolerance)

    print("Training transition model")
    mle_loss_fn = loss.conditional_mle_loss
    # MAF splitter at full lr_basis quickly pushes z[axis]<0 on all data
    # (set1→100%, boundary leaves the plot box). Keep it slow / near-identity.
    splitter_params = list(phi_psi_mutual._space_splitter.parameters())
    splitter_ids = {id(p) for p in splitter_params}
    deformer_params = [
        p for p in phi_psi_mutual.parameters() if id(p) not in splitter_ids
    ]
    optimizers = {
        "basis": torch.optim.Adam(
            [
                {
                    "params": deformer_params,
                    "lr": tran_params["lr_basis"],
                },
                {
                    "params": splitter_params,
                    "lr": tran_params["lr_basis"] * 0.02,
                },
                {
                    "params": param_group_iter(
                        (phi_means, phi_stds, psi_means, psi_stds)
                    ),
                    "lr": tran_params["lr_basis"],
                },
            ]
        ),
        "weights": torch.optim.Adam(
            param_group_iter((g_coeffs, B_coeffs)), lr=tran_params["lr_weights"]
        ),
    }

    tran_model, best_loss_tran, training_time_tran = train.train_iterate(
        tran_model,
        xp_dataloader,
        {"mle": mle_loss_fn},
        optimizers,
        device=device,
        epochs_per_group=tran_params["n_epochs_per_group"],
        iterations=tran_params["iterations"],
        verbose=True,
        use_best="mle",
    )
    print("Done! \n")

    for p in phi_psi_mutual.parameters():
        p.requires_grad_(False)
    phi_means.set_requires_grad(False)
    phi_stds.set_requires_grad(False)
    psi_means.set_requires_grad(False)
    psi_stds.set_requires_grad(False)
    g_coeffs.set_requires_grad(False)

    h0_basis = phi_psi_mutual.get_basis(1, coeffs=h0_coeffs)
    init_model = LinearFF.from_rff(tran_model, h0_basis).to(device)

    print("Training initial model")
    mle_loss_fn = loss.mle_loss
    optimizers = {
        "weights": torch.optim.Adam(h0_coeffs.parameters(), lr=init_params["lr_weights"]),
    }

    init_model, best_loss_init, training_time_init = train.train_iterate(
        init_model,
        x0_dataloader,
        {"mle": mle_loss_fn},
        optimizers,
        device=device,
        epochs_per_group=init_params["n_epochs_per_group"],
        iterations=init_params["iterations"],
        verbose=True,
        use_best="mle",
    )
    print("Done! \n")

    print(
        f"Transition model loss: {best_loss_tran:.4f}, "
        f"training time: {training_time_tran:.2f} seconds"
    )
    print(
        f"Initial model loss: {best_loss_init:.4f}, "
        f"training time: {training_time_init:.2f} seconds"
    )

    analysis_device = device
    init_model = init_model.to(analysis_device).eval()
    tran_model = tran_model.to(analysis_device).eval()

    output_dir = Path("figures/gram_preserving/vdp")
    output_dir.mkdir(parents=True, exist_ok=True)

    deformer_out = output_dir / "sigmoid_deformer.png"
    _plot_sigmoid_deformer(
        phi_psi_mutual,
        deformer_out,
        x_range=plot_x_range,
        y_range=plot_y_range,
    )
    print(f"Saved sigmoid deformer map to {deformer_out}")

    interval_out = output_dir / "deformer_interval_LU.png"
    _plot_deformer_interval_width(
        phi_psi_mutual,
        interval_out,
        x_range=plot_x_range,
        y_range=plot_y_range,
    )
    print(f"Saved deformer interval (L−U) map to {interval_out}")

    set_idx_out = output_dir / "set_indices.png"
    _plot_set_indices(
        phi_psi_mutual,
        set_idx_out,
        x_range=plot_x_range,
        y_range=plot_y_range,
    )
    print(f"Saved set-index map to {set_idx_out}")

    layerwise_dir = output_dir / "layerwise"
    layerwise_dir.mkdir(parents=True, exist_ok=True)
    _plot_layerwise_alpha_beta(
        phi_psi_mutual,
        layerwise_dir,
        x_range=plot_x_range,
        y_range=plot_y_range,
    )

    mutual_phi_out = output_dir / "mutual_basis_phi_2d.png"
    _plot_2d_pair_member_grid(
        phi_psi_mutual,
        mutual_phi_out,
        index=0,
        title="VDP deep Gram-preserving: phi on R^2 (no wrap transform)",
        x_range=plot_x_range,
        y_range=plot_y_range,
    )
    print(f"Saved 2D mutual phi grid to {mutual_phi_out}")

    mutual_psi_out = output_dir / "mutual_basis_psi_2d.png"
    _plot_2d_pair_member_grid(
        phi_psi_mutual,
        mutual_psi_out,
        index=1,
        title="VDP deep Gram-preserving: psi on R^2 (no wrap transform)",
        x_range=plot_x_range,
        y_range=plot_y_range,
    )
    print(f"Saved 2D mutual psi grid to {mutual_psi_out}")

    box_lows_tuple = tuple(box_lows.tolist())
    box_highs_tuple = tuple(box_highs.tolist())

    cond_slice_out_path = output_dir / "conditional_slices_model_vs_data.png"
    _plot_conditional_slices_model_vs_data(
        tran_model,
        x_k,
        x_kp1,
        cond_slice_out_path,
        x_range=plot_x_range,
        y_range=plot_y_range,
        n_random_points=10,
        n_grid=120,
        min_points_per_slice=20,
        bin_width=0.4,
        random_seed=0,
        title=(
            "VDP deep Gram-preserving: fixed x1, varied x2 conditional bins — "
            "empirical vs model vs KDE (R^2, no wrap transform)"
        ),
    )
    print(f"Saved conditional slice comparison to {cond_slice_out_path}")

    n_slices = n_timesteps_prop + 1

    base_belief_seq = propagate.propagate(
        init_model,
        tran_model,
        n_steps=n_timesteps_prop,
    )
    belief_seq = [belief.to(analysis_device).eval() for belief in base_belief_seq]

    ll_per_step = []
    for i in range(n_slices):
        data_i = traj_data[i].to(analysis_device)
        ll = avg_log_likelihood(belief_seq[i], data_i)
        ll_per_step.append(float(ll.detach().cpu()))
        print(f"Avg log-likelihood at time {i}: {ll_per_step[-1]:.6f}")
        check_pdf_valid(belief_seq[i], (box_lows_tuple, box_highs_tuple), device=analysis_device)

    n_plot = min(n_timesteps_prop, len(belief_seq))
    fig, axes = plt.subplots(2, n_plot, figsize=(3.2 * n_plot, 6.5), squeeze=False)
    fig.suptitle("VDP deep Gram-preserving: beliefs at each time step (R^2)")
    for i in range(n_plot):
        plot_belief(
            axes[1, i],
            belief_seq[i],
            x_range=plot_x_range,
            y_range=plot_y_range,
        )
        axes[0, i].scatter(traj_data[i][:, 0], traj_data[i][:, 1], s=1)
        axes[0, i].set_aspect("equal")
        axes[0, i].set_xlim(plot_x_range[0], plot_x_range[1])
        axes[0, i].set_ylim(plot_y_range[0], plot_y_range[1])
        axes[0, i].set_title(f"t = {i}")

    beliefs_out_path = output_dir / "beliefs.png"
    fig.tight_layout()
    fig.savefig(beliefs_out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved beliefs to {beliefs_out_path}")
