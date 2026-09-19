from pathlib import Path
from math import ceil, sqrt

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from rational_factor.models.basis_functions import BSpline1DBasis
from rational_factor.models.composite_model import CompositeConditionalModel, CompositeDensityModel
from rational_factor.models.domain_transformation import ErfSeparableTF
from rational_factor.models.factor_forms import SumProdRFF, LinearFF
from rational_factor.models.kde import GaussianKDE
from rational_factor.models.mlp import PairedMaskedMetzlerConeMLP
from rational_factor.models.parameters import (
    PositiveParameters,
    DenseMatrixFactorization,
    param_group_iter,
)
from rational_factor.models.autoregressive_basis import AutoregressiveMetzlerConeMutualBasis
from rational_factor.models.structured_matrices import DenseMatrix
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood, check_pdf_valid
from rational_factor.tools.metzler_cone_rays import RankDeficientGramConeRayFinder
from rational_factor.tools.visualization import plot_belief
import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.tools.propagate as propagate


def _random_nonneg_rank_r(
    m: int,
    r: int,
    rng: np.random.Generator,
    *,
    near_diag: bool = True,
) -> np.ndarray:
    """Strictly nonnegative ``m × m`` matrix of exact rank ``r``.

    Factors are lognormal. When ``near_diag`` is set, each factor column is
    softly localized around a diagonal center so entries decay away from the
    band (while staying strictly positive).
    """
    U = rng.lognormal(mean=0.0, sigma=0.65, size=(m, r))
    V = rng.lognormal(mean=0.0, sigma=0.65, size=(m, r))
    if near_diag:
        rows = np.arange(m, dtype=np.float64)[:, None]
        centers = np.linspace(0.0, m - 1.0, r)[None, :]
        width = max(m / r, 1.5)
        w = 0.15 + 0.85 * np.exp(-0.5 * ((rows - centers) / width) ** 2)
        U = U * w
        V = V * w
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    V /= np.linalg.norm(V, axis=0, keepdims=True)
    A = U @ V.T
    assert A.min() > 0.0
    assert np.linalg.matrix_rank(A, tol=1e-10) == r
    return A


def _uniform_nonneg_rank_r(
    m: int,
    r: int,
    rng: np.random.Generator,
    *,
    low: float = 0.1,
    high: float = 1.0,
) -> np.ndarray:
    """Strictly nonnegative ``m × m`` matrix of exact rank ``r`` via uniform factors.

    Draws ``U, V ∼ Unif[low, high]^{m × r}`` (column-normalized), returns ``U Vᵀ``.
    No near-diagonal localization — denser / more spatially uniform than the
    lognormal sampler.
    """
    if not (0.0 < low < high):
        raise ValueError(f"need 0 < low < high, got low={low}, high={high}")
    U = rng.uniform(low, high, size=(m, r))
    V = rng.uniform(low, high, size=(m, r))
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    V /= np.linalg.norm(V, axis=0, keepdims=True)
    A = U @ V.T
    assert A.min() > 0.0
    assert np.linalg.matrix_rank(A, tol=1e-10) == r
    return A


def _sample_A0_B0(
    m: int,
    r: int,
    d: int,
    rng: np.random.Generator,
    *,
    method: str = "uniform",
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``(A0, B0)`` each of shape ``(d, m, m)`` with the given method."""
    if method == "uniform":
        draw = lambda: _uniform_nonneg_rank_r(m, r, rng)
    elif method == "lognormal_near_diag":
        draw = lambda: _random_nonneg_rank_r(m, r, rng, near_diag=True)
    elif method == "lognormal":
        draw = lambda: _random_nonneg_rank_r(m, r, rng, near_diag=False)
    else:
        raise ValueError(
            f"unknown A0/B0 sampler {method!r}; expected "
            "'uniform', 'lognormal_near_diag', or 'lognormal'"
        )
    A0 = np.stack([draw() for _ in range(d)], axis=0)
    B0 = np.stack([draw() for _ in range(d)], axis=0)
    return A0, B0


def _build_paired_ray_banks(
    A0: torch.Tensor,
    B0: torch.Tensor,
    G: torch.Tensor,
    *,
    rank: int,
    n_rays: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each slot ``ℓ``, find paired Metzler rays of ``Γ_ℓ = A0_ℓ G B0_ℓᵀ``.

    Returns ``(R_bank, T_bank)`` each of shape ``(d, n_rays, m, m)``.
    """
    d, m, _ = A0.shape
    G_np = G.detach().cpu().numpy()
    A0_np = A0.detach().cpu().numpy()
    B0_np = B0.detach().cpu().numpy()

    R_list = []
    T_list = []
    for ell in range(d):
        gamma = A0_np[ell] @ G_np @ B0_np[ell].T
        s = np.linalg.svd(gamma, compute_uv=False)
        num_rank = int(np.linalg.matrix_rank(gamma, tol=1e-9 * s[0]))
        if num_rank != rank:
            raise RuntimeError(
                f"Gamma[{ell}] has numerical rank {num_rank}, expected {rank}"
            )
        print(
            f"  slot {ell}: finding {n_rays} paired rays for Gamma "
            f"shape {(m, m)}, rank={rank}, "
            f"sigma[:r]={np.array2string(s[:rank], precision=3)} ..."
        )
        finder = RankDeficientGramConeRayFinder(
            torch.as_tensor(gamma, dtype=torch.float64),
            rank=rank,
            m=m,
            support_mode="auto",
            candidate_factor=32,
            seed=seed + 17 * ell,
        )
        paired = finder.find(n_rays)
        R_list.append(paired.R.to_dense().detach())
        T_list.append(paired.T.to_dense().detach())
        print(f"  slot {ell}: got R/T shape {tuple(R_list[-1].shape)}")

    R_bank = torch.stack(R_list, dim=0)
    T_bank = torch.stack(T_list, dim=0)
    return R_bank, T_bank


def _plot_cone_ray_heatmaps(
    bank: torch.Tensor | np.ndarray,
    out_path: Path,
    *,
    slot: int,
    name: str,
    max_rays: int = 16,
    title: str = "",
) -> None:
    """Color heatmaps of ``expm`` of cone-ray matrices for one slot.

    ``bank`` has shape ``(d, k, m, m)`` or ``(k, m, m)``. Shows up to
    ``max_rays`` rays as ``exp(M_i)`` in a grid with a shared color scale
    (``vmin=0``) so diversity of the exponentials is easy to compare.
    """
    if isinstance(bank, torch.Tensor):
        mats = bank.detach().cpu()
    else:
        mats = torch.as_tensor(bank)
    if mats.ndim == 4:
        mats = mats[slot]
    if mats.ndim != 3:
        raise ValueError(f"expected (k,m,m) or (d,k,m,m), got {tuple(mats.shape)}")

    k = mats.shape[0]
    n_show = min(int(max_rays), k)
    # Batched matrix exp: (n_show, m, m)
    exps = torch.matrix_exp(mats[:n_show].to(dtype=torch.float64)).numpy()

    n_cols = max(1, ceil(sqrt(n_show)))
    n_rows = max(1, ceil(n_show / n_cols))
    vmax = float(np.max(exps))
    if vmax < 1e-12:
        vmax = 1.0

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.0 * n_cols, 1.85 * n_rows),
        squeeze=False,
    )
    if title:
        fig.suptitle(title, y=1.01)

    cmap = "viridis"
    im = None
    for idx in range(n_show):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        im = ax.imshow(
            exps[idx],
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(f"exp({name}[{idx}])", fontsize=7, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n_show, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].set_visible(False)

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_conditional_slices_model_vs_data(
    tran_model: CompositeConditionalModel,
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
) -> None:
    """Plot (n_pts, n_basis) values on the unit-square mesh as a subplot grid."""
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
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
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


def _plot_masked_metzler_slot_entries(
    get_M,
    out_path: Path,
    *,
    slot: int,
    n_slots: int,
    n_grid: int = 128,
    n_show: int = 9,
    title: str = "",
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> None:
    """Plot a few entries of ``M_slot(x)`` vs its free coordinate on ``[0, 1]``.

    ``get_M(x)`` maps ``(n_grid, d)`` inputs to a dense ``(n_grid, d, m, m)``
    tensor (or Matrix of that shape). Slot ``ℓ`` depends on ``x_<ℓ``.

    Instead of a full ``m × m`` grid, shows ``n_show`` entries on an evenly
    spaced index subgrid (default 9 → up to a 3×3 layout).
    """
    d = n_slots
    if not (0 <= slot < d):
        raise ValueError(f"slot must be in [0, {d}), got {slot}")
    if n_show < 1:
        raise ValueError(f"n_show must be positive, got {n_show}")

    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.device("cpu")

    t = torch.linspace(0.0, 1.0, n_grid, device=device, dtype=dtype)
    x = torch.zeros(n_grid, d, device=device, dtype=dtype)
    if slot == 0:
        xlabel = "(constant)"
    else:
        if slot >= 2:
            x[:, : slot - 1] = 0.5
        x[:, slot - 1] = t
        xlabel = f"x_{slot - 1}"

    with torch.no_grad():
        M = get_M(x)
        if hasattr(M, "to_dense"):
            M = M.to_dense()
        M = M[:, slot]  # (n_grid, m, m)

    m = M.shape[-1]
    n_side = max(1, ceil(sqrt(n_show)))
    idx = torch.linspace(0, m - 1, n_side).round().long().unique().tolist()
    pairs = [(int(i), int(j)) for i in idx for j in idx][:n_show]

    n_cols = min(n_side, len(pairs))
    n_rows = max(1, ceil(len(pairs) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.4 * n_cols, 2.1 * n_rows),
        sharex=True,
        squeeze=False,
    )
    if title:
        fig.suptitle(title, y=1.01)

    M_np = M.detach().cpu().numpy()
    t_np = t.detach().cpu().numpy()
    for p, (i, j) in enumerate(pairs):
        r, c = divmod(p, n_cols)
        ax = axes[r, c]
        ax.plot(t_np, M_np[:, i, j], color="C0", lw=1.0)
        ax.axhline(0.0, color="0.6", lw=0.4, zorder=0)
        ax.set_xlim(0.0, 1.0)
        ax.set_title(f"({i},{j})", fontsize=8, pad=2)
        ax.tick_params(labelsize=6)
        if r < n_rows - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(xlabel, fontsize=7)

    for p in range(len(pairs), n_rows * n_cols):
        r, c = divmod(p, n_cols)
        axes[r, c].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _unit_square_mesh(
    pair_basis,
    *,
    n_grid: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(X, Y, y_flat)`` on the unit square in the basis dtype/device."""
    dtype, device = pair_basis.dtype_device()
    lin = torch.linspace(0.0, 1.0, n_grid, device=device, dtype=dtype)
    X, Y = torch.meshgrid(lin, lin, indexing="xy")
    y = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    return X, Y, y


def _plot_2d_pair_member_grid(
    pair_basis,
    out_path: Path,
    *,
    index: int,
    title: str,
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

    X, Y, y = _unit_square_mesh(pair_basis, n_grid=n_grid)
    with torch.no_grad():
        if isinstance(pair_basis, torch.nn.Module):
            torch.nn.Module.eval(pair_basis)
        vals = pair_basis.eval(y, index).detach().cpu()

    _plot_2d_values_grid(vals, X, Y, out_path, title=title)


if __name__ == "__main__":
    problem = FULLY_OBSERVABLE_PROBLEMS["van_der_pol"]

    ###
    use_gpu = torch.cuda.is_available()
    n_basis = 10
    bspline_degree = 5
    n_rays = 20
    ray_seed = 0
    a0_b0_sampler = "uniform"  # "uniform" | "lognormal_near_diag" | "lognormal"
    # Finder returns unit-Frobenius (R,T) pairs; scale so exp(R) leaves near-I.
    # Same factor on R and T preserves R M + M Tᵀ = 0. Keep MLP coeffs O(1)
    # once rays are pre-scaled (otherwise expm can explode).
    mc_ray_scale = 25.0
    mc_mlp_hidden = 64
    mc_mlp_layers = 2
    # c = coeff_scale * softplus(base) * softplus(Δ(x)); base sets magnitude,
    # Δ carries x_<ℓ dependence (high last-layer gain, zero last-layer bias).
    mc_coeff_scale = 2.0
    mc_max_coeff = None  # do not clip away x-dependence
    mc_bias_init = 1.0  # softplus(1)≈1.3 base magnitude per ray
    tran_params = {
        "n_epochs_per_group": [3, 3],  # wrap + mc_mlps, weights
        "iterations": 20,
        "lr_mc_mlp": 3e-3,
        "lr_weights": 5e-2,
        "lr_wrap": 1e-3,
    }
    init_params = {
        "n_epochs_per_group": [10],  # h0 coeffs only
        "iterations": 10,
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

    if rank >= m:
        raise ValueError(f"rank={rank} must satisfy 1 <= rank < m={m}")
    print(f"n_basis={m}, dim={d}, rank=ceil(m^(1/d))={rank}, n_rays={n_rays}")

    n_cells = n_basis - bspline_degree
    if n_cells < 1:
        raise ValueError(f"n_basis={n_basis} too small for degree={bspline_degree}")

    nom_alpha_basis = BSpline1DBasis(
        n_cells=n_cells,
        degree=bspline_degree,
        device=device,
    )
    nom_beta_basis = nom_alpha_basis
    assert nom_alpha_basis.n_basis_functions() == n_basis

    G = nom_alpha_basis.Omega2(nom_beta_basis).to_dense()
    if G.dim() == 3:
        G = G.squeeze(0)
    G = G.detach().cpu().double()
    print(f"Nominal Gram G shape {tuple(G.shape)}")

    rng = np.random.default_rng(ray_seed)
    A0_np, B0_np = _sample_A0_B0(m, rank, d, rng, method=a0_b0_sampler)
    A0 = torch.as_tensor(A0_np, dtype=torch.float64)
    B0 = torch.as_tensor(B0_np, dtype=torch.float64)
    print(
        f"A0/B0 sampler={a0_b0_sampler!r}, shape {tuple(A0.shape)}, "
        f"min(A0)={float(A0.min()):.3e}, mean(A0)={float(A0.mean()):.3e}"
    )
    print("Computing paired Metzler cone rays per Gamma_l ...")
    R_bank, T_bank = _build_paired_ray_banks(
        A0, B0, G, rank=rank, n_rays=n_rays, seed=ray_seed
    )
    R_bank = R_bank * mc_ray_scale
    T_bank = T_bank * mc_ray_scale
    print(
        f"Scaled ray banks by mc_ray_scale={mc_ray_scale}: "
        f"||R||_F mean={R_bank.reshape(R_bank.shape[0], R_bank.shape[1], -1).norm(dim=-1).mean():.3f}"
    )

    output_dir = Path("figures/metzler/vdp")
    output_dir.mkdir(parents=True, exist_ok=True)
    max_rays_plot = min(16, n_rays)
    for name, bank in (("R", R_bank), ("T", T_bank)):
        for slot in range(d):
            out = output_dir / f"cone_rays_{name}_slot{slot}_expm_heatmaps.png"
            _plot_cone_ray_heatmaps(
                bank,
                out,
                slot=slot,
                name=name,
                max_rays=max_rays_plot,
                title=(
                    f"VDP Metzler: exp({name}) cone rays slot {slot} "
                    f"(scale={mc_ray_scale}, first {max_rays_plot}/{n_rays})"
                ),
            )
            print(f"Saved {name} slot-{slot} expm ray heatmaps to {out}")
    
    input("...")

    dtype_model = torch.float32
    A0 = A0.to(device=device, dtype=dtype_model)
    B0 = B0.to(device=device, dtype=dtype_model)
    R_bank = R_bank.to(device=device, dtype=dtype_model)
    T_bank = T_bank.to(device=device, dtype=dtype_model)

    paired_mc_mlp = PairedMaskedMetzlerConeMLP(
        R=DenseMatrix(R_bank),
        T=DenseMatrix(T_bank),
        hidden_features=mc_mlp_hidden,
        num_hidden_layers=mc_mlp_layers,
        zero_init_last=False,
        coeff_scale=mc_coeff_scale,
        max_coeff=mc_max_coeff,
        bias_init=mc_bias_init,
    ).to(device)

    phi_psi_mutual = AutoregressiveMetzlerConeMutualBasis(
        nom_alpha_basis,
        nom_beta_basis,
        A0,
        B0,
        paired_mc_mlp,
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

    wrap_tf = ErfSeparableTF.from_data(x_k, trainable=True).to(device)
    rff = SumProdRFF(g_basis, psi_basis, B, numerical_tolerance=problem.numerical_tolerance)
    tran_model = CompositeConditionalModel([wrap_tf], rff).to(device)

    print("Training transition model")
    mle_loss_fn = loss.conditional_mle_loss
    optimizers = {
        "basis": torch.optim.Adam(
            [
                {"params": paired_mc_mlp.parameters(), "lr": tran_params["lr_mc_mlp"]},
                {"params": wrap_tf.parameters(), "lr": tran_params["lr_wrap"]},
            ]
        ),
        "weights": torch.optim.Adam(
            param_group_iter((g_coeffs, B_coeffs)),
            lr=tran_params["lr_weights"],
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
    g_coeffs.set_requires_grad(False)
    trained_wrap_tf = ErfSeparableTF.copy_from_trainable(wrap_tf).to(device)

    h0_basis = phi_psi_mutual.get_basis(1, coeffs=h0_coeffs)
    init_model = CompositeDensityModel(
        [trained_wrap_tf],
        LinearFF.from_rff(tran_model.conditional_density_model, h0_basis),
    ).to(device)

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
    trained_wrap_tf = trained_wrap_tf.to(analysis_device).eval()

    output_dir = Path("figures/metzler/vdp")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Active slot for d=2 is ℓ=1 (ℓ=0 is excluded from the product).
    paired_mc_mlp.eval()
    dtype_plot, device_plot = paired_mc_mlp.R.dtype, paired_mc_mlp.R.device

    def _R_of(x):
        return paired_mc_mlp(x)[0]

    def _T_of(x):
        return paired_mc_mlp(x)[1]

    for name, get_M in (("R", _R_of), ("T", _T_of)):
        for slot in range(d):
            out = output_dir / f"metzler_cone_mlp_{name}_slot{slot}_entries.png"
            _plot_masked_metzler_slot_entries(
                get_M,
                out,
                slot=slot,
                n_slots=d,
                n_show=9,
                n_grid=128,
                title=(
                    f"VDP Metzler: paired {name} slot {slot} "
                    f"(9 of {n_basis}×{n_basis} entries) vs free coord"
                ),
                dtype=dtype_plot,
                device=device_plot,
            )
            print(f"Saved {name} slot-{slot} entry subsample to {out}")

    mutual_phi_out = output_dir / "mutual_basis_phi_2d.png"
    _plot_2d_pair_member_grid(
        phi_psi_mutual,
        mutual_phi_out,
        index=0,
        title="VDP Metzler: 2D mutual phi basis by index",
    )
    print(f"Saved 2D mutual phi grid to {mutual_phi_out}")

    mutual_psi_out = output_dir / "mutual_basis_psi_2d.png"
    _plot_2d_pair_member_grid(
        phi_psi_mutual,
        mutual_psi_out,
        index=1,
        title="VDP Metzler: 2D mutual psi basis by index",
    )
    print(f"Saved 2D mutual psi grid to {mutual_psi_out}")

    box_lows = tuple(problem.plot_bounds_low.tolist())
    box_highs = tuple(problem.plot_bounds_high.tolist())

    cond_slice_out_path = output_dir / "conditional_slices_model_vs_data.png"
    _plot_conditional_slices_model_vs_data(
        tran_model,
        x_k,
        x_kp1,
        cond_slice_out_path,
        x_range=(box_lows[0], box_highs[0]),
        y_range=(box_lows[1], box_highs[1]),
        n_random_points=10,
        n_grid=120,
        min_points_per_slice=20,
        bin_width=0.4,
        random_seed=0,
        title=(
            "VDP Metzler: fixed x1, varied x2 conditional bins — "
            "empirical vs model vs KDE"
        ),
    )
    print(f"Saved conditional slice comparison to {cond_slice_out_path}")

    n_slices = n_timesteps_prop + 1

    base_belief_seq = propagate.propagate(
        init_model.density_model,
        tran_model.conditional_density_model,
        n_steps=n_timesteps_prop,
    )
    belief_seq = [
        CompositeDensityModel([trained_wrap_tf], belief).to(analysis_device).eval()
        for belief in base_belief_seq
    ]

    ll_per_step = []
    for i in range(n_slices):
        data_i = traj_data[i].to(analysis_device)
        ll = avg_log_likelihood(belief_seq[i], data_i)
        ll_per_step.append(float(ll.detach().cpu()))
        print(f"Avg log-likelihood at time {i}: {ll_per_step[-1]:.6f}")
        check_pdf_valid(belief_seq[i], (box_lows, box_highs), device=analysis_device)

    n_plot = min(n_timesteps_prop, len(belief_seq))
    fig, axes = plt.subplots(2, n_plot, figsize=(3.2 * n_plot, 6.5), squeeze=False)
    fig.suptitle("VDP Metzler: beliefs at each time step")
    for i in range(n_plot):
        plot_belief(
            axes[1, i],
            belief_seq[i],
            x_range=(box_lows[0], box_highs[0]),
            y_range=(box_lows[1], box_highs[1]),
        )
        axes[0, i].scatter(traj_data[i][:, 0], traj_data[i][:, 1], s=1)
        axes[0, i].set_aspect("equal")
        axes[0, i].set_xlim(box_lows[0], box_highs[0])
        axes[0, i].set_ylim(box_lows[1], box_highs[1])
        axes[0, i].set_title(f"t = {i}")

    beliefs_out_path = output_dir / "beliefs.png"
    fig.tight_layout()
    fig.savefig(beliefs_out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved beliefs to {beliefs_out_path}")
