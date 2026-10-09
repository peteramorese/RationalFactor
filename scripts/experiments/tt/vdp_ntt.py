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
)
from rational_factor.models.tt.nested_tt_parameters import (
    NestedTTVectorParameters,
    RowStochasticNestedTTMatrixParameters,
)
from rational_factor.models.tt.tt_basis import TTBasis
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood, check_pdf_valid
from rational_factor.tools.visualization import plot_belief
import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.tools.propagate as propagate


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


def _unit_square_mesh(
    basis,
    *,
    n_grid: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(X, Y, y_flat)`` on the unit square in the basis dtype/device."""
    dtype, device = basis.dtype_device()
    lin = torch.linspace(0.0, 1.0, n_grid, device=device, dtype=dtype)
    X, Y = torch.meshgrid(lin, lin, indexing="xy")
    y = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    return X, Y, y


def _plot_2d_tt_basis_grid(
    basis: TTBasis,
    out_path: Path,
    *,
    title: str,
    n_grid: int = 64,
    max_basis: int = 64,
) -> None:
    """Plot densified TTBasis evaluations on a square subplot grid."""
    if basis.dim() != 2:
        raise ValueError(f"_plot_2d_tt_basis_grid expects a 2D basis, got dim={basis.dim()}")

    X, Y, y = _unit_square_mesh(basis, n_grid=n_grid)
    with torch.no_grad():
        # Evaluate one grid point at a time would be slow; densify each structured
        # evaluation. For n_basis = n_primitive^2 this is fine at modest sizes.
        vals_list = []
        block = 256
        for start in range(0, y.shape[0], block):
            end = min(start + block, y.shape[0])
            # TTBasis currently supports batch size 1 for structured eval.
            rows = []
            for i in range(start, end):
                v = basis(y[i : i + 1])
                dense = v.to_dense() if hasattr(v, "to_dense") else v
                rows.append(dense.detach().cpu().reshape(-1))
            vals_list.append(torch.stack(rows, dim=0))
        vals = torch.cat(vals_list, dim=0)

    n_show = min(int(vals.shape[1]), int(max_basis))
    _plot_2d_values_grid(vals[:, :n_show], X, Y, out_path, title=title)


if __name__ == "__main__":
    problem = FULLY_OBSERVABLE_PROBLEMS["van_der_pol"]

    ###
    use_gpu = torch.cuda.is_available()
    dtype = torch.float64
    n_primitive = 10
    rank = 10
    depth = 2
    leaf_separation_rank = 20
    gaussian_std_epsilon = 0.03
    tran_params = {
        "n_epochs_per_group": [5, 5],  # basis, weights
        "iterations": 30,
        "lr_basis": 4 * 3e-3,
        "lr_weights": 1 * 5e-2,
    }
    init_params = {
        "n_epochs_per_group": [10],  # h0 coeffs only
        "iterations": 50,
        "lr_weights": 1e-2,
    }

    batch_size = 512
    n_timesteps_prop = problem.n_timesteps

    ###

    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using GPU: ", use_gpu)
    print("Device: ", device)
    print("Dtype: ", dtype)

    system = problem.system
    dim = system.dim()
    if dim != 2:
        raise ValueError(f"VDP NestedTT script expects a 2D system, got dim={dim}")

    d = dim
    tt_modes = (n_primitive,) * d
    n_basis = n_primitive ** d
    m = n_basis

    x0 = problem.train_initial_state_data().to(dtype=dtype)
    x_k, x_kp1 = problem.train_state_transition_data()
    x_k = x_k.to(dtype=dtype)
    x_kp1 = x_kp1.to(dtype=dtype)
    traj_data = [t.to(dtype=dtype) for t in problem.test_data()]

    x0_dataloader = DataLoader(
        TensorDataset(x0), batch_size=batch_size, shuffle=True, pin_memory=use_gpu
    )
    xp_dataloader = DataLoader(
        TensorDataset(x_kp1, x_k), batch_size=batch_size, shuffle=True, pin_memory=use_gpu
    )

    print(
        f"n_basis={m}, n_primitive={n_primitive}, dim={d}, rank={rank}, "
        f"depth={depth}, modes={tt_modes}"
    )

    parameter_shape = (1, d, n_primitive)
    phi_means = TrainableParameters.random_init(
        shape=parameter_shape, mean=torch.tensor([0.0]), std=torch.tensor([3.5])
    ).to(device, dtype=dtype)
    phi_std = PositiveParameters.random_init(
        shape=parameter_shape,
        mean=torch.tensor([0.8]),
        std=torch.tensor([0.5]),
        epsilon=gaussian_std_epsilon,
    ).to(device, dtype=dtype)
    psi_means = TrainableParameters.random_init(
        shape=parameter_shape, mean=torch.tensor([0.0]), std=torch.tensor([3.5])
    ).to(device, dtype=dtype)
    psi_std = PositiveParameters.random_init(
        shape=parameter_shape,
        mean=torch.tensor([0.8]),
        std=torch.tensor([0.5]),
        epsilon=gaussian_std_epsilon,
    ).to(device, dtype=dtype)
    phi_primitive = GaussianBasis(mean_params=phi_means, std_params=phi_std)
    psi_primitive = GaussianBasis(mean_params=psi_means, std_params=psi_std)

    phi_basis = TTBasis(phi_primitive, nested_depth=depth)
    psi_basis = TTBasis(psi_primitive, nested_depth=depth)

    def positive_leaf_factory(shape, *, mean=1.0, std=1.0, epsilon=0.0):
        return PositiveParameters.random_init(
            shape=tuple(shape), mean=mean, std=std, epsilon=epsilon
        ).to(device, dtype=dtype)

    def trainable_leaf_factory(shape, *, mean=0.0, std=1.0):
        return TrainableParameters.random_init(
            shape=tuple(shape), mean=mean, std=std
        ).to(device, dtype=dtype)

    g_coeffs = NestedTTVectorParameters.from_core_spec(
        tt_modes,
        depth=depth,
        ranks=1,
        leaf_factory=lambda shape: positive_leaf_factory(
            shape, mean=1.0, std=1.0, epsilon=1e-3
        ),
    )
    h0_coeffs = NestedTTVectorParameters.from_core_spec(
        tt_modes,
        depth=depth,
        ranks=1,
        leaf_factory=lambda shape: positive_leaf_factory(shape, mean=1.0, std=1.0),
        separation_rank=leaf_separation_rank,
    )
    # Length depth-1 rank tuple → hierarchy (1, r, ..., r) for any depth >= 1.
    B = RowStochasticNestedTTMatrixParameters.from_core_spec(
        tt_modes,
        depth=depth,
        ranks=1 if depth == 1 else (rank,) * (depth - 1),
        separation_rank=leaf_separation_rank,
        leaf_factory=lambda shape: trainable_leaf_factory(shape, mean=0.0, std=1.0),
    )

    rff = SumProdRFF(g_coeffs, phi_basis, psi_basis, B, numerical_tolerance=1e-10)
    tran_model = rff

    print("Training transition model")
    mle_loss_fn = loss.conditional_mle_loss
    optimizers = {
        "basis": torch.optim.Adam(
            param_group_iter((phi_means, phi_std, psi_means, psi_std)),
            lr=tran_params["lr_basis"],
        ),
        "weights": torch.optim.Adam(
            param_group_iter((*g_coeffs.leaves, *B.leaves)),
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

    for p in param_group_iter((phi_means, phi_std, psi_means, psi_std, *g_coeffs.leaves)):
        p.requires_grad_(False)

    h0_basis = TTBasis(psi_primitive, coeffs=h0_coeffs)
    init_model = LinearFF.from_rff(tran_model, h0_basis).to(device=device, dtype=dtype)

    print("Training initial model")
    mle_loss_fn = loss.mle_loss
    optimizers = {
        "weights": torch.optim.Adam(
            param_group_iter(h0_coeffs.leaves),
            lr=init_params["lr_weights"],
        ),
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
    init_model = init_model.to(device=analysis_device, dtype=dtype).eval()
    tran_model = tran_model.to(device=analysis_device, dtype=dtype).eval()

    output_dir = Path("figures/tt/vdp_ntt")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_x_range = (
        float(problem.plot_bounds_low[0]),
        float(problem.plot_bounds_high[0]),
    )
    plot_y_range = (
        float(problem.plot_bounds_low[1]),
        float(problem.plot_bounds_high[1]),
    )
    box_lows = tuple(problem.plot_bounds_low.tolist())
    box_highs = tuple(problem.plot_bounds_high.tolist())

    #phi_out = output_dir / "tt_basis_phi_2d.png"
    #_plot_2d_tt_basis_grid(
    #    phi_basis,
    #    phi_out,
    #    title="VDP NestedTT: 2D phi TTBasis (densified)",
    #    max_basis=min(m, 64),
    #)
    #print(f"Saved 2D phi grid to {phi_out}")

    #psi_out = output_dir / "tt_basis_psi_2d.png"
    #_plot_2d_tt_basis_grid(
    #    psi_basis,
    #    psi_out,
    #    title="VDP NestedTT: 2D psi TTBasis (densified)",
    #    max_basis=min(m, 64),
    #)
    #print(f"Saved 2D psi grid to {psi_out}")

    #cond_slice_out_path = output_dir / "conditional_slices_model_vs_data.png"
    #_plot_conditional_slices_model_vs_data(
    #    tran_model,
    #    x_k,
    #    x_kp1,
    #    cond_slice_out_path,
    #    x_range=plot_x_range,
    #    y_range=plot_y_range,
    #    n_random_points=10,
    #    n_grid=120,
    #    min_points_per_slice=20,
    #    bin_width=0.4,
    #    random_seed=0,
    #    title=(
    #        "VDP NestedTT: fixed x1, varied x2 conditional bins — "
    #        "empirical vs model vs KDE"
    #    ),
    #)
    #print(f"Saved conditional slice comparison to {cond_slice_out_path}")

    n_slices = n_timesteps_prop + 1

    base_belief_seq = propagate.propagate(
        init_model,
        tran_model,
        n_steps=n_timesteps_prop,
    )
    belief_seq = [
        belief.to(device=analysis_device, dtype=dtype).eval()
        for belief in base_belief_seq
    ]

    ll_per_step = []
    for i in range(n_slices):
        data_i = traj_data[i].to(device=analysis_device, dtype=dtype)
        ll = avg_log_likelihood(belief_seq[i], data_i)
        ll_per_step.append(float(ll.detach().cpu()))
        print(f"Avg log-likelihood at time {i}: {ll_per_step[-1]:.6f}")
        check_pdf_valid(belief_seq[i], (box_lows, box_highs), device=analysis_device)

    n_plot = min(n_timesteps_prop, len(belief_seq))
    fig, axes = plt.subplots(2, n_plot, figsize=(3.2 * n_plot, 6.5), squeeze=False)
    fig.suptitle("VDP NestedTT: beliefs at each time step")
    for i in range(n_plot):
        plot_belief(
            axes[1, i],
            belief_seq[i],
            x_range=plot_x_range,
            y_range=plot_y_range,
        )
        axes[0, i].scatter(traj_data[i][:, 0].cpu(), traj_data[i][:, 1].cpu(), s=1)
        axes[0, i].set_aspect("equal")
        axes[0, i].set_xlim(plot_x_range[0], plot_x_range[1])
        axes[0, i].set_ylim(plot_y_range[0], plot_y_range[1])
        axes[0, i].set_title(f"t = {i}")

    beliefs_out_path = output_dir / "beliefs.png"
    fig.tight_layout()
    fig.savefig(beliefs_out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved beliefs to {beliefs_out_path}")
