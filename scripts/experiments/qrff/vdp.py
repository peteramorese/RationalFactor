"""2D Van der Pol: dense quadratic SOS (QRFF) with belief / basis visualization.

Training mirrors qrff/acc_comp.py; diagnostics follow scripts/experiments/tt/vdp.py
(2D basis grids, conditional slices, propagated beliefs).
"""
from math import ceil, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.factor_forms import QuadraticFF, QuadraticRFF
from rational_factor.models.kde import GaussianKDE
from rational_factor.models.parameters import TrainableParameters, param_group_iter
from rational_factor.models.train import Trainer
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood
from rational_factor.tools.misc import init_gaussian_basis_params_gmm
from rational_factor.tools.visualization import plot_belief
import rational_factor.models.loss as loss


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
    """Compare trained p(x'|x) to binned empirical samples and a conditional KDE."""
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
    centers = near_sorted[pick].clone()
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
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
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


def _plot_2d_gaussian_basis_grid(
    basis: GaussianBasis,
    out_path: Path,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    title: str,
    n_grid: int = 64,
) -> None:
    """Plot every 2D separable Gaussian basis function on a square subplot grid."""
    if basis.dim() != 2:
        raise ValueError(
            f"_plot_2d_gaussian_basis_grid expects a 2D basis, got dim={basis.dim()}"
        )

    dtype, device = basis.dtype_device()
    x_lin = torch.linspace(x_range[0], x_range[1], n_grid, device=device, dtype=dtype)
    y_lin = torch.linspace(y_range[0], y_range[1], n_grid, device=device, dtype=dtype)
    X, Y = torch.meshgrid(x_lin, y_lin, indexing="xy")
    y = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    with torch.no_grad():
        # Drop learned coeffs so the plot shows the raw feature shapes.
        vals = basis.eval_dim(y).prod(dim=1).detach().cpu()

    _plot_2d_values_grid(
        vals, X, Y, out_path, title=title, x_range=x_range, y_range=y_range
    )


if __name__ == "__main__":
    problem = FULLY_OBSERVABLE_PROBLEMS["van_der_pol"]

    use_gpu = torch.cuda.is_available()
    dtype = torch.float64
    n_basis = 20  # Q0 is (n_basis**2, n_basis**2); keep small for dense SOS.
    pd_jitter = 1e-8
    tran_params = dict(
        epochs=80,
        basis_iterations=3,
        weights_iterations=3,
        lr_basis=3e-3,
        lr_weights=5e-3,
    )
    init_params = dict(
        epochs=50,
        coeff_iterations=10,
        lr_coeffs=1e-2,
        basis_iterations=2,
        lr_basis=1e-3,
    )
    batch_size = 256
    n_timesteps_prop = problem.n_timesteps

    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"device={device}, dtype={dtype}, SOS basis count={n_basis}")

    system = problem.system
    dim = system.dim()
    if dim != 2:
        raise ValueError(f"VDP QRFF script expects a 2D system, got dim={dim}")

    x0 = problem.train_initial_state_data().to(dtype=dtype)
    x_k, x_kp1 = problem.train_state_transition_data()
    x_k, x_kp1 = x_k.to(dtype=dtype), x_kp1.to(dtype=dtype)
    traj_data = [x.to(dtype=dtype) for x in problem.test_data()]

    x0_loader = DataLoader(
        TensorDataset(x0), batch_size=batch_size, shuffle=True, pin_memory=use_gpu
    )
    xp_loader = DataLoader(
        TensorDataset(x_kp1, x_k),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_gpu,
    )

    phi_means, phi_stds = init_gaussian_basis_params_gmm(
        x_k, dim, n_basis, epsilon=0.0, device=device
    )
    psi_means, psi_stds = init_gaussian_basis_params_gmm(
        x_kp1, dim, n_basis, epsilon=0.0, device=device
    )
    phi = GaussianBasis(mean_params=phi_means, std_params=phi_stds, coeffs=None)
    psi = GaussianBasis(mean_params=psi_means, std_params=psi_stds, coeffs=None)

    LA = TrainableParameters.random_init(
        shape=(1, n_basis, n_basis), mean=0.0, std=0.1
    ).to(device=device, dtype=dtype)
    LQ0 = TrainableParameters.random_init(
        shape=(1, n_basis * n_basis, n_basis * n_basis), mean=0.0, std=0.05
    ).to(device=device, dtype=dtype)

    tran_model = QuadraticRFF(
        LA,
        phi,
        psi,
        LQ0,
        numerical_tolerance=problem.numerical_tolerance,
        pd_jitter=pd_jitter,
    )
    print("Training SOS transition model")
    transition_trainer = Trainer(tran_model, device=device)
    transition_trainer.add_dataset(xp_loader, "transitions")
    transition_trainer.add_optimizer(
        torch.optim.Adam(
            param_group_iter((phi_means, phi_stds, psi_means, psi_stds)),
            lr=tran_params["lr_basis"],
        ),
        "basis",
        n_iterations=tran_params["basis_iterations"],
    )
    transition_trainer.add_optimizer(
        torch.optim.Adam(param_group_iter((LA, LQ0)), lr=tran_params["lr_weights"]),
        "weights",
        n_iterations=tran_params["weights_iterations"],
    )
    transition_trainer.add_loss_function(loss.conditional_mle_loss, "mle", "transitions")
    tran_model, best_loss_tran, training_time_tran = transition_trainer.train(
        epochs=tran_params["epochs"], use_best="mle", verbose=True
    )
    if tran_model is None:
        raise SystemExit("No stable SOS transition checkpoint was found.")
    print("Transition model trained.")

    for parameter in tran_model.parameters():
        parameter.requires_grad_(False)

    psi0_means, psi0_stds = init_gaussian_basis_params_gmm(
        x0, dim, n_basis, epsilon=0.0, device=device
    )
    psi0 = GaussianBasis(mean_params=psi0_means, std_params=psi0_stds, coeffs=None)
    LV = TrainableParameters.random_init(
        shape=(1, n_basis, n_basis), mean=0.0, std=0.1
    ).to(device=device, dtype=dtype)
    init_model = QuadraticFF.from_rff(tran_model, LV, psi0).to(device)

    print("Training initial SOS belief")
    initial_trainer = Trainer(init_model, device=device)
    initial_trainer.add_dataset(x0_loader, "initial")
    initial_trainer.add_optimizer(
        torch.optim.Adam(param_group_iter((LV,)), lr=init_params["lr_coeffs"]),
        "coeffs",
        n_iterations=init_params["coeff_iterations"],
    )
    initial_trainer.add_optimizer(
        torch.optim.Adam(
            param_group_iter((psi0_means, psi0_stds)), lr=init_params["lr_basis"]
        ),
        "psi0_basis",
        n_iterations=init_params["basis_iterations"],
    )
    initial_trainer.add_loss_function(loss.mle_loss, "mle", "initial")
    init_model, best_loss_init, training_time_init = initial_trainer.train(
        epochs=init_params["epochs"], use_best="mle", verbose=True
    )
    if init_model is None:
        raise SystemExit("No stable SOS initial-density checkpoint was found.")

    print(
        f"Transition model loss: {best_loss_tran:.4f}, "
        f"training time: {training_time_tran:.2f} seconds"
    )
    print(
        f"Initial model loss: {best_loss_init:.4f}, "
        f"training time: {training_time_init:.2f} seconds"
    )

    output_dir = Path("figures/qrff/vdp")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_x_range = (
        float(problem.plot_bounds_low[0]),
        float(problem.plot_bounds_high[0]),
    )
    plot_y_range = (
        float(problem.plot_bounds_low[1]),
        float(problem.plot_bounds_high[1]),
    )

    tran_model = tran_model.to(device).eval()
    init_model = init_model.to(device).eval()

    phi_out = output_dir / "basis_phi_2d.png"
    _plot_2d_gaussian_basis_grid(
        tran_model.phi,
        phi_out,
        x_range=plot_x_range,
        y_range=plot_y_range,
        title="VDP QRFF: 2D phi basis by index",
    )
    print(f"Saved 2D phi grid to {phi_out}")

    psi_out = output_dir / "basis_psi_2d.png"
    _plot_2d_gaussian_basis_grid(
        tran_model.psi,
        psi_out,
        x_range=plot_x_range,
        y_range=plot_y_range,
        title="VDP QRFF: 2D psi basis by index",
    )
    print(f"Saved 2D psi grid to {psi_out}")

    psi0_out = output_dir / "basis_psi0_2d.png"
    _plot_2d_gaussian_basis_grid(
        init_model.psi,
        psi0_out,
        x_range=plot_x_range,
        y_range=plot_y_range,
        title="VDP QRFF: 2D psi0 basis by index",
    )
    print(f"Saved 2D psi0 grid to {psi0_out}")

    cond_slice_out = output_dir / "conditional_slices_model_vs_data.png"
    _plot_conditional_slices_model_vs_data(
        tran_model,
        x_k,
        x_kp1,
        cond_slice_out,
        x_range=plot_x_range,
        y_range=plot_y_range,
        n_random_points=10,
        n_grid=120,
        min_points_per_slice=20,
        bin_width=0.4,
        random_seed=0,
        title=(
            "VDP QRFF: fixed x1, varied x2 conditional bins — "
            "empirical vs model vs KDE"
        ),
    )
    print(f"Saved conditional slice comparison to {cond_slice_out}")

    beliefs = [init_model]
    for _ in range(n_timesteps_prop):
        beliefs.append(tran_model.propagate(beliefs[-1]))
    beliefs = [b.to(device).eval() for b in beliefs]

    ll_per_step = []
    with torch.no_grad():
        for t, (belief, samples) in enumerate(zip(beliefs, traj_data)):
            ll = avg_log_likelihood(belief, samples.to(device=device, dtype=dtype))
            print(f"Average SOS log-likelihood t={t}: {float(ll):.6f}")
            ll_per_step.append(float(ll.detach().cpu()))

    n_plot = min(len(beliefs), 8)
    fig, axes = plt.subplots(2, n_plot, figsize=(3.2 * n_plot, 6.5), squeeze=False)
    fig.suptitle("VDP QRFF: beliefs at each time step")
    for i in range(n_plot):
        plot_belief(
            axes[1, i],
            beliefs[i],
            x_range=plot_x_range,
            y_range=plot_y_range,
        )
        points = traj_data[i].detach().cpu()
        axes[0, i].scatter(points[:, 0], points[:, 1], s=1)
        axes[0, i].set_aspect("equal")
        axes[0, i].set_xlim(*plot_x_range)
        axes[0, i].set_ylim(*plot_y_range)
        axes[0, i].set_title(f"t = {i}")

    beliefs_out = output_dir / "beliefs.png"
    fig.tight_layout()
    fig.savefig(beliefs_out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved beliefs to {beliefs_out}")

    ll_path = output_dir / "avg_ll.png"
    plt.figure(figsize=(8, 4))
    plt.plot(ll_per_step, marker="o")
    for t, val in enumerate(ll_per_step):
        plt.annotate(
            f"{val:.3f}",
            (t, val),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    plt.xlabel("timestep")
    plt.ylabel("avg log-likelihood per timestep")
    plt.title(
        "van_der_pol - QRFF avg log-likelihood\n"
        f"tran loss={best_loss_tran:.4f}, init loss={best_loss_init:.4f}\n"
        f"n_basis={n_basis}"
    )
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(ll_path, dpi=200)
    plt.close()
    print(f"Saved plot to {ll_path}")
    print(f"Saved diagnostics in {output_dir}")
