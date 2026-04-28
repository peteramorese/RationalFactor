from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.tools.propagate as propagate
from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.composite_model import CompositeConditionalModel, CompositeDensityModel
from rational_factor.models.domain_transformation import MaskedRQSNFTF, IdentityTF, VolumePreservingNFTF, StackedTF
from rational_factor.models.density_model import LogisticSigmoid
from rational_factor.models.factor_forms import LinearFF, LinearRFF, LinearForm
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood
from rational_factor.tools.misc import data_bounds, train_test_split
from rational_factor.tools.visualization import plot_belief
from rational_factor.models.kde import GaussianKDE


def _plot_conditional_slices_model_vs_binned_data(
    tran_model: CompositeConditionalModel,
    x_k_data: torch.Tensor,
    x_kp1_data: torch.Tensor,
    out_path: Path,
    *,
    n_random_points: int = 10,
    n_grid: int = 120,
    min_points_per_slice: int = 40,
    bin_width: float = 0.4,
    random_seed: int = 0,
    title: str = "",
) -> None:
    if x_k_data.shape[1] != 2 or x_kp1_data.shape[1] != 2:
        raise ValueError("_plot_conditional_slices_model_vs_binned_data expects 2D state data.")

    param = next(iter(tran_model.parameters()), None)
    buffer = next(iter(tran_model.buffers()), None)
    dev = param.device if param is not None else (buffer.device if buffer is not None else torch.device("cpu"))
    dt = param.dtype if param is not None else (buffer.dtype if buffer is not None else torch.float32)

    xk_cpu = x_k_data.detach().cpu()
    xkp1_cpu = x_kp1_data.detach().cpu()
    lo = xkp1_cpu.min(dim=0).values
    hi = xkp1_cpu.max(dim=0).values
    pad = 0.05 * (hi - lo).clamp_min(1e-6)
    lo = lo - pad
    hi = hi + pad

    x_lin = torch.linspace(float(lo[0]), float(hi[0]), n_grid)
    y_lin = torch.linspace(float(lo[1]), float(hi[1]), n_grid)
    X, Y = torch.meshgrid(x_lin, y_lin, indexing="xy")
    xp_grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).to(device=dev, dtype=dt)

    n_data = xk_cpu.shape[0]
    if n_data == 0:
        raise ValueError("x_k_data is empty.")
    n_random_points = max(1, min(n_random_points, n_data))
    rng = torch.Generator(device="cpu")
    rng.manual_seed(random_seed)
    rand_idx = torch.randperm(n_data, generator=rng)[:n_random_points]
    centers = xk_cpu[rand_idx]
    half_width = torch.full((2,), 0.5 * float(bin_width), dtype=xk_cpu.dtype)

    fig, axes = plt.subplots(n_random_points, 1, figsize=(7.0, 4.0 * n_random_points), squeeze=False)
    cmap = "viridis"

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

            center_dev = center.to(device=dev, dtype=dt)
            cond = center_dev.unsqueeze(0).expand(xp_grid.shape[0], -1)
            model_pdf = tran_model.log_density(xp_grid, conditioner=cond).exp().reshape(n_grid, n_grid).detach().cpu().numpy()

            ax_model = axes[row, 0]
            cf = ax_model.contourf(X.numpy(), Y.numpy(), model_pdf, levels=40, cmap=cmap)
            fig.colorbar(cf, ax=ax_model, fraction=0.046, pad=0.04)
            ax_model.set_title(
                "model p(x'|x) with empirical x' overlay\n"
                f"x=({float(center[0]):.3f}, {float(center[1]):.3f}), "
                f"bin_w=({float(2 * half_width[0]):.3f}, {float(2 * half_width[1]):.3f}), "
                f"n={xp_slice.shape[0]}"
            )
            ax_model.set_xlim(float(lo[0]), float(hi[0]))
            ax_model.set_ylim(float(lo[1]), float(hi[1]))
            ax_model.set_aspect("equal")
            ax_model.set_ylabel("x'_2")
            if xp_slice.shape[0] >= min_points_per_slice:
                ax_model.scatter(
                    xp_slice[:, 0].numpy(),
                    xp_slice[:, 1].numpy(),
                    s=6,
                    alpha=0.9,
                    c="orange",
                    edgecolors="none",
                    rasterized=True,
                )
            elif xp_slice.shape[0] > 0:
                ax_model.scatter(
                    xp_slice[:, 0].numpy(),
                    xp_slice[:, 1].numpy(),
                    s=10,
                    alpha=0.9,
                    c="orange",
                    edgecolors="none",
                    rasterized=True,
                )
                ax_model.text(
                    0.02,
                    0.98,
                    f"low-sample slice (n={xp_slice.shape[0]})",
                    ha="left",
                    va="top",
                    transform=ax_model.transAxes,
                    color="white",
                    fontsize=9,
                    bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none"},
                )

    axes[-1, 0].set_xlabel("x'_1")
    if title:
        fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_domain_grid_morph(
    decorrupter_trained: torch.nn.Module,
    mover_trained: torch.nn.Module,
    x_data: torch.Tensor,
    out_path: Path,
    *,
    n_grid_lines: int = 11,
    n_points_per_line: int = 250,
    quantile_pad: float = 0.02,
    title: str = "",
) -> None:
    if x_data.shape[1] != 2:
        raise ValueError("_plot_domain_grid_morph expects 2D state data.")

    x_cpu = x_data.detach().cpu()
    q_lo = x_cpu.quantile(quantile_pad, dim=0)
    q_hi = x_cpu.quantile(1.0 - quantile_pad, dim=0)
    lo = q_lo.numpy()
    hi = q_hi.numpy()

    x_vals = np.linspace(lo[0], hi[0], n_points_per_line)
    y_vals = np.linspace(lo[1], hi[1], n_points_per_line)
    gx = np.linspace(lo[0], hi[0], n_grid_lines)
    gy = np.linspace(lo[1], hi[1], n_grid_lines)

    fig, axes = plt.subplots(3, 2, figsize=(12, 15), squeeze=False)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_grid_lines))

    # Build transformed datasets for visual context and for row-2 rectangular-grid bounds.
    with torch.no_grad():
        decorrupter_trained.eval()
        mover_trained.eval()
        y_data, _ = decorrupter_trained(x_cpu.to(dtype=torch.float32))
        z_data, _ = mover_trained(y_data)
        y_cpu = y_data.detach().cpu()
        z_cpu = z_data.detach().cpu()

    y_lo = y_cpu.quantile(quantile_pad, dim=0).numpy()
    y_hi = y_cpu.quantile(1.0 - quantile_pad, dim=0).numpy()
    yx_vals = np.linspace(y_lo[0], y_hi[0], n_points_per_line)
    yy_vals = np.linspace(y_lo[1], y_hi[1], n_points_per_line)
    ygx = np.linspace(y_lo[0], y_hi[0], n_grid_lines)
    ygy = np.linspace(y_lo[1], y_hi[1], n_grid_lines)

    # Row 1: original rectangular grid -> decorrupter only.
    ax_r1_l = axes[0, 0]
    ax_r1_r = axes[0, 1]
    ax_r1_l.set_title("Original state space: rectangular grid")
    ax_r1_r.set_title("Decorrupter effect only")
    ax_r1_l.scatter(x_cpu[:, 0].numpy(), x_cpu[:, 1].numpy(), s=1, alpha=0.08, c="gray", rasterized=True)
    ax_r1_r.scatter(y_cpu[:, 0].numpy(), y_cpu[:, 1].numpy(), s=1, alpha=0.08, c="gray", rasterized=True)

    # Row 2: fresh rectangular grid in decorrupter space -> mover only.
    ax_r2_l = axes[1, 0]
    ax_r2_r = axes[1, 1]
    ax_r2_l.set_title("Decorrupter space: new rectangular grid")
    ax_r2_r.set_title("Mover effect only (from new grid)")
    ax_r2_l.scatter(y_cpu[:, 0].numpy(), y_cpu[:, 1].numpy(), s=1, alpha=0.08, c="gray", rasterized=True)
    ax_r2_r.scatter(z_cpu[:, 0].numpy(), z_cpu[:, 1].numpy(), s=1, alpha=0.08, c="gray", rasterized=True)

    # Row 3: original rectangular grid -> full composition decorrupter then mover.
    ax_r3_l = axes[2, 0]
    ax_r3_r = axes[2, 1]
    ax_r3_l.set_title("Original state space: rectangular grid")
    ax_r3_r.set_title("Full map: decorrupter then mover")
    ax_r3_l.scatter(x_cpu[:, 0].numpy(), x_cpu[:, 1].numpy(), s=1, alpha=0.08, c="gray", rasterized=True)
    ax_r3_r.scatter(z_cpu[:, 0].numpy(), z_cpu[:, 1].numpy(), s=1, alpha=0.08, c="gray", rasterized=True)

    with torch.no_grad():
        decorrupter_trained.eval()
        mover_trained.eval()

        # Vertical + horizontal lines from original x-grid for rows 1 and 3.
        for i, xv in enumerate(gx):
            line_xy = np.stack([np.full_like(y_vals, xv), y_vals], axis=1)
            line_t = torch.from_numpy(line_xy).to(dtype=torch.float32)
            y_line, _ = decorrupter_trained(line_t)
            z_line, _ = mover_trained(y_line)
            y_np = y_line.detach().cpu().numpy()
            z_np = z_line.detach().cpu().numpy()

            ax_r1_l.plot(line_xy[:, 0], line_xy[:, 1], color=colors[i], lw=1.4)
            ax_r1_r.plot(y_np[:, 0], y_np[:, 1], color=colors[i], lw=1.4)
            ax_r3_l.plot(line_xy[:, 0], line_xy[:, 1], color=colors[i], lw=1.4)
            ax_r3_r.plot(z_np[:, 0], z_np[:, 1], color=colors[i], lw=1.4)

        for i, yv in enumerate(gy):
            line_xy = np.stack([x_vals, np.full_like(x_vals, yv)], axis=1)
            line_t = torch.from_numpy(line_xy).to(dtype=torch.float32)
            y_line, _ = decorrupter_trained(line_t)
            z_line, _ = mover_trained(y_line)
            y_np = y_line.detach().cpu().numpy()
            z_np = z_line.detach().cpu().numpy()

            ax_r1_l.plot(line_xy[:, 0], line_xy[:, 1], color=colors[i], lw=1.4, alpha=0.8)
            ax_r1_r.plot(y_np[:, 0], y_np[:, 1], color=colors[i], lw=1.4, alpha=0.8)
            ax_r3_l.plot(line_xy[:, 0], line_xy[:, 1], color=colors[i], lw=1.4, alpha=0.8)
            ax_r3_r.plot(z_np[:, 0], z_np[:, 1], color=colors[i], lw=1.4, alpha=0.8)

        # Vertical + horizontal lines from new y-grid for row 2.
        for i, yx in enumerate(ygx):
            line_y = np.stack([np.full_like(yy_vals, yx), yy_vals], axis=1)
            line_t = torch.from_numpy(line_y).to(dtype=torch.float32)
            z_line, _ = mover_trained(line_t)
            z_np = z_line.detach().cpu().numpy()
            ax_r2_l.plot(line_y[:, 0], line_y[:, 1], color=colors[i], lw=1.4)
            ax_r2_r.plot(z_np[:, 0], z_np[:, 1], color=colors[i], lw=1.4)

        for i, yy in enumerate(ygy):
            line_y = np.stack([yx_vals, np.full_like(yx_vals, yy)], axis=1)
            line_t = torch.from_numpy(line_y).to(dtype=torch.float32)
            z_line, _ = mover_trained(line_t)
            z_np = z_line.detach().cpu().numpy()
            ax_r2_l.plot(line_y[:, 0], line_y[:, 1], color=colors[i], lw=1.4, alpha=0.8)
            ax_r2_r.plot(z_np[:, 0], z_np[:, 1], color=colors[i], lw=1.4, alpha=0.8)

    # Axes formatting.
    ax_r1_l.set_xlabel("x_1")
    ax_r1_l.set_ylabel("x_2")
    ax_r1_r.set_xlabel("y_1")
    ax_r1_r.set_ylabel("y_2")
    ax_r2_l.set_xlabel("y_1")
    ax_r2_l.set_ylabel("y_2")
    ax_r2_r.set_xlabel("z_1")
    ax_r2_r.set_ylabel("z_2")
    ax_r3_l.set_xlabel("x_1")
    ax_r3_l.set_ylabel("x_2")
    ax_r3_r.set_xlabel("z_1")
    ax_r3_r.set_ylabel("z_2")

    ax_r1_l.set_xlim(float(lo[0]), float(hi[0]))
    ax_r1_l.set_ylim(float(lo[1]), float(hi[1]))
    ax_r3_l.set_xlim(float(lo[0]), float(hi[0]))
    ax_r3_l.set_ylim(float(lo[1]), float(hi[1]))
    ax_r1_r.set_xlim(float(y_lo[0]), float(y_hi[0]))
    ax_r1_r.set_ylim(float(y_lo[1]), float(y_hi[1]))
    ax_r2_l.set_xlim(float(y_lo[0]), float(y_hi[0]))
    ax_r2_l.set_ylim(float(y_lo[1]), float(y_hi[1]))

    for ax in (ax_r1_l, ax_r1_r, ax_r2_l, ax_r2_r, ax_r3_l, ax_r3_r):
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    problem = FULLY_OBSERVABLE_PROBLEMS["van_der_pol"]

    ######## USER CONFIG ########
    use_gpu = torch.cuda.is_available()
    batch_size = 1024

    decorrupter_params = {
        "epochs": 20,
        "lr": 1e-3,
    }

    mover_params = {
        "n_epochs_per_group": [1],
        "iterations": 5,
        "lr_basis": 5e-2,
        "lr_weights": 1e-1,
        "lr_dtf": 1e-3,
    }

    init_params = {
        "n_epochs_per_group": [15, 5],
        "iterations": 100,
        "lr_basis": 5e-2,
        "lr_weights": 1e-1,
    }

    ls_temp = 0.1
    n_basis = 50
    mover_iterations = 5
    reg_covar_joint = 1e-2
    decorrupter_weight_decay = 1e-5
    mover_weight_decay = 1e-5

    ######## SETUP ########
    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using device: ", device)

    system = problem.system
    x0_data = problem.train_initial_state_data()
    x_k_data, x_kp1_data = problem.train_state_transition_data()
    test_traj_data = problem.test_data()

    x_dataloader = DataLoader(TensorDataset(x_k_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    x0_dataloader = DataLoader(TensorDataset(x0_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)


    ######## TRAIN DECORRUPTER ########
    loc, scale = data_bounds(x_k_data, mode="center_lengths")
    loc = loc.to(device)
    scale = scale.to(device)
    base_distribution = LogisticSigmoid(system.dim(), temperature=ls_temp, loc=loc, scale=scale)
    decorrupter = MaskedRQSNFTF(system.dim(), trainable=True, hidden_features=128, n_layers=5).to(device)
    decorrupter_density = CompositeDensityModel([decorrupter], base_distribution).to(device)

    print("Training NF decorrupter")
    optimizer = torch.optim.Adam(decorrupter.parameters(), lr=decorrupter_params["lr"], weight_decay=decorrupter_weight_decay)
    dtf_concentration_loss_xk = lambda composite_model, x: 1.0 * loss.dtf_data_concentration_loss(composite_model.domain_tfs[0], x, concentration_point=loc, radius=scale)

    decorrupter_density, best_loss, training_time = train.train(
        decorrupter_density,
        x_dataloader,
        {"mle": loss.mle_loss, "dtf_conc": dtf_concentration_loss_xk},
        optimizer,
        epochs=decorrupter_params["epochs"],
        verbose=True,
        use_best="mle",
        clip_grad_norm=5.0,
        restore_loss_threshold=50.0,
    )
    print("Done.\n")

    decorrupter_trained = MaskedRQSNFTF.copy_from_trainable(decorrupter).to(device)

    x_k_data = x_k_data.to(device)
    x_kp1_data = x_kp1_data.to(device)
    x0_data = x0_data.to(device)
    y_k_data, _ = decorrupter_trained(x_k_data)
    y_kp1_data, _ = decorrupter_trained(x_kp1_data)
    y0_data, _ = decorrupter_trained(x0_data)
    
    ######## TRAIN MOVER ########
    print("Training mover")
    mover = VolumePreservingNFTF(system.dim(), trainable=True, hidden_features=256, n_layers=8).to(device)
    base_density_basis =  GaussianBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 20.0], device=device), variance=30.0, min_std=1e-4).to(device)
    
    # Initialize base density to LF GMM
    y_joint_data = torch.cat([y_k_data, y_kp1_data], dim=1)
    #gmm_lf = train.fit_gaussian_lf_em(y_joint_data.to(torch.device("cpu")), n_components=n_basis, reg_covar=1e-3, max_iter=100)
    #basis = GaussianBasis(uparams_init=gmm_lf.basis.uparams_init)
    #mover_base_density = LinearForm()
    mover_joint = StackedTF([mover, mover])

    # Train on the decorrupted x' marginal data
    z_joint_data = y_joint_data
    for i in range(mover_iterations):
        print("Fitting GMM LF...")
        gmm_lf = train.fit_gaussian_lf_em(z_joint_data.to(torch.device("cpu")), n_components=n_basis, reg_covar=reg_covar_joint, max_iter=100)
        mover_density = CompositeDensityModel([mover_joint], gmm_lf).to(device)
        y_joint_dataloader = DataLoader(TensorDataset(y_joint_data.to(torch.device("cpu"))), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

        #optimizers = {
        #    "basis_and_dtf": torch.optim.Adam([
        #        {"params": mover_density.conditional_density_model.basis_params(), "lr": mover_params["lr_basis"]},
        #        {"params": mover_density.domain_tfs.parameters(), "lr": mover_params["lr_dtf"]},
        #    ]),
        #    "weights": torch.optim.Adam(mover_density.density_model.weight_params(), lr=mover_params["lr_weights"]),
        #}
        optimizers = {
            "dtf": torch.optim.Adam([
                #{"params": mover_density.conditional_density_model.basis_params(), "lr": mover_params["lr_basis"]},
                {"params": mover_density.domain_tfs[0].parameters(), "lr": mover_params["lr_dtf"], "weight_decay": mover_weight_decay},
            ]),
            #"weights": torch.optim.Adam(mover_density.density_model.weight_params(), lr=mover_params["lr_weights"]),
        }
        mover_density, best_loss, training_time = train.train_iterate(
            mover_density,
            y_joint_dataloader,
            {"mle": loss.mle_loss},
            optimizers,
            epochs_per_group=mover_params["n_epochs_per_group"],
            iterations=mover_params["iterations"],
            verbose=True,
            use_best="mle",
            clip_grad_norm=5.0,
            restore_loss_threshold=50.0,
        )
        print("Done.\n")

        z_joint_data, _ = mover_joint(y_joint_data)

    mover_trained = VolumePreservingNFTF.copy_from_trainable(mover).to(device)

    weights = gmm_lf.get_w()
    z_marginal = gmm_lf.basis.marginal(marginal_dims=range(system.dim()))
    z_means, z_stds = z_marginal.means_stds()
    z_params = torch.stack([z_means, z_stds], dim=-1)

    zp_marginal = gmm_lf.basis.marginal(marginal_dims=range(system.dim(), 2 * system.dim()))
    zp_means, zp_stds = zp_marginal.means_stds()
    zp_params = torch.stack([zp_means, zp_stds], dim=-1)
    

    phi_basis = GaussianBasis(fixed_params=z_params).to(device)
    psi_basis = GaussianBasis(fixed_params=zp_params).to(device)
    psi0_basis = GaussianBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 20.0], device=device), variance=30.0, min_std=1e-4).to(device)

    lrff = LinearRFF(phi_basis, psi_basis, a_fixed=weights.to(device)).to(device)
    ff = LinearFF(lrff.get_a(), phi_basis, psi0_basis).to(device)

    z0_data, _ = mover_trained(y0_data)
    z0_dataloader = DataLoader(TensorDataset(z0_data.to(torch.device("cpu"))), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    optimizers = {
        "basis": torch.optim.Adam(ff.basis_params(), lr=init_params["lr_basis"]),
        "weights": torch.optim.Adam(ff.weight_params(), lr=init_params["lr_weights"]),
    }

    ff, best_loss_init, training_time_init = train.train_iterate(
        ff,
        z0_dataloader,
        {"mle": loss.mle_loss},
        optimizers,
        epochs_per_group=init_params["n_epochs_per_group"],
        iterations=init_params["iterations"],
        verbose=True,
        use_best="mle",
    )
    print("Done.\n")
    print(f"Initial model loss: {best_loss_init:.6f}, training time: {training_time_init:.2f}s")

    ######## ANALYSIS ########
    analysis_device = torch.device("cpu")
    ff = ff.to(analysis_device).eval()
    lrff = lrff.to(analysis_device).eval()
    decorrupter_trained = decorrupter_trained.to(analysis_device).eval()
    mover_trained = mover_trained.to(analysis_device).eval()

    base_belief_seq = propagate.propagate(ff, lrff, n_steps=problem.n_timesteps)
    belief_seq = [CompositeDensityModel([decorrupter_trained, mover_trained], belief).to(analysis_device).eval() for belief in base_belief_seq]

    tran_model = CompositeConditionalModel([decorrupter_trained, mover_trained], lrff).to(analysis_device).eval()

    cond_slice_out_path = Path("figures") / "vdp_lcmp__conditional_slices_model_vs_binned_data.png"
    _plot_conditional_slices_model_vs_binned_data(
        tran_model,
        x_k_data.detach().cpu(),
        x_kp1_data.detach().cpu(),
        cond_slice_out_path,
        n_random_points=20,
        n_grid=140,
        min_points_per_slice=25,
        bin_width=0.30,
        random_seed=0,
        title="van_der_pol lcmp: tran_model conditional slices vs binned empirical data",
    )
    print(f"Saved conditional slice comparison to {cond_slice_out_path}")

    grid_morph_out_path = Path("figures") / "vdp_lcmp__domain_grid_morph.png"
    _plot_domain_grid_morph(
        decorrupter_trained,
        mover_trained,
        x_k_data.detach().cpu(),
        grid_morph_out_path,
        n_grid_lines=11,
        n_points_per_line=280,
        quantile_pad=0.01,
        title="van_der_pol lcmp: state grid morphed by decorrupter ∘ mover",
    )
    print(f"Saved domain transformation grid morph to {grid_morph_out_path}")

    fig, axes = plt.subplots(2, problem.n_timesteps, figsize=(20, 10))
    fig.suptitle("Beliefs at each time step")
    for i in range(problem.n_timesteps):
        data_i = test_traj_data[i].to(analysis_device)
        ll = avg_log_likelihood(belief_seq[i], data_i)
        print(f"Log likelihood at time {i}: {ll:.4f}")
        plot_belief(axes[1, i], belief_seq[i], x_range=(problem.plot_bounds_low[0], problem.plot_bounds_high[0]), y_range=(problem.plot_bounds_low[1], problem.plot_bounds_high[1]))
        axes[0, i].scatter(test_traj_data[i][:, 0], test_traj_data[i][:, 1], s=1)
        axes[0, i].set_aspect("equal")
        axes[0, i].set_xlim(problem.plot_bounds_low[0], problem.plot_bounds_high[0])
        axes[0, i].set_ylim(problem.plot_bounds_low[1], problem.plot_bounds_high[1])
    


    plt.savefig("figures/vdp_lcmp_beliefs.png", dpi=1000)
    print("Saved beliefs to figures/vdp_lcmp_beliefs.png")

if __name__ == "__main__":
    main()

