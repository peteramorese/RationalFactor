from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.tools.propagate as propagate
from rational_factor.models.basis_functions import GaussianKernelBasis
from rational_factor.models.composite_model import CompositeConditionalModel, CompositeDensityModel
from rational_factor.models.domain_transformation import MaskedRQSNFTF, IdentityTF
from rational_factor.models.density_model import LogisticSigmoid
from rational_factor.models.factor_forms import LinearFF, LinearRFF
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood
from rational_factor.tools.misc import data_bounds, train_test_split
from rational_factor.tools.visualization import plot_belief
from rational_factor.models.kde import GaussianKDE


def _plot_conditional_slices_lrff_vs_data(
    tran_model: CompositeConditionalModel,
    x_k_data: torch.Tensor,
    x_kp1_data: torch.Tensor,
    out_path: Path,
    *,
    n_random_points: int = 10,
    n_grid: int = 120,
    n_bins: int = 50,
    min_points_per_slice: int = 40,
    bin_width: float = 1.0,
    random_seed: int = 0,
    title: str = "",
) -> None:
    """Compare model p(x'|x_i) to empirical p(x'|x in bin around x_i) for 2D Van der Pol."""
    if x_k_data.shape[1] != 2 or x_kp1_data.shape[1] != 2:
        raise ValueError("_plot_conditional_slices_lrff_vs_data expects 2D state data")

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

    # Randomly choose conditioner centers from observed x_k samples.
    n_data = xk_cpu.shape[0]
    if n_data == 0:
        raise ValueError("x_k_data is empty")
    n_random_points = max(1, min(n_random_points, n_data))
    rng = torch.Generator(device="cpu")
    rng.manual_seed(random_seed)
    rand_idx = torch.randperm(n_data, generator=rng)[:n_random_points]
    centers = xk_cpu[rand_idx]

    # Axis-aligned bins around each random center, using a fixed width.
    half_width = torch.full((2,), 0.5 * float(bin_width), dtype=xk_cpu.dtype)

    slice_specs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
    for row_id, center in enumerate(centers):
        lo_box = center - half_width
        hi_box = center + half_width
        slice_specs.append((center, lo_box, hi_box, row_id))

    n_rows = len(slice_specs)
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 3.2 * n_rows), squeeze=False)
    cmap = "viridis"
    eps = torch.finfo(dt).eps

    # "Traditional KDE" pane (as before): conditional estimate using data bandwidths.
    bw_x = GaussianKernelBasis.ss_bandwidth(xk_cpu.to(dtype=dt)).to(device=dev, dtype=dt).clamp_min(eps)
    bw_xp = GaussianKernelBasis.ss_bandwidth(xkp1_cpu.to(dtype=dt)).to(device=dev, dtype=dt).clamp_min(eps)
    xk_dev = xk_cpu.to(device=dev, dtype=dt)
    xkp1_dev = xkp1_cpu.to(device=dev, dtype=dt)

    with torch.no_grad():
        tran_model.eval()
        lrff = tran_model.conditional_density_model
        # Sanity check: basis centers should match transformed training pairs used at construction.
        xk_check = x_k_data.to(device=dev, dtype=dt)
        xkp1_check = x_kp1_data.to(device=dev, dtype=dt)
        for tf in tran_model.domain_tfs:
            xk_check, _ = tf(xk_check)
            xkp1_check, _ = tf(xkp1_check)
        phi_centers = lrff.phi_basis.kernel_centers().to(device=dev, dtype=dt)
        psi_centers = lrff.psi_basis.kernel_centers().to(device=dev, dtype=dt)
        max_diff_phi = (phi_centers - xk_check).abs().max().item()
        max_diff_psi = (psi_centers - xkp1_check).abs().max().item()
        print(
            "Kernel-center sanity:",
            f"max|phi_centers - tf(x_k)|={max_diff_phi:.3e},",
            f"max|psi_centers - tf(x_kp1)|={max_diff_psi:.3e}",
        )
        for row, (center, lo_box, hi_box, row_id) in enumerate(slice_specs):
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
                ax_emp.text(0.5, 0.5, f"Too few samples\nn={xp_slice.shape[0]}", ha="center", va="center", transform=ax_emp.transAxes)
            ax_emp.set_title(
                "empirical samples x' | x in bin\n"
                f"sample #{row_id}, n={xp_slice.shape[0]}"
            )
            ax_emp.set_xlim(float(lo[0]), float(hi[0]))
            ax_emp.set_ylim(float(lo[1]), float(hi[1]))
            ax_emp.set_aspect("equal")
            ax_emp.set_ylabel("x'_2")

            center_dev = center.to(device=dev, dtype=dt)
            cond = center_dev.unsqueeze(0).expand(xp_grid.shape[0], -1)
            model_pdf = tran_model.log_density(xp_grid, conditioner=cond).exp().reshape(n_grid, n_grid).detach().cpu().numpy()
            ax_model = axes[row, 1]
            cf = ax_model.contourf(
                X.numpy(),
                Y.numpy(),
                model_pdf,
                levels=40,
                cmap=cmap,
            )
            fig.colorbar(cf, ax=ax_model, fraction=0.046, pad=0.04)
            ax_model.set_title(
                "model p_lrff(x'|x_i)\n"
                f"x_i=({float(center[0]):.3f}, {float(center[1]):.3f}), "
                f"bin_w=({float(2*half_width[0]):.3f}, {float(2*half_width[1]):.3f})"
            )
            ax_model.set_xlim(float(lo[0]), float(hi[0]))
            ax_model.set_ylim(float(lo[1]), float(hi[1]))
            ax_model.set_aspect("equal")
            ax_model.set_ylabel("x'_2")
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

            # Basis-ratio conditional:
            # p_basis(x'|x_i) = sum_j phi_j(x_i) * psi_j(x') / sum_j phi_j(x_i)
            phi_xi = lrff.phi_basis(center_dev.unsqueeze(0)).squeeze(0)  # (n_basis,)
            psi_xp = lrff.psi_basis(xp_grid)  # (n_grid*n_grid, n_basis)
            denom = phi_xi.sum().clamp_min(torch.finfo(dt).eps)
            basis_pdf = (psi_xp * phi_xi.unsqueeze(0)).sum(dim=1) / denom
            basis_pdf = basis_pdf.reshape(n_grid, n_grid).detach().cpu().numpy()

            ax_basis = axes[row, 2]
            cf_basis = ax_basis.contourf(
                X.numpy(),
                Y.numpy(),
                basis_pdf,
                levels=40,
                cmap=cmap,
            )
            fig.colorbar(cf_basis, ax=ax_basis, fraction=0.046, pad=0.04)
            ax_basis.set_title(
                "basis-ratio p_basis(x'|x_i)\n"
                "sum_j phi_j(x_i)psi_j(x') / sum_j phi_j(x_i)"
            )
            ax_basis.set_xlim(float(lo[0]), float(hi[0]))
            ax_basis.set_ylim(float(lo[1]), float(hi[1]))
            ax_basis.set_aspect("equal")
            ax_basis.set_ylabel("x'_2")
            if xp_slice.shape[0] > 0:
                ax_basis.scatter(
                    xp_slice[:, 0].numpy(),
                    xp_slice[:, 1].numpy(),
                    s=5,
                    alpha=1.0,
                    c="orange",
                    edgecolors="none",
                    rasterized=True,
                )

            # Traditional KDE conditional slice at fixed x_i (original behavior).
            diff_x = xk_dev - center_dev.unsqueeze(0)  # (n_data, 2)
            sq_x = diff_x.square().sum(dim=1)
            w_x = torch.exp(-0.5 * sq_x / (bw_x * bw_x))  # (n_data,)
            w_sum = w_x.sum().clamp_min(eps)

            kde_vals = []
            block = 1024
            for start in range(0, xp_grid.shape[0], block):
                end = min(start + block, xp_grid.shape[0])
                xp_blk = xp_grid[start:end]  # (b, 2)
                diff_xp = xp_blk[:, None, :] - xkp1_dev[None, :, :]  # (b, n_data, 2)
                sq_xp = diff_xp.square().sum(dim=2)  # (b, n_data)
                k_xp = torch.exp(-0.5 * sq_xp / (bw_xp * bw_xp))
                kde_vals.append((k_xp * w_x.unsqueeze(0)).sum(dim=1) / w_sum)
            kde_pdf = torch.cat(kde_vals, dim=0).reshape(n_grid, n_grid).detach().cpu().numpy()

            ax_kde = axes[row, 3]
            cf_kde = ax_kde.contourf(
                X.numpy(),
                Y.numpy(),
                kde_pdf,
                levels=40,
                cmap=cmap,
            )
            fig.colorbar(cf_kde, ax=ax_kde, fraction=0.046, pad=0.04)
            ax_kde.set_title("traditional KDE p(x'|x_i)")
            ax_kde.set_xlim(float(lo[0]), float(hi[0]))
            ax_kde.set_ylim(float(lo[1]), float(hi[1]))
            ax_kde.set_aspect("equal")
            ax_kde.set_ylabel("x'_2")
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

    axes[-1, 0].set_xlabel("x'_1")
    axes[-1, 1].set_xlabel("x'_1")
    axes[-1, 2].set_xlabel("x'_1")
    axes[-1, 3].set_xlabel("x'_1")
    if title:
        fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_latent_marginal_hist_g_base(
    lrff: LinearRFF,
    base_distribution: LogisticSigmoid,
    hist_latent: torch.Tensor,
    x0_latent: torch.Tensor,
    out_path: Path,
    *,
    n_grid: int = 200,
    title: str = "",
) -> None:
    param = next(iter(lrff.parameters()), None)
    buffer = next(iter(lrff.buffers()), None)
    dev = param.device if param is not None else (buffer.device if buffer is not None else torch.device("cpu"))
    dt = param.dtype if param is not None else (buffer.dtype if buffer is not None else torch.float32)

    z_train = hist_latent.to(device=dev, dtype=dt)
    z0_train = x0_latent.to(device=dev, dtype=dt)
    state_dim = z_train.shape[1]
    phi_basis = lrff.phi_basis
    if not isinstance(phi_basis, GaussianKernelBasis):
        raise TypeError("_plot_latent_marginal_hist_g_base expects lrff.phi_basis to be GaussianKernelBasis")
    loc = base_distribution.loc.to(device=dev, dtype=dt)
    scale = base_distribution.scale.to(device=dev, dtype=dt)

    n_cols = 4
    n_rows = int(np.ceil(state_dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 2.8 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    with torch.no_grad():
        lrff.eval()
        a = lrff.get_a()
        for d in range(state_dim):
            ax = axes_flat[d]
            z_d_np = z_train[:, d].detach().cpu().numpy()
            z_lo = float(np.min(z_d_np))
            z_hi = float(np.max(z_d_np))
            if z_hi <= z_lo:
                z_hi = z_lo + 1e-6

            ax.hist(z_d_np, bins=50, density=True, alpha=0.35, color="C1")
            z0_d_np = z0_train[:, d].detach().cpu().numpy()
            ax.hist(z0_d_np, bins=50, density=True, alpha=0.25, color="C4")

            z_line = torch.linspace(z_lo, z_hi, n_grid, device=dev, dtype=dt)
            phi_d = phi_basis.marginal((d,))
            z_1d = z_line.unsqueeze(1)
            g_curve = (phi_d(z_1d) @ a).detach().cpu().numpy()
            z_axis = z_line.detach().cpu().numpy()

            base_1d = LogisticSigmoid(
                1,
                temperature=float(base_distribution.temperature),
                loc=loc[d : d + 1].detach(),
                scale=scale[d : d + 1].detach(),
            ).to(device=dev, dtype=dt)
            base_z = z_line.unsqueeze(1)
            base_curve = base_1d(base_z).squeeze(-1).detach().cpu().numpy()

            sort_idx = np.argsort(z_axis)
            ax.plot(z_axis[sort_idx], base_curve[sort_idx], color="C2", lw=1.8, ls="--")
            ax.plot(z_axis[sort_idx], g_curve[sort_idx], color="C0", lw=2.0)
            ax.set_xlim(z_lo, z_hi)
            ax.set_ylabel("density / g")
            ax.set_title(f"latent dim {d}")
            ax.grid(True, alpha=0.25)

        for j in range(state_dim, len(axes_flat)):
            axes_flat[j].set_visible(False)

    for c in range(n_cols):
        idx = (n_rows - 1) * n_cols + c
        if idx < state_dim:
            axes_flat[idx].set_xlabel("z (flow output)")
    if title:
        fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    problem = FULLY_OBSERVABLE_PROBLEMS["van_der_pol"]

    ######## USER CONFIG ########
    use_gpu = torch.cuda.is_available()
    batch_size = 1024

    dtf_params = {
        "epochs": 100,
        "lr": 1e-3,
    }
    ls_temp = 0.1

    init_params = {
        "n_epochs_per_group": [1],
        "iterations": 200,
        "lr_weights": 1e-1,
        "lr_kernel_bandwidth": 1e-3,
    }

    ######## SETUP ########
    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using device: ", device)

    system = problem.system
    x0_data = problem.train_initial_state_data()
    x_k_data, x_kp1_data = problem.train_state_transition_data()
    test_traj_data = problem.test_data()

    x_dataloader = DataLoader(TensorDataset(x_k_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(TensorDataset(x_kp1_data, x_k_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    x0_dataloader = DataLoader(TensorDataset(x0_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    loc, scale = data_bounds(x_k_data, mode="center_lengths")
    loc = loc.to(device)
    scale = scale.to(device)
    print("scale: ", scale)
    base_distribution = LogisticSigmoid(system.dim(), temperature=ls_temp, loc=loc, scale=scale)
    dtf = MaskedRQSNFTF(system.dim(), trainable=True, hidden_features=128, n_layers=5).to(device)
    decorrupter = CompositeDensityModel([dtf], base_distribution).to(device)

    ######## TRAIN LCM ########
    print("Training NF decorrupter")
    optimizer = torch.optim.Adam(decorrupter.parameters(), lr=dtf_params["lr"], weight_decay=1e-3)
    dtf_concentration_loss_xk = lambda composite_model, x: 1.0 * loss.dtf_data_concentration_loss(
        composite_model.domain_tfs[0], x, concentration_point=loc, radius=scale
    )
    decorrupter, best_loss, training_time = train.train(
        decorrupter,
        x_dataloader,
        {"mle": loss.mle_loss, "dtf_conc": dtf_concentration_loss_xk},
        optimizer,
        epochs=dtf_params["epochs"],
        verbose=True,
        use_best="mle",
        clip_grad_norm=5.0,
        restore_loss_threshold=50.0,
    )
    print("Done.\n")

    #dtf_trained = MaskedRQSNFTF.copy_from_trainable(dtf).to(device)
    #dtf = IdentityTF(system.dim()).to(device)

    x_k_data = x_k_data.to(device)
    x_kp1_data = x_kp1_data.to(device)
    x0_data = x0_data.to(device)
    x_k_transformed, _ = dtf(x_k_data)
    x_kp1_transformed, _ = dtf(x_kp1_data)
    x0_transformed, _ = dtf(x0_data)
    
    # Optimize the joint bandwidth
    joint_data = torch.cat([x_k_transformed, x_kp1_transformed], dim=1)
    joint_train_data, joint_val_data = train_test_split(joint_data, test_size=0.2)
    joint_kde = GaussianKDE(joint_train_data)
    best_bandwidth_joint, _ = joint_kde.fit_bandwidth_validation_mle(joint_val_data, epochs=200, threshold=0.5, lr=1e-1, min_step=1e-2, verbose=True)
    #best_bandwidth_joint = 0.5 * best_bandwidth_joint

    init_train_data, init_val_data = train_test_split(x0_transformed, test_size=0.2)
    init_kde = GaussianKDE(init_train_data)
    best_bandwidth_init, _ = init_kde.fit_bandwidth_validation_mle(init_val_data, epochs=200, threshold=0.5, lr=1e-1, min_step=1e-2, verbose=True)

    phi_basis = GaussianKernelBasis(x_k_transformed, kernel_bandwidth=best_bandwidth_joint, trainable=False)
    psi_basis = GaussianKernelBasis(x_kp1_transformed, kernel_bandwidth=best_bandwidth_joint, trainable=False)
    psi0_basis = GaussianKernelBasis(x0_transformed, kernel_bandwidth=best_bandwidth_init, trainable=True)

    lrff = LinearRFF(phi_basis, psi_basis, numerical_tolerance=problem.numerical_tolerance).to(device)
    tran_model = CompositeConditionalModel([dtf], lrff).to(device)

    ff = LinearFF.from_rff(lrff, psi0_basis).to(device)
    init_model = CompositeDensityModel([dtf], ff).to(device)

    print("Training initial model weights")
    optimizers = {
        "weights": torch.optim.Adam(
            [
                {"params": init_model.density_model.weight_params(), "lr": init_params["lr_weights"]},
            ]
        ),
    }
    init_model, best_loss_init, training_time_init = train.train_iterate(
        init_model,
        x0_dataloader,
        {"mle": loss.mle_loss},
        optimizers,
        epochs_per_group=init_params["n_epochs_per_group"],
        iterations=init_params["iterations"],
        verbose=True,
        use_best="mle",
        clip_grad_norm=5.0,
        restore_loss_threshold=50.0,
    )
    print("Done.\n")
    print(f"Initial model loss: {best_loss_init:.6f}, training time: {training_time_init:.2f}s")

    ######### ANALYSIS ########
    cpu = torch.device("cpu")
    init_model.to(cpu)
    tran_model.to(cpu)
    dtf.to(cpu)
    print("Propagating beliefs...")
    base_belief_seq = propagate.propagate(
        init_model.density_model,
        tran_model.conditional_density_model,
        n_steps=problem.n_timesteps,
        device=cpu,
    )
    print("Done.\n")
    belief_seq = [CompositeDensityModel([dtf], belief) for belief in base_belief_seq]

    ll_per_step = []
    for i in range(problem.n_timesteps):
        data_i = test_traj_data[i].to(cpu)
        ll = avg_log_likelihood(belief_seq[i], data_i)
        ll_per_step.append(ll.detach().cpu().reshape(()))
        print(f"Avg log-likelihood at time {i}: {float(ll_per_step[-1]):.6f}")
    prop_ll_vector = torch.stack(ll_per_step).to(dtype=torch.float32)
    box_lows = tuple(problem.plot_bounds_low.tolist())
    box_highs = tuple(problem.plot_bounds_high.tolist())

    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(__file__).stem

    ll_out_path = out_dir / f"{stem}__ll.png"
    plt.figure(figsize=(8, 4))
    plt.plot(prop_ll_vector.numpy(), marker="o")
    plt.xlabel("timestep")
    plt.ylabel("avg log-likelihood")
    plt.title(stem)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(ll_out_path, dpi=200)
    print(f"Saved LL plot to {ll_out_path}")

    #cond_slice_out_path = out_dir / f"{stem}__conditional_slices_lrff_vs_data.png"
    #_plot_conditional_slices_lrff_vs_data(
    #    tran_model,
    #    x_k_data.detach().cpu(),
    #    x_kp1_data.detach().cpu(),
    #    cond_slice_out_path,
    #    n_random_points=20,
    #    n_grid=120,
    #    n_bins=50,
    #    min_points_per_slice=1,
    #    bin_width=0.2,
    #    random_seed=0,
    #    title=f"{stem}: random conditional bins, empirical vs lrff model",
    #)
    #print(f"Saved conditional slice comparison plot to {cond_slice_out_path}")

    belief_out_path = out_dir / f"{stem}__beliefs_2d.png"
    fig, axes = plt.subplots(2, problem.n_timesteps, figsize=(4 * problem.n_timesteps, 8))
    if problem.n_timesteps == 1:
        axes = np.array(axes, dtype=object).reshape(2, 1)
    fig.suptitle(f"{stem}: propagated beliefs (Van der Pol)")
    for i in range(problem.n_timesteps):
        data_i = test_traj_data[i].to(cpu)
        axes[0, i].scatter(data_i[:, 0], data_i[:, 1], s=1)
        axes[0, i].set_aspect("equal")
        axes[0, i].set_xlim(box_lows[0], box_highs[0])
        axes[0, i].set_ylim(box_lows[1], box_highs[1])
        axes[0, i].set_title(f"t={i}")
        if i == 0:
            axes[0, i].set_ylabel("test data")

        plot_belief(
            axes[1, i],
            belief_seq[i],
            x_range=(box_lows[0], box_highs[0]),
            y_range=(box_lows[1], box_highs[1]),
        )
        if i == 0:
            axes[1, i].set_ylabel("propagated belief")
    fig.tight_layout()
    fig.savefig(belief_out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 2D propagated belief plot to {belief_out_path}")

    if use_gpu:
        torch.cuda.empty_cache()
    tran_model_cpu = tran_model.cpu().eval()
    base_dist_cpu = base_distribution.cpu()
    with torch.no_grad():
        dtf_c = tran_model_cpu.domain_tfs[0]
        x_k_latent, _ = dtf_c(x_k_data.detach().cpu())
        x_kp1_latent, _ = dtf_c(x_kp1_data.detach().cpu())
        x0_latent, _ = dtf_c(x0_data.detach().cpu())

    marg_path = out_dir / f"{stem}__latent_marginal_hist_g_base.png"
    _plot_latent_marginal_hist_g_base(
        tran_model_cpu.conditional_density_model,
        base_dist_cpu,
        x_k_latent,
        x0_latent,
        marg_path,
        n_grid=200,
        title=f"{stem}: z=f(x_k); hist(x_k) + hist(x0), g marginal = phi_d@a, LogisticSigmoid marginal",
    )
    print(f"Saved latent marginal hist / g(z) / base plot to {marg_path}")

    marg_xkp1_path = out_dir / f"{stem}__latent_marginal_hist_g_base__x_kp1.png"
    _plot_latent_marginal_hist_g_base(
        tran_model_cpu.conditional_density_model,
        base_dist_cpu,
        x_kp1_latent,
        x0_latent,
        marg_xkp1_path,
        n_grid=200,
        title=f"{stem}: z=f(x); hist(x_kp1) + hist(x0), g marginal = phi_d@a (same as x_k), LogisticSigmoid marginal",
    )
    print(f"Saved x_kp1-hist / g / base plot to {marg_xkp1_path}")


if __name__ == "__main__":
    main()
