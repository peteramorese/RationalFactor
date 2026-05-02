import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from rational_factor.systems.base import simulate, SystemObservationDistribution, SystemTransitionDistribution
from rational_factor.systems.problems import PARTIALLY_OBSERVABLE_PROBLEMS
from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.factor_forms import LinearRF, LinearR2FF, Linear2FF, LinearFF, LinearRFF
import rational_factor.models.train as train
import rational_factor.models.loss as loss
import rational_factor.tools.propagate as propagate
from rational_factor.models.density_model import LogisticSigmoid
from rational_factor.models.composite_model import CompositeDensityModel, CompositeConditionalModel
from rational_factor.models.domain_transformation import MaskedAffineNFTF
from rational_factor.tools.visualization import plot_belief, plot_particle_belief
from rational_factor.tools.analysis import check_pdf_valid
from rational_factor.tools.misc import data_bounds, train_test_split
from particle_filter.particle_set import WeightedParticleSet
from particle_filter.propagate import propagate_and_update
import matplotlib.pyplot as plt
from pathlib import Path


def _plot_state_vs_latent_grid(
    transform: torch.nn.Module,
    x_data: torch.Tensor,
    out_path: Path,
    *,
    n_grid_lines: int = 11,
    n_points_per_line: int = 260,
    quantile_pad: float = 0.02,
    title: str = "",
) -> None:
    if x_data.shape[1] != 2:
        raise ValueError("_plot_state_vs_latent_grid expects 2D state data.")

    x_cpu = x_data.detach().cpu()
    q_lo = x_cpu.quantile(quantile_pad, dim=0)
    q_hi = x_cpu.quantile(1.0 - quantile_pad, dim=0)
    lo = q_lo.numpy()
    hi = q_hi.numpy()

    x_vals = np.linspace(lo[0], hi[0], n_points_per_line)
    y_vals = np.linspace(lo[1], hi[1], n_points_per_line)
    gx = np.linspace(lo[0], hi[0], n_grid_lines)
    gy = np.linspace(lo[1], hi[1], n_grid_lines)

    try:
        transform_device = next(transform.parameters()).device
    except StopIteration:
        transform_device = torch.device("cpu")

    with torch.no_grad():
        transform.eval()
        z_data, _ = transform(x_cpu.to(device=transform_device, dtype=torch.float32))
        z_cpu = z_data.detach().cpu()
    z_lo = z_cpu.quantile(quantile_pad, dim=0).numpy()
    z_hi = z_cpu.quantile(1.0 - quantile_pad, dim=0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), squeeze=False)
    ax_state, ax_latent = axes[0, 0], axes[0, 1]
    ax_state.set_title("State space: rectangular grid")
    ax_latent.set_title("Latent space: transformed grid")
    ax_state.scatter(x_cpu[:, 0].numpy(), x_cpu[:, 1].numpy(), s=1, alpha=0.08, c="gray", rasterized=True)
    ax_latent.scatter(z_cpu[:, 0].numpy(), z_cpu[:, 1].numpy(), s=1, alpha=0.08, c="gray", rasterized=True)

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_grid_lines))
    with torch.no_grad():
        transform.eval()
        for i, xv in enumerate(gx):
            line_xy = np.stack([np.full_like(y_vals, xv), y_vals], axis=1)
            line_t = torch.from_numpy(line_xy).to(device=transform_device, dtype=torch.float32)
            z_line, _ = transform(line_t)
            z_np = z_line.detach().cpu().numpy()
            ax_state.plot(line_xy[:, 0], line_xy[:, 1], color=colors[i], lw=1.4)
            ax_latent.plot(z_np[:, 0], z_np[:, 1], color=colors[i], lw=1.4)

        for i, yv in enumerate(gy):
            line_xy = np.stack([x_vals, np.full_like(x_vals, yv)], axis=1)
            line_t = torch.from_numpy(line_xy).to(device=transform_device, dtype=torch.float32)
            z_line, _ = transform(line_t)
            z_np = z_line.detach().cpu().numpy()
            ax_state.plot(line_xy[:, 0], line_xy[:, 1], color=colors[i], lw=1.4, alpha=0.8)
            ax_latent.plot(z_np[:, 0], z_np[:, 1], color=colors[i], lw=1.4, alpha=0.8)

    ax_state.set_xlim(float(lo[0]), float(hi[0]))
    ax_state.set_ylim(float(lo[1]), float(hi[1]))
    ax_latent.set_xlim(float(z_lo[0]), float(z_hi[0]))
    ax_latent.set_ylim(float(z_lo[1]), float(z_hi[1]))

    ax_state.set_xlabel("x_1")
    ax_state.set_ylabel("x_2")
    ax_latent.set_xlabel("z_1")
    ax_latent.set_ylabel("z_2")
    ax_state.set_aspect("equal")
    ax_latent.set_aspect("equal")
    ax_state.grid(True, alpha=0.25)
    ax_latent.grid(True, alpha=0.25)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_initial_distribution_comparison(
    x0_data: torch.Tensor,
    y0_data: torch.Tensor,
    init_model: torch.nn.Module,
    out_path: Path,
    *,
    init_gmm_lf: torch.nn.Module | None = None,
    r_basis: torch.nn.Module | None = None,
    r_coeffs: torch.Tensor | None = None,
    g_basis: torch.nn.Module | None = None,
    g_coeffs: torch.Tensor | None = None,
    title: str = "",
) -> None:
    if x0_data.shape[1] != 2 or y0_data.shape[1] != 2:
        raise ValueError("_plot_initial_distribution_comparison expects 2D data.")

    x0_cpu = x0_data.detach().cpu()
    y0_cpu = y0_data.detach().cpu()
    all_xy = torch.cat([x0_cpu, y0_cpu], dim=0)
    lo = all_xy.quantile(0.01, dim=0).numpy()
    hi = all_xy.quantile(0.99, dim=0).numpy()
    x_range = (float(lo[0]), float(hi[0]))
    y_range = (float(lo[1]), float(hi[1]))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
    ax_y0, ax_xy, ax_rx = axes[0, 0], axes[0, 1], axes[0, 2]
    ax_gmm, ax_init, ax_gx = axes[1, 0], axes[1, 1], axes[1, 2]

    ax_y0.scatter(y0_cpu[:, 0].numpy(), y0_cpu[:, 1].numpy(), s=3, alpha=0.25, c="tab:green", rasterized=True)
    ax_y0.set_title("y0 data")

    ax_xy.scatter(x0_cpu[:, 0].numpy(), x0_cpu[:, 1].numpy(), s=3, alpha=0.2, c="tab:blue", label="x0", rasterized=True)
    ax_xy.scatter(y0_cpu[:, 0].numpy(), y0_cpu[:, 1].numpy(), s=3, alpha=0.2, c="tab:orange", label="y0", rasterized=True)
    ax_xy.set_title("x0 vs y0 data")
    ax_xy.legend(loc="upper right", fontsize=8)

    if init_gmm_lf is not None:
        plot_belief(ax_gmm, init_gmm_lf, x_range=x_range, y_range=y_range)
        ax_gmm.set_title("GMM init density")
    else:
        ax_gmm.text(0.5, 0.5, "GMM init disabled\n(gmm_init=False)", ha="center", va="center", fontsize=10)
        ax_gmm.set_title("GMM init density")

    plot_belief(ax_init, init_model, x_range=x_range, y_range=y_range)
    ax_init.set_title("Trained init model density")

    x_lin = np.linspace(x_range[0], x_range[1], 120)
    y_lin = np.linspace(y_range[0], y_range[1], 120)
    X, Y = np.meshgrid(x_lin, y_lin)
    xy = np.stack([X.ravel(), Y.ravel()], axis=1)

    if r_basis is not None and r_coeffs is not None:
        with torch.no_grad():
            basis_param = next(iter(r_basis.parameters()), None)
            basis_buffer = next(iter(r_basis.buffers()), None)
            basis_device = basis_param.device if basis_param is not None else (basis_buffer.device if basis_buffer is not None else torch.device("cpu"))
            basis_dtype = basis_param.dtype if basis_param is not None else (basis_buffer.dtype if basis_buffer is not None else torch.get_default_dtype())
            phi = r_basis(torch.tensor(xy, dtype=basis_dtype, device=basis_device))
            d = r_coeffs.to(device=basis_device, dtype=basis_dtype)
            r_vals = (phi @ d).detach().cpu().numpy().reshape(X.shape)
        ax_rx.contourf(X, Y, r_vals, levels=12)
        ax_rx.contour(X, Y, r_vals, levels=12, colors="white", linewidths=0.4)
        ax_rx.set_title("r(x) = xi(x)^T d")
    else:
        ax_rx.text(0.5, 0.5, "r(x) unavailable", ha="center", va="center", fontsize=10)
        ax_rx.set_title("r(x) = xi(x)^T d")

    if g_basis is not None and g_coeffs is not None:
        with torch.no_grad():
            basis_param = next(iter(g_basis.parameters()), None)
            basis_buffer = next(iter(g_basis.buffers()), None)
            basis_device = basis_param.device if basis_param is not None else (basis_buffer.device if basis_buffer is not None else torch.device("cpu"))
            basis_dtype = basis_param.dtype if basis_param is not None else (basis_buffer.dtype if basis_buffer is not None else torch.get_default_dtype())
            phi = g_basis(torch.tensor(xy, dtype=basis_dtype, device=basis_device))
            a = g_coeffs.to(device=basis_device, dtype=basis_dtype)
            g_vals = (phi @ a).detach().cpu().numpy().reshape(X.shape)
        ax_gx.contourf(X, Y, g_vals, levels=12)
        ax_gx.contour(X, Y, g_vals, levels=12, colors="white", linewidths=0.4)
        ax_gx.set_title("g(x) = phi(x)^T a")
    else:
        ax_gx.text(0.5, 0.5, "g(x) unavailable", ha="center", va="center", fontsize=10)
        ax_gx.set_title("g(x) = phi(x)^T a")

    for ax in [ax_y0, ax_xy, ax_gmm, ax_init, ax_rx, ax_gx]:
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False, bottom=False, left=False)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    
    ###
    problem = PARTIALLY_OBSERVABLE_PROBLEMS["po_van_der_pol"]

    use_gpu = torch.cuda.is_available()
    use_dtf = True
    n_basis = 300
    n_obs_basis = 300

    if use_dtf:
        
        obs_and_tran_params = {
            "n_epochs_per_group": [5, 5, 5, 5], # tran_basis+dtf, tran_weights, obs_basis, obs_weights
            "outer_iterations": 3,
            "iterations": 5,
            "pre_train_epochs": 5,
            "lr_basis_tran": 5e-3,
            "lr_basis_obs": 5e-3,
            "lr_weights": 1e-3,
            "lr_dtf": 5e-5,
            "dtf_weight_decay": 1e-2,
        }
        init_params = {
            "n_epochs_per_group": [20, 5], # basis, weights
            "iterations": 200,
            "lr_basis": 1e-2,
            "lr_weights": 1e-3,
        }
    else:
        obs_and_tran_params = {
            "n_epochs_per_group": [5, 5, 5, 5], # tran_basis, tran_weights, obs_basis, obs_weights
            "outer_iterations": 1,
            "iterations": 50,
            "lr_basis_tran": 5e-2,
            "lr_basis_obs": 5e-2,
            "lr_weights": 1e-2,
        }
        init_params = {
            "n_epochs_per_group": [20, 5], # basis, weights
            "iterations": 100,
            "lr_basis": 1e-3,
            "lr_weights": 1e-2,
        }

    batch_size = 256
    block_size = None

    reg_covar_joint = 1e-1
    reg_covar_obs = 1e-1
    reg_covar_init = 1e-1
    ls_temp = 0.1

    obs_loss_weight = 1.0

    n_simulation_tests = 3
    n_particles_true_pf = 5000
    figure_prefix = "vdp_nftf_filter_gaussian"
    gmm_init = True
    ###

    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using GPU: ", use_gpu)

    # Create system from shared partially observable problem parameters
    system = problem.system
    init_state_sampler = problem.initial_state_sampler
    n_timesteps_prop = problem.n_timesteps

    x0_data = problem.train_initial_state_data()
    x_k_data, x_kp1_data = problem.train_state_transition_data()
    xo_data, o_data = problem.train_obs_data()

    x0_train, x0_val = train_test_split(x0_data, test_size=0.2)
    x_k_train, x_k_val, x_kp1_train, x_kp1_val = train_test_split(x_k_data, x_kp1_data, test_size=0.2)
    xo_train, xo_val, o_train, o_val = train_test_split(xo_data, o_data, test_size=0.2)

    # Training data loaders
    x0_dataloader = DataLoader(TensorDataset(x0_train), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(TensorDataset(x_kp1_train, x_k_train), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    x_k_dataloader = DataLoader(TensorDataset(x_k_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    o_dataloader = DataLoader(TensorDataset(o_train, xo_train), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    # Validation data loaders
    x0_val_dataloader = DataLoader(TensorDataset(x0_val), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    xp_val_dataloader = DataLoader(TensorDataset(x_kp1_val, x_k_val), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    o_val_dataloader = DataLoader(TensorDataset(o_val, xo_val), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    nftf = MaskedAffineNFTF(system.dim(), trainable=True, hidden_features=256, n_layers=8).to(device)

    # Pre train the dtf
    if use_dtf:
        print("Pre training the dtf")
        loc, scale = data_bounds(x_k_data, mode="center_lengths")
        base_distribution = LogisticSigmoid(system.dim(), temperature=ls_temp, loc=loc.to(device), scale=scale.to(device))
        decorrupter_density = CompositeDensityModel([nftf], base_distribution).to(device)
        optimizer = torch.optim.Adam(nftf.parameters(), lr=obs_and_tran_params["lr_dtf"], weight_decay=obs_and_tran_params["lr_dtf"])
        decorrupter_density, best_loss, training_time = train.train(
            decorrupter_density,
            x_k_dataloader,
            {"mle": loss.mle_loss},
            optimizer,
            epochs=obs_and_tran_params["pre_train_epochs"],
            verbose=True,
            use_best="mle",
        )
        print("Done.\n")

    # Prefit the basis functions
    print("Prefitting the basis functions")
    if use_dtf:
        with torch.no_grad():
            y_k_data, _ = nftf(x_k_data.to(device))
            y_kp1_data, _ = nftf(x_kp1_data.to(device))
            y0_data, _ = nftf(x0_data.to(device))
            yo_data, _ = nftf(xo_data.to(device))
            y_k_data = y_k_data.to(torch.device("cpu"))
            y_kp1_data = y_kp1_data.to(torch.device("cpu"))
            y0_data = y0_data.to(torch.device("cpu"))
            yo_data = yo_data.to(torch.device("cpu"))
    else:
        y_k_data = x_k_data.to(torch.device("cpu"))
        y_kp1_data = x_kp1_data.to(torch.device("cpu"))
        y0_data = x0_data.to(torch.device("cpu"))
        yo_data = xo_data.to(torch.device("cpu"))

    # Transition model basis functions (fit to p(y, y'))
    y_joint_data = torch.cat([y_k_data, y_kp1_data], dim=1)
    tran_gmm_lf = train.fit_gaussian_lf_em(y_joint_data.to(torch.device("cpu")), n_components=n_basis, reg_covar=reg_covar_joint, max_iter=100)
    weights = tran_gmm_lf.get_w()
    phi_marginal = tran_gmm_lf.basis.marginal(marginal_dims=range(system.dim()))
    phi_means, phi_stds = phi_marginal.means_stds()
    phi_params = torch.stack([phi_means, phi_stds], dim=-1)

    psi_marginal = tran_gmm_lf.basis.marginal(marginal_dims=range(system.dim(), 2 * system.dim()))
    psi_means, psi_stds = psi_marginal.means_stds()
    psi_params = torch.stack([psi_means, psi_stds], dim=-1)

    phi_basis = GaussianBasis(uparams_init=phi_params).to(device)
    psi_basis = GaussianBasis(uparams_init=psi_params).to(device)

    # Observation model basis functions (fit to p(y, o))
    yo_joint_data = torch.cat([xo_data, o_data], dim=1)
    obs_gmm_lf = train.fit_gaussian_lf_em(yo_joint_data.to(torch.device("cpu")), n_components=n_obs_basis, reg_covar=reg_covar_obs, max_iter=100)
    weights = obs_gmm_lf.get_w()
    xi_marginal = obs_gmm_lf.basis.marginal(marginal_dims=range(system.dim()))
    xi_means, xi_stds = xi_marginal.means_stds()
    xi_params = torch.stack([xi_means, xi_stds], dim=-1)

    zeta_marginal = obs_gmm_lf.basis.marginal(marginal_dims=range(system.dim(), system.dim() + system.observation_dim()))
    zeta_means, zeta_stds = zeta_marginal.means_stds()
    zeta_params = torch.stack([zeta_means, zeta_stds], dim=-1)

    xi_basis = GaussianBasis(uparams_init=xi_params).to(device)
    zeta_basis = GaussianBasis(uparams_init=zeta_params).to(device)

    # Initial model basis functions (fit to p(y))
    init_gmm_lf = None
    if gmm_init:
        init_gmm_lf = train.fit_gaussian_lf_em(y0_data.to(torch.device("cpu")), n_components=n_basis, reg_covar=reg_covar_init, max_iter=100)
        weights = init_gmm_lf.get_w()
        psi0_means, psi0_stds = init_gmm_lf.basis.means_stds()
        psi0_params = torch.stack([psi0_means, psi0_stds], dim=-1)

        psi0_basis = GaussianBasis(uparams_init=psi0_params).to(device)
    else:
        psi0_basis = GaussianBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 20.0], device=device), variance=30.0, min_std=1e-4).to(device)

    # Stage 1: pretrain transition as a LinearRFF with the domain transformation
    if use_dtf:
        tran_pretrain_model = CompositeConditionalModel([nftf], LinearRFF(phi_basis, psi_basis)).to(device)
    else:
        tran_pretrain_model = LinearRFF(phi_basis, psi_basis).to(device)

    epochs_per_group = obs_and_tran_params["n_epochs_per_group"]
    if not isinstance(epochs_per_group, list):
        epochs_per_group = [epochs_per_group] * 4
    assert len(epochs_per_group) == 4, "Expected 4 epoch groups: tran_basis+dtf, tran_weights, obs_basis, obs_weights"
    outer_iterations = obs_and_tran_params.get("outer_iterations", 1)
    inner_iterations = obs_and_tran_params["iterations"]

    tran_conditional_mle_loss_fn = lambda model, xp, x : loss.conditional_mle_loss(model, xp, x, method_name="log_density")
    obs_conditional_mle_loss_fn = lambda model, o, x : obs_loss_weight * loss.conditional_mle_loss(model, o, x, method_name="log_density")

    tran_pretrain_optimizers = {
        "tran_basis": torch.optim.Adam(tran_pretrain_model.conditional_density_model.basis_params(), lr=obs_and_tran_params["lr_basis_tran"]) if use_dtf else torch.optim.Adam(tran_pretrain_model.basis_params(), lr=obs_and_tran_params["lr_basis_tran"]),
        "tran_weights": torch.optim.Adam(tran_pretrain_model.conditional_density_model.weight_params(), lr=obs_and_tran_params["lr_weights"]) if use_dtf else torch.optim.Adam(tran_pretrain_model.weight_params(), lr=obs_and_tran_params["lr_weights"]),
    }
    if use_dtf:
        tran_pretrain_optimizers["tran_basis_and_dtf"] = torch.optim.Adam(
            [
                {"params": tran_pretrain_model.conditional_density_model.basis_params(), "lr": obs_and_tran_params["lr_basis_tran"]},
                {"params": tran_pretrain_model.domain_tfs.parameters(), "lr": obs_and_tran_params["lr_dtf"], "weight_decay": obs_and_tran_params["dtf_weight_decay"]},
            ]
        )

    print("Pretraining transition model (LinearRFF)")
    best_loss_tran_pre = float("inf")
    training_time_tran_pre = 0.0
    for outer_iter in range(outer_iterations):
        print(f"\n=== Transition pretrain outer iteration {outer_iter + 1}/{outer_iterations} ===")
        tran_basis_key = "tran_basis_and_dtf" if use_dtf else "tran_basis"
        tran_pretrain_model, best_loss_phase, phase_time = train.train_iterate(
            tran_pretrain_model,
            xp_dataloader,
            labeled_loss_fns={"mle": tran_conditional_mle_loss_fn},
            labeled_optimizers={
                tran_basis_key: tran_pretrain_optimizers[tran_basis_key],
                "tran_weights": tran_pretrain_optimizers["tran_weights"],
            },
            labeled_validation_loss_fns={"val_mle": tran_conditional_mle_loss_fn},
            validation_data_loader=xp_val_dataloader,
            epochs_per_group=[epochs_per_group[0], epochs_per_group[1]],
            iterations=inner_iterations,
            verbose=True,
            use_best="val_mle",
        )
        best_loss_tran_pre = min(best_loss_tran_pre, best_loss_phase)
        training_time_tran_pre += phase_time
    print("Done pretraining transition model.\n")
    print("Transition pretrain valid: ", tran_pretrain_model.valid())

    # Freeze the domain transformation after pretraining.
    if use_dtf:
        for p in nftf.parameters():
            p.requires_grad = False
        nftf.eval()

    # Stage 2: train observation model as LinearRF with transformed conditioner only
    obs_rf_base = LinearRF(xi_basis, zeta_basis).to(device)
    if use_dtf:
        obs_model = CompositeConditionalModel([nftf], obs_rf_base, tf_conditioner_only=True).to(device)
    else:
        obs_model = obs_rf_base
    obs_optimizers = {
        "obs_basis": torch.optim.Adam(obs_rf_base.basis_params(), lr=obs_and_tran_params["lr_basis_obs"]),
        "obs_weights": torch.optim.Adam(obs_rf_base.weight_params(), lr=obs_and_tran_params["lr_weights"]),
    }
    print("Training observation model (LinearRF)")
    obs_model, best_loss_obs, training_time_obs = train.train_iterate(
        obs_model,
        o_dataloader,
        labeled_loss_fns={"obs_mle": obs_conditional_mle_loss_fn},
        labeled_optimizers={
            "obs_basis": obs_optimizers["obs_basis"],
            "obs_weights": obs_optimizers["obs_weights"],
        },
        labeled_validation_loss_fns={"val_obs_mle": obs_conditional_mle_loss_fn},
        validation_data_loader=o_val_dataloader,
        epochs_per_group=[epochs_per_group[2], epochs_per_group[3]],
        iterations=outer_iterations * inner_iterations,
        verbose=True,
        use_best="val_obs_mle",
    )
    print("Done training observation model.\n")
    print("Observation model valid: ", obs_model.valid())

    # Stage 3: train transition model as LinearR2FF from learned LinearRF (fixed r(x))
    obs_rf_for_r2ff = obs_model.conditional_density_model if use_dtf else obs_model
    tran_r2ff_base = LinearR2FF.from_rf(obs_rf_for_r2ff, phi_basis, psi_basis).to(device)
    if use_dtf:
        tran_model = CompositeConditionalModel([nftf], tran_r2ff_base).to(device)
    else:
        tran_model = tran_r2ff_base
    tran_r2ff_optimizers = {
        "tran_basis": torch.optim.Adam(tran_r2ff_base.basis_params(), lr=obs_and_tran_params["lr_basis_tran"]),
        "tran_weights": torch.optim.Adam(tran_r2ff_base.weight_params(), lr=obs_and_tran_params["lr_weights"]),
    }
    print("Training transition model (LinearR2FF with fixed r(x) and dtf)")
    tran_model, best_loss_tran, training_time_tran = train.train_iterate(
        tran_model,
        xp_dataloader,
        labeled_loss_fns={"mle": tran_conditional_mle_loss_fn},
        labeled_optimizers={
            "tran_basis": tran_r2ff_optimizers["tran_basis"],
            "tran_weights": tran_r2ff_optimizers["tran_weights"],
        },
        labeled_validation_loss_fns={"val_mle": tran_conditional_mle_loss_fn},
        validation_data_loader=xp_val_dataloader,
        epochs_per_group=[epochs_per_group[0], epochs_per_group[1]],
        iterations=outer_iterations * inner_iterations,
        verbose=True,
        use_best="val_mle",
    )
    print("Done training transition model.\n")
    print("Transition model valid: ", tran_model.valid())

    figure_dir = Path("figures")
    figure_dir.mkdir(parents=True, exist_ok=True)

    if use_dtf:
        grid_figure_path = figure_dir / f"{figure_prefix}__state_vs_latent_grid.png"
        _plot_state_vs_latent_grid(
            nftf,
            x_k_data,
            grid_figure_path,
            title=f"{figure_prefix}: state vs latent grid",
        )
        print(f"Saved state-vs-latent grid figure to {grid_figure_path}")

    base_tran_model = tran_model.conditional_density_model if use_dtf else tran_model
    init_model = LinearFF.from_r2ff(base_tran_model, psi0_basis).to(device)
    optimizers = {"basis": torch.optim.Adam(init_model.basis_params(), lr=init_params["lr_basis"]), "weights": torch.optim.Adam(init_model.weight_params(), lr=init_params["lr_weights"])}

    print("Training initial model")
    y0_dataloader = DataLoader(TensorDataset(y0_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu) if use_dtf else x0_dataloader
    init_model, best_loss_init, training_time_init = train.train_iterate(init_model, 
        y0_dataloader, 
        {"mle": loss.mle_loss}, 
        optimizers,
        epochs_per_group=init_params["n_epochs_per_group"],
        iterations=init_params["iterations"],
        verbose=True,
        use_best="mle")
    print("Done! \n")
    if x0_data.shape[1] == 2 and y0_data.shape[1] == 2:
        init_compare_figure_path = figure_dir / f"{figure_prefix}__initial_distribution_comparison.png"
        _plot_initial_distribution_comparison(
            x0_data,
            y0_data,
            init_model,
            init_compare_figure_path,
            init_gmm_lf=init_gmm_lf if gmm_init else None,
            r_basis=base_tran_model.xi_basis,
            r_coeffs=base_tran_model.get_d().detach().cpu(),
            g_basis=base_tran_model.phi_basis,
            g_coeffs=base_tran_model.get_a().detach().cpu(),
            title=f"{figure_prefix}: initial distribution comparison",
        )
        print(f"Saved initial distribution comparison figure to {init_compare_figure_path}")
    else:
        print("Skipping initial distribution comparison plot (only implemented for 2D state).")

    print(f"Transition pretrain loss: {best_loss_tran_pre:.4f}, training time: {training_time_tran_pre:.2f} seconds")
    print(f"Observation model loss : {best_loss_obs:.4f}, training time: {training_time_obs:.2f} seconds")
    print(f"Transition model loss  : {best_loss_tran:.4f}, training time: {training_time_tran:.2f} seconds")
    print(f"Initial model loss     : {best_loss_init:.4f}, training time: {training_time_init:.2f} seconds")

    # Analysis across repeated simulation tests using the same learned filter
    box_lows = (-5.0, -5.0)
    box_highs = (5.0, 5.0)
    t_dist = SystemTransitionDistribution(system).to(device=device)
    o_dist = SystemObservationDistribution(system).to(device=device)

    for sim_idx in range(n_simulation_tests):
        # Simulate a random true trajectory and observations
        sim_true_states, sim_observations = simulate(
            system,
            init_state_sampler,
            n_timesteps=n_timesteps_prop,
            device=device,
        )

        # Run learned filter
        with torch.no_grad():
            priors, posteriors = propagate.propagate_and_update(
                init_model,
                tran_model,
                obs_model,
                [obs for obs in sim_observations],
            )
        print(f"[sim {sim_idx + 1}/{n_simulation_tests}] priors: {len(priors)}, posteriors: {len(posteriors)}")

        # Run truth-model weighted particle filter baseline
        initial_particle_belief = WeightedParticleSet(
            particles=init_state_sampler(n_particles_true_pf).to(device=device),
            weights=torch.ones(n_particles_true_pf).to(device=device) / n_particles_true_pf,
        )
        _, wpf_posteriors = propagate_and_update(
            belief=initial_particle_belief,
            transition_model=t_dist,
            observation_model=o_dist,
            observations=[obs for obs in sim_observations],
        )


        fig, axes = plt.subplots(n_timesteps_prop, 3, figsize=(10, max(3 * n_timesteps_prop, 15)))
        fig.delaxes(axes[0, 0])  # no prior exists at k=0
        fig.suptitle(f"Beliefs at each time step ({figure_prefix}, sim {sim_idx + 1})")
        with torch.no_grad():
            for i in range(n_timesteps_prop):
                if i > 0:
                    #print("Testing prior ", i)
                    #check_pdf_valid(priors[i - 1], domain_bounds=(box_lows, box_highs), n_samples=100000, device=device)

                    plot_belief(axes[i, 0], priors[i - 1], x_range=(box_lows[0], box_highs[0]), y_range=(box_lows[1], box_highs[1]))
                    axes[i, 0].set_title(f"Prior at k={i}", fontsize=8)

                #print("Testing posterior", i)
                #check_pdf_valid(posteriors[i], domain_bounds=(box_lows, box_highs), n_samples=100000, device=device)

                plot_belief(axes[i, 1], posteriors[i], x_range=(box_lows[0], box_highs[0]), y_range=(box_lows[1], box_highs[1]))
                axes[i, 1].set_title(f"Posterior at k={i}", fontsize=8)

                plot_particle_belief(axes[i, 2], wpf_posteriors[i], x_range=(box_lows[0], box_highs[0]), y_range=(box_lows[1], box_highs[1]))
                axes[i, 2].set_title(f"WPF Posterior at k={i}", fontsize=8)
                cols = [1, 2] if i == 0 else [0, 1, 2]
                for j in cols:
                    if j > 0:  # Skip first prior plot
                        axes[i, j].scatter(sim_true_states[i, 0].item(), sim_true_states[i, 1].item(), marker="o", s=10, c="red")
                    if i > 0:
                        axes[i, j].scatter(sim_observations[i - 1, 0].item(), sim_observations[i - 1, 1].item(), marker="o", s=10, c="blue")
                    axes[i, j].set_xlabel("")
                    axes[i, j].set_ylabel("")
                    axes[i, j].tick_params(
                        axis="both",
                        which="both",
                        labelbottom=False,
                        labelleft=False,
                        bottom=False,
                        left=False,
                    )
                    axes[i, j].grid(alpha=0.25)

        figure_path = figure_dir / f"{figure_prefix}__sim_{sim_idx + 1:02d}.png"
        plt.savefig(figure_path, dpi=500, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure to {figure_path}")


