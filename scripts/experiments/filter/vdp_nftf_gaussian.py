import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
from rational_factor.systems.base import simulate, SystemObservationDistribution, SystemTransitionDistribution
from rational_factor.systems.problems import PARTIALLY_OBSERVABLE_PROBLEMS
from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.factor_forms import LinearRF, LinearR2FF, Linear2FF, LinearFF, LinearRFF, LinearRFandR2FF
import rational_factor.models.train as train
import rational_factor.models.loss as loss
import rational_factor.tools.propagate as propagate
from rational_factor.models.density_model import LogisticSigmoid
from rational_factor.models.composite_model import CompositeDensityModel, CompositeConditionalModel, CompositeRFandR2FF
from rational_factor.models.domain_transformation import MaskedAffineNFTF
from rational_factor.tools.visualization import plot_belief, plot_particle_belief
from rational_factor.tools.analysis import check_pdf_valid, avg_log_filter_score
from rational_factor.tools.misc import data_bounds, train_test_split
from rational_factor.models.filter import Filter
from particle_filter.particle_set import WeightedParticleSet
from particle_filter.propagate import propagate_and_update as pf_propagate_and_update
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy


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
    init_model_untrained: torch.nn.Module,
    init_model_trained: torch.nn.Module,
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

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), squeeze=False)
    ax_y0, ax_xy, ax_rx, ax_gx = axes[0, 0], axes[0, 1], axes[0, 2], axes[0, 3]
    ax_gmm, ax_init_untrained, ax_init_trained, ax_blank = axes[1, 0], axes[1, 1], axes[1, 2], axes[1, 3]

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
        ax_gmm.text(0.5, 0.5, "GMM init disabled", ha="center", va="center", fontsize=10)
        ax_gmm.set_title("GMM init density")

    plot_belief(ax_init_untrained, init_model_untrained, x_range=x_range, y_range=y_range)
    ax_init_untrained.set_title("Init model (before training)")

    plot_belief(ax_init_trained, init_model_trained, x_range=x_range, y_range=y_range)
    ax_init_trained.set_title("Init model (after training)")

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

    ax_blank.axis("off")

    for ax in [ax_y0, ax_xy, ax_gmm, ax_init_untrained, ax_init_trained, ax_rx, ax_gx]:
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-dtf",
        dest="use_dtf",
        action="store_false",
        default=True,
        help="Disable domain transformation flow (DTF).",
    )
    args = parser.parse_args()

    ###
    #problem = PARTIALLY_OBSERVABLE_PROBLEMS["bad_sensor_van_der_pol"]
    problem = PARTIALLY_OBSERVABLE_PROBLEMS["dubins_trailer"]

    use_gpu = torch.cuda.is_available()
    use_dtf = args.use_dtf
    n_basis = 300
    n_obs_basis = 100

    if use_dtf:
        
        obs_and_tran_params = {
            "n_epochs_per_group": [5, 5], # basis, weights
            "iterations": 15,
            "pre_train_epochs": 2,
            "lr_basis_tran": 5e-2,
            "lr_basis_obs": 5e-2,
            "lr_weights": 1e-2,
            "lr_dtf": 1e-4,
            "dtf_weight_decay": 0.0,
        }
        init_params = {
            "n_epochs_per_group": [20, 5], # basis, weights
            "iterations": 100,
            "lr_basis": 1e-3,
            "lr_weights": 1e-3,
        }
    else:
        obs_and_tran_params = {
            "n_epochs_per_group": [20, 5], # basis, weights
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
    reg_covar_obs = 1e-0
    reg_covar_init = 1e-0
    ls_temp = 0.1

    obs_loss_weight = 1.0
    tran_patience = 30
    init_patience = 30

    n_simulation_tests = 3
    n_test_sims = 30
    n_particles_true_pf = 5000
    figure_prefix = "vdp_nftf_filter_gaussian"
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
            yo_data, _ = nftf(xo_data.to(device))
            y_k_data = y_k_data.to(torch.device("cpu"))
            y_kp1_data = y_kp1_data.to(torch.device("cpu"))
            yo_data = yo_data.to(torch.device("cpu"))
    else:
        y_k_data = x_k_data.to(torch.device("cpu"))
        y_kp1_data = x_kp1_data.to(torch.device("cpu"))
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

    # Train the transition model and observation model simultaneously
    if use_dtf:
        tran_obs_model = CompositeRFandR2FF([nftf], LinearRFandR2FF(xi_basis, zeta_basis, phi_basis, psi_basis)).to(device)
        optimizers ={"dtf_and_basis": torch.optim.Adam([{'params': tran_obs_model.conditional_density_model.basis_params(), 'lr': obs_and_tran_params["lr_basis_tran"]}, {'params':tran_obs_model.domain_tfs.parameters(), 'lr': obs_and_tran_params["lr_dtf"], 'weight_decay': obs_and_tran_params["dtf_weight_decay"]}]), 
            "weights": torch.optim.Adam(tran_obs_model.conditional_density_model.weight_params(), lr=obs_and_tran_params["lr_weights"])} 
        tran_conditional_mle_loss_fn = lambda model, xp, x : loss.conditional_mle_loss(model, xp, x, method_name="log_density")
        obs_conditional_mle_loss_fn = lambda model, o, x : obs_loss_weight * loss.conditional_mle_loss(model, o, x, method_name="log_observation_density")
    else:
        tran_obs_model = LinearRFandR2FF(xi_basis, zeta_basis, phi_basis, psi_basis).to(device)
        optimizers ={"basis": torch.optim.Adam(tran_obs_model.basis_params(), lr=obs_and_tran_params["lr_basis_tran"]), "weights": torch.optim.Adam(tran_obs_model.weight_params(), lr=obs_and_tran_params["lr_weights"])} 
        tran_conditional_mle_loss_fn = lambda model, xp, x : loss.conditional_mle_loss(model, xp, x, method_name="log_density")
        obs_conditional_mle_loss_fn = lambda model, o, x : obs_loss_weight * loss.conditional_mle_loss(model, o, x, method_name="log_observation_density")

    print("Training transition and observation model")

    # Transition MLE uses (x_{k+1}, x_k); observation MLE uses (o, x) — separate loaders zipped per step.
    tran_obs_model, best_loss_tran_obs, training_time_tran_obs = train.train_iterate_multiset(
        tran_obs_model,
        data_loaders=[xp_dataloader, o_dataloader],
        labeled_loss_fns_list=[
            {"mle": tran_conditional_mle_loss_fn},
            {"obs_mle": obs_conditional_mle_loss_fn},
        ],
        labeled_optimizers=optimizers,
        labeled_validation_loss_fns_list=[
            {"val_mle": tran_conditional_mle_loss_fn},
            {"val_obs_mle": obs_conditional_mle_loss_fn},
        ],
        validation_data_loaders=[xp_val_dataloader, o_val_dataloader],
        epochs_per_group=obs_and_tran_params["n_epochs_per_group"],
        iterations=obs_and_tran_params["iterations"],
        verbose=True,
        use_best="val_mle",
        validation_early_stopping_patience=tran_patience,
    )
    print("Done! \n")
    print("Valid: ", tran_obs_model.valid())

    if use_dtf:
        grid_figure_path = Path("figures") / f"{figure_prefix}__state_vs_latent_grid.png"
        _plot_state_vs_latent_grid(
            nftf,
            x_k_data,
            grid_figure_path,
            title=f"{figure_prefix}: state vs latent grid",
        )
        print(f"Saved state-vs-latent grid figure to {grid_figure_path}")

    if use_dtf:
        with torch.no_grad():
            y0_data, _ = nftf(x0_data.to(device))
            y0_val_data, _ = nftf(x0_val.to(device))
        y0_data = y0_data.to(torch.device("cpu"))
        y0_val_data = y0_val_data.to(torch.device("cpu"))
    else:
        y0_data = x0_data.to(torch.device("cpu"))
        y0_val_data = x0_val.to(torch.device("cpu"))

    # Initial model basis functions (fit to p(y))
    init_gmm_lf = train.fit_gaussian_lf_em(y0_data.to(torch.device("cpu")), n_components=n_basis, reg_covar=reg_covar_init, max_iter=100)
    weights = init_gmm_lf.get_w()
    psi0_means, psi0_stds = init_gmm_lf.basis.means_stds()
    psi0_params = torch.stack([psi0_means, psi0_stds], dim=-1)

    psi0_basis = GaussianBasis(uparams_init=psi0_params).to(device)


    base_tran_obs_model = tran_obs_model.conditional_density_model if use_dtf else tran_obs_model
    init_model = LinearFF.from_r2ff(base_tran_obs_model, psi0_basis).to(device)
    init_model_untrained = deepcopy(init_model).to(device).eval()
    optimizers = {"basis": torch.optim.Adam(init_model.basis_params(), lr=init_params["lr_basis"]), "weights": torch.optim.Adam(init_model.weight_params(), lr=init_params["lr_weights"])}

    print("Training initial model")
    y0_dataloader = DataLoader(TensorDataset(y0_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    y0_val_dataloader = DataLoader(TensorDataset(y0_val_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    init_model, best_loss_init, training_time_init = train.train_iterate(init_model, 
        y0_dataloader,
        {"mle": loss.mle_loss}, 
        optimizers,
        labeled_validation_loss_fns={"val_mle": loss.mle_loss},
        validation_data_loader=y0_val_dataloader,
        epochs_per_group=init_params["n_epochs_per_group"],
        iterations=init_params["iterations"],
        verbose=True,
        use_best="val_mle",
        validation_early_stopping_patience=init_patience)
    print("Done! \n")

    print(f"Observation/transition model loss : {best_loss_tran_obs:.4f}, training time: {training_time_tran_obs:.2f} seconds")
    print(f"Initial model loss     : {best_loss_init:.4f}, training time: {training_time_init:.2f} seconds")

    # Extract the individual models for filtering
    if use_dtf:
        trained_nftf = MaskedAffineNFTF.copy_from_trainable(nftf).to(device).eval()
        tran_model = tran_obs_model.conditional_density_model.r2ff().to(device)
        obs_model = tran_obs_model.conditional_density_model.rf().to(device)
    else:
        trained_nftf = None
        tran_model = tran_obs_model.r2ff().to(device)
        obs_model = tran_obs_model.rf().to(device)

    figure_dir = Path("figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    if x0_data.shape[1] == 2 and y0_data.shape[1] == 2:
        init_compare_figure_path = figure_dir / f"{figure_prefix}__initial_distribution_comparison.png"
        if use_dtf:
            init_untrained_plot = CompositeDensityModel([trained_nftf], init_model_untrained).to(device).eval()
            init_trained_plot = CompositeDensityModel([trained_nftf], init_model).to(device).eval()
        else:
            init_untrained_plot = init_model_untrained
            init_trained_plot = init_model

        _plot_initial_distribution_comparison(
            x0_data,
            y0_data,
            init_untrained_plot,
            init_trained_plot,
            init_compare_figure_path,
            init_gmm_lf=init_gmm_lf,
            r_basis=base_tran_obs_model.xi_basis,
            r_coeffs=base_tran_obs_model.get_d().detach().cpu(),
            g_basis=base_tran_obs_model.phi_basis,
            g_coeffs=base_tran_obs_model.get_a().detach().cpu(),
            title=f"{figure_prefix}: initial distribution comparison",
        )
        print(f"Saved initial distribution comparison figure to {init_compare_figure_path}")
    else:
        print("Skipping initial distribution comparison plot (only implemented for 2D state).")

    # Analysis across repeated simulation tests using the same learned filter
    box_lows = (-5.0, -5.0)
    box_highs = (5.0, 5.0)
    t_dist = SystemTransitionDistribution(system).to(device=device)
    o_dist = SystemObservationDistribution(system).to(device=device)
    belief_plots_ok = system.dim() == 2
    if not belief_plots_ok:
        print("Skipping belief trajectory figures (only implemented for 2D state).")
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

        # Beliefs are learned/propagated in latent space when use_dtf=True.
        # Wrap them for visualization in original state space.
        if use_dtf:
            priors_plot = [CompositeDensityModel([trained_nftf], belief).to(device).eval() for belief in priors]
            posteriors_plot = [CompositeDensityModel([trained_nftf], belief).to(device).eval() for belief in posteriors]
        else:
            priors_plot = priors
            posteriors_plot = posteriors

        # Run truth-model weighted particle filter baseline
        initial_particle_belief = WeightedParticleSet(
            particles=init_state_sampler(n_particles_true_pf).to(device=device),
            weights=torch.ones(n_particles_true_pf).to(device=device) / n_particles_true_pf,
        )
        _, wpf_posteriors = pf_propagate_and_update(
            belief=initial_particle_belief,
            transition_model=t_dist,
            observation_model=o_dist,
            observations=[obs for obs in sim_observations],
        )

        if belief_plots_ok:
            fig, axes = plt.subplots(n_timesteps_prop, 3, figsize=(10, max(3 * n_timesteps_prop, 15)))
            fig.delaxes(axes[0, 0])  # no prior exists at k=0
            fig.suptitle(f"Beliefs at each time step ({figure_prefix}, sim {sim_idx + 1})")
            with torch.no_grad():
                for i in range(n_timesteps_prop):
                    if i > 0:
                        plot_belief(axes[i, 0], priors_plot[i - 1], x_range=(box_lows[0], box_highs[0]), y_range=(box_lows[1], box_highs[1]))
                        axes[i, 0].set_title(f"Prior at k={i}", fontsize=8)

                    plot_belief(axes[i, 1], posteriors_plot[i], x_range=(box_lows[0], box_highs[0]), y_range=(box_lows[1], box_highs[1]))
                    axes[i, 1].set_title(f"Posterior at k={i}", fontsize=8)

                    plot_particle_belief(axes[i, 2], wpf_posteriors[i], x_range=(box_lows[0], box_highs[0]), y_range=(box_lows[1], box_highs[1]))
                    axes[i, 2].set_title(f"WPF Posterior at k={i}", fontsize=8)
                    cols = [1, 2] if i == 0 else [0, 1, 2]
                    for j in cols:
                        if j > 0:  # Skip first prior plot
                            axes[i, j].scatter(sim_true_states[i, 0].item(), sim_true_states[i, 1].item(), marker="o", s=10, c="red")
                        if i > 0 and system.observation_dim() == 2:
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

    # Numerical comparison with avg_log_filter_score over an independent test batch.
    if n_test_sims > 0:
        test_state_trajectories: list[torch.Tensor] = []
        test_observation_trajectories: list[torch.Tensor] = []
        for _ in range(n_test_sims):
            sim_true_states, sim_observations = simulate(
                system,
                init_state_sampler,
                n_timesteps=n_timesteps_prop,
                device=device,
            )
            test_state_trajectories.append(sim_true_states)
            test_observation_trajectories.append(sim_observations)

        test_traj_data_state = [
            torch.stack([traj[k] for traj in test_state_trajectories], dim=0).to(device)
            for k in range(n_timesteps_prop)
        ]
        test_obs_data = [
            torch.stack([obs[k] for obs in test_observation_trajectories], dim=0).to(device)
            for k in range(n_timesteps_prop - 1)
        ]

        if use_dtf:
            with torch.no_grad():
                test_traj_data_learned = [trained_nftf(xk)[0] for xk in test_traj_data_state]
            learned_init_eval = init_model
        else:
            test_traj_data_learned = test_traj_data_state
            learned_init_eval = init_model

        learned_filter = Filter(
            transition_model=tran_model,
            observation_model=obs_model,
            prop_and_upd_fn=propagate.propagate_and_update,
        )
        learned_prior_scores, learned_posterior_scores = avg_log_filter_score(
            test_traj_data=test_traj_data_learned,
            test_obs_data=test_obs_data,
            filter=learned_filter,
            initial_belief=learned_init_eval,
        )

        initial_particle_belief_eval = WeightedParticleSet(
            particles=init_state_sampler(n_particles_true_pf).to(device=device),
            weights=torch.ones(n_particles_true_pf, device=device) / n_particles_true_pf,
        )
        particle_filter_model = Filter(
            transition_model=t_dist,
            observation_model=o_dist,
            prop_and_upd_fn=pf_propagate_and_update,
        )
        pf_prior_scores, pf_posterior_scores = avg_log_filter_score(
            test_traj_data=test_traj_data_state,
            test_obs_data=test_obs_data,
            filter=particle_filter_model,
            initial_belief=initial_particle_belief_eval,
        )

        print("=== avg_log_filter_score comparison ===")

        print("Learned posterior scores by timestep:")
        for k, score in enumerate(learned_posterior_scores):
            print(f"  k={k:02d}: {score.item():.4f}")

        print("True posterior scores by timestep:")
        for k, score in enumerate(pf_posterior_scores):
            print(f"  k={k:02d}: {score.item():.4f}")

        # Plot posterior score comparison over time.
        k_axis = np.arange(learned_posterior_scores.numel())
        learned_post_np = learned_posterior_scores.detach().cpu().numpy()
        pf_post_np = pf_posterior_scores.detach().cpu().numpy()
        delta_post_np = learned_post_np - pf_post_np

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.plot(k_axis, learned_post_np, marker="o", linewidth=1.8, label="Learned posterior")
        ax.plot(k_axis, pf_post_np, marker="s", linewidth=1.8, label="Particle posterior")
        ax.plot(k_axis, delta_post_np, linestyle="--", linewidth=1.2, label="Delta (learned - particle)")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Time step k")
        ax.set_ylabel("Average log filter score")
        ax.set_title("Posterior avg_log_filter_score over time")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        score_plot_path = figure_dir / f"{figure_prefix}__posterior_score_comparison.png"
        fig.savefig(score_plot_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved posterior score comparison plot to {score_plot_path}")

