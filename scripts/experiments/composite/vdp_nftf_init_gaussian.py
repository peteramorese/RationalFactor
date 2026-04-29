import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.factor_forms import LinearRFF, LinearFF
import rational_factor.models.train as train
import rational_factor.models.loss as loss
from rational_factor.models.density_model import LogisticSigmoid
import rational_factor.tools.propagate as propagate
from rational_factor.tools.misc import data_bounds, train_test_split
from rational_factor.tools.visualization import plot_belief
from rational_factor.models.domain_transformation import MaskedAffineNFTF
from rational_factor.models.composite_model import CompositeDensityModel, CompositeConditionalModel
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    problem = FULLY_OBSERVABLE_PROBLEMS["van_der_pol"]
    
    ###
    use_gpu = torch.cuda.is_available()
    use_dtf = True
    n_basis = 500
    if use_dtf:
        tran_params = {
            "n_epochs_per_group": [5, 5], # dtf_params and basis, weights
            "iterations": 40,
            "pre_train_epochs": 10,
            "lr_basis": 5e-3,
            "lr_weights": 1e-3,
            "lr_dtf": 5e-4,
        }
        init_params = {
            "n_epochs_per_group": [20, 5], # basis, weights
            "iterations": 50,
            "lr_basis": 5e-3,
            "lr_weights": 1e-3,
        }
    else:
        tran_params = {
            "n_epochs_per_group": [5, 5], # basis, weights
            "iterations": 40,
            "lr_basis": 5e-3,
            "lr_weights": 1e-3,
        }
        init_params = {
            "n_epochs_per_group": [20, 5], # basis, weights
            "iterations": 50,
            "lr_basis": 5e-3,
            "lr_weights": 1e-3,
        }

    train_batch_size = 256
    val_batch_size = 4096
    n_timesteps_prop = problem.n_timesteps
    reg_covar_joint = 1e-3
    ls_temp = 0.1
    ###

    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using GPU: ", use_gpu)
    print("Device: ", device)

    system = problem.system
    x0_data = problem.train_initial_state_data()
    x_k_data, x_kp1_data = problem.train_state_transition_data()
    traj_data = problem.test_data()

    x0_train, x0_val = train_test_split(x0_data, test_size=0.2)
    x_k_train, x_k_val, x_kp1_train, x_kp1_val = train_test_split(x_k_data, x_kp1_data, test_size=0.2)

    # Training data loaders
    x0_dataloader = DataLoader(TensorDataset(x0_train), batch_size=train_batch_size, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(TensorDataset(x_kp1_train, x_k_train), batch_size=train_batch_size, shuffle=True, pin_memory=use_gpu)
    x_k_dataloader = DataLoader(TensorDataset(x_k_data), batch_size=train_batch_size, shuffle=True, pin_memory=use_gpu)

    # Validation data loaders
    x0_val_dataloader = DataLoader(TensorDataset(x0_val), batch_size=val_batch_size, shuffle=True, pin_memory=use_gpu)
    xp_val_dataloader = DataLoader(TensorDataset(x_kp1_val, x_k_val), batch_size=val_batch_size, shuffle=True, pin_memory=use_gpu)

    # Create separable domain transformation
    nftf = MaskedAffineNFTF(system.dim(), trainable=True, hidden_features=256, n_layers=8).to(device) if use_dtf else None

    # Pre train the dtf
    if use_dtf:
        print("Pre training the dtf")
        loc, scale = data_bounds(x_k_data, mode="center_lengths")
        loc = loc.to(device)
        scale = scale.to(device)
        base_distribution = LogisticSigmoid(system.dim(), temperature=ls_temp, loc=loc, scale=scale)
        decorrupter_density = CompositeDensityModel([nftf], base_distribution).to(device)
        optimizer = torch.optim.Adam(nftf.parameters(), lr=tran_params["lr_dtf"], weight_decay=tran_params["lr_dtf"])
        decorrupter_density, best_loss, training_time = train.train(
            decorrupter_density,
            x_k_dataloader,
            {"mle": loss.mle_loss},
            optimizer,
            epochs=tran_params["pre_train_epochs"],
            verbose=True,
            use_best="mle",
        )
        print("Done.\n")

    # Prefit the basis functions
    print("Prefitting the basis functions")
    with torch.no_grad():
        y_k_data, _ = nftf(x_k_data.to(device))
        y_kp1_data, _ = nftf(x_kp1_data.to(device))
        y0_data, _ = nftf(x0_data.to(device))
        y_k_data = y_k_data.to(torch.device("cpu"))
        y_kp1_data = y_kp1_data.to(torch.device("cpu"))
        y0_data = y0_data.to(torch.device("cpu"))
    y_joint_data = torch.cat([y_k_data, y_kp1_data], dim=1)
    gmm_lf = train.fit_gaussian_lf_em(y_joint_data.to(torch.device("cpu")), n_components=n_basis, reg_covar=reg_covar_joint, max_iter=100)
    weights = gmm_lf.get_w()
    phi_marginal = gmm_lf.basis.marginal(marginal_dims=range(system.dim()))
    phi_means, phi_stds = phi_marginal.means_stds()
    phi_params = torch.stack([phi_means, phi_stds], dim=-1)

    psi_marginal = gmm_lf.basis.marginal(marginal_dims=range(system.dim(), 2 * system.dim()))
    psi_means, psi_stds = psi_marginal.means_stds()
    psi_params = torch.stack([psi_means, psi_stds], dim=-1)

    phi_basis = GaussianBasis(uparams_init=phi_params).to(device)
    psi_basis = GaussianBasis(uparams_init=psi_params).to(device)
    psi0_basis = GaussianBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 20.0], device=device), variance=30.0, min_std=1e-4).to(device)
    print("Done.\n")

    # Create and train the transition model
    if use_dtf:
        tran_model = CompositeConditionalModel([nftf], LinearRFF(phi_basis, psi_basis)).to(device)
    else:
        tran_model = LinearRFF(phi_basis, psi_basis).to(device)

    print("Training transition model")
    
    if use_dtf:
        optimizers ={"dtf_and_basis": torch.optim.Adam([{'params': tran_model.conditional_density_model.basis_params(), 'lr': tran_params["lr_basis"]}, {'params':tran_model.domain_tfs.parameters(), 'lr': tran_params["lr_dtf"]}]), 
            "weights": torch.optim.Adam(tran_model.conditional_density_model.weight_params(), lr=tran_params["lr_weights"])} 
    else:
        optimizers ={"basis": torch.optim.Adam(tran_model.basis_params(), lr=tran_params["lr_basis"]), "weights": torch.optim.Adam(tran_model.weight_params(), lr=tran_params["lr_weights"])} 

    tran_model, best_loss_tran, training_time_tran = train.train_iterate(tran_model,
        xp_dataloader,
        labeled_loss_fns={"mle": loss.conditional_mle_loss}, 
        labeled_optimizers=optimizers,
        labeled_validation_loss_fns={"val_mle": loss.conditional_mle_loss},
        validation_data_loader=xp_val_dataloader,
        epochs_per_group=tran_params["n_epochs_per_group"],
        iterations=tran_params["iterations"],
        verbose=True,
        use_best="val_mle")
    print("Done! \n")
    print("Valid: ", tran_model.valid())


    # Copy the domain transformation to fix it for training the initial state model
    trained_nftf = MaskedAffineNFTF.copy_from_trainable(nftf).to(device) if use_dtf else None

    if use_dtf:
        init_model = CompositeDensityModel([trained_nftf], LinearFF.from_rff(tran_model.conditional_density_model, psi0_basis)).to(device)
    else:
        init_model = LinearFF.from_rff(tran_model, psi0_basis).to(device)

    print("Training initial model")

    if use_dtf:
        optimizers = {"basis": torch.optim.Adam(init_model.density_model.basis_params(), lr=init_params["lr_basis"]), "weights": torch.optim.Adam(init_model.density_model.weight_params(), lr=init_params["lr_weights"])}
    else:
        optimizers = {"basis": torch.optim.Adam(init_model.basis_params(), lr=init_params["lr_basis"]), "weights": torch.optim.Adam(init_model.weight_params(), lr=init_params["lr_weights"])}

    init_model, best_loss_init, training_time_init = train.train_iterate(init_model, 
        x0_dataloader, 
        labeled_loss_fns={"mle": loss.mle_loss}, 
        labeled_optimizers=optimizers,
        labeled_validation_loss_fns={"val_mle": loss.mle_loss},
        validation_data_loader=x0_val_dataloader,
        epochs_per_group=init_params["n_epochs_per_group"],
        iterations=init_params["iterations"],
        verbose=True,
        use_best="val_mle")
    print("Done! \n")

    print(f"Transition model loss: {best_loss_tran:.4f}, training time: {training_time_tran:.2f} seconds")
    print(f"Initial model loss: {best_loss_init:.4f}, training time: {training_time_init:.2f} seconds")

    # Analysis
    analysis_device = torch.device("cpu")
    init_model = init_model.to(analysis_device).eval()
    tran_model = tran_model.to(analysis_device).eval() if use_dtf else tran_model.to(analysis_device).eval()
    if use_dtf:
        trained_nftf = trained_nftf.to(analysis_device).eval()

    box_lows = tuple(problem.plot_bounds_low.tolist())
    box_highs = tuple(problem.plot_bounds_high.tolist())

    if use_dtf:
        base_belief_seq = propagate.propagate(init_model.density_model, tran_model.conditional_density_model, n_steps=n_timesteps_prop)
        belief_seq = [CompositeDensityModel([trained_nftf], belief).to(analysis_device).eval() for belief in base_belief_seq]
    else:
        belief_seq = [belief.to(analysis_device).eval() for belief in propagate.propagate(init_model, tran_model, n_steps=n_timesteps_prop)]

    if use_dtf:
        out_dir = Path("figures")
        out_dir.mkdir(parents=True, exist_ok=True)
        grid_morph_out_path = out_dir / "vdp_nftf_group_gaussian__state_vs_latent_grid.png"
        _plot_state_vs_latent_grid(
            trained_nftf,
            x_k_data.detach().cpu(),
            grid_morph_out_path,
            n_grid_lines=11,
            n_points_per_line=280,
            quantile_pad=0.01,
            title="van_der_pol nftf_group_gaussian: state-space vs latent-space grid",
        )
        print(f"Saved state-vs-latent grid plot to {grid_morph_out_path}")

    fig, axes = plt.subplots(2, n_timesteps_prop, figsize=(20, 10))
    fig.suptitle("Beliefs at each time step")
    for i in range(n_timesteps_prop):
        #print("Printing belief: ", i)
        plot_belief(axes[1, i], belief_seq[i], x_range=(box_lows[0], box_highs[0]), y_range=(box_lows[1], box_highs[1]))
        axes[0, i].scatter(traj_data[i][:, 0], traj_data[i][:, 1], s=1)
        axes[0, i].set_aspect("equal")
        axes[0, i].set_xlim(box_lows[0], box_highs[0])
        axes[0, i].set_ylim(box_lows[1], box_highs[1])
    

    plt.savefig("figures/vdp_nfdf_gaussian_beliefs.png", dpi=1000)
    print(f"Saved beliefs to figures/vdp_nfdf_gaussian_beliefs.png")
    #plt.show()


