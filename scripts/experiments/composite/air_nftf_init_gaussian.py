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
from rational_factor.tools.analysis import avg_log_likelihood
from rational_factor.models.domain_transformation import MaskedAffineNFTF
from rational_factor.models.composite_model import CompositeDensityModel, CompositeConditionalModel
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
import matplotlib.pyplot as plt


if __name__ == "__main__":
    problem = FULLY_OBSERVABLE_PROBLEMS["aircraft"]
    
    ###
    use_gpu = torch.cuda.is_available()
    use_dtf = True
    n_basis = 500
    if use_dtf:
        tran_params = {
            "n_epochs_per_group": [5, 5], # dtf_params and basis, weights
            "iterations": 40,
            "pre_train_epochs": 2,
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

    if len(belief_seq) != len(traj_data):
        raise RuntimeError(
            f"belief_seq length ({len(belief_seq)}) must match traj_data length ({len(traj_data)})"
        )

    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    param_dtype = next(belief_seq[0].parameters()).dtype
    traj_eval = [t.to(device=analysis_device, dtype=param_dtype) for t in traj_data]

    log_liks = [
        avg_log_likelihood(belief, traj_eval[k]).item()
        for k, belief in enumerate(belief_seq)
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(len(log_liks)), log_liks, marker="o")
    ax.set_xlabel("Time step $k$")
    ax.set_ylabel("Mean log likelihood")
    ax.set_title(r"Test trajectories: $\mathbb{E}[\log p_{\mathrm{belief}_k}(x_k)]$ under propagated beliefs")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = figures_dir / "air_nftf_init_gaussian_belief_loglik.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved log-likelihood plot to {out_path}")

