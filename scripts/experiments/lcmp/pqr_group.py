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
from rational_factor.models.domain_transformation import MaskedAffineNFTF, IdentityTF, VolumePreservingNFTF, StackedTF
from rational_factor.models.density_model import LogisticSigmoid
from rational_factor.models.factor_forms import LinearFF, LinearRFF, LinearForm
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood
from rational_factor.tools.misc import data_bounds, train_test_split
from rational_factor.tools.visualization import plot_belief
from rational_factor.models.kde import GaussianKDE


def main() -> None:
    problem = FULLY_OBSERVABLE_PROBLEMS["planar_quadrotor"]

    ######## USER CONFIG ########
    use_gpu = torch.cuda.is_available()
    batch_size = 2048

    decorrupter_params = {
        "epochs": 50,
        "lr": 1e-3,
        "weight_decay": 1e-3,
    }

    mover_params = {
        "n_epochs_per_group": [5, 5],
        "iterations": 20,
        "lr_basis": 1e-4,
        "lr_weights": 1e-4,
        "lr_dtf": 1e-3,
    }

    init_params = {
        "n_epochs_per_group": [15, 5],
        "iterations": 100,
        "lr_basis": 5e-2,
        "lr_weights": 1e-1,
    }

    ls_temp = 0.1
    n_basis = 1000
    mover_iterations = 5
    reg_covar = 5e-2

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
    decorrupter = MaskedAffineNFTF(system.dim(), trainable=True, hidden_features=128, n_layers=5).to(device)
    decorrupter_density = CompositeDensityModel([decorrupter], base_distribution).to(device)

    print("Training NF decorrupter")
    optimizer = torch.optim.Adam(decorrupter.parameters(), lr=decorrupter_params["lr"], weight_decay=decorrupter_params["weight_decay"])
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

    decorrupter_trained = MaskedAffineNFTF.copy_from_trainable(decorrupter).to(device)

    x_k_data = x_k_data.to(device)
    x_kp1_data = x_kp1_data.to(device)
    x0_data = x0_data.to(device)
    y_k_data, _ = decorrupter_trained(x_k_data)
    y_kp1_data, _ = decorrupter_trained(x_kp1_data)
    y0_data, _ = decorrupter_trained(x0_data)
    
    ######## TRAIN MOVER ########
    print("Training mover")
    mover = VolumePreservingNFTF(system.dim(), trainable=True, hidden_features=256, n_layers=6).to(device)
    
    # Initialize base density to LF GMM
    y_joint_data = torch.cat([y_k_data, y_kp1_data], dim=1)

    # Train on the decorrupted x' marginal data
    z_joint_data = y_joint_data
    gmm_lf = train.fit_gaussian_lf_em(z_joint_data.to(torch.device("cpu")), n_components=n_basis, reg_covar=reg_covar, max_iter=200)

    weights = gmm_lf.get_w()
    z_marginal = gmm_lf.basis.marginal(marginal_dims=range(system.dim()))
    z_means, z_stds = z_marginal.means_stds()
    z_params = torch.stack([z_means, z_stds], dim=-1)

    zp_marginal = gmm_lf.basis.marginal(marginal_dims=range(system.dim(), 2 * system.dim()))
    zp_means, zp_stds = zp_marginal.means_stds()
    zp_params = torch.stack([zp_means, zp_stds], dim=-1)
    
    phi_basis = GaussianBasis(fixed_params=z_params).to(device)
    psi_basis = GaussianBasis(fixed_params=zp_params).to(device)
    lrff = LinearRFF(phi_basis, psi_basis, a_fixed=weights.to(device)).to(device)

    mover_density = CompositeConditionalModel([mover], lrff).to(device)

    yp_dataloader = DataLoader(TensorDataset(y_kp1_data.to(torch.device("cpu")), y_k_data.to(torch.device("cpu"))), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    optimizers = {
        "dtf_and_basis": torch.optim.Adam([
            {"params": mover_density.conditional_density_model.basis_params(), "lr": mover_params["lr_basis"]},
            {"params": mover_density.domain_tfs[0].parameters(), "lr": mover_params["lr_dtf"], "weight_decay": 1e-4},
        ]),
        "weights": torch.optim.Adam(mover_density.conditional_density_model.weight_params(), lr=mover_params["lr_weights"]),
    }
    mover_density, best_loss, training_time = train.train_iterate(
        mover_density,
        yp_dataloader,
        {"mle": loss.conditional_mle_loss},
        optimizers,
        epochs_per_group=mover_params["n_epochs_per_group"],
        iterations=mover_params["iterations"],
        verbose=True,
        use_best="mle",
        clip_grad_norm=5.0,
        restore_loss_threshold=50.0,
    )
    print("Done.\n")

    mover_trained = VolumePreservingNFTF.copy_from_trainable(mover).to(device)
    mover_joint = StackedTF([mover_trained, mover_trained])
    #z_joint_data, _ = mover_joint(y_joint_data)

    psi0_basis = GaussianBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 20.0], device=device), variance=30.0, min_std=1e-4).to(device)

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
    base_belief_seq = propagate.propagate(ff, lrff, n_steps=problem.n_timesteps)
    belief_seq = [CompositeDensityModel([decorrupter_trained,mover_trained], belief) for belief in base_belief_seq]

    for i in range(problem.n_timesteps):
        data_i = test_traj_data[i].to(device)
        ll = avg_log_likelihood(belief_seq[i], data_i)
        print(f"Log likelihood at time {i}: {ll:.4f}")


if __name__ == "__main__":
    main()

