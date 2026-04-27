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
from rational_factor.models.domain_transformation import MaskedRQSNFTF, IdentityTF, VolumePreservingNFTF
from rational_factor.models.density_model import LogisticSigmoid
from rational_factor.models.factor_forms import LinearFF, LinearRFF, LinearForm
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood
from rational_factor.tools.misc import data_bounds, train_test_split
from rational_factor.tools.visualization import plot_belief
from rational_factor.models.kde import GaussianKDE


def main() -> None:
    problem = FULLY_OBSERVABLE_PROBLEMS["van_der_pol"]

    ######## USER CONFIG ########
    use_gpu = torch.cuda.is_available()
    batch_size = 1024

    decorrupter_params = {
        "epochs": 100,
        "lr": 1e-3,
    }

    mover_params = {
        "n_epochs_per_group": [1],
        "iterations": 200,
        "lr_basis": 5e-2,
        "lr_weights": 1e-1,
        "lr_dtf": 1e-3,
    }

    init_params = {
        "n_epochs_per_group": [1],
        "iterations": 200,
        "lr_weights": 1e-1,
    }

    ls_temp = 0.1
    n_basis = 100

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
    optimizer = torch.optim.Adam(decorrupter.parameters(), lr=decorrupter_params["lr"], weight_decay=1e-3)
    dtf_concentration_loss_xk = lambda composite_model, x: 1.0 * loss.dtf_data_concentration_loss(composite_model.domain_tfs[0], x, concentration_point=loc, radius=scale)

    decorrupter, best_loss, training_time = train.train(
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
    y_kp1, _ = decorrupter_trained(x_kp1_data)
    y0, _ = decorrupter_trained(x0_data)
    
    ######## TRAIN MOVER ########
    print("Training mover")
    mover = VolumePreservingNFTF(system.dim(), trainable=True, hidden_features=128, n_layers=5).to(device)
    base_density_basis =  GaussianBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 20.0], device=device), variance=30.0, min_std=1e-4).to(device)
    mover_base_density = LinearForm(base_density_basis)
    mover_density = CompositeDensityModel([mover], mover_base_density).to(device)

    # Train on the decorrupted x' marginal data
    y_kp1_dataloader = DataLoader(TensorDataset(y_kp1), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    optimizers = {
        "basis_and_dtf": torch.optim.Adam([
            {"params": mover_density.conditional_density_model.basis_params(), "lr": mover_params["lr_basis"]},
            {"params": mover_density.domain_tfs.parameters(), "lr": mover_params["lr_dtf"]},
        ]),
        "weights": torch.optim.Adam(mover_density.density_model.weight_params(), lr=mover_params["lr_weights"]),
    }
    mover, best_loss, training_time = train.train_iterate(
        mover_density,
        y_kp1_dataloader,
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
