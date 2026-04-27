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
from rational_factor.models.domain_transformation import MaskedRQSNFTF
from rational_factor.models.factor_forms import LinearFF, LinearRFF
from rational_factor.models.density_model import LogisticSigmoid
from rational_factor.models.kde import GaussianKDE
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood, check_pdf_valid
from rational_factor.tools.misc import data_bounds, train_test_split
from rational_factor.tools.visualization import plot_marginal_trajectory_comparison



def main() -> None:
    problem = FULLY_OBSERVABLE_PROBLEMS["planar_quadrotor"]

    ######## USER CONFIG ########
    use_gpu = torch.cuda.is_available()
    batch_size = 1024

    dtf_params = {
        "epochs": 10,
        "lr": 1e-3,
    }
    ls_temp = 0.1

    init_params = {
        "n_epochs_per_group": [1],
        "iterations": 50,
        "lr_weights": 1e-1,
    }

    kernel_scale_tran = 0.3
    kernel_scale_init = 0.7

    bw_val_thresh_tran = 20.0
    bw_val_thresh_init = 5.0
    bw_val_lr_tran = 1e-2
    bw_val_lr_init = 1e-2

    ######## SETUP ########
    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using device: ", device)

    system = problem.system
    x0_data = problem.train_initial_state_data()
    x_k_data, x_kp1_data = problem.train_state_transition_data()
    test_traj_data = problem.test_data()

    x_dataloader = DataLoader(TensorDataset(x_k_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    x0_dataloader = DataLoader(TensorDataset(x0_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    loc, scale = data_bounds(x_k_data, mode="center_lengths")
    loc = loc.to(device)
    scale = scale.to(device)
    base_distribution = LogisticSigmoid(system.dim(), temperature=ls_temp, loc=loc, scale=scale)
    dtf = MaskedRQSNFTF(system.dim(), trainable=True, hidden_features=256, n_layers=5).to(device)
    decorrupter = CompositeDensityModel([dtf], base_distribution).to(device)

    ######## TRAIN LCM ########
    print("Training NF decorrupter")
    optimizer = torch.optim.Adam(decorrupter.parameters(), lr=dtf_params["lr"], weight_decay=1e-3)

    decorrupter, best_loss, training_time = train.train(
        decorrupter,
        x_dataloader,
        {"mle": loss.mle_loss},
        optimizer,
        epochs=dtf_params["epochs"],
        verbose=True,
        use_best="mle",
        clip_grad_norm=5.0,
        restore_loss_threshold=50.0,
    )
    print("Done.\n")

    dtf_trained = MaskedRQSNFTF.copy_from_trainable(dtf).to(device)


    x_k_data = x_k_data.to(device)
    x_kp1_data = x_kp1_data.to(device)
    x0_data = x0_data.to(device)
    with torch.no_grad():
        x_k_transformed, _ = dtf_trained(x_k_data)
        x_kp1_transformed, _ = dtf_trained(x_kp1_data)
        x0_transformed, _ = dtf_trained(x0_data)
    
    # Optimize joint bandwidth
    joint_data = torch.cat([x_k_transformed, x_kp1_transformed], dim=1)
    joint_train_data, joint_val_data = train_test_split(joint_data, test_size=0.1)
    joint_kde = GaussianKDE(joint_train_data)
    best_bandwidth_joint = joint_kde.bandwidth
    #best_bandwidth_joint, _ = joint_kde.fit_bandwidth_validation_mle(joint_val_data, 
    #        epochs=200, 
    #        threshold=bw_val_thresh_tran, 
    #        lr=bw_val_lr_tran, 
    #        min_step=1e-3, 
    #        verbose=True, 
    #        block_size=128)
    best_bandwidth_joint *= kernel_scale_tran

    # Optimize init bandwidth
    init_train_data, init_val_data = train_test_split(x0_transformed, test_size=0.1)
    init_kde = GaussianKDE(init_train_data)
    best_bandwidth_init = init_kde.bandwidth
    #best_bandwidth_init, _ = init_kde.fit_bandwidth_validation_mle(init_val_data, 
    #        epochs=200, 
    #        threshold=bw_val_thresh_init, 
    #        lr=bw_val_lr_init, 
    #        min_step=1e-3, 
    #        verbose=True, 
    #        block_size=128)
    best_bandwidth_init *= kernel_scale_init

    phi_basis = GaussianKernelBasis(x_k_transformed, kernel_bandwidth=best_bandwidth_joint, trainable=False)
    psi_basis = GaussianKernelBasis(x_kp1_transformed, kernel_bandwidth=best_bandwidth_joint, trainable=False)
    psi0_basis = GaussianKernelBasis(x0_transformed, kernel_bandwidth=best_bandwidth_init, trainable=True)

    lrff = LinearRFF(phi_basis, psi_basis, numerical_tolerance=problem.numerical_tolerance).to(device)
    tran_model = CompositeConditionalModel([dtf_trained], lrff).to(device)

    ff = LinearFF.from_rff(lrff, psi0_basis).to(device)
    init_model = CompositeDensityModel([dtf_trained], ff).to(device)

    print("Training initial model weights")
    optimizers = { "weights": torch.optim.Adam([{"params":init_model.density_model.weight_params(),"lr": init_params["lr_weights"]}]) }
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
    dtf_trained.to(cpu)
    print("Propagating beliefs...")
    base_belief_seq = propagate.propagate(
        init_model.density_model,
        tran_model.conditional_density_model,
        n_steps=problem.n_timesteps,
        device=cpu,
    )
    print("Done.\n")
    belief_seq = [CompositeDensityModel([dtf_trained], belief) for belief in base_belief_seq]

    ll_per_step = []
    for i in range(problem.n_timesteps):
        data_i = test_traj_data[i].to(cpu)
        ll = avg_log_likelihood(belief_seq[i], data_i)
        ll_per_step.append(ll.detach().cpu().reshape(()))
        print(f"Avg log-likelihood at time {i}: {float(ll_per_step[-1]):.6f}")
    prop_ll_vector = torch.stack(ll_per_step).to(dtype=torch.float32)

    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    ll_out_path = out_dir / f"{Path(__file__).stem}__ll.png"
    plt.figure(figsize=(8, 4))
    plt.plot(prop_ll_vector.numpy(), marker="o")
    plt.xlabel("timestep")
    plt.ylabel("avg log-likelihood")
    plt.title(f"{Path(__file__).stem}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(ll_out_path, dpi=200)
    print(f"Saved LL plot to {ll_out_path}")


if __name__ == "__main__":
    main()
