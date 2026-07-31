import torch
from rational_factor.models.parameters import TrainableParameters, PositiveParameters, param_group_iter
from rational_factor.tools.analysis import avg_log_likelihood, check_pdf_valid, check_conditional_pdf_valid
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.sum_prod_basis import SumProdBasis
from rational_factor.models.factor_forms import QuadraticRFF, QuadraticFF, LinearRFF, LinearFF, SumProdRFF
import rational_factor.models.train as train
import rational_factor.models.loss as loss
import rational_factor.tools.propagate as propagate
from rational_factor.tools.visualization import plot_belief
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
import matplotlib.pyplot as plt


if __name__ == "__main__":
    problem = FULLY_OBSERVABLE_PROBLEMS["cartpole"]
    
    ###
    sp_basis = True
    sp_model = False
    e2e = False

    use_gpu = torch.cuda.is_available()
    n_leaf_basis = 80
    n_output_basis = 600
    tran_params = {
        "n_epochs_per_group": [5, 5], # basis, weights
        "iterations": 20,
        "lr_basis": 5e-3,
        "lr_weights": 1e-1,
    }
    init_params = {
        "n_epochs_per_group": [20, 5], # basis, weights
        "iterations": 50,
        "lr_basis": 5e-3,
        "lr_weights": 5e-2,
    }
    tran_params_e2e = {
        "n_epochs_per_group": [1], # basis, weights
        "iterations": 200,
        "lr": 5e-3,
    }
    init_params_e2e = {
        "n_epochs_per_group": [1], # basis, weights
        "iterations": 400,
        "lr": 5e-3,
    }

    batch_size = 64
    n_timesteps_prop = problem.n_timesteps
    ###

    device = torch.device("cuda" if use_gpu else "cpu")
    print("System: ", problem.system.__class__.__name__)
    print("SP basis: ", sp_basis)
    print("SP model: ", sp_model)
    print("E2E: ", e2e)
    print("Using GPU: ", use_gpu)
    print("Device: ", device)

    system = problem.system
    x0 = problem.train_initial_state_data()
    x_k, x_kp1 = problem.train_state_transition_data()
    traj_data = problem.test_data()

    x0_dataloader = DataLoader(TensorDataset(x0), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(TensorDataset(x_kp1, x_k), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    # Create parameters
    phi_means = TrainableParameters.random_init(shape=(1, system.dim(), n_leaf_basis), mean=torch.tensor([0.0]), std=torch.tensor([5.0])).to(device)
    psi_means = TrainableParameters.random_init(shape=(1, system.dim(), n_leaf_basis), mean=torch.tensor([0.0]), std=torch.tensor([5.0])).to(device)
    psi0_means = TrainableParameters.random_init(shape=(1,system.dim(), n_leaf_basis), mean=torch.tensor([0.0]), std=torch.tensor([5.0])).to(device)
    phi_stds = PositiveParameters.random_init(shape=(1,system.dim(), n_leaf_basis), mean=torch.tensor([5.0]), std=torch.tensor([10.0]), epsilon=1e-1).to(device)
    psi_stds = PositiveParameters.random_init(shape=(1,system.dim(), n_leaf_basis), mean=torch.tensor([5.0]), std=torch.tensor([10.0]), epsilon=1e-1).to(device)
    psi0_stds = PositiveParameters.random_init(shape=(1, system.dim(), n_leaf_basis), mean=torch.tensor([5.0]), std=torch.tensor([10.0]), epsilon=1e-1).to(device)


    if sp_basis:
        g_coeffs = PositiveParameters.random_init(shape=(1, n_output_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0]), epsilon=1e-3).to(device)
        h0_coeffs = PositiveParameters.random_init(shape=(1, n_output_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0]), epsilon=1e-3).to(device)

        phi_matrix_coeffs = PositiveParameters.random_init(shape=(1, system.dim(), n_output_basis, n_leaf_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0]), epsilon=1e-6).to(device)
        psi_matrix_coeffs = PositiveParameters.random_init(shape=(1, system.dim(), n_output_basis, n_leaf_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0]), epsilon=1e-6).to(device)
        psi0_matrix_coeffs = PositiveParameters.random_init(shape=(1, system.dim(), n_output_basis, n_leaf_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0]), epsilon=1e-6).to(device)

        phi_leaf_basis = GaussianBasis(phi_means, phi_stds)
        psi_leaf_basis = GaussianBasis(psi_means, psi_stds)
        psi0_leaf_basis = GaussianBasis(psi0_means, psi0_stds)

        g_basis = SumProdBasis(n_output_basis, phi_leaf_basis, phi_matrix_coeffs, coeffs=g_coeffs)
        psi_basis = SumProdBasis(n_output_basis, psi_leaf_basis, psi_matrix_coeffs)
        h0_basis = SumProdBasis(n_output_basis, psi0_leaf_basis, psi0_matrix_coeffs, coeffs=h0_coeffs)

    else:
        g_coeffs = PositiveParameters.random_init(shape=(1, n_leaf_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0]), epsilon=1e-3).to(device)
        h0_coeffs = PositiveParameters.random_init(shape=(1, n_leaf_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0]), epsilon=1e-3).to(device)

        g_basis = GaussianBasis(phi_means, phi_stds, coeffs=g_coeffs)
        psi_basis = GaussianBasis(psi_means, psi_stds)
        h0_basis = GaussianBasis(psi0_means, psi0_stds, coeffs=h0_coeffs)
    
    if sp_model:
        B = PositiveParameters.random_init(shape=(1, n_output_basis, n_output_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0])).to(device)
        P = PositiveParameters.random_init(shape=(1, n_output_basis, n_output_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0])).to(device)

        # Create and train the transition model
        tran_model = SumProdRFF(g_basis, psi_basis, B, P)
    else:
        tran_model = LinearRFF(g_basis, psi_basis)

    print("Training transition model")
    mle_loss_fn = loss.conditional_mle_loss
    
    if e2e:
        param_list = [phi_means, phi_stds, psi_means, psi_stds, g_coeffs]
        if sp_basis:
            param_list.extend([phi_matrix_coeffs, psi_matrix_coeffs])
        if sp_model:
            param_list.extend([B, P])
        optimizers ={"basis": torch.optim.Adam(param_group_iter(param_list), lr=tran_params_e2e["lr"])}
        epochs_per_group = tran_params_e2e["n_epochs_per_group"]
        iterations = tran_params_e2e["iterations"]
    else:
        basis_params = param_group_iter([phi_means, phi_stds, psi_means, psi_stds])
        weight_param_list = [g_coeffs]
        if sp_basis:
            weight_param_list.extend([phi_matrix_coeffs, psi_matrix_coeffs])
        if sp_model:
            weight_param_list.extend([B, P])
        weight_params = param_group_iter(weight_param_list)
        optimizers ={"basis": torch.optim.Adam(basis_params, lr=tran_params["lr_basis"]), "weights": torch.optim.Adam(weight_params, lr=tran_params["lr_weights"])} 
        epochs_per_group = tran_params["n_epochs_per_group"]
        iterations = tran_params["iterations"]

    tran_model, best_loss_tran, training_time_tran = train.train_iterate(tran_model,
        xp_dataloader,
        {"mle": mle_loss_fn}, 
        optimizers,
        device=device,
        epochs_per_group=epochs_per_group,
        iterations=iterations,
        verbose=True,
        use_best="mle")
    print("Done! \n")

    # Freeze parameters of g
    phi_means.set_requires_grad(False)
    phi_stds.set_requires_grad(False)
    psi_means.set_requires_grad(False)
    psi_stds.set_requires_grad(False)
    g_coeffs.set_requires_grad(False)

    if sp_basis:
        phi_matrix_coeffs.set_requires_grad(False)
        psi_matrix_coeffs.set_requires_grad(False)
    if sp_model:
        B.set_requires_grad(False)
        P.set_requires_grad(False)

    init_model = LinearFF.from_rff(tran_model, h0_basis).to(device)

    print("Training initial model")
    mle_loss_fn = loss.mle_loss

    if e2e:
        if sp_basis:
            params = param_group_iter([psi0_means, psi0_stds, h0_coeffs, psi0_matrix_coeffs])
        else:
            params = param_group_iter([psi0_means, psi0_stds, h0_coeffs])
        optimizers = {"basis": torch.optim.Adam(params, lr=init_params_e2e["lr"])}
        epochs_per_group = init_params_e2e["n_epochs_per_group"]
        iterations = init_params_e2e["iterations"]
    else:
        ff_basis_params = param_group_iter([psi0_means, psi0_stds])
        ff_weight_params = h0_coeffs.parameters() if not sp_basis else param_group_iter([h0_coeffs, psi0_matrix_coeffs])
        optimizers = {"basis": torch.optim.Adam(ff_basis_params, lr=init_params["lr_basis"]), "weights": torch.optim.Adam(ff_weight_params, lr=init_params["lr_weights"])}
        epochs_per_group = init_params["n_epochs_per_group"]
        iterations = init_params["iterations"]

    init_model, best_loss_init, training_time_init = train.train_iterate(init_model, 
        x0_dataloader, 
        {"mle": mle_loss_fn}, 
        optimizers,
        device=device,
        epochs_per_group=epochs_per_group,
        iterations=iterations,
        verbose=True,
        use_best="mle")
    print("Done! \n")

    print(f"Transition model loss: {best_loss_tran:.4f}, training time: {training_time_tran:.2f} seconds")
    print(f"Initial model loss: {best_loss_init:.4f}, training time: {training_time_init:.2f} seconds")

    # Analysis
    #analysis_device = torch.device("cpu")
    analysis_device = torch.device("cuda")
    init_model = init_model.to(analysis_device).eval()
    tran_model = tran_model.to(analysis_device).eval()

    box_lows = tuple(problem.plot_bounds_low.tolist())
    box_highs = tuple(problem.plot_bounds_high.tolist())

    belief_seq = [belief.to(analysis_device).eval() for belief in propagate.propagate(init_model, tran_model, n_steps=n_timesteps_prop)]

    ll_per_step = []
    print("Finished propagating beliefs")
    print("System: ", problem.system.__class__.__name__)
    print("SP basis: ", sp_basis)
    print("SP model: ", sp_model)
    print("E2E: ", e2e)
    print("Number of leaf basis functions: ", n_leaf_basis)
    print("Number of output basis functions: ", n_output_basis)
    print("--------------------------------")
    print("Avg log-likelihood per step:")
    for i in range(n_timesteps_prop):
        data_i = traj_data[i].to(analysis_device)
        ll = avg_log_likelihood(belief_seq[i], data_i)
        ll_per_step.append(float(ll.detach().cpu()))
        print(f"  {i}: {ll_per_step[-1]:.6f}")



