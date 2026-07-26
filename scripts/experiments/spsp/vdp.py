import torch
from rational_factor.models.parameters import TrainableParameters, PositiveParameters, param_group_iter
from rational_factor.tools.analysis import check_pdf_valid, check_conditional_pdf_valid
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.factor_forms import QuadraticRFF, QuadraticFF, LinearRFF, LinearFF
import rational_factor.models.train as train
import rational_factor.models.loss as loss
import rational_factor.tools.propagate as propagate
from rational_factor.tools.visualization import plot_belief
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
import matplotlib.pyplot as plt


if __name__ == "__main__":
    problem = FULLY_OBSERVABLE_PROBLEMS["van_der_pol"]
    
    ###
    use_gpu = torch.cuda.is_available()
    n_basis = 500
    tran_params = {
        "n_epochs_per_group": [5, 5], # basis, weights
        "iterations": 10,
        "lr_basis": 5e-3,
        "lr_weights": 1e-2,
    }
    init_params = {
        "n_epochs_per_group": [20, 5], # basis, weights
        "iterations": 20,
        "lr_basis": 5e-3,
        "lr_weights": 1e-2,
    }

    batch_size = 256
    n_timesteps_prop = problem.n_timesteps
    var_reg_strength = 5e-1
    ###

    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using GPU: ", use_gpu)
    print("Device: ", device)

    system = problem.system
    x0 = problem.train_initial_state_data()
    x_k, x_kp1 = problem.train_state_transition_data()
    traj_data = problem.test_data()

    x0_dataloader = DataLoader(TensorDataset(x0), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(TensorDataset(x_kp1, x_k), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    # Create parameters
    phi_means = TrainableParameters.random_init(shape=(1, system.dim(), n_basis), mean=torch.tensor([0.0]), std=torch.tensor([5.0])).to(device)
    psi_means = TrainableParameters.random_init(shape=(1, system.dim(), n_basis), mean=torch.tensor([0.0]), std=torch.tensor([5.0])).to(device)
    psi0_means = TrainableParameters.random_init(shape=(1,system.dim(), n_basis), mean=torch.tensor([0.0]), std=torch.tensor([5.0])).to(device)
    phi_stds = PositiveParameters.random_init(shape=(1,system.dim(), n_basis), mean=torch.tensor([5.0]), std=torch.tensor([10.0]), epsilon=1e-1).to(device)
    psi_stds = PositiveParameters.random_init(shape=(1,system.dim(), n_basis), mean=torch.tensor([5.0]), std=torch.tensor([10.0]), epsilon=1e-1).to(device)
    psi0_stds = PositiveParameters.random_init(shape=(1, system.dim(), n_basis), mean=torch.tensor([5.0]), std=torch.tensor([10.0]), epsilon=1e-1).to(device)

    g_coeffs = PositiveParameters.random_init(shape=(1, n_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0]), epsilon=10.0).to(device)
    h0_coeffs = PositiveParameters.random_init(shape=(1, n_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0])).to(device)


    # Create basis functions
    g_basis = GaussianBasis(phi_means, phi_stds, coeffs=g_coeffs)
    psi_basis = GaussianBasis(psi_means, psi_stds)
    h0_basis = GaussianBasis(psi0_means, psi0_stds, coeffs=h0_coeffs)

    # Create and train the transition model
    tran_model = LinearRFF(g_basis, psi_basis)

    print("Training transition model")
    mle_loss_fn = loss.conditional_mle_loss
    
    rff_basis_params = param_group_iter([phi_means, phi_stds, psi_means, psi_stds])
    rff_weight_params = g_coeffs.parameters()
    optimizers ={"basis": torch.optim.Adam(rff_basis_params, lr=tran_params["lr_basis"]), "weights": torch.optim.Adam(rff_weight_params, lr=tran_params["lr_weights"])} 

    tran_model, best_loss_tran, training_time_tran = train.train_iterate(tran_model,
        xp_dataloader,
        {"mle": mle_loss_fn}, 
        optimizers,
        device=device,
        epochs_per_group=tran_params["n_epochs_per_group"],
        iterations=tran_params["iterations"],
        verbose=True,
        use_best="mle")
    print("Done! \n")

    # Freeze parameters of g
    phi_means.set_requires_grad(False)
    phi_stds.set_requires_grad(False)
    psi_means.set_requires_grad(False)
    psi_stds.set_requires_grad(False)
    g_coeffs.set_requires_grad(False)

    init_model = LinearFF.from_rff(tran_model, h0_basis).to(device)

    print("Training initial model")
    mle_loss_fn = loss.mle_loss

    ff_basis_params = param_group_iter([psi0_means, psi0_stds])
    ff_weight_params = h0_coeffs.parameters()
    optimizers = {"basis": torch.optim.Adam(ff_basis_params, lr=init_params["lr_basis"]), "weights": torch.optim.Adam(ff_weight_params, lr=init_params["lr_weights"])}

    init_model, best_loss_init, training_time_init = train.train_iterate(init_model, 
        x0_dataloader, 
        {"mle": mle_loss_fn}, 
        optimizers,
        device=device,
        epochs_per_group=init_params["n_epochs_per_group"],
        iterations=init_params["iterations"],
        verbose=True,
        use_best="mle")
    print("Done! \n")

    print(f"Transition model loss: {best_loss_tran:.4f}, training time: {training_time_tran:.2f} seconds")
    print(f"Initial model loss: {best_loss_init:.4f}, training time: {training_time_init:.2f} seconds")

    # Analysis
    analysis_device = torch.device("cpu")
    init_model = init_model.to(analysis_device).eval()
    tran_model = tran_model.to(analysis_device).eval()

    box_lows = tuple(problem.plot_bounds_low.tolist())
    box_highs = tuple(problem.plot_bounds_high.tolist())

    belief_seq = [belief.to(analysis_device).eval() for belief in propagate.propagate(init_model, tran_model, n_steps=n_timesteps_prop)]

    fig, axes = plt.subplots(2, n_timesteps_prop, figsize=(20, 10))
    fig.suptitle("Beliefs at each time step")
    for i in range(n_timesteps_prop):
        #print("Printing belief: ", i)
        check_pdf_valid(belief_seq[i], (box_lows, box_highs))
        plot_belief(axes[1, i], belief_seq[i], x_range=(box_lows[0], box_highs[0]), y_range=(box_lows[1], box_highs[1]))
        axes[0, i].scatter(traj_data[i][:, 0], traj_data[i][:, 1], s=1)
        axes[0, i].set_aspect("equal")
        axes[0, i].set_xlim(box_lows[0], box_highs[0])
        axes[0, i].set_ylim(box_lows[1], box_highs[1])
    

    output_dir = Path("figures/spsp/vdp")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "beliefs.png", dpi=1000)
    print(f"Saved beliefs to {output_dir / 'beliefs.png'}")
    #plt.show()


