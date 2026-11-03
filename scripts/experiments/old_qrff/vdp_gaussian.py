import torch
from torch.utils.data import DataLoader, TensorDataset
from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.factor_forms import QuadraticRFF, QuadraticFF
import rational_factor.models.train as train
import rational_factor.models.loss as loss
import rational_factor.tools.propagate as propagate
from rational_factor.tools.visualization import plot_belief
from rational_factor.tools.analysis import mc_integral_box

import matplotlib.pyplot as plt

from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS


if __name__ == "__main__":
    problem = FULLY_OBSERVABLE_PROBLEMS["van_der_pol"]
    
    ###
    use_gpu = torch.cuda.is_available()
    n_basis = 35
    n_epochs = 300
    batch_size = 512
    learning_rate = 1e-3
    n_timesteps_prop = problem.n_timesteps
    var_reg_strength = 0.5
    psd_reg_strength = 1e-1 #0.002
    ###

    system = problem.system
    x0_train, x_k, x_kp1 = problem.train_data()
    traj_data = problem.test_data()
    x0_data = TensorDataset(x0_train)
    xp_data = TensorDataset(x_k, x_kp1)

    x0_dataloader = DataLoader(TensorDataset(traj_data[0]), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(TensorDataset(x_kp1, x_k), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    # Create basis functions
    phi_basis =  GaussianBasis.set_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 0.5]))
    psi_basis =  GaussianBasis.set_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 0.5]))
    psi0_basis = GaussianBasis.set_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 0.5]))

    # Create and train the transition model
    tran_model = QuadraticRFF(phi_basis, psi_basis)
    print("Training transition model")
    mle_loss_fn = loss.conditional_mle_loss
    var_reg_loss_fn = lambda model, x, xp : var_reg_strength * (loss.gaussian_basis_var_reg_loss(model.phi_basis, mean=True) + loss.gaussian_basis_var_reg_loss(model.psi_basis, mean=True))
    psd_loss_fn = lambda model, x, xp : psd_reg_strength * loss.B_psd_loss(model)
    tran_model, _, _ = train.train(tran_model, 
        xp_dataloader, 
        {"mle": mle_loss_fn, "var_reg": var_reg_loss_fn, "psd": psd_loss_fn}, 
        torch.optim.Adam(tran_model.parameters(), lr=learning_rate), epochs=n_epochs, use_best="mle")
    print("Done! \n")
    print("PSD: ", tran_model.is_psd())

    mle_loss_fn = loss.mle_loss
    var_reg_loss_fn = lambda model, x : var_reg_strength * loss.gaussian_basis_var_reg_loss(model.psi0_basis, mean=True)
    init_model = QuadraticFF.from_rff(tran_model, psi0_basis)
    print("Training initial model")
    init_model = train.train(init_model, 
        x0_dataloader, 
        {"mle": mle_loss_fn, "var_reg": var_reg_loss_fn}, 
        torch.optim.Adam(init_model.parameters(), lr=learning_rate), epochs=n_epochs, use_best="mle")
    print("Done! \n")

    # Analysis
    box_lows = tuple(problem.plot_bounds_low.tolist())
    box_highs = tuple(problem.plot_bounds_high.tolist())

    belief_seq = propagate.propagate(init_model, tran_model, n_steps=n_timesteps_prop)

    fig, axes = plt.subplots(2, n_timesteps_prop)
    for i in range(n_timesteps_prop):
        #print("Printing belief: ", i)
        plot_belief(axes[1, i], belief_seq[i], x_range=(box_lows[0], box_highs[0]), y_range=(box_lows[1], box_highs[1]))
        axes[0, i].scatter(traj_data[i][:, 0], traj_data[i][:, 1], s=1)
        axes[0, i].set_aspect("equal")
        axes[0, i].set_xlim(box_lows[0], box_highs[0])
        axes[0, i].set_ylim(box_lows[1], box_highs[1])
    
    # Compute empirical AUC of each belief
    for i in range(n_timesteps_prop):
        auc = mc_integral_box(belief_seq[i], domain_bounds=(box_lows, box_highs), n_samples=100000)
        print("AUC of belief at time ", i, ": ", auc)

    plt.show()


