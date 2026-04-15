import torch
import rational_factor.systems.truth_models as truth_models
from rational_factor.systems.base import sample_trajectories, create_transition_data_matrix
from torch.utils.data import DataLoader, TensorDataset
from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.factor_forms import QuadraticRFF, QuadraticFF
import rational_factor.models.train as train
import rational_factor.models.loss as loss
import rational_factor.tools.propagate as propagate
from rational_factor.tools.visualization import plot_belief
from rational_factor.tools.analysis import mc_integral_box, avg_log_likelihood

import matplotlib.pyplot as plt

from rational_factor.tools.misc import make_mvnormal_init_sampler


if __name__ == "__main__":
    
    ###
    use_gpu = torch.cuda.is_available()
    n_basis = 30
    n_epochs = 100
    batch_size = 512
    learning_rate = 1e-1
    n_timesteps_train = 10
    n_timesteps_prop = 10
    n_trajectories_train = 1000
    n_trajectories_test = 1000
    var_reg_strength = 1e-3
    psd_reg_strength = 1e-4 
    ###

    # Create system
    system = truth_models.PlanarQuadrotor(dt=0.01, covariance=0.05 * torch.eye(6), waypoint=torch.tensor([5.0, 0.0]))
    #system = truth_models.PlanarQuadrotor(
    #    dt=0.03, 
    #    covariance=0.05 * torch.eye(6), 
    #    waypoint=torch.tensor([5.0, 5.0]),
    #    m=1.0,
    #    I=0.03,
    #    ell=0.2,
    #    g=9.81,
    #    c_v=0.05,
    #    c_w=0.12,
    #)
    #system.kp_pos = torch.tensor([1.0, 1.0])
    #system.kd_pos = torch.tensor([0.5, 0.5])
    #system.kp_theta = 3.0
    #system.kd_theta = 2.0

    # Generate data set from trajectories
    #mean = torch.tensor([0.0, 0.0, 0.1, 50.0, 0.0, 0.0])
    init_state_sampler = make_mvnormal_init_sampler(mean=torch.tensor([0.0, 0.0, 0.1, 0.0, 0.0, 0.0]), covariance=torch.diag(torch.tensor([0.1, 0.1, 0.05, 0.1, 0.1, 0.05])))

    traj_data = sample_trajectories(system, init_state_sampler, n_timesteps=n_timesteps_train, n_trajectories=n_trajectories_train)
    test_data = sample_trajectories(system, init_state_sampler, n_timesteps=n_timesteps_train, n_trajectories=n_trajectories_test)
    x0_data = TensorDataset(traj_data[0])
    x_k, x_kp1 = create_transition_data_matrix(traj_data, separate=True)

    x0_dataloader = DataLoader(TensorDataset(traj_data[0]), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(TensorDataset(x_kp1, x_k), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    # Create basis functions
    phi_basis =  GaussianBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 10.0]), variance=1.0)
    psi_basis =  GaussianBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 10.0]), variance=1.0)
    psi0_basis = GaussianBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 10.0]), variance=1.0)
    #phi_basis =  GaussianBasis.set_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 0.5]))
    #psi_basis =  GaussianBasis.set_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 0.5]))
    #psi0_basis = GaussianBasis.set_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 0.5]))

    #print("Phi basis: ", phi_basis.means_stds())
    #print("Psi basis: ", psi_basis.means_stds())
    #print("Psi0 basis: ", psi0_basis.means_stds())

    # Create and train the transition model
    tran_model = QuadraticRFF(phi_basis, psi_basis)
    print("Training transition model")
    print(x_k.min(), x_k.max())
    print(x_kp1.min(), x_kp1.max())
    test =tran_model(x_k, x_kp1) 
    print(test.min(), test.max())

    mle_loss_fn = loss.conditional_mle_loss
    var_reg_loss_fn = lambda model, x, xp : var_reg_strength * (loss.gaussian_basis_var_reg_loss(model.phi_basis, mean=True) + loss.gaussian_basis_var_reg_loss(model.psi_basis, mean=True))
    psd_loss_fn = lambda model, x, xp : psd_reg_strength * loss.B_psd_loss(model)
    tran_model, _, _ = train.train(tran_model, 
        xp_dataloader, 
        #{"mle": mle_loss_fn, "psd": psd_loss_fn}, 
        {"mle": mle_loss_fn, "var_reg": var_reg_loss_fn, "psd": psd_loss_fn}, 
        torch.optim.Adam(tran_model.parameters(), lr=learning_rate), epochs=n_epochs, use_best="mle")
    print("Done! \n")
    print("PSD: ", tran_model.is_psd())

    mle_loss_fn = loss.mle_loss
    #var_reg_loss_fn = lambda model, x : var_reg_strength * loss.gaussian_basis_var_reg_loss(model.psi0_basis, mean=True)
    init_model = QuadraticFF.from_rff(tran_model, psi0_basis)
    print("Training initial model")
    init_model = train.train(init_model, 
        x0_dataloader, 
        {"mle": mle_loss_fn}, 
        torch.optim.Adam(init_model.parameters(), lr=learning_rate), epochs=n_epochs, use_best="mle")
    print("Done! \n")

    # Analysis
    box_lows = (-5.0, -5.0)
    box_highs = (5.0, 5.0)

    belief_seq = propagate.propagate(init_model, tran_model, n_steps=n_timesteps_prop)

    fig, axes = plt.subplots(2, n_timesteps_prop)
    for i in range(n_timesteps_prop):
        #print("Printing belief: ", i)
        belief_marginal = belief_seq[i].marginal(marginal_dims=(0, 1))
        plot_belief(axes[1, i], belief_marginal, x_range=(box_lows[0], box_highs[0]), y_range=(box_lows[1], box_highs[1]))
        axes[0, i].scatter(traj_data[i][:, 0], traj_data[i][:, 1], s=1)
        axes[0, i].set_aspect("equal")
        axes[0, i].set_xlim(box_lows[0], box_highs[0])
        axes[0, i].set_ylim(box_lows[1], box_highs[1])
    
    # Compute empirical AUC of each belief
    for i in range(n_timesteps_prop):
        #auc = mc_integral_box(belief_seq[i], domain_bounds=(box_lows, box_highs), n_samples=100000)
        #print("AUC of belief at time ", i, ": ", auc)
        accuracy = avg_log_likelihood(belief_seq[i], test_data[i])
        print("Accuracy of belief at time ", i, ": ", accuracy)

    plt.show()


