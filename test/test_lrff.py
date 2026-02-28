import torch
import rational_factor.systems.truth_models as truth_models
from rational_factor.systems.base import sample_trajectories, create_transition_data_matrix
from torch.utils.data import DataLoader, TensorDataset
from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.rational_factor import LinearRFF, LinearFF
import rational_factor.models.train as train
import rational_factor.models.loss as loss
import rational_factor.models.propagate as propagate
from rational_factor.tools.visualization import plot_belief
from rational_factor.tools.analysis import mc_integral_box

import matplotlib.pyplot as plt

def make_mvnormal_init_sampler(mean: torch.Tensor, covariance: torch.Tensor):
    """
    Create an initial state sampler that draws n_samples i.i.d. from a multivariate normal.

    Args:
        mean: (d,) mean vector
        covariance: (d, d) covariance matrix

    Returns:
        callable(n_samples: int) -> torch.Tensor of shape (n_samples, d)
    """
    mean = torch.as_tensor(mean, dtype=torch.float32)
    covariance = torch.as_tensor(covariance, dtype=torch.float32)
    dist = torch.distributions.MultivariateNormal(mean, covariance)

    def sampler(n_samples: int) -> torch.Tensor:
        return dist.sample((n_samples,))

    return sampler


if __name__ == "__main__":
    
    ###
    use_gpu = torch.cuda.is_available()
    n_basis = 100
    n_epochs = 200
    batch_size = 256
    learning_rate = 2e-2
    n_timesteps_train = 10
    n_timesteps_prop = 10
    n_trajectories_train = 1000
    var_reg_strength = 0.1
    bomega_reg_strength = 0.02
    ###

    # Create system
    system = truth_models.VanDerPol(dt=0.3, mu=0.9, covariance=0.1*torch.eye(2))

    # Generate data set from trajectories
    mean = torch.tensor([0.2, 0.1])
    cov = torch.diag(torch.tensor([0.2, 0.2]))
    dist = torch.distributions.MultivariateNormal(mean, cov)
    def init_state_sampler(n_samples : int):
        return dist.sample((n_samples,))

    traj_data = sample_trajectories(system, init_state_sampler, n_timesteps=n_timesteps_train, n_trajectories=n_trajectories_train)
    x0_data = TensorDataset(traj_data[0])
    x_k, x_kp1 = create_transition_data_matrix(traj_data, separate=True)
    xp_data = TensorDataset(x_k, x_kp1)

    x0_dataloader = DataLoader(x0_data, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(xp_data, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    # Create basis functions
    phi_basis =  GaussianBasis.set_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 0.0]))
    psi_basis =  GaussianBasis.set_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 0.0]))
    psi0_basis = GaussianBasis.set_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 0.0]))

    # Create and train the transition model
    tran_model = LinearRFF(phi_basis, psi_basis)
    print("Training transition model")
    mle_loss_fn = loss.conditional_mle_loss
    var_reg_loss_fn = lambda model, x, xp : var_reg_strength * (loss.gaussian_basis_var_reg_loss(model.phi_basis, mean=True) + loss.gaussian_basis_var_reg_loss(model.psi_basis, mean=True))
    #bomega_eval_loss_fn = lambda model, x, xp : reg_strength * loss.BOmega_eval_loss(model)
    #bomega_trace_loss_fn = lambda model, x, xp : bomega_reg_strength * loss.BOmega_trace_loss(model)
    tran_model = train.train(tran_model, 
        xp_dataloader, 
        #{"mle": mle_loss_fn, "var_reg": var_reg_loss_fn, "bomega_trace": bomega_trace_loss_fn}, 
        {"mle": mle_loss_fn, "var_reg": var_reg_loss_fn}, 
        torch.optim.Adam(tran_model.parameters(), lr=learning_rate), epochs=n_epochs)
    print("Done! \n")


    mle_loss_fn = loss.mle_loss
    var_reg_loss_fn = lambda model, x : var_reg_strength * loss.gaussian_basis_var_reg_loss(model.psi0_basis, mean=True)
    init_model = LinearFF.from_rff(tran_model, psi0_basis)
    print("Training initial model")
    init_model = train.train(init_model, 
        x0_dataloader, 
        {"mle": mle_loss_fn, "var_reg": var_reg_loss_fn}, 
        torch.optim.Adam(init_model.parameters(), lr=learning_rate), epochs=n_epochs)
    print("Done! \n")

    # Analysis

    box_lows = (-5.0, -5.0)
    box_highs = (5.0, 5.0)

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
        auc = mc_integral_box(belief_seq[i], domain_bounds=(box_lows, box_highs), n_samples=10000)
        print("AUC of belief at time ", i, ": ", auc)

    plt.show()


