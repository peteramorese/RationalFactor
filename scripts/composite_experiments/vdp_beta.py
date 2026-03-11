import torch
import rational_factor.systems.truth_models as truth_models
from rational_factor.systems.base import sample_trajectories, create_transition_data_matrix
from torch.utils.data import DataLoader, TensorDataset
from rational_factor.models.basis_functions import BetaBasis
from rational_factor.models.density_model import QuadraticRFF, QuadraticFF
import rational_factor.models.train as train
import rational_factor.models.loss as loss
import rational_factor.models.propagate as propagate
from rational_factor.tools.visualization import plot_belief
from rational_factor.tools.analysis import mc_integral_box
from rational_factor.models.domain_transformation import ErfSeparableTF
from rational_factor.models.composite_model import CompositeDensityModel, CompositeConditionalModel
import matplotlib.pyplot as plt

def make_mvnormal_init_sampler(mean: torch.Tensor, covariance: torch.Tensor):
    mean = torch.as_tensor(mean, dtype=torch.float32)
    covariance = torch.as_tensor(covariance, dtype=torch.float32)
    dist = torch.distributions.MultivariateNormal(mean, covariance)

    def sampler(n_samples: int) -> torch.Tensor:
        return dist.sample((n_samples,))

    return sampler


if __name__ == "__main__":
    
    ###
    use_gpu = torch.cuda.is_available()
    n_basis = 20
    n_epochs = 100
    batch_size = 512
    learning_rate = 5e-3
    n_timesteps_train = 10
    n_timesteps_prop = 10
    n_trajectories_train = 2000
    var_reg_strength = 5e-3
    psd_reg_strength = 1e-1 #0.002
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
    phi_basis =  BetaBasis.set_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([5.0, 5.0]), min_concentration=1.0)
    psi_basis =  BetaBasis.set_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([5.0, 5.0]), min_concentration=1.0)
    psi0_basis = BetaBasis.set_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([5.0, 5.0]), min_concentration=1.0)

    # Create separable domain transformation
    domain_tf = ErfSeparableTF.from_data(x_k, trainable=True)

    # Create and train the transition model
    tran_model = CompositeConditionalModel(domain_tf, QuadraticRFF(phi_basis, psi_basis))
    print("Training transition model")
    mle_loss_fn = loss.conditional_mle_loss
    var_reg_loss_fn = lambda model, x, xp : var_reg_strength * (loss.beta_basis_concentration_reg_loss(model.conditional_density_model.phi_basis) + loss.beta_basis_concentration_reg_loss(model.conditional_density_model.psi_basis))
    psd_loss_fn = lambda model, x, xp : psd_reg_strength * loss.B_psd_loss(model.conditional_density_model)
    tran_model = train.train(tran_model, 
        xp_dataloader, 
        {"mle": mle_loss_fn, "var_reg": var_reg_loss_fn, "psd": psd_loss_fn}, 
        torch.optim.Adam(tran_model.parameters(), lr=learning_rate), epochs=n_epochs, use_best="mle")
    print("Done! \n")
    print("PSD: ", tran_model.conditional_density_model.is_psd())

    init_model = CompositeDensityModel(domain_tf, QuadraticFF.from_rff(tran_model.conditional_density_model, psi0_basis))
    print("Training initial model")
    mle_loss_fn = loss.mle_loss
    var_reg_loss_fn = lambda model, x : var_reg_strength * loss.beta_basis_concentration_reg_loss(model.density_model.psi0_basis)
    init_model = train.train(init_model, 
        x0_dataloader, 
        {"mle": mle_loss_fn, "var_reg": var_reg_loss_fn}, 
        torch.optim.Adam(init_model.parameters(), lr=learning_rate), epochs=n_epochs, use_best="mle")
    print("Done! \n")

    # Analysis
    box_lows = (-5.0, -5.0)
    box_highs = (5.0, 5.0)

    base_belief_seq = propagate.propagate(init_model.density_model, tran_model.conditional_density_model, n_steps=n_timesteps_prop)
    belief_seq = [CompositeDensityModel(domain_tf, belief) for belief in base_belief_seq]

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


