import torch
from torch.utils.data import DataLoader, TensorDataset
import rational_factor.systems.po_truth_models as po_truth_models
from rational_factor.systems.base import sample_io_pairs, sample_observation_pairs, simulate, SystemObservationDistribution, SystemTransitionDistribution
from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.factor_forms import LinearRF, LinearR2FF, Linear2FF
import rational_factor.models.train as train
import rational_factor.models.loss as loss
import rational_factor.tools.propagate as propagate
from rational_factor.tools.visualization import plot_belief, plot_particle_belief
from rational_factor.tools.analysis import mc_integral_box
from particle_filter.particle_set import WeightedParticleSet
from particle_filter.propagate import propagate_and_update
import matplotlib.pyplot as plt

from rational_factor.tools.misc import make_mvnormal_init_sampler


if __name__ == "__main__":
    
    ###
    use_gpu = torch.cuda.is_available()
    n_basis = 200
    obs_params = {
        "n_epochs_per_group": [2, 1], # basis, weights
        "iterations": 1,
        "lr_basis": 1e-2,
        "lr_weights": 1e-2,
    }
    tran_params = {
        "n_epochs_per_group": [2, 5], # basis, weights
        "iterations": 2,
        "lr_basis": 1e-2,
        "lr_weights": 1e-2,
    }
    init_params = {
        "n_epochs_per_group": [2, 1], # basis, weights
        "iterations": 2,
        "lr_basis": 1e-2,
        "lr_weights": 1e-2,
    }

    batch_size = 1024

    n_data_init = 3000
    n_data_tran = 30000
    n_data_obs = 30000

    n_timesteps_prop = 10
    n_particles_test = 5000
    var_reg_strength = 0 
    ###

    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using GPU: ", use_gpu)
    print("Device: ", device)

    # Create system
    system = po_truth_models.PartiallyObservableVanDerPol(dt=0.3, mu=0.9, process_covariance=0.1*torch.eye(2), observation_covariance=0.04*torch.eye(2))

    init_state_sampler = make_mvnormal_init_sampler(mean=torch.tensor([0.2, 0.1]), covariance=torch.diag(torch.tensor([0.2, 0.2])))

    # Generate data as input output pairs
    def prev_state_sampler(n_samples : int):
        mean = torch.tensor([0.0, 0.0])
        cov = torch.diag(4.0 * torch.ones(system.dim()))
        dist = torch.distributions.MultivariateNormal(mean, cov)
        return dist.sample((n_samples,))

    x0_data = init_state_sampler(n_data_init)
    x_k_data, x_kp1_data = sample_io_pairs(system, prev_state_sampler, n_pairs=n_data_tran)
    x_data, o_data = sample_observation_pairs(system, init_state_sampler, n_pairs=n_data_obs)

    x0_dataset = TensorDataset(x0_data)
    xp_dataset = TensorDataset(x_kp1_data, x_k_data)
    o_dataset = TensorDataset(o_data, x_data)

    x0_dataloader = DataLoader(x0_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(xp_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    o_dataloader = DataLoader(o_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    # Create basis functions
    phi_basis  = GaussianBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 30.0], device=device), variance=20.0, min_std=1e-3).to(device)
    psi_basis  = GaussianBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 30.0], device=device), variance=20.0, min_std=1e-3).to(device)
    psi0_basis = GaussianBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 30.0], device=device), variance=20.0, min_std=1e-3).to(device)
    xi_basis   = GaussianBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 30.0], device=device), variance=20.0, min_std=1e-3).to(device)
    zeta_basis = GaussianBasis.random_init(system.observation_dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 30.0], device=device), variance=20.0, min_std=1e-3).to(device)

    # Create and train the observation model
    obs_model = LinearRF(xi_basis, zeta_basis).to(device)
    reg_loss_fn = lambda model, o, x: var_reg_strength * (
        loss.gaussian_basis_var_reg_loss(model.xi_basis, mean=True)
        + loss.gaussian_basis_var_reg_loss(model.zeta_basis, mean=True)
    )
    optimizers = {
        "basis": torch.optim.Adam(obs_model.basis_params(), lr=obs_params["lr_basis"]),
        "weights": torch.optim.Adam(obs_model.weight_params(), lr=obs_params["lr_weights"]),
    }
    print("Training observation model")
    obs_model, best_loss_obs, training_time_obs = train.train_iterate(obs_model, 
        o_dataloader, 
        {"mle": loss.conditional_mle_loss, "var_reg": reg_loss_fn}, 
        optimizers,
        epochs_per_group=obs_params["n_epochs_per_group"],
        iterations=obs_params["iterations"],
        verbose=True,
        use_best="mle")
    print("Done! \n")

    # Create and train the transition model
    tran_model = LinearR2FF.from_rf(obs_model, phi_basis, psi_basis).to(device)
    reg_loss_fn = lambda model, xp, x : var_reg_strength * (loss.gaussian_basis_var_reg_loss(model.phi_basis, mean=True) + loss.gaussian_basis_var_reg_loss(model.psi_basis, mean=True))
    optimizers = {
        "basis": torch.optim.Adam(tran_model.basis_params(), lr=tran_params["lr_basis"]),
        "weights": torch.optim.Adam(tran_model.weight_params(), lr=tran_params["lr_weights"]),
    }
    print("Training transition model")
    tran_model, best_loss_tran, training_time_tran = train.train_iterate(tran_model, 
        xp_dataloader, 
        {"mle": loss.conditional_mle_loss, "var_reg": reg_loss_fn}, 
        optimizers,
        epochs_per_group=tran_params["n_epochs_per_group"],
        iterations=tran_params["iterations"],
        verbose=True,
        use_best="mle")
    print("Done! \n")


    init_model = Linear2FF.from_r2ff(tran_model, psi0_basis).to(device)
    reg_loss_fn = lambda model, x : var_reg_strength * loss.gaussian_basis_var_reg_loss(model.psi0_basis, mean=True)
    optimizers = {
        "basis": torch.optim.Adam(init_model.basis_params(), lr=init_params["lr_basis"]),
        "weights": torch.optim.Adam(init_model.weight_params(), lr=init_params["lr_weights"]),
    }
    print("Training initial model")
    init_model, best_loss_init, training_time_init = train.train_iterate(init_model, 
        x0_dataloader, 
        {"mle": loss.mle_loss, "var_reg": reg_loss_fn}, 
        optimizers,
        epochs_per_group=init_params["n_epochs_per_group"],
        iterations=init_params["iterations"],
        verbose=True,
        use_best="mle")
    print("Done! \n")

    print(f"Observation model loss : {best_loss_obs:.4f}, training time: {training_time_obs:.2f} seconds")
    print(f"Transition model loss  : {best_loss_tran:.4f}, training time: {training_time_tran:.2f} seconds")
    print(f"Initial model loss     : {best_loss_init:.4f}, training time: {training_time_init:.2f} seconds")

    # Simulate the true trajectory
    sim_true_states, sim_observations = simulate(system, init_state_sampler, n_timesteps=n_timesteps_prop)
    sim_true_states.to(device)
    sim_observations.to(device)

    # Run the filter
    priors, posteriors = propagate.propagate_and_update(init_model, tran_model, obs_model, sim_observations)

    # Run the comparison truth model WPF
    t_dist = SystemTransitionDistribution(system)
    o_dist = SystemObservationDistribution(system)
    initial_particle_belief = WeightedParticleSet(
        particles=init_state_sampler(n_particles_test),
        weights=torch.ones(n_particles_test) / n_particles_test,
    )
    wpf_priors, wpf_posteriors = propagate_and_update(
        belief=initial_particle_belief, 
        transition_model=t_dist, 
        observation_model=o_dist, 
        observations=[obs for obs in sim_observations])

    # Analysis
    box_lows = (-5.0, -5.0)
    box_highs = (5.0, 5.0)

    fig, axes = plt.subplots(n_timesteps_prop, 2, figsize=(20, max(8*n_timesteps_prop, 40)))
    fig.suptitle("Beliefs at each time step")
    for i in range(n_timesteps_prop):
        #print("Printing belief: ", i)
        plot_belief(axes[i, 0], posteriors[i], x_range=(box_lows[0], box_highs[0]), y_range=(box_lows[1], box_highs[1]))
        plot_particle_belief(axes[i, 1], wpf_posteriors[i], x_range=(box_lows[0], box_highs[0]), y_range=(box_lows[1], box_highs[1]))
        for j in [0, 1]:
            axes[i, j].scatter(sim_true_states[i, 0].item(), sim_true_states[i, 1].item(), marker="o", s=80, c="red")
            if i > 0:
                axes[i, j].scatter(sim_observations[i-1, 0].item(), sim_observations[i-1, 1].item(), marker="o", s=80, c="blue")
            axes[i, j].set_xlabel("x1")
            axes[i, j].set_ylabel("x2")
            axes[i, j].grid(alpha=0.25)
    
        axes[i, 0].set_title(f"Posterior at k={i}")
        axes[i, 1].set_title(f"WPF Posterior at k={i}")


    plt.savefig("figures/vdp_filter_gaussian.jpg", dpi=1000)


