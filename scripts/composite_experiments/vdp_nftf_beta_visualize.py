import torch
import rational_factor.systems.truth_models as truth_models
from rational_factor.systems.base import sample_trajectories, create_transition_data_matrix, sample_io_pairs
from torch.utils.data import DataLoader, TensorDataset
from rational_factor.models.basis_functions import BetaBasis
from rational_factor.models.factor_forms import QuadraticRFF, QuadraticFF, LinearRFF, LinearFF
import rational_factor.models.train as train
import rational_factor.models.loss as loss
import rational_factor.tools.propagate as propagate
from rational_factor.tools.visualization import plot_belief
from rational_factor.tools.analysis import avg_log_likelihood
from rational_factor.models.domain_transformation import MaskedAffineNFTF, ErfSeparableTF
from rational_factor.models.composite_model import CompositeDensityModel, CompositeConditionalModel
import matplotlib.pyplot as plt

from rational_factor.tools.misc import make_mvnormal_init_sampler


if __name__ == "__main__":
    
    ###
    use_gpu = torch.cuda.is_available()
    use_dtf = False
    n_basis = 500
    if use_dtf:
        tran_params = {
            "n_epochs_per_group": [20, 5], # dtf_params and basis, weights
            "iterations": 50,
            "lr_basis": 5e-2,
            "lr_weights": 1e-2,
            "lr_dtf": 1e-3,
            "lr_wrap": 1e-3,
        }
        init_params = {
            "n_epochs_per_group": [20, 5], # basis, weights
            "iterations": 70,
            "lr_basis": 1e-2,
            "lr_weights": 1e-2,
        }
    else:
        tran_params = {
            "n_epochs_per_group": [20, 5], # basis, weights
            "iterations": 50,
            "lr_basis": 5e-2,
            "lr_weights": 1e-2,
            "lr_wrap": 1e-3,
        }
        init_params = {
            "n_epochs_per_group": [20, 5], # basis, weights
            "iterations": 70,
            "lr_basis": 1e-2,
            "lr_weights": 1e-2,
        }

    batch_size = 256
    n_timesteps_prop = 10
    n_trajectories_test = 1000
    n_data_tran = 10000
    n_data_init = 1000
    #var_reg_strength = 5e-3
    var_reg_strength = 0#1e-2
    ###

    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using GPU: ", use_gpu)
    print("Device: ", device)

    # Create system
    system = truth_models.VanDerPol(dt=0.3, mu=0.9, covariance=0.1*torch.eye(2))

    init_state_sampler = make_mvnormal_init_sampler(mean=torch.tensor([0.2, 0.1]), covariance=torch.diag(torch.tensor([0.2, 0.2])))

    ## Generate data set from trajectories
    test_traj_data = sample_trajectories(system, init_state_sampler, n_timesteps=n_timesteps_prop, n_trajectories=n_trajectories_test)

    # Generate data as input output pairs
    def prev_state_sampler(n_samples : int):
        mean = torch.tensor([0.0, 0.0])
        cov = torch.diag(4.0 * torch.ones(system.dim()))
        dist = torch.distributions.MultivariateNormal(mean, cov)
        return dist.sample((n_samples,))

    x0_data = init_state_sampler(n_data_init)
    x_k_data, x_kp1_data = sample_io_pairs(system, prev_state_sampler, n_pairs=n_data_tran)

    x0_dataloader = DataLoader(TensorDataset(x0_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(TensorDataset(x_kp1_data, x_k_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    # Create basis functions
    phi_basis =  BetaBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([10.0, 10.0], device=device), variance=30.0, min_concentration=1.0).to(device)
    psi_basis =  BetaBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([10.0, 10.0], device=device), variance=30.0, min_concentration=1.0).to(device)
    psi0_basis = BetaBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([10.0, 10.0], device=device), variance=30.0, min_concentration=1.0).to(device)

    # Create separable domain transformation
    wrap_tf = ErfSeparableTF.from_data(x_k_data, trainable=True)
    
    nftf = MaskedAffineNFTF(system.dim(), trainable=True, hidden_features=128, n_layers=5).to(device) if use_dtf else None

    # Create and train the transition model
    if use_dtf:
        tran_model = CompositeConditionalModel([nftf, wrap_tf], LinearRFF(phi_basis, psi_basis)).to(device)
    else:
        tran_model = CompositeConditionalModel([wrap_tf], LinearRFF(phi_basis, psi_basis)).to(device)

    print("Training transition model")
    mle_loss_fn = loss.conditional_mle_loss
    
    if use_dtf:
        var_reg_loss_fn = lambda model, x, xp : var_reg_strength * (loss.beta_basis_concentration_reg_loss(model.conditional_density_model.phi_basis) + loss.beta_basis_concentration_reg_loss(model.conditional_density_model.psi_basis))
        optimizers ={"dtf_and_basis": torch.optim.Adam([{'params': tran_model.conditional_density_model.basis_params(), 'lr': tran_params["lr_basis"]}, 
                {'params':tran_model.domain_tfs[0].parameters(), 'lr': tran_params["lr_dtf"]}, 
                {'params':tran_model.domain_tfs[1].parameters(), 'lr': tran_params["lr_wrap"]}]), 
            "weights": torch.optim.Adam(tran_model.conditional_density_model.weight_params(), lr=tran_params["lr_weights"])} 
    else:
        var_reg_loss_fn = lambda model, x, xp : var_reg_strength * (loss.beta_basis_concentration_reg_loss(model.conditional_density_model.phi_basis) + loss.beta_basis_concentration_reg_loss(model.conditional_density_model.psi_basis))
        optimizers ={"basis": torch.optim.Adam([{'params': tran_model.conditional_density_model.basis_params(), 'lr': tran_params["lr_basis"]}, {'params': tran_model.domain_tfs.parameters(), 'lr': tran_params["lr_wrap"]}]),
            "weights": torch.optim.Adam(tran_model.conditional_density_model.weight_params(), lr=tran_params["lr_weights"])}

    tran_model, best_loss_tran, training_time_tran = train.train_iterate(tran_model,
        xp_dataloader,
        {"mle": mle_loss_fn, "var_reg": var_reg_loss_fn}, 
        optimizers,
        epochs_per_group=tran_params["n_epochs_per_group"],
        iterations=tran_params["iterations"],
        verbose=True,
        use_best="mle")
    print("Done! \n")
    print("Valid: ", tran_model.valid())


    # Copy the domain transformation to fix it for training the initial state model
    trained_nftf = MaskedAffineNFTF.copy_from_trainable(nftf).to(device) if use_dtf else None
    trained_domain_tf = ErfSeparableTF.copy_from_trainable(wrap_tf).to(device)

    #init_model = CompositeDensityModel(trained_domain_tf, QuadraticFF.from_rff(tran_model.conditional_density_model, psi0_basis))
    if use_dtf:
        init_model = CompositeDensityModel([trained_nftf, trained_domain_tf], LinearFF.from_rff(tran_model.conditional_density_model, psi0_basis)).to(device)
    else:
        init_model = CompositeDensityModel([trained_domain_tf], LinearFF.from_rff(tran_model.conditional_density_model, psi0_basis)).to(device)

    print("Training initial model")
    mle_loss_fn = loss.mle_loss

    if use_dtf:
        var_reg_loss_fn = lambda model, x : var_reg_strength * loss.beta_basis_concentration_reg_loss(model.density_model.psi0_basis)
        optimizers = {"basis": torch.optim.Adam(init_model.density_model.basis_params(), lr=init_params["lr_basis"]), "weights": torch.optim.Adam(init_model.density_model.weight_params(), lr=init_params["lr_weights"])}
    else:
        var_reg_loss_fn = lambda model, x : var_reg_strength * loss.beta_basis_concentration_reg_loss(model.density_model.psi0_basis)
        optimizers = {"basis": torch.optim.Adam(init_model.density_model.basis_params(), lr=init_params["lr_basis"]), "weights": torch.optim.Adam(init_model.density_model.weight_params(), lr=init_params["lr_weights"])}

    init_model, best_loss_init, training_time_init = train.train_iterate(init_model, 
        x0_dataloader, 
        {"mle": mle_loss_fn, "var_reg": var_reg_loss_fn}, 
        optimizers,
        epochs_per_group=init_params["n_epochs_per_group"],
        iterations=init_params["iterations"],
        verbose=True,
        use_best="mle")
    print("Done! \n")

    print(f"Transition model loss: {best_loss_tran:.4f}, training time: {training_time_tran:.2f} seconds")
    print(f"Initial model loss: {best_loss_init:.4f}, training time: {training_time_init:.2f} seconds")

    # Analysis
    state_box_lows = (-5.0, -5.0)
    state_box_highs = (5.0, 5.0)
    latent_box_lows = (-0.05, -0.05)
    latent_box_highs = (1.05, 1.05)

    base_belief_seq = propagate.propagate(
        init_model.density_model, tran_model.conditional_density_model, n_steps=n_timesteps_prop
    )
    if use_dtf:
        belief_seq = [CompositeDensityModel([trained_nftf, trained_domain_tf], belief) for belief in base_belief_seq]
        latent_tfs = [trained_nftf, trained_domain_tf]
    else:
        belief_seq = [CompositeDensityModel([trained_domain_tf], belief) for belief in base_belief_seq]
        latent_tfs = [trained_domain_tf]

    # Figure 1: state-space samples vs state-space beliefs (original)
    fig_state, axes_state = plt.subplots(2, n_timesteps_prop, figsize=(20, 10))
    fig_state.suptitle("State-space beliefs vs state-space samples")

    # Figure 2: latent samples vs latent beliefs
    fig_latent, axes_latent = plt.subplots(2, n_timesteps_prop, figsize=(20, 10))
    fig_latent.suptitle("Latent-space beliefs vs latent samples")

    for i in range(n_timesteps_prop):
        data_i = test_traj_data[i].to(device)
        ll = avg_log_likelihood(belief_seq[i], data_i)
        print(f"State log likelihood at time {i}: {ll:.4f}")

        # State-space plot
        plot_belief(
            axes_state[1, i],
            belief_seq[i],
            x_range=(state_box_lows[0], state_box_highs[0]),
            y_range=(state_box_lows[1], state_box_highs[1]),
        )
        axes_state[0, i].scatter(data_i[:, 0].detach().cpu(), data_i[:, 1].detach().cpu(), s=1)
        axes_state[0, i].set_aspect("equal")
        axes_state[0, i].set_xlim(state_box_lows[0], state_box_highs[0])
        axes_state[0, i].set_ylim(state_box_lows[1], state_box_highs[1])

        # Push trajectory samples to latent space
        z_i = data_i
        for tf in latent_tfs:
            z_i, _ = tf(z_i)
        ll_latent = avg_log_likelihood(base_belief_seq[i], z_i)
        print(f"Latent log likelihood at time {i}: {ll_latent:.4f}")

        # Latent-space plot
        plot_belief(
            axes_latent[1, i],
            base_belief_seq[i],
            x_range=(latent_box_lows[0], latent_box_highs[0]),
            y_range=(latent_box_lows[1], latent_box_highs[1]),
        )
        axes_latent[0, i].scatter(z_i[:, 0].detach().cpu(), z_i[:, 1].detach().cpu(), s=1)
        axes_latent[0, i].set_aspect("equal")
        axes_latent[0, i].set_xlim(latent_box_lows[0], latent_box_highs[0])
        axes_latent[0, i].set_ylim(latent_box_lows[1], latent_box_highs[1])

    fig_state.savefig("figures/vdp_nfdf_beta_beliefs_state_space.pdf", dpi=1000)
    fig_latent.savefig("figures/vdp_nfdf_beta_beliefs_latent_space.pdf", dpi=1000)
    #plt.show()


