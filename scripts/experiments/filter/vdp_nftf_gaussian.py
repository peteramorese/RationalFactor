import torch
from torch.utils.data import DataLoader, TensorDataset
from rational_factor.systems.base import simulate, SystemObservationDistribution, SystemTransitionDistribution
from rational_factor.systems.problems import PARTIALLY_OBSERVABLE_PROBLEMS
from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.factor_forms import LinearRF, LinearR2FF, Linear2FF, LinearFF, LinearRFF, LinearRFandR2FF
import rational_factor.models.train as train
import rational_factor.models.loss as loss
import rational_factor.tools.propagate as propagate
from rational_factor.models.density_model import LogisticSigmoid
from rational_factor.models.composite_model import CompositeDensityModel, CompositeConditionalModel
from rational_factor.models.domain_transformation import MaskedAffineNFTF
from rational_factor.tools.visualization import plot_belief, plot_particle_belief
from rational_factor.tools.analysis import check_pdf_valid
from rational_factor.tools.misc import data_bounds, train_test_split
from particle_filter.particle_set import WeightedParticleSet
from particle_filter.propagate import propagate_and_update
import matplotlib.pyplot as plt
from pathlib import Path


if __name__ == "__main__":
    
    ###
    problem = PARTIALLY_OBSERVABLE_PROBLEMS["po_van_der_pol"]

    use_gpu = torch.cuda.is_available()
    use_dtf = True
    n_basis = 300
    n_obs_basis = 50

    if use_dtf:
        
        obs_and_tran_params = {
            "n_epochs_per_group": [20, 5], # basis, weights
            "iterations": 50,
            "pre_train_epochs": 10,
            "lr_basis_tran": 5e-3,
            "lr_basis_obs": 5e-3,
            "lr_weights": 1e-3,
            "lr_dtf": 5e-4,
        }
        init_params = {
            "n_epochs_per_group": [20, 5], # basis, weights
            "iterations": 100,
            "lr_basis": 1e-2,
            "lr_weights": 1e-2,
        }
    else:
        obs_and_tran_params = {
            "n_epochs_per_group": [20, 5], # basis, weights
            "iterations": 50,
            "lr_basis_tran": 5e-2,
            "lr_basis_obs": 5e-2,
            "lr_weights": 1e-2,
        }
        init_params = {
            "n_epochs_per_group": [20, 5], # basis, weights
            "iterations": 100,
            "lr_basis": 1e-2,
            "lr_weights": 1e-2,
        }

    batch_size = 256
    block_size = None

    reg_covar_joint = 1e-3
    ls_temp = 0.1

    obs_loss_weight = 1.0

    n_simulation_tests = 3
    n_particles_true_pf = 5000
    figure_prefix = "vdp_nftf_filter_gaussian"
    ###

    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using GPU: ", use_gpu)

    # Create system from shared partially observable problem parameters
    system = problem.system
    init_state_sampler = problem.initial_state_sampler
    n_timesteps_prop = problem.n_timesteps

    x0_data = problem.train_initial_state_data()
    x_k_data, x_kp1_data = problem.train_state_transition_data()
    xo_data, o_data = problem.train_obs_data()

    x0_train, x0_val = train_test_split(x0_data, test_size=0.2)
    x_k_train, x_k_val, x_kp1_train, x_kp1_val = train_test_split(x_k_data, x_kp1_data, test_size=0.2)
    xo_train, xo_val, o_train, o_val = train_test_split(xo_data, o_data, test_size=0.2)

    # Training data loaders
    x0_dataloader = DataLoader(TensorDataset(x0_train), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(TensorDataset(x_kp1_train, x_k_train), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    x_k_dataloader = DataLoader(TensorDataset(x_k_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    o_dataloader = DataLoader(TensorDataset(o_train, xo_train), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    # Validation data loaders
    x0_val_dataloader = DataLoader(TensorDataset(x0_val), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    xp_val_dataloader = DataLoader(TensorDataset(x_kp1_val, x_k_val), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    o_val_dataloader = DataLoader(TensorDataset(o_val, xo_val), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    nftf = MaskedAffineNFTF(system.dim(), trainable=True, hidden_features=256, n_layers=8).to(device)

    # Pre train the dtf
    if use_dtf:
        print("Pre training the dtf")
        loc, scale = data_bounds(x_k_data, mode="center_lengths")
        base_distribution = LogisticSigmoid(system.dim(), temperature=ls_temp, loc=loc.to(device), scale=scale.to(device))
        decorrupter_density = CompositeDensityModel([nftf], base_distribution).to(device)
        optimizer = torch.optim.Adam(nftf.parameters(), lr=obs_and_tran_params["lr_dtf"], weight_decay=obs_and_tran_params["lr_dtf"])
        decorrupter_density, best_loss, training_time = train.train(
            decorrupter_density,
            x_k_dataloader,
            {"mle": loss.mle_loss},
            optimizer,
            epochs=obs_and_tran_params["pre_train_epochs"],
            verbose=True,
            use_best="mle",
        )
        print("Done.\n")

    # Prefit the basis functions
    print("Prefitting the basis functions")
    if use_dtf:
        with torch.no_grad():
            y_k_data, _ = nftf(x_k_data.to(device))
            y_kp1_data, _ = nftf(x_kp1_data.to(device))
            y0_data, _ = nftf(x0_data.to(device))
            yo_data, _ = nftf(xo_data.to(device))
            y_k_data = y_k_data.to(torch.device("cpu"))
            y_kp1_data = y_kp1_data.to(torch.device("cpu"))
            y0_data = y0_data.to(torch.device("cpu"))
            yo_data = yo_data.to(torch.device("cpu"))
    else:
        y_k_data = x_k_data.to(torch.device("cpu"))
        y_kp1_data = x_kp1_data.to(torch.device("cpu"))
        y0_data = x0_data.to(torch.device("cpu"))
        yo_data = xo_data.to(torch.device("cpu"))

    # Transition model basis functions (fit to p(y, y'))
    y_joint_data = torch.cat([y_k_data, y_kp1_data], dim=1)
    tran_gmm_lf = train.fit_gaussian_lf_em(y_joint_data.to(torch.device("cpu")), n_components=n_basis, reg_covar=reg_covar_joint, max_iter=100)
    weights = tran_gmm_lf.get_w()
    phi_marginal = tran_gmm_lf.basis.marginal(marginal_dims=range(system.dim()))
    phi_means, phi_stds = phi_marginal.means_stds()
    phi_params = torch.stack([phi_means, phi_stds], dim=-1)

    psi_marginal = tran_gmm_lf.basis.marginal(marginal_dims=range(system.dim(), 2 * system.dim()))
    psi_means, psi_stds = psi_marginal.means_stds()
    psi_params = torch.stack([psi_means, psi_stds], dim=-1)

    phi_basis = GaussianBasis(uparams_init=phi_params).to(device)
    psi_basis = GaussianBasis(uparams_init=psi_params).to(device)

    # Observation model basis functions (fit to p(y, o))
    yo_joint_data = torch.cat([xo_data, o_data], dim=1)
    obs_gmm_lf = train.fit_gaussian_lf_em(yo_joint_data.to(torch.device("cpu")), n_components=n_obs_basis, reg_covar=reg_covar_joint, max_iter=100)
    weights = obs_gmm_lf.get_w()
    xi_marginal = obs_gmm_lf.basis.marginal(marginal_dims=range(system.dim()))
    xi_means, xi_stds = xi_marginal.means_stds()
    xi_params = torch.stack([xi_means, xi_stds], dim=-1)

    zeta_marginal = obs_gmm_lf.basis.marginal(marginal_dims=range(system.dim(), system.dim() + system.observation_dim()))
    zeta_means, zeta_stds = zeta_marginal.means_stds()
    zeta_params = torch.stack([zeta_means, zeta_stds], dim=-1)

    xi_basis = GaussianBasis(uparams_init=xi_params).to(device)
    zeta_basis = GaussianBasis(uparams_init=zeta_params).to(device)

    # Initial model basis functions (fit to p(y))
    init_gmm_lf = train.fit_gaussian_lf_em(y0_data.to(torch.device("cpu")), n_components=n_basis, reg_covar=reg_covar_joint, max_iter=100)
    weights = init_gmm_lf.get_w()
    psi0_means, psi0_stds = init_gmm_lf.basis.means_stds()
    psi0_params = torch.stack([psi0_means, psi0_stds], dim=-1)

    psi0_basis = GaussianBasis(uparams_init=psi0_params).to(device)

    # Train the transition model and observation model simultaneously
    if use_dtf:
        tran_obs_model = CompositeConditionalModel([nftf], LinearRFandR2FF(xi_basis, zeta_basis, phi_basis, psi_basis)).to(device)
        optimizers ={"dtf_and_basis": torch.optim.Adam([{'params': tran_obs_model.conditional_density_model.basis_params(), 'lr': obs_and_tran_params["lr_basis_tran"]}, {'params':tran_obs_model.domain_tfs.parameters(), 'lr': obs_and_tran_params["lr_dtf"]}]), 
            "weights": torch.optim.Adam(tran_obs_model.conditional_density_model.weight_params(), lr=obs_and_tran_params["lr_weights"])} 
        tran_conditional_mle_loss_fn = lambda model, xp, x : loss.conditional_mle_loss(model, xp, x, method_name="log_density")
        obs_conditional_mle_loss_fn = lambda model, o, x : obs_loss_weight * loss.conditional_mle_loss(model, o, x, method_name="log_observation_density")
    else:
        tran_obs_model = LinearRFandR2FF(xi_basis, zeta_basis, phi_basis, psi_basis).to(device)
        optimizers ={"basis": torch.optim.Adam(tran_obs_model.basis_params(), lr=obs_and_tran_params["lr_basis_tran"]), "weights": torch.optim.Adam(tran_obs_model.weight_params(), lr=obs_and_tran_params["lr_weights"])} 
        tran_conditional_mle_loss_fn = lambda model, xp, x : loss.conditional_mle_loss(model, xp, x, method_name="log_density")
        obs_conditional_mle_loss_fn = lambda model, o, x : obs_loss_weight * loss.conditional_mle_loss(model, o, x, method_name="log_observation_density")

    print("Training transition and observation model")
    
    tran_obs_model, best_loss_tran_obs, training_time_tran_obs = train.train_iterate(tran_obs_model,
        xp_dataloader,
        labeled_loss_fns={"mle": tran_conditional_mle_loss_fn, "obs_mle": obs_conditional_mle_loss_fn}, 
        labeled_optimizers=optimizers,
        labeled_validation_loss_fns={"val_mle": tran_conditional_mle_loss_fn, "val_obs_mle": obs_conditional_mle_loss_fn},
        validation_data_loader=xp_val_dataloader,
        epochs_per_group=obs_and_tran_params["n_epochs_per_group"],
        iterations=obs_and_tran_params["iterations"],
        verbose=True,
        use_best="val_mle")
    print("Done! \n")
    print("Valid: ", tran_obs_model.valid())

    base_tran_obs_model = tran_obs_model.conditional_density_model if use_dtf else tran_obs_model
    init_model = LinearFF.from_r2ff(base_tran_obs_model, psi0_basis).to(device)
    optimizers = {"basis": torch.optim.Adam(init_model.basis_params(), lr=init_params["lr_basis"]), "weights": torch.optim.Adam(init_model.weight_params(), lr=init_params["lr_weights"])}

    print("Training initial model")
    y0_dataloader = DataLoader(TensorDataset(y0_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu) if use_dtf else x0_dataloader
    init_model, best_loss_init, training_time_init = train.train_iterate(init_model, 
        y0_dataloader, 
        {"mle": loss.mle_loss}, 
        optimizers,
        epochs_per_group=init_params["n_epochs_per_group"],
        iterations=init_params["iterations"],
        verbose=True,
        use_best="mle")
    print("Done! \n")

    print(f"Observation model loss : {best_loss_tran_obs:.4f}, training time: {training_time_tran_obs:.2f} seconds")
    print(f"Initial model loss     : {best_loss_init:.4f}, training time: {training_time_init:.2f} seconds")

    # Extract the individual models for filtering
    if use_dtf:
        tran_model = tran_obs_model.conditional_density_model.r2ff()
        obs_model = tran_obs_model.conditional_density_model.rf()
    else:
        tran_model = tran_obs_model.r2ff()
        obs_model = tran_obs_model.rf()

    # Analysis across repeated simulation tests using the same learned filter
    box_lows = (-5.0, -5.0)
    box_highs = (5.0, 5.0)
    figure_dir = Path("figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    t_dist = SystemTransitionDistribution(system).to(device=device)
    o_dist = SystemObservationDistribution(system).to(device=device)

    for sim_idx in range(n_simulation_tests):
        # Simulate a random true trajectory and observations
        sim_true_states, sim_observations = simulate(
            system,
            init_state_sampler,
            n_timesteps=n_timesteps_prop,
            device=device,
        )

        # Run learned filter
        with torch.no_grad():
            priors, posteriors = propagate.propagate_and_update(
                init_model,
                tran_model,
                obs_model,
                [obs for obs in sim_observations],
            )
        print(f"[sim {sim_idx + 1}/{n_simulation_tests}] priors: {len(priors)}, posteriors: {len(posteriors)}")

        # Run truth-model weighted particle filter baseline
        initial_particle_belief = WeightedParticleSet(
            particles=init_state_sampler(n_particles_true_pf).to(device=device),
            weights=torch.ones(n_particles_true_pf).to(device=device) / n_particles_true_pf,
        )
        _, wpf_posteriors = propagate_and_update(
            belief=initial_particle_belief,
            transition_model=t_dist,
            observation_model=o_dist,
            observations=[obs for obs in sim_observations],
        )


        fig, axes = plt.subplots(n_timesteps_prop, 3, figsize=(10, max(3 * n_timesteps_prop, 15)))
        fig.delaxes(axes[0, 0])  # no prior exists at k=0
        fig.suptitle(f"Beliefs at each time step ({figure_prefix}, sim {sim_idx + 1})")
        with torch.no_grad():
            for i in range(n_timesteps_prop):
                if i > 0:
                    print("Testing prior ", i)
                    check_pdf_valid(priors[i - 1], domain_bounds=(box_lows, box_highs), n_samples=100000, device=device)

                    plot_belief(axes[i, 0], priors[i - 1], x_range=(box_lows[0], box_highs[0]), y_range=(box_lows[1], box_highs[1]))
                    axes[i, 0].set_title(f"Prior at k={i}", fontsize=8)

                print("Testing posterior", i)
                check_pdf_valid(posteriors[i], domain_bounds=(box_lows, box_highs), n_samples=100000, device=device)

                plot_belief(axes[i, 1], posteriors[i], x_range=(box_lows[0], box_highs[0]), y_range=(box_lows[1], box_highs[1]))
                axes[i, 1].set_title(f"Posterior at k={i}", fontsize=8)

                plot_particle_belief(axes[i, 2], wpf_posteriors[i], x_range=(box_lows[0], box_highs[0]), y_range=(box_lows[1], box_highs[1]))
                axes[i, 2].set_title(f"WPF Posterior at k={i}", fontsize=8)
                cols = [1, 2] if i == 0 else [0, 1, 2]
                for j in cols:
                    if j > 0:  # Skip first prior plot
                        axes[i, j].scatter(sim_true_states[i, 0].item(), sim_true_states[i, 1].item(), marker="o", s=10, c="red")
                    if i > 0:
                        axes[i, j].scatter(sim_observations[i - 1, 0].item(), sim_observations[i - 1, 1].item(), marker="o", s=10, c="blue")
                    axes[i, j].set_xlabel("")
                    axes[i, j].set_ylabel("")
                    axes[i, j].tick_params(
                        axis="both",
                        which="both",
                        labelbottom=False,
                        labelleft=False,
                        bottom=False,
                        left=False,
                    )
                    axes[i, j].grid(alpha=0.25)

        figure_path = figure_dir / f"{figure_prefix}__sim_{sim_idx + 1:02d}.png"
        plt.savefig(figure_path, dpi=500, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure to {figure_path}")


