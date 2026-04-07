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
from rational_factor.tools.analysis import mc_integral_box
from rational_factor.models.domain_transformation import MaskedAffineNFTF, ErfSeparableTF
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
    use_dtf = True
    n_basis = 1000
    if use_dtf:
        tran_params = {
            "n_epochs_per_group": [10, 5], # dtf_params and basis, weights
            "iterations": 20,
            "lr_basis": 1e-2,
            "lr_weights": 1e-2,
            "lr_dtf": 1e-3,
            "lr_wrap": 1e-3,
        }
        init_params = {
            "n_epochs_per_group": [20, 5], # basis, weights
            "iterations": 15,
            "lr_basis": 1e-2,
            "lr_weights": 1e-2,
        }
    else:
        tran_params = {
            "n_epochs_per_group": [10, 5], # basis, weights
            "iterations": 20,
            "lr_basis": 1e-2,
            "lr_weights": 1e-2,
            "lr_wrap": 1e-3,
        }
        init_params = {
            "n_epochs_per_group": [20, 5], # basis, weights
            "iterations": 15,
            "lr_basis": 1e-2,
            "lr_weights": 1e-2,
        }

    batch_size = 256
    n_timesteps_train = 10
    n_timesteps_prop = 10
    n_trajectories_train = 1000
    n_pairs_train = n_timesteps_train * n_trajectories_train
    n_init_train = 5000
    #var_reg_strength = 5e-3
    var_reg_strength = 1e-3
    ###

    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using GPU: ", use_gpu)
    print("Device: ", device)

    # Create system
    system = truth_models.VanDerPol(dt=0.3, mu=0.9, covariance=0.1*torch.eye(2))

    def init_state_sampler(n_samples : int):
        mean = torch.tensor([0.2, 0.1])
        cov = torch.diag(torch.tensor([0.2, 0.2]))
        dist = torch.distributions.MultivariateNormal(mean, cov)
        return dist.sample((n_samples,))

    ## Generate data set from trajectories
    traj_data = sample_trajectories(system, init_state_sampler, n_timesteps=n_timesteps_train, n_trajectories=n_trajectories_train)
    #x0 = traj_data[0]
    #x_k, x_kp1 = create_transition_data_matrix(traj_data, separate=True)

    # Generate data as input output pairs
    def prev_state_sampler(n_samples : int):
        mean = torch.tensor([0.0, 0.0])
        cov = torch.diag(4.0 * torch.ones(system.dim()))
        dist = torch.distributions.MultivariateNormal(mean, cov)
        return dist.sample((n_samples,))

    x0 = init_state_sampler(n_init_train)
    x_k, x_kp1 = sample_io_pairs(system, prev_state_sampler, n_pairs=n_pairs_train)

    x0_data = TensorDataset(x0)
    xp_data = TensorDataset(x_k, x_kp1)

    x0_dataloader = DataLoader(x0_data, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(xp_data, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    # Create basis functions
    phi_basis =  BetaBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([10.0, 10.0], device=device), variance=30.0, min_concentration=1.0).to(device)
    psi_basis =  BetaBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([10.0, 10.0], device=device), variance=30.0, min_concentration=1.0).to(device)
    psi0_basis = BetaBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([10.0, 10.0], device=device), variance=30.0, min_concentration=1.0).to(device)

    # Create separable domain transformation
    wrap_tf = ErfSeparableTF.from_data(x_k, trainable=True)
    
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
    box_lows = (-5.0, -5.0)
    box_highs = (5.0, 5.0)

    if use_dtf:
        base_belief_seq = propagate.propagate(init_model.density_model, tran_model.conditional_density_model, n_steps=n_timesteps_prop)
        belief_seq = [CompositeDensityModel([trained_nftf, trained_domain_tf], belief) for belief in base_belief_seq]
    else:
        base_belief_seq = propagate.propagate(init_model.density_model, tran_model.conditional_density_model, n_steps=n_timesteps_prop)
        belief_seq = [CompositeDensityModel([trained_domain_tf], belief) for belief in base_belief_seq]

    fig, axes = plt.subplots(2, n_timesteps_prop, figsize=(20, 10))
    fig.suptitle("Beliefs at each time step")
    for i in range(n_timesteps_prop):
        #print("Printing belief: ", i)
        plot_belief(axes[1, i], belief_seq[i], x_range=(box_lows[0], box_highs[0]), y_range=(box_lows[1], box_highs[1]))
        axes[0, i].scatter(traj_data[i][:, 0], traj_data[i][:, 1], s=1)
        axes[0, i].set_aspect("equal")
        axes[0, i].set_xlim(box_lows[0], box_highs[0])
        axes[0, i].set_ylim(box_lows[1], box_highs[1])
    

    plt.savefig("figures/vdp_nfdf_beta_beliefs.pdf", dpi=1000)
    #plt.show()


