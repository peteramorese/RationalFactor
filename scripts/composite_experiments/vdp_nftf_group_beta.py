import torch
from torch.utils.data import DataLoader, TensorDataset
from rational_factor.models.basis_functions import UnnormalizedBetaBasis
from rational_factor.models.factor_forms import LinearRFF, LinearFF
import rational_factor.models.train as train
import rational_factor.models.loss as loss
import rational_factor.tools.propagate as propagate
from rational_factor.tools.visualization import plot_belief
from rational_factor.tools.analysis import avg_log_likelihood, check_pdf_valid
from rational_factor.models.domain_transformation import MaskedAffineNFTF, ErfSeparableTF
from rational_factor.models.composite_model import CompositeDensityModel, CompositeConditionalModel
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
import matplotlib.pyplot as plt

if __name__ == "__main__":
    problem = FULLY_OBSERVABLE_PROBLEMS["van_der_pol"]
    
    ###
    use_gpu = torch.cuda.is_available()
    use_dtf = False
    n_basis = 500
    if use_dtf:
        tran_params = {
            "n_epochs_per_group": [15, 5], # dtf_params and basis, weights
            "iterations": 30,
            "lr_basis": 5e-2,
            "lr_weights": 1e-2,
            "lr_dtf": 1e-3,
            "lr_wrap": 1e-3,
        }
        init_params = {
            "n_epochs_per_group": [20, 5], # basis, weights
            "iterations": 40,
            "lr_basis": 1e-2,
            "lr_weights": 1e-2,
        }
    else:
        tran_params = {
            "n_epochs_per_group": [15, 5], # basis, weights
            "iterations": 30,
            "lr_basis": 5e-2,
            "lr_weights": 1e-2,
            "lr_wrap": 1e-3,
        }
        init_params = {
            "n_epochs_per_group": [20, 5], # basis, weights
            "iterations": 40,
            "lr_basis": 1e-2,
            "lr_weights": 1e-2,
        }

    batch_size = 256
    n_timesteps_prop = problem.n_timesteps
    #var_reg_strength = 5e-3
    var_reg_strength = 0#1e-2
    ###

    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using GPU: ", use_gpu)
    print("Device: ", device)

    system = problem.system
    x0_data, x_k_data, x_kp1_data = problem.train_state_data()
    test_traj_data = problem.test_data()

    x0_dataloader = DataLoader(TensorDataset(x0_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(TensorDataset(x_kp1_data, x_k_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    # Create basis functions
    phi_basis =  UnnormalizedBetaBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([10.0, 10.0], device=device), variance=30.0, min_concentration=1.0).to(device)
    psi_basis =  UnnormalizedBetaBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([10.0, 10.0], device=device), variance=30.0, min_concentration=1.0).to(device)
    psi0_basis = UnnormalizedBetaBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([10.0, 10.0], device=device), variance=30.0, min_concentration=1.0).to(device)

    # Create separable domain transformation
    wrap_tf = ErfSeparableTF.from_data(x_k_data, trainable=True)
    
    nftf = MaskedAffineNFTF(system.dim(), trainable=True, hidden_features=128, n_layers=5).to(device) if use_dtf else None

    # Create and train the transition model
    if use_dtf:
        tran_model = CompositeConditionalModel([nftf, wrap_tf], LinearRFF(phi_basis, psi_basis, numerical_tolerance=problem.numerical_tolerance)).to(device)
    else:
        tran_model = CompositeConditionalModel([wrap_tf], LinearRFF(phi_basis, psi_basis, numerical_tolerance=problem.numerical_tolerance)).to(device)

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
    box_lows = tuple(problem.plot_bounds_low.tolist())
    box_highs = tuple(problem.plot_bounds_high.tolist())

    if use_dtf:
        base_belief_seq = propagate.propagate(init_model.density_model, tran_model.conditional_density_model, n_steps=n_timesteps_prop)
        belief_seq = [CompositeDensityModel([trained_nftf, trained_domain_tf], belief) for belief in base_belief_seq]
    else:
        base_belief_seq = propagate.propagate(init_model.density_model, tran_model.conditional_density_model, n_steps=n_timesteps_prop)
        belief_seq = [CompositeDensityModel([trained_domain_tf], belief) for belief in base_belief_seq]

    fig, axes = plt.subplots(2, n_timesteps_prop, figsize=(20, 10))
    fig.suptitle("Beliefs at each time step")
    for i in range(n_timesteps_prop):
        data_i = test_traj_data[i].to(device)
        ll = avg_log_likelihood(belief_seq[i], data_i)
        print(f"Log likelihood at time {i}: {ll:.4f}")
        check_pdf_valid(belief_seq[i], domain_bounds=(box_lows, box_highs), n_samples=100000, device=device)
        plot_belief(axes[1, i], belief_seq[i], x_range=(box_lows[0], box_highs[0]), y_range=(box_lows[1], box_highs[1]))
        axes[0, i].scatter(test_traj_data[i][:, 0], test_traj_data[i][:, 1], s=1)
        axes[0, i].set_aspect("equal")
        axes[0, i].set_xlim(box_lows[0], box_highs[0])
        axes[0, i].set_ylim(box_lows[1], box_highs[1])
    

    plt.savefig("figures/vdp_nfdf_beta_beliefs.pdf", dpi=1000)
    #plt.show()


