import torch
from torch.utils.data import DataLoader, TensorDataset
from rational_factor.models.basis_functions import QuadraticExpBasis
from rational_factor.models.factor_forms import QuadraticRFF, QuadraticFF, LinearRFF, LinearFF
import rational_factor.models.train as train
import rational_factor.models.loss as loss
import rational_factor.tools.propagate as propagate
from rational_factor.tools.visualization import plot_belief
from rational_factor.tools.analysis import mc_integral_box
from rational_factor.models.domain_transformation import MaskedAffineNFTF, ErfSeparableTF
from rational_factor.models.composite_model import CompositeDensityModel, CompositeConditionalModel
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
import matplotlib.pyplot as plt

if __name__ == "__main__":
    problem = FULLY_OBSERVABLE_PROBLEMS["van_der_pol"]
    
    ###
    use_gpu = torch.cuda.is_available()
    use_dtf = False
    n_basis = 80
    n_epochs_tran = 600
    n_epochs_init = 500
    batch_size = 1024
    lr_tran = 1e-1
    lr_init = 1e-2
    n_timesteps_prop = problem.n_timesteps
    psd_strength = 1e-2
    var_reg_strength = 0 #1e-2
    ###

    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using GPU: ", use_gpu)
    print("Device: ", device)

    system = problem.system
    x0, x_k, x_kp1 = problem.train_data()
    traj_data = problem.test_data()

    x0_dataloader = DataLoader(TensorDataset(x0), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(TensorDataset(x_kp1, x_k), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    # Create basis functions
    phi_basis =  QuadraticExpBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 10.0], device=device), variance=20.0, min_std=1e-5).to(device)
    psi_basis =  QuadraticExpBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 10.0], device=device), variance=20.0, min_std=1e-5).to(device)
    psi0_basis = QuadraticExpBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 10.0], device=device), variance=20.0, min_std=1e-5).to(device)

    # Create separable domain transformation
    nftf = MaskedAffineNFTF(system.dim(), trainable=True, hidden_features=128, n_layers=5).to(device) if use_dtf else None

    # Create and train the transition model
    if use_dtf:
        tran_model = CompositeConditionalModel([nftf], QuadraticRFF(phi_basis, psi_basis)).to(device)
    else:
        tran_model = QuadraticRFF(phi_basis, psi_basis).to(device)

    print("Training transition model")
    mle_loss_fn = loss.conditional_mle_loss
    
    if use_dtf:
        var_reg_loss_fn = lambda model, x, xp : var_reg_strength * (loss.gaussian_basis_var_reg_loss(model.conditional_density_model.phi_basis, mean=True) + loss.gaussian_basis_var_reg_loss(model.conditional_density_model.psi_basis, mean=True))
        psd_loss_fn = lambda model, x, xp : psd_strength * loss.B_psd_loss(model.conditional_density_model)
    else:
        var_reg_loss_fn = lambda model, x, xp : var_reg_strength * (loss.gaussian_basis_var_reg_loss(model.phi_basis, mean=True) + loss.gaussian_basis_var_reg_loss(model.psi_basis, mean=True))
        psd_loss_fn = lambda model, x, xp : psd_strength * loss.B_psd_loss(model)

    tran_model, best_loss_tran, training_time_tran = train.train(tran_model, 
        xp_dataloader, 
        {"mle": mle_loss_fn, "var_reg": var_reg_loss_fn, "psd": psd_loss_fn}, 
        torch.optim.Adam(tran_model.parameters(), lr=lr_tran), epochs=n_epochs_tran, use_best="mle")
    print("Done! \n")
    print("Valid: ", tran_model.valid())


    # Copy the domain transformation to fix it for training the initial state model
    trained_nftf = MaskedAffineNFTF.copy_from_trainable(nftf).to(device) if use_dtf else None

    #init_model = CompositeDensityModel(trained_domain_tf, QuadraticFF.from_rff(tran_model.conditional_density_model, psi0_basis))
    if use_dtf:
        init_model = CompositeDensityModel([trained_nftf], QuadraticFF.from_rff(tran_model.conditional_density_model, psi0_basis)).to(device)
    else:
        init_model = QuadraticFF.from_rff(tran_model, psi0_basis).to(device)

    print("Training initial model")
    mle_loss_fn = loss.mle_loss

    if use_dtf:
        var_reg_loss_fn = lambda model, x : var_reg_strength * loss.gaussian_basis_var_reg_loss(model.density_model.psi0_basis, mean=True)
    else:
        var_reg_loss_fn = lambda model, x : var_reg_strength * loss.gaussian_basis_var_reg_loss(model.psi0_basis, mean=True)

    init_model, best_loss_init, training_time_init = train.train(init_model, 
        x0_dataloader, 
        {"mle": mle_loss_fn, "var_reg": var_reg_loss_fn}, 
        torch.optim.Adam(init_model.parameters(), lr=lr_init), epochs=n_epochs_init, use_best="mle")
    print("Done! \n")

    print(f"Transition model loss: {best_loss_tran:.4f}, training time: {training_time_tran:.2f} seconds")
    print(f"Initial model loss: {best_loss_init:.4f}, training time: {training_time_init:.2f} seconds")

    # Analysis
    box_lows = tuple(problem.plot_bounds_low.tolist())
    box_highs = tuple(problem.plot_bounds_high.tolist())

    if use_dtf:
        base_belief_seq = propagate.propagate(init_model.density_model, tran_model.conditional_density_model, n_steps=n_timesteps_prop)
        belief_seq = [CompositeDensityModel([trained_nftf], belief) for belief in base_belief_seq]
    else:
        belief_seq = propagate.propagate(init_model, tran_model, n_steps=n_timesteps_prop)

    fig, axes = plt.subplots(2, n_timesteps_prop, figsize=(20, 10))
    fig.suptitle("Beliefs at each time step")
    for i in range(n_timesteps_prop):
        #print("Printing belief: ", i)
        plot_belief(axes[1, i], belief_seq[i], x_range=(box_lows[0], box_highs[0]), y_range=(box_lows[1], box_highs[1]))
        axes[0, i].scatter(traj_data[i][:, 0], traj_data[i][:, 1], s=1)
        axes[0, i].set_aspect("equal")
        axes[0, i].set_xlim(box_lows[0], box_highs[0])
        axes[0, i].set_ylim(box_lows[1], box_highs[1])
    

    #box_lows = (-20.0, -20.0)
    #box_highs = (20.0, 20.0)

    #fig, axes = plt.subplots(2, n_timesteps_prop)
    #fig.suptitle("Base beliefs at each time step")
    #for i in range(n_timesteps_prop):
    #    #print("Printing belief: ", i)
    #    plot_belief(axes[1, i], base_belief_seq[i], x_range=(box_lows[0], box_highs[0]), y_range=(box_lows[1], box_highs[1]))
    #    axes[0, i].scatter(traj_data[i][:, 0], traj_data[i][:, 1], s=1)
    #    axes[0, i].set_aspect("equal")
    #    axes[0, i].set_xlim(box_lows[0], box_highs[0])
    #    axes[0, i].set_ylim(box_lows[1], box_highs[1])

    plt.savefig("figures/vdp_nfdf_gaussian_beliefs.pdf", dpi=1000)
    #plt.show()


