import torch
from torch.utils.data import DataLoader, TensorDataset
from rational_factor.models.basis_functions import UnnormalizedBetaBasis
from rational_factor.models.factor_forms import QuadraticRFF, QuadraticFF, LinearRFF, LinearFF
import rational_factor.models.train as train
import rational_factor.models.loss as loss
import rational_factor.tools.propagate as propagate
from rational_factor.tools.visualization import plot_belief
from rational_factor.tools.analysis import mc_integral_box
from rational_factor.models.domain_transformation import ErfSeparableTF
from rational_factor.models.composite_model import CompositeDensityModel, CompositeConditionalModel
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
import matplotlib.pyplot as plt

if __name__ == "__main__":
    problem = FULLY_OBSERVABLE_PROBLEMS["van_der_pol"]
    
    ###
    use_gpu = torch.cuda.is_available()
    n_basis = 20
    n_epochs = 300
    batch_size = 256
    learning_rate = 1e-2
    n_timesteps_prop = problem.n_timesteps
    #var_reg_strength = 5e-3
    var_reg_strength = 0.0
    psd_reg_strength = 1e-2 #0.002
    ###

    system = problem.system
    x0_train, x_k, x_kp1 = problem.train_data()
    traj_data = problem.test_data()
    x0_data = TensorDataset(x0_train)
    xp_data = TensorDataset(x_k, x_kp1)

    x0_dataloader = DataLoader(x0_data, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(xp_data, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    # Create basis functions
    phi_basis =  UnnormalizedBetaBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([1.0, 1.0]), min_concentration=1.0, variance=10.0)
    psi_basis =  UnnormalizedBetaBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([1.0, 1.0]), min_concentration=1.0, variance=10.0)
    psi0_basis = UnnormalizedBetaBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([1.0, 1.0]), min_concentration=1.0, variance=10.0)

    print("alpha beta b4: ", phi_basis.alphas_betas())


    # Create separable domain transformation
    domain_tf = ErfSeparableTF.from_data(x_k, trainable=True)
    print("domain tf loc: ", domain_tf.params[:, 0])
    print("domain tf scale: ", torch.square(domain_tf.params[:, 1]))

    # Create and train the transition model
    tran_model = CompositeConditionalModel(domain_tf, QuadraticRFF(phi_basis, psi_basis))
    #tran_model = CompositeConditionalModel(domain_tf, LinearRFF(phi_basis, psi_basis))
    print("Training transition model")
    mle_loss_fn = loss.conditional_mle_loss
    var_reg_loss_fn = lambda model, x, xp : var_reg_strength * (loss.beta_basis_concentration_reg_loss(model.conditional_density_model.phi_basis) + loss.beta_basis_concentration_reg_loss(model.conditional_density_model.psi_basis))
    psd_loss_fn = lambda model, x, xp : psd_reg_strength * loss.B_psd_loss(model.conditional_density_model, penalty_offset=10.0, exponent=4.0)

    # Train model to feasible region
    vld_psd_loss_fn = lambda model: psd_reg_strength * loss.B_psd_loss(model.conditional_density_model, penalty_offset=10.0, exponent=4.0)
    tran_model = train.train_to_valid(tran_model, 
        {"psd": vld_psd_loss_fn}, 
        torch.optim.Adam(tran_model.parameters(), lr=1.0), epochs=10000, use_best="psd")

    print("Valid before training?: ", tran_model.valid())
    tran_model = train.train(tran_model, 
        xp_dataloader, 
        {"mle": mle_loss_fn, "var_reg": var_reg_loss_fn, "psd": psd_loss_fn}, 
        #{"mle": mle_loss_fn, "var_reg": var_reg_loss_fn}, 
        torch.optim.Adam(tran_model.parameters(), lr=learning_rate), epochs=n_epochs, use_best="mle")
    print("Done! \n")
    print("Valid: ", tran_model.valid())

    print("alpha beta: ", phi_basis.alphas_betas())
    print("domain tf loc: ", domain_tf.params[:, 0])
    print("domain tf scale: ", torch.square(domain_tf.params[:, 1]))

    # Copy the domain transformation to fix it for training the initial state model
    trained_domain_tf = ErfSeparableTF.copy_from_trainable(domain_tf)

    init_model = CompositeDensityModel(trained_domain_tf, QuadraticFF.from_rff(tran_model.conditional_density_model, psi0_basis))
    #init_model = CompositeDensityModel(trained_domain_tf, LinearFF.from_rff(tran_model.conditional_density_model, psi0_basis))
    print("Training initial model")
    mle_loss_fn = loss.mle_loss
    var_reg_loss_fn = lambda model, x : var_reg_strength * loss.beta_basis_concentration_reg_loss(model.density_model.psi0_basis)
    init_model = train.train(init_model, 
        x0_dataloader, 
        {"mle": mle_loss_fn, "var_reg": var_reg_loss_fn}, 
        torch.optim.Adam(init_model.parameters(), lr=learning_rate), epochs=n_epochs, use_best="mle")
    print("Done! \n")

    # Analysis
    box_lows = tuple(problem.plot_bounds_low.tolist())
    box_highs = tuple(problem.plot_bounds_high.tolist())

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


