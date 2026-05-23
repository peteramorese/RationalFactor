import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.factor_forms import LinearRFF, LinearFF
import rational_factor.models.train as train
import rational_factor.models.loss as loss
from rational_factor.models.density_model import LogisticSigmoid
import rational_factor.tools.propagate as propagate
from rational_factor.tools.misc import data_bounds, train_test_split
from rational_factor.tools.visualization import plot_belief
from rational_factor.models.domain_transformation import MaskedAffineNFTF
from rational_factor.models.composite_model import CompositeDensityModel, CompositeConditionalModel
from rational_factor.models.meta_forms import MLPMetaForm
from rational_factor.systems.problems import OPEN_LOOP_OBSERVABLE_PROBLEMS
import matplotlib.pyplot as plt

from rational_factor.tools.misc import make_mvnormal_state_sampler, make_uniform_state_sampler


if __name__ == "__main__":
    problem = OPEN_LOOP_OBSERVABLE_PROBLEMS["van_der_pol"]

    ###
    use_gpu = torch.cuda.is_available()
    n_basis = 500

    tran_params = {
        "epochs": 100,
        "lr": 5e-3,
    }
    init_params = {
        "n_epochs_per_group": [20, 5], # basis, weights
        "iterations": 50,
        "lr_basis": 5e-3,
        "lr_weights": 1e-3,
    }

    train_batch_size = 256
    val_batch_size = 4096
    n_timesteps_prop = 10
    reg_covar_joint = 1e-3
    n_controllers = 3
    ###

    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using GPU: ", use_gpu)
    print("Device: ", device)

    system = problem.system
    x0_data = problem.train_initial_state_data()
    u_k_data, x_k_data, x_kp1_data = problem.train_state_transition_data()
    test_controllers = problem.random_test_controllers(n_controllers)
    test_traj_data = [problem.test_data(controller) for controller in test_controllers]

    x0_train, x0_val = train_test_split(x0_data, test_size=0.2)
    u_k_train, u_k_val, x_k_train, x_k_val, x_kp1_train, x_kp1_val = train_test_split(x_k_data, x_kp1_data, u_k_data, test_size=0.2)

    # Training data loaders
    x0_dataloader = DataLoader(TensorDataset(x0_train), batch_size=train_batch_size, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(TensorDataset(u_k_train, x_kp1_train, x_k_train), batch_size=train_batch_size, shuffle=True, pin_memory=use_gpu)
    x_k_dataloader = DataLoader(TensorDataset(x_k_data), batch_size=train_batch_size, shuffle=True, pin_memory=use_gpu)

    # Validation data loaders
    x0_val_dataloader = DataLoader(TensorDataset(x0_val), batch_size=val_batch_size, shuffle=True, pin_memory=use_gpu)
    xp_val_dataloader = DataLoader(TensorDataset(u_k_val, x_kp1_val, x_k_val), batch_size=val_batch_size, shuffle=True, pin_memory=use_gpu)


    ## Prefit the basis functions
    #print("Prefitting the basis functions")
    #y_joint_data = torch.cat([y_k_data, y_kp1_data], dim=1)
    #gmm_lf = train.fit_gaussian_lf_em(y_joint_data.to(torch.device("cpu")), n_components=n_basis, reg_covar=reg_covar_joint, max_iter=100)
    #weights = gmm_lf.get_w()
    #phi_marginal = gmm_lf.basis.marginal(marginal_dims=range(system.dim()))
    #phi_means, phi_stds = phi_marginal.means_stds()
    #phi_params = torch.stack([phi_means, phi_stds], dim=-1)

    #psi_marginal = gmm_lf.basis.marginal(marginal_dims=range(system.dim(), 2 * system.dim()))
    #psi_means, psi_stds = psi_marginal.means_stds()
    #psi_params = torch.stack([psi_means, psi_stds], dim=-1)

    #phi_basis = GaussianBasis(uparams_init=phi_params).to(device)
    #psi_basis = GaussianBasis(uparams_init=psi_params).to(device)
    #psi0_basis = GaussianBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 20.0], device=device), variance=30.0, min_std=1e-4).to(device)
    #print("Done.\n")
    phi_basis = GaussianBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 20.0], device=device), variance=30.0, min_std=1e-4).to(device)
    psi_basis = GaussianBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 20.0], device=device), variance=30.0, min_std=1e-4).to(device)

    # Create and train the transition model
    tran_model = MLPMetaForm(context_dim=system.control_dim(), target_module=LinearRFF(phi_basis, psi_basis), hidden_features=128, num_hidden_layers=3).to(device)

    print("Training transition model")
    
    #optimizers ={
    #    "dtf_and_basis": torch.optim.Adam([{'params': tran_model.conditional_density_model.basis_params(), 'lr': tran_params["lr_basis"]}, {'params':tran_model.domain_tfs.parameters(), 'lr': tran_params["lr_dtf"]}]), 
    #    "weights": torch.optim.Adam(tran_model.conditional_density_model.weight_params(), lr=tran_params["lr_weights"])} 

    optimizer = torch.optim.Adam(tran_model.parameters(), lr=tran_params["lr"])

    def meta_conditional_mle_loss(model, u, xp, x):
        if u.ndim == 1:
            rff = model.instantiate(u)
            return loss.conditional_mle_loss(rff, xp, x)
        batch_losses = [
            loss.conditional_mle_loss(model.instantiate(u[i]), xp[i], x[i])
            for i in range(u.shape[0])
        ]
        return torch.stack(batch_losses).mean()

    tran_model, best_loss_tran, training_time_tran = train.train(tran_model,
        xp_dataloader,
        {"mle": meta_conditional_mle_loss},
        optimizer,
        epochs=tran_params["epochs"],
        verbose=True,
        use_best="mle")
    print("Done! \n")


    psi0_basis = GaussianBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 20.0], device=device), variance=30.0, min_std=1e-4).to(device)
    init_rff = tran_model.instantiate(torch.zeros(system.control_dim(), device=device))
    init_model = LinearFF.from_rff(init_rff, psi0_basis).to(device)

    print("Training initial model")

    optimizers = {"basis": torch.optim.Adam(init_model.basis_params(), lr=init_params["lr_basis"]), "weights": torch.optim.Adam(init_model.weight_params(), lr=init_params["lr_weights"])}

    init_model, best_loss_init, training_time_init = train.train_iterate(init_model, 
        x0_dataloader, 
        labeled_loss_fns={"mle": loss.mle_loss}, 
        labeled_optimizers=optimizers,
        labeled_validation_loss_fns={"val_mle": loss.mle_loss},
        validation_data_loader=x0_val_dataloader,
        epochs_per_group=init_params["n_epochs_per_group"],
        iterations=init_params["iterations"],
        verbose=True,
        use_best="val_mle")
    print("Done! \n")

    print(f"Transition model loss: {best_loss_tran:.4f}, training time: {training_time_tran:.2f} seconds")
    print(f"Initial model loss: {best_loss_init:.4f}, training time: {training_time_init:.2f} seconds")

    def propagate_with_controller(init_belief, meta_form, controller, n_steps, device):
        """Propagate beliefs with a time-varying LRFF proxy from MLPMetaForm.instantiate(u_k)."""
        belief_seq = [init_belief]
        for k in range(n_steps):
            u = torch.as_tensor(controller(k), device=device, dtype=torch.float32).flatten()
            rff = meta_form.instantiate(u)
            belief_seq.append(propagate.propagate(belief_seq[-1], rff, n_steps=1, device=device)[-1])
        return belief_seq

    def plot_controller_comparison(
        controller_idx,
        controller,
        true_traj,
        init_belief,
        meta_form,
        out_path,
        n_steps,
        device,
    ):
        analysis_device = torch.device("cpu")
        init_belief = init_belief.to(analysis_device).eval()
        meta_form = meta_form.to(analysis_device).eval()

        belief_seq = [
            belief.to(analysis_device).eval()
            for belief in propagate_with_controller(init_belief, meta_form, controller, n_steps, analysis_device)
        ]

        n_cols = min(len(true_traj), len(belief_seq))
        x_range = (problem.plot_bounds_low[0].item(), problem.plot_bounds_high[0].item())
        y_range = (problem.plot_bounds_low[1].item(), problem.plot_bounds_high[1].item())

        fig, axes = plt.subplots(2, n_cols, figsize=(2.8 * n_cols, 5.5), squeeze=False)
        fig.suptitle(f"Controller {controller_idx}: true trajectories vs LRFF belief propagation")

        for t in range(n_cols):
            ax_true = axes[0, t]
            ax_belief = axes[1, t]

            states = true_traj[t].detach().cpu()
            ax_true.scatter(states[:, 0], states[:, 1], s=0.6, alpha=0.35, c="C0")
            ax_true.set_aspect("equal")
            ax_true.set_xlim(*x_range)
            ax_true.set_ylim(*y_range)
            ax_true.set_title(f"t={t} true")

            plot_belief(ax_belief, belief_seq[t], x_range=x_range, y_range=y_range)
            ax_belief.set_title(f"t={t} belief")

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved comparison figure to {out_path}")

    # Analysis: compare ground-truth rollouts to belief propagation per test controller
    analysis_device = torch.device("cpu")
    init_model = init_model.to(analysis_device).eval()
    tran_model = tran_model.to(analysis_device).eval()

    n_plot_steps = min(problem.n_timesteps, n_timesteps_prop)
    out_dir = Path("figures") / "random_control_test"

    for controller_idx, (controller, true_traj) in enumerate(zip(test_controllers, test_traj_data)):
        plot_controller_comparison(
            controller_idx=controller_idx,
            controller=controller,
            true_traj=true_traj,
            init_belief=init_model,
            meta_form=tran_model,
            out_path=out_dir / f"controller_{controller_idx}_belief_vs_true.png",
            n_steps=n_plot_steps,
            device=analysis_device,
        )


