import torch
from torch.utils.data import DataLoader
from pathlib import Path

from rational_factor.models.basis_functions import GaussianBasis, Parameters, PositiveParameters
from rational_factor.models.factor_forms import MLPContextLinearRFF, MLPContextLinearFF
import rational_factor.models.train as train
import rational_factor.models.loss as loss
import rational_factor.tools.propagate as propagate
from rational_factor.tools.misc import train_test_split
from rational_factor.tools.visualization import plot_belief
from rational_factor.tools.analysis import check_pdf_valid
from rational_factor.models.meta_forms import MLPMetaForm
from rational_factor.systems.problems import OPEN_LOOP_OBSERVABLE_PROBLEMS
import matplotlib.pyplot as plt


def mlp_rff_mle_loss(model, xp, x, u, up):
    """MLE for p(x' | x; u, u') with batch fields (u, u', x', x)."""
    return loss.conditional_mle_loss(model, xp, x, u=u, up=up)


def mlp_init_mle_loss(model, x, up):
    """MLE for initial belief p(x; u') with batch fields (u', x)."""
    return -model.log_density(x, up=up).mean()


def mlp_meta_form_param_mse_loss(mlp_form, target_params, **context):
    """MSE between decoded MLP parameters and fixed decoded targets."""
    raw_params = mlp_form.get_params(**context)
    pred_params = mlp_form.decode_params(raw_params)
    return loss.mse_loss(
        pred_params,
        target_params,
        keys=tuple(mlp_form.param_shapes.keys()),
    )


def g_mlp_pretrain_loss(mlp_form, u, *, target_params):
    return mlp_meta_form_param_mse_loss(mlp_form, target_params, u=u)


def psi_mlp_pretrain_loss(mlp_form, u, up, *, target_params):
    return mlp_meta_form_param_mse_loss(mlp_form, target_params, u=u, up=up)


if __name__ == "__main__":
    problem = OPEN_LOOP_OBSERVABLE_PROBLEMS["van_der_pol"]

    use_gpu = torch.cuda.is_available()
    n_basis = 500
    control_dim = problem.system.control_dim()

    pretrain_params = {
        "epochs": 30,
        "lr": 1e-3,
    }
    tran_params = {
        "epochs": 50,
        "lr": 5e-5,
    }
    init_params = {
        "epochs": 50,
        "lr": 5e-5,
    }

    train_batch_size = 256
    val_batch_size = 256
    n_timesteps_prop = 10
    n_controllers = 3
    reg_covar_joint = 1e-5

    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using GPU: ", use_gpu)
    print("Device: ", device)

    system = problem.system
    x0_data = problem.train_initial_state_data()
    u_k_data, x_k_data, x_kp1_data = problem.train_state_transition_data()
    
    u_0_data = u_k_data[:len(x0_data)]
    test_controllers = problem.random_test_controllers(n_controllers)
    test_traj_data = [problem.test_data(controller) for controller in test_controllers]

    x_0_train, x_0_val, u_0_train, u_0_val = train_test_split(x0_data, u_0_data, test_size=0.2)
    x_k_train, x_k_val, x_kp1_train, x_kp1_val, u_k_train, u_k_val = train_test_split(x_k_data, x_kp1_data, u_k_data, test_size=0.2)
    u_k_train_copy = u_k_train.clone()
    u_k_val_copy = u_k_val.clone()

    # Training data loaders
    x0_augmented_dataloader = DataLoader(train.MixedAlignedRandomDataset((x_0_train, u_0_train), randomized_indices=(1,)), batch_size=train_batch_size, shuffle=True, pin_memory=use_gpu)
    xp_augmented_dataloader = DataLoader(train.MixedAlignedRandomDataset((x_kp1_train, x_k_train, u_k_train, u_k_train_copy), randomized_indices=(3,)), batch_size=train_batch_size, shuffle=True, pin_memory=use_gpu)
    # Validation data loaders
    x0_augmented_val_dataloader = DataLoader(train.MixedAlignedRandomDataset((x_0_val, u_0_val), randomized_indices=(1,)), batch_size=val_batch_size, shuffle=True, pin_memory=use_gpu)
    xp_augmented_val_dataloader = DataLoader(train.MixedAlignedRandomDataset((x_kp1_val, x_k_val, u_k_val, u_k_val_copy), randomized_indices=(3,)), batch_size=val_batch_size, shuffle=True, pin_memory=use_gpu)

    
    d = system.dim()
    basis_shape = (d, n_basis)

    g_means = Parameters.random_init(basis_shape, trainable=True, mean=0.0, std=1.0).to(device)
    g_stds = PositiveParameters.set_init(
        basis_shape, value=1.0, trainable=False, normalized=False, epsilon=1e-6
    ).to(device)
    g_coeffs = PositiveParameters.random_init((n_basis,), trainable=True, mean=1.0, std=0.1, normalized=True).to(device)
    g = GaussianBasis(mean_params=g_means, std_params=g_stds, coeffs=g_coeffs).to(device)
    g_mlp_form = MLPMetaForm(
        target_module_dim=system.dim(),
        context_dims={"u": control_dim},
        target_module=g,
        hidden_features=32,
        num_hidden_layers=2,
    ).to(device)

    psi_means = Parameters.random_init(basis_shape, trainable=True, mean=0.0, std=1.0).to(device)
    psi_stds = PositiveParameters.set_init(
        basis_shape, value=1.0, trainable=False, normalized=False, epsilon=1e-6
    ).to(device)
    psi = GaussianBasis(mean_params=psi_means, std_params=psi_stds).to(device)
    psi_mlp_form = MLPMetaForm(
        target_module_dim=system.dim(),
        context_dims={"u": control_dim, "up": control_dim},
        target_module=psi,
        hidden_features=32,
        num_hidden_layers=2,
    ).to(device)

    # Prefit the basis functions
    print("Prefitting the basis functions")
    x_joint_data = torch.cat([x_k_data, x_kp1_data], dim=1)
    gmm_lf = train.fit_gaussian_lf_em(x_joint_data.to(torch.device("cpu")), n_components=n_basis, reg_covar=reg_covar_joint, max_iter=100)
    pre_g_weights = gmm_lf.w.get_coeffs()
    phi_marginal = gmm_lf.marginal(marginal_dims=range(system.dim()))
    pre_g_means, pre_g_stds = phi_marginal.w.means_stds()
    #pre_phi_params = torch.stack([pre_phi_means, pre_phi_stds], dim=-1)

    psi_marginal = gmm_lf.marginal(marginal_dims=range(system.dim(), 2 * system.dim()))
    pre_psi_means, pre_psi_stds = psi_marginal.w.means_stds()
    #pre_psi_params = torch.stack([pre_psi_means, pre_psi_stds], dim=-1)

    pre_g_basis = GaussianBasis(
        mean_params=Parameters.from_values(pre_g_means.to(device), trainable=False),
        std_params=PositiveParameters.from_values(
            pre_g_stds.to(device), trainable=False, normalized=False, epsilon=1e-6
        ),
        coeffs=PositiveParameters.from_values(
            pre_g_weights.to(device), trainable=False, normalized=True
        ),
    ).to(device)
    pre_psi_basis = GaussianBasis(
        mean_params=Parameters.from_values(pre_psi_means.to(device), trainable=False),
        std_params=PositiveParameters.from_values(
            pre_psi_stds.to(device), trainable=False, normalized=False, epsilon=1e-6
        ),
    ).to(device)
    g_target_params = MLPMetaForm.target_param_dict_from_basis(pre_g_basis, layout=g)
    psi_target_params = MLPMetaForm.target_param_dict_from_basis(pre_psi_basis, layout=psi)

    u_g_train_dataloader = DataLoader(
        train.MixedAlignedRandomDataset((u_k_train,), randomized_indices=()),
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=use_gpu,
    )
    u_psi_train_dataloader = DataLoader(
        train.MixedAlignedRandomDataset((u_k_train, u_k_train_copy), randomized_indices=()),
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=use_gpu,
    )

    print("Pretraining g MLPMetaForm to EM-fitted parameters")
    g_pretrain_optimizer = torch.optim.Adam(g_mlp_form.mlp.parameters(), lr=pretrain_params["lr"])
    g_mlp_form, best_loss_g_pre, training_time_g_pre = train.train(
        g_mlp_form,
        u_g_train_dataloader,
        {"mse": lambda m, u: g_mlp_pretrain_loss(m, u, target_params=g_target_params)},
        g_pretrain_optimizer,
        epochs=pretrain_params["epochs"],
        verbose=True,
    )
    print(f"g pretrain loss: {best_loss_g_pre:.6f}, time: {training_time_g_pre:.2f}s\n")

    print("Pretraining psi MLPMetaForm to EM-fitted parameters")
    psi_pretrain_optimizer = torch.optim.Adam(psi_mlp_form.mlp.parameters(), lr=pretrain_params["lr"])
    psi_mlp_form, best_loss_psi_pre, training_time_psi_pre = train.train(
        psi_mlp_form,
        u_psi_train_dataloader,
        {"mse": lambda m, u, up: psi_mlp_pretrain_loss(m, u, up, target_params=psi_target_params)},
        psi_pretrain_optimizer,
        epochs=pretrain_params["epochs"],
        verbose=True,
    )
    print(f"psi pretrain loss: {best_loss_psi_pre:.6f}, time: {training_time_psi_pre:.2f}s\n")

    # Create and train the transition model
    tran_model = MLPContextLinearRFF(g_mlp_form=g_mlp_form, psi_mlp_form=psi_mlp_form).to(device)

    print("Training transition model")
    tran_optimizer = torch.optim.Adam(tran_model.parameters(), lr=tran_params["lr"])
    tran_model, best_loss_tran, training_time_tran = train.train(
        tran_model,
        xp_augmented_dataloader,
        {"mle": mlp_rff_mle_loss},
        tran_optimizer,
        labeled_validation_loss_fns={"val_mle": mlp_rff_mle_loss},
        validation_data_loader=xp_augmented_val_dataloader,
        epochs=tran_params["epochs"],
        verbose=True,
        use_best="val_mle",
        restore_loss_threshold=80,
    )
    print("Done! \n")

    h0_means = Parameters.random_init(basis_shape, trainable=True, mean=0.0, std=1.0).to(device)
    h0_stds = PositiveParameters.set_init(
        basis_shape, value=1.0, trainable=False, normalized=False, epsilon=1e-6
    ).to(device)
    h0_coeffs = PositiveParameters.random_init((n_basis,), trainable=True, mean=1.0, std=0.1, normalized=True).to(device)
    h0 = GaussianBasis(mean_params=h0_means, std_params=h0_stds, coeffs=h0_coeffs).to(device)
    h0_mlp_form = MLPMetaForm(
        target_module_dim=system.dim(),
        context_dims={"up": control_dim},
        target_module=h0,
        hidden_features=32,
        num_hidden_layers=2,
    ).to(device)
    init_model = MLPContextLinearFF(g_mlp_form=g_mlp_form, h_mlp_form=h0_mlp_form).to(device)

    print("Training initial model")
    init_optimizer = torch.optim.Adam(init_model.h_mlp_form.parameters(), lr=init_params["lr"])
    init_model, best_loss_init, training_time_init = train.train(
        init_model,
        x0_augmented_dataloader,
        {"mle": mlp_init_mle_loss},
        init_optimizer,
        labeled_validation_loss_fns={"val_mle": mlp_init_mle_loss},
        validation_data_loader=x0_augmented_val_dataloader,
        epochs=init_params["epochs"],
        verbose=True,
        use_best="val_mle",
    )
    print("Done! \n")

    print(f"Transition model loss: {best_loss_tran:.4f}, training time: {training_time_tran:.2f} seconds")
    print(f"Initial model loss: {best_loss_init:.4f}, training time: {training_time_init:.2f} seconds")

    domain_bounds = (
        tuple(problem.plot_bounds_low.tolist()),
        tuple(problem.plot_bounds_high.tolist()),
    )

    def plot_controller_comparison(
        controller_idx,
        controller,
        true_traj,
        init_belief,
        transition_model,
        out_path,
        n_steps,
    ):
        analysis_device = torch.device("cpu")
        init_belief = init_belief.to(analysis_device).eval()
        transition_model = transition_model.to(analysis_device).eval()

        controls = [controller(k).to(analysis_device) for k in range(n_steps + 1)]
        belief_seq = propagate.propagate_with_control(init_belief, transition_model, controls)

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

            belief_t = belief_seq[t].to(analysis_device).eval()
            print(f"Controller {controller_idx}, t={t}:")
            check_pdf_valid(belief_t, domain_bounds=domain_bounds, n_samples=10000, device=analysis_device)
            plot_belief(ax_belief, belief_t, x_range=x_range, y_range=y_range, n_points=50)
            ax_belief.set_title(f"t={t} belief")

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved comparison figure to {out_path}")

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
            transition_model=tran_model,
            out_path=out_dir / f"controller_{controller_idx}_belief_vs_true.png",
            n_steps=n_plot_steps,
        )
