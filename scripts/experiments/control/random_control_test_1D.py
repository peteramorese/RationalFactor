import torch
from torch.utils.data import DataLoader
from pathlib import Path
import copy

from rational_factor.models.basis_functions import GaussianBasis, Parameters, PositiveParameters
from rational_factor.models.factor_forms import MLPContextLinearRFF, MLPContextLinearFF, LinearRFF
import rational_factor.models.train as train
import rational_factor.models.loss as loss
import rational_factor.tools.propagate as propagate
from rational_factor.tools.misc import train_test_split
from rational_factor.tools.analysis import check_pdf_valid, mc_integral_box
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


def _print_mlp_param_minmax(mlp_form: MLPMetaForm, label: str, **context) -> None:
    """Print min/max for each raw and decoded MLP parameter tensor at the given context."""
    raw = mlp_form.get_params(**context)
    decoded = mlp_form.decode_params(raw)
    print(f"      {label} — raw MLP outputs:")
    for name, tensor in raw.items():
        print(f"         {name}: min={tensor.min().item():.6g}, max={tensor.max().item():.6g}")
    print(f"      {label} — decoded parameters:")
    for name, tensor in decoded.items():
        print(f"         {name}: min={tensor.min().item():.6g}, max={tensor.max().item():.6g}")


def check_mlp_context_transition_pdf_valid(
    tran_model: MLPContextLinearRFF,
    domain_bounds,
    conditioner_domain_bounds,
    u_up_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    n_samples: int = 50_000,
    n_conditioner_samples: int = 8,
    device: torch.device | None = None,
    atol: float = 0.25,
) -> bool:
    """MC-check that p(x' | x, u, up) integrates to 1 for fixed (u, up) pairs."""
    if device is None:
        device = torch.device("cpu")

    domain_lows = torch.as_tensor(domain_bounds[0], device=device, dtype=torch.float32)
    domain_highs = torch.as_tensor(domain_bounds[1], device=device, dtype=torch.float32)
    domain_bounds_dev = (domain_lows, domain_highs)
    conditioner_lows = torch.as_tensor(conditioner_domain_bounds[0], device=device, dtype=torch.float32)
    conditioner_highs = torch.as_tensor(conditioner_domain_bounds[1], device=device, dtype=torch.float32)
    conditioner_samples = (
        torch.rand(n_conditioner_samples, conditioner_lows.numel(), device=device, dtype=conditioner_lows.dtype)
        * (conditioner_highs - conditioner_lows)
        + conditioner_lows
    )

    tran_model = tran_model.to(device).eval()
    all_valid = True
    with torch.no_grad():
        for pair_idx, (u_vec, up_vec) in enumerate(u_up_pairs):
            u_ctx = torch.as_tensor(u_vec, device=device, dtype=torch.float32).reshape(1, -1)
            up_ctx = torch.as_tensor(up_vec, device=device, dtype=torch.float32).reshape(1, -1)
            integrals = []
            for i in range(n_conditioner_samples):
                conditioner = conditioner_samples[i]

                def density_xp(xp, conditioner=conditioner, u_ctx=u_ctx, up_ctx=up_ctx):
                    x_batch = conditioner.unsqueeze(0).expand(xp.shape[0], -1)
                    u_batch = u_ctx.expand(xp.shape[0], -1)
                    up_batch = up_ctx.expand(xp.shape[0], -1)
                    return tran_model.forward(xp, conditioner=x_batch, u=u_batch, up=up_batch)

                integrals.append(mc_integral_box(density_xp, domain_bounds_dev, n_samples, device=device))

            stacked = torch.stack(integrals)
            pair_valid = bool((stacked - 1.0).abs().max().item() < atol)
            all_valid = all_valid and pair_valid
            status = "ok" if pair_valid else "INVALID"
            u_val = float(u_ctx.reshape(-1)[0])
            up_val = float(up_ctx.reshape(-1)[0])
            print(
                f"   (u, up) pair {pair_idx + 1}: u={u_val:.4f}, up={up_val:.4f} — "
                f"integral mean={stacked.mean().item():.4f}, std={stacked.std(unbiased=False).item():.4f}, "
                f"min={stacked.min().item():.4f}, max={stacked.max().item():.4f} [{status}]"
            )
            if not pair_valid:
                print("      MLP parameter ranges for invalid pair:")
                _print_mlp_param_minmax(tran_model.g_mlp_form, "g_mlp_form @ u", u=u_ctx)
                _print_mlp_param_minmax(tran_model.g_mlp_form, "g_mlp_form @ up", u=up_ctx)
                _print_mlp_param_minmax(
                    tran_model.psi_mlp_form, "psi_mlp_form @ (u, up)", u=u_ctx, up=up_ctx
                )

    return all_valid


def plot_transition_u_up_comparison(
    tran_model,
    x_k_data,
    x_kp1_data,
    u_k_data,
    problem,
    out_path,
    *,
    baseline_rff=None,
    u_up_pairs=None,
    n_pairs=4,
    n_grid=120,
    u_tol=0.3,
    random_seed=0,
):
    """Compare EM-fitted RFF, MLP-context RFF, and empirical (x, x') samples near u."""
    lo = float(problem.plot_bounds_low[0].item())
    hi = float(problem.plot_bounds_high[0].item())
    control_dim = problem.system.control_dim()

    analysis_device = torch.device("cpu")
    tran_model = tran_model.to(analysis_device).eval()
    if baseline_rff is not None:
        baseline_rff = baseline_rff.to(analysis_device).eval()

    x_k_cpu = x_k_data.detach().cpu()
    x_kp1_cpu = x_kp1_data.detach().cpu()
    u_k_cpu = u_k_data.detach().cpu().view(-1, control_dim)

    if u_up_pairs is None:
        g = torch.Generator().manual_seed(random_seed)
        idx_u = torch.randint(0, len(u_k_cpu), (n_pairs,), generator=g)
        idx_up = torch.randint(0, len(u_k_cpu), (n_pairs,), generator=g)
        u_up_pairs = [(u_k_cpu[i].clone(), u_k_cpu[j].clone()) for i, j in zip(idx_u, idx_up)]

    nx = n_grid
    nxp = n_grid
    x_1d = torch.linspace(lo, hi, nx, device=analysis_device)
    xp_1d = torch.linspace(lo, hi, nxp, device=analysis_device)
    x_cond = x_1d.unsqueeze(1).expand(nx, nxp).reshape(-1, 1)
    xp_flat = xp_1d.unsqueeze(0).expand(nx, nxp).reshape(-1, 1)

    n_rows = len(u_up_pairs)
    n_cols = 3 if baseline_rff is not None else 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows), squeeze=False)
    cmap = "viridis"
    col_rff, col_model, col_data = (0, 1, 2) if baseline_rff is not None else (None, 0, 1)

    for row, (u_vec, up_vec) in enumerate(u_up_pairs):
        n_pts = x_cond.shape[0]
        u_batch = u_vec.unsqueeze(0).expand(n_pts, -1).to(analysis_device)
        up_batch = up_vec.unsqueeze(0).expand(n_pts, -1).to(analysis_device)

        with torch.no_grad():
            log_pdf_model = tran_model.log_density(
                xp_flat, conditioner=x_cond, u=u_batch, up=up_batch
            )
            pdf_model = log_pdf_model.exp().reshape(nx, nxp).cpu().numpy()
            if baseline_rff is not None:
                log_pdf_rff = baseline_rff.log_density(xp_flat, conditioner=x_cond)
                pdf_rff = log_pdf_rff.exp().reshape(nx, nxp).cpu().numpy()

        x_np = x_1d.cpu().numpy()
        xp_np = xp_1d.cpu().numpy()
        u_s, up_s = float(u_vec[0]), float(up_vec[0])

        if baseline_rff is not None:
            vmax = float(max(pdf_rff.max(), pdf_model.max()))
            vmin = 0.0
            ax_rff = axes[row, col_rff]
            cf_rff = ax_rff.contourf(
                x_np, xp_np, pdf_rff.T, levels=40, cmap=cmap, vmin=vmin, vmax=vmax
            )
            fig.colorbar(cf_rff, ax=ax_rff, fraction=0.046, pad=0.04)
            ax_rff.set_xlabel("x")
            ax_rff.set_ylabel("x'")
            ax_rff.set_title("EM LinearRFF p(x'|x)\n(indep. of u, u')")
            ax_rff.set_xlim(lo, hi)
            ax_rff.set_ylim(lo, hi)

        ax_model = axes[row, col_model]
        contour_kw = {"levels": 40, "cmap": cmap}
        if baseline_rff is not None:
            contour_kw["vmin"] = vmin
            contour_kw["vmax"] = vmax
        cf = ax_model.contourf(x_np, xp_np, pdf_model.T, **contour_kw)
        fig.colorbar(cf, ax=ax_model, fraction=0.046, pad=0.04)
        ax_model.set_xlabel("x")
        ax_model.set_ylabel("x'")
        ax_model.set_title(f"MLP RFF p(x'|x, u={u_s:.2f}, u'={up_s:.2f})")
        ax_model.set_xlim(lo, hi)
        ax_model.set_ylim(lo, hi)

        u_diff = (u_k_cpu - u_vec).abs()
        mask = (u_diff.max(dim=-1).values if u_diff.dim() > 1 else u_diff.squeeze(-1)) < u_tol
        x_slice = x_k_cpu[mask, 0].numpy()
        xp_slice = x_kp1_cpu[mask, 0].numpy()

        ax_data = axes[row, col_data]
        if mask.any():
            ax_data.scatter(
                x_slice,
                xp_slice,
                s=6,
                alpha=0.35,
                c="C1",
                edgecolors="none",
                rasterized=True,
            )
        ax_data.set_xlim(lo, hi)
        ax_data.set_ylim(lo, hi)
        ax_data.set_aspect("auto")
        ax_data.set_xlabel("x")
        ax_data.set_ylabel("x'")
        ax_data.set_title(f"data |u - {u_s:.2f}| < {u_tol:.2f}, n={int(mask.sum())}")

    title = "EM RFF vs MLP RFF vs data (1D)" if baseline_rff is not None else "Transition model vs data (1D)"
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved transition comparison figure to {out_path}")


if __name__ == "__main__":
    problem = OPEN_LOOP_OBSERVABLE_PROBLEMS["scalar_nonlinear_drift"]

    use_gpu = torch.cuda.is_available()
    n_basis = 200
    control_dim = problem.system.control_dim()

    pretrain_params = {
        "epochs": 30,
        "lr": 1e-3,
    }
    tran_params = {
        "epochs": 100,
        "lr": 5e-5,
        "clip_grad_norm": 1.0,
        "weight_decay": 0.1,
    }
    init_params = {
        "epochs": 100,
        "lr": 5e-5,
        "clip_grad_norm": 1.0,
        "weight_decay": 0.1,
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
    # trainable=True: softplus at forward; pretrain MSE uses decode_params (decoded std space).
    g_stds = PositiveParameters.set_init(
        basis_shape, value=1.0, trainable=True, normalized=False, epsilon=1e-3
    ).to(device)
    g_coeffs = PositiveParameters.random_init((n_basis,), trainable=True, mean=1.0, std=0.1, normalized=False, epsilon=1e-2).to(device)
    g = GaussianBasis(mean_params=g_means, std_params=g_stds, coeffs=g_coeffs).to(device)
    g_mlp_form = MLPMetaForm(
        target_module_dim=system.dim(),
        context_dims={"u": control_dim},
        target_module=g,
        hidden_features=32,
        num_hidden_layers=3,
    ).to(device)

    psi_means = Parameters.random_init(basis_shape, trainable=True, mean=0.0, std=1.0).to(device)
    psi_stds = PositiveParameters.set_init(
        basis_shape, value=1.0, trainable=True, normalized=False, epsilon=1e-3
    ).to(device)
    psi = GaussianBasis(mean_params=psi_means, std_params=psi_stds).to(device)
    psi_mlp_form = MLPMetaForm(
        target_module_dim=system.dim(),
        context_dims={"u": control_dim, "up": control_dim},
        target_module=psi,
        hidden_features=32,
        num_hidden_layers=3,
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
    #print("Pre g means: ", pre_g_means)
    #print("Pre g stds: ", pre_g_stds)
    #print("Pre g weights: ", pre_g_weights)
    #print("Pre psi means: ", pre_psi_means)
    #print("Pre psi stds: ", pre_psi_stds)

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

    pre_g_basis_copy = copy.deepcopy(pre_g_basis)
    pre_psi_basis_copy = copy.deepcopy(pre_psi_basis)
    em_rff = LinearRFF(pre_g_basis_copy, pre_psi_basis_copy).to(device)


    g_target_params = MLPMetaForm.target_param_dict_from_basis(pre_g_basis, layout=g)
    psi_target_params = MLPMetaForm.target_param_dict_from_basis(pre_psi_basis, layout=psi)

    u_g_train_dataloader = DataLoader(
        train.MixedAlignedRandomDataset((u_k_train,), randomized_indices=()),
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=use_gpu,
    )
    u_psi_train_dataloader = DataLoader(
        train.MixedAlignedRandomDataset((u_k_train, u_k_train_copy), randomized_indices=(1,)),
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
    tran_model = MLPContextLinearRFF(g_mlp_form=g_mlp_form, psi_mlp_form=psi_mlp_form, numerical_tolerance=1e-10).to(device)

    #torch.autograd.set_detect_anomaly(True)

    print("Training transition model")
    tran_optimizer = torch.optim.Adam(tran_model.parameters(), lr=tran_params["lr"], weight_decay=tran_params["weight_decay"])
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
        clip_grad_norm=tran_params["clip_grad_norm"],
    )
    print("Done! \n")

    state_bounds = (
        tuple(problem.plot_bounds_low.tolist()),
        tuple(problem.plot_bounds_high.tolist()),
    )
    print("\n=== Transition model: conditional PDF validity (MC) ===")
    u_up_check_pairs = [
        (u_k_train[0].cpu(), u_k_train[0].cpu()),
        (u_k_train[len(u_k_train) // 2].cpu(), u_k_train[len(u_k_train) // 2].cpu()),
        (u_0_train[0].cpu(), u_k_val[0].cpu()),
    ]
    check_rng = torch.Generator().manual_seed(0)
    for _ in range(3):
        i, j = torch.randint(0, len(u_k_train), (2,), generator=check_rng).tolist()
        u_up_check_pairs.append((u_k_train[i].cpu(), u_k_train[j].cpu()))

    #tran_pdf_ok = check_mlp_context_transition_pdf_valid(
    #    tran_model,
    #    domain_bounds=state_bounds,
    #    conditioner_domain_bounds=state_bounds,
    #    u_up_pairs=u_up_check_pairs,
    #    n_samples=50_000,
    #    n_conditioner_samples=8,
    #    device=torch.device("cpu"),
    #    atol=0.25,
    #)
    #print(f"\nTransition conditional PDF check: {'PASSED' if tran_pdf_ok else 'FAILED'}\n")

    ## DEBUG
    #post_g_means, post_g_stds = g_mlp_form.instantiate(u=u_k_train[0].to(device)).means_stds()
    #print("Post g means: ", post_g_means)
    #print("Post g stds: ", post_g_stds)
    #post_psi_means, post_psi_stds = psi_mlp_form.instantiate(u=u_k_train[0].to(device), up=u_k_train[0].to(device)).means_stds()
    #print("Post psi means: ", post_psi_means)
    #print("Post psi stds: ", post_psi_stds)

    g_mlp_form.set_requires_grad(False)

    h0_means = Parameters.random_init(basis_shape, trainable=True, mean=0.0, std=1.0).to(device)
    h0_stds = PositiveParameters.set_init(
        basis_shape, value=1.0, trainable=True, normalized=False, epsilon=1e-3
    ).to(device)
    h0_coeffs = PositiveParameters.random_init((n_basis,), trainable=True, mean=1.0, std=0.1, normalized=True).to(device)
    h0 = GaussianBasis(mean_params=h0_means, std_params=h0_stds, coeffs=h0_coeffs).to(device)
    h0_mlp_form = MLPMetaForm(
        target_module_dim=system.dim(),
        context_dims={"up": control_dim},
        target_module=h0,
        hidden_features=32,
        num_hidden_layers=3,
    ).to(device)
    init_model = MLPContextLinearFF(g_mlp_form=g_mlp_form, h_mlp_form=h0_mlp_form, numerical_tolerance=1e-10).to(device)

    print("Training initial model")
    init_optimizer = torch.optim.Adam(init_model.h_mlp_form.parameters(), lr=init_params["lr"], weight_decay=init_params["weight_decay"])
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
        clip_grad_norm=init_params["clip_grad_norm"],
    )
    print("Done! \n")

    print(f"Transition model loss: {best_loss_tran:.4f}, training time: {training_time_tran:.2f} seconds")
    print(f"Initial model loss: {best_loss_init:.4f}, training time: {training_time_init:.2f} seconds")

    domain_bounds = state_bounds

    def plot_controller_comparison(
        controller_idx,
        controller,
        true_traj,
        init_belief,
        transition_model,
        out_path,
        n_steps,
        n_grid=400,
    ):
        analysis_device = torch.device("cpu")
        init_belief = init_belief.to(analysis_device).eval()
        transition_model = transition_model.to(analysis_device).eval()

        controls = [controller(k).to(analysis_device) for k in range(n_steps + 1)]
        belief_seq = propagate.propagate_with_control(init_belief, transition_model, controls)

        n_cols = min(len(true_traj), len(belief_seq))
        x_lo = problem.plot_bounds_low[0].item()
        x_hi = problem.plot_bounds_high[0].item()
        x_grid = torch.linspace(x_lo, x_hi, n_grid, device=analysis_device).unsqueeze(1)

        fig, axes = plt.subplots(n_cols, 1, figsize=(9, 2.4 * n_cols), squeeze=False)
        fig.suptitle(f"Controller {controller_idx}: true trajectories vs LRFF belief propagation")

        for t in range(n_cols):
            ax = axes[t, 0]
            samples = true_traj[t][:, 0].detach().cpu().numpy()
            belief_t = belief_seq[t].to(analysis_device).eval()
            print(f"Controller {controller_idx}, t={t}:")
            check_pdf_valid(belief_t, domain_bounds=domain_bounds, n_samples=10000, device=analysis_device)
            with torch.no_grad():
                pdf = belief_t(x_grid).squeeze(-1).detach().cpu().numpy()
            x_np = x_grid.squeeze(-1).detach().cpu().numpy()
            ax.plot(x_np, pdf, color="C0", lw=2.0, label="propagated belief")
            ax.hist(samples, bins=50, density=True, alpha=0.35, color="C1", label="test traj.")
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylabel("density")
            ax.set_title(f"t={t}")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right", fontsize=8)

        axes[-1, 0].set_xlabel(problem.system.state_label(0))
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved comparison figure to {out_path}")

    analysis_device = torch.device("cpu")
    init_model = init_model.to(analysis_device).eval()
    tran_model = tran_model.to(analysis_device).eval()

    n_plot_steps = min(problem.n_timesteps, n_timesteps_prop)
    out_dir = Path("figures") / "random_control_test_1D"

    plot_transition_u_up_comparison(
        tran_model,
        x_k_data,
        x_kp1_data,
        u_k_data,
        problem,
        out_dir / "transition_model_vs_data.png",
        baseline_rff=em_rff,
        n_pairs=10,
        n_grid=120,
        u_tol=0.3,
        random_seed=42,
    )

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
