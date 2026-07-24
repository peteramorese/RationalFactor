import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
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


def _plot_domain_grid_morph(
    first_tf: torch.nn.Module,
    second_tf: torch.nn.Module | None,
    x_data: torch.Tensor,
    out_path: Path,
    *,
    n_grid_lines: int = 11,
    n_points_per_line: int = 250,
    quantile_pad: float = 0.02,
    title: str = "",
) -> None:
    if x_data.shape[1] != 2:
        raise ValueError("_plot_domain_grid_morph expects 2D state data.")

    x_cpu = x_data.detach().cpu()
    q_lo = x_cpu.quantile(quantile_pad, dim=0)
    q_hi = x_cpu.quantile(1.0 - quantile_pad, dim=0)
    lo = q_lo.numpy()
    hi = q_hi.numpy()

    x_vals = np.linspace(lo[0], hi[0], n_points_per_line)
    y_vals = np.linspace(lo[1], hi[1], n_points_per_line)
    gx = np.linspace(lo[0], hi[0], n_grid_lines)
    gy = np.linspace(lo[1], hi[1], n_grid_lines)

    fig, axes = plt.subplots(3, 2, figsize=(12, 15), squeeze=False)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_grid_lines))

    with torch.no_grad():
        first_tf.eval()
        if second_tf is not None:
            second_tf.eval()
        y_data, _ = first_tf(x_cpu.to(dtype=torch.float32))
        if second_tf is None:
            z_data = y_data
        else:
            z_data, _ = second_tf(y_data)
        y_cpu = y_data.detach().cpu()
        z_cpu = z_data.detach().cpu()

    y_lo = y_cpu.quantile(quantile_pad, dim=0).numpy()
    y_hi = y_cpu.quantile(1.0 - quantile_pad, dim=0).numpy()
    yx_vals = np.linspace(y_lo[0], y_hi[0], n_points_per_line)
    yy_vals = np.linspace(y_lo[1], y_hi[1], n_points_per_line)
    ygx = np.linspace(y_lo[0], y_hi[0], n_grid_lines)
    ygy = np.linspace(y_lo[1], y_hi[1], n_grid_lines)

    ax_r1_l, ax_r1_r = axes[0, 0], axes[0, 1]
    ax_r2_l, ax_r2_r = axes[1, 0], axes[1, 1]
    ax_r3_l, ax_r3_r = axes[2, 0], axes[2, 1]

    ax_r1_l.set_title("Original state space: rectangular grid")
    ax_r1_r.set_title("First transform effect")
    ax_r2_l.set_title("First-transform space: new rectangular grid")
    ax_r2_r.set_title("Second transform effect (from new grid)")
    ax_r3_l.set_title("Original state space: rectangular grid")
    ax_r3_r.set_title("Full map: first then second transform")

    for ax, pts in (
        (ax_r1_l, x_cpu),
        (ax_r1_r, y_cpu),
        (ax_r2_l, y_cpu),
        (ax_r2_r, z_cpu),
        (ax_r3_l, x_cpu),
        (ax_r3_r, z_cpu),
    ):
        ax.scatter(pts[:, 0].numpy(), pts[:, 1].numpy(), s=1, alpha=0.08, c="gray", rasterized=True)

    with torch.no_grad():
        for i, xv in enumerate(gx):
            line_xy = np.stack([np.full_like(y_vals, xv), y_vals], axis=1)
            line_t = torch.from_numpy(line_xy).to(dtype=torch.float32)
            y_line, _ = first_tf(line_t)
            if second_tf is None:
                z_line = y_line
            else:
                z_line, _ = second_tf(y_line)
            y_np = y_line.detach().cpu().numpy()
            z_np = z_line.detach().cpu().numpy()

            ax_r1_l.plot(line_xy[:, 0], line_xy[:, 1], color=colors[i], lw=1.4)
            ax_r1_r.plot(y_np[:, 0], y_np[:, 1], color=colors[i], lw=1.4)
            ax_r3_l.plot(line_xy[:, 0], line_xy[:, 1], color=colors[i], lw=1.4)
            ax_r3_r.plot(z_np[:, 0], z_np[:, 1], color=colors[i], lw=1.4)

        for i, yv in enumerate(gy):
            line_xy = np.stack([x_vals, np.full_like(x_vals, yv)], axis=1)
            line_t = torch.from_numpy(line_xy).to(dtype=torch.float32)
            y_line, _ = first_tf(line_t)
            if second_tf is None:
                z_line = y_line
            else:
                z_line, _ = second_tf(y_line)
            y_np = y_line.detach().cpu().numpy()
            z_np = z_line.detach().cpu().numpy()

            ax_r1_l.plot(line_xy[:, 0], line_xy[:, 1], color=colors[i], lw=1.4, alpha=0.8)
            ax_r1_r.plot(y_np[:, 0], y_np[:, 1], color=colors[i], lw=1.4, alpha=0.8)
            ax_r3_l.plot(line_xy[:, 0], line_xy[:, 1], color=colors[i], lw=1.4, alpha=0.8)
            ax_r3_r.plot(z_np[:, 0], z_np[:, 1], color=colors[i], lw=1.4, alpha=0.8)

        for i, yx in enumerate(ygx):
            line_y = np.stack([np.full_like(yy_vals, yx), yy_vals], axis=1)
            line_t = torch.from_numpy(line_y).to(dtype=torch.float32)
            if second_tf is None:
                z_line = line_t
            else:
                z_line, _ = second_tf(line_t)
            z_np = z_line.detach().cpu().numpy()
            ax_r2_l.plot(line_y[:, 0], line_y[:, 1], color=colors[i], lw=1.4)
            ax_r2_r.plot(z_np[:, 0], z_np[:, 1], color=colors[i], lw=1.4)

        for i, yy in enumerate(ygy):
            line_y = np.stack([yx_vals, np.full_like(yx_vals, yy)], axis=1)
            line_t = torch.from_numpy(line_y).to(dtype=torch.float32)
            if second_tf is None:
                z_line = line_t
            else:
                z_line, _ = second_tf(line_t)
            z_np = z_line.detach().cpu().numpy()
            ax_r2_l.plot(line_y[:, 0], line_y[:, 1], color=colors[i], lw=1.4, alpha=0.8)
            ax_r2_r.plot(z_np[:, 0], z_np[:, 1], color=colors[i], lw=1.4, alpha=0.8)

    ax_r1_l.set_xlim(float(lo[0]), float(hi[0]))
    ax_r1_l.set_ylim(float(lo[1]), float(hi[1]))
    ax_r3_l.set_xlim(float(lo[0]), float(hi[0]))
    ax_r3_l.set_ylim(float(lo[1]), float(hi[1]))
    ax_r1_r.set_xlim(float(y_lo[0]), float(y_hi[0]))
    ax_r1_r.set_ylim(float(y_lo[1]), float(y_hi[1]))
    ax_r2_l.set_xlim(float(y_lo[0]), float(y_hi[0]))
    ax_r2_l.set_ylim(float(y_lo[1]), float(y_hi[1]))

    for ax in (ax_r1_l, ax_r1_r, ax_r2_l, ax_r2_r, ax_r3_l, ax_r3_r):
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)

    ax_r1_l.set_xlabel("x_1"); ax_r1_l.set_ylabel("x_2")
    ax_r1_r.set_xlabel("y_1"); ax_r1_r.set_ylabel("y_2")
    ax_r2_l.set_xlabel("y_1"); ax_r2_l.set_ylabel("y_2")
    ax_r2_r.set_xlabel("z_1"); ax_r2_r.set_ylabel("z_2")
    ax_r3_l.set_xlabel("x_1"); ax_r3_l.set_ylabel("x_2")
    ax_r3_r.set_xlabel("z_1"); ax_r3_r.set_ylabel("z_2")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    problem = FULLY_OBSERVABLE_PROBLEMS["van_der_pol"]
    
    ###
    use_gpu = torch.cuda.is_available()
    use_dtf = True
    n_basis = 500
    if use_dtf:
        tran_params = {
            "n_epochs_per_group": [15, 5], # dtf_params and basis, weights
            "iterations": 20,
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
            "iterations": 20,
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
    ###

    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using GPU: ", use_gpu)
    print("Device: ", device)

    system = problem.system
    x0_data = problem.train_initial_state_data()
    x_k_data, x_kp1_data = problem.train_state_transition_data()
    test_traj_data = problem.test_data()

    x0_dataloader = DataLoader(TensorDataset(x0_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(TensorDataset(x_kp1_data, x_k_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    # Create basis functions
    phi_basis =  UnnormalizedBetaBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([10.0, 10.0], device=device), variance=30.0, min_concentration=1.0).to(device)
    psi_basis =  UnnormalizedBetaBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([10.0, 10.0], device=device), variance=30.0, min_concentration=1.0).to(device)
    psi0_basis = UnnormalizedBetaBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([10.0, 10.0], device=device), variance=30.0, min_concentration=1.0).to(device)

    # Create separable domain transformation
    wrap_tf = ErfSeparableTF.from_data(x_k_data, trainable=True)
    
    nftf = MaskedAffineNFTF(system.dim(), trainable=True, hidden_features=256, n_layers=8).to(device) if use_dtf else None

    # Create and train the transition model
    if use_dtf:
        tran_model = CompositeConditionalModel([nftf, wrap_tf], LinearRFF(phi_basis, psi_basis, numerical_tolerance=problem.numerical_tolerance)).to(device)
    else:
        tran_model = CompositeConditionalModel([wrap_tf], LinearRFF(phi_basis, psi_basis, numerical_tolerance=problem.numerical_tolerance)).to(device)

    print("Training transition model")
    mle_loss_fn = loss.conditional_mle_loss
    
    if use_dtf:
        optimizers ={"dtf_and_basis": torch.optim.Adam([{'params': tran_model.conditional_density_model.basis_params(), 'lr': tran_params["lr_basis"]}, 
                {'params':tran_model.domain_tfs[0].parameters(), 'lr': tran_params["lr_dtf"]}, 
                {'params':tran_model.domain_tfs[1].parameters(), 'lr': tran_params["lr_wrap"]}]), 
            "weights": torch.optim.Adam(tran_model.conditional_density_model.weight_params(), lr=tran_params["lr_weights"])} 
    else:
        optimizers ={"basis": torch.optim.Adam([{'params': tran_model.conditional_density_model.basis_params(), 'lr': tran_params["lr_basis"]}, {'params': tran_model.domain_tfs.parameters(), 'lr': tran_params["lr_wrap"]}]),
            "weights": torch.optim.Adam(tran_model.conditional_density_model.weight_params(), lr=tran_params["lr_weights"])}

    tran_model, best_loss_tran, training_time_tran = train.train_iterate(tran_model,
        xp_dataloader,
        {"mle": mle_loss_fn}, 
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
        optimizers = {"basis": torch.optim.Adam(init_model.density_model.basis_params(), lr=init_params["lr_basis"]), "weights": torch.optim.Adam(init_model.density_model.weight_params(), lr=init_params["lr_weights"])}
    else:
        optimizers = {"basis": torch.optim.Adam(init_model.density_model.basis_params(), lr=init_params["lr_basis"]), "weights": torch.optim.Adam(init_model.density_model.weight_params(), lr=init_params["lr_weights"])}

    init_model, best_loss_init, training_time_init = train.train_iterate(init_model, 
        x0_dataloader, 
        {"mle": mle_loss_fn}, 
        optimizers,
        epochs_per_group=init_params["n_epochs_per_group"],
        iterations=init_params["iterations"],
        verbose=True,
        use_best="mle")
    print("Done! \n")

    print(f"Transition model loss: {best_loss_tran:.4f}, training time: {training_time_tran:.2f} seconds")
    print(f"Initial model loss: {best_loss_init:.4f}, training time: {training_time_init:.2f} seconds")

    # Analysis
    analysis_device = torch.device("cpu")
    init_model = init_model.to(analysis_device).eval()
    tran_model = tran_model.to(analysis_device).eval()
    trained_domain_tf = trained_domain_tf.to(analysis_device).eval()
    if use_dtf:
        trained_nftf = trained_nftf.to(analysis_device).eval()

    box_lows = tuple(problem.plot_bounds_low.tolist())
    box_highs = tuple(problem.plot_bounds_high.tolist())

    if use_dtf:
        base_belief_seq = propagate.propagate(init_model.density_model, tran_model.conditional_density_model, n_steps=problem.n_timesteps)
        belief_seq = [CompositeDensityModel([trained_nftf, trained_domain_tf], belief).to(analysis_device).eval() for belief in base_belief_seq]
    else:
        base_belief_seq = propagate.propagate(init_model.density_model, tran_model.conditional_density_model, n_steps=problem.n_timesteps)
        belief_seq = [CompositeDensityModel([trained_domain_tf], belief).to(analysis_device).eval() for belief in base_belief_seq]

    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_morph_out_path = out_dir / "vdp_nftf_group_beta__domain_grid_morph.png"
    _plot_domain_grid_morph(
        trained_domain_tf,
        (trained_nftf if use_dtf else None),
        x_k_data.detach().cpu(),
        grid_morph_out_path,
        n_grid_lines=11,
        n_points_per_line=280,
        quantile_pad=0.01,
        title="van_der_pol nftf_group_beta: staged state/latent grid morph",
    )
    print(f"Saved domain transformation grid morph to {grid_morph_out_path}")

    fig, axes = plt.subplots(2, problem.n_timesteps, figsize=(20, 10))
    fig.suptitle("Beliefs at each time step")
    for i in range(problem.n_timesteps):
        data_i = test_traj_data[i].to(analysis_device)
        ll = avg_log_likelihood(belief_seq[i], data_i)
        print(f"Log likelihood at time {i}: {ll:.4f}")
        check_pdf_valid(belief_seq[i], domain_bounds=(box_lows, box_highs), n_samples=100000, device=analysis_device)
        plot_belief(axes[1, i], belief_seq[i], x_range=(box_lows[0], box_highs[0]), y_range=(box_lows[1], box_highs[1]))
        axes[0, i].scatter(test_traj_data[i][:, 0], test_traj_data[i][:, 1], s=1)
        axes[0, i].set_aspect("equal")
        axes[0, i].set_xlim(box_lows[0], box_highs[0])
        axes[0, i].set_ylim(box_lows[1], box_highs[1])
    

    plt.savefig("figures/vdp_nfdf_beta_beliefs.pdf", dpi=1000)
    #plt.show()


