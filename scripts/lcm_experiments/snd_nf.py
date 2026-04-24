from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.tools.propagate as propagate
from rational_factor.models.basis_functions import GaussianKernelBasis
from rational_factor.models.composite_model import CompositeConditionalModel, CompositeDensityModel
from rational_factor.models.density_model import DensityModel
from rational_factor.models.domain_transformation import ErfSeparableTF, MaskedRQSNFTF, IdentityTF
from rational_factor.models.factor_forms import LinearFF, LinearRFF
from rational_factor.models.density_model import LogisticSigmoid
from normalizing_flow.normalizing_flow import NormalizingFlow
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood, check_pdf_valid


def _plot_snd_1d_marginal_trajectory_comparison(
    problem,
    beliefs: list[CompositeDensityModel],
    test_traj_data: list[torch.Tensor],
    out_path: Path,
    *,
    n_grid: int = 400,
    title: str = "",
) -> None:
    """1D-only: learned marginal pdf vs normalized histogram of test trajectories per timestep."""
    n_slices = problem.n_timesteps + 1
    assert len(beliefs) == n_slices
    lo = float(problem.plot_bounds_low[0].item())
    hi = float(problem.plot_bounds_high[0].item())
    x_grid = torch.linspace(lo, hi, n_grid).unsqueeze(1)

    param = next(iter(beliefs[0].parameters()), None)
    buffer = next(iter(beliefs[0].buffers()), None)
    dev = param.device if param is not None else (buffer.device if buffer is not None else torch.device("cpu"))
    dt = param.dtype if param is not None else (buffer.dtype if buffer is not None else torch.float32)
    x_grid = x_grid.to(device=dev, dtype=dt)

    fig, axes = plt.subplots(n_slices, 1, figsize=(8, 2.2 * n_slices), squeeze=False)
    for t in range(n_slices):
        ax = axes[t, 0]
        samples = test_traj_data[t][:, 0].detach().cpu().numpy()
        with torch.no_grad():
            beliefs[t].eval()
            pdf = beliefs[t](x_grid).squeeze(-1).detach().cpu().numpy()
        x_np = x_grid.squeeze(-1).detach().cpu().numpy()
        ax.plot(x_np, pdf, color="C0", lw=2.0, label="learned marginal")
        ax.hist(samples, bins=50, density=True, alpha=0.35, color="C1", label="test traj. (hist)")
        ax.set_xlim(lo, hi)
        ax.set_ylabel("density")
        ax.set_title(f"t = {t}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel(problem.system.state_label(0))
    if title:
        fig.suptitle(title, y=1.002)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_snd_conditional_true_vs_learned(
    tran_model: CompositeConditionalModel,
    system,
    out_path: Path,
    x_cond_train: torch.Tensor,
    xp_train: torch.Tensor,
    x_cond_train_latent: torch.Tensor | None = None,
    base_density_model: DensityModel | None = None,
    *,
    problem,
    n_grid: int = 200,
    title: str = "",
) -> None:
    """
    2D heatmaps of p(x'|x): horizontal axis = conditioner x, vertical axis = next state x'.
    Left: true (Gaussian around drift); right: learned LinearRFF (+ domain transforms).
    """
    lo = float(problem.plot_bounds_low[0].item())
    hi = float(problem.plot_bounds_high[0].item())

    param = next(iter(tran_model.parameters()), None)
    buffer = next(iter(tran_model.buffers()), None)
    dev = param.device if param is not None else (buffer.device if buffer is not None else torch.device("cpu"))
    dt = param.dtype if param is not None else (buffer.dtype if buffer is not None else torch.float32)

    nx = n_grid
    nxp = n_grid
    x_1d = torch.linspace(lo, hi, nx, device=dev, dtype=dt)
    xp_1d = torch.linspace(lo, hi, nxp, device=dev, dtype=dt)
    x_cond = x_1d.unsqueeze(1).expand(nx, nxp).reshape(-1, 1)
    xp_flat = xp_1d.unsqueeze(0).expand(nx, nxp).reshape(-1, 1)

    with torch.no_grad():
        tran_model.eval()
        log_learned = tran_model.log_density(xp_flat, conditioner=x_cond)
        learned = log_learned.exp().detach().cpu().numpy().reshape(nx, nxp)

    x_cond_np = x_cond_train[:, 0].detach().cpu().numpy()
    xp_train_np = xp_train[:, 0].detach().cpu().numpy()
    log_true = system.log_transition_density(
        x_cond.detach().cpu().float(),
        xp_flat.detach().cpu().float(),
    )
    true_pdf = log_true.exp().detach().numpy().reshape(nx, nxp)

    # Shared color scale for direct comparison
    vmax = float(max(true_pdf.max(), learned.max()))
    vmin = 0.0
    if vmax <= 0:
        vmax = 1.0

    x_np = x_1d.detach().cpu().numpy()
    xp_np = xp_1d.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 5, figsize=(30, 5), squeeze=False)
    cmap = "viridis"
    for ax, Z, subt in zip(
        axes[0, :2],
        [true_pdf.T, learned.T],
        ["true p(x'|x)", "learned p(x'|x)"],
    ):
        # contourf expects Z with shape (len(yp), len(xp)) for values at (x, y) = (columns, rows)
        cf = ax.contourf(x_np, xp_np, Z, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_aspect("auto")
        ax.set_xlabel("x (conditioner)")
        ax.set_ylabel("x'")
        ax.set_title(subt)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

    # Third panel: scatter of training transition samples.
    ax_scatter = axes[0, 2]
    ax_scatter.scatter(
        x_cond_np,
        xp_train_np,
        s=4,
        alpha=0.25,
        c="C1",
        edgecolors="none",
        rasterized=True,
    )
    ax_scatter.set_xlim(lo, hi)
    ax_scatter.set_ylim(lo, hi)
    ax_scatter.set_aspect("auto")
    ax_scatter.set_xlabel("x (conditioner)")
    ax_scatter.set_ylabel("x'")
    ax_scatter.set_title("training samples scatter")
    ax_scatter.grid(True, alpha=0.2)

    # Fourth panel: KDE estimate of joint p(x, x') from training samples.
    ax_kde = axes[0, 3]
    joint_train = torch.cat([x_cond_train, xp_train], dim=1).to(device=dev, dtype=dt)
    joint_grid = torch.stack([x_cond, xp_flat], dim=1).reshape(-1, 2).to(device=dev, dtype=dt)
    n_joint = joint_train.shape[0]
    d_joint = joint_train.shape[1]
    eps = torch.finfo(dt).eps
    if n_joint > 1:
        sample_std = joint_train.std(dim=0, unbiased=True)
        scale_joint = sample_std.mean().clamp_min(eps)
    else:
        scale_joint = torch.tensor(1.0, device=dev, dtype=dt)
    silverman = (4.0 / (d_joint + 2.0)) ** (1.0 / (d_joint + 4.0))
    h_joint = (silverman * (n_joint ** (-1.0 / (d_joint + 4.0))) * scale_joint).clamp_min(eps)
    h_joint = 0.2 * h_joint
    norm_const = torch.pow(joint_grid.new_tensor(2.0 * torch.pi), -0.5 * d_joint) * h_joint.pow(-d_joint)

    block = 4096
    kde_vals = []
    for start in range(0, joint_grid.shape[0], block):
        end = min(start + block, joint_grid.shape[0])
        grid_blk = joint_grid[start:end]  # (b, 2)
        diff = grid_blk[:, None, :] - joint_train[None, :, :]  # (b, n_joint, 2)
        sq_norm = diff.square().sum(dim=2)  # (b, n_joint)
        k_blk = norm_const * torch.exp(-0.5 * sq_norm / (h_joint * h_joint))
        kde_vals.append(k_blk.mean(dim=1))
    joint_kde = torch.cat(kde_vals, dim=0).reshape(nx, nxp).detach().cpu().numpy()

    cf_kde = ax_kde.contourf(x_np, xp_np, joint_kde.T, levels=50, cmap=cmap)
    ax_kde.set_aspect("auto")
    ax_kde.set_xlabel("x (conditioner)")
    ax_kde.set_ylabel("x'")
    ax_kde.set_title("training joint KDE")
    fig.colorbar(cf_kde, ax=ax_kde, fraction=0.046, pad=0.04)

    # Fifth panel: LRFF quantities in latent conditioner coordinates z.
    ax_g = axes[0, 4]
    with torch.no_grad():
        lrff = tran_model.conditional_density_model
        if hasattr(lrff, "get_a") and hasattr(lrff, "phi_basis"):
            if x_cond_train_latent is not None:
                z_train_np = x_cond_train_latent[:, 0].detach().cpu().numpy()
            else:
                z_train = x_cond_train.to(device=dev, dtype=dt)
                for tf in tran_model.domain_tfs:
                    z_train, _ = tf(z_train)
                z_train_np = z_train[:, 0].detach().cpu().numpy()

            # Set latent axis bounds from latent training samples.
            z_lo = float(np.min(z_train_np))
            z_hi = float(np.max(z_train_np))
            if z_hi <= z_lo:
                z_hi = z_lo + 1e-6

            hist_counts, hist_edges = np.histogram(z_train_np, bins=n_grid // 2, range=(z_lo, z_hi), density=True)
            hist_centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
            hist_width = hist_edges[1] - hist_edges[0]

            # Evaluate g(z) and base density on a latent-space grid
            z_plot = torch.linspace(z_lo, z_hi, nx, device=dev, dtype=dt).unsqueeze(1)
            z_axis = z_plot[:, 0].detach().cpu().numpy()
            g_curve = (lrff.phi_basis(z_plot) @ lrff.get_a()).detach().cpu().numpy()

            # Use normalized histogram density directly (area under bars is 1).
            if hist_counts.size > 0:
                ax_g.bar(
                    hist_centers,
                    hist_counts,
                    width=hist_width,
                    color="C0",
                    alpha=0.25,
                    edgecolor="none",
                    label="hist(transformed x_k)",
                )
            if base_density_model is not None:
                base_log = base_density_model.log_density(z_plot)
                base_curve = base_log.exp().detach().cpu().numpy()

                sort_idx = np.argsort(z_axis)
                z_axis_sorted = z_axis[sort_idx]
                g_sorted = g_curve[sort_idx]
                base_sorted = base_curve[sort_idx]
                ax_g.plot(z_axis_sorted, base_sorted, color="C2", lw=1.8, ls="--", label="base density")
                ax_g.plot(z_axis_sorted, g_sorted, color="C3", lw=2.0, label="g(z)")
            else:
                sort_idx = np.argsort(z_axis)
                z_axis_sorted = z_axis[sort_idx]
                g_sorted = g_curve[sort_idx]
                ax_g.plot(z_axis_sorted, g_sorted, color="C3", lw=2.0, label="g(z)")

            ax_g.set_xlim(z_lo, z_hi)
            ax_g.set_ylabel("density / g(z)")
            ax_g.grid(True, alpha=0.25)
            ax_g.legend(loc="upper right", fontsize=8)
        else:
            ax_g.text(0.5, 0.5, "LRFF g(z)\nnot available", ha="center", va="center", transform=ax_g.transAxes)
            ax_g.set_ylabel("value")
    ax_g.set_xlabel("z (latent conditioner)")
    ax_g.set_title("LRFF g(z) in latent space")

    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    problem = FULLY_OBSERVABLE_PROBLEMS["scalar_nonlinear_drift"]

    ######## USER CONFIG ########
    use_gpu = torch.cuda.is_available()
    batch_size = 256

    dtf_params = {
        "n_epochs_per_group": [1],  
        "iterations": 50,
        "lr": 1e-3,
    }
    ls_temp = 0.01

    ######## SETUP ########
    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using device: ", device)

    system = problem.system
    x0_data = problem.train_initial_state_data()
    x_k_data, x_kp1_data = problem.train_state_transition_data()
    test_traj_data = problem.test_data()

    x_dataloader = DataLoader(TensorDataset(x_k_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(TensorDataset(x_kp1_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)


    all_train_data = torch.cat([x0_data, x_k_data, x_kp1_data], dim=0)
    data_min = all_train_data.min(dim=0).values
    data_max = all_train_data.max(dim=0).values
    loc = 0.5 * (data_min + data_max)
    scale = (data_max - data_min).clamp_min(torch.finfo(all_train_data.dtype).eps)
    loc = loc.to(device)
    scale = scale.to(device)
    base_distribution = LogisticSigmoid(system.dim(), temperature=ls_temp, loc=loc, scale=scale)
    print("Base distribution loc: ", base_distribution.loc)
    print("Base distribution scale: ", base_distribution.scale)
    dtf = MaskedRQSNFTF(system.dim(), trainable=True, hidden_features=256, n_layers=5).to(device)
    decorrupter = CompositeDensityModel([dtf], base_distribution).to(device)
    test_decorrupter = NormalizingFlow(system.dim(), num_layers=5, hidden_features=256).to(device)

    ######## TRAIN LCM ########

    print("Training NF decorrupter")

    optimizers = {
        "dec": torch.optim.Adam(decorrupter.parameters(), lr=dtf_params["lr"]),
        #"test_dec": torch.optim.Adam(test_decorrupter.parameters(), lr=dtf_params["lr"]),
    }

    dtf_concentration_loss = lambda composite_model, x: loss.dtf_data_concentration_loss(composite_model.domain_tfs[0], x, loc, scale)
    decorrupter, best_loss, training_time = train.train_iterate(
        decorrupter,
        #test_decorrupter,
        x_dataloader,
        {"mle": loss.mle_loss, "dtf_conc": dtf_concentration_loss},
        optimizers,
        epochs_per_group=dtf_params["n_epochs_per_group"],
        iterations=dtf_params["iterations"],
        verbose=True,
        use_best="mle",
    )
    print("Done.\n")

    x_k_data = x_k_data.to(device)
    x_kp1_data = x_kp1_data.to(device)
    x_k_transformed, _ = dtf(x_k_data)
    x_kp1_transformed, _ = dtf(x_kp1_data)
    phi_basis = GaussianKernelBasis(x_k_transformed, trainable=False)
    psi_basis = GaussianKernelBasis(x_kp1_transformed, trainable=False)
    phi_basis.kernel_bandwidth *= 0.2
    psi_basis.kernel_bandwidth *= 0.2

    lrff = LinearRFF(phi_basis, psi_basis, numerical_tolerance=problem.numerical_tolerance).to(device)
    tran_model = CompositeConditionalModel([dtf], lrff).to(device)
    


    #tran_model = CompositeConditionalModel([IdentityTF(system.dim())], lrff).to(device)

    ######### TRAIN INITIAL ########
    #if use_dtf:
    #    init_model = CompositeDensityModel(
    #        [trained_nftf, trained_domain_tf],
    #        LinearFF.from_rff(tran_model.conditional_density_model, psi0_basis),
    #    ).to(device)
    #else:
    #    init_model = CompositeDensityModel(
    #        [trained_domain_tf],
    #        LinearFF.from_rff(tran_model.conditional_density_model, psi0_basis),
    #    ).to(device)

    #print("Training initial model")
    #optimizers = {
    #    "basis": torch.optim.Adam(init_model.density_model.basis_params(), lr=init_params["lr_basis"]),
    #    "weights": torch.optim.Adam(init_model.density_model.weight_params(), lr=init_params["lr_weights"]),
    #}

    #init_model, best_loss_init, training_time_init = train.train_iterate(
    #    init_model,
    #    x0_dataloader,
    #    {"mle": loss.mle_loss},
    #    optimizers,
    #    epochs_per_group=init_params["n_epochs_per_group"],
    #    iterations=init_params["iterations"],
    #    verbose=True,
    #    use_best="mle",
    #)
    #print("Done.\n")

    #print(f"Transition model loss: {best_loss_tran:.6f}, training time: {training_time_tran:.2f}s")
    #print(f"Initial model loss: {best_loss_init:.6f}, training time: {training_time_init:.2f}s")


    ######### ANALYSIS ########
    #base_belief_seq = propagate.propagate(init_model.density_model, tran_model.conditional_density_model, n_steps=problem.n_timesteps)
    #if use_dtf:
    #    belief_seq = [CompositeDensityModel([trained_nftf, trained_domain_tf], belief) for belief in base_belief_seq]
    #else:
    #    belief_seq = [CompositeDensityModel([trained_domain_tf], belief) for belief in base_belief_seq]

    #ll_per_step = []
    #for i in range(problem.n_timesteps):
    #    data_i = test_traj_data[i].to(device)
    #    ll = avg_log_likelihood(belief_seq[i], data_i)
    #    check_pdf_valid(belief_seq[i], domain_bounds=(tuple(problem.plot_bounds_low.tolist()), tuple(problem.plot_bounds_high.tolist())), n_samples=10000, device=device)
    #    ll_per_step.append(ll.detach().cpu().reshape(()))
    #    print(f"Avg log-likelihood at time {i}: {float(ll_per_step[-1]):.6f}")
    #prop_ll_vector = torch.stack(ll_per_step).to(dtype=torch.float32)

    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    #ll_out_path = out_dir / f"{Path(__file__).stem}__ll.png"
    #plt.figure(figsize=(8, 4))
    #plt.plot(prop_ll_vector.numpy(), marker="o")
    #plt.xlabel("timestep")
    #plt.ylabel("avg log-likelihood")
    #plt.title(f"{Path(__file__).stem} (use_dtf={use_dtf}, n_basis={n_basis})")
    #plt.grid(True, alpha=0.25)
    #plt.tight_layout()
    #plt.savefig(ll_out_path, dpi=200)
    #print(f"Saved LL plot to {ll_out_path}")

    # 1D-only figures: marginal pdfs + true vs learned conditional densities (CPU for plotting).
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    #belief_seq_cpu = [belief.to("cpu").eval() for belief in belief_seq]
    tran_model_cpu = tran_model.cpu().eval()

    stem = Path(__file__).stem
    #marg_out_path = out_dir / f"{stem}__marginal_comparison.png"
    #_plot_snd_1d_marginal_trajectory_comparison(
    #    problem,
    #    belief_seq_cpu,
    #    test_traj_data,
    #    marg_out_path,
    #    n_grid=400,
    #    title=f"{stem} marginal (1D): learned vs test samples",
    #)
    #print(f"Saved 1D marginal comparison plot to {marg_out_path}")

    cond_out_path = out_dir / f"{stem}__conditional_comparison.png"
    _plot_snd_conditional_true_vs_learned(
        tran_model_cpu,
        system,
        cond_out_path,
        x_k_data,
        x_kp1_data,
        x_cond_train_latent=x_k_transformed,
        base_density_model=base_distribution,
        problem=problem,
        n_grid=200,
        title=f"{stem}: true vs learned conditional density (2D)",
    )
    print(f"Saved conditional density comparison plot to {cond_out_path}")


if __name__ == "__main__":
    main()
