from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.tools.propagate as propagate
from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.composite_model import CompositeConditionalModel, CompositeDensityModel
from rational_factor.models.density_model import LogisticSigmoid
from rational_factor.models.domain_transformation import MaskedRQSNFTF, StackedTF, VolumePreservingNFTF
from rational_factor.models.factor_forms import LinearFF, LinearRFF
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood
from rational_factor.tools.misc import data_bounds
from rational_factor.tools.visualization import plot_belief


def _plot_snd_conditional_true_vs_learned(
    tran_model: CompositeConditionalModel,
    system,
    out_path: Path,
    x_cond_train: torch.Tensor,
    xp_train: torch.Tensor,
    *,
    problem,
    n_grid: int = 200,
    title: str = "",
) -> None:
    """Compare true and learned p(x'|x) on a 2D grid (conditioner vs next state)."""
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

    log_true = system.log_transition_density(
        x_cond.detach().cpu().float(),
        xp_flat.detach().cpu().float(),
    )
    true_pdf = log_true.exp().detach().numpy().reshape(nx, nxp)

    x_cond_np = x_cond_train[:, 0].detach().cpu().numpy()
    xp_train_np = xp_train[:, 0].detach().cpu().numpy()
    x_np = x_1d.detach().cpu().numpy()
    xp_np = xp_1d.detach().cpu().numpy()

    vmax = float(max(true_pdf.max(), learned.max()))
    vmin = 0.0
    if vmax <= 0:
        vmax = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), squeeze=False)
    cmap = "viridis"
    for ax, Z, subt in zip(
        axes[0, :2],
        [true_pdf.T, learned.T],
        ["true p(x'|x)", "learned p(x'|x)"],
    ):
        cf = ax.contourf(x_np, xp_np, Z, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_aspect("auto")
        ax.set_xlabel("x (conditioner)")
        ax.set_ylabel("x'")
        ax.set_title(subt)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

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

    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_snd_1d_belief_trajectory(
    problem,
    beliefs: list[CompositeDensityModel],
    test_traj_data: list[torch.Tensor],
    out_path: Path,
    *,
    n_grid: int = 400,
    title: str = "",
) -> None:
    """Plot propagated 1D beliefs against test-trajectory histograms for each timestep."""
    n_slices = problem.n_timesteps + 1
    if len(beliefs) != n_slices:
        raise ValueError(f"Expected {n_slices} beliefs, got {len(beliefs)}")

    lo = float(problem.plot_bounds_low[0].item())
    hi = float(problem.plot_bounds_high[0].item())
    x_grid = torch.linspace(lo, hi, n_grid).unsqueeze(1)

    param = next(iter(beliefs[0].parameters()), None)
    buffer = next(iter(beliefs[0].buffers()), None)
    dev = param.device if param is not None else (buffer.device if buffer is not None else torch.device("cpu"))
    dt = param.dtype if param is not None else (buffer.dtype if buffer is not None else torch.float32)
    x_grid = x_grid.to(device=dev, dtype=dt)

    fig, axes = plt.subplots(n_slices, 1, figsize=(9, 2.4 * n_slices), squeeze=False)
    for t in range(n_slices):
        ax = axes[t, 0]
        samples = test_traj_data[t][:, 0].detach().cpu().numpy()
        with torch.no_grad():
            beliefs[t].eval()
            pdf = beliefs[t](x_grid).squeeze(-1).detach().cpu().numpy()
        x_np = x_grid.squeeze(-1).detach().cpu().numpy()
        ax.plot(x_np, pdf, color="C0", lw=2.0, label="propagated belief")
        ax.hist(samples, bins=50, density=True, alpha=0.35, color="C1", label="test traj. (hist)")
        ax.set_xlim(lo, hi)
        ax.set_ylabel("density")
        ax.set_title(f"t = {t}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1, 0].set_xlabel(problem.system.state_label(0))
    if title:
        fig.suptitle(title, y=1.002)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    problem = FULLY_OBSERVABLE_PROBLEMS["scalar_nonlinear_drift"]

    ######## USER CONFIG ########
    use_gpu = torch.cuda.is_available()
    batch_size = 1024

    decorrupter_params = {
        "epochs": 50,
        "lr": 1e-3,
    }

    mover_params = {
        "n_epochs_per_group": [1],
        "iterations": 50,
        "lr_basis": 5e-2,
        "lr_weights": 1e-1,
        "lr_dtf": 1e-3,
    }

    init_params = {
        "n_epochs_per_group": [15, 5],
        "iterations": 100,
        "lr_basis": 5e-2,
        "lr_weights": 1e-1,
    }

    ls_temp = 0.05
    n_basis = 50
    mover_iterations = 3

    ######## SETUP ########
    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using device: ", device)

    system = problem.system
    x0_data = problem.train_initial_state_data()
    x_k_data, x_kp1_data = problem.train_state_transition_data()
    test_traj_data = problem.test_data()

    x_dataloader = DataLoader(TensorDataset(x_k_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    ######## TRAIN DECORRUPTER ########
    loc, scale = data_bounds(x_k_data, mode="center_lengths")
    loc = loc.to(device)
    scale = scale.to(device)
    base_distribution = LogisticSigmoid(system.dim(), temperature=ls_temp, loc=loc, scale=scale)
    decorrupter = MaskedRQSNFTF(system.dim(), trainable=True, hidden_features=128, n_layers=5).to(device)
    decorrupter_density = CompositeDensityModel([decorrupter], base_distribution).to(device)

    print("Training NF decorrupter")
    optimizer = torch.optim.Adam(decorrupter.parameters(), lr=decorrupter_params["lr"], weight_decay=1e-3)
    dtf_concentration_loss_xk = (
        lambda composite_model, x: loss.dtf_data_concentration_loss(
            composite_model.domain_tfs[0], x, concentration_point=loc, radius=scale
        )
    )
    decorrupter_density, _, _ = train.train(
        decorrupter_density,
        x_dataloader,
        {"mle": loss.mle_loss, "dtf_conc": dtf_concentration_loss_xk},
        optimizer,
        epochs=decorrupter_params["epochs"],
        verbose=True,
        use_best="mle",
        clip_grad_norm=5.0,
        restore_loss_threshold=50.0,
    )
    print("Done.\n")

    decorrupter_trained = MaskedRQSNFTF.copy_from_trainable(decorrupter).to(device)

    x_k_data = x_k_data.to(device)
    x_kp1_data = x_kp1_data.to(device)
    x0_data = x0_data.to(device)
    y_k_data, _ = decorrupter_trained(x_k_data)
    y_kp1_data, _ = decorrupter_trained(x_kp1_data)
    y0_data, _ = decorrupter_trained(x0_data)

    ######## TRAIN MOVER ########
    print("Training mover")
    mover = VolumePreservingNFTF(system.dim(), trainable=True, hidden_features=128, n_layers=5).to(device)
    mover_joint = StackedTF([mover, mover])

    y_joint_data = torch.cat([y_k_data, y_kp1_data], dim=1)
    z_joint_data = y_joint_data
    for _ in range(mover_iterations):
        print("Fitting GMM LF...")
        gmm_lf = train.fit_gaussian_lf_em(
            z_joint_data.to(torch.device("cpu")),
            n_components=n_basis,
            reg_covar=5e-3,
            max_iter=100,
        )
        mover_density = CompositeDensityModel([mover_joint], gmm_lf).to(device)
        y_joint_dataloader = DataLoader(
            TensorDataset(y_joint_data.to(torch.device("cpu"))),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=use_gpu,
        )

        optimizers = {
            "dtf": torch.optim.Adam(
                [
                    {"params": mover_density.domain_tfs[0].parameters(), "lr": mover_params["lr_dtf"], "weight_decay": 1e-3},
                ]
            ),
        }
        mover_density, _, _ = train.train_iterate(
            mover_density,
            y_joint_dataloader,
            {"mle": loss.mle_loss},
            optimizers,
            epochs_per_group=mover_params["n_epochs_per_group"],
            iterations=mover_params["iterations"],
            verbose=True,
            use_best="mle",
            clip_grad_norm=5.0,
            restore_loss_threshold=50.0,
        )
        print("Done.\n")
        z_joint_data, _ = mover_joint(y_joint_data)

    mover_trained = VolumePreservingNFTF.copy_from_trainable(mover).to(device)

    weights = gmm_lf.get_w()
    z_marginal = gmm_lf.basis.marginal(marginal_dims=range(system.dim()))
    z_means, z_stds = z_marginal.means_stds()
    z_params = torch.stack([z_means, z_stds], dim=-1)

    zp_marginal = gmm_lf.basis.marginal(marginal_dims=range(system.dim(), 2 * system.dim()))
    zp_means, zp_stds = zp_marginal.means_stds()
    zp_params = torch.stack([zp_means, zp_stds], dim=-1)

    phi_basis = GaussianBasis(fixed_params=z_params).to(device)
    psi_basis = GaussianBasis(fixed_params=zp_params).to(device)
    psi0_basis = GaussianBasis.random_init(
        system.dim(),
        n_basis=n_basis,
        offsets=torch.tensor([0.0], device=device),
        variance=4.0,
        min_std=1e-4,
    ).to(device)

    lrff = LinearRFF(phi_basis, psi_basis, a_fixed=weights.to(device)).to(device)
    ff = LinearFF(lrff.get_a(), phi_basis, psi0_basis).to(device)

    z0_data, _ = mover_trained(y0_data)
    z0_dataloader = DataLoader(TensorDataset(z0_data.to(torch.device("cpu"))), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    optimizers = {
        "basis": torch.optim.Adam(ff.basis_params(), lr=init_params["lr_basis"]),
        "weights": torch.optim.Adam(ff.weight_params(), lr=init_params["lr_weights"]),
    }
    ff, best_loss_init, training_time_init = train.train_iterate(
        ff,
        z0_dataloader,
        {"mle": loss.mle_loss},
        optimizers,
        epochs_per_group=init_params["n_epochs_per_group"],
        iterations=init_params["iterations"],
        verbose=True,
        use_best="mle",
    )
    print("Done.\n")
    print(f"Initial model loss: {best_loss_init:.6f}, training time: {training_time_init:.2f}s")

    ######## ANALYSIS ########
    base_belief_seq = propagate.propagate(ff, lrff, n_steps=problem.n_timesteps)
    belief_seq = [CompositeDensityModel([decorrupter_trained, mover_trained], belief) for belief in base_belief_seq]
    tran_model = CompositeConditionalModel([decorrupter_trained, mover_trained], lrff).to(device)

    ll_per_step = []
    for i in range(problem.n_timesteps):
        data_i = test_traj_data[i].to(device)
        ll = avg_log_likelihood(belief_seq[i], data_i)
        ll_per_step.append(ll.detach().cpu().reshape(()))
        print(f"Avg log-likelihood at time {i}: {float(ll_per_step[-1]):.6f}")
    prop_ll_vector = torch.stack(ll_per_step).to(dtype=torch.float32)

    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(__file__).stem
    lo = float(problem.plot_bounds_low[0].item())
    hi = float(problem.plot_bounds_high[0].item())

    ll_out_path = out_dir / f"{stem}__ll.png"
    plt.figure(figsize=(8, 4))
    plt.plot(prop_ll_vector.numpy(), marker="o")
    plt.xlabel("timestep")
    plt.ylabel("avg log-likelihood")
    plt.title(f"{stem}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(ll_out_path, dpi=200)
    plt.close()
    print(f"Saved LL plot to {ll_out_path}")

    beliefs_out_path = out_dir / f"{stem}__beliefs.png"
    _plot_snd_1d_belief_trajectory(
        problem,
        belief_seq,
        test_traj_data,
        beliefs_out_path,
        n_grid=400,
        title=f"{stem}: propagated beliefs vs test trajectories",
    )
    print(f"Saved propagated beliefs plot to {beliefs_out_path}")

    # Keep the same 2D-style belief figure API used in other experiments.
    # For 1D SND this is less informative, so we only generate it if plotting succeeds.
    try:
        fig, axes = plt.subplots(2, problem.n_timesteps, figsize=(4 * problem.n_timesteps, 8))
        fig.suptitle("Beliefs at each time step")
        for i in range(problem.n_timesteps):
            data_i = test_traj_data[i]
            axes[0, i].hist(data_i[:, 0].detach().cpu().numpy(), bins=60, density=True, alpha=0.5, color="C1")
            plot_belief(axes[1, i], belief_seq[i], x_range=(lo, hi))
            axes[0, i].set_xlim(lo, hi)
            axes[1, i].set_xlim(lo, hi)
            axes[0, i].set_title(f"t={i} samples")
            axes[1, i].set_title(f"t={i} belief")
            axes[0, i].grid(True, alpha=0.2)
            axes[1, i].grid(True, alpha=0.2)
        fig.tight_layout()
        beliefs_legacy_out_path = out_dir / f"{stem}__beliefs_legacy.png"
        fig.savefig(beliefs_legacy_out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved legacy beliefs panel plot to {beliefs_legacy_out_path}")
    except Exception as exc:
        print(f"Skipping legacy beliefs panel plot: {exc}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tran_model_cpu = tran_model.cpu().eval()

    cond_out_path = out_dir / f"{stem}__conditional_comparison.png"
    _plot_snd_conditional_true_vs_learned(
        tran_model_cpu,
        system,
        cond_out_path,
        x_k_data,
        x_kp1_data,
        problem=problem,
        n_grid=200,
        title=f"{stem}: true vs learned conditional density",
    )
    print(f"Saved conditional density comparison plot to {cond_out_path}")


if __name__ == "__main__":
    main()
