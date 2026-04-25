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
from rational_factor.models.domain_transformation import MaskedRQSNFTF, IdentityTF
from rational_factor.models.factor_forms import LinearFF, LinearRFF
from rational_factor.models.density_model import LogisticSigmoid
from normalizing_flow.normalizing_flow import NormalizingFlow
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood, check_pdf_valid
from rational_factor.tools.misc import data_bounds, data_mean_std
from rational_factor.tools.visualization import plot_marginal_trajectory_comparison


def _plot_quad_latent_marginal_hist_g_base(
    lrff: LinearRFF,
    base_distribution: LogisticSigmoid,
    hist_latent: torch.Tensor,
    out_path: Path,
    *,
    n_grid: int = 200,
    title: str = "",
) -> None:
    """
    Per latent dimension d: normalized histogram of ``hist_latent`` (latent z marginals),
    the 1D LogisticSigmoid marginal, and g as the marginal of g(z)=phi(z)@a (``phi_d @ a``)
    for isotropic Gaussian ``lrff.phi_basis``; axis range is set from the histogram data per dim.
    """
    param = next(iter(lrff.parameters()), None)
    buffer = next(iter(lrff.buffers()), None)
    dev = param.device if param is not None else (buffer.device if buffer is not None else torch.device("cpu"))
    dt = param.dtype if param is not None else (buffer.dtype if buffer is not None else torch.float32)

    z_train = hist_latent.to(device=dev, dtype=dt)
    state_dim = z_train.shape[1]
    phi_basis = lrff.phi_basis
    if not isinstance(phi_basis, GaussianKernelBasis):
        raise TypeError("_plot_quad_latent_marginal_hist_g_base expects lrff.phi_basis to be GaussianKernelBasis")
    loc = base_distribution.loc.to(device=dev, dtype=dt)
    scale = base_distribution.scale.to(device=dev, dtype=dt)

    n_cols = 4
    n_rows = int(np.ceil(state_dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 2.8 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    with torch.no_grad():
        lrff.eval()
        a = lrff.get_a()
        for d in range(state_dim):
            ax = axes_flat[d]
            z_d_np = z_train[:, d].detach().cpu().numpy()
            z_lo = float(np.min(z_d_np))
            z_hi = float(np.max(z_d_np))
            if z_hi <= z_lo:
                z_hi = z_lo + 1e-6

            ax.hist(z_d_np, bins=50, density=True, alpha=0.35, color="C1")

            z_line = torch.linspace(z_lo, z_hi, n_grid, device=dev, dtype=dt)
            phi_d = phi_basis.marginal((d,))
            z_1d = z_line.unsqueeze(1)
            g_curve = (phi_d(z_1d) @ a).detach().cpu().numpy()
            z_axis = z_line.detach().cpu().numpy()

            base_1d = LogisticSigmoid(
                1,
                temperature=float(base_distribution.temperature),
                loc=loc[d : d + 1].detach(),
                scale=scale[d : d + 1].detach(),
            ).to(device=dev, dtype=dt)
            base_z = z_line.unsqueeze(1)
            base_curve = base_1d(base_z).squeeze(-1).detach().cpu().numpy()

            sort_idx = np.argsort(z_axis)
            ax.plot(z_axis[sort_idx], base_curve[sort_idx], color="C2", lw=1.8, ls="--")
            ax.plot(z_axis[sort_idx], g_curve[sort_idx], color="C0", lw=2.0)
            ax.set_xlim(z_lo, z_hi)
            ax.set_ylabel("density / g")
            ax.set_title(f"latent dim {d}")
            ax.grid(True, alpha=0.25)

        for j in range(state_dim, len(axes_flat)):
            axes_flat[j].set_visible(False)

    for c in range(n_cols):
        idx = (n_rows - 1) * n_cols + c
        if idx < state_dim:
            axes_flat[idx].set_xlabel("z (flow output)")
    if title:
        fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    problem = FULLY_OBSERVABLE_PROBLEMS["quadcopter"]

    ######## USER CONFIG ########
    use_gpu = torch.cuda.is_available()
    batch_size = 512

    dtf_params = {
        "epochs": 30,
        "lr": 1e-3,
    }
    ls_temp = 0.01
    kernel_bandwidth_factor = 0.01

    init_params = {
        "n_epochs_per_group": [1],
        "iterations": 10,
        "lr_weights": 1e-2,
    }

    ######## SETUP ########
    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using device: ", device)

    system = problem.system
    x0_data = problem.train_initial_state_data()
    x_k_data, x_kp1_data = problem.train_state_transition_data()
    test_traj_data = problem.test_data()

    x_dataloader = DataLoader(TensorDataset(x_k_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    x_kp1_dataloader = DataLoader(TensorDataset(x_kp1_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    x0_dataloader = DataLoader(TensorDataset(x0_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    loc, scale = data_bounds(torch.cat([x0_data, x_k_data, x_kp1_data], dim=0), mode="center_lengths")
    #loc, scale = data_mean_std(torch.cat([x0_data, x_k_data, x_kp1_data], dim=0))
    loc = loc.to(device)
    scale = scale.to(device)
    print("scale: ", scale)
    base_distribution = LogisticSigmoid(system.dim(), temperature=ls_temp, loc=loc, scale=scale)
    dtf = MaskedRQSNFTF(system.dim(), trainable=True, hidden_features=256, n_layers=5).to(device)
    decorrupter = CompositeDensityModel([dtf], base_distribution).to(device)

    ######## TRAIN LCM ########

    print("Training NF decorrupter")

    optimizer = torch.optim.Adam(decorrupter.parameters(), lr=dtf_params["lr"])

    dtf_concentration_loss_xk = lambda composite_model, x: 1.0 * loss.dtf_data_concentration_loss(composite_model.domain_tfs[0], x, concentration_point=loc, radius=scale)
    dtf_concentration_loss_xkp1 = lambda composite_model, x: 0.0 *loss.dtf_data_concentration_loss(composite_model.domain_tfs[0], x, concentration_point=loc, radius=scale)
    decorrupter, best_loss, training_time = train.train_multiset(
        decorrupter,
        [x_dataloader, x_kp1_dataloader],
        [{"mle": loss.mle_loss, "dtf_conc": dtf_concentration_loss_xk}, {"dtf_conc": dtf_concentration_loss_xkp1}],
        optimizer,
        epochs=dtf_params["epochs"],
        verbose=True,
        use_best="mle",
        clip_grad_norm=5.0,
        restore_loss_threshold=50.0,
    )

    #decorrupter, best_loss, training_time = train.train(
    #    decorrupter,
    #    x_dataloader,
    #    {"mle": loss.mle_loss, "dtf_conc": dtf_concentration_loss_xk},
    #    optimizers,
    #    epochs=dtf_params["epochs"],
    #    iterations=dtf_params["iterations"],
    #    verbose=True,
    #    use_best="mle",
    #    clip_grad_norm=5.0,
    #    restore_loss_threshold=50.0,
    #)
    print("Done.\n")


    x_k_data = x_k_data.to(device)
    x_kp1_data = x_kp1_data.to(device)
    x0_data = x0_data.to(device)
    x_k_transformed, _ = dtf(x_k_data)
    x_kp1_transformed, _ = dtf(x_kp1_data)
    x0_transformed, _ = dtf(x0_data)
    phi_basis = GaussianKernelBasis(x_k_transformed, trainable=False)
    psi_basis = GaussianKernelBasis(x_kp1_transformed, trainable=False)
    psi0_basis = GaussianKernelBasis(x0_transformed, trainable=False)
    phi_basis.kernel_bandwidth *= kernel_bandwidth_factor
    psi_basis.kernel_bandwidth *= kernel_bandwidth_factor
    psi0_basis.kernel_bandwidth *= kernel_bandwidth_factor

    lrff = LinearRFF(phi_basis, psi_basis, numerical_tolerance=problem.numerical_tolerance).to(device)
    tran_model = CompositeConditionalModel([dtf], lrff).to(device)

    ff = LinearFF.from_rff(lrff, psi0_basis).to(device)
    init_model = CompositeDensityModel([dtf], ff).to(device)
    

    print("Training initial model weights")
    optimizers = {
        "weights": torch.optim.Adam(init_model.density_model.weight_params(), lr=init_params["lr_weights"]),
    }
    init_model, best_loss_init, training_time_init = train.train_iterate(
        init_model,
        x0_dataloader,
        {"mle": loss.mle_loss},
        optimizers,
        epochs_per_group=init_params["n_epochs_per_group"],
        iterations=init_params["iterations"],
        verbose=True,
        use_best="mle",
        clip_grad_norm=5.0,
        restore_loss_threshold=50.0,
    )
    print("Done.\n")
    print(f"Initial model loss: {best_loss_init:.6f}, training time: {training_time_init:.2f}s")

    ######### ANALYSIS ########
    #cpu = torch.device("cpu")
    #init_model.to(cpu)
    #tran_model.to(cpu)
    #base_belief_seq = propagate.propagate(
    #    init_model.density_model,
    #    tran_model.conditional_density_model,
    #    n_steps=problem.n_timesteps,
    #    device=cpu,
    #)
    #belief_seq = [CompositeDensityModel([dtf], belief) for belief in base_belief_seq]

    #ll_per_step = []
    #for i in range(problem.n_timesteps):
    #    #data_i = test_traj_data[i].to(cpu)
    #    data_i = test_traj_data[i].to(cpu)
    #    ll = avg_log_likelihood(belief_seq[i], data_i)
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
    #plt.title(f"{Path(__file__).stem}")
    #plt.grid(True, alpha=0.25)
    #plt.tight_layout()
    #plt.savefig(ll_out_path, dpi=200)
    #print(f"Saved LL plot to {ll_out_path}")

    if use_gpu:
        torch.cuda.empty_cache()
    tran_model_cpu = tran_model.cpu().eval()
    base_dist_cpu = base_distribution.cpu()
    with torch.no_grad():
        dtf_c = tran_model_cpu.domain_tfs[0]
        x_k_latent, _ = dtf_c(x_k_data.detach().cpu())
        x_kp1_latent, _ = dtf_c(x_kp1_data.detach().cpu())

    stem = Path(__file__).stem
    marg_path = out_dir / f"{stem}__latent_marginal_hist_g_base.png"
    _plot_quad_latent_marginal_hist_g_base(
        tran_model_cpu.conditional_density_model,
        base_dist_cpu,
        x_k_latent,
        marg_path,
        n_grid=200,
        title=f"{stem}: z=f(x_k); hist(x_k), g marginal = phi_d@a, LogisticSigmoid marginal",
    )
    print(f"Saved latent marginal hist / g(z) / base plot to {marg_path}")

    marg_xkp1_path = out_dir / f"{stem}__latent_marginal_hist_g_base__x_kp1.png"
    _plot_quad_latent_marginal_hist_g_base(
        tran_model_cpu.conditional_density_model,
        base_dist_cpu,
        x_kp1_latent,
        marg_xkp1_path,
        n_grid=200,
        title=f"{stem}: z=f(x); hist(x_kp1), g marginal = phi_d@a (same as x_k), LogisticSigmoid marginal",
    )
    print(f"Saved x_kp1-hist / g / base plot to {marg_xkp1_path}")


if __name__ == "__main__":
    main()
