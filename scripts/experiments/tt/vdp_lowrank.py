"""Van der Pol: low-rank SumProdRFF with unwrapped Gaussian tensor-product bases.

B = U V^T is a row-stochastic LowRankFactorizationMPO.  The factor and basis
coefficient tensor operations, including prediction, remain structured TT/MPO
operations, and recursive propagation uses the structured matrix/vector interfaces.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.tools.misc import init_gaussian_basis_params_gmm
try:
    from rational_factor.models.tt.tt_basis import TTBasis
except ModuleNotFoundError as exc:
    if exc.name not in ('rational_factor.models.tt.tt_basis', 'rational_factor.models.tt'):
        raise
    from rational_factor.models.tt_basis import TTBasis
try:
    from rational_factor.models.tt.tt_parameters import (
        LowRankFactorizationMPOParameters, TTVectorParameters,
    )
except ModuleNotFoundError as exc:
    if exc.name not in ('rational_factor.models.tt.tt_parameters', 'rational_factor.models.tt'):
        raise
    from rational_factor.models.tt_parameters import (
        LowRankFactorizationMPOParameters, TTVectorParameters,
    )
from rational_factor.models.parameters import (
    PositiveParameters, TrainableParameters, param_group_iter,
)
from rational_factor.models.factor_forms import SumProdRFF, LinearFF
from rational_factor.models.kde import GaussianKDE
from rational_factor.models.train import Trainer, LossVisualizer
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood
from rational_factor.tools.visualization import plot_belief
import rational_factor.models.loss as loss


def _plot_conditional_slices_model_vs_data(
    tran_model,
    x_k_data: torch.Tensor,
    x_kp1_data: torch.Tensor,
    out_path: Path,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    n_random_points: int = 10,
    n_grid: int = 120,
    min_points_per_slice: int = 40,
    bin_width: float = 0.4,
    random_seed: int = 0,
    title: str = "",
) -> None:
    """Compare trained p(x'|x) to binned empirical samples and a conditional KDE.

    Conditioners share one randomly chosen x1 and span different x2 values so
    rows can be compared for conditional dependence on x2.
    """
    if x_k_data.shape[1] != 2 or x_kp1_data.shape[1] != 2:
        raise ValueError("_plot_conditional_slices_model_vs_data expects 2D state data")

    param = next(iter(tran_model.parameters()), None)
    buffer = next(iter(tran_model.buffers()), None)
    dev = (
        param.device
        if param is not None
        else (buffer.device if buffer is not None else torch.device("cpu"))
    )
    dt = (
        param.dtype
        if param is not None
        else (buffer.dtype if buffer is not None else torch.float32)
    )

    xk_cpu = x_k_data.detach().cpu()
    xkp1_cpu = x_kp1_data.detach().cpu()
    lo = torch.tensor([x_range[0], y_range[0]], dtype=xk_cpu.dtype)
    hi = torch.tensor([x_range[1], y_range[1]], dtype=xk_cpu.dtype)

    x_lin = torch.linspace(float(lo[0]), float(hi[0]), n_grid)
    y_lin = torch.linspace(float(lo[1]), float(hi[1]), n_grid)
    X, Y = torch.meshgrid(x_lin, y_lin, indexing="xy")
    xp_grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).to(device=dev, dtype=dt)

    n_data = xk_cpu.shape[0]
    if n_data == 0:
        raise ValueError("x_k_data is empty")
    n_random_points = max(1, min(n_random_points, n_data))
    rng = torch.Generator(device="cpu")
    rng.manual_seed(random_seed)
    half_width = torch.full((2,), 0.5 * float(bin_width), dtype=xk_cpu.dtype)

    ref_idx = int(torch.randint(0, n_data, (1,), generator=rng).item())
    x1_fixed = 0.0
    near_x1 = (xk_cpu[:, 0] - x1_fixed).abs() <= half_width[0]
    near_pts = xk_cpu[near_x1]
    if near_pts.shape[0] < n_random_points:
        near_pts = xk_cpu
        x1_fixed = float(near_pts[ref_idx, 0])
    order = torch.argsort(near_pts[:, 1])
    near_sorted = near_pts[order]
    if near_sorted.shape[0] == n_random_points:
        pick = torch.arange(n_random_points)
    else:
        pick = torch.linspace(
            0, near_sorted.shape[0] - 1, n_random_points
        ).round().long()
        pick = torch.unique(pick)
        if pick.numel() < n_random_points:
            need = n_random_points - pick.numel()
            all_idx = torch.arange(near_sorted.shape[0])
            mask = torch.ones(near_sorted.shape[0], dtype=torch.bool)
            mask[pick] = False
            extra = all_idx[mask][:need]
            pick = torch.sort(torch.cat([pick, extra]))[0]
    centers = near_sorted[pick]
    centers = centers.clone()
    centers[:, 0] = x1_fixed

    fig, axes = plt.subplots(
        n_random_points, 3, figsize=(15, 3.2 * n_random_points), squeeze=False
    )
    cmap = "viridis"
    eps = torch.finfo(dt).eps

    xk_dev = xk_cpu.to(device=dev, dtype=dt)
    xkp1_dev = xkp1_cpu.to(device=dev, dtype=dt)
    bw_x = GaussianKDE.scott_bandwidth(xk_dev).clamp_min(eps)
    bw_xp = GaussianKDE.scott_bandwidth(xkp1_dev).clamp_min(eps)

    with torch.no_grad():
        tran_model.eval()
        for row, center in enumerate(centers):
            lo_box = center - half_width
            hi_box = center + half_width
            mask = (
                (xk_cpu[:, 0] >= lo_box[0])
                & (xk_cpu[:, 0] < hi_box[0])
                & (xk_cpu[:, 1] >= lo_box[1])
                & (xk_cpu[:, 1] < hi_box[1])
            )
            xp_slice = xkp1_cpu[mask]

            ax_emp = axes[row, 0]
            if xp_slice.shape[0] >= min_points_per_slice:
                ax_emp.scatter(
                    xp_slice[:, 0].numpy(),
                    xp_slice[:, 1].numpy(),
                    s=5,
                    alpha=1.0,
                    c="orange",
                    edgecolors="none",
                    rasterized=True,
                )
            else:
                ax_emp.text(
                    0.5,
                    0.5,
                    f"Too few samples\nn={xp_slice.shape[0]}",
                    ha="center",
                    va="center",
                    transform=ax_emp.transAxes,
                )
            ax_emp.set_title(
                "empirical samples x' | x in bin\n"
                f"n={xp_slice.shape[0]}, "
                f"x1={float(center[0]):.3f}, x2={float(center[1]):.3f}"
            )
            ax_emp.set_xlim(float(lo[0]), float(hi[0]))
            ax_emp.set_ylim(float(lo[1]), float(hi[1]))
            ax_emp.set_aspect("equal")
            ax_emp.set_ylabel("x'_2")

            center_dev = center.to(device=dev, dtype=dt)
            cond = center_dev.unsqueeze(0).expand(xp_grid.shape[0], -1)
            model_pdf = (
                tran_model.log_density(xp_grid, conditioner=cond)
                .exp()
                .reshape(n_grid, n_grid)
                .detach()
                .cpu()
                .numpy()
            )
            ax_model = axes[row, 1]
            cf = ax_model.contourf(X.numpy(), Y.numpy(), model_pdf, levels=40, cmap=cmap)
            fig.colorbar(cf, ax=ax_model, fraction=0.046, pad=0.04)
            ax_model.set_title(
                "model p(x'|x1 fixed, x2)\n"
                f"x2={float(center[1]):.3f}, "
                f"bin_w=({float(2 * half_width[0]):.3f}, {float(2 * half_width[1]):.3f})"
            )
            ax_model.set_xlim(float(lo[0]), float(hi[0]))
            ax_model.set_ylim(float(lo[1]), float(hi[1]))
            ax_model.set_aspect("equal")
            if xp_slice.shape[0] > 0:
                ax_model.scatter(
                    xp_slice[:, 0].numpy(),
                    xp_slice[:, 1].numpy(),
                    s=5,
                    alpha=1.0,
                    c="orange",
                    edgecolors="none",
                    rasterized=True,
                )

            diff_x = xk_dev - center_dev.unsqueeze(0)
            w_x = torch.exp(-0.5 * diff_x.square().sum(dim=1) / (bw_x * bw_x))
            w_sum = w_x.sum().clamp_min(eps)
            kde_vals = []
            block = 1024
            for start in range(0, xp_grid.shape[0], block):
                end = min(start + block, xp_grid.shape[0])
                xp_blk = xp_grid[start:end]
                diff_xp = xp_blk[:, None, :] - xkp1_dev[None, :, :]
                k_xp = torch.exp(-0.5 * diff_xp.square().sum(dim=2) / (bw_xp * bw_xp))
                kde_vals.append((k_xp * w_x.unsqueeze(0)).sum(dim=1) / w_sum)
            kde_pdf = (
                torch.cat(kde_vals, dim=0).reshape(n_grid, n_grid).detach().cpu().numpy()
            )

            ax_kde = axes[row, 2]
            cf_kde = ax_kde.contourf(X.numpy(), Y.numpy(), kde_pdf, levels=40, cmap=cmap)
            fig.colorbar(cf_kde, ax=ax_kde, fraction=0.046, pad=0.04)
            ax_kde.set_title(
                f"conditional KDE p(x'|x)\n"
                f"x1={float(center[0]):.3f}, x2={float(center[1]):.3f}"
            )
            ax_kde.set_xlim(float(lo[0]), float(hi[0]))
            ax_kde.set_ylim(float(lo[1]), float(hi[1]))
            ax_kde.set_aspect("equal")
            if xp_slice.shape[0] > 0:
                ax_kde.scatter(
                    xp_slice[:, 0].numpy(),
                    xp_slice[:, 1].numpy(),
                    s=5,
                    alpha=1.0,
                    c="orange",
                    edgecolors="none",
                    rasterized=True,
                )

    for col in range(3):
        axes[-1, col].set_xlabel("x'_1")
    if title:
        fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_gaussian_primitives(primitives: GaussianBasis, out_path: Path,
                              ranges: tuple[tuple[float, float], ...], title: str) -> None:
    """Visualize the local 1D Gaussian factors, not N tensor-product functions."""
    with torch.no_grad():
        mu, std = primitives.means_stds()
        mu, std = mu[0].detach().cpu(), std[0].detach().cpu()
        fig, axes = plt.subplots(1, len(ranges), figsize=(5.0 * len(ranges), 3.3), squeeze=False)
        for k, (lo, hi) in enumerate(ranges):
            grid = torch.linspace(lo, hi, 250, dtype=mu.dtype)
            curves = torch.exp(-0.5 * ((grid[:, None] - mu[k]) / std[k]) ** 2) / (
                std[k] * (2.0 * torch.pi) ** 0.5
            )
            axes[0, k].plot(grid.numpy(), curves.numpy(), lw=1)
            axes[0, k].set_title(f"Dimension {k+1}: {curves.shape[1]} Gaussian factors")
            axes[0, k].set_xlabel("State")
            axes[0, k].set_ylabel("Density")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=170, bbox_inches="tight")
        plt.close(fig)


def _new_gaussian_basis(
    data: torch.Tensor, d: int, n_local: int, *,
    dtype: torch.dtype, device: torch.device,
    bandwidth_floor: float = 0.0,
    gmm_reg_covar: float = 1e-5,
    gmm_max_iter: int = 100,
    gmm_seed: int = 0,
):
    """EM-fit Gaussian primitives to the supplied state distribution."""
    means, stds = init_gaussian_basis_params_gmm(
        data, d, n_local,
        reg_covar=gmm_reg_covar, max_iter=gmm_max_iter,
        random_state=gmm_seed, epsilon=bandwidth_floor,
        dtype=dtype, device=device,
    )
    return TTBasis(GaussianBasis(mean_params=means, std_params=stds)), (means, stds)

def build_transition_model(x_k: torch.Tensor, x_kp1: torch.Tensor, d: int, *, n_local: int, latent_rank: int,
                           mpo_rank: int, dtype: torch.dtype, device: torch.device,
                           numerical_tolerance: float, gaussian_std_floor: float = 0.25):
    """Create transition model and disjoint basis/weight parameter groups."""
    phi, phi_modules = _new_gaussian_basis(x_k, d, n_local, dtype=dtype, device=device, bandwidth_floor=gaussian_std_floor)
    psi, psi_modules = _new_gaussian_basis(x_kp1, d, n_local, dtype=dtype, device=device, bandwidth_floor=gaussian_std_floor)
    modes = (n_local,) * d
    a = TTVectorParameters.from_core_spec(
        modes, ranks=1, param_cls=PositiveParameters, mean=1.0, std=0.15,
        epsilon=1e-3, dtype=dtype, device=device,
    )
    B = LowRankFactorizationMPOParameters.from_core_spec(
        modes, rank=latent_rank, ranks_u=mpo_rank, ranks_v=mpo_rank,
        row_stochastic=True, mean=0.0, std=0.2, dtype=dtype, device=device,
    )
    rff = SumProdRFF(a, phi, psi, B, numerical_tolerance=numerical_tolerance)
    basis_params = param_group_iter((*phi_modules, *psi_modules))
    weight_params = param_group_iter((*a.parameter_modules(), *B.parameter_modules()))
    return rff, a, B, basis_params, weight_params


def build_initial_model(rff: SumProdRFF, x0: torch.Tensor, *, n_local: int, dtype: torch.dtype,
                        device: torch.device, init_tt_rank: int = 1,
                        gaussian_std_floor: float = 0.25):
    """Train an independent initial Gaussian psi0 and arbitrary-rank TT h0."""
    modes = (n_local,) * rff.phi.dim()
    h0 = TTVectorParameters.from_core_spec(
        modes, ranks=init_tt_rank, param_cls=PositiveParameters, mean=1.0, std=0.15,
        epsilon=1e-3, dtype=dtype, device=device,
    )
    # Initial psi0 has its own Gaussian locations and bandwidths.
    h0_basis, psi0_modules = _new_gaussian_basis(
        x0, rff.phi.dim(), n_local, dtype=dtype, device=device,
        bandwidth_floor=gaussian_std_floor,
    )
    return (LinearFF.from_rff(rff, h0, h0_basis).to(device=device, dtype=dtype),
            h0, psi0_modules)


if __name__ == "__main__":
    problem = FULLY_OBSERVABLE_PROBLEMS["van_der_pol"]

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    dtype = torch.float32
    # N = n_local ** d: *never* materialize this high-dimensional coefficient space.
    n_local = 20
    latent_rank = 10  # ordinary rank of B; propagation is latent_rank x latent_rank
    mpo_rank = 10     # internal TT/MPO bond ranks of the factors U, V
    tran_params = {
        "epochs": 20,
        "basis_iterations": 5,  # full dataset passes per outer epoch
        "weights_iterations": 2,
        "lr_basis": 1e-2,
        "lr_weights": 5e-2,
    }
    gaussian_std_epsilon = 0.25
    init_tt_rank = 8
    init_params = {"epochs": 20, "coeff_iterations": 10, "lr_coeffs": 1e-2,
                   "basis_iterations": 2, "lr_basis": 1e-3}
    batch_size = 256
    n_timesteps_prop = problem.n_timesteps

    dim = problem.system.dim()
    if dim != 2:
        raise ValueError(f"Expected a 2D Van der Pol system, got dim={dim}")
    x0 = problem.train_initial_state_data().to(dtype=dtype)
    x_k, x_kp1 = problem.train_state_transition_data()
    x_k, x_kp1 = x_k.to(dtype=dtype), x_kp1.to(dtype=dtype)
    traj_data = [x.to(dtype=dtype) for x in problem.test_data()]
    x0_loader = DataLoader(TensorDataset(x0), batch_size=batch_size, shuffle=True,
                           pin_memory=use_gpu)
    transition_loader = DataLoader(TensorDataset(x_kp1, x_k), batch_size=batch_size,
                                   shuffle=True, pin_memory=use_gpu)
    output_dir = Path("figures/tt/vdp")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}, modes={(n_local,) * dim}, latent_rank={latent_rank}, MPO rank={mpo_rank}")

    rff, a_params, B_params, basis_params, weight_params = build_transition_model(
        x_k, x_kp1, dim, n_local=n_local, latent_rank=latent_rank, mpo_rank=mpo_rank,
        dtype=dtype, device=device, numerical_tolerance=problem.numerical_tolerance,
        gaussian_std_floor=gaussian_std_epsilon,
    )
    transition_trainer = Trainer(
        rff, device=device,
        loss_visualizer=LossVisualizer(output_dir, buffer_size=5, filename="transition_losses.png"),
    )
    transition_trainer.add_dataset(transition_loader, "transitions")
    transition_trainer.add_optimizer(
        torch.optim.Adam(basis_params, lr=tran_params["lr_basis"]),
        "basis", n_iterations=tran_params["basis_iterations"],
    )
    transition_trainer.add_optimizer(
        torch.optim.Adam(weight_params, lr=tran_params["lr_weights"]),
        "weights", n_iterations=tran_params["weights_iterations"],
    )
    transition_trainer.add_loss_function(loss.conditional_mle_loss, "mle", "transitions")
    rff, best_loss_tran, training_time_tran = transition_trainer.train(
        epochs=tran_params["epochs"], use_best="mle", verbose=True,
    )
    if rff is None:
        raise SystemExit("No stable transition checkpoint was found.")

    # Freeze *all* transition parameters, including the Gaussian means/stds.
    for parameter in rff.parameters():
        parameter.requires_grad_(False)
    init_model, h0_params, psi0_modules = build_initial_model(
        rff, x0, n_local=n_local, dtype=dtype, device=device, init_tt_rank=init_tt_rank,
        gaussian_std_floor=gaussian_std_epsilon,
    )
    initial_trainer = Trainer(
        init_model, device=device,
        loss_visualizer=LossVisualizer(output_dir, buffer_size=5, filename="initial_losses.png"),
    )
    initial_trainer.add_dataset(x0_loader, "initial")
    initial_trainer.add_optimizer(
        torch.optim.Adam(param_group_iter(tuple(h0_params.parameter_modules())),
                         lr=init_params["lr_coeffs"]),
        "coeffs", n_iterations=init_params["coeff_iterations"],
    )
    initial_trainer.add_optimizer(
        torch.optim.Adam(param_group_iter(psi0_modules), lr=init_params["lr_basis"]),
        "psi0_basis", n_iterations=init_params["basis_iterations"],
    )
    initial_trainer.add_loss_function(loss.mle_loss, "mle", "initial")
    init_model, best_loss_init, training_time_init = initial_trainer.train(
        epochs=init_params["epochs"], use_best="mle", verbose=True,
    )
    if init_model is None:
        raise SystemExit("No stable initial-density checkpoint was found.")

    print(f"Transition loss: {best_loss_tran:.5f} ({training_time_tran:.1f}s)")
    print(f"Initial loss: {best_loss_init:.5f} ({training_time_init:.1f}s)")
    rff.eval()
    init_model.eval()
    ranges = tuple(
        (float(lo), float(hi)) for lo, hi in zip(problem.plot_bounds_low, problem.plot_bounds_high)
    )
    _plot_gaussian_primitives(rff.phi.primitives, output_dir / "phi_gaussians.png",
                              ranges, "Phi: local Gaussian factors")
    _plot_gaussian_primitives(rff.psi.primitives, output_dir / "psi_gaussians.png",
                              ranges, "Psi: local Gaussian factors")
    _plot_gaussian_primitives(init_model.psi.primitives, output_dir / "psi0_gaussians.png",
                              ranges, "Psi0: independently trained initial Gaussian factors")
    _plot_conditional_slices_model_vs_data(
        rff, x_k, x_kp1, output_dir / "conditional_slices_model_vs_data.png",
        x_range=ranges[0], y_range=ranges[1], n_random_points=10,
        n_grid=120, min_points_per_slice=20, bin_width=0.4,
        random_seed=0, title="VDP low-rank SumProdRFF: conditional slices",
    )

    # One-step transition propagation automatically uses the current h basis
    # (initially psi0, subsequently psi) to construct the correct Gram.
    with torch.no_grad():
        belief_seq = [init_model]
        for _ in range(n_timesteps_prop):
            belief_seq.append(rff.propagate(belief_seq[-1]))
        belief_seq = [belief.to(device=device, dtype=dtype).eval() for belief in belief_seq]
        for i, (belief, samples) in enumerate(zip(belief_seq, traj_data)):
            ll = avg_log_likelihood(belief, samples.to(device=device))
            print(f"Avg log-likelihood at time {i}: {float(ll):.6f}")

    # Gaussian bases integrate over R^d, so finite-box check_pdf_valid is not
    # a global normalization test; intentionally do not call it here.
    n_plot = min(len(belief_seq), 8)
    if n_plot:
        fig, axes = plt.subplots(2, n_plot, figsize=(3.2 * n_plot, 6.5), squeeze=False)
        for i, belief in enumerate(belief_seq[:n_plot]):
            plot_belief(axes[1, i], belief, x_range=ranges[0], y_range=ranges[1])
            points = traj_data[i].detach().cpu()
            axes[0, i].scatter(points[:, 0], points[:, 1], s=1)
            axes[0, i].set_aspect("equal")
            axes[0, i].set_xlim(*ranges[0])
            axes[0, i].set_ylim(*ranges[1])
            axes[0, i].set_title(f"t={i}")
        fig.suptitle("Van der Pol: propagated low-rank TT beliefs")
        fig.tight_layout()
        fig.savefig(output_dir / "beliefs.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved diagnostics in {output_dir}")
