from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.factor_forms import QuadraticFF
from rational_factor.models.mdp import NeuralActionQRFF
from rational_factor.models.parameters import (
    NeuralParameters,
    TrainableParameters,
    param_group_iter,
)
from rational_factor.models.train import Trainer
from rational_factor.systems.base import sample_trajectories
from rational_factor.systems.problems import OPEN_LOOP_OBSERVABLE_PROBLEMS
from rational_factor.tools.misc import init_gaussian_basis_params_gmm
import rational_factor.models.loss as loss


def _action_conditional_mle_loss(model, xp, conditioner, action):
    return loss.conditional_mle_loss(model, xp, conditioner, action=action)


def _plot_control_slices(
    model: NeuralActionQRFF,
    u: torch.Tensor,
    x: torch.Tensor,
    xp: torch.Tensor,
    *,
    n_slices: int = 20,
    u_bandwidth: float = 0.2,
    grid_n: int = 160,
    out_path: Path,
    title: str,
    dpi: int = 400,
):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    u_cpu = u.detach().cpu()
    x_cpu = x.detach().cpu()
    xp_cpu = xp.detach().cpu()

    rng = torch.Generator().manual_seed(0)
    perm = torch.randperm(u_cpu.shape[0], generator=rng)
    selected = []
    for idx in perm.tolist():
        u_s = u_cpu[idx]
        if all((u_s - s).norm() > 0.5 * u_bandwidth for s in selected):
            selected.append(u_s)
        if len(selected) >= n_slices:
            break
    while len(selected) < n_slices:
        selected.append(u_cpu[perm[len(selected) % len(perm)]])

    lo = float(min(x_cpu.min(), xp_cpu.min()))
    hi = float(max(x_cpu.max(), xp_cpu.max()))
    pad = 0.05 * (hi - lo + 1e-6)
    lo, hi = lo - pad, hi + pad
    xs = torch.linspace(lo, hi, grid_n)
    xps = torch.linspace(lo, hi, grid_n)
    X, XP = torch.meshgrid(xs, xps, indexing="xy")
    x_flat = X.reshape(-1, 1).to(device=device, dtype=dtype)
    xp_flat = XP.reshape(-1, 1).to(device=device, dtype=dtype)

    ncols = 5
    nrows = (n_slices + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows), squeeze=False)

    model.eval()
    with torch.no_grad():
        for ax, u_s in zip(axes.ravel(), selected):
            dist = (u_cpu - u_s).norm(dim=-1)
            mask = dist <= u_bandwidth

            u_flat = u_s.to(device=device, dtype=dtype).reshape(1, -1).expand(x_flat.shape[0], -1)
            log_p = model.log_density(xp_flat, conditioner=x_flat, action=u_flat)
            dens = log_p.exp().reshape(grid_n, grid_n).detach().cpu()
            ax.pcolormesh(
                X.numpy(),
                XP.numpy(),
                dens.numpy(),
                shading="auto",
                cmap="viridis",
                zorder=1,
            )
            ax.scatter(
                x_cpu[mask, 0],
                xp_cpu[mask, 0],
                s=0.8,
                c="C1",
                alpha=0.55,
                linewidths=0,
                rasterized=True,
                zorder=2,
            )
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.set_title(f"u={float(u_s.squeeze()):.2f}  (n={int(mask.sum())})", fontsize=9)
            ax.set_xlabel("x")
            ax.set_ylabel("x'")

        for ax in axes.ravel()[len(selected) :]:
            ax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def _plot_multi_controller_beliefs(
    rollouts: list[dict],
    *,
    x_range: tuple[float, float],
    out_path: Path,
    title: str,
    n_grid: int = 500,
    dpi: int = 400,
):
    """One row per controller: belief curve over MC particle histogram at each t."""
    n_ctrl = len(rollouts)
    n_slices = len(rollouts[0]["beliefs"])
    lo, hi = x_range
    dtype, device = rollouts[0]["beliefs"][0].dtype_device()
    x_grid = torch.linspace(lo, hi, n_grid, dtype=dtype, device=device).unsqueeze(1)
    x_np = x_grid.squeeze(1).detach().cpu().numpy()

    fig, axes = plt.subplots(
        n_ctrl,
        n_slices,
        figsize=(2.4 * n_slices, 2.3 * n_ctrl),
        squeeze=False,
        sharex=True,
    )
    for r, rollout in enumerate(rollouts):
        beliefs = rollout["beliefs"]
        traj_data = rollout["traj"]
        controls = rollout["controls"]
        for t in range(n_slices):
            ax = axes[r, t]
            samples = traj_data[t][:, 0].detach().cpu().numpy()
            ax.hist(samples, bins=60, density=True, color="C1", alpha=0.35, zorder=1)
            with torch.no_grad():
                beliefs[t].eval()
                pdf = beliefs[t](x_grid).detach().cpu().numpy()
            ax.plot(x_np, pdf, color="C0", lw=1.6, zorder=2)
            ax.set_xlim(lo, hi)
            ax.grid(True, alpha=0.25)
            u_label = (
                f"u={float(controls[t - 1].reshape(-1)[0]):.2f}"
                if t > 0
                else "init"
            )
            ax.set_title(f"t={t} ({u_label})", fontsize=7)
            if t == 0:
                ax.set_ylabel(f"ctrl {r}\ndensity", fontsize=8)
            if r == n_ctrl - 1:
                ax.set_xlabel("x", fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    problem_name = "scalar_nonlinear_drift"
    problem = OPEN_LOOP_OBSERVABLE_PROBLEMS[problem_name]

    use_gpu = torch.cuda.is_available()
    dtype = torch.float64
    # Q0 / LQ0 are (n_basis**2, n_basis**2); keep smaller than the linear test.
    n_basis = 12
    pd_jitter = 1e-6
    gaussian_std_epsilon = 1e-6
    action_features = problem.system.control_dim()
    hidden_features = 64
    num_hidden_layers = 2

    tran_params = {
        "epochs": 10,
        "basis_iterations": 2,
        "neural_iterations": 3,
        "weights_iterations": 3,
        "lr_basis": 3e-3,
        "lr_neural": 2e-3,
        "lr_weights": 1e-2,
    }
    init_params = {
        "epochs": 40,
        "coeff_iterations": 8,
        "lr_coeffs": 1e-2,
        "basis_iterations": 2,
        "lr_basis": 1e-3,
    }
    batch_size = 256
    u_bandwidth = 0.25
    n_plot_slices = 20
    n_controllers = 5
    figure_dpi = 400

    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"device={device}, dtype={dtype}, n_basis={n_basis}")

    system = problem.system
    dim = system.dim()

    u, x_k, x_kp1 = problem.train_state_transition_data()
    u = u.to(dtype=dtype)
    x_k = x_k.to(dtype=dtype)
    x_kp1 = x_kp1.to(dtype=dtype)
    print(f"transition data: u={tuple(u.shape)}, x={tuple(x_k.shape)}, xp={tuple(x_kp1.shape)}")

    xp_dataloader = DataLoader(
        TensorDataset(x_kp1, x_k, u),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_gpu,
    )

    phi_means, phi_stds = init_gaussian_basis_params_gmm(
        x_k, dim, n_basis, epsilon=gaussian_std_epsilon, device=device, dtype=dtype,
    )
    phi = GaussianBasis(mean_params=phi_means, std_params=phi_stds, coeffs=None)

    mlp_kw = dict(
        in_features=action_features,
        out_features=(dim, n_basis),
        batch_shape=(1,),
        context_key="action",
        hidden_features=hidden_features,
        num_hidden_layers=num_hidden_layers,
        activation=torch.nn.Tanh,
        zero_init_last=False,
    )
    psi_means = NeuralParameters(**mlp_kw, positive=False)
    psi_stds = NeuralParameters(**mlp_kw, positive=True)
    psi = GaussianBasis(mean_params=psi_means, std_params=psi_stds, coeffs=None)

    LA = TrainableParameters.random_init(
        shape=(1, n_basis, n_basis), mean=0.0, std=0.1
    ).to(device=device, dtype=dtype)

    tran_model = NeuralActionQRFF(
        LA,
        phi,
        psi,
        action_features=action_features,
        numerical_tolerance=problem.numerical_tolerance,
        pd_jitter=pd_jitter,
        hidden_features=hidden_features,
        num_hidden_layers=num_hidden_layers,
        zero_init_last=False,
        positive=False,
    ).to(device=device, dtype=dtype)

    print("Training NeuralActionQRFF transition model")
    trainer = Trainer(tran_model, device=device)
    trainer.add_dataset(xp_dataloader, "transitions")
    trainer.add_optimizer(
        torch.optim.Adam(param_group_iter((phi_means, phi_stds)), lr=tran_params["lr_basis"]),
        "phi_basis",
        n_iterations=tran_params["basis_iterations"],
    )
    trainer.add_optimizer(
        torch.optim.Adam(
            list(psi_means.parameters())
            + list(psi_stds.parameters())
            + list(tran_model.LQ0.parameter_modules()[0].parameters()),
            lr=tran_params["lr_neural"],
        ),
        "neural",
        n_iterations=tran_params["neural_iterations"],
    )
    trainer.add_optimizer(
        torch.optim.Adam(param_group_iter((LA,)), lr=tran_params["lr_weights"]),
        "weights",
        n_iterations=tran_params["weights_iterations"],
    )
    trainer.add_loss_function(_action_conditional_mle_loss, "mle", "transitions")
    tran_model, best_loss, train_time = trainer.train(
        epochs=tran_params["epochs"],
        use_best="mle",
        verbose=True,
    )
    if tran_model is None:
        raise SystemExit("No stable transition checkpoint was found.")
    print(f"Done. best mle loss={best_loss:.4f}, time={train_time:.1f}s")

    for parameter in tran_model.parameters():
        parameter.requires_grad_(False)

    out_dir = Path("figures/control")
    _plot_control_slices(
        tran_model,
        u,
        x_k,
        x_kp1,
        n_slices=n_plot_slices,
        u_bandwidth=u_bandwidth,
        dpi=figure_dpi,
        out_path=out_dir / f"{problem_name}_neural_action_qrff_slices.png",
        title=(
            f"{problem_name} — NeuralActionQRFF  "
            f"n_basis={n_basis}, loss={best_loss:.4f}, Δu≤{u_bandwidth}"
        ),
    )

    # --- Initial belief ---
    x0 = problem.train_initial_state_data().to(dtype=dtype)
    x0_dataloader = DataLoader(
        TensorDataset(x0), batch_size=batch_size, shuffle=True, pin_memory=use_gpu
    )
    psi0_means, psi0_stds = init_gaussian_basis_params_gmm(
        x0, dim, n_basis, epsilon=gaussian_std_epsilon, device=device, dtype=dtype,
    )
    psi0 = GaussianBasis(mean_params=psi0_means, std_params=psi0_stds, coeffs=None)
    LV = TrainableParameters.random_init(
        shape=(1, n_basis, n_basis), mean=0.0, std=0.1
    ).to(device=device, dtype=dtype)
    init_model = QuadraticFF.from_rff(tran_model, LV, psi0).to(device=device, dtype=dtype)

    print("Training initial QuadraticFF belief")
    init_trainer = Trainer(init_model, device=device)
    init_trainer.add_dataset(x0_dataloader, "initial")
    init_trainer.add_optimizer(
        torch.optim.Adam(param_group_iter((LV,)), lr=init_params["lr_coeffs"]),
        "coeffs",
        n_iterations=init_params["coeff_iterations"],
    )
    init_trainer.add_optimizer(
        torch.optim.Adam(
            param_group_iter((psi0_means, psi0_stds)), lr=init_params["lr_basis"]
        ),
        "psi0_basis",
        n_iterations=init_params["basis_iterations"],
    )
    init_trainer.add_loss_function(loss.mle_loss, "mle", "initial")
    init_model, best_loss_init, train_time_init = init_trainer.train(
        epochs=init_params["epochs"],
        use_best="mle",
        verbose=True,
    )
    if init_model is None:
        raise SystemExit("No stable initial-density checkpoint was found.")
    print(f"Done. best init mle={best_loss_init:.4f}, time={train_time_init:.1f}s")

    # Several fixed open-loop controllers; same u_k for every particle at step k.
    rollouts = []
    lo = float(problem.plot_bounds_low[0])
    hi = float(problem.plot_bounds_high[0])
    for ctrl_i in range(n_controllers):
        with torch.random.fork_rng():
            torch.manual_seed(problem.seed + 17 + 101 * ctrl_i)
            controls = [
                problem.control_sampler(1).reshape(-1).to(dtype=dtype)
                for _ in range(problem.n_timesteps)
            ]
        controller = lambda k, controls=controls: controls[k]
        with torch.random.fork_rng():
            torch.manual_seed(problem.seed + 1 + 101 * ctrl_i)
            traj_data = [
                x.to(dtype=dtype)
                for x in sample_trajectories(
                    system,
                    problem.initial_state_sampler,
                    controller=controller,
                    n_timesteps=problem.n_timesteps,
                    n_trajectories=problem.n_trajectories_test,
                )
            ]
        print(
            f"controller {ctrl_i}: "
            + ", ".join(f"{float(c.reshape(-1)[0]):.3f}" for c in controls)
        )

        belief_seq = [init_model.eval()]
        with torch.no_grad():
            belief = init_model
            for u_k in controls:
                belief = tran_model.propagate(belief, u_k.to(device=device))
                belief_seq.append(belief.eval())

        for slice_x in traj_data:
            lo = min(lo, float(slice_x.min()))
            hi = max(hi, float(slice_x.max()))
        rollouts.append(
            {"beliefs": belief_seq, "traj": traj_data, "controls": controls}
        )

    pad = 0.05 * (hi - lo + 1e-6)
    lo, hi = lo - pad, hi + pad

    _plot_multi_controller_beliefs(
        rollouts,
        x_range=(lo, hi),
        dpi=figure_dpi,
        out_path=out_dir / f"{problem_name}_neural_action_qrff_beliefs.png",
        title=(
            f"{problem_name} — open-loop belief propagation ({n_controllers} controllers)  "
            f"tran={best_loss:.3f}, init={best_loss_init:.3f}"
        ),
    )
