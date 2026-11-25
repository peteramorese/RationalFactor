from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from normalizing_flow.composite_model import CompositeConditionalModel, CompositeDensityModel
from normalizing_flow.transforms import Transforms
from rational_factor.models.basis_functions import BetaBasis
from rational_factor.models.factor_forms import LinearFF
from rational_factor.models.mdp import NeuralActionLRFF
from rational_factor.models.parameters import (
    NeuralParameters,
    PositiveParameters,
    param_group_iter,
)
from rational_factor.models.train import Trainer
from rational_factor.systems.base import sample_trajectories
from rational_factor.systems.problems import OPEN_LOOP_OBSERVABLE_PROBLEMS
from rational_factor.tools.visualization import plot_belief
import rational_factor.models.loss as loss


def _action_conditional_mle_loss(model, xp, conditioner, action):
    return loss.conditional_mle_loss(model, xp, conditioner, action=action)


def _format_control(u: torch.Tensor) -> str:
    vals = u.detach().cpu().reshape(-1).tolist()
    return "[" + ", ".join(f"{v:.2f}" for v in vals) + "]"


def _plot_multi_controller_beliefs_2d(
    rollouts: list[dict],
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    out_path: Path,
    title: str,
    n_points: int = 100,
    dpi: int = 400,
    max_particles: int = 1500,
):
    """For each controller: particle cloud row above a 2D belief contour row."""
    n_ctrl = len(rollouts)
    n_slices = len(rollouts[0]["beliefs"])
    fig, axes = plt.subplots(
        2 * n_ctrl,
        n_slices,
        figsize=(2.3 * n_slices, 2.2 * 2 * n_ctrl),
        squeeze=False,
    )

    for r, rollout in enumerate(rollouts):
        beliefs = rollout["beliefs"]
        traj_data = rollout["traj"]
        controls = rollout["controls"]
        ax_part_row = 2 * r
        ax_bel_row = 2 * r + 1
        for t in range(n_slices):
            ax_p = axes[ax_part_row, t]
            ax_b = axes[ax_bel_row, t]
            samples = traj_data[t].detach().cpu()
            if samples.shape[0] > max_particles:
                idx = torch.randperm(samples.shape[0])[:max_particles]
                samples = samples[idx]

            ax_p.scatter(
                samples[:, 0].numpy(),
                samples[:, 1].numpy(),
                s=1.2,
                c="C1",
                alpha=0.45,
                linewidths=0,
                rasterized=True,
            )
            ax_p.set_aspect("equal")
            ax_p.set_xlim(*x_range)
            ax_p.set_ylim(*y_range)
            ax_p.grid(True, alpha=0.2)

            plot_belief(
                ax_b,
                beliefs[t],
                x_range=x_range,
                y_range=y_range,
                n_points=n_points,
            )
            ax_b.grid(True, alpha=0.2)

            u_label = _format_control(controls[t - 1]) if t > 0 else "init"
            ax_p.set_title(f"ctrl {r}  t={t} particles", fontsize=7)
            ax_b.set_title(f"t={t} belief ({u_label})", fontsize=7)
            if t == 0:
                ax_p.set_ylabel(f"ctrl {r}\nx2", fontsize=8)
                ax_b.set_ylabel("x2", fontsize=8)
            if r == n_ctrl - 1:
                ax_b.set_xlabel("x1", fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    problem_name = "van_der_pol"
    problem = OPEN_LOOP_OBSERVABLE_PROBLEMS[problem_name]

    use_gpu = torch.cuda.is_available()
    dtype = torch.float64
    n_basis = 100
    action_features = problem.system.control_dim()
    hidden_features = 32
    num_hidden_layers = 3

    tran_params = {
        "epochs": 100,
        "basis_iterations": 2,
        "neural_iterations": 3,
        "weights_iterations": 3,
        "wrap_iterations": 1,
        "lr_basis": 3e-3,
        "lr_neural": 2e-3,
        "lr_weights": 5e-2,
        "lr_wrap": 1e-3,
    }
    init_params = {
        "epochs": 100,
        "coeff_iterations": 8,
        "lr_coeffs": 1e-2,
        "basis_iterations": 2,
        "lr_basis": 1e-3,
    }
    batch_size = 256
    n_controllers = 5
    figure_dpi = 400

    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"device={device}, dtype={dtype}, n_basis={n_basis}, action_dim={action_features}")

    system = problem.system
    dim = system.dim()
    assert dim == 2, "lrff_vdp_beta expects a 2D system"

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

    # Static Beta phi on the unit box (after erf wrap).
    param_shape = (1, dim, n_basis)
    phi_alpha = PositiveParameters.random_init(
        shape=param_shape, mean=5.0, std=10.0, epsilon=1.0
    ).to(device=device, dtype=dtype)
    phi_beta = PositiveParameters.random_init(
        shape=param_shape, mean=5.0, std=10.0, epsilon=1.0
    ).to(device=device, dtype=dtype)
    phi = BetaBasis(alpha_params=phi_alpha, beta_params=phi_beta)

    # Action-conditioned Beta psi (neural alpha/beta).
    mlp_kw = dict(
        in_features=action_features,
        out_features=(dim, n_basis),
        batch_shape=(1,),
        context_key="action",
        hidden_features=hidden_features,
        num_hidden_layers=num_hidden_layers,
        activation=torch.nn.Tanh,
        zero_init_last=False,
        positive=True,
    )
    psi_alpha = NeuralParameters(**mlp_kw)
    psi_beta = NeuralParameters(**mlp_kw)
    psi = BetaBasis(alpha_params=psi_alpha, beta_params=psi_beta)

    a = PositiveParameters.random_init(
        shape=(1, n_basis), mean=1.0, std=1.0, epsilon=1e-3
    ).to(device=device, dtype=dtype)

    base_tran = NeuralActionLRFF(
        a,
        phi,
        psi,
        action_features=action_features,
        numerical_tolerance=problem.numerical_tolerance,
        hidden_features=hidden_features,
        num_hidden_layers=num_hidden_layers,
        zero_init_last=False,
        positive=False,
    )

    # Erf wrap maps raw states <-> unit box; wraps both x' and conditioner x.
    tran_model = CompositeConditionalModel(
        base_tran,
        context_features=dim,
        transform="erf",
        x_data=x_k,
        trainable=True,
        transform_conditioner=True,
    ).to(device=device, dtype=dtype)
    wrap_tf = tran_model.transform

    print("Training NeuralActionLRFF + erf wrap transition model")
    trainer = Trainer(tran_model, device=device)
    trainer.add_dataset(xp_dataloader, "transitions")
    trainer.add_optimizer(
        torch.optim.Adam(wrap_tf.parameters(), lr=tran_params["lr_wrap"]),
        "wrap",
        n_iterations=tran_params["wrap_iterations"],
    )
    trainer.add_optimizer(
        torch.optim.Adam(
            param_group_iter((phi_alpha, phi_beta)), lr=tran_params["lr_basis"]
        ),
        "phi_basis",
        n_iterations=tran_params["basis_iterations"],
    )
    trainer.add_optimizer(
        torch.optim.Adam(
            list(psi_alpha.parameters())
            + list(psi_beta.parameters())
            + list(base_tran.B.parameter_modules()[0].parameters()),
            lr=tran_params["lr_neural"],
        ),
        "neural",
        n_iterations=tran_params["neural_iterations"],
    )
    trainer.add_optimizer(
        torch.optim.Adam(param_group_iter((a,)), lr=tran_params["lr_weights"]),
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

    base_tran = tran_model.base_density
    for parameter in tran_model.parameters():
        parameter.requires_grad_(False)

    # Freeze wrap for the initial belief (same unit-box map, not trained further).
    trained_wrap_tf = Transforms.freeze(tran_model.transform).to(device=device, dtype=dtype)

    # --- Initial belief in latent (unit-box) space behind the frozen wrap ---
    x0 = problem.train_initial_state_data().to(dtype=dtype)
    x0_dataloader = DataLoader(
        TensorDataset(x0), batch_size=batch_size, shuffle=True, pin_memory=use_gpu
    )
    psi0_alpha = PositiveParameters.random_init(
        shape=param_shape, mean=5.0, std=10.0, epsilon=1.0
    ).to(device=device, dtype=dtype)
    psi0_beta = PositiveParameters.random_init(
        shape=param_shape, mean=5.0, std=10.0, epsilon=1.0
    ).to(device=device, dtype=dtype)
    psi0 = BetaBasis(alpha_params=psi0_alpha, beta_params=psi0_beta)
    h0 = PositiveParameters.random_init(
        shape=(1, n_basis), mean=1.0, std=1.0, epsilon=1e-3
    ).to(device=device, dtype=dtype)
    init_base = LinearFF.from_rff(base_tran, h0, psi0)
    init_model = CompositeDensityModel(
        init_base,
        transform="stacked",
        transforms=[trained_wrap_tf],
    ).to(device=device, dtype=dtype)

    print("Training initial LinearFF belief (frozen erf wrap)")
    init_trainer = Trainer(init_model, device=device)
    init_trainer.add_dataset(x0_dataloader, "initial")
    init_trainer.add_optimizer(
        torch.optim.Adam(param_group_iter((h0,)), lr=init_params["lr_coeffs"]),
        "coeffs",
        n_iterations=init_params["coeff_iterations"],
    )
    init_trainer.add_optimizer(
        torch.optim.Adam(
            param_group_iter((psi0_alpha, psi0_beta)), lr=init_params["lr_basis"]
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

    init_base = init_model.base_density

    # Propagate in latent space; wrap beliefs for plotting in original coordinates.
    rollouts = []
    x_range = (
        float(problem.plot_region().low[0]),
        float(problem.plot_region().high[0]),
    )
    y_range = (
        float(problem.plot_region().low[1]),
        float(problem.plot_region().high[1]),
    )
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
            + " | ".join(_format_control(c) for c in controls)
        )

        belief_seq = [
            CompositeDensityModel(
                init_base.eval(),
                transform="stacked",
                transforms=[trained_wrap_tf],
            ).eval()
        ]
        with torch.no_grad():
            belief = init_base
            for u_k in controls:
                belief = base_tran.propagate(belief, u_k.to(device=device))
                belief_seq.append(
                    CompositeDensityModel(
                        belief.eval(),
                        transform="stacked",
                        transforms=[trained_wrap_tf],
                    ).eval()
                )

        rollouts.append(
            {"beliefs": belief_seq, "traj": traj_data, "controls": controls}
        )

    out_dir = Path("figures/control")
    _plot_multi_controller_beliefs_2d(
        rollouts,
        x_range=x_range,
        y_range=y_range,
        dpi=figure_dpi,
        out_path=out_dir / f"{problem_name}_neural_action_lrff_beta_beliefs.png",
        title=(
            f"{problem_name} — NeuralActionLRFF (Beta + erf)  "
            f"({n_controllers} controllers)  "
            f"tran={best_loss:.3f}, init={best_loss_init:.3f}"
        ),
    )
