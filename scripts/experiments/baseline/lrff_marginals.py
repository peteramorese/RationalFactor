from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.tools.misc import init_gaussian_basis_params_gmm
from rational_factor.models.factor_forms import LinearRFF, LinearFF
from rational_factor.models.parameters import (
    PositiveParameters,
    TrainableParameters,
    param_group_iter,
    RowStochasticDenseMatrixParameters,
)
from rational_factor.models.train import Trainer
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood
import rational_factor.models.loss as loss


def _square_ranges(
    x_range: tuple[float, float], y_range: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_mid = 0.5 * (x_range[0] + x_range[1])
    y_mid = 0.5 * (y_range[0] + y_range[1])
    half_span = 0.5 * max(x_range[1] - x_range[0], y_range[1] - y_range[0])
    return (x_mid - half_span, x_mid + half_span), (y_mid - half_span, y_mid + half_span)


def plot_marginal_mc_vs_belief(
    problem,
    beliefs,
    traj_data,
    *,
    n_points: int = 50,
    contour_levels: int = 10,
    scatter_kwargs: dict | None = None,
):
    """One figure: for each 2D plot marginal, MC samples (top) vs belief density (bottom).

    Layout is ``(2 * n_marginals, n_timesteps + 1)``. Columns are timesteps.
    Belief panels evaluate ``belief.marginal_density`` on the kept 2D coordinates
    (integrating out all other dimensions).
    """
    marginals_list = problem.plot_marginals_list
    if not marginals_list:
        raise ValueError("problem.plot_marginals_list is empty")

    n_slices = problem.n_timesteps + 1
    n_marginals = len(marginals_list)
    assert len(beliefs) == n_slices, "beliefs must have length n_timesteps + 1"
    assert len(traj_data) == n_slices, "traj_data must have length n_timesteps + 1"

    scatter_kwargs = {"s": 1, "alpha": 0.5, **(scatter_kwargs or {})}
    fig, axes = plt.subplots(
        2 * n_marginals,
        n_slices,
        figsize=(2.4 * n_slices, 2.4 * 2 * n_marginals),
        squeeze=False,
    )

    dim = problem.system.dim()
    for m_idx, (x_dim, y_dim) in enumerate(marginals_list):
        keep = (x_dim, y_dim)
        integrate_out = tuple(d for d in range(dim) if d not in keep)
        # marginal_density expects x columns in ascending remaining-dim order.
        remaining = tuple(d for d in range(dim) if d not in integrate_out)

        x_range, y_range = _square_ranges(
            (problem.plot_bounds_low[x_dim].item(), problem.plot_bounds_high[x_dim].item()),
            (problem.plot_bounds_low[y_dim].item(), problem.plot_bounds_high[y_dim].item()),
        )
        x_lin = np.linspace(x_range[0], x_range[1], n_points)
        y_lin = np.linspace(y_range[0], y_range[1], n_points)
        X, Y = np.meshgrid(x_lin, y_lin)

        # Pack mesh columns in `remaining` order for marginal_density.
        col_map = {x_dim: X.ravel(), y_dim: Y.ravel()}
        xy = np.stack([col_map[d] for d in remaining], axis=1)

        mc_row = 2 * m_idx
        belief_row = 2 * m_idx + 1
        pair_label = (
            f"{problem.system.state_label(x_dim)}-{problem.system.state_label(y_dim)}"
        )

        for t in range(n_slices):
            ax_mc = axes[mc_row, t]
            ax_b = axes[belief_row, t]
            samples = traj_data[t]
            ax_mc.scatter(
                samples[:, x_dim].detach().cpu().numpy(),
                samples[:, y_dim].detach().cpu().numpy(),
                **scatter_kwargs,
            )
            ax_mc.set_xlim(*x_range)
            ax_mc.set_ylim(*y_range)
            ax_mc.set_aspect("equal")

            belief = beliefs[t]
            dtype, device = belief.dtype_device()
            with torch.no_grad():
                belief.eval()
                density = belief.marginal_density(
                    torch.tensor(xy, dtype=dtype, device=device),
                    integrate_out,
                )
                Z = density.detach().cpu().numpy().reshape(X.shape)

            ax_b.contourf(X, Y, Z, levels=contour_levels)
            ax_b.contour(X, Y, Z, levels=contour_levels, colors="white", linewidths=0.5)
            ax_b.set_xlim(*x_range)
            ax_b.set_ylim(*y_range)
            ax_b.set_aspect("equal")

            if t == 0:
                ax_mc.set_ylabel(f"{pair_label}\nMC")
                ax_b.set_ylabel(f"{pair_label}\nbelief")
            if m_idx == 0:
                ax_mc.set_title(f"t={t}")
            if m_idx == n_marginals - 1:
                ax_b.set_xlabel(problem.system.state_label(x_dim))
            ax_mc.set_xticks([])
            if t > 0:
                ax_mc.set_yticks([])
                ax_b.set_yticks([])

    fig.tight_layout()
    return fig, axes


if __name__ == "__main__":
    problem_name = "cartpole"
    problem = FULLY_OBSERVABLE_PROBLEMS[problem_name]

    ###
    use_gpu = torch.cuda.is_available()
    dtype = torch.float64
    n_basis = 500
    gaussian_std_epsilon = 1e-3
    tran_params = {
        "epochs": 100,
        "basis_iterations": 3,
        "weights_iterations": 3,
        "lr_basis": 3e-3,
        "lr_weights": 5e-2,
    }
    init_params = {
        "epochs": 50,
        "coeff_iterations": 10,
        "lr_coeffs": 1e-2,
        "basis_iterations": 2,
        "lr_basis": 1e-3,
    }

    batch_size = 256
    n_timesteps_prop = problem.n_timesteps

    ###

    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"device={device}, dtype={dtype}")

    system = problem.system
    dim = system.dim()

    x0 = problem.train_initial_state_data().to(dtype=dtype)
    x_k, x_kp1 = problem.train_state_transition_data()
    x_k, x_kp1 = x_k.to(dtype=dtype), x_kp1.to(dtype=dtype)
    traj_data = [x.to(dtype=dtype) for x in problem.test_data()]

    x0_dataloader = DataLoader(
        TensorDataset(x0), batch_size=batch_size, shuffle=True, pin_memory=use_gpu
    )
    xp_dataloader = DataLoader(
        TensorDataset(x_kp1, x_k), batch_size=batch_size, shuffle=True, pin_memory=use_gpu
    )

    # Fit independent EM dictionaries to each training marginal.
    phi_means, phi_stds = init_gaussian_basis_params_gmm(
        x_k, dim, n_basis, epsilon=0.0, device=device,
    )
    psi_means, psi_stds = init_gaussian_basis_params_gmm(
        x_kp1, dim, n_basis, epsilon=0.0, device=device,
    )
    phi_basis = GaussianBasis(mean_params=phi_means, std_params=phi_stds, coeffs=None)
    psi_basis = GaussianBasis(mean_params=psi_means, std_params=psi_stds, coeffs=None)

    g_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis), mean=1.0, std=1.0, epsilon=1e-3
    ).to(device=device, dtype=dtype)
    h0_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis), mean=1.0, std=1.0, epsilon=1e-3
    ).to(device=device, dtype=dtype)

    # LinearRFF is normalized iff B @ 1 = 1.
    B_logits = TrainableParameters.random_init(
        shape=(1, n_basis, n_basis),
        mean=0.0,
        std=0.1,
    ).to(device=device, dtype=dtype)
    B = RowStochasticDenseMatrixParameters(B_logits)

    tran_model = LinearRFF(
        g_coeffs, phi_basis, psi_basis, B, numerical_tolerance=problem.numerical_tolerance
    )

    print("Training transition model")
    transition_trainer = Trainer(tran_model, device=device)
    transition_trainer.add_dataset(xp_dataloader, "transitions")
    transition_trainer.add_optimizer(
        torch.optim.Adam(
            param_group_iter((phi_means, phi_stds, psi_means, psi_stds)),
            lr=tran_params["lr_basis"],
        ),
        "basis",
        n_iterations=tran_params["basis_iterations"],
    )
    transition_trainer.add_optimizer(
        torch.optim.Adam(
            param_group_iter((g_coeffs, B_logits)),
            lr=tran_params["lr_weights"],
        ),
        "weights",
        n_iterations=tran_params["weights_iterations"],
    )
    transition_trainer.add_loss_function(loss.conditional_mle_loss, "mle", "transitions")
    tran_model, best_loss_tran, training_time_tran = transition_trainer.train(
        epochs=tran_params["epochs"],
        use_best="mle",
        verbose=True,
    )
    if tran_model is None:
        raise SystemExit("No stable transition checkpoint was found.")
    print("Done! \n")

    for parameter in tran_model.parameters():
        parameter.requires_grad_(False)

    psi0_means, psi0_stds = init_gaussian_basis_params_gmm(
        x0, dim, n_basis, epsilon=0.0, device=device,
    )
    psi0_basis = GaussianBasis(mean_params=psi0_means, std_params=psi0_stds, coeffs=None)
    init_model = LinearFF.from_rff(tran_model, h0_coeffs, psi0_basis).to(device)

    print("Training initial model")
    initial_trainer = Trainer(init_model, device=device)
    initial_trainer.add_dataset(x0_dataloader, "initial")

    initial_trainer.add_optimizer(
        torch.optim.Adam(
            param_group_iter((h0_coeffs,)),
            lr=init_params["lr_coeffs"],
        ),
        "coeffs",
        n_iterations=init_params["coeff_iterations"],
    )
    initial_trainer.add_optimizer(
        torch.optim.Adam(param_group_iter((psi0_means, psi0_stds)), lr=1e-3),
        "psi0_basis", n_iterations=2,
    )
    initial_trainer.add_loss_function(loss.mle_loss, "mle", "initial")
    init_model, best_loss_init, training_time_init = initial_trainer.train(
        epochs=init_params["epochs"],
        use_best="mle",
        verbose=True,
    )
    if init_model is None:
        raise SystemExit("No stable initial-density checkpoint was found.")
    print("Done! \n")

    print(
        f"Transition model loss: {best_loss_tran:.4f}, "
        f"training time: {training_time_tran:.2f} seconds"
    )
    print(
        f"Initial model loss: {best_loss_init:.4f}, "
        f"training time: {training_time_init:.2f} seconds"
    )

    n_slices = n_timesteps_prop + 1

    belief_seq = [init_model]
    for _ in range(n_timesteps_prop):
        belief_seq.append(tran_model.propagate(belief_seq[-1]))
    belief_seq = [belief.eval() for belief in belief_seq]

    ll_per_step = []
    with torch.no_grad():
        beliefs = [init_model]
        for _ in range(n_timesteps_prop):
            beliefs.append(tran_model.propagate(beliefs[-1]))
        for i, (belief, samples) in enumerate(zip(beliefs, traj_data)):
            loglik = avg_log_likelihood(belief, samples.to(device=device, dtype=dtype))
            print(f"Avg log-likelihood at time {i}: {float(loglik):.6f}")
            ll_per_step.append(float(loglik.detach().cpu()))

    out_dir = Path("figures/baseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    ll_path = out_dir / f"{problem_name}_lrff__avg_ll.png"
    plt.figure(figsize=(8, 4))
    plt.plot(ll_per_step, marker="o")
    for t, ll in enumerate(ll_per_step):
        plt.annotate(
            f"{ll:.3f}",
            (t, ll),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    plt.xlabel("timestep")
    plt.ylabel("avg log-likelihood per timestep")
    plt.title(
        f"{problem_name} — LRFF avg log-likelihood\n"
        f"tran loss={best_loss_tran:.4f}, init loss={best_loss_init:.4f}\n"
        f"n_basis={n_basis}"
    )
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(ll_path, dpi=200)
    print(f"Saved plot to {ll_path}")

    marg_path = out_dir / f"{problem_name}_lrff__marginal_comparison.png"
    fig, _ = plot_marginal_mc_vs_belief(
        problem,
        belief_seq,
        traj_data,
        n_points=50,
        scatter_kwargs={"s": 1, "alpha": 0.5},
    )
    fig.suptitle(
        f"{problem_name} — LRFF marginal comparison (MC vs belief)",
        y=1.01,
    )
    fig.savefig(marg_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved marginal comparison plot to {marg_path}")
