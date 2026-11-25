from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from motion_planning.controller import TrainableController
from motion_planning.mdp import NeuralActionLRFF
from motion_planning.objectives import (
    probability_of_reach_avoid,
    reach_avoid_probabilities,
)
from normalizing_flow.composite_model import (
    CompositeConditionalModel,
    CompositeDensityModel,
)
from normalizing_flow.transforms import Transforms
from rational_factor.models.basis_functions import BetaBasis
from rational_factor.models.factor_forms import LinearFF
from rational_factor.models.parameters import (
    NeuralParameters,
    PositiveParameters,
    param_group_iter,
)
from rational_factor.models.train import Trainer
from rational_factor.systems.base import sample_trajectories_without_collisions
from rational_factor.systems.problems import OPEN_LOOP_OBSERVABLE_PROBLEMS
from rational_factor.tools.set import Hyperrectangle
import rational_factor.models.loss as loss


def _action_conditional_mle_loss(model, xp, conditioner, action):
    return loss.conditional_mle_loss(model, xp, conditioner, action=action)


def transform_hyperrectangle(
    region: Hyperrectangle, transform: torch.nn.Module
) -> Hyperrectangle:
    """Push an axis-aligned box through a coordinate-wise monotone transform."""
    with torch.no_grad():
        z_low, _ = transform(region.low)
        z_high, _ = transform(region.high)
    return Hyperrectangle(z_low, z_high)


def _square_ranges(
    x_range: tuple[float, float], y_range: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_mid = 0.5 * (x_range[0] + x_range[1])
    y_mid = 0.5 * (y_range[0] + y_range[1])
    half_span = 0.5 * max(x_range[1] - x_range[0], y_range[1] - y_range[0])
    return (x_mid - half_span, x_mid + half_span), (y_mid - half_span, y_mid + half_span)


def make_hyperrectangle_obstacles(
    n_obstacles: int,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    relative_width: float = 0.06,
    seed: int | None = None,
    dtype: torch.dtype = torch.float64,
) -> Hyperrectangle:
    """Sample ``n_obstacles`` axis-aligned boxes uniformly in ``[low, high]``.

    Box centers are drawn independently and uniformly from the shrunk domain
    so each box stays inside the plot bounds. Each side length is
    ``relative_width * (high - low)`` per axis. Returns a batched
    ``Hyperrectangle``.
    """
    if n_obstacles < 1:
        raise ValueError("n_obstacles must be >= 1")
    low = torch.as_tensor(low, dtype=dtype)
    high = torch.as_tensor(high, dtype=dtype)
    if low.shape != high.shape or low.ndim != 1:
        raise ValueError("low/high must be 1D tensors of equal shape")
    if bool((low >= high).any()):
        raise ValueError("low must be strictly less than high in every coordinate")
    if not (0.0 < relative_width < 1.0):
        raise ValueError("relative_width must lie in (0, 1)")

    span = high - low
    width = relative_width * span
    half = 0.5 * width
    # Centers uniform in the interior so every box stays inside plot_bounds.
    center_low = low + half
    center_high = high - half
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    u = torch.rand((n_obstacles, low.numel()), generator=gen, dtype=dtype)
    centers = center_low + u * (center_high - center_low)
    box_low = centers - half
    box_high = centers + half
    return Hyperrectangle(box_low, box_high)


def _draw_boxes(ax, region: Hyperrectangle | None, x_dim: int, y_dim: int, **rect_kw):
    if region is None:
        return
    lows = region.low.detach().cpu().numpy()
    highs = region.high.detach().cpu().numpy()
    if lows.ndim == 1:
        lows, highs = lows[None, :], highs[None, :]
    for lo, hi in zip(lows, highs):
        ax.add_patch(
            plt.Rectangle(
                (lo[x_dim], lo[y_dim]),
                hi[x_dim] - lo[x_dim],
                hi[y_dim] - lo[y_dim],
                fill=False,
                **rect_kw,
            )
        )


def plot_marginal_mc_vs_belief(
    problem,
    beliefs,
    traj_data,
    *,
    wrap_tf=None,
    obstacles: Hyperrectangle | None = None,
    goal: Hyperrectangle | None = None,
    reach_belief: list[float] | None = None,
    reach_mc: list[float] | None = None,
    n_points: int = 50,
    contour_levels: int = 10,
    scatter_kwargs: dict | None = None,
):
    """One figure: for each 2D plot marginal, MC samples (top) vs belief density (bottom).

    Layout is ``(2 * n_marginals, n_timesteps + 1)``. Columns are timesteps.
    Belief panels evaluate ``belief.marginal_density`` on the kept 2D coordinates
    (integrating out all other dimensions). When ``wrap_tf`` (an
    ``ErfSeparableTransform``) is provided, the beliefs are assumed to live in
    the transformed unit-box space and are pulled back to original coordinates
    via the separable change-of-variables formula.
    """
    marginals_list = problem.plot_marginals_list
    if not marginals_list:
        raise ValueError("problem.plot_marginals_list is empty")

    n_slices = problem.n_timesteps + 1
    n_marginals = len(marginals_list)
    assert len(beliefs) == n_slices, "beliefs must have length n_timesteps + 1"
    assert len(traj_data) == n_slices, "traj_data must have length n_timesteps + 1"
    if reach_belief is not None:
        assert len(reach_belief) == n_slices
    if reach_mc is not None:
        assert len(reach_mc) == n_slices

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
            (problem.plot_region().low[x_dim].item(), problem.plot_region().high[x_dim].item()),
            (problem.plot_region().low[y_dim].item(), problem.plot_region().high[y_dim].item()),
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
            if samples.shape[0] > 0:
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
                xy_t = torch.tensor(xy, dtype=dtype, device=device)
                if wrap_tf is not None:
                    loc, scale = wrap_tf.loc_scale()
                    rem_idx = list(remaining)
                    loc_r = loc[rem_idx]
                    scale_r = scale[rem_idx]
                    sqrt_2 = torch.sqrt(xy_t.new_tensor(2.0))
                    u = (xy_t - loc_r) / (scale_r * sqrt_2)
                    z_r = 0.5 * (1.0 + torch.special.erf(u))
                    z_r = z_r.clamp(
                        wrap_tf.numerical_tolerance, 1.0 - wrap_tf.numerical_tolerance
                    )
                    ladj_r = (
                        -torch.log(scale_r)
                        - 0.5 * torch.log(xy_t.new_tensor(2.0 * torch.pi))
                        - u**2
                    ).sum(dim=-1)
                    density = belief.marginal_density(z_r, integrate_out)
                    density = density * torch.exp(ladj_r).reshape(density.shape)
                else:
                    density = belief.marginal_density(xy_t, integrate_out)
                Z = density.detach().cpu().numpy().reshape(X.shape)

            ax_b.contourf(X, Y, Z, levels=contour_levels)
            ax_b.contour(X, Y, Z, levels=contour_levels, colors="white", linewidths=0.5)
            ax_b.set_xlim(*x_range)
            ax_b.set_ylim(*y_range)
            ax_b.set_aspect("equal")

            _draw_boxes(
                ax_mc, obstacles, x_dim, y_dim,
                edgecolor="crimson", linewidth=1.0, linestyle="--",
            )
            _draw_boxes(
                ax_b, obstacles, x_dim, y_dim,
                edgecolor="crimson", linewidth=1.0, linestyle="--",
            )
            _draw_boxes(
                ax_mc, goal, x_dim, y_dim,
                edgecolor="limegreen", linewidth=1.5, linestyle="-",
            )
            _draw_boxes(
                ax_b, goal, x_dim, y_dim,
                edgecolor="limegreen", linewidth=1.5, linestyle="-",
            )

            if t == 0:
                ax_mc.set_ylabel(f"{pair_label}\nMC (free)")
                ax_b.set_ylabel(f"{pair_label}\nbelief (set complement)")
            if m_idx == 0:
                if reach_belief is not None and reach_mc is not None:
                    ax_mc.set_title(
                        f"t={t}\n"
                        f"P_reach belief={reach_belief[t]:.3f}\n"
                        f"P_reach MC={reach_mc[t]:.3f}"
                    )
                else:
                    ax_mc.set_title(f"t={t} (n={samples.shape[0]})")
            if m_idx == n_marginals - 1:
                ax_b.set_xlabel(problem.system.state_label(x_dim))
            ax_mc.set_xticks([])
            if t > 0:
                ax_mc.set_yticks([])
                ax_b.set_yticks([])

    fig.tight_layout()
    return fig, axes


def plot_reach_avoid_comparison(
    reach_belief: torch.Tensor | list[float],
    reach_mc: torch.Tensor | list[float],
    *,
    out_path: Path,
    title: str,
):
    """Incremental and cumulative reach-avoid probability: belief vs MC."""
    rb = torch.as_tensor(reach_belief, dtype=torch.float64).detach().cpu()
    rm = torch.as_tensor(reach_mc, dtype=torch.float64).detach().cpu()
    t = torch.arange(rb.numel())
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), squeeze=False)

    ax = axes[0, 0]
    ax.plot(t.numpy(), rb.numpy(), "o-", label="belief", color="C0")
    ax.plot(t.numpy(), rm.numpy(), "s--", label="MC", color="C1")
    ax.set_xlabel("t")
    ax.set_ylabel("first-hit P(goal)")
    ax.set_title("Per-timestep")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(t.numpy(), torch.cumsum(rb, 0).numpy(), "o-", label="belief", color="C0")
    ax.plot(t.numpy(), torch.cumsum(rm, 0).numpy(), "s--", label="MC", color="C1")
    ax.set_xlabel("t")
    ax.set_ylabel("cumulative P(reach-avoid)")
    ax.set_title("Cumulative")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved reach-avoid comparison to {out_path}")


if __name__ == "__main__":
    problem_name = "van_der_pol"
    problem = OPEN_LOOP_OBSERVABLE_PROBLEMS[problem_name]

    ###
    use_gpu = torch.cuda.is_available()
    dtype = torch.float64
    n_basis = 80
    hidden_features = 64
    num_hidden_layers = 2
    action_features = problem.system.control_dim()

    tran_params = {
        "epochs": 40,
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
        "epochs": 40,
        "coeff_iterations": 8,
        "lr_coeffs": 1e-2,
        "basis_iterations": 2,
        "lr_basis": 1e-3,
    }
    ctrl_params = {
        "epochs": 200,
        "lr": 5e-1,
    }

    batch_size = 128
    n_timesteps_prop = problem.n_timesteps

    ###

    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"device={device}, dtype={dtype}, n_basis={n_basis}")

    system = problem.system
    dim = system.dim()

    obstacles = problem.obstacles
    goal = problem.goal
    n_obstacle_boxes = (
        obstacles.low.shape[0] if obstacles.is_batched else 1
    )
    print(
        f"obstacles ({n_obstacle_boxes}):\n  low={obstacles.low}\n  high={obstacles.high}"
    )
    print(f"goal:\n  low={goal.low}\n  high={goal.high}")
    print(
        f"control_limits:\n  low={problem.control_limits.low}\n  "
        f"high={problem.control_limits.high}"
    )

    u, x_k, x_kp1 = problem.train_state_transition_data()
    u = u.to(dtype=dtype)
    x_k = x_k.to(dtype=dtype)
    x_kp1 = x_kp1.to(dtype=dtype)
    x0 = problem.train_initial_state_data().to(dtype=dtype)
    print(
        f"transition data: u={tuple(u.shape)}, x={tuple(x_k.shape)}, "
        f"xp={tuple(x_kp1.shape)}; x0={tuple(x0.shape)}"
    )

    xp_dataloader = DataLoader(
        TensorDataset(x_kp1, x_k, u),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_gpu,
    )
    x0_dataloader = DataLoader(
        TensorDataset(x0), batch_size=batch_size, shuffle=True, pin_memory=use_gpu
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
        epsilon=1.0,
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

    print("Training NeuralActionLRFF (Beta + erf) transition model")
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
    tran_model, best_loss_tran, train_time_tran = trainer.train(
        epochs=tran_params["epochs"],
        use_best="mle",
        verbose=True,
    )
    if tran_model is None:
        raise SystemExit("No stable transition checkpoint was found.")
    print(f"Done. best mle={best_loss_tran:.4f}, time={train_time_tran:.1f}s")

    base_tran = tran_model.base_density
    for parameter in tran_model.parameters():
        parameter.requires_grad_(False)

    # Freeze wrap for the initial belief (same unit-box map, not trained further).
    trained_wrap_tf = Transforms.freeze(tran_model.transform).to(
        device=device, dtype=dtype
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

    print("Training initial LinearFF belief (Beta + frozen erf wrap)")
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

    # Freeze all density-model parameters before absorbing norm constant
    # (absorb_norm_constant uses register_modules=False and FixedParameters(c_scaled),
    # so freezing afterwards would leave psi0 params trainable and c_scaled attached
    # to a graph that gets freed on the first backward pass).
    init_base = init_model.base_density
    for parameter in init_model.parameters():
        parameter.requires_grad_(False)
    for parameter in tran_model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        init_base = init_base.absorb_norm_constant().eval()

    # Transform rectangular obstacles and goal into the unit-box space.
    obstacles_dev = Hyperrectangle(
        obstacles.low.to(device=device, dtype=dtype),
        obstacles.high.to(device=device, dtype=dtype),
    )
    goal_dev = Hyperrectangle(
        goal.low.to(device=device, dtype=dtype),
        goal.high.to(device=device, dtype=dtype),
    )
    obstacles_z = transform_hyperrectangle(obstacles_dev, trained_wrap_tf)
    goal_z = transform_hyperrectangle(goal_dev, trained_wrap_tf)
    forbidden_z = obstacles_z.union(goal_z)

    action_limits = Hyperrectangle(
        problem.control_limits.low.to(device=device, dtype=dtype),
        problem.control_limits.high.to(device=device, dtype=dtype),
    )

    controller = TrainableController.random_init(
        action_features=action_features,
        timesteps=n_timesteps_prop,
        mean=0.0,
        std=0.1,
        action_feature_limits=action_limits,
    ).to(device=device, dtype=dtype)

    # Sanity: only controller params should be trainable.
    assert all(not p.requires_grad for p in base_tran.parameters())
    assert all(not p.requires_grad for p in init_base.parameters())
    assert any(p.requires_grad for p in controller.parameters())

    print("Training open-loop controller (maximize reach-avoid probability)")
    ctrl_opt = torch.optim.Adam(controller.parameters(), lr=ctrl_params["lr"])
    best_pra = None
    best_ctrl_state = None
    for epoch in range(ctrl_params["epochs"]):
        ctrl_opt.zero_grad()
        pra = probability_of_reach_avoid(
            init_base, base_tran, controller, obstacles_z, goal_z
        )
        loss_ctrl = -pra
        loss_ctrl.backward()
        ctrl_opt.step()
        pra_val = float(pra.detach().cpu())
        if best_pra is None or pra_val > best_pra:
            best_pra = pra_val
            best_ctrl_state = {
                k: v.detach().clone() for k, v in controller.state_dict().items()
            }
        if epoch % 10 == 0 or epoch == ctrl_params["epochs"] - 1:
            print(f"  epoch {epoch:3d}: P_reach-avoid={pra_val:.6f}")

    if best_ctrl_state is not None:
        controller.load_state_dict(best_ctrl_state)
    print(f"Best controller P_reach-avoid (belief)={best_pra:.6f}")
    print(
        "controls=\n"
        + "\n".join(
            f"  t={t}: {controller(t).detach().cpu().numpy()}"
            for t in range(controller.n_timesteps())
        )
    )

    # --- Evaluate / visualize under the trained controller ---
    def _ctrl_k(k, ctrl=controller):
        return ctrl(k).detach().cpu()

    with torch.no_grad():
        reach_belief = reach_avoid_probabilities(
            init_base, base_tran, controller, obstacles_z, goal_z
        ).detach().cpu()

        belief_seq = [init_base]
        for t in range(n_timesteps_prop):
            belief_seq.append(
                base_tran.propagate_set_complement(
                    belief_seq[-1], controller(t), forbidden_z
                )
            )
        belief_seq = [b.eval() for b in belief_seq]

    reach_mc = problem.test_reach_avoid_probabilities(_ctrl_k).cpu()
    traj_free = [
        x.to(dtype=dtype)
        for x in sample_trajectories_without_collisions(
            system,
            problem.initial_state_sampler,
            obstacles.union(goal),
            n_timesteps=n_timesteps_prop,
            n_trajectories=problem.n_trajectories_test,
            controller=_ctrl_k,
        )
    ]

    print("Reach-avoid probabilities (incremental first-hit goal mass):")
    for t in range(n_timesteps_prop + 1):
        print(
            f"  t={t}: belief={float(reach_belief[t]):.6f}  "
            f"MC={float(reach_mc[t]):.6f}"
        )
    print(
        f"Total P_reach-avoid: belief={float(reach_belief.sum()):.6f}  "
        f"MC={float(reach_mc.sum()):.6f}"
    )

    out_dir = Path("figures/obstacles")
    out_dir.mkdir(parents=True, exist_ok=True)

    cum_belief = torch.cumsum(reach_belief, 0).tolist()
    cum_mc = torch.cumsum(reach_mc, 0).tolist()

    marg_path = out_dir / f"{problem_name}_lrff_beta_reach_avoid_marginals.png"
    fig, _ = plot_marginal_mc_vs_belief(
        problem,
        belief_seq,
        traj_free,
        wrap_tf=trained_wrap_tf,
        obstacles=obstacles,
        goal=goal,
        reach_belief=cum_belief,
        reach_mc=cum_mc,
        n_points=50,
        scatter_kwargs={"s": 1, "alpha": 0.5},
    )
    fig.suptitle(
        f"{problem_name} — NeuralActionLRFF (Beta + erf) reach-avoid set-complement "
        f"(goal ∪ obstacles)\n"
        f"n_obstacles={n_obstacle_boxes}, P_ra belief={float(reach_belief.sum()):.3f}, "
        f"MC={float(reach_mc.sum()):.3f}",
        y=1.04,
    )
    fig.savefig(marg_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved marginal comparison plot to {marg_path}")

    plot_reach_avoid_comparison(
        reach_belief,
        reach_mc,
        out_path=out_dir / f"{problem_name}_lrff_beta_reach_avoid_probs.png",
        title=(
            f"{problem_name} — reach-avoid probabilities (Beta + erf) "
            f"(belief vs MC under trained controller)"
        ),
    )
