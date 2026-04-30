import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from rational_factor.systems.problems import FullyObservableProblem
from rational_factor.models.density_model import DensityModel
from particle_filter.particle_set import WeightedParticleSet
from rational_factor.tools.analysis import check_pdf_valid

def _square_ranges(x_range: tuple[float, float], y_range: tuple[float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
    x_mid = 0.5 * (x_range[0] + x_range[1])
    y_mid = 0.5 * (y_range[0] + y_range[1])
    half_span = 0.5 * max(x_range[1] - x_range[0], y_range[1] - y_range[0])
    return (x_mid - half_span, x_mid + half_span), (y_mid - half_span, y_mid + half_span)

def plot_belief(ax: plt.Axes, belief : DensityModel, x_range: tuple[float, float], y_range: tuple[float, float], n_points: int = 100, contour_levels: int = 10, contourf_kwargs: dict = None):
    assert belief.dim == 2, "Belief must be 2D"
    x_range, y_range = _square_ranges(x_range, y_range)
    ax.set_aspect("equal")
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    x_lin = np.linspace(x_range[0], x_range[1], n_points)
    y_lin = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x_lin, y_lin)

    # Flatten grid to (n_points^2, 2) and evaluate density
    xy = np.stack([X.ravel(), Y.ravel()], axis=1)
    with torch.no_grad():
        belief.eval()
        param = next(iter(belief.parameters()), None)
        buffer = next(iter(belief.buffers()), None)
        belief_device = param.device if param is not None else (buffer.device if buffer is not None else torch.device("cpu"))
        belief_dtype = param.dtype if param is not None else (buffer.dtype if buffer is not None else torch.get_default_dtype())
        density = belief(torch.tensor(xy, dtype=belief_dtype, device=belief_device))
        density = density.cpu().numpy()
    Z = density.reshape(X.shape)

    contourf_kwargs = contourf_kwargs or {}
    default_contour_kwargs = dict(colors="white", linewidths=0.5)

    ax.contourf(X, Y, Z, levels=contour_levels, **contourf_kwargs)
    ax.contour(X, Y, Z, levels=contour_levels, **default_contour_kwargs)

def plot_particle_belief(ax: plt.Axes, belief : WeightedParticleSet, x_range: tuple[float, float], y_range: tuple[float, float], scatter_kwargs: dict = None):
    ax.set_aspect("equal")
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    scatter_kwargs = scatter_kwargs or {}
    weights = belief.weights.detach().cpu()
    w_min = torch.min(weights)
    w_max = torch.max(weights)
    w_range = w_max - w_min
    # Raw weights with vmin==vmax (e.g. uniform prior) makes ScalarMappable put every point at one colormap end.
    nearly_uniform = w_range <= 1e-12 * torch.clamp(w_max, min=1e-300)
    if nearly_uniform:
        w_norm = torch.full_like(weights, 0.5)
        default_sizes = np.full(weights.shape[0], 5.5, dtype=np.float64)
    else:
        w_norm = (weights - w_min) / w_range
        default_sizes = 0.2 * (1.0 + 10.0 * w_norm).numpy()

    default_scatter_kwargs = dict(
        cmap="viridis",
        s=default_sizes,
        alpha=0.9,
        norm=Normalize(vmin=0.0, vmax=1.0),
        c=w_norm.numpy(),
    )
    default_scatter_kwargs.update(scatter_kwargs)
    sc = ax.scatter(
        belief.particles[:, 0].detach().cpu().numpy(),
        belief.particles[:, 1].detach().cpu().numpy(),
        **default_scatter_kwargs,
    )
    return sc

def plot_problem_test_trajectories(problem: FullyObservableProblem, marginals_list : list[tuple[int, int]] = None, scatter_kwargs: dict = None, axes = None):
    n_slices = problem.n_timesteps + 1  # include initial-state marginal at t=0
    if marginals_list is None:
        marginals_list = problem.plot_marginals_list
    n_marginals = len(marginals_list)
    if axes is None:
        fig, axes = plt.subplots(
        n_slices,
        n_marginals,
            figsize=(10, max(8 * n_slices, 40)),
            squeeze=False,
        )
    else:
        fig = None
        assert axes.shape == (n_slices, n_marginals), "Axes must be a 2D array with shape (n_timesteps + 1, n_marginals)"
    scatter_kwargs = scatter_kwargs or {}
    traj_data = problem.test_data()
    assert len(traj_data) == n_slices, "problem.test_data() length must equal problem.n_timesteps + 1"
    for t in range(n_slices):
        for i in range(n_marginals):
            x_dim, y_dim = marginals_list[i]
            axes[t, i].scatter(traj_data[t][:, x_dim].detach().cpu().numpy(), traj_data[t][:, y_dim].detach().cpu().numpy(), **scatter_kwargs)
            axes[t, i].set_xlabel(problem.system.state_label(x_dim))
            axes[t, i].set_ylabel(problem.system.state_label(y_dim))
            x_range, y_range = _square_ranges(
                (problem.plot_bounds_low[x_dim].item(), problem.plot_bounds_high[x_dim].item()),
                (problem.plot_bounds_low[y_dim].item(), problem.plot_bounds_high[y_dim].item()),
            )
            axes[t, i].set_xlim(*x_range)
            axes[t, i].set_ylim(*y_range)
            axes[t, i].set_aspect("equal")
    if fig is not None:
        fig.tight_layout()
    return fig, axes if fig is not None else axes

def plot_marginal_trajectory_comparison(problem: FullyObservableProblem, beliefs : list[DensityModel], marginals_list : list[tuple[int, int]] = None, scatter_kwargs: dict = None, axes = None, n_points: int = 50):
    n_slices = problem.n_timesteps + 1  # include initial belief at t=0
    assert len(beliefs) == n_slices, "Beliefs must have length equal to problem.n_timesteps + 1"

    if marginals_list is None:
        marginals_list = problem.plot_marginals_list
    n_marginals = len(marginals_list)

    if axes is None:
        fig, axes = plt.subplots(
            n_slices,
            2 * n_marginals,
            figsize=(16, max(4 * n_slices, 40)),
            squeeze=False,
        )
    else:
        fig = None
        assert axes.shape == (n_slices, 2 * n_marginals), "Axes must be a 2D array with shape (n_timesteps + 1, 2 * n_marginals)"

    traj_data = problem.test_data()
    assert len(traj_data) == n_slices, "problem.test_data() length must equal problem.n_timesteps + 1"
    scatter_kwargs = scatter_kwargs or {}
    fig_i = 0
    for t in range(n_slices):
        for i, marginal_dims in enumerate(marginals_list):
            print(f"Creating figure {fig_i + 1} of {n_slices * n_marginals}")
            fig_i += 1
            x_dim, y_dim = marginal_dims
            x_range, y_range = _square_ranges(
                (problem.plot_bounds_low[x_dim].item(), problem.plot_bounds_high[x_dim].item()),
                (problem.plot_bounds_low[y_dim].item(), problem.plot_bounds_high[y_dim].item()),
            )

            traj_ax = axes[t, 2 * i]
            belief_ax = axes[t, 2 * i + 1]

            traj_ax.scatter(
                traj_data[t][:, x_dim].detach().cpu().numpy(),
                traj_data[t][:, y_dim].detach().cpu().numpy(),
                **scatter_kwargs,
            )
            traj_ax.set_xlabel(problem.system.state_label(x_dim))
            traj_ax.set_ylabel(problem.system.state_label(y_dim))
            traj_ax.set_xlim(*x_range)
            traj_ax.set_ylim(*y_range)
            traj_ax.set_aspect("equal")

            belief_marginal = beliefs[t].marginal(marginal_dims=marginal_dims)
            plot_belief(
                belief_ax,
                belief_marginal,
                x_range=x_range,
                y_range=y_range,
                n_points=n_points,
            )
            belief_ax.set_xlabel(problem.system.state_label(x_dim))
            belief_ax.set_ylabel(problem.system.state_label(y_dim))
            belief_ax.set_xlim(*x_range)
            belief_ax.set_ylim(*y_range)

            if t == 0:
                traj_ax.set_title(f"{problem.system.state_label(x_dim)}-{problem.system.state_label(y_dim)} traj")
                belief_ax.set_title(f"{problem.system.state_label(x_dim)}-{problem.system.state_label(y_dim)} belief")

    if fig is not None:
        fig.tight_layout()
    return fig, axes if fig is not None else axes
