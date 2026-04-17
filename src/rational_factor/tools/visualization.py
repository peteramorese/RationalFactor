import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from rational_factor.systems.problems import FullyObservableProblem
from rational_factor.models.density_model import DensityModel
from particle_filter.particle_set import WeightedParticleSet

def plot_belief(ax: plt.Axes, belief : DensityModel, x_range: tuple[float, float], y_range: tuple[float, float], n_points: int = 100, contour_levels: int = 10, contourf_kwargs: dict = None):
    assert belief.dim == 2, "Belief must be 2D"
    ax.set_aspect("equal")
    x_lin = np.linspace(x_range[0], x_range[1], n_points)
    y_lin = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x_lin, y_lin)

    # Flatten grid to (n_points^2, 2) and evaluate density
    xy = np.stack([X.ravel(), Y.ravel()], axis=1)
    with torch.no_grad():
        belief.eval()
        density = belief(torch.tensor(xy, dtype=torch.get_default_dtype(), device=next(belief.parameters()).device))
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
        default_sizes = (1.0 + 10.0 * w_norm).numpy()

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
    n_timesteps = problem.n_timesteps
    if marginals_list is None:
        marginals_list = problem.plot_marginals_list
    n_marginals = len(marginals_list)
    if axes is None:
        fig, axes = plt.subplots(
        n_timesteps,
        n_marginals,
            figsize=(10, max(8 * n_timesteps, 40)),
            squeeze=False,
        )
    else:
        fig = None
        assert axes.shape == (n_timesteps, n_marginals), "Axes must be a 2D array with shape (n_timesteps, n_marginals)"
    scatter_kwargs = scatter_kwargs or {}
    traj_data = problem.test_data()
    for t in range(n_timesteps):
        for i in range(n_marginals):
            axes[t, i].scatter(traj_data[t][:, marginals_list[i][0]].detach().cpu().numpy(), traj_data[t][:, marginals_list[i][1]].detach().cpu().numpy(), **scatter_kwargs)
            axes[t, i].set_xlabel(problem.system.state_label(marginals_list[i][0]))
            axes[t, i].set_ylabel(problem.system.state_label(marginals_list[i][1]))
            axes[t, i].set_xlim(problem.plot_bounds_low[marginals_list[i][0]].item(), problem.plot_bounds_high[marginals_list[i][0]].item())
            axes[t, i].set_ylim(problem.plot_bounds_low[marginals_list[i][1]].item(), problem.plot_bounds_high[marginals_list[i][1]].item())
            axes[t, i].set_box_aspect(1)
    if fig is not None:
        fig.tight_layout()
    return fig, axes if fig is not None else axes

def plot_marginal_trajectory_comparison(problem: FullyObservableProblem, beliefs : list[DensityModel], marginals_list : list[tuple[int, int]] = None, scatter_kwargs: dict = None, axes = None):
    n_timesteps = problem.n_timesteps
    assert len(beliefs) == n_timesteps, "Beliefs must have length equal to problem.n_timesteps"

    if marginals_list is None:
        marginals_list = problem.plot_marginals_list
    n_marginals = len(marginals_list)

    if axes is None:
        fig, axes = plt.subplots(
            n_timesteps,
            n_marginals,
            figsize=(10, max(8 * n_timesteps, 40)),
            squeeze=False,
        )
    else:
        fig = None
        assert axes.shape == (n_timesteps, n_marginals), "Axes must be a 2D array with shape (n_timesteps, n_marginals)"

    # First layer: empirical test trajectory marginals.
    plot_problem_test_trajectories(
        problem=problem,
        marginals_list=marginals_list,
        scatter_kwargs=scatter_kwargs,
        axes=axes,
    )

    # Second layer: model-implied belief marginals.
    for t in range(n_timesteps):
        for i, marginal_dims in enumerate(marginals_list):
            belief_marginal = beliefs[t].marginal(marginal_dims=marginal_dims)
            x_dim, y_dim = marginal_dims
            plot_belief(
                axes[t, i],
                belief_marginal,
                x_range=(problem.plot_bounds_low[x_dim].item(), problem.plot_bounds_high[x_dim].item()),
                y_range=(problem.plot_bounds_low[y_dim].item(), problem.plot_bounds_high[y_dim].item()),
            )
            axes[t, i].set_xlabel(problem.system.state_label(x_dim))
            axes[t, i].set_ylabel(problem.system.state_label(y_dim))
            axes[t, i].set_xlim(problem.plot_bounds_low[x_dim].item(), problem.plot_bounds_high[x_dim].item())
            axes[t, i].set_ylim(problem.plot_bounds_low[y_dim].item(), problem.plot_bounds_high[y_dim].item())
            axes[t, i].set_box_aspect(1)

    if fig is not None:
        fig.tight_layout()
    return fig, axes if fig is not None else axes
