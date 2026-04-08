import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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