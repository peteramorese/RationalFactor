import torch
import numpy as np
import matplotlib.pyplot as plt
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
    w_span = torch.clamp(w_max - w_min, min=1e-12)
    w_norm = (weights - w_min) / w_span
    default_sizes = (1.0 + 10.0 * w_norm).numpy()

    default_scatter_kwargs = dict(cmap="viridis", s=default_sizes, alpha=0.9)
    default_scatter_kwargs.update(scatter_kwargs)
    sc = ax.scatter(
        belief.particles[:, 0].detach().cpu().numpy(),
        belief.particles[:, 1].detach().cpu().numpy(),
        c=weights.numpy(),
        **default_scatter_kwargs,
    )
    return sc