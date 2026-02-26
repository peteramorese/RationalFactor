import torch
import numpy as np
import matplotlib.pyplot as plt
from ..models.rational_factor import LinearFF

def plot_belief(ax: plt.Axes, belief, x_range: tuple[float, float], y_range: tuple[float, float], n_points: int = 100, contour_levels: int = 10, contourf_kwargs: dict = None):
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
