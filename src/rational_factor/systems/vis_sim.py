"""Visualizations for the six-state ControlledIcyRoadCar benchmark."""

from __future__ import annotations

import math
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection, PolyCollection

from .controlled_systems import ControlledIcyRoadCar
from .problems import ControlledProblem


def simulate_icy_road_mc(
    problem: ControlledProblem,
    controls: torch.Tensor | Sequence[Sequence[float]],
    n_particles: int = 100,
    seed: int | None = 42,
) -> torch.Tensor:
    """Return (T+1, n_particles, 6) states under an *open-loop* control sequence.

    ``controls[k] = [steering_rad, requested_longitudinal_accel_m_s2]`` is
    applied to every particle at time k. The icy patches are fixed, and
    independent Gaussian control/state disturbances are sampled per particle
    and time step.
    """
    if not isinstance(problem.system, ControlledIcyRoadCar):
        raise TypeError("Expected a ControlledProblem containing ControlledIcyRoadCar")
    if n_particles <= 0:
        raise ValueError("n_particles must be positive")
    u = torch.as_tensor(controls, dtype=torch.float32)
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError("controls must have shape (n_timesteps, 2)")
    system = problem.system
    device = system.additive_state_noise_std.device
    u = u.to(device=device, dtype=system.additive_state_noise_std.dtype)
    states = torch.empty((len(u) + 1, n_particles, 6), dtype=u.dtype, device=device)

    # fork_rng leaves the calling code's RNG unchanged on CPU.
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        if seed is not None:
            torch.manual_seed(seed)
        states[0] = problem.initial_state_sampler(n_particles).to(device=device, dtype=u.dtype)
        with torch.no_grad():
            for k in range(len(u)):
                states[k + 1] = system(states[k], u[k])
    return states.cpu()


def _car_polygons(states: np.ndarray, car_length: float, car_width: float):
    """Oriented footprint vertices and front-bumper segments for a state batch."""
    q = states[:, :2]
    heading = states[:, 2]
    forward = np.stack([np.sin(heading), np.cos(heading)], axis=-1)
    left = np.stack([np.cos(heading), -np.sin(heading)], axis=-1)
    f = forward * (0.5 * car_length)
    l = left * (0.5 * car_width)
    corners = np.stack([q + f + l, q + f - l, q - f - l, q - f + l], axis=1)
    bumpers = np.stack([q + f + l, q + f - l], axis=1)
    return corners, bumpers


def plot_icy_road_mc(
    problem: ControlledProblem,
    controls: torch.Tensor | Sequence[Sequence[float]],
    n_particles: int = 100,
    max_car_bodies: int = 80,
    ncols: int = 6,
    seed: int | None = 42,
    figsize: tuple[float, float] | None = None,
    save_path: str | None = None,
):
    """Plot a distinct full-road panel for every Monte-Carlo timestep.

    A background map shows the *nominal* road's smooth icy patches; each
    panel shows all MC car centers and up to ``max_car_bodies`` oriented
    rectangular car footprints.  A single control sequence is shared by all
    vehicles, which experience independent control/state noise over the same
    deterministic road friction map.

    Returns:
        fig: matplotlib Figure
        axes: 1D numpy array of axes (one per step, including k=0)
        trajectories: torch.Tensor of shape (T+1, n_particles, 6)

    Example:
        from rational_factor.systems.problems import OPEN_LOOP_OBSERVABLE_PROBLEMS
        from rational_factor.systems.vis_sim import plot_icy_road_mc
        problem = OPEN_LOOP_OBSERVABLE_PROBLEMS['icy_road_car']
        controls = torch.zeros(20, 2)
        controls[4:11, 0] = 0.12  # attempt to turn left across an ice patch
        fig, axes, states = plot_icy_road_mc(problem, controls, n_particles=120)
        fig.savefig('icy_road_mc.png', dpi=120)
    """
    if max_car_bodies < 0 or ncols <= 0:
        raise ValueError("max_car_bodies must be nonnegative and ncols positive")

    trajectories = simulate_icy_road_mc(problem, controls, n_particles, seed)
    system = problem.system
    n_steps = len(trajectories)
    ncols = min(ncols, n_steps)
    nrows = math.ceil(n_steps / ncols)
    if figsize is None:
        # The road is five times longer than it is wide, so tall panels help.
        figsize = (2.75 * ncols, 5.0 * nrows)
    fig, grid = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = grid.ravel()

    # Display the same deterministic road friction map used in the dynamics.
    xs = torch.linspace(0, system.road_width, 75)
    ys = torch.linspace(0, system.road_length, 350)
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    with torch.no_grad():
        mu = system.friction_at(xx, yy).cpu().numpy()
    ice_strength = np.clip((system.mu_dry - mu) / (system.mu_dry - system.mu_ice), 0, 1)
    dry_rgb = np.array([0.30, 0.32, 0.34])
    ice_rgb = np.array([0.67, 0.81, 0.92])
    road_rgb = dry_rgb + ice_strength[..., None] * (ice_rgb - dry_rgb)

    np_states = trajectories.numpy()
    for k in range(n_steps):
        ax = axes[k]
        ax.imshow(road_rgb, extent=(0, system.road_width, 0, system.road_length),
                  origin="lower", interpolation="bilinear", zorder=0)
        # lane marker + road edges; mirrored x display places x=1.5
        # in the right-hand lane when traveling up the page (+y).
        ax.plot([system.road_width * 0.5] * 2, [0, system.road_length],
                linestyle=(0, (6, 8)), lw=1.2, color="white", alpha=0.85, zorder=1)
        for road_edge in (0, system.road_width):
            ax.plot([road_edge] * 2, [0, system.road_length],
                    lw=2, color="white", alpha=0.9, zorder=1)

        frame_states = np_states[k]
        ax.scatter(frame_states[:, 0], frame_states[:, 1], s=6, c="#ffbc62",
                   alpha=0.33, edgecolors="none", zorder=2)
        if max_car_bodies:
            # Show actual oriented car rectangles rather than point particles.
            # Uniform subsampling only affects rendering, never propagation.
            chosen = np.linspace(0, n_particles - 1,
                                 min(max_car_bodies, n_particles), dtype=int)
            polys, front_edges = _car_polygons(
                frame_states[chosen], system.car_length, system.car_width
            )
            ax.add_collection(PolyCollection(polys, facecolors=(1.0, 0.56, 0.24, 0.19),
                                             edgecolors=(1.0, 0.76, 0.39, 0.66),
                                             linewidths=0.7, zorder=3))
            ax.add_collection(LineCollection(front_edges, colors=[(0.90, 0.22, 0.12, 0.62)],
                                             linewidths=1, zorder=4))
        ax.set_xlim(system.road_width, 0)
        ax.set_ylim(0, system.road_length)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"k={k}   t={k * system.dt:.2f} s", fontsize=10)
        ax.set_xlabel("road x [m]", fontsize=9)
        ax.set_ylabel("road y [m]", fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in axes[n_steps:]:
        ax.set_visible(False)
    fig.suptitle("Icy-road car: Monte Carlo body poses at every timestep\n"
                 "Pale blue = fixed ice patches; translucent orange = sampled cars", y=1.0)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig, axes[:n_steps], trajectories
