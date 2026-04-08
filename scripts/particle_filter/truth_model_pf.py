import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from particle_filter.particle_set import WeightedParticleSet
from particle_filter.propagate import propagate_and_update
from rational_factor.systems.base import SystemObservationDistribution, SystemTransitionDistribution, simulate
from rational_factor.systems.po_truth_models import PartiallyObservableVanDerPol
from rational_factor.tools.misc import make_mvnormal_init_sampler
from rational_factor.tools.visualization import plot_particle_belief




def main():
    timesteps = 5
    n_particles = 4000
    seed = 42
    dt = 0.3
    mu = 0.9

    torch.manual_seed(seed)

    process_cov = 0.1 * torch.eye(2)
    observation_cov = 0.04 * torch.eye(2)
    init_mean = torch.tensor([0.2, 0.1])
    init_cov = torch.diag(torch.tensor([0.2, 0.2]))

    truth_system = PartiallyObservableVanDerPol(
        dt=dt,
        mu=mu,
        process_covariance=process_cov,
        observation_covariance=observation_cov,
    )

    transition_dist = SystemTransitionDistribution(truth_system)
    observation_dist = SystemObservationDistribution(truth_system)
    initial_state_sampler = make_mvnormal_init_sampler(mean=init_mean, covariance=init_cov)
    initial_belief = WeightedParticleSet(
        particles=initial_state_sampler(n_particles),
        weights=torch.ones(n_particles) / n_particles,
    )

    true_states, observations = simulate(truth_system, initial_state_sampler, n_timesteps=timesteps)

    _, posteriors = propagate_and_update(
        belief=initial_belief,
        transition_model=transition_dist,
        observation_model=observation_dist,
        observations=[obs for obs in observations],
    )

    box_lows = (-5.0, -5.0)
    box_highs = (5.0, 5.0)


    fig, axes = plt.subplots(timesteps, 1, figsize=(10, max(8*timesteps, 40)))
    fig.suptitle("Beliefs at each time step")
    for i in range(timesteps):
        plot_particle_belief(
            ax=axes[i],
            belief=posteriors[i],
            x_range=(box_lows[0], box_highs[0]),
            y_range=(box_lows[1], box_highs[1]),
            scatter_kwargs={"cmap": "viridis", "alpha": 0.9},
        )
        axes[i].scatter(true_states[i, 0].item(), true_states[i, 1].item(), marker="o", s=80, c="red")
        if i > 0:
            axes[i].scatter(observations[i-1, 0].item(), observations[i-1, 1].item(), marker="o", s=80, c="blue")
        axes[i].set_title(f"Posterior at k={i}")
        axes[i].set_xlabel("x1")
        axes[i].set_ylabel("x2")
        axes[i].grid(alpha=0.25)

    fig.tight_layout()
    plt.savefig("figures/vdp_particle_posteriors.jpg", dpi=1000)


if __name__ == "__main__":
    main()
