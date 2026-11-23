from __future__ import annotations

import torch

from .controller import TrainableController
from .mdp import MarkovDecisionProcess
from rational_factor.models.density_model import DensityModel
from rational_factor.tools.set import Hyperrectangle


def _mass(value) -> torch.Tensor:
    """Reduce ``integrate`` outputs (scalar or per-box tensor) to one mass."""
    if not torch.is_tensor(value):
        return torch.as_tensor(value)
    return value.reshape(-1).sum()


def probability_of_collision(
    init_belief: DensityModel,
    mdp: MarkovDecisionProcess,
    controller: TrainableController,
    obstacles: Hyperrectangle,
) -> torch.Tensor:
    """Obstacle mass remaining after set-complement propagation under ``controller``."""
    belief = init_belief
    for t in range(controller.n_timesteps()):
        belief = mdp.propagate_set_complement(belief, controller(t), obstacles)
    return _mass(belief.integrate(obstacles))


def reach_avoid_probabilities(
    init_belief: DensityModel,
    mdp: MarkovDecisionProcess,
    controller: TrainableController,
    obstacles: Hyperrectangle,
    goal: Hyperrectangle,
) -> torch.Tensor:
    """Per-timestep first-hit goal mass under reach-avoid absorbing semantics.

    At ``t = 0`` this is ``init_belief`` mass on ``goal``. For each later step the
    current free belief is fully propagated, goal mass is recorded, and
    propagation continues from the set complement of ``goal ∪ obstacles``.
    Length is ``controller.n_timesteps() + 1``.
    """
    forbidden = obstacles.union(goal)
    probs = [_mass(init_belief.integrate(goal))]
    belief = init_belief
    for t in range(controller.n_timesteps()):
        u = controller(t)
        full = mdp.propagate(belief, u)
        probs.append(_mass(full.integrate(goal)))
        belief = mdp.propagate_set_complement(belief, u, forbidden)
    return torch.stack(probs)


def probability_of_reach_avoid(
    init_belief: DensityModel,
    mdp: MarkovDecisionProcess,
    controller: TrainableController,
    obstacles: Hyperrectangle,
    goal: Hyperrectangle,
) -> torch.Tensor:
    """Total reach-avoid probability (sum of per-timestep first-hit goal masses)."""
    return reach_avoid_probabilities(
        init_belief, mdp, controller, obstacles, goal
    ).sum()
