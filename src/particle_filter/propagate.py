import torch
from particle_filter.particle_set import WeightedParticleSet
from rational_factor.models.density_model import ConditionalDensityModel

@torch.no_grad()
def propagate(belief: WeightedParticleSet, transition_model: ConditionalDensityModel, n_steps: int, copy_belief: bool = True):
    """
    Propagate the particle belief forward through p(x' | x).

    If `copy_belief` is True, this returns an updated copy and leaves `belief`
    unchanged. Otherwise, this mutates `belief` in place and also returns it.
    """
    assert n_steps >= 1, "n_steps must be >= 1"

    belief_next = belief.clone() if copy_belief else belief
    particles = belief_next.particles
    for _ in range(n_steps):
        particles = transition_model.sample(particles)
        assert particles.ndim == 2, "transition_model.sample must return shape (N, dim)"
        assert particles.shape[0] == belief_next.n_particles, \
            "transition_model.sample must return one sample per input particle"

    belief_next.particles = particles
    belief_next.normalize_weights()
    return belief_next

@torch.no_grad()
def update(
    belief: WeightedParticleSet,
    observation_model: ConditionalDensityModel,
    observation: torch.Tensor,
    copy_belief: bool = True,
):
    """
    Update particle weights using the observation likelihood p(o | x).

    If `copy_belief` is True, this returns an updated copy and leaves `belief`
    unchanged. Otherwise, this mutates `belief` in place and also returns it.
    """
    belief_next = belief.clone() if copy_belief else belief
    N = belief_next.n_particles

    if observation.ndim == 1:
        obs = observation.unsqueeze(0).expand(N, -1)
    elif observation.ndim == 2 and observation.shape[0] == 1:
        obs = observation.expand(N, -1)
    else:
        raise ValueError("observation must have shape (obs_dim,) or (1, obs_dim)")

    # likelihoods[i] = p(o | x_i)
    likelihoods = observation_model(obs, conditioner=belief_next.particles).reshape(-1)

    assert likelihoods.shape[0] == N, "observation_model must return one likelihood per particle"

    # Numerical guard
    likelihoods = torch.clamp(likelihoods, min=0.0)

    belief_next.weights = belief_next.weights * likelihoods
    belief_next.normalize_weights()
    return belief_next

def propagate_and_update(belief : WeightedParticleSet, transition_model : ConditionalDensityModel, observation_model : ConditionalDensityModel, observations : list[torch.Tensor]):
    priors = []
    posteriors = [belief]

    for observation in observations:
        
        # Propagate the previous posterior belief to get the prior for the current timestep
        prior = propagate(posteriors[-1], transition_model, 1, copy_belief=True)
        
        if observation is not None:
            posterior = update(prior, observation_model, observation, copy_belief=True)
        else:
            posterior = prior

        priors.append(prior)
        posteriors.append(posterior)
    
    return priors, posteriors