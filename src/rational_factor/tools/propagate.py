"""Thin compatibility entry point for one-step Markov propagation.

Each MarkovTransitionModel now owns its one-step propagation implementation.
Use ``belief = transition_model.propagate(belief)`` recursively.
"""
from rational_factor.models.density_model import (
    DensityModel, ConditionalDensityModel, MarkovTransitionModel,
)
from rational_factor.models.factor_forms import LinearFF, Linear2FF, LinearRF
import torch


def propagate(belief: DensityModel, transition_model: MarkovTransitionModel) -> DensityModel:
    """Backward-compatible one-step alias; no type dispatch or time loop."""
    return transition_model.propagate(belief)


def update(belief : DensityModel, observation_model : ConditionalDensityModel, observation : torch.Tensor, device : torch.device = None):
    if device is None:
        device = belief.dtype_device()[1]

    try:
        from normalizing_flow.composite_model import CompositeConditionalModel
    except ImportError:
        CompositeConditionalModel = ()
    if CompositeConditionalModel and isinstance(observation_model, CompositeConditionalModel):
        return update(belief, observation_model.conditional_density_model, observation, device)

    if isinstance(observation_model, LinearRF):
        assert isinstance(belief, Linear2FF), "Belief must be Linear2FF for LinearRF observation model"

        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        # Evaluate likelihood numerator to get updated coefficients d
        zeta_o = observation_model.zeta_basis(observation).squeeze(0)
        d_unnormalized = observation_model.get_e() * zeta_o

        c_fixed = belief.c_fixed
        denom_vec = belief.xi_basis.Omega3_contract(
            belief.phi_basis,
            belief.psi0_basis,
            d_unnormalized,
            belief.a,
        )
        norm_constant = 1.0 / (denom_vec @ c_fixed)
        d_updated = norm_constant * d_unnormalized

        belief_posterior = Linear2FF(d_updated, 
            belief.xi_basis, 
            belief.a, 
            belief.phi_basis, 
            belief.psi0_basis, 
            c_fixed=belief.c_fixed, 
            numerical_tolerance=belief.numerical_tolerance).to(device=device)
        return belief_posterior
    else:
        raise ValueError(f"Unrecognized observation model type '{type(observation_model)}'")

    
def propagate_and_update(belief : DensityModel, transition_model : ConditionalDensityModel, observation_model : ConditionalDensityModel, observations : list[torch.Tensor]):
    """
    Propagate and update the belief given observation data

    Args:
        belief : LinearFF | Linear2FF starting belief (k=0)
        transition_model : LinearR2FF transition model
        observations : list[torch.Tensor] sequential observation data for timesteps k=1, ..., k=len(observations)-1. 
            If observations[k] is None, no observation is available and the belief is propagated without update
    """

    priors = []
    posteriors = [belief]

    for observation in observations:
        
        # Propagate the previous posterior belief to get the prior for the current timestep
        prior = transition_model.propagate(posteriors[-1])
        
        if observation is not None:
            posterior = update(prior, observation_model, observation)
        else:
            posterior = prior

        priors.append(prior)
        posteriors.append(posterior)
    
    return priors, posteriors