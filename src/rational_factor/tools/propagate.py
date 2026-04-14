from rational_factor.models.density_model import DensityModel, ConditionalDensityModel
import torch
from copy import deepcopy
from rational_factor.models.factor_forms import LinearFF, LinearRFF, QuadraticFF, QuadraticRFF, Linear2FF, LinearR2FF, LinearRF

def propagate(init_belief : DensityModel, transition_model : ConditionalDensityModel, n_steps : int, device : torch.device = None):
    if device is None:
        device = next(init_belief.parameters()).device

    if isinstance(transition_model, LinearRFF):
        assert isinstance(init_belief, LinearFF), "Belief must be LinearFF for LinearRFF transition model"

        Omega_0 = init_belief.phi_basis.Omega2(init_belief.psi0_basis)
        Omega = transition_model.phi_basis.Omega2(transition_model.psi_basis)
        b = transition_model.get_b(Omega=Omega)
        bOmega_0 = b.unsqueeze(1) * Omega_0
        bOmega = b.unsqueeze(1) * Omega

        c0 = init_belief.get_c0(Omega_0=Omega_0)

        c_seq = [c0]
        c_seq.append(bOmega_0 @ c0)
        for _ in range(1, n_steps):
            c_seq.append(bOmega @ c_seq[-1])
        
        #print("C_seq:", C_seq)
        belief_seq = [LinearFF(init_belief.a, init_belief.phi_basis, transition_model.psi_basis, c0_fixed=c_seq[i + 1]).to(device=device) for i in range(n_steps)]
        belief_seq.insert(0, init_belief) # Add the initial belief
        return belief_seq
    
    elif isinstance(transition_model, QuadraticRFF):
        assert isinstance(init_belief, QuadraticFF), "Belief must be QuadraticFF for QuadraticRFF transition model"

        Omega_0 = init_belief.phi_basis.Omega22(init_belief.psi0_basis)
        Omega = transition_model.phi_basis.Omega22(transition_model.psi_basis)
        B = transition_model.get_B(Omega=Omega)
        #BOmega0 = torch.einsum("ij,klij->klij", B, Omega0)
        #BOmega = torch.einsum("ij,klij->klij", B, Omega)
        BOmega_0 = torch.einsum("ij,ijkl->ijkl", B, Omega_0)
        BOmega = torch.einsum("ij,ijkl->ijkl", B, Omega)
        
        C0 = init_belief.get_C0(Omega_0=Omega_0)

        C_seq = [C0]
        C_seq.append(torch.einsum("ij,klij->kl", C0, BOmega_0))
        for _ in range(1, n_steps):
            C_seq.append(torch.einsum("ij,klij->kl", C_seq[-1], BOmega))
        
        belief_seq = [QuadraticFF(init_belief.A, init_belief.phi_basis, transition_model.psi_basis, C0_fixed=C_seq[i + 1]).to(device=device) for i in range(n_steps)]
        belief_seq.insert(0, init_belief) # Add the initial belief
        return belief_seq

    elif isinstance(transition_model, LinearR2FF):
        def _prop(curr_belief : LinearFF | Linear2FF):
            # Compute first belief propagation
            if isinstance(curr_belief, LinearFF):
                Omega_0 = curr_belief.phi_basis.Omega2(curr_belief.psi0_basis)
                b = transition_model.get_b()
                bOmega_0 = b.unsqueeze(1) * Omega_0

                c0 = curr_belief.get_c0(Omega_0=Omega_0)

                c1 = bOmega_0 @ c0

            elif isinstance(curr_belief, Linear2FF):
                b = transition_model.get_b()
                d = curr_belief.d
                c0 = curr_belief.get_c0()

                # c1[j] = b[j] * sum_{i,k} d[i] c0[k] Omega3[i,j,k]
                contracted_j = curr_belief.xi_basis.Omega3_contract(
                    curr_belief.psi0_basis,
                    curr_belief.phi_basis,
                    d,
                    c0,
                )
                c1 = b * contracted_j

            else:
                raise ValueError(f"Unrecognized belief type '{type(curr_belief)}'")

            return Linear2FF(transition_model.d, transition_model.xi_basis, 
                transition_model.get_a(), transition_model.phi_basis, 
                transition_model.psi_basis, c0_fixed=c1, 
                numerical_tolerance=curr_belief.numerical_tolerance).to(device=device)
        belief_seq = [init_belief]
        for _ in range(0, n_steps):
            belief_seq.append(_prop(belief_seq[-1]))
        return belief_seq

    else:
        raise ValueError(f"Unrecognized transition model type '{type(transition_model)}'")


def update(belief : DensityModel, observation_model : ConditionalDensityModel, observation : torch.Tensor, device : torch.device = None):
    if device is None:
        device = next(belief.parameters()).device

    if isinstance(observation_model, LinearRF):
        assert isinstance(belief, Linear2FF), "Belief must be Linear2FF for LinearRF observation model"

        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        # Evaluate likelihood numerator to get updated coefficients d
        zeta_o = observation_model.zeta_basis(observation).squeeze(0)
        d_unnormalized = observation_model.get_e() * zeta_o

        c0_fixed = belief.c0_fixed
        denom_vec = belief.xi_basis.Omega3_contract(
            belief.phi_basis,
            belief.psi0_basis,
            d_unnormalized,
            belief.a,
        )
        norm_constant = 1.0 / (denom_vec @ c0_fixed)
        d_updated = norm_constant * d_unnormalized

        belief_posterior = Linear2FF(d_updated, 
            belief.xi_basis, 
            belief.a, 
            belief.phi_basis, 
            belief.psi0_basis, 
            c0_fixed=belief.c0_fixed, 
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
        prior = propagate(posteriors[-1], transition_model, 1)[1]
        
        if observation is not None:
            posterior = update(prior, observation_model, observation)
        else:
            posterior = prior

        priors.append(prior)
        posteriors.append(posterior)
    
    return priors, posteriors