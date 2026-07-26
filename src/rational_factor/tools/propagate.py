from rational_factor.models.density_model import DensityModel, ConditionalDensityModel
from rational_factor.models.composite_model import CompositeConditionalModel
from rational_factor.models.parameters import Parameters
import torch
import copy 
from rational_factor.models.factor_forms import LinearFF, LinearRFF, QuadraticFF, QuadraticRFF, Linear2FF, LinearR2FF, LinearRF

def propagate(init_belief : DensityModel, transition_model : ConditionalDensityModel, n_steps : int, device : torch.device = None):
    if device is None:
        device = next(init_belief.parameters()).device

    if isinstance(transition_model, CompositeConditionalModel):
        return propagate(init_belief, transition_model.conditional_density_model, n_steps, device)

    if isinstance(transition_model, LinearRFF):
        assert isinstance(init_belief, LinearFF), "Belief must be LinearFF for LinearRFF transition model"

        phi = copy.copy(transition_model.g)
        phi.set_coeffs_to_one()

        psi0 = copy.copy(init_belief.h)
        psi0.set_coeffs_to_one()

        Omega2_0 = phi.Omega2(psi0)
        Omega2 = phi.Omega2(transition_model.psi)

        b = transition_model.get_b(Omega2=Omega2)

        bOmega2T_0 = torch.einsum("...ji,...j->...ij", Omega2_0, b)
        bOmega2T = torch.einsum("...ji,...j->...ij", Omega2, b)

        c0_norm_constant = torch.exp(init_belief.log_norm_constant())
        c0 = c0_norm_constant * init_belief.h.coeffs()

        h0 = copy.copy(init_belief.h)
        h0.set_coeffs(Parameters(c0))
        h_seq = [h0]
        c1 = torch.einsum("...ij,...i->...j", bOmega2T_0, c0)
        h1 = copy.copy(transition_model.psi)
        h1.set_coeffs(Parameters(c1))
        h_seq.append(h1)
        for _ in range(1, n_steps):
            ck = torch.einsum("...ij,...i->...j", bOmega2T, h_seq[-1].coeffs())
            hk = copy.copy(transition_model.psi)
            hk.set_coeffs(Parameters(ck))
            h_seq.append(hk)
        
        belief_seq = [LinearFF(init_belief.g, h, numerical_tolerance=init_belief.numerical_tolerance, renormalize_h=False, register_modules=False) for h in h_seq]
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

#def propagate_with_control(init_belief : DensityModel, transition_model : ConditionalDensityModel, controls : list[torch.Tensor]):
#    if isinstance(transition_model, MLPContextLinearRFF):
#        assert isinstance(init_belief, MLPContextLinearFF), "Belief must be MLPContextLinearFF for MLPContextLinearRFF transition model"
#
#
#        g_mlp_form = transition_model.g_mlp_form # same g as in init
#
#        curr_g = g_mlp_form.instantiate(u=controls[0])
#        curr_h_inst = init_belief.h_mlp_form.instantiate(up=controls[0])
#        init_norm_constant = torch.exp(init_belief.log_norm_constant(up=controls[0]))
#
#        Omega_curr = curr_g.Omega2(curr_h_inst, ignore_coeffs=True)
#        c_curr = init_norm_constant * curr_h_inst.get_coeffs()
#        if c_curr is None:
#            raise ValueError(
#                "Initial h0 basis must have coeffs set on the template before propagation "
#                "(psi_mlp_form uses coeffs=None; h_mlp_form should provide c_curr)."
#            )
#
#        init_belief = LinearFF(curr_g, curr_h_inst, numerical_tolerance=init_belief.numerical_tolerance, renormalize_h=True)
#        belief_seq = [init_belief]
#
#        for k, u in enumerate(controls[:-1]):
#            up = controls[k + 1]
#
#            curr_g = g_mlp_form.instantiate(u=up)
#            curr_psi_inst = transition_model.psi_mlp_form.instantiate(u=u, up=up)
#            Omega_next = curr_g.Omega2(curr_psi_inst, ignore_coeffs=True)
#            b_curr = transition_model.get_b(u=u, up=up, Omega=Omega_next)
#
#            c_next = torch.einsum("i,ij,j->i", b_curr, Omega_curr, c_curr)
#            curr_h = curr_psi_inst.shallow_copy_target_module()
#            curr_h.set_coeffs(c_next)
#            belief_seq.append(LinearFF(curr_g, curr_h, numerical_tolerance=init_belief.numerical_tolerance, renormalize_h=False))
#
#            Omega_curr = Omega_next
#            c_curr = c_next
#
#
#        return belief_seq
#    
#    else:
#        raise ValueError(f"Unrecognized transition model type '{type(transition_model)}'")

def update(belief : DensityModel, observation_model : ConditionalDensityModel, observation : torch.Tensor, device : torch.device = None):
    if device is None:
        device = next(belief.parameters()).device

    if isinstance(observation_model, CompositeConditionalModel):
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
        prior = propagate(posteriors[-1], transition_model, 1)[1]
        
        if observation is not None:
            posterior = update(prior, observation_model, observation)
        else:
            posterior = prior

        priors.append(prior)
        posteriors.append(posterior)
    
    return priors, posteriors