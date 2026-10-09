from rational_factor.models.density_model import DensityModel, ConditionalDensityModel
from rational_factor.models.parameters import FixedParameters, TTVectorParameters
from rational_factor.models.structured_matrices import Matrix, as_matrix
from rational_factor.models.structured_vectors import TTVector, as_vector
from rational_factor.models.tt.nested_tt import NestedTTMatrix, NestedTTVector
from rational_factor.models.tt.nested_tt_parameters import FixedNestedTTVectorParameters
import torch
import copy 
from rational_factor.models.factor_forms import LinearFF, LinearRFF, SumProdRFF, QuadraticFF, QuadraticRFF, Linear2FF, LinearR2FF, LinearRF


def _coeff_params_from_values(values):
    """Wrap vector values as Fixed / TT / NestedTT coefficient Parameters."""
    values = as_vector(values)
    if isinstance(values, NestedTTVector):
        return FixedNestedTTVectorParameters(values)
    if isinstance(values, TTVector):
        return TTVectorParameters.from_cores(
            [FixedParameters(core.detach().clone()) for core in values.cores]
        )
    dense = values.to_dense() if hasattr(values, "to_dense") else values
    return FixedParameters(dense.detach().clone() if hasattr(dense, "detach") else dense)


def _outer_rank_one_factors(v: NestedTTVector) -> tuple[torch.Tensor, ...]:
    """Physical factors of an outer-rank-one NestedTTVector.

    Prefer tracked ``rank_one_factors`` when present.  Otherwise materialize the
    per-mode outer cores (shape ``(1, n_k, 1)``) without forming the full
    Kronecker product.
    """
    known = v.rank_one_factors
    if known is not None:
        return known
    if any(int(r) != 1 for r in v.ranks):
        raise ValueError(
            f"expected outer rank-one NestedTTVector, got ranks={tuple(v.ranks)}"
        )
    factors = []
    for core, n in zip(v.materialize_cores(), v.modes):
        if core.ndim == 3:
            factors.append(core.reshape(int(n)))
        elif core.ndim == 4:
            factors.append(core.reshape(int(core.shape[0]), int(n)))
        else:
            raise ValueError(
                f"unexpected materialized core shape {tuple(core.shape)} for mode {n}"
            )
    return tuple(factors)


def propagate(init_belief : DensityModel, transition_model : ConditionalDensityModel, n_steps : int, device : torch.device = None):
    if device is None:
        device = init_belief.dtype_device()[1]

    try:
        from normalizing_flow.composite_model import CompositeConditionalModel
    except ImportError:
        CompositeConditionalModel = ()  # type: ignore[misc, assignment]

    if CompositeConditionalModel and isinstance(transition_model, CompositeConditionalModel):
        return propagate(init_belief, transition_model.conditional_density_model, n_steps, device)

    ##### LINEAR RATIONAL FACTOR #####
    if isinstance(transition_model, LinearRFF):
        assert isinstance(init_belief, LinearFF), "Belief must be LinearFF for LinearRFF transition model"

        phi = transition_model.phi

        psi0 = copy.copy(init_belief.h)
        psi0.set_coeffs_to_one()

        Omega2_0 = phi.Omega2(psi0)
        Omega2 = phi.Omega2(transition_model.psi)

        b = transition_model.get_b(Omega2=Omega2)

        c0_norm_constant = torch.exp(init_belief.log_norm_constant())
        c0 = c0_norm_constant * init_belief.h.coeffs()

        h0 = copy.copy(init_belief.h)
        h0.set_coeffs(_coeff_params_from_values(c0))
        h_seq = [h0]
        c1 = b * Omega2_0.matvec(c0)
        h1 = copy.copy(transition_model.psi)
        h1.set_coeffs(_coeff_params_from_values(c1))
        h_seq.append(h1)
        for _ in range(1, n_steps):
            ck = b * Omega2.matvec(h_seq[-1].coeffs())
            hk = copy.copy(transition_model.psi)
            hk.set_coeffs(_coeff_params_from_values(ck))
            h_seq.append(hk)
        
        belief_seq = [LinearFF(init_belief.g, h, numerical_tolerance=init_belief.numerical_tolerance, renormalize_h=False, register_modules=False) for h in h_seq]
        return belief_seq
    
    ##### SUM PRODUCT RATIONAL FACTOR #####
    elif isinstance(transition_model, SumProdRFF):
        assert isinstance(init_belief, LinearFF), "Belief must be LinearFF for SumProdRFF transition model"

        phi = transition_model.phi

        psi0 = copy.copy(init_belief.h)
        psi0.set_coeffs_to_one()

        Omega2_0_raw = phi.Omega2(psi0)
        Omega2_raw = phi.Omega2(transition_model.psi)

        # Q = diag(a) @ B @ diag(q)^{-1}, so Q^T s = (B^T (a * s)) / q.
        B = transition_model.B()
        a = as_vector(transition_model.a())

        if isinstance(a, NestedTTVector):
            if not isinstance(B, NestedTTMatrix):
                raise TypeError(
                    "NestedTTVector coefficients require NestedTTMatrix B, got "
                    f"{type(B).__name__}"
                )
            if not isinstance(Omega2_0_raw, NestedTTMatrix) or not isinstance(
                Omega2_raw, NestedTTMatrix
            ):
                raise TypeError(
                    "NestedTT coefficients require NestedTT Gram matrices; "
                    "construct TTBasis with nested_depth matching the "
                    "coefficient depth"
                )
            Omega2_0 = Omega2_0_raw
            Omega2 = Omega2_raw
            # NestedTTMatrix has no transpose; use rev_matvec for Q^T actions.
            a_factors = _outer_rank_one_factors(a)
            q = Omega2.rev_matvec(a)
            q_factors = _outer_rank_one_factors(q)

            def _as_nested(c):
                c_vec = as_vector(c)
                if isinstance(c_vec, NestedTTVector):
                    return c_vec
                raise TypeError(
                    "NestedTT propagation expects NestedTTVector coefficients, "
                    f"got {type(c_vec).__name__}"
                )

            c0_norm_constant = torch.exp(init_belief.log_norm_constant())
            c0 = c0_norm_constant * as_vector(init_belief.h.coeffs())

            h0 = copy.copy(init_belief.h)
            h0.set_coeffs(_coeff_params_from_values(_as_nested(c0)))
            h_seq = [h0]

            def _Omega2_QT_matmul(c, _Omega2: NestedTTMatrix):
                c_vec = _as_nested(c)
                s1 = _Omega2.matvec(c_vec)
                # a * s1 via rank-one factors of a (NestedTT has no general Hadamard).
                return B.rev_matvec(s1.elementwise_multiply(a_factors)).elementwise_divide(
                    q_factors
                )

            c1 = _Omega2_QT_matmul(c0, Omega2_0)
            h1 = copy.copy(transition_model.psi)
            h1.set_coeffs(_coeff_params_from_values(c1))
            h_seq.append(h1)
            for _ in range(1, n_steps):
                ck = _Omega2_QT_matmul(h_seq[-1].coeffs(), Omega2)
                hk = copy.copy(transition_model.psi)
                hk.set_coeffs(_coeff_params_from_values(ck))
                h_seq.append(hk)
            g = transition_model.g_basis()
            belief_seq = [
                LinearFF(
                    g,
                    h,
                    numerical_tolerance=init_belief.numerical_tolerance,
                    renormalize_h=False,
                    register_modules=False,
                )
                for h in h_seq
            ]
            return belief_seq

        Omega2_0 = as_matrix(Omega2_0_raw)
        Omega2 = as_matrix(Omega2_raw)
        q = as_vector(Omega2.rev_matvec(a))

        c0_norm_constant = torch.exp(init_belief.log_norm_constant())
        c0 = c0_norm_constant * as_vector(init_belief.h.coeffs())

        h0 = copy.copy(init_belief.h)
        h0.set_coeffs(_coeff_params_from_values(c0))
        h_seq = [h0]
        
        def _Omega2_QT_matmul(c, _Omega2 : Matrix):
            c_vec = as_vector(c)
            s1 = _Omega2.matvec(c_vec)
            return B.rev_matvec(a * s1) / q
        
        c1 = _Omega2_QT_matmul(c0, Omega2_0)
        h1 = copy.copy(transition_model.psi)
        h1.set_coeffs(_coeff_params_from_values(c1))
        h_seq.append(h1)
        for _ in range(1, n_steps):
            ck = _Omega2_QT_matmul(h_seq[-1].coeffs(), Omega2)
            hk = copy.copy(transition_model.psi)
            hk.set_coeffs(_coeff_params_from_values(ck))
            h_seq.append(hk)
        g = transition_model.g_basis()
        belief_seq = [LinearFF(g, h, numerical_tolerance=init_belief.numerical_tolerance, renormalize_h=False, register_modules=False) for h in h_seq]

        return belief_seq

    ##### QUADRATIC RATIONAL FACTOR #####
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
                c0 = curr_belief.get_c0(Omega_0=Omega_0.to_dense())
                c1 = b * Omega_0.matvec(c0)

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
        device = belief.dtype_device()[1]

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