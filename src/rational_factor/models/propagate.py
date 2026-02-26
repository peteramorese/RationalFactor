import torch
from .rational_factor import LinearFF, LinearRFF

def propagate(belief : LinearFF, transition_model : LinearRFF, n_steps : int):
    Omega0 = belief.phi_basis.inner_prod_matrix(belief.psi0_basis)
    Omega = transition_model.phi_basis.inner_prod_matrix(transition_model.psi_basis)
    b = transition_model.get_b(Omega=Omega)
    BOmega0 = b.unsqueeze(1) * Omega0
    BOmega = b.unsqueeze(1) * Omega

    print("BOmega0: evals: ", torch.linalg.eigvals(BOmega0))
    print("BOmega: evals: ", torch.linalg.eigvals(BOmega))

    c0 = belief.get_c0(Omega0=Omega0)

    c_seq = [c0]
    c_seq.append(BOmega0 @ c0)
    for _ in range(1, n_steps):
        c_seq.append(BOmega @ c_seq[-1])
    
    #print("C_seq:", C_seq)
    belief_seq = [LinearFF(belief.a, belief.phi_basis, transition_model.psi_basis, c0_fixed=c_seq[i + 1]) for i in range(n_steps)]
    belief_seq.insert(0, belief) # Add the initial belief
    return belief_seq