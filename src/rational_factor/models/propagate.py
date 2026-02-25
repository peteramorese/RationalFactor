import torch
from .rational_factor import LinearFF, LinearRFF

def propagate(belief : LinearFF, transition_model : LinearRFF, n_steps : int):
    Omega0 = belief.phi_basis.inner_prod_matrix(belief.psi0_basis)
    Omega = transition_model.phi_basis.inner_prod_matrix(transition_model.psi_basis)
    B = transition_model.get_B(Omega=Omega)
    BOmega0 = B.unsqueeze(1) * Omega0
    BOmega = B.unsqueeze(1) * Omega

    C0 = belief.get_C0(Omega0=Omega0)

    C_seq = [C0]
    C_seq.append(BOmega0 @ C0)
    for _ in range(1, n_steps):
        C_seq.append(BOmega @ C_seq[-1])
    
    #print("C_seq:", C_seq)
    belief_seq = [LinearFF(belief.A, belief.phi_basis, transition_model.psi_basis, C0_fixed=C_seq[i + 1]) for i in range(n_steps)]
    belief_seq.insert(0, belief) # Add the initial belief
    return belief_seq