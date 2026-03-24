import torch
from .density_model import LinearFF, LinearRFF, QuadraticFF, QuadraticRFF

def propagate(belief : LinearFF | QuadraticFF, transition_model : LinearRFF | QuadraticRFF, n_steps : int):
    if isinstance(transition_model, LinearRFF):
        assert isinstance(belief, LinearFF), "Belief must be LinearFF for LinearRFF transition model"

        Omega0 = belief.phi_basis.inner_prod_matrix(belief.psi0_basis)
        Omega = transition_model.phi_basis.inner_prod_matrix(transition_model.psi_basis)
        b = transition_model.get_b(Omega=Omega)
        bOmega0 = b.unsqueeze(1) * Omega0
        bOmega = b.unsqueeze(1) * Omega

        c0 = belief.get_c0(Omega0=Omega0)

        #print("bOmega eigvals: ", torch.linalg.eigvals(bOmega))

        c_seq = [c0]
        c_seq.append(bOmega0 @ c0)
        for _ in range(1, n_steps):
            c_seq.append(bOmega @ c_seq[-1])
        
        #print("C_seq:", C_seq)
        belief_seq = [LinearFF(belief.a, belief.phi_basis, transition_model.psi_basis, c0_fixed=c_seq[i + 1]) for i in range(n_steps)]
        belief_seq.insert(0, belief) # Add the initial belief
        return belief_seq
    
    elif isinstance(transition_model, QuadraticRFF):
        assert isinstance(belief, QuadraticFF), "Belief must be QuadraticFF for QuadraticRFF transition model"

        Omega0 = belief.phi_basis.inner_prod_tensor(belief.psi0_basis)
        Omega = transition_model.phi_basis.inner_prod_tensor(transition_model.psi_basis)
        B = transition_model.get_B(Omega=Omega)
        #BOmega0 = torch.einsum("ij,klij->klij", B, Omega0)
        #BOmega = torch.einsum("ij,klij->klij", B, Omega)
        BOmega0 = torch.einsum("ij,ijkl->ijkl", B, Omega0)
        BOmega = torch.einsum("ij,ijkl->ijkl", B, Omega)
        
        C0 = belief.get_C0(Omega0=Omega0)

        C_seq = [C0]
        C_seq.append(torch.einsum("ij,klij->kl", C0, BOmega0))
        for _ in range(1, n_steps):
            C_seq.append(torch.einsum("ij,klij->kl", C_seq[-1], BOmega))
        
        belief_seq = [QuadraticFF(belief.A, belief.phi_basis, transition_model.psi_basis, C0_fixed=C_seq[i + 1]) for i in range(n_steps)]
        belief_seq.insert(0, belief) # Add the initial belief
        return belief_seq