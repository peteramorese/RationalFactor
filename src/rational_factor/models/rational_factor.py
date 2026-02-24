import torch
from copy import deepcopy
from .basis_functions import SeparableBasis

class LinearRFF(torch.nn.Module):
    def __init__(self, phi_basis : SeparableBasis, psi_basis : SeparableBasis, numerical_tolerance : float = 1e-10):
        super().__init__()

        assert isinstance(phi_basis, SeparableBasis), "phi_basis must be a SeparableBasis"
        assert isinstance(psi_basis, SeparableBasis), "psi_basis must be a SeparableBasis"

        assert phi_basis.dim() == psi_basis.dim(), "phi_basis and psi_basis must have the same dimension"
        self.d = phi_basis.dim()

        self.n_phi = phi_basis.n_basis_functions()
        self.n_psi = psi_basis.n_basis_functions()
        assert self.n_phi == self.n_psi, "Currently only supported for n_phi == n_psi"

        self.phi_basis = phi_basis
        self.psi_basis = psi_basis

        self.__Au = torch.nn.Parameter(torch.randn(self.n_phi)) # g

        self.numerical_tolerance = numerical_tolerance

    def get_A(self):
        return torch.nn.functional.softplus(self.__Au)

    def get_B(self, A : torch.Tensor = None, Omega : torch.Tensor = None):
        if Omega is None:
            Omega = self.phi_basis.inner_prod_matrix(self.psi_basis)

        if A is None:
            A = self.get_A()

        B = A / (Omega @ A.T)

        return B
    
    def forward(self, x : torch.Tensor, xp : torch.Tensor):    
        assert x.shape[1] == self.d, "x must have shape (n_data, d)"
        assert xp.shape[1] == self.d, "xp must have shape (n_data, d)"
        assert x.shape[0] == xp.shape[0], "x and xp must have the same number of data points"
        
        phi_x = self.phi_basis(x) # (n_data, n_phi)
        phi_xp = self.phi_basis(xp) # (n_data, n_phi)
        psi_xp = self.psi_basis(xp) # (n_data, n_psi)
        
        A = self.get_A()
        B = self.get_B(A=A)
        
        # Calculate g(x)
        log_g_x = torch.log(phi_x @ A + self.numerical_tolerance) # (n_data)
        log_g_xp = torch.log(phi_xp @ A + self.numerical_tolerance) # (n_data)

        # Calculate f(x, x')
        log_f = torch.log((phi_x * psi_xp) @ B + self.numerical_tolerance) # (n_data)

        return torch.exp(log_g_xp + log_f - log_g_x)
    
class LinearFF(torch.nn.Module):
    def __init__(self, rff : LinearRFF, psi0_basis : SeparableBasis = None):
        super().__init__()
        assert isinstance(rff, LinearRFF), "Must pass in a trained rational rational factor model"

        if psi0_basis is not None:
            assert isinstance(psi0_basis, SeparableBasis), "psi0_basis must be a SeparableBasis"
            assert psi0_basis.dim() == rff.d, "psi0_basis and rff must have the same dimension"
            self.psi0_basis = psi0_basis
        else:
            self.psi0_basis = deepcopy(rff.psi_basis)
            
            # If using the psi basis from the rff, fix the parameters (not trainable)
            for p in self.psi0_basis.params:
                p.requires_grad = False

        basis_type = type(rff.phi_basis)
        self.phi_basis = basis_type.freeze_params(rff.phi_basis)
        
        # Disable training of phi (g factor function) basis functions
        for p in self.phi_basis.params:
            p.requires_grad = False

        self.n_phi = rff.n_phi
        self.n_psi0 = self.psi0_basis.n_basis_functions()

        self.register_buffer("A", rff.get_A().detach().clone()) 
        self.__C0u = torch.nn.Parameter(torch.randn(self.n_psi0))
        
        self.numerical_tolerance = rff.numerical_tolerance

    def get_C0(self, Omega : torch.Tensor = None):
        if Omega is None:
            Omega = self.phi_basis.inner_prod_matrix(self.psi0_basis)
        
        C0_unnormalized = torch.nn.functional.softplus(self.__C0u)
        
        norm_constant = 1.0 / (self.A.T @ Omega @ C0_unnormalized)
        
        return norm_constant * C0_unnormalized
        
    def forward(self, x : torch.Tensor):
        phi_x = self.phi_basis(x) # (n_data, n_phi)
        psi0_x = self.psi0_basis(x) # (n_data, n_psi)

        C0 = self.get_C0()

        log_g_x = torch.log(phi_x @ self.A + self.numerical_tolerance) # (n_data)
        log_h0_x = torch.log(psi0_x @ C0 + self.numerical_tolerance)

        return torch.exp(log_g_x + log_h0_x)
    
    #def marginal(self, )
    