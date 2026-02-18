import torch
from .BasisFunctions import SeparableBasis

class LinearRFF(torch.nn.Module):
    def __init__(self, phi_basis : SeparableBasis, psi_basis : SeparableBasis, numerical_tolerance : float = 1e-10):
        super().__init__()

        assert isinstance(phi_basis, SeparableBasis), "phi_basis must be a SeparableBasis"
        assert isinstance(psi_basis, SeparableBasis), "psi_basis must be a SeparableBasis"

        assert phi_basis.d == psi_basis.d, "phi_basis and psi_basis must have the same dimension"
        self.d = phi_basis.d

        self.n_phi = phi_basis.n_basis_functions()
        self.n_psi = psi_basis.n_basis_functions()
        assert self.n_phi == self.n_psi, "Currently only supported for n_phi == n_psi"

        self.phi_basis = phi_basis
        self.psi_basis = psi_basis

        self.A = torch.nn.Parameter(torch.randn(self.n_phi)) # f
        self.B = torch.nn.Parameter(torch.randn(self.n_psi)) # g

        self.numerical_tolerance = numerical_tolerance
    
    def forward(self, x : torch.Tensor, xp : torch.Tensor):    
        assert x.shape[1] == self.d, "x must have shape (n_data, d)"
        assert xp.shape[1] == self.d, "xp must have shape (n_data, d)"
        assert x.shape[0] == xp.shape[0], "x and xp must have the same number of data points"
        
        phi_x = self.phi_basis(x) # (n_data, n_phi)
        phi_xp = self.phi_basis(xp) # (n_data, n_phi)
        psi_xp = self.psi_basis(xp) # (n_data, n_psi)
        
        # Calculate g(x)
        log_g_x = torch.log(self.B @ phi_x + self.numerical_tolerance) # (n_data)
        log_g_xp = torch.log(self.B @ phi_xp + self.numerical_tolerance) # (n_data)

        # Calculate f(x, x')
        log_f = torch.log(self.A @ (phi_x * psi_xp) + self.numerical_tolerance) # (n_data)

        return torch.exp(log_g_xp + log_f - log_g_x)
    
class LinearFF(torch.nn.Module):
    def __init__(self, rff : LinearRFF, psi_basis : SeparableBasis = None):
        super().__init__()
        assert isinstance(rff, LinearRFF), "Must pass in a trained rational rational factor model"

        if psi_basis is not None:
            assert isinstance(psi_basis, SeparableBasis), "psi_basis must be a SeparableBasis"
            assert psi_basis.d == rff.d, "psi_basis and rff must have the same dimension"
            self.psi_basis = psi_basis
        else:
            self.psi_basis = rff.psi_basis

        self.n_psi = self.psi_basis.n_basis_functions()
        # FIX THIS
        self.register_buffer("B", rff.B.detach().clone()) 
        self.C = torch.nn.Parameter(torch.randn(self.n_psi))
        

        self.numerical_tolerance = rff.numerical_tolerance
        
    def forward(self, x : torch.Tensor):
        psi_x = self.psi_basis(x) # (n_data, n_psi)

        log_g_x = torch.log(rff.B @ phi_x + self.numerical_tolerance) # (n_data)
        log_h_x = torch.log(self.C @ psi_x + self.numerical_tolerance)


        return self.C @ psi_x * psi_xp
    