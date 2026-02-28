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

        self.__au = torch.nn.Parameter(torch.randn(self.n_phi)) # g

        self.numerical_tolerance = numerical_tolerance

    def get_a(self):
        return torch.nn.functional.softmax(self.__au)

    def get_b(self, a : torch.Tensor = None, Omega : torch.Tensor = None):
        if Omega is None:
            Omega = self.phi_basis.inner_prod_matrix(self.psi_basis)

        if a is None:
            a = self.get_a()

        b = a / (Omega.T @ a + self.numerical_tolerance)

        return b
    
    def forward(self, x : torch.Tensor, xp : torch.Tensor):    
        assert x.shape[1] == self.d, "x must have shape (n_data, d)"
        assert xp.shape[1] == self.d, "xp must have shape (n_data, d)"
        assert x.shape[0] == xp.shape[0], "x and xp must have the same number of data points"
        
        phi_x = self.phi_basis(x) # (n_data, n_phi)
        phi_xp = self.phi_basis(xp) # (n_data, n_phi)
        psi_xp = self.psi_basis(xp) # (n_data, n_psi)
        
        a = self.get_a()
        b = self.get_b(a=a)
        
        # Calculate g(x)
        log_g_x = torch.log(phi_x @ a + self.numerical_tolerance) # (n_data)
        log_g_xp = torch.log(phi_xp @ a + self.numerical_tolerance) # (n_data)

        # Calculate f(x, x')
        log_f = torch.log((phi_x * psi_xp) @ b + self.numerical_tolerance) # (n_data)

        if torch.isnan(log_g_xp + log_f - log_g_x).any():
            print("NaN in forward pass")
            print("log g(x'):", log_g_xp)
            print("log f(x, x'):", log_f)
            print("log g(x):", log_g_x)
            print("phi(x):", phi_x)
            print("phi(x'):", phi_xp)
            print("psi(x'):", psi_xp)
            print("b:", b)
            raise ValueError("NaN in forward pass")
        return torch.exp(log_g_xp + log_f - log_g_x)


class LinearFF(torch.nn.Module):
    #def __init__(self, rff : LinearRFF, psi0_basis : SeparableBasis = None):
    def __init__(self, a : torch.Tensor, phi_basis : SeparableBasis, psi0_basis : SeparableBasis, numerical_tolerance : float = 1e-10, c0_fixed : torch.Tensor = None):
        super().__init__()
        assert isinstance(phi_basis, SeparableBasis), "phi_basis must be a SeparableBasis"
        assert a.shape[0] == phi_basis.n_basis_functions(), "a must have n_phi elements"

        assert isinstance(psi0_basis, SeparableBasis), "psi0_basis must be a SeparableBasis"
        assert psi0_basis.dim() == phi_basis.dim(), "psi0_basis and phi_basis must have the same dimension"
        self.phi_basis = phi_basis 
        self.psi0_basis = psi0_basis
        
        self.n_phi = phi_basis.n_basis_functions()
        self.n_psi0 = self.psi0_basis.n_basis_functions()

        self.register_buffer("a", a) 
        if c0_fixed is not None:
            self.register_buffer("c0_fixed", c0_fixed)
        else:
            self.__c0u = torch.nn.Parameter(torch.randn(self.n_psi0))
        
        self.numerical_tolerance = numerical_tolerance
    
    @classmethod
    def from_rff(cls, rff : LinearRFF, psi0_basis : SeparableBasis = None):
        basis_type = type(rff.phi_basis)
        phi_basis = basis_type.freeze_params(rff.phi_basis)
        a = rff.get_a().detach().clone()
        return cls(a, phi_basis, psi0_basis, rff.numerical_tolerance)

    def get_c0(self, Omega0 : torch.Tensor = None):
        if hasattr(self, "c0_fixed"):
            return self.c0_fixed

        if Omega0 is None:
            Omega0 = self.phi_basis.inner_prod_matrix(self.psi0_basis)
        
        c0_unnormalized = torch.nn.functional.softplus(self.__c0u)
        
        norm_constant = 1.0 / (self.a @ Omega0 @ c0_unnormalized)
        
        return norm_constant * c0_unnormalized
        
    def forward(self, x : torch.Tensor):
        phi_x = self.phi_basis(x) # (n_data, n_phi)
        psi0_x = self.psi0_basis(x) # (n_data, n_psi)

        c0 = self.get_c0()

        log_g_x = torch.log(phi_x @ self.a + self.numerical_tolerance) # (n_data)
        log_h0_x = torch.log(psi0_x @ c0 + self.numerical_tolerance)

        if torch.isnan(log_g_x + log_h0_x).any():
            print("NaN in forward pass")
            print("log g(x):", log_g_x)
            print("log h0(x):", log_h0_x)
            print("phi(x):", phi_x)
            print("psi0(x):", psi0_x)
            print("c0:", c0)
            raise ValueError("NaN in forward pass")

        return torch.exp(log_g_x + log_h0_x)
    
    def marginal(self, marginal_dims : tuple[int, ...]):
        return LinearFF(self.a, self.phi_basis.marginal(marginal_dims), self.psi0_basis.marginal(marginal_dims), self.numerical_tolerance)
    
    
class QuadraticRFF(torch.nn.Module):
    def __init__(self, phi_basis : SeparableBasis, psi_basis : SeparableBasis, numerical_tolerance : float = 1e-6):
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

        self.__LAu = torch.nn.Parameter(torch.randn(self.n_phi, self.n_phi)) # g

        self.numerical_tolerance = numerical_tolerance
    
    def get_A(self):
        return self.__LAu.T @ self.__LAu
    
    def get_B(self, A : torch.Tensor = None, Omega : torch.Tensor = None):
        if Omega is None:
            Omega = self.phi_basis.inner_prod_tensor(self.psi_basis)
        
        if A is None:
            A = self.get_A()

        den = torch.einsum('klij,kl->ij', Omega, A) 
        B = A / (den + self.numerical_tolerance)

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

        log_g_x = torch.log(phi_x @ A @ phi_x.T + self.numerical_tolerance) # (n_data)
        log_g_xp = torch.log(phi_xp @ A @ phi_xp.T + self.numerical_tolerance) # (n_data)

        log_f = torch.log((phi_x * psi_xp) @ B @ (phi_x * psi_xp).T + self.numerical_tolerance) # (n_data)

        return torch.exp(log_g_xp + log_f - log_g_x)


class QuadraticFF(torch.nn.Module):
    def __init__(self, A : torch.Tensor, phi_basis : SeparableBasis, psi0_basis : SeparableBasis = None, numerical_tolerance : float = 1e-10, C0_fixed : torch.Tensor = None):
        super().__init__()
        assert isinstance(phi_basis, SeparableBasis), "phi_basis must be a SeparableBasis"
        assert A.shape[0] == phi_basis.n_basis_functions(), "A must have n_phi rows"
        assert A.shape[1] == phi_basis.n_basis_functions(), "A must have n_phi columns"

        assert isinstance(psi0_basis, SeparableBasis), "psi0_basis must be a SeparableBasis"
        assert psi0_basis.dim() == phi_basis.dim(), "psi0_basis and phi_basis must have the same dimension"
        self.phi_basis = phi_basis 
        self.psi0_basis = psi0_basis
        
        self.n_phi = phi_basis.n_basis_functions()
        self.n_psi0 = self.psi0_basis.n_basis_functions()

        self.register_buffer("A", A) 
        if C0_fixed is not None:
            self.register_buffer("C0_fixed", C0_fixed)
        else:
            self.__LC0u = torch.nn.Parameter(torch.randn(self.n_psi0, self.n_psi0))
        
        self.numerical_tolerance = numerical_tolerance

    @classmethod
    def from_rff(cls, rff : QuadraticRFF, psi0_basis : SeparableBasis = None):
        basis_type = type(rff.phi_basis)
        phi_basis = basis_type.freeze_params(rff.phi_basis)
        A = rff.get_A().detach().clone()
        return cls(A, phi_basis, psi0_basis, rff.numerical_tolerance)

    def get_C0(self, Omega0 : torch.Tensor = None):
        if hasattr(self, "C0_fixed"):
            return self.C0_fixed

        if Omega0 is None:
            Omega0 = self.phi_basis.inner_prod_tensor(self.psi0_basis)
        
        C0_unnormalized = self.__LC0u.T @ self.__LC0u
        
        norm_constant = 1.0 / torch.einsum('ij,ijkl,kl->', self.A, Omega0, C0_unnormalized)
        
        return norm_constant * C0_unnormalized
    
    def forward(self, x : torch.Tensor):
        phi_x = self.phi_basis(x)
        psi0_x = self.psi0_basis(x) # (n_data, n_psi)

        C0 = self.get_C0()

        log_g_x = torch.log(phi_x @ self.A @ phi_x + self.numerical_tolerance) # (n_data)
        log_h0_x = torch.log(psi0_x @ C0 @ psi0_x + self.numerical_tolerance)

        if torch.isnan(log_g_x + log_h0_x).any():
            print("NaN in forward pass")
            print("log g(x):", log_g_x)
            print("log h0(x):", log_h0_x)
            print("phi(x):", phi_x)
            print("psi0(x):", psi0_x)
            print("C0:", C0)
            raise ValueError("NaN in forward pass")

        return torch.exp(log_g_x + log_h0_x)

    def marginal(self, marginal_dims : tuple[int, ...]):
        return LinearFF(self.A, self.phi_basis.marginal(marginal_dims), self.psi0_basis.marginal(marginal_dims), self.numerical_tolerance)