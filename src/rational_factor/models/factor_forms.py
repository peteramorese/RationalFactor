import torch
from copy import deepcopy
import itertools
from .basis_functions import Basis, SeparableBasis, NonnegativeBasis
from .density_model import DensityModel, ConditionalDensityModel

# Linear models #

class LinearForm(DensityModel):
    def __init__(self, basis : SeparableBasis, w_fixed : torch.Tensor = None, numerical_tolerance : float = 1e-20):
        super().__init__(basis.dim())

        self.basis = basis
        self.numerical_tolerance = numerical_tolerance

        if w_fixed is not None:
            self.register_buffer("w_fixed", w_fixed)
        else:
            self.__wu = torch.nn.Parameter(torch.ones(basis.n_basis_functions()))
    
    def get_w(self, Omega : torch.Tensor = None):
        if hasattr(self, "w_fixed"):
            return self.w_fixed

        if self.basis.normalized():
            return torch.nn.functional.softmax(self.__wu, dim=0)
        
        if Omega is None:
            Omega = self.basis.Omega1()

        w_unnormalized = torch.nn.functional.softplus(self.__wu)

        w = w_unnormalized / (Omega @ w_unnormalized + self.numerical_tolerance)
        return w
    
    def log_density(self, x : torch.Tensor):
        w = self.get_w()
        log_g_x = torch.log(self.basis(x) @ w + self.numerical_tolerance) # (n_data)

        return log_g_x
    
    def weight_params(self):
        if hasattr(self, "w_fixed"):
            return [self.w_fixed]
        else:
            return [self.__wu]
    
    def basis_params(self):
        return self.basis.parameters()


class LinearRFF(ConditionalDensityModel):
    """
    Linear Rational Factor Form

    Used for Markov transition distribution for propagation only models
    """
    def __init__(self, phi_basis : SeparableBasis, psi_basis : SeparableBasis, numerical_tolerance : float = 1e-20):
        assert phi_basis.dim() == psi_basis.dim(), "phi_basis and psi_basis must have the same dimension"
        assert isinstance(phi_basis, SeparableBasis), "phi_basis must be a SeparableBasis"
        assert isinstance(psi_basis, SeparableBasis), "psi_basis must be a SeparableBasis"
        assert isinstance(phi_basis, NonnegativeBasis), "phi_basis must be a NonnegativeBasis"
        assert isinstance(psi_basis, NonnegativeBasis), "psi_basis must be a NonnegativeBasis"
        super().__init__(phi_basis.dim(), psi_basis.dim())

        self.n_phi = phi_basis.n_basis_functions()
        self.n_psi = psi_basis.n_basis_functions()
        assert self.n_phi == self.n_psi, "Currently only supported for n_phi == n_psi"

        self.phi_basis = phi_basis
        self.psi_basis = psi_basis

        self.__au = torch.nn.Parameter(torch.ones(self.n_phi)) # g

        self.numerical_tolerance = numerical_tolerance
    
    def log_density(self, xp : torch.Tensor, *, conditioner : torch.Tensor):
        x = conditioner
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

        return log_g_xp + log_f - log_g_x

    def get_a(self):
        return torch.nn.functional.softmax(self.__au, dim=0)

    def get_b(self, a : torch.Tensor = None, Omega : torch.Tensor = None):
        if Omega is None:
            Omega = self.phi_basis.Omega2(self.psi_basis)

        if a is None:
            a = self.get_a()

        b = a / (Omega.T @ a + self.numerical_tolerance)

        return b

    def weight_params(self):
        return [self.__au]
    
    def basis_params(self):
        return itertools.chain(self.phi_basis.parameters(), self.psi_basis.parameters())


class LinearRF(ConditionalDensityModel):
    """
    Linear Rational Form

    Used for time-invariant observation distribution for filtering models
    """
    def __init__(self, xi_basis : SeparableBasis, zeta_basis : Basis, numerical_tolerance : float = 1e-20):
        assert isinstance(xi_basis, SeparableBasis), "xi_basis must be a SeparableBasis"
        assert isinstance(zeta_basis, Basis), "zeta_basis must be a Basis"
        assert isinstance(xi_basis, NonnegativeBasis), "xi_basis must be a NonnegativeBasis"
        assert isinstance(zeta_basis, NonnegativeBasis), "zeta_basis must be a NonnegativeBasis"
        super().__init__(xi_basis.dim(), zeta_basis.dim())

        self.xi_basis = xi_basis
        self.zeta_basis = zeta_basis

        assert xi_basis.n_basis_functions() == zeta_basis.n_basis_functions(), "xi_basis and zeta_basis must have the same number of basis functions"

        self.__du = torch.nn.Parameter(torch.ones(xi_basis.n_basis_functions()))

        self.numerical_tolerance = numerical_tolerance
    
    def get_d(self):
        return torch.nn.functional.softmax(self.__du, dim=0)
    
    def get_e(self, d : torch.Tensor = None):
        if d is None:
            d = self.get_d()

        if not self.zeta_basis.normalized():
            Omega = self.zeta_basis.Omega1()
            return d / (Omega + self.numerical_tolerance)
        return d

    def log_density(self, o : torch.Tensor, *, conditioner : torch.Tensor):
        x = conditioner
        xi_x = self.xi_basis(x)
        zeta_o = self.zeta_basis(o)

        d = self.get_d()
        e = self.get_e(d=d)

        log_r_x = torch.log(xi_x @ d + self.numerical_tolerance) # (n_data)
        log_l_o_x = torch.log((zeta_o * xi_x) @ e + self.numerical_tolerance) # (n_data)

        return log_l_o_x - log_r_x
    
    def weight_params(self):
        return [self.__du]
    
    def basis_params(self):
        return itertools.chain(self.xi_basis.parameters(), self.zeta_basis.parameters())


class LinearR2FF(ConditionalDensityModel):
    """
    Linear Rational Two-Factor Form 

    Used for Markov transition distribution for filtering models
    """
    def __init__(self, d : torch.Tensor, xi_basis : SeparableBasis, phi_basis : SeparableBasis, psi_basis : SeparableBasis, numerical_tolerance : float = 1e-20):
        assert phi_basis.dim() == psi_basis.dim(), "Input bases must have the same dimension"
        assert phi_basis.dim() == xi_basis.dim(), "Input bases must have the same dimension"
        assert isinstance(phi_basis, SeparableBasis), "phi_basis must be a SeparableBasis"
        assert isinstance(psi_basis, SeparableBasis), "psi_basis must be a SeparableBasis"
        assert isinstance(xi_basis, SeparableBasis), "xi_basis must be a SeparableBasis"
        assert isinstance(phi_basis, NonnegativeBasis), "phi_basis must be a NonnegativeBasis"
        assert isinstance(psi_basis, NonnegativeBasis), "psi_basis must be a NonnegativeBasis"
        assert isinstance(xi_basis, NonnegativeBasis), "xi_basis must be a NonnegativeBasis"
        super().__init__(phi_basis.dim(), psi_basis.dim())


        assert phi_basis.n_basis_functions() == psi_basis.n_basis_functions(), "phi_basis and psi_basis must have the same number of basis functions"

        self.xi_basis = xi_basis.freeze_params()
        self.phi_basis = phi_basis
        self.psi_basis = psi_basis

        self.register_buffer("d", d)
        self.__au = torch.nn.Parameter(torch.ones(phi_basis.n_basis_functions())) # g

        self.numerical_tolerance = numerical_tolerance
    
    @classmethod
    def from_rf(cls, rf : LinearRF, phi_basis : SeparableBasis, psi_basis : SeparableBasis):
        xi_basis = rf.xi_basis.freeze_params()
        d = rf.get_d().detach().clone()
        return cls(d, xi_basis, phi_basis, psi_basis, rf.numerical_tolerance)
    
    def log_density(self, xp : torch.Tensor, *, conditioner : torch.Tensor):
        x = conditioner
        phi_x = self.phi_basis(x) # (n_data, n_phi)
        xi_xp = self.xi_basis(xp)  # (n_data, n_xi)
        phi_xp = self.phi_basis(xp) # (n_data, n_phi)
        psi_xp = self.psi_basis(xp) # (n_data, n_psi)
        
        a = self.get_a()
        b = self.get_b(a=a)
        
        # Calculate g(x)
        log_g_x = torch.log(phi_x @ a + self.numerical_tolerance) # (n_data)
        log_g_xp = torch.log(phi_xp @ a + self.numerical_tolerance) # (n_data)

        # Calculate r(x')
        log_r_xp = torch.log(xi_xp @ self.d + self.numerical_tolerance) # (n_data)

        # Calculate f(x, x')
        log_f = torch.log((phi_x * psi_xp) @ b + self.numerical_tolerance) # (n_data)

        return log_r_xp + log_g_xp + log_f - log_g_x

    def get_a(self):
        return torch.nn.functional.softmax(self.__au, dim=0)

    def get_b(self, a : torch.Tensor = None, Omega : torch.Tensor = None):
        if a is None:
            a = self.get_a()

        if Omega is not None:
            denom = torch.einsum('i,j,ijk->k', self.d, a, Omega)
        else:
            denom = self.xi_basis.Omega3_contract(self.phi_basis, self.psi_basis, self.d, a)

        b = a / (denom + self.numerical_tolerance)

        return b

    def weight_params(self):
        return [self.__au]
    
    def basis_params(self):
        return itertools.chain(self.phi_basis.parameters(), self.psi_basis.parameters())


class LinearFF(DensityModel):
    """
    Linear Factor Form

    Used for belief representation for propagation only models
    """
    def __init__(self, a : torch.Tensor, phi_basis : SeparableBasis, psi0_basis : SeparableBasis, c0_fixed : torch.Tensor = None, numerical_tolerance : float = 1e-20, renormalize_c0_fixed : bool = True):
        assert phi_basis.dim() == psi0_basis.dim(), "phi_basis and psi0_basis must have the same dimension"
        assert isinstance(phi_basis, SeparableBasis), "phi_basis must be a SeparableBasis"
        assert isinstance(psi0_basis, SeparableBasis), "psi0_basis must be a SeparableBasis"
        assert isinstance(phi_basis, NonnegativeBasis), "phi_basis must be a NonnegativeBasis"
        assert isinstance(psi0_basis, NonnegativeBasis), "psi0_basis must be a NonnegativeBasis"
        assert a.shape[0] == phi_basis.n_basis_functions(), "a must have n_phi elements"
        super().__init__(phi_basis.dim())

        self.phi_basis = phi_basis.freeze_params() 
        self.psi0_basis = psi0_basis
        
        self.n_phi = phi_basis.n_basis_functions()
        self.n_psi0 = self.psi0_basis.n_basis_functions()

        self.register_buffer("a", a) 
        
        self.numerical_tolerance = numerical_tolerance
        if c0_fixed is not None:
            if renormalize_c0_fixed:
                # Renormalize for numerical stability
                Omega_0 = self.phi_basis.Omega2(self.psi0_basis)
                norm_constant = 1.0 / (a @ Omega_0 @ c0_fixed + self.numerical_tolerance)
                c0_fixed = norm_constant * c0_fixed
            self.register_buffer("c0_fixed", c0_fixed)
        else:
            self.__c0u = torch.nn.Parameter(torch.ones(self.n_psi0))
    
    @classmethod
    def from_rff(cls, rff : LinearRFF, psi0_basis : SeparableBasis):
        phi_basis = rff.phi_basis.freeze_params()
        a = rff.get_a().detach().clone()
        return cls(a, phi_basis, psi0_basis, numerical_tolerance=rff.numerical_tolerance)

    @classmethod
    def from_r2ff(cls, r2ff : LinearR2FF, psi0_basis : SeparableBasis):
        phi_basis = r2ff.phi_basis.freeze_params()
        a = r2ff.get_a().detach().clone()
        return cls(a, phi_basis, psi0_basis, numerical_tolerance=r2ff.numerical_tolerance)

    def get_c0(self, Omega_0 : torch.Tensor = None):
        if hasattr(self, "c0_fixed"):
            return self.c0_fixed

        if Omega_0 is None:
            Omega_0 = self.phi_basis.Omega2(self.psi0_basis)
        
        c0_unnormalized = torch.nn.functional.softplus(self.__c0u)
        
        norm_constant = 1.0 / (self.a @ Omega_0 @ c0_unnormalized + self.numerical_tolerance)

        return norm_constant * c0_unnormalized
        
    def log_density(self, x : torch.Tensor):
        phi_x = self.phi_basis(x) # (n_data, n_phi)
        psi0_x = self.psi0_basis(x) # (n_data, n_psi)

        c0 = self.get_c0()

        log_g_x = torch.log(phi_x @ self.a + self.numerical_tolerance) # (n_data)
        log_h0_x = torch.log(psi0_x @ c0 + self.numerical_tolerance)

        return log_g_x + log_h0_x

    def marginal(self, marginal_dims : tuple[int, ...]):
        phi_basis_copy = self.phi_basis.freeze_params()
        psi0_basis_copy = self.psi0_basis.freeze_params()
        phi_basis_copy.set_coeffs(self.a)
        psi0_basis_copy.set_coeffs(self.get_c0())
        expanded_basis_marginal = phi_basis_copy.product_basis([psi0_basis_copy]).marginal(marginal_dims)
        dtype, device = expanded_basis_marginal.param_dtype_device()
        w_fixed = torch.ones(
            expanded_basis_marginal.n_basis_functions(),
            device=device,
            dtype=dtype,
        )
        return LinearForm(expanded_basis_marginal, w_fixed=w_fixed, numerical_tolerance=self.numerical_tolerance)

    def weight_params(self):
        if hasattr(self, "c0_fixed"):
            return [self.c0_fixed]
        else:
            return [self.__c0u]
    
    def basis_params(self):
        return self.psi0_basis.parameters()


class Linear2FF(DensityModel):
    """
    Linear Two-Factor Form 

    Used for belief representation for filtering models
    """
    def __init__(self, d : torch.Tensor, xi_basis : SeparableBasis, a : torch.Tensor, phi_basis : SeparableBasis, psi0_basis : SeparableBasis, c0_fixed : torch.Tensor = None, numerical_tolerance : float = 1e-20, renormalize_c0_fixed : bool = True):
        assert phi_basis.dim() == psi0_basis.dim(), "phi_basis and psi0_basis must have the same dimension"
        assert isinstance(phi_basis, SeparableBasis), "phi_basis must be a SeparableBasis"
        assert isinstance(xi_basis, SeparableBasis), "xi_basis must be a SeparableBasis"
        assert isinstance(psi0_basis, SeparableBasis), "psi0_basis must be a SeparableBasis"
        assert isinstance(phi_basis, NonnegativeBasis), "phi_basis must be a NonnegativeBasis"
        assert isinstance(psi0_basis, NonnegativeBasis), "psi0_basis must be a NonnegativeBasis"
        assert isinstance(xi_basis, NonnegativeBasis), "xi_basis must be a NonnegativeBasis"
        assert a.shape[0] == phi_basis.n_basis_functions(), "a must have n_phi elements"
        super().__init__(phi_basis.dim())

        self.xi_basis = xi_basis.freeze_params()
        self.phi_basis = phi_basis.freeze_params()
        self.psi0_basis = psi0_basis
        
        self.register_buffer("d", d) 
        self.register_buffer("a", a) 
        
        self.numerical_tolerance = numerical_tolerance
        if c0_fixed is not None:
            if renormalize_c0_fixed:
                # Renormalize for numerical stability
                Omega3_0 = self.xi_basis.Omega3(self.phi_basis, self.psi0_basis)
                denom_vec = self.xi_basis.Omega3_contract(self.phi_basis, self.psi0_basis, self.d, self.a)
                norm_denom = denom_vec @ c0_fixed
                norm_constant = 1.0 / (norm_denom + self.numerical_tolerance)
                c0_fixed = norm_constant * c0_fixed
            self.register_buffer("c0_fixed", c0_fixed)
        else:
            self.__c0u = torch.nn.Parameter(torch.ones(psi0_basis.n_basis_functions()))
    
    @classmethod
    def from_r2ff(cls, r2ff : LinearRFF, psi0_basis : SeparableBasis):
        # g(x)
        phi_basis = r2ff.phi_basis.freeze_params()
        a = r2ff.get_a().detach().clone()

        # r(x)
        xi_basis = r2ff.xi_basis.freeze_params()
        d = r2ff.d.detach().clone()
        return cls(d, xi_basis, a, phi_basis, psi0_basis, numerical_tolerance=r2ff.numerical_tolerance)

    def get_c0(self, Omega3_0 : torch.Tensor = None):
        if hasattr(self, "c0_fixed"):
            return self.c0_fixed

        c0_unnormalized = torch.nn.functional.softplus(self.__c0u)

        if Omega3_0 is not None:
            norm_denom = torch.einsum("i,j,k,ijk->", self.d, self.a, c0_unnormalized, Omega3_0)
        else:
            denom_vec = self.xi_basis.Omega3_contract(self.phi_basis, self.psi0_basis, self.d, self.a)
            norm_denom = denom_vec @ c0_unnormalized

        norm_constant = 1.0 / (norm_denom + self.numerical_tolerance)

        return norm_constant * c0_unnormalized
        
    def log_density(self, x : torch.Tensor):
        xi_x = self.xi_basis(x) # (n_data, n_xi)
        phi_x = self.phi_basis(x) # (n_data, n_phi)
        psi0_x = self.psi0_basis(x) # (n_data, n_psi)

        c0 = self.get_c0()

        log_r_x = torch.log(xi_x @ self.d + self.numerical_tolerance) # (n_data)
        log_g_x = torch.log(phi_x @ self.a + self.numerical_tolerance) # (n_data)
        log_h0_x = torch.log(psi0_x @ c0 + self.numerical_tolerance)

        return log_r_x + log_g_x + log_h0_x

    def marginal(self, marginal_dims : tuple[int, ...]):
        xi_basis_copy = self.xi_basis.freeze_params()
        phi_basis_copy = self.phi_basis.freeze_params()
        psi0_basis_copy = self.psi0_basis.freeze_params()
        xi_basis_copy.set_coeffs(self.d)
        phi_basis_copy.set_coeffs(self.a)
        psi0_basis_copy.set_coeffs(self.get_c0())
        expanded_basis_marginal = xi_basis_copy.product_basis([phi_basis_copy, psi0_basis_copy]).marginal(marginal_dims)
        dtype, device = expanded_basis_marginal.param_dtype_device()
        w_fixed = torch.ones(
            expanded_basis_marginal.n_basis_functions(),
            device=device,
            dtype=dtype,
        )
        return LinearForm(expanded_basis_marginal, w_fixed=w_fixed, numerical_tolerance=self.numerical_tolerance)

    def weight_params(self):
        if hasattr(self, "c0_fixed"):
            return [self.c0_fixed]
        else:
            return [self.__c0u]
    
    def basis_params(self):
        return self.psi0_basis.parameters()


# Quadratic models #
    
class QuadraticRFF(ConditionalDensityModel):
    def __init__(self, phi_basis : SeparableBasis, psi_basis : SeparableBasis, numerical_tolerance : float = 1e-8):
        assert phi_basis.dim() == psi_basis.dim(), "phi_basis and psi_basis must have the same dimension"
        assert isinstance(phi_basis, SeparableBasis), "phi_basis must be a SeparableBasis"
        assert isinstance(psi_basis, SeparableBasis), "psi_basis must be a SeparableBasis"
        super().__init__(phi_basis.dim(), psi_basis.dim())

        assert phi_basis.n_basis_functions() == psi_basis.n_basis_functions(), "phi_basis and psi_basis must have the same number of basis functions"

        self.phi_basis = phi_basis
        self.psi_basis = psi_basis

        self.__LAu = torch.nn.Parameter(torch.randn(phi_basis.n_basis_functions(), phi_basis.n_basis_functions())) # g

        self.numerical_tolerance = numerical_tolerance
    
    def get_A(self):
        bounded_A = torch.tanh(self.__LAu)
        return bounded_A @ bounded_A.T
    
    def get_B(self, A : torch.Tensor = None, Omega : torch.Tensor = None):
        if Omega is None:
            Omega = self.phi_basis.Omega22(self.psi_basis)
        
        if A is None:
            A = self.get_A()

        den = torch.einsum('ij,ijkl->kl', A, Omega) 

        #print("den min max: ", den.min(), den.max())

        B = A / (den + self.numerical_tolerance)

        return B

    def log_density(self, xp : torch.Tensor, *, conditioner : torch.Tensor):
        x = conditioner
        phi_x = self.phi_basis(x) # (n_data, n_phi)
        phi_xp = self.phi_basis(xp) # (n_data, n_phi)
        psi_xp = self.psi_basis(xp) # (n_data, n_psi)
        
        A = self.get_A()
        B = self.get_B(A=A)

        log_g_x = torch.log(torch.relu(torch.einsum("pi,ij,pj->p", phi_x, A, phi_x)) + self.numerical_tolerance) # (n_data)
        log_g_xp = torch.log(torch.relu(torch.einsum("pi,ij,pj->p", phi_xp, A, phi_xp)) + self.numerical_tolerance) # (n_data)

        f_quad = torch.einsum("pi,ij,pj->p", phi_x * psi_xp, B, phi_x * psi_xp)
        log_f = torch.log(torch.relu(f_quad - self.numerical_tolerance) + self.numerical_tolerance) # (n_data)

        return log_g_xp + log_f - log_g_x
    
    def valid(self):
        return self.is_psd()
    
    def is_psd(self):
        B = self.get_B()
        #if not torch.all(torch.linalg.eigvalsh(B) > 0):
        #    print("Min eigval: ", torch.min(torch.linalg.eigvalsh(B)))
        return torch.all(torch.linalg.eigvalsh(B) > 0)

    def weight_params(self):
        return [self.__LAu]
    
    def basis_params(self):
        return itertools.chain(self.phi_basis.parameters(), self.psi_basis.parameters())


class QuadraticFF(DensityModel):
    def __init__(self, A : torch.Tensor, phi_basis : SeparableBasis, psi0_basis : SeparableBasis = None, C0_fixed : torch.Tensor = None, numerical_tolerance : float = 1e-10):
        assert phi_basis.dim() == psi0_basis.dim(), "phi_basis and psi0_basis must have the same dimension"
        assert isinstance(phi_basis, SeparableBasis), "phi_basis must be a SeparableBasis"
        assert isinstance(psi0_basis, SeparableBasis), "psi0_basis must be a SeparableBasis"
        assert A.shape[0] == phi_basis.n_basis_functions(), "A must have n_phi rows"
        assert A.shape[1] == phi_basis.n_basis_functions(), "A must have n_phi columns"
        super().__init__(phi_basis.dim())

        self.phi_basis = phi_basis.freeze_params()
        self.psi0_basis = psi0_basis
        
        self.register_buffer("A", A) 
        if C0_fixed is not None:
            self.register_buffer("C0_fixed", C0_fixed)
        else:
            self.__LC0u = torch.nn.Parameter(torch.randn(psi0_basis.n_basis_functions(), psi0_basis.n_basis_functions()))
        
        self.numerical_tolerance = numerical_tolerance

    @classmethod
    def from_rff(cls, rff : QuadraticRFF, psi0_basis : SeparableBasis = None):
        phi_basis = rff.phi_basis.freeze_params()
        A = rff.get_A().detach().clone()
        return cls(A, phi_basis, psi0_basis, numerical_tolerance=rff.numerical_tolerance)

    def get_C0(self, Omega_0 : torch.Tensor = None):
        if hasattr(self, "C0_fixed"):
            return self.C0_fixed

        if Omega_0 is None:
            Omega_0 = self.phi_basis.Omega22(self.psi0_basis)
        
        C0_unnormalized = torch.tanh(self.__LC0u) @ torch.tanh(self.__LC0u).T
        
        norm_constant = 1.0 / torch.einsum('ij,ijkl,kl->', self.A, Omega_0, C0_unnormalized)
        
        return norm_constant * C0_unnormalized
    
    def log_density(self, x : torch.Tensor):
        phi_x = self.phi_basis(x)
        psi0_x = self.psi0_basis(x) # (n_data, n_psi)

        C0 = self.get_C0()

        log_g_x = torch.log(torch.relu(torch.einsum("pi,ij,pj->p", phi_x, self.A, phi_x)) + self.numerical_tolerance) # (n_data)
        log_h0_x = torch.log(torch.relu(torch.einsum("pi,ij,pj->p", psi0_x, C0, psi0_x)) + self.numerical_tolerance)

        return log_g_x + log_h0_x

    def marginal(self, marginal_dims : tuple[int, ...]):
        phi_basis_1 = self.phi_basis.freeze_params()
        phi_basis_2 = self.phi_basis.freeze_params()
        psi0_basis_1 = self.psi0_basis.freeze_params()
        psi0_basis_2 = self.psi0_basis.freeze_params()

        phi_quad_basis = phi_basis_1.product_basis([phi_basis_2])
        psi0_quad_basis = psi0_basis_1.product_basis([psi0_basis_2])

        phi_quad_basis.set_coeffs(self.A.reshape(-1))
        psi0_quad_basis.set_coeffs(self.get_C0().reshape(-1))

        expanded_basis_marginal = phi_quad_basis.product_basis([psi0_quad_basis]).marginal(marginal_dims)
        dtype, device = expanded_basis_marginal.param_dtype_device()
        w_fixed = torch.ones(
            expanded_basis_marginal.n_basis_functions(),
            device=device,
            dtype=dtype,
        )
        return LinearForm(expanded_basis_marginal, w_fixed=w_fixed, numerical_tolerance=self.numerical_tolerance)

    def weight_params(self):
        if hasattr(self, "C0_fixed"):
            return [self.C0_fixed]
        else:
            return [self.__LC0u]
    
    def basis_params(self):
        return self.psi0_basis.parameters()