import torch
import copy
import itertools
from .basis_functions import Basis, SeparableBasis, NonnegativeBasis
from .density_model import DensityModel, ConditionalDensityModel
from .parameters import Parameters, RowStochasticMatrixParameters
from .structured_matrices import Matrix, as_matrix
from .structured_vectors import TTVector, as_vector

# Linear models #

class LinearForm(DensityModel):
    def __init__(self, w : SeparableBasis, numerical_tolerance : float = 1e-20, register_modules : bool = True):
        super().__init__(w.dim())

        self.w = w

        if register_modules:
            # Register the parameter modules to use parameters() and to() methods
            param_modules, coeff_modules = Basis.get_deduplicated_module_list([w])
            self._param_modules = torch.nn.ModuleList(param_modules)
            self._coeff_modules = torch.nn.ModuleList(coeff_modules)

        self.numerical_tolerance = numerical_tolerance

    def log_norm_constant(self, Omega : torch.Tensor = None):
        if Omega is None:
            Omega = self.w.Omega1()

        return -torch.log(torch.sum(Omega) + self.numerical_tolerance)
    
    def log_density(self, x : torch.Tensor):
        log_norm_constant = self.log_norm_constant()
        log_g_x = torch.log(self.w(x) + self.numerical_tolerance) # (n_data)

        return log_norm_constant + log_g_x
    
    def marginal(self, marginal_dims : tuple[int, ...]):
        w_marginal = self.w.marginal(marginal_dims)
        return LinearForm(w_marginal, numerical_tolerance=self.numerical_tolerance)


class LinearRFF(ConditionalDensityModel):
    """
    Linear Rational Factor Form

    Used for Markov transition distribution for propagation only models.
    """
    def __init__(self, a : Parameters, phi : Basis, psi : Basis, numerical_tolerance : float = 1e-20, register_modules : bool = True):
        assert phi.dim() == psi.dim(), "phi and psi must have the same dimension"
        assert isinstance(a, Parameters), "a must be a Parameters"
        assert isinstance(phi, Basis), "phi must be a Basis"
        assert isinstance(psi, Basis), "psi must be a Basis"
        a_vals = a()
        a_shape = a_vals.shape if hasattr(a_vals, "shape") else a_vals.size()
        assert tuple(a_shape) == (phi.batch_size(), phi.n_basis_functions()), (
            "a must have shape (batch_size, n_basis) matching phi"
        )
        super().__init__(phi.dim(), psi.dim())

        self.a = a
        self.phi = phi
        self.psi = psi
        self.numerical_tolerance = numerical_tolerance

        if register_modules:
            # Register the parameter modules to use parameters() and to() methods
            param_modules, coeff_modules = Basis.get_deduplicated_module_list([phi, psi])
            coeff_modules = list(coeff_modules)
            if a.is_module() and id(a) not in {id(c) for c in coeff_modules}:
                coeff_modules.append(a)
            self._param_modules = torch.nn.ModuleList(param_modules)
            self._coeff_modules = torch.nn.ModuleList(coeff_modules)

    def dtype_device(self):
        return self.phi.dtype_device()

    def g_basis(self) -> Basis:
        """Return a Basis view of ``g = a * phi`` (phi shallow-copied with coeffs ``a``)."""
        g = copy.copy(self.phi)
        g.set_coeffs(self.a)
        return g

    def log_density(self, xp : torch.Tensor, *, conditioner : torch.Tensor):
        x = conditioner
        tol = self.numerical_tolerance

        a = as_vector(self.a())
        phi_x = as_vector(self.phi(x))
        phi_xp = as_vector(self.phi(xp))
        log_g_x = torch.log((a * phi_x).sum() + tol)
        log_g_xp = torch.log((a * phi_xp).sum() + tol)

        psi_xp = as_vector(self.psi(xp))
        
        b = self.get_b(a=a)

        log_f = torch.log(as_vector(phi_x * psi_xp * b).sum() + tol)

        return log_g_xp + log_f - log_g_x

    def get_b(self, a : torch.Tensor = None, Omega2 : torch.Tensor = None):
        if a is None:
            a = as_vector(self.a())

        if Omega2 is None:
            Omega2 = self.phi.Omega2(self.psi)

        return a / (as_matrix(Omega2).rev_matvec(a) + self.numerical_tolerance)


class SumProdRFF(ConditionalDensityModel):
    def __init__(self, a : Parameters, phi : Basis, psi : Basis, B : Parameters,
                numerical_tolerance : float = 1e-20, register_modules : bool = True):
        assert phi.dim() == psi.dim(), "phi and psi must have the same dimension"
        assert isinstance(a, Parameters), "a must be a Parameters"
        assert isinstance(phi, Basis), "phi must be a Basis"
        assert isinstance(psi, Basis), "psi must be a Basis"
        a_vals = a()
        a_shape = tuple(a_vals.shape) if hasattr(a_vals, "shape") else tuple(a_vals.size())
        batch_size = phi.batch_size()
        n_basis = phi.n_basis_functions()
        a_ok = a_shape == (batch_size, n_basis) or (
            batch_size == 1 and a_shape == (n_basis,)
        )
        assert a_ok, (
            "a must have shape "
            f"({batch_size}, {n_basis})"
            + (f" or ({n_basis},)" if batch_size == 1 else "")
            + f", got {a_shape}"
        )
        super().__init__(phi.dim(), psi.dim())

        self.a = a
        self.phi = phi
        self.psi = psi
        
        expected = (batch_size, n_basis, n_basis)
        # Unbatched structured matrices (e.g. TTMatrix) expose shape (n, n);
        # that is accepted when the basis batch size is 1.
        expected_unbatched = (n_basis, n_basis)

        B_m = B()
        assert isinstance(B_m, RowStochasticMatrixParameters), "B must be a RowStochasticMatrixParameters"
        shape_ok = isinstance(B_m, Matrix) and (
            B_m.shape == expected
            or (batch_size == 1 and B_m.shape == expected_unbatched)
        )
        assert shape_ok, (
            f"B() must be a Matrix of shape {expected}"
            + (f" or {expected_unbatched}" if batch_size == 1 else "")
            + f", got {type(B_m).__name__} {tuple(getattr(B_m, 'shape', ()))}"
        )

        self.B = B
        self.numerical_tolerance = numerical_tolerance

        if register_modules:
            param_modules, coeff_modules = Basis.get_deduplicated_module_list([phi, psi])
            coeff_modules = list(coeff_modules)
            seen = {id(c) for c in coeff_modules}
            if a.is_module() and id(a) not in seen:
                coeff_modules.append(a)
                seen.add(id(a))
            for module in a.parameter_modules():
                if id(module) not in seen:
                    coeff_modules.append(module)
                    seen.add(id(module))
            self._param_modules = torch.nn.ModuleList(param_modules)
            self._coeff_modules = torch.nn.ModuleList(coeff_modules)
            matrix_modules = B.parameter_modules()
            self._matrix_param_modules = torch.nn.ModuleList(dict.fromkeys(matrix_modules))

    def dtype_device(self):
        return self.phi.dtype_device()

    def g_basis(self) -> Basis:
        """Return a Basis view of ``g = a * phi`` (phi shallow-copied with coeffs ``a``)."""
        g = copy.copy(self.phi)
        g.set_coeffs(self.a)
        return g

    def log_density(self, xp : torch.Tensor, *, conditioner : torch.Tensor):
        # f(x, xp) = phi(x)^T Q psi(xp) with Q = diag(a) @ B @ diag(q)^{-1},
        # q = Omega2^T a.  Applied as elementwise scales around B.matvec.
        # For TT, avoid ``q + tol`` (raises TT rank); tol is only used in logs.
        x = conditioner
        tol = self.numerical_tolerance

        a = as_vector(self.a())
        phi_x = as_vector(self.phi(x))
        phi_xp = as_vector(self.phi(xp))
        log_g_x = torch.log((a * phi_x).sum() + tol)
        log_g_xp = torch.log((a * phi_xp).sum() + tol)

        psi_xp = as_vector(self.psi(xp))

        B = self.B()
        q = as_matrix(self.phi.Omega2(self.psi)).rev_matvec(a)

        # Q @ psi = a * (B @ (psi / q))
        Q_psi_xp = a * B.matvec(psi_xp / (q + tol))
        log_f = torch.log(as_vector(phi_x * Q_psi_xp).sum() + tol)

        return log_g_xp + log_f - log_g_x


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

    def log_density(self, o : torch.Tensor, *, conditioner : torch.Tensor, **contexts : torch.Tensor):
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
    
    def log_density(self, xp : torch.Tensor, *, conditioner : torch.Tensor, **contexts : torch.Tensor):
        x = conditioner
        phi_x = self.phi_basis(x) # (n_data, n_phi)
        xi_xp = self.xi_basis(xp)  # (n_data, n_xi)
        phi_xp = self.phi_basis(xp) # (n_data, n_phi)
        psi_xp = self.psi_basis(xp) # (n_data, n_psi)
        
        a = self.get_a()
        b = self.get_b(a=a)
        d = self.get_d()
        
        # Calculate g(x)
        log_g_x = torch.log(phi_x @ a + self.numerical_tolerance) # (n_data)
        log_g_xp = torch.log(phi_xp @ a + self.numerical_tolerance) # (n_data)

        # Calculate r(x')
        log_r_xp = torch.log(xi_xp @ d + self.numerical_tolerance) # (n_data)

        # Calculate f(x, x')
        log_f = torch.log((phi_x * psi_xp) @ b + self.numerical_tolerance) # (n_data)

        return log_r_xp + log_g_xp + log_f - log_g_x

    def get_a(self):
        return torch.nn.functional.softmax(self.__au, dim=0)

    def get_b(self, a : torch.Tensor = None, d : torch.Tensor = None, Omega : torch.Tensor = None):
        if a is None:
            a = self.get_a()

        if d is None:
            d = self.get_d()

        if Omega is not None:
            denom = torch.einsum('i,j,ijk->k', d, a, Omega)
        else:
            denom = self.xi_basis.Omega3_contract(self.phi_basis, self.psi_basis, d, a)

        b = a / (denom + self.numerical_tolerance)

        return b
    
    def get_d(self):
        return self.d

    def weight_params(self):
        return [self.__au]
    
    def basis_params(self):
        return itertools.chain(self.phi_basis.parameters(), self.psi_basis.parameters())

class LinearRFandR2FF(LinearR2FF):
    """
    Linear Rational Form and Two-Factor Form 

    Used for combined Markov transition distribution and observation distribution for filtering models

    Treated as state transition conditional distribution with special observation methods
    """
    def __init__(self, xi_basis : SeparableBasis, zeta_basis : Basis, phi_basis : SeparableBasis, psi_basis : SeparableBasis, numerical_tolerance : float = 1e-20):
        assert phi_basis.dim() == psi_basis.dim(), "Input bases must have the same dimension"
        assert phi_basis.dim() == xi_basis.dim(), "Input bases must have the same dimension"
        assert isinstance(phi_basis, SeparableBasis), "phi_basis must be a SeparableBasis"
        assert isinstance(psi_basis, SeparableBasis), "psi_basis must be a SeparableBasis"
        assert isinstance(xi_basis, SeparableBasis), "xi_basis must be a SeparableBasis"
        assert isinstance(phi_basis, NonnegativeBasis), "phi_basis must be a NonnegativeBasis"
        assert isinstance(psi_basis, NonnegativeBasis), "psi_basis must be a NonnegativeBasis"
        assert isinstance(xi_basis, NonnegativeBasis), "xi_basis must be a NonnegativeBasis"
        ConditionalDensityModel.__init__(self, phi_basis.dim(), psi_basis.dim())


        assert phi_basis.n_basis_functions() == psi_basis.n_basis_functions(), "phi_basis and psi_basis must have the same number of basis functions"
        assert xi_basis.n_basis_functions() == zeta_basis.n_basis_functions(), "xi_basis and zeta_basis must have the same number of basis functions"

        self.xi_basis = xi_basis
        self.zeta_basis = zeta_basis
        self.phi_basis = phi_basis
        self.psi_basis = psi_basis

        self.__du = torch.nn.Parameter(torch.ones(xi_basis.n_basis_functions())) # d
        self._LinearR2FF__au = torch.nn.Parameter(torch.ones(phi_basis.n_basis_functions()))  # g TODO FIX THIS

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

    def log_observation_density(self, o : torch.Tensor, *, conditioner : torch.Tensor, **contexts : torch.Tensor):
        x = conditioner
        xi_x = self.xi_basis(x)
        zeta_o = self.zeta_basis(o)

        d = self.get_d()
        e = self.get_e(d=d)

        log_r_x = torch.log(xi_x @ d + self.numerical_tolerance) # (n_data)
        log_l_o_x = torch.log((zeta_o * xi_x) @ e + self.numerical_tolerance) # (n_data)

        #print("log_r_x min: ", log_r_x.min(), "max: ", log_r_x.max())
        #print("log_l_o_x min: ", log_l_o_x.min(), "max: ", log_l_o_x.max())

        if torch.abs(log_l_o_x - log_r_x).max() < 1e-8:
            _, xi_stds = self.xi_basis.means_stds()
            _, zeta_stds = self.zeta_basis.means_stds()
            print(
                "obs basis scales | "
                f"xi std min/max: {xi_stds.min().item():.3e}/{xi_stds.max().item():.3e}, "
                f"zeta std min/max: {zeta_stds.min().item():.3e}/{zeta_stds.max().item():.3e}"
            )
            print("x min: ", x.min(), "max: ", x.max())
            print("o min: ", o.min(), "max: ", o.max())

        return log_l_o_x - log_r_x

    def weight_params(self):
        return [self._LinearR2FF__au]
    
    def basis_params(self):
        return itertools.chain(self.xi_basis.parameters(), self.zeta_basis.parameters(), self.phi_basis.parameters(), self.psi_basis.parameters())
    
    def tran_weight_params(self):
        return [self._LinearR2FF__au]
    
    def obs_weight_params(self):
        return [self.__du]
    
    def tran_basis_params(self):
        return itertools.chain(self.phi_basis.parameters(), self.psi_basis.parameters())
    
    def obs_basis_params(self):
        return itertools.chain(self.xi_basis.parameters(), self.zeta_basis.parameters())

    def rf(self):
        return LinearRF(self.xi_basis, self.zeta_basis, numerical_tolerance=self.numerical_tolerance)

    def r2ff(self):
        return LinearR2FF(
            self.get_d().detach().clone(),
            self.xi_basis,
            self.phi_basis,
            self.psi_basis,
            numerical_tolerance=self.numerical_tolerance,
        )


class LinearFF(DensityModel):
    """
    Linear Factor Form

    Used for belief representation for propagation only models
    """
    def __init__(self, g : Basis, h : Basis, numerical_tolerance : float = 1e-20, renormalize_h : bool = True, register_modules : bool = True):
        assert g.dim() == h.dim(), "g and h must have the same dimension"
        assert isinstance(g, Basis), "g must be a Basis"
        assert isinstance(h, Basis), "h must be a Basis"
        super().__init__(g.dim())

        self.g = g
        self.h = h
        
        self.numerical_tolerance = numerical_tolerance
        self._renormalize_h = renormalize_h

        if register_modules:
            # Register the parameter modules to use parameters() and to() methods
            param_modules, coeff_modules = Basis.get_deduplicated_module_list([g, h])
            self._param_modules = torch.nn.ModuleList(param_modules)
            self._coeff_modules = torch.nn.ModuleList(coeff_modules)

    def dtype_device(self):
        return self.g.dtype_device()

    @classmethod
    def from_rff(cls, rff : LinearRFF | SumProdRFF, h : Basis, renormalize_h : bool = True, register_modules : bool = True):
        assert isinstance(rff, LinearRFF) or isinstance(rff, SumProdRFF), "rff must be a LinearRFF or SumProdRFF"
        return cls(rff.g_basis(), h, numerical_tolerance=rff.numerical_tolerance, renormalize_h=renormalize_h, register_modules=register_modules)

    #TODO
    #@classmethod
    #def from_r2ff(cls, r2ff : LinearR2FF | LinearRFandR2FF, psi0_basis : SeparableBasis):

    def log_norm_constant(self, Omega2 : torch.Tensor = None):
        if not self._renormalize_h:
            return 0.0
        if Omega2 is None:
            Omega2 = self.g.Omega2(self.h)
        return -torch.log(as_matrix(Omega2).sum() + self.numerical_tolerance)
        
    def log_density(self, x : torch.Tensor):
        # Use machine tiny for log floors — NOT numerical_tolerance.
        # SumProdRFF often passes a coarse tol (e.g. 1e-5) that is only meant for
        # conditional ratios; flooring log(h+tol) while Omega2 uses the true tiny h
        # makes p(x)=g h/Z appear enormously peaked (bogus large negative NLL).
        g_vals = self.g(x)
        h_vals = self.h(x)
        if isinstance(g_vals, torch.Tensor):
            g_sum = g_vals.sum(dim=-1).clamp_min(0)
            h_sum = h_vals.sum(dim=-1).clamp_min(0)
        else:
            g_sum = as_vector(g_vals).sum().clamp_min(0)
            h_sum = as_vector(h_vals).sum().clamp_min(0)
        log_eps = torch.finfo(g_sum.dtype).tiny
        log_g_x = torch.log(g_sum + log_eps)
        log_h_x = torch.log(h_sum + log_eps)

        return self.log_norm_constant() + log_g_x + log_h_x

    def marginal(self, marginal_dims : tuple[int, ...]):
        pass
        # TODO
        #g_copy = self.g.freeze_params()
        #h_copy = self.h.freeze_params()
        #expanded_basis_marginal = g_copy.product_basis([h_copy]).marginal(marginal_dims)
        #dtype, device = expanded_basis_marginal.param_dtype_device()
        #w_fixed = torch.ones(
        #    expanded_basis_marginal.n_basis_functions(),
        #    device=device,
        #    dtype=dtype,
        #)
        #return LinearForm(expanded_basis_marginal, w_fixed=w_fixed, numerical_tolerance=self.numerical_tolerance)


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
                denom_vec = self.xi_basis.Omega3_contract(self.phi_basis, self.psi0_basis, self.d, self.a)
                norm_denom = denom_vec @ c0_fixed
                norm_constant = 1.0 / (norm_denom + self.numerical_tolerance)
                c0_fixed = norm_constant * c0_fixed
            self.register_buffer("c0_fixed", c0_fixed)
        else:
            self.__c0u = torch.nn.Parameter(torch.ones(psi0_basis.n_basis_functions()))
    
    @classmethod
    def from_r2ff(cls, r2ff : LinearR2FF | LinearRFandR2FF, psi0_basis : SeparableBasis):
        # g(x)
        phi_basis = r2ff.phi_basis.freeze_params()
        a = r2ff.get_a().detach().clone()

        # r(x)
        xi_basis = r2ff.xi_basis.freeze_params()
        d = r2ff.d.detach().clone() if isinstance(r2ff, LinearR2FF) else r2ff.get_d().detach().clone()

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
        
    def log_density(self, x : torch.Tensor, **contexts : torch.Tensor):
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

    def log_density(self, xp : torch.Tensor, *, conditioner : torch.Tensor, **contexts : torch.Tensor):
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
    
    def log_density(self, x : torch.Tensor, **contexts : torch.Tensor):
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
