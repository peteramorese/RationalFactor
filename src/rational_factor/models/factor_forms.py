import torch
import copy
import itertools
import math
from .basis_functions import Basis, SeparableBasis, NonnegativeBasis
from .density_model import DensityModel, ConditionalDensityModel, MarkovTransitionModel
from .parameters import Parameters, FixedParameters
from .structured_matrices import Matrix, DenseMatrix, as_matrix
from .structured_vectors import as_vector
from rational_factor.models.tt.tt import TTVector


def _is_parameters_like(obj) -> bool:
    """Duck-type check for Parameters / NestedTT parameter objects."""
    return callable(obj) and hasattr(obj, "is_trainable") and hasattr(
        obj, "parameter_modules"
    )


def _fixed_vector_parameters(values, *, modes=None):
    """Wrap detached propagated coefficients, preserving structured TT storage.

    A structured Matrix.rev_matvec yields TT coefficients directly. When a
    *dense* Matrix instead yields a dense vector, tensor-product coefficient
    storage with ``modes`` requires a TT: convert it via exact TT-SVD.
    This dense fallback is not intended for exponentially large spaces.
    """
    values = as_vector(values)
    if isinstance(values, TTVector):
        from rational_factor.models.tt.tt_parameters import TTVectorParameters
        return TTVectorParameters.from_cores([
            FixedParameters(core.detach().clone()) for core in values.cores
        ])
    dense = values.to_dense().detach().clone()
    if modes is None:
        return FixedParameters(dense)
    modes = tuple(int(n) for n in modes)
    if dense.ndim != 1 or dense.numel() != math.prod(modes):
        raise ValueError("dense-to-TT coefficient conversion requires an unbatched vector")
    from rational_factor.models.tt.tt_parameters import TTVectorParameters
    cores = []
    tensor = dense.reshape(*modes)
    left_rank = 1
    for n in modes[:-1]:
        matrix = tensor.reshape(left_rank * n, -1)
        u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        rank = singular.numel()
        cores.append(FixedParameters(u.reshape(left_rank, n, rank)))
        tensor = singular[:, None] * vh
        left_rank = rank
    cores.append(FixedParameters(tensor.reshape(left_rank, modes[-1], 1)))
    return TTVectorParameters.from_cores(cores)


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

        a = as_vector(self.a(), numerical_tolerance=tol)
        phi_x = as_vector(self.phi(x), numerical_tolerance=tol)
        phi_xp = as_vector(self.phi(xp), numerical_tolerance=tol)
        psi_xp = as_vector(self.psi(xp), numerical_tolerance=tol)
        b = self.get_b(a=a)

        # Floor only the combined ratio. Per-term ``+ numerical_tolerance``
        # floors destroy cancellation when g(x) and f underflow but g(xp) does not
        # (common in high-d products of peaked Gaussians) and invent huge log-densities.
        g_x = (a * phi_x).sum().clamp_min(0)
        g_xp = (a * phi_xp).sum().clamp_min(0)
        f = as_vector(phi_x * psi_xp * b).sum().clamp_min(0)
        eps = torch.finfo(g_x.dtype).tiny
        return torch.log(g_xp * f + eps) - torch.log(g_x + eps)

    def get_b(self, a : torch.Tensor = None, Omega2 : torch.Tensor = None):
        tol = self.numerical_tolerance
        if a is None:
            a = as_vector(self.a(), numerical_tolerance=tol)
        else:
            a = as_vector(a, numerical_tolerance=tol)

        if Omega2 is None:
            Omega2 = self.phi.Omega2(self.psi)

        # Denominator floor lives on the vector's elementwise division.
        q = as_vector(as_matrix(Omega2).rev_matvec(a), numerical_tolerance=tol)
        return a / q


class SumProdRFF(MarkovTransitionModel):
    def __init__(self, a : Parameters, phi : Basis, psi : Basis, B : Parameters,
                numerical_tolerance : float = 1e-20, register_modules : bool = True):
        assert phi.dim() == psi.dim(), "phi and psi must have the same dimension"
        assert _is_parameters_like(a), "a must be a Parameters-like object"
        assert isinstance(phi, Basis), "phi must be a Basis"
        assert isinstance(psi, Basis), "psi must be a Basis"
        a_vals = a()
        a_shape = tuple(a_vals.shape) if hasattr(a_vals, "shape") else tuple(a_vals.size())
        if isinstance(a_vals, TTVector) and not a_vals.is_rank_one:
            raise ValueError("SumProdRFF requires rank-one TT coefficients a for structured q scaling")
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
        super().__init__(phi.dim())

        self.a = a
        self.phi = phi
        self.psi = psi
        
        expected = (batch_size, n_basis, n_basis)
        # Unbatched structured matrices (e.g. LowRankFactorizationMPO) expose shape (n, n);
        # that is accepted when the basis batch size is 1.
        expected_unbatched = (n_basis, n_basis)

        B_m = B()
        shape_ok = (
            B_m.shape == expected
            or (batch_size == 1 and B_m.shape == expected_unbatched)
        ) and hasattr(B_m, "matvec")
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
            if getattr(a, "is_module", lambda: False)() and id(a) not in seen:
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
        # Denominator flooring for ``psi / q`` is handled by the vector types'
        # elementwise division (via ``numerical_tolerance`` / ``eps``).
        x = conditioner
        tol = self.numerical_tolerance

        a = as_vector(self.a(), numerical_tolerance=tol)
        phi_x = as_vector(self.phi(x), numerical_tolerance=tol)
        phi_xp = as_vector(self.phi(xp), numerical_tolerance=tol)
        psi_xp = as_vector(self.psi(xp), numerical_tolerance=tol)

        B = self.B()
        Omega2 = as_matrix(self.phi.Omega2(self.psi))
        q = as_vector(Omega2.rev_matvec(a), numerical_tolerance=tol)

        # Q @ psi = a * (B @ (psi / q))
        Q_psi_xp = a * B.matvec(psi_xp / q)
        # Floor only the combined ratio. Per-term ``+ numerical_tolerance``
        # floors destroy cancellation when g(x) and f underflow but g(xp) does not
        # (common in high-d products of peaked Gaussians) and invent huge log-densities.
        g_x = (a * phi_x).sum().clamp_min(0)
        g_xp = (a * phi_xp).sum().clamp_min(0)
        f = as_vector(phi_x * Q_psi_xp).sum().clamp_min(0)
        eps = torch.finfo(g_x.dtype).tiny
        return (torch.log(g_xp * f + eps) - torch.log(g_x + eps)).reshape(-1)


    def propagate(self, belief: "LinearFF") -> "LinearFF":
        r"""Propagate a LinearFF for one Markov step using base vector/matrix APIs.

        Omega_current[i,j] = integral phi_i(x) belief.psi_j(x) dx
        q[k] = sum_i a[i] integral phi_i(x) psi_k(x) dx
        c_next = (B.T @ (a * (Omega_current @ (c/Z)))) / q

        No coefficient-bearing basis copies are needed. The first propagation
        naturally uses the initial psi0; subsequent calls use self.psi.
        The belief's g factor must coincide with the transition's a*phi.
        """
        if not isinstance(belief, LinearFF):
            raise TypeError(
                "SumProdRFF.propagate expects a LinearFF belief, got "
                f"{type(belief).__name__}"
            )

        tol = self.numerical_tolerance
        a = as_vector(self.a(), numerical_tolerance=tol)
        B = as_matrix(self.B())
        omega_current = as_matrix(self.phi.Omega2(belief.psi))
        omega_next = as_matrix(self.phi.Omega2(self.psi))
        q = as_vector(omega_next.rev_matvec(a), numerical_tolerance=tol)

        c = as_vector(belief.c(), numerical_tolerance=tol)
        log_scale = torch.as_tensor(belief.log_norm_constant(), dtype=a.dtype, device=a.device)
        c = c * torch.exp(log_scale)
        overlap = as_vector(omega_current.matvec(c), numerical_tolerance=tol)
        c_next = as_vector(B.rev_matvec(a * overlap), numerical_tolerance=tol) / q

        return LinearFF(
            self.a, self.phi,
            _fixed_vector_parameters(c_next, modes=getattr(self.psi, "modes", None)),
            self.psi,
            numerical_tolerance=belief.numerical_tolerance,
            renormalize_h=False, register_modules=False,
        )

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
    def __init__(self, a: Parameters, phi: Basis, c: Parameters, psi: Basis,
                 numerical_tolerance: float = 1e-20, renormalize_h: bool = True,
                 register_modules: bool = True):
        if not isinstance(phi, Basis) or not isinstance(psi, Basis):
            raise TypeError("phi and psi must be Basis instances")
        if phi.dim() != psi.dim():
            raise ValueError("phi and psi must have the same dimension")
        for name, provider, basis in (("a", a, phi), ("c", c, psi)):
            if not _is_parameters_like(provider):
                raise TypeError(f"{name} must be a Parameters-like provider")
            values = provider()
            actual = tuple(values.shape)
            expected = (basis.batch_size(), basis.n_basis_functions())
            if actual != expected and not (basis.batch_size() == 1 and
                                           actual == (basis.n_basis_functions(),)):
                raise ValueError(f"{name} has shape {actual}; expected {expected}"
                                 + (f" or {(basis.n_basis_functions(),)}"
                                    if basis.batch_size() == 1 else ""))

        super().__init__(phi.dim())
        self.a, self.phi = a, phi
        self.c, self.psi = c, psi
        self.numerical_tolerance = numerical_tolerance
        self._renormalize_h = renormalize_h

        if register_modules:
            param_modules, coeff_modules = Basis.get_deduplicated_module_list([phi, psi])
            # Register every provider (including TTVectorParameters' leaf modules)
            # exactly once. Bases themselves remain coefficient-free.
            seen = {id(m) for m in (*param_modules, *coeff_modules)}
            coeff_modules = list(coeff_modules)
            for provider in (a, c):
                if getattr(provider, "is_module", lambda: False)() and id(provider) not in seen:
                    coeff_modules.append(provider)
                    seen.add(id(provider))
                for module in provider.parameter_modules():
                    if id(module) not in seen:
                        coeff_modules.append(module)
                        seen.add(id(module))
            self._param_modules = torch.nn.ModuleList(param_modules)
            self._coeff_modules = torch.nn.ModuleList(coeff_modules)

    def dtype_device(self):
        return self.phi.dtype_device()

    @classmethod
    def from_rff(cls, rff: LinearRFF | SumProdRFF, c: Parameters, psi: Basis,
                 renormalize_h: bool = True, register_modules: bool = True):
        if not isinstance(rff, (LinearRFF, SumProdRFF)):
            raise TypeError("rff must be a LinearRFF or SumProdRFF")
        return cls(rff.a, rff.phi, c, psi,
                   numerical_tolerance=rff.numerical_tolerance,
                   renormalize_h=renormalize_h,
                   register_modules=register_modules)

    def log_norm_constant(self, Omega2=None):
        if not self._renormalize_h:
            return 0.0
        if Omega2 is None:
            Omega2 = self.phi.Omega2(self.psi)
        tol = self.numerical_tolerance
        a = as_vector(self.a(), numerical_tolerance=tol)
        c = as_vector(self.c(), numerical_tolerance=tol)
        overlap = as_vector(as_matrix(Omega2).matvec(c), numerical_tolerance=tol)
        return -torch.log((a * overlap).sum() + tol)

    def log_density(self, x: torch.Tensor):
        tol = self.numerical_tolerance
        a = as_vector(self.a(), numerical_tolerance=tol)
        c = as_vector(self.c(), numerical_tolerance=tol)
        phi_x = as_vector(self.phi(x), numerical_tolerance=tol)
        psi_x = as_vector(self.psi(x), numerical_tolerance=tol)
        g_sum = (a * phi_x).sum().clamp_min(0)
        h_sum = (c * psi_x).sum().clamp_min(0)
        # Avoid a coarse numerical_tolerance floor when taking the logarithm.
        eps = torch.finfo(g_sum.dtype).tiny
        return self.log_norm_constant() + torch.log(g_sum + eps) + torch.log(h_sum + eps)

    def marginal(self, marginal_dims: tuple[int, ...]):
        raise NotImplementedError("LinearFF.marginal is not implemented")

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
    
class OldQuadraticRFF(ConditionalDensityModel):
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
            Omega = as_matrix(self.phi_basis.Omega22(self.psi_basis)).to_dense()
        
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


class OldQuadraticFF(DensityModel):
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
    def from_rff(cls, rff : OldQuadraticRFF, psi0_basis : SeparableBasis = None):
        phi_basis = rff.phi_basis.freeze_params()
        A = rff.get_A().detach().clone()
        return cls(A, phi_basis, psi0_basis, numerical_tolerance=rff.numerical_tolerance)

    def get_C0(self, Omega_0 : torch.Tensor = None):
        if hasattr(self, "C0_fixed"):
            return self.C0_fixed

        if Omega_0 is None:
            Omega_0 = as_matrix(self.phi_basis.Omega22(self.psi0_basis)).to_dense()
        
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

# Dense SOS quadratic rational factor form.  The old models above remain
# available as OldQuadraticRFF and OldQuadraticFF.

def _quadratic_parameter_matrix(provider, size: int, batch_size: int, name: str) -> DenseMatrix:
    """Read a dense matrix parameter provider; retain its autograd graph."""
    if not _is_parameters_like(provider):
        raise TypeError(f"{name} must be a Parameters-like matrix provider")
    value = as_matrix(provider())
    if not isinstance(value, DenseMatrix):
        raise TypeError(f"{name} must return a DenseMatrix or a dense torch.Tensor")
    dense = value.to_dense()
    if dense.shape == (size, size) and batch_size == 1:
        return DenseMatrix(dense.unsqueeze(0))
    if dense.shape != (batch_size, size, size):
        raise ValueError(
            f"{name} has shape {tuple(dense.shape)}; expected "
            f"({batch_size}, {size}, {size})"
            + (f" or ({size}, {size})" if batch_size == 1 else "")
        )
    return value


def _quadratic_register_modules(model, bases, providers):
    param_modules, coeff_modules = Basis.get_deduplicated_module_list(bases)
    param_modules = list(param_modules)
    coeff_modules = list(coeff_modules)
    seen = {id(mod) for mod in (*param_modules, *coeff_modules)}
    for provider in providers:
        if provider is None:
            continue
        if getattr(provider, 'is_module', lambda: False)() and id(provider) not in seen:
            coeff_modules.append(provider)
            seen.add(id(provider))
        for mod in provider.parameter_modules():
            if id(mod) not in seen:
                coeff_modules.append(mod)
                seen.add(id(mod))
    model._param_modules = torch.nn.ModuleList(param_modules)
    model._coeff_modules = torch.nn.ModuleList(coeff_modules)


def _quadratic_spd_from_factor(factor: DenseMatrix, jitter: float) -> DenseMatrix:
    """Make SPD (rather than merely PSD) to ensure the inverse sqrt exists."""
    x = factor.to_dense()
    result = factor @ factor.T
    if jitter:
        n = x.shape[-1]
        result = DenseMatrix(
            result.to_dense() + jitter * torch.eye(n, dtype=x.dtype, device=x.device)
        )
    return result


def _quadratic_evaluate(basis, x: torch.Tensor):
    result = basis(x)
    if hasattr(result, 'to_dense'):
        result = result.to_dense()
    return result


def _quadratic_form(features: torch.Tensor, coeff: Matrix) -> torch.Tensor:
    matrix = coeff.to_dense()
    return torch.einsum('bi,bij,bj->b', features, matrix, features)


class QuadraticRFF(MarkovTransitionModel):
    r"""Normalized, positive quadratic rational Markov transition.

    p(y|x) = g(y)/g(x) * (phi(x) (x) psi(y))^T Q (phi(x) (x) psi(y)).

    LA : trainable dense factor (batch, m, m); A = LA LA^T + jitter I.
    LQ0: trainable dense factor (batch, m*n, m*n);
         Q0 = LQ0 LQ0^T + jitter I.

    Omega22(phi, psi) is a Matrix of shape (batch, m, m, n, n).
    R0 = contract_psi(Q0, C(A)); T = R0^{-1/2} A^{1/2};
    Q = (T (x) I)^T Q0 (T (x) I).

    Both factors are learned independently.  The principal matrix square roots
    are exact up to numerical floating-point error, not an approximation/truncation.
    """

    def __init__(self, LA: Parameters, phi: Basis, psi: Basis, LQ0: Parameters,
                 numerical_tolerance: float = 1e-20, pd_jitter: float = 1e-6,
                 register_modules: bool = True):
        if not isinstance(phi, Basis) or not isinstance(psi, Basis):
            raise TypeError('phi and psi must be Basis instances')
        if phi.dim() != psi.dim() or phi.batch_size() != psi.batch_size():
            raise ValueError('phi and psi must have matching dimensions and parameter batches')
        if pd_jitter <= 0:
            raise ValueError('pd_jitter must be strictly positive')
        m, n, batch = phi.n_basis_functions(), psi.n_basis_functions(), phi.batch_size()
        _quadratic_parameter_matrix(LA, m, batch, 'LA')
        _quadratic_parameter_matrix(LQ0, m*n, batch, 'LQ0')
        super().__init__(phi.dim())
        self.LA, self.LQ0 = LA, LQ0
        self.phi, self.psi = phi, psi
        # Consistency aliases for existing quadratic code.
        self.phi_basis, self.psi_basis = phi, psi
        self.numerical_tolerance = numerical_tolerance
        self.pd_jitter = pd_jitter
        if register_modules:
            _quadratic_register_modules(self, (phi, psi), (LA, LQ0))

    def dtype_device(self):
        return self.phi.dtype_device()

    def get_A(self) -> Matrix:
        return _quadratic_spd_from_factor(
            _quadratic_parameter_matrix(self.LA, self.phi.n_basis_functions(),
                                        self.phi.batch_size(), 'LA'), self.pd_jitter)

    def get_Q0(self) -> Matrix:
        m, n = self.phi.n_basis_functions(), self.psi.n_basis_functions()
        return _quadratic_spd_from_factor(
            _quadratic_parameter_matrix(self.LQ0, m*n, self.phi.batch_size(), 'LQ0'),
            self.pd_jitter)

    def get_C(self, A: Matrix = None) -> Matrix:
        """C(A)[a,b] = integral g(y) psi_a(y) psi_b(y) dy."""
        A = self.get_A() if A is None else as_matrix(A)
        omega = as_matrix(self.phi.Omega22(self.psi))
        return omega.contract_batch_matrix(A)

    def get_R0(self, Q0: Matrix = None, C: Matrix = None) -> Matrix:
        """R0[i,j] = Tr(Q0[i,:,j,:] C)."""
        Q0 = self.get_Q0() if Q0 is None else as_matrix(Q0)
        C = self.get_C() if C is None else as_matrix(C)
        m, n = self.phi.n_basis_functions(), self.psi.n_basis_functions()
        batch = Q0.to_dense().shape[:-2]
        blocks = Q0.to_dense().reshape(*batch, m, n, m, n).movedim(-3, -2)
        return DenseMatrix((blocks * C.to_dense()[..., None, None, :, :]).sum((-2, -1)))

    def get_T(self, A: Matrix = None, R0: Matrix = None) -> Matrix:
        A = self.get_A() if A is None else as_matrix(A)
        R0 = self.get_R0(C=self.get_C(A)) if R0 is None else as_matrix(R0)
        # The order is essential: T^T R0 T = A.
        return R0.sqrt(inverse=True) @ A.sqrt()

    def get_normalization(self):
        A = self.get_A()
        C = self.get_C(A)
        Q0 = self.get_Q0()
        R0 = self.get_R0(Q0, C)
        T = self.get_T(A, R0)
        return A, C, Q0, R0, T

    def get_Q(self, Q0: Matrix = None, T: Matrix = None) -> Matrix:
        """Return the normalized SOS Gram Q, without a dense Kronecker matrix."""
        if Q0 is None or T is None:
            _, _, q0, _, t = self.get_normalization()
            Q0 = q0 if Q0 is None else as_matrix(Q0)
            T = t if T is None else as_matrix(T)
        else:
            Q0, T = as_matrix(Q0), as_matrix(T)
        m, n = self.phi.n_basis_functions(), self.psi.n_basis_functions()
        v = Q0.to_dense()
        qb = v.reshape(*v.shape[:-2], m, n, m, n)
        transformed = torch.einsum(
            '...ki,...kalb,...lj->...iajb', T.to_dense(), qb, T.to_dense()
        )
        return DenseMatrix(transformed.reshape(*v.shape[:-2], m*n, m*n))

    def log_density(self, xp: torch.Tensor, *, conditioner: torch.Tensor, **contexts):
        A, _, Q0, _, T = self.get_normalization()
        Q = self.get_Q(Q0, T)
        phi_x = _quadratic_evaluate(self.phi, conditioner)
        phi_y = _quadratic_evaluate(self.phi, xp)
        psi_y = _quadratic_evaluate(self.psi, xp)
        u = torch.einsum('bi,bj->bij', phi_x, psi_y).reshape(phi_x.shape[0], -1)
        g_x = _quadratic_form(phi_x, A).clamp_min(torch.finfo(phi_x.dtype).tiny)
        g_y = _quadratic_form(phi_y, A).clamp_min(torch.finfo(phi_x.dtype).tiny)
        f = _quadratic_form(u, Q).clamp_min(torch.finfo(phi_x.dtype).tiny)
        return (torch.log(g_y) + torch.log(f) - torch.log(g_x)).reshape(-1)

    def propagate(self, belief: 'QuadraticFF') -> 'QuadraticFF':
        """Exact one-step dense SOS propagation, preserving PSD and total mass.

        D(V)[i,j] = integral phi_i(x) phi_j(x) psi0(x)^T V psi0(x) dx.
        V_next = sum_kl [T D(V) T^T]_kl Q0[k,:,l,:].
        """
        if not isinstance(belief, QuadraticFF):
            raise TypeError(f'QuadraticRFF.propagate expects QuadraticFF, got {type(belief).__name__}')
        if belief.LA is not self.LA or belief.phi is not self.phi:
            raise ValueError('belief must share LA and phi with transition (use QuadraticFF.from_rff)')
        _, _, Q0, _, T = self.get_normalization()
        V = belief.get_V(normalized=True)
        omega = as_matrix(self.phi.Omega22(belief.psi))
        D = DenseMatrix(omega.frobenius_inner(
            DenseMatrix(V.to_dense()[..., None, None, :, :])
        ))
        weighted_D = (T @ D) @ T.T
        m, n = self.phi.n_basis_functions(), self.psi.n_basis_functions()
        qb = Q0.to_dense().reshape(self.phi.batch_size(), m, n, m, n)
        qb = qb.movedim(-3, -2)  # (batch, m, m, n, n)
        Vnext = DenseMatrix(qb).contract_batch_matrix(weighted_D)
        # Use the already normalized, positive result, not a newly learned factor.
        return QuadraticFF(
            self.LA, self.phi, None, self.psi,
            V_fixed=Vnext.to_dense().detach().clone(),
            numerical_tolerance=belief.numerical_tolerance,
            pd_jitter=self.pd_jitter, renormalize_h=False,
            register_modules=False,
        )

    def weight_params(self):
        return itertools.chain(self.LA.parameter_modules(), self.LQ0.parameter_modules())

    def basis_params(self):
        return itertools.chain(self.phi.parameters(), self.psi.parameters())


class QuadraticFF(DensityModel):
    r"""SOS belief: p(x) = (phi^T A phi)(psi^T V psi).

    LA is the *same* provider used by QuadraticRFF. LV is a learned dense
    factor (batch, n, n). For propagated beliefs, pass V_fixed instead.
    Trainable V is automatically normalized by its exact fourth-order moment.
    """

    def __init__(self, LA: Parameters, phi: Basis, LV: Parameters | None, psi: Basis,
                 numerical_tolerance: float = 1e-20, pd_jitter: float = 1e-6,
                 renormalize_h: bool = True, register_modules: bool = True,
                 V_fixed: torch.Tensor | Matrix | None = None):
        if not isinstance(phi, Basis) or not isinstance(psi, Basis):
            raise TypeError('phi and psi must be Basis instances')
        if phi.dim() != psi.dim() or phi.batch_size() != psi.batch_size():
            raise ValueError('phi and psi must have matching dimension and parameter batch')
        m, n, batch = phi.n_basis_functions(), psi.n_basis_functions(), phi.batch_size()
        _quadratic_parameter_matrix(LA, m, batch, 'LA')
        if (LV is None) == (V_fixed is None):
            raise ValueError('provide exactly one of LV (trainable) or V_fixed')
        if LV is not None:
            _quadratic_parameter_matrix(LV, n, batch, 'LV')
        else:
            fixed = as_matrix(V_fixed).to_dense()
            if fixed.shape == (n, n) and batch == 1:
                fixed = fixed.unsqueeze(0)
            if fixed.shape != (batch, n, n):
                raise ValueError(f'V_fixed has shape {tuple(fixed.shape)}, expected {(batch,n,n)}')
        super().__init__(phi.dim())
        self.LA, self.LV = LA, LV
        self.phi, self.psi = phi, psi
        self.phi_basis, self.psi_basis = phi, psi
        self.numerical_tolerance = numerical_tolerance
        self.pd_jitter = pd_jitter
        self._renormalize_h = renormalize_h
        if V_fixed is not None:
            self.register_buffer('V_fixed', fixed)
        if register_modules:
            _quadratic_register_modules(self, (phi, psi), (LA, LV))

    @classmethod
    def from_rff(cls, rff: QuadraticRFF, LV: Parameters, psi0_basis: Basis,
                 renormalize_h: bool = True, register_modules: bool = True):
        if not isinstance(rff, QuadraticRFF):
            raise TypeError('rff must be QuadraticRFF')
        return cls(
            rff.LA, rff.phi, LV, psi0_basis,
            numerical_tolerance=rff.numerical_tolerance, pd_jitter=rff.pd_jitter,
            renormalize_h=renormalize_h, register_modules=register_modules,
        )

    def dtype_device(self):
        return self.phi.dtype_device()

    def get_A(self) -> Matrix:
        return _quadratic_spd_from_factor(
            _quadratic_parameter_matrix(self.LA, self.phi.n_basis_functions(),
                                        self.phi.batch_size(), 'LA'), self.pd_jitter)

    def get_V(self, normalized: bool = False) -> Matrix:
        if self.LV is None:
            V = DenseMatrix(self.V_fixed)
        else:
            V = _quadratic_spd_from_factor(
                _quadratic_parameter_matrix(self.LV, self.psi.n_basis_functions(),
                                            self.psi.batch_size(), 'LV'), self.pd_jitter)
        if normalized and self._renormalize_h:
            mass = self._mass(V)
            if bool((mass <= 0).any()):
                raise ValueError('belief has zero or negative integral')
            V = DenseMatrix(V.to_dense() / mass[..., None, None])
        return V

    def _mass(self, V: Matrix, omega=None):
        omega = as_matrix(self.phi.Omega22(self.psi)) if omega is None else as_matrix(omega)
        C = omega.contract_batch_matrix(self.get_A())
        return C.frobenius_inner(V)

    def log_norm_constant(self, Omega22=None):
        if not self._renormalize_h:
            return 0.0
        mass = self._mass(self.get_V(), omega=Omega22)
        if bool((mass <= 0).any()):
            raise ValueError('belief has zero or negative integral')
        return -torch.log(mass)

    def log_density(self, x: torch.Tensor, **contexts):
        phi = _quadratic_evaluate(self.phi, x)
        psi = _quadratic_evaluate(self.psi, x)
        g = _quadratic_form(phi, self.get_A()).clamp_min(torch.finfo(phi.dtype).tiny)
        h = _quadratic_form(psi, self.get_V()).clamp_min(torch.finfo(phi.dtype).tiny)
        return (torch.log(g) + torch.log(h) + self.log_norm_constant()).reshape(-1)

    def weight_params(self):
        if self.LV is None:
            return []
        return self.LV.parameter_modules()

    def basis_params(self):
        return self.psi.parameters()

    def marginal(self, marginal_dims: tuple[int, ...]):
        raise NotImplementedError('QuadraticFF.marginal requires a product-basis implementation')
