import torch
import copy
import itertools
from .basis_functions import Basis, SeparableBasis, NonnegativeBasis
from .density_model import DensityModel, ConditionalDensityModel, MarkovTransitionModel
from .parameters import Parameters, FixedParameters
from .structured_matrices import Matrix, DenseMatrix, as_matrix
from .structured_vectors import as_vector
from rational_factor.tools.misc import Hyperrectangle


# Linear models #

class LinearRFF(MarkovTransitionModel):
    def __init__(self, a : Parameters, phi : Basis, psi : Basis, B : Parameters,
                numerical_tolerance : float = 1e-20, register_modules : bool = True):
        assert phi.dim() == psi.dim(), "phi and psi must have the same dimension"
        assert (callable(a) and hasattr(a, "is_trainable") and
                hasattr(a, "parameter_modules")), "a must be a Parameters-like object"
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
        super().__init__(phi.dim())

        self.a = a
        self.phi = phi
        self.psi = psi
        
        expected = (batch_size, n_basis, n_basis)
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

    def log_density(self, xp : torch.Tensor, *, conditioner : torch.Tensor, **context : torch.Tensor):
        # f(x, xp) = phi(x)^T Q psi(xp) with Q = diag(a) @ B @ diag(q)^{-1},
        # q = Omega2^T a.  Applied as elementwise scales around B.matvec.
        x = conditioner
        tol = self.numerical_tolerance

        a = as_vector(self.a(**context), numerical_tolerance=tol)
        phi_x = as_vector(self.phi(x, **context), numerical_tolerance=tol)
        phi_xp = as_vector(self.phi(xp, **context), numerical_tolerance=tol)
        psi_xp = as_vector(self.psi(xp, **context), numerical_tolerance=tol)

        B = self.B(**context)
        Omega2 = as_matrix(self.psi.Omega2(self.phi, **context))
        q = as_vector(Omega2.matvec(a), numerical_tolerance=tol)

        # Q @ psi = a * (B @ (psi / q))
        Q_psi_xp = a * B.matvec(psi_xp / q)
        g_x = (a * phi_x).sum().clamp_min(0)
        g_xp = (a * phi_xp).sum().clamp_min(0)
        f = as_vector(phi_x * Q_psi_xp).sum().clamp_min(0)
        eps = torch.finfo(g_x.dtype).tiny
        return (torch.log(g_xp * f + eps) - torch.log(g_x + eps)).reshape(-1)


    def propagate(self, belief: "LinearFF", starting_region: Hyperrectangle = None) -> "LinearFF":
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
                "LinearRFF.propagate expects a LinearFF belief, got "
                f"{type(belief).__name__}"
            )

        tol = self.numerical_tolerance
        a = as_vector(self.a(), numerical_tolerance=tol)
        B = as_matrix(self.B())
        
        # Omega2 between RFF bases used for normalization
        Omega2_normalization = as_matrix(self.psi.Omega2(self.phi))

        # Omega2 between RFF input basis and belief basis for transition
        Omega2_transition = as_matrix(belief.psi.Omega2(self.phi))
        q = as_vector(Omega2_normalization.matvec(a), numerical_tolerance=tol)

        c = as_vector(belief.norm_constant() * belief.c(), numerical_tolerance=tol)

        # v = c^T Omega2_transition diag(a)
        v = a * Omega2_transition.rev_matvec(c)

        # c' = v^T B diag(q)^{-1}
        cp = B.rev_matvec(v) / q

        return LinearFF(
            self.a, self.phi,
            FixedParameters(cp.to_dense()),
            self.psi,
            numerical_tolerance=belief.numerical_tolerance,
            renormalize_h=False, register_modules=False,
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
            if not (callable(provider) and hasattr(provider, "is_trainable")
                    and hasattr(provider, "parameter_modules")):
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
    def from_rff(cls, rff: "LinearRFF", c: Parameters, psi: Basis,
                 renormalize_h: bool = True, register_modules: bool = True):
        if not isinstance(rff, LinearRFF):
            raise TypeError("rff must be a LinearRFF")
        return cls(rff.a, rff.phi, c, psi,
                   numerical_tolerance=rff.numerical_tolerance,
                   renormalize_h=renormalize_h,
                   register_modules=register_modules)

    def norm_constant(self, Omega2=None, log=False):
        if not self._renormalize_h:
            return 0.0 if log else 1.0
        if Omega2 is None:
            Omega2 = self.phi.Omega2(self.psi)
        tol = self.numerical_tolerance
        a = as_vector(self.a(), numerical_tolerance=tol)
        c = as_vector(self.c(), numerical_tolerance=tol)
        overlap = as_vector(as_matrix(Omega2).matvec(c), numerical_tolerance=tol)
        return -torch.log((a * overlap).sum() + tol) if log else torch.exp(-torch.log((a * overlap).sum() + tol))

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
        return self.norm_constant(log=True) + torch.log(g_sum + eps) + torch.log(h_sum + eps)

    def integrate(self, region):
        """Integral of the (possibly normalized) belief over a Hyperrectangle."""
        dtype, device = self.dtype_device()
        lows = torch.as_tensor(region.low, dtype=dtype, device=device).reshape(-1)
        highs = torch.as_tensor(region.high, dtype=dtype, device=device).reshape(-1)
        if lows.numel() != self.features or highs.numel() != self.features:
            raise ValueError(f"region bounds must each have {self.features} entries")
        if bool((lows > highs).any()):
            raise ValueError("region.low must not exceed region.high")
        tol = self.numerical_tolerance
        gram = as_matrix(self.phi.Omega2(self.psi, lows=lows, highs=highs))
        a = as_vector(self.a(), numerical_tolerance=tol)
        c = as_vector(self.c(), numerical_tolerance=tol)
        unnormalized_mass = (a * as_vector(gram.matvec(c), numerical_tolerance=tol)).sum()
        return unnormalized_mass * torch.exp(torch.as_tensor(self.norm_constant(log=True), dtype=dtype, device=device))

    def log_marginal_density(self, x: torch.Tensor, marginal_dims: tuple[int, ...]):
        """Log p(x_remaining), integrating out the coordinates in marginal_dims.

        The marginal generally cannot be factorized into independent g and h
        factors. Its exact expression uses their cross-Gram on removed axes.
        """
        if not isinstance(self.phi, SeparableBasis) or not isinstance(self.psi, SeparableBasis):
            raise TypeError("marginal density requires SeparableBasis for phi and psi")
        marginal_dims = tuple(int(d) for d in marginal_dims)
        if len(set(marginal_dims)) != len(marginal_dims) or any(
            d < 0 or d >= self.features for d in marginal_dims
        ):
            raise ValueError("marginal_dims must contain distinct indices in [0, dim)")
        remaining = tuple(d for d in range(self.features) if d not in marginal_dims)
        if x.ndim != 2 or x.shape[1] != len(remaining):
            raise ValueError(f"x must have shape (n_data, {len(remaining)})")
        if not marginal_dims:
            return self.log_density(x)

        logged = isinstance(self.phi, NonnegativeBasis) and isinstance(self.psi, NonnegativeBasis)
        gram_dim = self.phi.Omega2_dim(self.psi, return_log=logged)
        selected = gram_dim[:, list(marginal_dims)]
        gram = selected.sum(dim=1).exp() if logged else selected.prod(dim=1)

        # Evaluate only the retained coordinate factors. The remaining entries
        # are interior placeholders required by the full-dimensional eval_dim API.
        full_x = x.new_full((x.shape[0], self.features), 0.5)
        index = torch.tensor(remaining, dtype=torch.long, device=x.device)
        full_x = full_x.index_copy(1, index, x)
        phi_remaining = (self.phi.eval_dim(full_x)[:, list(remaining)].prod(dim=1)
                         * Basis._coeff_tensor(self.phi.coeffs()))
        psi_remaining = (self.psi.eval_dim(full_x)[:, list(remaining)].prod(dim=1)
                         * Basis._coeff_tensor(self.psi.coeffs()))
        a = Basis._coeff_tensor(self.a())
        c = Basis._coeff_tensor(self.c())
        # (n_data,): a_i*c_j*Omega_ij(marginal_dims)*phi_i(x_R)*psi_j(x_R)
        marginal = torch.einsum('...ij,...i,...j->...',
                                gram * a[..., :, None] * c[..., None, :],
                                phi_remaining, psi_remaining)
        eps = torch.finfo(marginal.dtype).tiny
        return (torch.log(marginal.clamp_min(0) + eps) + self.log_norm_constant()).reshape(-1)


# Quadratic models #

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
        self._parameter_matrix(LA, m, batch, 'LA')
        self._parameter_matrix(LQ0, m*n, batch, 'LQ0')
        super().__init__(phi.dim())
        self.LA, self.LQ0 = LA, LQ0
        self.phi, self.psi = phi, psi
        # Consistency aliases for existing quadratic code.
        self.phi_basis, self.psi_basis = phi, psi
        self.numerical_tolerance = numerical_tolerance
        self.pd_jitter = pd_jitter
        if register_modules:
            param_modules, coeff_modules = Basis.get_deduplicated_module_list([phi, psi])
            seen = {id(mod) for mod in (*param_modules, *coeff_modules)}
            coeff_modules = list(coeff_modules)
            for provider in (LA, LQ0):
                if provider is None:
                    continue
                if getattr(provider, 'is_module', lambda: False)() and id(provider) not in seen:
                    coeff_modules.append(provider)
                    seen.add(id(provider))
                for mod in provider.parameter_modules():
                    if id(mod) not in seen:
                        coeff_modules.append(mod)
                        seen.add(id(mod))
            self._param_modules = torch.nn.ModuleList(param_modules)
            self._coeff_modules = torch.nn.ModuleList(coeff_modules)

    def dtype_device(self):
        return self.phi.dtype_device()

    @staticmethod
    def _parameter_matrix(provider, size: int, batch_size: int, name: str, **context) -> DenseMatrix:
        """Read a dense matrix parameter, checking its trailing dimensions."""
        if not (callable(provider) and hasattr(provider, 'is_trainable')
                and hasattr(provider, 'parameter_modules')):
            raise TypeError(f"{name} must be a Parameters-like matrix provider")
        value = as_matrix(provider(**context))
        if not isinstance(value, DenseMatrix):
            raise TypeError(f"{name} must return a dense matrix")
        dense = value.to_dense()
        if dense.shape[-2:] != (size, size):
            raise ValueError(
                f"{name} trailing shape must be ({size}, {size}), got {tuple(dense.shape)}"
            )
        if dense.ndim == 2:
            if batch_size != 1 and not context:
                raise ValueError(
                    f"{name} has shape {tuple(dense.shape)}; expected ({batch_size}, {size}, {size})"
                )
            return DenseMatrix(dense.unsqueeze(0))
        return value

    def _spd_matrix(self, provider, size: int, name: str, **context) -> DenseMatrix:
        """Build A, Q0, or V as L L^T + jitter I with full autograd support."""
        factor = self._parameter_matrix(provider, size, self.phi.batch_size(), name, **context)
        dense = factor.to_dense()
        spd = factor @ factor.T
        return DenseMatrix(spd.to_dense() + self.pd_jitter * torch.eye(
            size, dtype=dense.dtype, device=dense.device
        ))

    def get_A(self, **context) -> Matrix:
        return self._spd_matrix(self.LA, self.phi.n_basis_functions(), 'LA', **context)

    def get_Q0(self, **context) -> Matrix:
        m, n = self.phi.n_basis_functions(), self.psi.n_basis_functions()
        return self._spd_matrix(self.LQ0, m*n, 'LQ0', **context)

    def get_C(self, A: Matrix = None, **context) -> Matrix:
        """C(A)[a,b] = integral g(y) psi_a(y) psi_b(y) dy."""
        A = self.get_A(**context) if A is None else as_matrix(A)
        omega = as_matrix(self.phi.Omega22(self.psi, **context))
        return omega.contract_batch_matrix(A)

    def get_R0(self, Q0: Matrix = None, C: Matrix = None, **context) -> Matrix:
        """R0[i,j] = Tr(Q0[i,:,j,:] C)."""
        Q0 = self.get_Q0(**context) if Q0 is None else as_matrix(Q0)
        C = self.get_C(**context) if C is None else as_matrix(C)
        m, n = self.phi.n_basis_functions(), self.psi.n_basis_functions()
        batch = Q0.to_dense().shape[:-2]
        blocks = Q0.to_dense().reshape(*batch, m, n, m, n).movedim(-3, -2)
        return DenseMatrix((blocks * C.to_dense()[..., None, None, :, :]).sum((-2, -1)))

    def get_T(self, A: Matrix = None, R0: Matrix = None, **context) -> Matrix:
        A = self.get_A(**context) if A is None else as_matrix(A)
        R0 = self.get_R0(C=self.get_C(A, **context), **context) if R0 is None else as_matrix(R0)
        # The order is essential: T^T R0 T = A.
        return R0.sqrt(inverse=True) @ A.sqrt()

    def get_normalization(self, **context):
        A = self.get_A(**context)
        C = self.get_C(A, **context)
        Q0 = self.get_Q0(**context)
        R0 = self.get_R0(Q0, C, **context)
        T = self.get_T(A, R0, **context)
        return A, C, Q0, R0, T

    def get_Q(self, Q0: Matrix = None, T: Matrix = None, **context) -> Matrix:
        """Return the normalized SOS Gram Q, without a dense Kronecker matrix."""
        if Q0 is None or T is None:
            _, _, q0, _, t = self.get_normalization(**context)
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

    def log_density(self, xp: torch.Tensor, *, conditioner: torch.Tensor, **context : torch.Tensor):
        A, _, Q0, _, T = self.get_normalization(**context)
        Q = self.get_Q(Q0, T)
        phi_x = self.phi(conditioner, **context)
        phi_y = self.phi(xp, **context)
        psi_y = self.psi(xp, **context)
        phi_x = phi_x.to_dense() if hasattr(phi_x, 'to_dense') else phi_x
        phi_y = phi_y.to_dense() if hasattr(phi_y, 'to_dense') else phi_y
        psi_y = psi_y.to_dense() if hasattr(psi_y, 'to_dense') else psi_y
        u = torch.einsum('bi,bj->bij', phi_x, psi_y).reshape(phi_x.shape[0], -1)
        eps = torch.finfo(phi_x.dtype).tiny
        g_x = torch.einsum('...i,...ij,...j->...', phi_x, A.to_dense(), phi_x).clamp_min(eps)
        g_y = torch.einsum('...i,...ij,...j->...', phi_y, A.to_dense(), phi_y).clamp_min(eps)
        f = torch.einsum('...i,...ij,...j->...', u, Q.to_dense(), u).clamp_min(eps)
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
            V_fixed=FixedParameters(Vnext.to_dense()),
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
    factor (batch, n, n). Propagated beliefs hold their computed Gram matrix
    in a FixedParameters provider, retaining the autograd graph.
    Trainable V is automatically normalized by its exact fourth-order moment.
    """

    def __init__(self, LA: Parameters, phi: Basis, LV: Parameters | None, psi: Basis,
                 numerical_tolerance: float = 1e-20, pd_jitter: float = 1e-6,
                 renormalize_h: bool = True, register_modules: bool = True,
                 V_fixed: Parameters | torch.Tensor | Matrix | None = None):
        if not isinstance(phi, Basis) or not isinstance(psi, Basis):
            raise TypeError('phi and psi must be Basis instances')
        if phi.dim() != psi.dim() or phi.batch_size() != psi.batch_size():
            raise ValueError('phi and psi must have matching dimension and parameter batch')
        m, n, batch = phi.n_basis_functions(), psi.n_basis_functions(), phi.batch_size()
        self._parameter_matrix(LA, m, batch, 'LA')
        if (LV is None) == (V_fixed is None):
            raise ValueError('provide exactly one of LV (trainable) or V_fixed')
        if LV is not None:
            self._parameter_matrix(LV, n, batch, 'LV')
        else:
            # A fixed provider holds the computed Gram matrix, not its factor.
            # FixedParameters stores the tensor unchanged, preserving autograd.
            if not isinstance(V_fixed, Parameters):
                V_fixed = FixedParameters(as_matrix(V_fixed).to_dense())
            fixed = self._parameter_matrix(V_fixed, n, batch, 'V_fixed').to_dense()
            if fixed.shape != (batch, n, n):
                raise ValueError(f'V_fixed has shape {tuple(fixed.shape)}, expected {(batch, n, n)}')
        super().__init__(phi.dim())
        self.LA, self.LV = LA, LV
        self.phi, self.psi = phi, psi
        self.phi_basis, self.psi_basis = phi, psi
        self.numerical_tolerance = numerical_tolerance
        self.pd_jitter = pd_jitter
        self._renormalize_h = renormalize_h
        self.V_fixed = V_fixed
        if register_modules:
            param_modules, coeff_modules = Basis.get_deduplicated_module_list([phi, psi])
            seen = {id(mod) for mod in (*param_modules, *coeff_modules)}
            coeff_modules = list(coeff_modules)
            for provider in (LA, LV, V_fixed):
                if provider is None:
                    continue
                if getattr(provider, 'is_module', lambda: False)() and id(provider) not in seen:
                    coeff_modules.append(provider)
                    seen.add(id(provider))
                for mod in provider.parameter_modules():
                    if id(mod) not in seen:
                        coeff_modules.append(mod)
                        seen.add(id(mod))
            self._param_modules = torch.nn.ModuleList(param_modules)
            self._coeff_modules = torch.nn.ModuleList(coeff_modules)

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

    @staticmethod
    def _parameter_matrix(provider, size: int, batch_size: int, name: str, **context) -> DenseMatrix:
        """Read a dense matrix parameter, checking its trailing dimensions."""
        if not (callable(provider) and hasattr(provider, 'is_trainable')
                and hasattr(provider, 'parameter_modules')):
            raise TypeError(f"{name} must be a Parameters-like matrix provider")
        value = as_matrix(provider(**context))
        if not isinstance(value, DenseMatrix):
            raise TypeError(f"{name} must return a dense matrix")
        dense = value.to_dense()
        if dense.shape[-2:] != (size, size):
            raise ValueError(
                f"{name} trailing shape must be ({size}, {size}), got {tuple(dense.shape)}"
            )
        if dense.ndim == 2:
            if batch_size != 1 and not context:
                raise ValueError(
                    f"{name} has shape {tuple(dense.shape)}; expected ({batch_size}, {size}, {size})"
                )
            return DenseMatrix(dense.unsqueeze(0))
        return value

    def _spd_matrix(self, provider, size: int, name: str, **context) -> DenseMatrix:
        """Build A, Q0, or V as L L^T + jitter I with full autograd support."""
        factor = self._parameter_matrix(provider, size, self.phi.batch_size(), name, **context)
        dense = factor.to_dense()
        spd = factor @ factor.T
        return DenseMatrix(spd.to_dense() + self.pd_jitter * torch.eye(
            size, dtype=dense.dtype, device=dense.device
        ))

    def get_A(self) -> Matrix:
        return self._spd_matrix(self.LA, self.phi.n_basis_functions(), 'LA')

    def get_V(self, normalized: bool = False) -> Matrix:
        if self.LV is None:
            V = self._parameter_matrix(self.V_fixed, self.psi.n_basis_functions(),
                                       self.psi.batch_size(), 'V_fixed')
        else:
            V = self._spd_matrix(self.LV, self.psi.n_basis_functions(), 'LV')
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
        phi = self.phi(x)
        psi = self.psi(x)
        phi = phi.to_dense() if hasattr(phi, 'to_dense') else phi
        psi = psi.to_dense() if hasattr(psi, 'to_dense') else psi
        eps = torch.finfo(phi.dtype).tiny
        g = torch.einsum('...i,...ij,...j->...', phi, self.get_A().to_dense(), phi).clamp_min(eps)
        h = torch.einsum('...i,...ij,...j->...', psi, self.get_V().to_dense(), psi).clamp_min(eps)
        return (torch.log(g) + torch.log(h) + self.log_norm_constant()).reshape(-1)

    def weight_params(self):
        if self.LV is None:
            return []
        return self.LV.parameter_modules()

    def basis_params(self):
        return self.psi.parameters()

    def integrate(self, region):
        """Integral over a Hyperrectangle via the restricted fourth-order Gram."""
        dtype, device = self.dtype_device()
        lows = torch.as_tensor(region.low, dtype=dtype, device=device).reshape(-1)
        highs = torch.as_tensor(region.high, dtype=dtype, device=device).reshape(-1)
        if lows.numel() != self.features or highs.numel() != self.features:
            raise ValueError(f"region bounds must each have {self.features} entries")
        if bool((lows > highs).any()):
            raise ValueError("region.low must not exceed region.high")
        omega = self.phi.Omega22(self.psi, lows=lows, highs=highs)
        return self._mass(self.get_V(normalized=True), omega=omega)

    def log_marginal_density(self, x: torch.Tensor, marginal_dims: tuple[int, ...]):
        """Exact SOS marginal using per-coordinate fourth-order Gram tensors."""
        if not isinstance(self.phi, SeparableBasis) or not isinstance(self.psi, SeparableBasis):
            raise TypeError('marginal density requires SeparableBasis for phi and psi')
        marginal_dims = tuple(int(d) for d in marginal_dims)
        if len(set(marginal_dims)) != len(marginal_dims) or any(
            d < 0 or d >= self.features for d in marginal_dims
        ):
            raise ValueError("marginal_dims must contain distinct indices in [0, dim)")
        remaining = tuple(d for d in range(self.features) if d not in marginal_dims)
        if x.ndim != 2 or x.shape[1] != len(remaining):
            raise ValueError(f"x must have shape (n_data, {len(remaining)})")
        if not marginal_dims:
            return self.log_density(x)

        logged = isinstance(self.phi, NonnegativeBasis) and isinstance(self.psi, NonnegativeBasis)
        gram_dim = self.phi.Omega22_dim(self.psi, return_log=logged)
        selected = gram_dim[:, list(marginal_dims)]
        gram = selected.sum(dim=1).exp() if logged else selected.prod(dim=1)

        full_x = x.new_full((x.shape[0], self.features), 0.5)
        index = torch.tensor(remaining, dtype=torch.long, device=x.device)
        full_x = full_x.index_copy(1, index, x)
        phi_remaining = (self.phi.eval_dim(full_x)[:, list(remaining)].prod(dim=1)
                         * Basis._coeff_tensor(self.phi.coeffs()))
        psi_remaining = (self.psi.eval_dim(full_x)[:, list(remaining)].prod(dim=1)
                         * Basis._coeff_tensor(self.psi.coeffs()))
        A = self.get_A().to_dense()
        V = self.get_V(normalized=True).to_dense()
        # Contract phi_i*phi_j and psi_k*psi_l on retained coordinates,
        # and Omega22_dim over all marginalized coordinates.
        p = torch.einsum('...ijkl,...ij,...kl->...', gram,
                         A * phi_remaining[..., :, None] * phi_remaining[..., None, :],
                         V * psi_remaining[..., :, None] * psi_remaining[..., None, :])
        eps = torch.finfo(p.dtype).tiny
        return torch.log(p.clamp_min(0) + eps).reshape(-1)
