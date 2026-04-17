import torch
from abc import abstractmethod

from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform



class Basis(torch.nn.Module):
    def __init__(self, dim : int, n_basis : int, params_init : torch.Tensor = None, coeffs_init : torch.Tensor = None, trainable : bool = True):
        '''
        Args:
            dim : int, number of dimensions
            n_basis : int, number of basis functions
            params_init : torch.Tensor of initial parameters
            coeffs_init : torch.Tensor of initial coefficients
            trainable : bool indicating if parameters are trainable
        '''
        super().__init__()
        self._dim = dim
        self._n_basis = n_basis
        self._trainable = trainable

        if params_init is not None:
            self.set_params(params_init, trainable)
        if coeffs_init is not None:
            assert coeffs_init.dim() == 1, "coeffs_init must have shape (n_basis,)"
            assert coeffs_init.shape[0] == n_basis, "coeffs_init must have shape (n_basis,)"
            self.set_coeffs(coeffs_init, trainable)
        
    def set_params(self, params_init : torch.Tensor, trainable : bool):
        assert not hasattr(self, "params"), "params already set"
        if trainable:
            self.params = torch.nn.Parameter(params_init)
        else:
            self.register_buffer("params", params_init)

    def set_coeffs(self, coeffs_init : torch.Tensor, trainable : bool):
        assert not hasattr(self, "coeffs"), "coeffs already set"
        if trainable:
            self.coeffs = torch.nn.Parameter(coeffs_init)
        else:
            self.register_buffer("coeffs", coeffs_init)

    def dim(self):
        return self._dim
    
    def n_basis_functions(self):
        return self._n_basis
    
    def trainable(self):
        return self._trainable
    
    def freeze_params(self):
        raise NotImplementedError("freeze_params is not implemented for this basis function")
    
    def normalized(self):
        raise NotImplementedError("normalized is not implemented for this basis function")
    
    def Omega1(self, ignore_coeffs : bool = False):
        '''
        Computes the integral of the basis functions.
        omega[i] = <this_i, 1>

        Returns:
            Tensor of shape (n_basis,)
        '''
        raise NotImplementedError("Omega1 is not implemented for this basis function")

    def Omega2(self, other: 'Basis', ignore_coeffs : bool = False):
        '''
        Computes the function inner product matrix with another basis function vector. 
        omega[i, j] = <this_i, other_j>
        
        Returns:
            Tensor of shape (n_basis, other.n_basis)
        '''
        raise NotImplementedError("Omega2 is not implemented for this basis function")

    def Omega3_contract(self, other1: 'Basis', other2: 'Basis'):
        '''
        Computes the 3D quadratic function inner product tensor with another basis function vector. 
        omega[i, j, k] = <this_i, other1_j, other2_k>
        
        Returns:
            Tensor of shape (n_basis, other1.n_basis, other2.n_basis)
        '''
        raise NotImplementedError("Omega3 is not implemented for this basis function")

    def Omega22(self, other: 'Basis'):
        '''
        Computes the 4D quadratic function inner product tensor with another basis function vector. 
        omega[i, j, k, l] = <this_i * this_j, other_k * other_l>
        
        Returns:
            Tensor of shape (n_basis, n_basis, other.n_basis, other.n_basis)
        '''
        raise NotImplementedError("Omega4 is not implemented for this basis function")
    
    def product_basis(self, other_basis_factors : list['Basis'], ignore_coeffs : bool = False):
        '''
        Returns the broadcasted (flattened) product of the basis functions in the other_basis_factors list.
        '''
        raise NotImplementedError("product_basis is not implemented for this basis function")
    
    @abstractmethod
    def marginal(self, marginal_dims : tuple[int, ...], ignore_coeffs : bool = False):
        '''
        Computes the marginalized basis functions over the given dimensions

        Returns:
            A new SeparableBasis object with the marginalized dimensions
        '''
        pass

class SeparableBasis(Basis):
    def __init__(self, params_init : torch.Tensor, trainable : bool = True, coeffs_init : torch.Tensor | None = None):
        assert params_init.dim() == 3, "params_init must have shape (d, n_basis, n_params_per_basis)"
        super().__init__(
            params_init.shape[0],
            params_init.shape[1],
            params_init=params_init,
            coeffs_init=coeffs_init,
            trainable=trainable,
        )

    def n_params_per_basis(self):
        return self.params.shape[2]
    
# Nonnegative basis functions 
class NonnegativeBasis:
    pass


class QuadraticExpBasis(SeparableBasis, NonnegativeBasis):
    """
    Separable basis with 1D factors of the form exp(a x^2 + b x).
    Optional per-basis scaling is handled by self.coeffs.
    """

    def __init__(self, params_init: torch.Tensor, coeffs_init: torch.Tensor | None = None, trainable: bool = True, eps: float = 1e-6):
        assert params_init.shape[2] == 2, "params_init must have shape (d, n_basis, 2)"
        super().__init__(params_init, trainable=trainable, coeffs_init=coeffs_init)
        self.eps = eps

    @classmethod
    def random_init(
        cls,
        d: int,
        n_basis: int,
        offsets: torch.Tensor = torch.zeros(2),
        variance: float = 1.0,
        eps: float = 1e-6,
        device=None,
        coeffs_init: torch.Tensor | None = None,
    ):
        if device is None:
            device = offsets.device
        offsets = offsets.repeat(d, n_basis, 1)
        params_init = (
            torch.randn(d, n_basis, 2, device=device)
            * torch.sqrt(torch.tensor(variance, device=device))
            + offsets
        )
        return cls(params_init, coeffs_init=coeffs_init, eps=eps)

    @classmethod
    def set_init(
        cls,
        d: int,
        n_basis: int,
        offsets: torch.Tensor = torch.zeros(2),
        eps: float = 1e-6,
        coeffs_init: torch.Tensor | None = None,
    ):
        offsets = offsets.repeat(d, n_basis, 1)
        return cls(offsets, coeffs_init=coeffs_init, eps=eps)

    def freeze_params(self):
        coeffs = self.coeffs.detach().clone() if hasattr(self, "coeffs") else None
        return QuadraticExpBasis(self.params.detach().clone(), coeffs_init=coeffs, trainable=False, eps=self.eps)

    def ab(self):
        raw_a = self.params[..., 0]
        b = self.params[..., 1]
        a = -torch.nn.functional.softplus(raw_a) - self.eps
        return a, b

    def normalized(self):
        return False

    def forward(self, y: torch.Tensor):
        assert y.shape[1] == self.dim(), "y must have shape (n_data, d)"
        y = y[:, :, None]  # (n_data, d, n_basis)

        a, b = self.ab()  # (d, n_basis)
        log_dim_factors = a * y.square() + b * y
        out = torch.exp(log_dim_factors.sum(dim=1))  # (n_data, n_basis)
        if hasattr(self, "coeffs"):
            out = out * self.coeffs[None, :]
        return out

    def Omega1(self, ignore_coeffs: bool = False):
        a, b = self.ab()

        log_dim_int = -b.square() / (4.0 * a) + 0.5 * (
            torch.log(torch.tensor(torch.pi, dtype=a.dtype, device=a.device)) - torch.log(-a)
        )
        out = torch.exp(log_dim_int.sum(dim=0))  # (n_basis,)
        if (not ignore_coeffs) and hasattr(self, "coeffs"):
            out = out * self.coeffs
        return out
    
    def Omega2(self, other: "QuadraticExpBasis", ignore_coeffs: bool = False):
        assert isinstance(other, QuadraticExpBasis), "other must be QuadraticExpBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"

        a1, b1 = self.ab()
        a2, b2 = other.ab()

        # broadcast to (d, n1, n2)
        a = a1[:, :, None] + a2[:, None, :]
        b = b1[:, :, None] + b2[:, None, :]

        log_dim_int = -b.square() / (4.0 * a) + 0.5 * (
            torch.log(torch.tensor(torch.pi, dtype=a.dtype, device=a.device)) - torch.log(-a)
        )
        out = torch.exp(log_dim_int.sum(dim=0))  # (n1, n2)
        if not ignore_coeffs:
            if hasattr(self, "coeffs"):
                out = out * self.coeffs[:, None]
            if hasattr(other, "coeffs"):
                out = out * other.coeffs[None, :]
        return out
    
    def Omega3_contract(
        self,
        other1: "QuadraticExpBasis",
        other2: "QuadraticExpBasis",
        left_i: torch.Tensor,
        left_j: torch.Tensor,
        block_size: int | None = None,
    ):
        """
        Computes v[k] = sum_{i,j} left_i[i] * left_j[j] * Omega3[i,j,k]
        without materializing Omega3.
        """
        assert isinstance(other1, QuadraticExpBasis), "other1 must be QuadraticExpBasis"
        assert isinstance(other2, QuadraticExpBasis), "other2 must be QuadraticExpBasis"
        assert self.dim() == other1.dim() == other2.dim(), "Basis functions must have the same dimension"
        assert left_i.dim() == 1 and left_i.shape[0] == self.n_basis_functions(), "left_i has wrong shape"
        assert left_j.dim() == 1 and left_j.shape[0] == other1.n_basis_functions(), "left_j has wrong shape"

        a0, b0 = self.ab()
        a1, b1 = other1.ab()
        a2, b2 = other2.ab()

        n0 = self.n_basis_functions()
        n1 = other1.n_basis_functions()
        n2 = other2.n_basis_functions()

        log_pi = torch.log(torch.tensor(torch.pi, dtype=a0.dtype, device=a0.device))

        if block_size is None:
            A = a0[:, :, None, None] + a1[:, None, :, None] + a2[:, None, None, :]
            B = b0[:, :, None, None] + b1[:, None, :, None] + b2[:, None, None, :]
            log_dim_int = -(B * B) / (4.0 * A) + 0.5 * (log_pi - torch.log(-A))
            omega_full = torch.exp(log_dim_int.sum(dim=0))
            return torch.einsum("i,j,ijk->k", left_i, left_j, omega_full)

        assert block_size > 0, "block_size must be positive"
        denom = torch.zeros(n2, dtype=a0.dtype, device=a0.device)
        for j_start in range(0, n1, block_size):
            j_end = min(j_start + block_size, n1)
            left_j_blk = left_j[j_start:j_end]
            for k_start in range(0, n2, block_size):
                k_end = min(k_start + block_size, n2)
                log_chunk = torch.zeros((n0, j_end - j_start, k_end - k_start), dtype=a0.dtype, device=a0.device)
                for r in range(self.dim()):
                    A = (
                        a0[r, :, None, None]
                        + a1[r, None, j_start:j_end, None]
                        + a2[r, None, None, k_start:k_end]
                    )
                    B = (
                        b0[r, :, None, None]
                        + b1[r, None, j_start:j_end, None]
                        + b2[r, None, None, k_start:k_end]
                    )
                    log_chunk += -(B * B) / (4.0 * A) + 0.5 * (log_pi - torch.log(-A))

                omega_chunk = torch.exp(log_chunk)
                denom[k_start:k_end] += torch.einsum("i,j,ijk->k", left_i, left_j_blk, omega_chunk)

        return denom
    
    def Omega22(self, other: "QuadraticExpBasis"):
        """
        omega[i, j, k, l] = ∫ self_i(x) self_j(x) other_k(x) other_l(x) dx

        Returns:
            Tensor of shape (n_self, n_self, n_other, n_other)
        """
        assert isinstance(other, QuadraticExpBasis), "other must be QuadraticExpBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"

        a1, b1 = self.ab()    # (d, n_self)
        a2, b2 = other.ab()   # (d, n_other)

        # Broadcast to (d, n_self, n_self, n_other, n_other)
        A = (
            a1[:, :, None, None, None]
            + a1[:, None, :, None, None]
            + a2[:, None, None, :, None]
            + a2[:, None, None, None, :]
        )
        B = (
            b1[:, :, None, None, None]
            + b1[:, None, :, None, None]
            + b2[:, None, None, :, None]
            + b2[:, None, None, None, :]
        )
        log_pi = torch.log(torch.tensor(torch.pi, dtype=A.dtype, device=A.device))
        log_dim_int = -(B * B) / (4.0 * A) + 0.5 * (log_pi - torch.log(-A))

        log_Omega = log_dim_int.sum(dim=0)   # (n_self, n_self, n_other, n_other)
        return torch.exp(log_Omega)
    
    def marginal(self, marginal_dims: tuple[int, ...], ignore_coeffs: bool = False) -> "QuadraticExpBasis":
        marginal_dims = tuple(marginal_dims)
        assert all(0 <= i < self.dim() for i in marginal_dims), "marginal_dims must be in [0, d)"

        # In this codebase, marginal_dims are the coordinates to keep.
        keep_dims = marginal_dims
        integrate_dims = tuple(i for i in range(self.dim()) if i not in keep_dims)

        a, b = self.ab()  # (d, n_basis)

        # --- compute log integral contribution from marginalized dims ---
        if len(integrate_dims) > 0:
            a_m = a[integrate_dims, :]
            b_m = b[integrate_dims, :]

            log_pi = torch.log(torch.tensor(torch.pi, dtype=a.dtype, device=a.device))

            log_int_m = (
                -(b_m * b_m) / (4.0 * a_m)
                + 0.5 * (log_pi - torch.log(-a_m))
            )  # (|M|, n_basis)

            log_int_sum = log_int_m.sum(dim=0)  # (n_basis,)
        else:
            log_int_sum = torch.zeros(
                self.n_basis_functions(),
                dtype=a.dtype,
                device=a.device,
            )

        a_k = a[keep_dims, :]
        b_k = b[keep_dims, :]
        coeffs_new = torch.exp(log_int_sum)
        if (not ignore_coeffs) and hasattr(self, "coeffs"):
            coeffs_new = coeffs_new * self.coeffs

        s = (-a_k - self.eps).clamp_min(1e-12)
        raw_a = torch.log(torch.expm1(s))

        params = torch.stack([raw_a, b_k], dim=-1)

        return QuadraticExpBasis(
            params.detach().clone(),
            coeffs_init=coeffs_new.detach().clone(),
            trainable=self.params.requires_grad,
            eps=self.eps,
        )
    
    def product_basis(self, other_basis_factors : list['Basis'], ignore_coeffs : bool = False) -> "QuadraticExpBasis":
        factors: list[QuadraticExpBasis] = [self]
        for other in other_basis_factors:
            assert isinstance(other, QuadraticExpBasis), "all factors must be QuadraticExpBasis"
            assert other.dim() == self.dim(), "Basis functions must have the same dimension"
            factors.append(other)

        n_factors = len(factors)
        dtype = self.params.dtype
        device = self.params.device
        dim = self.dim()

        ab_terms: list[tuple[torch.Tensor, torch.Tensor]] = [basis.ab() for basis in factors]
        n_per_factor = [basis.n_basis_functions() for basis in factors]

        coeff_shape = [dim, *n_per_factor]
        a_sum = torch.zeros(coeff_shape, dtype=dtype, device=device)
        b_sum = torch.zeros_like(a_sum)

        for k, (a_k, b_k) in enumerate(ab_terms):
            view_shape = [dim] + [1] * n_factors
            view_shape[k + 1] = n_per_factor[k]
            a_sum = a_sum + a_k.reshape(view_shape)
            b_sum = b_sum + b_k.reshape(view_shape)

        n_total = 1
        for n in n_per_factor:
            n_total *= n

        a_flat = a_sum.reshape(dim, n_total)
        b_flat = b_sum.reshape(dim, n_total)

        s = (-a_flat - self.eps).clamp_min(1e-12)
        raw_a = torch.log(torch.expm1(s))
        params = torch.stack([raw_a, b_flat], dim=-1)

        coeffs_new = None
        if not ignore_coeffs:
            coeff_terms = []
            for basis in factors:
                if not hasattr(basis, "coeffs"):
                    coeff_terms.append(torch.ones(basis.n_basis_functions(), dtype=dtype, device=device))
                else:
                    coeff_terms.append(basis.coeffs)

            coeff_prod = torch.ones(n_per_factor, dtype=dtype, device=device)
            for k, c_k in enumerate(coeff_terms):
                view_shape = [1] * n_factors
                view_shape[k] = n_per_factor[k]
                coeff_prod = coeff_prod * c_k.reshape(view_shape)
            coeffs_new = coeff_prod.reshape(n_total)

            if all(not hasattr(basis, "coeffs") for basis in factors):
                coeffs_new = None

        return QuadraticExpBasis(
            params.detach().clone(),
            coeffs_init=None if coeffs_new is None else coeffs_new.detach().clone(),
            trainable=self.params.requires_grad,
            eps=self.eps,
        )
    

class UnnormalizedBetaBasis(SeparableBasis, NonnegativeBasis):
    def __init__(
        self,
        params_init: torch.Tensor,
        coeffs_init: torch.Tensor | None = None,
        trainable: bool = True,
        min_concentration: float = 1.0,
        eps: float = 1e-6,
    ):
        assert min_concentration > 0.0, "min_concentration must be positive"
        assert params_init.shape[2] == 2, "params_init must have shape (d, n_basis, 2)"
        super().__init__(params_init, coeffs_init=coeffs_init, trainable=trainable)
        self.min_concentration = min_concentration
        self.eps = eps

    @classmethod
    def random_init(
        cls,
        d: int,
        n_basis: int,
        offsets: torch.Tensor = torch.zeros(2),
        variance: float = 1.0,
        min_concentration: float = 1.0,
        eps: float = 1e-6,
        device = None,
        coeffs_init: torch.Tensor | None = None,
    ):
        if device is None:
            device = offsets.device
        else:
            offsets = offsets.to(device)
        offsets = offsets.repeat(d, n_basis, 1)
        return cls(
            torch.randn(d, n_basis, 2, device=device) * torch.sqrt(torch.tensor(variance, device=device)) + offsets,
            coeffs_init=coeffs_init,
            min_concentration=min_concentration,
            eps=eps,
        )

    @classmethod
    def set_init(
        cls,
        d: int,
        n_basis: int,
        offsets: torch.Tensor = torch.zeros(2),
        min_concentration: float = 1.0,
        eps: float = 1e-6,
        device = None,
        coeffs_init: torch.Tensor | None = None,
    ):
        if device is None:
            device = offsets.device
        else:
            offsets = offsets.to(device)
        offsets = offsets.repeat(d, n_basis, 1)
        return cls(
            offsets,
            coeffs_init=coeffs_init,
            min_concentration=min_concentration,
            eps=eps,
        )

    def freeze_params(self):
        coeffs = self.coeffs.detach().clone() if hasattr(self, "coeffs") else None
        return UnnormalizedBetaBasis(
            self.params.detach().clone(),
            trainable=False,
            min_concentration=self.min_concentration,
            eps=self.eps,
            coeffs_init=coeffs,
        )

    def alphas_betas(self):
        alpha = torch.nn.functional.softplus(self.params[..., 0] - 1.0) + self.min_concentration
        beta = torch.nn.functional.softplus(self.params[..., 1] - 1.0) + self.min_concentration
        return alpha, beta

    @staticmethod
    def _log_beta_fn(a: torch.Tensor, b: torch.Tensor):
        return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)

    def forward(self, y: torch.Tensor):
        assert y.shape[1] == self.dim(), "y must have shape (n_data, d)"

        y = y.clamp(self.eps, 1.0 - self.eps)
        y = y[:, :, None]  # (n_data, d, n_basis)

        alpha, beta = self.alphas_betas()  # (d, n_basis), (d, n_basis)
        log_dim_factors = (alpha - 1.0) * torch.log(y) + (beta - 1.0) * torch.log1p(-y)
        out = torch.exp(log_dim_factors.sum(dim=1))  # (n_data, n_basis)
        if hasattr(self, "coeffs"):
            out = out * self.coeffs[None, :]
        return out

    def normalized(self):
        return False

    def Omega1(self, ignore_coeffs: bool = False):
        alpha, beta = self.alphas_betas()  # (d, n_basis)
        log_dim_int = self._log_beta_fn(alpha, beta)
        out = torch.exp(log_dim_int.sum(dim=0))
        if (not ignore_coeffs) and hasattr(self, "coeffs"):
            out = out * self.coeffs
        return out

    def Omega2(self, other: "UnnormalizedBetaBasis", ignore_coeffs: bool = False):
        assert isinstance(other, UnnormalizedBetaBasis), "other must be UnnormalizedBetaBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"

        a1, b1 = self.alphas_betas()
        a2, b2 = other.alphas_betas()

        a_sum = a1[:, :, None] + a2[:, None, :] - 1.0
        b_sum = b1[:, :, None] + b2[:, None, :] - 1.0
        out = torch.exp(self._log_beta_fn(a_sum, b_sum).sum(dim=0))
        if not ignore_coeffs:
            if hasattr(self, "coeffs"):
                out = out * self.coeffs[:, None]
            if hasattr(other, "coeffs"):
                out = out * other.coeffs[None, :]
        return out
    
    def Omega3_contract(
        self,
        other1: "UnnormalizedBetaBasis",
        other2: "UnnormalizedBetaBasis",
        left_i: torch.Tensor,
        left_j: torch.Tensor,
        block_size: int | None = None,
    ):
        assert isinstance(other1, UnnormalizedBetaBasis), "other1 must be UnnormalizedBetaBasis"
        assert isinstance(other2, UnnormalizedBetaBasis), "other2 must be UnnormalizedBetaBasis"
        assert self.dim() == other1.dim() == other2.dim(), "Basis functions must have the same dimension"
        assert left_i.dim() == 1 and left_i.shape[0] == self.n_basis_functions(), "left_i has wrong shape"
        assert left_j.dim() == 1 and left_j.shape[0] == other1.n_basis_functions(), "left_j has wrong shape"

        a0, b0 = self.alphas_betas()
        a1, b1 = other1.alphas_betas()
        a2, b2 = other2.alphas_betas()

        n0 = self.n_basis_functions()
        n1 = other1.n_basis_functions()
        n2 = other2.n_basis_functions()

        if block_size is None:
            a_sum = a0[:, :, None, None] + a1[:, None, :, None] + a2[:, None, None, :] - 2.0
            b_sum = b0[:, :, None, None] + b1[:, None, :, None] + b2[:, None, None, :] - 2.0
            omega_full = torch.exp(self._log_beta_fn(a_sum, b_sum).sum(dim=0))
            return torch.einsum("i,j,ijk->k", left_i, left_j, omega_full)

        assert block_size > 0, "block_size must be positive"
        denom = torch.zeros(n2, dtype=a0.dtype, device=a0.device)
        for j_start in range(0, n1, block_size):
            j_end = min(j_start + block_size, n1)
            left_j_blk = left_j[j_start:j_end]
            for k_start in range(0, n2, block_size):
                k_end = min(k_start + block_size, n2)
                log_chunk = torch.zeros((n0, j_end - j_start, k_end - k_start), dtype=a0.dtype, device=a0.device)
                for r in range(self.dim()):
                    a_sum = (
                        a0[r, :, None, None]
                        + a1[r, None, j_start:j_end, None]
                        + a2[r, None, None, k_start:k_end]
                        - 2.0
                    )
                    b_sum = (
                        b0[r, :, None, None]
                        + b1[r, None, j_start:j_end, None]
                        + b2[r, None, None, k_start:k_end]
                        - 2.0
                    )
                    log_chunk += self._log_beta_fn(a_sum, b_sum)

                omega_chunk = torch.exp(log_chunk)
                denom[k_start:k_end] += torch.einsum("i,j,ijk->k", left_i, left_j_blk, omega_chunk)
        return denom

    def Omega22(self, other: "UnnormalizedBetaBasis"):
        assert isinstance(other, UnnormalizedBetaBasis), "other must be UnnormalizedBetaBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"

        a1, b1 = self.alphas_betas()
        a2, b2 = other.alphas_betas()

        a_sum = (
            a1[:, :, None, None, None]
            + a1[:, None, :, None, None]
            + a2[:, None, None, :, None]
            + a2[:, None, None, None, :]
            - 3.0
        )
        b_sum = (
            b1[:, :, None, None, None]
            + b1[:, None, :, None, None]
            + b2[:, None, None, :, None]
            + b2[:, None, None, None, :]
            - 3.0
        )
        return torch.exp(self._log_beta_fn(a_sum, b_sum).sum(dim=0))

    def marginal(self, marginal_dims: tuple[int, ...], ignore_coeffs: bool = False) -> "UnnormalizedBetaBasis":
        marginal_dims = tuple(marginal_dims)
        assert all(0 <= i < self.dim() for i in marginal_dims), "marginal_dims must be in [0, d)"

        # In this codebase, marginal_dims are the coordinates to keep.
        keep_dims = marginal_dims
        integrate_dims = tuple(i for i in range(self.dim()) if i not in keep_dims)
        alpha, beta = self.alphas_betas()
        if len(integrate_dims) > 0:
            log_int_sum = self._log_beta_fn(alpha[integrate_dims, :], beta[integrate_dims, :]).sum(dim=0)
            coeffs_new = torch.exp(log_int_sum)
        else:
            coeffs_new = torch.ones(self.n_basis_functions(), dtype=self.params.dtype, device=self.params.device)
        if (not ignore_coeffs) and hasattr(self, "coeffs"):
            coeffs_new = coeffs_new * self.coeffs

        return UnnormalizedBetaBasis(
            self.params[keep_dims, :, :].detach().clone(),
            trainable=self.params.requires_grad,
            min_concentration=self.min_concentration,
            eps=self.eps,
            coeffs_init=coeffs_new.detach().clone(),
        )

    def product_basis(self, other_basis_factors : list['Basis'], ignore_coeffs : bool = False) -> "UnnormalizedBetaBasis":
        factors: list[UnnormalizedBetaBasis] = [self]
        for other in other_basis_factors:
            assert isinstance(other, UnnormalizedBetaBasis), "all factors must be UnnormalizedBetaBasis"
            assert other.dim() == self.dim(), "Basis functions must have the same dimension"
            assert abs(other.min_concentration - self.min_concentration) < 1e-12, "all factors must share min_concentration"
            factors.append(other)

        n_factors = len(factors)
        dtype = self.params.dtype
        device = self.params.device
        dim = self.dim()

        alpha_beta_terms: list[tuple[torch.Tensor, torch.Tensor]] = [basis.alphas_betas() for basis in factors]
        n_per_factor = [basis.n_basis_functions() for basis in factors]

        coeff_shape = [dim, *n_per_factor]
        alpha_sum = torch.zeros(coeff_shape, dtype=dtype, device=device)
        beta_sum = torch.zeros_like(alpha_sum)

        for k, (alpha_k, beta_k) in enumerate(alpha_beta_terms):
            view_shape = [dim] + [1] * n_factors
            view_shape[k + 1] = n_per_factor[k]
            alpha_sum = alpha_sum + alpha_k.reshape(view_shape)
            beta_sum = beta_sum + beta_k.reshape(view_shape)

        # For unnormalized Beta products:
        # prod_t x^(a_t-1)(1-x)^(b_t-1) = x^(a_new-1)(1-x)^(b_new-1)
        # with a_new = sum_t a_t - (m-1), b_new = sum_t b_t - (m-1), m=n_factors.
        shift = float(n_factors - 1)
        alpha_new = alpha_sum - shift
        beta_new = beta_sum - shift

        min_c = self.min_concentration
        alpha_new = alpha_new.clamp_min(min_c + 1e-8)
        beta_new = beta_new.clamp_min(min_c + 1e-8)

        n_total = 1
        for n in n_per_factor:
            n_total *= n

        alpha_flat = alpha_new.reshape(dim, n_total)
        beta_flat = beta_new.reshape(dim, n_total)

        # Invert alpha = softplus(raw-1) + min_concentration
        z_alpha = (alpha_flat - min_c).clamp_min(1e-12)
        z_beta = (beta_flat - min_c).clamp_min(1e-12)
        raw_alpha = 1.0 + torch.log(torch.expm1(z_alpha))
        raw_beta = 1.0 + torch.log(torch.expm1(z_beta))
        params = torch.stack([raw_alpha, raw_beta], dim=-1)

        coeffs_new = None
        if not ignore_coeffs:
            coeff_terms = []
            for basis in factors:
                if not hasattr(basis, "coeffs"):
                    coeff_terms.append(torch.ones(basis.n_basis_functions(), dtype=dtype, device=device))
                else:
                    coeff_terms.append(basis.coeffs)

            coeff_prod = torch.ones(n_per_factor, dtype=dtype, device=device)
            for k, c_k in enumerate(coeff_terms):
                view_shape = [1] * n_factors
                view_shape[k] = n_per_factor[k]
                coeff_prod = coeff_prod * c_k.reshape(view_shape)
            coeffs_new = coeff_prod.reshape(n_total)

            if all(not hasattr(basis, "coeffs") for basis in factors):
                coeffs_new = None

        return UnnormalizedBetaBasis(
            params.detach().clone(),
            trainable=self.params.requires_grad,
            min_concentration=self.min_concentration,
            eps=self.eps,
            coeffs_init=None if coeffs_new is None else coeffs_new.detach().clone(),
        )


class NFBasis(Basis, NonnegativeBasis):
    def __init__(self, dim : int, n_basis : int, n_layers : int = 5, hidden_features : int = 128, embedding_dim : int = 16, trainable : bool = True):
        super().__init__(dim, n_basis, trainable=trainable)

        self.embedding_dim = embedding_dim
        self.index_embedding = torch.nn.Embedding(num_embeddings=n_basis, embedding_dim=embedding_dim)

        transforms = []
        for _ in range(n_layers):
            transforms.append(ReversePermutation(features=dim))
            transforms.append(MaskedAffineAutoregressiveTransform(
                features=dim,
                hiddenfeatures=hidden_features,
                context_features=embedding_dim,
                num_blocks=2,
                use_residual_blocks=True,
                random_mask=False,
                activation=torch.tanh,
                dropout_probability=0.0,
                use_batch_norm=False,
            ))

        transform = CompositeTransform(transforms)
        base_dist = StandardNormal(shape=[dim])
        self.flow = Flow(transform, base_dist)

        torch.nn.init.normal_(self.index_embedding.weight, mean=0.0, std=0.05)

        self.register_buffer("indices", torch.arange(n_basis))
    
    def forward(self, y: torch.Tensor):
        index_embeddings = self.index_embedding(self.indices)
        return torch.exp(self.flow.log_prob(y, context=index_embeddings))

    def normalized(self):
        return True

    def Omega1(self):
        return torch.ones(
            self.n_basis_functions(),
            dtype=self.params.dtype,
            device=self.params.device,
        )
