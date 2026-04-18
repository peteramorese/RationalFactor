import math

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

    def set_coeffs(self, coeffs_init: torch.Tensor, trainable: bool):
        if hasattr(self, "coeffs"):
            assert coeffs_init.shape == self.coeffs.shape, "set_coeffs: shape must match existing coeffs"
            with torch.no_grad():
                self.coeffs.copy_(coeffs_init.to(device=self.coeffs.device, dtype=self.coeffs.dtype))
            return
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
    
    def product_basis(self, other_basis_factors: list["Basis"]):
        """
        Returns the broadcasted (flattened) product of the basis functions in ``other_basis_factors``.

        Implementations always attach ``coeffs`` on the returned basis (including analytic
        prefactors and factor ``coeffs``).
        """
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


class GaussianBasis(SeparableBasis, NonnegativeBasis):
    """
    Separable product of 1D normal PDFs N(x | μ, σ²).
    Optional per-basis scaling is handled by self.coeffs.
    """

    def __init__(
        self,
        params_init: torch.Tensor,
        coeffs_init: torch.Tensor | None = None,
        trainable: bool = True,
        min_std: float = 1e-5,
        block_size: int = None,
    ):
        assert params_init.shape[2] == 2, "params_init must have shape (d, n_basis, 2)"
        super().__init__(params_init, trainable=trainable, coeffs_init=coeffs_init)
        self.min_std = min_std
        self.block_size = block_size

    @classmethod
    def random_init(
        cls,
        d: int,
        n_basis: int,
        offsets: torch.Tensor = torch.zeros(2),
        min_std: float = 1e-5,
        variance: float = 1.0,
        device=None,
        block_size: int = None,
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
            min_std=min_std,
            block_size=block_size,
        )

    @classmethod
    def set_init(
        cls,
        d: int,
        n_basis: int,
        offsets: torch.Tensor = torch.zeros(2),
        min_std: float = 1e-5,
        block_size: int = None,
        coeffs_init: torch.Tensor | None = None,
    ):
        offsets = offsets.repeat(d, n_basis, 1)
        return cls(offsets, coeffs_init=coeffs_init, min_std=min_std, block_size=block_size)

    def freeze_params(self):
        coeffs = self.coeffs.detach().clone() if hasattr(self, "coeffs") else None
        return GaussianBasis(
            self.params.detach().clone(),
            coeffs_init=coeffs,
            trainable=False,
            min_std=self.min_std,
            block_size=self.block_size,
        )

    def means_stds(self):
        return self.params[..., 0], torch.nn.functional.softplus(self.params[..., 1] - 1.0) + self.min_std
    
    def forward(self, y: torch.Tensor):
        assert y.shape[1] == self.dim(), "y must have shape (n_data, d)"
        y = y[:, :, None]  # (n_data, d, n_basis)
        mu, std = self.means_stds()

        log_dim_factors = (
            -0.5 * torch.log(y.new_tensor(2.0 * torch.pi))
            - torch.log(std)
            - (y - mu) ** 2 / (2 * std ** 2)
        )
        out = torch.exp(log_dim_factors.sum(dim=1))  # (n_data, n_basis)
        if hasattr(self, "coeffs"):
            out = out * self.coeffs[None, :]
        return out

    def normalized(self):
        return not hasattr(self, "coeffs")

    def Omega1(self, ignore_coeffs: bool = False):
        out = torch.ones(
            self.n_basis_functions(),
            dtype=self.params.dtype,
            device=self.params.device,
        )
        if (not ignore_coeffs) and hasattr(self, "coeffs"):
            out = out * self.coeffs
        return out

    def Omega2(self, other: "GaussianBasis", ignore_coeffs: bool = False):
        assert isinstance(other, GaussianBasis), "other must be GaussianBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"

        # (d, n1), (d, n2)
        mu1, std1 = self.means_stds()
        mu2, std2 = other.means_stds()

        # Broadcast to (d, n1, n2)
        diff = mu1[:, :, None] - mu2[:, None, :]
        var_sum = (std1[:, :, None] ** 2) + (std2[:, None, :] ** 2)

        # log of 1D Gaussian pdf evaluated at diff with variance var_sum
        log_dim_ip = -0.5 * (torch.log(2 * torch.pi * var_sum) + (diff * diff) / var_sum)

        log_Omega = log_dim_ip.sum(dim=0)
        out = torch.exp(log_Omega)
        if not ignore_coeffs:
            if hasattr(self, "coeffs"):
                out = out * self.coeffs[:, None]
            if hasattr(other, "coeffs"):
                out = out * other.coeffs[None, :]
        return out

    def Omega3_contract(
        self,
        other1: "GaussianBasis",
        other2: "GaussianBasis",
        left_i: torch.Tensor,
        left_j: torch.Tensor,
        block_size: int | None = None,
        ignore_coeffs: bool = False,
    ):
        """
        Computes v[k] = sum_{i,j} left_i[i] * left_j[j] * Omega3[i,j,k]
        without materializing Omega3.
        """
        assert isinstance(other1, GaussianBasis), "other1 must be GaussianBasis"
        assert isinstance(other2, GaussianBasis), "other2 must be GaussianBasis"
        assert self.dim() == other1.dim() == other2.dim(), "Basis functions must have the same dimension"
        assert left_i.dim() == 1 and left_i.shape[0] == self.n_basis_functions(), "left_i has wrong shape"
        assert left_j.dim() == 1 and left_j.shape[0] == other1.n_basis_functions(), "left_j has wrong shape"

        c0 = (
            torch.ones(self.n_basis_functions(), dtype=self.params.dtype, device=self.params.device)
            if ignore_coeffs or (not hasattr(self, "coeffs"))
            else self.coeffs
        )
        c1 = (
            torch.ones(other1.n_basis_functions(), dtype=self.params.dtype, device=self.params.device)
            if ignore_coeffs or (not hasattr(other1, "coeffs"))
            else other1.coeffs
        )
        c2 = (
            torch.ones(other2.n_basis_functions(), dtype=self.params.dtype, device=self.params.device)
            if ignore_coeffs or (not hasattr(other2, "coeffs"))
            else other2.coeffs
        )
        coeff_scale = c0[:, None, None] * c1[None, :, None] * c2[None, None, :]

        mu0, std0 = self.means_stds()
        mu1, std1 = other1.means_stds()
        mu2, std2 = other2.means_stds()

        var0 = std0.square()
        var1 = std1.square()
        var2 = std2.square()

        inv0 = var0.reciprocal()
        inv1 = var1.reciprocal()
        inv2 = var2.reciprocal()

        n0 = self.n_basis_functions()
        n1 = other1.n_basis_functions()
        n2 = other2.n_basis_functions()

        log2pi = torch.log(mu0.new_tensor(2.0 * torch.pi))

        if block_size is None:
            block_size = self.block_size
        if block_size is None:
            # Full vectorized path (equivalent to building Omega3 then contracting).
            mu_i = mu0[:, :, None, None]
            mu_j = mu1[:, None, :, None]
            mu_k = mu2[:, None, None, :]

            inv_i = inv0[:, :, None, None]
            inv_j = inv1[:, None, :, None]
            inv_k = inv2[:, None, None, :]

            var_i = var0[:, :, None, None]
            var_j = var1[:, None, :, None]
            var_k = var2[:, None, None, :]

            S = inv_i + inv_j + inv_k
            T = mu_i * inv_i + mu_j * inv_j + mu_k * inv_k
            U = mu_i.square() * inv_i + mu_j.square() * inv_j + mu_k.square() * inv_k

            log_dim = (
                -0.5 * (3.0 * log2pi + torch.log(var_i) + torch.log(var_j) + torch.log(var_k))
                + 0.5 * (log2pi - torch.log(S))
                - 0.5 * (U - T.square() / S)
            )
            omega_full = torch.exp(log_dim.sum(dim=0)) * coeff_scale
            return torch.einsum("i,j,ijk->k", left_i, left_j, omega_full)

        assert block_size > 0, "block_size must be positive"
        denom = torch.zeros(n2, dtype=mu0.dtype, device=mu0.device)

        for j_start in range(0, n1, block_size):
            j_end = min(j_start + block_size, n1)
            left_j_blk = left_j[j_start:j_end]
            for k_start in range(0, n2, block_size):
                k_end = min(k_start + block_size, n2)
                log_chunk = torch.zeros((n0, j_end - j_start, k_end - k_start), dtype=mu0.dtype, device=mu0.device)
                for r in range(self.dim()):
                    mu_i = mu0[r, :, None, None]
                    mu_j = mu1[r, None, j_start:j_end, None]
                    mu_k = mu2[r, None, None, k_start:k_end]

                    inv_i = inv0[r, :, None, None]
                    inv_j = inv1[r, None, j_start:j_end, None]
                    inv_k = inv2[r, None, None, k_start:k_end]

                    var_i = var0[r, :, None, None]
                    var_j = var1[r, None, j_start:j_end, None]
                    var_k = var2[r, None, None, k_start:k_end]

                    S = inv_i + inv_j + inv_k
                    T = mu_i * inv_i + mu_j * inv_j + mu_k * inv_k
                    U = mu_i.square() * inv_i + mu_j.square() * inv_j + mu_k.square() * inv_k

                    log_chunk += (
                        -0.5 * (3.0 * log2pi + torch.log(var_i) + torch.log(var_j) + torch.log(var_k))
                        + 0.5 * (log2pi - torch.log(S))
                        - 0.5 * (U - T.square() / S)
                    )

                omega_chunk = torch.exp(log_chunk) * coeff_scale[:, j_start:j_end, k_start:k_end]
                denom[k_start:k_end] += torch.einsum("i,j,ijk->k", left_i, left_j_blk, omega_chunk)

        return denom

    def Omega22(self, other: "GaussianBasis", ignore_coeffs: bool = False):
        assert isinstance(other, GaussianBasis), "other must be GaussianBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"

        # (d, n_phi), (d, n_psi)
        mu1, std1 = self.means_stds()
        mu2, std2 = other.means_stds()

        var1 = std1 * std1
        var2 = std2 * std2
        inv_var1 = 1.0 / var1
        inv_var2 = 1.0 / var2

        # Broadcast everything to (d, n_phi, n_phi, n_psi, n_psi)
        mu_i = mu1[:, :, None, None, None]
        mu_j = mu1[:, None, :, None, None]
        mu_k = mu2[:, None, None, :, None]
        mu_l = mu2[:, None, None, None, :]

        inv_i = inv_var1[:, :, None, None, None]
        inv_j = inv_var1[:, None, :, None, None]
        inv_k = inv_var2[:, None, None, :, None]
        inv_l = inv_var2[:, None, None, None, :]

        var_i = var1[:, :, None, None, None]
        var_j = var1[:, None, :, None, None]
        var_k = var2[:, None, None, :, None]
        var_l = var2[:, None, None, None, :]

        S = inv_i + inv_j + inv_k + inv_l
        T = mu_i * inv_i + mu_j * inv_j + mu_k * inv_k + mu_l * inv_l
        U = mu_i.square() * inv_i + mu_j.square() * inv_j + mu_k.square() * inv_k + mu_l.square() * inv_l

        two_pi = mu1.new_tensor(2.0 * torch.pi)
        log2pi = torch.log(two_pi)

        log_pref = -0.5 * (4.0 * log2pi + torch.log(var_i) + torch.log(var_j) + torch.log(var_k) + torch.log(var_l))
        log_gauss_int = 0.5 * (log2pi - torch.log(S))

        quad = -0.5 * (U - (T * T) / S)

        log_dim = log_pref + log_gauss_int + quad              # (d, nf, nf, ng, ng)
        log_Omega = log_dim.sum(dim=0)                         # (nf, nf, ng, ng)

        out = torch.exp(log_Omega)
        if not ignore_coeffs:
            c_self = (
                torch.ones(mu1.shape[1], dtype=mu1.dtype, device=mu1.device)
                if not hasattr(self, "coeffs")
                else self.coeffs
            )
            c_other = (
                torch.ones(mu2.shape[1], dtype=mu2.dtype, device=mu2.device)
                if not hasattr(other, "coeffs")
                else other.coeffs
            )
            out = out * (c_self[:, None, None, None] * c_self[None, :, None, None])
            out = out * (c_other[None, None, :, None] * c_other[None, None, None, :])
        return out

    def marginal(self, marginal_dims: tuple[int, ...], ignore_coeffs: bool = False) -> "GaussianBasis":
        marginal_dims = tuple(marginal_dims)
        assert all(0 <= i < self.dim() for i in marginal_dims), "marginal_dims must be in [0, d)"
        coeffs_out = None
        if (not ignore_coeffs) and hasattr(self, "coeffs"):
            coeffs_out = self.coeffs.detach().clone()
        return GaussianBasis(
            self.params[marginal_dims, :, :].detach().clone(),
            coeffs_init=coeffs_out,
            trainable=self.params.requires_grad,
            min_std=self.min_std,
            block_size=self.block_size,
        )

    def product_basis(self, other_basis_factors: list["Basis"]) -> "GaussianBasis":
        """
        Cartesian product of factor bases along the flattened product index.

        Each factor is a *normalized* separable Gaussian PDF. Their pointwise product is
        proportional to the Gaussian described by the returned ``params``. The ratio
        ``∏_k N_k / N_*`` (per axis, then multiplied across axes) and any factor ``coeffs``
        are stored on the returned basis as ``coeffs`` (including the all-ones case when the
        prefactor is trivial).
        """
        factors: list[GaussianBasis] = [self]
        for other in other_basis_factors:
            assert isinstance(other, GaussianBasis), "all factors must be GaussianBasis"
            assert other.dim() == self.dim(), "Basis functions must have the same dimension"
            factors.append(other)

        n_factors = len(factors)
        dtype = self.params.dtype
        device = self.params.device
        dim = self.dim()

        mus_stds = [basis.means_stds() for basis in factors]
        n_per_factor = [basis.n_basis_functions() for basis in factors]

        coeff_shape = [dim, *n_per_factor]
        tau_sum = torch.zeros(coeff_shape, dtype=dtype, device=device)
        mu_tau_sum = torch.zeros_like(tau_sum)
        mu_sq_tau_sum = torch.zeros_like(tau_sum)
        log_std_sum = torch.zeros_like(tau_sum)

        log2pi = torch.log(torch.tensor(2.0 * torch.pi, dtype=dtype, device=device))

        for k, (mu_k, std_k) in enumerate(mus_stds):
            inv_var = 1.0 / (std_k * std_k)
            view_shape = [dim] + [1] * n_factors
            view_shape[k + 1] = n_per_factor[k]

            mu_b = mu_k.reshape(view_shape)
            inv_b = inv_var.reshape(view_shape)
            std_b = std_k.reshape(view_shape)

            tau_sum = tau_sum + inv_b
            mu_tau_sum = mu_tau_sum + mu_b * inv_b
            mu_sq_tau_sum = mu_sq_tau_sum + (mu_b * mu_b) * inv_b
            log_std_sum = log_std_sum + torch.log(std_b)

        n_total = 1
        for n in n_per_factor:
            n_total *= n

        mu_star = mu_tau_sum / tau_sum
        surplus = mu_sq_tau_sum - tau_sum * mu_star.square()
        log_const_dim = (
            -0.5 * float(n_factors - 1) * log2pi
            - log_std_sum
            - 0.5 * torch.log(tau_sum)
            - 0.5 * surplus
        )
        log_const = log_const_dim.sum(dim=0)

        sigma_star = torch.sqrt(1.0 / tau_sum)

        mu_flat = mu_star.reshape(dim, n_total)
        sigma_flat = sigma_star.reshape(dim, n_total)
        log_const_flat = log_const.reshape(n_total)

        std_inner = (sigma_flat - self.min_std).clamp_min(1e-12)
        raw_std = torch.log(torch.expm1(std_inner)) + 1.0
        params = torch.stack([mu_flat, raw_std], dim=-1)

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
        coeffs_new = (coeff_prod * torch.exp(log_const_flat.reshape(n_per_factor))).reshape(n_total)

        return GaussianBasis(
            params.detach().clone(),
            coeffs_init=coeffs_new.detach().clone(),
            trainable=self.params.requires_grad,
            min_std=self.min_std,
            block_size=self.block_size,
        )


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
        finfo = torch.finfo(a.dtype)

        # --- compute log integral contribution from marginalized dims ---
        if len(integrate_dims) > 0:
            a_m = a[integrate_dims, :]
            b_m = b[integrate_dims, :]
            # Stay away from a → 0^- where -(b^2)/(4a) and log(-a) are singular.
            a_m = torch.clamp(a_m, max=-self.eps)

            log_pi = torch.log(torch.tensor(torch.pi, dtype=a.dtype, device=a.device))

            log_int_m = (
                -(b_m * b_m) / (4.0 * a_m)
                + 0.5 * (log_pi - torch.log(-a_m))
            )  # (|M|, n_basis)

            log_int_sum = log_int_m.sum(dim=0)  # (n_basis,)
            log_int_sum = torch.clamp(log_int_sum, max=0.99 * math.log(finfo.max))
        else:
            log_int_sum = torch.zeros(
                self.n_basis_functions(),
                dtype=a.dtype,
                device=a.device,
            )

        coeffs_new = torch.exp(log_int_sum)
        if (not ignore_coeffs) and hasattr(self, "coeffs"):
            coeffs_new = coeffs_new * self.coeffs
        coeffs_new = torch.nan_to_num(coeffs_new, nan=0.0, posinf=finfo.max, neginf=0.0)

        # Slice stored parameters on kept axes. Re-inverting a_k -> raw_a can drift in
        # float32 so (a,b) on kept dims no longer match coeffs_new from the true integral.
        params = self.params[keep_dims, :, :].detach().clone()

        return QuadraticExpBasis(
            params,
            coeffs_init=coeffs_new.detach().clone(),
            trainable=self.params.requires_grad,
            eps=self.eps,
        )
    
    def product_basis(self, other_basis_factors: list["Basis"]) -> "QuadraticExpBasis":
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
        print("coeffs new is nan: ", torch.isnan(coeffs_new).any())

        return QuadraticExpBasis(
            params.detach().clone(),
            coeffs_init=coeffs_new.detach().clone(),
            trainable=self.params.requires_grad,
            eps=self.eps,
        )
    

class BetaBasis(SeparableBasis, NonnegativeBasis):
    """
    Separable product of 1D Beta PDFs on (0, 1).
    Optional per-basis scaling is handled by self.coeffs.
    """

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
        super().__init__(params_init, trainable=trainable, coeffs_init=coeffs_init)
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
        device=None,
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
        device=None,
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
        return BetaBasis(
            self.params.detach().clone(),
            coeffs_init=coeffs,
            trainable=False,
            min_concentration=self.min_concentration,
            eps=self.eps,
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

        #y_input = y.clone()

        y = y.clamp(self.eps, 1.0 - self.eps)
        y = y[:, :, None]  # (n_data, d, n_basis)

        alpha, beta = self.alphas_betas()  # (d, n_basis), (d, n_basis)

        log_dim_factors = (
            (alpha - 1.0) * torch.log(y)
            + (beta - 1.0) * torch.log1p(-y)
            - self._log_beta_fn(alpha, beta)
        )

        out = torch.exp(log_dim_factors.sum(dim=1))  # (n_data, n_basis)
        if hasattr(self, "coeffs"):
            out = out * self.coeffs[None, :]
        return out

    def normalized(self):
        return not hasattr(self, "coeffs")

    def Omega1(self, ignore_coeffs: bool = False):
        out = torch.ones(
            self.n_basis_functions(),
            dtype=self.params.dtype,
            device=self.params.device,
        )
        if (not ignore_coeffs) and hasattr(self, "coeffs"):
            out = out * self.coeffs
        return out

    def Omega2(self, other: "BetaBasis", ignore_coeffs: bool = False):
        assert isinstance(other, BetaBasis), "other must be BetaBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"

        a1, b1 = self.alphas_betas()   # (d, n1)
        a2, b2 = other.alphas_betas()  # (d, n2)

        # Broadcast to (d, n1, n2)
        a_sum = a1[:, :, None] + a2[:, None, :] - 1.0
        b_sum = b1[:, :, None] + b2[:, None, :] - 1.0

        log_dim_ip = (
            self._log_beta_fn(a_sum, b_sum)
            - self._log_beta_fn(a1[:, :, None], b1[:, :, None])
            - self._log_beta_fn(a2[:, None, :], b2[:, None, :])
        )

        log_Omega = log_dim_ip.sum(dim=0)  # (n1, n2)
        out = torch.exp(log_Omega)
        if not ignore_coeffs:
            if hasattr(self, "coeffs"):
                out = out * self.coeffs[:, None]
            if hasattr(other, "coeffs"):
                out = out * other.coeffs[None, :]
        return out

    def Omega3_contract(
        self,
        other1: "BetaBasis",
        other2: "BetaBasis",
        left_i: torch.Tensor,
        left_j: torch.Tensor,
        block_size: int | None = None,
        ignore_coeffs: bool = False,
    ):
        """
        Computes v[k] = sum_{i,j} left_i[i] * left_j[j] * Omega3[i,j,k]
        without materializing Omega3.
        """
        assert isinstance(other1, BetaBasis), "other1 must be BetaBasis"
        assert isinstance(other2, BetaBasis), "other2 must be BetaBasis"
        assert self.dim() == other1.dim() == other2.dim(), "Basis functions must have the same dimension"
        assert left_i.dim() == 1 and left_i.shape[0] == self.n_basis_functions(), "left_i has wrong shape"
        assert left_j.dim() == 1 and left_j.shape[0] == other1.n_basis_functions(), "left_j has wrong shape"

        c0 = (
            torch.ones(self.n_basis_functions(), dtype=self.params.dtype, device=self.params.device)
            if ignore_coeffs or (not hasattr(self, "coeffs"))
            else self.coeffs
        )
        c1 = (
            torch.ones(other1.n_basis_functions(), dtype=self.params.dtype, device=self.params.device)
            if ignore_coeffs or (not hasattr(other1, "coeffs"))
            else other1.coeffs
        )
        c2 = (
            torch.ones(other2.n_basis_functions(), dtype=self.params.dtype, device=self.params.device)
            if ignore_coeffs or (not hasattr(other2, "coeffs"))
            else other2.coeffs
        )
        coeff_scale = c0[:, None, None] * c1[None, :, None] * c2[None, None, :]

        a0, b0 = self.alphas_betas()
        a1, b1 = other1.alphas_betas()
        a2, b2 = other2.alphas_betas()

        n0 = self.n_basis_functions()
        n1 = other1.n_basis_functions()
        n2 = other2.n_basis_functions()

        if block_size is None:
            a_i = a0[:, :, None, None]
            a_j = a1[:, None, :, None]
            a_k = a2[:, None, None, :]
            b_i = b0[:, :, None, None]
            b_j = b1[:, None, :, None]
            b_k = b2[:, None, None, :]
            a_sum = a_i + a_j + a_k - 2.0
            b_sum = b_i + b_j + b_k - 2.0
            log_dim = (
                self._log_beta_fn(a_sum, b_sum)
                - self._log_beta_fn(a_i, b_i)
                - self._log_beta_fn(a_j, b_j)
                - self._log_beta_fn(a_k, b_k)
            )
            omega_full = torch.exp(log_dim.sum(dim=0)) * coeff_scale
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
                    a_i = a0[r, :, None, None]
                    a_j = a1[r, None, j_start:j_end, None]
                    a_k = a2[r, None, None, k_start:k_end]

                    b_i = b0[r, :, None, None]
                    b_j = b1[r, None, j_start:j_end, None]
                    b_k = b2[r, None, None, k_start:k_end]

                    a_sum = a_i + a_j + a_k - 2.0
                    b_sum = b_i + b_j + b_k - 2.0

                    log_chunk += (
                        self._log_beta_fn(a_sum, b_sum)
                        - self._log_beta_fn(a_i, b_i)
                        - self._log_beta_fn(a_j, b_j)
                        - self._log_beta_fn(a_k, b_k)
                    )

                omega_chunk = torch.exp(log_chunk) * coeff_scale[:, j_start:j_end, k_start:k_end]
                denom[k_start:k_end] += torch.einsum("i,j,ijk->k", left_i, left_j_blk, omega_chunk)

        return denom

    def Omega22(self, other: "BetaBasis", ignore_coeffs: bool = False):
        assert isinstance(other, BetaBasis), "other must be BetaBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"

        a1, b1 = self.alphas_betas()   # (d, n_phi)
        a2, b2 = other.alphas_betas()  # (d, n_psi)

        # Broadcast to (d, n_phi, n_phi, n_psi, n_psi)
        a_i = a1[:, :, None, None, None]
        a_j = a1[:, None, :, None, None]
        a_k = a2[:, None, None, :, None]
        a_l = a2[:, None, None, None, :]

        b_i = b1[:, :, None, None, None]
        b_j = b1[:, None, :, None, None]
        b_k = b2[:, None, None, :, None]
        b_l = b2[:, None, None, None, :]

        a_sum = a_i + a_j + a_k + a_l - 3.0
        b_sum = b_i + b_j + b_k + b_l - 3.0

        log_dim = (
            self._log_beta_fn(a_sum, b_sum)
            - self._log_beta_fn(a_i, b_i)
            - self._log_beta_fn(a_j, b_j)
            - self._log_beta_fn(a_k, b_k)
            - self._log_beta_fn(a_l, b_l)
        )

        log_Omega = log_dim.sum(dim=0)  # (n_phi, n_phi, n_psi, n_psi)
        out = torch.exp(log_Omega)
        if not ignore_coeffs:
            c_self = (
                torch.ones(a1.shape[1], dtype=a1.dtype, device=a1.device)
                if not hasattr(self, "coeffs")
                else self.coeffs
            )
            c_other = (
                torch.ones(a2.shape[1], dtype=a2.dtype, device=a2.device)
                if not hasattr(other, "coeffs")
                else other.coeffs
            )
            out = out * (c_self[:, None, None, None] * c_self[None, :, None, None])
            out = out * (c_other[None, None, :, None] * c_other[None, None, None, :])
        return out

    def marginal(self, marginal_dims: tuple[int, ...], ignore_coeffs: bool = False) -> "BetaBasis":
        marginal_dims = tuple(marginal_dims)
        assert all(0 <= i < self.dim() for i in marginal_dims), "marginal_dims must be in [0, d)"

        coeffs_out = None
        if (not ignore_coeffs) and hasattr(self, "coeffs"):
            coeffs_out = self.coeffs.detach().clone()

        return BetaBasis(
            self.params[marginal_dims, :, :].detach().clone(),
            coeffs_init=coeffs_out,
            trainable=self.params.requires_grad,
            min_concentration=self.min_concentration,
            eps=self.eps,
        )

    def product_basis(self, other_basis_factors: list["Basis"]) -> "BetaBasis":
        """
        Cartesian product of factor bases along the flattened product index.

        Each factor is a *normalized* separable Beta PDF. The product is proportional to the
        Beta PDF described by the returned ``params`` (merged ``α``, ``β`` as for
        ``UnnormalizedBetaBasis.product_basis``). The ratio ``B(α_*,β_*) / ∏_k B(α_k,β_k)``
        over axes, times any factor ``coeffs``, is stored as ``coeffs`` (all-ones when only
        trivial prefactors apply).
        """
        factors: list["BetaBasis"] = [self]
        for other in other_basis_factors:
            assert isinstance(other, BetaBasis), "all factors must be BetaBasis"
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

        shift = float(n_factors - 1)
        alpha_new = alpha_sum - shift
        beta_new = beta_sum - shift

        min_c = self.min_concentration
        alpha_new = alpha_new.clamp_min(min_c + 1e-8)
        beta_new = beta_new.clamp_min(min_c + 1e-8)

        log_b_new = self._log_beta_fn(alpha_new, beta_new)
        log_b_factors = torch.zeros_like(log_b_new)
        for k, (alpha_k, beta_k) in enumerate(alpha_beta_terms):
            view_shape = [dim] + [1] * n_factors
            view_shape[k + 1] = n_per_factor[k]
            log_b_factors = log_b_factors + self._log_beta_fn(alpha_k.reshape(view_shape), beta_k.reshape(view_shape))
        log_const_dim = log_b_new - log_b_factors

        n_total = 1
        for n in n_per_factor:
            n_total *= n

        alpha_flat = alpha_new.reshape(dim, n_total)
        beta_flat = beta_new.reshape(dim, n_total)

        z_alpha = (alpha_flat - min_c).clamp_min(1e-12)
        z_beta = (beta_flat - min_c).clamp_min(1e-12)
        raw_alpha = 1.0 + torch.log(torch.expm1(z_alpha))
        raw_beta = 1.0 + torch.log(torch.expm1(z_beta))
        params = torch.stack([raw_alpha, raw_beta], dim=-1)

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

        log_const_flat = log_const_dim.sum(dim=0).reshape(n_total)
        intrinsic = torch.exp(log_const_flat.reshape(n_per_factor))
        coeffs_new = (coeff_prod * intrinsic).reshape(n_total)

        return BetaBasis(
            params.detach().clone(),
            coeffs_init=coeffs_new.detach().clone(),
            trainable=self.params.requires_grad,
            min_concentration=self.min_concentration,
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

    def product_basis(self, other_basis_factors: list["Basis"]) -> "UnnormalizedBetaBasis":
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

        return UnnormalizedBetaBasis(
            params.detach().clone(),
            trainable=self.params.requires_grad,
            min_concentration=self.min_concentration,
            eps=self.eps,
            coeffs_init=coeffs_new.detach().clone(),
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
