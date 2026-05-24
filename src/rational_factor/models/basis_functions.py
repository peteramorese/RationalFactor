import math

import torch
from abc import abstractmethod

from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform

class Parameters(torch.nn.Module):
    def __init__(self, trainable_init_values: torch.Tensor = None, fixed_values: torch.Tensor = None):
        super().__init__()
        assert not (trainable_init_values is not None and fixed_values is not None)
        self._trainable = trainable_init_values is not None
        if self._trainable:
            self._p = torch.nn.Parameter(trainable_init_values)
        else:
            self.register_buffer("_p", fixed_values)

    @classmethod
    def random_init(
        cls,
        shape: tuple[int, ...],
        trainable: bool = True,
        mean: float = 0.0,
        std: float = 1.0,
    ):
        values = torch.randn(*shape) * std + mean
        if trainable:
            return cls(trainable_init_values=values)
        return cls(fixed_values=values)

    @classmethod
    def set_init(cls, shape: tuple[int, ...], value: float, trainable: bool = True):
        values = torch.ones(shape) * value
        if trainable:
            return cls(trainable_init_values=values)
        return cls(fixed_values=values)

    @classmethod
    def from_values(cls, values: torch.Tensor, trainable: bool = True):
        if trainable:
            return cls(trainable_init_values=values)
        return cls(fixed_values=values.detach().clone())

    def size(self):
        return self._p.size()

    def is_trainable(self):
        return self._trainable

    def dtype_device(self):
        return self._p.dtype, self._p.device

    def forward(self):
        return self._p

    def freeze_params(self):
        return self.__class__(fixed_values=self.forward().detach().clone())


class PositiveParameters(Parameters):
    def __init__(
        self,
        trainable_init_values: torch.Tensor = None,
        fixed_values: torch.Tensor = None,
        normalized: bool = True,
        epsilon: float = 0.0,
    ):
        super().__init__(trainable_init_values=trainable_init_values, fixed_values=fixed_values)
        self._normalized = normalized
        self._epsilon = epsilon
        if not self._trainable:
            assert torch.all(fixed_values >= 0), "fixed_values must be nonnegative"
            if normalized:
                self._p = self._normalize(self._p)
    
    @classmethod
    def random_init(
        cls,
        shape: tuple[int, ...],
        trainable: bool = True,
        mean: float = 0.0,
        std: float = 1.0,
        normalized: bool = True,
        epsilon: float = 0.0,
    ):
        values = torch.randn(*shape) * std + mean
        if trainable:
            return cls(trainable_init_values=values, normalized=normalized, epsilon=epsilon)
        return cls(fixed_values=values, normalized=normalized, epsilon=epsilon)

    @classmethod
    def set_init(cls, shape: tuple[int, ...], value: float, trainable: bool = True, normalized: bool = True, epsilon: float = 0.0):
        values = torch.ones(shape) * value
        if trainable:
            return cls(trainable_init_values=values, normalized=normalized, epsilon=epsilon)
        return cls(fixed_values=values, normalized=normalized, epsilon=epsilon)

    @classmethod
    def from_values(cls, values: torch.Tensor, trainable: bool = True, normalized: bool = True, epsilon: float = 0.0):
        if trainable:
            return cls(trainable_init_values=values, normalized=normalized, epsilon=epsilon)
        return cls(fixed_values=values.detach().clone(), normalized=normalized, epsilon=epsilon)

    def _normalize(self, p: torch.Tensor, dim: int = 0):
        return self._epsilon + (1.0 - p.shape[dim] * self._epsilon) * torch.nn.functional.softmax(p, dim=dim)

    def forward(self):
        if self._trainable:
            if self._normalized:
                return self._normalize(self._p)
            return self._epsilon + torch.nn.functional.softplus(self._p)
        return self._p

    def freeze_params(self):
        return PositiveParameters(
            fixed_values=self.forward().detach().clone(),
            normalized=self._normalized,
            epsilon=self._epsilon,
        )

    def with_fixed_values(self, values: torch.Tensor) -> "PositiveParameters":
        return PositiveParameters(
            fixed_values=values.detach().clone(),
            normalized=self._normalized,
            epsilon=self._epsilon,
        )

    def is_normalized(self):
        return self._normalized


def _is_functional_proxy_of(obj, basis_cls: type) -> bool:
    """True if ``obj`` is a basis instance or a FunctionalModuleProxy wrapping one."""
    if isinstance(obj, basis_cls):
        return True
    target = getattr(obj, "_target_module", None)
    return isinstance(target, basis_cls)


class Basis(torch.nn.Module):
    def __init__(self, 
            dim : int, 
            n_basis : int,
            params : tuple[torch.Tensor, ...] | tuple[Parameters, ...],
            coeffs : torch.Tensor | Parameters = None):
        '''
        Args:
            dim : int, number of dimensions
            n_basis : int, number of basis functions
            params : list of tensors where each represents a parameter group for a basis function
            coeffs : PositiveParameters of coefficients
        '''
        super().__init__()
        self._dim = dim
        self._n_basis = n_basis

        self.set_params(params)

        if coeffs is not None:
            assert coeffs.size() == (n_basis,), "coeffs must have shape (n_basis,)"
            self.set_coeffs(coeffs)
        else:
            self.coeffs = None
            self._module_coeffs = False

    def forward(self, y : torch.Tensor, ignore_coeffs : bool = False):
        raise NotImplementedError("forward is not implemented for this basis function")

    def set_params(self, params : tuple[torch.Tensor, ...] | tuple[Parameters, ...]):
        if isinstance(params[0], torch.Tensor):
            self._params = params
            self._module_params = False
        elif isinstance(params[0], Parameters):
            self._params = torch.nn.ModuleList(params)
            self._module_params = True
        else:
            raise ValueError("params must be a tuple of tensors or Parameters")

    def get_params(self):
        if self._module_params:
            return [param() for param in self._params]
        return self._params

    def has_coeffs(self):
        return self.coeffs is not None

    def set_coeffs(self, coeffs : torch.Tensor | Parameters):
        if isinstance(coeffs, torch.Tensor):
            self.coeffs = coeffs
            self._module_coeffs = False
        elif isinstance(coeffs, Parameters):
            self.add_module("coeffs", coeffs)
            self._module_coeffs = True
        else:
            raise ValueError("coeffs must be a tensor or Parameters")
        assert coeffs.size() == (self._n_basis,)
    
    def get_coeffs(self):
        if not self.has_coeffs():
            raise ValueError("Basis has no coefficients.")
        if self._module_coeffs:
            return self.coeffs()
        return self.coeffs

    def coeff_values(self):
        return self.get_coeffs()

    def dim(self):
        return self._dim
    
    def n_basis_functions(self):
        return self._n_basis
    
    def param_dtype_device(self):
        params = self.get_params()
        if len(params) > 0:
            return params[0].dtype(), params[0].device()
        return None, None

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

    def Omega2(self, other: 'Basis', ignore_coeffs : bool = False, lows : torch.Tensor = None, highs : torch.Tensor = None):
        '''
        Computes the function inner product matrix with another basis function vector. 
        omega[i, j] = <this_i, other_j>

        Args:
            other : Basis function to compute the inner product with
            ignore_coeffs : whether to ignore the coefficients
            lows : lower bounds of the domain, if None, the domain is the entire real line
            highs : upper bounds of the domain, if None, the domain is the entire real line
        
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
    def __init__(
        self,
        params: tuple[torch.Tensor, ...] | tuple[Parameters, ...],
        coeffs: torch.Tensor = None,
    ):
        param_tensor = params[0] if isinstance(params[0], torch.Tensor) else params[0]()
        dim = param_tensor.size()[0]
        n_basis = param_tensor.size()[1]
        super().__init__(dim=dim, n_basis=n_basis, params=params, coeffs=coeffs)

        self_params = self.get_params()
        for param_set in self_params:
            assert param_set.size() == (dim, n_basis), "Each parameter tensor must have shape (dim, n_basis)"

    def n_params_per_basis(self):
        return len(self.get_params())


# Nonnegative basis functions 
class NonnegativeBasis:
    pass


class GaussianBasis(SeparableBasis, NonnegativeBasis):
    """Separable product of 1D normal PDFs N(x | mean, std^2)."""

    def __init__(
        self,
        mean_params : torch.Tensor | Parameters,
        std_params : torch.Tensor | PositiveParameters,
        coeffs: torch.Tensor | PositiveParameters = None,
        block_size: int | None = None,
    ):
        super().__init__(params=(mean_params, std_params), coeffs=coeffs)
        self.block_size = block_size

    def means_stds(self) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.get_params()
        return params[0], params[1]

    @staticmethod
    def omega2_from_means_stds(
        mu1: torch.Tensor,
        std1: torch.Tensor,
        mu2: torch.Tensor,
        std2: torch.Tensor,
        ignore_coeffs: bool = False,
        coeffs1: torch.Tensor | None = None,
        coeffs2: torch.Tensor | None = None,
        lows: torch.Tensor | None = None,
        highs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inner product matrix Omega2 from mean/std tensors, optionally batched over leading dim."""
        if mu1.dim() == 2:
            mu1_b = mu1[:, :, None]
            std1_b = std1[:, :, None]
            mu2_b = mu2[:, None, :]
            std2_b = std2[:, None, :]
            sum_dim = 0
        elif mu1.dim() == 3:
            mu1_b = mu1[:, :, :, None]
            std1_b = std1[:, :, :, None]
            mu2_b = mu2[:, :, None, :]
            std2_b = std2[:, :, None, :]
            sum_dim = 1
        else:
            raise ValueError(
                f"means/stds must have shape (d, n_basis) or (batch, d, n_basis), got ndim={mu1.dim()}."
            )

        lows_b, highs_b = None, None
        if lows is not None or highs is not None:
            assert lows is not None and highs is not None, "lows and highs must both be set for a bounded domain"
            dtype, device = mu1.dtype, mu1.device
            lows_b = lows.to(dtype=dtype, device=device).reshape(-1)
            highs_b = highs.to(dtype=dtype, device=device).reshape(-1)
            if mu1.dim() == 2:
                lows_b = lows_b[:, None, None]
                highs_b = highs_b[:, None, None]
            else:
                lows_b = lows_b[None, :, None, None]
                highs_b = highs_b[None, :, None, None]

        log_dim = GaussianBasis._log_gaussian_pair_ip(mu1_b, std1_b, mu2_b, std2_b, lows_b, highs_b)
        out = torch.exp(log_dim.sum(dim=sum_dim))
        if not ignore_coeffs:
            if coeffs1 is not None:
                out = out * (coeffs1[:, None] if coeffs1.dim() == 1 else coeffs1[:, :, None])
            if coeffs2 is not None:
                out = out * (coeffs2[None, :] if coeffs2.dim() == 1 else coeffs2[:, None, :])
        return out

    @staticmethod
    def _log_gaussian_pair_ip(
        mu1: torch.Tensor,
        std1: torch.Tensor,
        mu2: torch.Tensor,
        std2: torch.Tensor,
        lows: torch.Tensor | None,
        highs: torch.Tensor | None,
    ) -> torch.Tensor:
        var_sum = std1.square() + std2.square()
        log_pref = -0.5 * (torch.log(2 * torch.pi * var_sum) + (mu1 - mu2).square() / var_sum)
        if lows is None:
            return log_pref
        m = (mu1 * std2.square() + mu2 * std1.square()) / var_sum
        std_m = torch.sqrt(std1.square() * std2.square() / var_sum)
        cdf = torch.special.ndtr((highs - m) / std_m) - torch.special.ndtr((lows - m) / std_m)
        return log_pref + cdf.clamp_min(torch.finfo(mu1.dtype).tiny).log()

    def forward(self, y: torch.Tensor, ignore_coeffs: bool = False):
        assert y.dim() == 2 and y.shape[1] == self.dim(), "y must have shape (n_data, d)"
        mu, std = self.means_stds()
        y = y[:, :, None]
        log_dim_factors = (
            -0.5 * torch.log(y.new_tensor(2.0 * torch.pi))
            - torch.log(std)
            - (y - mu) ** 2 / (2 * std ** 2)
        )
        out = torch.exp(log_dim_factors.sum(dim=1))
        if not ignore_coeffs and self.has_coeffs():
            coeffs = self.get_coeffs()
            if coeffs.dim() == 1:
                out = out * coeffs[None, :]
            else:
                out = out * coeffs
        return out

    def normalized(self):
        return not self.has_coeffs()

    def Omega1(self, ignore_coeffs: bool = False):
        dtype, device = self.param_dtype_device()
        out = torch.ones(
            self.n_basis_functions(),
            dtype=dtype,
            device=device,
        )
        if (not ignore_coeffs) and self.has_coeffs():
            out = out * self.get_coeffs()
        return out

    def Omega2(self, other: "GaussianBasis", ignore_coeffs: bool = False, lows: torch.Tensor | None = None, highs: torch.Tensor | None = None):
        assert _is_functional_proxy_of(other, GaussianBasis), "other must be GaussianBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"
        mu1, std1 = self.means_stds()
        mu2, std2 = other.means_stds()
        coeffs1 = None if ignore_coeffs or (not self.has_coeffs()) else self.get_coeffs()
        coeffs2 = None if ignore_coeffs or (not other.has_coeffs()) else other.get_coeffs()
        return self.omega2_from_means_stds(
            mu1, std1, mu2, std2,
            ignore_coeffs=ignore_coeffs,
            coeffs1=coeffs1,
            coeffs2=coeffs2,
            lows=lows,
            highs=highs,
        )

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
        assert _is_functional_proxy_of(other1, GaussianBasis), "other1 must be GaussianBasis"
        assert _is_functional_proxy_of(other2, GaussianBasis), "other2 must be GaussianBasis"
        assert self.dim() == other1.dim() == other2.dim(), "Basis functions must have the same dimension"
        assert left_i.dim() == 1 and left_i.shape[0] == self.n_basis_functions(), "left_i has wrong shape"
        assert left_j.dim() == 1 and left_j.shape[0] == other1.n_basis_functions(), "left_j has wrong shape"

        mu0, std0 = self.means_stds()
        mu1, std1 = other1.means_stds()
        mu2, std2 = other2.means_stds()

        c0 = (
            torch.ones(self.n_basis_functions(), dtype=mu0.dtype, device=mu0.device)
            if ignore_coeffs or (not self.has_coeffs())
            else self.get_coeffs()
        )
        c1 = (
            torch.ones(other1.n_basis_functions(), dtype=mu0.dtype, device=mu0.device)
            if ignore_coeffs or (not other1.has_coeffs())
            else other1.get_coeffs()
        )
        c2 = (
            torch.ones(other2.n_basis_functions(), dtype=mu0.dtype, device=mu0.device)
            if ignore_coeffs or (not other2.has_coeffs())
            else other2.get_coeffs()
        )
        coeff_scale = c0[:, None, None] * c1[None, :, None] * c2[None, None, :]

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
        assert _is_functional_proxy_of(other, GaussianBasis), "other must be GaussianBasis"
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
                if not self.has_coeffs()
                else self.get_coeffs()
            )
            c_other = (
                torch.ones(mu2.shape[1], dtype=mu2.dtype, device=mu2.device)
                if not other.has_coeffs()
                else other.get_coeffs()
            )
            out = out * (c_self[:, None, None, None] * c_self[None, :, None, None])
            out = out * (c_other[None, None, :, None] * c_other[None, None, None, :])
        return out

    def marginal(self, marginal_dims: tuple[int, ...], ignore_coeffs: bool = False) -> "GaussianBasis":
        dims = tuple(marginal_dims)
        assert all(0 <= i < self.dim() for i in dims), "marginal_dims must be in [0, d)"
        coeffs_out = self.get_coeffs() if ((not ignore_coeffs) and self.has_coeffs()) else None
        
        mu, std = self.means_stds()
        mu_out = mu[dims, :]
        std_out = std[dims, :]
        return GaussianBasis(mean_params=mu_out, std_params=std_out, coeffs=coeffs_out, block_size=self.block_size)

    def product_basis(self, other_basis_factors: list["Basis"]) -> "GaussianBasis":
        pass # TODO
        """
        Cartesian product of factor bases along the flattened product index.

        Each factor is a *normalized* separable Gaussian PDF. Their pointwise product is
        proportional to the Gaussian described by the returned ``params``. The ratio
        ``product_k N_k / N_*`` (per axis, then multiplied across axes) and any factor ``coeffs``
        are stored on the returned basis as ``coeffs`` (including the all-ones case when the
        prefactor is trivial).
        """
        factors: list[GaussianBasis] = [self]
        for other in other_basis_factors:
            assert _is_functional_proxy_of(other, GaussianBasis), "all factors must be GaussianBasis"
            assert other.dim() == self.dim(), "Basis functions must have the same dimension"
            factors.append(other)

        n_factors = len(factors)
        dtype, device = self.param_dtype_device()
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

        coeff_terms = []
        for basis in factors:
            if not basis.has_coeffs():
                coeff_terms.append(torch.ones(basis.n_basis_functions(), dtype=dtype, device=device))
            else:
                coeff_terms.append(basis.get_coeffs())

        coeff_prod = torch.ones(n_per_factor, dtype=dtype, device=device)
        for k, c_k in enumerate(coeff_terms):
            view_shape = [1] * n_factors
            view_shape[k] = n_per_factor[k]
            coeff_prod = coeff_prod * c_k.reshape(view_shape)
        coeffs_new = (coeff_prod * torch.exp(log_const_flat.reshape(n_per_factor))).reshape(n_total)

        return GaussianBasis(
            params=(
                Parameters.from_values(mu_flat, trainable=False),
                self.get_params()[1].with_fixed_values(sigma_flat),
            ),
            coeffs=PositiveParameters.from_values(coeffs_new, trainable=False),
            block_size=self.block_size,
        )


class QuadraticExpBasis(SeparableBasis, NonnegativeBasis):
    """Separable basis with 1D factors exp(a x^2 + b x), a < 0."""

    def __init__(
        self,
        params: tuple[Parameters, Parameters],
        coeffs: PositiveParameters | None = None,
    ):
        assert len(params) == 2 and all(isinstance(p, Parameters) for p in params)
        super().__init__(params=list(params), coeffs=coeffs)

    @classmethod
    def random_init(
        cls,
        d: int,
        n_basis: int,
        offsets: torch.Tensor = torch.zeros(2),
        variance: float = 1.0,
        device=None,
        coeffs: PositiveParameters | None = None,
    ):
        if device is None:
            device = offsets.device
        else:
            offsets = offsets.to(device)
        scale = torch.sqrt(torch.tensor(variance, device=device))
        return cls(
            params=(
                Parameters.random_init((d, n_basis), mean=offsets[0].item(), std=scale.item()),
                Parameters.random_init((d, n_basis), mean=offsets[1].item(), std=scale.item()),
            ),
            coeffs=coeffs,
        )

    @classmethod
    def set_init(
        cls,
        d: int,
        n_basis: int,
        offsets: torch.Tensor = torch.zeros(2),
        coeffs: PositiveParameters | None = None,
    ):
        return cls(
            params=(
                Parameters.set_init((d, n_basis), offsets[0].item()),
                Parameters.set_init((d, n_basis), offsets[1].item()),
            ),
            coeffs=coeffs,
        )

    def _a_values(self) -> torch.Tensor:
        raw = self._params[0]()
        if self._params[0].is_trainable():
            return -torch.nn.functional.softplus(raw)
        return raw

    def freeze_params(self):
        return QuadraticExpBasis(
            params=(
                Parameters.from_values(self._a_values().detach(), trainable=False),
                self._params[1].freeze_params(),
            ),
            coeffs=self._coeffs.freeze_params() if self.has_coeffs() else None,
        )

    def _ab_axes(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._a_values(), self._params[1]()

    @staticmethod
    def _log_quadratic_exp_pair_ip(
        a: torch.Tensor,
        b: torch.Tensor,
        lows: torch.Tensor | None,
        highs: torch.Tensor | None,
    ) -> torch.Tensor:
        log_pref = -b.square() / (4.0 * a) + 0.5 * (
            torch.log(torch.tensor(torch.pi, dtype=a.dtype, device=a.device)) - torch.log(-a)
        )
        if lows is None:
            return log_pref
        scale = torch.sqrt(-4.0 * a)
        erf_diff = torch.special.erf((2.0 * a * highs + b) / scale) - torch.special.erf(
            (2.0 * a * lows + b) / scale
        )
        return log_pref + erf_diff.clamp_min(torch.finfo(a.dtype).tiny).log()

    def normalized(self):
        return False

    def forward(self, y: torch.Tensor):
        assert y.shape[1] == self.dim(), "y must have shape (n_data, d)"
        a, b = self._ab_axes()
        y = y[:, :, None]
        out = torch.exp((a * y.square() + b * y).sum(dim=1))
        if self.has_coeffs():
            out = out * self.coeff_values()[None, :]
        return out

    def Omega1(self, ignore_coeffs: bool = False):
        a, b = self._ab_axes()
        log_dim_int = -b.square() / (4.0 * a) + 0.5 * (
            torch.log(torch.tensor(torch.pi, dtype=a.dtype, device=a.device)) - torch.log(-a)
        )
        out = torch.exp(log_dim_int.sum(dim=0))
        if (not ignore_coeffs) and self.has_coeffs():
            out = out * self.coeff_values()
        return out

    def Omega2(
        self,
        other: "QuadraticExpBasis",
        ignore_coeffs: bool = False,
        lows: torch.Tensor | None = None,
        highs: torch.Tensor | None = None,
    ):
        assert isinstance(other, QuadraticExpBasis), "other must be QuadraticExpBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"
        a1, b1 = self._ab_axes()
        a2, b2 = other._ab_axes()
        lows_b, highs_b = None, None
        if lows is not None or highs is not None:
            assert lows is not None and highs is not None, "lows and highs must both be set for a bounded domain"
            dtype, device = self.param_dtype_device()
            lows_b = lows.to(dtype=dtype, device=device).reshape(self.dim())[:, None, None]
            highs_b = highs.to(dtype=dtype, device=device).reshape(self.dim())[:, None, None]
            assert torch.all(lows_b < highs_b), "each low must be strictly less than the corresponding high"
        log_dim = self._log_quadratic_exp_pair_ip(
            a1[:, :, None] + a2[:, None, :], b1[:, :, None] + b2[:, None, :], lows_b, highs_b
        )
        out = torch.exp(log_dim.sum(dim=0))
        if not ignore_coeffs:
            if self.has_coeffs():
                out = out * self.coeff_values()[:, None]
            if other.has_coeffs():
                out = out * other.coeff_values()[None, :]
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

        a0, b0 = self._ab_axes()
        a1, b1 = other1._ab_axes()
        a2, b2 = other2._ab_axes()

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

        a1, b1 = self._ab_axes()
        a2, b2 = other._ab_axes()

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
        return torch.exp(log_dim_int.sum(dim=0))

    def marginal(self, marginal_dims: tuple[int, ...], ignore_coeffs: bool = False) -> "QuadraticExpBasis":
        marginal_dims = tuple(marginal_dims)
        assert all(0 <= i < self.dim() for i in marginal_dims), "marginal_dims must be in [0, d)"

        # In this codebase, marginal_dims are the coordinates to keep.
        keep_dims = marginal_dims
        integrate_dims = tuple(i for i in range(self.dim()) if i not in keep_dims)

        a, b = self._ab_axes()
        finfo = torch.finfo(a.dtype)

        if len(integrate_dims) > 0:
            a_m = torch.clamp(a[integrate_dims, :], max=-torch.finfo(a.dtype).eps)
            b_m = b[integrate_dims, :]
            log_pi = torch.log(torch.tensor(torch.pi, dtype=a.dtype, device=a.device))
            log_int_m = -(b_m * b_m) / (4.0 * a_m) + 0.5 * (log_pi - torch.log(-a_m))
            log_int_sum = torch.clamp(log_int_m.sum(dim=0), max=0.99 * math.log(finfo.max))
        else:
            log_int_sum = torch.zeros(self.n_basis_functions(), dtype=a.dtype, device=a.device)

        coeffs_new = torch.exp(log_int_sum)
        if (not ignore_coeffs) and self.has_coeffs():
            coeffs_new = coeffs_new * self.coeff_values()
        coeffs_new = torch.nan_to_num(coeffs_new, nan=0.0, posinf=finfo.max, neginf=0.0)

        return QuadraticExpBasis(
            params=(
                Parameters.from_values(a[keep_dims, :], trainable=False),
                Parameters.from_values(self._params[1]()[keep_dims, :], trainable=False),
            ),
            coeffs=PositiveParameters.from_values(coeffs_new, trainable=False),
        )

    def product_basis(self, other_basis_factors: list["Basis"]) -> "QuadraticExpBasis":
        factors: list[QuadraticExpBasis] = [self]
        for other in other_basis_factors:
            assert isinstance(other, QuadraticExpBasis), "all factors must be QuadraticExpBasis"
            assert other.dim() == self.dim(), "Basis functions must have the same dimension"
            factors.append(other)

        n_factors = len(factors)
        dtype, device = self.param_dtype_device()
        dim = self.dim()

        ab_terms: list[tuple[torch.Tensor, torch.Tensor]] = [basis._ab_axes() for basis in factors]
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

        coeff_terms = []
        for basis in factors:
            if not basis.has_coeffs():
                coeff_terms.append(torch.ones(basis.n_basis_functions(), dtype=dtype, device=device))
            else:
                coeff_terms.append(basis.coeff_values())

        coeff_prod = torch.ones(n_per_factor, dtype=dtype, device=device)
        for k, c_k in enumerate(coeff_terms):
            view_shape = [1] * n_factors
            view_shape[k] = n_per_factor[k]
            coeff_prod = coeff_prod * c_k.reshape(view_shape)
        coeffs_new = coeff_prod.reshape(n_total)
        return QuadraticExpBasis(
            params=(
                Parameters.from_values(a_flat, trainable=False),
                Parameters.from_values(b_flat, trainable=False),
            ),
            coeffs=PositiveParameters.from_values(coeffs_new, trainable=False),
        )


class BetaBasis(SeparableBasis, NonnegativeBasis):
    """Separable product of normalized 1D Beta PDFs on (0, 1)."""

    def __init__(
        self,
        params: tuple[PositiveParameters, PositiveParameters],
        coeffs: PositiveParameters | None = None,
        eps: float = 1e-6,
    ):
        assert len(params) == 2 and all(isinstance(p, PositiveParameters) for p in params)
        super().__init__(params=list(params), coeffs=coeffs)
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
        coeffs: PositiveParameters | None = None,
    ):
        if device is None:
            device = offsets.device
        else:
            offsets = offsets.to(device)
        scale = torch.sqrt(torch.tensor(variance, device=device))
        return cls(
            params=(
                PositiveParameters(
                    trainable_init_values=torch.randn(d, n_basis, device=device) * scale + offsets[0],
                    normalized=False,
                ),
                PositiveParameters(
                    trainable_init_values=torch.randn(d, n_basis, device=device) * scale + offsets[1],
                    normalized=False,
                ),
            ),
            coeffs=coeffs,
            eps=eps,
        )

    @classmethod
    def set_init(
        cls,
        d: int,
        n_basis: int,
        offsets: torch.Tensor = torch.zeros(2),
        eps: float = 1e-6,
        device=None,
        coeffs: PositiveParameters | None = None,
    ):
        return cls(
            params=(
                PositiveParameters(
                    trainable_init_values=torch.full((d, n_basis), offsets[0].item()),
                    normalized=False,
                ),
                PositiveParameters(
                    trainable_init_values=torch.full((d, n_basis), offsets[1].item()),
                    normalized=False,
                ),
            ),
            coeffs=coeffs,
            eps=eps,
        )

    def freeze_params(self):
        return BetaBasis(
            params=tuple(p.freeze_params() for p in self._params),
            coeffs=self._coeffs.freeze_params() if self.has_coeffs() else None,
            eps=self.eps,
        )

    def _alphas_betas_axes(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._params[0](), self._params[1]()

    def alphas_betas(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._params[0](), self._params[1]()

    @staticmethod
    def _log_beta_fn(a: torch.Tensor, b: torch.Tensor):
        return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)

    @staticmethod
    def _log_beta_pair_ip(
        a1: torch.Tensor,
        b1: torch.Tensor,
        a2: torch.Tensor,
        b2: torch.Tensor,
        lows: torch.Tensor | None,
        highs: torch.Tensor | None,
        normalized: bool,
    ) -> torch.Tensor:
        a = a1 + a2 - 1.0
        b = b1 + b2 - 1.0
        log_ip = BetaBasis._log_beta_fn(a, b)
        if normalized:
            log_ip = log_ip - BetaBasis._log_beta_fn(a1, b1) - BetaBasis._log_beta_fn(a2, b2)
        if lows is None:
            return log_ip
        inc = torch.special.betainc(highs, a, b) - torch.special.betainc(lows, a, b)
        return log_ip + inc.clamp_min(torch.finfo(a.dtype).tiny).log()

    def forward(self, y: torch.Tensor):
        assert y.shape[1] == self.dim(), "y must have shape (n_data, d)"
        y = y.clamp(self.eps, 1.0 - self.eps)[:, :, None]
        alpha, beta = self._alphas_betas_axes()
        log_dim_factors = (
            (alpha - 1.0) * torch.log(y)
            + (beta - 1.0) * torch.log1p(-y)
            - self._log_beta_fn(alpha, beta)
        )
        out = torch.exp(log_dim_factors.sum(dim=1))
        if self.has_coeffs():
            out = out * self.coeff_values()[None, :]
        return out

    def normalized(self):
        return not self.has_coeffs()

    def Omega1(self, ignore_coeffs: bool = False):
        dtype, device = self.param_dtype_device()
        out = torch.ones(self.n_basis_functions(), dtype=dtype, device=device)
        if (not ignore_coeffs) and self.has_coeffs():
            out = out * self.coeff_values()
        return out

    def Omega2(
        self,
        other: "BetaBasis",
        ignore_coeffs: bool = False,
        lows: torch.Tensor | None = None,
        highs: torch.Tensor | None = None,
    ):
        assert isinstance(other, BetaBasis), "other must be BetaBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"
        a1, b1 = self._alphas_betas_axes()
        a2, b2 = other._alphas_betas_axes()
        lows_b, highs_b = None, None
        if lows is not None or highs is not None:
            assert lows is not None and highs is not None, "lows and highs must both be set for a bounded domain"
            dtype, device = self.param_dtype_device()
            lows_b = lows.to(dtype=dtype, device=device).reshape(self.dim())[:, None, None]
            highs_b = highs.to(dtype=dtype, device=device).reshape(self.dim())[:, None, None]
            assert torch.all(lows_b < highs_b), "each low must be strictly less than the corresponding high"
        log_dim = self._log_beta_pair_ip(
            a1[:, :, None], b1[:, :, None], a2[:, None, :], b2[:, None, :], lows_b, highs_b, True
        )
        out = torch.exp(log_dim.sum(dim=0))
        if not ignore_coeffs:
            if self.has_coeffs():
                out = out * self.coeff_values()[:, None]
            if other.has_coeffs():
                out = out * other.coeff_values()[None, :]
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

        a0, b0 = self._alphas_betas_axes()
        a1, b1 = other1._alphas_betas_axes()
        a2, b2 = other2._alphas_betas_axes()
        c0 = (
            torch.ones(self.n_basis_functions(), dtype=a0.dtype, device=a0.device)
            if ignore_coeffs or (not self.has_coeffs())
            else self.coeff_values()
        )
        c1 = (
            torch.ones(other1.n_basis_functions(), dtype=a0.dtype, device=a0.device)
            if ignore_coeffs or (not other1.has_coeffs())
            else other1.coeff_values()
        )
        c2 = (
            torch.ones(other2.n_basis_functions(), dtype=a0.dtype, device=a0.device)
            if ignore_coeffs or (not other2.has_coeffs())
            else other2.coeff_values()
        )
        coeff_scale = c0[:, None, None] * c1[None, :, None] * c2[None, None, :]

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

        a1, b1 = self._alphas_betas_axes()
        a2, b2 = other._alphas_betas_axes()

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
                if not self.has_coeffs()
                else self.coeff_values()
            )
            c_other = (
                torch.ones(a2.shape[1], dtype=a2.dtype, device=a2.device)
                if not other.has_coeffs()
                else other.coeff_values()
            )
            out = out * (c_self[:, None, None, None] * c_self[None, :, None, None])
            out = out * (c_other[None, None, :, None] * c_other[None, None, None, :])
        return out

    def marginal(self, marginal_dims: tuple[int, ...], ignore_coeffs: bool = False) -> "BetaBasis":
        dims = tuple(marginal_dims)
        assert all(0 <= i < self.dim() for i in dims), "marginal_dims must be in [0, d)"
        coeffs_out = self._coeffs.freeze_params() if ((not ignore_coeffs) and self.has_coeffs()) else None
        return BetaBasis(
            params=(
                self._params[0].with_fixed_values(self._params[0]()[dims, :]),
                self._params[1].with_fixed_values(self._params[1]()[dims, :]),
            ),
            coeffs=coeffs_out,
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
        min_c = self._params[0]._epsilon
        for other in other_basis_factors:
            assert isinstance(other, BetaBasis), "all factors must be BetaBasis"
            assert other.dim() == self.dim(), "Basis functions must have the same dimension"
            assert abs(other._params[0]._epsilon - min_c) < 1e-12
            factors.append(other)

        n_factors = len(factors)
        dtype, device = self.param_dtype_device()
        dim = self.dim()

        alpha_beta_terms = [basis._alphas_betas_axes() for basis in factors]
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

        coeff_terms = []
        for basis in factors:
            if not basis.has_coeffs():
                coeff_terms.append(torch.ones(basis.n_basis_functions(), dtype=dtype, device=device))
            else:
                coeff_terms.append(basis.coeff_values())

        coeff_prod = torch.ones(n_per_factor, dtype=dtype, device=device)
        for k, c_k in enumerate(coeff_terms):
            view_shape = [1] * n_factors
            view_shape[k] = n_per_factor[k]
            coeff_prod = coeff_prod * c_k.reshape(view_shape)

        log_const_flat = log_const_dim.sum(dim=0).reshape(n_total)
        intrinsic = torch.exp(log_const_flat.reshape(n_per_factor))
        coeffs_new = (coeff_prod * intrinsic).reshape(n_total)

        return BetaBasis(
            params=(
                self._params[0].with_fixed_values(alpha_flat),
                self._params[1].with_fixed_values(beta_flat),
            ),
            coeffs=PositiveParameters.from_values(coeffs_new, trainable=False),
            eps=self.eps,
        )


class UnnormalizedBetaBasis(SeparableBasis, NonnegativeBasis):
    """Separable unnormalized Beta kernel x^(α-1) (1-x)^(β-1) on (0, 1)."""

    def __init__(
        self,
        params: tuple[PositiveParameters, PositiveParameters],
        coeffs: PositiveParameters | None = None,
        eps: float = 1e-6,
    ):
        assert len(params) == 2 and all(isinstance(p, PositiveParameters) for p in params)
        super().__init__(params=list(params), coeffs=coeffs)
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
        coeffs: PositiveParameters | None = None,
    ):
        beta = BetaBasis.random_init(d, n_basis, offsets, variance, eps, device, coeffs)
        return cls(params=tuple(beta._params), coeffs=coeffs, eps=eps)

    @classmethod
    def set_init(
        cls,
        d: int,
        n_basis: int,
        offsets: torch.Tensor = torch.zeros(2),
        eps: float = 1e-6,
        device=None,
        coeffs: PositiveParameters | None = None,
    ):
        beta = BetaBasis.set_init(d, n_basis, offsets, eps, device, coeffs)
        return cls(params=tuple(beta._params), coeffs=coeffs, eps=eps)

    def freeze_params(self):
        return UnnormalizedBetaBasis(
            params=tuple(p.freeze_params() for p in self._params),
            coeffs=self._coeffs.freeze_params() if self.has_coeffs() else None,
            eps=self.eps,
        )

    def _alphas_betas_axes(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._params[0](), self._params[1]()

    def alphas_betas(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._params[0](), self._params[1]()

    def forward(self, y: torch.Tensor):
        assert y.shape[1] == self.dim(), "y must have shape (n_data, d)"
        y = y.clamp(self.eps, 1.0 - self.eps)[:, :, None]
        alpha, beta = self._alphas_betas_axes()
        out = torch.exp(((alpha - 1.0) * torch.log(y) + (beta - 1.0) * torch.log1p(-y)).sum(dim=1))
        if self.has_coeffs():
            out = out * self.coeff_values()[None, :]
        return out

    def normalized(self):
        return False

    def Omega1(self, ignore_coeffs: bool = False):
        alpha, beta = self._alphas_betas_axes()
        out = torch.exp(BetaBasis._log_beta_fn(alpha, beta).sum(dim=0))
        if (not ignore_coeffs) and self.has_coeffs():
            out = out * self.coeff_values()
        return out

    def Omega2(
        self,
        other: "UnnormalizedBetaBasis",
        ignore_coeffs: bool = False,
        lows: torch.Tensor | None = None,
        highs: torch.Tensor | None = None,
    ):
        assert isinstance(other, UnnormalizedBetaBasis), "other must be UnnormalizedBetaBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"
        a1, b1 = self._alphas_betas_axes()
        a2, b2 = other._alphas_betas_axes()
        lows_b, highs_b = None, None
        if lows is not None or highs is not None:
            assert lows is not None and highs is not None, "lows and highs must both be set for a bounded domain"
            dtype, device = self.param_dtype_device()
            lows_b = lows.to(dtype=dtype, device=device).reshape(self.dim())[:, None, None]
            highs_b = highs.to(dtype=dtype, device=device).reshape(self.dim())[:, None, None]
            assert torch.all(lows_b < highs_b), "each low must be strictly less than the corresponding high"
        log_dim = BetaBasis._log_beta_pair_ip(
            a1[:, :, None], b1[:, :, None], a2[:, None, :], b2[:, None, :], lows_b, highs_b, False
        )
        out = torch.exp(log_dim.sum(dim=0))
        if not ignore_coeffs:
            if self.has_coeffs():
                out = out * self.coeff_values()[:, None]
            if other.has_coeffs():
                out = out * other.coeff_values()[None, :]
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

        a0, b0 = self._alphas_betas_axes()
        a1, b1 = other1._alphas_betas_axes()
        a2, b2 = other2._alphas_betas_axes()

        n0 = self.n_basis_functions()
        n1 = other1.n_basis_functions()
        n2 = other2.n_basis_functions()

        if block_size is None:
            a_sum = a0[:, :, None, None] + a1[:, None, :, None] + a2[:, None, None, :] - 2.0
            b_sum = b0[:, :, None, None] + b1[:, None, :, None] + b2[:, None, None, :] - 2.0
            omega_full = torch.exp(BetaBasis._log_beta_fn(a_sum, b_sum).sum(dim=0))
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
                    log_chunk += BetaBasis._log_beta_fn(a_sum, b_sum)

                omega_chunk = torch.exp(log_chunk)
                denom[k_start:k_end] += torch.einsum("i,j,ijk->k", left_i, left_j_blk, omega_chunk)
        return denom

    def Omega22(self, other: "UnnormalizedBetaBasis"):
        assert isinstance(other, UnnormalizedBetaBasis), "other must be UnnormalizedBetaBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"

        a1, b1 = self._alphas_betas_axes()
        a2, b2 = other._alphas_betas_axes()

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
        return torch.exp(BetaBasis._log_beta_fn(a_sum, b_sum).sum(dim=0))

    def marginal(self, marginal_dims: tuple[int, ...], ignore_coeffs: bool = False) -> "UnnormalizedBetaBasis":
        keep_dims = tuple(marginal_dims)
        assert all(0 <= i < self.dim() for i in keep_dims), "marginal_dims must be in [0, d)"
        integrate_dims = tuple(i for i in range(self.dim()) if i not in keep_dims)
        alpha, beta = self._alphas_betas_axes()
        if len(integrate_dims) > 0:
            coeffs_new = torch.exp(
                BetaBasis._log_beta_fn(alpha[integrate_dims, :], beta[integrate_dims, :]).sum(dim=0)
            )
        else:
            dtype, device = self.param_dtype_device()
            coeffs_new = torch.ones(self.n_basis_functions(), dtype=dtype, device=device)
        if (not ignore_coeffs) and self.has_coeffs():
            coeffs_new = coeffs_new * self.coeff_values()
        return UnnormalizedBetaBasis(
            params=(
                self._params[0].with_fixed_values(self._params[0]()[keep_dims, :]),
                self._params[1].with_fixed_values(self._params[1]()[keep_dims, :]),
            ),
            coeffs=PositiveParameters.from_values(coeffs_new, trainable=False),
            eps=self.eps,
        )

    def product_basis(self, other_basis_factors: list["Basis"]) -> "UnnormalizedBetaBasis":
        factors: list[UnnormalizedBetaBasis] = [self]
        min_c = self._params[0]._epsilon
        for other in other_basis_factors:
            assert isinstance(other, UnnormalizedBetaBasis), "all factors must be UnnormalizedBetaBasis"
            assert other.dim() == self.dim(), "Basis functions must have the same dimension"
            assert abs(other._params[0]._epsilon - min_c) < 1e-12
            factors.append(other)

        n_factors = len(factors)
        dtype, device = self.param_dtype_device()
        dim = self.dim()

        alpha_beta_terms = [basis._alphas_betas_axes() for basis in factors]
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

        alpha_new = alpha_new.clamp_min(min_c + 1e-8)
        beta_new = beta_new.clamp_min(min_c + 1e-8)

        n_total = 1
        for n in n_per_factor:
            n_total *= n

        alpha_flat = alpha_new.reshape(dim, n_total)
        beta_flat = beta_new.reshape(dim, n_total)

        coeff_terms = []
        for basis in factors:
            if not basis.has_coeffs():
                coeff_terms.append(torch.ones(basis.n_basis_functions(), dtype=dtype, device=device))
            else:
                coeff_terms.append(basis.coeff_values())

        coeff_prod = torch.ones(n_per_factor, dtype=dtype, device=device)
        for k, c_k in enumerate(coeff_terms):
            view_shape = [1] * n_factors
            view_shape[k] = n_per_factor[k]
            coeff_prod = coeff_prod * c_k.reshape(view_shape)
        coeffs_new = coeff_prod.reshape(n_total)

        return UnnormalizedBetaBasis(
            params=(
                self._params[0].with_fixed_values(alpha_flat),
                self._params[1].with_fixed_values(beta_flat),
            ),
            coeffs=PositiveParameters.from_values(coeffs_new, trainable=False),
            eps=self.eps,
        )


class NFBasis(Basis, NonnegativeBasis):
    def __init__(
        self,
        dim: int,
        n_basis: int,
        n_layers: int = 5,
        hidden_features: int = 128,
        embedding_dim: int = 16,
    ):
        super().__init__(dim=dim, n_basis=n_basis, params=[])

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

    def param_dtype_device(self):
        return self.index_embedding.weight.dtype, self.index_embedding.weight.device

    def Omega1(self):
        dtype, device = self.param_dtype_device()
        return torch.ones(self.n_basis_functions(), dtype=dtype, device=device)


class GaussianKernelBasis(Basis, NonnegativeBasis):
    def __init__(
        self,
        x: torch.Tensor,
        kernel_bandwidth: torch.Tensor | Parameters | None = None,
        coeffs: PositiveParameters | None = None,
        *,
        trainable: bool = True,
    ):
        if coeffs is not None:
            assert not trainable, "GaussianKernelBasis: coeffs requires trainable=False"
        assert x.dim() == 2, "x must have shape (n_samples, d)"
        centers = Parameters.from_values(x.detach().clone().transpose(0, 1).contiguous(), trainable=trainable)
        if kernel_bandwidth is None:
            bw = GaussianKernelBasis.ss_bandwidth(x.detach())
            bandwidth = Parameters.from_values(bw, trainable=trainable)
        elif isinstance(kernel_bandwidth, Parameters):
            bandwidth = kernel_bandwidth
        else:
            bandwidth = Parameters.from_values(kernel_bandwidth.detach().clone(), trainable=trainable)
        super().__init__(dim=x.shape[1], n_basis=x.shape[0], params=(centers,), coeffs=coeffs)
        self.add_module("_bandwidth", bandwidth)

    @property
    def kernel_bandwidth(self) -> torch.Tensor:
        """Shared scalar bandwidth (trainable parameter or buffer)."""
        return self._bandwidth._p

    def freeze_params(self):
        return GaussianKernelBasis(
            self.kernel_centers().transpose(0, 1).detach().clone(),
            kernel_bandwidth=self._bandwidth.freeze_params(),
            coeffs=self._coeffs.freeze_params() if self.has_coeffs() else None,
            trainable=False,
        )

    @staticmethod
    def ss_bandwidth(x : torch.Tensor):
        """
        Isotropic Silverman/Scott-style rule of thumb.
        Returns a scalar bandwidth shared by all kernels.
        """
        assert x.dim() == 2, "x must have shape (n_samples, d)"
        n, d = x.shape
        eps = torch.finfo(x.dtype).eps

        if n <= 1:
            return torch.tensor(1.0, dtype=x.dtype, device=x.device)

        sample_std = x.std(dim=0, unbiased=True)
        scale = sample_std.mean().clamp_min(eps)
        silverman = (4.0 / (d + 2.0)) ** (1.0 / (d + 4.0))
        h = silverman * (n ** (-1.0 / (d + 4.0))) * scale
        return h.clamp_min(eps)
    
    def kernel_centers(self) -> torch.Tensor:
        return self._params[0]()

    def bandwidth(self) -> torch.Tensor:
        return self._bandwidth().reshape(())

    def _kernel_mean_std_axes(self) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.kernel_centers()
        h = self.bandwidth().clamp_min(torch.finfo(mu.dtype).eps)
        return mu, torch.full_like(mu, h)

    def forward(self, y: torch.Tensor):
        assert y.shape[1] == self.dim(), "y must have shape (n_data, d)"

        h = self.bandwidth().clamp_min(torch.finfo(y.dtype).eps)
        d = self.dim()

        diff = y[:, None, :] - self.kernel_centers()[None, :, :]
        sq_norm = diff.square().sum(dim=2)  # (n_data, n_kernels)

        two_pi = torch.as_tensor(2.0 * math.pi, device=y.device, dtype=y.dtype)
        norm_const = torch.pow(two_pi, -0.5 * d) * h.pow(-d)
        kernels = norm_const * torch.exp(-0.5 * sq_norm / (h * h))  # (n_data, n_kernels)
        if self.has_coeffs():
            kernels = kernels * self.coeff_values()[None, :]
        return kernels  # (n_data, n_basis)

    def marginal(self, marginal_dims: tuple[int, ...], ignore_coeffs: bool = False) -> "GaussianKernelBasis":
        dims = tuple(marginal_dims)
        assert all(0 <= i < self.dim() for i in dims), "marginal_dims must be in [0, d)"
        coeffs_out = self._coeffs.freeze_params() if ((not ignore_coeffs) and self.has_coeffs()) else None
        return GaussianKernelBasis(
            self.kernel_centers()[dims, :].transpose(0, 1).detach().clone(),
            kernel_bandwidth=self._bandwidth.freeze_params(),
            coeffs=coeffs_out,
            trainable=False,
        )

    def product_basis(self, other_basis_factors: list["Basis"]) -> "GaussianKernelBasis":
        """
        Cartesian product index over factors. Each factor is an isotropic normalized Gaussian
        kernel; their pointwise product is proportional to another such kernel. Analytic
        prefactors and factor ``coeffs`` are stored on the returned basis as ``coeffs``.
        """
        factors: list[GaussianKernelBasis] = [self]
        for other in other_basis_factors:
            assert isinstance(other, GaussianKernelBasis), "all factors must be GaussianKernelBasis"
            assert other.dim() == self.dim(), "Basis functions must have the same dimension"
            factors.append(other)

        n_factors = len(factors)
        dtype, device = self.param_dtype_device()
        dim = self.dim()

        mus_stds = []
        for basis in factors:
            mu_k = basis.kernel_centers()
            h_k = basis.bandwidth().clamp_min(torch.finfo(dtype).eps)
            std_k = torch.full_like(mu_k, h_k)
            mus_stds.append((mu_k, std_k))

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

        coeff_terms = []
        for basis in factors:
            if not basis.has_coeffs():
                coeff_terms.append(torch.ones(basis.n_basis_functions(), dtype=dtype, device=device))
            else:
                coeff_terms.append(basis.coeff_values())

        coeff_prod = torch.ones(n_per_factor, dtype=dtype, device=device)
        for k, c_k in enumerate(coeff_terms):
            view_shape = [1] * n_factors
            view_shape[k] = n_per_factor[k]
            coeff_prod = coeff_prod * c_k.reshape(view_shape)
        coeffs_new = (coeff_prod * torch.exp(log_const_flat.reshape(n_per_factor))).reshape(n_total)

        centers_new = mu_flat.transpose(0, 1).contiguous().detach().clone()
        h_new = sigma_flat[0, 0].detach().clone()

        return GaussianKernelBasis(
            centers_new,
            kernel_bandwidth=Parameters.from_values(h_new, trainable=False),
            coeffs=PositiveParameters.from_values(coeffs_new, trainable=False),
            trainable=False,
        )

    def Omega1(self, ignore_coeffs: bool = False):
        dtype, device = self.param_dtype_device()
        out = torch.ones(self.n_basis_functions(), dtype=dtype, device=device)
        if (not ignore_coeffs) and self.has_coeffs():
            out = out * self.coeff_values()
        return out

    def Omega2(
        self,
        other: "GaussianKernelBasis",
        ignore_coeffs: bool = False,
        lows: torch.Tensor | None = None,
        highs: torch.Tensor | None = None,
    ):
        assert isinstance(other, GaussianKernelBasis), "other must be GaussianKernelBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"

        mu1, std1 = self._kernel_mean_std_axes()
        mu2, std2 = other._kernel_mean_std_axes()
        lows_b, highs_b = None, None
        if lows is not None or highs is not None:
            assert lows is not None and highs is not None, "lows and highs must both be set for a bounded domain"
            dtype, device = self.param_dtype_device()
            lows_b = lows.to(dtype=dtype, device=device).reshape(self.dim())[:, None, None]
            highs_b = highs.to(dtype=dtype, device=device).reshape(self.dim())[:, None, None]
            assert torch.all(lows_b < highs_b), "each low must be strictly less than the corresponding high"
        log_dim = GaussianBasis._log_gaussian_pair_ip(
            mu1[:, :, None], std1[:, :, None], mu2[:, None, :], std2[:, None, :], lows_b, highs_b
        )
        out = torch.exp(log_dim.sum(dim=0))
        if not ignore_coeffs:
            if self.has_coeffs():
                out = out * self.coeff_values()[:, None]
            if other.has_coeffs():
                out = out * other.coeff_values()[None, :]
        return out

    def Omega3_contract(
        self,
        other1: "GaussianKernelBasis",
        other2: "GaussianKernelBasis",
        left_i: torch.Tensor,
        left_j: torch.Tensor,
        block_size: int | None = None,
        ignore_coeffs: bool = False,
    ):
        assert isinstance(other1, GaussianKernelBasis), "other1 must be GaussianKernelBasis"
        assert isinstance(other2, GaussianKernelBasis), "other2 must be GaussianKernelBasis"
        assert self.dim() == other1.dim() == other2.dim(), "Basis functions must have the same dimension"
        assert left_i.dim() == 1 and left_i.shape[0] == self.n_basis_functions(), "left_i has wrong shape"
        assert left_j.dim() == 1 and left_j.shape[0] == other1.n_basis_functions(), "left_j has wrong shape"

        x0 = self.kernel_centers().transpose(0, 1)     # (n0, d)
        x1 = other1.kernel_centers().transpose(0, 1)   # (n1, d)
        x2 = other2.kernel_centers().transpose(0, 1)   # (n2, d)
        d = self.dim()
        n0 = x0.shape[0]
        n1 = x1.shape[0]
        n2 = x2.shape[0]

        h0 = self.bandwidth().clamp_min(torch.finfo(x0.dtype).eps)
        h1 = other1.bandwidth().clamp_min(torch.finfo(x1.dtype).eps)
        h2 = other2.bandwidth().clamp_min(torch.finfo(x2.dtype).eps)

        c0 = (
            torch.ones(n0, dtype=x0.dtype, device=x0.device)
            if ignore_coeffs or (not self.has_coeffs())
            else self.coeff_values()
        )
        c1 = (
            torch.ones(n1, dtype=x0.dtype, device=x0.device)
            if ignore_coeffs or (not other1.has_coeffs())
            else other1.coeff_values()
        )
        c2 = (
            torch.ones(n2, dtype=x0.dtype, device=x0.device)
            if ignore_coeffs or (not other2.has_coeffs())
            else other2.coeff_values()
        )
        coeff_scale = c0[:, None, None] * c1[None, :, None] * c2[None, None, :]

        var0 = h0.square()
        var1 = h1.square()
        var2 = h2.square()
        tau0 = 1.0 / var0
        tau1 = 1.0 / var1
        tau01 = tau0 + tau1
        var01 = 1.0 / tau01

        # Product identity:
        # N(x|x0_i,var0I)N(x|x1_j,var1I) = c01_ij * N(x|m01_ij,var01I),
        # then integrate with the third Gaussian against x2_k.
        pair_var = var0 + var1
        pair_norm = torch.pow(2.0 * math.pi * pair_var, -0.5 * d)
        tail_var = var01 + var2
        tail_norm = torch.pow(2.0 * math.pi * tail_var, -0.5 * d)

        if block_size is None:
            block_size = 256
        assert block_size > 0, "block_size must be positive"

        out = torch.zeros(n2, dtype=x0.dtype, device=x0.device)

        for j_start in range(0, n1, block_size):
            j_end = min(j_start + block_size, n1)
            x1_blk = x1[j_start:j_end, :]              # (bj, d)
            left_j_blk = left_j[j_start:j_end]         # (bj,)

            diff01 = x0[:, None, :] - x1_blk[None, :, :]         # (n0, bj, d)
            sq01 = diff01.square().sum(dim=2)                    # (n0, bj)
            c01 = pair_norm * torch.exp(-0.5 * sq01 / pair_var)  # (n0, bj)

            m01 = (tau0 * x0[:, None, :] + tau1 * x1_blk[None, :, :]) / tau01  # (n0, bj, d)

            for k_start in range(0, n2, block_size):
                k_end = min(k_start + block_size, n2)
                x2_blk = x2[k_start:k_end, :]                                   # (bk, d)

                diff2 = m01[:, :, None, :] - x2_blk[None, None, :, :]           # (n0, bj, bk, d)
                sq2 = diff2.square().sum(dim=3)                                  # (n0, bj, bk)
                tail = tail_norm * torch.exp(-0.5 * sq2 / tail_var)              # (n0, bj, bk)

                omega_blk = (
                    c01[:, :, None]
                    * tail
                    * coeff_scale[:, j_start:j_end, k_start:k_end]
                )  # (n0, bj, bk)
                out[k_start:k_end] += torch.einsum("i,j,ijk->k", left_i, left_j_blk, omega_blk)

        return out

    def normalized(self):
        return not self.has_coeffs()