import math

import torch
from abc import abstractmethod

from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform

from .parameters import Parameters, PositiveParameters
from .gram import BetaGram, GaussianGram


class Basis:
    def __init__(self, 
            dim : int, 
            batch_size : int,
            n_basis : int,
            params : tuple[Parameters, ...],
            coeffs : Parameters = None):
        '''
        Args:
            dim : int, number of dimensions
            batch_size : int, number of data points in the batch
            n_basis : int, number of basis functions
            params : list of tensors where each represents a parameter group for a basis function
            coeffs : coefficients
        '''

        self._dim = dim
        self._batch_size = batch_size
        self._n_basis = n_basis

        for i, param in enumerate(params):
            assert isinstance(param, Parameters), "Parameter at index " + str(i) + " must be a Parameters object"
        self._params = params

        if coeffs is not None:
            self.set_coeffs(coeffs)
        else:
            self.set_coeffs_to_one()

    @staticmethod
    def get_deduplicated_module_list(bases : list["Basis"]) -> list[torch.nn.Module]:
        '''
        Returns a list of unique parameters and coefficients modules from the given list of bases.
        '''
        params_seen = {}
        coeffs_seen = {}
        unique_params = []
        unique_coeffs = []
        for basis in bases:
            if basis.coeffs.is_module() and id(basis.coeffs) not in coeffs_seen:
                unique_coeffs.append(basis.coeffs)
            for param in basis._params:
                if param.is_module() and id(param) not in params_seen:
                    unique_params.append(param)
                    params_seen[id(param)] = param
        return unique_params, unique_coeffs

    def forward(self, y : torch.Tensor):
        raise NotImplementedError("__call__ is not implemented for this basis function")

    def set_coeffs(self, coeffs : Parameters):
        assert isinstance(coeffs, Parameters), "coeffs must be a Parameters object"
        assert coeffs().dim() == 2, "coeffs must have shape (batch_size, n_basis)"
        assert coeffs().size() == (self._batch_size, self._n_basis), "coeffs must have shape (batch_size, n_basis)"
        self.coeffs = coeffs
    
    def set_coeffs_to_one(self):
        dtype, device = self.param_dtype_device()
        self.set_coeffs(Parameters(torch.ones(self._batch_size, self._n_basis, dtype=dtype, device=device)))
    
    def dim(self):
        return self._dim
    
    def batch_size(self):
        return self._batch_size
    
    def n_basis_functions(self):
        return self._n_basis
    
    def param_dtype_device(self):
        return self._params[0]().dtype, self._params[0]().device

    def normalized(self):
        raise NotImplementedError("normalized is not implemented for this basis function")
    
    def Omega1(self, lows : torch.Tensor = None, highs : torch.Tensor = None):
        '''
        Computes the integral of the basis functions.
        omega[i] = <this_i, 1>

        Args:
            lows : lower bounds of the integration domain, if None, the domain is the entire real line
            highs : upper bounds of the integration domain, if None, the domain is the entire real line
        '''
        raise NotImplementedError("Omega1 is not implemented for this basis function")

    def Omega2(self, other: 'Basis', lows : torch.Tensor = None, highs : torch.Tensor = None):
        '''
        Computes the inner product matrix of the basis functions with another basis function vector. 
        omega[i, j] = <this_i, other_j>

        Args:
            other : Basis function to compute the inner product with
            lows : lower bounds of the integration domain, if None, the domain is the entire real line
            highs : upper bounds of the integration domain, if None, the domain is the entire real line
        '''
        raise NotImplementedError("Omega2 is not implemented for this basis function")

    def Omega3(self, other1: 'Basis', other2: 'Basis', lows : torch.Tensor = None, highs : torch.Tensor = None):
        '''
        Computes the inner product tensor of the basis functions with another basis function vector. 
        omega[i, j, k] = <this_i, other1_j, other2_k>

        Args:
            other1 : Basis function to compute the inner product with
            other2 : Basis function to compute the inner product with
            lows : lower bounds of the integration domain, if None, the domain is the entire real line
            highs : upper bounds of the integration domain, if None, the domain is the entire real line
        '''

    def Omega22(self, other: 'Basis', v : torch.Tensor = None, lows : torch.Tensor = None, highs : torch.Tensor = None):
        '''
        Computes the inner product tensor of the basis functions with another basis function vector. 
        omega[i, j, k, l] = <this_i * this_j, other_k * other_l>

        Args:
            other : Basis function to compute the inner product with
            v : vector to contract with
            lows : lower bounds of the integration domain, if None, the domain is the entire real line
            highs : upper bounds of the integration domain, if None, the domain is the entire real line
        '''

    def product_basis(self, other_basis_factors: list["Basis"]):
        """
        Returns the broadcasted (flattened) product of the basis functions in ``other_basis_factors``.

        Implementations always attach ``coeffs`` on the returned basis (including analytic
        prefactors and factor ``coeffs``).
        """
        raise NotImplementedError("product_basis is not implemented for this basis function")
    
    def marginal(self, marginal_dims : tuple[int, ...]):
        '''
        Computes the marginalized basis functions over the given dimensions

        Returns:
            A new SeparableBasis object with the marginalized dimensions
        '''
        raise NotImplementedError("marginal is not implemented for this basis function")


class NonnegativeBasis:
    pass


class SeparableBasis(Basis):
    def __init__(
        self,
        params: tuple[Parameters, ...],
        coeffs: torch.Tensor = None,
    ):
        for param in params:
            assert param().dim() == 3, "Each parameter tensor must have shape (batch_size, dim, n_basis)"
        
        param_tensor = params[0]()
        #assert param_tensor.dim() == 3, "Each parameter tensor must have shape (batch_size, dim, n_basis)"
        batch_size = param_tensor.size()[0]
        dim = param_tensor.size()[1]
        n_basis = param_tensor.size()[2]
        super().__init__(dim=dim, batch_size=batch_size, n_basis=n_basis, params=params, coeffs=coeffs)

        for param_set in self._params:
            assert param_set.size() == (batch_size, dim, n_basis), "Each parameter tensor must have shape (batch_size, dim, n_basis)"

    def n_params_per_basis(self):
        return len(self._params)

    @staticmethod
    def _gram_domain_bounds(
        lows: torch.Tensor | None,
        highs: torch.Tensor | None,
        axes: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Broadcast per-dimension domain bounds to ``(*batch, dim)`` for gram params ``(*batch, dim, n)``."""
        dtype, device = axes.dtype, axes.device
        lows_b, highs_b = None, None
        if lows is not None:
            lows_b = torch.as_tensor(lows, dtype=dtype, device=device).reshape(-1)
            lows_b = lows_b[(None,) * (axes.dim() - 2) + (slice(None),) + (None,)].squeeze(-1)
        if highs is not None:
            highs_b = torch.as_tensor(highs, dtype=dtype, device=device).reshape(-1)
            highs_b = highs_b[(None,) * (axes.dim() - 2) + (slice(None),) + (None,)].squeeze(-1)
        return lows_b, highs_b

    def log_Omega1_dim(self, lows : torch.Tensor = None, highs : torch.Tensor = None):
        '''
        Per-coordinate log integrals of each basis function.

        Returns:
            ``(batch_size, dim, n_basis)``
        '''
        raise NotImplementedError("log_Omega1_dim is not implemented for this basis function")
    
    def log_Omega2_dim(self, other : 'Basis', lows : torch.Tensor = None, highs : torch.Tensor = None):
        '''
        Per-coordinate log inner-product matrices.

        Returns:
            ``(batch_size, dim, n_basis, other.n_basis)`` with
            ``out[..., d, i, j] = log <this_i, other_j>_d``
        '''
        raise NotImplementedError("log_Omega2_dim is not implemented for this basis function")

    def log_Omega3_dim(self, other1 : 'Basis', other2 : 'Basis', lows : torch.Tensor = None, highs : torch.Tensor = None):
        '''
        Per-coordinate log triple-product tensors.

        Returns:
            ``(batch_size, dim, n_basis, other1.n_basis, other2.n_basis)``
        '''
        raise NotImplementedError("log_Omega3_dim is not implemented for this basis function")

    def log_Omega22_dim(self, other : 'Basis', lows : torch.Tensor = None, highs : torch.Tensor = None):
        '''
        Per-coordinate log grams for ``<this_i * this_j, other_k * other_l>``.

        Returns:
            ``(batch_size, dim, n_basis, n_basis, other.n_basis, other.n_basis)``
        '''
        raise NotImplementedError("log_Omega22_dim is not implemented for this basis function")

    def eval_dim(self, y : torch.Tensor):
        '''
        Per-coordinate factor values (no product over dims, no coeffs).

        ``y`` has shape ``(batch, dim)``. Parameter tensors have leading size
        ``batch_size``; that axis is the same batch as ``y`` when parameters are
        functions of ``x``. If ``batch_size == 1``, shared parameters broadcast
        over ``y``.

        Returns:
            ``(batch, dim, n_basis)``
        '''
        raise NotImplementedError("eval_dim is not implemented for this basis function")

    def _check_eval_batch(self, y : torch.Tensor):
        assert y.dim() == 2 and y.shape[1] == self.dim(), "y must have shape (batch, dim)"
        n, b = y.shape[0], self.batch_size()
        assert b == n or b == 1, (
            f"y batch {n} must match parameter batch {b} (or parameter batch must be 1 for shared params)"
        )
        return n, b

    def Omega1(self, lows : torch.Tensor = None, highs : torch.Tensor = None):
        return torch.exp(self.log_Omega1_dim(lows, highs).sum(dim=1)) * self.coeffs()
    
    def Omega2(self, other : 'Basis', lows : torch.Tensor = None, highs : torch.Tensor = None):
        return (
            torch.exp(self.log_Omega2_dim(other, lows, highs).sum(dim=1))
            * self.coeffs()[:, :, None]
            * other.coeffs()[:, None, :]
        )

    def Omega3(self, other1 : 'Basis', other2 : 'Basis', lows : torch.Tensor = None, highs : torch.Tensor = None):
        return (
            torch.exp(self.log_Omega3_dim(other1, other2, lows, highs).sum(dim=1))
            * self.coeffs()[:, :, None, None]
            * other1.coeffs()[:, None, :, None]
            * other2.coeffs()[:, None, None, :]
        )

    def Omega22(self, other : 'Basis', lows : torch.Tensor = None, highs : torch.Tensor = None):
        c1 = self.coeffs()
        c2 = other.coeffs()
        return (
            torch.exp(self.log_Omega22_dim(other, lows, highs).sum(dim=1))
            * c1[:, :, None, None, None]
            * c1[:, None, :, None, None]
            * c2[:, None, None, :, None]
            * c2[:, None, None, None, :]
        )


class GaussianBasis(SeparableBasis, NonnegativeBasis):
    """Separable product of 1D normal PDFs N(x | mean, std^2)."""

    def __init__(
        self,
        mean_params : Parameters,
        std_params : PositiveParameters,
        coeffs: Parameters = None,
    ):
        super().__init__(params=(mean_params, std_params), coeffs=coeffs)

    def means_stds(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._params[0](), self._params[1]()

    def __call__(self, y: torch.Tensor):
        out = self.eval_dim(y).prod(dim=1)  # (batch, n_basis)
        return out * self.coeffs()

    def eval_dim(self, y: torch.Tensor):
        self._check_eval_batch(y)
        mu, std = self.means_stds()  # (batch_size, dim, n_basis)
        y_e = y[:, :, None]  # (batch, dim, 1)
        std = std.clamp_min(torch.finfo(y.dtype).eps)
        log_two_pi = y.new_tensor(2.0 * math.pi).log()
        # batch_size==1 broadcasts shared params over y's batch
        log_dim = -0.5 * (log_two_pi + 2.0 * torch.log(std) + ((y_e - mu) / std).square())
        return torch.exp(log_dim)  # (batch, dim, n_basis)

    def log_Omega1_dim(self, lows: torch.Tensor = None, highs: torch.Tensor = None):
        mu, std = self.means_stds()
        lows_b, highs_b = self._gram_domain_bounds(lows, highs, mu)
        return GaussianGram.log_gram((mu,), (std,), lows=lows_b, highs=highs_b)

    def log_Omega2_dim(self, other: "GaussianBasis", lows: torch.Tensor = None, highs: torch.Tensor = None):
        assert isinstance(other, GaussianBasis), "other must be GaussianBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"
        assert self.batch_size() == other.batch_size(), "Basis functions must have the same batch size"
        mu1, std1 = self.means_stds()
        mu2, std2 = other.means_stds()
        lows_b, highs_b = self._gram_domain_bounds(lows, highs, mu1)
        return GaussianGram.log_gram((mu1, mu2), (std1, std2), lows=lows_b, highs=highs_b)

    def log_Omega3_dim(
        self,
        other1: "GaussianBasis",
        other2: "GaussianBasis",
        lows: torch.Tensor = None,
        highs: torch.Tensor = None,
    ):
        assert isinstance(other1, GaussianBasis), "other1 must be GaussianBasis"
        assert isinstance(other2, GaussianBasis), "other2 must be GaussianBasis"
        assert self.dim() == other1.dim() == other2.dim(), "Basis functions must have the same dimension"
        assert self.batch_size() == other1.batch_size() == other2.batch_size(), "Basis functions must have the same batch size"
        mu1, std1 = self.means_stds()
        mu2, std2 = other1.means_stds()
        mu3, std3 = other2.means_stds()
        lows_b, highs_b = self._gram_domain_bounds(lows, highs, mu1)
        return GaussianGram.log_gram((mu1, mu2, mu3), (std1, std2, std3), lows=lows_b, highs=highs_b)

    def log_Omega22_dim(self, other: "GaussianBasis", lows: torch.Tensor = None, highs: torch.Tensor = None):
        assert isinstance(other, GaussianBasis), "other must be GaussianBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"
        assert self.batch_size() == other.batch_size(), "Basis functions must have the same batch size"
        mu1, std1 = self.means_stds()
        mu2, std2 = other.means_stds()
        lows_b, highs_b = self._gram_domain_bounds(lows, highs, mu1)
        return GaussianGram.log_gram((mu1, mu1, mu2, mu2), (std1, std1, std2, std2), lows=lows_b, highs=highs_b)

    def marginal(self, marginal_dims: tuple[int, ...]) -> "GaussianBasis":
        dims = tuple(marginal_dims)
        assert all(0 <= i < self.dim() for i in dims), "marginal_dims must be in [0, d)"
        mu, std = self.means_stds()
        return GaussianBasis(
            mean_params=mu[:, dims, :],
            std_params=std[:, dims, :],
            coeffs=self.coeffs(),
        )

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
            assert isinstance(other, GaussianBasis), "all factors must be GaussianBasis"
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
            coeff_terms.append(basis.coeffs)

        coeff_prod = torch.ones(n_per_factor, dtype=dtype, device=device)
        for k, c_k in enumerate(coeff_terms):
            view_shape = [1] * n_factors
            view_shape[k] = n_per_factor[k]
            coeff_prod = coeff_prod * c_k.reshape(view_shape)
        coeffs_new = (coeff_prod * torch.exp(log_const_flat.reshape(n_per_factor))).reshape(n_total)

        return GaussianBasis(
            params=(
                Parameters.from_values(mu_flat, trainable=False),
                self._params[1].with_fixed_values(sigma_flat),
            ),
            coeffs=PositiveParameters.from_values(coeffs_new, trainable=False),
        )


class BetaBasis(SeparableBasis, NonnegativeBasis):
    """Separable product of normalized 1D Beta PDFs on (0, 1)."""

    def __init__(
        self,
        alpha_params: PositiveParameters,
        beta_params: PositiveParameters,
        coeffs: Parameters = None,
        eps: float = 1e-6,
    ):
        super().__init__(params=(alpha_params, beta_params), coeffs=coeffs)
        self.eps = eps

    def alphas_betas(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._params[0](), self._params[1]()

    def __call__(self, y: torch.Tensor):
        out = self.eval_dim(y).prod(dim=1)  # (batch, n_basis)
        return out * self.coeffs()

    def eval_dim(self, y: torch.Tensor):
        self._check_eval_batch(y)
        alpha, beta = self.alphas_betas()  # (batch_size, dim, n_basis)
        y_c = y.clamp(self.eps, 1.0 - self.eps)[:, :, None]  # (batch, dim, 1)
        log_dim = (
            (alpha - 1.0) * torch.log(y_c)
            + (beta - 1.0) * torch.log1p(-y_c)
            - BetaGram.log_beta(alpha, beta)
        )
        return torch.exp(log_dim)  # (batch, dim, n_basis)

    def log_Omega1_dim(self, lows: torch.Tensor = None, highs: torch.Tensor = None):
        alpha, beta = self.alphas_betas()
        lows_b, highs_b = self._gram_domain_bounds(lows, highs, alpha)
        return BetaGram.log_gram((alpha,), (beta,), lows=lows_b, highs=highs_b)

    def log_Omega2_dim(self, other: "BetaBasis", lows: torch.Tensor = None, highs: torch.Tensor = None):
        assert isinstance(other, BetaBasis), "other must be BetaBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"
        assert self.batch_size() == other.batch_size(), "Basis functions must have the same batch size"
        a1, b1 = self.alphas_betas()
        a2, b2 = other.alphas_betas()
        lows_b, highs_b = self._gram_domain_bounds(lows, highs, a1)
        return BetaGram.log_gram((a1, a2), (b1, b2), lows=lows_b, highs=highs_b)

    def log_Omega3_dim(
        self,
        other1: "BetaBasis",
        other2: "BetaBasis",
        lows: torch.Tensor = None,
        highs: torch.Tensor = None,
    ):
        assert isinstance(other1, BetaBasis), "other1 must be BetaBasis"
        assert isinstance(other2, BetaBasis), "other2 must be BetaBasis"
        assert self.dim() == other1.dim() == other2.dim(), "Basis functions must have the same dimension"
        assert self.batch_size() == other1.batch_size() == other2.batch_size(), "Basis functions must have the same batch size"
        a1, b1 = self.alphas_betas()
        a2, b2 = other1.alphas_betas()
        a3, b3 = other2.alphas_betas()
        lows_b, highs_b = self._gram_domain_bounds(lows, highs, a1)
        return BetaGram.log_gram((a1, a2, a3), (b1, b2, b3), lows=lows_b, highs=highs_b)

    def log_Omega22_dim(self, other: "BetaBasis", lows: torch.Tensor = None, highs: torch.Tensor = None):
        assert isinstance(other, BetaBasis), "other must be BetaBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"
        assert self.batch_size() == other.batch_size(), "Basis functions must have the same batch size"
        a1, b1 = self.alphas_betas()
        a2, b2 = other.alphas_betas()
        lows_b, highs_b = self._gram_domain_bounds(lows, highs, a1)
        return BetaGram.log_gram((a1, a1, a2, a2), (b1, b1, b2, b2), lows=lows_b, highs=highs_b)

    def marginal(self, marginal_dims: tuple[int, ...]) -> "BetaBasis":
        dims = tuple(marginal_dims)
        assert all(0 <= i < self.dim() for i in dims), "marginal_dims must be in [0, d)"
        alpha, beta = self.alphas_betas()
        return BetaBasis(
            alpha_params=alpha[:, dims, :],
            beta_params=beta[:, dims, :],
            coeffs=self.coeffs(),
            eps=self.eps,
        )


class GaussianKernelBasis(SeparableBasis, NonnegativeBasis):
    """Separable isotropic Gaussian kernels N(x_r | mean, std^2) per coordinate."""

    def __init__(
        self,
        mean_params: Parameters,
        std_params: PositiveParameters,
        coeffs: Parameters = None,
    ):
        super().__init__(params=(mean_params, std_params), coeffs=coeffs)

    def means_stds(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._params[0](), self._params[1]()

    def forward(self, y: torch.Tensor, ignore_coeffs: bool = False):
        out = self.eval_dim(y).prod(dim=1)  # (batch, n_basis)
        if not ignore_coeffs:
            out = out * self.coeffs()
        return out

    def eval_dim(self, y: torch.Tensor):
        self._check_eval_batch(y)
        mu, std = self.means_stds()  # (batch_size, dim, n_basis)
        y_e = y[:, :, None]  # (batch, dim, 1)
        std = std.clamp_min(torch.finfo(y.dtype).eps)
        log_two_pi = y.new_tensor(2.0 * math.pi).log()
        log_dim = -0.5 * (log_two_pi + 2.0 * torch.log(std) + ((y_e - mu) / std).square())
        return torch.exp(log_dim)  # (batch, dim, n_basis)

    def log_Omega1_dim(self, lows: torch.Tensor = None, highs: torch.Tensor = None):
        mu, std = self.means_stds()
        lows_b, highs_b = self._gram_domain_bounds(lows, highs, mu)
        return GaussianGram.log_gram((mu,), (std,), lows=lows_b, highs=highs_b)

    def log_Omega2_dim(self, other: "GaussianKernelBasis", lows: torch.Tensor = None, highs: torch.Tensor = None):
        assert isinstance(other, GaussianKernelBasis), "other must be GaussianKernelBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"
        assert self.batch_size() == other.batch_size(), "Basis functions must have the same batch size"
        mu1, std1 = self.means_stds()
        mu2, std2 = other.means_stds()
        lows_b, highs_b = self._gram_domain_bounds(lows, highs, mu1)
        return GaussianGram.log_gram((mu1, mu2), (std1, std2), lows=lows_b, highs=highs_b)

    def log_Omega3_dim(
        self,
        other1: "GaussianKernelBasis",
        other2: "GaussianKernelBasis",
        lows: torch.Tensor = None,
        highs: torch.Tensor = None,
    ):
        assert isinstance(other1, GaussianKernelBasis), "other1 must be GaussianKernelBasis"
        assert isinstance(other2, GaussianKernelBasis), "other2 must be GaussianKernelBasis"
        assert self.dim() == other1.dim() == other2.dim(), "Basis functions must have the same dimension"
        assert self.batch_size() == other1.batch_size() == other2.batch_size(), "Basis functions must have the same batch size"
        mu1, std1 = self.means_stds()
        mu2, std2 = other1.means_stds()
        mu3, std3 = other2.means_stds()
        lows_b, highs_b = self._gram_domain_bounds(lows, highs, mu1)
        return GaussianGram.log_gram((mu1, mu2, mu3), (std1, std2, std3), lows=lows_b, highs=highs_b)

    def log_Omega22_dim(self, other: "GaussianKernelBasis", lows: torch.Tensor = None, highs: torch.Tensor = None):
        assert isinstance(other, GaussianKernelBasis), "other must be GaussianKernelBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"
        assert self.batch_size() == other.batch_size(), "Basis functions must have the same batch size"
        mu1, std1 = self.means_stds()
        mu2, std2 = other.means_stds()
        lows_b, highs_b = self._gram_domain_bounds(lows, highs, mu1)
        return GaussianGram.log_gram((mu1, mu1, mu2, mu2), (std1, std1, std2, std2), lows=lows_b, highs=highs_b)

    def marginal(self, marginal_dims: tuple[int, ...]) -> "GaussianKernelBasis":
        dims = tuple(marginal_dims)
        assert all(0 <= i < self.dim() for i in dims), "marginal_dims must be in [0, d)"
        mu, std = self.means_stds()
        return GaussianKernelBasis(
            mean_params=mu[:, dims, :],
            std_params=std[:, dims, :],
            coeffs=self.coeffs(),
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
