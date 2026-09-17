import math

import torch
from abc import abstractmethod
from numpy.polynomial.legendre import leggauss

from .parameters import Parameters, FixedParameters, TrainableParameters, PositiveParameters
from .gram import BetaGram, GaussianGram
from .structured_matrices import Banded, DenseMatrix


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

    def _coeffs_register(self):
        return [self.coeffs]
    
    def _params_register(self):
        return [self._params]

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
            for coeffs in basis._coeffs_register():
                if coeffs.is_module() and id(coeffs) not in coeffs_seen:
                    unique_coeffs.append(coeffs)
                    coeffs_seen[id(coeffs)] = coeffs
            owner = getattr(basis, "owner", None)
            if isinstance(owner, torch.nn.Module):
                if id(owner) not in params_seen:
                    unique_params.append(owner)
                    params_seen[id(owner)] = owner
                continue
            for params in basis._params_register():
                for param in params:
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
        dtype, device = self.dtype_device()
        self.set_coeffs(FixedParameters(torch.ones(self._batch_size, self._n_basis, dtype=dtype, device=device)))
    
    def dim(self):
        return self._dim
    
    def batch_size(self):
        return self._batch_size
    
    def n_basis_functions(self):
        return self._n_basis
    
    def dtype_device(self):
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
        return DenseMatrix(
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

    def supremum_bound(self) -> torch.Tensor:
        """Product of per-coordinate Gaussian PDF maxima, shape ``(batch, n_basis)``.

        Each 1-D factor ``N(x | μ, σ²)`` peaks at ``x = μ`` with height
        ``(2π σ²)^{-1/2}``, so the separable product peaks at the mean with
        height ``∏_d (2π σ_d²)^{-1/2}``.
        """
        _, std = self.means_stds()
        std = std.clamp_min(torch.finfo(std.dtype).eps)
        log_two_pi = std.new_tensor(2.0 * math.pi).log()
        log_sup = (-0.5 * (log_two_pi + 2.0 * torch.log(std))).sum(dim=1)
        return log_sup.exp() * self.coeffs()

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
        dtype, device = self.dtype_device()
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
                TrainableParameters.from_values(mu_flat, trainable=False),
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

    def supremum_bound(self) -> torch.Tensor:
        """Product of per-coordinate Beta PDF suprema, shape ``(batch, n_basis)``."""
        alpha, beta = self.alphas_betas()
        a1 = (alpha - 1.0).clamp(min=0.0)
        b1 = (beta - 1.0).clamp(min=0.0)
        s = a1 + b1

        # 0 log 0 = 0 with zero subgradient; torch.xlogy(0,0) has NaN grads.
        def _xlogx(x: torch.Tensor) -> torch.Tensor:
            safe = torch.where(x > 0, x, torch.ones_like(x))
            return torch.where(x > 0, x * torch.log(safe), torch.zeros_like(x))

        log_sup = _xlogx(a1) + _xlogx(b1) - _xlogx(s) - BetaGram.log_beta(alpha, beta)
        finite = (alpha >= 1.0) & (beta >= 1.0)
        log_sup = torch.where(finite, log_sup, torch.full_like(log_sup, math.inf))
        return log_sup.sum(dim=1).exp() * self.coeffs()

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


class BSpline1DBasis(Basis, NonnegativeBasis):
    """Open-uniform B-splines on ``[0, 1]`` with ``n_cells`` equal spans.

    There are ``n_basis = n_cells + degree`` cardinal B-splines of degree
    ``degree``. Each interior basis is supported on ``degree + 1`` cells, so the
    Gram ``G_ij = ∫ N_i N_j`` is banded with half-bandwidth ``degree``.
    """

    def __init__(
        self,
        n_cells: int,
        degree: int = 3,
        batch_size: int = 1,
        coeffs: Parameters = None,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ):
        if degree < 0:
            raise ValueError("degree must be nonnegative")
        if n_cells < 1:
            raise ValueError("n_cells must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        n_basis = n_cells + degree
        self._degree = int(degree)
        self._n_cells = int(n_cells)
        self._n_basis = n_basis  # needed by _cell_gram before Basis.__init__
        self._knots = self.open_uniform_knots(n_basis, degree, dtype=dtype, device=device)
        self._breaks = torch.linspace(0.0, 1.0, n_cells + 1, dtype=dtype, device=device)
        self._mass = self.basis_mass(self._knots, degree)
        self._gram = self._pack_banded(self._cell_gram())

        super().__init__(dim=1, batch_size=batch_size, n_basis=n_basis, params=(), coeffs=coeffs)

    # ------------------------------------------------------------------
    # Knot / evaluation primitives (shared with BSpline1D densities)
    # ------------------------------------------------------------------

    @staticmethod
    def open_uniform_knots(
        n_basis: int,
        degree: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Open-uniform knot vector on ``[0, 1]`` with ``n_basis`` degree-``degree`` B-splines."""
        if degree < 0:
            raise ValueError("degree must be nonnegative")
        if n_basis < degree + 1:
            raise ValueError("n_basis must be at least degree + 1")
        n_spans = n_basis - degree
        breaks = torch.linspace(0.0, 1.0, n_spans + 1, dtype=dtype, device=device)
        return torch.cat(
            [
                torch.zeros(degree + 1, dtype=dtype, device=device),
                breaks[1:-1],
                torch.ones(degree + 1, dtype=dtype, device=device),
            ]
        )

    @staticmethod
    def basis_mass(knots: torch.Tensor, degree: int) -> torch.Tensor:
        """Exact integrals ``∫ N_{i,p}`` via ``(t_{i+p+1} - t_i) / (p + 1)``."""
        n_basis = knots.numel() - degree - 1
        return (knots[degree + 1 : degree + 1 + n_basis] - knots[:n_basis]) / (degree + 1.0)

    @staticmethod
    def eval_basis(
        x: torch.Tensor,
        knots: torch.Tensor,
        degree: int,
        n_basis: int,
    ) -> torch.Tensor:
        """Cox–de Boor evaluation of every open-uniform B-spline at ``x``.

        ``x`` has shape ``(n,)``. Returns ``(n, n_basis)``, nonnegative and
        summing to 1 (partition of unity). Built without in-place writes so
        gradients w.r.t. ``x`` are well-defined.
        """
        x = x.reshape(-1)
        x = torch.nan_to_num(x, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        t = knots
        p = degree
        n = n_basis - 1

        span = torch.searchsorted(t, x, right=True) - 1
        span = torch.where(x >= t[n + 1], torch.full_like(span, n), span)
        span = span.clamp(p, n)

        N_curr = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
        eps = torch.finfo(x.dtype).eps
        for j in range(1, p + 1):
            ks = torch.arange(1, j + 1, device=x.device)
            left = x.unsqueeze(1) - t[span.unsqueeze(1) + 1 - ks]
            right = t[span.unsqueeze(1) + ks] - x.unsqueeze(1)
            cols: list[torch.Tensor] = []
            saved = torch.zeros_like(x)
            for r in range(j):
                den = right[:, r] + left[:, j - 1 - r]
                tmp = torch.where(den.abs() > eps, N_curr[:, r] / den, torch.zeros_like(den))
                cols.append(saved + right[:, r] * tmp)
                saved = left[:, j - 1 - r] * tmp
            cols.append(saved)
            N_curr = torch.stack(cols, dim=1)

        idx = span.unsqueeze(1) - p + torch.arange(p + 1, device=x.device)
        idx = idx.clamp(0, n_basis - 1)
        return x.new_zeros(x.shape[0], n_basis).scatter(1, idx, N_curr)

    # ------------------------------------------------------------------
    # Properties / evaluation
    # ------------------------------------------------------------------

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def bandwidth(self) -> int:
        """Half-bandwidth of the Gram matrix (equal to ``degree``)."""
        return self._degree

    @property
    def n_cells(self) -> int:
        return self._n_cells

    @property
    def knots(self) -> torch.Tensor:
        return self._knots

    @property
    def breakpoints(self) -> torch.Tensor:
        return self._breaks

    @property
    def mass(self) -> torch.Tensor:
        return self._mass

    def dtype_device(self):
        return self._knots.dtype, self._knots.device

    def eval(self, y: torch.Tensor) -> torch.Tensor:
        """Evaluate all B-splines. ``y`` is ``(n,)`` or ``(n, 1)``; returns ``(n, n_basis)``."""
        y = torch.as_tensor(y, dtype=self._knots.dtype, device=self._knots.device)
        return self.eval_basis(y.reshape(-1), self._knots, self._degree, self._n_basis)

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        vals = self.eval(y)  # (n, n_basis)
        n = vals.shape[0]
        b = self.batch_size()
        assert b == n or b == 1, (
            f"y batch {n} must match parameter batch {b} (or parameter batch must be 1)"
        )
        return vals * self.coeffs()

    def _cell_gram(self, lo: float | torch.Tensor = 0.0, hi: float | torch.Tensor = 1.0) -> torch.Tensor:
        """Dense ``∫_{[lo,hi]} N_i N_j`` via cellwise Gauss–Legendre (exact for degree ``2p``)."""
        p = self._degree
        m = self._n_basis
        dtype, device = self.dtype_device()
        qx, qw = leggauss(p + 1)
        qx = torch.as_tensor(qx, dtype=dtype, device=device)
        qw = torch.as_tensor(qw, dtype=dtype, device=device)

        lo_t = torch.as_tensor(lo, dtype=dtype, device=device).reshape(())
        hi_t = torch.as_tensor(hi, dtype=dtype, device=device).reshape(())
        left = torch.maximum(self._breaks[:-1], lo_t)
        right = torch.minimum(self._breaks[1:], hi_t)
        half = (0.5 * (right - left)).clamp(min=0.0)
        mid = 0.5 * (right + left)
        x = mid[:, None] + half[:, None] * qx[None, :]
        N = self.eval_basis(x.reshape(-1), self._knots, p, m).reshape(self._n_cells, -1, m)
        w = (half[:, None] * qw[None, :]).unsqueeze(-1)
        G = torch.einsum("cqi,cqj->ij", N * w, N)
        return 0.5 * (G + G.T)

    def _pack_banded(self, G: torch.Tensor) -> Banded:
        p, m = self._degree, self._n_basis
        offsets = torch.arange(-p, p + 1, device=G.device)
        data = G.new_zeros(2 * p + 1, m)
        for r, off in enumerate(range(-p, p + 1)):
            diag = G.diagonal(offset=-off)
            if off >= 0:
                data[r, : m - off] = diag
            else:
                data[r, -off:] = diag
        return Banded(offsets, data)

    def Omega2(
        self,
        other: "BSpline1DBasis",
        lows: torch.Tensor = None,
        highs: torch.Tensor = None,
    ) -> Banded:
        assert isinstance(other, BSpline1DBasis), "other must be BSpline1DBasis"
        assert self._n_cells == other._n_cells and self._degree == other._degree, (
            "BSpline1DBasis Omega2 requires matching n_cells and degree"
        )
        assert self.batch_size() == other.batch_size(), "batch sizes must match"

        if lows is None and highs is None:
            geom = self._gram
        else:
            lo = 0.0 if lows is None else lows
            hi = 1.0 if highs is None else highs
            geom = self._pack_banded(self._cell_gram(lo, hi))

        data = geom.data.unsqueeze(0).expand(self._batch_size, -1, -1)
        return (
            Banded(geom.offsets, data)
            .mul_diag_left(self.coeffs())
            .mul_diag_right(other.coeffs())
        )
