import math

import torch
from abc import abstractmethod
from functools import lru_cache
from numpy.polynomial.legendre import leggauss

from .parameters import Parameters, FixedParameters, TrainableParameters, PositiveParameters, OneVectorParameters
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
            params : list of tensors where each represents a parameter group 
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
            coeffs = basis.coeffs
            if coeffs.is_module() and id(coeffs) not in coeffs_seen:
                unique_coeffs.append(coeffs)
                coeffs_seen[id(coeffs)] = coeffs
            for module in coeffs.parameter_modules():
                if id(module) not in coeffs_seen:
                    unique_coeffs.append(module)
                    coeffs_seen[id(module)] = module
            owner = getattr(basis, "owner", None)
            if isinstance(owner, torch.nn.Module):
                if id(owner) not in params_seen:
                    unique_params.append(owner)
                    params_seen[id(owner)] = owner
                continue
            for param in basis._params:
                if param.is_module() and id(param) not in params_seen:
                    unique_params.append(param)
                    params_seen[id(param)] = param
                for module in param.parameter_modules():
                    if id(module) not in params_seen:
                        unique_params.append(module)
                        params_seen[id(module)] = module
        return unique_params, unique_coeffs

    def forward(self, y : torch.Tensor):
        raise NotImplementedError("__call__ is not implemented for this basis function")

    def set_coeffs(self, coeffs : Parameters):
        assert isinstance(coeffs, Parameters), "coeffs must be a Parameters object"
        vals = coeffs()
        shape = tuple(vals.shape) if hasattr(vals, "shape") else tuple(vals.size())
        # Dense / OneVector coeffs are (batch_size, n_basis). Unbatched structured
        # vectors (e.g. TTVector) expose shape (n_basis,) when batch_size == 1.
        if shape == (self._batch_size, self._n_basis):
            pass
        elif self._batch_size == 1 and shape == (self._n_basis,):
            pass
        else:
            raise AssertionError(
                "coeffs must have shape "
                f"({self._batch_size}, {self._n_basis})"
                + (f" or ({self._n_basis},)" if self._batch_size == 1 else "")
                + f", got {shape}"
            )
        self.coeffs = coeffs
    
    def set_coeffs_to_one(self):
        dtype, device = self.dtype_device()
        self.set_coeffs(OneVectorParameters(self._n_basis, (self._batch_size,), dtype, device))
    
    @staticmethod
    def _coeff_tensor(coeffs_vals):
        """Materialize coeffs to a dense tensor (supports structured Vectors)."""
        if hasattr(coeffs_vals, "to_dense"):
            return coeffs_vals.to_dense()
        return coeffs_vals 
    def params(self):
        return self._params
    
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

    def Omega1_dim(self, lows : torch.Tensor = None, highs : torch.Tensor = None, return_log : bool = False):
        '''
        Per-coordinate integrals of each basis function.

        Returns:
            ``(batch_size, dim, n_basis)``
        '''
        raise NotImplementedError("Omega1_dim is not implemented for this basis function")
    
    def Omega2_dim(self, other : 'Basis', lows : torch.Tensor = None, highs : torch.Tensor = None, return_log : bool = False):
        '''
        Per-coordinate inner-product matrices.

        Returns:
            ``(batch_size, dim, n_basis, other.n_basis)`` with
            ``out[..., d, i, j] = <this_i, other_j>_d``, or its log when
            ``return_log=True`` (only well-defined for nonnegative entries).
        '''
        raise NotImplementedError("Omega2_dim is not implemented for this basis function")

    def Omega3_dim(self, other1 : 'Basis', other2 : 'Basis', lows : torch.Tensor = None, highs : torch.Tensor = None, return_log : bool = False):
        '''
        Per-coordinate triple-product tensors.

        Returns:
            ``(batch_size, dim, n_basis, other1.n_basis, other2.n_basis)``
        '''
        raise NotImplementedError("Omega3_dim is not implemented for this basis function")

    def Omega22_dim(self, other : 'Basis', lows : torch.Tensor = None, highs : torch.Tensor = None, return_log : bool = False):
        '''
        Per-coordinate grams for ``<this_i * this_j, other_k * other_l>``.

        Returns:
            ``(batch_size, dim, n_basis, n_basis, other.n_basis, other.n_basis)``
        '''
        raise NotImplementedError("Omega22_dim is not implemented for this basis function")

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
        if isinstance(self, NonnegativeBasis):
            return torch.exp(self.Omega1_dim(lows, highs, return_log=True).sum(dim=1)) * self.coeffs()
        else:
            return self.Omega1_dim(lows, highs, return_log=False).prod(dim=1) * self.coeffs()

    def Omega2(self, other : 'Basis', lows : torch.Tensor = None, highs : torch.Tensor = None):
        if isinstance(self, NonnegativeBasis):
            raw = torch.exp(self.Omega2_dim(other, lows, highs, return_log=True).sum(dim=1))
        else:
            raw = self.Omega2_dim(other, lows, highs, return_log=False).prod(dim=1)
        base = DenseMatrix(raw)
        return base.mul_diag_left(Basis._coeff_tensor(self.coeffs())).mul_diag_right(
            Basis._coeff_tensor(other.coeffs())
        )

    def Omega3(self, other1 : 'Basis', other2 : 'Basis', lows : torch.Tensor = None, highs : torch.Tensor = None):
        c0 = Basis._coeff_tensor(self.coeffs())
        c1 = Basis._coeff_tensor(other1.coeffs())
        c2 = Basis._coeff_tensor(other2.coeffs())
        if isinstance(self, NonnegativeBasis):
            core = torch.exp(self.Omega3_dim(other1, other2, lows, highs, return_log=True).sum(dim=1))
        else:
            core = self.Omega3_dim(other1, other2, lows, highs, return_log=False).prod(dim=1)
        return (
            core
            * c0[:, :, None, None]
            * c1[:, None, :, None]
            * c2[:, None, None, :]
        )

    def Omega22(self, other : 'Basis', lows : torch.Tensor = None, highs : torch.Tensor = None):
        c1 = Basis._coeff_tensor(self.coeffs())
        c2 = Basis._coeff_tensor(other.coeffs())
        if isinstance(self, NonnegativeBasis):
            core = torch.exp(self.Omega22_dim(other, lows, highs, return_log=True).sum(dim=1))
        else:
            core = self.Omega22_dim(other, lows, highs, return_log=False).prod(dim=1)
        return (
            core
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

    def Omega1_dim(self, lows: torch.Tensor = None, highs: torch.Tensor = None, return_log: bool = False):
        mu, std = self.means_stds()
        lows_b, highs_b = self._gram_domain_bounds(lows, highs, mu)
        log_val = GaussianGram.log_gram((mu,), (std,), lows=lows_b, highs=highs_b)
        return log_val if return_log else torch.exp(log_val)

    def Omega2_dim(self, other: "GaussianBasis", lows: torch.Tensor = None, highs: torch.Tensor = None, return_log: bool = False):
        assert isinstance(other, GaussianBasis), "other must be GaussianBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"
        assert self.batch_size() == other.batch_size(), "Basis functions must have the same batch size"
        mu1, std1 = self.means_stds()
        mu2, std2 = other.means_stds()
        lows_b, highs_b = self._gram_domain_bounds(lows, highs, mu1)
        log_val = GaussianGram.log_gram((mu1, mu2), (std1, std2), lows=lows_b, highs=highs_b)
        return log_val if return_log else torch.exp(log_val)

    def Omega3_dim(
        self,
        other1: "GaussianBasis",
        other2: "GaussianBasis",
        lows: torch.Tensor = None,
        highs: torch.Tensor = None,
        return_log: bool = False,
    ):
        assert isinstance(other1, GaussianBasis), "other1 must be GaussianBasis"
        assert isinstance(other2, GaussianBasis), "other2 must be GaussianBasis"
        assert self.dim() == other1.dim() == other2.dim(), "Basis functions must have the same dimension"
        assert self.batch_size() == other1.batch_size() == other2.batch_size(), "Basis functions must have the same batch size"
        mu1, std1 = self.means_stds()
        mu2, std2 = other1.means_stds()
        mu3, std3 = other2.means_stds()
        lows_b, highs_b = self._gram_domain_bounds(lows, highs, mu1)
        log_val = GaussianGram.log_gram((mu1, mu2, mu3), (std1, std2, std3), lows=lows_b, highs=highs_b)
        return log_val if return_log else torch.exp(log_val)

    def Omega22_dim(self, other: "GaussianBasis", lows: torch.Tensor = None, highs: torch.Tensor = None, return_log: bool = False):
        assert isinstance(other, GaussianBasis), "other must be GaussianBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"
        assert self.batch_size() == other.batch_size(), "Basis functions must have the same batch size"
        mu1, std1 = self.means_stds()
        mu2, std2 = other.means_stds()
        lows_b, highs_b = self._gram_domain_bounds(lows, highs, mu1)
        log_val = GaussianGram.log_gram((mu1, mu1, mu2, mu2), (std1, std1, std2, std2), lows=lows_b, highs=highs_b)
        return log_val if return_log else torch.exp(log_val)

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

    def Omega1_dim(self, lows: torch.Tensor = None, highs: torch.Tensor = None, return_log: bool = False):
        alpha, beta = self.alphas_betas()
        lows_b, highs_b = self._gram_domain_bounds(lows, highs, alpha)
        log_val = BetaGram.log_gram((alpha,), (beta,), lows=lows_b, highs=highs_b)
        return log_val if return_log else torch.exp(log_val)

    def Omega2_dim(self, other: "BetaBasis", lows: torch.Tensor = None, highs: torch.Tensor = None, return_log: bool = False):
        assert isinstance(other, BetaBasis), "other must be BetaBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"
        assert self.batch_size() == other.batch_size(), "Basis functions must have the same batch size"
        a1, b1 = self.alphas_betas()
        a2, b2 = other.alphas_betas()
        lows_b, highs_b = self._gram_domain_bounds(lows, highs, a1)
        log_val = BetaGram.log_gram((a1, a2), (b1, b2), lows=lows_b, highs=highs_b)
        return log_val if return_log else torch.exp(log_val)

    def Omega3_dim(
        self,
        other1: "BetaBasis",
        other2: "BetaBasis",
        lows: torch.Tensor = None,
        highs: torch.Tensor = None,
        return_log: bool = False,
    ):
        assert isinstance(other1, BetaBasis), "other1 must be BetaBasis"
        assert isinstance(other2, BetaBasis), "other2 must be BetaBasis"
        assert self.dim() == other1.dim() == other2.dim(), "Basis functions must have the same dimension"
        assert self.batch_size() == other1.batch_size() == other2.batch_size(), "Basis functions must have the same batch size"
        a1, b1 = self.alphas_betas()
        a2, b2 = other1.alphas_betas()
        a3, b3 = other2.alphas_betas()
        lows_b, highs_b = self._gram_domain_bounds(lows, highs, a1)
        log_val = BetaGram.log_gram((a1, a2, a3), (b1, b2, b3), lows=lows_b, highs=highs_b)
        return log_val if return_log else torch.exp(log_val)

    def Omega22_dim(self, other: "BetaBasis", lows: torch.Tensor = None, highs: torch.Tensor = None, return_log: bool = False):
        assert isinstance(other, BetaBasis), "other must be BetaBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"
        assert self.batch_size() == other.batch_size(), "Basis functions must have the same batch size"
        a1, b1 = self.alphas_betas()
        a2, b2 = other.alphas_betas()
        lows_b, highs_b = self._gram_domain_bounds(lows, highs, a1)
        log_val = BetaGram.log_gram((a1, a1, a2, a2), (b1, b1, b2, b2), lows=lows_b, highs=highs_b)
        return log_val if return_log else torch.exp(log_val)

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

    def Omega1_dim(self, lows: torch.Tensor = None, highs: torch.Tensor = None, return_log: bool = False):
        mu, std = self.means_stds()
        lows_b, highs_b = self._gram_domain_bounds(lows, highs, mu)
        log_val = GaussianGram.log_gram((mu,), (std,), lows=lows_b, highs=highs_b)
        return log_val if return_log else torch.exp(log_val)

    def Omega2_dim(self, other: "GaussianKernelBasis", lows: torch.Tensor = None, highs: torch.Tensor = None, return_log: bool = False):
        assert isinstance(other, GaussianKernelBasis), "other must be GaussianKernelBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"
        assert self.batch_size() == other.batch_size(), "Basis functions must have the same batch size"
        mu1, std1 = self.means_stds()
        mu2, std2 = other.means_stds()
        lows_b, highs_b = self._gram_domain_bounds(lows, highs, mu1)
        log_val = GaussianGram.log_gram((mu1, mu2), (std1, std2), lows=lows_b, highs=highs_b)
        return log_val if return_log else torch.exp(log_val)

    def Omega3_dim(
        self,
        other1: "GaussianKernelBasis",
        other2: "GaussianKernelBasis",
        lows: torch.Tensor = None,
        highs: torch.Tensor = None,
        return_log: bool = False,
    ):
        assert isinstance(other1, GaussianKernelBasis), "other1 must be GaussianKernelBasis"
        assert isinstance(other2, GaussianKernelBasis), "other2 must be GaussianKernelBasis"
        assert self.dim() == other1.dim() == other2.dim(), "Basis functions must have the same dimension"
        assert self.batch_size() == other1.batch_size() == other2.batch_size(), "Basis functions must have the same batch size"
        mu1, std1 = self.means_stds()
        mu2, std2 = other1.means_stds()
        mu3, std3 = other2.means_stds()
        lows_b, highs_b = self._gram_domain_bounds(lows, highs, mu1)
        log_val = GaussianGram.log_gram((mu1, mu2, mu3), (std1, std2, std3), lows=lows_b, highs=highs_b)
        return log_val if return_log else torch.exp(log_val)

    def Omega22_dim(self, other: "GaussianKernelBasis", lows: torch.Tensor = None, highs: torch.Tensor = None, return_log: bool = False):
        assert isinstance(other, GaussianKernelBasis), "other must be GaussianKernelBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"
        assert self.batch_size() == other.batch_size(), "Basis functions must have the same batch size"
        mu1, std1 = self.means_stds()
        mu2, std2 = other.means_stds()
        lows_b, highs_b = self._gram_domain_bounds(lows, highs, mu1)
        log_val = GaussianGram.log_gram((mu1, mu1, mu2, mu2), (std1, std1, std2, std2), lows=lows_b, highs=highs_b)
        return log_val if return_log else torch.exp(log_val)

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
    """Uniform B-splines on [0, 1].

    ``cellwise=False``:
        Standard open-uniform B-spline basis.

    ``cellwise=True``:
        Independent degree-p Bernstein basis on every cell.  This is the
        exact representation used by products/shifts.
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
        cellwise: bool = False,
    ):
        if degree < 0:
            raise ValueError("degree must be nonnegative")
        if n_cells < 1:
            raise ValueError("n_cells must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self._degree = int(degree)
        self._n_cells = int(n_cells)
        self._cellwise = bool(cellwise)

        if cellwise:
            self._n_basis = n_cells * (degree + 1)
            self._knots = self.cellwise_knots(
                n_cells, degree, dtype=dtype, device=device
            )
        else:
            self._n_basis = n_cells + degree
            self._knots = self.open_uniform_knots(
                self._n_basis, degree, dtype=dtype, device=device
            )

        self._breaks = torch.linspace(
            0.0, 1.0, n_cells + 1, dtype=dtype, device=device
        )
        self._mass = self.basis_mass(self._knots, degree)
        self._gram = self._pack_banded(self._cell_gram())

        super().__init__(
            dim=1,
            batch_size=batch_size,
            n_basis=self._n_basis,
            params=(),
            coeffs=coeffs,
        )

    @property
    def cellwise(self) -> bool:
        return self._cellwise

    @staticmethod
    def cellwise_knots(
        n_cells: int,
        degree: int,
        *,
        dtype: torch.dtype,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Degree-p Bernstein block independently on every cell."""
        breaks = torch.linspace(
            0.0, 1.0, n_cells + 1, dtype=dtype, device=device
        )
        return breaks.repeat_interleave(degree + 1)

    # ------------------------------------------------------------------
    # Cellwise conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _bernstein_matrix(
        degree: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interior interpolation nodes and Bernstein evaluation matrix."""
        p = degree
        if p == 0:
            u = torch.tensor([0.5], dtype=dtype, device=device)
        else:
            j = torch.arange(p + 1, dtype=dtype, device=device)
            u = 0.5 * (
                1.0
                - torch.cos(torch.pi * (j + 0.5) / (p + 1))
            )

        i = torch.arange(p + 1, dtype=dtype, device=device)
        comb = torch.tensor(
            [math.comb(p, k) for k in range(p + 1)],
            dtype=dtype,
            device=device,
        )

        B = (
            comb[None, :]
            * u[:, None] ** i[None, :]
            * (1.0 - u[:, None]) ** (p - i)[None, :]
        )
        return u, B

    def _cell_coeffs(self) -> torch.Tensor:
        """Return coefficients as ``(batch, n_cells, degree+1)``."""
        c = self.coeffs()
        if c.ndim == 1:
            c = c.unsqueeze(0)

        if self._cellwise:
            return c.reshape(c.shape[0], self._n_cells, self._degree + 1)

        p = self._degree
        dtype, device = self.dtype_device()

        u, E = self._bernstein_matrix(
            p, dtype=dtype, device=device
        )

        h = 1.0 / self._n_cells
        x = self._breaks[:-1, None] + h * u[None, :]

        # (cells, p+1, n_basis)
        B = self.eval_basis(
            x.reshape(-1),
            self._knots,
            p,
            self._n_basis,
        ).reshape(self._n_cells, p + 1, self._n_basis)

        # (batch, cells, p+1)
        values = torch.einsum("bi,cji->bcj", c, B)

        # E @ bernstein_coeffs = sampled_values
        return torch.linalg.solve(
            E, values.transpose(-1, -2)
        ).transpose(-1, -2)

    @classmethod
    def _from_cells(
        cls,
        coeffs: torch.Tensor,
        n_cells: int,
        degree: int,
    ) -> "BSpline1DBasis":
        """Construct an exact cellwise basis from (..., cells, degree+1)."""
        coeffs = coeffs.flatten(-2)
        return cls(
            n_cells=n_cells,
            degree=degree,
            batch_size=coeffs.shape[0],
            coeffs=FixedParameters(coeffs),
            dtype=coeffs.dtype,
            device=coeffs.device,
            cellwise=True,
        )

    # ------------------------------------------------------------------
    # Integer-cell shift
    # ------------------------------------------------------------------

    def shift_cells(self, shift: int) -> "BSpline1DBasis":
        """Return ``g(x) = f(x + shift / n_cells)``.

        Values shifted outside [0,1] are exactly zero.
        """
        shift = int(shift)
        c = self._cell_coeffs()
        n = self._n_cells

        if abs(shift) >= n:
            out = torch.zeros_like(c)

        elif shift > 0:
            z = c.new_zeros(c.shape[0], shift, c.shape[-1])
            out = torch.cat((c[:, shift:], z), dim=1)

        elif shift < 0:
            s = -shift
            z = c.new_zeros(c.shape[0], s, c.shape[-1])
            out = torch.cat((z, c[:, : n - s]), dim=1)

        else:
            out = c

        return self._from_cells(out, n, self._degree)

    # ------------------------------------------------------------------
    # Exact product
    # ------------------------------------------------------------------

    def product(self, other: "BSpline1DBasis") -> "BSpline1DBasis":
        """Exact pointwise product using FFT Bernstein convolution."""
        if not isinstance(other, BSpline1DBasis):
            raise TypeError("other must be BSpline1DBasis")
        if self._n_cells != other._n_cells:
            raise ValueError("product requires matching n_cells")

        a = self._cell_coeffs()
        b = other._cell_coeffs()

        ba, bb = a.shape[0], b.shape[0]
        if ba != bb and ba != 1 and bb != 1:
            raise ValueError(
                f"incompatible batch sizes {ba} and {bb}"
            )

        p, q = self._degree, other._degree
        r = p + q

        dtype, device = a.dtype, a.device

        cp = torch.tensor(
            [math.comb(p, i) for i in range(p + 1)],
            dtype=dtype,
            device=device,
        )
        cq = torch.tensor(
            [math.comb(q, i) for i in range(q + 1)],
            dtype=dtype,
            device=device,
        )
        cr = torch.tensor(
            [math.comb(r, i) for i in range(r + 1)],
            dtype=dtype,
            device=device,
        )

        # Bernstein multiplication reduces to ordinary convolution after
        # weighting by binomial coefficients.
        a = a * cp
        b = b * cq

        n_out = r + 1
        n_fft = 1 << (n_out - 1).bit_length()

        work_dtype = (
            torch.float32
            if dtype in (torch.float16, torch.bfloat16)
            else dtype
        )

        A = torch.fft.rfft(a.to(work_dtype), n=n_fft, dim=-1)
        B = torch.fft.rfft(b.to(work_dtype), n=n_fft, dim=-1)

        out = torch.fft.irfft(
            A * B, n=n_fft, dim=-1
        )[..., :n_out]

        out = (out / cr.to(work_dtype)).to(dtype)

        return self._from_cells(out, self._n_cells, r)

@lru_cache(maxsize=32)
def _polynomial_quad(order: int):
    return leggauss(order)


class UnitBoxPolynomialBasis(SeparableBasis):
    """Fixed, per-coordinate polynomial features (the full Kronecker basis stays implicit)."""

    def __init__(self, dim: int, n_basis: int, batch_size: int = 1, *,
                 dtype: torch.dtype = torch.float32, device=None):
        if dim < 1 or n_basis < 1 or batch_size < 1:
            raise ValueError("dim, n_basis and batch_size must be positive")
        # SeparableBasis requires a parameter tensor; this is fixed bookkeeping only.
        fixed = torch.zeros(batch_size, dim, n_basis, dtype=dtype, device=device)
        super().__init__(params=(FixedParameters(fixed),), coeffs=None)

    def _bounds(self, lows, highs):
        dtype, device = self.dtype_device()
        def bound(value, default):
            value = torch.as_tensor(default if value is None else value, dtype=dtype, device=device)
            return value.expand(self.dim()) if value.numel() == 1 else value.reshape(self.dim())
        lo, hi = bound(lows, 0.), bound(highs, 1.)
        if bool(((lo < 0) | (hi > 1) | (lo > hi)).any()):
            raise ValueError("Expected 0 <= lows <= highs <= 1 in each dimension")
        return lo, hi

    def _quadrature(self, degree, lows, highs):
        lo, hi = self._bounds(lows, highs)
        nodes, weights = _polynomial_quad(degree // 2 + 1)  # exact through 'degree'
        z = torch.as_tensor(nodes, dtype=lo.dtype, device=lo.device)[:, None]
        w = torch.as_tensor(weights, dtype=lo.dtype, device=lo.device)[:, None]
        return (lo + hi)[None, :] / 2 + (hi - lo)[None, :] * z / 2, w * (hi - lo)[None, :] / 2

    def gram_dim(self, other, lows=None, highs=None):
        """Real, signed 1D Gram matrices, shape (batch_size, dim, n_self, n_other)."""
        if type(self) is not type(other) or self.dim() != other.dim() or self.batch_size() != other.batch_size():
            raise ValueError("Gram requires the same basis type, dimension, and batch size")
        if lows is None and highs is None:
            g = self._full_gram(other)
            return g.expand(self.batch_size(), self.dim(), -1, -1)
        degree = self.n_basis_functions() + other.n_basis_functions() - 2
        x, w = self._quadrature(degree, lows, highs)
        g = torch.einsum('kdi,kdj,kd->dij', self._eval_local(x), other._eval_local(x), w)
        return g.unsqueeze(0).expand(self.batch_size(), -1, -1, -1)

    def _integrals_dim(self, lows=None, highs=None):
        if lows is None and highs is None:
            return self._full_integrals().expand(self.batch_size(), self.dim(), -1)
        x, w = self._quadrature(self.n_basis_functions() - 1, lows, highs)
        g = torch.einsum('kdi,kd->di', self._eval_local(x), w)
        return g.unsqueeze(0).expand(self.batch_size(), -1, -1)

    def eval_dim(self, y: torch.Tensor):
        self._check_eval_batch(y)
        return self._eval_local(y)

    @staticmethod
    def _maybe_log(values, return_log: bool):
        if not return_log:
            return values
        if bool((values < 0).any()):
            raise ValueError(
                "return_log=True is only defined when the per-dimension "
                "integrals are nonnegative (e.g. a NonnegativeBasis); "
                "signed bases such as ShiftedLegendreBasis must use "
                "return_log=False so negative values are preserved"
            )
        return torch.log(values)

    def _check_polynomial_other(self, other):
        if type(self) is not type(other) or self.dim() != other.dim() or self.batch_size() != other.batch_size():
            raise ValueError("Gram requires the same basis type, dimension, and batch size")

    def Omega1_dim(self, lows=None, highs=None, return_log: bool = False):
        return self._maybe_log(self._integrals_dim(lows, highs), return_log)

    def Omega2_dim(self, other, lows=None, highs=None, return_log: bool = False):
        return self._maybe_log(self.gram_dim(other, lows, highs), return_log)

    def Omega3_dim(self, other1, other2, lows=None, highs=None, return_log: bool = False):
        self._check_polynomial_other(other1)
        self._check_polynomial_other(other2)
        degree = (
            (self.n_basis_functions() - 1)
            + (other1.n_basis_functions() - 1)
            + (other2.n_basis_functions() - 1)
        )
        x, w = self._quadrature(degree, lows, highs)
        g = torch.einsum(
            'qdi,qdj,qdk,qd->dijk',
            self._eval_local(x), other1._eval_local(x), other2._eval_local(x), w,
        )
        g = g.unsqueeze(0).expand(self.batch_size(), -1, -1, -1, -1)
        return self._maybe_log(g, return_log)

    def Omega22_dim(self, other, lows=None, highs=None, return_log: bool = False):
        self._check_polynomial_other(other)
        degree = 2 * (self.n_basis_functions() - 1) + 2 * (other.n_basis_functions() - 1)
        x, w = self._quadrature(degree, lows, highs)
        a = self._eval_local(x)
        b = other._eval_local(x)
        g = torch.einsum('qdi,qdj,qdk,qdl,qd->dijkl', a, a, b, b, w)
        g = g.unsqueeze(0).expand(self.batch_size(), -1, -1, -1, -1, -1)
        return self._maybe_log(g, return_log)


class ShiftedLegendreBasis(UnitBoxPolynomialBasis):
    """Orthonormal shifted Legendre polynomials sqrt(2j+1) P_j(2x-1)."""

    def __init__(self, dim: int, n_basis: int, batch_size: int = 1, **kwargs):
        super().__init__(dim, n_basis, batch_size, **kwargs)
        dtype, device = self.dtype_device()
        self._scales = (2 * torch.arange(n_basis, dtype=dtype, device=device) + 1).sqrt()

    def _eval_local(self, y):
        z = 2 * y - 1
        polys = [torch.ones_like(z)]
        if self.n_basis_functions() > 1:
            polys.append(z)
        for j in range(1, self.n_basis_functions() - 1):
            polys.append(((2*j + 1)*z*polys[-1] - j*polys[-2]) / (j + 1))
        out = torch.stack(polys, dim=-1)
        return out * self._scales.to(dtype=y.dtype, device=y.device)

    def _full_gram(self, other):
        a = torch.arange(self.n_basis_functions(), device=self.dtype_device()[1])
        b = torch.arange(other.n_basis_functions(), device=a.device)
        return (a[:, None] == b[None, :]).to(self.dtype_device()[0])[None, None]

    def _full_integrals(self):
        dtype, device = self.dtype_device()
        return torch.nn.functional.one_hot(torch.tensor(0, device=device), self.n_basis_functions()).to(dtype)[None, None]


class BernsteinBasis(UnitBoxPolynomialBasis, NonnegativeBasis):
    """Degree-(n_basis-1) Bernstein polynomials on [0, 1]."""

    def __init__(self, dim: int, n_basis: int, batch_size: int = 1, **kwargs):
        super().__init__(dim, n_basis, batch_size, **kwargs)
        dtype, device = self.dtype_device()
        self._binom = torch.tensor([math.comb(n_basis - 1, j) for j in range(n_basis)],
                                   dtype=dtype, device=device)

    def _eval_local(self, y):
        p = self.n_basis_functions() - 1
        j = torch.arange(p + 1, dtype=y.dtype, device=y.device)
        x = y.unsqueeze(-1)
        return self._binom.to(dtype=y.dtype, device=y.device) * x.pow(j) * (1 - x).pow(p - j)

    def _full_gram(self, other):
        return self._full_log_gram(other).exp()[None, None]

    def _full_log_gram(self, other):
        dtype, device = self.dtype_device()
        p, q = self.n_basis_functions() - 1, other.n_basis_functions() - 1
        i = torch.arange(p + 1, dtype=dtype, device=device)[:, None]
        j = torch.arange(q + 1, dtype=dtype, device=device)[None, :]
        logbinom = lambda n, k: math.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
        return logbinom(p, i) + logbinom(q, j) - logbinom(p + q, i + j) - math.log(p + q + 1)

    def _full_integrals(self):
        dtype, device = self.dtype_device()
        return torch.full((1, 1, self.n_basis_functions()), 1 / self.n_basis_functions(),
                          dtype=dtype, device=device)

    def Omega2_dim(self, other, lows=None, highs=None, return_log: bool = False):
        if lows is None and highs is None:
            # Analytic log-gram avoids an exp/log round-trip on the full box.
            self._check_polynomial_other(other)
            log_g = self._full_log_gram(other)[None, None].expand(self.batch_size(), self.dim(), -1, -1)
            return log_g if return_log else torch.exp(log_g)
        return super().Omega2_dim(other, lows, highs, return_log=return_log)
