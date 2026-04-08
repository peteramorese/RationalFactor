import torch
from abc import abstractmethod

from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform



class Basis(torch.nn.Module):
    def __init__(self, dim : int, n_basis : int, params_init : torch.Tensor = None, trainable : bool = True):
        '''
        Args:
            dim : int, number of dimensions
            n_basis : int, number of basis functions
            params_init : torch.Tensor of initial parameters
            trainable : bool indicating if parameters are trainable
        '''
        super().__init__()
        self._dim = dim
        self._n_basis = n_basis
        self._trainable = trainable

        if params_init is not None:
            self._init_params(params_init, trainable)
        
    def _init_params(self, params_init : torch.Tensor, trainable : bool):
        if trainable:
            self.params = torch.nn.Parameter(params_init)
        else:
            self.register_buffer("params", params_init)

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
    
    def Omega1(self):
        '''
        Computes the integral of the basis functions.
        omega[i] = <this_i, 1>

        Returns:
            Tensor of shape (n_basis,)
        '''
        raise NotImplementedError("Omega1 is not implemented for this basis function")

    def Omega2(self, other: 'Basis'):
        '''
        Computes the function inner product matrix with another basis function vector. 
        omega[i, j] = <this_i, other_j>
        
        Returns:
            Tensor of shape (n_basis, other.n_basis)
        '''
        raise NotImplementedError("Omega2 is not implemented for this basis function")

    def Omega3(self, other1: 'Basis', other2: 'Basis'):
        '''
        Computes the 3D quadratic function inner product tensor with another basis function vector. 
        omega[i, j, k] = <this_i, other1_j, other2_k>
        
        Returns:
            Tensor of shape (n_basis, n_basis, other1.n_basis, other2.n_basis)
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
    
    def product_basis(self, other_basis_factors : list['Basis']):
        '''
        Returns the broadcasted (flattened) product of the basis functions in the other_basis_factors list.
        '''
        raise NotImplementedError("product_basis is not implemented for this basis function")
    
    @abstractmethod
    def marginal(self, marginal_dims : tuple[int, ...]):
        '''
        Computes the marginalized basis functions over the given dimensions

        Returns:
            A new SeparableBasis object with the marginalized dimensions
        '''
        pass

class SeparableBasis(Basis):
    def __init__(self, params_init : torch.Tensor, trainable : bool = True):
        assert params_init.dim() == 3, "params_init must have shape (d, n_basis, n_params_per_basis)"
        super().__init__(params_init.shape[0], params_init.shape[1], params_init, trainable)

    def n_params_per_basis(self):
        return self.params.shape[2]
    
# Nonnegative basis functions 
class NonnegativeBasis:
    pass


class GaussianBasis(SeparableBasis, NonnegativeBasis):
    def __init__(self, params_init : torch.Tensor, trainable : bool = True, min_std : float = 1e-5):
        assert params_init.shape[2] == 2, "params_init must have shape (d, n_basis, 2)"
        super().__init__(params_init, trainable)
        self.min_std = min_std

    @classmethod
    def random_init(cls, d : int, n_basis : int, offsets : torch.Tensor = torch.zeros(2), min_std : float = 1e-5, variance: float = 1.0, device = None):
        if device is None:
            device = offsets.device
        offsets = offsets.repeat(d, n_basis, 1)
        return cls(torch.randn(d, n_basis, 2, device=device) * torch.sqrt(torch.tensor(variance, device=device)) + offsets, min_std=min_std)

    @classmethod
    def set_init(cls, d : int, n_basis : int, offsets : torch.Tensor = torch.zeros(2), min_std : float = 1e-5):
        offsets = offsets.repeat(d, n_basis, 1)
        return cls(offsets, min_std=min_std)

    def freeze_params(self):
        return GaussianBasis(self.params.detach().clone(), trainable=False, min_std=self.min_std)

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
        return torch.exp(log_dim_factors.sum(dim=1))  # (n_data, n_basis)

    def normalized(self):
        return True

    def Omega1(self):
        return torch.ones(
            self.n_basis_functions(),
            dtype=self.params.dtype,
            device=self.params.device,
        )

    def Omega2(self, other: 'GaussianBasis'):
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
        return torch.exp(log_Omega)

    def Omega3(self, other1: "GaussianBasis", other2: "GaussianBasis"):
        assert isinstance(other1, GaussianBasis), "other1 must be GaussianBasis"
        assert isinstance(other2, GaussianBasis), "other2 must be GaussianBasis"
        assert self.dim() == other1.dim() == other2.dim(), "Basis functions must have the same dimension"

        # (d, n0), (d, n1), (d, n2)
        mu0, std0 = self.means_stds()
        mu1, std1 = other1.means_stds()
        mu2, std2 = other2.means_stds()

        var0 = std0 * std0
        var1 = std1 * std1
        var2 = std2 * std2

        inv0 = 1.0 / var0
        inv1 = 1.0 / var1
        inv2 = 1.0 / var2

        # Broadcast to (d, n0, n1, n2)
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

        two_pi = mu0.new_tensor(2.0 * torch.pi)
        log2pi = torch.log(two_pi)

        log_pref = -0.5 * (
            3.0 * log2pi
            + torch.log(var_i)
            + torch.log(var_j)
            + torch.log(var_k)
        )
        log_gauss_int = 0.5 * (log2pi - torch.log(S))
        quad = -0.5 * (U - (T * T) / S)

        log_dim = log_pref + log_gauss_int + quad   # (d, n0, n1, n2)
        log_Omega = log_dim.sum(dim=0)              # (n0, n1, n2)

        return torch.exp(log_Omega)

    def Omega22(self, other: "GaussianBasis"):
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

        return torch.exp(log_Omega)

    def product_basis(self, other_basis_factors : list['Basis']) -> 'QuadraticExpBasis':
        pass

    def marginal(self, marginal_dims: tuple[int, ...]) -> 'GaussianBasis':
        marginal_dims = tuple(marginal_dims)
        assert all(0 <= i < self.dim() for i in marginal_dims), "marginal_dims must be in [0, d)"
        return GaussianBasis(
            self.params[marginal_dims, :, :].detach().clone(),
            trainable=self.params.requires_grad,
            min_std=self.min_std,
        )

class QuadraticExpBasis(SeparableBasis, NonnegativeBasis):
    """
    Separable basis with 1D factors of the form exp(a x^2 + b x + c)
    """

    def __init__(self, params_init: torch.Tensor, trainable: bool = True, eps: float = 1e-6):
        assert params_init.shape[2] == 3, "params_init must have shape (d, n_basis, 3)"
        super().__init__(params_init, trainable)
        self.eps = eps

    @classmethod
    def random_init(cls, d: int, n_basis: int, offsets: torch.Tensor = torch.zeros(3), variance: float = 1.0, eps: float = 1e-6, device=None):
        if device is None:
            device = offsets.device
        offsets = offsets.repeat(d, n_basis, 1)
        params_init = (
            torch.randn(d, n_basis, 3, device=device)
            * torch.sqrt(torch.tensor(variance, device=device))
            + offsets
        )
        return cls(params_init, eps=eps)

    @classmethod
    def set_init(cls, d: int, n_basis: int, offsets: torch.Tensor = torch.zeros(3), eps: float = 1e-6):
        offsets = offsets.repeat(d, n_basis, 1)
        return cls(offsets, eps=eps)

    def freeze_params(self):
        return QuadraticExpBasis(self.params.detach().clone(), trainable=False, eps=self.eps)

    def abc(self):
        raw_a = self.params[..., 0]
        b = self.params[..., 1]
        c = self.params[..., 2]
        a = -torch.nn.functional.softplus(raw_a) - self.eps
        return a, b, c

    def normalized(self):
        return False

    def forward(self, y: torch.Tensor):
        assert y.shape[1] == self.dim(), "y must have shape (n_data, d)"
        y = y[:, :, None]  # (n_data, d, n_basis)

        a, b, c = self.abc()  # (d, n_basis)

        log_dim_factors = a * y.square() + b * y + c
        return torch.exp(log_dim_factors.sum(dim=1))  # (n_data, n_basis)

    def Omega1(self):
        a, b, c = self.abc()

        log_dim_int = c - b.square() / (4.0 * a) + 0.5 * (
            torch.log(torch.tensor(torch.pi, dtype=a.dtype, device=a.device)) - torch.log(-a)
        )
        return torch.exp(log_dim_int.sum(dim=0))  # (n_basis,)
    
    def Omega2(self, other: "QuadraticExpBasis"):
        assert isinstance(other, QuadraticExpBasis), "other must be QuadraticExpBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"

        a1, b1, c1 = self.abc()
        a2, b2, c2 = other.abc()

        # broadcast to (d, n1, n2)
        a = a1[:, :, None] + a2[:, None, :]
        b = b1[:, :, None] + b2[:, None, :]
        c = c1[:, :, None] + c2[:, None, :]

        log_dim_int = c - b.square() / (4.0 * a) + 0.5 * (
            torch.log(torch.tensor(torch.pi, dtype=a.dtype, device=a.device)) - torch.log(-a)
        )
        return torch.exp(log_dim_int.sum(dim=0))  # (n1, n2)
    
    def Omega3(self, other1: "QuadraticExpBasis", other2: "QuadraticExpBasis"):
        assert isinstance(other1, QuadraticExpBasis), "other1 must be QuadraticExpBasis"
        assert isinstance(other2, QuadraticExpBasis), "other2 must be QuadraticExpBasis"
        assert self.dim() == other1.dim() == other2.dim(), "Basis functions must have the same dimension"

        a0, b0, c0 = self.abc()     # (d, n0)
        a1, b1, c1 = other1.abc()   # (d, n1)
        a2, b2, c2 = other2.abc()   # (d, n2)

        # Broadcast to (d, n0, n1, n2)
        A = a0[:, :, None, None] + a1[:, None, :, None] + a2[:, None, None, :]
        B = b0[:, :, None, None] + b1[:, None, :, None] + b2[:, None, None, :]
        C = c0[:, :, None, None] + c1[:, None, :, None] + c2[:, None, None, :]

        log_pi = torch.log(torch.tensor(torch.pi, dtype=A.dtype, device=A.device))
        log_dim_int = C - (B * B) / (4.0 * A) + 0.5 * (log_pi - torch.log(-A))

        log_Omega = log_dim_int.sum(dim=0)   # (n0, n1, n2)
        return torch.exp(log_Omega)
    
    def Omega22(self, other: "QuadraticExpBasis"):
        """
        omega[i, j, k, l] = ∫ self_i(x) self_j(x) other_k(x) other_l(x) dx

        Returns:
            Tensor of shape (n_self, n_self, n_other, n_other)
        """
        assert isinstance(other, QuadraticExpBasis), "other must be QuadraticExpBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"

        a1, b1, c1 = self.abc()    # (d, n_self)
        a2, b2, c2 = other.abc()   # (d, n_other)

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
        C = (
            c1[:, :, None, None, None]
            + c1[:, None, :, None, None]
            + c2[:, None, None, :, None]
            + c2[:, None, None, None, :]
        )

        log_pi = torch.log(torch.tensor(torch.pi, dtype=A.dtype, device=A.device))
        log_dim_int = C - (B * B) / (4.0 * A) + 0.5 * (log_pi - torch.log(-A))

        log_Omega = log_dim_int.sum(dim=0)   # (n_self, n_self, n_other, n_other)
        return torch.exp(log_Omega)
    
    def marginal(self, marginal_dims: tuple[int, ...]) -> "QuadraticExpBasis":
        marginal_dims = tuple(marginal_dims)
        assert all(0 <= i < self.dim() for i in marginal_dims), "marginal_dims must be in [0, d)"

        keep_dims = tuple(i for i in range(self.dim()) if i not in marginal_dims)

        a, b, c = self.abc()  # (d, n_basis)

        # --- compute log integral contribution from marginalized dims ---
        if len(marginal_dims) > 0:
            a_m = a[marginal_dims, :]
            b_m = b[marginal_dims, :]
            c_m = c[marginal_dims, :]

            log_pi = torch.log(torch.tensor(torch.pi, dtype=a.dtype, device=a.device))

            log_int_m = (
                c_m
                - (b_m * b_m) / (4.0 * a_m)
                + 0.5 * (log_pi - torch.log(-a_m))
            )  # (|M|, n_basis)

            log_int_sum = log_int_m.sum(dim=0)  # (n_basis,)
        else:
            log_int_sum = torch.zeros(
                self.n_basis_functions(),
                dtype=a.dtype,
                device=a.device,
            )

        # --- keep remaining dims ---
        a_k = a[keep_dims, :]
        b_k = b[keep_dims, :]
        c_k = c[keep_dims, :]

        # --- add marginalized contribution into c ---
        c_new = c_k + log_int_sum[None, :]

        # --- convert back to raw parameterization ---
        # a = -softplus(raw_a) - eps  => invert
        s = (-a_k - self.eps).clamp_min(1e-12)
        raw_a = torch.log(torch.expm1(s))

        params = torch.stack([raw_a, b_k, c_new], dim=-1)

        return QuadraticExpBasis(
            params.detach().clone(),
            trainable=self.params.requires_grad,
            eps=self.eps,
    )
    

class BetaBasis(SeparableBasis, NonnegativeBasis):
    def __init__(self, params_init: torch.Tensor, trainable: bool = True, min_concentration: float = 1.0, eps: float = 1e-6):
        assert min_concentration > 0.0, "min_concentration must be positive"
        assert params_init.shape[2] == 2, "params_init must have shape (d, n_basis, 2)"
        super().__init__(params_init, trainable)
        self.min_concentration = min_concentration
        self.eps = eps

    @classmethod
    def random_init(cls, d: int, n_basis: int, offsets: torch.Tensor = torch.zeros(2), variance: float = 1.0, min_concentration: float = 1.0, eps: float = 1e-6):
        offsets = offsets.repeat(d, n_basis, 1)
        return cls(
            torch.randn(d, n_basis, 2) * torch.sqrt(torch.tensor(variance)) + offsets,
            min_concentration=min_concentration,
            eps=eps,
        )

    @classmethod
    def set_init(cls, d: int, n_basis: int, offsets: torch.Tensor = torch.zeros(2), min_concentration: float = 1.0, eps: float = 1e-6):
        offsets = offsets.repeat(d, n_basis, 1)
        return cls(
            offsets,
            min_concentration=min_concentration,
            eps=eps,
        )

    def freeze_params(self):
        return BetaBasis(
            self.params.detach().clone(),
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

        if torch.isnan(log_dim_factors).any():
            print("log_dim_factors: ", log_dim_factors)
            print("alpha: ", alpha)
            print("beta: ", beta)
            print("y: ", y)
            #print("y_input: ", y_input)
            raise ValueError("log_dim_factors is nan")

        return torch.exp(log_dim_factors.sum(dim=1))  # (n_data, n_basis)

    def normalized(self):
        return True

    def Omega1(self):
        return torch.ones(
            self.n_basis_functions(),
            dtype=self.params.dtype,
            device=self.params.device,
        )

    def Omega2(self, other: "BetaBasis"):
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
        return torch.exp(log_Omega)

    def Omega3(self, other1: "BetaBasis", other2: "BetaBasis"):
        assert isinstance(other1, BetaBasis), "other1 must be BetaBasis"
        assert isinstance(other2, BetaBasis), "other2 must be BetaBasis"
        assert self.dim() == other1.dim() == other2.dim(), "Basis functions must have the same dimension"

        a0, b0 = self.alphas_betas()    # (d, n0)
        a1, b1 = other1.alphas_betas()  # (d, n1)
        a2, b2 = other2.alphas_betas()  # (d, n2)

        # Broadcast to (d, n0, n1, n2)
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

        log_Omega = log_dim.sum(dim=0)   # (n0, n1, n2)
        return torch.exp(log_Omega)

    def Omega22(self, other: "BetaBasis"):
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
        return torch.exp(log_Omega)

    def marginal(self, marginal_dims: tuple[int, ...]) -> "BetaBasis":
        marginal_dims = tuple(marginal_dims)
        assert all(0 <= i < self.dim() for i in marginal_dims), "marginal_dims must be in [0, d)"

        return BetaBasis(
            self.params[marginal_dims, :, :].detach().clone(),
            trainable=self.params.requires_grad,
            min_concentration=self.min_concentration,
            eps=self.eps,
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
