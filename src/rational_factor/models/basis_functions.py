import torch
from abc import abstractmethod

class SeparableBasis(torch.nn.Module):
    def __init__(self, params_init : torch.Tensor, trainable : bool = True):
        '''
        Args:
            params_init : torch.Tensor of shape (d, n_basis, n_params_per_basis)
            trainable : bool indicating if parameters are trainable
        '''
        super().__init__()
        #self.params = torch.nn.Parameter(torch.randn(d, n_basis, n_params_per_basis))
        if trainable:
            self.params = torch.nn.Parameter(params_init)
        else:
            self.register_buffer("params", params_init)

    def dim(self):
        return self.params.shape[0]
    
    def n_basis_functions(self):
        return self.params.shape[1]
    
    @abstractmethod
    def inner_prod_matrix(self, other: 'SeparableBasis'):
        '''
        Computes the function inner product matrix with another basis function vector. The argument `other` occupies the column index
        
        Returns:
            n_basis x other.n_basis matrix of inner product values
        '''
        pass

    def inner_prod_tensor(self, other: 'SeparableBasis'):
        '''
        Computes the 4D quadratic function inner product tensor with another basis function vector. The argument `other` occupies the last two indices
        
        Returns:
            n_basis x b_basis x other.n_basis x other.n_basis tensor of inner product values
        '''
        pass
    
    @abstractmethod
    def marginal(self, marginal_dims : tuple[int, ...]):
        '''
        Computes the marginalized basis functions over the given dimensions

        Returns:
            A new SeparableBasis object with the marginalized dimensions
        '''
        pass


class GaussianBasis(SeparableBasis):
    def __init__(self, params_init : torch.Tensor, trainable : bool = True, min_std : float = 1e-5):
        assert params_init.shape[2] == 2, "params_init must have shape (d, n_basis, 2)"
        super().__init__(params_init, trainable)
        self.min_std = min_std

    @classmethod
    def random_init(cls, d : int, n_basis : int, offsets : torch.Tensor = torch.zeros(2), min_std : float = 1e-5):
        offsets = offsets.repeat(d, n_basis, 1)
        return cls(torch.randn(d, n_basis, 2) + offsets, min_std=min_std)

    @classmethod
    def set_init(cls, d : int, n_basis : int, offsets : torch.Tensor = torch.zeros(2), min_std : float = 1e-5):
        offsets = offsets.repeat(d, n_basis, 1)
        return cls(offsets, min_std=min_std)

    @classmethod 
    def freeze_params(cls, other : 'GaussianBasis'):
        return cls(other.params.detach().clone(), trainable=False, min_std=other.min_std)

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

    def inner_prod_matrix(self, other: 'GaussianBasis'):
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

    def inner_prod_tensor(self, other: "GaussianBasis"):
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

    def marginal(self, marginal_dims: tuple[int, ...]) -> 'GaussianBasis':
        marginal_dims = tuple(marginal_dims)
        assert all(0 <= i < self.dim() for i in marginal_dims), "marginal_dims must be in [0, d)"
        d_new = len(marginal_dims)
        n_basis = self.n_basis_functions()
        out = GaussianBasis(d_new, n_basis)
        out.params.data.copy_(self.params[marginal_dims, :, :].detach())
        return out


class BetaBasis(SeparableBasis):
    def __init__(self, params_init: torch.Tensor, trainable: bool = True, min_concentration: float = 1e-5, eps: float = 1e-6):
        assert params_init.shape[2] == 2, "params_init must have shape (d, n_basis, 2)"
        super().__init__(params_init, trainable)
        self.min_concentration = min_concentration
        self.eps = eps

    @classmethod
    def random_init(cls, d: int, n_basis: int, offsets: torch.Tensor = torch.zeros(2), min_concentration: float = 1e-5, eps: float = 1e-6):
        offsets = offsets.repeat(d, n_basis, 1)
        return cls(
            torch.randn(d, n_basis, 2) + offsets,
            min_concentration=min_concentration,
            eps=eps,
        )

    @classmethod
    def set_init(cls, d: int, n_basis: int, offsets: torch.Tensor = torch.zeros(2), min_concentration: float = 1e-5, eps: float = 1e-6):
        offsets = offsets.repeat(d, n_basis, 1)
        return cls(
            offsets,
            min_concentration=min_concentration,
            eps=eps,
        )

    @classmethod
    def freeze_params(cls, other: "BetaBasis"):
        return cls(
            other.params.detach().clone(),
            trainable=False,
            min_concentration=other.min_concentration,
            eps=other.eps,
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
            raise ValueError("log_dim_factors is nan")

        return torch.exp(log_dim_factors.sum(dim=1))  # (n_data, n_basis)

    def inner_prod_matrix(self, other: "BetaBasis"):
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

    def inner_prod_tensor(self, other: "BetaBasis"):
        """
        Omega[i,j,k,l] = <phi_i phi_j, psi_k psi_l>
                       = \int phi_i(x)phi_j(x)psi_k(x)psi_l(x) dx
        """
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