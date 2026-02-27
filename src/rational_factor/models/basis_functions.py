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
    def __init__(self, params_init : torch.Tensor, trainable : bool = True):
        assert params_init.shape[2] == 2, "params_init must have shape (d, n_basis, 2)"
        super().__init__(params_init, trainable)

    @classmethod
    def random_init(cls, d : int, n_basis : int, offsets : torch.Tensor = torch.zeros(2)):
        offsets = offsets.repeat(d, n_basis, 1)
        return cls(torch.randn(d, n_basis, 2) + offsets)

    @classmethod
    def set_init(cls, d : int, n_basis : int, offsets : torch.Tensor = torch.zeros(2)):
        offsets = offsets.repeat(d, n_basis, 1)
        return cls(offsets)

    @classmethod 
    def freeze_params(cls, other : 'GaussianBasis'):
        return cls(other.params.detach().clone(), trainable=False)

    def means_stds(self):
        return self.params[..., 0], torch.nn.functional.softplus(self.params[..., 1]) + 1e-8
    
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

        # (d, nf), (d, ng)
        mu_f, std_f = self.means_stds()
        mu_g, std_g = other.means_stds()

        var_f = std_f * std_f
        var_g = std_g * std_g
        inv_var_f = 1.0 / var_f
        inv_var_g = 1.0 / var_g

        # Broadcast everything to (d, nf, nf, ng, ng)
        mu_i = mu_f[:, :, None, None, None]
        mu_j = mu_f[:, None, :, None, None]
        mu_k = mu_g[:, None, None, :, None]
        mu_l = mu_g[:, None, None, None, :]

        inv_i = inv_var_f[:, :, None, None, None]
        inv_j = inv_var_f[:, None, :, None, None]
        inv_k = inv_var_g[:, None, None, :, None]
        inv_l = inv_var_g[:, None, None, None, :]

        var_i = var_f[:, :, None, None, None]
        var_j = var_f[:, None, :, None, None]
        var_k = var_g[:, None, None, :, None]
        var_l = var_g[:, None, None, None, :]

        S = inv_i + inv_j + inv_k + inv_l
        T = mu_i * inv_i + mu_j * inv_j + mu_k * inv_k + mu_l * inv_l
        U = (mu_i * mu_i) * inv_i + (mu_j * mu_j) * inv_j + (mu_k * mu_k) * inv_k + (mu_l * mu_l) * inv_l

        log_pref = -0.5 * (
            4.0 * torch.log(torch.tensor(2.0 * torch.pi, device=mu_f.device, dtype=mu_f.dtype))
            + torch.log(var_i) + torch.log(var_j) + torch.log(var_k) + torch.log(var_l)
        )

        log_gauss_int = 0.5 * (
            torch.log(torch.tensor(2.0 * torch.pi, device=mu_f.device, dtype=mu_f.dtype))
            - torch.log(S)
        )

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
