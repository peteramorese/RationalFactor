import torch

class SeparableBasis(torch.nn.Module):
    def __init__(self, d: int, n_basis: int, n_params_per_basis: int):
        super().__init__()
        self.params = torch.nn.Parameter(torch.randn(d, n_basis, n_params_per_basis))
        self.d = d
    
    def n_basis_functions(self):
        return self.params.shape[1]
    
    def inner_prod_matrix(self, other: 'SeparableBasis'):
        raise NotImplementedError("Subclass must implement inner_prod_matrix")

class GaussianBasis(SeparableBasis):
    def __init__(self, d: int, n_basis: int):
        super().__init__(d, n_basis, 2)
    
    def __means_stds(self):
        return self.params[..., 0], torch.nn.functional.softplus(self.params[..., 1]) + 1e-8
    
    def forward(self, y: torch.Tensor):
        assert y.shape[1] == self.d, "y must have shape (n_data, d)"
        y = y[:, :, None]  # (n_data, d, n_basis)
        mu, std = self.__means_stds()

        log_dim_factors = (
            -0.5 * torch.log(2 * torch.pi)
            - torch.log(std)
            - (y - mu) ** 2 / (2 * std ** 2)
        )
        return log_dim_factors.sum(dim=1)  # (n_data, n_basis)

    def inner_prod_matrix(self, other: 'GaussianBasis') -> torch.Tensor:
        """
        Returns inner product matrix with another basis function vector
        """
        assert isinstance(other, GaussianBasis), "other must be GaussianBasis"
        assert self.d == other.d, "Basis functions must have the same dimension"

        # (d, n1), (d, n2)
        mu1, std1 = self.__means_stds()
        mu2, std2 = other.__means_stds()

        # Broadcast to (d, n1, n2)
        diff = mu1[:, :, None] - mu2[:, None, :]
        var_sum = (std1[:, :, None] ** 2) + (std2[:, None, :] ** 2)

        # log of 1D Gaussian pdf evaluated at diff with variance var_sum
        log_dim_ip = -0.5 * (torch.log(2 * torch.pi * var_sum) + (diff * diff) / var_sum)

        log_Omega = log_dim_ip.sum(dim=0)
        return torch.exp(log_Omega)
