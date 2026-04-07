import torch
from copy import deepcopy
import itertools
from .basis_functions import Basis, SeparableBasis, NonnegativeBasis

#### Base Classes ####

class DensityModel(torch.nn.Module):
    def __init__(self, dim : int):
        super().__init__()
        self.dim = dim

    def forward(self, x : torch.Tensor):    
        assert x.shape[1] == self.dim, "x must have shape (n_data, dim)"
        return torch.exp(self.log_density(x))

    def log_density(self, x : torch.Tensor):
        raise NotImplementedError("log_density not implemented")

    def valid(self):
        return True
    
    def marginal(self, marginal_dims : tuple[int, ...]):
        raise NotImplementedError("marginal not implemented")
    
    def sample(self, n_samples : int):
        raise NotImplementedError("sample not implemented")

class ConditionalDensityModel(torch.nn.Module):
    def __init__(self, dim : int, conditioner_dim : int):
        super().__init__()
        self.dim = dim
        self.conditioner_dim = conditioner_dim

    def forward(self, x : torch.Tensor, *, conditioner : torch.Tensor):
        """
        Returns density of p(x | conditioner)
        """
        assert x.shape[1] == self.dim, "x must have shape (n_data, dim)"
        assert conditioner.shape[1] == self.conditioner_dim, "conditioner must have shape (n_data, conditioner_dim)"
        assert x.shape[0] == conditioner.shape[0], "x and conditioner must have the same number of data points"
        
        return torch.exp(self.log_density(x, conditioner=conditioner))

    def log_density(self, x : torch.Tensor, *, conditioner : torch.Tensor):
        raise NotImplementedError("log_density not implemented")

    def valid(self):
        return True
    
    def sample(self, conditioner : torch.Tensor):
        """
        Returns (n_samples, dim) tensor of samples
        """
        raise NotImplementedError("sample not implemented")
    