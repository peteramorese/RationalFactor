import torch
from copy import deepcopy
import itertools
from .basis_functions import Basis, SeparableBasis, NonnegativeBasis

#### Base Classes ####

class DensityModel(torch.nn.Module):
    def __init__(self, dim : int):
        super().__init__()
        self.dim = dim
        self.min_log_density = -30

    def _clip_log_density(self, log_density : torch.Tensor):
        return torch.nan_to_num(log_density, nan=self.min_log_density, neginf=self.min_log_density)
    
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
        self.min_log_density = -30

    def _clip_log_density(self, log_density : torch.Tensor):
        return torch.nan_to_num(log_density, nan=self.min_log_density, neginf=self.min_log_density)

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
    

###### Special Distributions ######
class LogisticSigmoid(DensityModel):
    def __init__(
        self,
        dim: int,
        temperature: float = 0.1,
        loc: torch.Tensor = None,
        scale: torch.Tensor = None,
    ):
        super().__init__(dim)
        assert temperature > 0.0, "temperature must be positive"
        self.temperature = temperature

        if loc is None:
            loc = torch.full((dim,), 0.5)
        if scale is None:
            scale = torch.ones(dim)

        assert torch.all(scale > 0), "scale must be positive"

        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    def log_density(self, x: torch.Tensor):
        tau = torch.as_tensor(self.temperature, dtype=x.dtype, device=x.device)

        loc = self.loc.to(dtype=x.dtype, device=x.device)
        scale = self.scale.to(dtype=x.dtype, device=x.device)

        # map to unit-box coordinates
        x_norm = (x - loc) / scale + 0.5

        term1 = torch.nn.functional.softplus(-x_norm / tau)
        term2 = torch.nn.functional.softplus(-(1.0 - x_norm) / tau)

        # stable log(1 - exp(-1/tau))
        log_inv_Z_1d = torch.log(-torch.expm1(-1.0 / tau))

        log_p = (
            self.dim * log_inv_Z_1d
            - torch.log(scale).sum()
            - (term1 + term2).sum(dim=-1)
        )

        return log_p