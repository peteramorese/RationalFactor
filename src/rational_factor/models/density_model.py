import torch

class DensityModel(torch.nn.Module):
    def __init__(self, features: int, batch_shape: tuple[int, ...] | int = 1):
        super().__init__()
        self.features = features
        self.batch_shape = batch_shape
        self.min_log_density = -30
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
        else:
            batch_shape = tuple(batch_shape)
        if any(s < 1 for s in batch_shape):
            raise ValueError(f"batch_shape entries must be positive, got {batch_shape}")
        self.batch_shape = batch_shape
    
    def _expand_data(self, x : torch.Tensor) -> torch.Tensor:
        """``(n, d)`` → ``(n, *ones(batch_ndim), d)`` for broadcasting against batched params."""
        return x.reshape(x.shape[0], *([1] * len(self.batch_shape)), x.shape[-1])

    def _clip_log_density(self, log_density : torch.Tensor):
        return torch.nan_to_num(log_density, nan=self.min_log_density, neginf=self.min_log_density)
    
    def forward(self, x : torch.Tensor, **contexts : torch.Tensor):
        assert x.shape[1] == self.features, "x must have shape (n_data, features)"
        return torch.exp(self.log_density(x, **contexts))

    def log_density(self, x : torch.Tensor, **contexts : torch.Tensor):
        raise NotImplementedError("log_density not implemented")

    def valid(self):
        return True
    
    def marginal(self, marginal_dims : tuple[int, ...]):
        raise NotImplementedError("marginal not implemented")
    
    def sample(self, n_samples : int, **contexts : torch.Tensor):
        raise NotImplementedError("sample not implemented")

    def dtype_device(self):
        raise NotImplementedError("dtype_device not implemented")

    def supremum_bound(self):
        raise NotImplementedError("supremum_bound not implemented")

class ConditionalDensityModel(torch.nn.Module):
    def __init__(self, features : int, context_features : int, batch_shape: tuple[int, ...] = (1,)):
        super().__init__()
        self.features = features
        self.context_features = context_features
        self.min_log_density = -30
        self.batch_shape = batch_shape

    def _clip_log_density(self, log_density : torch.Tensor):
        return torch.nan_to_num(log_density, nan=self.min_log_density, neginf=self.min_log_density)

    def forward(self, x : torch.Tensor, *, conditioner : torch.Tensor, **contexts : torch.Tensor):
        """
        Returns density of p(x | conditioner, contexts).
        """
        assert x.shape[1] == self.features, "x must have shape (n_data, features)"
        assert conditioner.shape[1] == self.context_features, "conditioner must have shape (n_data, context_features)"
        assert x.shape[0] == conditioner.shape[0], "x and conditioner must have the same number of data points"
        
        return torch.exp(self.log_density(x, conditioner=conditioner, **contexts))

    def log_density(self, x : torch.Tensor, *, conditioner : torch.Tensor, **contexts : torch.Tensor):
        raise NotImplementedError("log_density not implemented")

    def valid(self):
        return True
    
    def sample(self, conditioner : torch.Tensor, **contexts : torch.Tensor):
        """
        Returns (n_samples, features) tensor of samples.
        """
        raise NotImplementedError("sample not implemented")
    
    def dtype_device(self):
        raise NotImplementedError("dtype_device not implemented")
    
    def supremum_bound(self, conditioner : torch.Tensor | None):
        f"""
        Returns the supremum bound of the density. If conditioner is provided, it returns b \geq sup_x p(x | conditioner),
        otherwise if conditoner is None, it returns the supremum across all possible conditioners.
        """
        raise NotImplementedError("supremum_bound not implemented")