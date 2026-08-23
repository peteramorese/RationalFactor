import torch
import itertools

class Parameters:
    def __init__(self, fixed_values: torch.Tensor = None):
        self._p = fixed_values

    def size(self):
        return self._p.size()

    def is_trainable(self):
        return False

    def dtype_device(self):
        return self._p.dtype, self._p.device

    def __call__(self):
        return self._p
    
    def is_module(self):
        return False


class TrainableParameters(Parameters, torch.nn.Module):
    def __init__(self, trainable_init_values: torch.Tensor = None, fixed_values: torch.Tensor = None):
        torch.nn.Module.__init__(self)
        assert not (trainable_init_values is not None and fixed_values is not None)
        self._trainable = trainable_init_values is not None
        if self._trainable:
            self._p = torch.nn.Parameter(trainable_init_values)
        else:
            self.register_buffer("_p", fixed_values)

    @classmethod
    def random_init(
        cls,
        shape: tuple[int, ...],
        trainable: bool = True,
        mean: float = 0.0,
        std: float = 1.0,
    ):
        values = torch.randn(*shape) * std + mean
        if trainable:
            return cls(trainable_init_values=values)
        return cls(fixed_values=values)

    @classmethod
    def set_init(cls, shape: tuple[int, ...], value: float, trainable: bool = True):
        values = torch.ones(shape) * value
        if trainable:
            return cls(trainable_init_values=values)
        return cls(fixed_values=values)

    @classmethod
    def from_values(cls, values: torch.Tensor, trainable: bool = True):
        if trainable:
            return cls(trainable_init_values=values)
        return cls(fixed_values=values.detach().clone())

    def is_trainable(self):
        return self._trainable

    def forward(self) -> torch.Tensor:
        """Unconstrained values. Subclasses override to apply constraints."""
        return self._p

    def __call__(self) -> torch.Tensor:
        return torch.nn.Module.__call__(self)

    def set_requires_grad(self, requires_grad: bool):
        if self._p.is_leaf:
            self._p.requires_grad_(requires_grad)
        
    def is_module(self):
        return True


class PositiveParameters(TrainableParameters):
    def __init__(
        self,
        trainable_init_values: torch.Tensor = None,
        fixed_values: torch.Tensor = None,
        normalization_dim: int = None,
        epsilon: float = 0.0,
    ):
        super().__init__(trainable_init_values=trainable_init_values, fixed_values=fixed_values)
        self._normalization_dim = normalization_dim
        self._epsilon = epsilon
        if not self._trainable:
            assert torch.all(fixed_values >= 0), "fixed_values must be nonnegative"
            if normalization_dim is not None:
                with torch.no_grad():
                    self._p.copy_(self._normalize(self._p))
    
    @classmethod
    def random_init(
        cls,
        shape: tuple[int, ...],
        trainable: bool = True,
        mean: float = 0.0,
        std: float = 1.0,
        normalization_dim: int = None,
        epsilon: float = 0.0,
    ):
        values = torch.randn(*shape) * std + mean
        if trainable:
            return cls(trainable_init_values=values, normalization_dim=normalization_dim, epsilon=epsilon)
        return cls(fixed_values=values, normalization_dim=normalization_dim, epsilon=epsilon)

    @classmethod
    def set_init(cls, shape: tuple[int, ...], value: float, trainable: bool = True, normalization_dim: int = None, epsilon: float = 0.0):
        values = torch.ones(shape) * value
        if trainable:
            return cls(trainable_init_values=values, normalization_dim=normalization_dim, epsilon=epsilon)
        return cls(fixed_values=values, normalization_dim=normalization_dim, epsilon=epsilon)

    @classmethod
    def from_values(cls, values: torch.Tensor, trainable: bool = True, normalization_dim: int = None, epsilon: float = 0.0):
        if trainable:
            return cls(trainable_init_values=values, normalization_dim=normalization_dim, epsilon=epsilon)
        return cls(fixed_values=values.detach().clone(), normalization_dim=normalization_dim, epsilon=epsilon)

    def _normalize(self, p: torch.Tensor):
        return self._epsilon + (1.0 - p.shape[self._normalization_dim] * self._epsilon) * torch.nn.functional.softmax(p, dim=self._normalization_dim)

    def forward(self) -> torch.Tensor:
        if self._trainable:
            if self._normalization_dim is not None:
                return self._normalize(self._p)
            return self._epsilon + torch.nn.functional.softplus(self._p)
        return self._p

    def freeze_params(self):
        return PositiveParameters(
            fixed_values=self.forward().detach().clone(),
            normalization_dim=self._normalization_dim,
            epsilon=self._epsilon,
        )

    def with_fixed_values(self, values: torch.Tensor) -> "PositiveParameters":
        return PositiveParameters(
            fixed_values=values.detach().clone(),
            normalization_dim=self._normalization_dim,
            epsilon=self._epsilon,
        )

    def get_normalization_dim(self):
        return self._normalization_dim

def param_group_iter(params: tuple[Parameters, ...]):
    return itertools.chain(*[param.parameters() for param in params])

