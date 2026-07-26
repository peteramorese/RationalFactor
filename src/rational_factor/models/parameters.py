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

    def __call__(self) -> torch.Tensor:
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
        normalized: bool = False,
        epsilon: float = 0.0,
    ):
        super().__init__(trainable_init_values=trainable_init_values, fixed_values=fixed_values)
        self._normalized = normalized
        self._epsilon = epsilon
        if not self._trainable:
            assert torch.all(fixed_values >= 0), "fixed_values must be nonnegative"
            if normalized:
                with torch.no_grad():
                    self._p.copy_(self._normalize(self._p, dim=-1))
    
    @classmethod
    def random_init(
        cls,
        shape: tuple[int, ...],
        trainable: bool = True,
        mean: float = 0.0,
        std: float = 1.0,
        normalized: bool = False,
        epsilon: float = 0.0,
    ):
        values = torch.randn(*shape) * std + mean
        if trainable:
            return cls(trainable_init_values=values, normalized=normalized, epsilon=epsilon)
        return cls(fixed_values=values, normalized=normalized, epsilon=epsilon)

    @classmethod
    def set_init(cls, shape: tuple[int, ...], value: float, trainable: bool = True, normalized: bool = False, epsilon: float = 0.0):
        values = torch.ones(shape) * value
        if trainable:
            return cls(trainable_init_values=values, normalized=normalized, epsilon=epsilon)
        return cls(fixed_values=values, normalized=normalized, epsilon=epsilon)

    @classmethod
    def from_values(cls, values: torch.Tensor, trainable: bool = True, normalized: bool = False, epsilon: float = 0.0):
        if trainable:
            return cls(trainable_init_values=values, normalized=normalized, epsilon=epsilon)
        return cls(fixed_values=values.detach().clone(), normalized=normalized, epsilon=epsilon)

    def _normalize(self, p: torch.Tensor, dim: int = -1):
        return self._epsilon + (1.0 - p.shape[dim] * self._epsilon) * torch.nn.functional.softmax(p, dim=dim)

    def forward(self) -> torch.Tensor:
        if self._trainable:
            if self._normalized:
                return self._normalize(self._p, dim=-1)
            return self._epsilon + torch.nn.functional.softplus(self._p)
        return self._p

    def freeze_params(self):
        return PositiveParameters(
            fixed_values=self.forward().detach().clone(),
            normalized=self._normalized,
            epsilon=self._epsilon,
        )

    def with_fixed_values(self, values: torch.Tensor) -> "PositiveParameters":
        return PositiveParameters(
            fixed_values=values.detach().clone(),
            normalized=self._normalized,
            epsilon=self._epsilon,
        )

    def is_normalized(self):
        return self._normalized

def param_group_iter(params: tuple[Parameters, ...]):
    return itertools.chain(*[param.parameters() for param in params])