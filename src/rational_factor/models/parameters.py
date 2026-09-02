import torch
import itertools
from abc import ABC, abstractmethod

from rational_factor.models.structured_matrices import (
    DenseMatrix,
    Order1Quasiseparable,
    Rank1PlusDiagonal,
)


class Parameters(ABC):
    @abstractmethod
    def __call__(self): ...

    @abstractmethod
    def is_trainable(self): ...


class FixedParameters(Parameters):
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

    def size(self):
        return self._p.size()

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


def param_group_iter(params: tuple[TrainableParameters, ...]):
    return itertools.chain(*[param.parameters() for param in params])


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


class Rank1PlusDiagonalFactorization(Parameters):
    """Stores R1PD factor tensors ``d, u, v`` with shape ``(T, n)``.

    Call ``()`` to get a ``Rank1PlusDiagonal``; apply products with
    ``sequential_matvec`` / ``inverse`` / ``T`` / ``flip`` on that object.
    """

    def __init__(
        self,
        u: Parameters,
        v: Parameters,
        d: Parameters | None = None,
        normalization_dim: int | None = None,
    ):
        assert u.size() == v.size(), "u and v must have the same shape"
        assert u.size().__len__() == 2, "u and v must have shape (n_factors, n)"
        if d is not None:
            assert d.size() == u.size(), "d, u, and v must have the same shape"
        self.d = d
        self.u = u
        self.v = v
        self.normalization_dim = normalization_dim

    def __call__(self) -> Rank1PlusDiagonal:
        d = None if self.d is None else self.d()
        return Rank1PlusDiagonal(self.u(), self.v(), d, normalization_dim=self.normalization_dim)
    
    def is_trainable(self):
        trainable = self.u.is_trainable() or self.v.is_trainable()
        if self.d is not None:
            trainable = trainable or self.d.is_trainable()
        return trainable

    def is_module(self):
        return False


class DenseMatrixFactorization(Parameters):
    """Wraps a ``(..., n, m)`` tensor parameter; ``()`` returns a ``DenseMatrix``."""

    def __init__(self, values: Parameters):
        self._values = values

    def __call__(self) -> DenseMatrix:
        return DenseMatrix(self._values())

    def is_trainable(self):
        return self._values.is_trainable()

    def is_module(self):
        return False


class Order1QuasiseparableFactorization(Parameters):
    """Trainable ``P = L D U``. Diagonal is softplus so ``P`` stays nonsingular.

    ``transition_bound`` (e.g. ``0.99``) maps ``la, ub`` through ``bound * tanh``
    so long products of the transition generators cannot explode.
    """

    def __init__(
        self,
        lower_p: Parameters,
        lower_a: Parameters,
        lower_q: Parameters,
        diag: Parameters,
        upper_g: Parameters,
        upper_b: Parameters,
        upper_h: Parameters,
        min_diag: float = 1e-4,
        transition_bound: float | None = None,
    ):
        self.lower_p, self.lower_a, self.lower_q = lower_p, lower_a, lower_q
        self.diag = diag
        self.upper_g, self.upper_b, self.upper_h = upper_g, upper_b, upper_h
        self.min_diag = min_diag
        self.transition_bound = transition_bound

        shape = diag().shape
        if any(p().shape != shape for p in self.parameters):
            raise ValueError("all factor Parameters must share the same shape")

    @property
    def parameters(self) -> tuple[Parameters, ...]:
        return (
            self.lower_p, self.lower_a, self.lower_q, self.diag,
            self.upper_g, self.upper_b, self.upper_h,
        )

    def __call__(self) -> Order1Quasiseparable:
        la, ub = self.lower_a(), self.upper_b()
        if self.transition_bound is not None:
            la = self.transition_bound * torch.tanh(la)
            ub = self.transition_bound * torch.tanh(ub)
        d = torch.nn.functional.softplus(self.diag()) + self.min_diag
        return Order1Quasiseparable(
            self.lower_p(), la, self.lower_q(), d,
            self.upper_g(), ub, self.upper_h(),
        )
    
    def is_trainable(self):
        return any(p.is_trainable() for p in self.parameters)

    def is_module(self):
        return False