import torch
import itertools
from abc import ABC, abstractmethod
from collections.abc import Sequence

from rational_factor.models.structured_matrices import (
    DenseMatrix,
    Identity,
    LowRankFactorization,
    Order1Quasiseparable,
    Quasiseparable,
    R1PDFactorization,
)
from rational_factor.models.structured_vectors import OneVector
from rational_factor.models.mlp import MLP


class Parameters(ABC):
    def __init__(self, shape: tuple[int, ...], batch_shape: tuple[int, ...] = (1,)):
        self._shape = shape
        self._batch_shape = batch_shape

    def shape(self):
        return self._shape

    def batch_shape(self):
        return self._batch_shape

    @abstractmethod
    def __call__(self, **context): ...

    @abstractmethod
    def is_trainable(self): ...

    def parameter_modules(self) -> list[torch.nn.Module]:
        """Leaf modules owned by this parameterization."""
        return [self] if isinstance(self, torch.nn.Module) else []


class FixedParameters(Parameters):
    def __init__(self, fixed_values: torch.Tensor = None):
        self._p = fixed_values
        super().__init__(shape=tuple(fixed_values.shape))

    def is_trainable(self):
        return self._p.requires_grad

    def dtype_device(self):
        return self._p.dtype, self._p.device

    def __call__(self, **context):
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
        Parameters.__init__(self, shape=tuple(self._p.shape))

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

    def forward(self, **context) -> torch.Tensor:
        """Unconstrained values. Subclasses override to apply constraints."""
        return self._p

    def __call__(self, **context) -> torch.Tensor:
        return torch.nn.Module.__call__(self, **context)

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

    def forward(self, **context) -> torch.Tensor:
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


class IdentityParameters(Parameters, torch.nn.Module):
    """Structured identity; empty buffer tracks dtype/device through ``Module.to``."""

    def __init__(self, n: int, dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")):
        torch.nn.Module.__init__(self)
        Parameters.__init__(self, shape=(n, n))
        self._n = n
        # Anchor buffer so model.to(device/dtype) relocates structured identity params.
        self.register_buffer("_anchor", torch.empty((), dtype=dtype, device=device))

    def __call__(self, **context):
        return Identity(self._n, self.batch_shape(), self._anchor.dtype, self._anchor.device)

    def dtype_device(self):
        return self._anchor.dtype, self._anchor.device

    def is_trainable(self):
        return False

    def is_module(self):
        return True


class OneVectorParameters(Parameters, torch.nn.Module):
    """All-ones vector; empty buffer tracks dtype/device through ``Module.to``."""

    def __init__(self, n: int, dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")):
        torch.nn.Module.__init__(self)
        Parameters.__init__(self, shape=(n,))
        self._n = n
        # Anchor buffer so model.to(device/dtype) relocates default ones-coeffs.
        self.register_buffer("_anchor", torch.empty((), dtype=dtype, device=device))

    def __call__(self, **context) -> OneVector:
        return OneVector(self._n, self.batch_shape(), self._anchor.dtype, self._anchor.device)

    def dtype_device(self):
        return self._anchor.dtype, self._anchor.device

    def is_trainable(self):
        return False

    def is_module(self):
        return True


class DenseMatrixParameters(Parameters):
    """Wraps a ``(..., n, m)`` tensor parameter; ``()`` returns a ``DenseMatrix``."""

    def __init__(self, values: Parameters):
        super().__init__(shape=values.shape())
        self._values = values

    def __call__(self, **context) -> DenseMatrix:
        return DenseMatrix(self._values(**context))

    def is_trainable(self):
        return self._values.is_trainable()

    def parameter_modules(self) -> list[torch.nn.Module]:
        return self._values.parameter_modules()

    def is_module(self):
        return False


class RowStochasticMatrixParameters(Parameters):
    pass


class RowStochasticDenseMatrixParameters(DenseMatrixParameters, RowStochasticMatrixParameters):
    def __call__(self, **context) -> DenseMatrix:
        row_stochastic_values = torch.softmax(self._values(**context), dim=-1)
        return DenseMatrix(row_stochastic_values)


class R1PDFactorizationParameters(Parameters):
    def __init__(
        self,
        u: Parameters,
        v: Parameters,
        d: Parameters | None = None,
        *,
        seq_dim: int = -2,
        normalization: str | None = None,
    ):
        assert u.shape() == v.shape(), "u and v must have the same shape"
        assert len(u.shape()) >= 2, "u and v must have shape (..., T, ..., n) with a sequence axis"
        if normalization is not None and d is not None:
            raise ValueError(
                "d is ignored when normalization is set; omit d and pass "
                "unconstrained TrainableParameters for u and v"
            )
        if d is not None:
            assert d.shape() == u.shape(), "d, u, and v must have the same shape"
        batch_ndim = len(u.shape()) - 1
        if not (-batch_ndim <= seq_dim < batch_ndim):
            raise ValueError(
                f"seq_dim must index a batch axis of u/v/d, got seq_dim={seq_dim} "
                f"for shape {tuple(u.shape())}"
            )
        super().__init__(shape=u.shape())
        self.d = d
        self.u = u
        self.v = v
        self.seq_dim = seq_dim % batch_ndim
        self.normalization = normalization

    def __call__(self, **context) -> R1PDFactorization:
        d = None if self.d is None else self.d(**context)
        return R1PDFactorization(
            self.u(**context),
            self.v(**context),
            d,
            seq_dim=self.seq_dim,
            normalization=self.normalization,
        )

    def is_trainable(self):
        trainable = self.u.is_trainable() or self.v.is_trainable()
        if self.d is not None:
            trainable = trainable or self.d.is_trainable()
        return trainable

    def is_module(self):
        return False

    def parameter_modules(self) -> list[torch.nn.Module]:
        params = (self.u, self.v) if self.d is None else (self.d, self.u, self.v)
        return [module for param in params for module in param.parameter_modules()]


class LowRankFactorizationParameters(Parameters):
    """Wraps a ``(..., n, k, r)`` tensor parameter; ``()`` returns a ``LowRankFactorization``."""

    def __init__(self, U: Parameters, V: Parameters):
        ush, vsh = U.shape(), V.shape()
        super().__init__(shape=ush[:-1] + (vsh[-2],))
        self.U = U
        self.V = V

    def __call__(self, **context) -> LowRankFactorization:
        return LowRankFactorization(self.U(**context), self.V(**context))

    def is_trainable(self):
        return self.U.is_trainable() and self.V.is_trainable()

    def parameter_modules(self) -> list[torch.nn.Module]:
        return [module for param in (self.U, self.V) for module in param.parameter_modules()]

    def is_module(self):
        return False


class Order1Quasiseparable1Parameters(Parameters):
    """Trainable ``P = L D U``.

    ``diag`` must already be positive (e.g. :class:`PositiveParameters`); it is
    used as the factor diagonal ``D`` with no further transform.

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
        transition_bound: float | None = None,
    ):
        self.lower_p, self.lower_a, self.lower_q = lower_p, lower_a, lower_q
        self.diag = diag
        self.upper_g, self.upper_b, self.upper_h = upper_g, upper_b, upper_h
        self.transition_bound = transition_bound

        shape = diag.shape()
        if any(p.shape() != shape for p in self.parameters):
            raise ValueError("all factor Parameters must share the same shape")
        super().__init__(shape=shape)

    @property
    def parameters(self) -> tuple[Parameters, ...]:
        return (
            self.lower_p, self.lower_a, self.lower_q, self.diag,
            self.upper_g, self.upper_b, self.upper_h,
        )

    def __call__(self, **context) -> Order1Quasiseparable:
        la, ub = self.lower_a(**context), self.upper_b(**context)
        if self.transition_bound is not None:
            la = self.transition_bound * torch.tanh(la)
            ub = self.transition_bound * torch.tanh(ub)
        return Order1Quasiseparable(
            self.lower_p(**context), la, self.lower_q(**context), self.diag(**context),
            self.upper_g(**context), ub, self.upper_h(**context),
        )
    
    def is_trainable(self):
        return any(p.is_trainable() for p in self.parameters)

    def is_module(self):
        return False

    def parameter_modules(self) -> list[torch.nn.Module]:
        return [module for param in self.parameters for module in param.parameter_modules()]


class QuasiseparableParameters(Parameters):
    """Trainable order-``k`` ``P = L D U`` with diagonal transitions.

    Generator parameters have shape ``(..., m, k)``; ``diag`` has shape ``(..., m)``.
    ``diag`` must already be positive (e.g. :class:`PositiveParameters`); it is
    used as the factor diagonal ``D`` with no further transform.

    Same ``transition_bound`` stabilization as
    :class:`Order1QuasiseparableFactorization`.
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
        transition_bound: float | None = None,
    ):
        self.lower_p, self.lower_a, self.lower_q = lower_p, lower_a, lower_q
        self.diag = diag
        self.upper_g, self.upper_b, self.upper_h = upper_g, upper_b, upper_h
        self.transition_bound = transition_bound

        d_shape = diag.shape()
        gen_shape = lower_p.shape()
        if gen_shape != d_shape + (gen_shape[-1],):
            raise ValueError(
                f"generators must have shape diag.shape + (k,), got diag={tuple(d_shape)}, "
                f"gen={tuple(gen_shape)}"
            )
        if any(p.shape() != gen_shape for p in (
            self.lower_p, self.lower_a, self.lower_q,
            self.upper_g, self.upper_b, self.upper_h,
        )):
            raise ValueError("all generator Parameters must share shape (..., m, k)")
        super().__init__(shape=d_shape)

    @property
    def parameters(self) -> tuple[Parameters, ...]:
        return (
            self.lower_p, self.lower_a, self.lower_q, self.diag,
            self.upper_g, self.upper_b, self.upper_h,
        )

    def __call__(self, **context) -> Quasiseparable:
        la, ub = self.lower_a(**context), self.upper_b(**context)
        if self.transition_bound is not None:
            la = self.transition_bound * torch.tanh(la)
            ub = self.transition_bound * torch.tanh(ub)
        return Quasiseparable(
            self.lower_p(**context), la, self.lower_q(**context), self.diag(**context),
            self.upper_g(**context), ub, self.upper_h(**context),
        )

    def is_trainable(self):
        return any(p.is_trainable() for p in self.parameters)

    def is_module(self):
        return False

    def parameter_modules(self) -> list[torch.nn.Module]:
        return [module for param in self.parameters for module in param.parameter_modules()]


class NeuralParameters(MLP, Parameters):
    """MLP whose outputs are used as :class:`Parameters` values.

    ``__call__(**context)`` feeds ``context[context_key]`` through the network.
    With no context (e.g. construction-time shape checks), a single zero input
    is used so the returned tensor has leading data-batch size 1.

    Context-conditioned outputs have shape ``(*data_batch, *out_shape)`` and
    broadcast against static (non-context) parameters whose leading ensemble
    batch is typically ``1``.
    """

    def __init__(
        self,
        in_features: int | Sequence[int],
        out_features: int | Sequence[int],
        batch_shape: tuple[int, ...] = (1,),
        context_key: str = "action",
        hidden_features: int = 128,
        num_hidden_layers: int = 2,
        activation=torch.nn.Tanh,
        zero_init_last: bool = True,
        positive: bool = False,
    ):
        MLP.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
            zero_init_last=zero_init_last,
        )
        Parameters.__init__(self, shape=self.out_shape, batch_shape=batch_shape)
        self.context_key = context_key
        self.positive = positive

    def __call__(self, **context) -> torch.Tensor:
        if self.context_key in context:
            x = context[self.context_key]
        elif not context:
            param = next(self.parameters())
            x = torch.zeros((1, *self.in_shape), dtype=param.dtype, device=param.device)
        else:
            raise KeyError(
                f"NeuralParameters requires context key '{self.context_key}', "
                f"got {sorted(context)}"
            )
        out = MLP.forward(self, x)
        if self.positive:
            out = torch.nn.functional.softplus(out)
        return out

    def is_trainable(self) -> bool:
        return True

    def is_module(self) -> bool:
        return True

    def parameter_modules(self) -> list[torch.nn.Module]:
        return [self]
