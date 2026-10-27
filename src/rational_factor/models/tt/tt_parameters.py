"""Parameterized TT vectors and MPOs, including exactly invertible MPO layers.

These lightweight wrappers follow ``Parameters``: ``p()`` constructs a fresh
TT/MPO connected to the current leaf tensors' PyTorch autograd graphs. Parameter
ownership and optimizer registration remain in the supplied Parameters objects.
"""

from collections.abc import Sequence
import math
import torch

from rational_factor.models.parameters import (
    Parameters, FixedParameters, TrainableParameters, PositiveParameters,
)
from rational_factor.models.tt.tt import (
    TTVector, MPO, LowRankFactorizationMPO, KroneckerPlusLowRankFactorizationMPO, NilpotentMPO, GateLayerMPO, InvertibleMPO,
    InvertibleMPOProduct,
)


def _bond_ranks(d: int, ranks: int | Sequence[int]) -> tuple[int, ...]:
    if d < 1:
        raise ValueError('at least one site is required')
    if isinstance(ranks, int):
        result = (1,) + (ranks,) * (d - 1) + (1,)
    else:
        r = tuple(int(x) for x in ranks)
        if len(r) == d - 1:
            result = (1,) + r + (1,)
        elif len(r) == d + 1:
            result = r
        else:
            raise ValueError(f'ranks must have {d-1} or {d+1} entries')
    if result[0] != 1 or result[-1] != 1 or any(x < 1 for x in result):
        raise ValueError(f'invalid TT boundary/interior ranks: {result}')
    return result


def _random_core(cls, shape, *, trainable, mean, std, epsilon, dtype, device):
    if not issubclass(cls, TrainableParameters):
        raise TypeError('param_cls must derive from TrainableParameters')
    kw = dict(trainable=trainable, mean=mean, std=std)
    if issubclass(cls, PositiveParameters):
        kw['epsilon'] = epsilon
    # PositiveParameters.random_init(trainable=False) samples signed values,
    # which violates its fixed-value constructor's nonnegativity assertion.
    if issubclass(cls, PositiveParameters) and not trainable:
        values = torch.nn.functional.softplus(torch.randn(shape) * std + mean) + epsilon
        p = cls.from_values(values, trainable=False, epsilon=epsilon)
    else:
        p = cls.random_init(shape, **kw)
    if isinstance(p, torch.nn.Module):
        p.to(dtype=dtype, device=device)
    return p


def _modules(*providers):
    seen, result = set(), []
    for p in providers:
        if isinstance(p, Parameters):
            for module in p.parameter_modules():
                if id(module) not in seen:
                    seen.add(id(module))
                    result.append(module)
    return result


class TTVectorParameters(Parameters):
    """Each core provider returns ``(r_left, n, r_right)`` (or batched)."""

    def __init__(self, cores: Sequence[Parameters]):
        self._cores = tuple(cores)
        if not self._cores:
            raise ValueError('at least one TT vector core is required')
        self()  # Validate complete core layout.

    @classmethod
    def from_cores(cls, cores: Sequence[Parameters]):
        return cls(cores)

    @classmethod
    def from_core_spec(cls, modes: Sequence[int], ranks: int | Sequence[int] = 1, *,
                       param_cls: type[TrainableParameters] = TrainableParameters,
                       trainable=True, mean=0., std=1., epsilon=0.,
                       dtype=torch.float32, device=None):
        modes = tuple(int(n) for n in modes)
        if not modes or any(n < 1 for n in modes):
            raise ValueError('modes must be nonempty positive integers')
        r = _bond_ranks(len(modes), ranks)
        return cls([_random_core(param_cls, (r[i], n, r[i+1]), trainable=trainable,
                                 mean=mean, std=std, epsilon=epsilon, dtype=dtype,
                                 device=device) for i, n in enumerate(modes)])

    @property
    def cores(self):
        return self._cores

    @property
    def modes(self):
        return tuple(int(p.size()[-2]) for p in self._cores)

    @property
    def ranks(self):
        return (1,) + tuple(int(p.size()[-1]) for p in self._cores)

    @property
    def n(self):
        return math.prod(self.modes)

    def __call__(self) -> TTVector:
        return TTVector.from_cores([p() for p in self._cores])

    def is_trainable(self):
        return any(p.is_trainable() for p in self._cores)

    def parameter_modules(self):
        return _modules(*self._cores)

    def is_module(self):
        return False


class MPOParameters(Parameters):
    """Each core provider returns ``(r_left, n_row, n_col, r_right)``."""

    def __init__(self, cores: Sequence[Parameters]):
        self._cores = tuple(cores)
        if not self._cores:
            raise ValueError('at least one MPO core is required')
        self()  # Validate complete core layout.

    @classmethod
    def from_cores(cls, cores: Sequence[Parameters]):
        return cls(cores)

    @classmethod
    def from_core_spec(cls, row_modes: Sequence[int], ranks: int | Sequence[int] = 1,
                       col_modes: Sequence[int] | None = None, *,
                       param_cls: type[TrainableParameters] = TrainableParameters,
                       trainable=True, mean=0., std=1., epsilon=0.,
                       dtype=torch.float32, device=None):
        rows = tuple(int(x) for x in row_modes)
        cols = rows if col_modes is None else tuple(int(x) for x in col_modes)
        if not rows or len(rows) != len(cols) or any(x < 1 for x in rows + cols):
            raise ValueError('row_modes and col_modes must have matching, nonempty positive sizes')
        r = _bond_ranks(len(rows), ranks)
        return cls([_random_core(param_cls, (r[i], m, n, r[i+1]),
                                 trainable=trainable, mean=mean, std=std,
                                 epsilon=epsilon, dtype=dtype, device=device)
                    for i, (m, n) in enumerate(zip(rows, cols))])

    @property
    def cores(self):
        return self._cores

    @property
    def row_modes(self):
        return tuple(int(p.size()[-3]) for p in self._cores)

    @property
    def col_modes(self):
        return tuple(int(p.size()[-2]) for p in self._cores)

    @property
    def ranks(self):
        return (1,) + tuple(int(p.size()[-1]) for p in self._cores)

    def __call__(self) -> MPO:
        return MPO.from_cores([p() for p in self._cores])

    def is_trainable(self):
        return any(p.is_trainable() for p in self._cores)

    def parameter_modules(self):
        return _modules(*self._cores)

    def is_module(self):
        return False



class LowRankFactorizationMPOParameters(Parameters):
    r"""Structured rank-r ``B = U @ V.T`` with rectangular MPO factors.

    Both factors have column modes ``(1, ..., 1, r)`` and physical row modes
    matching ``row_modes`` / ``col_modes`` respectively. With the default
    ``row_stochastic=True``, positive local softmax constraints guarantee

        U @ ones(r) = ones(N_row),
        V.T @ ones(N_col) = ones(r),
        B >= 0, B @ ones(N_col) = ones(N_row).

    These are exact algebraic constraints (up to roundoff), without expanding
    exponentially large vectors or matrices. The V columns are normalized
    globally by an r-dimensional contraction, rather than by forcing an
    identical early-coordinate marginal across latent components.
    The normalization imposes a
    particular nonnegative MPO gauge and is not a universal representation of
    all nonnegative rank-r matrices at fixed MPO bond rank.
    """

    def __init__(self, U: Parameters, V: Parameters, *, row_stochastic: bool = True):
        if not isinstance(U, Parameters) or not isinstance(V, Parameters):
            raise TypeError("U and V must be Parameters objects")
        self.U, self.V = U, V
        self.row_stochastic = bool(row_stochastic)
        # Validate only shape/meta: avoid keeping parameter-dependent graphs.
        u, v = U(), V()
        LowRankFactorizationMPO(u, v)

    @classmethod
    def from_factors(cls, U: Parameters, V: Parameters, *, row_stochastic=True):
        return cls(U, V, row_stochastic=row_stochastic)

    @classmethod
    def from_core_spec(
        cls, row_modes: Sequence[int], rank: int,
        ranks: int | Sequence[int] = 1,
        col_modes: Sequence[int] | None = None, *,
        ranks_u: int | Sequence[int] | None = None,
        ranks_v: int | Sequence[int] | None = None,
        row_stochastic: bool = True,
        param_cls: type[TrainableParameters] = TrainableParameters,
        trainable=True, mean=0., std=1., epsilon=0.,
        dtype=torch.float32, device=None,
    ):
        rows = tuple(int(m) for m in row_modes)
        cols = rows if col_modes is None else tuple(int(m) for m in col_modes)
        if not rows or len(rows) != len(cols) or any(m < 1 for m in rows + cols):
            raise ValueError("row_modes and col_modes must have matching positive site sizes")
        if not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be a positive integer")
        latent_modes = (1,) * (len(rows) - 1) + (rank,)
        U = MPOParameters.from_core_spec(rows, ranks=ranks if ranks_u is None else ranks_u,
                col_modes=latent_modes, param_cls=param_cls, trainable=trainable,
                mean=mean, std=std, epsilon=epsilon, dtype=dtype, device=device)
        V = MPOParameters.from_core_spec(cols, ranks=ranks if ranks_v is None else ranks_v,
                col_modes=latent_modes, param_cls=param_cls, trainable=trainable,
                mean=mean, std=std, epsilon=epsilon, dtype=dtype, device=device)
        return cls(U, V, row_stochastic=row_stochastic)

    @staticmethod
    def _normalized_u(factor: MPO) -> MPO:
        result = []
        for k, core in enumerate(factor.cores):
            # Intermediate U cores define conditional distributions over
            # the next bond for every (current bond, row mode).
            dim = 2 if k == len(factor.cores) - 1 else -1
            result.append(core.softmax(dim=dim))
        return MPO(result)

    @staticmethod
    def _normalized_v(factor: MPO) -> MPO:
        # Normalize FULL latent columns, not the last core separately at each
        # incoming bond. The latter would make the marginal of the first d-1
        # coordinates identical for all latent components (a severe limitation).
        result = []
        for core in factor.cores[:-1]:
            left, n, _, right = core.shape
            result.append(core.reshape(left, n * right).softmax(dim=-1)
                          .reshape(left, n, 1, right))
        result.append(torch.nn.functional.softplus(factor.cores[-1]))
        raw = MPO(result)
        ones = TTVector([core.new_ones(1, n, 1)
                         for core, n in zip(raw.cores, raw.row_modes)])
        column_sums = raw.T.matvec(ones).to_dense()
        result[-1] = result[-1] / column_sums[None, None, :, None]
        return MPO(result)

    def __call__(self) -> LowRankFactorizationMPO:
        U, V = self.U(), self.V()
        if self.row_stochastic:
            U, V = self._normalized_u(U), self._normalized_v(V)
        return LowRankFactorizationMPO(U, V)

    def is_row_stochastic(self) -> bool:
        return self.row_stochastic

    @property
    def row_modes(self):
        return self.U.row_modes

    @property
    def col_modes(self):
        return self.V.row_modes

    @property
    def rank(self):
        return self.U.col_modes[-1]

    def is_trainable(self):
        return self.U.is_trainable() or self.V.is_trainable()

    def parameter_modules(self):
        return _modules(self.U, self.V)

    def is_module(self):
        return False


class KroneckerPlusLowRankFactorizationMPOParameters(LowRankFactorizationMPOParameters):
    r"""Parameterize ``K + U @ V.T`` with a Kronecker-rank-one MPO ``K``.

    By default ``row_stochastic=True`` constructs a convex combination

        B = (1 - mixture_weight) * K_normalized
            + mixture_weight * U_normalized @ V_normalized.T

    where ``K_normalized`` is a product of local row-stochastic matrices.
    This guarantees ``B >= 0`` and ``B @ 1 = 1`` without dense matrices.
    When ``row_stochastic=False`` this constructs the unscaled sum
    ``K + U @ V.T``. The mixture weight is unused in that mode.

    ``mixture_weight`` may be a fixed probability in [0, 1] or a scalar
    Parameters provider whose raw value is mapped through sigmoid, allowing
    a trainable mixture while preserving the stochastic constraint.
    ``from_core_spec(..., trainable=True)`` converts a float init into such a
    logit parameter automatically.
    """

    def __init__(self, kronecker: Parameters, U: Parameters, V: Parameters, *,
                 row_stochastic: bool = True,
                 mixture_weight: float | Parameters = 0.5):
        super().__init__(U, V, row_stochastic=row_stochastic)
        if not isinstance(kronecker, Parameters):
            raise TypeError("kronecker must be a Parameters provider")
        self.kronecker = kronecker
        self.mixture_weight = mixture_weight
        if isinstance(mixture_weight, Parameters):
            if mixture_weight().numel() != 1:
                raise ValueError("mixture_weight Parameters must produce a scalar")
        elif not (0.0 <= float(mixture_weight) <= 1.0):
            raise ValueError("fixed mixture_weight must be in [0, 1]")
        KroneckerPlusLowRankFactorizationMPO(kronecker(), self.U(), self.V())

    @classmethod
    def from_factors(cls, kronecker: Parameters, U: Parameters, V: Parameters, *,
                     row_stochastic: bool = True,
                     mixture_weight: float | Parameters = 0.5):
        return cls(kronecker, U, V, row_stochastic=row_stochastic,
                   mixture_weight=mixture_weight)

    @staticmethod
    def _as_mixture_weight_parameters(
        mixture_weight: float | Parameters, *, trainable: bool,
        dtype=torch.float32, device=None,
    ) -> float | Parameters:
        """Keep an explicit Parameters provider, else turn a probability into a logit.

        When ``trainable=True``, a float in ``[0, 1]`` becomes a scalar
        ``TrainableParameters`` initialized so ``sigmoid(logit) ≈ mixture_weight``.
        When ``trainable=False``, the float is left as a fixed probability.
        """
        if isinstance(mixture_weight, Parameters):
            return mixture_weight
        w = float(mixture_weight)
        if not (0.0 <= w <= 1.0):
            raise ValueError("mixture_weight must be in [0, 1]")
        if not trainable:
            return w
        # Avoid ±inf logits at the endpoints while matching the requested init.
        eps = 1e-4
        w_c = min(max(w, eps), 1.0 - eps)
        logit = math.log(w_c / (1.0 - w_c))
        p = TrainableParameters.from_values(
            torch.tensor(logit, dtype=dtype, device=device), trainable=True,
        )
        return p

    @classmethod
    def from_core_spec(
        cls, row_modes: Sequence[int], rank: int,
        ranks: int | Sequence[int] = 1,
        col_modes: Sequence[int] | None = None, *,
        ranks_u: int | Sequence[int] | None = None,
        ranks_v: int | Sequence[int] | None = None,
        row_stochastic: bool = True,
        mixture_weight: float | Parameters = 0.5,
        param_cls: type[TrainableParameters] = TrainableParameters,
        trainable=True, mean=0., std=1., epsilon=0.,
        dtype=torch.float32, device=None,
    ):
        rows = tuple(int(n) for n in row_modes)
        cols = rows if col_modes is None else tuple(int(n) for n in col_modes)
        lowrank = LowRankFactorizationMPOParameters.from_core_spec(
            rows, rank, ranks, cols, ranks_u=ranks_u, ranks_v=ranks_v,
            row_stochastic=row_stochastic, param_cls=param_cls,
            trainable=trainable, mean=mean, std=std, epsilon=epsilon,
            dtype=dtype, device=device,
        )
        kronecker = MPOParameters.from_core_spec(
            rows, ranks=1, col_modes=cols, param_cls=param_cls,
            trainable=trainable, mean=mean, std=std, epsilon=epsilon,
            dtype=dtype, device=device,
        )
        mw = cls._as_mixture_weight_parameters(
            mixture_weight, trainable=trainable, dtype=dtype, device=device,
        )
        return cls(kronecker, lowrank.U, lowrank.V,
                   row_stochastic=row_stochastic, mixture_weight=mw)

    @staticmethod
    def _normalized_kronecker(kronecker: MPO) -> MPO:
        # Row stochasticity factorizes over local row indices.
        return MPO([core.softmax(dim=2) for core in kronecker.cores])

    def __call__(self) -> KroneckerPlusLowRankFactorizationMPO:
        K, U, V = self.kronecker(), self.U(), self.V()
        if self.row_stochastic:
            K = self._normalized_kronecker(K)
            U, V = self._normalized_u(U), self._normalized_v(V)
            if isinstance(self.mixture_weight, Parameters):
                weight = torch.sigmoid(self.mixture_weight().reshape(()))
            else:
                weight = torch.as_tensor(self.mixture_weight, dtype=U.dtype, device=U.device)
            K = K.scale(1 - weight)
            # Absorb low-rank mixture scale in the first U core.
            u_cores = list(U.cores)
            u_cores[0] = u_cores[0] * weight
            U = MPO(u_cores)
        return KroneckerPlusLowRankFactorizationMPO(K, U, V)

    def is_trainable(self):
        return (super().is_trainable() or self.kronecker.is_trainable() or
                (isinstance(self.mixture_weight, Parameters) and
                 self.mixture_weight.is_trainable()))

    def parameter_modules(self):
        return _modules(self.kronecker, self.U, self.V, self.mixture_weight)


KroneckerPlusLowRankFactorizationParameters = KroneckerPlusLowRankFactorizationMPOParameters


class ConstantModeRank1MPOParameters(Parameters):
    """Rank-one MPO S with e_0.T @ S == e_0.T, without positivity constraints.

    Each Parameters object supplies the free lower (n_k-1) rows of a local
    n_k x n_k matrix; the first row [1, 0, ..., 0] is inserted exactly.
    The resulting S is their Kronecker product, with MPO ranks all equal to 1.
    """

    def __init__(self, lower_rows: Sequence[Parameters]):
        self._lower_rows = tuple(lower_rows)
        if not self._lower_rows:
            raise ValueError('at least one local matrix is required')
        for p in self._lower_rows:
            if not isinstance(p, Parameters):
                raise TypeError('each local block must be a Parameters object')
            shape = tuple(p().shape)
            if len(shape) != 2 or shape[1] != shape[0] + 1:
                raise ValueError(f'expected local free rows of shape (n-1, n), got {shape}')

    @classmethod
    def from_cores(cls, lower_rows: Sequence[Parameters]):
        return cls(lower_rows)

    @classmethod
    def from_core_spec(cls, row_modes: Sequence[int], ranks: int | Sequence[int] = 1,
                       col_modes: Sequence[int] | None = None, *,
                       param_cls: type[TrainableParameters] = TrainableParameters,
                       trainable=True, mean=0., std=0.1, epsilon=0.,
                       dtype=torch.float32, device=None):
        modes = tuple(int(n) for n in row_modes)
        if not modes or any(n < 1 for n in modes):
            raise ValueError('row_modes must be nonempty positive integers')
        if col_modes is not None and tuple(col_modes) != modes:
            raise ValueError('local matrices must be square (col_modes == row_modes)')
        if any(r != 1 for r in _bond_ranks(len(modes), ranks)):
            raise ValueError('ConstantModeRank1MPOParameters requires ranks=1')
        return cls([FixedParameters(torch.empty((0, 1), dtype=dtype, device=device))
                   if n == 1 else _random_core(param_cls, (n - 1, n), trainable=trainable,
                   mean=mean, std=std, epsilon=epsilon, dtype=dtype, device=device)
                   for n in modes])

    @property
    def lower_rows(self):
        return self._lower_rows

    @property
    def cores(self):
        """Free lower-row parameter providers (not full four-axis MPO cores)."""
        return self._lower_rows

    @property
    def row_modes(self):
        return tuple(int(p.size()[1]) for p in self._lower_rows)

    @property
    def col_modes(self):
        return self.row_modes

    @property
    def ranks(self):
        return (1,) * (len(self._lower_rows) + 1)

    def __call__(self) -> MPO:
        cores = []
        for p in self._lower_rows:
            lower = p()
            n = lower.shape[1]
            top = torch.cat((lower.new_ones(1, 1), lower.new_zeros(1, n - 1)), dim=1)
            cores.append(torch.cat((top, lower), dim=0)[None, :, :, None])
        return MPO(cores)

    def is_trainable(self):
        return any(p.is_trainable() for p in self._lower_rows)

    def parameter_modules(self):
        return _modules(*self._lower_rows)

    def is_module(self):
        return False


class NilpotentMPOParameters(Parameters):
    """Generate ``I + strength * |output><input|_control ⊗ B`` with exact inverse.

    ``block`` is an MPOParameters (or another Parameters returning a square MPO)
    on all sites except ``control_site``. No normalization constraint on B.
    """

    def __init__(self, modes: Sequence[int], block: Parameters | None, *,
                 control_site=0, input_mode=0, output_mode=1,
                 strength: Parameters | torch.Tensor | float = 1.,
                 dtype=torch.float32, device=None):
        self.modes = tuple(int(n) for n in modes)
        self.block = block
        self.control_site = int(control_site)
        self.input_mode, self.output_mode = int(input_mode), int(output_mode)
        self.strength = strength
        self.dtype, self.device = dtype, device
        if not self.modes or not 0 <= self.control_site < len(self.modes):
            raise ValueError('invalid modes/control_site')
        if self.modes[self.control_site] < 2 or not (0 <= self.input_mode < self.modes[self.control_site]
                 and 0 <= self.output_mode < self.modes[self.control_site]
                 and self.input_mode != self.output_mode):
            raise ValueError('input and output must be distinct valid control modes')
        if (block is None) != (len(self.modes) == 1):
            raise ValueError('block must be supplied exactly when there are other sites')
        if block is not None and (not isinstance(block, Parameters) or
                                  not isinstance(block(), MPO)):
            raise TypeError('block must be a Parameters object returning an MPO')
        other = self.modes[:self.control_site] + self.modes[self.control_site+1:]
        if block is not None and (block().row_modes != other or block().col_modes != other):
            raise ValueError(f'block must have square MPO modes {other}')

    @classmethod
    def from_core_spec(cls, modes: Sequence[int], ranks: int | Sequence[int] = 1, *,
                       control_site=0, input_mode=0, output_mode=1, strength=1.,
                       param_cls: type[TrainableParameters] = TrainableParameters,
                       trainable=True, mean=0., std=0.1, epsilon=0.,
                       dtype=torch.float32, device=None):
        modes = tuple(int(x) for x in modes)
        if not modes or not 0 <= control_site < len(modes):
            raise ValueError('invalid modes/control_site')
        other = modes[:control_site] + modes[control_site+1:]
        B = MPOParameters.from_core_spec(other, ranks, param_cls=param_cls,
                    trainable=trainable, mean=mean, std=std, epsilon=epsilon,
                    dtype=dtype, device=device) if other else None
        return cls(modes, B, control_site=control_site, input_mode=input_mode,
                   output_mode=output_mode, strength=strength, dtype=dtype, device=device)

    def __call__(self) -> NilpotentMPO:
        return NilpotentMPO(self.modes, None if self.block is None else self.block(),
                             control_site=self.control_site, input_mode=self.input_mode,
                             output_mode=self.output_mode,
                             strength=self.strength() if isinstance(self.strength, Parameters)
                             else self.strength, dtype=self.dtype, device=self.device)

    def is_trainable(self):
        return ((self.block is not None and self.block.is_trainable()) or
                (isinstance(self.strength, Parameters) and self.strength.is_trainable()))

    def parameter_modules(self):
        return _modules(self.block, self.strength)

    def is_module(self):
        return False


class ConstantModeNilpotentMPOParameters(NilpotentMPOParameters):
    """Preserve ``e0.T @ V == e0.T`` by restricting the control output mode."""

    def __init__(self, *args, output_mode=1, **kwargs):
        if output_mode == 0:
            raise ValueError('output_mode=0 does not preserve the constant left mode')
        super().__init__(*args, output_mode=output_mode, **kwargs)


class GateLayerMPOParameters(Parameters):
    """Parameterize invertible, disjoint, contiguous k-site gates.

    Each ``(sites, raw_parameters)`` provides a square (prod local modes)^2
    tensor. Default ``parameterization='expm'`` makes each gate ``exp(raw)``,
    guaranteeing invertibility. ``'direct'`` treats raw as a known-invertible
    gate and can lose invertibility during unconstrained optimization.
    """

    def __init__(self, modes: Sequence[int], gates: Sequence[tuple[Sequence[int], Parameters]],
                 *, parameterization='expm', dtype=torch.float32, device=None):
        self.modes = tuple(int(n) for n in modes)
        self.gates = tuple((tuple(int(i) for i in sites), p) for sites, p in gates)
        self.parameterization, self.dtype, self.device = parameterization, dtype, device
        if parameterization not in ('expm', 'direct'):
            raise ValueError('parameterization must be expm or direct')
        used = set()
        if not self.modes or any(n < 1 for n in self.modes):
            raise ValueError('modes must be nonempty positive integers')
        for sites, p in self.gates:
            if (not sites or sites != tuple(range(sites[0], sites[0] + len(sites)))
                    or sites[0] < 0 or sites[-1] >= len(self.modes)
                    or not used.isdisjoint(sites)):
                raise ValueError('gate sites must be contiguous, disjoint and in range')
            used.update(sites)
            m = math.prod(self.modes[i] for i in sites)
            if not isinstance(p, Parameters) or tuple(p().shape) != (m, m):
                raise ValueError(f'gate parameter on {sites} must have shape ({m}, {m})')

    @classmethod
    def from_gate_spec(cls, modes: Sequence[int], groups: Sequence[Sequence[int]], *,
                       param_cls: type[TrainableParameters] = TrainableParameters,
                       trainable=True, mean=0., std=0.1, epsilon=0.,
                       dtype=torch.float32, device=None, **kwargs):
        modes = tuple(int(x) for x in modes)
        gates = []
        for group in groups:
            group = tuple(int(i) for i in group)
            if not group or min(group) < 0 or max(group) >= len(modes):
                raise ValueError('invalid gate group')
            size = math.prod(modes[i] for i in group)
            gates.append((group, _random_core(param_cls, (size, size), trainable=trainable,
                mean=mean, std=std, epsilon=epsilon, dtype=dtype, device=device)))
        return cls(modes, gates, dtype=dtype, device=device, **kwargs)

    @classmethod
    def from_gates(cls, modes: Sequence[int], gates, **kwargs):
        return cls(modes, gates, **kwargs)

    def _gate_matrix(self, raw: torch.Tensor) -> torch.Tensor:
        return torch.matrix_exp(raw) if self.parameterization == 'expm' else raw

    def __call__(self) -> GateLayerMPO:
        gates = [(sites, self._gate_matrix(p())) for sites, p in self.gates]
        return GateLayerMPO(self.modes, gates, dtype=self.dtype, device=self.device)

    def is_trainable(self):
        return any(p.is_trainable() for _, p in self.gates)

    def parameter_modules(self):
        return _modules(*(p for _, p in self.gates))

    def is_module(self):
        return False


class ConstantModeGateLayerMPOParameters(GateLayerMPOParameters):
    """Invertible gates with first row e_0^T, preserving orthonormal-basis mass.

    Raw gate tensors are converted to ``[[1,0],[b, exp(H)]]``.
    """

    def _gate_matrix(self, raw: torch.Tensor) -> torch.Tensor:
        size = raw.shape[-1]
        if size == 1:
            return torch.ones_like(raw)
        bottom = torch.cat((raw[1:, :1], torch.matrix_exp(raw[1:, 1:])), dim=1)
        top = torch.cat((torch.ones_like(raw[:1, :1]), torch.zeros_like(raw[:1, 1:])), dim=1)
        return torch.cat((top, bottom), dim=0)


class InvertibleMPOProductParameters(Parameters):
    """Keep invertible MPO parameter providers in right-to-left application order."""

    def __init__(self, layers: Sequence[Parameters]):
        self.layers = tuple(layers)
        if not self.layers or not all(isinstance(p, Parameters) for p in self.layers):
            raise ValueError('layers must be a nonempty sequence of Parameters')

    def __call__(self) -> InvertibleMPOProduct:
        return compose_invertible_mpo_parameters(*self.layers)

    def is_trainable(self):
        return any(p.is_trainable() for p in self.layers)

    def parameter_modules(self):
        return _modules(*self.layers)

    def is_module(self):
        return False




class SimilarityMPOParameters(Parameters):
    """Build ``M = V @ S @ V.inverse()`` from independently parameterized MPOs.

    ``transform`` must construct an InvertibleMPO (including an
    InvertibleMPOProduct); ``center`` must construct a square MPO with matching
    local modes. In particular, use ConstantModeNilpotentMPOParameters (or
    ConstantModeGateLayerMPOParameters) for V and
    ConstantModeRank1MPOParameters for S to guarantee ``e0.T @ M == e0.T``.

    Both factors are freshly evaluated on every call, preserving gradients
    through the exact MPO multiplication and specialized inverse. The MPO
    product is not compressed: its bond ranks can grow with circuit depth.
    """

    def __init__(self, transform: Parameters, center: Parameters):
        if not isinstance(transform, Parameters) or not isinstance(center, Parameters):
            raise TypeError('transform and center must be Parameters providers')
        self.transform = transform
        self.center = center
        V = transform()
        S = center()
        if not isinstance(V, InvertibleMPO):
            raise TypeError('transform must construct an InvertibleMPO')
        if not isinstance(S, MPO):
            raise TypeError('center must construct an MPO')
        if V.row_modes != V.col_modes or S.row_modes != S.col_modes or V.row_modes != S.row_modes:
            raise ValueError('transform and center must have identical square local modes')

    @property
    def row_modes(self):
        return self.center().row_modes

    @property
    def col_modes(self):
        return self.center().col_modes

    def __call__(self) -> MPO:
        V = self.transform()
        S = self.center()
        return V @ S @ V.inverse()

    def is_trainable(self):
        return self.transform.is_trainable() or self.center.is_trainable()

    def parameter_modules(self):
        return _modules(self.transform, self.center)

    def is_module(self):
        return False


def compose_invertible_mpo_parameters(*layers: Parameters) -> InvertibleMPOProduct:
    """Materialize ``L_last @ ... @ L_first`` with a structured exact inverse.

    Accepts either positional layer providers or one sequence of providers.
    Calling again rebuilds all MPOs and autograd graphs from current values.
    """
    if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
        layers = tuple(layers[0])
    factors = [layer() for layer in layers]
    if not factors or any(not isinstance(f, InvertibleMPO) for f in factors):
        raise TypeError('layers must produce a nonempty sequence of InvertibleMPO objects')
    return InvertibleMPOProduct(factors)


# Friendly names for the layer-based parameter API.
InvertibleMPOCircuitParameters = InvertibleMPOProductParameters
