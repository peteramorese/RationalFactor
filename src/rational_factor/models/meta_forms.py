import torch
import torch.nn as nn
from torch.func import functional_call
from collections import OrderedDict
from typing import Mapping
from copy import copy, deepcopy
from .domain_transformation import MLP
from .basis_functions import PositiveParameters, GaussianBasis, Parameters


def _coeff_param_name(param_shapes: dict[str, tuple[int, ...]]) -> str:
    coeff_names = [name for name in param_shapes if name.endswith("coeffs._p")]
    if not coeff_names:
        raise ValueError("Target module has no coefficient parameters in the meta-form layout.")
    return coeff_names[0]

class MethodCaller(nn.Module):
    """Invoke a method on ``target_module`` under ``functional_call`` param swapping."""

    def __init__(self, target_module: nn.Module, method_name: str):
        super().__init__()
        self._target_module = target_module
        self.method_name = method_name
        for child_name, child in target_module.named_children():
            self.add_module(child_name, child)

    def forward(self, *args, **kwargs):
        method = getattr(self._target_module, self.method_name)
        return method(*args, **kwargs)

class FunctionalModuleProxy:
    def __init__(
        self,
        target_module: nn.Module,
        params: Mapping[str, torch.Tensor],
        buffers: Mapping[str, torch.Tensor] | None = None,
        strict: bool = False,
    ):
        self._target_module = target_module
        self._params = OrderedDict(params)
        self._buffers = OrderedDict(buffers or {})
        self._strict = strict

    def state(self):
        state = OrderedDict()
        state.update(self._params)
        state.update(self._buffers)
        return state

    def parameters_dict(self):
        return self._params

    def buffers_dict(self):
        return self._buffers
    
    def __class__(self):
        return self._target_module.__class__

    def __call__(self, *args, **kwargs):
        return functional_call(
            self._target_module,
            self.state(),
            args,
            kwargs,
            strict=self._strict,
        )
    
    def shallow_copy_target_module(self) -> "FunctionalModuleProxy":
        """
        Return a new proxy whose basis module is a shallow copy of ``_target_module``.

        MLP-predicted tensors in ``_params`` / ``_buffers`` are shared with this proxy
        (gradients still flow to the meta-form). 
        """
        return FunctionalModuleProxy(
            target_module=copy(self._target_module),
            params=self._params,
            buffers=self._buffers,
            strict=self._strict,
        )

    def __getattr__(self, name: str):
        attr = getattr(self._target_module, name)

        if not callable(attr):
            return attr

        def wrapped_method(*args, **kwargs):
            method_module = MethodCaller(self._target_module, name)

            return functional_call(
                method_module,
                self.state(),
                args,
                kwargs,
                strict=self._strict,
            )

        return wrapped_method

class MLPMetaForm(nn.Module):
    def __init__(
        self,
        target_module_dim : int,
        context_dims: dict[str, int],
        target_module: nn.Module,
        hidden_features: int = 128,
        num_hidden_layers: int = 2,
        activation: type = nn.SiLU,
        zero_init_last: bool = False,
    ):
        super().__init__()

        if not context_dims:
            raise ValueError("context_dims must contain at least one named context entry.")

        self._context_keys = tuple(context_dims.keys())
        self._context_dims = dict(context_dims)

        self.target_module = target_module
        # Layout-only placeholders; MLP outputs replace these via functional_call.
        for param in self.target_module.parameters():
            param.requires_grad_(False)

        self.param_shapes = {
            name: p.shape for name, p in target_module.named_parameters()
        }
        self.param_sizes = {
            name: p.numel() for name, p in target_module.named_parameters()
        }

        self._dim = target_module_dim
        output_dim = sum(self.param_sizes.values())

        self.mlp = MLP(
            in_features=self.context_dim(),
            out_features=output_dim,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
            zero_init_last=zero_init_last,
        )
    
    def dim(self) -> int:
        return self._dim

    def context_dim(self) -> int:
        return sum(self._context_dims[key] for key in self._context_keys)

    def context_keys(self) -> tuple[str, ...]:
        return self._context_keys

    def context_dims(self) -> dict[str, int]:
        return dict(self._context_dims)
    
    def valid(self):
        return True

    def _validate_context_kwargs(self, **context: torch.Tensor) -> dict[str, torch.Tensor]:
        if set(context.keys()) != set(self._context_keys):
            raise ValueError(
                f"Expected context keyword arguments {self._context_keys}, got {tuple(context.keys())}."
            )

        return {key: torch.as_tensor(context[key]) for key in self._context_keys}

    def _pack_context(self, **context: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """
        Stack context tensors in registered key order.

        Returns:
            packed: [context_dim] or [batch, context_dim]
            single: True when packed has no batch dimension
        """
        context = self._validate_context_kwargs(**context)

        parts = []
        batch_size = None

        for key in self._context_keys:
            t = context[key]
            expected = self._context_dims[key]

            if t.ndim == 0:
                t = t.reshape(1)
            elif t.ndim == 1:
                if t.shape[0] != expected:
                    raise ValueError(
                        f"Context '{key}' expects {expected} elements, got shape {tuple(t.shape)}."
                    )
                t = t.unsqueeze(0)
            elif t.ndim == 2:
                if t.shape[1] != expected:
                    raise ValueError(
                        f"Context '{key}' expects feature dimension {expected}, got shape {tuple(t.shape)}."
                    )
            else:
                raise ValueError(
                    f"Context '{key}' must be a scalar, vector, or batched matrix, got ndim={t.ndim}."
                )

            if batch_size is None:
                batch_size = t.shape[0]
            elif t.shape[0] != batch_size:
                raise ValueError("All context keyword arguments must share the same batch size.")

            parts.append(t)

        packed = torch.cat(parts, dim=-1)
        if batch_size == 1:
            return packed.squeeze(0), True
        return packed, False

    def get_params(self, **context) -> dict[str, torch.Tensor]:
        packed, single = self._pack_context(**context)

        if packed.ndim == 1:
            flat_params = self.mlp(packed)
            batch_size = 1
        else:
            flat_params = self.mlp(packed)
            batch_size = packed.shape[0]
            single = False

        params = {}
        idx = 0
        for name, shape in self.param_shapes.items():
            size = self.param_sizes[name]
            chunk = flat_params[..., idx : idx + size].reshape(batch_size, *shape)
            params[name] = chunk.squeeze(0) if single else chunk
            idx += size

        return params

    def coeff_values(self, **context) -> torch.Tensor:
        """Return normalized basis coefficients for single or batched context."""
        params = self.get_params(**context)
        raw = params[_coeff_param_name(self.param_shapes)]
        coeffs_holder = self.target_module.coeffs
        if isinstance(coeffs_holder, PositiveParameters):
            normalize_dim = -1 if raw.dim() > 1 else 0
            if coeffs_holder.is_normalized():
                return coeffs_holder._normalize(raw, dim=normalize_dim)
            return coeffs_holder._epsilon + torch.nn.functional.softplus(raw)
        return raw

    def means_stds(self, **context) -> tuple[torch.Tensor, torch.Tensor]:
        """Decoded mean/std tensors for single or batched context."""
        params = self.get_params(**context)
        mean = params["_params.0._p"]
        std_raw = params["_params.1._p"]
        std_holder = self.target_module._params[1]
        if isinstance(std_holder, PositiveParameters):
            std = std_holder._epsilon + torch.nn.functional.softplus(std_raw)
        else:
            std = std_raw
        return mean, std

    def omega2(
        self,
        other: "MLPMetaForm",
        ignore_coeffs: bool = False,
        lows: torch.Tensor | None = None,
        highs: torch.Tensor | None = None,
        **context: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized Omega2 between this basis and another meta-form basis."""
        self_ctx = {key: context[key] for key in self._context_keys}
        other_ctx = {key: context[key] for key in other._context_keys}
        mu1, std1 = self.means_stds(**self_ctx)
        mu2, std2 = other.means_stds(**other_ctx)

        coeffs1 = None
        coeffs2 = None
        if not ignore_coeffs:
            if self.target_module.has_coeffs():
                coeffs1 = self.coeff_values(**self_ctx)
            if other.target_module.has_coeffs():
                coeffs2 = other.coeff_values(**other_ctx)

        return GaussianBasis.omega2_from_means_stds(
            mu1, std1, mu2, std2,
            ignore_coeffs=ignore_coeffs,
            coeffs1=coeffs1,
            coeffs2=coeffs2,
            lows=lows,
            highs=highs,
        )

    def forward(self, *inputs, **kwargs) -> torch.Tensor:
        """
        Predict parameters from context keyword arguments and run the target module.

        Context must be supplied as keyword arguments (e.g. ``u=..., up=...``), stacked
        in registered key order before the MLP. Remaining keyword arguments are
        forwarded to ``target_module`` after context keys are removed.
        """
        context = {key: kwargs.pop(key) for key in self._context_keys}

        params = self.get_params(**context)
        return functional_call(self.target_module, params, inputs, kwargs)

    def instantiate(self, **context) -> FunctionalModuleProxy:
        context = self._validate_context_kwargs(**context)
        for key, value in context.items():
            if value.ndim > 1:
                raise ValueError(
                    f"instantiate expects a single context for '{key}'. "
                    "Use get_params for batched contexts."
                )

        params = self.get_params(**context)
        buffers = dict(self.target_module.named_buffers())

        return FunctionalModuleProxy(
            target_module=self.target_module,
            params=params,
            buffers=buffers,
            strict=False,
        )
    
    def freeze_params(self) -> "MLPMetaForm":
        """Deep copy with MLP and target frozen for use as a fixed hypernetwork."""
        frozen = deepcopy(self)
        for param in frozen.parameters():
            param.requires_grad_(False)
        return frozen

    @staticmethod
    def target_param_dict_from_basis(basis: nn.Module) -> dict[str, torch.Tensor]:
        """Snapshot basis layout tensors for MSE targets (matches ``param_shapes`` keys)."""
        targets = {name: param.detach().clone() for name, param in basis.named_parameters()}
        for name, module in basis.named_modules():
            if not isinstance(module, Parameters):
                continue
            key = f"{name}._p" if name else "_p"
            if key not in targets:
                targets[key] = module._p.detach().clone()
        return targets
