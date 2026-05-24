import math

import torch
import torch.nn as nn
from torch.func import functional_call
from collections import OrderedDict
from typing import Mapping
from copy import copy, deepcopy
from .domain_transformation import MLP
from .basis_functions import Parameters

def _holder_path_from_param_name(param_name: str) -> str:
    if param_name.endswith("._p"):
        return param_name[: -len("._p")]
    return param_name


def layout_param_shapes(target_module: nn.Module) -> OrderedDict[str, tuple[int, ...]]:
    """
    Shapes of all ``._p`` tensors the hypernetwork predicts for ``target_module``.

    Includes ``Parameters`` holders stored as buffers (``trainable=False``) as well as
    ``nn.Parameter`` entries, so ``functional_call`` can replace every layout slot.
    """
    shapes = OrderedDict((name, tuple(param.shape)) for name, param in target_module.named_parameters())
    for name, module in target_module.named_modules():
        if not isinstance(module, Parameters):
            continue
        key = f"{name}._p" if name else "_p"
        if key not in shapes:
            shapes[key] = tuple(module._p.shape)
    return shapes


def _clone_module(module: nn.Module) -> nn.Module:
    """Clone an ``nn.Module`` tree with detached leaf parameter/buffer tensors."""
    cloned = module.__new__(type(module))
    nn.Module.__init__(cloned)

    for name, child in module.named_children():
        cloned.add_module(name, _clone_module(child))

    for name, param in module._parameters.items():
        if param is not None:
            cloned.register_parameter(
                name,
                nn.Parameter(param.detach().clone(), requires_grad=param.requires_grad),
            )

    for name, buf in module._buffers.items():
        if buf is not None:
            cloned.register_buffer(name, buf.detach().clone())

    skip_names = {
        "_parameters",
        "_buffers",
        "_modules",
        "_backward_hooks",
        "_backward_pre_hooks",
        "_forward_hooks",
        "_forward_hooks_with_kwargs",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "_load_state_dict_pre_hooks",
        "_non_persistent_buffers_set",
    }
    for name, value in vars(module).items():
        if name in skip_names or isinstance(value, nn.Module):
            continue
        if isinstance(value, torch.Tensor):
            setattr(cloned, name, value.detach().clone())
            continue
        if isinstance(value, dict):
            setattr(cloned, name, deepcopy(value))
            continue
        if isinstance(value, (list, tuple)):
            setattr(cloned, name, type(value)(deepcopy(value)))
            continue
        setattr(cloned, name, copy.copy(value))

    return cloned


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

        self.param_shapes = dict(layout_param_shapes(target_module))
        self.param_sizes = {name: math.prod(shape) for name, shape in self.param_shapes.items()}

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

    def decode_params(
        self,
        params: Mapping[str, torch.Tensor] | None = None,
        **context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Apply each layout holder's ``forward`` to raw MLP outputs (same decode as inference).

        Keys match ``get_params`` / ``param_shapes``. Gradients flow through softplus,
        softmax, etc., into the MLP when ``params`` comes from ``get_params``.
        """
        if params is None:
            params = self.get_params(**context)

        decoded: dict[str, torch.Tensor] = {}
        for name in self.param_shapes:
            holder = self.target_module.get_submodule(_holder_path_from_param_name(name))
            decoded[name] = functional_call(holder, {"_p": params[name]}, (), strict=False)
        return decoded

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
        """
        Build a functional proxy for the target module at the given context.

        Context may be scalar or batched (leading batch dimension); batched params
        are passed through ``functional_call`` so target methods (e.g. ``coeff_values``,
        ``Omega2``) run with the correct per-context decode.
        """
        params = self.get_params(**context)
        buffers = dict(self.target_module.named_buffers())

        return FunctionalModuleProxy(
            target_module=self.target_module,
            params=params,
            buffers=buffers,
            strict=False,
        )
    
    def freeze_params(self) -> "MLPMetaForm":
        """Detached copy with all parameters frozen for use as a fixed hypernetwork."""
        frozen = _clone_module(self)
        for param in frozen.parameters():
            param.requires_grad_(False)
        return frozen

    def set_requires_grad(self, requires_grad: bool):
        """
        Toggle gradients on the hypernetwork MLP only.

        ``target_module`` layout tensors are placeholders (overwritten by
        ``functional_call``); they are already frozen in ``__init__`` and may
        include non-leaf storages if misconfigured.
        """
        for param in self.mlp.parameters():
            param.requires_grad_(requires_grad)

    @staticmethod
    def target_param_dict_from_basis(
        basis: nn.Module,
        layout: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Decoded parameter targets for meta-form MSE pretraining (keys match ``layout``).

        Each entry is ``basis`` holder ``forward()`` output (e.g. EM std scales, mixture
        weights), aligned with ``decode_params`` on MLP-predicted raw ``_p`` values.
        """
        layout = layout if layout is not None else basis
        targets: dict[str, torch.Tensor] = {}

        for name in layout_param_shapes(layout):
            holder = basis.get_submodule(_holder_path_from_param_name(name))
            if not isinstance(holder, Parameters):
                raise TypeError(
                    f"Expected Parameters holder for '{name}', got {type(holder).__name__}."
                )
            targets[name] = holder().detach()
        return targets
