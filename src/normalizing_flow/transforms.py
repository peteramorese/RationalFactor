"""Invertible transforms for normalizing flows and domain maps.

All public factories return ``nflows.transforms.base.Transform`` instances.
Use ``Transforms.make_transform(name, **kwargs)`` as the single entry point.
"""

from __future__ import annotations

import copy
from typing import Sequence

import torch
from nflows.nn.nets import ResidualNet
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from nflows.transforms.base import CompositeTransform, Transform
from nflows.transforms.coupling import (
    AffineCouplingTransform,
    PiecewiseRationalQuadraticCouplingTransform,
)
from nflows.transforms.normalization import ActNorm
from nflows.transforms.permutations import RandomPermutation, ReversePermutation
from nflows.transforms.lu import LULinear

from rational_factor.models.mlp import MLP


# ---------------------------------------------------------------------------
# Small / custom transforms
# ---------------------------------------------------------------------------


class IdentityTransform(Transform):
    def __init__(self, features: int):
        super().__init__()
        self.features = features

    def forward(self, inputs: torch.Tensor, context: torch.Tensor | None = None):
        return inputs, inputs.new_zeros(inputs.shape[0])

    def inverse(self, inputs: torch.Tensor, context: torch.Tensor | None = None):
        return inputs, inputs.new_zeros(inputs.shape[0])


class StackedTransform(Transform):
    """Apply child transforms to contiguous feature blocks and concatenate."""

    def __init__(self, transforms: Sequence[Transform]):
        super().__init__()
        if not transforms:
            raise ValueError("StackedTransform requires at least one child")
        feats = []
        for tf in transforms:
            if not hasattr(tf, "features"):
                raise ValueError(
                    f"{type(tf).__name__} must expose `.features` to be stacked"
                )
            feats.append(tf.features)
        self.features = sum(feats)
        self.tfs = torch.nn.ModuleList(list(transforms))

    def forward(self, inputs: torch.Tensor, context: torch.Tensor | None = None):
        z_parts = []
        ladj = inputs.new_zeros(inputs.shape[0])
        cursor = 0
        for tf in self.tfs:
            z_part, ladj_part = tf.forward(
                inputs[:, cursor : cursor + tf.features], context=context
            )
            z_parts.append(z_part)
            ladj = ladj + ladj_part
            cursor += tf.features
        return torch.cat(z_parts, dim=1), ladj

    def inverse(self, inputs: torch.Tensor, context: torch.Tensor | None = None):
        x_parts = []
        ladj = inputs.new_zeros(inputs.shape[0])
        cursor = 0
        for tf in self.tfs:
            x_part, ladj_part = tf.inverse(
                inputs[:, cursor : cursor + tf.features], context=context
            )
            x_parts.append(x_part)
            ladj = ladj + ladj_part
            cursor += tf.features
        return torch.cat(x_parts, dim=1), ladj


class ErfSeparableTransform(Transform):
    """Per-dimension Gaussian CDF: ``z_d = Phi((x_d - loc_d) / scale_d)``."""

    def __init__(
        self,
        features: int,
        loc: torch.Tensor,
        scale: torch.Tensor,
        trainable: bool = True,
        min_scale: float = 1e-3,
        numerical_tolerance: float = 1e-20,
    ):
        super().__init__()
        self.features = features
        self.trainable = trainable
        self.min_scale = min_scale
        self.numerical_tolerance = numerical_tolerance

        scale = torch.as_tensor(scale, dtype=loc.dtype, device=loc.device).clamp(
            min=min_scale
        )
        if trainable:
            scale_params = torch.sqrt(scale)
            self.params = torch.nn.Parameter(
                torch.hstack([loc.unsqueeze(1), scale_params.unsqueeze(1)])
            )
        else:
            self.register_buffer(
                "params", torch.hstack([loc.unsqueeze(1), scale.unsqueeze(1)])
            )

    @classmethod
    def from_data(
        cls,
        x_data: torch.Tensor,
        trainable: bool = True,
        min_scale: float = 1e-3,
    ) -> ErfSeparableTransform:
        mean = x_data.mean(dim=0)
        std = x_data.std(dim=0).clamp_min(min_scale)
        return cls(x_data.shape[1], mean, std, trainable=trainable, min_scale=min_scale)  # features

    @classmethod
    def copy_from_trainable(cls, other: ErfSeparableTransform) -> ErfSeparableTransform:
        loc, scale = other.loc_scale()
        return cls(
            other.features,
            loc.detach().clone(),
            scale.detach().clone(),
            trainable=False,
            min_scale=other.min_scale,
            numerical_tolerance=other.numerical_tolerance,
        )

    def loc_scale(self):
        if self.trainable:
            loc = self.params[:, 0]
            scale = torch.square(self.params[:, 1])
        else:
            loc = self.params[:, 0]
            scale = self.params[:, 1]
        return loc, torch.clamp(scale, min=self.min_scale)

    def forward(self, inputs: torch.Tensor, context: torch.Tensor | None = None):
        loc, scale = self.loc_scale()
        sqrt_2 = torch.sqrt(inputs.new_tensor(2.0))
        u = (inputs - loc) / (scale * sqrt_2)
        z = 0.5 * (1.0 + torch.special.erf(u))
        z = z.clamp(self.numerical_tolerance, 1.0 - self.numerical_tolerance)
        ladj = (
            -torch.log(scale)
            - 0.5 * torch.log(inputs.new_tensor(2.0 * torch.pi))
            - u**2
        ).sum(dim=-1)
        return z, ladj

    def inverse(self, inputs: torch.Tensor, context: torch.Tensor | None = None):
        loc, scale = self.loc_scale()
        sqrt_2 = torch.sqrt(inputs.new_tensor(2.0))
        u = torch.special.erfinv(
            2.0
            * inputs.clamp(self.numerical_tolerance, 1.0 - self.numerical_tolerance)
            - 1.0
        )
        x = loc + scale * sqrt_2 * u
        ladj = (
            torch.log(scale)
            + 0.5 * (torch.log(inputs.new_tensor(2.0 * torch.pi)) + u**2)
        ).sum(dim=-1)
        return x, ladj


class ClampInputsTransform(Transform):
    """Clamp into a closed interval; log-det contribution is zero."""

    def __init__(self, left: float = 0.0, right: float = 1.0, eps: float = 0.0):
        super().__init__()
        if not left < right:
            raise ValueError(f"Need left < right, got left={left}, right={right}")
        if eps < 0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        if 2.0 * eps >= right - left:
            raise ValueError(f"eps={eps} too large for interval [{left}, {right}]")
        self.left = float(left + eps)
        self.right = float(right - eps)

    def forward(self, inputs, context=None):
        return inputs.clamp(self.left, self.right), inputs.new_zeros(inputs.shape[0])

    def inverse(self, inputs, context=None):
        return inputs.clamp(self.left, self.right), inputs.new_zeros(inputs.shape[0])


class AdditiveCoupling(Transform):
    """NICE-style additive coupling; ``log |det J| = 0``."""

    def __init__(
        self,
        features: int,
        mask: torch.Tensor,
        hidden_features: int = 128,
        num_hidden_layers: int = 2,
        activation=torch.nn.Tanh,
        zero_init: bool = True,
    ):
        super().__init__()
        self.features = features
        self.register_buffer("mask", mask.float())
        self.register_buffer("inv_mask", 1.0 - mask.float())
        self.shift_net = MLP(
            in_features=features,
            out_features=features,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
            zero_init_last=zero_init,
        )

    def forward(self, inputs, context=None):
        shift = self.shift_net(inputs * self.mask) * self.inv_mask
        return inputs + shift, inputs.new_zeros(inputs.shape[0])

    def inverse(self, inputs, context=None):
        shift = self.shift_net(inputs * self.mask) * self.inv_mask
        return inputs - shift, inputs.new_zeros(inputs.shape[0])


class HouseholderTransform(Transform):
    """Product of Householder reflections; ``|det J| = 1``."""

    def __init__(self, features: int, num_reflections: int = 4, eps: float = 1e-8):
        super().__init__()
        self.features = features
        self.num_reflections = num_reflections
        self.eps = eps
        self.vectors = torch.nn.Parameter(
            torch.randn(num_reflections, features) / features**0.5
        )

    def _apply_reflection(self, x, v):
        denom = torch.sum(v * v).clamp_min(self.eps)
        projection = (x @ v)[:, None] * v[None, :] / denom
        return x - 2.0 * projection

    def forward(self, inputs, context=None):
        outputs = inputs
        for k in range(self.num_reflections):
            outputs = self._apply_reflection(outputs, self.vectors[k])
        return outputs, inputs.new_zeros(inputs.shape[0])

    def inverse(self, inputs, context=None):
        outputs = inputs
        for k in reversed(range(self.num_reflections)):
            outputs = self._apply_reflection(outputs, self.vectors[k])
        return outputs, inputs.new_zeros(inputs.shape[0])


class DimCompositeTransform(CompositeTransform):
    """``CompositeTransform`` that records ``features`` / optional ``context_features``."""

    def __init__(
        self,
        transforms: Sequence[Transform],
        features: int,
        context_features: int | None = None,
    ):
        super().__init__(list(transforms))
        self.features = features
        self.context_features = context_features


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class Transforms:
    """Static factory for named invertible transforms."""

    @staticmethod
    def make_transform(name: str, features: int, context_features: int | None = None, **kwargs) -> Transform:
        key = name.lower().replace("-", "_")
        builders = {
            "identity": Transforms.identity,
            "stacked": Transforms.stacked,
            "erf": Transforms.erf,
            "erf_separable": Transforms.erf,
            "maf": Transforms.maf,
            "nsf": Transforms.nsf,
            "masked_rqs": Transforms.nsf,
            "realnvp": Transforms.realnvp,
            "rq_coupling": Transforms.rq_coupling,
            "volume_preserving": Transforms.volume_preserving,
            "vp": Transforms.volume_preserving,
            "householder": Transforms.householder,
            "additive_coupling": Transforms.additive_coupling,
            "clamp": Transforms.clamp,
        }
        if key not in builders:
            raise ValueError(
                f"Unknown transform {name!r}. Choose from: {sorted(builders)}"
            )
        return builders[key](features=features, context_features=context_features, **kwargs)

    @staticmethod
    def freeze(transform: Transform) -> Transform:
        """Deep-copy a transform and disable gradients."""
        if isinstance(transform, ErfSeparableTransform):
            return ErfSeparableTransform.copy_from_trainable(transform)
        frozen = copy.deepcopy(transform)
        for p in frozen.parameters():
            p.requires_grad_(False)
        return frozen

    # -- atomic -------------------------------------------------------------

    @staticmethod
    def identity(*, features: int, **_unused) -> IdentityTransform:
        return IdentityTransform(features)

    @staticmethod
    def stacked(*, transforms: Sequence[Transform], **_unused) -> StackedTransform:
        return StackedTransform(transforms)

    @staticmethod
    def erf(
        *,
        features: int | None = None,
        loc: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        x_data: torch.Tensor | None = None,
        trainable: bool = True,
        min_scale: float = 1e-3,
        numerical_tolerance: float = 1e-20,
        **_unused,
    ) -> ErfSeparableTransform:
        if x_data is not None:
            return ErfSeparableTransform.from_data(
                x_data, trainable=trainable, min_scale=min_scale
            )
        if loc is None or scale is None:
            raise ValueError("erf requires loc/scale, or x_data")
        if features is None:
            features = loc.shape[0]
        return ErfSeparableTransform(
            features,
            loc,
            scale,
            trainable=trainable,
            min_scale=min_scale,
            numerical_tolerance=numerical_tolerance,
        )

    @staticmethod
    def clamp(
        *,
        left: float = 0.0,
        right: float = 1.0,
        eps: float = 0.0,
        **_unused,
    ) -> ClampInputsTransform:
        return ClampInputsTransform(left=left, right=right, eps=eps)

    @staticmethod
    def householder(
        *,
        features: int,
        num_reflections: int = 4,
        eps: float = 1e-8,
        **_unused,
    ) -> HouseholderTransform:
        return HouseholderTransform(features=features, num_reflections=num_reflections, eps=eps)

    @staticmethod
    def additive_coupling(
        *,
        features: int,
        mask: torch.Tensor,
        hidden_features: int = 128,
        num_hidden_layers: int = 2,
        activation=torch.nn.Tanh,
        zero_init: bool = True,
        **_unused,
    ) -> AdditiveCoupling:
        return AdditiveCoupling(
            features=features,
            mask=mask,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
            zero_init=zero_init,
        )

    # -- flow stacks --------------------------------------------------------

    @staticmethod
    def maf(
        *,
        features: int,
        num_layers: int = 5,
        hidden_features: int = 64,
        context_features: int | None = None,
        num_blocks: int = 2,
        use_residual_blocks: bool = True,
        permutation: str = "reverse",
        init_identity: bool = False,
        trainable: bool = True,
        **_unused,
    ) -> DimCompositeTransform:
        transforms: list[Transform] = []
        for _ in range(num_layers):
            transforms.append(_permutation(features, permutation))
            maf = MaskedAffineAutoregressiveTransform(
                features=features,
                hidden_features=hidden_features,
                context_features=context_features,
                num_blocks=num_blocks,
                use_residual_blocks=use_residual_blocks,
                random_mask=False,
                activation=torch.tanh,
                dropout_probability=0.0,
                use_batch_norm=False,
            )
            if init_identity:
                _zero_init_last_linear(maf)
            transforms.append(maf)
        out = DimCompositeTransform(transforms, features, context_features)
        if not trainable:
            for p in out.parameters():
                p.requires_grad_(False)
        return out

    @staticmethod
    def nsf(
        *,
        features: int,
        num_layers: int = 5,
        hidden_features: int = 64,
        context_features: int | None = None,
        num_blocks: int = 2,
        num_bins: int = 8,
        tails: str | None = "linear",
        tail_bound: float = 3.0,
        domain_eps: float = 0.0,
        use_residual_blocks: bool = True,
        permutation: str = "reverse",
        trainable: bool = True,
        **_unused,
    ) -> DimCompositeTransform:
        bounded = tails is None
        transforms: list[Transform] = []
        for _ in range(num_layers):
            transforms.append(_permutation(features, permutation))
            if bounded:
                transforms.append(ClampInputsTransform(0.0, 1.0, eps=domain_eps))
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=features,
                    hidden_features=hidden_features,
                    context_features=context_features,
                    num_bins=num_bins,
                    tails=tails,
                    tail_bound=tail_bound,
                    num_blocks=num_blocks,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=False,
                    activation=torch.tanh,
                    dropout_probability=0.0,
                    use_batch_norm=False,
                )
            )
        out = DimCompositeTransform(transforms, features, context_features)
        if not trainable:
            for p in out.parameters():
                p.requires_grad_(False)
        return out

    @staticmethod
    def realnvp(
        *,
        features: int,
        num_layers: int = 5,
        hidden_features: int = 64,
        context_features: int | None = None,
        num_blocks: int = 2,
        trainable: bool = True,
        **_unused,
    ) -> DimCompositeTransform:
        if features < 2:
            raise ValueError(f"RealNVP coupling flows require features >= 2, got {features}")

        def create_resnet(in_features: int, out_features: int) -> ResidualNet:
            return ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                context_features=context_features,
                num_blocks=num_blocks,
                use_batch_norm=False,
            )

        mask = torch.ones(features)
        mask[::2] = -1
        transforms: list[Transform] = []
        for _ in range(num_layers):
            transforms.append(ActNorm(features=features))
            transforms.append(LULinear(features=features))
            transforms.append(
                AffineCouplingTransform(mask=mask, transform_net_create_fn=create_resnet)
            )
            mask = -mask
        out = DimCompositeTransform(transforms, features, context_features)
        if not trainable:
            for p in out.parameters():
                p.requires_grad_(False)
        return out

    @staticmethod
    def rq_coupling(
        *,
        features: int,
        num_layers: int = 5,
        hidden_features: int = 64,
        context_features: int | None = None,
        num_blocks: int = 2,
        num_bins: int = 8,
        tail_bound: float = 3.0,
        trainable: bool = True,
        **_unused,
    ) -> DimCompositeTransform:
        if features < 2:
            raise ValueError(f"RQ coupling flows require features >= 2, got {features}")

        def create_resnet(in_features: int, out_features: int) -> ResidualNet:
            return ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                context_features=context_features,
                num_blocks=num_blocks,
                use_batch_norm=False,
            )

        mask = torch.ones(features)
        mask[::2] = -1
        transforms: list[Transform] = []
        for _ in range(num_layers):
            transforms.append(ActNorm(features=features))
            transforms.append(LULinear(features=features))
            transforms.append(
                PiecewiseRationalQuadraticCouplingTransform(
                    mask=mask,
                    transform_net_create_fn=create_resnet,
                    num_bins=num_bins,
                    tails="linear",
                    tail_bound=tail_bound,
                )
            )
            mask = -mask
        out = DimCompositeTransform(transforms, features, context_features)
        if not trainable:
            for p in out.parameters():
                p.requires_grad_(False)
        return out

    @staticmethod
    def volume_preserving(
        *,
        features: int,
        num_layers: int = 6,
        hidden_features: int = 128,
        num_hidden_layers: int = 2,
        num_householder_reflections: int = 4,
        use_random_permutation: bool = False,
        zero_init: bool = True,
        trainable: bool = True,
        **_unused,
    ) -> DimCompositeTransform:
        base_mask = (torch.arange(features) % 2).float()
        transforms: list[Transform] = []
        for layer_idx in range(num_layers):
            mask = base_mask if layer_idx % 2 == 0 else 1.0 - base_mask
            transforms.append(
                AdditiveCoupling(
                    features=features,
                    mask=mask,
                    hidden_features=hidden_features,
                    num_hidden_layers=num_hidden_layers,
                    activation=torch.nn.Tanh,
                    zero_init=zero_init,
                )
            )
            transforms.append(
                HouseholderTransform(
                    features=features, num_reflections=num_householder_reflections
                )
            )
            if use_random_permutation:
                transforms.append(RandomPermutation(features=features))
        out = DimCompositeTransform(transforms, features)
        if not trainable:
            for p in out.parameters():
                p.requires_grad_(False)
        return out


def _permutation(features: int, kind: str) -> Transform:
    kind = kind.lower()
    if kind == "reverse":
        return ReversePermutation(features=features)
    if kind == "random":
        return RandomPermutation(features=features)
    raise ValueError(f"Unknown permutation {kind!r}; use 'reverse' or 'random'")


def _zero_init_last_linear(module: torch.nn.Module) -> None:
    last_linear = None
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            last_linear = m
    if last_linear is None:
        raise RuntimeError("Could not find final Linear layer")
    torch.nn.init.zeros_(last_linear.weight)
    torch.nn.init.zeros_(last_linear.bias)
