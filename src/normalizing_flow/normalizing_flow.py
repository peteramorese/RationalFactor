"""Normalizing-flow density models built on ``nflows``.

Unconditional and conditional variants are provided for several transform
families, ordered roughly by expressivity of the elementwise map:

- **MAF** — masked affine autoregressive (Papamakarios et al., 2017)
- **RealNVP** — affine coupling with alternating masks (Dinh et al., 2017)
- **NSF** — masked rational-quadratic spline autoregressive (Durkan et al., 2019)
- **RQ-coupling** — rational-quadratic spline coupling (Neural Spline Flows)

``NormalizingFlow`` / ``ConditionalNormalizingFlow`` remain aliases for the MAF
variants for backward compatibility.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from nflows.nn.nets import ResidualNet
from nflows.transforms import (
    ActNorm,
    AffineCouplingTransform,
    CompositeTransform,
    LULinear,
    ReversePermutation,
)
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform

from rational_factor.models.density_model import ConditionalDensityModel, DensityModel


def _make_flow(dim: int, transforms: Sequence) -> Flow:
    return Flow(
        transform=CompositeTransform(list(transforms)),
        distribution=StandardNormal(shape=[dim]),
    )


def _maf_transforms(
    dim: int,
    num_layers: int,
    hidden_features: int,
    *,
    context_features: int | None = None,
    num_blocks: int = 2,
) -> list:
    transforms = []
    for _ in range(num_layers):
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=dim,
                hidden_features=hidden_features,
                context_features=context_features,
                num_blocks=num_blocks,
                use_batch_norm=False,
            )
        )
        transforms.append(ReversePermutation(features=dim))
    return transforms


def _nsf_transforms(
    dim: int,
    num_layers: int,
    hidden_features: int,
    *,
    context_features: int | None = None,
    num_blocks: int = 2,
    num_bins: int = 8,
    tail_bound: float = 3.0,
) -> list:
    transforms = []
    for _ in range(num_layers):
        transforms.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=dim,
                hidden_features=hidden_features,
                context_features=context_features,
                num_blocks=num_blocks,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
                use_batch_norm=False,
            )
        )
        transforms.append(ReversePermutation(features=dim))
    return transforms


def _coupling_net_factory(
    hidden_features: int,
    *,
    context_features: int | None = None,
    num_blocks: int = 2,
) -> Callable:
    def create_resnet(in_features: int, out_features: int) -> ResidualNet:
        return ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            use_batch_norm=False,
        )

    return create_resnet


def _checkerboard_mask(dim: int) -> torch.Tensor:
    mask = torch.ones(dim)
    mask[::2] = -1
    return mask


def _realnvp_transforms(
    dim: int,
    num_layers: int,
    hidden_features: int,
    *,
    context_features: int | None = None,
    num_blocks: int = 2,
) -> list:
    if dim < 2:
        raise ValueError(f"RealNVP coupling flows require dim >= 2, got {dim}")

    create_resnet = _coupling_net_factory(
        hidden_features, context_features=context_features, num_blocks=num_blocks
    )
    mask = _checkerboard_mask(dim)
    transforms = []
    for _ in range(num_layers):
        transforms.append(ActNorm(features=dim))
        transforms.append(LULinear(features=dim))
        transforms.append(
            AffineCouplingTransform(mask=mask, transform_net_create_fn=create_resnet)
        )
        mask = -mask
    return transforms


def _rq_coupling_transforms(
    dim: int,
    num_layers: int,
    hidden_features: int,
    *,
    context_features: int | None = None,
    num_blocks: int = 2,
    num_bins: int = 8,
    tail_bound: float = 3.0,
) -> list:
    if dim < 2:
        raise ValueError(f"RQ coupling flows require dim >= 2, got {dim}")

    create_resnet = _coupling_net_factory(
        hidden_features, context_features=context_features, num_blocks=num_blocks
    )
    mask = _checkerboard_mask(dim)
    transforms = []
    for _ in range(num_layers):
        transforms.append(ActNorm(features=dim))
        transforms.append(LULinear(features=dim))
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
    return transforms


def _sample_conditional(
    flow: Flow,
    conditioner: torch.Tensor,
    num_samples_per: int = 1,
) -> torch.Tensor:
    if conditioner.ndim == 1:
        conditioner = conditioner.unsqueeze(0)
    elif conditioner.ndim > 2:
        conditioner = conditioner.view(-1, conditioner.shape[-1])

    x_samples = flow.sample(num_samples=num_samples_per, context=conditioner)
    if num_samples_per == 1:
        x_samples = x_samples[:, 0, :]
    return x_samples


# ---------------------------------------------------------------------------
# MAF (affine autoregressive)
# ---------------------------------------------------------------------------


class MAFNormalizingFlow(DensityModel):
    def __init__(self, dim: int, num_layers: int = 5, hidden_features: int = 64):
        super().__init__(dim=dim)
        self.flow = _make_flow(dim, _maf_transforms(dim, num_layers, hidden_features))

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.flow.log_prob(inputs=x)

    def sample(self, n_samples: int) -> torch.Tensor:
        return self.flow.sample(num_samples=n_samples)


class ConditionalMAFNormalizingFlow(ConditionalDensityModel):
    def __init__(
        self,
        dim: int,
        conditioner_dim: int,
        num_layers: int = 5,
        hidden_features: int = 64,
    ):
        super().__init__(dim=dim, conditioner_dim=conditioner_dim)
        self.flow = _make_flow(
            dim,
            _maf_transforms(
                dim, num_layers, hidden_features, context_features=conditioner_dim
            ),
        )

    def log_density(self, x: torch.Tensor, *, conditioner: torch.Tensor) -> torch.Tensor:
        return self.flow.log_prob(inputs=x, context=conditioner)

    def sample(self, conditioner: torch.Tensor, num_samples_per: int = 1) -> torch.Tensor:
        return _sample_conditional(self.flow, conditioner, num_samples_per)


# ---------------------------------------------------------------------------
# NSF (rational-quadratic spline autoregressive)
# ---------------------------------------------------------------------------


class NSFNormalizingFlow(DensityModel):
    """Masked autoregressive flow with rational-quadratic spline transforms."""

    def __init__(
        self,
        dim: int,
        num_layers: int = 5,
        hidden_features: int = 64,
        num_bins: int = 8,
        tail_bound: float = 3.0,
    ):
        super().__init__(dim=dim)
        self.flow = _make_flow(
            dim,
            _nsf_transforms(
                dim,
                num_layers,
                hidden_features,
                num_bins=num_bins,
                tail_bound=tail_bound,
            ),
        )

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.flow.log_prob(inputs=x)

    def sample(self, n_samples: int) -> torch.Tensor:
        return self.flow.sample(num_samples=n_samples)


class ConditionalNSFNormalizingFlow(ConditionalDensityModel):
    """Conditional NSF with rational-quadratic spline autoregressive transforms."""

    def __init__(
        self,
        dim: int,
        conditioner_dim: int,
        num_layers: int = 5,
        hidden_features: int = 64,
        num_bins: int = 8,
        tail_bound: float = 3.0,
    ):
        super().__init__(dim=dim, conditioner_dim=conditioner_dim)
        self.flow = _make_flow(
            dim,
            _nsf_transforms(
                dim,
                num_layers,
                hidden_features,
                context_features=conditioner_dim,
                num_bins=num_bins,
                tail_bound=tail_bound,
            ),
        )

    def log_density(self, x: torch.Tensor, *, conditioner: torch.Tensor) -> torch.Tensor:
        return self.flow.log_prob(inputs=x, context=conditioner)

    def sample(self, conditioner: torch.Tensor, num_samples_per: int = 1) -> torch.Tensor:
        return _sample_conditional(self.flow, conditioner, num_samples_per)


# ---------------------------------------------------------------------------
# RealNVP (affine coupling)
# ---------------------------------------------------------------------------


class RealNVPNormalizingFlow(DensityModel):
    """RealNVP-style flow: ActNorm → LU → affine coupling, alternating masks."""

    def __init__(self, dim: int, num_layers: int = 5, hidden_features: int = 64):
        super().__init__(dim=dim)
        self.flow = _make_flow(dim, _realnvp_transforms(dim, num_layers, hidden_features))

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.flow.log_prob(inputs=x)

    def sample(self, n_samples: int) -> torch.Tensor:
        return self.flow.sample(num_samples=n_samples)


class ConditionalRealNVPNormalizingFlow(ConditionalDensityModel):
    """Conditional RealNVP with context-conditioned residual coupling nets."""

    def __init__(
        self,
        dim: int,
        conditioner_dim: int,
        num_layers: int = 5,
        hidden_features: int = 64,
    ):
        super().__init__(dim=dim, conditioner_dim=conditioner_dim)
        self.flow = _make_flow(
            dim,
            _realnvp_transforms(
                dim, num_layers, hidden_features, context_features=conditioner_dim
            ),
        )

    def log_density(self, x: torch.Tensor, *, conditioner: torch.Tensor) -> torch.Tensor:
        return self.flow.log_prob(inputs=x, context=conditioner)

    def sample(self, conditioner: torch.Tensor, num_samples_per: int = 1) -> torch.Tensor:
        return _sample_conditional(self.flow, conditioner, num_samples_per)


# ---------------------------------------------------------------------------
# RQ coupling (neural spline coupling)
# ---------------------------------------------------------------------------


class RQCouplingNormalizingFlow(DensityModel):
    """Coupling flow with rational-quadratic splines (NSF coupling variant)."""

    def __init__(
        self,
        dim: int,
        num_layers: int = 5,
        hidden_features: int = 64,
        num_bins: int = 8,
        tail_bound: float = 3.0,
    ):
        super().__init__(dim=dim)
        self.flow = _make_flow(
            dim,
            _rq_coupling_transforms(
                dim,
                num_layers,
                hidden_features,
                num_bins=num_bins,
                tail_bound=tail_bound,
            ),
        )

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.flow.log_prob(inputs=x)

    def sample(self, n_samples: int) -> torch.Tensor:
        return self.flow.sample(num_samples=n_samples)


class ConditionalRQCouplingNormalizingFlow(ConditionalDensityModel):
    """Conditional rational-quadratic coupling flow."""

    def __init__(
        self,
        dim: int,
        conditioner_dim: int,
        num_layers: int = 5,
        hidden_features: int = 64,
        num_bins: int = 8,
        tail_bound: float = 3.0,
    ):
        super().__init__(dim=dim, conditioner_dim=conditioner_dim)
        self.flow = _make_flow(
            dim,
            _rq_coupling_transforms(
                dim,
                num_layers,
                hidden_features,
                context_features=conditioner_dim,
                num_bins=num_bins,
                tail_bound=tail_bound,
            ),
        )

    def log_density(self, x: torch.Tensor, *, conditioner: torch.Tensor) -> torch.Tensor:
        return self.flow.log_prob(inputs=x, context=conditioner)

    def sample(self, conditioner: torch.Tensor, num_samples_per: int = 1) -> torch.Tensor:
        return _sample_conditional(self.flow, conditioner, num_samples_per)


# Backward-compatible aliases (original MAF implementations).
NormalizingFlow = MAFNormalizingFlow
ConditionalNormalizingFlow = ConditionalMAFNormalizingFlow
