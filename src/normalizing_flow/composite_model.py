"""Change-of-variables density models with pluggable base densities.

``CompositeDensityModel`` / ``CompositeConditionalModel`` push data through an
invertible transform from ``Transforms.make_transform``, then evaluate a user-
supplied base density in latent space:

    log p(x) = log p_base(T(x)) + log |det J_T(x)|

Unlike ``nflows.Flow``, the base is any ``DensityModel`` /
``ConditionalDensityModel`` (e.g. ``StandardNormalDensity``, ``SeparableBeta``).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from nflows.transforms.base import CompositeTransform, Transform

from normalizing_flow.base_distributions import StandardNormalDensity
from normalizing_flow.transforms import Transforms
from rational_factor.models.density_model import ConditionalDensityModel, DensityModel


class CompositeDensityModel(DensityModel):
    """Unconditional density ``p(x) = p_base(T(x)) |det J_T(x)|``."""

    def __init__(
        self,
        base_density: DensityModel,
        transform: str | Transform | Sequence[Transform] = "maf",
        **transform_kwargs,
    ):
        super().__init__(features=base_density.features)
        self.base_density = base_density
        self.transform = Transforms.make_transform(
            transform, features=base_density.features, **transform_kwargs
        )

    def log_density(self, x: torch.Tensor, **contexts: torch.Tensor) -> torch.Tensor:
        z, ladj = self.transform(x)
        return self._clip_log_density(
            self.base_density.log_density(z, **contexts) + ladj
        )

    def sample(self, n_samples: int, **contexts: torch.Tensor) -> torch.Tensor:
        z = self.base_density.sample(n_samples, **contexts)
        x, _ = self.transform.inverse(z)
        return x

    def valid(self):
        return self.base_density.valid()

    def dtype_device(self):
        return self.base_density.dtype_device()

    def supremum_bound(self):
        raise NotImplementedError(
            "supremum_bound requires a bound on |det J|; not available in general"
        )


class CompositeConditionalModel(ConditionalDensityModel):
    """Conditional density ``p(x|c) = p_base(T(x;c)|c) |det J_{T(·;c)}(x)|``.

    ``base_density`` may be unconditional (ignores ``c``) or conditional.
    The transform always receives ``c`` as ``context``.
    """

    def __init__(
        self,
        base_density: DensityModel | ConditionalDensityModel,
        context_features: int,
        transform: str | Transform | Sequence[Transform] = "maf",
        **transform_kwargs,
    ):
        super().__init__(features=base_density.features, context_features=context_features)
        if isinstance(base_density, ConditionalDensityModel):
            if base_density.context_features != context_features:
                raise ValueError(
                    f"base context_features {base_density.context_features} must match "
                    f"context_features {context_features}"
                )
        self.base_density = base_density
        self.transform = Transforms.make_transform(
            transform,
            features=base_density.features,
            context_features=context_features,
            **transform_kwargs,
        )

    def log_density(
        self, x: torch.Tensor, *, conditioner: torch.Tensor, **contexts: torch.Tensor
    ) -> torch.Tensor:
        z, ladj = self.transform(x, context=conditioner)
        if isinstance(self.base_density, ConditionalDensityModel):
            log_base = self.base_density.log_density(
                z, conditioner=conditioner, **contexts
            )
        else:
            log_base = self.base_density.log_density(z, **contexts)
        return self._clip_log_density(log_base + ladj)

    def sample(
        self,
        conditioner: torch.Tensor,
        num_samples_per: int = 1,
        **contexts: torch.Tensor,
    ) -> torch.Tensor:
        if conditioner.ndim == 1:
            conditioner = conditioner.unsqueeze(0)
        elif conditioner.ndim > 2:
            conditioner = conditioner.view(-1, conditioner.shape[-1])

        n_cond = conditioner.shape[0]
        if num_samples_per > 1:
            conditioner = conditioner.repeat_interleave(num_samples_per, dim=0)

        if isinstance(self.base_density, ConditionalDensityModel):
            z = self.base_density.sample(conditioner, **contexts)
        else:
            z = self.base_density.sample(conditioner.shape[0], **contexts)

        x, _ = self.transform.inverse(z, context=conditioner)
        if num_samples_per > 1:
            x = x.view(n_cond, num_samples_per, self.features)
        return x

    def valid(self):
        return self.base_density.valid()

    def dtype_device(self):
        return self.base_density.dtype_device()

    def supremum_bound(self, conditioner: torch.Tensor | None = None):
        raise NotImplementedError(
            "supremum_bound requires a bound on |det J|; not available in general"
        )
