"""Volume-preserving normalizing-flow density models.

Two families are provided:

- **Unit-box CNF** — continuous normalizing flow on ``[0, 1]^d`` whose velocity
  is divergence-free and tangent to the boundary, so trajectories stay in the
  cube and ``log |det J| = 0``. Paired with a unit-box base (default:
  ``SeparableBeta(1, 1)``).
- **Additive coupling (NICE)** — exact volume-preserving discrete flow on
  ``R^d``: additive coupling layers interleaved with Householder orthogonal
  mixings, paired with a standard normal base.

Because every map has ``|det J| = 1``, ``supremum_bound()`` equals the base
density's supremum. Unconditional and conditional variants inherit
``DensityModel`` / ``ConditionalDensityModel``.
"""

from __future__ import annotations

import copy
import math

import torch
from nflows.nn.nets import ResidualNet
from nflows.transforms import CompositeTransform
from nflows.transforms.coupling import AdditiveCouplingTransform

from normalizing_flow.base_distributions import SeparableBeta, StandardNormalDensity
from rational_factor.models.density_model import ConditionalDensityModel, DensityModel
from rational_factor.models.domain_transformation import HouseholderTransform, MLP


class DivergenceFreeBoxVelocity(torch.nn.Module):
    """Divergence-free velocity on [0, 1]^d with zero flux through the boundary."""

    def __init__(
        self,
        dim: int,
        conditioner_dim: int = 0,
        hidden_features: int = 64,
        num_hidden_layers: int = 2,
        time_dependent: bool = True,
        zero_init: bool = True,
    ):
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be at least 1")
        if conditioner_dim < 0:
            raise ValueError("conditioner_dim must be nonnegative")

        self.dim = dim
        self.conditioner_dim = conditioner_dim
        self.time_dependent = time_dependent

        in_features = dim + conditioner_dim + (1 if time_dependent else 0)
        self.net = MLP(
            in_features=in_features,
            out_features=dim * dim,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            activation=torch.nn.Tanh,
            zero_init_last=zero_init,
        )

    def _embed(
        self,
        x: torch.Tensor,
        t: torch.Tensor | float,
        conditioner: torch.Tensor | None,
    ) -> torch.Tensor:
        parts = [x]
        if self.conditioner_dim > 0:
            if conditioner is None:
                raise ValueError("conditioner is required when conditioner_dim > 0")
            parts.append(conditioner)
        if self.time_dependent:
            t = torch.as_tensor(t, dtype=x.dtype, device=x.device).reshape(-1, 1)
            if t.shape[0] == 1:
                t = t.expand(x.shape[0], 1)
            parts.append(t)
        return torch.cat(parts, dim=-1)

    def skew_potential(
        self,
        x: torch.Tensor,
        t: torch.Tensor | float,
        conditioner: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Skew matrix Ψ(x, t, c) that vanishes on every pair of coordinate faces."""
        raw = self.net(self._embed(x, t, conditioner)).view(-1, self.dim, self.dim)
        A = raw - raw.transpose(-1, -2)
        h = x * (1.0 - x)
        return A * h.unsqueeze(-1) * h.unsqueeze(-2)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | float,
        conditioner: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate v(x, t, c). x has shape (batch, dim)."""
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"x must have shape (batch, {self.dim}), got {tuple(x.shape)}")

        def potential(x_):
            return self.skew_potential(x_, t, conditioner)

        v = x.new_zeros(x.shape)
        for j in range(self.dim):
            tangent = torch.zeros_like(x)
            tangent[:, j] = 1.0
            _, dPsi = torch.func.jvp(potential, (x,), (tangent,))
            v = v + dPsi[:, :, j]
        return v


class UnitBoxVolumePreservingFlow(DensityModel):
    """Unconditional volume-preserving CNF density on the unit box.

    ``transform`` integrates t: 0 → 1 (physical → latent).
    ``inverse_transform`` integrates t: 1 → 0 (latent → physical).
    Both have ``log |det J_x| = 0``. Density is ``p_0(T(x))`` for base ``p_0``.
    """

    def __init__(
        self,
        dim: int,
        n_steps: int = 16,
        hidden_features: int = 64,
        num_hidden_layers: int = 2,
        time_dependent: bool = True,
        zero_init: bool = True,
        base: DensityModel | None = None,
        trainable: bool = True,
    ):
        super().__init__(dim=dim)
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        if base is None:
            base = SeparableBeta(dim=dim, alpha=1.0, beta=1.0)
        elif not isinstance(base, DensityModel):
            raise TypeError("base must be a DensityModel")
        elif base.dim != dim:
            raise ValueError("base.dim must match flow dim")

        self.n_steps = n_steps
        self.base = base
        self.velocity_field = DivergenceFreeBoxVelocity(
            dim=dim,
            conditioner_dim=0,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            time_dependent=time_dependent,
            zero_init=zero_init,
        )

        if not trainable:
            for p in self.parameters():
                p.requires_grad_(False)

    @classmethod
    def copy_from_trainable(cls, other: "UnitBoxVolumePreservingFlow"):
        new_module = copy.deepcopy(other)
        for p in new_module.parameters():
            p.requires_grad_(False)
        return new_module

    def _rk4(self, x: torch.Tensor, t0: float, t1: float) -> torch.Tensor:
        dt = (t1 - t0) / self.n_steps
        t = x.new_tensor(t0)
        for _ in range(self.n_steps):
            k1 = self.velocity_field(x, t, None)
            k2 = self.velocity_field(x + 0.5 * dt * k1, t + 0.5 * dt, None)
            k3 = self.velocity_field(x + 0.5 * dt * k2, t + 0.5 * dt, None)
            k4 = self.velocity_field(x + dt * k3, t + dt, None)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t = t + dt
        return x

    def velocity(self, x: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
        return self.velocity_field(x, t, None)

    def transform(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert x.shape[1] == self.dim, "x must have shape (n_data, dim)"
        if self.dim == 1:
            return x, x.new_zeros(x.shape[0])
        return self._rk4(x, 0.0, 1.0), x.new_zeros(x.shape[0])

    def inverse_transform(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert z.shape[1] == self.dim, "z must have shape (n_data, dim)"
        if self.dim == 1:
            return z, z.new_zeros(z.shape[0])
        return self._rk4(z, 1.0, 0.0), z.new_zeros(z.shape[0])

    def log_density(self, x: torch.Tensor, **contexts: torch.Tensor) -> torch.Tensor:
        z, ladj = self.transform(x)
        return self._clip_log_density(self.base.log_density(z) + ladj)

    def sample(self, n_samples: int, **contexts: torch.Tensor) -> torch.Tensor:
        z = self.base.sample(n_samples)
        x, _ = self.inverse_transform(z)
        return x

    def supremum_bound(self) -> torch.Tensor:
        return self.base.supremum_bound()

    def dtype_device(self):
        p = next(self.parameters())
        return p.dtype, p.device


class ConditionalUnitBoxVolumePreservingFlow(ConditionalDensityModel):
    """Conditional volume-preserving CNF density on the unit box.

    The conditioner is held fixed along each ODE trajectory. ``transform`` /
    ``inverse_transform`` expose the volume-preserving map used by pair bases.
    """

    def __init__(
        self,
        dim: int,
        conditioner_dim: int,
        n_steps: int = 16,
        hidden_features: int = 64,
        num_hidden_layers: int = 2,
        time_dependent: bool = True,
        zero_init: bool = True,
        base: DensityModel | None = None,
        trainable: bool = True,
    ):
        super().__init__(dim=dim, conditioner_dim=conditioner_dim)
        if conditioner_dim < 1:
            raise ValueError("conditioner_dim must be at least 1")
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        if base is None:
            base = SeparableBeta(dim=dim, alpha=1.0, beta=1.0)
        elif not isinstance(base, DensityModel):
            raise TypeError("base must be a DensityModel")
        elif base.dim != dim:
            raise ValueError("base.dim must match flow dim")

        self.n_steps = n_steps
        self.base = base
        self.velocity_field = DivergenceFreeBoxVelocity(
            dim=dim,
            conditioner_dim=conditioner_dim,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            time_dependent=time_dependent,
            zero_init=zero_init,
        )

        if not trainable:
            for p in self.parameters():
                p.requires_grad_(False)

    @classmethod
    def copy_from_trainable(cls, other: "ConditionalUnitBoxVolumePreservingFlow"):
        new_module = copy.deepcopy(other)
        for p in new_module.parameters():
            p.requires_grad_(False)
        return new_module

    def _prepare_conditioner(self, x: torch.Tensor, conditioner: torch.Tensor) -> torch.Tensor:
        if conditioner.ndim == 1:
            conditioner = conditioner.unsqueeze(0)
        elif conditioner.ndim != 2:
            raise ValueError(
                f"conditioner must have shape ({self.conditioner_dim},) or "
                f"(batch, {self.conditioner_dim}), got {tuple(conditioner.shape)}"
            )
        if conditioner.shape[-1] != self.conditioner_dim:
            raise ValueError(
                f"conditioner must have {self.conditioner_dim} features, "
                f"got {conditioner.shape[-1]}"
            )
        if conditioner.shape[0] == 1 and x.shape[0] > 1:
            conditioner = conditioner.expand(x.shape[0], -1)
        if conditioner.shape[0] != x.shape[0]:
            raise ValueError(
                f"conditioner batch {conditioner.shape[0]} does not match x batch {x.shape[0]}"
            )
        return conditioner.to(dtype=x.dtype, device=x.device)

    def _rk4(
        self,
        x: torch.Tensor,
        t0: float,
        t1: float,
        conditioner: torch.Tensor,
    ) -> torch.Tensor:
        dt = (t1 - t0) / self.n_steps
        t = x.new_tensor(t0)
        for _ in range(self.n_steps):
            k1 = self.velocity_field(x, t, conditioner)
            k2 = self.velocity_field(x + 0.5 * dt * k1, t + 0.5 * dt, conditioner)
            k3 = self.velocity_field(x + 0.5 * dt * k2, t + 0.5 * dt, conditioner)
            k4 = self.velocity_field(x + dt * k3, t + dt, conditioner)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t = t + dt
        return x

    def velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor | float,
        *,
        conditioner: torch.Tensor,
    ) -> torch.Tensor:
        conditioner = self._prepare_conditioner(x, conditioner)
        return self.velocity_field(x, t, conditioner)

    def transform(
        self,
        x: torch.Tensor,
        *,
        conditioner: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert x.shape[1] == self.dim, "x must have shape (n_data, dim)"
        if self.dim == 1:
            return x, x.new_zeros(x.shape[0])
        conditioner = self._prepare_conditioner(x, conditioner)
        return self._rk4(x, 0.0, 1.0, conditioner), x.new_zeros(x.shape[0])

    def inverse_transform(
        self,
        z: torch.Tensor,
        *,
        conditioner: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert z.shape[1] == self.dim, "z must have shape (n_data, dim)"
        if self.dim == 1:
            return z, z.new_zeros(z.shape[0])
        conditioner = self._prepare_conditioner(z, conditioner)
        return self._rk4(z, 1.0, 0.0, conditioner), z.new_zeros(z.shape[0])

    def log_density(self, x: torch.Tensor, *, conditioner: torch.Tensor, **contexts) -> torch.Tensor:
        z, ladj = self.transform(x, conditioner=conditioner)
        return self._clip_log_density(self.base.log_density(z) + ladj)

    def sample(self, conditioner: torch.Tensor, num_samples_per: int = 1, **contexts) -> torch.Tensor:
        if conditioner.ndim == 1:
            conditioner = conditioner.unsqueeze(0)
        n_ctx = conditioner.shape[0]
        z = self.base.sample(n_ctx * num_samples_per)
        c = conditioner.repeat_interleave(num_samples_per, dim=0)
        x, _ = self.inverse_transform(z, conditioner=c)
        if num_samples_per == 1:
            return x
        return x.view(n_ctx, num_samples_per, self.dim)

    def supremum_bound(self) -> torch.Tensor:
        return self.base.supremum_bound()

    def dtype_device(self):
        p = next(self.parameters())
        return p.dtype, p.device


class VolumePreservingFlow(DensityModel):
    """NICE-style volume-preserving flow on ``R^d`` (additive coupling + Householder)."""

    def __init__(
        self,
        dim: int,
        num_layers: int = 5,
        hidden_features: int = 64,
        num_householder_reflections: int = 4,
        base: DensityModel | None = None,
    ):
        super().__init__(dim=dim)
        if dim < 2:
            raise ValueError(f"Additive coupling VP flows require dim >= 2, got {dim}")
        if base is None:
            base = StandardNormalDensity(dim=dim)
        elif not isinstance(base, DensityModel):
            raise TypeError("base must be a DensityModel")
        elif base.dim != dim:
            raise ValueError("base.dim must match flow dim")

        self.base = base

        def create_resnet(in_features: int, out_features: int) -> ResidualNet:
            return ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                context_features=None,
                num_blocks=2,
                use_batch_norm=False,
            )

        mask = torch.ones(dim)
        mask[::2] = -1
        transforms = []
        for _ in range(num_layers):
            transforms.append(
                AdditiveCouplingTransform(mask=mask, transform_net_create_fn=create_resnet)
            )
            transforms.append(
                HouseholderTransform(features=dim, num_reflections=num_householder_reflections)
            )
            mask = -mask
        self._transform = CompositeTransform(transforms)

    def transform(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._transform(x, context=None)

    def inverse_transform(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._transform.inverse(z, context=None)

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        z, ladj = self.transform(x)
        return self._clip_log_density(self.base.log_density(z) + ladj)

    def sample(self, n_samples: int) -> torch.Tensor:
        z = self.base.sample(n_samples)
        x, _ = self.inverse_transform(z)
        return x

    def supremum_bound(self) -> torch.Tensor:
        return self.base.supremum_bound()

    def dtype_device(self):
        p = next(self.parameters())
        return p.dtype, p.device


class ConditionalVolumePreservingFlow(ConditionalDensityModel):
    """Conditional NICE-style volume-preserving flow on ``R^d``."""

    def __init__(
        self,
        dim: int,
        conditioner_dim: int,
        num_layers: int = 5,
        hidden_features: int = 64,
        num_householder_reflections: int = 4,
        base: DensityModel | None = None,
    ):
        super().__init__(dim=dim, conditioner_dim=conditioner_dim)
        if dim < 2:
            raise ValueError(f"Additive coupling VP flows require dim >= 2, got {dim}")
        if base is None:
            base = StandardNormalDensity(dim=dim)
        elif not isinstance(base, DensityModel):
            raise TypeError("base must be a DensityModel")
        elif base.dim != dim:
            raise ValueError("base.dim must match flow dim")

        self.base = base

        def create_resnet(in_features: int, out_features: int) -> ResidualNet:
            return ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                context_features=conditioner_dim,
                num_blocks=2,
                use_batch_norm=False,
            )

        mask = torch.ones(dim)
        mask[::2] = -1
        transforms = []
        for _ in range(num_layers):
            transforms.append(
                AdditiveCouplingTransform(mask=mask, transform_net_create_fn=create_resnet)
            )
            transforms.append(
                HouseholderTransform(features=dim, num_reflections=num_householder_reflections)
            )
            mask = -mask
        self._transform = CompositeTransform(transforms)

    def transform(
        self,
        x: torch.Tensor,
        *,
        conditioner: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._transform(x, context=conditioner)

    def inverse_transform(
        self,
        z: torch.Tensor,
        *,
        conditioner: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._transform.inverse(z, context=conditioner)

    def log_density(self, x: torch.Tensor, *, conditioner: torch.Tensor) -> torch.Tensor:
        z, ladj = self.transform(x, conditioner=conditioner)
        return self._clip_log_density(self.base.log_density(z) + ladj)

    def sample(self, conditioner: torch.Tensor, num_samples_per: int = 1) -> torch.Tensor:
        if conditioner.ndim == 1:
            conditioner = conditioner.unsqueeze(0)
        elif conditioner.ndim > 2:
            conditioner = conditioner.view(-1, conditioner.shape[-1])

        n_ctx = conditioner.shape[0]
        z = self.base.sample(n_ctx * num_samples_per)
        c = conditioner.repeat_interleave(num_samples_per, dim=0)
        x, _ = self.inverse_transform(z, conditioner=c)
        if num_samples_per == 1:
            return x
        return x.view(n_ctx, num_samples_per, self.dim)

    def supremum_bound(self) -> torch.Tensor:
        return self.base.supremum_bound()

    def dtype_device(self):
        p = next(self.parameters())
        return p.dtype, p.device
