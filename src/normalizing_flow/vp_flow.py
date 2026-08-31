"""Volume-preserving continuous normalizing flow on the unit box [0, 1]^d.

A continuous normalizing flow is the time-1 map of an ODE

    dx/dt = v_θ(x, t, c).

Volume preservation is equivalent to ∇ · v = 0. Trajectories stay in the unit
box if the field is tangent to the boundary, i.e. the normal component vanishes
on every face:

    v_i(x, t, c) = 0  whenever  x_i ∈ {0, 1}.

Both constraints are built into the parameterization. Let A_θ(x, t, c) be a
skew-symmetric matrix produced by a neural net, and set

    Ψ_{ij}(x, t, c) = x_i (1 - x_i) x_j (1 - x_j) A_{ij}(x, t, c).

Then the velocity

    v_i = Σ_j ∂Ψ_{ij} / ∂x_j

is divergence-free (mixed partials of a skew matrix cancel) and has zero normal
component on ∂[0, 1]^d (Ψ_{i·} vanishes identically on the faces x_i ∈ {0, 1},
so its tangential derivatives do too). Because ∇ · v = 0, log |det J_x| = 0
along the flow. The conditioner c is held fixed along each trajectory.
"""

from __future__ import annotations

import copy

import torch

from rational_factor.models.domain_transformation import DomainTF, MLP


class DivergenceFreeBoxVelocity(torch.nn.Module):
    """Divergence-free velocity on [0, 1]^d with zero flux through the boundary."""

    def __init__(
        self,
        dim: int,
        conditioner_dim: int,
        hidden_features: int = 64,
        num_hidden_layers: int = 2,
        time_dependent: bool = True,
        zero_init: bool = True,
    ):
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be at least 1")
        if conditioner_dim < 1:
            raise ValueError("conditioner_dim must be at least 1")

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
        conditioner: torch.Tensor,
    ) -> torch.Tensor:
        parts = [x, conditioner]
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
        conditioner: torch.Tensor,
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
        conditioner: torch.Tensor,
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


class VolumePreservingFlow(DomainTF):
    """Conditional time-1 map of a divergence-free no-flux velocity on the unit box.

    ``forward`` integrates t: 0 → 1 (physical → latent).
    ``inverse`` integrates t: 1 → 0 (latent → physical).
    Both return log |det J_x| = 0. The conditioner is constant along trajectories.
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
        trainable: bool = True,
    ):
        super().__init__(dim)
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")

        self.conditioner_dim = conditioner_dim
        self.n_steps = n_steps
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
    def copy_from_trainable(cls, other: "VolumePreservingFlow"):
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
                f"conditioner must have {self.conditioner_dim} features, got {conditioner.shape[-1]}"
            )
        if conditioner.shape[0] == 1 and x.shape[0] > 1:
            conditioner = conditioner.expand(x.shape[0], -1)
        if conditioner.shape[0] != x.shape[0]:
            raise ValueError(
                f"conditioner batch {conditioner.shape[0]} does not match x batch {x.shape[0]}"
            )
        return conditioner.to(dtype=x.dtype, device=x.device)

    def velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor | float,
        *,
        conditioner: torch.Tensor,
    ) -> torch.Tensor:
        conditioner = self._prepare_conditioner(x, conditioner)
        return self.velocity_field(x, t, conditioner)

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

    def forward(self, x: torch.Tensor, *, conditioner: torch.Tensor):
        assert x.shape[1] == self.dim, "x must have shape (n_data, dim)"
        # 1D skew-symmetric A is identically 0, so v ≡ 0 and T is the identity.
        if self.dim == 1:
            return x, x.new_zeros(x.shape[0])
        conditioner = self._prepare_conditioner(x, conditioner)
        z = self._rk4(x, 0.0, 1.0, conditioner)
        ladj = x.new_zeros(x.shape[0])
        return z, ladj

    def inverse(self, z: torch.Tensor, *, conditioner: torch.Tensor):
        assert z.shape[1] == self.dim, "z must have shape (n_data, dim)"
        if self.dim == 1:
            return z, z.new_zeros(z.shape[0])
        conditioner = self._prepare_conditioner(z, conditioner)
        x = self._rk4(z, 1.0, 0.0, conditioner)
        ladj = z.new_zeros(z.shape[0])
        return x, ladj
