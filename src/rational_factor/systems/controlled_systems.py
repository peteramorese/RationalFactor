"""
Controllable counterparts to the passive / closed-loop systems in ``systems.py``.

Passive plants receive an explicit control input ``u`` each step. Systems that
previously embedded a feedback controller (quadrotors, aircraft, Dubins trailer)
expose the former actuator commands directly.
"""

import math

import torch

from .base import ControllableStochasticSystem
from . import systems


class ControlledScalarNonlinearDrift(ControllableStochasticSystem, systems.ScalarNonlinearDrift):
    """x_{k+1} = f(x_k) + u + v,  u is additive scalar forcing."""

    def __init__(
        self,
        sigma: float = 0.1,
        covariance: torch.Tensor | None = None,
    ):
        if covariance is None:
            covariance = torch.tensor([[sigma**2]], dtype=torch.float32)
        else:
            covariance = torch.as_tensor(covariance, dtype=torch.float32)

        dist = torch.distributions.MultivariateNormal(torch.zeros(1), covariance)
        ControllableStochasticSystem.__init__(
            self,
            state_dim=1,
            control_dim=1,
            state_labels=["x"],
            control_labels=["u"],
            v_dist=dist,
        )
        self.cov = covariance

    def next_state(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
        mean = self._drift(x) + u.reshape(-1)
        return mean + v.reshape(-1)


class ControlledVanDerPol(ControllableStochasticSystem, systems.VanDerPol):
    """
    Van der Pol with two additive forcing channels on the velocity equations:
      dx1 = x2 + u_x1
      dx2 = mu (1 - x1^2) x2 - x1 + u_x2
    """

    def __init__(
        self,
        dt: float,
        mu: float = 1.0,
        covariance: torch.Tensor | None = None,
    ):
        if covariance is None:
            covariance = torch.eye(2)
        else:
            covariance = torch.as_tensor(covariance, dtype=torch.float32)

        dist = torch.distributions.MultivariateNormal(torch.zeros(2), covariance)
        ControllableStochasticSystem.__init__(
            self,
            state_dim=2,
            control_dim=2,
            state_labels=["x1", "x2"],
            control_labels=["u_x1", "u_x2"],
            v_dist=dist,
        )
        self.dt = dt
        self.mu = mu
        self.cov = covariance

    def next_state(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
        u = u.flatten()
        x1, x2 = x[0], x[1]
        dx1 = x2 + u[0]
        dx2 = self.mu * (1 - x1**2) * x2 - x1 + u[1]

        x1_next = x1 + self.dt * dx1
        x2_next = x2 + self.dt * dx2
        return torch.stack([x1_next, x2_next]) + v.flatten()


class ControlledCartPole(ControllableStochasticSystem, systems.CartPole):
    """Cart-pole with horizontal cart force u (same plant as ``CartPole``)."""

    def __init__(
        self,
        dt: float,
        m_c: float = 1.0,
        m_p: float = 0.1,
        l: float = 0.5,
        g: float = 9.81,
        covariance: torch.Tensor | None = None,
    ):
        if covariance is None:
            covariance = 0.001 * torch.eye(4)
        else:
            covariance = torch.as_tensor(covariance, dtype=torch.float32)

        dist = torch.distributions.MultivariateNormal(torch.zeros(4), covariance)
        ControllableStochasticSystem.__init__(
            self,
            state_dim=4,
            control_dim=1,
            state_labels=["p", "p_dot", "theta", "theta_dot"],
            control_labels=["F"],
            v_dist=dist,
        )
        self.dt = dt
        self.m_c = m_c
        self.m_p = m_p
        self.l = l
        self.g = g
        self.cov = covariance

    def next_state(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
        x_flat = x.flatten()
        p, p_dot, theta, theta_dot = x_flat[0], x_flat[1], x_flat[2], x_flat[3]
        force = u.flatten()[0]

        p_acc, theta_acc = self._accelerations(theta, theta_dot, force)

        p_next = p + self.dt * p_dot
        p_dot_next = p_dot + self.dt * p_acc
        theta_next = theta + self.dt * theta_dot
        theta_dot_next = theta_dot + self.dt * theta_acc

        return torch.stack([p_next, p_dot_next, theta_next, theta_dot_next]) + v.flatten()


class ControlledPlanarQuadrotor(ControllableStochasticSystem, systems.PlanarQuadrotor):
    """Planar quadrotor with rotor thrusts u = [u1, u2] (no waypoint tracking)."""

    def __init__(
        self,
        dt: float,
        m: float = 1.0,
        I: float = 0.02,
        ell: float = 0.2,
        g: float = 9.81,
        c_v: float = 0.05,
        c_w: float = 0.02,
        covariance: torch.Tensor | None = None,
        thrust_min: float = 0.0,
        thrust_max: float = 20.0,
    ):
        if covariance is None:
            covariance = 0.01 * torch.eye(6)
        else:
            covariance = torch.as_tensor(covariance, dtype=torch.float32)

        dist = torch.distributions.MultivariateNormal(torch.zeros(6), covariance)
        ControllableStochasticSystem.__init__(
            self,
            state_dim=6,
            control_dim=2,
            state_labels=["px", "pz", "theta", "vx", "vz", "omega"],
            control_labels=["u1", "u2"],
            v_dist=dist,
        )
        self.dt = dt
        self.m = m
        self.I = I
        self.ell = ell
        self.g = g
        self.c_v = c_v
        self.c_w = c_w
        self.thrust_min = thrust_min
        self.thrust_max = thrust_max
        self.cov = covariance

    def next_state(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
        u = torch.clamp(u.flatten(), min=self.thrust_min, max=self.thrust_max)
        dx = self._dynamics(x, u)
        return x.flatten() + self.dt * dx + v.flatten()


class ControlledSecondOrderDubinsTrailer(
    ControllableStochasticSystem, systems.SecondOrderDubinsTrailer
):
    """
    Tractor–trailer with explicit control u = [u_v, u_omega]:
      u_v drives longitudinal acceleration, u_omega drives steering-rate acceleration.
    Multiplicative noise on controls matches the passive model.
    """

    def __init__(
        self,
        dt: float,
        L_t: float = 1.0,
        sigma_v: float = 0.1,
        sigma_omega: float = 0.1,
        sigma_speed_state: float = 0.1,
        cov_scale: float = 0.01,
    ):
        cov1 = cov_scale * torch.tensor([[1.0, 0.2], [0.2, 1.0]], dtype=torch.float32)
        cov2 = cov_scale * torch.tensor([[1.0, -0.2], [-0.2, 1.0]], dtype=torch.float32)
        loc = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        mix = torch.distributions.Categorical(probs=torch.tensor([0.6, 0.4], dtype=torch.float32))
        comp = torch.distributions.MultivariateNormal(
            loc=loc,
            covariance_matrix=torch.stack([cov1, cov2]),
        )
        v_dist = torch.distributions.MixtureSameFamily(mix, comp)

        ControllableStochasticSystem.__init__(
            self,
            state_dim=6,
            control_dim=2,
            state_labels=["px", "py", "theta_c", "theta_t", "v", "omega"],
            control_labels=["u_v", "u_omega"],
            v_dist=v_dist,
        )
        self.dt = dt
        self.L_t = L_t
        self.sigma_v = sigma_v
        self.sigma_omega = sigma_omega
        self.sigma_speed_state = sigma_speed_state

    def next_state(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
        x = x.flatten()
        u = u.flatten()
        v = v.flatten()
        px, py, theta_c, theta_t, speed, omega = x[0], x[1], x[2], x[3], x[4], x[5]
        xi_v, xi_omega = v[0], v[1]
        dt = self.dt

        u_v_noisy = u[0] * (1.0 + self.sigma_v * xi_v)
        u_omega_noisy = u[1] * (1.0 + self.sigma_omega * xi_omega)

        px_next = px + dt * speed * torch.cos(theta_c)
        py_next = py + dt * speed * torch.sin(theta_c)
        theta_c_next = theta_c + dt * omega
        theta_t_next = theta_t + dt * (speed / self.L_t) * torch.sin(theta_c - theta_t)
        speed_next = speed + dt * u_v_noisy + self.sigma_speed_state * speed * xi_v
        omega_next = omega + dt * u_omega_noisy

        return torch.stack(
            [px_next, py_next, theta_c_next, theta_t_next, speed_next, omega_next],
            dim=0,
        )


class ControlledQuadcopter(ControllableStochasticSystem, systems.Quadcopter):
    """12D quadcopter with u = [T, tau_x, tau_y, tau_z] (no autopilot)."""

    def __init__(
        self,
        dt: float,
        m: float = 1.0,
        J: torch.Tensor | None = None,
        g: float = 9.81,
        c_v: float = 0.05,
        c_w: float = 0.05,
        thrust_min: float = 0.0,
        thrust_max: float = 20.0,
        torque_limits: torch.Tensor | None = None,
        covariance: torch.Tensor | None = None,
    ):
        if covariance is None:
            covariance = 0.001 * torch.eye(12)
        else:
            covariance = torch.as_tensor(covariance, dtype=torch.float32)

        dist = torch.distributions.MultivariateNormal(torch.zeros(12), covariance)
        ControllableStochasticSystem.__init__(
            self,
            state_dim=12,
            control_dim=4,
            state_labels=[
                "px",
                "py",
                "pz",
                "vx",
                "vy",
                "vz",
                "phi",
                "theta",
                "psi",
                "p",
                "q",
                "r",
            ],
            control_labels=["T", "tau_x", "tau_y", "tau_z"],
            v_dist=dist,
        )

        self.dt = dt
        self.m = m
        self.g = g
        self.c_v = c_v
        self.c_w = c_w
        self.cov = covariance
        self.thrust_min = thrust_min
        self.thrust_max = thrust_max

        if J is None:
            J = torch.diag(torch.tensor([0.02, 0.02, 0.04], dtype=torch.float32))
        self.J = torch.as_tensor(J, dtype=torch.float32).reshape(3, 3)
        self.Jinv = torch.linalg.inv(self.J)

        if torque_limits is None:
            torque_limits = torch.tensor([1.0, 1.0, 0.5], dtype=torch.float32)
        self.torque_limits = torch.as_tensor(torque_limits, dtype=torch.float32).reshape(3)

    def next_state(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
        x = x.flatten()
        v = v.flatten()
        u = u.flatten()
        T = torch.clamp(u[0], min=self.thrust_min, max=self.thrust_max)
        tau = torch.clamp(u[1:4], min=-self.torque_limits, max=self.torque_limits)

        xdot = self._dynamics(x, T, tau)
        x_next = x + self.dt * xdot

        x_next = x_next.clone()
        x_next[6] = torch.remainder(x_next[6] + math.pi, 2 * math.pi) - math.pi
        x_next[7] = torch.remainder(x_next[7] + math.pi, 2 * math.pi) - math.pi
        x_next[8] = torch.remainder(x_next[8] + math.pi, 2 * math.pi) - math.pi

        return x_next + v


class ControlledAircraft(ControllableStochasticSystem, systems.Aircraft):
    """
    12D fixed-wing aircraft with u = [T, delta_e, delta_a, delta_r].
    Multiplicative process noise matches ``Aircraft``.
    """

    def __init__(
        self,
        dt: float,
        V_ref: float = 22.0,
        m: float = 25.0,
        S: float = 0.55,
        b: float = 2.2,
        c_bar: float = 0.18,
        rho: float = 1.225,
        g: float = 9.81,
        J: torch.Tensor | None = None,
        thrust_min: float = 5.0,
        thrust_max: float = 600.0,
        delta_max: float = 0.35,
        sigma_thrust: float = 0.04,
        sigma_surface: float = 0.05,
        sigma_CL: float = 0.03,
        sigma_CM: float = 0.04,
        noise_cov_scale: float = 1.0,
        state_scale: torch.Tensor | None = None,
    ):
        cov = noise_cov_scale * torch.eye(6, dtype=torch.float32)
        dist = torch.distributions.MultivariateNormal(torch.zeros(6), cov)
        ControllableStochasticSystem.__init__(
            self,
            state_dim=12,
            control_dim=4,
            state_labels=[
                "px",
                "py",
                "pz",
                "u",
                "v",
                "w",
                "phi",
                "theta",
                "psi",
                "p",
                "q",
                "r",
            ],
            control_labels=["T", "de", "da", "dr"],
            v_dist=dist,
        )

        self.dt = dt
        self.m = m
        self.S = S
        self.b = b
        self.c_bar = c_bar
        self.rho = rho
        self.g = g
        self.V_ref = V_ref
        self.thrust_min = thrust_min
        self.thrust_max = thrust_max
        self.delta_max = delta_max

        self.sigma_thrust = sigma_thrust
        self.sigma_surface = sigma_surface
        self.sigma_CL = sigma_CL
        self.sigma_CM = sigma_CM

        if state_scale is None:
            state_scale = torch.ones(12, dtype=torch.float32)
        state_scale = torch.as_tensor(state_scale, dtype=torch.float32).reshape(12).clamp_min(1e-6)
        self.register_buffer("state_scale", state_scale)

        if J is None:
            J = torch.diag(torch.tensor([15.0, 22.0, 32.0], dtype=torch.float32))
        self.J = torch.as_tensor(J, dtype=torch.float32).reshape(3, 3)
        self.Jinv = torch.linalg.inv(self.J)

        self.CL0 = 0.28
        self.CLa = 5.1
        self.CD0 = 0.028
        self.k_ind = 0.045
        self.CY_beta = -0.95

        self.Cm0 = -0.05
        self.Cma = -0.65
        self.Cmq = -18.0
        self.Cmd = -1.35

        self.Clb = -0.12
        self.Clp = -0.55
        self.Clda = 0.14

        self.Cnb = 0.12
        self.Cnr = -0.18
        self.Cndr = -0.085

    def next_state(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
        x_phys = self._to_physical(x).flatten()
        u = u.flatten()
        v = v.flatten()
        assert v.numel() == 6, "noise vector must be 6-dimensional"

        T, de, da, dr = u[0], u[1], u[2], u[3]
        xi = v

        T_n = T * (1.0 + self.sigma_thrust * xi[0])
        T_n = torch.clamp(T_n, min=self.thrust_min, max=self.thrust_max)

        de_n = de * (1.0 + self.sigma_surface * xi[1])
        da_n = da * (1.0 + self.sigma_surface * xi[2])
        dr_n = dr * (1.0 + self.sigma_surface * xi[3])
        de_n = torch.clamp(de_n, -self.delta_max, self.delta_max)
        da_n = torch.clamp(da_n, -self.delta_max, self.delta_max)
        dr_n = torch.clamp(dr_n, -self.delta_max, self.delta_max)

        CL_scale = 1.0 + self.sigma_CL * xi[4]
        CM_scale = 1.0 + self.sigma_CM * xi[5]

        xdot = self._dynamics(x_phys, T_n, de_n, da_n, dr_n, CL_scale, CM_scale)
        x_next_phys = x_phys + self.dt * xdot

        x_next_phys = x_next_phys.clone()
        x_next_phys[6] = torch.remainder(x_next_phys[6] + math.pi, 2 * math.pi) - math.pi
        x_next_phys[7] = torch.remainder(x_next_phys[7] + math.pi, 2 * math.pi) - math.pi
        x_next_phys[8] = torch.remainder(x_next_phys[8] + math.pi, 2 * math.pi) - math.pi

        return self._to_normalized(x_next_phys)
