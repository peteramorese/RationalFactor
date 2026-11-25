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
from rational_factor.tools.set import Hyperrectangle




class ControlledKinematicUnicycle(ControllableStochasticSystem, systems.KinematicUnicycle):
    """3D kinematic unicycle with one control: angle displacement per step.

    State [px, py, theta], control [dtheta]. Forward speed is fixed.
    Actual angular displacement = u[0] + angle_noise_bound * eta_theta,
    with eta_theta ~ Uniform(-1, 1). Position noise is independent Gaussian.
    See ``systems.KinematicUnicycle`` for the exact-arc position update.
    """

    def __init__(
        self,
        dt: float = 0.3,
        speed: float = 1.0,
        angle_noise_bound: float = 0.2,
        position_noise_std: float | tuple[float, float] = 0.05,
        bounded_support: bool = False,
        support: Hyperrectangle | None = None,
    ):
        std = systems.KinematicUnicycle._check_parameters(
            dt, speed, angle_noise_bound, position_noise_std
        )
        if support is None:
            support = Hyperrectangle(
                systems.KinematicUnicycle.DEFAULT_SUPPORT_LOW.clone(),
                systems.KinematicUnicycle.DEFAULT_SUPPORT_HIGH.clone(),
            )
        ControllableStochasticSystem.__init__(
            self,
            state_dim=3,
            control_dim=1,
            state_labels=["px", "py", "theta"],
            control_labels=["dtheta"],
            v_dist=systems.UnicycleProcessNoise(),
            support=support,
            bounded_support=bounded_support,
            wrap_dims=systems.KinematicUnicycle.WRAP_DIMS,
        )
        self.dt = float(dt)
        self.speed = float(speed)
        self.angle_noise_bound = float(angle_noise_bound)
        self.register_buffer("position_noise_std", std)

    def next_state(self, x: torch.Tensor, u: torch.Tensor,
                   v: torch.Tensor) -> torch.Tensor:
        if u.shape[-1] != 1:
            raise ValueError("Expected angle displacement control with shape (..., 1)")
        return self._step(x, u[..., 0], v)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # Independent random innovations for every state in a batch.
        noise = self._v_dist.sample(x.shape[:-1]).to(device=x.device, dtype=x.dtype)
        return self.next_state(x, u, noise)


class ControlledScalarNonlinearDrift(ControllableStochasticSystem, systems.ScalarNonlinearDrift):
    """x_{k+1} = f(x_k) + u + v,  u is additive scalar forcing."""

    def __init__(
        self,
        sigma: float = 0.1,
        covariance: torch.Tensor | None = None,
        bounded_support: bool = False,
        support: Hyperrectangle | None = None,
    ):
        if covariance is None:
            covariance = torch.tensor([[sigma**2]], dtype=torch.float32)
        else:
            covariance = torch.as_tensor(covariance, dtype=torch.float32)

        if support is None:
            support = Hyperrectangle(
                systems.ScalarNonlinearDrift.DEFAULT_SUPPORT_LOW.clone(),
                systems.ScalarNonlinearDrift.DEFAULT_SUPPORT_HIGH.clone(),
            )
        dist = torch.distributions.MultivariateNormal(torch.zeros(1), covariance)
        ControllableStochasticSystem.__init__(
            self,
            state_dim=1,
            control_dim=1,
            state_labels=["x"],
            control_labels=["u"],
            v_dist=dist,
            support=support,
            bounded_support=bounded_support,
            wrap_dims=systems.ScalarNonlinearDrift.WRAP_DIMS,
        )
        self.cov = covariance

    def next_state(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
        mean = self._drift(x) + u.reshape(-1)
        return self._apply_support(x, mean + v.reshape(-1))


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
        bounded_support: bool = False,
        support: Hyperrectangle | None = None,
    ):
        if covariance is None:
            covariance = torch.eye(2)
        else:
            covariance = torch.as_tensor(covariance, dtype=torch.float32)

        if support is None:
            support = Hyperrectangle(
                systems.VanDerPol.DEFAULT_SUPPORT_LOW.clone(),
                systems.VanDerPol.DEFAULT_SUPPORT_HIGH.clone(),
            )
        dist = torch.distributions.MultivariateNormal(torch.zeros(2), covariance)
        ControllableStochasticSystem.__init__(
            self,
            state_dim=2,
            control_dim=2,
            state_labels=["x1", "x2"],
            control_labels=["u_x1", "u_x2"],
            v_dist=dist,
            support=support,
            bounded_support=bounded_support,
            wrap_dims=systems.VanDerPol.WRAP_DIMS,
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
        return self._apply_support(x, torch.stack([x1_next, x2_next]) + v.flatten())


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
        bounded_support: bool = False,
        support: Hyperrectangle | None = None,
    ):
        if covariance is None:
            covariance = 0.001 * torch.eye(4)
        else:
            covariance = torch.as_tensor(covariance, dtype=torch.float32)

        if support is None:
            support = Hyperrectangle(
                systems.CartPole.DEFAULT_SUPPORT_LOW.clone(),
                systems.CartPole.DEFAULT_SUPPORT_HIGH.clone(),
            )
        dist = torch.distributions.MultivariateNormal(torch.zeros(4), covariance)
        ControllableStochasticSystem.__init__(
            self,
            state_dim=4,
            control_dim=1,
            state_labels=["p", "p_dot", "theta", "theta_dot"],
            control_labels=["F"],
            v_dist=dist,
            support=support,
            bounded_support=bounded_support,
            wrap_dims=systems.CartPole.WRAP_DIMS,
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

        x_next = torch.stack([p_next, p_dot_next, theta_next, theta_dot_next]) + v.flatten()
        return self._apply_support(x_flat, x_next)


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
        bounded_support: bool = False,
        support: Hyperrectangle | None = None,
    ):
        if covariance is None:
            covariance = 0.01 * torch.eye(6)
        else:
            covariance = torch.as_tensor(covariance, dtype=torch.float32)

        if support is None:
            support = Hyperrectangle(
                systems.PlanarQuadrotor.DEFAULT_SUPPORT_LOW.clone(),
                systems.PlanarQuadrotor.DEFAULT_SUPPORT_HIGH.clone(),
            )
        dist = torch.distributions.MultivariateNormal(torch.zeros(6), covariance)
        ControllableStochasticSystem.__init__(
            self,
            state_dim=6,
            control_dim=2,
            state_labels=["px", "pz", "theta", "vx", "vz", "omega"],
            control_labels=["u1", "u2"],
            v_dist=dist,
            support=support,
            bounded_support=bounded_support,
            wrap_dims=systems.PlanarQuadrotor.WRAP_DIMS,
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
        x_next = x.flatten() + self.dt * dx + v.flatten()
        return self._apply_support(x.flatten(), x_next)


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
        bounded_support: bool = False,
        support: Hyperrectangle | None = None,
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

        if support is None:
            support = Hyperrectangle(
                systems.SecondOrderDubinsTrailer.DEFAULT_SUPPORT_LOW.clone(),
                systems.SecondOrderDubinsTrailer.DEFAULT_SUPPORT_HIGH.clone(),
            )
        ControllableStochasticSystem.__init__(
            self,
            state_dim=6,
            control_dim=2,
            state_labels=["px", "py", "theta_c", "theta_t", "v", "omega"],
            control_labels=["u_v", "u_omega"],
            v_dist=v_dist,
            support=support,
            bounded_support=bounded_support,
            wrap_dims=systems.SecondOrderDubinsTrailer.WRAP_DIMS,
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

        x_next = torch.stack(
            [px_next, py_next, theta_c_next, theta_t_next, speed_next, omega_next],
            dim=0,
        )
        return self._apply_support(x, x_next)


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
        bounded_support: bool = False,
        support: Hyperrectangle | None = None,
    ):
        if covariance is None:
            covariance = 0.001 * torch.eye(12)
        else:
            covariance = torch.as_tensor(covariance, dtype=torch.float32)

        if support is None:
            support = Hyperrectangle(
                systems.Quadcopter.DEFAULT_SUPPORT_LOW.clone(),
                systems.Quadcopter.DEFAULT_SUPPORT_HIGH.clone(),
            )
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
            support=support,
            bounded_support=bounded_support,
            wrap_dims=systems.Quadcopter.WRAP_DIMS,
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

        return self._apply_support(x, x_next + v)


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
        bounded_support: bool = False,
        support: Hyperrectangle | None = None,
    ):
        if state_scale is None:
            state_scale = torch.ones(12, dtype=torch.float32)
        state_scale = torch.as_tensor(state_scale, dtype=torch.float32).reshape(12).clamp_min(1e-6)
        if support is None:
            support = Hyperrectangle(
                (systems.Aircraft.PLOT_BOUNDS_LOW_PHYS / state_scale).clone(),
                (systems.Aircraft.PLOT_BOUNDS_HIGH_PHYS / state_scale).clone(),
            )
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
            support=support,
            bounded_support=bounded_support,
            wrap_dims=systems.Aircraft.WRAP_DIMS,
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

        x_next = self._to_normalized(x_next_phys)
        return self._apply_support(x.flatten(), x_next)


class ControlledIcyRoadCar(ControllableStochasticSystem):
    """Six-state dynamic bicycle / small car on a 6 x 30 m icy road.

    State = [x, y, theta, vx, vy, r]; control = [steering angle (rad),
    requested longitudinal acceleration (m/s^2)].  x is across the road,
    y is along the road.  The car initially faces +y; positive steering/yaw
    turns toward increasing x.  vx and vy are body-frame longitudinal and
    lateral velocities, respectively.

    The icy regions are fixed in space and have smooth boundaries: friction
    depends only on the front/rear tire contact positions.  Stochasticity
    instead enters through the applied steering angle and requested
    acceleration.  A small, optional additive Gaussian state disturbance is
    applied after numerical integration.  All noise is independent between
    particles and discrete timesteps, so the six-state process is Markov.

    Noise vector v has eight independent standard-normal components:
    [steering, acceleration, x, y, theta, vx, vy, r].  Each component is
    scaled by its corresponding configurable standard deviation.

    The default dimensions are those of a small research vehicle / go-kart,
    allowing its entire body to start near y=1 on a short 30 m road.
    """

    DEFAULT_ICE_PATCHES = (
        (0.35, 2.85, 6.5, 11.8),
        (2.30, 5.70, 13.8, 18.7),
        (0.65, 3.45, 21.0, 25.7),
    )  # each (x_min, x_max, y_min, y_max), meters
    def __init__(
        self,
        dt: float = 0.3,
        substeps: int = 4,
        road_width: float = 6.0,
        road_length: float = 30.0,
        car_length: float = 1.9,
        car_width: float = 0.95,
        mass: float = 400.0,
        yaw_inertia: float = 190.0,
        lf: float = 0.70,
        lr: float = 0.75,
        gravity: float = 9.81,
        cornering_stiffness_front: float = 11000.0,
        cornering_stiffness_rear: float = 11500.0,
        drag_coefficient: float = 0.35,
        rolling_resistance: float = 0.012,
        min_slip_speed: float = 0.8,
        mu_dry: float = 0.9,
        mu_ice: float = 0.09,
        ice_transition_width: float = 0.28,
        ice_patches: tuple[tuple[float, float, float, float], ...] = DEFAULT_ICE_PATCHES,
        steering_noise_std: float = 0.025,
        acceleration_noise_std: float = 0.35,
        additive_state_noise_std: tuple[float, float, float, float, float, float] = (
            0.002, 0.002, 0.001, 0.015, 0.015, 0.008,
        ),
        steering_max: float = 0.35,
        acceleration_min: float = -3.0,
        acceleration_max: float = 2.0,
        bounded_support: bool = False,
        support: Hyperrectangle | None = None,
    ):
        if dt <= 0 or substeps < 1 or int(substeps) != substeps:
            raise ValueError("dt must be positive and substeps a positive integer")
        if min(road_width, road_length, car_length, car_width, mass, yaw_inertia,
               lf, lr, gravity, cornering_stiffness_front,
               cornering_stiffness_rear, min_slip_speed, mu_ice,
               ice_transition_width) <= 0:
            raise ValueError("Vehicle, friction, and road scales must be positive")
        if not (mu_dry > mu_ice):
            raise ValueError("Require mu_dry > mu_ice > 0")
        if steering_max <= 0 or acceleration_min >= acceleration_max:
            raise ValueError("Invalid steering or acceleration limits")
        if not (math.isfinite(steering_noise_std) and steering_noise_std >= 0
                and math.isfinite(acceleration_noise_std) and acceleration_noise_std >= 0):
            raise ValueError("Control noise standard deviations must be nonnegative and finite")
        state_noise_std = torch.as_tensor(additive_state_noise_std, dtype=torch.float32)
        if (state_noise_std.shape != (6,) or not torch.isfinite(state_noise_std).all()
                or (state_noise_std < 0).any()):
            raise ValueError("additive_state_noise_std must contain six finite nonnegative values")
        if not ice_patches:
            raise ValueError("At least one ice patch is required")
        for x0, x1, y0, y1 in ice_patches:
            if not (0 <= x0 < x1 <= road_width and 0 <= y0 < y1 <= road_length):
                raise ValueError("Ice patches must be inside the road, with ordered bounds")

        if support is None:
            support = Hyperrectangle(
                torch.tensor([0.0, 0.0, -math.pi, -3.0, -5.0, -4.0], dtype=torch.float32),
                torch.tensor([float(road_width), float(road_length), math.pi, 13.0, 5.0, 4.0], dtype=torch.float32),
            )
        v_dist = torch.distributions.Independent(
            torch.distributions.Normal(torch.zeros(8), torch.ones(8)), 1
        )
        super().__init__(
            state_dim=6,
            control_dim=2,
            state_labels=["x", "y", "theta", "vx", "vy", "r"],
            control_labels=["steering", "acceleration"],
            v_dist=v_dist,
            support=support,
            bounded_support=bounded_support,
            wrap_dims=(2,),
        )
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.road_width = float(road_width)
        self.road_length = float(road_length)
        self.car_length = float(car_length)
        self.car_width = float(car_width)
        self.mass = float(mass)
        self.yaw_inertia = float(yaw_inertia)
        self.lf = float(lf)
        self.lr = float(lr)
        self.gravity = float(gravity)
        self.cornering_stiffness_front = float(cornering_stiffness_front)
        self.cornering_stiffness_rear = float(cornering_stiffness_rear)
        self.drag_coefficient = float(drag_coefficient)
        self.rolling_resistance = float(rolling_resistance)
        self.min_slip_speed = float(min_slip_speed)
        self.mu_dry = float(mu_dry)
        self.mu_ice = float(mu_ice)
        self.ice_transition_width = float(ice_transition_width)
        self.ice_patches = tuple(tuple(map(float, p)) for p in ice_patches)
        self.steering_noise_std = float(steering_noise_std)
        self.acceleration_noise_std = float(acceleration_noise_std)
        self.register_buffer("additive_state_noise_std", state_noise_std.clone())
        self.steering_max = float(steering_max)
        self.acceleration_min = float(acceleration_min)
        self.acceleration_max = float(acceleration_max)

    def friction_at(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Deterministic, smooth tire-road friction at world position(s) (x,y).

        The icy patches have fixed positions and smooth sigmoid transitions.
        This map also extends outside the road, where it approaches dry friction.
        """
        x = torch.as_tensor(x)
        y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
        clear = torch.ones_like(x + y)
        w = self.ice_transition_width
        for xlo, xhi, ylo, yhi in self.ice_patches:
            mask = (torch.sigmoid((x - xlo) / w)
                    * torch.sigmoid((xhi - x) / w)
                    * torch.sigmoid((y - ylo) / w)
                    * torch.sigmoid((yhi - y) / w))
            clear = clear * (1.0 - mask)
        return self.mu_ice + (self.mu_dry - self.mu_ice) * clear

    def _derivatives(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        pos_x, pos_y, theta, vx, vy, r = state.unbind(dim=-1)
        steering = torch.clamp(u[..., 0], -self.steering_max, self.steering_max)
        requested_a = torch.clamp(u[..., 1], self.acceleration_min,
                                   self.acceleration_max)

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        # Tire centers: the car's forward direction is (sin(theta), cos(theta)).
        mu_f = self.friction_at(pos_x + self.lf * sin_theta,
                                pos_y + self.lf * cos_theta)
        mu_r = self.friction_at(pos_x - self.lr * sin_theta,
                                pos_y - self.lr * cos_theta)
        load_f = self.mass * self.gravity * self.lr / (self.lf + self.lr)
        load_r = self.mass * self.gravity * self.lf / (self.lf + self.lr)
        cap_f = mu_f * load_f
        cap_r = mu_r * load_r

        # Small-speed regularization prevents singular slip-angle computations.
        speed = torch.sqrt(vx.square() + self.min_slip_speed ** 2)
        alpha_f = steering - torch.atan((vy + self.lf * r) / speed)
        alpha_r = -torch.atan((vy - self.lr * r) / speed)

        # Saturating lateral force and a rear-tire friction circle; the
        # longitudinal acceleration command can also become traction-limited.
        fy_f = cap_f * torch.tanh(self.cornering_stiffness_front * alpha_f / cap_f)
        fx_r = cap_r * torch.tanh(self.mass * requested_a / cap_r)
        cap_yr = torch.sqrt(torch.clamp(cap_r.square() - fx_r.square(), min=1.0))
        fy_r = cap_yr * torch.tanh(self.cornering_stiffness_rear * alpha_r / cap_yr)

        fx_body = (fx_r - fy_f * torch.sin(steering)
                   - self.drag_coefficient * vx * torch.abs(vx)
                   - self.rolling_resistance * self.mass * self.gravity
                   * torch.tanh(vx / self.min_slip_speed))
        fy_body = fy_f * torch.cos(steering) + fy_r
        return torch.stack([
            vx * sin_theta + vy * cos_theta,
            vx * cos_theta - vy * sin_theta,
            r,
            fx_body / self.mass + r * vy,
            fy_body / self.mass - r * vx,
            (self.lf * fy_f * torch.cos(steering) - self.lr * fy_r)
            / self.yaw_inertia,
        ], dim=-1)

    def next_state(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
        """One stochastic Markov step; supports (6,) and (batch,6) states."""
        if x.shape[-1] != 6 or u.shape[-1] != 2 or v.shape[-1] != 8:
            raise ValueError("Expected state (...,6), control (...,2), Gaussian noise (...,8)")
        # Control noise is sampled once per step and held constant across the
        # integration substeps.  Nonlinearity comes from slip and tire forces.
        effective_u = torch.stack([
            u[..., 0] + self.steering_noise_std * v[..., 0],
            u[..., 1] + self.acceleration_noise_std * v[..., 1],
        ], dim=-1)
        h = self.dt / self.substeps
        state = x
        for _ in range(self.substeps):
            state = state + h * self._derivatives(state, effective_u)
        # Optional independent additive Gaussian state noise (zero std disables it).
        x_next = state + v[..., 2:] * self.additive_state_noise_std
        return self._apply_support(x, x_next)

    def forward(self, x: torch.Tensor, u: torch.Tensor):
        """Draw independent steering, acceleration, and additive state noise."""
        v = self._v_dist.sample(x.shape[:-1]).to(device=x.device, dtype=x.dtype)
        return self.next_state(x, u, v)
