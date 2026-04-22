import math

import torch
from .base import DiscreteTimeStochasticSystem


class ScalarNonlinearDrift(DiscreteTimeStochasticSystem):
    """
    One-dimensional state with a fixed nonlinear drift f(x) and additive Gaussian noise:
    x_{k+1} = f(x_k) + v,  v ~ N(0, Sigma).

    f is a deliberately irregular mix of sines, cosines, tanh, and rational terms (for testing).
    The transition density is p(x' | x) = N(x' | f(x), Sigma).
    """

    def __init__(self, sigma: float = 0.1, covariance: torch.Tensor | None = None):
        if covariance is None:
            covariance = torch.tensor([[sigma**2]], dtype=torch.float32)
        else:
            covariance = torch.as_tensor(covariance, dtype=torch.float32)
        assert covariance.shape == (1, 1), "covariance must be 1x1"

        dist = torch.distributions.MultivariateNormal(torch.zeros(1), covariance)
        super().__init__(dim=1, state_labels=["x"], v_dist=dist)
        self.cov = covariance

    @staticmethod
    def _drift(x: torch.Tensor) -> torch.Tensor:
        """Deterministic next state f(x), same batch shape as `x.reshape(-1)`."""
        t = x.reshape(-1)
        u = torch.tanh(0.5 * t)
        return (
            0.42 * torch.sin(2.1 * t + 0.31)
            + 0.28 * torch.cos(1.37 * t * t - 0.73)
            + 0.22 * u
            + 0.17 * (t / (1.0 + t * t))
            + 0.11 * torch.sin(3.05 * t) * torch.cos(0.48 * t)
            + 0.09 * torch.sin(t * torch.cos(1.2 * t))
        )

    def next_state(self, x: torch.Tensor, v: torch.Tensor):
        mean = self._drift(x)
        return mean + v.reshape(-1)

    def predict(self, x: torch.Tensor):
        mean = self._drift(x)
        return mean, self.cov

    def log_transition_density(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """
        log p(x' | x) under additive Gaussian noise: x' = f(x) + v, v ~ N(0, Sigma).
        Same batch size for x and x' (each row is one state if batched as (n, 1) or (n,)).
        """
        mean = self._drift(x).reshape(-1, 1)
        xp = x_prime.reshape(-1, 1)
        if mean.shape[0] != xp.shape[0]:
            raise ValueError("x and x_prime must have the same number of elements in the batch dimension")
        mvn = torch.distributions.MultivariateNormal(mean, self.cov)
        return mvn.log_prob(xp)

    def transition_density(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        """p(x' | x) = exp(log_transition_density(...))."""
        return self.log_transition_density(x, x_prime).exp()


class VanDerPol(DiscreteTimeStochasticSystem):
    def __init__(self, dt: float, mu: float = 1.0, covariance: torch.Tensor | None = None):
        """
        Van der Pol oscillator with additive Gaussian noise.
        State x = [x1, x2] (first-order realization of the oscillator).

        Args:
            dt : time step
            mu : nonlinearity parameter
            covariance : 2x2 covariance matrix for process noise (default: identity)
        """
        if covariance is None:
            covariance = torch.eye(2)
        else:
            covariance = torch.as_tensor(covariance, dtype=torch.float32)

        dist = torch.distributions.MultivariateNormal(torch.zeros(2), covariance)
        super().__init__(
            dim=2,
            state_labels=["x1", "x2"],
            v_dist=dist,
        )

        self.dt = dt
        self.mu = mu
        self.cov = covariance

    def next_state(self, x: torch.Tensor, v: torch.Tensor):
        x1, x2 = x[0], x[1]
        dx1 = x2
        dx2 = self.mu * (1 - x1**2) * x2 - x1

        x1_next = x1 + self.dt * dx1
        x2_next = x2 + self.dt * dx2

        return torch.stack([x1_next, x2_next]) + v

    def predict(self, x: torch.Tensor):
        x_flat = x.flatten()
        x1, x2 = x_flat[0], x_flat[1]
        dx1 = x2
        dx2 = self.mu * (1 - x1**2) * x2 - x1

        x1_next = x1 + self.dt * dx1
        x2_next = x2 + self.dt * dx2

        return torch.stack([x1_next, x2_next]), self.cov

    def jacobian(self, x: torch.Tensor):
        x_flat = x.flatten()
        x1, x2 = x_flat[0], x_flat[1]
        mu = self.mu
        dt = self.dt

        device, dtype = x.device, x.dtype
        dfdx1 = torch.zeros(2, 2, device=device, dtype=dtype)
        dfdx1[0, 0] = 1
        dfdx1[0, 1] = dt
        dfdx1[1, 0] = dt * (-2 * mu * x1 * x2 - 1)
        dfdx1[1, 1] = 1 + dt * mu * (1 - x1**2)

        return dfdx1

    def hessian_tensor(self, x: torch.Tensor):
        x_flat = x.flatten()
        x1, x2 = x_flat[0], x_flat[1]
        mu = self.mu
        dt = self.dt

        device, dtype = x.device, x.dtype
        H = torch.zeros(2, 2, 2, device=device, dtype=dtype)
        H[1, 0, 0] = dt * (-2 * mu * x2)
        H[1, 0, 1] = dt * (-2 * mu * x1)
        H[1, 1, 0] = dt * (-2 * mu * x1)

        return H


class CartPole(DiscreteTimeStochasticSystem):
    """
    Cart-pole (open-loop, no horizontal force). State angles follow the usual
    convention: theta is the pole angle from vertical; theta = 0 is upright.
    State x = [p, p_dot, theta, theta_dot].

    Dynamics match the classic uniform-pole model (pole half-length l, moment
    term 4/3 in the angular equation); see e.g. Barto/Sutton cart-pole.
    """

    def __init__(
        self,
        dt: float,
        m_c: float = 1.0,
        m_p: float = 0.1,
        l: float = 0.5,
        g: float = 9.81,
        covariance: torch.Tensor | None = None,
    ):
        """
        Args:
            dt : time step
            m_c : cart mass
            m_p : pole mass
            l : half-length of the pole (hinge to center of mass)
            g : gravity
            covariance : 4x4 process noise covariance (additive Gaussian)
        """
        if covariance is None:
            covariance = 0.001 * torch.eye(4)
        else:
            covariance = torch.as_tensor(covariance, dtype=torch.float32)

        dist = torch.distributions.MultivariateNormal(torch.zeros(4), covariance)
        super().__init__(
            dim=4,
            state_labels=["p", "p_dot", "theta", "theta_dot"],
            v_dist=dist,
        )

        self.dt = dt
        self.m_c = m_c
        self.m_p = m_p
        self.l = l
        self.g = g
        self.cov = covariance

    def _accelerations(
        self,
        theta: torch.Tensor,
        theta_dot: torch.Tensor,
        u: torch.Tensor | float = 0.0,
    ):
        """Horizontal cart acceleration and pole angular acceleration (no velocity in force terms)."""
        m_c, m_p, l, g = self.m_c, self.m_p, self.l, self.g
        total_mass = m_c + m_p
        sin_th = torch.sin(theta)
        cos_th = torch.cos(theta)
        if not isinstance(u, torch.Tensor):
            u = theta.new_tensor(float(u))

        temp = (u + m_p * l * theta_dot**2 * sin_th) / total_mass
        theta_acc = (g * sin_th - cos_th * temp) / (
            l * (4.0 / 3.0 - (m_p * cos_th**2) / total_mass)
        )
        p_acc = temp - (m_p * l * theta_acc * cos_th) / total_mass
        return p_acc, theta_acc

    def next_state(self, x: torch.Tensor, v: torch.Tensor):
        x_flat = x.flatten()
        p, p_dot, theta, theta_dot = x_flat[0], x_flat[1], x_flat[2], x_flat[3]

        p_acc, theta_acc = self._accelerations(theta, theta_dot, 0.0)

        p_next = p + self.dt * p_dot
        p_dot_next = p_dot + self.dt * p_acc
        theta_next = theta + self.dt * theta_dot
        theta_dot_next = theta_dot + self.dt * theta_acc

        return torch.stack([p_next, p_dot_next, theta_next, theta_dot_next]) + v.flatten()

    def predict(self, x: torch.Tensor):
        """Mean and additive noise covariance under forward Euler (no noise in mean)."""
        x_flat = x.flatten()
        p, p_dot, theta, theta_dot = x_flat[0], x_flat[1], x_flat[2], x_flat[3]
        p_acc, theta_acc = self._accelerations(theta, theta_dot, 0.0)
        mean = torch.stack(
            [
                p + self.dt * p_dot,
                p_dot + self.dt * p_acc,
                theta + self.dt * theta_dot,
                theta_dot + self.dt * theta_acc,
            ]
        )
        return mean, self.cov


class PlanarQuadrotor(DiscreteTimeStochasticSystem):
    def __init__(
        self,
        dt: float,
        waypoint: torch.Tensor | None = None,  # [px_ref, pz_ref]
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
        """
        Planar quadrotor with state-feedback waypoint tracking (6D autonomous system).
        State x = [px, pz, theta, vx, vz, omega]

        Controller drives (px, pz) -> waypoint using PD position control, pitch control from desired horizontal accel.
        """
        if covariance is None:
            covariance = 0.01 * torch.eye(6)
        else:
            covariance = torch.as_tensor(covariance, dtype=torch.float32)

        dist = torch.distributions.MultivariateNormal(torch.zeros(6), covariance)
        super().__init__(
            dim=6,
            state_labels=["px", "pz", "theta", "vx", "vz", "omega"],
            v_dist=dist,
        )

        self.dt = dt
        self.m, self.I, self.ell, self.g = m, I, ell, g
        self.c_v, self.c_w = c_v, c_w
        self.thrust_min, self.thrust_max = thrust_min, thrust_max
        self.cov = covariance

        # Fixed waypoint (parameter, not part of the state)
        if waypoint is None:
            waypoint = torch.zeros(2)
        self.waypoint = torch.as_tensor(waypoint, dtype=torch.float32).reshape(2,)

        # PD gains 
        self.kp_pos = torch.tensor([1.35, 1.35])  # [x, z]
        self.kd_pos = torch.tensor([0.55, 0.55])
        self.kp_theta = 3.5
        self.kd_theta = 2.25

        # Convenience: hover thrust per rotor (not used directly but useful bound)
        self.u_hover = torch.tensor([m * g / 2, m * g / 2])

    def set_waypoint(self, waypoint: torch.Tensor):
        self.waypoint = torch.as_tensor(waypoint, dtype=torch.float32).reshape(2,)

    def _dynamics(self, x: torch.Tensor, u: torch.Tensor):
        x = x.flatten()
        u = u.flatten()
        device, dtype = x.device, x.dtype

        px, pz, th, vx, vz, w = x
        u1, u2 = u
        T = u1 + u2
        tau = self.ell * (u2 - u1)

        dx = torch.zeros(6, device=device, dtype=dtype)
        dx[0] = vx
        dx[1] = vz
        dx[2] = w
        dx[3] = -(T / self.m) * torch.sin(th) - self.c_v * vx
        dx[4] = (T / self.m) * torch.cos(th) - self.g - self.c_v * vz
        dx[5] = (tau / self.I) - self.c_w * w
        return dx

    def _state_feedback(self, x: torch.Tensor):
        """
        State-feedback thrusts to move toward self.waypoint.
        Returns rotor thrusts u = [u1, u2] with saturation and non-negativity.
        """
        x = x.flatten()
        px, pz, th, vx, vz, w = x
        px_ref, pz_ref = self.waypoint

        # Position & velocity errors
        ex, ez = (px_ref - px), (pz_ref - pz)
        evx, evz = (-vx), (-vz)

        # Desired accelerations
        ax_des = self.kp_pos[0] * ex + self.kd_pos[0] * evx
        az_des = self.kp_pos[1] * ez + self.kd_pos[1] * evz + self.g  # add g so az_des = g at zero error

        # Desired pitch from horizontal accel (small-angle compatible, globally well-defined)
        theta_des = -torch.atan2(ax_des, az_des)

        # Inner-loop attitude control -> desired torque
        e_theta = theta_des - th
        e_w = -w
        tau_des = self.kp_theta * e_theta + self.kd_theta * e_w

        # Total thrust to realize resultant accel magnitude
        T_des = self.m * torch.sqrt(ax_des**2 + az_des**2)

        # Map to rotor thrusts
        u1 = 0.5 * (T_des - tau_des / self.ell)
        u2 = 0.5 * (T_des + tau_des / self.ell)
        u = torch.stack([u1, u2])

        # Enforce actuator limits and non-negativity
        u = torch.clamp(u, min=self.thrust_min, max=self.thrust_max)
        return u

    def next_state(self, x: torch.Tensor, v: torch.Tensor):
        u = self._state_feedback(x)
        dx = self._dynamics(x, u)
        x_next = x + self.dt * dx  # Euler step (matches style of your other systems)
        return x_next + v

    def predict(self, x: torch.Tensor):
        """
        Predict mean and covariance for additive Gaussian system.
        Returns (mean, covariance) where mean = f(x) = x + dt * dx(x, u(x))
        """
        u = self._state_feedback(x)
        dx = self._dynamics(x, u)
        mean = x + self.dt * dx
        return mean, self.cov


class SecondOrderDubinsTrailer(DiscreteTimeStochasticSystem):
    """
    6D second-order Dubins tractor–trailer with multiplicative noise in steering control, speed control, and the speed state itself 
    State x = [px, py, theta_c, theta_t, v, omega].
    """
    def __init__(
        self,
        dt: float,
        L_t: float = 1.0,
        v_ref: float = 1.0,
        k_v: float = 1.0,
        k_theta: float = 1.0,
        sigma_v: float = 0.1,
        sigma_omega: float = 0.1,
        sigma_speed_state: float = 0.1,
        cov_scale: float = 0.01,
    ):
        # Mixture-of-Gaussians non-Gaussian noise (2D): xi = [xi_v, xi_omega]
        cov1 = cov_scale * torch.tensor([[1.0, 0.2], [0.2, 1.0]], dtype=torch.float32)
        cov2 = cov_scale * torch.tensor([[1.0, -0.2], [-0.2, 1.0]], dtype=torch.float32)
        loc = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        mix = torch.distributions.Categorical(probs=torch.tensor([0.6, 0.4], dtype=torch.float32))
        comp = torch.distributions.MultivariateNormal(
            loc=loc,
            covariance_matrix=torch.stack([cov1, cov2]),
        )
        v_dist = torch.distributions.MixtureSameFamily(mix, comp)

        super().__init__(
            dim=6,
            state_labels=["px", "py", "theta_c", "theta_t", "v", "omega"],
            v_dist=v_dist,
        )

        self.dt = dt
        self.L_t = L_t

        # Control parameters
        self.v_ref = v_ref
        self.k_v = k_v
        self.k_theta = k_theta

        # Noise parameters
        self.sigma_v = sigma_v                # multiplicative noise in speed control
        self.sigma_omega = sigma_omega        # multiplicative noise in steering control
        self.sigma_speed_state = sigma_speed_state  # multiplicative noise directly on speed state

    def next_state(self, x: torch.Tensor, v: torch.Tensor):
        x = x.flatten()
        v = v.flatten()
        px, py, theta_c, theta_t, speed, omega = x[0], x[1], x[2], x[3], x[4], x[5]
        xi_v, xi_omega = v[0], v[1]

        dt = self.dt

        # --- Deterministic control laws ---

        # Speed control (toward v_ref)
        u_v = self.k_v * (self.v_ref - speed)

        # Steering rate control (oscillator)
        #   theta_c'' + k_theta * theta_c = 0 (continuous limit)
        u_omega = -self.k_theta * theta_c

        # --- Multiplicative noise on controls ---

        u_v_noisy = u_v * (1.0 + self.sigma_v * xi_v)
        u_omega_noisy = u_omega * (1.0 + self.sigma_omega * xi_omega)

        # --- Tractor–trailer kinematics ---

        px_next = px + dt * speed * torch.cos(theta_c)
        py_next = py + dt * speed * torch.sin(theta_c)
        theta_c_next = theta_c + dt * omega
        theta_t_next = theta_t + dt * (speed / self.L_t) * torch.sin(theta_c - theta_t)

        # --- Speed update: second-order + multiplicative noise in state ---
        #
        #   v_{k+1} = v_k + dt*u_v_noisy + (multiplicative state noise)
        #
        speed_next = (
            speed
            + dt * u_v_noisy
            + self.sigma_speed_state * speed * xi_v
        )

        # --- Steering rate update ---
        omega_next = omega + dt * u_omega_noisy

        return torch.stack(
            [px_next, py_next, theta_c_next, theta_t_next, speed_next, omega_next],
            dim=0,
        )


class Quadcopter(DiscreteTimeStochasticSystem):
    """
    12D quadcopter with closed-loop waypoint tracking (autonomous).
    State x = [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r]
        - positions (world), linear velocities (world),
          Euler angles ZYX = (roll=phi, pitch=theta, yaw=psi),
          body rates (p,q,r) in body frame.
    Control (internal): total thrust T and body torques tau = [tau_x, tau_y, tau_z].
    Noise: additive Gaussian (12D).
    """

    def __init__(
        self,
        dt: float,
        waypoint: torch.Tensor | None = None,
        yaw_ref: float = 0.0,
        m: float = 1.0,
        J: torch.Tensor | None = None,
        g: float = 9.81,
        c_v: float = 0.05,
        c_w: float = 0.05,
        thrust_min: float = 0.0,
        thrust_max: float = 20.0,
        torque_limits: torch.Tensor | None = None,
        covariance: torch.Tensor | None = None,
        rate_filter_alpha: float = 0.3,
    ):
        if covariance is None:
            covariance = 0.001 * torch.eye(12)
        else:
            covariance = torch.as_tensor(covariance, dtype=torch.float32)

        dist = torch.distributions.MultivariateNormal(torch.zeros(12), covariance)
        super().__init__(
            dim=12,
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
            v_dist=dist,
        )

        self.dt = dt
        self.m = m
        self.g = g
        self.c_v = c_v
        self.c_w = c_w
        self.cov = covariance
        self.yaw_ref = yaw_ref
        self.thrust_min = thrust_min
        self.thrust_max = thrust_max

        if J is None:
            J = torch.diag(torch.tensor([0.02, 0.02, 0.04], dtype=torch.float32))
        self.J = torch.as_tensor(J, dtype=torch.float32).reshape(3, 3)
        self.Jinv = torch.linalg.inv(self.J)

        if waypoint is None:
            waypoint = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        self.waypoint = torch.as_tensor(waypoint, dtype=torch.float32).reshape(3)

        if torque_limits is None:
            torque_limits = torch.tensor([1.0, 1.0, 0.5], dtype=torch.float32)
        self.torque_limits = torch.as_tensor(torque_limits, dtype=torch.float32).reshape(3)

        self.rate_filter_alpha = rate_filter_alpha
        self.filtered_rates = torch.zeros(3, dtype=torch.float32)

        self.kp_pos = torch.tensor([2.0, 2.0, 4.0], dtype=torch.float32)
        self.kd_pos = torch.tensor([1.2, 1.2, 2.0], dtype=torch.float32)
        self.kp_ang = torch.tensor([2.0, 2.0, 1.5], dtype=torch.float32)
        self.kd_ang = torch.tensor([1.5, 1.5, 0.5], dtype=torch.float32)

    def set_waypoint(self, waypoint: torch.Tensor):
        self.waypoint = torch.as_tensor(waypoint, dtype=torch.float32).reshape(3)

    @staticmethod
    def _rot_zyx(phi: torch.Tensor, theta: torch.Tensor, psi: torch.Tensor, device, dtype):
        """Rotation matrix R (world <- body) from ZYX Euler angles."""
        cphi, sphi = torch.cos(phi), torch.sin(phi)
        cth, sth = torch.cos(theta), torch.sin(theta)
        cpsi, spsi = torch.cos(psi), torch.sin(psi)

        Rz = torch.stack(
            [
                torch.stack([cpsi, -spsi, torch.zeros((), device=device, dtype=dtype)]),
                torch.stack([spsi, cpsi, torch.zeros((), device=device, dtype=dtype)]),
                torch.stack(
                    [
                        torch.zeros((), device=device, dtype=dtype),
                        torch.zeros((), device=device, dtype=dtype),
                        torch.ones((), device=device, dtype=dtype),
                    ]
                ),
            ]
        )
        Ry = torch.stack(
            [
                torch.stack([cth, torch.zeros((), device=device, dtype=dtype), sth]),
                torch.stack(
                    [
                        torch.zeros((), device=device, dtype=dtype),
                        torch.ones((), device=device, dtype=dtype),
                        torch.zeros((), device=device, dtype=dtype),
                    ]
                ),
                torch.stack([-sth, torch.zeros((), device=device, dtype=dtype), cth]),
            ]
        )
        Rx = torch.stack(
            [
                torch.stack(
                    [
                        torch.ones((), device=device, dtype=dtype),
                        torch.zeros((), device=device, dtype=dtype),
                        torch.zeros((), device=device, dtype=dtype),
                    ]
                ),
                torch.stack([torch.zeros((), device=device, dtype=dtype), cphi, -sphi]),
                torch.stack([torch.zeros((), device=device, dtype=dtype), sphi, cphi]),
            ]
        )
        return Rz @ Ry @ Rx

    @staticmethod
    def _euler_rate_matrix(phi: torch.Tensor, theta: torch.Tensor, device, dtype):
        """Map body rates Ω=[p,q,r] to Euler angle rates [phi_dot, theta_dot, psi_dot] (ZYX)."""
        cphi, sphi = torch.cos(phi), torch.sin(phi)
        cth, sth = torch.cos(theta), torch.sin(theta)
        cth_safe = torch.where(
            cth.abs() < 0.2,
            torch.where(cth >= 0, cth.new_tensor(0.2), cth.new_tensor(-0.2)),
            cth,
        )

        row0 = torch.stack(
            [
                torch.ones((), device=device, dtype=dtype),
                sphi * sth / cth_safe,
                cphi * sth / cth_safe,
            ]
        )
        row1 = torch.stack([torch.zeros((), device=device, dtype=dtype), cphi, -sphi])
        row2 = torch.stack(
            [torch.zeros((), device=device, dtype=dtype), sphi / cth_safe, cphi / cth_safe]
        )
        return torch.stack([row0, row1, row2])

    def _state_feedback(self, x: torch.Tensor):
        """Compute (T, tau) from state. Tracks self.waypoint at yaw = self.yaw_ref."""
        x = x.flatten()
        device, dtype = x.device, x.dtype

        wp = self.waypoint.to(device=device, dtype=dtype)
        kp_pos = self.kp_pos.to(device=device, dtype=dtype)
        kd_pos = self.kd_pos.to(device=device, dtype=dtype)
        kp_ang = self.kp_ang.to(device=device, dtype=dtype)
        kd_ang = self.kd_ang.to(device=device, dtype=dtype)
        tlim = self.torque_limits.to(device=device, dtype=dtype)

        p = x[0:3]
        v = x[3:6]
        phi, theta, psi = x[6], x[7], x[8]
        pqr = x[9:12]

        pos_err = wp - p
        vel_err = -v
        a_des = kp_pos * pos_err + kd_pos * vel_err

        yaw = x.new_tensor(self.yaw_ref)
        cpsi, spsi = torch.cos(yaw), torch.sin(yaw)
        ax, ay, az = a_des[0], a_des[1], a_des[2]
        g = x.new_tensor(self.g)
        phi_des = (ax * spsi - ay * cpsi) / g
        theta_des = (ax * cpsi + ay * spsi) / g
        T_des = self.m * (g + az)
        psi_des = yaw

        ang_err = torch.stack([phi_des - phi, theta_des - theta, psi_des - psi])
        ang_err = ang_err.clone()
        ang_err[2] = torch.remainder(ang_err[2] + math.pi, 2 * math.pi) - math.pi

        rate_err = -pqr
        tau = kp_ang * ang_err + kd_ang * rate_err

        T = torch.clamp(T_des, min=self.thrust_min, max=self.thrust_max)
        tau = torch.clamp(tau, min=-tlim, max=tlim)
        return T, tau

    def _dynamics(self, x: torch.Tensor, T: torch.Tensor, tau: torch.Tensor):
        x = x.flatten()
        device, dtype = x.device, x.dtype
        vx, vy, vz = x[3], x[4], x[5]
        phi, theta, psi = x[6], x[7], x[8]
        p, q, r = x[9], x[10], x[11]

        J = self.J.to(device=device, dtype=dtype)
        Jinv = self.Jinv.to(device=device, dtype=dtype)

        R = self._rot_zyx(phi, theta, psi, device, dtype)
        e3 = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        Omega = torch.stack([p, q, r])
        a_world = (T / self.m) * (R @ e3) - self.g * e3 - self.c_v * torch.stack([vx, vy, vz])
        J_omega = J @ Omega
        Omega_dot = Jinv @ (tau - torch.cross(Omega, J_omega) - self.c_w * Omega)
        E = self._euler_rate_matrix(phi, theta, device, dtype)
        euler_dot = E @ Omega

        xdot = torch.zeros(12, device=device, dtype=dtype)
        xdot[0:3] = torch.stack([vx, vy, vz])
        xdot[3:6] = a_world
        xdot[6:9] = euler_dot
        xdot[9:12] = Omega_dot
        return xdot

    def next_state(self, x: torch.Tensor, v: torch.Tensor):
        """Autonomous closed-loop: x_{k+1} = x_k + dt * f(x_k, u(x_k)) + v_k (forward Euler)."""
        x = x.flatten()
        v = v.flatten()
        T, tau = self._state_feedback(x)
        xdot = self._dynamics(x, T, tau)
        x_next = x + self.dt * xdot

        raw_rates = x_next[9:12]
        fr = self.filtered_rates.to(device=x.device, dtype=x.dtype)
        fr = self.rate_filter_alpha * raw_rates + (1 - self.rate_filter_alpha) * fr
        x_next = x_next.clone()
        x_next[9:12] = fr
        self.filtered_rates = fr.detach()

        x_next[6] = torch.remainder(x_next[6] + math.pi, 2 * math.pi) - math.pi
        x_next[7] = torch.remainder(x_next[7] + math.pi, 2 * math.pi) - math.pi
        x_next[8] = torch.remainder(x_next[8] + math.pi, 2 * math.pi) - math.pi

        return x_next + v

    def predict(self, x: torch.Tensor):
        """
        Predict mean and covariance for additive Gaussian noise.
        Mean matches the deterministic part of next_state (including rate filter and angle wrap).
        """
        x = x.flatten()
        T, tau = self._state_feedback(x)
        xdot = self._dynamics(x, T, tau)
        mean = x + self.dt * xdot

        fr = self.filtered_rates.to(device=x.device, dtype=x.dtype)
        mean = mean.clone()
        mean[9:12] = self.rate_filter_alpha * mean[9:12] + (1 - self.rate_filter_alpha) * fr

        mean[6] = torch.remainder(mean[6] + math.pi, 2 * math.pi) - math.pi
        mean[7] = torch.remainder(mean[7] + math.pi, 2 * math.pi) - math.pi
        mean[8] = torch.remainder(mean[8] + math.pi, 2 * math.pi) - math.pi
        return mean, self.cov