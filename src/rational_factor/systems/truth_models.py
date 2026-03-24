import torch
from .base import DiscreteTimeStochasticSystem


class VanDerPol(DiscreteTimeStochasticSystem):
    def __init__(self, dt: float, mu: float = 1.0, covariance: torch.Tensor | None = None):
        """
        Van der Pol oscillator with additive Gaussian noise.

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

        def additive_gaussian():
            return dist.sample()

        super().__init__(dim=2, v_dist=additive_gaussian)

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
        Inputs are internal (no extra args to next_state), so the system is autonomous.
        """
        if covariance is None:
            covariance = 0.01 * torch.eye(6)
        else:
            covariance = torch.as_tensor(covariance, dtype=torch.float32)

        dist = torch.distributions.MultivariateNormal(torch.zeros(6), covariance)

        def additive_gaussian():
            return dist.sample()

        super().__init__(dim=6, v_dist=additive_gaussian)

        self.dt = dt
        self.m, self.I, self.ell, self.g = m, I, ell, g
        self.c_v, self.c_w = c_v, c_w
        self.thrust_min, self.thrust_max = thrust_min, thrust_max
        self.cov = covariance

        # Fixed waypoint (parameter, not part of the state)
        if waypoint is None:
            waypoint = torch.zeros(2)
        self.waypoint = torch.as_tensor(waypoint, dtype=torch.float32).reshape(2,)

        # PD gains (tune as needed)
        self.kp_pos = torch.tensor([2.0, 2.0])  # [x, z]
        self.kd_pos = torch.tensor([1.0, 1.0])
        self.kp_theta = 5.0
        self.kd_theta = 3.0

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
        """
        Autonomous closed-loop: x_{k+1} = f(x_k) + v_k
        """
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