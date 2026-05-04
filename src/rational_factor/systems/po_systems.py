import torch
from .base import PartiallyObservableSystem
from . import systems


class PartiallyObservableVanDerPol(PartiallyObservableSystem):
    def __init__(self, dt: float, mu: float = 1.0, process_covariance: torch.Tensor = None, observation_covariance: torch.Tensor = None):
        """
        Van der Pol oscillator with additive Gaussian noise.

        Args:
            dt : time step
            mu : nonlinearity parameter
            covariance : 2x2 covariance matrix for process noise (default: identity)
        """
        if process_covariance is None:
            process_covariance = torch.eye(2)
        else:
            process_covariance = torch.as_tensor(process_covariance, dtype=torch.float32)

        if observation_covariance is None:
            observation_covariance = torch.eye(2)
        else:
            observation_covariance = torch.as_tensor(observation_covariance, dtype=torch.float32)

        process_dist = torch.distributions.MultivariateNormal(torch.zeros(2), process_covariance)
        observation_dist = torch.distributions.MultivariateNormal(torch.zeros(2), observation_covariance)

        super().__init__(
            state_dim=2,
            observation_dim=2,
            v_dist=process_dist,
            w_dist=observation_dist,
        )

        self.dt = dt
        self.mu = mu
        self.process_cov = process_covariance
        self.observation_cov = observation_covariance

    def next_state(self, x: torch.Tensor, v: torch.Tensor):
        x1, x2 = x[0], x[1]
        dx1 = x2
        dx2 = self.mu * (1 - x1**2) * x2 - x1

        x1_next = x1 + self.dt * dx1
        x2_next = x2 + self.dt * dx2

        return torch.stack([x1_next, x2_next]) + v

    def observe(self, x : torch.Tensor):
        w = self._sample_w()
        return x + w.to(device=x.device, dtype=x.dtype)
    
    def log_observation_likelihood(self, x : torch.Tensor, o : torch.Tensor):
        observation_cov = self.observation_cov.to(device=x.device, dtype=x.dtype)
        return torch.distributions.MultivariateNormal(x, observation_cov).log_prob(o)

    def observation_likelihood(self, x : torch.Tensor, o : torch.Tensor):
        return torch.exp(self.log_observation_likelihood(x, o))

class BadSensorVanDerPol(PartiallyObservableSystem):
    def __init__(self, dt: float, mu: float = 1.0, process_covariance: torch.Tensor = None, observation_covariance: torch.Tensor = None):
        """
        Van der Pol oscillator with additive Gaussian noise.

        Args:
            dt : time step
            mu : nonlinearity parameter
            covariance : 2x2 covariance matrix for process noise (default: identity)
        """
        if process_covariance is None:
            process_covariance = torch.eye(2)
        else:
            process_covariance = torch.as_tensor(process_covariance, dtype=torch.float32)

        if observation_covariance is None:
            observation_covariance = torch.eye(2)
        else:
            observation_covariance = torch.as_tensor(observation_covariance, dtype=torch.float32)

        process_dist = torch.distributions.MultivariateNormal(torch.zeros(2), process_covariance)
        observation_dist = torch.distributions.MultivariateNormal(torch.zeros(2), observation_covariance)

        super().__init__(
            state_dim=2,
            observation_dim=2,
            v_dist=process_dist,
            w_dist=observation_dist,
        )

        self.dt = dt
        self.mu = mu
        self.process_cov = process_covariance
        self.observation_cov = observation_covariance

    def next_state(self, x: torch.Tensor, v: torch.Tensor):
        x1, x2 = x[0], x[1]
        dx1 = x2
        dx2 = self.mu * (1 - x1**2) * x2 - x1

        x1_next = x1 + self.dt * dx1
        x2_next = x2 + self.dt * dx2

        return torch.stack([x1_next, x2_next]) + v

    def _observation_mode_means(self, x: torch.Tensor):
        x1 = x[..., 0]
        x2 = x[..., 1]

        # Nonlinear baseline sensor distortion.
        baseline = torch.stack(
            [
                x1 + 0.25 * torch.sin(1.4 * x2),
                x2 + 0.2 * x1**2,
            ],
            dim=-1,
        )

        # State-dependent nonlinear mode split controls bimodality.
        mode_offset = torch.stack(
            [
                0.8 * torch.tanh(0.6 * x1),
                0.5 * torch.sin(x1 * x2),
            ],
            dim=-1,
        )

        mean_mode_a = baseline - mode_offset
        mean_mode_b = baseline + mode_offset
        return mean_mode_a, mean_mode_b

    def observe(self, x : torch.Tensor):
        mean_mode_a, mean_mode_b = self._observation_mode_means(x)

        mode_prob = torch.full(mean_mode_a.shape[:-1], 0.5, device=x.device, dtype=x.dtype)
        mode_selector = torch.bernoulli(mode_prob).unsqueeze(-1)
        mean = mode_selector * mean_mode_b + (1.0 - mode_selector) * mean_mode_a

        w = self._sample_w().to(device=x.device, dtype=x.dtype)
        return mean + w
    
    def log_observation_likelihood(self, x : torch.Tensor, o : torch.Tensor):
        observation_cov = self.observation_cov.to(device=x.device, dtype=x.dtype)
        mean_mode_a, mean_mode_b = self._observation_mode_means(x)

        dist_a = torch.distributions.MultivariateNormal(mean_mode_a, observation_cov)
        dist_b = torch.distributions.MultivariateNormal(mean_mode_b, observation_cov)
        log_prob_a = dist_a.log_prob(o)
        log_prob_b = dist_b.log_prob(o)

        mixture_logs = torch.stack([log_prob_a, log_prob_b], dim=0)
        return torch.logsumexp(mixture_logs, dim=0) - torch.log(
            torch.tensor(2.0, device=x.device, dtype=x.dtype)
        )

    def observation_likelihood(self, x : torch.Tensor, o : torch.Tensor):
        return torch.exp(self.log_observation_likelihood(x, o))


class PartiallyObservableCartPole(PartiallyObservableSystem):
    def __init__(
        self,
        dt: float,
        m_c: float = 1.0,
        m_p: float = 0.1,
        l: float = 0.5,
        g: float = 9.81,
        process_covariance: torch.Tensor = None,
        observation_covariance: torch.Tensor = None,
    ):
        if process_covariance is None:
            process_covariance = 0.1 * torch.eye(4)
        else:
            process_covariance = torch.as_tensor(process_covariance, dtype=torch.float32)

        if observation_covariance is None:
            observation_covariance = 0.1 * torch.eye(4)
        else:
            observation_covariance = torch.as_tensor(observation_covariance, dtype=torch.float32)

        self.state_system = systems.CartPole(
            dt=dt,
            m_c=m_c,
            m_p=m_p,
            l=l,
            g=g,
            covariance=process_covariance,
        )
        observation_dist = torch.distributions.MultivariateNormal(torch.zeros(4), observation_covariance)

        super().__init__(
            state_dim=4,
            observation_dim=4,
            v_dist=self.state_system._v_dist,
            w_dist=observation_dist,
        )
        self.observation_cov = observation_covariance

    def next_state(self, x: torch.Tensor, v: torch.Tensor):
        return self.state_system.next_state(x, v)

    def _observation_mean(self, x: torch.Tensor):
        p = x[..., 0]
        p_dot = x[..., 1]
        theta = x[..., 2]
        theta_dot = x[..., 3]
        return torch.stack(
            [
                p + 0.15 * torch.sin(theta),
                0.7 * p_dot + 0.25 * torch.tanh(p * theta),
                torch.sin(theta) + 0.12 * p_dot * theta_dot,
                theta_dot + 0.1 * p**2 - 0.08 * torch.cos(theta),
            ],
            dim=-1,
        )

    def observe(self, x: torch.Tensor):
        w = self._sample_w().to(device=x.device, dtype=x.dtype)
        return self._observation_mean(x) + w

    def log_observation_likelihood(self, x: torch.Tensor, o: torch.Tensor):
        observation_cov = self.observation_cov.to(device=x.device, dtype=x.dtype)
        return torch.distributions.MultivariateNormal(self._observation_mean(x), observation_cov).log_prob(o)

    def observation_likelihood(self, x: torch.Tensor, o: torch.Tensor):
        return torch.exp(self.log_observation_likelihood(x, o))


class PartiallyObservableDubinsTrailer(PartiallyObservableSystem):
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
        observation_covariance: torch.Tensor = None,
    ):
        self.state_system = systems.SecondOrderDubinsTrailer(
            dt=dt,
            L_t=L_t,
            v_ref=v_ref,
            k_v=k_v,
            k_theta=k_theta,
            sigma_v=sigma_v,
            sigma_omega=sigma_omega,
            sigma_speed_state=sigma_speed_state,
            cov_scale=cov_scale,
        )

        if observation_covariance is None:
            observation_covariance = 0.1 * torch.eye(6)
        else:
            observation_covariance = torch.as_tensor(observation_covariance, dtype=torch.float32)

        observation_dist = torch.distributions.MultivariateNormal(torch.zeros(6), observation_covariance)

        super().__init__(
            state_dim=6,
            observation_dim=6,
            v_dist=self.state_system._v_dist,
            w_dist=observation_dist,
        )
        self.observation_cov = observation_covariance

    def next_state(self, x: torch.Tensor, v: torch.Tensor):
        return self.state_system.next_state(x, v)

    def _observation_mean(self, x: torch.Tensor):
        px = x[..., 0]
        py = x[..., 1]
        theta_c = x[..., 2]
        theta_t = x[..., 3]
        speed = x[..., 4]
        omega = x[..., 5]
        return torch.stack(
            [
                px + 0.1 * speed * torch.cos(theta_c),
                py + 0.1 * speed * torch.sin(theta_c),
                torch.sin(theta_c) + 0.08 * omega,
                torch.sin(theta_t) + 0.08 * (theta_c - theta_t) ** 2,
                speed + 0.05 * omega**2,
                omega + 0.12 * torch.tanh(speed),
            ],
            dim=-1,
        )

    def observe(self, x: torch.Tensor):
        w = self._sample_w().to(device=x.device, dtype=x.dtype)
        return self._observation_mean(x) + w

    def log_observation_likelihood(self, x: torch.Tensor, o: torch.Tensor):
        observation_cov = self.observation_cov.to(device=x.device, dtype=x.dtype)
        return torch.distributions.MultivariateNormal(self._observation_mean(x), observation_cov).log_prob(o)

    def observation_likelihood(self, x: torch.Tensor, o: torch.Tensor):
        return torch.exp(self.log_observation_likelihood(x, o))


class PartiallyObservableAircraft(PartiallyObservableSystem):
    def __init__(
        self,
        dt: float,
        waypoint: torch.Tensor | None = None,
        V_ref: float = 22.0,
        sigma_thrust: float = 0.04,
        sigma_surface: float = 0.05,
        sigma_CL: float = 0.03,
        sigma_CM: float = 0.04,
        noise_cov_scale: float = 1.0,
        state_scale: torch.Tensor | None = None,
        observation_covariance: torch.Tensor = None,
    ):
        self.state_system = systems.Aircraft(
            dt=dt,
            waypoint=waypoint,
            V_ref=V_ref,
            sigma_thrust=sigma_thrust,
            sigma_surface=sigma_surface,
            sigma_CL=sigma_CL,
            sigma_CM=sigma_CM,
            noise_cov_scale=noise_cov_scale,
            state_scale=state_scale,
        )

        if observation_covariance is None:
            observation_covariance = 0.1 * torch.eye(12)
        else:
            observation_covariance = torch.as_tensor(observation_covariance, dtype=torch.float32)

        observation_dist = torch.distributions.MultivariateNormal(torch.zeros(12), observation_covariance)
        super().__init__(
            state_dim=12,
            observation_dim=12,
            v_dist=self.state_system._v_dist,
            w_dist=observation_dist,
        )
        self.observation_cov = observation_covariance

    def next_state(self, x: torch.Tensor, v: torch.Tensor):
        return self.state_system.next_state(x, v)

    def _observation_mean(self, x: torch.Tensor):
        obs = x.clone()
        obs[..., 0] = x[..., 0] + 0.04 * torch.sin(0.7 * x[..., 8]) + 0.02 * x[..., 3] * x[..., 6]
        obs[..., 1] = x[..., 1] + 0.04 * torch.cos(0.6 * x[..., 8]) - 0.02 * x[..., 4] * x[..., 7]
        obs[..., 2] = x[..., 2] + 0.03 * x[..., 5] ** 2
        obs[..., 3] = x[..., 3] + 0.08 * torch.tanh(x[..., 7])
        obs[..., 4] = x[..., 4] + 0.08 * torch.sin(x[..., 6])
        obs[..., 5] = x[..., 5] + 0.05 * torch.tanh(x[..., 10])
        obs[..., 6] = torch.sin(x[..., 6]) + 0.03 * x[..., 9]
        obs[..., 7] = torch.sin(x[..., 7]) + 0.03 * x[..., 10]
        obs[..., 8] = torch.sin(x[..., 8]) + 0.02 * x[..., 11]
        obs[..., 9] = x[..., 9] + 0.04 * x[..., 3] * x[..., 7]
        obs[..., 10] = x[..., 10] + 0.04 * x[..., 4] * x[..., 6]
        obs[..., 11] = x[..., 11] + 0.04 * torch.tanh(x[..., 5])
        return obs

    def observe(self, x: torch.Tensor):
        w = self._sample_w().to(device=x.device, dtype=x.dtype)
        return self._observation_mean(x) + w

    def log_observation_likelihood(self, x: torch.Tensor, o: torch.Tensor):
        observation_cov = self.observation_cov.to(device=x.device, dtype=x.dtype)
        return torch.distributions.MultivariateNormal(self._observation_mean(x), observation_cov).log_prob(o)

    def observation_likelihood(self, x: torch.Tensor, o: torch.Tensor):
        return torch.exp(self.log_observation_likelihood(x, o))
