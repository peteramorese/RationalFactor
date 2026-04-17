import torch
from .base import PartiallyObservableSystem


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
