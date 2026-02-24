import torch
from .base import DiscreteTimeStochasticSystem

class VanDerPol(DiscreteTimeStochasticSystem):
    def __init__(self, dt : float, mu : float = 1.0, covariance : torch.Tensor | None = None):
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

    def next_state(self, x : torch.Tensor, v : torch.Tensor):
        x1, x2 = x[0], x[1]
        dx1 = x2
        dx2 = self.mu * (1 - x1**2) * x2 - x1

        x1_next = x1 + self.dt * dx1
        x2_next = x2 + self.dt * dx2

        return torch.stack([x1_next, x2_next]) + v

    # Methods for matching GP model
    def predict(self, x : torch.Tensor):
        x_flat = x.flatten()
        x1, x2 = x_flat[0], x_flat[1]
        dx1 = x2
        dx2 = self.mu * (1 - x1**2) * x2 - x1

        x1_next = x1 + self.dt * dx1
        x2_next = x2 + self.dt * dx2

        return torch.stack([x1_next, x2_next]), self.cov

    def jacobian(self, x : torch.Tensor):
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

    def hessian_tensor(self, x : torch.Tensor):
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