import math
import torch
from rational_factor.models.density_model import DensityModel


class ParticleSet(DensityModel):
    def __init__(self, particles: torch.Tensor):
        super().__init__(particles.shape[1])
        self.particles = particles

    def clone(self):
        return ParticleSet(particles=self.particles.clone())
    
    def marginal(self, marginal_dims: tuple[int, ...]):
        marginal_particles = self.particles[:, marginal_dims]
        return ParticleSet(particles=marginal_particles)
    
    def sample(self, n_samples: int):
        idx = torch.randint(
            low=0,
            high=self.n_particles(),
            size=(n_samples,),
            device=self.particles.device,
        )
        return self.particles[idx]

    def _bandwidth(self) -> torch.Tensor:
        """
        Scott's rule with per-dimension bandwidths:
            h_j = sigma_j * n^(-1/(d+4))

        Returns:
            h: (d,)
        """
        n, d = self.particles.shape
        if n <= 1:
            # fallback if only one particle
            return torch.ones(
                d, device=self.particles.device, dtype=self.particles.dtype
            )

        std = self.particles.std(dim=0, unbiased=True)
        factor = n ** (-1.0 / (d + 4))

        # Avoid zero bandwidth in degenerate dimensions
        eps = torch.finfo(self.particles.dtype).eps
        h = torch.clamp(std * factor, min=eps)
        return h

    def log_density(self, x: torch.Tensor):
        """
        Standard Gaussian KDE with diagonal bandwidth.
        """
        assert x.ndim == 2 and x.shape[1] == self.dim, \
            f"x must have shape (n_data, {self.dim})"

        particles = self.particles 
        n, d = particles.shape
        h = self._bandwidth()

        # z: (m, n, d)
        z = (x[:, None, :] - particles[None, :, :]) / h[None, None, :]

        # log Gaussian kernel per particle:
        log_kernel = (
            -0.5 * torch.sum(z ** 2, dim=-1)
            - 0.5 * d * math.log(2.0 * math.pi)
            - torch.sum(torch.log(h))
        ) 

        # Uniform mixture over particles
        return torch.logsumexp(log_kernel, dim=1) - math.log(n)

    def n_particles(self) -> int:
        return self.particles.shape[0]

    def mean(self) -> torch.Tensor:
        return self.particles.mean(dim=0)


class WeightedParticleSet(ParticleSet):
    def __init__(self, particles: torch.Tensor, weights: torch.Tensor, resample: bool = True):
        self.weights = weights
        super().__init__(particles)
        self.normalize_weights()
        self.resample = resample

    def clone(self):
        return WeightedParticleSet(
            particles=self.particles.clone(),
            weights=self.weights.clone()
        )
    
    def marginal(self, marginal_dims: tuple[int, ...]):
        marginal_particles = self.particles[:, marginal_dims]
        return WeightedParticleSet(
            particles=marginal_particles,
            weights=self.weights.clone(),
        )
    
    def sample(self, n_samples: int):
        idx = torch.multinomial(self.weights, n_samples, replacement=True)
        return self.particles[idx]

    def _bandwidth(self) -> torch.Tensor:
        """
        Weighted Scott-style bandwidth using effective sample size:
            h_j = sigma_j * neff^(-1/(d+4))

        where sigma_j is the weighted std per dimension.
        """
        n, d = self.particles.shape
        w = self.weights

        neff = self.effective_sample_size()

        if n <= 1 or neff <= 1:
            return torch.ones(
                d, device=self.particles.device, dtype=self.particles.dtype
            )

        mean = torch.sum(w[:, None] * self.particles, dim=0)  # (d,)
        var = torch.sum(w[:, None] * (self.particles - mean) ** 2, dim=0)
        std = torch.sqrt(var)

        factor = neff ** (-1.0 / (d + 4))

        eps = torch.finfo(self.particles.dtype).eps
        h = torch.clamp(std * factor, min=eps)
        return h
    
    def log_density(self, x: torch.Tensor):
        """
        Weighted Gaussian KDE with diagonal bandwidth.

        Args:
            x: (m, d)

        Returns:
            log p(x): (m,)
        """
        assert x.ndim == 2 and x.shape[1] == self.dim, \
            f"x must have shape (n_data, {self.dim})"

        particles = self.particles
        w = self.weights
        _, d = particles.shape
        h = self._bandwidth()

        z = (x[:, None, :] - particles[None, :, :]) / h[None, None, :]  # (m, n, d)

        log_kernel = (
            -0.5 * torch.sum(z ** 2, dim=-1)
            - 0.5 * d * math.log(2.0 * math.pi)
            - torch.sum(torch.log(h))
        ) 

        log_w = torch.log(torch.clamp(w, min=torch.finfo(w.dtype).tiny))
        return torch.logsumexp(log_kernel + log_w[None, :], dim=1)

    def normalize_weights(self):
        s = self.weights.sum()
        if (not torch.isfinite(s)) or s <= 0:
            self.weights = torch.ones_like(self.weights) / self.weights.numel()
        else:
            self.weights = self.weights / s

    def effective_sample_size(self) -> torch.Tensor:
        return 1.0 / torch.sum(self.weights ** 2)

    def resample(self):
        """
        Multinomial resampling.
        """
        if not self.resample:
            return
        idx = torch.multinomial(self.weights, self.n_particles(), replacement=True)
        self.particles = self.particles[idx]
        self.weights = torch.ones_like(self.weights) / self.n_particles()

    def mean(self) -> torch.Tensor:
        return torch.sum(self.weights[:, None] * self.particles, dim=0)