import torch
from dataclasses import dataclass

@dataclass
class WeightedParticleSet:
    particles: torch.Tensor   # shape: (N, state_dim)
    weights: torch.Tensor     # shape: (N,)

    def __post_init__(self):
        assert self.particles.ndim == 2, "particles must have shape (N, dim)"
        assert self.weights.ndim == 1, "weights must have shape (N,)"
        assert self.particles.shape[0] == self.weights.shape[0], \
            "particles and weights must have the same number of rows"
        self.normalize_weights()

    @property
    def n_particles(self) -> int:
        return self.particles.shape[0]

    @property
    def dim(self) -> int:
        return self.particles.shape[1]

    def clone(self):
        return WeightedParticleSet(
            particles=self.particles.clone(),
            weights=self.weights.clone()
        )

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
        idx = torch.multinomial(self.weights, self.n_particles, replacement=True)
        self.particles = self.particles[idx]
        self.weights = torch.ones_like(self.weights) / self.n_particles

    def mean(self) -> torch.Tensor:
        return torch.sum(self.weights[:, None] * self.particles, dim=0)