import torch
from rational_factor.models.density_model import ConditionalDensityModel, DensityModel

from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, ReversePermutation
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform


class NormalizingFlow(DensityModel):
    def __init__(self, dim: int, num_layers: int = 5, hidden_features: int = 64):
        super().__init__(dim=dim)

        transforms = []
        for _ in range(num_layers):
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=dim,
                    hidden_features=hidden_features,
                    num_blocks=2,
                    use_batch_norm=False,
                )
            )
            transforms.append(ReversePermutation(features=dim))

        transform = CompositeTransform(transforms)
        base_distribution = StandardNormal(shape=[dim])
        self.flow = Flow(transform=transform, distribution=base_distribution)

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.flow.log_prob(inputs=x)

    def sample(self, n_samples: int) -> torch.Tensor:
        return self.flow.sample(num_samples=n_samples)


class ConditionalNormalizingFlow(ConditionalDensityModel):
    def __init__(self, dim: int, conditioner_dim: int, num_layers: int = 5, hidden_features: int = 64):
        super().__init__(dim=dim, conditioner_dim=conditioner_dim)

        transforms = []
        for _ in range(num_layers):
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=dim,
                    hidden_features=hidden_features,
                    context_features=conditioner_dim,
                    num_blocks=2,
                    use_batch_norm=False,
                )
            )
            transforms.append(ReversePermutation(features=dim))

        transform = CompositeTransform(transforms)
        base_distribution = StandardNormal(shape=[dim])
        self.flow = Flow(transform=transform, distribution=base_distribution)

    def log_density(self, x: torch.Tensor, *, conditioner: torch.Tensor) -> torch.Tensor:
        return self.flow.log_prob(inputs=x, context=conditioner)
    
    def sample(self, conditioner: torch.Tensor, num_samples_per: int = 1) -> torch.Tensor:
        if conditioner.ndim == 1:
            conditioner = conditioner.unsqueeze(0)
        elif conditioner.ndim > 2:
            conditioner = conditioner.view(-1, conditioner.shape[-1])

        x_samples = self.flow.sample(num_samples=num_samples_per, context=conditioner)

        if num_samples_per == 1:
            x_samples = x_samples[:, 0, :]

        return x_samples

