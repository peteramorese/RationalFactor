import torch

def make_mvnormal_state_sampler(mean: torch.Tensor, covariance: torch.Tensor):
    mean = torch.as_tensor(mean, dtype=torch.float32)
    covariance = torch.as_tensor(covariance, dtype=torch.float32)
    dist = torch.distributions.MultivariateNormal(mean, covariance)

    def sampler(n_samples: int) -> torch.Tensor:
        return dist.sample((n_samples,))

    return sampler


def make_unform_state_sampler(low: torch.Tensor, high: torch.Tensor):
    low = torch.as_tensor(low, dtype=torch.float32)
    high = torch.as_tensor(high, dtype=torch.float32)

    if low.shape != high.shape:
        raise ValueError("low and high must have the same shape")
    if not torch.all(high > low):
        raise ValueError("all high entries must be strictly greater than low entries")

    span = high - low

    def sampler(n_samples: int) -> torch.Tensor:
        noise = torch.rand((n_samples, low.numel()), dtype=low.dtype, device=low.device)
        return low + noise * span

    return sampler
