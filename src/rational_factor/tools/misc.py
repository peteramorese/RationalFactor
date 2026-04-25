import torch

def make_mvnormal_state_sampler(mean: torch.Tensor, covariance: torch.Tensor):
    mean = torch.as_tensor(mean, dtype=torch.float32)
    covariance = torch.as_tensor(covariance, dtype=torch.float32)
    dist = torch.distributions.MultivariateNormal(mean, covariance)

    def sampler(n_samples: int) -> torch.Tensor:
        return dist.sample((n_samples,))

    return sampler


def make_uniform_state_sampler(low: torch.Tensor, high: torch.Tensor):
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

def data_bounds(data: torch.Tensor, mode: str = "center_lengths") -> tuple[torch.Tensor, torch.Tensor]:
    """Smallest axis-aligned box containing all rows of ``data`` (shape ``(n, d)``).

    ``mode``:
        ``"min_max"`` — return ``(data_min, data_max)`` per dimension.
        ``"center_lengths"`` — return ``(center, lengths)`` where
        ``center = (min + max) / 2`` and ``lengths = max - min`` per dimension.
    """
    x = torch.as_tensor(data)
    data_min = x.min(dim=0).values
    data_max = x.max(dim=0).values
    if mode == "min_max":
        return data_min, data_max
    if mode == "center_lengths":
        return 0.5 * (data_min + data_max), data_max - data_min
    raise ValueError(f"mode must be 'min_max' or 'center_lengths', got {mode!r}")

def data_mean_std(data: torch.Tensor, n_sigma : float = 3.0) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.as_tensor(data)
    return x.mean(dim=0), n_sigma * x.std(dim=0)