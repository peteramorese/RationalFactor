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


def train_test_split(
    *arrays: torch.Tensor,
    test_size: float = 0.2,
    shuffle: bool = True,
    seed: int | None = None,
):
    """
    Split one or more tensors along the first dimension into train/test parts.

    Returns:
        If one array is provided: (train, test)
        If multiple arrays are provided: (a0_train, a0_test, a1_train, a1_test, ...)
    """
    if len(arrays) == 0:
        raise ValueError("At least one array must be provided")
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must satisfy 0 < test_size < 1")

    tensors = [torch.as_tensor(a) for a in arrays]
    n = tensors[0].shape[0]
    if n < 2:
        raise ValueError("Need at least 2 samples to split")
    if any(t.shape[0] != n for t in tensors):
        raise ValueError("All arrays must have the same number of samples (axis 0)")

    n_test = max(1, int(round(test_size * n)))
    n_test = min(n - 1, n_test)
    n_train = n - n_test

    if shuffle:
        if seed is None:
            perm = torch.randperm(n, device=tensors[0].device)
        else:
            gen = torch.Generator(device=tensors[0].device.type)
            gen.manual_seed(seed)
            perm = torch.randperm(n, generator=gen, device=tensors[0].device)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]
    else:
        train_idx = torch.arange(n_train, device=tensors[0].device)
        test_idx = torch.arange(n_train, n, device=tensors[0].device)

    split = []
    for t in tensors:
        split.extend([t[train_idx], t[test_idx]])

    if len(arrays) == 1:
        return split[0], split[1]
    return tuple(split)