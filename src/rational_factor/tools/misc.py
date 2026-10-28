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


def init_gaussian_basis_params_gmm(
    data: torch.Tensor,
    d: int,
    n_local: int,
    *,
    reg_covar: float = 1e-5,
    max_iter: int = 100,
    random_state: int | None = 0,
    epsilon: float = 0.0,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
):
    r"""Initialize trainable Gaussian basis parameters using diagonal-GMM EM.

    Fit one ``n_local``-component, ``d``-dimensional diagonal-covariance GMM
    (as in ``train.fit_gaussian_lf_em``) and extract its component means and
    marginal standard deviations into tensors of shape ``(1, d, n_local)``.
    These are suitable for ``GaussianBasis`` and for the local primitive basis
    of a ``TTBasis``; the full tensor-product basis is never constructed.

    Returns
    -------
    (TrainableParameters, PositiveParameters)
        Separate *trainable* mean and std providers, ready to pass to
        ``GaussianBasis(mean_params=..., std_params=...)``.

    Notes
    -----
    PositiveParameters applies ``epsilon + softplus(raw)`` when trainable.
    Following the requested initialization policy, the EM standard deviations
    are passed directly via ``from_values(trainable=True)`` WITHOUT applying an
    inverse softplus. Thus the initial *effective* standard deviations are
    larger than those returned by EM; the means are initialized exactly.

    GMM mixture weights are intentionally not returned: in a TT basis the
    coefficients are trained separately, and ``n_local`` specifies the number
    of 1-D Gaussian factors in every state dimension.
    """
    from sklearn.mixture import GaussianMixture
    from rational_factor.models.parameters import TrainableParameters, PositiveParameters

    x = torch.as_tensor(data)
    if x.ndim != 2 or not x.is_floating_point() or x.shape[1] != d:
        raise ValueError(f"data must be a floating tensor of shape (N, {d})")
    if not isinstance(d, int) or d < 1:
        raise ValueError("d must be a positive integer")
    if not isinstance(n_local, int) or n_local < 1:
        raise ValueError("n_local must be a positive integer")
    if x.shape[0] < n_local:
        raise ValueError(f"GMM with {n_local} components needs at least {n_local} samples; got {x.shape[0]}")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("data must contain only finite values")
    if reg_covar < 0 or max_iter < 1 or epsilon < 0:
        raise ValueError("require reg_covar >= 0, max_iter >= 1, epsilon >= 0")

    out_dtype = x.dtype if dtype is None else dtype
    out_device = x.device if device is None else torch.device(device)
    mixture = GaussianMixture(
        n_components=n_local,
        covariance_type="diag",
        reg_covar=reg_covar,
        max_iter=max_iter,
        random_state=random_state,
        init_params="kmeans",
    )
    mixture.fit(x.detach().cpu().numpy())

    means = torch.as_tensor(mixture.means_, dtype=out_dtype, device=out_device).T.contiguous().unsqueeze(0)
    variances = torch.as_tensor(mixture.covariances_, dtype=out_dtype, device=out_device)
    stds = variances.clamp_min(torch.finfo(out_dtype).eps).sqrt().T.contiguous().unsqueeze(0)

    mean_params = TrainableParameters.from_values(means, trainable=True)
    std_params = PositiveParameters.from_values(stds, trainable=True, epsilon=epsilon)
    return mean_params, std_params
