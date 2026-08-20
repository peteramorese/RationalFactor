import torch


class LowRankPlusDiagonal:
    """Batched low-rank-plus-diagonal matrix ``M = diag(D) + Uᵀ V``.

    Factors are stored with the thin axis first:

    - ``D``: ``(..., n)`` diagonal entries
    - ``U``, ``V``: ``(..., k, n)`` with ``k << n``

    Matrix–vector products use ``Mx = D ⊙ x + Uᵀ (V x)`` and cost ``O(n k)``
    per batch element (linear in ``n`` for fixed rank ``k``), never forming the
    dense ``n × n`` matrix.
    """

    def __init__(self, D: torch.Tensor, U: torch.Tensor, V: torch.Tensor):
        D = torch.as_tensor(D)
        U = torch.as_tensor(U)
        V = torch.as_tensor(V)

        if D.dim() < 1:
            raise ValueError("D must have shape (..., n)")
        if U.dim() < 2 or V.dim() < 2:
            raise ValueError("U and V must have shape (..., k, n)")
        if U.shape != V.shape:
            raise ValueError(f"U and V must have the same shape, got {tuple(U.shape)} and {tuple(V.shape)}")
        if U.shape[:-2] != D.shape[:-1]:
            raise ValueError(
                f"batch shapes of D and U/V must match, got D[..., n]={tuple(D.shape)} "
                f"and U/V[..., k, n]={tuple(U.shape)}"
            )
        if U.shape[-1] != D.shape[-1]:
            raise ValueError(
                f"trailing size of D and U/V must match (n), got n={D.shape[-1]} and U/V n={U.shape[-1]}"
            )
        if D.dtype != U.dtype or D.dtype != V.dtype:
            raise ValueError("D, U, and V must share the same dtype")
        if D.device != U.device or D.device != V.device:
            raise ValueError("D, U, and V must share the same device")

        self.D = D
        self.U = U
        self.V = V

    @property
    def n(self) -> int:
        return self.D.shape[-1]

    @property
    def k(self) -> int:
        return self.U.shape[-2]

    @property
    def batch_shape(self) -> torch.Size:
        return self.D.shape[:-1]

    @property
    def shape(self) -> torch.Size:
        return self.batch_shape + torch.Size([self.n, self.n])

    @property
    def dtype(self) -> torch.dtype:
        return self.D.dtype

    @property
    def device(self) -> torch.device:
        return self.D.device

    @property
    def T(self) -> "LowRankPlusDiagonal":
        """Transpose: ``(diag(D) + Uᵀ V)ᵀ = diag(D) + Vᵀ U``."""
        return LowRankPlusDiagonal(self.D, self.V, self.U)

    def to(self, *args, **kwargs) -> "LowRankPlusDiagonal":
        return LowRankPlusDiagonal(
            self.D.to(*args, **kwargs),
            self.U.to(*args, **kwargs),
            self.V.to(*args, **kwargs),
        )

    def to_dense(self) -> torch.Tensor:
        """Materialize the full ``(..., n, n)`` matrix (``O(n² k)``)."""
        return torch.diag_embed(self.D) + self.U.transpose(-2, -1) @ self.V

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``M @ x`` in ``O(n k)`` time (per batch / RHS).

        Args:
            x: ``(..., n)`` or ``(..., n, m)``. Leading dims broadcast with
                ``batch_shape``.
        """
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        n = self.n
        batch_ndim = self.D.dim() - 1
        # Multi-RHS: (..., n, m) — one more axis than a batched vector.
        if x.dim() >= batch_ndim + 2 and x.shape[-2] == n:
            Vx = torch.einsum("...ki,...im->...km", self.V, x)
            return self.D.unsqueeze(-1) * x + torch.einsum("...ki,...km->...im", self.U, Vx)
        if x.shape[-1] == n:
            Vx = torch.einsum("...ki,...i->...k", self.V, x)
            return self.D * x + torch.einsum("...ki,...k->...i", self.U, Vx)
        raise ValueError(
            f"x must have shape (..., n) or (..., n, m) with n={n}, got {tuple(x.shape)}"
        )

    def __matmul__(self, other: torch.Tensor) -> torch.Tensor:
        return self.matvec(other)

    def __rmatmul__(self, other: torch.Tensor) -> torch.Tensor:
        """Row-vector / left multiply: ``x @ M == (Mᵀ @ xᵀ)ᵀ``."""
        other = torch.as_tensor(other, dtype=self.dtype, device=self.device)
        if other.shape[-1] == self.n and other.dim() == self.D.dim():
            return self.T.matvec(other)
        if other.shape[-1] == self.n:
            # other: (..., m, n) → treat rows as vectors
            return self.T.matvec(other.transpose(-2, -1)).transpose(-2, -1)
        raise ValueError(
            f"left operand must have shape (..., n) or (..., m, n) with n={self.n}, got {tuple(other.shape)}"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(batch_shape={tuple(self.batch_shape)}, "
            f"n={self.n}, k={self.k}, dtype={self.dtype}, device={self.device})"
        )


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