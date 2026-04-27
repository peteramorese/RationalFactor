import math
import torch

from .density_model import DensityModel


class BaseGaussianKDE(DensityModel):
    def __init__(self, data: torch.Tensor):
        assert data.ndim == 2, "data must have shape (n_samples, dim)"
        assert data.shape[0] >= 1, "data must contain at least one sample"
        super().__init__(dim=data.shape[1])
        self.register_buffer("data", data.detach().clone())
        self._eps = torch.finfo(self.data.dtype).eps

    def _kernel_bandwidths(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """
        Return per-kernel bandwidths as shape (n_samples,).
        """
        raise NotImplementedError

    def _build_marginal(self, marginal_data: torch.Tensor):
        raise NotImplementedError

    def _log_kernel(self, x: torch.Tensor, centers: torch.Tensor, kernel_bandwidths: torch.Tensor) -> torch.Tensor:
        d = x.shape[1]
        diff = x[:, None, :] - centers[None, :, :]
        z = diff / kernel_bandwidths[None, :, None]
        sq_norm = z.square().sum(dim=-1)
        log_norm = -0.5 * d * math.log(2.0 * math.pi) - d * torch.log(kernel_bandwidths)[None, :]
        return log_norm - 0.5 * sq_norm

    def log_density(self, x: torch.Tensor):
        assert x.ndim == 2 and x.shape[1] == self.dim, f"x must have shape (n_data, {self.dim})"
        n = self.data.shape[0]
        h = self._kernel_bandwidths(dtype=x.dtype, device=x.device).clamp_min(self._eps)
        log_k = self._log_kernel(x=x, centers=self.data, kernel_bandwidths=h)
        return torch.logsumexp(log_k, dim=1) - math.log(n)

    def sample(self, n_samples: int):
        assert n_samples > 0, "n_samples must be positive"
        idx = torch.randint(low=0, high=self.data.shape[0], size=(n_samples,), device=self.data.device)
        h = self._kernel_bandwidths(dtype=self.data.dtype, device=self.data.device).clamp_min(self._eps)
        noise = torch.randn(n_samples, self.dim, dtype=self.data.dtype, device=self.data.device)
        return self.data[idx] + h[idx, None] * noise

    def marginal(self, marginal_dims: tuple[int, ...]):
        return self._build_marginal(self.data[:, marginal_dims].detach().clone())

    def loo_log_likelihood_per_sample(self) -> torch.Tensor:
        n = self.data.shape[0]
        if n <= 1:
            raise ValueError("leave-one-out likelihood is undefined for n <= 1")

        h = self._kernel_bandwidths(dtype=self.data.dtype, device=self.data.device).clamp_min(self._eps)
        log_k = self._log_kernel(x=self.data, centers=self.data, kernel_bandwidths=h)
        log_k = log_k.masked_fill(torch.eye(n, dtype=torch.bool, device=self.data.device), float("-inf"))
        return torch.logsumexp(log_k, dim=1) - math.log(n - 1)

    def loo_log_likelihood(self) -> torch.Tensor:
        return torch.mean(self.loo_log_likelihood_per_sample())


class GaussianKDE(BaseGaussianKDE):
    """
    Gaussian-kernel density estimator with a shared isotropic bandwidth.
    """

    def __init__(self, data: torch.Tensor, bandwidth: float | torch.Tensor | None = None):
        super().__init__(data=data)

        if bandwidth is None:
            bandwidth_tensor = self.scott_bandwidth(self.data)
        elif torch.is_tensor(bandwidth):
            bandwidth_tensor = bandwidth.detach().to(dtype=self.data.dtype, device=self.data.device)
        else:
            bandwidth_tensor = torch.tensor(float(bandwidth), dtype=self.data.dtype, device=self.data.device)

        self.register_buffer("bandwidth", bandwidth_tensor.clamp_min(self._eps))

    @staticmethod
    def scott_bandwidth(data: torch.Tensor) -> torch.Tensor:
        """
        Scott/Silverman-style isotropic rule of thumb for an initial bandwidth.
        """
        n, d = data.shape
        eps = torch.finfo(data.dtype).eps
        if n <= 1:
            return torch.tensor(1.0, dtype=data.dtype, device=data.device)

        std = data.std(dim=0, unbiased=True)
        scale = std.mean().clamp_min(eps)
        factor = n ** (-1.0 / (d + 4.0))
        return (scale * factor).clamp_min(eps)

    def _kernel_bandwidths(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        h = self.bandwidth.to(dtype=dtype, device=device).clamp_min(self._eps)
        return h.expand(self.data.shape[0])

    def _build_marginal(self, marginal_data: torch.Tensor):
        return GaussianKDE(
            data=marginal_data,
            bandwidth=self.bandwidth.detach().clone(),
        )

    def fit_bandwidth_loo_mle(
        self,
        epochs: int = 200,
        lr: float = 0.05,
        verbose: bool = False,
    ) -> tuple[torch.Tensor, float]:
        """
        Optimize bandwidth by maximizing leave-one-out log-likelihood.

        Returns:
            (best_bandwidth, best_objective_value)
        """
        assert epochs > 0, "epochs must be positive"
        assert lr > 0, "lr must be positive"

        n = self.data.shape[0]
        if n <= 1:
            raise ValueError("Cannot fit leave-one-out bandwidth with fewer than 2 samples")



        log_h = torch.nn.Parameter(torch.log(self.bandwidth.clamp_min(self._eps)).detach().clone())
        optimizer = torch.optim.Adam([log_h], lr=lr)

        best_obj = -float("inf")
        best_h = None

        for epoch in range(epochs):
            optimizer.zero_grad()
            h = torch.exp(log_h).clamp_min(self._eps)
            log_k = self._log_kernel(
                x=self.data,
                centers=self.data,
                kernel_bandwidths=h.expand(self.data.shape[0]),
            )
            n = self.data.shape[0]
            log_k = log_k.masked_fill(torch.eye(n, dtype=torch.bool, device=self.data.device), float("-inf"))
            obj = torch.mean(torch.logsumexp(log_k, dim=1) - math.log(n - 1))
            loss = -obj
            loss.backward()
            optimizer.step()

            obj_value = obj.detach().item()
            if obj_value > best_obj:
                best_obj = obj_value
                best_h = torch.exp(log_h.detach()).clamp_min(self._eps)

            if verbose:
                print(
                    f"Bandwidth epoch {epoch + 1}/{epochs}, "
                    f"loo_loglik={obj_value:.6f}, h={torch.exp(log_h.detach()).item():.6g}"
                )

        assert best_h is not None
        self.bandwidth.copy_(best_h)
        return self.bandwidth.detach().clone(), best_obj

    def fit_bandwidth_validation_mle(
        self,
        validation_data: torch.Tensor,
        epochs: int = 200,
        lr: float = 0.05,
        threshold: float = 0.05,
        verbose: bool = False,
        min_step : float = 1e-4,
        block_size: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        """
        Heuristic bandwidth update using train-vs-validation log-likelihood gap.

        At each iteration:
          train_score = mean(log p_h(train_data))
          val_score   = mean(log p_h(validation_data))
          diff = train_score - val_score

        Update rule:
          if diff > threshold: h <- h + lr * diff
          else:                h <- h - lr * diff
        """
        assert validation_data.ndim == 2, "validation_data must have shape (n_samples, dim)"
        assert validation_data.shape[1] == self.dim, f"validation_data must have shape (n_samples, {self.dim})"
        assert validation_data.shape[0] > 0, "validation_data must contain at least one sample"
        assert epochs > 0, "epochs must be positive"
        assert lr > 0, "lr must be positive"
        if block_size is not None:
            assert block_size > 0, "block_size must be positive when provided"

        val = validation_data.to(dtype=self.data.dtype, device=self.data.device)
        h = self.bandwidth.detach().clone().clamp_min(self._eps)
        best_val = -float("inf")
        best_h = h.clone()

        def _mean_log_density_batched(x: torch.Tensor) -> torch.Tensor:
            if block_size is None:
                return torch.mean(self.log_density(x))
            total = torch.zeros((), dtype=self.data.dtype, device=self.data.device)
            n = x.shape[0]
            for start in range(0, n, block_size):
                stop = min(start + block_size, n)
                xb = x[start:stop]
                total = total + self.log_density(xb).sum()
            return total / n

        with torch.no_grad():
            for epoch in range(epochs):
                self.bandwidth.copy_(h)
                train_score = _mean_log_density_batched(self.data)
                val_score = _mean_log_density_batched(val)
                diff = train_score - val_score

                if diff.item() > threshold:
                    h = h + lr * (torch.abs(diff) + min_step)
                else:
                    h = h - lr * (torch.abs(diff) + min_step)
                h = h.clamp_min(self._eps)

                val_value = float(val_score.item())
                train_value = float(train_score.item())
                diff_value = float(diff.item())
                if val_value > best_val:
                    best_val = val_value
                    best_h = h.clone()

                if verbose:
                    print(
                        f"Val-bandwidth epoch {epoch + 1}/{epochs}, "
                        f"train_loglik={train_value:.6f}, val_loglik={val_value:.6f}, "
                        f"gap={diff_value:.6f}, h={float(h.item()):.6g}"
                    )

        self.bandwidth.copy_(best_h.clamp_min(self._eps))
        return self.bandwidth.detach().clone(), best_val

class AdaptiveGaussianKDE(BaseGaussianKDE):
    """
    Adaptive Gaussian KDE with per-kernel bandwidths using Abramson's square-root rule.
    """

    def __init__(self, data: torch.Tensor, kernel_bandwidths: torch.Tensor):
        assert data.ndim == 2, "data must have shape (n_samples, dim)"
        assert kernel_bandwidths.ndim == 1, "kernel_bandwidths must have shape (n_samples,)"
        assert data.shape[0] == kernel_bandwidths.shape[0], "bandwidth count must match number of samples"
        super().__init__(data=data)
        self.register_buffer(
            "kernel_bandwidths",
            kernel_bandwidths.detach().to(dtype=data.dtype, device=data.device).clamp_min(self._eps),
        )

    @classmethod
    def from_gaussian_kde(
        cls,
        kde: GaussianKDE,
        alpha: float = 0.5,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
        use_loo_pilot: bool = True,
    ) -> "AdaptiveGaussianKDE":
        """
        Build adaptive KDE from a global-bandwidth KDE.

        Abramson rule:
            h_i = h * (g / f_pilot(x_i))^alpha
        with alpha=1/2 by default (square-root law) and geometric-mean scale g.
        """
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must satisfy 0 < alpha <= 1")
        if min_factor <= 0.0 or max_factor <= 0.0 or min_factor > max_factor:
            raise ValueError("min_factor and max_factor must be positive and satisfy min_factor <= max_factor")

        with torch.no_grad():
            if use_loo_pilot and kde.data.shape[0] > 1:
                pilot_log = kde.loo_log_likelihood_per_sample()
            else:
                pilot_log = kde.log_density(kde.data)
            pilot = torch.exp(pilot_log).clamp_min(kde._eps)

            log_g = torch.mean(torch.log(pilot))
            g = torch.exp(log_g).clamp_min(kde._eps)
            local_factor = (g / pilot).pow(alpha).clamp(min=min_factor, max=max_factor)
            local_h = kde.bandwidth * local_factor

        return cls(data=kde.data, kernel_bandwidths=local_h)

    def _kernel_bandwidths(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return self.kernel_bandwidths.to(dtype=dtype, device=device).clamp_min(self._eps)

    def _build_marginal(self, marginal_data: torch.Tensor):
        return AdaptiveGaussianKDE(
            data=marginal_data,
            kernel_bandwidths=self.kernel_bandwidths.detach().clone(),
        )
