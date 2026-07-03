import math

import torch
from torch import nn
from torch.nn import functional as F

from nflows.transforms import CompositeTransform, ReversePermutation
from nflows.transforms.autoregressive import AutoregressiveTransform
from nflows.transforms.made import (
    MaskedFeedforwardBlock,
    MaskedLinear,
    MaskedResidualBlock,
    _get_input_degrees,
)
from nflows.utils import torchutils

from rational_factor.models.density_model import ConditionalDensityModel, DensityModel


def bernstein_gram_matrix(degree: int, *, dtype=torch.float64, device=None):
    """
    Return G where

        G[i, j] = int_0^1 B_{i,degree}(x) B_{j,degree}(x) dx

    Shape: (degree + 1, degree + 1)
    """
    n = degree
    device = device or torch.device("cpu")

    i = torch.arange(n + 1, dtype=dtype, device=device)
    j = torch.arange(n + 1, dtype=dtype, device=device)

    I, J = torch.meshgrid(i, j, indexing="ij")

    # log binomial coefficients
    def log_binom(n, k):
        n_t = torch.as_tensor(float(n), dtype=dtype, device=device)
        return (
            torch.lgamma(n_t + 1.0)
            - torch.lgamma(k + 1.0)
            - torch.lgamma(n_t - k + 1.0)
        )

    log_C_i = log_binom(n, I)
    log_C_j = log_binom(n, J)

    # Integral:
    # int B_{i,n}(x) B_{j,n}(x) dx
    # = C(n,i) C(n,j) Beta(i+j+1, 2n-i-j+1)
    log_beta = (
        torch.lgamma(I + J + 1.0)
        + torch.lgamma(2.0 * n - I - J + 1.0)
        - torch.lgamma(torch.tensor(2.0 * n + 2.0, dtype=dtype, device=device))
    )

    G = torch.exp(log_C_i + log_C_j + log_beta)
    return G


class CoupledMABernsteinConditioner(nn.Module):
    """Masked conditioner mapping x in [0, 1]^d to Bernstein control points.

    For each autoregressive dimension, outputs ``num_couplings`` coefficient vectors
    for degree-m Bernstein polynomials. The autoregressive mask enforces that output
    i depends only on x_1, ..., x_{i-1}; the first output vector is constant in x.

    Use ``num_couplings=1`` for a single flow; ``num_couplings=2`` shares backbone
    parameters across two coupled flows.
    """

    def __init__(
        self,
        features: int,
        degree: int,
        num_couplings: int = 2,
        hidden_features: int = 64,
        context_features: int | None = None,
        num_blocks: int = 2,
        use_residual_blocks: bool = True,
        activation=F.relu,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
    ):
        if degree < 1:
            raise ValueError("Bernstein degree must be at least 1.")
        if num_couplings < 1:
            raise ValueError("num_couplings must be at least 1.")

        super().__init__()
        self.features = features
        self.degree = degree
        self.num_couplings = num_couplings
        self.hidden_features = hidden_features
        self.use_residual_blocks = use_residual_blocks

        self.initial_layer = MaskedLinear(
            in_degrees=_get_input_degrees(features),
            out_features=hidden_features,
            autoregressive_features=features,
            random_mask=False,
            is_output=False,
        )

        if context_features is not None:
            self.context_layer = nn.Linear(context_features, hidden_features)
        else:
            self.context_layer = None

        block_constructor = (
            MaskedResidualBlock if use_residual_blocks else MaskedFeedforwardBlock
        )
        blocks = []
        prev_out_degrees = self.initial_layer.degrees
        for _ in range(num_blocks):
            blocks.append(
                block_constructor(
                    in_degrees=prev_out_degrees,
                    autoregressive_features=features,
                    context_features=context_features,
                    random_mask=False,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
            )
            prev_out_degrees = blocks[-1].degrees
        self.blocks = nn.ModuleList(blocks)

        self.final_layer = MaskedLinear(
            in_degrees=prev_out_degrees,
            out_features=features * degree * num_couplings,
            autoregressive_features=features,
            random_mask=False,
            is_output=True,
        )
        self.activation = activation

    @staticmethod
    def _simplex_weights_to_coefficients(weights: torch.Tensor) -> torch.Tensor:
        cumulative = torch.cumsum(weights, dim=-1)
        zeros = torch.zeros(
            *weights.shape[:-1],
            1,
            device=weights.device,
            dtype=weights.dtype,
        )
        return torch.cat([zeros, cumulative], dim=-1)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None, return_simplex_weights: bool = False
    ) -> torch.Tensor:
        """Return control points with shape (batch, features, num_couplings, degree + 1)."""
        hidden = self.initial_layer(x)
        if context is not None:
            hidden = hidden + self.activation(self.context_layer(context))
        if not self.use_residual_blocks:
            hidden = self.activation(hidden)
        for block in self.blocks:
            hidden = block(hidden, context)

        logits = self.final_layer(hidden)
        logits = logits.view(
            x.shape[0], self.features, self.num_couplings, self.degree
        )
        logits = logits / math.sqrt(self.hidden_features)
        weights = F.softmax(logits, dim=-1)
        if return_simplex_weights:
            return weights
        return self._simplex_weights_to_coefficients(weights)


class MABernsteinConditioner(CoupledMABernsteinConditioner):
    """Single-flow Bernstein conditioner (``num_couplings=1``)."""

    def __init__(
        self,
        features: int,
        degree: int,
        hidden_features: int = 64,
        context_features: int | None = None,
        num_blocks: int = 2,
        use_residual_blocks: bool = True,
        activation=F.relu,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
    ):
        super().__init__(
            features=features,
            degree=degree,
            num_couplings=1,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            use_residual_blocks=use_residual_blocks,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None, return_simplex_weights: bool = False
    ) -> torch.Tensor:
        """Return control points with shape (batch, features, degree + 1)."""
        return super().forward(x, context, return_simplex_weights).squeeze(2)


class ConstantBilinearMABernsteinConditioner(CoupledMABernsteinConditioner):
    """Coupled conditioner enforcing a constant bilinear form beta^T W alpha across flows."""

    def __init__(
        self,
        features: int,
        degree: int,
        hidden_features: int = 64,
        context_features: int | None = None,
        num_blocks: int = 2,
        use_residual_blocks: bool = True,
        activation=F.relu,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__(
            features=features,
            degree=degree,
            num_couplings=2,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            use_residual_blocks=use_residual_blocks,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

        self.register_buffer(
            "bernstein_operation_matrix",
            degree ** 2 * bernstein_gram_matrix(degree),
        )
        self.tf_inner_prod_values_u = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        weights = super().forward(x, context, return_simplex_weights=True)

        alpha = weights[..., 0, :]
        u = weights[..., 1, :]

        # a_i = (W alpha_i); feasible Z_i satisfies amin <= Z_i <= amax for beta on the simplex.
        operation_matrix = self.bernstein_operation_matrix.to(dtype=alpha.dtype)
        a = torch.einsum("ij,...j->...i", operation_matrix, alpha)
        a_min, _ = a.min(dim=-1, keepdim=True)
        a_max, _ = a.max(dim=-1, keepdim=True)

        mix = torch.softmax(self.tf_inner_prod_values_u, dim=0).view(
            *([1] * (a.ndim - 2)), self.features
        )
        inner_prod_values = (a_max - a_min).squeeze(-1) * mix + a_min.squeeze(-1)

        # obtain a beta_z that is strictly positive, sums to 1 and satisfies a^T beta_z = z
        beta_z = self._feasible_beta_center(a, inner_prod_values)
        v = self._project_onto_constraint_nullspace(u, a)
        tau = self._positive_tau(beta_z, v)
        beta = beta_z + tau * v

        alpha_coefficients = self._simplex_weights_to_coefficients(alpha)
        beta_coefficients = self._simplex_weights_to_coefficients(beta)
        return torch.stack([alpha_coefficients, beta_coefficients], dim=-2)

    @staticmethod
    def _normalize_simplex_weights(weights: torch.Tensor) -> torch.Tensor:
        weights = torch.clamp(weights, min=0.0)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).eps
        )

    def _project_onto_constraint_nullspace(
        self, u: torch.Tensor, a: torch.Tensor
    ) -> torch.Tensor:
        """Project u onto {v | 1^T v = 0 and a^T v = 0} via v = u - A^T (A A^T)^{-1} A u."""
        ones_row = torch.ones(
            *u.shape[:-1], 1, u.shape[-1], dtype=u.dtype, device=u.device
        )
        constraint_matrix = torch.cat([ones_row, a.unsqueeze(-2)], dim=-2)

        constraint_rhs = torch.einsum("...ij,...j->...i", constraint_matrix, u)
        constraint_gram = torch.matmul(
            constraint_matrix, constraint_matrix.transpose(-1, -2)
        )
        eye = torch.eye(2, dtype=u.dtype, device=u.device).expand_as(constraint_gram)
        constraint_gram = constraint_gram + self.eps * eye

        coefficients = torch.linalg.solve(constraint_gram, constraint_rhs)
        return u - torch.einsum("...ij,...i->...j", constraint_matrix, coefficients)

    def _feasible_beta_center(self, a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Construct beta_z on the simplex with a^T beta_z = z and amin <= z <= amax."""
        z = z.to(dtype=a.dtype, device=a.device)
        if z.ndim == a.ndim - 1:
            z = z.unsqueeze(-1)

        a_min, idx_min = torch.min(a, dim=-1, keepdim=True)
        a_max, idx_max = torch.max(a, dim=-1, keepdim=True)

        e_min = torch.zeros_like(a).scatter(-1, idx_min, 1.0)
        e_max = torch.zeros_like(a).scatter(-1, idx_max, 1.0)

        lam = (z - a_min) / (a_max - a_min + self.eps)
        lam = torch.clamp(lam, 0.0, 1.0)
        return (1.0 - lam) * e_min + lam * e_max

    def _positive_tau(self, beta_center: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """
        Return tau such that

            beta = beta_center + tau * direction

        remains positive.
        """
        floor = torch.as_tensor(self.positivity_floor, dtype=beta_center.dtype, device=beta_center.device)

        positive_slack = beta_center - floor

        ratios = torch.where(
            direction < 0,
            positive_slack / (-direction + self.eps),
            torch.full_like(direction, float("inf")),
        )

        tau_max = ratios.min(dim=-1, keepdim=True).values

        # If direction has no negative components, tau_max is inf.
        # But because direction should be zero-sum, this usually means direction is zero.
        tau_max = torch.where(torch.isfinite(tau_max), tau_max, torch.zeros_like(tau_max))

        tau_max = torch.clamp(tau_max, min=0.0)

        # Per-feature learned safety fraction in (0, 1).
        # Shape should broadcast over batch/context dimensions.
        rho = torch.sigmoid(self.nullspace_scale_logits).view(
            *([1] * (direction.ndim - 2)), self.features, 1
        )

        tau = rho * tau_max
        return tau

class MaskedBernsteinAutoregressiveTransform(AutoregressiveTransform):
    """Autoregressive layer with degree-m Bernstein polynomial couplings on [0, 1]."""

    def __init__(
        self,
        features: int,
        degree: int,
        hidden_features: int = 64,
        context_features: int | None = None,
        num_blocks: int = 2,
        use_residual_blocks: bool = True,
        activation=F.relu,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
        min_derivative: float = 1e-6,
    ):
        self.features = features
        self.degree = degree
        self.min_derivative = min_derivative

        conditioner = MABernsteinConditioner(
            features=features,
            degree=degree,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            use_residual_blocks=use_residual_blocks,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
        super().__init__(conditioner)

    @staticmethod
    def _de_casteljau(x: torch.Tensor, control_points: torch.Tensor) -> torch.Tensor:
        values = control_points
        for _ in range(control_points.shape[-1] - 1):
            values = (1.0 - x).unsqueeze(-1) * values[..., :-1] + x.unsqueeze(
                -1
            ) * values[..., 1:]
        return values[..., 0]

    def _bernstein_derivative(
        self, x: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        delta_coefficients = coefficients[..., 1:] - coefficients[..., :-1]
        derivative = self.degree * self._de_casteljau(x, delta_coefficients)
        return derivative.clamp_min(self.min_derivative)

    def forward(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.autoregressive_net(inputs, context)
        outputs = self._de_casteljau(inputs, coefficients)
        logabsdet = torch.log(self._bernstein_derivative(inputs, coefficients))
        return outputs, torchutils.sum_except_batch(logabsdet, num_batch_dims=1)


class BernsteinNormalizingFlow(DensityModel):
    """Autoregressive normalizing flow on the unit box with a uniform base distribution."""

    def __init__(
        self,
        dim: int,
        degree: int = 5,
        num_layers: int = 5,
        hidden_features: int = 64,
    ):
        super().__init__(dim=dim)

        transforms = []
        for _ in range(num_layers):
            transforms.append(
                MaskedBernsteinAutoregressiveTransform(
                    features=dim,
                    degree=degree,
                    hidden_features=hidden_features,
                )
            )
            transforms.append(ReversePermutation(features=dim))

        self.transform = CompositeTransform(transforms)

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        _, logabsdet = self.transform(x)
        return self._clip_log_density(logabsdet)


class ConditionalBernsteinNormalizingFlow(ConditionalDensityModel):
    """Conditional Bernstein autoregressive flow on the unit box."""

    def __init__(
        self,
        dim: int,
        conditioner_dim: int,
        degree: int = 5,
        num_layers: int = 5,
        hidden_features: int = 64,
    ):
        super().__init__(dim=dim, conditioner_dim=conditioner_dim)

        transforms = []
        for _ in range(num_layers):
            transforms.append(
                MaskedBernsteinAutoregressiveTransform(
                    features=dim,
                    degree=degree,
                    hidden_features=hidden_features,
                    context_features=conditioner_dim,
                )
            )
            transforms.append(ReversePermutation(features=dim))

        self.transform = CompositeTransform(transforms)

    def log_density(
        self, x: torch.Tensor, *, conditioner: torch.Tensor
    ) -> torch.Tensor:
        _, logabsdet = self.transform(x, context=conditioner)
        return self._clip_log_density(logabsdet)
