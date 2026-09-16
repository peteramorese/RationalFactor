from __future__ import annotations

import torch

from rational_factor.models.structured_matrices import Matrix


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 128,
        num_hidden_layers: int = 2,
        activation=torch.nn.Tanh,
        zero_init_last: bool = True,
    ):
        super().__init__()

        layers = []
        last = in_features
        for _ in range(num_hidden_layers):
            layers.append(torch.nn.Linear(last, hidden_features))
            layers.append(activation())
            last = hidden_features

        layers.append(torch.nn.Linear(last, out_features))
        self.net = torch.nn.Sequential(*layers)

        if zero_init_last:
            final = self.net[-1]
            torch.nn.init.zeros_(final.weight)
            torch.nn.init.zeros_(final.bias)

    def in_features(self):
        return self.net[0].in_features

    def out_features(self):
        return self.net[-1].out_features

    def forward(self, x):
        return self.net(x)


class MetzlerConeMLP(MLP):
    """MLP constrained to a Metzler cone via nonnegative ray coefficients.

    Parameters
    ----------
    K :
        Cone generator matrix of shape ``(m, k)``, typically the
        ``DenseMatrix`` returned by ``metzler_cone_rays``. The base MLP
        produces ``k`` logits; the module output is
        ``K @ softplus(mlp(x))`` with shape ``(..., m)``.
    """

    def __init__(
        self,
        in_features: int,
        K: Matrix | torch.Tensor,
        hidden_features: int = 128,
        num_hidden_layers: int = 2,
        activation=torch.nn.Tanh,
        zero_init_last: bool = True,
    ):
        K_dense = K.to_dense() if isinstance(K, Matrix) else torch.as_tensor(K)
        if K_dense.dim() != 2:
            raise ValueError(
                f"K must have shape (m, k), got {tuple(K_dense.shape)}"
            )
        m, k = K_dense.shape
        super().__init__(
            in_features=in_features,
            out_features=k,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
            zero_init_last=zero_init_last,
        )
        self.register_buffer("K", K_dense.detach().clone())
        self.out_dim = m

    def in_features(self):
        return self.net[0].in_features

    def out_features(self):
        return self.K.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = torch.nn.functional.softplus(super().forward(x))
        return torch.einsum("mk,...k->...m", self.K, c)
