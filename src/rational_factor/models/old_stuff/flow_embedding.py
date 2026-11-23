"""Index-conditioned normalizing flow: ``p(x | e(i))`` for ``i = 1..k``.

Self-contained module (depends only on ``torch`` and ``nflows``). A shared
conditional flow is conditioned on a learned embedding table with ``k`` rows;
each index ``i`` defines a density over ``R^d``.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from nflows.nn.nets import ResidualNet
from nflows.transforms import (
    ActNorm,
    AffineCouplingTransform,
    CompositeTransform,
    LULinear,
    ReversePermutation,
    RandomPermutation,
)
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform


def _permutation(features: int, kind: str):
    kind = kind.lower()
    if kind == "reverse":
        return ReversePermutation(features=features)
    if kind == "random":
        return RandomPermutation(features=features)
    raise ValueError(f"Unknown permutation {kind!r}; use 'reverse' or 'random'")


def _coupling_net_factory(
    hidden_features: int,
    *,
    context_features: int | None = None,
    num_blocks: int = 2,
) -> Callable:
    def create_resnet(in_features: int, out_features: int) -> ResidualNet:
        return ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            use_batch_norm=False,
        )

    return create_resnet


def _checkerboard_mask(dim: int) -> torch.Tensor:
    mask = torch.ones(dim)
    mask[::2] = -1
    return mask


def _maf_transforms(
    dim: int,
    num_layers: int,
    hidden_features: int,
    *,
    context_features: int,
    num_blocks: int = 2,
    permutation: str = "reverse",
) -> list:
    transforms = []
    for _ in range(num_layers):
        transforms.append(_permutation(dim, permutation))
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=dim,
                hidden_features=hidden_features,
                context_features=context_features,
                num_blocks=num_blocks,
                use_residual_blocks=True,
                random_mask=False,
                activation=torch.tanh,
                dropout_probability=0.0,
                use_batch_norm=False,
            )
        )
    return transforms


def _nsf_transforms(
    dim: int,
    num_layers: int,
    hidden_features: int,
    *,
    context_features: int,
    num_blocks: int = 2,
    num_bins: int = 8,
    tail_bound: float = 3.0,
    permutation: str = "reverse",
) -> list:
    transforms = []
    for _ in range(num_layers):
        transforms.append(_permutation(dim, permutation))
        transforms.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=dim,
                hidden_features=hidden_features,
                context_features=context_features,
                num_blocks=num_blocks,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
                use_residual_blocks=True,
                random_mask=False,
                activation=torch.tanh,
                dropout_probability=0.0,
                use_batch_norm=False,
            )
        )
    return transforms


def _realnvp_transforms(
    dim: int,
    num_layers: int,
    hidden_features: int,
    *,
    context_features: int,
    num_blocks: int = 2,
) -> list:
    if dim < 2:
        raise ValueError(f"RealNVP coupling flows require dim >= 2, got {dim}")
    create_resnet = _coupling_net_factory(
        hidden_features, context_features=context_features, num_blocks=num_blocks
    )
    mask = _checkerboard_mask(dim)
    transforms = []
    for _ in range(num_layers):
        transforms.append(ActNorm(features=dim))
        transforms.append(LULinear(features=dim))
        transforms.append(
            AffineCouplingTransform(mask=mask, transform_net_create_fn=create_resnet)
        )
        mask = -mask
    return transforms


def _rq_coupling_transforms(
    dim: int,
    num_layers: int,
    hidden_features: int,
    *,
    context_features: int,
    num_blocks: int = 2,
    num_bins: int = 8,
    tail_bound: float = 3.0,
) -> list:
    if dim < 2:
        raise ValueError(f"RQ coupling flows require dim >= 2, got {dim}")
    create_resnet = _coupling_net_factory(
        hidden_features, context_features=context_features, num_blocks=num_blocks
    )
    mask = _checkerboard_mask(dim)
    transforms = []
    for _ in range(num_layers):
        transforms.append(ActNorm(features=dim))
        transforms.append(LULinear(features=dim))
        transforms.append(
            PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=create_resnet,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
            )
        )
        mask = -mask
    return transforms


def _build_conditional_flow(
    dim: int,
    context_features: int,
    transform: str,
    num_layers: int,
    hidden_features: int,
    num_blocks: int,
    num_bins: int,
    tail_bound: float,
    permutation: str,
) -> Flow:
    key = transform.lower().replace("-", "_")
    builders = {
        "maf": lambda: _maf_transforms(
            dim,
            num_layers,
            hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            permutation=permutation,
        ),
        "nsf": lambda: _nsf_transforms(
            dim,
            num_layers,
            hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            num_bins=num_bins,
            tail_bound=tail_bound,
            permutation=permutation,
        ),
        "masked_rqs": lambda: _nsf_transforms(
            dim,
            num_layers,
            hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            num_bins=num_bins,
            tail_bound=tail_bound,
            permutation=permutation,
        ),
        "realnvp": lambda: _realnvp_transforms(
            dim,
            num_layers,
            hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
        ),
        "rq_coupling": lambda: _rq_coupling_transforms(
            dim,
            num_layers,
            hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            num_bins=num_bins,
            tail_bound=tail_bound,
        ),
    }
    if key not in builders:
        raise ValueError(
            f"Unknown transform {transform!r}. Choose from: {sorted(builders)}"
        )
    transforms: Sequence = builders[key]()
    return Flow(
        transform=CompositeTransform(list(transforms)),
        distribution=StandardNormal(shape=[dim]),
    )


class FlowEmbedding(torch.nn.Module):
    """Conditional normalizing flow indexed by a learned embedding table.

    Models ``p(x | e(i))`` for ``i = 0, ..., k-1``, where ``e(i)`` is row ``i``
    of an embedding of size ``(k, embedding_dim)`` and ``x ∈ R^d``. The same
    flow parameters are shared across indices; only the conditioner differs.

    Parameters
    ----------
    dim :
        Ambient dimension ``d`` of the particles ``x``.
    n_indices :
        Number of unique indices ``k`` (rows of the embedding table).
    embedding_dim :
        Dimension of each index embedding ``e(i)``.
    transform :
        Flow family: ``"maf"``, ``"nsf"``, ``"realnvp"``, or ``"rq_coupling"``.
    num_layers :
        Number of flow layers.
    hidden_features :
        Hidden width of the conditioner / MADE / residual nets.
    num_blocks :
        Number of residual / MADE blocks per layer.
    num_bins :
        Number of spline bins (NSF / RQ coupling only).
    tail_bound :
        Linear-tail bound for rational-quadratic splines.
    permutation :
        Inter-layer permutation for autoregressive flows (``"reverse"`` or
        ``"random"``).
    embedding_init_std :
        Std of the normal initialization for the embedding table.
    """

    def __init__(
        self,
        dim: int,
        n_indices: int,
        embedding_dim: int,
        transform: str = "maf",
        num_layers: int = 5,
        hidden_features: int = 64,
        num_blocks: int = 2,
        num_bins: int = 8,
        tail_bound: float = 3.0,
        permutation: str = "reverse",
        embedding_init_std: float = 0.05,
    ):
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if n_indices < 1:
            raise ValueError(f"n_indices must be >= 1, got {n_indices}")
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim}")

        self.dim = dim
        self.n_indices = n_indices
        self.embedding_dim = embedding_dim
        self.transform_name = transform
        self.num_layers = num_layers
        self.hidden_features = hidden_features
        self.num_blocks = num_blocks
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.permutation = permutation

        self.index_embedding = torch.nn.Embedding(n_indices, embedding_dim)
        torch.nn.init.normal_(
            self.index_embedding.weight, mean=0.0, std=embedding_init_std
        )

        self.flow = _build_conditional_flow(
            dim=dim,
            context_features=embedding_dim,
            transform=transform,
            num_layers=num_layers,
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            num_bins=num_bins,
            tail_bound=tail_bound,
            permutation=permutation,
        )

    def _index_conditioners(
        self, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        idx = torch.arange(self.n_indices, device=device)
        return self.index_embedding(idx).to(dtype=dtype)

    def _as_data(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.index_embedding.weight
        x = torch.as_tensor(x, dtype=weight.dtype, device=weight.device)
        if x.ndim == 1:
            x = x.unsqueeze(-1) if self.dim == 1 else x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(
                f"x must have shape (n_particles, {self.dim}), got {tuple(x.shape)}"
            )
        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Per-particle log densities ``log p(x_n | e(i))``, shape ``(n, k)``."""
        x = self._as_data(x)
        n, k = x.shape[0], self.n_indices
        cond = self._index_conditioners(x.dtype, x.device)
        x_rep = x.unsqueeze(1).expand(-1, k, -1).reshape(n * k, self.dim)
        c_rep = cond.unsqueeze(0).expand(n, -1, -1).reshape(n * k, -1)
        log_p = self.flow.log_prob(inputs=x_rep, context=c_rep)
        return log_p.reshape(n, k)

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Average log likelihood under each ``p(x | e(i))``.

        Parameters
        ----------
        x :
            Particles of shape ``(n_particles, dim)``.

        Returns
        -------
        Tensor of shape ``(n_indices,)``, where entry ``i`` is
        ``mean_n log p(x_n | e(i))``.
        """
        return self.log_prob(x).mean(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`evaluate`."""
        return self.evaluate(x)
