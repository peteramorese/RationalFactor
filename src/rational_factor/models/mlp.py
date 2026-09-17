from __future__ import annotations

import torch

from rational_factor.models.structured_matrices import (
    Banded,
    DenseMatrix,
    Diagonal,
    Matrix,
)


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
    """MLP that outputs a nonnegative combination of Metzler cone ray matrices.

    Parameters
    ----------
    K :
        Batched cone generators of shape ``(k, m, m)``, typically from
        ``MetzlerConeRayFinder.find`` (``Diagonal``, ``Banded``, or
        ``DenseMatrix``). Rays live on the leading batch axis.
    coeff_scale :
        Multiplies the softplus ray weights. Default ``1`` so unit-Frobenius
        rays produce ``M`` entries on the same order as the rays themselves.
    max_coeff :
        Optional per-ray softplus cap after scaling (safety for ``expm``).

    The base MLP produces ``k`` logits; ``forward`` returns

        ``sum_i softplus(mlp(x))_i · K_i``

    as a :class:`~rational_factor.models.structured_matrices.Matrix` whose
    structure matches ``K`` (batch size 1 when ``x`` is unbatched, otherwise
    one matrix per leading data batch element).
    """

    def __init__(
        self,
        in_features: int,
        K: Matrix | torch.Tensor,
        hidden_features: int = 128,
        num_hidden_layers: int = 2,
        activation=torch.nn.Tanh,
        zero_init_last: bool = True,
        coeff_scale: float = 1.0,
        max_coeff: float | None = 0.25,
        bias_init: float = -1.0,
    ):
        if isinstance(K, torch.Tensor):
            K = DenseMatrix(K)
        if not isinstance(K, Matrix):
            raise TypeError(f"K must be a Matrix or Tensor, got {type(K)!r}")
        if len(K.shape) != 3 or K.shape[-1] != K.shape[-2]:
            raise ValueError(
                f"K must have shape (k, m, m), got {tuple(K.shape)}"
            )
        if coeff_scale <= 0:
            raise ValueError("coeff_scale must be positive")
        if max_coeff is not None and max_coeff <= 0:
            raise ValueError("max_coeff must be positive when set")

        k, m, _ = K.shape
        # Skip parent zero-init; last layer is set below.
        super().__init__(
            in_features=in_features,
            out_features=k,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
            zero_init_last=False,
        )
        self._m = m
        self._n_rays = k
        self.coeff_scale = float(coeff_scale)
        self.max_coeff = None if max_coeff is None else float(max_coeff)
        self._register_K(K)

        if zero_init_last:
            final = self.net[-1]
            torch.nn.init.zeros_(final.weight)
            # softplus(-1) ≈ 0.31 with σ(-1)≈0.27 — O(1) coeffs with usable
            # gradients, then capped by max_coeff so expm(M) stays stable.
            torch.nn.init.constant_(final.bias, float(bias_init))

    def _register_K(self, K: Matrix) -> None:
        if isinstance(K, Diagonal):
            self._K_kind = "diagonal"
            self.register_buffer("_K_storage", K.d.detach().clone())
        elif isinstance(K, Banded):
            self._K_kind = "banded"
            self.register_buffer("_K_storage", K.data.detach().clone())
            self.register_buffer("_K_offsets", K.offsets.detach().clone())
        else:
            self._K_kind = "dense"
            self.register_buffer("_K_storage", K.to_dense().detach().clone())

    @property
    def K(self) -> Matrix:
        if self._K_kind == "diagonal":
            return Diagonal(self._K_storage)
        if self._K_kind == "banded":
            return Banded(self._K_offsets, self._K_storage)
        return DenseMatrix(self._K_storage)

    def in_features(self):
        return self.net[0].in_features

    def out_features(self):
        """Matrix size ``m`` (each ray is ``m × m``)."""
        return self._m

    def n_rays(self) -> int:
        return self._n_rays

    def forward(self, x: torch.Tensor) -> Matrix:
        c = self.coeff_scale * torch.nn.functional.softplus(super().forward(x))
        if self.max_coeff is not None:
            c = c.clamp(max=self.max_coeff)
        return self.K.batch_linear_combine(c)
