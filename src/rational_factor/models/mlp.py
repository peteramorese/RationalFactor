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


class MaskedMetzlerConeMLP(MLP):
    """Triangularly masked MLP over banks of Metzler cone rays.

    Parameters
    ----------
    K :
        Cone generators of shape ``(d, k, m, m)`` (``Matrix`` with batch shape
        ``(d, k)``, or a dense tensor). Slot ``ℓ`` has its own ``k`` rays.
    coeff_scale / max_coeff / bias_init :
        Same coefficient softplus scaling as :class:`MetzlerConeMLP`.

    For input ``x`` with shape ``(..., d)``, each output slot ``ℓ`` uses the
    masked view

        ``x̃_ℓ = (x_0, …, x_{ℓ-1}, 0, …, 0)``

    so ``M_ℓ`` depends only on ``x_<ℓ`` (slot ``0`` is input-independent).
    Softplus MLP weights give

        ``c_ℓ = coeff_scale · softplus(base_ℓ) · softplus(Δ_ℓ(x̃_ℓ))``,
        ``M_ℓ = Σ_i c_{ℓ,i} · K_{ℓ,i}``

    so overall magnitude is controlled by ``base`` while ``Δ`` (the masked net)
    provides dependence on ``x_<ℓ``. Returned as a Matrix of shape
    ``(..., d, m, m)`` (structure matching ``K``).
    """

    def __init__(
        self,
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
        if len(K.shape) != 4 or K.shape[-1] != K.shape[-2]:
            raise ValueError(
                f"K must have shape (d, k, m, m), got {tuple(K.shape)}"
            )
        if coeff_scale <= 0:
            raise ValueError("coeff_scale must be positive")
        if max_coeff is not None and max_coeff <= 0:
            raise ValueError("max_coeff must be positive when set")

        d, k, m, _ = K.shape
        super().__init__(
            in_features=d,
            out_features=k,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
            zero_init_last=False,
        )
        self._d = d
        self._m = m
        self._n_rays = k
        self.coeff_scale = float(coeff_scale)
        self.max_coeff = None if max_coeff is None else float(max_coeff)
        self._register_K(K)

        # Row ℓ keeps x_<ℓ (first ℓ coordinates); row 0 is all zeros.
        mask = torch.tril(torch.ones(d, d), diagonal=-1)
        self.register_buffer("_input_mask", mask)

        # Per-slot / per-ray base magnitude, separate from input-dependent Δ(x).
        # c = coeff_scale * softplus(base) * softplus(Δ(x)) so scale can be large
        # while Δ still modulates with the masked coordinates.
        self.base_logit = torch.nn.Parameter(
            torch.full((d, k), float(bias_init))
        )

        final = self.net[-1]
        if zero_init_last:
            # Constant coeffs at init (no x-dependence until training moves Δ).
            torch.nn.init.zeros_(final.weight)
            torch.nn.init.zeros_(final.bias)
        else:
            # Strong last-layer sensitivity to masked inputs; no last-layer bias
            # so Δ(0)=0 for slot 0 and Δ varies with x_<ℓ for ℓ≥1.
            torch.nn.init.xavier_uniform_(final.weight, gain=3.0)
            torch.nn.init.zeros_(final.bias)

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
        return self._d

    def out_features(self):
        """Matrix size ``m`` (each ray is ``m × m``)."""
        return self._m

    def n_rays(self) -> int:
        return self._n_rays

    def n_slots(self) -> int:
        return self._d

    def _masked_inputs(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x)
        if x.shape[-1] != self._d:
            raise ValueError(
                f"x trailing size must be d={self._d}, got {tuple(x.shape)}"
            )
        # (..., d) -> (..., d, d) with triangular zeros.
        return x.unsqueeze(-2) * self._input_mask

    def _coeffs(self, x: torch.Tensor) -> torch.Tensor:
        """Nonnegative ray weights ``(..., d, k)`` with triangular input dependence.

        ``c = coeff_scale · softplus(base) · softplus(Δ(x̃))`` where ``base`` is a
        learned per-(slot, ray) logit (magnitude) and ``Δ`` is the masked MLP
        (variation with ``x_<ℓ``).
        """
        x_masked = self._masked_inputs(x)
        delta = MLP.forward(self, x_masked)
        base = torch.nn.functional.softplus(self.base_logit)
        c = self.coeff_scale * base * torch.nn.functional.softplus(delta)
        if self.max_coeff is not None:
            c = c.clamp(max=self.max_coeff)
        return c

    def _combine(self, c: torch.Tensor) -> Matrix:
        """``c`` has shape ``(..., d, k)``; returns ``(..., d, m, m)``."""
        if self._K_kind == "diagonal":
            out = torch.einsum("...dk,dkn->...dn", c, self._K_storage)
            return Diagonal(out)
        if self._K_kind == "banded":
            out = torch.einsum("...dk,dkab->...dab", c, self._K_storage)
            return Banded(self._K_offsets, out)
        out = torch.einsum("...dk,dkij->...dij", c, self._K_storage)
        return DenseMatrix(out)

    def forward(self, x: torch.Tensor) -> Matrix:
        return self._combine(self._coeffs(x))


class PairedMaskedMetzlerConeMLP(MaskedMetzlerConeMLP):
    """Triangular MLP with shared coeffs on paired ``(R, T)`` ray banks.

    Paired Metzler rays satisfy ``R_k M + M T_k^T = 0``. Nonnegative combinations
    preserve the identity only when the **same** weights multiply ``R_k`` and
    ``T_k``. This module stores banks of shape ``(d, k, m, m)`` for both and
    applies one masked softplus MLP ``c(x)`` so

        ``R(x)_ℓ = Σ_i c(x_<ℓ)_i R_{ℓ,i}``,
        ``T(x)_ℓ = Σ_i c(x_<ℓ)_i T_{ℓ,i}``.

    ``forward`` returns ``(R(x), T(x))`` as a pair of Matrices.
    """

    def __init__(
        self,
        R: Matrix | torch.Tensor,
        T: Matrix | torch.Tensor,
        hidden_features: int = 128,
        num_hidden_layers: int = 2,
        activation=torch.nn.Tanh,
        zero_init_last: bool = True,
        coeff_scale: float = 1.0,
        max_coeff: float | None = 0.25,
        bias_init: float = -1.0,
    ):
        if isinstance(R, torch.Tensor):
            R = DenseMatrix(R)
        if isinstance(T, torch.Tensor):
            T = DenseMatrix(T)
        if not isinstance(R, Matrix) or not isinstance(T, Matrix):
            raise TypeError("R and T must be Matrix or Tensor")
        if R.shape != T.shape:
            raise ValueError(
                f"R and T must have the same shape, got R={tuple(R.shape)}, "
                f"T={tuple(T.shape)}"
            )
        super().__init__(
            K=R,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
            zero_init_last=zero_init_last,
            coeff_scale=coeff_scale,
            max_coeff=max_coeff,
            bias_init=bias_init,
        )
        self._register_T(T)

    def _register_T(self, T: Matrix) -> None:
        if isinstance(T, Diagonal):
            if self._K_kind != "diagonal":
                raise TypeError("T must match R structure (diagonal)")
            self.register_buffer("_T_storage", T.d.detach().clone())
        elif isinstance(T, Banded):
            if self._K_kind != "banded":
                raise TypeError("T must match R structure (banded)")
            if not torch.equal(T.offsets, self._K_offsets):
                raise ValueError("T banded offsets must match R")
            self.register_buffer("_T_storage", T.data.detach().clone())
        else:
            if self._K_kind != "dense":
                raise TypeError("T must match R structure (dense)")
            self.register_buffer("_T_storage", T.to_dense().detach().clone())

    @property
    def R(self) -> Matrix:
        return self.K

    @property
    def T(self) -> Matrix:
        if self._K_kind == "diagonal":
            return Diagonal(self._T_storage)
        if self._K_kind == "banded":
            return Banded(self._K_offsets, self._T_storage)
        return DenseMatrix(self._T_storage)

    def _combine_T(self, c: torch.Tensor) -> Matrix:
        """Same as ``_combine`` but against the ``T`` ray bank."""
        if self._K_kind == "diagonal":
            out = torch.einsum("...dk,dkn->...dn", c, self._T_storage)
            return Diagonal(out)
        if self._K_kind == "banded":
            out = torch.einsum("...dk,dkab->...dab", c, self._T_storage)
            return Banded(self._K_offsets, out)
        out = torch.einsum("...dk,dkij->...dij", c, self._T_storage)
        return DenseMatrix(out)

    def forward(self, x: torch.Tensor) -> tuple[Matrix, Matrix]:
        c = self._coeffs(x)
        return self._combine(c), self._combine_T(c)
