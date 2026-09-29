from __future__ import annotations

from collections.abc import Sequence

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


class MaskedInputMLP(MLP):
    r"""Shared autoregressive-masked MLP over all slots and cone layers.

    Takes ``x`` of shape ``(batch, D)`` and applies the triangular mask

        ``x̃_ℓ = (x_0, …, x_{ℓ-1}, 0, …, 0)``

    so every slot shares one network (slot ``0`` is input-independent).  The
    shared head produces, for each slot and each of ``n_layers`` cone updates,

        ``C_{ℓ,k} ∈ R^{m×r}``,   ``raw_s_{ℓ,k} ∈ R^r``,

    returned as

        ``C`` of shape ``(batch, D, n_layers, m, r)``,
        ``raw_s`` of shape ``(batch, D, n_layers, r)``.

    Multiple hidden layers use ``num_hidden_layers`` with uniform width
    ``hidden_features``, or a sequence of per-layer widths.
    """

    def __init__(
        self,
        n_features: int,
        n_layers: int,
        m: int,
        r: int,
        hidden_features: int | Sequence[int] = 128,
        num_hidden_layers: int = 2,
        activation=torch.nn.SiLU,
        zero_init_last: bool = True,
    ):
        if n_features < 1:
            raise ValueError(f"n_features must be positive, got {n_features}")
        if n_layers < 0:
            raise ValueError(f"n_layers must be nonnegative, got {n_layers}")
        if m < 1 or r < 1:
            raise ValueError(f"m and r must be positive, got m={m}, r={r}")

        out_per_slot = n_layers * (m * r + r)

        if isinstance(hidden_features, Sequence) and not isinstance(
            hidden_features, (str, bytes)
        ):
            widths = tuple(int(h) for h in hidden_features)
            if not widths:
                raise ValueError("hidden_features sequence must be nonempty")
            if any(h != widths[0] for h in widths):
                uniform_hidden = False
                custom_widths = widths
                hidden_width = widths[0]
                n_hidden = len(widths)
            else:
                uniform_hidden = True
                custom_widths = None
                hidden_width = widths[0]
                n_hidden = len(widths)
        else:
            uniform_hidden = True
            custom_widths = None
            hidden_width = int(hidden_features)
            n_hidden = int(num_hidden_layers)
            if n_hidden < 0:
                raise ValueError("num_hidden_layers must be nonnegative")

        super().__init__(
            in_features=n_features,
            out_features=out_per_slot,
            hidden_features=hidden_width,
            num_hidden_layers=max(n_hidden, 0),
            activation=activation,
            zero_init_last=False,
        )

        if not uniform_hidden:
            layers: list[torch.nn.Module] = []
            last = n_features
            for width in custom_widths:
                layers += [torch.nn.Linear(last, width), activation()]
                last = width
            layers.append(torch.nn.Linear(last, out_per_slot))
            self.net = torch.nn.Sequential(*layers)

        if zero_init_last:
            final = self.net[-1]
            torch.nn.init.zeros_(final.weight)
            torch.nn.init.zeros_(final.bias)

        self.n_features = int(n_features)
        self.n_layers = int(n_layers)
        self.m = int(m)
        self.r = int(r)
        # Row ℓ keeps x_<ℓ; row 0 is all zeros.
        self.register_buffer(
            "_input_mask",
            torch.tril(torch.ones(n_features, n_features), diagonal=-1),
        )

    def _masked_inputs(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(
                f"x must have shape (batch, {self.n_features}), got {tuple(x.shape)}"
            )
        # (batch, D) -> (batch, D, D) with triangular zeros.
        return x.unsqueeze(-2) * self._input_mask

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = MLP.forward(self, self._masked_inputs(x))  # (batch, D, L*(m*r+r))
        batch, d, _ = out.shape
        per = self.m * self.r + self.r
        out = out.reshape(batch, d, self.n_layers, per)
        C = out[..., : self.m * self.r].reshape(
            batch, d, self.n_layers, self.m, self.r
        )
        raw_s = out[..., self.m * self.r :]
        return C, raw_s


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
