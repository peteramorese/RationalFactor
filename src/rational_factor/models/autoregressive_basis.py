import torch
import torch.nn.functional as F
from collections.abc import Sequence
import math

from rational_factor.models.basis_functions import Basis
from rational_factor.models.mlp import PairedMaskedMetzlerConeMLP
from rational_factor.models.mutual_bases import MutualPairBasis
from rational_factor.models.parameters import LowRankFactorizationParameters
from rational_factor.models.structured_matrices import DenseMatrix, Matrix, as_matrix, LowRankFactorization


class AutoregressiveMetzlerConeMutualBasis(torch.nn.Module, MutualPairBasis):
    r"""Product mutual pair with triangular paired Metzler-cone factors.

    Nominal 1D bases ``nom_alpha``, ``nom_beta`` share Gram ``G``. Fixed
    nonnegative factors ``A0, B0`` of shape ``(d, m, m)`` and a
    :class:`~rational_factor.models.mlp.PairedMaskedMetzlerConeMLP` produce

        R(x), T(x) \in \mathbb{R}^{d \times m \times m}

    with a **shared** coefficient MLP so each paired ray ``(R_{ℓ,k}, T_{ℓ,k})``
    gets the same weight. Slot ``ℓ`` depends only on ``x_<ℓ``. For ``ℓ ≥ 1``,

        \alpha_ℓ(x_{≤ℓ}) = \exp(R_ℓ(x_<ℓ))\, A0_ℓ\, \mathrm{nom\_alpha}(x_ℓ),
        \beta_ℓ(x_{≤ℓ}) = \exp(T_ℓ(x_<ℓ))\, B0_ℓ\, \mathrm{nom\_beta}(x_ℓ),

    and ``α_0 = β_0 = 1`` (excluded from the product for now). The pair is

        \alpha(x) = \prod_{ℓ=1}^{d-1} \alpha_ℓ(x_{≤ℓ}),
        \beta(x) = \prod_{ℓ=1}^{d-1} \beta_ℓ(x_{≤ℓ})

    (elementwise over basis index). Because paired rays satisfy
    ``R_ℓ Γ_ℓ + Γ_ℓ T_ℓ^\top = 0`` with the same weights,

        \exp(R_ℓ)\, Γ_ℓ\, \exp(T_ℓ)^\top = Γ_ℓ,

    independent of ``x_<ℓ``. Integrating in reverse coordinate order therefore
    yields the Hadamard product over the **same** slots that appear in the
    product (currently ``ℓ = 1, …, d-1``; slot ``0`` is omitted while
    ``α_0 = β_0 = 1``):

        \Omega^2 = \bigodot_{ℓ=1}^{d-1} \Gamma_ℓ,
        \Gamma_ℓ = A0_ℓ\, G\, B0_ℓ^\top.
    """

    def __init__(
        self,
        nom_alpha_basis: Basis,
        nom_beta_basis: Basis,
        A0: Matrix | torch.Tensor,
        B0: Matrix | torch.Tensor,
        paired_mc_mlp: PairedMaskedMetzlerConeMLP,
    ):
        if nom_alpha_basis.dim() != 1:
            raise ValueError("nom_alpha_basis must be 1D")
        if nom_beta_basis.dim() != 1:
            raise ValueError("nom_beta_basis must be 1D")
        if nom_alpha_basis.batch_size() != nom_beta_basis.batch_size():
            raise ValueError(
                "nom_alpha_basis and nom_beta_basis must have the same batch size"
            )
        n_basis = nom_alpha_basis.n_basis_functions()
        if nom_beta_basis.n_basis_functions() != n_basis:
            raise ValueError(
                "nom_alpha_basis and nom_beta_basis must have the same n_basis"
            )

        A0 = as_matrix(A0)
        B0 = as_matrix(B0)
        d = paired_mc_mlp.n_slots()
        m = paired_mc_mlp.out_features()
        if m != n_basis:
            raise ValueError(
                f"MLP matrix size m={m} must match n_basis={n_basis}"
            )
        if A0.shape != (d, m, m) or B0.shape != (d, m, m):
            raise ValueError(
                f"A0 and B0 must have shape ({d}, {m}, {m}), "
                f"got A0={tuple(A0.shape)}, B0={tuple(B0.shape)}"
            )

        torch.nn.Module.__init__(self)
        MutualPairBasis.__init__(
            self,
            d,
            nom_alpha_basis.batch_size(),
            n_basis,
            (),
        )

        self.nom_alpha_basis = nom_alpha_basis
        self.nom_beta_basis = nom_beta_basis
        self.paired_mc_mlp = paired_mc_mlp

        A0_dense = A0.to_dense().detach().clone()
        B0_dense = B0.to_dense().detach().clone()
        self.register_buffer("_A0", A0_dense)
        self.register_buffer("_B0", B0_dense)

        G = nom_alpha_basis.Omega2(nom_beta_basis).to_dense().detach()
        if G.dim() == 3:
            if G.shape[0] != 1:
                raise ValueError(
                    f"nominal Gram batch size must be 1, got {tuple(G.shape)}"
                )
            G = G.squeeze(0)
        G = G.to(dtype=A0_dense.dtype, device=A0_dense.device)
        # Γ_ℓ = A0_ℓ G B0_ℓᵀ.  Eval skips ℓ=0 (α_0=β_0=1), so Ω² must too.
        gamma = torch.matmul(torch.matmul(A0_dense, G), B0_dense.transpose(-1, -2))
        if d == 1:
            # No active product factors; α=β=1 ⇒ Ω² = J (all-ones) on the unit box.
            omega2 = torch.ones(m, m, dtype=A0_dense.dtype, device=A0_dense.device)
        else:
            omega2 = gamma[1:].prod(dim=0)
        self.register_buffer("_omega2", omega2)
        self.register_buffer("_gamma", gamma)
        self.register_buffer("_G", G)

    @property
    def A0(self) -> DenseMatrix:
        return DenseMatrix(self._A0)

    @property
    def B0(self) -> DenseMatrix:
        return DenseMatrix(self._B0)

    @property
    def G(self) -> DenseMatrix:
        return DenseMatrix(self._G)

    def dtype_device(self):
        return self._A0.dtype, self._A0.device

    def _eval_nom(self, basis: Basis, x_l: torch.Tensor) -> torch.Tensor:
        return basis(x_l)

    def _factor_product(
        self,
        y: torch.Tensor,
        generators: Matrix,
        factors: torch.Tensor,
        nom_basis: Basis,
    ) -> torch.Tensor:
        """Elementwise product of ``expm(M_ℓ) @ F_ℓ @ nom(x_ℓ)`` over ``ℓ ≥ 1``."""
        y = torch.as_tensor(y)
        if y.ndim == 1:
            y = y.unsqueeze(0)
        if y.ndim != 2 or y.shape[1] != self._dim:
            raise ValueError(
                f"y must have shape (n_data, {self._dim}), got {tuple(y.shape)}"
            )

        n_data, d = y.shape
        m = self._n_basis
        M = generators.to_dense()
        if M.shape[-3:] != (d, m, m):
            raise ValueError(
                f"generator batch must end in ({d}, {m}, {m}), got {tuple(M.shape)}"
            )
        # Broadcast unbatched generators over the data batch.
        if M.dim() == 3:
            M = M.unsqueeze(0).expand(n_data, -1, -1, -1)
        elif M.shape[0] != n_data:
            raise ValueError(
                f"generator leading batch {M.shape[0]} must match n_data={n_data}"
            )

        out = y.new_ones(n_data, m)
        for ell in range(1, d):
            nom = self._eval_nom(nom_basis, y[:, ell : ell + 1])
            # v = F_ℓ nom(x_ℓ), then exp(M_ℓ) v
            v = torch.einsum("ij,...j->...i", factors[ell], nom)
            v = torch.einsum("...ij,...j->...i", torch.matrix_exp(M[:, ell]), v)
            out = out * v
        return out

    def eval(self, y: torch.Tensor | None = None, index: int | None = None):
        if y is None:
            return torch.nn.Module.eval(self)
        if index not in (0, 1, None):
            raise ValueError("index must be 0, 1, or None")

        y = torch.as_tensor(y)
        if y.ndim == 1:
            y = y.unsqueeze(0)

        R, T = self.paired_mc_mlp(y)

        def _alpha():
            alpha = self._factor_product(y, R, self._A0, self.nom_alpha_basis)
            assert (alpha > 0).all(), "alpha must be positive"
            return alpha

        def _beta():
            beta = self._factor_product(y, T, self._B0, self.nom_beta_basis)
            assert (beta > 0).all(), "beta must be positive"
            return beta

        if index == 0:
            return _alpha()
        if index == 1:
            return _beta()
        return torch.stack([_alpha(), _beta()], dim=1)

    def Omega2(self, lows: torch.Tensor = None, highs: torch.Tensor = None) -> Matrix:
        if lows is not None or highs is not None:
            raise NotImplementedError(
                "AutoregressiveMetzlerConeMutualBasis.Omega2 is only defined "
                "on the full domain"
            )
        return DenseMatrix(self._omega2)


class _PrefixUpdateMLP(torch.nn.Module):
    """MLP on x_<d producing C in R^{m x r} and r raw row-wise steps."""

    def __init__(
        self,
        in_features: int,
        m: int,
        r: int,
        hidden_features: Sequence[int],
    ):
        super().__init__()
        self.m = m
        self.r = r
        out_features = m * r + r

        if in_features == 0:
            # Slot d=0 has an empty autoregressive context, so its update is constant.
            self.constant = torch.nn.Parameter(torch.zeros(out_features))
            self.net = None
            return

        widths = [in_features, *hidden_features, out_features]
        layers: list[torch.nn.Module] = []
        for i in range(len(widths) - 2):
            layers += [torch.nn.Linear(widths[i], widths[i + 1]), torch.nn.SiLU()]
        final = torch.nn.Linear(widths[-2], widths[-1])
        # Start from the supplied A0/B0. The network learns residual cone moves.
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = torch.nn.Sequential(*layers)
        self.register_parameter("constant", None)

    def forward(self, prefix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.net is None:
            out = self.constant.unsqueeze(0).expand(prefix.shape[0], -1)
        else:
            out = self.net(prefix)
        split = self.m * self.r
        C = out[..., :split].reshape(prefix.shape[0], self.m, self.r)
        raw_s = out[..., split:]  # (batch, r)
        return C, raw_s


class AutoregressiveConeLayerBasis(torch.nn.Module, MutualPairBasis):
    r"""Autoregressive mutual basis with alternating exact nullspace updates.

    For each coordinate d, the nominal one-dimensional bases share a fixed Gram G
    and the initial factors are rank-r matrices

        A0_d = L_A,d X_A,d,       B0_d = L_B,d X_B,d,

    with L in R^{m x r}, X in R^{r x m}.  The initial factors are trainable,
    hence

        M_d = A0_d G B0_d^T

    is trainable too.  Cone layers preserve this M_d pointwise in x_<d.

    Layers alternate B and A updates, starting with B.  For a B update,

        K_A = X_A G,
        Z   = P_null(K_A) C(x_<d),
        X_B <- X_B + diag(s(x_<d)) Z^T,

    where

        P_null(K) C = C - K^T (K K^T)^{-1} K C.

    Therefore K_A Z = 0 and A G B^T is unchanged.  A updates are symmetric with
    K_B = X_B G^T.  The vector s in R_+^r is bounded independently for each
    latent row so that the updated right factor X remains entrywise nonnegative.
    Since L is nonnegative, this is sufficient to keep A = L_A X_A and
    B = L_B X_B entrywise nonnegative.

    The final autoregressive bases are

        alpha(x) = prod_d A_d(x_<d) nom_alpha(x_d),
        beta(x)  = prod_d B_d(x_<d) nom_beta(x_d),

    with elementwise products over d.  Reverse autoregressive integration gives

        Omega2 = odot_d M_d.

    Notes
    -----
    * ``A0`` and ``B0`` must be :class:`LowRankFactorizationParameters` whose
      ``()`` yields factors with one batch axis of length D and shape
      ``(D, m, m)``.  Their factor rank must be ``ceil(m ** (1 / D))``.
    * Use :class:`~rational_factor.models.parameters.PositiveParameters` for the
      U/V factors so the starting A0/B0 stay entrywise nonnegative under
      optimization.  Positivity of updated factors is enforced by the bounded
      cone steps.
    * Exact nullspace projection assumes K_A and K_B retain full row rank r.
      This is generic but can fail at a rank-degenerate parameter setting.
    """

    def __init__(
        self,
        nom_alpha_basis: Basis,
        nom_beta_basis: Basis,
        A0: LowRankFactorizationParameters,
        B0: LowRankFactorizationParameters,
        n_layers: int,
        hidden_features: int | Sequence[int] = (64, 64),
        *,
        positivity_margin: float = 1e-6,
    ):
        if nom_alpha_basis.dim() != 1 or nom_beta_basis.dim() != 1:
            raise ValueError("nom_alpha_basis and nom_beta_basis must both be 1D")
        if nom_alpha_basis.batch_size() != nom_beta_basis.batch_size():
            raise ValueError("nom_alpha_basis and nom_beta_basis must share batch size")

        m = nom_alpha_basis.n_basis_functions()
        if nom_beta_basis.n_basis_functions() != m:
            raise ValueError("nom_alpha_basis and nom_beta_basis must have the same n_basis")
        if not isinstance(A0, LowRankFactorizationParameters) or not isinstance(
            B0, LowRankFactorizationParameters
        ):
            raise TypeError("A0 and B0 must be LowRankFactorizationParameters instances")

        A0_mat = A0()
        B0_mat = B0()
        if A0_mat.shape != B0_mat.shape:
            raise ValueError(
                f"A0 and B0 must have the same shape, got {A0_mat.shape} and {B0_mat.shape}"
            )
        if len(A0_mat.batch_shape) != 1 or A0_mat.n != m or A0_mat.m != m:
            raise ValueError(
                f"A0/B0 must have shape (D, {m}, {m}); got A0.shape={tuple(A0_mat.shape)}"
            )
        if A0_mat.r != B0_mat.r:
            raise ValueError(
                f"A0 and B0 must share factor rank, got {A0_mat.r} and {B0_mat.r}"
            )
        if n_layers < 0:
            raise ValueError("n_layers must be nonnegative")
        if not (0.0 <= positivity_margin < 1.0):
            raise ValueError("positivity_margin must lie in [0, 1)")

        D = A0_mat.batch_shape[0]
        r = A0_mat.r
        expected_r = math.ceil(m ** (1.0 / D))
        if r != expected_r:
            raise ValueError(
                f"factor rank must be ceil(m**(1/D))={expected_r}, got r={r} "
                f"for m={m}, D={D}"
            )

        # A0 = U V^T = L X, so L=U and X=V^T; require nonnegative starting factors.
        if min(
            A0_mat.U.min(),
            A0_mat.V.min(),
            B0_mat.U.min(),
            B0_mat.V.min(),
        ).item() < 0:
            raise ValueError(
                "A0/B0 low-rank factors must be entrywise nonnegative; "
                "use PositiveParameters for U and V"
            )

        if isinstance(hidden_features, int):
            hidden_features = (hidden_features, hidden_features)
        else:
            hidden_features = tuple(hidden_features)

        torch.nn.Module.__init__(self)
        MutualPairBasis.__init__(
            self,
            D,
            nom_alpha_basis.batch_size(),
            m,
            (),
        )

        self.nom_alpha_basis = nom_alpha_basis
        self.nom_beta_basis = nom_beta_basis
        self._A0 = A0
        self._B0 = B0
        self.n_layers = n_layers
        self.rank = r
        self.positivity_margin = float(positivity_margin)

        # Nominal basis parameters are fixed by assumption.
        for basis in (nom_alpha_basis, nom_beta_basis):
            for params in basis._params_register():
                for param in params:
                    if hasattr(param, "set_requires_grad"):
                        param.set_requires_grad(False)

        # Own the factor parameter modules so they appear in parameters()/to().
        factor_modules = A0.parameter_modules() + B0.parameter_modules()
        self._factor_param_modules = torch.nn.ModuleList(dict.fromkeys(factor_modules))

        G = nom_alpha_basis.Omega2(nom_beta_basis).to_dense().detach()
        if G.dim() == 3:
            if G.shape[0] != 1:
                raise ValueError(f"nominal Gram batch size must be 1, got {tuple(G.shape)}")
            G = G.squeeze(0)
        if G.shape != (m, m):
            raise ValueError(f"nominal Gram must have shape ({m}, {m}), got {tuple(G.shape)}")
        G = G.to(dtype=A0_mat.dtype, device=A0_mat.device)
        if torch.linalg.matrix_rank(G).item() != m:
            raise ValueError("nominal Gram G must be full rank")
        self.register_buffer("_G", G)

        # One autoregressive update network per (layer, dimension).  Layer parity
        # determines which side it updates: even -> B, odd -> A.
        self.update_mlps = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [
                        _PrefixUpdateMLP(d, m, r, hidden_features)
                        for d in range(D)
                    ]
                )
                for _ in range(n_layers)
            ]
        )

    @property
    def L_A(self) -> torch.Tensor:
        # A0 = U V^T = L X  ⇒  L = U.
        return self._A0().U

    @property
    def X_A0(self) -> torch.Tensor:
        # A0 = U V^T = L X  ⇒  X = V^T.
        return self._A0().V.transpose(-2, -1)

    @property
    def L_B(self) -> torch.Tensor:
        return self._B0().U

    @property
    def X_B0(self) -> torch.Tensor:
        return self._B0().V.transpose(-2, -1)

    @property
    def A0(self) -> LowRankFactorization:
        return self._A0()

    @property
    def B0(self) -> LowRankFactorization:
        return self._B0()

    @property
    def G(self) -> DenseMatrix:
        return DenseMatrix(self._G)

    def dtype_device(self):
        return self._G.dtype, self._G.device

    @staticmethod
    def _project_null(K: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        r"""Project columns of C onto ker(K), batched over the leading axis.

        K: (batch, r, m), C: (batch, m, r)
        returns Z: (batch, m, r) with K @ Z = 0 up to floating-point error.
        """
        KKT = K @ K.transpose(-2, -1)               # (batch, r, r)
        KC = K @ C                                   # (batch, r, r)
        coeff = torch.linalg.solve(KKT, KC)          # (batch, r, r)
        return C - K.transpose(-2, -1) @ coeff       # (batch, m, r)

    def _bounded_positive_step(
        self,
        Y: torch.Tensor,
        dY: torch.Tensor,
        raw_s: torch.Tensor,
    ) -> torch.Tensor:
        r"""Return row-wise s >= 0 with Y + diag(s) dY >= 0.

        ``Y`` and ``dY`` have shape ``(batch, r, m)`` and ``raw_s`` has shape
        ``(batch, r)``.  Each latent row k gets its own exact pointwise bound

            s_max[k] = min_{j: dY[k,j] < 0} Y[k,j] / (-dY[k,j]).

        Thus one restrictive entry only limits its own latent row rather than
        all r rows.  If a row of dY has no negative entry, positivity imposes
        no upper bound on that row and we use softplus(raw_s[k]).

        ``s_max`` is detached and capped: backprop through the nondifferentiable
        ``amin`` / reciprocal of near-zero ``dY`` can be numerically unstable,
        and an uncapped bound can make a tiny negative entry yield a huge step.
        """
        neg = dY < 0
        # Floor away from 0 so barely-negative entries cannot explode s_max.
        denom = (-dY).clamp_min(1e-3)
        ratios = torch.where(
            neg,
            Y.clamp_min(0.0) / denom,
            torch.full_like(dY, torch.inf),
        )
        s_max = ratios.amin(dim=-1)  # (batch, r)
        has_bound = torch.isfinite(s_max)
        s_cap = (
            torch.nan_to_num(s_max, nan=0.0, posinf=0.0, neginf=0.0)
            .clamp(0.0, 10.0)
            .detach()
        )
        bounded = (1.0 - self.positivity_margin) * s_cap * torch.sigmoid(raw_s)
        unbounded = F.softplus(raw_s)
        return torch.where(has_bound, bounded, unbounded)

    def _slot_state(
        self,
        y: torch.Tensor,
        d: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pointwise X_A(x_<d), X_B(x_<d) for one coordinate slot."""
        n_data = y.shape[0]
        G = self._G
        XA = self.X_A0[d].unsqueeze(0).expand(n_data, -1, -1)  # (b, r, m)
        XB = self.X_B0[d].unsqueeze(0).expand(n_data, -1, -1)
        prefix = y[:, :d]

        for ell in range(self.n_layers):
            C, raw_s = self.update_mlps[ell][d](prefix)         # C: (b, m, r)

            if ell % 2 == 0:
                # B update. Z lies in ker(X_A G).
                K = XA @ G                                      # (b, r, m)
                Z = self._project_null(K, C)                     # (b, m, r)
                dX = Z.transpose(-2, -1)                         # (b, r, m)

                # Row-wise positivity bound on the right factor X_B.
                # X_B >= 0 and L_B >= 0 imply B = L_B X_B >= 0.
                s = self._bounded_positive_step(XB, dX, raw_s)  # (b, r)
                XB = XB + s.unsqueeze(-1) * dX
            else:
                # A update. Z lies in ker(X_B G^T).
                K = XB @ G.transpose(-2, -1)                    # (b, r, m)
                Z = self._project_null(K, C)                     # (b, m, r)
                dX = Z.transpose(-2, -1)                         # (b, r, m)

                # Row-wise positivity bound on the right factor X_A.
                # X_A >= 0 and L_A >= 0 imply A = L_A X_A >= 0.
                s = self._bounded_positive_step(XA, dX, raw_s)  # (b, r)
                XA = XA + s.unsqueeze(-1) * dX

        return XA, XB

    def _eval_nom(self, basis: Basis, x_d: torch.Tensor) -> torch.Tensor:
        return basis(x_d)

    def eval(self, y: torch.Tensor | None = None, index: int | None = None):
        if y is None:
            return torch.nn.Module.eval(self)
        if index not in (0, 1, None):
            raise ValueError("index must be 0 (alpha), 1 (beta), or None")

        dtype, device = self.dtype_device()
        y = torch.as_tensor(y, dtype=dtype, device=device)
        if y.ndim == 1:
            y = y.unsqueeze(0)
        if y.ndim != 2 or y.shape[1] != self._dim:
            raise ValueError(
                f"y must have shape (n_data, {self._dim}), got {tuple(y.shape)}"
            )

        n_data = y.shape[0]
        m = self._n_basis
        alpha = y.new_ones(n_data, m)
        beta = y.new_ones(n_data, m)

        for d in range(self._dim):
            XA, XB = self._slot_state(y, d)
            nom_a = self._eval_nom(self.nom_alpha_basis, y[:, d : d + 1])
            nom_b = self._eval_nom(self.nom_beta_basis, y[:, d : d + 1])

            # A nom = L_A (X_A nom), avoiding materialization of dense A/B here.
            u_a = torch.einsum("brm,bm->br", XA, nom_a)
            u_b = torch.einsum("brm,bm->br", XB, nom_b)
            a_d = torch.einsum("mr,br->bm", self.L_A[d], u_a)
            b_d = torch.einsum("mr,br->bm", self.L_B[d], u_b)
            alpha = alpha * a_d
            beta = beta * b_d

        if index == 0:
            return alpha
        if index == 1:
            return beta
        return torch.stack([alpha, beta], dim=1)

    def Omega2(self, lows: torch.Tensor = None, highs: torch.Tensor = None) -> Matrix:
        if lows is not None or highs is not None:
            raise NotImplementedError(
                "AutoregressiveConeLayerBasis.Omega2 is only defined on the full domain"
            )

        # M_d = L_A (X_A G X_B^T) L_B^T, using the trainable *initial* factors.
        # Every cone layer preserves M_d pointwise, so the total Gram is the
        # Hadamard product across coordinates.
        middle = (self.X_A0 @ self._G) @ self.X_B0.transpose(-2, -1)  # (D,r,r)
        M = (self.L_A @ middle) @ self.L_B.transpose(-2, -1)          # (D,m,m)
        return DenseMatrix(M.prod(dim=0))