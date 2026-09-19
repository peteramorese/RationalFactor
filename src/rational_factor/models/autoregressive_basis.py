import torch

from rational_factor.models.basis_functions import Basis
from rational_factor.models.mlp import PairedMaskedMetzlerConeMLP
from rational_factor.models.mutual_bases import MutualPairBasis
from rational_factor.models.structured_matrices import DenseMatrix, Matrix, as_matrix


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
