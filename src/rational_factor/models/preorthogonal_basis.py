import torch

from rational_factor.models.basis_functions import Basis
from rational_factor.models.mlp import MetzlerConeMLP
from rational_factor.models.mutual_bases import MutualPairBasis
from rational_factor.models.structured_matrices import Matrix


class PreorthogonalMutualBasis(torch.nn.Module, MutualPairBasis):
    r"""Mutual pair with a Metzler-cone free factor and fixed nominal 1D bases.

    With sacrificial coordinate ``y_s`` and free coordinates ``z = y_{\neq s}``,

        C(z) = \exp(M(z)), \qquad M(z) = \mathrm{mc\_mlp}(z),
        A(z) = \mathrm{diag}(n(z))\, C(z),
        B(z) = C(z)^{-T} = \exp(-M(z)^T),

    where ``n(z)`` is the product basis. Then

        \alpha(y) = A(z)\,\mathrm{nom\_alpha}(y_s),
        \beta(y) = B(z)\,\mathrm{nom\_beta}(y_s).
    """

    def __init__(
        self,
        nom_alpha_basis: Basis,
        nom_beta_basis: Basis,
        mc_mlp: MetzlerConeMLP,
        product_basis: Basis,
        sacrificial_index: int = 0,
    ):
        assert nom_alpha_basis.dim() == 1, "nom_alpha_basis must be 1D"
        assert nom_beta_basis.dim() == 1, "nom_beta_basis must be 1D"
        assert nom_alpha_basis.batch_size() == nom_beta_basis.batch_size(), (
            "nom_alpha_basis and nom_beta_basis must have the same batch size"
        )
        n_basis = nom_alpha_basis.n_basis_functions()
        assert nom_beta_basis.n_basis_functions() == n_basis, (
            "nom_alpha_basis and nom_beta_basis must have the same n_basis"
        )
        assert product_basis.n_basis_functions() == n_basis, (
            "product_basis n_basis must match the nominal bases"
        )
        assert mc_mlp.in_features() == product_basis.dim(), (
            "mc_mlp input dim must match product_basis.dim()"
        )
        assert mc_mlp.out_features() == n_basis, (
            "mc_mlp matrix size must match n_basis"
        )

        total_dim = product_basis.dim() + 1
        if not (0 <= sacrificial_index < total_dim):
            raise ValueError(
                f"sacrificial_index must be in [0, {total_dim}), got {sacrificial_index}"
            )

        torch.nn.Module.__init__(self)
        MutualPairBasis.__init__(
            self,
            total_dim,
            nom_alpha_basis.batch_size(),
            n_basis,
            (),
        )

        self.nom_alpha_basis = nom_alpha_basis
        self.nom_beta_basis = nom_beta_basis
        self._product_basis = product_basis
        self._mc_mlp = mc_mlp
        self.sacrificial_index = sacrificial_index
        self._G: Matrix = nom_alpha_basis.Omega2(nom_beta_basis).T

    def dtype_device(self):
        return self._mc_mlp.K.dtype, self._mc_mlp.K.device

    def _split_coords(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = torch.as_tensor(y)
        if y.ndim == 1:
            y = y.unsqueeze(0)
        if y.ndim != 2 or y.shape[1] != self._dim:
            raise ValueError(
                f"y must have shape (n_data, {self._dim}), got {tuple(y.shape)}"
            )
        l = self.sacrificial_index
        rest = [i for i in range(self._dim) if i != l]
        return y[:, l : l + 1], y[:, rest]

    def _eval_basis(self, basis: Basis, x: torch.Tensor) -> torch.Tensor:
        return basis(x)

    def _A(self, X: Matrix, z: torch.Tensor) -> Matrix:
        n = self._eval_basis(self._product_basis, z)
        return X.expm().mul_diag_left(n)

    def _B(self, X: Matrix) -> Matrix:
        return X.T.scale(-1.0).expm()
    
    def eval(self, y: torch.Tensor | None = None, index: int | None = None):
        if y is None:
            return torch.nn.Module.eval(self)
        if index not in (0, 1, None):
            raise ValueError("index must be 0, 1, or None")

        y0, z = self._split_coords(y)
        X = self._mc_mlp(z)

        def _alpha():
            A = self._A(X, z)
            assert (A.to_dense() >= 0).all(), "A must be non-negative"
            return self._A(X, z).matvec(self._eval_basis(self.nom_alpha_basis, y0))

        def _beta():
            # beta_perp = G^{-1} beta(y0)
            beta_perp = self._G.inverse_matvec(self._eval_basis(self.nom_beta_basis, y0))

            # beta_pre = B(z) beta_perp(x0)
            beta_pre = self._B(X).matvec(beta_perp)

            # beta = G beta_pre
            return self._G.matvec(beta_pre)

        if index == 0:
            return _alpha()
        if index == 1:
            return _beta()

        return torch.stack([_alpha(), _beta()], dim=1)

    def Omega2(self, lows: torch.Tensor = None, highs: torch.Tensor = None) -> Matrix:
        if lows is not None or highs is not None:
            raise NotImplementedError(
                "PreorthogonalMutualBasis.Omega2 is only defined on the full domain"
            )
        try:
            if self._product_basis.normalized():
                return self._G
        except NotImplementedError:
            pass
        prod_Omega1 = self._product_basis.Omega1()
        return self._G.mul_diag_left(prod_Omega1)
