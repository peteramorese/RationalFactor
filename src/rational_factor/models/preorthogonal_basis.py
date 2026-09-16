import torch

from rational_factor.models.basis_functions import Basis
from rational_factor.models.mlp import MetzlerConeMLP
from rational_factor.models.mutual_bases import MutualPairBasis
from rational_factor.models.structured_matrices import Matrix


class PreorthogonalMutualBasis(torch.nn.Module, MutualPairBasis):
    r"""Mutual pair with a Metzler-cone free factor and fixed nominal 1D bases.

    With sacrificial coordinate ``y_s`` and free coordinates ``z = y_{\neq s}``,

        b(z) = \exp(\mathrm{mc\_mlp}(z)),
        a(z) = \exp(\log n(z) - \mathrm{mc\_mlp}(z)) = n(z) / b(z),

    where ``n(z)`` is the product basis. Then

        \alpha(y) = a(z) \circ \mathrm{nom\_alpha}(y_s),
        \beta(y) = G\,\mathrm{diag}(b(z))\,G^{-1}\,\mathrm{nom\_beta}(y_s),

    with ``G = \Omega_2(\mathrm{nom\_alpha}, \mathrm{nom\_beta})^\top``. The beta
    map applies ``G^{-1}`` via linear solves, not an explicit inverse.
    """

    def __init__(
        self,
        nom_alpha_basis: Basis,
        nom_beta_basis: Basis,
        mc_mlp: MetzlerConeMLP,
        product_basis: Basis,
        sacrificial_index: int = 0,
        eps: float = 1e-12,
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
            "mc_mlp output dim must match n_basis"
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
        self.eps = eps
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

    def _solve_G(self, rhs: torch.Tensor) -> torch.Tensor:
        """Solve ``G x = rhs`` along the last axis without forming ``G^{-1}``."""
        G = self._G
        if hasattr(G, "inverse_matvec"):
            return G.inverse_matvec(rhs)
        if hasattr(G, "solve"):
            return G.solve(rhs)
        A = G.to_dense()
        while A.dim() > rhs.dim():
            A = A.squeeze(0)
        return torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)

    def _matvec_G(self, x: torch.Tensor) -> torch.Tensor:
        G = self._G
        try:
            return G.matvec(x)
        except ValueError:
            A = G.to_dense()
            while A.dim() > x.dim():
                A = A.squeeze(0)
            return (A @ x.unsqueeze(-1)).squeeze(-1)

    def _ab(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(a(z), b(z))`` with shape ``(n_data, n_basis)``."""
        n = self._eval_basis(self._product_basis, z)
        log_b = self._mc_mlp(z)
        b = torch.exp(log_b)
        a = torch.exp(torch.log(n.clamp_min(self.eps)) - log_b)
        return a, b

    def eval(self, y: torch.Tensor | None = None, index: int | None = None):
        if y is None:
            return torch.nn.Module.eval(self)
        if index not in (0, 1, None):
            raise ValueError("index must be 0, 1, or None")

        y0, z = self._split_coords(y)
        a, b = self._ab(z)

        if index == 0:
            return a * self._eval_basis(self.nom_alpha_basis, y0)

        nom_beta = self._eval_basis(self.nom_beta_basis, y0)
        # beta = G diag(b) G^{-1} nom_beta
        beta = self._matvec_G(b * self._solve_G(nom_beta))
        if index == 1:
            return beta

        alpha = a * self._eval_basis(self.nom_alpha_basis, y0)
        return torch.stack([alpha, beta], dim=1)

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
