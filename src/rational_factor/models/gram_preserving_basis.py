import torch

from rational_factor.models.basis_functions import Basis
from rational_factor.models.mutual_bases import MutualPairBasis
from rational_factor.models.space_splitter import LatentReflectionSpaceSplitter
from rational_factor.models.mlp import MLP


class DeepGramPreservingBasis(MutualPairBasis, torch.nn.Module):
    def __init__(
        self,
        base_phi: Basis,
        base_psi: Basis,
        space_splitter: LatentReflectionSpaceSplitter,
        deformer: MLP,
        fixed_base_basis: bool = False,
        smooth_boundary: bool = False,
        eps: float = 1e-8,
    ):
        torch.nn.Module.__init__(self)
        n_basis = base_phi.n_basis_functions()
        assert n_basis == base_psi.n_basis_functions(), (
            "Base phi and psi must have the same number of basis functions"
        )
        n_layers = deformer.out_features()
        n_maps = space_splitter.tf.n_mappings
        if n_layers != n_maps:
            raise ValueError(
                f"deformer out_features {n_layers} must match "
                f"space_splitter n_mappings {n_maps}"
            )

        super().__init__(
            base_phi.dim(),
            base_phi.batch_size(),
            n_basis,
            tuple(base_phi._params) + tuple(base_psi._params),
        )

        self._base_phi = base_phi
        self._base_psi = base_psi
        self._space_splitter = space_splitter
        self._deformer = deformer
        self._n_layers = n_layers
        self._fixed_base_basis = fixed_base_basis
        self._base_gram = None
        self._smooth_boundary = smooth_boundary
        self._eps = eps

        if fixed_base_basis:
            self._base_gram = base_phi.Omega2(base_psi)

    def _eval_base(self, basis: Basis, y: torch.Tensor) -> torch.Tensor:
        """Evaluate a base basis on ``y``, shape ``(n_data, n_basis)``."""
        out = basis(y)
        if out.ndim != 2 or out.shape[0] != y.shape[0] or out.shape[1] != self._n_basis:
            raise ValueError(
                f"base basis must return shape {(y.shape[0], self._n_basis)}, "
                f"got {tuple(out.shape)}"
            )
        return out

    def _constrained_s(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        alpha_p: torch.Tensor,
        beta_p: torch.Tensor,
        J: torch.Tensor,
        sigma: torch.Tensor,
        d_to_boundary: torch.Tensor,
    ) -> torch.Tensor:
        """Scalar deformer ``s ∈ [L, U]`` from positivity bounds at a set-0 point.

        ``L = max_j (-β_j / (J β_p,j))``, ``U = min_j α_p,j / α_j`` (with
        ``α_j > 0``), and ``s = (U - L) σ + L`` with ``σ ∈ (0, 1)``.
        """
        eps = self._eps
        J = J.unsqueeze(-1)
        L = (-beta / (J * beta_p).clamp_min(eps)).amax(dim=-1)
        ratios_u = torch.where(
            alpha > eps,
            alpha_p / alpha.clamp_min(eps),
            torch.full_like(alpha, float("inf")),
        )
        U = ratios_u.amin(dim=-1)
        U = torch.where(torch.isfinite(U), torch.maximum(U, L), L)
        s = (U - L) * sigma + L
        if self._smooth_boundary:
            return s * d_to_boundary.unsqueeze(-1)
        return s

    def _apply_layers(self, y: torch.Tensor, n_layers: int):
        """Return ``(alpha, beta)`` at ``y`` after ``n_layers`` reflections."""
        alpha = self._eval_base(self._base_phi, y)
        beta = self._eval_base(self._base_psi, y)
        if n_layers == 0:
            return alpha, beta

        y_partner, set_index, ladj, d_to_boundary = self._space_splitter.partner(y)
        if y_partner.shape[1] < n_layers:
            raise ValueError(
                f"need {n_layers} layers but splitter returned "
                f"{y_partner.shape[1]} mappings"
            )

        # σ_l(y) ∈ (0, 1); also need σ_l(y_partner) so s is shared across each pair.
        sigma = torch.sigmoid(self._deformer(y))
        if sigma.shape[-1] < n_layers:
            raise ValueError(
                f"deformer out_features {sigma.shape[-1]} < n_layers {n_layers}"
            )

        for l in range(n_layers):
            yp = y_partner[:, l, :]
            alpha_p, beta_p = self._apply_layers(yp, l)

            J = torch.exp(ladj[:, l])  # (n,)
            si = set_index[:, l]
            sig = sigma[:, l]
            sig_p = torch.sigmoid(self._deformer(yp))[:, l]

            # s is defined on set 0; set-1 points reuse s(partner).
            s_at_y = self._constrained_s(alpha, beta, alpha_p, beta_p, J, sig, d_to_boundary)
            s_at_p = self._constrained_s(
                alpha_p, beta_p, alpha, beta, torch.exp(-ladj[:, l]), sig_p, d_to_boundary
            )
            s = torch.where(si == 0, s_at_y, s_at_p).unsqueeze(-1)
            J = J.unsqueeze(-1)
            set0 = (si == 0).unsqueeze(-1)

            # Unimodular pair update (preserves ∫ α_i β_j):
            #   set 0: α' = α,           β' = β + J s β(partner)
            #   set 1: α' = α - s α(partner), β' = β
            alpha = torch.where(set0, alpha, alpha - s * alpha_p)
            beta = torch.where(set0, beta + J * s * beta_p, beta)

        return alpha, beta

    def eval(self, y: torch.Tensor, index: int = None):
        if index not in (0, 1, None):
            raise ValueError("index must be 0, 1, or None")
        if y.ndim == 1:
            y = y.unsqueeze(-1) if self._dim == 1 else y.unsqueeze(0)
        if y.ndim != 2 or y.shape[1] != self._dim:
            raise ValueError(
                f"y must have shape (n_data, {self._dim}), got {tuple(y.shape)}"
            )

        alpha, beta = self._apply_layers(y, self._n_layers)
        if index == 0:
            return alpha
        if index == 1:
            return beta
        return torch.stack([alpha, beta], dim=1)

    def Omega2(self, lows: torch.Tensor = None, highs: torch.Tensor = None):
        assert lows is None and highs is None, (
            "Omega2 over arbitrary regions is not implemented for this basis"
        )

        if self._fixed_base_basis:
            return self._base_gram
        else:
            return self._base_phi.Omega2(self._base_psi)
