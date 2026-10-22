import torch

from .basis_functions import Basis, SeparableBasis
from .parameters import Parameters


class SumProdBasis(SeparableBasis):
    def __init__(self, n_output_basis : int, leaf_basis : SeparableBasis, matrix_coeffs : Parameters, coeffs : Parameters = None):
        dim = leaf_basis.dim()
        batch_size = leaf_basis.batch_size()

        self.leaf_basis = leaf_basis

        # n_basis is the mixed output size; params come from the leaf.
        Basis.__init__(
            self,
            dim=dim,
            batch_size=batch_size,
            n_basis=n_output_basis,
            params=leaf_basis._params,
            coeffs=coeffs,
        )

        self.set_matrix_coeffs(matrix_coeffs)

    def _coeffs_register(self):
        return [self.coeffs, self.matrix_coeffs]
    
    def _params_register(self):
        return [self.leaf_basis._params]

    def set_matrix_coeffs(self, matrix_coeffs : Parameters):
        assert isinstance(matrix_coeffs, Parameters), "matrix coeffs must be a Parameters object"
        assert matrix_coeffs().dim() == 4, "matrix coeffs must have shape (batch_size, dim, n_output_basis, n_leaf_basis)"
        assert matrix_coeffs().size() == (
            self._batch_size,
            self._dim,
            self._n_basis,
            self.leaf_basis.n_basis_functions(),
        ), "matrix coeffs must have shape (batch_size, dim, n_output_basis, n_leaf_basis)"
        self.matrix_coeffs = matrix_coeffs

    def __call__(self, x : torch.Tensor):
        self._check_eval_batch(x)
        # (batch, dim, n_leaf)
        leaf_dim = self.leaf_basis.eval_dim(x)
        M = self.matrix_coeffs()  # (batch_size, dim, n_output, n_leaf)
        if M.shape[0] == 1 and leaf_dim.shape[0] > 1:
            M = M.expand(leaf_dim.shape[0], -1, -1, -1)
        # (batch, dim, n_output)
        mixed_dim = torch.einsum("bdl,bdil->bdi", leaf_dim, M)
        return mixed_dim.prod(dim=1) * self.coeffs()

    def Omega1_dim(self, lows : torch.Tensor = None, highs : torch.Tensor = None, return_log : bool = False):
        # Mixed values can be signed, so combine in linear space and only
        # take a log if the caller explicitly requests it.
        leaf_Omega1_dim = self.leaf_basis.Omega1_dim(lows, highs, return_log=False)  # (batch, dim, n_leaf)
        # Per (batch, dim): matrix_coeffs[b,d] @ leaf_Omega1[b,d]
        out = torch.einsum("bdij,bdj->bdi", self.matrix_coeffs(), leaf_Omega1_dim)
        return torch.log(out) if return_log else out

    def Omega2_dim(self, other : "SumProdBasis", lows : torch.Tensor = None, highs : torch.Tensor = None, return_log : bool = False):
        leaf_Omega2_dim = self.leaf_basis.Omega2_dim(other.leaf_basis, lows, highs, return_log=False)  # (batch, dim, n_leaf, n_leaf)
        # Per (batch, dim): A @ Omega @ B^T
        out = torch.einsum("bdij,bdjk,bdlk->bdil", self.matrix_coeffs(), leaf_Omega2_dim, other.matrix_coeffs())
        return torch.log(out) if return_log else out
