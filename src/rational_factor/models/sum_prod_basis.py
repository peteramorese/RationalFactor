import torch

from .basis_functions import Basis, SeparableBasis
from .parameters import Parameters


class SumProdBasis(SeparableBasis):
    def __init__(self, n_output_basis : int, leaf_basis : SeparableBasis, matrix_coeffs : Parameters, coeffs : Parameters = None):
        dim = leaf_basis.dim()
        batch_size = leaf_basis.batch_size()
        n_basis = leaf_basis.n_basis_functions()

        self.leaf_basis = leaf_basis

        super().__init__(dim, batch_size, n_output_basis, leaf_basis._params, coeffs=coeffs)

        self.set_matrix_coeffs(matrix_coeffs)
    
    def set_matrix_coeffs(self, matrix_coeffs : Parameters):
        assert isinstance(matrix_coeffs, Parameters), "matrix coeffs must be a Parameters object"
        assert matrix_coeffs().dim() == 4, "matrix coeffs must have shape (batch_size, dim, n_basis, n_basis)"
        assert matrix_coeffs().size() == (self._batch_size, self._dim, self._n_basis, self._n_basis), "matrix coeffs must have shape (batch_size, dim, n_basis, n_basis)"
        self.matrix_coeffs = matrix_coeffs
    
    def log_Omega1_dim(self, lows : torch.Tensor = None, highs : torch.Tensor = None):
        leaf_Omega1_dim = torch.exp(self.leaf_basis.log_Omega1_dim(lows, highs))  # (batch, dim, n_basis)
        # Per (batch, dim): matrix_coeffs[b,d] @ log_Omega1[b,d]
        return torch.log(torch.einsum("bdij,bdj->bdi", self.matrix_coeffs(), leaf_Omega1_dim))

    def log_Omega2_dim(self, other : "SumProdBasis", lows : torch.Tensor = None, highs : torch.Tensor = None):
        leaf_Omega2_dim = torch.exp(self.leaf_basis.log_Omega2_dim(other.leaf_basis, lows, highs))  # (batch, dim, n_basis)
        # Per (batch, dim): matrix_coeffs[b,d] @ log_Omega2[b,d] @ other.matrix_coeffs[b,d]^T
        return torch.log(torch.einsum("bdij,bdjk,bdlk->bdil", self.matrix_coeffs(), leaf_Omega2_dim, other.matrix_coeffs()))
