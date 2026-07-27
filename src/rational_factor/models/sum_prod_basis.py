import torch

from .basis_functions import Basis, SeparableBasis
from .parameters import Parameters


class SumProdBasis(Basis):
    def __init__(self, leaf_basis : SeparableBasis, matrix_coeffs : Parameters):
        dim = leaf_basis.dim()
        batch_size = leaf_basis.batch_size()
        n_basis = leaf_basis.n_basis_functions()

        self.leaf_basis = leaf_basis

        super().__init__(dim, batch_size, n_basis, leaf_basis._params, coeffs=matrix_coeffs)
    
    def set_coeffs(self, coeffs : Parameters):
        assert isinstance(coeffs, Parameters), "coeffs must be a Parameters object"
        assert coeffs().dim() == 4, "matrix coeffs must have shape (batch_size, dim, n_basis, n_basis)"
        assert coeffs().size() == (self._batch_size, self._dim, self._n_basis, self._n_basis), "coeffs must have shape (batch_size, dim, n_basis, n_basis)"
        self.coeffs = coeffs
    
    def set_coeffs_to_one(self):
        dtype, device = self.param_dtype_device()
        self.set_coeffs(Parameters(torch.ones(self._batch_size, self._dim, self._n_basis, self._n_basis, dtype=dtype, device=device)))
    
    def Omega1(self, lows : torch.Tensor = None, highs : torch.Tensor = None):
        log_Omega1 = self.leaf_basis.log_Omega1(lows, highs)
