import torch
from rational_factor.models.basis_functions import Basis
from rational_factor.models.parameters import Parameters, PositiveParameters
from rational_factor.models.lrpd import Rank1PlusDiagonal, Rank1PlusDiagonalFactorization

class MutualPairBasis:
    def __init__(self, 
            dim : int, 
            batch_size : int,
            n_basis : int,
            params : tuple[Parameters, ...],
            coeffs : tuple[Parameters, Parameters] = None):

        self._dim = dim
        self._batch_size = batch_size
        self._n_basis = n_basis
        self._params = params
        self._coeffs = coeffs

    def get_basis(self, index: int) -> Basis:
        raise NotImplementedError("get_basis is not implemented for this mutual basis")
    
    def Omega1(self, index : int, lows : torch.Tensor = None, highs : torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError("Omega1 is not implemented for this mutual basis")

    def Omega2(self, lows : torch.Tensor = None, highs : torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError("Omega2 is not implemented for this mutual basis")
    
    def eval(self, y : torch.Tensor, index : int = None):
        raise NotImplementedError("eval is not implemented for this mutual basis")
    
    def dim(self) -> int:
        return self._dim

class MutualPairMemberBasis(Basis):
    def __init__(self, owner : MutualPairBasis, index : int):
        super().__init__(owner.dim())
        self.owner = owner

        assert index in [0, 1], "index must be 0 or 1"
        self.index = index
    
    def _is_mate(self, other : MutualPairMemberBasis | list[MutualPairMemberBasis]) -> bool:
        if isinstance(other, MutualPairMemberBasis):
            return self.owner is other.owner and self.index != other.index
        else:
            same_owner = all(self.owner is basis.owner for basis in other)
            unique_indices = len(set(basis.index for basis in other)) == len(other)
            return same_owner and unique_indices

    def Omega1(self, lows : torch.Tensor = None, highs : torch.Tensor = None):
        return self.owner.Omega1(lows, highs)
    
    def Omega2(self, other : MutualPairMemberBasis, lows : torch.Tensor = None, highs : torch.Tensor = None):
        if not self._is_mate(other):
            raise ValueError("Omega2 is not defined for non-mate bases")
        Omega2 = self.owner.Omega2(other, lows, highs)
        if self.index == 0:
            return Omega2
        elif self.index == 1:
            return torch.transpose(Omega2, -2, -1)
    
    def __call__(self, y : torch.Tensor):
        return self.owner.eval(y, self.index)


class Orthogonal1DPWCBasis(MutualPairBasis):
    def __init__(self, value_params_1 : Parameters, value_params_2 : Parameters, cell_width_params : PositiveParameters, r1pd_factorization : Rank1PlusDiagonalFactorization, zero_mean: bool = True):
        value_params_tensor_1 = value_params_1()
        
        batch_size = value_params_tensor_1.shape[0]
        n_basis = value_params_tensor_1.shape[1]

        assert value_params_2().shape == (batch_size, n_basis), "value_params_2 must have shape (batch_size, n_basis)"
        assert cell_width_params().shape == (batch_size, n_basis), "cell_width_params must have shape (batch_size, n_basis)"

        super().__init__(1, 1, n_basis, 
            (value_params_1, value_params_2, cell_width_params, r1pd_factorization.d, r1pd_factorization.u, r1pd_factorization.v), 
            (None, None))

        self._zero_mean = zero_mean
        self._r1pd_factorization = r1pd_factorization
    
    def get_basis(self, index : int) -> MutualPairMemberBasis:
        return MutualPairMemberBasis(self, index)
    
    def Omega1(self, index : int, lows : torch.Tensor = None, highs : torch.Tensor = None):
        pass
    
    def _get_alpha(self):
        if self._zero_mean:
            pre_proj_alpha = self._params[0]()
            cell_widths = self._params[2]()

            device = pre_proj_alpha.device
            dtype = pre_proj_alpha.dtype

            # Integral over each pre-projected piece
            sigma_alpha = cell_widths * pre_proj_alpha

            u = -sigma_alpha / sigma_alpha[:, -1].unsqueeze(-1)
            d = torch.ones_like(pre_proj_alpha)
            v = torch.zeros_like(pre_proj_alpha)
            v[..., -1] = 1.0

            A_r1pd = Rank1PlusDiagonal()
            return A_r1pd.matvec(pre_proj_alpha)
        return self._params[0]()

    def _get_beta(self):
        if self._zero_mean:
        return self._params[1]()

    def eval(self, y)

    