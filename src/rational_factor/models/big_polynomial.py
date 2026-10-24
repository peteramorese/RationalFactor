import math
import torch

# -----------------------------------------------------------------------------
# Tensor-train polynomial densities on [0, 1]^d
# -----------------------------------------------------------------------------

from rational_factor.models.density_model import DensityModel, ConditionalDensityModel
from rational_factor.models.basis_functions import BernsteinBasis, ShiftedLegendreBasis
from rational_factor.models.parameters import Parameters, PositiveParameters
from rational_factor.models.tt.tt import MPO, TTVector
from rational_factor.models.tt.tt_basis import TTBasis
from rational_factor.models.tt.tt_parameters import TTVectorParameters


def _tt_log_contraction(v: TTVector, factors: torch.Tensor) -> torch.Tensor:
    """Log of a nonnegative TT contraction, scaling at every bond.

    ``factors`` is ``(n_data, dim, n_local)``; output is
    ``(n_data, *v.batch_shape)``. Never materializes the full TT tensor.
    """
    state = factors.new_ones((factors.shape[0], *v.batch_shape, 1))
    log_value = state[..., 0] * 0
    for k, core in enumerate(v.cores):
        state = torch.einsum('s...a,...aib,si->s...b', state, core, factors[:, k])
        scale = state.sum(dim=-1)
        positive = scale > 0
        safe = torch.where(positive, scale, torch.ones_like(scale))
        log_value = log_value + torch.where(positive, safe.log(), -torch.inf)
        state = state / safe.unsqueeze(-1)
    return log_value


class BigBernstein(DensityModel):
    """Normalized nonnegative TT Bernstein density on the unit box.

    Core providers must be PositiveParameters, or fixed nonnegative tensors.
    The normalization and evaluation are rank-preserving TT contractions.
    """

    def __init__(self, coeffs: TTVectorParameters):
        if not isinstance(coeffs, TTVectorParameters):
            raise TypeError('coeffs must be TTVectorParameters')
        for core in coeffs.cores:
            if not isinstance(core, PositiveParameters):
                if core.is_trainable() or bool((core() < 0).any()):
                    raise ValueError('Bernstein TT cores must be nonnegative: use PositiveParameters')
        v = coeffs()
        if len(set(v.modes)) != 1:
            raise ValueError('BernsteinBasis requires equal local mode sizes')
        super().__init__(features=len(v.modes), batch_shape=v.batch_shape or (1,))
        self.coeff_params = coeffs
        self._parameter_modules = torch.nn.ModuleList(coeffs.parameter_modules())
        self.basis = TTBasis(BernsteinBasis(len(v.modes), v.modes[0], dtype=v.dtype, device=v.device))

    def dtype_device(self):
        v = self.coeff_params()
        return v.dtype, v.device

    def log_density(self, x: torch.Tensor, **contexts: torch.Tensor):
        v = self.coeff_params()
        x = x.to(dtype=v.dtype, device=v.device)
        factors = self.basis.primitives.eval_dim(x.clamp(0, 1))
        mass = _tt_log_contraction(v, factors)
        integrals = self.basis.primitives.Omega1_dim(return_log=False)[0].to(x)
        log_z = _tt_log_contraction(v, integrals.unsqueeze(0))[0]
        if bool(torch.isneginf(log_z).any()):
            raise ValueError('Bernstein coefficient tensor has zero integral')
        out = mass - log_z
        if not v.batch_shape:
            out = out.unsqueeze(-1)
        inside = ((x >= 0) & (x <= 1)).all(dim=-1)
        return torch.where(inside.reshape(-1, *([1] * (out.ndim - 1))), out, -torch.inf)


class ConditionalBigLegendre(ConditionalDensityModel):
    """Signed Legendre kernel phi(y)^T M phi(x) on the unit box.

    ``matrix_params()`` must return an MPO satisfying e0^T M = e0^T.
    Its polynomial integrates to one in y, but can be negative until a
    separate positivity loss is imposed.
    """

    def __init__(self, matrix_params: Parameters):
        if not isinstance(matrix_params, Parameters):
            raise TypeError('matrix_params must be a Parameters object returning an MPO')
        M = matrix_params()
        if not isinstance(M, MPO) or M.row_modes != M.col_modes:
            raise ValueError('matrix_params must return a square MPO')
        if len(set(M.row_modes)) != 1:
            raise ValueError('ShiftedLegendreBasis requires equal local mode sizes')
        d, n = len(M.row_modes), M.row_modes[0]
        super().__init__(features=d, context_features=d, batch_shape=(1,))
        self.matrix_params = matrix_params
        self._parameter_modules = torch.nn.ModuleList(matrix_params.parameter_modules())
        self.basis = TTBasis(ShiftedLegendreBasis(d, n, dtype=M.dtype, device=M.device))

    def dtype_device(self):
        M = self.matrix_params()
        return M.dtype, M.device

    def forward(self, x: torch.Tensor, *, conditioner: torch.Tensor, **contexts: torch.Tensor):
        """Return the *signed* polynomial, not exp of a clipped log-density."""
        if x.ndim != 2 or conditioner.ndim != 2 or x.shape != conditioner.shape or x.shape[1] != self.features:
            raise ValueError('x and conditioner must have matching shape (n_data, features)')
        M = self.matrix_params()
        x = x.to(dtype=M.dtype, device=M.device)
        conditioner = conditioner.to(dtype=M.dtype, device=M.device)
        y_local = self.basis.primitives.eval_dim(x)
        x_local = self.basis.primitives.eval_dim(conditioner)
        state = x.new_ones(x.shape[0], 1)
        for k, core in enumerate(M.cores):
            state = torch.einsum('sa,si,aijb,sj->sb', state, y_local[:, k], core, x_local[:, k])
        result = state  # (n_data, 1)
        inside = ((x >= 0) & (x <= 1)).all(dim=-1, keepdim=True)
        return torch.where(inside, result, torch.zeros_like(result))

    def log_density(self, x: torch.Tensor, *, conditioner: torch.Tensor, **contexts: torch.Tensor):
        """Clipped log for likelihoods; call forward() for the signed polynomial."""
        raw = self.forward(x, conditioner=conditioner, **contexts)
        return raw.clamp_min(math.exp(self.min_log_density)).log()
