import torch
import math
from rational_factor.models.structured_matrices import DenseMatrix


def legendre_to_bernstein(
    leg_deg: int, *, dtype: torch.dtype = torch.float64, device=None
) -> DenseMatrix:
    """Degree-preserving orthonormal shifted-Legendre -> Bernstein coefficients.

    ``bernstein_coeffs = T @ legendre_coeffs`` on [0, 1].
    """
    if not isinstance(leg_deg, int) or isinstance(leg_deg, bool) or leg_deg < 0:
        raise ValueError('leg_deg must be a nonnegative integer')
    n = leg_deg
    choose = torch.tensor(
        [[math.comb(a, b) if b <= a else 0 for b in range(n + 1)]
         for a in range(n + 1)], dtype=dtype, device=device,
    )
    j = torch.arange(n + 1, device=choose.device)[:, None]
    out = choose.new_empty(n + 1, n + 1)
    for k in range(n + 1):
        i = torch.arange(k + 1, device=choose.device)[None, :]
        signs = 1 - 2 * ((k + i) % 2)
        weights = signs * choose[k, i] / choose[n, k]
        out[:, k] = (choose[j, i] * choose[n - j, k - i] * weights).sum(dim=-1) * math.sqrt(2*k + 1)
    return DenseMatrix(out)