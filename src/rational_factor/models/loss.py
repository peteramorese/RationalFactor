import torch
from rational_factor.models.tt.tt import MPO
from rational_factor.models.big_polynomial import ConditionalBigLegendre
from .density_model import DensityModel, ConditionalDensityModel
from .factor_forms import LinearRFF, QuadraticRFF
#### General ####

def conditional_mle_loss(
    model : ConditionalDensityModel,
    x : torch.Tensor,
    conditioner : torch.Tensor,
    method_name : str = "log_density",
    **contexts : torch.Tensor,
):
    loss_method = getattr(model, method_name, None)
    if loss_method is None or not callable(loss_method):
        raise AttributeError(f"{type(model).__name__} has no callable method '{method_name}'")
    return -loss_method(x, conditioner=conditioner, **contexts).mean()

def mle_loss(model : DensityModel, x : torch.Tensor):
    return -model.log_density(x).mean()


def mse_loss(
    pred: torch.Tensor | dict[str, torch.Tensor],
    target: torch.Tensor | dict[str, torch.Tensor],
    *,
    keys: tuple[str, ...] | list[str] | None = None,
) -> torch.Tensor:
    """
    Mean squared error between predictions and targets.

    Accepts either a single tensor pair or dicts of tensors with matching keys.
    For dict inputs, if ``keys`` is given, only those entries are compared (in order);
    otherwise all keys in ``pred`` must appear in ``target``. Batched ``pred`` tensors
    broadcast unbatched ``target`` along a leading batch dimension when needed.
    """
    if isinstance(pred, torch.Tensor):
        if not isinstance(target, torch.Tensor):
            raise TypeError("mse_loss expects both arguments to be tensors or both to be dicts.")
        if pred.dim() > target.dim():
            target = target.unsqueeze(0).expand_as(pred)
        return torch.nn.functional.mse_loss(pred, target)

    if not isinstance(pred, dict) or not isinstance(target, dict):
        raise TypeError("mse_loss expects both arguments to be tensors or both to be dicts.")

    key_order = tuple(keys) if keys is not None else tuple(pred.keys())
    missing = set(key_order) - set(target.keys())
    if missing:
        raise KeyError(f"target is missing keys: {sorted(missing)}")

    losses = []
    for name in key_order:
        if name not in pred:
            raise KeyError(f"pred is missing key '{name}'.")
        p = pred[name]
        t = target[name].to(device=p.device, dtype=p.dtype)
        if p.dim() > t.dim():
            t = t.unsqueeze(0).expand_as(p)
        losses.append(torch.nn.functional.mse_loss(p, t))
    return torch.stack(losses).mean()

#### Linear models ####

def lrff_bOmega_eval_loss(model : LinearRFF):
    Omega = model.phi_basis.inner_prod_matrix(model.psi_basis)
    b = model.get_b(Omega=Omega)
    bOmega = b.unsqueeze(1) * Omega
    evals_abs = torch.linalg.eigvals(bOmega).abs()
    return -torch.log(evals_abs).mean()

def lrff_bOmega_trace_loss(model : LinearRFF):
    Omega = model.phi_basis.inner_prod_matrix(model.psi_basis)
    b = model.get_b(Omega=Omega)
    bOmega = b.unsqueeze(1) * Omega

    return bOmega.shape[0] - torch.trace(bOmega)

#### Quadratic models ####

def B_psd_loss(model : QuadraticRFF, min_eigval : float = 0.0, penalty_offset : float = 2e0, eval_condition : float = 1e-8, exponent : float = 2.0):
    B = model.get_B()
    eigvals = torch.linalg.eigvalsh(B) + eval_condition * torch.eye(B.shape[0], device=B.device, dtype=B.dtype)
    if torch.all(eigvals > min_eigval):
        return -exponent * torch.log(eigvals).mean()
    else:
        penalty = torch.relu(-eigvals + penalty_offset)
        return torch.sum(penalty**exponent + penalty)


#### Regularization ####

def dtf_data_concentration_loss(dtf, x : torch.Tensor, concentration_point : torch.Tensor, radius : float):
    x_transformed, _ = dtf(x)
    assert x_transformed.dim() == 2, "dtf(x) must have shape (n_data, d)"
    assert torch.all(radius >= 0.0), "radius must be non-negative"

    concentration_point = concentration_point.to(device=x_transformed.device, dtype=x_transformed.dtype)
    if concentration_point.dim() == 0:
        concentration_point = concentration_point.expand(x_transformed.shape[1])

    assert concentration_point.dim() == 1, "concentration_point must be a scalar or 1D tensor"
    assert concentration_point.shape[0] == x_transformed.shape[1], "concentration_point must have shape (d,)"

    distance_from_center = torch.abs(x_transformed - concentration_point.unsqueeze(0))
    outside_distance = torch.relu(distance_from_center - radius)
    return outside_distance.mean()

#### Conditional Big Legendre Polynomial ####

from functools import lru_cache
from rational_factor.models.basis_functions import legendre_to_bernstein


@lru_cache(maxsize=32)
def _legendre_to_bernstein_tensor(degree: int, dtype: torch.dtype, device: torch.device):
    return legendre_to_bernstein(degree, dtype=dtype, device=device).to_dense()


def cbl_bernstein_mpo(model: ConditionalBigLegendre) -> MPO:
    """MPO of Bernstein coefficients A=C M C.T, with unchanged bond ranks.

    A[i, j] multiplies b_i(y) b_j(x). Its nonnegative entries suffice for
    nonnegativity of the conditional polynomial on the unit box.
    """
    if not isinstance(model, ConditionalBigLegendre):
        raise TypeError('model must be ConditionalBigLegendre')
    M = model.matrix_params()
    cores = []
    for G in M.cores:
        C = _legendre_to_bernstein_tensor(G.shape[1] - 1, M.dtype, M.device)
        cores.append(torch.einsum('pi,aijb,qj->apqb', C, G, C))
    return MPO(cores)


def cbl_positivity_loss(
    model: ConditionalBigLegendre, *, method: str = 'cores',
    n_samples: int = 1024, row_indices: torch.Tensor | None = None,
    col_indices: torch.Tensor | None = None, reduction: str = 'mean',
) -> torch.Tensor:
    """Squared-negative Bernstein penalty for a ConditionalBigLegendre model.

    ``method='cores'`` penalizes negative entries of the transformed MPO cores.
    Zero loss *certifies* nonnegative Bernstein coefficients, but is conservative
    and MPO-gauge dependent.

    ``method='coefficients'`` penalizes actual Bernstein coefficients sampled
    at pairs of tensor-product indices. It is gauge independent, but zero
    sampled loss does not certify the remaining coefficients. Specify matching
    ``row_indices``/``col_indices`` of shape (n_samples, dim) for deterministic
    tests or targeted coefficient checks; otherwise indices are uniform random.

    Neither criterion is necessary for polynomial nonnegativity. ``reduction``
    is 'mean' or 'sum'. Gradients propagate to all matrix parameters.
    """
    if method not in ('cores', 'coefficients'):
        raise ValueError("method must be 'cores' or 'coefficients'")
    if reduction not in ('mean', 'sum'):
        raise ValueError("reduction must be 'mean' or 'sum'")
    A = cbl_bernstein_mpo(model)
    if method == 'cores':
        if row_indices is not None or col_indices is not None:
            raise ValueError('indices are only supported for method=\'coefficients\'')
        penalties = [torch.relu(-G).square() for G in A.cores]
        terms = [p.mean() if reduction == 'mean' else p.sum() for p in penalties]
        return torch.stack(terms).mean() if reduction == 'mean' else torch.stack(terms).sum()

    if (row_indices is None) != (col_indices is None):
        raise ValueError('row_indices and col_indices must be supplied together')
    if row_indices is None:
        if not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError('n_samples must be a positive integer')
        row_indices = torch.stack([
            torch.randint(n, (n_samples,), device=A.device) for n in A.row_modes
        ], dim=1)
        col_indices = torch.stack([
            torch.randint(n, (n_samples,), device=A.device) for n in A.col_modes
        ], dim=1)
    else:
        row_indices = torch.as_tensor(row_indices, device=A.device)
        col_indices = torch.as_tensor(col_indices, device=A.device)
        if (row_indices.ndim != 2 or row_indices.shape != col_indices.shape
                or row_indices.shape[1] != len(A.cores) or row_indices.shape[0] == 0
                or row_indices.dtype not in (torch.int32, torch.int64)
                or col_indices.dtype not in (torch.int32, torch.int64)):
            raise ValueError('indices must be matching nonempty integer tensors (samples, dim)')
        for k, (m, n) in enumerate(zip(A.row_modes, A.col_modes)):
            if ((row_indices[:, k] < 0) | (row_indices[:, k] >= m)
                    | (col_indices[:, k] < 0) | (col_indices[:, k] >= n)).any():
                raise ValueError(f'coefficient indices out of range at site {k}')

    state = A.cores[0].new_ones(row_indices.shape[0], 1)
    for k, G in enumerate(A.cores):
        selected = G[:, row_indices[:, k].long(), col_indices[:, k].long(), :]
        state = torch.einsum('sa,asb->sb', state, selected)
    penalty = torch.relu(-state[:, 0]).square()
    return penalty.mean() if reduction == 'mean' else penalty.sum()