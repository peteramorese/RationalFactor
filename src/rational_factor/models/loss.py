import torch
from .density_model import DensityModel, ConditionalDensityModel
from .factor_forms import LinearRFF, QuadraticRFF
from .basis_functions import UnnormalizedBetaBasis

#### General ####

def conditional_mle_loss(model : ConditionalDensityModel, x : torch.Tensor, conditioner : torch.Tensor):
    return -model.log_density(x, conditioner=conditioner).mean()

def mle_loss(model : DensityModel, x : torch.Tensor):
    return -model.log_density(x).mean()

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
    #print("B: ", B)
    eigvals = torch.linalg.eigvalsh(B) + eval_condition * torch.eye(B.shape[0], device=B.device, dtype=B.dtype)
    if torch.all(eigvals > min_eigval):
        #print("torch.min(eigvals): ", torch.min(eigvals))
        #print("torch.max(eigvals): ", torch.max(eigvals))
        #print("torch.prod(eigvals): ", torch.prod(eigvals))
        #print("eigvals: ", eigvals, " loss: ", torch.sum(torch.log(eigvals + eval_condition)))

        #print("eigvals: ", eigvals, " loss: ", torch.min(eigvals).log())
        return -exponent * torch.log(eigvals).mean()
        #return torch.min(eigvals).log()
    else:
        #print("invalid evals", eigvals.min(), eigvals.max())
        penalty = torch.relu(-eigvals + penalty_offset)
        return torch.sum(penalty**exponent + penalty)


#### Regularization ####

#def gaussian_basis_var_reg_loss(basis : GaussianBasis, mean=True):
#    mu, std = basis.means_stds()
#    log_det_cov = 2.0 * torch.log(std).sum(dim=0)
#    return -log_det_cov.mean() if mean else -log_det_cov.sum()

def beta_basis_concentration_reg_loss(basis : UnnormalizedBetaBasis):
    alpha, beta = basis.alphas_betas()
    return torch.square(alpha).mean() + torch.square(beta).mean()

def dtf_data_concentration_loss(dtf, x : torch.Tensor, concentration_point : torch.Tensor, radius : float):
    x_transformed = dtf(x)
    assert x_transformed.dim() == 2, "dtf(x) must have shape (n_data, d)"
    assert radius >= 0.0, "radius must be non-negative"

    concentration_point = concentration_point.to(device=x_transformed.device, dtype=x_transformed.dtype)
    if concentration_point.dim() == 0:
        concentration_point = concentration_point.expand(x_transformed.shape[1])

    assert concentration_point.dim() == 1, "concentration_point must be a scalar or 1D tensor"
    assert concentration_point.shape[0] == x_transformed.shape[1], "concentration_point must have shape (d,)"

    distance_from_center = torch.abs(x_transformed - concentration_point.unsqueeze(0))
    outside_distance = torch.relu(distance_from_center - radius)
    return outside_distance.mean()

