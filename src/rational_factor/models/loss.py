import torch
from .rational_factor import LinearRFF, LinearFF
from .basis_functions import GaussianBasis

#### Linear models ####

def lrff_mle_loss(model : LinearRFF, x : torch.Tensor, xp : torch.Tensor):
    return -model(x, xp).log().mean()

def lff_mle_loss(model : LinearFF, x : torch.Tensor):
    return -model(x).log().mean()

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

def gaussian_basis_var_reg_loss(basis : GaussianBasis, mean=True):
    mu, std = basis.means_stds()
    log_det_cov = 2.0 * torch.log(std).sum(dim=0)
    return -log_det_cov.mean() if mean else -log_det_cov.sum()
