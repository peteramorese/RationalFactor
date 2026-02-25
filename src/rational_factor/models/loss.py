import torch
from .rational_factor import LinearRFF, LinearFF
from .basis_functions import GaussianBasis

def rff_mle_loss(model : LinearRFF, x : torch.Tensor, xp : torch.Tensor):
    return -model(x, xp).log().mean()

def ff_mle_loss(model : LinearFF, x : torch.Tensor):
    return -model(x).log().mean()

def gaussian_basis_var_reg_loss(basis : GaussianBasis, mean=True):
    mu, std = basis.means_stds()
    log_det_cov = 2.0 * torch.log(std).sum(dim=0)
    return -log_det_cov.mean() if mean else -log_det_cov.sum()

def BOmega_eval_loss(model : LinearRFF):
    Omega = model.phi_basis.inner_prod_matrix(model.psi_basis)
    B = model.get_B(Omega=Omega)
    BOmega = B.unsqueeze(1) * Omega
    evals_abs = torch.linalg.eigvals(BOmega).abs()
    return -torch.log(evals_abs).mean()

def BOmega_trace_loss(model : LinearRFF):
    Omega = model.phi_basis.inner_prod_matrix(model.psi_basis)
    B = model.get_B(Omega=Omega)
    BOmega = B.unsqueeze(1) * Omega

    return BOmega.shape[0] - torch.trace(BOmega)