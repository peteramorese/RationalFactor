import torch
from .density_model import DensityModel, ConditionalDensityModel, LinearRFF, QuadraticRFF
from .basis_functions import GaussianBasis, BetaBasis

#### General ####

def conditional_mle_loss(model : ConditionalDensityModel, x : torch.Tensor, xp : torch.Tensor):
    #return -torch.log(torch.relu(model(x, xp) + 1e-15)).mean()
    #if torch.isnan(model(x, xp).log()).any():
    #    print("NaN in conditional_mle_loss")
    #    print("min model(x, xp): ", model(x, xp).min())
    #    #print("model basis functions: ", model.phi_basis.means_stds())
    #    print("B: ", model.get_B())
    #    raise ValueError("NaN in conditional_mle_loss")
    return -model.log_density(x, xp).mean()

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

def B_psd_loss(model : QuadraticRFF, min_eigval : float = 0.0, penalty_offset : float = 2e0, eval_condition : float = 1e-8):
    B = model.get_B()
    #print("B: ", B)
    eigvals = torch.linalg.eigvalsh(B) + eval_condition * torch.eye(B.shape[0])
    if torch.all(eigvals > min_eigval):
        #print("torch.min(eigvals): ", torch.min(eigvals))
        #print("torch.max(eigvals): ", torch.max(eigvals))
        #print("torch.prod(eigvals): ", torch.prod(eigvals))
        #print("eigvals: ", eigvals, " loss: ", torch.sum(torch.log(eigvals + eval_condition)))

        #print("eigvals: ", eigvals, " loss: ", torch.min(eigvals).log())
        return -torch.log(eigvals).mean()
        #return torch.min(eigvals).log()
    else:
        #print("invalid evals", eigvals.min(), eigvals.max())
        penalty = torch.relu(-eigvals + penalty_offset)
        return torch.sum(penalty**2 + penalty)


#### Regularization ####

def gaussian_basis_var_reg_loss(basis : GaussianBasis, mean=True):
    mu, std = basis.means_stds()
    log_det_cov = 2.0 * torch.log(std).sum(dim=0)
    return -log_det_cov.mean() if mean else -log_det_cov.sum()

def beta_basis_concentration_reg_loss(basis : BetaBasis):
    alpha, beta = basis.alphas_betas()
    return torch.square(alpha).mean() + torch.square(beta).mean()
