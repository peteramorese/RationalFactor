import torch
from .rational_factor import LinearRFF, LinearFF
from .basis_functions import GaussianBasis

def rff_mle_loss(model : LinearRFF, x : torch.Tensor, xp : torch.Tensor):
    return -model(x, xp).log().mean()

def ff_mle_loss(model : LinearFF, x : torch.Tensor):
    return -model(x).log().mean()

def gaussian_basis_var_reg_loss(basis : GaussianBasis, exponent : float = 1.0):
    mu, std = basis.means_stds()
    return (1.0 / std**exponent).mean()