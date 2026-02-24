import torch
from .rational_factor import LinearRFF, LinearFF

def rff_mle_loss(model : LinearRFF, x : torch.Tensor, xp : torch.Tensor):
    return -model(x, xp).log().mean()

def ff_mle_loss(model : LinearFF, x : torch.Tensor):
    return -model(x).log().mean()