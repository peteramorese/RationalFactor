import torch
from ..models.loss import mle_loss
from ..models.density_model import DensityModel

def mc_integral_box(f, domain_bounds, n_samples=1000):
    lows = torch.as_tensor(domain_bounds[0])
    highs = torch.as_tensor(domain_bounds[1])

    d = lows.numel()
    x = torch.rand(n_samples, d, device=lows.device, dtype=lows.dtype) * (highs - lows) + lows
    y = f(x).squeeze(-1)  # if needed

    vol = torch.prod(highs - lows)
    return vol * y.mean()

def avg_log_likelihood(belief : DensityModel, test_data : torch.Tensor):
    with torch.no_grad():
        belief.eval()
        return -mle_loss(belief, test_data).mean()