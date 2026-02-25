import torch

def mc_integral_box(f, domain_bounds, n_samples=1000):
    lows = torch.as_tensor(domain_bounds[0])
    highs = torch.as_tensor(domain_bounds[1])

    d = lows.numel()
    x = torch.rand(n_samples, d, device=lows.device, dtype=lows.dtype) * (highs - lows) + lows
    y = f(x).squeeze(-1)  # if needed

    vol = torch.prod(highs - lows)
    return vol * y.mean()
