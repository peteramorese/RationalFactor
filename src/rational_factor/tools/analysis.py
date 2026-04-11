import torch
from ..models.loss import mle_loss
from ..models.density_model import DensityModel, ConditionalDensityModel

def mc_integral_box(f, domain_bounds, n_samples=1000, device=None):
    lows = torch.as_tensor(domain_bounds[0], device=device)
    highs = torch.as_tensor(domain_bounds[1], device=device)


    d = lows.numel()
    x = torch.rand(n_samples, d, device=device) * (highs - lows) + lows
    y = f(x).squeeze(-1)  # if needed

    vol = torch.prod(highs - lows)
    return vol * y.mean()

def avg_log_likelihood(belief : DensityModel, test_data : torch.Tensor):
    with torch.no_grad():
        belief.eval()
        return -mle_loss(belief, test_data).mean()

def check_pdf_valid(pdf : DensityModel | ConditionalDensityModel, domain_bounds, n_samples=1000, atol=0.2, device=None):
    assert isinstance(pdf, DensityModel)
    integral = mc_integral_box(pdf.forward, domain_bounds, n_samples, device=device)
    err = abs(float(integral) - 1.0)
    if err < atol:
        print(f"   Check PDF:  Integral over x (MC): {integral}")
    else:
        print(f"   Check PDF:  Integral over x (MC): {integral} (INVALID PDF)")

def check_conditional_pdf_valid(pdf : ConditionalDensityModel, domain_bounds, conditioner_domain_bounds, n_samples=1000, n_conditioner_samples=10, device=None):
    assert isinstance(pdf, ConditionalDensityModel)
    print("Testing conditional density model...")
    if device is None:
        device = next(pdf.parameters()).device
    domain_lows = torch.as_tensor(domain_bounds[0], device=device)
    domain_highs = torch.as_tensor(domain_bounds[1], device=device)
    domain_bounds_dev = (domain_lows, domain_highs)
    conditioner_lows = torch.as_tensor(conditioner_domain_bounds[0], device=device)
    conditioner_highs = torch.as_tensor(conditioner_domain_bounds[1], device=device)
    conditioner_samples = torch.rand(n_conditioner_samples, conditioner_lows.numel(), device=device, dtype=conditioner_lows.dtype) * (conditioner_highs - conditioner_lows) + conditioner_lows
    print("num samples: ", n_samples)
    print("num conditioner samples: ", n_conditioner_samples)
    integrals = []
    with torch.no_grad():
        pdf.eval()
        for i in range(n_conditioner_samples):
            c = conditioner_samples[i]

            def density_cond(x, conditioner=c):
                y = conditioner.unsqueeze(0).expand(x.shape[0], -1)
                return pdf.forward(x, conditioner=y)

            integral = mc_integral_box(density_cond, domain_bounds_dev, n_samples, device=device)
            integrals.append(integral)

    stacked = torch.stack(integrals)
    print(f"   Check Conditional PDF:  Integral over x (MC) — mean: {stacked.mean().item()}, std: {stacked.std(unbiased=False).item()}")
