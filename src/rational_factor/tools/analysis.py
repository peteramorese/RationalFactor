import torch
from ..models.loss import mle_loss
from ..models.density_model import DensityModel, ConditionalDensityModel

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

def check_pdf_valid(pdf : DensityModel | ConditionalDensityModel, domain_bounds, n_samples=1000):
    assert isinstance(pdf, DensityModel)
    print("Testing density model...")
    integral = mc_integral_box(pdf.forward, domain_bounds, n_samples)
    print(f"   Integral over x (MC): {integral}")
    #err = abs(float(integral) - 1.0)
    #if err > atol:
    #    raise AssertionError(f"PDF integrates to {float(integral):.6f}, expected ~1 (|error|={err:.6f}, atol={atol})")

def check_conditional_pdf_valid(pdf : ConditionalDensityModel, domain_bounds, conditioner_domain_bounds, n_samples=1000, n_conditioner_samples=10):
    assert isinstance(pdf, ConditionalDensityModel)
    print("Testing conditional density model...")
    device = next(pdf.parameters()).device
    domain_lows = torch.as_tensor(domain_bounds[0], device=device)
    domain_highs = torch.as_tensor(domain_bounds[1], device=device)
    domain_bounds_dev = (domain_lows, domain_highs)
    conditioner_lows = torch.as_tensor(conditioner_domain_bounds[0], device=device)
    conditioner_highs = torch.as_tensor(conditioner_domain_bounds[1], device=device)
    conditioner_samples = torch.rand(n_conditioner_samples, conditioner_lows.numel(), device=device, dtype=conditioner_lows.dtype) * (conditioner_highs - conditioner_lows) + conditioner_lows

    integrals = []
    with torch.no_grad():
        pdf.eval()
        for i in range(n_conditioner_samples):
            c = conditioner_samples[i]

            def log_density_cond(x, conditioner=c):
                y = conditioner.unsqueeze(0).expand(x.shape[0], -1)
                return pdf.log_density(x, y)

            integral = mc_integral_box(log_density_cond, domain_bounds_dev, n_samples)
            integrals.append(integral)

    stacked = torch.stack(integrals)
    print(f"   Integral over x (MC) — mean: {stacked.mean().item()}, std: {stacked.std(unbiased=False).item()}")
