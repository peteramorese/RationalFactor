"""
Checks for separable Beta and Bernstein base densities on the unit cube.

Run:
  PYTHONPATH=src python test/test_base_distributions.py
"""

from __future__ import annotations

import torch

from normalizing_flow.base_distributions import SeparableBernstein, SeparableBeta
from rational_factor.tools.analysis import mc_integral_box


SEED = 0
INTEGRAL_TOL = 0.04
SUP_REL_TOL = 0.03
GRID = 400


def _unit_box(dim: int):
    return (torch.zeros(dim), torch.ones(dim))


def _grid_1d(n: int = GRID) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, n)


def main() -> None:
    g = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)

    # --- Beta: uniform product ---------------------------------------------
    uniform = SeparableBeta(dim=2)
    x = torch.rand(16, 2, generator=g)
    logp = uniform.log_density(x)
    print(f"uniform beta log-density max abs: {logp.abs().max().item():.3e}")
    assert logp.abs().max().item() < 1e-5
    assert torch.allclose(uniform.supremum_bound(), torch.tensor(1.0), atol=1e-5)

    # --- Beta: closed-form 1D supremum -------------------------------------
    beta_25 = SeparableBeta(dim=1, alpha=2.0, beta=5.0)
    analytic = beta_25.supremum_bound()
    xs = _grid_1d()
    log_pdf = torch.distributions.Beta(torch.tensor(2.0), torch.tensor(5.0)).log_prob(
        xs.clamp(1e-6, 1 - 1e-6)
    )
    grid_max = log_pdf.exp().max()
    print(f"Beta(2,5) analytic supremum_bound:    {analytic.item():.6f}")
    print(f"Beta(2,5) grid max:               {grid_max.item():.6f}")
    assert torch.allclose(analytic, grid_max, rtol=1e-3, atol=1e-4)

    unbounded = SeparableBeta(
        dim=1, alpha=0.5, beta=0.5, min_concentration=0.0
    ).supremum_bound()
    print(f"Beta(0.5,0.5) supremum_bound:     {unbounded.item()}")
    assert torch.isinf(unbounded)

    assert torch.allclose(SeparableBeta(dim=1, alpha=1.0, beta=3.0).supremum_bound(), torch.tensor(3.0), atol=1e-5)
    assert torch.allclose(SeparableBeta(dim=1, alpha=4.0, beta=1.0).supremum_bound(), torch.tensor(4.0), atol=1e-5)

    # --- Beta: product integral and joint supremum -------------------------
    product = SeparableBeta(dim=2, alpha=torch.tensor([2.0, 3.0]), beta=torch.tensor([5.0, 2.0]))
    integral = mc_integral_box(product.forward, _unit_box(2), n_samples=200_000)
    print(f"Beta product MC integral:         {integral.item():.4f}")
    assert abs(integral.item() - 1.0) < INTEGRAL_TOL

    print(f"Beta product supremum_bound:      {product.supremum_bound().item():.6f}")
    gx, gy = torch.meshgrid(_grid_1d(200), _grid_1d(200), indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    emp_max = product.forward(grid).max()
    print(f"Beta product grid max:            {emp_max.item():.6f}")
    assert emp_max <= product.supremum_bound() * (1.0 + SUP_REL_TOL)

    samples = product.sample(1024)
    assert samples.shape == (1024, 2)
    assert torch.all(samples > 0) and torch.all(samples < 1)

    a, _ = product.concentrations()
    marg = product.marginal((1,))
    assert marg.dim == 1
    assert torch.allclose(marg.concentrations()[0], a[1:])

    # --- Bernstein: uniform (zero logits) ---------------------------------
    bern_uniform = SeparableBernstein(dim=2, degree=4)
    logp_b = bern_uniform.log_density(x)
    print(f"uniform Bernstein log-density:    {logp_b.abs().max().item():.3e}")
    assert logp_b.abs().max().item() < 1e-5
    assert torch.allclose(bern_uniform.coefficients().sum(dim=-1), torch.full((2,), 5.0))
    assert torch.allclose(bern_uniform.supremum_bound(), torch.tensor(1.0), atol=1e-5)

    integral_u = mc_integral_box(bern_uniform.forward, _unit_box(2), n_samples=50_000)
    print(f"uniform Bernstein MC integral:    {integral_u.item():.4f}")
    assert abs(integral_u.item() - 1.0) < 0.02

    # --- Bernstein: endpoint spike, bound is exact ------------------------
    logits_end = torch.tensor([[8.0, -8.0, -8.0, -8.0, -8.0]])
    bern_end = SeparableBernstein(dim=1, degree=4, logits=logits_end)
    xs = _grid_1d().unsqueeze(1)
    emp = bern_end.forward(xs).max()
    bound = bern_end.supremum_bound()
    print(f"endpoint Bernstein supremum_bound:    {bound.item():.6f}")
    print(f"endpoint Bernstein grid max:      {emp.item():.6f}")
    assert abs(emp.item() / bound.item() - 1.0) < 0.02

    # --- Bernstein: interior mode, bound is valid and ≥ empirical max -----
    logits_int = torch.tensor([[-4.0, -4.0, 6.0, -4.0, -4.0], [-2.0, 3.0, 3.0, -2.0, -4.0]])
    bern = SeparableBernstein(dim=2, degree=4, logits=logits_int)
    integral_b = mc_integral_box(bern.forward, _unit_box(2), n_samples=200_000)
    print(f"Bernstein product MC integral:    {integral_b.item():.4f}")
    assert abs(integral_b.item() - 1.0) < INTEGRAL_TOL

    emp_b = bern.forward(grid).max()
    bound_b = bern.supremum_bound()
    print(f"Bernstein product supremum_bound:     {bound_b.item():.6f}")
    print(f"Bernstein product grid max:       {emp_b.item():.6f}")
    assert emp_b <= bound_b * (1.0 + 1e-5)

    samples_b = bern.sample(1024)
    assert samples_b.shape == (1024, 2)
    assert torch.all(samples_b >= 0) and torch.all(samples_b <= 1)

    product.supremum_bound().backward()
    assert product.raw_alpha.grad is not None and product.raw_alpha.grad.abs().sum() > 0
    bern.supremum_bound().backward()
    assert bern.logits.grad is not None and bern.logits.grad.abs().sum() > 0

    print("ok")


if __name__ == "__main__":
    main()
