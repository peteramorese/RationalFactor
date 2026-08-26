"""
Checks for VolumePreservingPairBasis.

Run:
  PYTHONPATH=src python test/test_vp_pair_basis.py
"""

from __future__ import annotations

import torch

from normalizing_flow.base_distributions import SeparableBeta
from normalizing_flow.vp_flow import VolumePreservingFlow
from rational_factor.models.domain_transformation import MLP
from rational_factor.models.mutual_bases import VolumePreservingPairBasis
from rational_factor.tools.analysis import mc_integral_box


SEED = 0
DIM = 2
N_BASIS = 3
CONDITIONER_DIM = 4
N_POINTS = 32
PRODUCT_TOL = 1e-5
BOUND_TOL = 1e-5
INTEGRAL_TOL = 0.12


def main() -> None:
    torch.manual_seed(SEED)

    flow = VolumePreservingFlow(
        dim=DIM,
        conditioner_dim=CONDITIONER_DIM,
        n_steps=4,
        hidden_features=16,
        num_hidden_layers=2,
        zero_init=False,
    )
    base = SeparableBeta(dim=DIM, alpha=2.0, beta=2.0)
    embedding = torch.nn.Embedding(N_BASIS, CONDITIONER_DIM)
    splitter = MLP(
        in_features=DIM,
        out_features=N_BASIS,
        hidden_features=16,
        num_hidden_layers=2,
        zero_init_last=True,
    )
    pair = VolumePreservingPairBasis(flow, base, splitter, embedding)
    pair.eval()

    y = 0.05 + 0.9 * torch.rand(N_POINTS, DIM)
    n = pair.flow_density(y)
    alpha = pair.eval(y, index=0)
    beta = pair.eval(y, index=1)
    stacked = pair.eval(y, index=None)

    print(f"n range:          {n.min().item():.4f} .. {n.max().item():.4f}")
    print(f"alpha range:      {alpha.min().item():.4f} .. {alpha.max().item():.4f}")
    print(f"beta range:       {beta.min().item():.4f} .. {beta.max().item():.4f}")

    product_err = (alpha * beta - n).abs().max().item()
    print(f"max |alpha*beta - n|: {product_err:.3e}")
    assert product_err < PRODUCT_TOL

    assert torch.allclose(stacked[:, 0, :], alpha)
    assert torch.allclose(stacked[:, 1, :], beta)

    Nbound = base.supremum_bound()
    print(f"base supremum N:  {Nbound.item():.6f}")
    assert torch.allclose(pair.supremum(0), torch.ones(1, N_BASIS))
    assert torch.allclose(pair.supremum(1), torch.full((1, N_BASIS), Nbound.item()))

    assert (alpha <= 1.0 + BOUND_TOL).all(), "alpha exceeded bound 1"
    assert (alpha >= (n / Nbound) - BOUND_TOL).all(), "alpha below n/N"
    assert (beta <= Nbound + BOUND_TOL).all(), "beta exceeded bound N"

    alpha_b = pair.get_basis(0)(y)
    beta_b = pair.get_basis(1)(y)
    assert torch.allclose(alpha_b, alpha)
    assert torch.allclose(beta_b, beta)

    # Each n_i is a density on the unit cube.
    for i in range(N_BASIS):
        def n_i(x, idx=i):
            return pair.flow_density(x)[:, idx]

        integral = mc_integral_box(n_i, (torch.zeros(DIM), torch.ones(DIM)), n_samples=4_000)
        print(f"MC ∫ n_{i}:         {integral.item():.4f}")
        assert abs(integral.item() - 1.0) < INTEGRAL_TOL

    # Shared flow: one module, distinct index embeddings.
    assert pair.flow is flow
    assert pair.splitter is splitter
    assert pair.index_embedding is embedding
    e = pair.index_embedding.weight
    off_diag = (e.unsqueeze(0) - e.unsqueeze(1)).norm(dim=-1)
    off_diag = off_diag + torch.eye(N_BASIS)
    print(f"min embedding gap: {off_diag.min().item():.3e}")
    assert off_diag.min().item() > 0.0

    pair.train()
    y_b = y.detach()
    n_b = pair.flow_density(y_b)
    alpha_b = pair.eval(y_b, index=0)
    N_b = pair.supremum(1)
    (alpha_b.sum() + n_b.sum() + N_b.sum()).backward()
    assert base.raw_alpha.grad is not None and base.raw_alpha.grad.abs().sum() > 0
    assert base.raw_beta.grad is not None and base.raw_beta.grad.abs().sum() > 0
    print(f"base raw_alpha grad l1: {base.raw_alpha.grad.abs().sum().item():.3e}")

    print("ok")


if __name__ == "__main__":
    main()
