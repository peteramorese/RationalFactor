"""
Checks for VolumePreservingPairBasis.

Run:
  PYTHONPATH=src python test/test_vp_pair_basis.py
"""

from __future__ import annotations

import torch

from normalizing_flow.vp_flow import VolumePreservingFlow
from rational_factor.models.basis_functions import BetaBasis
from rational_factor.models.domain_transformation import MLP
from rational_factor.models.mutual_bases import VolumePreservingPairBasis
from rational_factor.models.parameters import PositiveParameters
from rational_factor.tools.analysis import mc_integral_box


SEED = 0
DIM = 2
N_BASIS = 3
CONDITIONER_DIM = 4
N_POINTS = 32
PRODUCT_TOL = 1e-5
BOUND_TOL = 1e-5
INTEGRAL_TOL = 0.12


def _beta_basis(dim: int, n_basis: int = 1) -> BetaBasis:
    shape = (1, dim, n_basis)
    return BetaBasis(
        PositiveParameters.set_init(shape, 0.0, epsilon=1.0),
        PositiveParameters.set_init(shape, 0.0, epsilon=1.0),
    )


def main() -> None:
    torch.manual_seed(SEED)

    bb_25 = BetaBasis(
        PositiveParameters(fixed_values=torch.full((1, 1, 1), 2.0)),
        PositiveParameters(fixed_values=torch.full((1, 1, 1), 5.0)),
    )
    xs = torch.linspace(0.0, 1.0, 400).unsqueeze(1)
    assert torch.allclose(bb_25.supremum_bound(), bb_25(xs).max(), rtol=1e-3, atol=1e-4)
    assert torch.allclose(
        BetaBasis(
            PositiveParameters(fixed_values=torch.ones(1, 1, 1)),
            PositiveParameters(fixed_values=torch.full((1, 1, 1), 3.0)),
        ).supremum_bound(),
        torch.tensor(3.0),
        atol=1e-5,
    )

    flow = VolumePreservingFlow(
        dim=DIM,
        conditioner_dim=CONDITIONER_DIM,
        n_steps=4,
        hidden_features=16,
        num_hidden_layers=2,
        zero_init=False,
    )
    base = _beta_basis(DIM)
    embedding = torch.nn.Embedding(N_BASIS, CONDITIONER_DIM)
    splitter = MLP(
        in_features=DIM + CONDITIONER_DIM,
        out_features=1,
        hidden_features=16,
        num_hidden_layers=2,
        zero_init_last=True,
    )
    pair = VolumePreservingPairBasis(base, splitter, embedding, flow)
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
    print(f"base supremum N:  {Nbound.reshape(-1)[0].item():.6f}")
    assert torch.allclose(pair.supremum(0), torch.ones(1, N_BASIS))
    assert torch.allclose(pair.supremum(1), torch.ones(1, N_BASIS) * Nbound)

    assert (alpha <= 1.0 + BOUND_TOL).all(), "alpha exceeded bound 1"
    assert (alpha >= (n / Nbound) - BOUND_TOL).all(), "alpha below n/N"
    assert (beta <= Nbound + BOUND_TOL).all(), "beta exceeded bound N"

    alpha_b = pair.get_basis(0)(y)
    beta_b = pair.get_basis(1)(y)
    assert torch.allclose(alpha_b, alpha)
    assert torch.allclose(beta_b, beta)

    for i in range(N_BASIS):
        def n_i(x, idx=i):
            return pair.flow_density(x)[:, idx]

        integral = mc_integral_box(n_i, (torch.zeros(DIM), torch.ones(DIM)), n_samples=4_000)
        print(f"MC ∫ n_{i}:         {integral.item():.4f}")
        assert abs(integral.item() - 1.0) < INTEGRAL_TOL

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
    alpha_raw = base._params[0]._p
    beta_raw = base._params[1]._p
    assert alpha_raw.grad is not None and alpha_raw.grad.abs().sum() > 0
    assert beta_raw.grad is not None and beta_raw.grad.abs().sum() > 0
    print(f"base alpha grad l1: {alpha_raw.grad.abs().sum().item():.3e}")

    pair_1d = VolumePreservingPairBasis(
        _beta_basis(1),
        MLP(in_features=1 + CONDITIONER_DIM, out_features=1, hidden_features=16, num_hidden_layers=2, zero_init_last=True),
        torch.nn.Embedding(N_BASIS, CONDITIONER_DIM),
        VolumePreservingFlow(dim=1, conditioner_dim=CONDITIONER_DIM, n_steps=8, zero_init=False),
    )
    y1 = 0.05 + 0.9 * torch.rand(N_POINTS, 1)
    n1 = pair_1d.flow_density(y1)
    n1_base = pair_1d.base(y1)
    assert torch.allclose(n1, n1_base.expand_as(n1)), "1D flow density must be p0(y) for every index"

    pair_noflo = VolumePreservingPairBasis(
        _beta_basis(DIM),
        MLP(in_features=DIM + CONDITIONER_DIM, out_features=1, hidden_features=16, num_hidden_layers=2, zero_init_last=False),
        torch.nn.Embedding(N_BASIS, CONDITIONER_DIM),
    )
    y_nf = 0.05 + 0.9 * torch.rand(N_POINTS, DIM)
    n_nf = pair_noflo.flow_density(y_nf)
    n_nf_base = pair_noflo.base(y_nf)
    assert pair_noflo.flow is None
    assert torch.allclose(n_nf, n_nf_base.expand_as(n_nf)), "shared base, flow=None: n_i = p0(y)"
    alpha_nf, beta_nf = pair_noflo.eval(y_nf, index=0), pair_noflo.eval(y_nf, index=1)
    assert (alpha_nf * beta_nf - n_nf).abs().max().item() < PRODUCT_TOL
    col_gap = (alpha_nf[:, :1] - alpha_nf[:, 1:]).abs().max().item()
    assert col_gap > 1e-6, "splitter should make alpha differ across indices"

    alpha_init = torch.zeros(1, DIM, N_BASIS)
    alpha_init[0, 0] = torch.tensor([0.0, 1.0, 2.0])
    base_unique = BetaBasis(
        PositiveParameters(trainable_init_values=alpha_init, epsilon=1.0),
        PositiveParameters.set_init((1, DIM, N_BASIS), 0.0, epsilon=1.0),
    )
    pair_unique = VolumePreservingPairBasis(
        base_unique,
        MLP(in_features=DIM + CONDITIONER_DIM, out_features=1, hidden_features=16, num_hidden_layers=2, zero_init_last=True),
        torch.nn.Embedding(N_BASIS, CONDITIONER_DIM),
    )
    n_u = pair_unique.flow_density(y_nf)
    assert torch.allclose(n_u, base_unique(y_nf))
    assert (n_u[:, 0] - n_u[:, 1]).abs().max() > 1e-4
    N_u = pair_unique.supremum(1)
    assert N_u.shape == (1, N_BASIS)
    n_u.sum().backward()
    unique_raw = base_unique._params[0]._p
    assert unique_raw.grad is not None
    assert unique_raw.grad.abs().sum() > 0
    print(f"unique base n col gap: {(n_u[:, 0] - n_u[:, 1]).abs().max().item():.3e}")

    print("ok")


if __name__ == "__main__":
    main()
