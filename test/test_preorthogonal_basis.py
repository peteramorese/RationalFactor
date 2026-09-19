"""Checks for AutoregressiveMetzlerConeMutualBasis and PairedMaskedMetzlerConeMLP."""

from __future__ import annotations

import torch

from rational_factor.models.autoregressive_basis import AutoregressiveMetzlerConeMutualBasis
from rational_factor.models.basis_functions import BSpline1DBasis
from rational_factor.models.mlp import PairedMaskedMetzlerConeMLP


def _make_pair(d: int = 3, m: int = 4, k: int = 2):
    torch.manual_seed(0)
    nom_alpha = BSpline1DBasis(n_cells=m - 1, degree=1)
    nom_beta = BSpline1DBasis(n_cells=m - 1, degree=1)
    assert nom_alpha.n_basis_functions() == m

    A0 = torch.eye(m).expand(d, m, m).contiguous() + 0.1 * torch.randn(d, m, m).abs()
    B0 = torch.eye(m).expand(d, m, m).contiguous() + 0.1 * torch.randn(d, m, m).abs()
    KR = torch.randn(d, k, m, m) * 0.05
    KT = torch.randn(d, k, m, m) * 0.05
    paired = PairedMaskedMetzlerConeMLP(
        KR, KT, hidden_features=8, num_hidden_layers=1, max_coeff=None, bias_init=-40.0
    )
    pair = AutoregressiveMetzlerConeMutualBasis(
        nom_alpha, nom_beta, A0, B0, paired
    )
    return pair, nom_alpha, nom_beta, A0, B0


def test_eval_shapes():
    d, m = 3, 4
    pair, *_ = _make_pair(d=d, m=m)
    y = torch.rand(5, d)
    alpha = pair.eval(y, index=0)
    beta = pair.eval(y, index=1)
    both = pair.eval(y, index=None)
    assert alpha.shape == (5, m)
    assert beta.shape == (5, m)
    assert both.shape == (5, 2, m)


def test_omega2_is_hadamard_of_active_gammas():
    """Omega2 must omit slot 0 while α_0=β_0=1."""
    pair, nom_alpha, nom_beta, A0, B0 = _make_pair()
    G = nom_alpha.Omega2(nom_beta).to_dense()
    if G.dim() == 3:
        G = G.squeeze(0)
    G = G.to(dtype=A0.dtype)
    gamma = torch.matmul(torch.matmul(A0, G), B0.transpose(-1, -2))
    expected = gamma[1:].prod(dim=0)
    assert torch.allclose(pair.Omega2().to_dense(), expected, atol=1e-6)
    # Including Gamma_0 would be wrong:
    assert not torch.allclose(pair.Omega2().to_dense(), gamma.prod(dim=0), atol=1e-3)


def test_shared_coeffs_on_R_and_T():
    d, k, m = 3, 4, 5
    torch.manual_seed(0)
    R = torch.randn(d, k, m, m)
    T = torch.randn(d, k, m, m)
    mlp = PairedMaskedMetzlerConeMLP(
        R, T, hidden_features=16, num_hidden_layers=1, max_coeff=None, zero_init_last=False
    )
    torch.nn.init.xavier_uniform_(mlp.net[-1].weight)
    x = torch.randn(7, d)
    c = mlp._coeffs(x)
    R_out, T_out = mlp(x)
    assert torch.allclose(
        R_out.to_dense(),
        torch.einsum("...dk,dkij->...dij", c, R),
        atol=1e-6,
    )
    assert torch.allclose(
        T_out.to_dense(),
        torch.einsum("...dk,dkij->...dij", c, T),
        atol=1e-6,
    )


def test_excludes_slot_zero_from_product():
    """With zero MLP logits, α = ⊙_{ℓ≥1} (A0_ℓ nom(x_ℓ)); A0_0 unused."""
    d, m, k = 3, 4, 2
    torch.manual_seed(1)
    nom_alpha = BSpline1DBasis(n_cells=m - 1, degree=1)
    nom_beta = BSpline1DBasis(n_cells=m - 1, degree=1)
    A0 = torch.randn(d, m, m).abs() + 0.1 * torch.eye(m)
    B0 = torch.randn(d, m, m).abs() + 0.1 * torch.eye(m)
    paired = PairedMaskedMetzlerConeMLP(
        torch.randn(d, k, m, m),
        torch.randn(d, k, m, m),
        hidden_features=4,
        num_hidden_layers=1,
        max_coeff=None,
        bias_init=-40.0,
    )
    pair = AutoregressiveMetzlerConeMutualBasis(nom_alpha, nom_beta, A0, B0, paired)

    y = torch.rand(6, d)
    alpha = pair.eval(y, index=0)
    expected = torch.ones(6, m)
    for ell in range(1, d):
        nom = nom_alpha(y[:, ell : ell + 1])
        expected = expected * torch.einsum("ij,nj->ni", A0[ell], nom)
    assert torch.allclose(alpha, expected, atol=1e-5)


def test_triangular_dependence_of_factors():
    d, m = 4, 4
    pair, *_ = _make_pair(d=d, m=m)
    torch.nn.init.xavier_uniform_(pair.paired_mc_mlp.net[-1].weight)
    torch.nn.init.zeros_(pair.paired_mc_mlp.net[-1].bias)

    y = torch.rand(d, requires_grad=True)
    R, _ = pair.paired_mc_mlp(y.unsqueeze(0))
    R = R.to_dense()[0]
    for ell in range(d):
        g = torch.autograd.grad(R[ell].sum(), y, retain_graph=True)[0]
        assert torch.allclose(g[ell:], torch.zeros_like(g[ell:]), atol=1e-7)


if __name__ == "__main__":
    test_eval_shapes()
    test_omega2_is_hadamard_of_active_gammas()
    test_shared_coeffs_on_R_and_T()
    test_excludes_slot_zero_from_product()
    test_triangular_dependence_of_factors()
    print("ok")
