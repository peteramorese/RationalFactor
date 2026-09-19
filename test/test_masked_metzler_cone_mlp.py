"""Checks for MaskedMetzlerConeMLP / PairedMaskedMetzlerConeMLP."""

from __future__ import annotations

import torch

from rational_factor.models.mlp import MaskedMetzlerConeMLP, PairedMaskedMetzlerConeMLP
from rational_factor.models.structured_matrices import Banded, Diagonal


def _rand_rays(d: int, k: int, m: int, *, kind: str = "dense") -> torch.Tensor | Diagonal | Banded:
    torch.manual_seed(0)
    if kind == "diagonal":
        return Diagonal(torch.randn(d, k, m))
    if kind == "banded":
        offsets = torch.tensor([-1, 0, 1])
        return Banded(offsets, torch.randn(d, k, offsets.numel(), m))
    return torch.randn(d, k, m, m)


def test_shape_unbatched_and_batched():
    d, k, m = 3, 4, 5
    K = _rand_rays(d, k, m)
    mlp = MaskedMetzlerConeMLP(K, hidden_features=16, num_hidden_layers=1, max_coeff=None)

    out = mlp(torch.randn(d))
    assert out.shape == (d, m, m)

    out_b = mlp(torch.randn(7, d))
    assert out_b.shape == (7, d, m, m)


def test_nonnegative_combination_of_slot_rays():
    d, k, m = 2, 3, 4
    K = torch.randn(d, k, m, m)
    mlp = MaskedMetzlerConeMLP(
        K, hidden_features=8, num_hidden_layers=1, max_coeff=None, coeff_scale=1.0
    )
    x = torch.randn(d)
    c = mlp._coeffs(x)
    expected = torch.einsum("dk,dkij->dij", c, K)
    got = mlp(x).to_dense()
    assert torch.allclose(got, expected, atol=1e-6)


def test_slot1_coeffs_vary_with_x0():
    d, k, m = 2, 4, 5
    mlp = MaskedMetzlerConeMLP(
        torch.randn(d, k, m, m),
        hidden_features=32,
        num_hidden_layers=2,
        max_coeff=None,
        zero_init_last=False,
        bias_init=1.0,
        coeff_scale=2.0,
    )
    x = torch.zeros(64, d)
    x[:, 0] = torch.linspace(0.0, 1.0, 64)
    c = mlp._coeffs(x)[:, 1]  # slot 1 depends on x0
    spread = (c.max(0).values - c.min(0).values).max().item()
    assert spread > 0.05, f"expected visible x0 modulation of coeffs, got spread={spread}"


def test_triangular_input_dependence():
    d, k, m = 4, 3, 3
    mlp = MaskedMetzlerConeMLP(
        _rand_rays(d, k, m),
        hidden_features=32,
        num_hidden_layers=2,
        max_coeff=None,
        zero_init_last=False,
    )
    # Nonzero last-layer already set by zero_init_last=False

    x = torch.randn(d, requires_grad=True)
    M = mlp(x).to_dense()
    for ell in range(d):
        grad = torch.autograd.grad(M[ell].sum(), x, retain_graph=True)[0]
        blocked = grad[ell:]
        assert torch.allclose(blocked, torch.zeros_like(blocked), atol=1e-7), (
            f"slot {ell} should not depend on x[{ell}:], got grad={grad}"
        )


def test_slot_zero_is_input_independent():
    d, k, m = 3, 4, 3
    mlp = MaskedMetzlerConeMLP(
        _rand_rays(d, k, m),
        hidden_features=16,
        num_hidden_layers=1,
        max_coeff=None,
        zero_init_last=False,
    )
    torch.nn.init.xavier_uniform_(mlp.net[-1].weight)
    x0 = torch.zeros(d)
    x1 = torch.randn(d)
    assert torch.allclose(mlp(x0).to_dense()[0], mlp(x1).to_dense()[0], atol=1e-6)


def test_diagonal_and_banded_structure_preserved():
    d, k, m = 3, 2, 5
    for kind in ("diagonal", "banded"):
        K = _rand_rays(d, k, m, kind=kind)
        mlp = MaskedMetzlerConeMLP(K, hidden_features=8, num_hidden_layers=1, max_coeff=None)
        out = mlp(torch.randn(d))
        if kind == "diagonal":
            assert isinstance(out, Diagonal)
            assert out.d.shape == (d, m)
        else:
            assert isinstance(out, Banded)
            assert out.data.shape == (d, K.data.shape[-2], m)


def test_paired_returns_matching_coeff_combines():
    d, k, m = 2, 3, 4
    R = torch.randn(d, k, m, m)
    T = torch.randn(d, k, m, m)
    mlp = PairedMaskedMetzlerConeMLP(R, T, hidden_features=8, num_hidden_layers=1, max_coeff=None)
    x = torch.randn(5, d)
    c = mlp._coeffs(x)
    R_out, T_out = mlp(x)
    assert torch.allclose(R_out.to_dense(), torch.einsum("bdk,dkij->bdij", c, R), atol=1e-6)
    assert torch.allclose(T_out.to_dense(), torch.einsum("bdk,dkij->bdij", c, T), atol=1e-6)


if __name__ == "__main__":
    test_shape_unbatched_and_batched()
    test_nonnegative_combination_of_slot_rays()
    test_slot1_coeffs_vary_with_x0()
    test_triangular_input_dependence()
    test_slot_zero_is_input_independent()
    test_diagonal_and_banded_structure_preserved()
    test_paired_returns_matching_coeff_combines()
    print("ok")
