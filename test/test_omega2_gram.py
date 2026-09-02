"""
Checks for DenseMatrix vs Rank1PlusDiagonal matvec (the Omega2 matrix-like API).

Run:
  PYTHONPATH=src python test/test_omega2_gram.py
"""

from __future__ import annotations

import torch

from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.factor_forms import LinearFF, LinearRFF, SumProdRFF
from rational_factor.models.structured_matrices import DenseMatrix, Rank1PlusDiagonal, as_matrix
from rational_factor.models.parameters import FixedParameters, PositiveParameters, TrainableParameters, DenseMatrixFactorization, Rank1PlusDiagonalFactorization
import rational_factor.tools.propagate as propagate


SEED = 0
N = 5
BATCH = 2


def _dense_and_structured() -> tuple[torch.Tensor, Rank1PlusDiagonal]:
    torch.manual_seed(SEED)
    d = 0.5 + torch.rand(BATCH, N)
    u = torch.randn(BATCH, N)
    v = torch.randn(BATCH, N)
    structured = Rank1PlusDiagonal(u, v, d)
    return structured.to_dense(), structured


def main() -> None:
    dense, structured = _dense_and_structured()
    G_dense = DenseMatrix(dense)
    G_struct = structured
    assert isinstance(G_dense, DenseMatrix)
    assert isinstance(G_struct, Rank1PlusDiagonal)
    assert torch.allclose(G_struct.to_dense(), dense)

    x = torch.randn(BATCH, N)
    y_e = torch.einsum("...ij,...j->...i", dense, x)
    y_t = torch.einsum("...ij,...i->...j", dense, x)
    assert torch.allclose(G_dense.matvec(x), y_e)
    assert torch.allclose(G_struct.matvec(x), y_e)
    assert torch.allclose(G_dense.rev_matvec(x), y_t)
    assert torch.allclose(G_struct.rev_matvec(x), y_t)
    assert torch.allclose(G_dense @ x, y_e)
    assert torch.allclose(G_dense.T.matvec(x), y_t)

    rhs = torch.randn(BATCH, N, 3)
    assert torch.allclose(G_dense.matvec(rhs), torch.einsum("...ij,...jk->...ik", dense, rhs))
    assert torch.allclose(G_struct.matvec(rhs), torch.einsum("...ij,...jk->...ik", dense, rhs))

    a = torch.randn(BATCH, N)
    left = G_dense.mul_diag_left(a)
    assert torch.allclose(left.to_dense(), a.unsqueeze(-1) * dense)
    assert torch.allclose(
        G_struct.mul_diag_left(a).to_dense(),
        a.unsqueeze(-1) * dense,
    )

    assert torch.allclose(G_dense.sum(), dense.sum())
    assert torch.allclose(G_struct.sum(), dense.sum())

    wrapped = as_matrix(G_struct)
    assert wrapped is G_struct
    assert torch.allclose(wrapped.matvec(x), y_e)

    # LinearRFF.get_b / one-step propagate use the same matvec API.
    torch.manual_seed(SEED)
    n_basis, dim = 4, 2
    g = GaussianBasis(
        FixedParameters(torch.randn(1, dim, n_basis)),
        PositiveParameters.set_init((1, dim, n_basis), 0.8),
        coeffs=FixedParameters(0.4 + torch.rand(1, n_basis)),
    )
    psi = GaussianBasis(
        FixedParameters(torch.randn(1, dim, n_basis)),
        PositiveParameters.set_init((1, dim, n_basis), 0.8),
        coeffs=FixedParameters(0.4 + torch.rand(1, n_basis)),
    )
    h0 = GaussianBasis(
        FixedParameters(torch.randn(1, dim, n_basis)),
        PositiveParameters.set_init((1, dim, n_basis), 0.8),
        coeffs=FixedParameters(0.4 + torch.rand(1, n_basis)),
    )
    rff = LinearRFF(g, psi, register_modules=False)
    Omega = g.Omega2(psi)
    assert isinstance(Omega, DenseMatrix)
    a_coeff = g.coeffs()
    b = rff.get_b(a=a_coeff, Omega2=Omega)
    b_einsum = a_coeff / (torch.einsum("...ij,...i->...j", Omega.to_dense(), a_coeff) + rff.numerical_tolerance)
    assert torch.allclose(b, b_einsum)

    init = LinearFF(g, h0, numerical_tolerance=rff.numerical_tolerance, register_modules=False)
    seq = propagate.propagate(init, rff, n_steps=2)
    assert len(seq) == 3
    for belief in seq:
        assert torch.isfinite(belief.h.coeffs()).all()

    # Structured gram through the LinearRFF b-update formula.
    b_struct = a / (G_struct.rev_matvec(a) + 1e-20)
    b_from_dense = a / (torch.einsum("...ij,...i->...j", dense, a) + 1e-20)
    assert torch.allclose(b_struct, b_from_dense)
    c = torch.randn(BATCH, N)
    c_next = b_struct * G_struct.matvec(c)
    M = torch.einsum("...ji,...j->...ij", dense, b_struct)
    c_old = torch.einsum("...ij,...i->...j", M, c)
    assert torch.allclose(c_next, c_old)

    # SumProdRFF B/P are matrix factorizations.
    n = n_basis
    raw = torch.eye(n).unsqueeze(0)
    B = DenseMatrixFactorization(PositiveParameters.from_values(raw, trainable=True))
    P = Rank1PlusDiagonalFactorization(
        TrainableParameters.random_init((1, n)),
        TrainableParameters.random_init((1, n)),
        normalization_dim=1,
    )
    sp = SumProdRFF(g, psi, B, P, register_modules=False)
    Gamma = sp.get_Gamma()
    assert isinstance(Gamma, Rank1PlusDiagonal)
    assert Gamma.shape == (1, n, n)
    seq_sp = propagate.propagate(init, sp, n_steps=2)
    assert len(seq_sp) == 3
    logp = sp.log_density(torch.randn(4, dim), conditioner=torch.randn(4, dim))
    assert torch.isfinite(logp).all()

    print("ok")


if __name__ == "__main__":
    main()
