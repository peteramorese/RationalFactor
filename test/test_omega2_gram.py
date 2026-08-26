"""
Checks for Omega2Gram dense vs structured matvec.

Run:
  PYTHONPATH=src python test/test_omega2_gram.py
"""

from __future__ import annotations

import torch

from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.factor_forms import LinearFF, LinearRFF
from rational_factor.models.gram import Omega2Gram
from rational_factor.models.lrpd import Rank1PlusDiagonal
from rational_factor.models.parameters import Parameters, PositiveParameters
import rational_factor.tools.propagate as propagate


SEED = 0
N = 5
BATCH = 2


def _dense_and_structured() -> tuple[torch.Tensor, Rank1PlusDiagonal]:
    torch.manual_seed(SEED)
    d = 0.5 + torch.rand(BATCH, N)
    u = torch.randn(BATCH, N)
    v = torch.randn(BATCH, N)
    structured = Rank1PlusDiagonal(d, u, v)
    return structured.to_dense(), structured


def main() -> None:
    dense, structured = _dense_and_structured()
    G_dense = Omega2Gram(dense)
    G_struct = Omega2Gram(structured)
    assert not G_dense.is_structured
    assert G_struct.is_structured
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

    wrapped = Omega2Gram(G_struct)
    assert wrapped.is_structured
    assert torch.allclose(wrapped.matvec(x), y_e)

    # LinearRFF.get_b / one-step propagate use the same matvec API.
    torch.manual_seed(SEED)
    n_basis, dim = 4, 2
    g = GaussianBasis(
        Parameters(torch.randn(1, dim, n_basis)),
        PositiveParameters.set_init((1, dim, n_basis), 0.8),
        coeffs=Parameters(0.4 + torch.rand(1, n_basis)),
    )
    psi = GaussianBasis(
        Parameters(torch.randn(1, dim, n_basis)),
        PositiveParameters.set_init((1, dim, n_basis), 0.8),
        coeffs=Parameters(0.4 + torch.rand(1, n_basis)),
    )
    h0 = GaussianBasis(
        Parameters(torch.randn(1, dim, n_basis)),
        PositiveParameters.set_init((1, dim, n_basis), 0.8),
        coeffs=Parameters(0.4 + torch.rand(1, n_basis)),
    )
    rff = LinearRFF(g, psi, register_modules=False)
    Omega = g.Omega2(psi)
    assert isinstance(Omega, Omega2Gram)
    assert not Omega.is_structured
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

    print("ok")


if __name__ == "__main__":
    main()
