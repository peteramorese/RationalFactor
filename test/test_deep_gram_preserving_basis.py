"""
Checks for DeepGramPreservingBasis.

Verifies positivity of (alpha, beta) and that a Monte-Carlo empirical Gram
matches ``Omega2`` (which equals the base Gaussian Gram).

Run:
  PYTHONPATH=src python test/test_deep_gram_preserving_basis.py
"""

from __future__ import annotations

import torch

from normalizing_flow.transforms import IdentityTransform
from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.gram_preserving_basis import DeepGramPreservingBasis
from rational_factor.models.index_embedding_model import IndexEmbeddingTransform
from rational_factor.models.mlp import MLP
from rational_factor.models.parameters import FixedParameters, PositiveParameters
from rational_factor.models.space_splitter import LatentReflectionSpaceSplitter


SEED = 0
DIM = 2
N_BASIS = 3
N_LAYERS = 2
N_MC = 120_000
MC_SCALE = 4.0
GRAM_ATOL = 0.015
POS_ATOL = 1e-7


def _random_gaussian(dim: int, n_basis: int) -> GaussianBasis:
    return GaussianBasis(
        FixedParameters(torch.randn(1, dim, n_basis) * 0.5),
        PositiveParameters.set_init((1, dim, n_basis), 0.7 + 0.6 * torch.rand(1).item()),
        coeffs=FixedParameters(0.5 + torch.rand(1, n_basis)),
    )


def _make_basis() -> DeepGramPreservingBasis:
    phi = _random_gaussian(DIM, N_BASIS)
    psi = _random_gaussian(DIM, N_BASIS)
    embedding = torch.nn.Embedding(N_LAYERS, 1)
    tf = IndexEmbeddingTransform(IdentityTransform(DIM), embedding)
    splitter = LatentReflectionSpaceSplitter(tf, reflection_axis=0)
    deformer = MLP(
        in_features=DIM,
        out_features=N_LAYERS,
        hidden_features=32,
        num_hidden_layers=2,
        zero_init_last=False,
    )
    for p in deformer.parameters():
        torch.nn.init.normal_(p, std=0.15)
    return DeepGramPreservingBasis(phi, psi, splitter, deformer)


def _importance_gram(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    y: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Estimate ``∫ α_i β_j`` with ``y ~ N(0, scale^2 I)`` importance weights."""
    n, d = y.shape
    log_q = (
        -0.5 * d * torch.log(torch.tensor(2.0 * torch.pi, dtype=y.dtype, device=y.device))
        - d * torch.log(torch.tensor(scale, dtype=y.dtype, device=y.device))
        - 0.5 * (y / scale).pow(2).sum(dim=-1)
    )
    w = torch.exp(-log_q)
    return torch.einsum("ni,nj,n->ij", alpha, beta, w) / n


def main() -> None:
    torch.manual_seed(SEED)
    basis = _make_basis()

    y_probe = MC_SCALE * torch.randn(4_000, DIM)
    alpha = basis.eval(y_probe, index=0)
    beta = basis.eval(y_probe, index=1)
    stacked = basis.eval(y_probe, index=None)

    assert alpha.shape == (y_probe.shape[0], N_BASIS)
    assert beta.shape == (y_probe.shape[0], N_BASIS)
    assert stacked.shape == (y_probe.shape[0], 2, N_BASIS)
    assert torch.allclose(stacked[:, 0], alpha)
    assert torch.allclose(stacked[:, 1], beta)

    print(f"alpha range: {alpha.min().item():.4e} .. {alpha.max().item():.4e}")
    print(f"beta range:  {beta.min().item():.4e} .. {beta.max().item():.4e}")
    assert (alpha >= -POS_ATOL).all(), "alpha has negative values"
    assert (beta >= -POS_ATOL).all(), "beta has negative values"

    y_mc = MC_SCALE * torch.randn(N_MC, DIM)
    alpha_mc = basis.eval(y_mc, index=0)
    beta_mc = basis.eval(y_mc, index=1)
    assert (alpha_mc >= -POS_ATOL).all()
    assert (beta_mc >= -POS_ATOL).all()

    G_emp = _importance_gram(alpha_mc, beta_mc, y_mc, MC_SCALE)
    G = basis.Omega2().to_dense()[0].detach()
    G_base = basis._base_phi.Omega2(basis._base_psi).to_dense()[0].detach()
    assert torch.allclose(G, G_base), "Omega2 should equal the base Gram"

    err = (G_emp - G).abs().max().item()
    print(f"Omega2:\n{G}")
    print(f"empirical Gram:\n{G_emp}")
    print(f"max |G_emp - Omega2|: {err:.4e}")
    assert err < GRAM_ATOL, f"empirical Gram mismatch: {err} >= {GRAM_ATOL}"

    # Member bases should agree with eval.
    assert torch.allclose(basis.get_basis(0)(y_probe), alpha)
    assert torch.allclose(basis.get_basis(1)(y_probe), beta)

    print("ok")


if __name__ == "__main__":
    main()
