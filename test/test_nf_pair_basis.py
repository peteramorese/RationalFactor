"""Checks for NormalizedProductPairBasis (NFPairBasis) evaluation.

Run:
  PYTHONPATH=src python test/test_nf_pair_basis.py
"""

import torch

from normalizing_flow.conditional_base_distributions import ConditionalBernstein1D
from normalizing_flow.transforms import Transforms
from rational_factor.models.mlp import MLP
from rational_factor.models.mutual_bases import NFPairBasis


def test_nf_pair_basis() -> None:
    torch.manual_seed(0)
    dim, n_basis, embedding_dim, n_data = 2, 4, 3, 11

    embedding = torch.nn.Embedding(n_basis, embedding_dim)
    base_mlp = MLP(
        in_features=embedding_dim,
        out_features=9,
        hidden_features=12,
        num_hidden_layers=1,
        zero_init_last=False,
    )
    base = ConditionalBernstein1D(
        features=dim,
        context_features=embedding_dim,
        degree=8,
        mlp=base_mlp,
    )
    domain_tf = Transforms.make_transform(
        "nsf",
        features=dim,
        context_features=embedding_dim,
        num_layers=2,
        hidden_features=12,
        tails=None,
        num_bins=4,
        permutation="random",
    )
    pair = NFPairBasis(base, domain_tf, embedding)

    y = torch.rand(n_data, dim)
    both = pair.eval(y, index=None)
    assert both.shape == (n_data, 2, n_basis)

    alpha = pair.eval(y, index=0)
    beta = pair.eval(y, index=1)
    assert alpha.shape == (n_data, n_basis)
    assert beta.shape == (n_data, n_basis)
    assert torch.allclose(both[:, 0], alpha)
    assert torch.allclose(both[:, 1], beta)

    assert torch.allclose(pair.Omega2_diag(), torch.ones(1, n_basis))
    assert torch.allclose(pair.Omega1(0), torch.ones(1, n_basis))

    loss = both.sum()
    loss.backward()
    assert embedding.weight.grad is not None
    assert embedding.weight.grad.abs().sum() > 0
    assert next(domain_tf.parameters()).grad is not None
    assert next(base.parameters()).grad is not None
    print("nf pair basis: ok")


def main() -> None:
    test_nf_pair_basis()
    print("ok")


if __name__ == "__main__":
    main()
