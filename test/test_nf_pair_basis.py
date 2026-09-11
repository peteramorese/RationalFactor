"""Checks for vectorized NFPairBasis evaluation.

Run:
  PYTHONPATH=src python test/test_nf_pair_basis.py
"""

import torch

from normalizing_flow.normalizing_flow import ConditionalNormalizingFlow
from normalizing_flow.vp_flow import ConditionalVolumePreservingFlow
from rational_factor.models.mutual_bases import NFPairBasis


def _module_splitter(dim: int, embedding_dim: int) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(dim + embedding_dim, 12),
        torch.nn.Tanh(),
        torch.nn.Linear(12, 1),
        torch.nn.Softplus(),
    )


def test_module_splitter() -> None:
    torch.manual_seed(0)
    dim, n_basis, embedding_dim, n_data = 2, 4, 3, 11

    product = ConditionalNormalizingFlow(
        dim=dim,
        conditioner_dim=embedding_dim,
        num_layers=2,
        hidden_features=12,
    )
    embedding = torch.nn.Embedding(n_basis, embedding_dim)
    splitter = _module_splitter(dim, embedding_dim)
    pair = NFPairBasis(product, splitter, embedding)

    calls = {"product": 0, "splitter": 0}
    product_hook = product.register_forward_hook(
        lambda *_: calls.__setitem__("product", calls["product"] + 1)
    )
    splitter_hook = splitter.register_forward_hook(
        lambda *_: calls.__setitem__("splitter", calls["splitter"] + 1)
    )

    y = torch.randn(n_data, dim)
    both = pair.eval(y, index=None)
    assert both.shape == (n_data, 2, n_basis)
    assert calls == {"product": 1, "splitter": 1}

    density = pair.flow_density(y)
    assert density.shape == (n_data, n_basis)
    assert torch.allclose(both[:, 0] * both[:, 1], density)

    calls.update(product=0, splitter=0)
    alpha = pair.eval(y, index=0)
    assert alpha.shape == (n_data, n_basis)
    assert calls == {"product": 0, "splitter": 1}

    calls.update(product=0, splitter=0)
    beta = pair.eval(y, index=1)
    assert beta.shape == (n_data, n_basis)
    assert calls == {"product": 1, "splitter": 1}

    assert torch.allclose(pair.Omega2_diag(), torch.ones(1, n_basis))
    assert torch.allclose(pair.get_basis(0)(y), alpha)
    assert torch.allclose(pair.get_basis(1)(y), beta)

    try:
        pair.Omega1(0)
        raise AssertionError("Omega1 should require a density splitter")
    except NotImplementedError:
        pass

    loss = both.sum()
    loss.backward()
    assert embedding.weight.grad is not None
    assert embedding.weight.grad.abs().sum() > 0
    assert next(product.parameters()).grad is not None
    assert next(splitter.parameters()).grad is not None

    product_hook.remove()
    splitter_hook.remove()
    print("module splitter: ok")


def test_density_splitter() -> None:
    torch.manual_seed(1)
    dim, n_basis, embedding_dim, n_data = 2, 3, 4, 8

    product = ConditionalNormalizingFlow(
        dim=dim,
        conditioner_dim=embedding_dim,
        num_layers=2,
        hidden_features=12,
    )
    splitter = ConditionalVolumePreservingFlow(
        dim=dim,
        conditioner_dim=embedding_dim,
        num_layers=2,
        hidden_features=12,
    )
    embedding = torch.nn.Embedding(n_basis, embedding_dim)
    pair = NFPairBasis(product, splitter, embedding)

    y = torch.randn(n_data, dim)
    both = pair.eval(y, index=None)
    density = pair.flow_density(y)
    assert both.shape == (n_data, 2, n_basis)
    assert torch.allclose(both[:, 0] * both[:, 1], density)

    assert torch.allclose(pair.Omega1(0), torch.ones(1, n_basis))
    assert torch.allclose(pair.Omega2_diag(), torch.ones(1, n_basis))
    assert torch.allclose(
        pair.supremum(0),
        torch.ones(1, n_basis) * splitter.supremum_bound(),
    )
    print("density splitter: ok")


def main() -> None:
    test_module_splitter()
    test_density_splitter()
    print("ok")


if __name__ == "__main__":
    main()
