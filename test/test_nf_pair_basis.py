"""Checks for vectorized NFPairBasis evaluation.

Run:
  PYTHONPATH=src python test/test_nf_pair_basis.py
"""

import torch

from normalizing_flow.normalizing_flow import ConditionalNormalizingFlow
from rational_factor.models.mutual_bases import NFPairBasis


def main() -> None:
    torch.manual_seed(0)
    dim, n_basis, embedding_dim, n_data = 2, 4, 3, 11

    nf = ConditionalNormalizingFlow(
        dim=dim,
        conditioner_dim=embedding_dim,
        num_layers=2,
        hidden_features=12,
    )
    embedding = torch.nn.Embedding(n_basis, embedding_dim)
    splitter = torch.nn.Sequential(
        torch.nn.Linear(dim + embedding_dim, 12),
        torch.nn.Tanh(),
        torch.nn.Linear(12, 1),
        torch.nn.Softplus(),
    )
    pair = NFPairBasis(nf, splitter, embedding)

    calls = {"nf": 0, "splitter": 0}
    nf_hook = nf.register_forward_hook(
        lambda *_: calls.__setitem__("nf", calls["nf"] + 1)
    )
    splitter_hook = splitter.register_forward_hook(
        lambda *_: calls.__setitem__("splitter", calls["splitter"] + 1)
    )

    y = torch.randn(n_data, dim)
    both = pair.eval(y, index=None)
    assert both.shape == (n_data, 2, n_basis)
    assert calls == {"nf": 1, "splitter": 1}

    density = pair.flow_density(y)
    assert density.shape == (n_data, n_basis)
    assert torch.allclose(both[:, 0] * both[:, 1], density)

    calls.update(nf=0, splitter=0)
    alpha = pair.eval(y, index=0)
    assert alpha.shape == (n_data, n_basis)
    assert calls == {"nf": 0, "splitter": 1}

    calls.update(nf=0, splitter=0)
    beta = pair.eval(y, index=1)
    assert beta.shape == (n_data, n_basis)
    assert calls == {"nf": 1, "splitter": 1}

    assert torch.allclose(pair.Omega2_diag(), torch.ones(1, n_basis))
    assert torch.allclose(pair.get_basis(0)(y), alpha)
    assert torch.allclose(pair.get_basis(1)(y), beta)

    loss = both.sum()
    loss.backward()
    assert embedding.weight.grad is not None
    assert embedding.weight.grad.abs().sum() > 0
    assert next(nf.parameters()).grad is not None
    assert next(splitter.parameters()).grad is not None

    nf_hook.remove()
    splitter_hook.remove()
    print("ok")


if __name__ == "__main__":
    main()
