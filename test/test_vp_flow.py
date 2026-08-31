"""
Checks for the volume-preserving CNF on the unit box.

Run:
  PYTHONPATH=src python test/test_vp_flow.py
"""

from __future__ import annotations

import torch

from normalizing_flow.vp_flow import VolumePreservingFlow


SEED = 0
DIM = 3
CONDITIONER_DIM = 2
N_POINTS = 64
DIV_TOL = 1e-5
NORMAL_TOL = 1e-6
BOX_TOL = 1e-4
ROUNDTRIP_TOL = 5e-4
LOGDET_TOL = 5e-3


def _random_interior(n: int, dim: int, g: torch.Generator, margin: float = 0.05) -> torch.Tensor:
    return margin + (1.0 - 2.0 * margin) * torch.rand(n, dim, generator=g)


def _divergence(velocity, x: torch.Tensor, t: float) -> torch.Tensor:
    x = x.detach().requires_grad_(True)
    v = velocity(x, t)
    div = x.new_zeros(x.shape[0])
    for i in range(x.shape[1]):
        (grad_i,) = torch.autograd.grad(v[:, i].sum(), x, create_graph=False, retain_graph=True)
        div = div + grad_i[:, i]
    return div


def _jacobian_logabsdet_single(
    flow: VolumePreservingFlow,
    x: torch.Tensor,
    conditioner: torch.Tensor,
) -> torch.Tensor:
    assert x.shape == (1, flow.dim)
    x = x.detach().requires_grad_(True)
    z, _ = flow.forward(x, conditioner=conditioner)
    rows = []
    for i in range(flow.dim):
        (grad_i,) = torch.autograd.grad(z[0, i], x, retain_graph=True)
        rows.append(grad_i[0])
    J = torch.stack(rows, dim=0)
    sign, logabsdet = torch.linalg.slogdet(J)
    assert sign > 0, f"expected orientation-preserving map, got sign={sign.item()}"
    return logabsdet


def main() -> None:
    g = torch.Generator().manual_seed(SEED)
    flow = VolumePreservingFlow(
        dim=DIM,
        conditioner_dim=CONDITIONER_DIM,
        n_steps=24,
        hidden_features=32,
        num_hidden_layers=2,
        time_dependent=True,
        zero_init=False,
    )
    flow.eval()

    x = _random_interior(N_POINTS, DIM, g)
    conditioner = torch.randn(N_POINTS, CONDITIONER_DIM, generator=g)
    t = 0.37

    div = _divergence(lambda u, s: flow.velocity(u, s, conditioner=conditioner), x, t)
    max_div = div.abs().max().item()
    print(f"max |div v|:              {max_div:.3e}")
    assert max_div < DIV_TOL, f"velocity is not divergence-free: {max_div}"

    max_normal = 0.0
    for i in range(DIM):
        for face in (0.0, 1.0):
            xf = x.clone()
            xf[:, i] = face
            v = flow.velocity(xf, t, conditioner=conditioner)
            max_normal = max(max_normal, v[:, i].abs().max().item())
    print(f"max |v · n| on faces:     {max_normal:.3e}")
    assert max_normal < NORMAL_TOL, f"nonzero normal velocity on the boundary: {max_normal}"

    with torch.no_grad():
        z, ladj_fwd = flow.forward(x, conditioner=conditioner)
        x_rec, ladj_inv = flow.inverse(z, conditioner=conditioner)

    print(f"ladj forward (max abs):   {ladj_fwd.abs().max().item():.3e}")
    print(f"ladj inverse (max abs):   {ladj_inv.abs().max().item():.3e}")
    assert torch.equal(ladj_fwd, torch.zeros_like(ladj_fwd))
    assert torch.equal(ladj_inv, torch.zeros_like(ladj_inv))

    below = (-z).max().clamp(min=0)
    above = (z - 1.0).max().clamp(min=0)
    overshoot = torch.maximum(below, above).item()
    print(f"forward overshoot:        {overshoot:.3e}")
    assert overshoot < BOX_TOL, f"flow left the unit box: {overshoot}"

    roundtrip = (x_rec - x).abs().max().item()
    print(f"roundtrip max error:      {roundtrip:.3e}")
    assert roundtrip < ROUNDTRIP_TOL, f"inverse(forward(x)) != x: {roundtrip}"

    logdets = []
    for k in range(8):
        logdets.append(_jacobian_logabsdet_single(flow, x[k : k + 1], conditioner[k : k + 1]))
    max_logdet = torch.stack(logdets).abs().max().item()
    print(f"max |log|det J||:         {max_logdet:.3e}")
    assert max_logdet < LOGDET_TOL, f"discrete map is not volume-preserving: {max_logdet}"

    flow_1d = VolumePreservingFlow(dim=1, conditioner_dim=CONDITIONER_DIM, n_steps=8, zero_init=False)
    x1 = _random_interior(N_POINTS, 1, g)
    c1 = torch.randn(N_POINTS, CONDITIONER_DIM, generator=g)
    z1, ladj1 = flow_1d.forward(x1, conditioner=c1)
    assert torch.equal(z1, x1) and torch.equal(ladj1, torch.zeros(N_POINTS))

    print("1D flow is identity:      ok")

    ident = VolumePreservingFlow(
        dim=DIM,
        conditioner_dim=CONDITIONER_DIM,
        n_steps=4,
        zero_init=True,
    )
    with torch.no_grad():
        z_id, _ = ident.forward(x, conditioner=conditioner)
    ident_err = (z_id - x).abs().max().item()
    print(f"zero-init identity error: {ident_err:.3e}")
    assert ident_err == 0.0

    other = torch.randn(N_POINTS, CONDITIONER_DIM, generator=g)
    with torch.no_grad():
        z_other, _ = flow.forward(x, conditioner=other)
    cond_gap = (z_other - z).abs().max().item()
    print(f"conditioner sensitivity:  {cond_gap:.3e}")
    assert cond_gap > 1e-4, "flow ignored the conditioner"

    print("ok")


if __name__ == "__main__":
    main()
