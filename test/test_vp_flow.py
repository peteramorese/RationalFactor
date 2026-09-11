"""
Checks for the volume-preserving CNF on the unit box and additive VP on R^n.

Run:
  PYTHONPATH=src python test/test_vp_flow.py
"""

from __future__ import annotations

import torch

from normalizing_flow.vp_flow import (
    ConditionalUnitBoxVolumePreservingFlow,
    ConditionalVolumePreservingFlow,
    UnitBoxVolumePreservingFlow,
    VolumePreservingFlow,
)


SEED = 0
DIM = 3
CONDITIONER_DIM = 2
N_POINTS = 64
DIV_TOL = 1e-5
NORMAL_TOL = 1e-6
BOX_TOL = 1e-4
ROUNDTRIP_TOL = 5e-4
LOGDET_TOL = 5e-3
VP_LOGDET_TOL = 1e-5


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
    flow: ConditionalUnitBoxVolumePreservingFlow,
    x: torch.Tensor,
    conditioner: torch.Tensor,
) -> torch.Tensor:
    assert x.shape == (1, flow.dim)
    x = x.detach().requires_grad_(True)
    z, _ = flow.transform(x, conditioner=conditioner)
    rows = []
    for i in range(flow.dim):
        (grad_i,) = torch.autograd.grad(z[0, i], x, retain_graph=True)
        rows.append(grad_i[0])
    J = torch.stack(rows, dim=0)
    sign, logabsdet = torch.linalg.slogdet(J)
    assert sign > 0, f"expected orientation-preserving map, got sign={sign.item()}"
    return logabsdet


def test_unit_box_conditional() -> None:
    g = torch.Generator().manual_seed(SEED)
    flow = ConditionalUnitBoxVolumePreservingFlow(
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
        z, ladj_fwd = flow.transform(x, conditioner=conditioner)
        x_rec, ladj_inv = flow.inverse_transform(z, conditioner=conditioner)
        log_p = flow.log_density(x, conditioner=conditioner)

    print(f"ladj forward (max abs):   {ladj_fwd.abs().max().item():.3e}")
    print(f"ladj inverse (max abs):   {ladj_inv.abs().max().item():.3e}")
    assert torch.equal(ladj_fwd, torch.zeros_like(ladj_fwd))
    assert torch.equal(ladj_inv, torch.zeros_like(ladj_inv))
    assert log_p.shape == (N_POINTS,)

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

    flow_1d = ConditionalUnitBoxVolumePreservingFlow(
        dim=1, conditioner_dim=CONDITIONER_DIM, n_steps=8, zero_init=False
    )
    x1 = _random_interior(N_POINTS, 1, g)
    c1 = torch.randn(N_POINTS, CONDITIONER_DIM, generator=g)
    z1, ladj1 = flow_1d.transform(x1, conditioner=c1)
    assert torch.equal(z1, x1) and torch.equal(ladj1, torch.zeros(N_POINTS))

    print("1D flow is identity:      ok")

    ident = ConditionalUnitBoxVolumePreservingFlow(
        dim=DIM,
        conditioner_dim=CONDITIONER_DIM,
        n_steps=4,
        zero_init=True,
    )
    with torch.no_grad():
        z_id, _ = ident.transform(x, conditioner=conditioner)
    ident_err = (z_id - x).abs().max().item()
    print(f"zero-init identity error: {ident_err:.3e}")
    assert ident_err == 0.0

    other = torch.randn(N_POINTS, CONDITIONER_DIM, generator=g)
    with torch.no_grad():
        z_other, _ = flow.transform(x, conditioner=other)
    cond_gap = (z_other - z).abs().max().item()
    print(f"conditioner sensitivity:  {cond_gap:.3e}")
    assert cond_gap > 1e-4, "flow ignored the conditioner"

    samples = flow.sample(conditioner[:4], num_samples_per=2)
    assert samples.shape == (4, 2, DIM)
    assert torch.allclose(flow.supremum_bound(), flow.base.supremum_bound())
    print("conditional unit-box sample shape: ok")


def test_unit_box_unconditional() -> None:
    g = torch.Generator().manual_seed(SEED)
    flow = UnitBoxVolumePreservingFlow(
        dim=DIM, n_steps=8, hidden_features=32, zero_init=False
    )
    x = _random_interior(N_POINTS, DIM, g)
    with torch.no_grad():
        z, ladj = flow.transform(x)
        log_p = flow.log_density(x)
        samples = flow.sample(8)
    assert ladj.abs().max().item() == 0.0
    assert log_p.shape == (N_POINTS,)
    assert samples.shape == (8, DIM)
    assert torch.allclose(flow.supremum_bound(), flow.base.supremum_bound())
    print(f"unconditional unit-box log_p mean: {log_p.mean().item():.3e}")


def test_additive_vp() -> None:
    g = torch.Generator().manual_seed(SEED)
    flow = VolumePreservingFlow(dim=DIM, num_layers=3, hidden_features=32)
    x = torch.randn(N_POINTS, DIM, generator=g)
    with torch.no_grad():
        z, ladj = flow.transform(x)
        x_rec, ladj_inv = flow.inverse_transform(z)
        log_p = flow.log_density(x)
        samples = flow.sample(8)

    assert ladj.abs().max().item() < VP_LOGDET_TOL
    assert ladj_inv.abs().max().item() < VP_LOGDET_TOL
    assert (x_rec - x).abs().max().item() < 1e-5
    assert log_p.shape == (N_POINTS,)
    assert samples.shape == (8, DIM)
    assert torch.allclose(flow.supremum_bound(), flow.base.supremum_bound())
    print(f"additive VP log_p mean:   {log_p.mean().item():.3e}")

    cond = ConditionalVolumePreservingFlow(
        dim=DIM, conditioner_dim=CONDITIONER_DIM, num_layers=3, hidden_features=32
    )
    c = torch.randn(N_POINTS, CONDITIONER_DIM, generator=g)
    with torch.no_grad():
        z_c, ladj_c = cond.transform(x, conditioner=c)
        log_pc = cond.log_density(x, conditioner=c)
        samples_c = cond.sample(c[:4], num_samples_per=2)
    assert ladj_c.abs().max().item() < VP_LOGDET_TOL
    assert z_c.shape == x.shape
    assert log_pc.shape == (N_POINTS,)
    assert samples_c.shape == (4, 2, DIM)
    assert torch.allclose(cond.supremum_bound(), cond.base.supremum_bound())
    print("conditional additive VP:   ok")


def main() -> None:
    test_unit_box_conditional()
    test_unit_box_unconditional()
    test_additive_vp()
    print("ok")


if __name__ == "__main__":
    main()
