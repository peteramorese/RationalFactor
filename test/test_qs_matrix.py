"""
Checks that vectorized Order-1 / Order-k QS ops match the sequential recurrences.

Run:
  PYTHONPATH=src python test/test_qs_matrix.py
"""

from __future__ import annotations

import torch

from rational_factor.models.structured_matrices import (
    Order1Quasiseparable,
    Quasiseparable,
    Semiseparable,
)


def _loop_semiseparable(x, p, a, q, *, reverse=False, solve=False):
    """Reference scan. ``p,a,q`` are flat ``(..., m)`` or modes ``(..., m, k)``."""
    n = x.shape[-1]
    if p.dim() == x.dim() + 1 and p.shape[-2] == n:
        pass  # already (..., m, k), possibly with matching batch
    elif p.shape[-1] == n:
        p, a, q = p.unsqueeze(-1), a.unsqueeze(-1), q.unsqueeze(-1)
    elif p.dim() >= 2 and p.shape[-2] == n:
        pass  # (..., m, k) with fewer batch dims than x
    else:
        raise ValueError(f"cannot align generators {tuple(p.shape)} with x {tuple(x.shape)}")
    if reverse:
        x = x.flip(-1)
        p, a, q = p.flip(-2), a.flip(-2), q.flip(-2)
    m = x.shape[-1]
    state = x.new_zeros(x.shape[:-1] + (p.shape[-1],))
    sign = -1.0 if solve else 1.0
    cols = []
    for i in range(m):
        yi = x[..., i] + sign * (p[..., i, :] * state).sum(-1)
        cols.append(yi)
        if i < m - 1:
            src = yi if solve else x[..., i]
            state = a[..., i, :] * state + q[..., i, :] * src.unsqueeze(-1)
    out = torch.stack(cols, dim=-1)
    return out.flip(-1) if reverse else out


def _ss_flat(p, a, q, *, upper=False) -> Semiseparable:
    return Semiseparable(p.unsqueeze(-1), a.unsqueeze(-1), q.unsqueeze(-1), upper=upper)


def _loop_direct(P: Order1Quasiseparable):
    S = torch.zeros_like(P.d[..., 0])
    qP, dP, gP = [], [], []
    for i in range(P.n):
        lp, la, lq = P.lp[..., i], P.la[..., i], P.lq[..., i]
        d, ug, ub, uh = P.d[..., i], P.ug[..., i], P.ub[..., i], P.uh[..., i]
        dP.append(d + lp * uh * S)
        qP.append(lq * d + la * uh * S)
        gP.append(d * ug + lp * ub * S)
        S = lq * d * ug + la * ub * S
    return torch.stack(qP, -1), torch.stack(dP, -1), torch.stack(gP, -1)


def _loop_invT(P: Order1Quasiseparable):
    A_g, A_b, A_h = P.lq, P.la - P.lq * P.lp, -P.lp
    B_p, B_a, B_q = P.uh, P.ub - P.uh * P.ug, -P.ug
    Dinv = P.d.reciprocal()
    T = torch.zeros_like(P.d[..., 0])
    pQ, dQ, hQ = [], [], []
    for i in range(P.n - 1, -1, -1):
        Ag, Ab, Ah = A_g[..., i], A_b[..., i], A_h[..., i]
        Bp, Ba, Bq = B_p[..., i], B_a[..., i], B_q[..., i]
        dinv = Dinv[..., i]
        dQ.append(dinv + Ag * Bq * T)
        pQ.append(dinv * Bp + Ag * Ba * T)
        hQ.append(Ah * dinv + Ab * Bq * T)
        T = Ah * dinv * Bp + Ab * Ba * T
    pQ.reverse()
    dQ.reverse()
    hQ.reverse()
    return torch.stack(pQ, -1), torch.stack(dQ, -1), torch.stack(hQ, -1)


def _loop_to_dense(gen):
    m = gen.n
    M = torch.diag_embed(gen.d)
    if gen.p.shape == gen.d.shape:
        p, a, q = gen.p, gen.a, gen.q
        g, b, h = gen.g, gen.b, gen.h
        prod = q.clone()
        for i in range(1, m):
            M[..., i, :i] = p[..., i].unsqueeze(-1) * prod[..., :i]
            prod[..., :i] = prod[..., :i] * a[..., i].unsqueeze(-1)
        prod = h.clone()
        for i in range(m - 2, -1, -1):
            M[..., i, i + 1 :] = g[..., i].unsqueeze(-1) * prod[..., i + 1 :]
            prod[..., i + 1 :] = prod[..., i + 1 :] * b[..., i].unsqueeze(-1)
        return M
    # Order-k diagonal transitions.
    p, a, q = gen.p, gen.a, gen.q
    g, b, h = gen.g, gen.b, gen.h
    prod = q.clone()
    for i in range(1, m):
        M[..., i, :i] = (p[..., i, :].unsqueeze(-2) * prod[..., :i, :]).sum(-1)
        prod[..., :i, :] = prod[..., :i, :] * a[..., i, :].unsqueeze(-2)
    prod = h.clone()
    for i in range(m - 2, -1, -1):
        M[..., i, i + 1 :] = (g[..., i, :].unsqueeze(-2) * prod[..., i + 1 :, :]).sum(-1)
        prod[..., i + 1 :, :] = prod[..., i + 1 :, :] * b[..., i, :].unsqueeze(-2)
    return M


def _loop_row_minmax(gen):
    m = gen.n
    row_min_cols = [gen.d[..., i] for i in range(m)]
    row_max_cols = [gen.d[..., i] for i in range(m)]

    def _sweep(p, a, q, forward: bool) -> None:
        state_min = state_max = None
        idx = range(m) if forward else range(m - 1, -1, -1)
        for i in idx:
            has_off = (i > 0) if forward else (i < m - 1)
            grow = (i < m - 1) if forward else (i > 0)
            start = (i == 0) if forward else (i == m - 1)
            if has_off:
                z1, z2 = p[..., i] * state_min, p[..., i] * state_max
                row_min_cols[i] = torch.minimum(row_min_cols[i], torch.minimum(z1, z2))
                row_max_cols[i] = torch.maximum(row_max_cols[i], torch.maximum(z1, z2))
            if grow:
                if start:
                    state_min = state_max = q[..., i]
                else:
                    z1, z2 = a[..., i] * state_min, a[..., i] * state_max
                    state_min = torch.minimum(q[..., i], torch.minimum(z1, z2))
                    state_max = torch.maximum(q[..., i], torch.maximum(z1, z2))

    if gen.p.shape != gen.d.shape:
        # Order-k: fall back to dense extrema (closed-form sweep is order-1 only).
        M = _loop_to_dense(gen)
        return M.amin(-1), M.amax(-1)

    _sweep(gen.p, gen.a, gen.q, forward=True)
    _sweep(gen.g, gen.b, gen.h, forward=False)
    return torch.stack(row_min_cols, dim=-1), torch.stack(row_max_cols, dim=-1)


def _make_P(m: int, batch: tuple[int, ...] = (), zero_a: bool = False) -> Order1Quasiseparable:
    g = torch.Generator().manual_seed(0)
    shape = batch + (m,)
    def r():
        return torch.randn(*shape, generator=g)
    la = 0.99 * torch.tanh(r())
    ub = 0.99 * torch.tanh(r())
    if zero_a and m > 3:
        la = la.clone()
        la[..., 2] = 0.0
    d = torch.nn.functional.softplus(r()) + 1e-4
    return Order1Quasiseparable(r(), la, r(), d, r(), ub, r())


def _make_Pk(
    m: int,
    k: int,
    batch: tuple[int, ...] = (),
) -> Quasiseparable:
    g = torch.Generator().manual_seed(1)
    shape = batch + (m, k)
    def r():
        return torch.randn(*shape, generator=g)
    la = 0.99 * torch.tanh(r())
    ub = 0.99 * torch.tanh(r())
    d = torch.nn.functional.softplus(torch.randn(*batch, m, generator=g)) + 1e-4
    return Quasiseparable(r(), la, r(), d, r(), ub, r())


def _dense_LDU(P: Quasiseparable) -> torch.Tensor:
    L = Semiseparable(P.lp, P.la, P.lq).to_dense()
    U = Semiseparable(P.ug, P.ub, P.uh, upper=True).to_dense()
    return L @ torch.diag_embed(P.d) @ U


def main() -> None:
    for m in (1, 2, 5, 16):
        for batch in ((), (3,), (2, 3)):
            P = _make_P(m, batch)
            x = torch.randn(*(batch + (m,)))
            for upper, solve in ((False, False), (False, True), (True, False), (True, True)):
                S = _ss_flat(P.lp, P.la, P.lq, upper=upper)
                y = S.solve(x) if solve else S.matvec(x)
                y_ref = _loop_semiseparable(x, P.lp, P.la, P.lq, reverse=upper, solve=solve)
                assert torch.allclose(y, y_ref, atol=1e-5, rtol=1e-4), f"semiseparable m={m} {batch} {upper} {solve}"

            q, d, g = _loop_direct(P)
            gen = P.direct_generators()
            assert torch.allclose(gen.q, q, atol=1e-5, rtol=1e-4)
            assert torch.allclose(gen.d, d, atol=1e-5, rtol=1e-4)
            assert torch.allclose(gen.g, g, atol=1e-5, rtol=1e-4)

            pQ, dQ, hQ = _loop_invT(P)
            inv = P.inverse_transpose_generators()
            assert torch.allclose(inv.p, pQ, atol=1e-5, rtol=1e-4)
            assert torch.allclose(inv.d, dQ, atol=1e-5, rtol=1e-4)
            assert torch.allclose(inv.h, hQ, atol=1e-5, rtol=1e-4)

            dense = gen.to_dense()
            dense_ref = _loop_to_dense(gen)
            assert torch.allclose(dense, dense_ref, atol=1e-5, rtol=1e-4), f"to_dense m={m} {batch}"

            lo, hi = gen.row_minmax()
            lo_r, hi_r = _loop_row_minmax(gen)
            assert torch.allclose(lo, lo_r, atol=1e-5, rtol=1e-4)
            assert torch.allclose(hi, hi_r, atol=1e-5, rtol=1e-4)

            if m >= 2:
                Px = P.matvec(x)
                assert torch.allclose(Px, (P.to_dense() @ x.unsqueeze(-1)).squeeze(-1), atol=1e-4, rtol=1e-4)

    # Data batch that does not match the generator batch (eval path: (N, B, m) vs (B, m)).
    P = _make_P(5, (1,))
    x_nb = torch.randn(7, 1, 5)
    y = P.matvec(x_nb)
    y_ref = _loop_semiseparable(
        P.d * _loop_semiseparable(x_nb, P.ug, P.ub, P.uh, reverse=True),
        P.lp, P.la, P.lq,
    )
    assert y.shape == (7, 1, 5)
    assert torch.allclose(y, y_ref, atol=1e-5, rtol=1e-4)

    Pz = _make_P(8, (2,), zero_a=True)
    gen_z = Pz.direct_generators()
    assert torch.allclose(gen_z.to_dense(), _loop_to_dense(gen_z), atol=1e-5, rtol=1e-4)
    lo, hi = gen_z.row_minmax()
    lo_r, hi_r = _loop_row_minmax(gen_z)
    assert torch.allclose(lo, lo_r, atol=1e-5, rtol=1e-4)
    assert torch.allclose(hi, hi_r, atol=1e-5, rtol=1e-4)

    P = _make_P(6)
    x = torch.randn(6, requires_grad=True)
    P.matvec(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0

    # Order-k checks.
    g_x = torch.Generator().manual_seed(2)
    for m, k, batch in ((5, 2, ()), (8, 3, (2,)), (4, 4, (2, 3))):
        P = _make_Pk(m, k, batch)
        x = torch.randn(*(batch + (m,)), generator=g_x)
        assert P.order == k
        for upper, solve in ((False, False), (False, True), (True, False), (True, True)):
            S = Semiseparable(P.lp, P.la, P.lq, upper=upper)
            y = S.solve(x) if solve else S.matvec(x)
            y_ref = _loop_semiseparable(x, P.lp, P.la, P.lq, reverse=upper, solve=solve)
            assert torch.allclose(y, y_ref, atol=1e-5, rtol=1e-4), f"order-{k} ss {upper} {solve}"

        dense_ldu = _dense_LDU(P)
        assert torch.allclose(P.matvec(x), (dense_ldu @ x.unsqueeze(-1)).squeeze(-1), atol=1e-4, rtol=1e-4)
        assert torch.allclose(
            P.inverse_matvec(P.matvec(x)), x, atol=1e-3, rtol=1e-3
        ), f"order-{k} inverse"
        gen = P.direct_generators()
        assert gen.order == k
        assert torch.allclose(gen.to_dense(), dense_ldu, atol=1e-4, rtol=1e-4), f"order-{k} direct gens"
        assert torch.allclose(gen.to_dense(), _loop_to_dense(gen), atol=1e-4, rtol=1e-4)
        # invT generators need full (non-diagonal) transitions for k>1; matvec path covers inv.
        PinvT_x = P.invT_matvec(x)
        assert torch.allclose(
            PinvT_x,
            (torch.linalg.inv(dense_ldu).transpose(-1, -2) @ x.unsqueeze(-1)).squeeze(-1),
            atol=1e-3,
            rtol=1e-3,
        ), f"order-{k} invT matvec"

    print("ok")


if __name__ == "__main__":
    main()
