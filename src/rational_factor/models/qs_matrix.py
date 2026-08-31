"""Order-1 quasiseparable factorization ``P = L D U`` and a feasible PWC matrix ``F``."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as torch_F

from rational_factor.models.parameters import Parameters


def _semiseparable(
    x: torch.Tensor,
    p: torch.Tensor,
    a: torch.Tensor,
    q: torch.Tensor,
    *,
    reverse: bool = False,
    solve: bool = False,
) -> torch.Tensor:
    """Unit semiseparable matvec (``solve=False``) or triangular solve (``solve=True``).

    Lower generators ``(p, a, q)`` run left → right. Upper generators are the
    same recurrence on flipped tensors (``reverse=True``).
    """
    if reverse:
        x, p, a, q = x.flip(-1), p.flip(-1), a.flip(-1), q.flip(-1)

    m = x.shape[-1]
    state = torch.zeros_like(x[..., 0])
    sign = -1.0 if solve else 1.0
    cols = []
    for i in range(m):
        yi = x[..., i] + sign * p[..., i] * state
        cols.append(yi)
        if i < m - 1:
            src = yi if solve else x[..., i]
            state = a[..., i] * state + q[..., i] * src
    out = torch.stack(cols, dim=-1)
    return out.flip(-1) if reverse else out


def _stack_cols(cols: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(cols, dim=-1)


@dataclass
class Order1QSGenerators:
    """Direct scalar generators of an order-1 quasiseparable matrix

        Q[i,j] =
            p[i] * prod_{k=j+1}^{i-1} a[k] * q[j],   i > j
            d[i],                                     i = j
            g[i] * prod_{k=i+1}^{j-1} b[k] * h[j],   i < j

    Every tensor has shape ``(..., m)``. Used for row extrema and debug densify.
    """

    p: torch.Tensor
    a: torch.Tensor
    q: torch.Tensor
    d: torch.Tensor
    g: torch.Tensor
    b: torch.Tensor
    h: torch.Tensor

    @property
    def n(self) -> int:
        return self.d.shape[-1]

    def row_minmax(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact per-row min/max over columns in ``O(m)`` time."""
        m = self.n
        row_min_cols = [self.d[..., i] for i in range(m)]
        row_max_cols = [self.d[..., i] for i in range(m)]

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

        _sweep(self.p, self.a, self.q, forward=True)
        _sweep(self.g, self.b, self.h, forward=False)
        return torch.stack(row_min_cols, dim=-1), torch.stack(row_max_cols, dim=-1)

    def to_dense(self) -> torch.Tensor:
        """Materialize the full matrix (``O(m^2)``, debug only)."""
        m = self.n
        M = torch.diag_embed(self.d)

        prod = self.q.clone()
        for i in range(1, m):
            M[..., i, :i] = self.p[..., i].unsqueeze(-1) * prod[..., :i]
            prod[..., :i] = prod[..., :i] * self.a[..., i].unsqueeze(-1)

        prod = self.h.clone()
        for i in range(m - 2, -1, -1):
            M[..., i, i + 1 :] = self.g[..., i].unsqueeze(-1) * prod[..., i + 1 :]
            prod[..., i + 1 :] = prod[..., i + 1 :] * self.b[..., i].unsqueeze(-1)

        return M


class Order1Quasiseparable:
    """Invertible order-1 quasiseparable ``P = L D U``.

    ``L`` / ``U`` are unit lower / upper semiseparable of order 1; ``D`` is a
    nonzero diagonal. All generators have shape ``(..., m)``.

    Matvecs ``P x``, ``P^{-1} x``, ``P^T x``, ``P^{-T} x`` are ``O(m)``.
    """

    def __init__(
        self,
        lower_p: torch.Tensor,
        lower_a: torch.Tensor,
        lower_q: torch.Tensor,
        diag: torch.Tensor,
        upper_g: torch.Tensor,
        upper_b: torch.Tensor,
        upper_h: torch.Tensor,
    ):
        tensors = (lower_p, lower_a, lower_q, diag, upper_g, upper_b, upper_h)
        if any(x.shape != diag.shape for x in tensors):
            raise ValueError("all generators must share the same shape")
        self.lp, self.la, self.lq = lower_p, lower_a, lower_q
        self.d = diag
        self.ug, self.ub, self.uh = upper_g, upper_b, upper_h

    @property
    def n(self) -> int:
        return self.d.shape[-1]

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """``P x = L D U x``."""
        x = _semiseparable(x, self.ug, self.ub, self.uh, reverse=True)
        return _semiseparable(self.d * x, self.lp, self.la, self.lq)

    def inverse_matvec(self, x: torch.Tensor) -> torch.Tensor:
        """``P^{-1} x = U^{-1} D^{-1} L^{-1} x``."""
        x = _semiseparable(x, self.lp, self.la, self.lq, solve=True)
        return _semiseparable(x / self.d, self.ug, self.ub, self.uh, reverse=True, solve=True)

    def T_matvec(self, x: torch.Tensor) -> torch.Tensor:
        """``P^T x = U^T D L^T x`` (swap lower/upper generators)."""
        x = _semiseparable(x, self.lq, self.la, self.lp, reverse=True)
        return _semiseparable(self.d * x, self.uh, self.ub, self.ug)

    def invT_matvec(self, x: torch.Tensor) -> torch.Tensor:
        """``P^{-T} x = L^{-T} D^{-1} U^{-T} x``."""
        x = _semiseparable(x, self.uh, self.ub, self.ug, solve=True)
        return _semiseparable(x / self.d, self.lq, self.la, self.lp, reverse=True, solve=True)

    def direct_generators(self) -> Order1QSGenerators:
        """Direct QS generators of ``P`` itself (``O(m)``)."""
        S = torch.zeros_like(self.d[..., 0])
        qP, dP, gP = [], [], []
        for i in range(self.n):
            lp, la, lq = self.lp[..., i], self.la[..., i], self.lq[..., i]
            d, ug, ub, uh = self.d[..., i], self.ug[..., i], self.ub[..., i], self.uh[..., i]
            dP.append(d + lp * uh * S)
            qP.append(lq * d + la * uh * S)
            gP.append(d * ug + lp * ub * S)
            S = lq * d * ug + la * ub * S
        return Order1QSGenerators(
            p=self.lp, a=self.la, q=_stack_cols(qP),
            d=_stack_cols(dP),
            g=_stack_cols(gP), b=self.ub, h=self.uh,
        )

    def inverse_transpose_generators(self) -> Order1QSGenerators:
        """Direct QS generators of ``P^{-T}`` (``O(m)``)."""
        # L^{-T} upper generators from L^{-1} lower generators (-lp, la - lq lp, lq).
        A_g, A_b, A_h = self.lq, self.la - self.lq * self.lp, -self.lp
        # U^{-T} lower generators from U^{-1} upper generators (uh, ub - uh ug, -ug).
        B_p, B_a, B_q = self.uh, self.ub - self.uh * self.ug, -self.ug
        Dinv = self.d.reciprocal()

        T = torch.zeros_like(self.d[..., 0])
        pQ, dQ, hQ = [], [], []
        for i in range(self.n - 1, -1, -1):
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
        return Order1QSGenerators(
            p=_stack_cols(pQ), a=B_a, q=B_q,
            d=_stack_cols(dQ),
            g=A_g, b=A_b, h=_stack_cols(hQ),
        )

    def to_dense(self) -> torch.Tensor:
        """Materialize ``P`` (``O(m^2)``, debug only)."""
        return self.direct_generators().to_dense()


class Order1QuasiseparableFactorization:
    """Trainable ``P = L D U``. Diagonal is softplus so ``P`` stays nonsingular.

    ``transition_bound`` (e.g. ``0.99``) maps ``la, ub`` through ``bound * tanh``
    so long products of the transition generators cannot explode.
    """

    def __init__(
        self,
        lower_p: Parameters,
        lower_a: Parameters,
        lower_q: Parameters,
        diag: Parameters,
        upper_g: Parameters,
        upper_b: Parameters,
        upper_h: Parameters,
        min_diag: float = 1e-4,
        transition_bound: float | None = None,
    ):
        self.lower_p, self.lower_a, self.lower_q = lower_p, lower_a, lower_q
        self.diag = diag
        self.upper_g, self.upper_b, self.upper_h = upper_g, upper_b, upper_h
        self.min_diag = min_diag
        self.transition_bound = transition_bound

        shape = diag().shape
        if any(p().shape != shape for p in self.parameters):
            raise ValueError("all factor Parameters must share the same shape")

    @property
    def parameters(self) -> tuple[Parameters, ...]:
        return (
            self.lower_p, self.lower_a, self.lower_q, self.diag,
            self.upper_g, self.upper_b, self.upper_h,
        )

    def __call__(self) -> Order1Quasiseparable:
        la, ub = self.lower_a(), self.upper_b()
        if self.transition_bound is not None:
            la = self.transition_bound * torch.tanh(la)
            ub = self.transition_bound * torch.tanh(ub)
        d = torch_F.softplus(self.diag()) + self.min_diag
        return Order1Quasiseparable(
            self.lower_p(), la, self.lower_q(), d,
            self.upper_g(), ub, self.upper_h(),
        )
