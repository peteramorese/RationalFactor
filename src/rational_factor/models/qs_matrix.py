"""Order-1 quasiseparable factorization ``P = L D U`` and a feasible PWC matrix ``F``."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as torch_F

from rational_factor.models.parameters import Parameters


def _inclusive_affine_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inclusive scan of maps ``x ↦ a x + b``. Returns ``(op_i ∘ ⋯ ∘ op_0)(0)``."""
    n = a.shape[-1]
    k = 1
    while k < n:
        a_l, b_l = a[..., :-k], b[..., :-k]
        a_r, b_r = a[..., k:], b[..., k:]
        b = torch.cat((b[..., :k], a_r * b_l + b_r), dim=-1)
        a = torch.cat((a[..., :k], a_r * a_l), dim=-1)
        k *= 2
    return b


def _affine_scan_from_zero(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``s[..., 0] = 0``, ``s[..., i+1] = a[..., i] s[..., i] + b[..., i]``.

    ``a, b`` have shape ``(..., n)`` (``n`` transitions). Result is ``(..., n+1)``.
    """
    a, b = torch.broadcast_tensors(a, b)
    zero = b.new_zeros(b.shape[:-1] + (1,))
    if a.shape[-1] == 0:
        return zero
    return torch.cat((zero, _inclusive_affine_scan(a, b)), dim=-1)


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
    if m == 1:
        y = x
    else:
        if solve:
            trans_a = a[..., :-1] - q[..., :-1] * p[..., :-1]
            trans_b = q[..., :-1] * x[..., :-1]
        else:
            trans_a = a[..., :-1]
            trans_b = q[..., :-1] * x[..., :-1]
        s = _affine_scan_from_zero(trans_a, trans_b)
        sign = -1.0 if solve else 1.0
        y = x + sign * p * s
    return y.flip(-1) if reverse else y


def _strict_lower_scale(a: torch.Tensor) -> torch.Tensor:
    """``S[..., i, j] = prod_{k=j+1}^{i-1} a[..., k]`` for ``i > j`` (empty product is 1)."""
    m = a.shape[-1]
    ones = torch.ones_like(a)
    idx_i = torch.arange(m, device=a.device).view(*([1] * (a.ndim - 1)), m, 1)
    idx_j = torch.arange(m, device=a.device).view(*([1] * (a.ndim - 1)), 1, m)
    factors = torch.where(idx_i > idx_j, a.unsqueeze(-1), ones.unsqueeze(-1))
    cprod = torch.cumprod(factors, dim=-2)
    scale = a.new_zeros(a.shape + (m,))
    scale[..., 1:, :] = cprod[..., :-1, :]
    return scale


def _qs_strict_lower(p: torch.Tensor, a: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Strict lower triangle ``M[i, j] = p[i] (prod_{k=j+1}^{i-1} a[k]) q[j]``."""
    return torch.tril(p.unsqueeze(-1) * _strict_lower_scale(a) * q.unsqueeze(-2), diagonal=-1)


def _qs_strict_upper(g: torch.Tensor, b: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Strict upper triangle ``M[i, j] = g[i] (prod_{k=i+1}^{j-1} b[k]) h[j]``."""
    return _qs_strict_lower(g.flip(-1), b.flip(-1), h.flip(-1)).flip(-1).flip(-2)


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
        """Exact per-row min/max over columns."""
        M = self.to_dense()
        return M.amin(dim=-1), M.amax(dim=-1)

    def to_dense(self) -> torch.Tensor:
        """Materialize the full matrix (``O(m^2)``, debug / row extrema)."""
        return (
            torch.diag_embed(self.d)
            + _qs_strict_lower(self.p, self.a, self.q)
            + _qs_strict_upper(self.g, self.b, self.h)
        )


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
        trans_a = self.la[..., :-1] * self.ub[..., :-1]
        trans_b = self.lq[..., :-1] * self.d[..., :-1] * self.ug[..., :-1]
        S = _affine_scan_from_zero(trans_a, trans_b)
        return Order1QSGenerators(
            p=self.lp,
            a=self.la,
            q=self.lq * self.d + self.la * self.uh * S,
            d=self.d + self.lp * self.uh * S,
            g=self.d * self.ug + self.lp * self.ub * S,
            b=self.ub,
            h=self.uh,
        )

    def inverse_transpose_generators(self) -> Order1QSGenerators:
        """Direct QS generators of ``P^{-T}`` (``O(m)``)."""
        # L^{-T} upper generators from L^{-1} lower generators (-lp, la - lq lp, lq).
        A_g, A_b, A_h = self.lq, self.la - self.lq * self.lp, -self.lp
        # U^{-T} lower generators from U^{-1} upper generators (uh, ub - uh ug, -ug).
        B_p, B_a, B_q = self.uh, self.ub - self.uh * self.ug, -self.ug
        Dinv = self.d.reciprocal()

        trans_a = (A_b * B_a).flip(-1)[..., :-1]
        trans_b = (A_h * Dinv * B_p).flip(-1)[..., :-1]
        T = _affine_scan_from_zero(trans_a, trans_b).flip(-1)
        return Order1QSGenerators(
            p=Dinv * B_p + A_g * B_a * T,
            a=B_a,
            q=B_q,
            d=Dinv + A_g * B_q * T,
            g=A_g,
            b=A_b,
            h=A_h * Dinv + A_b * B_q * T,
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
