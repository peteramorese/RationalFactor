"""Commuting row-stochastic matrix-polynomial leaves.

At physical site k, CP channel q uses P[k,q] = sum_j c[k,q,j] S[k,q]**j.
Different S[k,q] at one site are permitted only if they commute pairwise.
The local matrices are finite, dense matrices on *virtual/physical slot* indices;
no full tensor-product matrix is constructed in production contractions.
"""
from __future__ import annotations
import torch


def _fft_convolve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Differentiable convolution of polynomial coefficients, trailing dimension."""
    n = a.shape[-1] + b.shape[-1] - 1
    size = 1 << (n - 1).bit_length()
    dtype = torch.promote_types(a.dtype, b.dtype)
    work = torch.float64 if dtype == torch.float64 else torch.float32
    fa = torch.fft.rfft(a.to(work), n=size)
    fb = torch.fft.rfft(b.to(work), n=size)
    return torch.fft.irfft(fa * fb, n=size)[..., :n].to(dtype)


class StochasticPolynomialLeaf:
    """One site: ``bases`` (n,n) or (Q,n,n); ``coeffs`` (Q,L).

    Optional leading batch dimension on either argument. Matrix coefficients
    are probabilities, not logits. They are *not* renormalized after products.
    """

    def __init__(self, bases: torch.Tensor, coeffs: torch.Tensor, *, check=True):
        bases = torch.as_tensor(bases)
        coeffs = torch.as_tensor(coeffs, dtype=bases.dtype, device=bases.device)
        if coeffs.ndim == 2:
            coeffs = coeffs.unsqueeze(0)
        if coeffs.ndim != 3:
            raise ValueError('coeffs must be (Q,L) or (B,Q,L)')
        B, Q, L = coeffs.shape
        if bases.ndim == 2:
            bases = bases[None, None]
        elif bases.ndim == 3:
            # (Q,n,n) when Q>1; (B,n,n) with a single channel is ambiguous.
            bases = bases[None] if bases.shape[0] == Q else bases[:, None]
        if bases.ndim != 4 or bases.shape[-1] != bases.shape[-2]:
            raise ValueError('bases must be (n,n), (Q,n,n), (B,n,n) or (B,Q,n,n)')
        if bases.shape[0] not in (1, B) or bases.shape[1] not in (1, Q):
            raise ValueError('incompatible base/coeff batch and channel axes')
        bases = bases.expand(B, Q, -1, -1)
        if check:
            tol = 1e-6 if bases.dtype == torch.float32 else 1e-10
            if bool((bases.detach() < -tol).any()) or not torch.allclose(
                bases.detach().sum(-1), torch.ones_like(bases[..., 0].detach()), atol=tol, rtol=tol
            ):
                raise ValueError('bases must be nonnegative and row-stochastic')
            if bool((coeffs.detach() < -tol).any()) or not torch.allclose(
                coeffs.detach().sum(-1), torch.ones_like(coeffs[..., 0].detach()), atol=tol, rtol=tol
            ):
                raise ValueError('coefficient vectors must be nonnegative and sum to 1')
            for q in range(Q):
                for j in range(q):
                    a, b = bases[:, q].detach(), bases[:, j].detach()
                    if not torch.allclose(a @ b, b @ a, atol=tol * 10, rtol=tol * 10):
                        raise ValueError('bases at one slot must commute across channels')
        self.bases, self.coeffs = bases, coeffs
        self._powers = None

    @property
    def n(self):
        return self.bases.shape[-1]

    @property
    def batch_size(self):
        return self.coeffs.shape[0]

    @property
    def channels(self):
        return self.coeffs.shape[-2]

    @property
    def degree(self):
        return self.coeffs.shape[-1] - 1

    @property
    def monomials(self):
        if self._powers is None:
            I = torch.eye(self.n, dtype=self.bases.dtype, device=self.bases.device)
            p = I.expand(self.batch_size, self.channels, -1, -1)
            powers = [p]
            for _ in range(self.degree):
                p = p @ self.bases
                powers.append(p)
            self._powers = torch.stack(powers, dim=-3)  # (B,Q,L,n,n)
        return self._powers

    def matrices(self):
        return torch.einsum('bql,bqlij->bqij', self.coeffs, self.monomials)

    @staticmethod
    def multiply_coeffs(a: torch.Tensor, b: torch.Tensor):
        return _fft_convolve(a, b)

    @staticmethod
    def power_coeffs(c: torch.Tensor, exponent: int):
        """Exact polynomial coefficient exponentiation via differentiable FFT."""
        if exponent < 0:
            raise ValueError('exponent must be nonnegative')
        if exponent == 0:
            return torch.ones((*c.shape[:-1], 1), dtype=c.dtype, device=c.device)
        if exponent == 1:
            return c
        n = exponent * (c.shape[-1] - 1) + 1
        size = 1 << (n - 1).bit_length()
        work = torch.float64 if c.dtype == torch.float64 else torch.float32
        spectrum = torch.fft.rfft(c.to(work), n=size)
        return torch.fft.irfft(spectrum ** exponent, n=size)[..., :n].to(c.dtype)

    def canonical_coefficients(self, counts: tuple[int, ...]) -> torch.Tensor:
        """FFT normal form ``prod_q c_q(z)**counts[q]``, if all bases match.

        Returns (B, 1 + degree * sum(counts)) coefficient vectors for the
        common stochastic generator. This is *not* valid for distinct bases,
        even if those matrices commute: they are different indeterminates.
        """
        if len(counts) != self.channels or min(counts) < 0:
            raise ValueError('nonnegative count required for every channel')
        tol=1e-6 if self.coeffs.dtype==torch.float32 else 1e-10
        if not torch.allclose(self.bases.detach(),
                              self.bases[:, :1].detach().expand_as(self.bases),
                              atol=tol,rtol=tol):
            raise ValueError('one-variable FFT normal form requires a shared base')
        n=self.degree*sum(counts)+1
        size=1 << (n-1).bit_length()
        dtype=torch.float64 if self.coeffs.dtype==torch.float64 else torch.float32
        spectrum=torch.fft.rfft(self.coeffs.to(dtype),n=size,dim=-1)
        exponents=torch.as_tensor(counts,device=self.coeffs.device).reshape(1,-1,1)
        result=torch.fft.irfft(torch.prod(spectrum**exponents,dim=1),n=size,dim=-1)[...,:n]
        # FFT roundoff can produce tiny negative coefficients at exact zeros.
        if bool((self.coeffs.detach()>=0).all()):
            result=result.clamp_min(0)
            result=result/result.sum(-1,keepdim=True)
        return result.to(self.coeffs.dtype)

    def accumulated(self, counts: tuple[int, ...], *, transpose: bool = False):
        """Product of powers of the commuting CP-channel matrices, shape (B,n,n).

        Evaluates powers of the already assembled polynomial matrices with
        exponentiation by squaring, avoiding unnecessary degree*t monomial
        storage. ``multiply_coeffs``/``power_coeffs`` expose FFT operations
        where the polynomial coefficient representation is actually required.
        """
        if len(counts) != self.channels:
            raise ValueError('channel count mismatch')
        B = self.batch_size
        result = torch.eye(self.n, dtype=self.coeffs.dtype, device=self.coeffs.device).expand(B, -1, -1)
        channels = self.matrices()
        for q, c in enumerate(counts):
            if c:
                result = result @ torch.linalg.matrix_power(channels[:, q], int(c))
        return result.transpose(-2, -1) if transpose else result

    def to_dense(self):
        return self.matrices()


# Descriptive alias for old imports; this is intentionally not a B-spline class.
CommutingStochasticPolynomialLeaf = StochasticPolynomialLeaf
