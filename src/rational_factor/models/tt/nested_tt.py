"""Depth-three commuting-polynomial nested TT/MPO, with batched symbolic API.

Guaranteed polynomial-in-time contractions apply to a fixed number of commuting
matrix-power blocks, e.g. repeated propagation by one row-stochastic MPO,
followed by rank-one pointwise evaluation. Arbitrary alternating noncommuting
maps admit exact symbolic storage but NOT a uniform polynomial-time bound.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import reduce
from itertools import product
from collections.abc import Sequence
import math
import torch
try:
    from rational_factor.models.structured_matrices import Matrix
except ImportError:
    class Matrix:  # allows standalone unit tests without installing the repository
        pass
try:
    from .nested_tt_leaf import StochasticPolynomialLeaf
except ImportError:
    from nested_tt_leaf import StochasticPolynomialLeaf


def _batch(*sizes):
    out = 1
    for n in sizes:
        if n not in (1, out) and out != 1:
            raise ValueError(f'incompatible batch dimensions: {sizes}')
        out = max(out, n)
    return out


def _as_factor(x, n, dtype, device):
    x = torch.as_tensor(x, dtype=dtype, device=device)
    if x.ndim == 1:
        x = x[None]
    if x.ndim != 2 or x.shape[-1] != n:
        raise ValueError(f'expected (n,) or (B,n), n={n}, got {tuple(x.shape)}')
    return x


def _count_vectors(t, q):
    if q == 1:
        yield (t,)
    else:
        for a in range(t + 1):
            for rest in _count_vectors(t - a, q - 1):
                yield (a, *rest)


def _resolve_factors(other, modes):
    known = getattr(other, 'rank_one_factors', None)
    if known is not None:
        return tuple(known)
    if isinstance(other, (int, float)) or (isinstance(other, torch.Tensor) and other.ndim == 0):
        out = [torch.ones(n) for n in modes]
        out[0] = out[0] * other
        return tuple(out)
    if isinstance(other, torch.Tensor):
        if len(modes) == 1:
            return (other,)
        raise ValueError('pass one separable factor per physical mode')
    if isinstance(other, Sequence) and len(other) == len(modes):
        return tuple(other)
    raise TypeError('only rank-one/separable physical factors are supported')


def _factor_three(n):
    dims = [1, 1, 1]
    p = 2
    while p * p <= n:
        while n % p == 0:
            dims[dims.index(min(dims))] *= p
            n //= p
        p += 1
    if n > 1:
        dims[dims.index(min(dims))] *= n
    return tuple(dims)


@dataclass(frozen=True)
class VirtualIndexMPO:
    """Three-layer exact MPO of a diagonal virtual-index operator."""
    cores: tuple[torch.Tensor, ...]
    modes: tuple[int, ...]

    @classmethod
    def from_diagonal(cls, values):
        if values.ndim != 1:
            raise ValueError('one diagonal vector required')
        modes = _factor_three(values.numel())
        # Explicit Q-channel CP representation; avoids differentiating an SVD.
        q = values.numel()
        cores = []
        strides = [math.prod(modes[k+1:]) for k in range(3)]
        # CP chain: bond Q, physical diagonal, first core carries values.
        for k, n in enumerate(modes):
            left, right = (1 if k == 0 else q), (1 if k == 2 else q)
            m = torch.zeros(left, n, n, right, dtype=values.dtype, device=values.device)
            for z in range(q):
                l, r = (0 if k == 0 else z), (0 if k == 2 else z)
                digit = (z // strides[k]) % n
                m[l, digit, digit, r] = values[z] if k == 0 else 1
            cores.append(m)
        return cls(tuple(cores), modes)

    def to_dense(self):
        x = self.cores[0].squeeze(0)
        for g in self.cores[1:]:
            x = torch.einsum('...r,rijs->...ijs', x, g)
        t = x.squeeze(-1).reshape(*self.modes, *self.modes)
        # Interleaved physical pair axes become all rows then all columns.
        perm = tuple(range(0, 2*len(self.modes), 2)) + tuple(range(1, 2*len(self.modes), 2))
        return t.permute(perm).reshape(math.prod(self.modes), -1)


@dataclass(frozen=True)
class ContractionStats:
    operands: int
    pair_contractions: int
    max_intermediate_numel: int
    max_intermediate_ndim: int


@dataclass(frozen=True)
class NestedTTMatrixSpec:
    row_modes: tuple[int, ...]
    col_modes: tuple[int, ...]
    depth: int = 3
    channel_rank: int = 2
    polynomial_degree: int = 2

    def __post_init__(self):
        if self.depth != 3 or not self.row_modes or self.row_modes != self.col_modes:
            raise ValueError('only depth=3, square row/column modes supported')
        if min(self.row_modes) < 1 or self.channel_rank < 1 or self.polynomial_degree < 0:
            raise ValueError('positive dimensions/ranks and nonnegative polynomial degree required')

    @property
    def d(self):
        return len(self.row_modes)


@dataclass(frozen=True)
class NestedTTVectorSpec:
    modes: tuple[int, ...]
    depth: int = 3
    bond_rank: int = 2

    def __post_init__(self):
        if self.depth != 3 or not self.modes or min(self.modes) < 1 or self.bond_rank < 1:
            raise ValueError('invalid depth/modes/bond rank')


@dataclass(frozen=True)
class NestedMPOCore:
    """Top MPO slice, with a 3-level CP/diagonal virtual-index routing MPO."""
    parent: 'NestedTTMatrix'
    site: int

    @property
    def virtual_shape(self):
        q, d = self.parent.channels, self.parent.d
        return (1 if self.site == 0 else q, 1 if self.site == d-1 else q)

    def slice(self, row, col):
        p = self.parent.leaves[self.site].matrices()  # (B,Q,n,n)
        v = p[..., row, col]  # (B,Q)
        c = self.parent.channel_weights
        if self.parent.d == 1:
            return (v * c).sum(-1)[..., None, None]
        if self.site == 0:
            return (v * c)[:, None, :]
        if self.site == self.parent.d - 1:
            return v[..., :, None]
        return torch.diag_embed(v)

    def virtual_mpo(self, row, col, batch_index=0):
        qv = self.parent.leaves[self.site].matrices()[batch_index, :, row, col]
        if self.site == 0:
            qv = qv * self.parent.channel_weights[batch_index]
        return VirtualIndexMPO.from_diagonal(qv)


class NestedTTMatrix(Matrix):
    """Row-stochastic mixture of Kronecker products of commuting polynomials.

    M = sum_q w[q] * kron_k (sum_j c[k,q,j] S[k,q]**j)

    S[k,q] may differ with q, but must commute for each fixed site k.
    """
    def __init__(self, spec: NestedTTMatrixSpec, leaves: Sequence[StochasticPolynomialLeaf],
                 channel_logits: torch.Tensor):
        self.spec = spec
        self.leaves = tuple(leaves)
        if len(leaves) != spec.d or any(a.n != n for a, n in zip(leaves, spec.row_modes)):
            raise ValueError('one leaf per physical mode, with matching square dimension')
        logits = torch.as_tensor(channel_logits)
        if logits.ndim == 1:
            logits = logits[None]
        if logits.ndim != 2 or logits.shape[-1] != spec.channel_rank:
            raise ValueError('channel_logits must have shape (Q,) or (B,Q)')
        self.channel_logits = logits
        self._batch_size = _batch(logits.shape[0], *(a.batch_size for a in leaves))
        if any(a.channels != spec.channel_rank for a in leaves):
            raise ValueError('leaf CP channel count mismatch')

    @property
    def channels(self): return self.spec.channel_rank
    @property
    def d(self): return self.spec.d
    @property
    def row_modes(self): return self.spec.row_modes
    @property
    def col_modes(self): return self.spec.col_modes
    @property
    def modes(self): return self.row_modes
    @property
    def depth(self): return self.spec.depth
    @property
    def batch_size(self): return self._batch_size
    @property
    def dtype(self): return self.leaves[0].coeffs.dtype
    @property
    def device(self): return self.leaves[0].coeffs.device
    @property
    def shape(self):
        n = math.prod(self.row_modes)
        return torch.Size((n, n) if self.batch_size == 1 else (self.batch_size, n, n))
    @property
    def channel_weights(self): return self.channel_logits.softmax(-1).expand(self.batch_size, -1)
    @property
    def cores(self): return tuple(NestedMPOCore(self,k) for k in range(self.d))
    @property
    def leaf_count(self): return sum(a.channels for a in self.leaves)
    @property
    def ranks(self): return (1,) + (self.channels,) * (self.d-1) + (1,)
    @property
    def T(self): return _TransposeNestedMatrix(self)

    def matvec(self, x):
        if isinstance(x, NestedTTVector):
            if x.modes != self.col_modes:
                raise ValueError('matrix-vector modes differ')
            return x._append(('power', self, False, 1), self.row_modes)
        return torch.matmul(self.to_dense(), torch.as_tensor(x, dtype=self.dtype, device=self.device).unsqueeze(-1)).squeeze(-1)

    def rev_matvec(self, x):
        if isinstance(x, NestedTTVector):
            if x.modes != self.row_modes:
                raise ValueError('matrix-vector modes differ')
            return x._append(('power', self, True, 1), self.col_modes)
        return torch.matmul(self.to_dense().transpose(-1,-2), torch.as_tensor(x, dtype=self.dtype, device=self.device).unsqueeze(-1)).squeeze(-1)

    def __matmul__(self, other): return self.matvec(other)

    def to_dense(self, *, max_elements=1_000_000):
        n = math.prod(self.modes)
        if n*n*self.batch_size > max_elements:
            raise ValueError('dense matrix materialization exceeds max_elements')
        local = [a.matrices().expand(self.batch_size, -1, -1, -1) for a in self.leaves]
        pieces = []
        for b in range(self.batch_size):
            result = torch.zeros(n,n,dtype=self.dtype,device=self.device)
            for q in range(self.channels):
                block = local[0][b,q]
                for x in local[1:]:
                    block = torch.kron(block, x[b,q])
                result = result + self.channel_weights[b,q] * block
            pieces.append(result)
        out = torch.stack(pieces)
        return out[0] if self.batch_size == 1 else out

    def sum(self):
        n = math.prod(self.modes)
        return self.channel_weights.sum(-1) * n if self.batch_size > 1 else self.channel_weights.sum() * n

    def row_sums(self):
        """Return the analytic row sum 1, without expanding physical states."""
        return self.channel_weights.sum(-1)

    def __rmatmul__(self, x):
        if isinstance(x, NestedTTVector):
            return self.rev_matvec(x)
        return torch.as_tensor(x, dtype=self.dtype, device=self.device) @ self.to_dense()

    def diag(self): return self.to_dense().diagonal(dim1=-2,dim2=-1)
    def elementwise_multiply(self, factors): return _SchurNestedMatrix(self, factors)
    __mul__ = elementwise_multiply
    def __rmul__(self, factors): return self.elementwise_multiply(factors)
    def mul_diag_left(self, a): return _DenseFallbackMatrix(self.to_dense()).mul_diag_left(a)
    def mul_diag_right(self, a): return _DenseFallbackMatrix(self.to_dense()).mul_diag_right(a)
    def scale(self, x): return _DenseFallbackMatrix(self.to_dense() * x)
    def materialize_cores(self):
        out = []
        for k, (leaf, core) in enumerate(zip(self.leaves,self.cores)):
            n = leaf.n
            v = leaf.matrices().expand(self.batch_size,-1,-1,-1)
            if self.d == 1:
                a = (v * self.channel_weights[...,None,None]).sum(1)[:,None,:,: ,None]
            elif k == 0:
                a = (v * self.channel_weights[...,None,None])[:,None,:,:,:].permute(0,1,3,4,2)
            elif k == self.d - 1:
                a = v.unsqueeze(-1)
            else:
                eye = torch.eye(self.channels,dtype=self.dtype,device=self.device)
                a = torch.einsum('bqij,qr->bqijr',v,eye)
            out.append(a[0] if self.batch_size == 1 else a)
        return tuple(out)


class _TransposeNestedMatrix(Matrix):
    def __init__(self, base): self.base=base
    @property
    def shape(self): return self.base.shape
    @property
    def dtype(self): return self.base.dtype
    @property
    def device(self): return self.base.device
    @property
    def T(self): return self.base
    def matvec(self,x): return self.base.rev_matvec(x)
    def rev_matvec(self,x): return self.base.matvec(x)
    def to_dense(self): return self.base.to_dense().transpose(-1,-2)


class _DenseFallbackMatrix(Matrix):
    def __init__(self, tensor): self.values = tensor
    @property
    def shape(self): return self.values.shape
    @property
    def dtype(self): return self.values.dtype
    @property
    def device(self): return self.values.device
    @property
    def T(self): return _DenseFallbackMatrix(self.values.transpose(-1,-2))
    def to_dense(self): return self.values
    def matvec(self,x): return self.values @ x
    def mul_diag_left(self,a): return _DenseFallbackMatrix(torch.as_tensor(a,device=self.device,dtype=self.dtype).unsqueeze(-1)*self.values)
    def mul_diag_right(self,a): return _DenseFallbackMatrix(self.values*torch.as_tensor(a,device=self.device,dtype=self.dtype).unsqueeze(-2))


class _WeightedMatrix(_DenseFallbackMatrix):
    """Schur-product adapter. Dense by design, for small reference operations."""
    def __init__(self, base, factors):
        factors = tuple(factors)
        if len(factors) != base.d:
            raise ValueError('one matrix factor per site')
        n = math.prod(base.row_modes)
        if n*n > 1_000_000:
            raise NotImplementedError('general Schur-scaled MPO has no guaranteed compact form')
        factors = [torch.as_tensor(x,dtype=base.dtype,device=base.device) for x in factors]
        batch = _batch(base.batch_size, *(1 if x.ndim==2 else x.shape[0] for x in factors))
        rows=[]
        for b in range(batch):
            local = [x if x.ndim==2 else x[b if x.shape[0]>1 else 0] for x in factors]
            f = reduce(torch.kron,local)
            m = base.to_dense()
            rows.append((m if m.ndim==2 else m[b if m.shape[0]>1 else 0])*f)
        super().__init__(torch.stack(rows) if batch>1 else rows[0])


class _SchurNestedMatrix(Matrix):
    """General CP MPO after Schur multiplication by rank-one matrix factors.

    No full spatial tensor product is ever formed by matvec/contraction. Local
    channel matrices are no longer required to commute or be stochastic.
    """
    def __init__(self,base, factors):
        factors=tuple(factors)
        if len(factors)!=base.d:
            raise ValueError('one separable matrix factor per physical mode')
        self.base=base
        self.factors=tuple(torch.as_tensor(f,dtype=base.dtype,device=base.device) for f in factors)
        for f,n in zip(self.factors,base.row_modes):
            if f.ndim not in (2,3) or f.shape[-2:]!=(n,n):
                raise ValueError('expected (n,n) or (B,n,n) physical factor')
        self._batch_size=_batch(base.batch_size,*(1 if f.ndim==2 else f.shape[0] for f in self.factors))
    @property
    def batch_size(self): return self._batch_size
    @property
    def d(self): return self.base.d
    @property
    def row_modes(self): return self.base.row_modes
    @property
    def col_modes(self): return self.base.col_modes
    @property
    def dtype(self): return self.base.dtype
    @property
    def device(self): return self.base.device
    @property
    def channel_weights(self): return self.base.channel_weights.expand(self.batch_size,-1)
    @property
    def channels(self): return self.base.channels
    @property
    def shape(self):
        n=math.prod(self.row_modes)
        return torch.Size((n,n) if self.batch_size==1 else (self.batch_size,n,n))
    @property
    def T(self): return _TransposeNestedMatrix(self)
    def channel_matrices(self):
        return tuple(leaf.matrices().expand(self.batch_size,-1,-1,-1)
                     * (f[None] if f.ndim==2 else f).expand(self.batch_size,-1,-1)[:,None]
                     for leaf,f in zip(self.base.leaves,self.factors))
    def matvec(self,x):
        if isinstance(x,NestedTTVector):
            if x.modes!=self.col_modes: raise ValueError('physical modes differ')
            return x._append(('cp',self,False),self.row_modes)
        return torch.matmul(self.to_dense(),torch.as_tensor(x).unsqueeze(-1)).squeeze(-1)
    def rev_matvec(self,x):
        if isinstance(x,NestedTTVector):
            if x.modes!=self.row_modes: raise ValueError('physical modes differ')
            return x._append(('cp',self,True),self.col_modes)
        return self.T.matvec(x)
    def elementwise_multiply(self,factors):
        return _SchurNestedMatrix(self.base,[a*b for a,b in zip(self.factors,factors)])
    __mul__=elementwise_multiply
    def __matmul__(self,x): return self.matvec(x)
    def to_dense(self,*,max_elements=1_000_000):
        n=math.prod(self.row_modes)
        if n*n*self.batch_size>max_elements: raise ValueError('dense matrix limit exceeded')
        locals=self.channel_matrices()
        out=[]
        for b in range(self.batch_size):
            terms=[]
            for q in range(self.channels):
                terms.append(self.channel_weights[b,q]*reduce(torch.kron,[P[b,q] for P in locals]))
            out.append(sum(terms))
        return out[0] if self.batch_size==1 else torch.stack(out)
    def sum(self):
        locals=self.channel_matrices()
        z=self.channel_weights
        for P in locals: z=z*P.sum(dim=(-2,-1))
        out=z.sum(-1)
        return out.squeeze(0) if self.batch_size==1 else out
    def diag(self): return self.to_dense().diagonal(dim1=-2,dim2=-1)


class NestedTTVector:
    """Initial TT plus symbolic ordered maps, with exact rank-one contraction.

    Consecutive commuting powers of the same operator/direction are collected.
    Other event sequences remain exact but may have exponentially costly
    contractions; a branch budget prevents unnoticed exponential expansion.
    """
    def __init__(self, cores, *, depth=3, events=(), modes=None, history_length=0):
        if depth != 3 or not cores:
            raise ValueError('requires depth=3 and nonempty TT cores')
        self.initial_cores = tuple(torch.as_tensor(x) for x in cores)
        self.depth = depth
        self.events = tuple(events)
        self._modes = tuple(x.shape[-2] for x in self.initial_cores) if modes is None else tuple(modes)
        self._history_length = int(history_length)
        self.spec = None
        self._last_contract_stats = None
        if any(c.ndim not in (3,4) for c in self.initial_cores):
            raise ValueError('TT cores must have (r,n,r) or (B,r,n,r) shape')
        if self.initial_cores[0].shape[-3] != 1 or self.initial_cores[-1].shape[-1] != 1:
            raise ValueError('TT boundary ranks must be one')
        for a,b in zip(self.initial_cores,self.initial_cores[1:]):
            if a.shape[-1] != b.shape[-3]:
                raise ValueError('incompatible TT ranks')

    @property
    def modes(self): return self._modes
    @property
    def cores(self): return self.initial_cores
    @property
    def leaves(self): return self.initial_cores
    @property
    def last_contract_stats(self): return self._last_contract_stats
    @property
    def d(self): return len(self.modes)
    @property
    def n(self): return math.prod(self.modes)
    @property
    def ranks(self): return (1,) + tuple(x.shape[-1] for x in self.initial_cores)
    @property
    def history_length(self): return self._history_length
    @property
    def leaf_count(self): return sum(x.numel() for x in self.initial_cores)
    @property
    def storage_numel(self): return self.leaf_count + sum(1 for _ in self.events)
    @property
    def batch_size(self):
        b = _batch(*(1 if x.ndim==3 else x.shape[0] for x in self.initial_cores))
        for event in self.events:
            if event[0] in ('power','cp'): b = _batch(b,event[1].batch_size)
            else: b = _batch(b,*(1 if t.ndim == (2 if event[0]=='map' else 1) else t.shape[0] for t in event[1]))
        return b
    @property
    def shape(self): return torch.Size((self.n,) if self.batch_size==1 else (self.batch_size,self.n))
    @property
    def dtype(self):
        dtype=self.initial_cores[0].dtype
        for ev in self.events:
            tensors=(ev[1].channel_weights,) if ev[0] in ('power','cp') else ev[1]
            for x in tensors:
                dtype=torch.promote_types(dtype,x.dtype)
        return dtype
    @property
    def device(self): return self.initial_cores[0].device
    @property
    def rank_one_factors(self):
        """Recover separable factors without constructing a dense basis tensor.

        In particular, TTBasis(coeffs=None).forward(y) has pending
        pointwise scales, and must remain usable as a rank-one divisor.
        """
        if any(c.shape[-3]!=1 or c.shape[-1]!=1 for c in self.initial_cores):
            return None
        fs=tuple(c[...,0,:,0] for c in self.initial_cores)
        for event in self.events:
            if event[0]=='scale':
                fs=tuple(a*b for a,b in zip(fs,event[1]))
            elif event[0]=='map':
                maps,reverse=event[1:]
                fs=tuple(torch.matmul(A.transpose(-1,-2) if reverse else A,
                                      _as_factor(a,A.shape[-2] if reverse else A.shape[-1],self.dtype,self.device).unsqueeze(-1)).squeeze(-1)
                         for a,A in zip(fs,maps))
            else:
                return None
        return tuple(a.squeeze(0) if a.ndim==2 and a.shape[0]==1 else a for a in fs)
    @property
    def normal_form_terms(self):
        return math.prod(math.comb(ev[3]+ev[1].channels-1, ev[1].channels-1)
                         if ev[0]=='power' else ev[1].channels if ev[0]=='cp' else 1
                         for ev in self.events)

    def _append(self,event,modes=None):
        events=list(self.events)
        if (event[0]=='power' and events and events[-1][0]=='power'
            and events[-1][1] is event[1] and events[-1][2]==event[2]):
            old=events[-1]
            events[-1]=('power',old[1],old[2],old[3]+event[3])
        else:
            events.append(event)
        return NestedTTVector(self.initial_cores,depth=self.depth,events=events,
                              modes=self.modes if modes is None else modes,
                              history_length=self.history_length+(event[3] if event[0]=='power' else 0))

    def elementwise_multiply(self, other):
        raw=_resolve_factors(other,self.modes)
        factors=tuple(_as_factor(f,n,self.dtype,self.device) for f,n in zip(raw,self.modes))
        return self._append(('scale',factors))
    __mul__ = elementwise_multiply
    def __rmul__(self,other): return self.elementwise_multiply(other)

    def elementwise_divide(self, other, *, eps=1e-12):
        if eps<=0: raise ValueError('eps must be positive')
        raw=_resolve_factors(other,self.modes)
        inv=[]
        for f,n in zip(raw,self.modes):
            x=_as_factor(f,n,self.dtype,self.device)
            z=torch.where(x.abs()<eps,torch.where(x<0,-eps,eps),x)
            inv.append(z.reciprocal())
        return self._append(('scale',tuple(inv)))

    def _dense_state(self, *, end=None, max_elements=4096):
        """Evaluate an ordered history in the physical state space when it is small.

        Complexity is O(T Q d n_max N) rather than O(Q**T), where
        N=prod(modes). This is a *small physical dimension* optimization,
        not a dimension-independent algorithm: N is exponential in d.
        """
        events = self.events if end is None else self.events[:end]
        dims = tuple(int(c.shape[-2]) for c in self.initial_cores)
        if math.prod(dims) > max_elements:
            raise ValueError('dense-state contraction would exceed the physical dimension guard')
        batch = _batch(*(1 if c.ndim == 3 else c.shape[0] for c in self.initial_cores))
        cores = [(c[None] if c.ndim == 3 else c) for c in self.initial_cores]
        z = cores[0].expand(batch,-1,-1,-1)[:,0]
        for core in cores[1:]:
            z = torch.einsum('b...r,brns->b...ns', z, core.expand(batch,-1,-1,-1))
        state = z.squeeze(-1)

        def apply_local(state, matrices, *, reverse=False):
            nonlocal dims
            # state: (B, n_0, ..., n_{d-1}); matrices[k]: (B, out, in)
            for k, A in enumerate(matrices):
                if A.ndim == 2:
                    A = A[None]
                if reverse:
                    A = A.transpose(-1, -2)
                batch = _batch(int(state.shape[0]), int(A.shape[0]))
                dtype = torch.promote_types(state.dtype, A.dtype)
                incoming = state.to(dtype).expand(batch,*state.shape[1:]).movedim(k+1,-1)
                outer = incoming.shape[1:-1]
                A = A.to(dtype).expand(batch,-1,-1)
                result = torch.bmm(incoming.reshape(batch,-1,incoming.shape[-1]), A.transpose(-1,-2))
                state = result.reshape(batch,*outer,A.shape[-2]).movedim(-1,k+1)
            dims = tuple(int(state.shape[k+1]) for k in range(self.d))
            return state

        for event in events:
            kind, *args = event
            if kind == 'scale':
                fs = args[0]
                for k,f in enumerate(fs):
                    f = _as_factor(f, dims[k], state.dtype, state.device)
                    shape = (f.shape[0],) + (1,)*k + (dims[k],) + (1,)*(self.d-k-1)
                    state = state * f.reshape(shape)
            elif kind == 'map':
                matrices, reverse = args
                state = apply_local(state, matrices, reverse=reverse)
            else:
                if kind == 'power':
                    op, reverse, repetitions = args
                    mats = tuple(leaf.matrices() for leaf in op.leaves)
                    weights = op.channel_weights
                elif kind == 'cp':
                    op, reverse = args
                    repetitions = 1
                    mats = op.channel_matrices()
                    weights = op.channel_weights
                else:
                    raise ValueError(f'unknown history event: {kind!r}')
                # Contract the CP channel as an *internal batch axis*.
                # Work remains O(B Q N) and never enumerates channel histories.
                for _ in range(repetitions):
                    original_dims = dims
                    B = _batch(int(state.shape[0]), int(weights.shape[0]), *(int(x.shape[0]) for x in mats))
                    Q = weights.shape[-1]
                    z = state[:,None].expand(B,Q,*dims).reshape(B*Q,*dims)
                    local = [P.expand(B,-1,-1,-1).reshape(B*Q,*P.shape[-2:]) for P in mats]
                    z = apply_local(z, local, reverse=reverse)
                    new_dims = dims
                    z = z.reshape(B,Q,*new_dims)
                    state = (z * weights.expand(B,Q).reshape(B,Q,*([1]*self.d))).sum(dim=1)
                    dims = new_dims
        return state

    def _dense_contract(self, factors, *, max_elements=4096):
        # Pull final pointwise factors back into the query. In particular,
        # TTBasis.forward(points).sum() must not propagate one state per point.
        end = len(self.events)
        while end and self.events[end-1][0] == 'scale':
            fs = self.events[end-1][1]
            factors = tuple(f*z for f,z in zip(factors,fs))
            end -= 1
        state = self._dense_state(end=end, max_elements=max_elements)
        dims = tuple(int(x) for x in state.shape[1:])
        B = _batch(state.shape[0], *(f.shape[0] for f in factors))
        state = state.expand(B,*dims)
        for k,f in enumerate(factors):
            f=f.expand(B,-1)
            state = state * f.reshape((B,)+(1,)*k+(dims[k],)+(1,)*(self.d-k-1))
        result = state.sum(dim=tuple(range(1,self.d+1)))
        self._last_contract_stats=ContractionStats(
            operands=len(self.initial_cores)+len(self.events),
            pair_contractions=len(self.events),
            max_intermediate_numel=max(state.numel(), *(c.numel() for c in self.initial_cores)),
            max_intermediate_ndim=self.d+2,
        )
        return result.squeeze(0) if B==1 else result

    def contract(self, factors, *, max_terms=100_000):
        raw=_resolve_factors(factors,self.modes)
        fs=tuple(_as_factor(f,n,self.dtype,self.device) for f,n in zip(raw,self.modes))
        # The real VDP experiment has only 10**2=100 physical coefficients.
        # Ordered dense-state updates avoid exponential CP-history enumeration
        # while retaining full batched sample evaluation at the output.
        if self.n <= 4096:
            return self._dense_contract(fs)
        if self.normal_form_terms > max_terms:
            raise RuntimeError(f'contract requires {self.normal_form_terms} normal-form terms; '
                               f'budget {max_terms}. No polynomial bound for this mixed event sequence.')
        B=_batch(self.batch_size,*(f.shape[0] for f in fs))
        fs=tuple(f.expand(B,-1) for f in fs)
        out=torch.zeros(B,dtype=self.dtype,device=self.device)
        # Cache within call only so no saved autograd graph becomes stale.
        matrix_cache={}
        def recurse(i, factors, weight):
            if i < 0:
                return weight * self._initial_contract(factors,B)
            kind,*args=self.events[i]
            if kind=='scale':
                f=args[0]
                return recurse(i-1,tuple(x*z.expand(B,-1) for x,z in zip(factors,f)),weight)
            if kind=='map':
                maps, reverse=args
                new=[]
                for x,A in zip(factors,maps):
                    A=A.expand(B,-1,-1)
                    new.append(torch.bmm(A if reverse else A.transpose(-1,-2),x.unsqueeze(-1)).squeeze(-1))
                return recurse(i-1,tuple(new),weight)
            if kind=='cp':
                op,reverse=args
                terms=torch.zeros_like(weight)
                local=op.channel_matrices()
                for q in range(op.channels):
                    pulled=tuple(torch.bmm(P[:,q] if reverse else P[:,q].transpose(-1,-2),
                                           f.unsqueeze(-1)).squeeze(-1)
                                 for P,f in zip(local,factors))
                    terms=terms+recurse(i-1,pulled,weight*op.channel_weights[:,q].expand(B))
                return terms
            op, reverse, t=args
            terms=torch.zeros_like(weight)
            w=op.channel_weights.expand(B,-1)
            for alpha in _count_vectors(t,op.channels):
                # lgamma multinomial avoids incorrect integer division ordering.
                logm=math.lgamma(t+1)-sum(math.lgamma(a+1) for a in alpha)
                coef=torch.exp(w.new_full((B,),logm)+sum(a*w[:,q].log() for q,a in enumerate(alpha)))
                pulled=[]
                for k,(f,leaf) in enumerate(zip(factors,op.leaves)):
                    key=(id(op),k,alpha,not reverse)
                    if key not in matrix_cache:
                        matrix_cache[key]=leaf.accumulated(alpha,transpose=not reverse).expand(B,-1,-1)
                    pulled.append(torch.bmm(matrix_cache[key],f.unsqueeze(-1)).squeeze(-1))
                terms=terms+recurse(i-1,tuple(pulled),weight*coef)
            return terms
        result=recurse(len(self.events)-1,fs,torch.ones_like(out))
        # A useful conservative bound on local workspaces. This is not a proof
        # of bounded cost for arbitrary mixed noncommuting event histories.
        local_max=max((leaf.n**2 * B for e in self.events if e[0]=='power'
                       for leaf in e[1].leaves), default=0)
        core_max=max(c.numel() for c in self.initial_cores)
        self._last_contract_stats=ContractionStats(
            operands=len(self.initial_cores)+len(self.events),
            pair_contractions=self.normal_form_terms * max(1,len(self.modes)),
            max_intermediate_numel=max(local_max,core_max),
            max_intermediate_ndim=4,
        )
        return result.squeeze(0) if B==1 else result

    def _initial_contract(self,factors,B):
        z=torch.ones(B,1,1,dtype=self.dtype,device=self.device)
        for g,f in zip(self.initial_cores,factors):
            g=(g.unsqueeze(0) if g.ndim==3 else g).to(dtype=self.dtype).expand(B,-1,-1,-1)
            z=torch.bmm(z,torch.einsum('brns,bn->brs',g,f))
        return z.squeeze(-1).squeeze(-1)

    def sum(self):
        return self.contract(tuple(torch.ones(n,dtype=self.dtype,device=self.device) for n in self.modes))

    def contract_basis(self,basis,points):
        if getattr(basis,'_default_coeffs',False) is not True:
            raise ValueError('requires coefficient-free TTBasis')
        return self.contract(basis._primitive_factors(points))

    def to_dense(self, *, max_elements=1_000_000):
        B=self.batch_size
        if B*self.n>max_elements:
            raise ValueError('dense vector materialization exceeds max_elements')
        if self.n <= 4096:
            state=self._dense_state().reshape(self.batch_size,self.n)
            return state.squeeze(0) if self.batch_size==1 else state
        onehot=tuple(torch.eye(n,dtype=self.dtype,device=self.device) for n in self.modes)
        # Debug-only, one coordinate at a time. Production sum/contract does not call this.
        values=[]
        for idx in product(*(range(n) for n in self.modes)):
            factors=tuple(onehot[k][j] for k,j in enumerate(idx))
            values.append(self.contract(factors))
        output=torch.stack(values,-1)
        return output.squeeze(0) if B==1 else output

    def materialize_cores(self, *, max_elements=1_000_000):
        """Dense TT-SVD reconstruction for small/debug cases only.

        Exact up to floating-point roundoff; it intentionally has a dense
        materialization guard because exact output TT ranks can be exponential.
        """
        if not self.events:
            return self.initial_cores
        x=self.to_dense(max_elements=max_elements)
        if x.ndim==1: x=x[None]
        B=x.shape[0]
        all_cores=[]
        for b in range(B):
            rem=x[b].reshape(1,-1)
            left=1
            pieces=[]
            for n in self.modes[:-1]:
                mat=rem.reshape(left*n,-1)
                U,s,Vh=torch.linalg.svd(mat,full_matrices=False)
                rank=s.shape[0]
                pieces.append(U.reshape(left,n,rank).permute(0,1,2))
                rem=s[:,None]*Vh
                left=rank
            pieces.append(rem.reshape(left,self.modes[-1],1))
            all_cores.append(pieces)
        return tuple(torch.stack([all_cores[b][k] for b in range(B)]) if B>1
                     else all_cores[0][k] for k in range(self.d))

    def __repr__(self):
        return f'NestedTTVector(modes={self.modes}, batch={self.batch_size}, events={len(self.events)}, steps={self.history_length})'


class SeparableNestedTTMatrix(Matrix):
    """Rank-one MPO: exact batch-compatible separable Gram/physical linear map."""
    def __init__(self,cores,*,depth=3):
        if depth!=3 or not cores: raise ValueError('depth must be 3')
        self.cores=tuple(torch.as_tensor(x) for x in cores)
        if any(x.ndim not in (2,3) for x in self.cores): raise ValueError('matrix factors must be (m,n) or (B,m,n)')
        self.depth=depth
        self.row_modes=tuple(x.shape[-2] for x in self.cores)
        self.col_modes=tuple(x.shape[-1] for x in self.cores)
        self.batch_size=_batch(*(1 if x.ndim==2 else x.shape[0] for x in self.cores))
    @property
    def dtype(self): return self.cores[0].dtype
    @property
    def device(self): return self.cores[0].device
    @property
    def shape(self):
        sh=(math.prod(self.row_modes),math.prod(self.col_modes))
        return torch.Size(sh if self.batch_size==1 else (self.batch_size,*sh))
    @property
    def T(self): return SeparableNestedTTMatrix([x.transpose(-1,-2) for x in self.cores])
    @property
    def separable_cores(self): return self.cores
    def matvec(self,x):
        if not isinstance(x,NestedTTVector):
            return torch.matmul(self.to_dense(),torch.as_tensor(x).unsqueeze(-1)).squeeze(-1)
        if x.modes!=self.col_modes: raise ValueError('physical modes do not match')
        return x._append(('map',tuple(x.unsqueeze(0) if x.ndim==2 else x for x in self.cores),False),self.row_modes)
    def rev_matvec(self,x):
        if isinstance(x,NestedTTVector):
            if x.modes!=self.row_modes: raise ValueError('physical modes do not match')
            return x._append(('map',tuple(x.unsqueeze(0) if x.ndim==2 else x for x in self.cores),True),self.col_modes)
        return self.T.matvec(x)
    def to_dense(self,*,max_elements=1_000_000):
        n,m=math.prod(self.row_modes),math.prod(self.col_modes)
        if n*m*self.batch_size>max_elements: raise ValueError('dense matrix limit exceeded')
        vals=[]
        for b in range(self.batch_size):
            a=[x if x.ndim==2 else x[b if x.shape[0]>1 else 0] for x in self.cores]
            vals.append(reduce(torch.kron,a))
        return vals[0] if self.batch_size==1 else torch.stack(vals)
    def sum(self):
        values=[x.sum(dim=(-2,-1)) for x in self.cores]
        return reduce(torch.mul,values)
    def elementwise_multiply(self,factors):
        return SeparableNestedTTMatrix([a*b for a,b in zip(self.cores,factors)])
    __mul__=elementwise_multiply
    def __rmul__(self,factors): return self.elementwise_multiply(factors)
    def __matmul__(self,x): return self.matvec(x)


def nested_tt_matrix_from_separable_cores(cores, *, depth=3):
    return SeparableNestedTTMatrix(cores,depth=depth)


def nested_tt_vector_from_factors(factors, *, depth=3):
    cores=[]
    for f in factors:
        f=torch.as_tensor(f)
        cores.append(f.reshape(1,-1,1) if f.ndim==1 else f[:,None,:,None])
    return NestedTTVector(cores,depth=depth)


def ones_nested_tt_vector(modes, *, depth=3, dtype=torch.float32,device=None):
    return nested_tt_vector_from_factors([torch.ones(n,dtype=dtype,device=device) for n in modes],depth=depth)
