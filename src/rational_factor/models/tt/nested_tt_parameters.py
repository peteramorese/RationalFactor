"""Parameter wrappers for the commuting matrix-polynomial nested TT model."""
from __future__ import annotations
from collections.abc import Sequence
import torch
try:
    from .nested_tt import (NestedTTMatrix,NestedTTMatrixSpec,NestedTTVector,
                            NestedTTVectorSpec)
    from .nested_tt_leaf import StochasticPolynomialLeaf
except ImportError:
    from nested_tt import (NestedTTMatrix,NestedTTMatrixSpec,NestedTTVector,
                           NestedTTVectorSpec)
    from nested_tt_leaf import StochasticPolynomialLeaf


def _v(p): return p() if callable(p) and not isinstance(p,torch.Tensor) else p


def _modules(seq):
    out=[]
    for x in seq:
        if hasattr(x,'parameter_modules'): out.extend(x.parameter_modules())
        elif isinstance(x,torch.nn.Module): out.append(x)
    return out


class NestedTTVectorParameters:
    def __init__(self,spec:NestedTTVectorSpec,cores:Sequence,*,positive=False):
        self.spec=spec
        self.cores=tuple(cores)
        self.positive=positive
        d=len(spec.modes)
        bond=(1,)+(spec.bond_rank,)*(d-1)+(1,)
        if len(cores)!=d or any(tuple(_v(c).shape)[-3:]!=(bond[k],spec.modes[k],bond[k+1]) for k,c in enumerate(cores)):
            raise ValueError('vector core shapes do not match specification')
    @classmethod
    def from_core_spec(cls,modes,*,depth=3,bond_rank=None,ranks=2,dtype=torch.float64,
                       device=None,positive=False,leaf_factory=None,trainable=True,**kwargs):
        r = ranks if bond_rank is None else bond_rank
        if not isinstance(r,int):
            r=tuple(r)[-1]
        spec=NestedTTVectorSpec(tuple(modes),depth,r)
        bond=(1,)+(r,)*(len(modes)-1)+(1,)
        tensors=[]
        for k,n in enumerate(modes):
            shape=(bond[k],n,bond[k+1])
            z=leaf_factory(shape) if leaf_factory is not None else 0.2*torch.randn(*shape,dtype=dtype,device=device)
            if trainable and leaf_factory is None: z=torch.nn.Parameter(z)
            tensors.append(z)
        return cls(spec,tensors,positive=positive)
    @property
    def modes(self): return self.spec.modes
    @property
    def depth(self): return self.spec.depth
    @property
    def ranks(self): return (self.spec.bond_rank,)*self.depth
    def __call__(self):
        cores=tuple(torch.nn.functional.softplus(_v(c)) if self.positive else _v(c) for c in self.cores)
        v=NestedTTVector(cores)
        if self.positive:
            z=v.sum()
            cores=(cores[0]/z,*cores[1:])
        return NestedTTVector(cores)
    def is_trainable(self): return any(bool(_v(x).requires_grad) for x in self.cores)
    def parameters(self): return tuple(x for c in self.cores for x in ([c] if isinstance(c,torch.Tensor) and c.requires_grad else list(c.parameters()) if isinstance(c,torch.nn.Module) else []))
    def parameter_modules(self): return _modules(self.cores)
    def is_module(self): return False


class RowStochasticNestedTTMatrixParameters:
    """Independent row-softmax Markov bases and simplex polynomial coefficients.

    ``shared_bases=True``: one S_k per site, shared across all Q CP channels.
    ``shared_bases=False``: separate S[k,q], which are required to commute.
    By default independent trainable S[k,q] are incompatible with that condition,
    so nonshared bases must be supplied explicitly and checked at call time.
    """
    def __init__(self,spec:NestedTTMatrixSpec,base_logits,coefficient_logits,
                 channel_logits,*,shared_bases=True,already_stochastic=False):
        self.spec=spec
        self.base_logits=tuple(base_logits)
        self.coefficient_logits=tuple(coefficient_logits)
        self.channel_logits=channel_logits
        self.shared_bases=shared_bases
        self.already_stochastic=already_stochastic
        if len(self.base_logits)!=spec.d or len(self.coefficient_logits)!=spec.d:
            raise ValueError('one base/coeff tensor per physical site')
    @classmethod
    def from_core_spec(cls,row_modes,*,col_modes=None,depth=3,channel_rank=None,
                       ranks=None,polynomial_degree=2,degree=None,shared_bases=True,
                       base_matrices=None,coefficient_logits=None,channel_logits=None,
                       dtype=torch.float64,device=None,trainable=True,
                       trainable_bases=None,**kwargs):
        row_modes=tuple(int(n) for n in row_modes)
        col_modes=row_modes if col_modes is None else tuple(col_modes)
        if trainable_bases is None:
            trainable_bases = trainable and shared_bases
        if trainable_bases and not shared_bases:
            raise ValueError('freely training separate S[k,q] would break commutativity; '
                             'train polynomial coefficients or use a shared trainable base')
        if channel_rank is None:
            channel_rank=int(ranks) if isinstance(ranks,int) else int(tuple(ranks)[-1]) if ranks is not None else 2
        if degree is not None: polynomial_degree=int(degree)
        spec=NestedTTMatrixSpec(row_modes,col_modes,depth,int(channel_rank),int(polynomial_degree))
        bases=[]; coeffs=[]
        for k,n in enumerate(row_modes):
            if base_matrices is None:
                if not shared_bases:
                    raise ValueError('independent channel bases must be supplied explicitly and commute')
                z=torch.randn(n,n,dtype=dtype,device=device)*0.2
            else:
                z=torch.as_tensor(base_matrices[k],dtype=dtype,device=device)
                if z.shape not in ((n,n),(spec.channel_rank,n,n)):
                    raise ValueError('base shape must be (n,n) or (Q,n,n)')
                # Supplied matrices are stochastic probabilities; turn into
                # equivalent logits, with exact zeros approximated by tiny mass.
                z=z.clamp_min(torch.finfo(z.dtype).tiny).log()
            if trainable_bases: z=torch.nn.Parameter(z)
            bases.append(z)
            # Independent channel logits are essential. Identical zeros make
            # every CP channel identical under shared_bases=True, and their
            # exact gradient symmetry keeps the entire operator rank one.
            c=torch.randn(spec.channel_rank,polynomial_degree+1,dtype=dtype,device=device) if coefficient_logits is None else torch.as_tensor(coefficient_logits[k],dtype=dtype,device=device)
            if trainable: c=torch.nn.Parameter(c)
            coeffs.append(c)
        w=torch.zeros(spec.channel_rank,dtype=dtype,device=device) if channel_logits is None else torch.as_tensor(channel_logits,dtype=dtype,device=device)
        if trainable: w=torch.nn.Parameter(w)
        return cls(spec,bases,coeffs,w,shared_bases=shared_bases)
    @property
    def row_modes(self): return self.spec.row_modes
    @property
    def col_modes(self): return self.spec.col_modes
    @property
    def depth(self): return self.spec.depth
    @property
    def ranks(self): return (self.spec.channel_rank,)*self.depth
    @property
    def separation_rank(self): return self.spec.channel_rank
    def __call__(self):
        leaves=[]
        for b,c in zip(self.base_logits,self.coefficient_logits):
            bs=_v(b)
            if self.already_stochastic:
                base=bs
            else:
                base=bs.softmax(-1)
            # Resolve ambiguous (B,n,n) versus (Q,n,n) dimensions.
            if self.shared_bases and base.ndim==3:
                base=base[:,None,:,:]
            elif not self.shared_bases and base.ndim==3:
                base=base[None,:,:,:]
            leaves.append(StochasticPolynomialLeaf(base,_v(c).softmax(-1),check=True))
        return NestedTTMatrix(self.spec,leaves,_v(self.channel_logits))
    def parameters(self):
        return tuple(x for p in (*self.base_logits,*self.coefficient_logits,self.channel_logits)
                     for x in ([p] if isinstance(p,torch.Tensor) and p.requires_grad else list(p.parameters()) if isinstance(p,torch.nn.Module) else []))
    def is_trainable(self): return bool(self.parameters())
    def parameter_modules(self): return _modules((*self.base_logits,*self.coefficient_logits,self.channel_logits))
    def is_module(self): return False


NestedTTMatrixParameters=RowStochasticNestedTTMatrixParameters


class FixedNestedTTVectorParameters:
    def __init__(self,value:NestedTTVector): self.value=value
    @property
    def modes(self): return self.value.modes
    @property
    def depth(self): return self.value.depth
    @property
    def ranks(self): return self.value.ranks
    def __call__(self): return self.value
    def parameters(self): return ()
    def is_trainable(self): return False
    def parameter_modules(self): return []
    def is_module(self): return False
