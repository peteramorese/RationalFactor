import math
import pytest
import torch
from .nested_tt import (NestedTTVector,NestedTTMatrix,VirtualIndexMPO,
                        nested_tt_vector_from_factors,nested_tt_matrix_from_separable_cores,
                        ones_nested_tt_vector)
from .nested_tt_leaf import StochasticPolynomialLeaf
from .nested_tt_parameters import (NestedTTVectorParameters,RowStochasticNestedTTMatrixParameters)


def setup(d=3,n=3,q=2,degree=2,B=1,independent=False):
    torch.manual_seed(51)
    m=RowStochasticNestedTTMatrixParameters.from_core_spec(
        (n,)*d,channel_rank=q,polynomial_degree=degree,trainable=True,dtype=torch.float64)
    v=NestedTTVectorParameters.from_core_spec((n,)*d,bond_rank=2,dtype=torch.float64,positive=True)
    if B>1:
        m.channel_logits=torch.nn.Parameter(torch.randn(B,q,dtype=torch.float64))
        m.base_logits=tuple(torch.nn.Parameter(x.expand(B,*x.shape).clone()) for x in m.base_logits)
        m.coefficient_logits=tuple(torch.nn.Parameter(x.expand(B,*x.shape).clone()) for x in m.coefficient_logits)
    return m(),v(),m,v


def test_dense_row_stochastic_and_virtual_mpo():
    M,v,_,_=setup(d=3,q=3)
    a=M.to_dense()
    assert (a>=0).all()
    torch.testing.assert_close(a.sum(-1),torch.ones(a.shape[0],dtype=a.dtype))
    C=M.materialize_cores()
    for k,c in enumerate(C):
        assert c.shape==(1 if k==0 else 3,3,3,1 if k==2 else 3)
    core=M.cores[1]
    mpo=core.virtual_mpo(0,1)
    torch.testing.assert_close(mpo.to_dense(),torch.diag(M.leaves[1].matrices()[0,:,0,1]))


def test_homogeneous_forward_reverse_and_large_t():
    M,v,_,_=setup(d=2,q=2)
    a=M.to_dense()
    ref=v.to_dense().reshape(-1)
    f=(torch.tensor([.3,.4,.7],dtype=torch.float64),torch.tensor([.1,.9,.4],dtype=torch.float64))
    phi=torch.kron(*f)
    for t in range(1,7):
        v=M.matvec(v)
        ref=a@ref
        torch.testing.assert_close(v.to_dense().reshape(-1),ref,atol=1e-11,rtol=1e-11)
        torch.testing.assert_close(v.contract(f),phi@ref,atol=1e-11,rtol=1e-11)
        assert v.history_length==t
        assert len(v.events)==1
    x=ones_nested_tt_vector((3,3),dtype=torch.float64)
    torch.testing.assert_close(M.matvec(x).to_dense().reshape(-1),torch.ones(9,dtype=torch.float64))
    x=M.rev_matvec(x)
    torch.testing.assert_close(x.to_dense().reshape(-1),a.T@torch.ones(9,dtype=torch.float64))


def test_pointwise_divide_then_forward_and_reverse_and_gram():
    M,v,_,_=setup(d=2)
    phi=(torch.tensor([0.3,1.3,1.8],dtype=torch.float64),torch.tensor([.5,1.5,.9],dtype=torch.float64))
    x=v.to_dense().reshape(-1)
    divided=v.elementwise_divide(phi)
    ref=x/torch.kron(*phi)
    torch.testing.assert_close(divided.to_dense().reshape(-1),ref)
    z=M.matvec(divided)
    torch.testing.assert_close(z.to_dense().reshape(-1),M.to_dense()@ref)
    z2=M.rev_matvec(divided)
    torch.testing.assert_close(z2.to_dense().reshape(-1),M.to_dense().T@ref)
    mats=(torch.randn(3,3,dtype=torch.float64),torch.randn(3,3,dtype=torch.float64))
    G=nested_tt_matrix_from_separable_cores(mats)
    x3=M.matvec(G.matvec(divided.elementwise_multiply(phi)))
    expected=M.to_dense()@torch.kron(*mats)@(ref*torch.kron(*phi))
    torch.testing.assert_close(x3.to_dense().reshape(-1),expected)
    y=G.rev_matvec(M.matvec(divided))
    torch.testing.assert_close(y.to_dense().reshape(-1),torch.kron(*mats).T@M.to_dense()@ref)


def test_batched_pointwise_divide_and_gram():
    M,v,_,_=setup(d=2,B=3)
    a=M.to_dense()
    vf=(torch.randn(3,3,dtype=torch.float64).abs()+0.2,torch.randn(3,3,dtype=torch.float64).abs()+0.2)
    vb=v.elementwise_divide(vf)
    initial=v.to_dense().reshape(-1)
    ref=torch.stack([initial/torch.kron(vf[0][b],vf[1][b]) for b in range(3)])
    got=M.matvec(vb)
    torch.testing.assert_close(got.to_dense(),torch.einsum('bij,bj->bi',a,ref))
    gram=(torch.randn(3,3,3,dtype=torch.float64),torch.randn(3,3,3,dtype=torch.float64))
    G=nested_tt_matrix_from_separable_cores(gram)
    out=G.matvec(got)
    ref_gram=torch.stack([torch.kron(gram[0][b],gram[1][b])@a[b]@ref[b] for b in range(3)])
    torch.testing.assert_close(out.to_dense(),ref_gram)
    assert out.shape==torch.Size((3,9))


def test_polynomial_fft_coefficients_and_differentiability():
    a=torch.tensor([[.2,.3,.5]],dtype=torch.float64,requires_grad=True)
    b=torch.tensor([[.6,.4]],dtype=torch.float64,requires_grad=True)
    c=StochasticPolynomialLeaf.multiply_coeffs(a,b)
    ref=torch.tensor([[.12,.26,.42,.20]],dtype=torch.float64)
    torch.testing.assert_close(c,ref,atol=1e-12,rtol=1e-12)
    p=StochasticPolynomialLeaf.power_coeffs(a,3)
    assert p.shape[-1]==7
    (c.square().sum()+p.square().sum()).backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


def test_independent_commuting_bases_per_channel():
    n=3
    raw=torch.tensor([[.6,.2,.2],[.1,.7,.2],[.2,.2,.6]],dtype=torch.float64)
    S=raw
    S2=0.4*torch.eye(n,dtype=S.dtype)+0.6*S
    bases=torch.stack((S,S2))
    c=torch.tensor([[.1,.7,.2],[.3,.4,.3]],dtype=S.dtype)
    leaf=StochasticPolynomialLeaf(bases,c)
    torch.testing.assert_close(leaf.accumulated((2,3)),
        leaf.matrices()[:,0].matrix_power(2)@leaf.matrices()[:,1].matrix_power(3))
    bad=torch.eye(n,dtype=S.dtype).roll(1,0)
    with pytest.raises(ValueError,match='commute'):
        StochasticPolynomialLeaf(torch.stack((S,bad)),c)


def test_trainable_gradients_and_no_dense_contract():
    M,v,mp,vp=setup(d=3,q=2)
    x=(torch.tensor([0.1,.8,.3],dtype=torch.float64),)*3
    for _ in range(3): v=M.matvec(v)
    original=NestedTTMatrix.to_dense
    try:
        NestedTTMatrix.to_dense=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('no global dense matrix'))
        val=v.contract(x)
    finally:
        NestedTTMatrix.to_dense=original
    val.backward()
    for p in (*mp.parameters(),*vp.parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_large_dimension_linear_storage_and_polynomial_work():
    M,v,_,_=setup(d=15,n=2,q=2,degree=1)
    storage=v.storage_numel
    for _ in range(18): v=M.matvec(v)
    assert v.storage_numel==storage+1 and len(v.events)==1
    f=(torch.tensor([.2,.9],dtype=torch.float64),)*15
    got=v.contract(f)
    assert got.ndim==0 and torch.isfinite(got)
    assert v.normal_form_terms==19


def test_normal_form_branch_budget_for_mixed_sequences():
    M,v,_,_=setup(d=2,q=2)
    for _ in range(2):v=M.matvec(v)
    v=v.elementwise_multiply((torch.ones(3),torch.ones(3)))
    for _ in range(2):v=M.matvec(v)
    with pytest.raises(RuntimeError,match='budget'):
        v.contract([torch.ones(3)]*2,max_terms=3)


def test_sep_transpose_and_batched_basis():
    mats=(torch.randn(2,3,3,dtype=torch.float64),torch.randn(2,3,3,dtype=torch.float64))
    M=nested_tt_matrix_from_separable_cores(mats)
    v=nested_tt_vector_from_factors((torch.randn(2,3,dtype=torch.float64),torch.randn(2,3,dtype=torch.float64)))
    got=M.rev_matvec(v).to_dense()
    ref=torch.stack([torch.kron(mats[0][b],mats[1][b]).T@torch.kron(v.initial_cores[0][b,0,:,0],v.initial_cores[1][b,0,:,0]) for b in range(2)])
    torch.testing.assert_close(got,ref)



def test_ttbasis_coeff_free_rankone_division_batched():
    v=ones_nested_tt_vector((3,3),dtype=torch.float64)
    f=(torch.tensor([[.2,.4,.8],[.3,.6,.9]],dtype=torch.float64),
       torch.tensor([[.5,1.2,1.8],[1.5,.7,.6]],dtype=torch.float64))
    phi=v.elementwise_multiply(f)
    assert phi.rank_one_factors is not None
    y=v.elementwise_divide(phi)
    expected=torch.stack([torch.kron(f[0][b],f[1][b]).reciprocal() for b in range(2)])
    torch.testing.assert_close(y.to_dense(),expected)


def test_nested_mpo_core_reconstruction_batched():
    M,_,_,_=setup(d=3,n=2,q=3,B=2)
    cores=M.materialize_cores()
    for b in range(2):
        out=[]
        for rows in __import__('itertools').product(range(2),repeat=3):
            row=[]
            for cols in __import__('itertools').product(range(2),repeat=3):
                acc=cores[0][b,0,rows[0],cols[0],:]
                for k in range(1,3):
                    acc=acc @ cores[k][b,:,rows[k],cols[k],:]
                row.append(acc.squeeze())
            out.append(torch.stack(row))
        torch.testing.assert_close(torch.stack(out),M.to_dense()[b])


def test_batched_same_size_as_number_of_channels():
    M,v,_,_=setup(d=2,n=3,q=2,B=2)
    dense=M.to_dense()
    ref=v.to_dense()
    got=M.matvec(v).to_dense()
    torch.testing.assert_close(got,torch.einsum('bij,j->bi',dense,ref))
    torch.testing.assert_close(dense.sum(-1),torch.ones(2,9,dtype=torch.float64))


def test_rectangular_gram_after_rankone_scaling():
    mats=(torch.randn(3,2,dtype=torch.float64),torch.randn(4,3,dtype=torch.float64))
    v=nested_tt_vector_from_factors((torch.tensor([.3,.5]),torch.tensor([.1,.2,.8])))
    f=(torch.tensor([.4,.8]),torch.tensor([.5,.7,.9]))
    G=nested_tt_matrix_from_separable_cores(mats)
    z=G.matvec(v.elementwise_multiply(f))
    assert z.modes==(3,4)
    torch.testing.assert_close(z.to_dense().reshape(-1),torch.kron(*mats)@((torch.kron(v.initial_cores[0][0,:,0],v.initial_cores[1][0,:,0])*torch.kron(*f)).double()))
    z2=G.rev_matvec(z)
    torch.testing.assert_close(z2.to_dense().reshape(-1),torch.kron(*mats).T@z.to_dense().reshape(-1))


def test_schur_product_large_dim_no_dense_and_batched():
    M,v,_,_=setup(d=3,n=3,q=2,B=2)
    Fs=tuple(torch.rand(2,3,3,dtype=torch.float64) for _ in range(3))
    schur=M.elementwise_multiply(Fs)
    expected=torch.stack([M.to_dense()[b]*torch.kron(torch.kron(Fs[0][b],Fs[1][b]),Fs[2][b]) for b in range(2)])
    torch.testing.assert_close(schur.to_dense(),expected)
    result=schur.matvec(v).to_dense()
    torch.testing.assert_close(result,torch.einsum('bij,j->bi',expected,v.to_dense()))
    assert len(schur.matvec(v).events)==1
    assert schur.matvec(v).normal_form_terms==2
    bigger,vb,_,_=setup(d=15,n=2,q=2)
    Fs_big=tuple(torch.eye(2,dtype=torch.float64) for _ in range(15))
    huge=bigger.elementwise_multiply(Fs_big)
    original=type(huge).to_dense
    try:
        type(huge).to_dense=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('dense conversion'))
        val=huge.matvec(vb).sum()
    finally:
        type(huge).to_dense=original
    assert torch.isfinite(val)


def test_leaf_fft_normal_form_equal_dense_and_gradient():
    M,_,mp,_=setup(d=2,n=3,q=2,degree=2)
    L=M.leaves[0]
    c=L.canonical_coefficients((2,3))
    assert c.shape==(1,11)
    S=L.bases[:,0]
    eye=torch.eye(L.n,dtype=S.dtype,device=S.device).expand(1,-1,-1)
    powers=[eye]
    for _ in range(c.shape[-1]-1): powers.append(powers[-1]@S)
    polynomial=torch.einsum('bl,blij->bij',c,torch.stack(powers,dim=1))
    torch.testing.assert_close(polynomial,L.accumulated((2,3)),atol=1e-10,rtol=1e-10)
    polynomial.sum().backward()
    for p in (mp.base_logits[0], mp.coefficient_logits[0]):
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_independent_fixed_commuting_bases_train_coefficients():
    from .nested_tt_parameters import RowStochasticNestedTTMatrixParameters
    n=3
    S=torch.tensor([[.6,.2,.2],[.1,.7,.2],[.2,.2,.6]],dtype=torch.float64)
    S2=.3*torch.eye(n,dtype=S.dtype)+.7*S
    params=RowStochasticNestedTTMatrixParameters.from_core_spec(
        (3,3),channel_rank=2,polynomial_degree=2,
        base_matrices=[torch.stack((S,S2))]*2,shared_bases=False,
        dtype=torch.float64,trainable=True)
    M=params()
    torch.testing.assert_close(M.to_dense().sum(-1),torch.ones(9,dtype=torch.float64))
    assert all(not isinstance(p,torch.nn.Parameter) for p in params.base_logits)
    with pytest.raises(ValueError,match='break commutativity'):
        RowStochasticNestedTTMatrixParameters.from_core_spec(
            (3,),channel_rank=2,base_matrices=[torch.stack((S,S2))],
            shared_bases=False,trainable_bases=True)


def test_materialize_small_propagated_tt_debug_and_batch():
    M,v,_,_=setup(d=3,n=2,q=2,B=2)
    w=M.matvec(M.matvec(v))
    cores=w.materialize_cores()
    assert all(g.ndim==4 for g in cores)
    recon=[]
    for b in range(2):
        x=cores[0][b,0]
        for g in cores[1:]:
            x=torch.einsum('pr,rns->pns',x.reshape(-1,x.shape[-1]),g[b])
        recon.append(x.reshape(-1))
    torch.testing.assert_close(torch.stack(recon),w.to_dense(),atol=1e-10,rtol=1e-10)
