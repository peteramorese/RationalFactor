import importlib.util
import math
import sys
import types

import torch

import rational_factor.models.tt.nested_tt
import rational_factor.models.tt.nested_tt_parameters


# Minimal package stubs so the supplied TTBasis module can be integration-tested
# in this isolated environment.
rf = types.ModuleType("rational_factor")
models = types.ModuleType("rational_factor.models")
tt_pkg = types.ModuleType("rational_factor.models.tt")
basis_mod = types.ModuleType("rational_factor.models.basis_functions")
params_mod = types.ModuleType("rational_factor.models.parameters")
mat_mod = types.ModuleType("rational_factor.models.structured_matrices")
vec_mod = types.ModuleType("rational_factor.models.structured_vectors")


class Basis:
    def __init__(self, *, dim, batch_size, n_basis, params, coeffs):
        self._dim = dim
        self._batch_size = batch_size
        self._n_basis = n_basis
        self._params = params
        self.coeffs = coeffs

    def dim(self):
        return self._dim


class SeparableBasis:
    pass


class FixedParameters:
    def __init__(self, x):
        self.x = x

    def __call__(self):
        return self.x


class TTVectorParameters:
    def __init__(self, cores):
        self._cores = tuple(cores)
        self.modes = tuple(int(c().shape[1]) for c in self._cores)
        self.ranks = tuple([1] * (len(self._cores) + 1))

    @classmethod
    def from_cores(cls, cores):
        return cls(cores)

    def __call__(self):
        return TTVector([c() for c in self._cores])


class TTVector:
    def __init__(self, cores, numerical_tolerance=1e-20):
        self.cores = tuple(cores)
        self.numerical_tolerance = numerical_tolerance
        self.ranks = tuple([1] * (len(self.cores) + 1))


class TTMatrix:
    def __init__(self, cores):
        self.cores = tuple(cores)


basis_mod.Basis = Basis
basis_mod.SeparableBasis = SeparableBasis
params_mod.FixedParameters = FixedParameters
params_mod.TTVectorParameters = TTVectorParameters
mat_mod.TTMatrix = TTMatrix
vec_mod.TTVector = TTVector

for name, module in {
    "rational_factor": rf,
    "rational_factor.models": models,
    "rational_factor.models.tt": tt_pkg,
    "rational_factor.models.basis_functions": basis_mod,
    "rational_factor.models.parameters": params_mod,
    "rational_factor.models.structured_matrices": mat_mod,
    "rational_factor.models.structured_vectors": vec_mod,
    "rational_factor.models.tt.nested_tt": rational_factor.models.tt.nested_tt,
    "rational_factor.models.tt.nested_tt_parameters": rational_factor.models.tt.nested_tt_parameters,
}.items():
    sys.modules[name] = module

spec = importlib.util.spec_from_file_location("tt_basis_under_test", "/mnt/data/tt_basis.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
TTBasis = mod.TTBasis


class FakePrimitives(SeparableBasis):
    def __init__(self, d=2, n=2, dtype=torch.float64):
        self.d = d
        self.n = n
        self.dtype = dtype
        self._gram = torch.tensor(
            [[[1.2, 0.3], [0.4, 0.9]], [[0.8, 0.1], [0.2, 1.1]]],
            dtype=dtype,
        )[:d, :n, :n]

    def batch_size(self):
        return 1

    def n_basis_functions(self):
        return self.n

    def dim(self):
        return self.d

    def params(self):
        return ()

    def dtype_device(self):
        return self.dtype, torch.device("cpu")

    def eval_dim(self, y):
        y = torch.as_tensor(y, dtype=self.dtype)
        # Positive, nontrivial values; shape (B,d,n).
        vals = []
        for k in range(self.d):
            vals.append(torch.stack((1.0 + y[:, k], 2.0 - 0.5 * y[:, k]), dim=-1))
        return torch.stack(vals, dim=1)

    def log_Omega1_dim(self, lows=None, highs=None):
        vals = torch.tensor([[[1.1, 0.7], [0.9, 1.3]]], dtype=self.dtype)
        return vals[:, : self.d, : self.n].log()

    def log_Omega2_dim(self, other, lows=None, highs=None):
        return self._gram[None].log()


def kron_all(xs):
    out = xs[0]
    for x in xs[1:]:
        out = torch.kron(out, x)
    return out


def test_nested_ttbasis_batched_forward_and_omega2():
    primitives = FakePrimitives()
    basis = TTBasis(primitives, nested_depth=3)
    y = torch.tensor([[0.1, -0.2], [0.4, 0.3], [-0.1, 0.5]], dtype=torch.float64)

    phi = basis(y)
    factors = tuple(primitives.eval_dim(y)[:, k] for k in range(primitives.d))
    expected_phi = torch.stack([kron_all([f[b] for f in factors]) for b in range(y.shape[0])])
    assert phi.batch_size == y.shape[0]
    torch.testing.assert_close(phi.to_dense(), expected_phi)

    Omega = basis.Omega2(basis)
    expected_omega = kron_all([primitives._gram[k] for k in range(primitives.d)])
    torch.testing.assert_close(Omega.to_dense(), expected_omega)
    assert Omega.separable_cores is not None

    # Rank-one Gram MPO acts locally and does not add nested history/rank modes.
    out = Omega.matvec(phi)
    assert out.history_length == phi.history_length
    assert out.leaf_count == phi.leaf_count
    torch.testing.assert_close(
        out.to_dense(),
        torch.einsum("mn,bn->bm", expected_omega, expected_phi),
    )


def test_nested_ttbasis_omega1_is_rank_one_and_batch_compatible():
    primitives = FakePrimitives()
    basis = TTBasis(primitives, nested_depth=2)
    omega1 = basis.Omega1()
    factors = tuple(torch.exp(primitives.log_Omega1_dim())[0, k] for k in range(primitives.d))
    torch.testing.assert_close(omega1.to_dense(), kron_all(factors))
    assert omega1.rank_one_factors is not None
