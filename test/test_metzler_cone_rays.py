"""
Checks that MetzlerConeRayFinder returns feasible, diverse Metzler directions
for diagonal, dense, and banded ray types.

For every returned matrix ray X_i and random nonnegative combination

    X = sum_i c_i X_i,   c_i >= 0,

verifies

    X is Metzler,
    M X^T M^{-1} is Metzler,

where M = gram.T if transpose=True, otherwise gram.

Also checks:
  - correct structured return type / batch shape,
  - removal of the scalar-identity lineality direction,
  - cosine diversity of returned rays,
  - requested support structure for diagonal / banded rays,
  - feasibility after adding gamma * I.

The diagonal algorithm is tested at large m. Dense and banded rays use an exact
LP with O(m^2) auxiliary variables, so they are intentionally tested at much
smaller m.

Run:
  PYTHONPATH=src python test/test_metzler_cone_rays.py
"""

from __future__ import annotations

import numpy as np
import torch

from rational_factor.models.structured_matrices import (
    Banded,
    DenseMatrix,
    Diagonal,
)
from rational_factor.tools.metzler_cone_rays import MetzlerConeRayFinder


SEED = 0
FEAS_TOL = 1e-6
N_COMBOS = 32

# Sampled diagonal algorithm can still be tested large.
DIAG_M = 100
DIAG_K = 20

# Exact dense/banded LPs scale quadratically in matrix dimension.
EXACT_M = 12
EXACT_K = 4
BANDED_BW = 2

# Returned rays should not contain near-duplicates.
MAX_COSINE = 1.0 - 1e-6


# ======================================================================
# Gram fixtures
# ======================================================================

def _eye_gram(n: int) -> DenseMatrix:
    return DenseMatrix(torch.eye(n, dtype=torch.float64))


def _lower_bidiagonal(n: int, sub: float = -0.35) -> DenseMatrix:
    M = torch.eye(n, dtype=torch.float64)
    idx = torch.arange(1, n)
    M[idx, idx - 1] = sub
    return DenseMatrix(M)


def _banded_bidiagonal(n: int, sub: float = -0.4) -> Banded:
    # Banded convention:
    # data[r, j] = A[j + offset[r], j].
    #
    # Therefore offset +1 is the lower/subdiagonal.
    data = torch.zeros(2, n, dtype=torch.float64)
    data[0] = 1.0
    data[1, :-1] = sub

    return Banded(
        torch.tensor([0, 1], dtype=torch.long),
        data,
    )


def _lower_triangular(n: int, fill: float = -0.05) -> DenseMatrix:
    M = torch.eye(n, dtype=torch.float64)
    i, j = torch.tril_indices(n, n, offset=-1)
    M[i, j] = fill
    return DenseMatrix(M)


# ======================================================================
# Verification helpers
# ======================================================================

def _min_offdiag(A: np.ndarray) -> float:
    off = A.copy()
    np.fill_diagonal(off, np.inf)
    return float(off.min())


def _transform(M: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Q = M X^T M^{-1}, computed without explicitly forming M^{-1}.
    """
    A = M @ X.T

    # Q M = A  ->  M^T Q^T = A^T
    return np.linalg.solve(M.T, A.T).T


def _assert_metzler(
    A: np.ndarray,
    *,
    label: str,
    tol: float = FEAS_TOL,
) -> float:
    vmin = _min_offdiag(A)

    assert vmin >= -tol, (
        f"{label}: min off-diag={vmin:.3e} < -{tol}"
    )

    return vmin


def _assert_feasible(
    M: np.ndarray,
    X: np.ndarray,
    *,
    label: str,
) -> tuple[float, float]:
    xmin = _assert_metzler(
        X,
        label=f"{label}: X",
    )

    Y = _transform(M, X)

    ymin = _assert_metzler(
        Y,
        label=f"{label}: M X^T M^-1",
    )

    # Verify the similarity equation independently.
    residual = np.max(np.abs(Y @ M - M @ X.T))

    assert residual < 1e-7, (
        f"{label}: Y M != M X^T, "
        f"max residual={residual:.3e}"
    )

    return xmin, ymin


def _cosine_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise Frobenius cosine similarity of (k,m,m) rays."""
    V = X.reshape(X.shape[0], -1)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return np.clip(V @ V.T, -1.0, 1.0)


# ======================================================================
# Main generalized checker
# ======================================================================

def _check_rays(
    gram: DenseMatrix | Banded,
    *,
    ray_type: str,
    k: int,
    transpose: bool = True,
    bandwidth: int | None = None,
    seed: int = SEED,
) -> None:
    m = gram.shape[-1]

    finder = MetzlerConeRayFinder(
        gram,
        transpose=transpose,

        # For the sampled diagonal method, cover every column.
        n_constraint_cols=m,
        n_verify_cols=m,
        candidate_factor=10,
        cut_rounds=4,
        max_new_cols=m,

        feasibility_tol=FEAS_TOL,
        cosine_tol=1e-7,
        seed=seed,
    )

    kwargs = {}
    if ray_type == "banded":
        kwargs["bandwidth"] = bandwidth

    rays = finder.find(
        k,
        ray_type=ray_type,
        **kwargs,
    )

    # ------------------------------------------------------------------
    # Structured return type
    # ------------------------------------------------------------------

    expected_type = {
        "diagonal": Diagonal,
        "dense": DenseMatrix,
        "banded": Banded,
    }[ray_type]

    assert isinstance(rays, expected_type), (
        f"{ray_type}: expected {expected_type.__name__}, "
        f"got {type(rays).__name__}"
    )

    assert rays.shape == torch.Size((k, m, m)), (
        f"{ray_type}: expected {(k, m, m)}, got {tuple(rays.shape)}"
    )

    if ray_type == "diagonal":
        assert rays.d.shape == (k, m)

    elif ray_type == "banded":
        assert bandwidth is not None
        assert rays.data.shape == (
            k,
            2 * bandwidth + 1,
            m,
        )

        expected_offsets = torch.arange(
            -bandwidth,
            bandwidth + 1,
            device=rays.offsets.device,
        )

        assert torch.equal(rays.offsets, expected_offsets)

    # Dense only for verification.
    X = rays.to_dense().detach().cpu().double().numpy()

    # ------------------------------------------------------------------
    # Nonzero / lineality
    # ------------------------------------------------------------------

    norms = np.linalg.norm(X.reshape(k, -1), axis=1)

    assert np.all(norms > 1e-10), (
        f"{ray_type}: zero ray found: norms={norms}"
    )

    # Scalar identity lineality has been removed.
    traces = np.trace(X, axis1=-2, axis2=-1)

    assert np.all(np.abs(traces) < 1e-6), (
        f"{ray_type}: rays not trace-zero: "
        f"max |trace|={np.abs(traces).max():.3e}"
    )

    # ------------------------------------------------------------------
    # Representation-specific structure
    # ------------------------------------------------------------------

    if ray_type == "diagonal":
        off = X.copy()

        for i in range(k):
            np.fill_diagonal(off[i], 0.0)

        assert np.max(np.abs(off)) < 1e-12, (
            "diagonal ray contains non-diagonal entries"
        )

    elif ray_type == "banded":
        assert bandwidth is not None

        row = np.arange(m)[:, None]
        col = np.arange(m)[None, :]
        outside = np.abs(row - col) > bandwidth

        assert np.max(np.abs(X[:, outside])) < 1e-12, (
            f"banded rays contain entries outside bandwidth={bandwidth}"
        )

    # ------------------------------------------------------------------
    # M used by the finder
    # ------------------------------------------------------------------

    M_torch = (
        gram.T.to_dense()
        if transpose
        else gram.to_dense()
    )

    M = M_torch.detach().cpu().double().numpy()

    # ------------------------------------------------------------------
    # Individual rays
    # ------------------------------------------------------------------

    ray_x_mins = []
    ray_y_mins = []

    for i in range(k):
        xmin, ymin = _assert_feasible(
            M,
            X[i],
            label=f"{ray_type} ray[{i}]",
        )

        ray_x_mins.append(xmin)
        ray_y_mins.append(ymin)

    # ------------------------------------------------------------------
    # Nonnegative combinations
    # ------------------------------------------------------------------

    rng = np.random.default_rng(seed + 17)

    combo_x_mins = []
    combo_y_mins = []

    for t in range(N_COMBOS):
        if t < k:
            # Single ray.
            c = np.zeros(k)
            c[t] = 1.0

        elif t < 2 * k:
            # Sparse two-ray combination.
            c = np.zeros(k)

            i = t - k
            j = (i + 1) % k

            c[i] = rng.uniform(0.1, 2.0)
            c[j] = rng.uniform(0.1, 2.0)

        else:
            # Dense positive combination.
            c = rng.random(k)

        Xc = np.tensordot(c, X, axes=(0, 0))

        xmin, ymin = _assert_feasible(
            M,
            Xc,
            label=f"{ray_type} combo[{t}] c={np.round(c, 3)}",
        )

        combo_x_mins.append(xmin)
        combo_y_mins.append(ymin)

    # ------------------------------------------------------------------
    # Identity lineality direction
    # ------------------------------------------------------------------

    I = np.eye(m)

    for t in range(8):
        c = rng.random(k)

        # Lineality really is unrestricted, so test both signs.
        gamma = rng.uniform(-2.0, 2.0)

        Xc = (
            np.tensordot(c, X, axes=(0, 0))
            + gamma * I
        )

        _assert_feasible(
            M,
            Xc,
            label=f"{ray_type} combo+I[{t}]",
        )

    # ------------------------------------------------------------------
    # Cosine diversity
    # ------------------------------------------------------------------

    cosine = _cosine_matrix(X)

    offdiag_mask = ~np.eye(k, dtype=bool)
    pair_cosines = cosine[offdiag_mask]

    max_cos = float(pair_cosines.max())
    min_cos = float(pair_cosines.min())

    assert max_cos < MAX_COSINE, (
        f"{ray_type}: near-duplicate rays: "
        f"max pairwise cosine={max_cos:.9f}"
    )

    # SVD/rank are still useful diagnostics, but cosine selection does
    # not mathematically guarantee linear independence.
    V = X.reshape(k, -1).T
    s = np.linalg.svd(V, compute_uv=False)
    rank = int(np.linalg.matrix_rank(V, tol=1e-8))

    print(
        f"  ok  type={ray_type:<8} "
        f"shape={tuple(rays.shape)}  "
        f"X_min={min(ray_x_mins):.3e}  "
        f"Y_min={min(ray_y_mins):.3e}  "
        f"combo_X_min={min(combo_x_mins):.3e}  "
        f"combo_Y_min={min(combo_y_mins):.3e}  "
        f"cos=[{min_cos:.3f}, {max_cos:.3f}]  "
        f"rank={rank}/{k}  "
        f"sv={np.array2string(s, precision=3)}"
    )


# ======================================================================
# API checks
# ======================================================================

def _check_banded_api() -> None:
    gram = _eye_gram(6)
    finder = MetzlerConeRayFinder(gram)

    try:
        finder.find(2, ray_type="banded")
    except ValueError as exc:
        assert "bandwidth is required" in str(exc)
    else:
        raise AssertionError(
            "banded ray type should require bandwidth"
        )

    for bw in (-1, 6):
        try:
            finder.find(
                2,
                ray_type="banded",
                bandwidth=bw,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"invalid bandwidth={bw} should raise ValueError"
            )


# ======================================================================
# Test driver
# ======================================================================

def main() -> None:
    torch.manual_seed(SEED)

    # ------------------------------------------------------------------
    # Diagonal: preserve the old large-m test coverage.
    # ------------------------------------------------------------------

    print(
        f"Diagonal rays / identity, "
        f"m={DIAG_M}, k={DIAG_K}"
    )
    _check_rays(
        _eye_gram(DIAG_M),
        ray_type="diagonal",
        k=DIAG_K,
    )

    print(
        f"Diagonal rays / dense lower bidiagonal, "
        f"m={DIAG_M}, k={DIAG_K}"
    )
    G = _lower_bidiagonal(DIAG_M)

    _check_rays(
        G,
        ray_type="diagonal",
        k=DIAG_K,
        transpose=True,
    )

    _check_rays(
        G,
        ray_type="diagonal",
        k=DIAG_K,
        transpose=False,
        seed=SEED + 1,
    )

    print(
        f"Diagonal rays / banded lower bidiagonal, "
        f"m={DIAG_M}, k={DIAG_K}"
    )
    _check_rays(
        _banded_bidiagonal(DIAG_M),
        ray_type="diagonal",
        k=DIAG_K,
        transpose=True,
        seed=SEED + 2,
    )

    # ------------------------------------------------------------------
    # Dense exact LP.
    # ------------------------------------------------------------------

    print(
        f"Dense rays / identity, "
        f"m={EXACT_M}, k={EXACT_K}"
    )
    _check_rays(
        _eye_gram(EXACT_M),
        ray_type="dense",
        k=EXACT_K,
        seed=SEED + 10,
    )

    print(
        f"Dense rays / lower bidiagonal, "
        f"m={EXACT_M}, k={EXACT_K}"
    )
    G = _lower_bidiagonal(EXACT_M)

    _check_rays(
        G,
        ray_type="dense",
        k=EXACT_K,
        transpose=True,
        seed=SEED + 11,
    )

    _check_rays(
        G,
        ray_type="dense",
        k=EXACT_K,
        transpose=False,
        seed=SEED + 12,
    )

    # ------------------------------------------------------------------
    # Banded exact LP.
    # ------------------------------------------------------------------

    print(
        f"Banded rays / identity, "
        f"m={EXACT_M}, k={EXACT_K}, bw={BANDED_BW}"
    )
    _check_rays(
        _eye_gram(EXACT_M),
        ray_type="banded",
        bandwidth=BANDED_BW,
        k=EXACT_K,
        seed=SEED + 20,
    )

    print(
        f"Banded rays / dense lower bidiagonal, "
        f"m={EXACT_M}, k={EXACT_K}, bw={BANDED_BW}"
    )
    _check_rays(
        _lower_bidiagonal(EXACT_M),
        ray_type="banded",
        bandwidth=BANDED_BW,
        k=EXACT_K,
        transpose=True,
        seed=SEED + 21,
    )

    print(
        f"Banded rays / banded lower bidiagonal, "
        f"m={EXACT_M}, k={EXACT_K}, bw={BANDED_BW}"
    )
    _check_rays(
        _banded_bidiagonal(EXACT_M),
        ray_type="banded",
        bandwidth=BANDED_BW,
        k=EXACT_K,
        transpose=True,
        seed=SEED + 22,
    )

    _check_banded_api()

    print("All MetzlerConeRayFinder checks passed.")


if __name__ == "__main__":
    main()