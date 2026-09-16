"""
Checks that metzler_cone_rays returns feasible, diverse Metzler directions.

For each returned ray (and random nonnegative combinations), verifies that
M diag(v) M^{-1} is Metzler (nonnegative off-diagonals). Diversity is checked
via the SVD of the returned (m, k) ray matrix.

Uses matrices with a nontrivial Metzler cone (diagonal, bidiagonal, triangular).
Generic dense SPD Grams typically only admit the all-ones lineality direction
once 1^T r = 0 is imposed, so they are not used as the primary fixtures.

Run:
  PYTHONPATH=src python test/test_metzler_cone_rays.py
"""

from __future__ import annotations

import numpy as np
import torch

from rational_factor.models.structured_matrices import Banded, DenseMatrix
from rational_factor.tools.metzler_cone_rays import metzler_cone_rays


SEED = 0
M_SIZE = 1000
K = 20
FEAS_TOL = 1e-6
# Algorithm uses sampled constraints; for small m cover all columns.
N_COLS = M_SIZE
N_COMBOS = 32
MIN_SV_RATIO = 1e-2


def _eye_gram(n: int) -> DenseMatrix:
    return DenseMatrix(torch.eye(n, dtype=torch.float64))


def _lower_bidiagonal(n: int, sub: float = -0.35) -> DenseMatrix:
    M = torch.eye(n, dtype=torch.float64)
    for i in range(1, n):
        M[i, i - 1] = sub
    return DenseMatrix(M)


def _banded_bidiagonal(n: int, sub: float = -0.4) -> Banded:
    data = torch.zeros(2, n, dtype=torch.float64)
    data[0] = 1.0
    data[1, 1:] = sub
    return Banded(torch.tensor([0, -1], dtype=torch.long), data)


def _lower_triangular(n: int, fill: float = -0.05) -> DenseMatrix:
    M = torch.eye(n, dtype=torch.float64)
    for i in range(n):
        for j in range(i):
            M[i, j] = fill
    return DenseMatrix(M)


def _similarity(M: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Q = M diag(r) M^{-1}."""
    return M @ (r[:, None] * np.linalg.solve(M, np.eye(M.shape[0])))


def _min_offdiag(Q: np.ndarray) -> float:
    off = Q.copy()
    np.fill_diagonal(off, np.inf)
    return float(np.min(off))


def _assert_metzler(
    M: np.ndarray,
    v: np.ndarray,
    *,
    label: str,
    tol: float = FEAS_TOL,
) -> float:
    vmin = _min_offdiag(_similarity(M, v))
    assert vmin >= -tol, f"{label}: min off-diag={vmin:.3e} < -{tol}"
    return vmin


def _check_rays(
    gram: DenseMatrix | Banded,
    *,
    k: int = K,
    transpose: bool = True,
    seed: int = SEED,
) -> None:
    R_mat = metzler_cone_rays(
        gram,
        k,
        transpose=transpose,
        n_constraint_cols=N_COLS,
        n_verify_cols=N_COLS,
        candidate_factor=8,
        cut_rounds=4,
        max_new_cols=N_COLS,
        feasibility_tol=FEAS_TOL,
        seed=seed,
    )
    R = R_mat.to_dense().detach().cpu().double().numpy()
    assert R.shape == (gram.shape[-1], k), f"expected {(gram.shape[-1], k)}, got {R.shape}"

    M_torch = gram.T.to_dense() if transpose else gram.to_dense()
    M = M_torch.detach().cpu().double().numpy()

    # Lineality removed: each ray should be (nearly) mean-zero.
    means = np.abs(R.mean(axis=0))
    assert np.all(means < 1e-6), f"rays not mean-zero: max |mean|={means.max():.3e}"

    # Individual rays.
    ray_mins = []
    for i in range(k):
        ray_mins.append(_assert_metzler(M, R[:, i], label=f"ray[{i}]"))

    # Nonnegative combinations (including sparse and dense coeffs).
    rng = np.random.default_rng(seed + 17)
    combo_mins = []
    for t in range(N_COMBOS):
        if t < k:
            c = np.zeros(k)
            c[t] = 1.0
        elif t < 2 * k:
            c = np.zeros(k)
            c[t - k] = rng.uniform(0.1, 2.0)
            c[(t - k + 1) % k] = rng.uniform(0.1, 2.0)
        else:
            c = rng.random(k)
        v = R @ c
        combo_mins.append(_assert_metzler(M, v, label=f"combo[{t}] c={np.round(c, 3)}"))

    # Nonnegative mix with the all-ones lineality direction.
    ones = np.ones(M.shape[0])
    for t in range(8):
        c = rng.random(k)
        gamma = rng.uniform(0.0, 2.0)
        v = R @ c + gamma * ones
        _assert_metzler(M, v, label=f"combo+ones[{t}]")

    # Diversity via SVD of R (m x k).
    s = np.linalg.svd(R, compute_uv=False)
    assert len(s) == k
    assert np.all(s > 0), f"singular values not all positive: {s}"
    sv_ratio = float(s.min() / s.max())
    assert sv_ratio >= MIN_SV_RATIO, (
        f"rays not diverse enough: singular values={s}, "
        f"min/max={sv_ratio:.3e} < {MIN_SV_RATIO}"
    )

    rank = int(np.linalg.matrix_rank(R, tol=1e-8))
    assert rank == k, f"expected rank {k}, got {rank}"

    print(
        f"  ok  shape={R.shape}  ray_min_off={min(ray_mins):.3e}  "
        f"combo_min_off={min(combo_mins):.3e}  "
        f"svd={np.array2string(s, precision=3)}  sv_ratio={sv_ratio:.3e}"
    )


def main() -> None:
    torch.manual_seed(SEED)

    print(f"Identity, m={M_SIZE}, k={K}")
    _check_rays(_eye_gram(M_SIZE), transpose=True)

    print(f"Dense lower bidiagonal, m={M_SIZE}, k={K}")
    G = _lower_bidiagonal(M_SIZE)
    _check_rays(G, transpose=True)
    _check_rays(G, transpose=False, seed=SEED + 1)

    print(f"Banded lower bidiagonal, m={M_SIZE}, k={K}")
    _check_rays(_banded_bidiagonal(M_SIZE), transpose=True, seed=SEED + 2)

    print(f"Dense lower triangular, m={M_SIZE}, k={K}")
    _check_rays(_lower_triangular(M_SIZE), transpose=True, seed=SEED + 3)

    print("All metzler_cone_rays checks passed.")


if __name__ == "__main__":
    main()
