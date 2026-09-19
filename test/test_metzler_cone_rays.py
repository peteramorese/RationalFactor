"""Regression test for RankDeficientGramConeRayFinder.

The fixture builds

    phi = A(z) alpha,
    psi = B(z) beta,

with a fixed rank-r target Gram

    M = A0 G B0^T.

A0 and B0 are random positive rank-r matrices and G is the exact Gram of a
positive Bernstein basis.  The test then verifies that the direct paired cone
finder produces genuinely nontrivial Metzler generator pairs (R,T) satisfying

    R M + M T^T = 0.

It additionally checks random nonnegative combinations and the finite-time
identities

    A(z) = exp(R(z)) A0 >= 0,
    B(z) = exp(T(z)) B0 >= 0,
    A(z) G B(z)^T = M.

Run standalone:

    python test_rank_deficient_gram_cone.py

or in the project after placing the finder under rational_factor/tools.
"""

from __future__ import annotations

from math import comb

import numpy as np
import torch
from scipy.linalg import expm

from rational_factor.tools.metzler_cone_rays import RankDeficientGramConeRayFinder


SEED = 7
M_DIM = 10
RANK = 3
N_RAYS = 8
N_COMBOS = 24
TOL = 2e-7


def _bernstein_gram(m: int) -> np.ndarray:
    """Exact Gram of degree-(m-1) Bernstein basis on [0,1]."""
    n = m - 1
    G = np.empty((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            G[i, j] = (
                comb(n, i)
                * comb(n, j)
                / ((2 * n + 1) * comb(2 * n, i + j))
            )
    return G


def _random_positive_rank_r(
    m: int, r: int, rng: np.random.Generator
) -> np.ndarray:
    """Random strictly-positive m x m matrix of exact rank r."""
    U = rng.lognormal(mean=0.0, sigma=0.65, size=(m, r))
    V = rng.lognormal(mean=0.0, sigma=0.65, size=(m, r))
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    V /= np.linalg.norm(V, axis=0, keepdims=True)
    A = U @ V.T
    assert np.min(A) > 0.0
    assert np.linalg.matrix_rank(A, tol=1e-10) == r
    return A


def _min_offdiag(A: np.ndarray) -> float:
    mask = ~np.eye(A.shape[0], dtype=bool)
    return float(A[mask].min())


def _pair_cosines(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    V = np.concatenate(
        [R.reshape(R.shape[0], -1), T.reshape(T.shape[0], -1)], axis=1
    )
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return np.clip(V @ V.T, -1.0, 1.0)


def _check_pair(M: np.ndarray, R: np.ndarray, T: np.ndarray, label: str) -> None:
    assert _min_offdiag(R) >= -TOL, f"{label}: R is not Metzler"
    assert _min_offdiag(T) >= -TOL, f"{label}: T is not Metzler"

    residual = np.linalg.norm(R @ M + M @ T.T) / max(np.linalg.norm(M), 1.0)
    assert residual < TOL, f"{label}: Gram generator residual={residual:.3e}"


def main() -> None:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    G = _bernstein_gram(M_DIM)
    assert np.min(G) > 0.0
    assert np.linalg.matrix_rank(G, tol=1e-12) == M_DIM

    A0 = _random_positive_rank_r(M_DIM, RANK, rng)
    B0 = _random_positive_rank_r(M_DIM, RANK, rng)
    M = A0 @ G @ B0.T

    s = np.linalg.svd(M, compute_uv=False)
    numerical_rank = np.linalg.matrix_rank(M, tol=1e-9 * s[0])
    assert numerical_rank == RANK

    print(
        f"fixture: m={M_DIM}, rank={RANK}, "
        f"sigma[:r]={np.array2string(s[:RANK], precision=3)}, "
        f"sigma[r]={s[RANK]:.3e}"
    )

    finder = RankDeficientGramConeRayFinder(
        torch.as_tensor(M, dtype=torch.float64),
        rank=RANK,
        m=M_DIM,
        support_mode="full",
        candidate_factor=24,
        seed=SEED,
        feasibility_tol=1e-9,
        rank_tol=1e-9,
    )
    rays = finder.find(N_RAYS)

    R = rays.R.to_dense().detach().cpu().numpy()
    T = rays.T.to_dense().detach().cpu().numpy()
    C = rays.C.detach().cpu().numpy()

    assert R.shape == (N_RAYS, M_DIM, M_DIM)
    assert T.shape == (N_RAYS, M_DIM, M_DIM)
    assert C.shape == (N_RAYS, RANK, RANK)

    # tr(C)=0 removes only the reciprocal scalar lineality.  C=0 rays are
    # allowed and correspond to pure Gram-null dynamics.
    assert np.max(np.abs(np.trace(C, axis1=1, axis2=2))) < 5e-7

    offdiag_energy = []
    for k in range(N_RAYS):
        _check_pair(M, R[k], T[k], f"ray[{k}]")
        Roff = R[k].copy()
        Toff = T[k].copy()
        np.fill_diagonal(Roff, 0.0)
        np.fill_diagonal(Toff, 0.0)
        offdiag_energy.append(np.linalg.norm(Roff) + np.linalg.norm(Toff))

    assert max(offdiag_energy) > 1e-3, "cone contains only diagonal directions"

    # The returned set should span several genuinely different paired directions.
    pair_flat = np.concatenate(
        [R.reshape(N_RAYS, -1), T.reshape(N_RAYS, -1)], axis=1
    )
    span_rank = int(np.linalg.matrix_rank(pair_flat, tol=1e-8))
    assert span_rank >= min(4, N_RAYS), f"paired cone span rank only {span_rank}"

    cosine = _pair_cosines(R, T)
    mask = ~np.eye(N_RAYS, dtype=bool)
    max_cosine = float(cosine[mask].max())
    assert max_cosine < 1.0 - 1e-6

    # Random nonnegative combinations remain feasible and preserve the finite
    # Gram exactly under exponentiation.
    worst_generator_residual = 0.0
    worst_finite_gram_residual = 0.0
    worst_A_min = np.inf
    worst_B_min = np.inf

    for q in range(N_COMBOS):
        coeff = rng.uniform(0.0, 0.35, size=N_RAYS)
        Rc = np.tensordot(coeff, R, axes=(0, 0))
        Tc = np.tensordot(coeff, T, axes=(0, 0))
        _check_pair(M, Rc, Tc, f"combo[{q}]")

        gen_res = np.linalg.norm(Rc @ M + M @ Tc.T) / max(np.linalg.norm(M), 1.0)
        worst_generator_residual = max(worst_generator_residual, gen_res)

        A = expm(Rc) @ A0
        B = expm(Tc) @ B0
        worst_A_min = min(worst_A_min, float(A.min()))
        worst_B_min = min(worst_B_min, float(B.min()))
        assert A.min() >= -TOL
        assert B.min() >= -TOL

        M_new = A @ G @ B.T
        gram_res = np.linalg.norm(M_new - M) / max(np.linalg.norm(M), 1.0)
        worst_finite_gram_residual = max(worst_finite_gram_residual, gram_res)
        assert gram_res < 5e-7, f"combo[{q}]: finite Gram residual={gram_res:.3e}"

    print(
        "  found nontrivial paired cone: "
        f"rays={N_RAYS}, span_rank={span_rank}, "
        f"max_cos={max_cosine:.4f}, "
        f"max_generator_residual={worst_generator_residual:.3e}, "
        f"max_finite_gram_residual={worst_finite_gram_residual:.3e}, "
        f"min(A)={worst_A_min:.3e}, min(B)={worst_B_min:.3e}"
    )
    print("All RankDeficientGramConeRayFinder checks passed.")


if __name__ == "__main__":
    main()
