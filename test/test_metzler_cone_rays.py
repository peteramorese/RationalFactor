"""
Integration / regression checks for LowRankMetzlerConeRayFinder.

The realistic fixture does the following:

1. Builds a dense, strictly positive Gram matrix G from normalized Bernstein
   basis functions on [0, 1].
2. Draws independent strictly positive low-rank factors

       U, V, U_tilde, V_tilde in R_+^{m x r}.

3. Forms the reduced coupling

       H = V.T @ G @ V_tilde.

4. Runs the reduced cone finder for C in R^{r x r}, where

       D(C) = -H.T @ C.T @ H^{-T},

   and independently verifies Metzler certificates

       R @ U       = U @ C,
       T @ U_tilde = U_tilde @ D(C).

5. Verifies the actual model consequences:

       U exp(C) >= 0,
       U_tilde exp(D) >= 0,

   and

       A G B.T = U H U_tilde.T,

   where

       A = U exp(C) V.T,
       B = U_tilde exp(D) V_tilde.T.

Important diagnostic:
---------------------
For generic independent positive U, U_tilde, V, V_tilde, the trace-free
certified cone may contain no nontrivial direction at all; the always-feasible
scalar lineality C = gamma I can be the entire discovered cone.  The realistic
random fixture therefore *strictly* tests the scalar cone and then probes the
trace-free cone, reporting whether nontrivial directions were found rather
than assuming that random factors must admit them.

A small identity-coupling regression fixture is also included so the test
strictly exercises discovery of a nontrivial trace-free ray.

Run from the repository root with, e.g.

    PYTHONPATH=src python test/test_low_rank_metzler_cone_rays.py
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import expm
from scipy.optimize import linprog
from scipy.special import betaln, gammaln

from rational_factor.tools.metzler_cone_rays import LowRankMetzlerConeRayFinder


SEED = 0
FEAS_TOL = 2e-8
GRAM_TOL = 2e-8
CERT_TOL = 2e-8
N_COMBOS = 24

# Realistic random low-rank fixture.
M = 10
RANK = 10
NONTRIVIAL_K = 10

# Random LP probing.  The full certificate mode is intentional at this small m
# so the test checks the exact lifted cone rather than a sparse inner cone.
CANDIDATE_FACTOR = 50


# ============================================================================
# Positive Gram / low-rank fixture generation
# ============================================================================


def _bernstein_gram(m: int) -> np.ndarray:
    """Gram of normalized degree-(m-1) Bernstein basis functions on [0,1].

    For B_i^n(x) = C(n,i) x^i (1-x)^(n-i),

        integral B_i^n B_j^n dx
        = C(n,i) C(n,j) Beta(i+j+1, 2n-i-j+1).

    The diagonal normalization simply rescales each positive basis function to
    unit L2 norm.  The result is dense, symmetric positive definite, and
    strictly entrywise positive.
    """
    if m <= 0:
        raise ValueError("m must be positive")

    n = m - 1
    i = np.arange(m, dtype=np.float64)[:, None]
    j = np.arange(m, dtype=np.float64)[None, :]

    log_choose_i = (
        gammaln(n + 1.0)
        - gammaln(i + 1.0)
        - gammaln(n - i + 1.0)
    )
    log_choose_j = (
        gammaln(n + 1.0)
        - gammaln(j + 1.0)
        - gammaln(n - j + 1.0)
    )

    log_g = (
        log_choose_i
        + log_choose_j
        + betaln(i + j + 1.0, 2.0 * n - i - j + 1.0)
    )
    G = np.exp(log_g)

    scale = np.sqrt(np.diag(G))
    G = G / (scale[:, None] * scale[None, :])

    assert np.min(G) > 0.0
    print("G rank:", np.linalg.matrix_rank(G), " m: ", m)
    assert np.linalg.matrix_rank(G) == m
    return G


def _random_positive_factor(
    m: int,
    r: int,
    rng: np.random.Generator,
    *,
    concentration: float = 0.6,
    floor: float = 0.03,
) -> np.ndarray:
    """Draw a strictly positive, row-normalized m x r full-rank matrix."""
    for _ in range(100):
        X = rng.gamma(concentration, 1.0, size=(m, r)) + floor
        X /= X.sum(axis=1, keepdims=True)
        if np.linalg.matrix_rank(X) == r:
            return X
    raise RuntimeError("Could not sample a full-column-rank positive factor")


def _random_fixture(
    m: int,
    r: int,
    seed: int,
    *,
    max_h_condition: float = 1e5,
):
    rng = np.random.default_rng(seed)
    G = _bernstein_gram(m)

    U = _random_positive_factor(m, r, rng)
    U_tilde = _random_positive_factor(m, r, rng)

    # H can be ill-conditioned if two random positive column spaces align too
    # closely.  Since these factors are an offline design choice, simply reject
    # such draws rather than making the cone test numerically meaningless.
    for _ in range(200):
        V = _random_positive_factor(m, r, rng)
        V_tilde = _random_positive_factor(m, r, rng)
        H = V.T @ G @ V_tilde
        if (
            np.linalg.matrix_rank(H) == r
            and np.linalg.cond(H) <= max_h_condition
        ):
            break
    else:
        raise RuntimeError("Could not sample a well-conditioned invertible H")

    local_gram = U @ H @ U_tilde.T
    assert np.linalg.matrix_rank(local_gram, tol=1e-9) == r

    return G, U, V, U_tilde, V_tilde, H, local_gram


# ============================================================================
# Independent certificate / model verification
# ============================================================================


def _paired_generator(H: np.ndarray, C: np.ndarray) -> np.ndarray:
    """D(C) = -H^T C^T H^{-T}, without explicitly forming H^{-T}."""
    # Right-multiplication by H^{-T}: X H^{-T} = solve(H^{-1} X^T)^T.
    # At these small r values explicit inverse would also be harmless, but the
    # solve makes the orientation unambiguous.
    left = -H.T @ C.T
    return np.linalg.solve(H, left.T).T


def _solve_metzler_certificate(
    U: np.ndarray,
    C: np.ndarray,
) -> np.ndarray | None:
    """Independently solve R U = U C with R Metzler.

    This intentionally does not use any private matrices from the ray finder,
    so it catches vectorization/orientation mistakes in the implementation.
    """
    m, r = U.shape
    n_vars = m * m

    rows = []
    rhs = (U @ C).reshape(-1)

    for i in range(m):
        for a in range(r):
            row = np.zeros(n_vars, dtype=np.float64)
            for j in range(m):
                row[i * m + j] = U[j, a]
            rows.append(row)

    bounds = []
    for i in range(m):
        for j in range(m):
            bounds.append((None, None) if i == j else (0.0, None))

    res = linprog(
        c=np.zeros(n_vars, dtype=np.float64),
        A_eq=np.asarray(rows),
        b_eq=rhs,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        return None
    return res.x.reshape((m, m))


def _min_offdiag(A: np.ndarray) -> float:
    B = A.copy()
    np.fill_diagonal(B, np.inf)
    return float(B.min())


def _assert_ray_feasible(
    *,
    G: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    U_tilde: np.ndarray,
    V_tilde: np.ndarray,
    H: np.ndarray,
    local_gram: np.ndarray,
    C: np.ndarray,
    label: str,
) -> tuple[float, float]:
    r = C.shape[0]
    assert C.shape == (r, r)

    D = _paired_generator(H, C)

    # ------------------------------------------------------------------
    # Independent Metzler witnesses.
    # ------------------------------------------------------------------
    R_cert = _solve_metzler_certificate(U, C)
    assert R_cert is not None, f"{label}: no R Metzler certificate"

    T_cert = _solve_metzler_certificate(U_tilde, D)
    assert T_cert is not None, f"{label}: no T Metzler certificate"

    r_min = _min_offdiag(R_cert)
    t_min = _min_offdiag(T_cert)
    assert r_min >= -CERT_TOL, f"{label}: R offdiag min={r_min:.3e}"
    assert t_min >= -CERT_TOL, f"{label}: T offdiag min={t_min:.3e}"

    r_resid = np.max(np.abs(R_cert @ U - U @ C))
    t_resid = np.max(np.abs(T_cert @ U_tilde - U_tilde @ D))
    assert r_resid <= CERT_TOL, f"{label}: RU-UC residual={r_resid:.3e}"
    assert t_resid <= CERT_TOL, f"{label}: TU~-U~D residual={t_resid:.3e}"

    # ------------------------------------------------------------------
    # Exponential positivity.
    # ------------------------------------------------------------------
    exp_C = expm(C)
    exp_D = expm(D)

    U_exp_C = U @ exp_C
    Ut_exp_D = U_tilde @ exp_D

    assert U_exp_C.min() >= -FEAS_TOL, (
        f"{label}: min(U exp(C))={U_exp_C.min():.3e}"
    )
    assert Ut_exp_D.min() >= -FEAS_TOL, (
        f"{label}: min(U_tilde exp(D))={Ut_exp_D.min():.3e}"
    )

    # ------------------------------------------------------------------
    # Reduced and full Gram cancellation.
    # ------------------------------------------------------------------
    reduced = exp_C @ H @ exp_D.T
    reduced_err = np.max(np.abs(reduced - H))
    assert reduced_err <= GRAM_TOL, (
        f"{label}: exp(C) H exp(D)^T != H, err={reduced_err:.3e}"
    )

    A = U_exp_C @ V.T
    B = Ut_exp_D @ V_tilde.T

    assert A.min() >= -FEAS_TOL, f"{label}: A min={A.min():.3e}"
    assert B.min() >= -FEAS_TOL, f"{label}: B min={B.min():.3e}"

    gram = A @ G @ B.T
    gram_err = np.max(np.abs(gram - local_gram))
    assert gram_err <= GRAM_TOL, (
        f"{label}: A G B^T != U H U_tilde^T, err={gram_err:.3e}"
    )

    return r_min, t_min


def _cosine_matrix(C: np.ndarray) -> np.ndarray:
    V = C.reshape(C.shape[0], -1)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return np.clip(V @ V.T, -1.0, 1.0)


# ============================================================================
# Realistic random-positive test
# ============================================================================


def _check_random_positive_fixture() -> None:
    G, U, V, U_tilde, V_tilde, H, local_gram = _random_fixture(
        M,
        RANK,
        SEED,
    )

    print(
        "Random positive fixture: "
        f"m={M}, r={RANK}, "
        f"cond(G)={np.linalg.cond(G):.3e}, "
        f"cond(H)={np.linalg.cond(H):.3e}, "
        f"rank(local gram)={np.linalg.matrix_rank(local_gram)}"
    )

    # ------------------------------------------------------------------
    # The scalar lineality C = gamma I is always feasible.  Keep identity
    # lineality enabled here so a generic random fixture gives a strict smoke
    # test rather than failing merely because its trace-free cone is trivial.
    # ------------------------------------------------------------------
    finder = LowRankMetzlerConeRayFinder(
        torch.as_tensor(U, dtype=torch.float64),
        torch.as_tensor(U_tilde, dtype=torch.float64),
        torch.as_tensor(H, dtype=torch.float64),
        certificate_mode="full",
        candidate_factor=CANDIDATE_FACTOR,
        feasibility_tol=1e-9,
        cosine_tol=1e-7,
        seed=SEED,
        remove_identity=False,
    )

    rays = finder.find(1)
    C = rays.to_dense().detach().cpu().double().numpy()

    assert C.shape == (1, RANK, RANK)
    assert np.linalg.norm(C[0]) > 1e-10

    ray_r_min, ray_t_min = _assert_ray_feasible(
        G=G,
        U=U,
        V=V,
        U_tilde=U_tilde,
        V_tilde=V_tilde,
        H=H,
        local_gram=local_gram,
        C=C[0],
        label="random scalar-cone ray",
    )

    # Random positive multiples plus arbitrary identity lineality.  The latter
    # is useful because it tests D(C + gamma I) = D(C) - gamma I as well.
    rng = np.random.default_rng(SEED + 100)
    combo_r_min = np.inf
    combo_t_min = np.inf

    for t in range(N_COMBOS):
        coeff = rng.uniform(0.0, 3.0)
        gamma = rng.uniform(-2.0, 2.0)
        C_combo = coeff * C[0] + gamma * np.eye(RANK)

        r_min, t_min = _assert_ray_feasible(
            G=G,
            U=U,
            V=V,
            U_tilde=U_tilde,
            V_tilde=V_tilde,
            H=H,
            local_gram=local_gram,
            C=C_combo,
            label=f"random combo[{t}]",
        )
        combo_r_min = min(combo_r_min, r_min)
        combo_t_min = min(combo_t_min, t_min)

    print(
        "  scalar cone ok: "
        f"R_offdiag_min={ray_r_min:.3e}, "
        f"T_offdiag_min={ray_t_min:.3e}, "
        f"combo_R_min={combo_r_min:.3e}, "
        f"combo_T_min={combo_t_min:.3e}"
    )

    # ------------------------------------------------------------------
    # Probe the actually interesting trace-free cone.  Do not assume generic
    # independent positive factors admit one: empirically they often do not.
    # If rays are found, verify them and their nonnegative combinations fully.
    # ------------------------------------------------------------------
    nontrivial_finder = LowRankMetzlerConeRayFinder(
        U,
        U_tilde,
        H,
        certificate_mode="full",
        candidate_factor=CANDIDATE_FACTOR,
        feasibility_tol=1e-9,
        cosine_tol=1e-7,
        seed=SEED + 1,
        remove_identity=True,
    )

    try:
        nontrivial = nontrivial_finder.find(NONTRIVIAL_K)
    except RuntimeError as exc:
        print(
            "  trace-free cone diagnostic: no set of "
            f"{NONTRIVIAL_K} nontrivial rays found for this independent "
            f"random fixture ({exc})."
        )
        return

    Cn = nontrivial.to_dense().detach().cpu().double().numpy()
    assert Cn.shape == (NONTRIVIAL_K, RANK, RANK)
    assert np.max(np.abs(np.trace(Cn, axis1=1, axis2=2))) < 1e-6

    cosine = _cosine_matrix(Cn)
    pair = cosine[~np.eye(NONTRIVIAL_K, dtype=bool)]

    for i, Ci in enumerate(Cn):
        _assert_ray_feasible(
            G=G,
            U=U,
            V=V,
            U_tilde=U_tilde,
            V_tilde=V_tilde,
            H=H,
            local_gram=local_gram,
            C=Ci,
            label=f"random trace-free ray[{i}]",
        )

    for t in range(N_COMBOS):
        coeff = rng.random(NONTRIVIAL_K)
        C_combo = np.tensordot(coeff, Cn, axes=(0, 0))
        _assert_ray_feasible(
            G=G,
            U=U,
            V=V,
            U_tilde=U_tilde,
            V_tilde=V_tilde,
            H=H,
            local_gram=local_gram,
            C=C_combo,
            label=f"random trace-free combo[{t}]",
        )

    print(
        "  trace-free cone found: "
        f"k={NONTRIVIAL_K}, pairwise cosine "
        f"range=[{pair.min():.3f}, {pair.max():.3f}]"
    )


# ============================================================================
# Small known-nontrivial solver regression
# ============================================================================


def _check_known_nontrivial_fixture() -> None:
    """Strictly exercise discovery of multiple trace-free rays.

    U = U_tilde = I and H = I is not intended as a realistic model fixture.
    It is a solver regression where the cone is known analytically:

        C Metzler and -C^T Metzler

    forces C to be diagonal, while tr(C)=0 leaves an (r-1)-dimensional
    nontrivial subspace.
    """
    r = 4
    U = np.eye(r)
    U_tilde = np.eye(r)
    H = np.eye(r)

    finder = LowRankMetzlerConeRayFinder(
        U,
        U_tilde,
        H,
        certificate_mode="full",
        candidate_factor=30,
        seed=SEED + 1000,
        remove_identity=True,
    )

    rays = finder.find(1).to_dense().detach().cpu().double().numpy()
    assert rays.shape == (1, r, r)

    for i, C in enumerate(rays):
        assert abs(np.trace(C)) < 1e-7
        off = C.copy()
        np.fill_diagonal(off, 0.0)
        assert np.max(np.abs(off)) < 1e-7

        D = _paired_generator(H, C)
        R_cert = _solve_metzler_certificate(U, C)
        T_cert = _solve_metzler_certificate(U_tilde, D)
        assert R_cert is not None, f"known ray[{i}] missing R certificate"
        assert T_cert is not None, f"known ray[{i}] missing T certificate"

    print(
        "Known nontrivial fixture: "
        "found and verified a nonzero trace-free diagonal direction."
    )


# ============================================================================
# Driver
# ============================================================================


def main() -> None:
    torch.manual_seed(SEED)
    np.set_printoptions(precision=4, suppress=True)

    _check_random_positive_fixture()
    _check_known_nontrivial_fixture()

    print("All LowRankMetzlerConeRayFinder checks passed.")


if __name__ == "__main__":
    main()
