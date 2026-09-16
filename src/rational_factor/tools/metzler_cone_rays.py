from __future__ import annotations

import numpy as np
import torch
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import linprog

from rational_factor.models.structured_matrices import Banded, DenseMatrix, Matrix


def metzler_cone_rays(
    gram: Matrix,
    k: int,
    *,
    transpose: bool = True,
    n_constraint_cols: int = 32,
    n_verify_cols: int = 128,
    candidate_factor: int = 6,
    cut_rounds: int = 3,
    max_new_cols: int = 16,
    feasibility_tol: float = 1e-8,
    independence_tol: float = 1e-5,
    seed: int = 0,
) -> Matrix:
    """
    Find k diverse feasible directions r satisfying approximately

        M diag(r) M^{-1}  is Metzler,

    where M = gram.T if transpose=True, else gram.

    Returns:
        DenseMatrix R of shape (m, k), so v(z) = R @ c(z), c(z) >= 0.

    Notes:
        - The free lineality direction 1 is removed; add gamma(z) * 1 separately.
        - For Banded input and fixed sample counts, storage is O(m).
        - Feasibility is enforced on sampled/cutting-plane columns, not all m^2
          inequalities. Increase n_verify_cols / cut_rounds for a tighter cone.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    rng = np.random.default_rng(seed)

    M = gram.T if transpose else gram
    n = M.shape[-1]

    if M.shape[-2] != n:
        raise ValueError("gram must be square")
    if len(M.shape) != 2:
        raise ValueError("batched matrices are not supported by this offline routine")
    if k >= n:
        raise ValueError("use k < m; the all-ones lineality direction is handled separately")

    # ------------------------------------------------------------------
    # Convert M to a SciPy sparse matrix without densifying Banded.
    # ------------------------------------------------------------------
    if isinstance(M, Banded):
        rows, cols, vals = [], [], []

        offsets = M.offsets.detach().cpu().numpy()
        data = M.data.detach().cpu().numpy()

        for d, off in enumerate(offsets):
            off = int(off)

            if off >= 0:
                c = np.arange(0, n - off)
            else:
                c = np.arange(-off, n)

            r = c + off

            rows.append(r)
            cols.append(c)
            vals.append(data[d, c])

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)

        A = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))

    elif isinstance(M, DenseMatrix):
        A = sp.csc_matrix(
            M.to_dense().detach().cpu().double().numpy()
        )

    else:
        # A generic Matrix has matvec but no solve(), so there is no
        # structure-preserving way to apply M^{-1}.
        A = sp.csc_matrix(
            M.to_dense().detach().cpu().double().numpy()
        )

    A = A.astype(np.float64)

    # One sparse LU factorization reused for every inverse-column query.
    lu = spla.splu(A)

    # ------------------------------------------------------------------
    # G^{-1} columns needed for selected constraints.
    # ------------------------------------------------------------------
    def inverse_columns(js: np.ndarray) -> np.ndarray:
        E = np.zeros((n, len(js)), dtype=np.float64)
        E[js, np.arange(len(js))] = 1.0
        return lu.solve(E)  # shape (n, len(js))

    # ------------------------------------------------------------------
    # Build H such that H @ r >= 0 represents all off-diagonal entries
    # from the selected columns of M diag(r) M^{-1}.
    #
    # For column j:
    #   x_j = M^{-1} e_j
    #   Q[:, j] = M diag(x_j) r
    # so H_j = M diag(x_j).
    # ------------------------------------------------------------------
    def build_constraints(js: np.ndarray) -> sp.csr_matrix:
        X = inverse_columns(js)
        blocks = []

        for t, j in enumerate(js):
            H = A @ sp.diags(X[:, t], format="csc")

            # Remove the diagonal constraint Q[j, j].
            if j == 0:
                H = H[1:, :]
            elif j == n - 1:
                H = H[:-1, :]
            else:
                H = sp.vstack(
                    [H[:j, :], H[j + 1:, :]],
                    format="csr",
                )

            blocks.append(H)

        return sp.vstack(blocks, format="csr")

    # ------------------------------------------------------------------
    # Random LP probes.
    #
    # The cone contains span{1}, so impose 1^T r = 0 and recover the
    # unrestricted gamma(z) * 1 term separately.
    #
    # Box bounds make the homogeneous LP bounded.
    # ------------------------------------------------------------------
    def solve_candidates(
        H: sp.csr_matrix,
        count: int,
    ) -> list[np.ndarray]:
        rays = []

        A_ub = -H
        b_ub = np.zeros(H.shape[0])

        A_eq = sp.csr_matrix(np.ones((1, n)))
        b_eq = np.zeros(1)

        for _ in range(count):
            q = rng.standard_normal(n)
            q -= q.mean()

            res = linprog(
                c=-q,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=[(-1.0, 1.0)] * n,
                method="highs",
            )

            if not res.success:
                continue

            r = res.x
            norm = np.linalg.norm(r)

            if norm > 1e-10:
                rays.append(r / norm)

        return rays

    # ------------------------------------------------------------------
    # Evaluate candidate rays on additional columns and return:
    #   minimum off-diagonal value,
    #   worst offending column.
    # ------------------------------------------------------------------
    def verify(
        rays: list[np.ndarray],
        js: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not rays:
            return np.empty(0), np.empty(0, dtype=int)

        X = inverse_columns(js)

        mins = np.full(len(rays), np.inf)
        worst_cols = np.full(len(rays), -1, dtype=int)

        for r_idx, r in enumerate(rays):
            # For all sampled columns simultaneously:
            #
            #   Q[:, js] = A @ (diag(r) X)
            Y = A @ (r[:, None] * X)

            for t, j in enumerate(js):
                col = Y[:, t]

                # Exclude diagonal element.
                if j == 0:
                    vmin = col[1:].min()
                elif j == n - 1:
                    vmin = col[:-1].min()
                else:
                    vmin = min(col[:j].min(), col[j + 1:].min())

                if vmin < mins[r_idx]:
                    mins[r_idx] = vmin
                    worst_cols[r_idx] = j

        return mins, worst_cols

    # ------------------------------------------------------------------
    # Initial constraint columns: random + boundaries.
    # ------------------------------------------------------------------
    n0 = min(n, max(2, n_constraint_cols))

    constraint_cols = set(
        rng.choice(n, size=n0, replace=False).tolist()
    )
    constraint_cols.add(0)
    constraint_cols.add(n - 1)

    candidate_count = max(candidate_factor * k, k + 4)

    rays: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Cutting-plane rounds.
    # ------------------------------------------------------------------
    for _ in range(cut_rounds):
        cols = np.array(sorted(constraint_cols), dtype=int)
        H = build_constraints(cols)

        rays = solve_candidates(H, candidate_count)

        if not rays:
            continue

        remaining = np.array(
            [j for j in range(n) if j not in constraint_cols],
            dtype=int,
        )

        if len(remaining) == 0:
            break

        nv = min(n_verify_cols, len(remaining))
        verify_cols = rng.choice(remaining, size=nv, replace=False)

        mins, worst = verify(rays, verify_cols)

        bad = np.where(mins < -feasibility_tol)[0]

        if len(bad) == 0:
            break

        # Add the most frequently / severely violated columns.
        order = bad[np.argsort(mins[bad])]
        new_cols = []

        for idx in order:
            j = int(worst[idx])
            if j >= 0 and j not in constraint_cols:
                new_cols.append(j)

            if len(new_cols) >= max_new_cols:
                break

        if not new_cols:
            break

        constraint_cols.update(new_cols)

    if not rays:
        raise RuntimeError("Could not find any nontrivial feasible cone directions")

    # ------------------------------------------------------------------
    # Final sampled feasibility filter.
    # ------------------------------------------------------------------
    nv = min(n_verify_cols, n)
    verify_cols = rng.choice(n, size=nv, replace=False)

    mins, _ = verify(rays, verify_cols)

    feasible = [
        r for r, mn in zip(rays, mins)
        if mn >= -feasibility_tol
    ]

    if len(feasible) < k:
        # Keep least-violating candidates too; this is deliberately an
        # approximate offline dictionary search.
        order = np.argsort(mins)[::-1]
        feasible = [rays[i] for i in order[:max(k, len(feasible))]]

    # ------------------------------------------------------------------
    # Greedy maximum-residual selection.
    #
    # This maximizes novelty relative to the span of already selected
    # rays and strongly favors linear independence.
    # ------------------------------------------------------------------
    C = np.stack(feasible, axis=1)  # (n, n_candidates)

    selected: list[np.ndarray] = []
    Qbasis = np.empty((n, 0), dtype=np.float64)

    available = list(range(C.shape[1]))

    while available and len(selected) < k:
        if Qbasis.shape[1] == 0:
            # First ray: arbitrary feasible candidate.
            best = available[0]
            score = 1.0
        else:
            best = None
            score = -np.inf

            for j in available:
                r = C[:, j]
                residual = r - Qbasis @ (Qbasis.T @ r)
                s = np.linalg.norm(residual)

                if s > score:
                    score = s
                    best = j

        if best is None or score < independence_tol:
            break

        r = C[:, best]

        # Numerically remove lineality once more.
        r = r - r.mean()
        r /= np.linalg.norm(r)

        selected.append(r)
        available.remove(best)

        Qbasis, _ = np.linalg.qr(np.stack(selected, axis=1))

    if len(selected) < k:
        raise RuntimeError(
            f"Only found {len(selected)} sufficiently independent rays; "
            f"requested k={k}. Increase candidate_factor / constraint samples, "
            f"or reduce k."
        )

    R = np.stack(selected[:k], axis=1)

    out = torch.as_tensor(
        R,
        dtype=gram.dtype,
        device=gram.device,
    )

    return DenseMatrix(out)