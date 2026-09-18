from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from scipy.optimize import linprog

from rational_factor.models.structured_matrices import Banded, Diagonal, DenseMatrix, Matrix


RayType = Literal["diagonal", "dense", "banded"]


class MetzlerConeRayFinder:
    """
    Find diverse rays X satisfying

        X Metzler,
        -M X^T M^{-1} Metzler,

    where M = gram.T if transpose=True, else gram.

    Returns a Matrix with batch size k:

        diagonal -> Diagonal:    (k, m, m), storage (k, m)
        dense    -> DenseMatrix: (k, m, m), storage (k, m, m)
        banded   -> Banded:      (k, m, m), storage (k, 2*bw+1, m)
    """

    def __init__(
        self,
        gram: Matrix,
        *,
        transpose: bool = True,
        n_constraint_cols: int = 32,
        n_verify_cols: int = 128,
        candidate_factor: int = 10,
        cut_rounds: int = 3,
        max_new_cols: int = 16,
        feasibility_tol: float = 1e-8,
        cosine_tol: float = 1e-6,
        seed: int = 0,
    ):
        self.gram = gram
        self.M = gram.T if transpose else gram

        if len(self.M.shape) != 2 or self.M.shape[-2] != self.M.shape[-1]:
            raise ValueError("gram must be a non-batched square matrix")

        self.m = self.M.shape[-1]
        self.n_constraint_cols = n_constraint_cols
        self.n_verify_cols = n_verify_cols
        self.candidate_factor = candidate_factor
        self.cut_rounds = cut_rounds
        self.max_new_cols = max_new_cols
        self.feasibility_tol = feasibility_tol
        self.cosine_tol = cosine_tol
        self.seed = seed

        self.A = self._to_scipy(self.M)
        self._lu = None
        self._exact_cache = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def find(
        self,
        k: int,
        ray_type: RayType = "diagonal",
        *,
        bandwidth: int | None = None,
    ) -> Matrix:
        if k <= 0:
            raise ValueError("k must be positive")

        rng = np.random.default_rng(self.seed)

        if ray_type == "diagonal":
            return self._find_diagonal(k, rng)

        if ray_type == "dense":
            return self._find_dense(k, rng)

        if ray_type == "banded":
            if bandwidth is None:
                raise ValueError("bandwidth is required for banded rays")
            if not 0 <= bandwidth < self.m:
                raise ValueError(
                    f"bandwidth must satisfy 0 <= bandwidth < {self.m}"
                )
            return self._find_banded(k, bandwidth, rng)

        raise ValueError(f"unknown ray_type={ray_type!r}")

    __call__ = find

    # ------------------------------------------------------------------
    # Matrix conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _to_scipy(M: Matrix) -> sp.csc_matrix:
        if isinstance(M, Banded):
            n = M.shape[-1]
            offsets = M.offsets.detach().cpu().numpy()
            data = M.data.detach().cpu().double().numpy()

            rows, cols, vals = [], [], []
            for d, off in enumerate(offsets):
                off = int(off)
                c = (
                    np.arange(n - off)
                    if off >= 0
                    else np.arange(-off, n)
                )
                rows.append(c + off)
                cols.append(c)
                vals.append(data[d, c])

            return sp.csc_matrix(
                (
                    np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols)),
                ),
                shape=(n, n),
                dtype=np.float64,
            )

        return sp.csc_matrix(
            M.to_dense().detach().cpu().double().numpy(),
            dtype=np.float64,
        )

    @property
    def lu(self):
        if self._lu is None:
            self._lu = spla.splu(self.A)
        return self._lu

    # ------------------------------------------------------------------
    # Cosine-diverse subset selection
    # ------------------------------------------------------------------

    def _select_diverse(
        self,
        rays: list[np.ndarray],
        k: int,
    ) -> list[np.ndarray]:
        """
        Greedy max-min angular diversity.

        At each step choose the candidate minimizing

            max_{s in selected} cos(candidate, s).

        Since cone rays are equivalent only under positive scaling,
        ordinary cosine similarity (not |cosine|) is used.
        """
        if len(rays) < k:
            raise RuntimeError(
                f"Only found {len(rays)} candidates; requested {k}."
            )

        V = np.stack([r.ravel() for r in rays])
        norms = np.linalg.norm(V, axis=1)

        valid = norms > 1e-12
        V = V[valid]
        rays = [r for r, keep in zip(rays, valid) if keep]
        V /= np.linalg.norm(V, axis=1, keepdims=True)

        # Pairwise cosine similarities.
        S = np.clip(V @ V.T, -1.0, 1.0)

        # Deduplicate nearly identical rays.
        keep = []
        for i in range(len(rays)):
            if not keep or np.max(S[i, keep]) < 1.0 - self.cosine_tol:
                keep.append(i)

        V = V[keep]
        rays = [rays[i] for i in keep]
        S = np.clip(V @ V.T, -1.0, 1.0)

        if len(rays) < k:
            raise RuntimeError(
                f"Only found {len(rays)} distinct rays; requested {k}. "
                "Increase candidate_factor."
            )

        if k == 1:
            # Most isolated ray in the candidate population.
            if len(rays) == 1:
                return rays

            T = S.copy()
            np.fill_diagonal(T, -np.inf)
            return [rays[int(np.argmin(T.max(axis=1)))]]

        # Start with globally least-similar pair.
        T = S.copy()
        np.fill_diagonal(T, np.inf)
        i, j = np.unravel_index(np.argmin(T), T.shape)

        selected = [int(i), int(j)]
        available = np.ones(len(rays), dtype=bool)
        available[selected] = False

        # Similarity to closest already-selected ray.
        max_similarity = S[:, selected].max(axis=1)

        while len(selected) < k:
            candidates = np.flatnonzero(available)
            best = candidates[np.argmin(max_similarity[candidates])]

            selected.append(int(best))
            available[best] = False

            max_similarity = np.maximum(
                max_similarity,
                S[:, best],
            )

        return [rays[i] for i in selected]

    # ==================================================================
    # Diagonal: existing sampled/cutting-plane algorithm
    # ==================================================================

    def _inverse_columns(self, js: np.ndarray) -> np.ndarray:
        E = np.zeros((self.m, len(js)))
        E[js, np.arange(len(js))] = 1.0
        return self.lu.solve(E)

    def _diagonal_constraints(self, js: np.ndarray) -> sp.csr_matrix:
        """Linear forms for off-diagonals of ``Y = -M diag(r) M^{-1}``.

        Column ``j`` of ``M diag(r) M^{-1}`` is ``A @ (r ⊙ M^{-1} e_j)``, so
        the off-diagonal rows of ``-A @ diag(M^{-1} e_j)`` are the Metzler
        inequalities for the dual condition.
        """
        inv = self._inverse_columns(js)
        blocks = []

        for t, j in enumerate(js):
            # Off-diag of -M diag(r) M^{-1} >= 0  <=>  H r >= 0 with H = -A diag(inv).
            H = -(self.A @ sp.diags(inv[:, t], format="csc"))
            blocks.append(H[np.arange(self.m) != j])

        return sp.vstack(blocks, format="csr")

    def _diagonal_candidates(
        self,
        H: sp.csr_matrix,
        count: int,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        n = self.m
        A_eq = sp.csr_matrix(np.ones((1, n)))
        rays = []

        for _ in range(count):
            q = rng.standard_normal(n)
            q -= q.mean()

            res = linprog(
                c=-q,
                A_ub=-H,
                b_ub=np.zeros(H.shape[0]),
                A_eq=A_eq,
                b_eq=np.zeros(1),
                bounds=[(-1.0, 1.0)] * n,
                method="highs",
            )

            if res.success:
                r = res.x - res.x.mean()
                norm = np.linalg.norm(r)
                if norm > 1e-10:
                    rays.append(r / norm)

        return rays

    def _verify_diagonal(
        self,
        rays: list[np.ndarray],
        js: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Check off-diagonals of ``Y = -M diag(r) M^{-1}`` for sampled columns."""
        if not rays:
            return np.empty(0), np.empty(0, dtype=int)

        inv = self._inverse_columns(js)
        mins = np.full(len(rays), np.inf)
        worst = np.full(len(rays), -1, dtype=int)

        for r_idx, r in enumerate(rays):
            # Y = -M diag(r) M^{-1}; columns over the sampled M^{-1} e_j.
            Y = -(self.A @ (r[:, None] * inv))

            for t, j in enumerate(js):
                vmin = Y[np.arange(self.m) != j, t].min()
                if vmin < mins[r_idx]:
                    mins[r_idx] = vmin
                    worst[r_idx] = j

        return mins, worst

    def _find_diagonal(
        self,
        k: int,
        rng: np.random.Generator,
    ) -> Diagonal:
        n = self.m
        if k >= n:
            raise ValueError(
                "For diagonal rays use k < m; identity lineality is separate."
            )

        n0 = min(n, max(2, self.n_constraint_cols))
        cols = set(rng.choice(n, n0, replace=False).tolist())
        cols.update((0, n - 1))

        count = max(self.candidate_factor * k, k + 4)
        rays: list[np.ndarray] = []

        for _ in range(self.cut_rounds):
            H = self._diagonal_constraints(
                np.array(sorted(cols), dtype=int)
            )
            rays = self._diagonal_candidates(H, count, rng)

            remaining = np.array(
                [j for j in range(n) if j not in cols],
                dtype=int,
            )
            if not rays or not len(remaining):
                break

            verify_cols = rng.choice(
                remaining,
                min(self.n_verify_cols, len(remaining)),
                replace=False,
            )
            mins, worst = self._verify_diagonal(rays, verify_cols)
            bad = np.where(mins < -self.feasibility_tol)[0]

            if not len(bad):
                break

            new_cols = []
            for idx in bad[np.argsort(mins[bad])]:
                j = int(worst[idx])
                if j >= 0 and j not in cols:
                    new_cols.append(j)
                if len(new_cols) >= self.max_new_cols:
                    break

            if not new_cols:
                break

            cols.update(new_cols)

        if not rays:
            raise RuntimeError("Could not find any diagonal cone rays")

        verify_cols = rng.choice(
            n,
            min(self.n_verify_cols, n),
            replace=False,
        )
        mins, _ = self._verify_diagonal(rays, verify_cols)

        feasible = [
            r for r, mn in zip(rays, mins)
            if mn >= -self.feasibility_tol
        ]

        if len(feasible) < k:
            order = np.argsort(mins)[::-1]
            feasible = [rays[i] for i in order[:max(k, len(feasible))]]

        selected = self._select_diverse(feasible, k)

        return Diagonal(
            torch.as_tensor(
                np.stack(selected),          # (k, m)
                dtype=self.gram.dtype,
                device=self.gram.device,
            )
        )

    # ==================================================================
    # Exact dense / banded LP
    # ==================================================================

    def _support(
        self,
        bandwidth: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Valid (row, col) locations of Z = X^T, ordered column-major.
        bandwidth=None gives dense support.
        """
        n = self.m

        if bandwidth is None:
            cols = np.repeat(np.arange(n), n)
            rows = np.tile(np.arange(n), n)
            return rows, cols

        rows, cols = [], []
        for j in range(n):
            i = np.arange(
                max(0, j - bandwidth),
                min(n, j + bandwidth + 1),
            )
            rows.append(i)
            cols.append(np.full(len(i), j))

        return np.concatenate(rows), np.concatenate(cols)

    def _exact_lp(
        self,
        bandwidth: int | None,
    ):
        """
        Variables:

            z = valid entries of Z = X^T
            y = vec(Y)

        Constraints:

            Y M = -M Z   (i.e. Y = -M X^T M^{-1})
            Z, Y Metzler
            tr(Z) = 0
        """
        if bandwidth in self._exact_cache:
            return self._exact_cache[bandwidth]

        n = self.m
        n2 = n * n
        rows, cols = self._support(bandwidth)
        p = len(rows)

        # vec(Z) = E z
        flat = rows + n * cols
        E = sp.csc_matrix(
            (np.ones(p), (flat, np.arange(p))),
            shape=(n2, p),
        )

        I = sp.eye(n, format="csc")

        # vec(YM + MZ) = 0  <=>  Y M = -M Z.
        # vec(YM) = (M^T ⊗ I) y,  vec(MZ) = (I ⊗ M) vec(Z).
        dynamics = sp.hstack(
            [
                sp.kron(I, self.A, format="csr") @ E,
                sp.kron(self.A.T, I, format="csr"),
            ],
            format="csr",
        )

        # Remove scalar-I lineality.
        diag_idx = np.flatnonzero(rows == cols)
        trace = sp.csr_matrix(
            (
                np.ones(len(diag_idx)),
                (
                    np.zeros(len(diag_idx), dtype=int),
                    diag_idx,
                ),
            ),
            shape=(1, p + n2),
        )

        A_eq = sp.vstack([dynamics, trace], format="csr")
        b_eq = np.zeros(n2 + 1)

        z_bounds = [
            (-1.0, 1.0) if i == j else (0.0, 1.0)
            for i, j in zip(rows, cols)
        ]

        y_bounds = [
            (-1.0, 1.0) if i == j else (0.0, 1.0)
            for j in range(n)
            for i in range(n)
        ]

        result = (A_eq, b_eq, z_bounds + y_bounds, rows, cols)
        self._exact_cache[bandwidth] = result
        return result

    def _exact_candidates(
        self,
        k: int,
        bandwidth: int | None,
        rng: np.random.Generator,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        A_eq, b_eq, bounds, rows, cols = self._exact_lp(bandwidth)

        p = len(rows)
        n2 = self.m * self.m
        count = max(self.candidate_factor * k, k + 4)

        rays = []

        for _ in range(count):
            # Probe both representations to expose different cone faces.
            c = -rng.standard_normal(p + n2)

            res = linprog(
                c=c,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
            )

            if not res.success:
                continue

            z = res.x[:p]
            norm = np.linalg.norm(z)

            if norm > 1e-10:
                rays.append(z / norm)

        if not rays:
            raise RuntimeError("Could not find any nontrivial cone rays")

        return rays, rows, cols

    # ------------------------------------------------------------------
    # Dense
    # ------------------------------------------------------------------

    def _find_dense(
        self,
        k: int,
        rng: np.random.Generator,
    ) -> DenseMatrix:
        rays, _, _ = self._exact_candidates(k, None, rng)
        rays = self._select_diverse(rays, k)

        # Dense support is column-major vec(Z), where Z = X^T.
        X = np.stack([
            z.reshape((self.m, self.m), order="F").T
            for z in rays
        ])

        return DenseMatrix(
            torch.as_tensor(
                X,
                dtype=self.gram.dtype,
                device=self.gram.device,
            )
        )

    # ------------------------------------------------------------------
    # Banded
    # ------------------------------------------------------------------

    def _find_banded(
        self,
        k: int,
        bandwidth: int,
        rng: np.random.Generator,
    ) -> Banded:
        rays, rows, cols = self._exact_candidates(k, bandwidth, rng)
        rays = self._select_diverse(rays, k)

        offsets = np.arange(-bandwidth, bandwidth + 1)
        offset_to_idx = {
            int(off): i for i, off in enumerate(offsets)
        }

        # Banded stores:
        #     data[r, j] = X[j + offset[r], j]
        #
        # LP variable z_q = Z[i,j] = X[j,i].
        data = np.zeros(
            (k, len(offsets), self.m),
            dtype=np.float64,
        )

        for batch, z in enumerate(rays):
            for q, (i, j) in enumerate(zip(rows, cols)):
                x_offset = int(j - i)
                x_col = int(i)
                data[batch, offset_to_idx[x_offset], x_col] = z[q]

        data = torch.as_tensor(
            data,
            dtype=self.gram.dtype,
            device=self.gram.device,
        )

        return Banded(
            torch.as_tensor(
                offsets,
                dtype=torch.long,
                device=self.gram.device,
            ),
            data,
        )