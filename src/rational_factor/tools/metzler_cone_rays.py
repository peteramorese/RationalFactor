from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import linprog

try:
    from rational_factor.models.structured_matrices import DenseMatrix
except ModuleNotFoundError:  # standalone fallback
    class DenseMatrix:
        def __init__(self, data):
            self.data = data

        @property
        def shape(self):
            return self.data.shape

        @property
        def dtype(self):
            return self.data.dtype

        @property
        def device(self):
            return self.data.device

        def to_dense(self):
            return self.data

        @property
        def T(self):
            return DenseMatrix(self.data.transpose(-1, -2))


SupportMode = Literal["auto", "full", "neighbors"]


@dataclass(frozen=True)
class PairedMetzlerConeRays:
    r"""Batch of paired Metzler generator rays.

    For every batch index k,

        R[k] M + M T[k]^T = 0,

    and R[k], T[k] are Metzler.  Therefore every nonnegative combination

        R(z) = sum_k a_k(z) R[k],
        T(z) = sum_k a_k(z) T[k],       a_k(z) >= 0,

    is again feasible.  If A0,B0 >= 0 and A0 G B0^T = M, then

        A(z) = exp(R(z)) A0,
        B(z) = exp(T(z)) B0

    are nonnegative and satisfy A(z) G B(z)^T = M exactly.

    ``C`` is the r x r active-subspace witness used internally:

        R L = L C,
        T F = -F C^T,

    for an internal rank factorization M = L F^T.
    """

    R: DenseMatrix
    T: DenseMatrix
    C: torch.Tensor
    singular_values: torch.Tensor
    factor_residual: float

    def __len__(self) -> int:
        return int(self.C.shape[0])


@dataclass(frozen=True)
class _Layout:
    c: slice
    r: slice
    t: slice
    n_vars: int


class RankDeficientGramConeRayFinder:
    r"""Find diverse paired Metzler rays preserving a singular Gram target.

    Parameters
    ----------
    M:
        Fixed target Gram matrix of shape ``(m, m)`` and numerical rank
        ``rank``.
    rank:
        Intended rank r of M, with 1 <= r < m.
    m:
        Ambient matrix dimension.  Passed explicitly to make dimension
        mismatches fail early; it must agree with M.shape.

    Feasible generator pairs satisfy

        R M + M T^T = 0,
        R Metzler,
        T Metzler.

    Rather than impose the m x m matrix equality directly, the implementation
    computes a compact SVD factorization

        M = L F^T,    L,F in R^{m x r},

    and introduces an auxiliary active generator C in R^{r x r}:

        R L = L C,
        T F = -F C^T.

    These equations are equivalent to R M + M T^T = 0 when L and F have full
    column rank.  They reduce the equality count from O(m^2) to O(m r).

    ``support_mode='full'`` searches the exact dense paired cone.  For large m,
    ``'neighbors'`` restricts each row of R/T to a fixed number of off-diagonal
    neighbors, yielding a rigorous inner approximation with O(m * neighbors)
    variables. ``'auto'`` chooses full support for m <= max_full_m.

    The trivial reciprocal scaling line

        (R,T,C) = (gamma I, -gamma I, gamma I)

    is removed during discovery with tr(C)=0.  Gram-null directions with C=0
    are *not* removed; those are precisely useful rank-deficiency freedoms.
    """

    def __init__(
        self,
        M: np.ndarray | torch.Tensor | DenseMatrix,
        rank: int,
        m: int,
        *,
        support_mode: SupportMode = "auto",
        n_neighbors: int = 32,
        max_full_m: int = 96,
        candidate_factor: int = 16,
        feasibility_tol: float = 1e-8,
        rank_tol: float = 1e-9,
        factor_tol: float = 1e-8,
        cosine_tol: float = 1e-6,
        seed: int = 0,
        remove_scalar_lineality: bool = True,
    ):
        self._torch_ref = self._pick_torch_reference(M)
        self.M = self._to_numpy(M)
        self.m = int(m)
        self.rank = int(rank)

        if self.M.ndim != 2 or self.M.shape != (self.m, self.m):
            raise ValueError(
                f"M must have shape ({self.m}, {self.m}); got {self.M.shape}"
            )
        if not 1 <= self.rank < self.m:
            raise ValueError("rank must satisfy 1 <= rank < m")
        if candidate_factor <= 0:
            raise ValueError("candidate_factor must be positive")
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        if support_mode not in ("auto", "full", "neighbors"):
            raise ValueError("support_mode must be 'auto', 'full', or 'neighbors'")

        self.feasibility_tol = float(feasibility_tol)
        self.rank_tol = float(rank_tol)
        self.factor_tol = float(factor_tol)
        self.cosine_tol = float(cosine_tol)
        self.candidate_factor = int(candidate_factor)
        self.seed = int(seed)
        self.remove_scalar_lineality = bool(remove_scalar_lineality)

        # Compact rank-r factorization M = L F^T from the SVD.
        U, s, Vh = np.linalg.svd(self.M, full_matrices=False)
        if s[0] <= 0:
            raise ValueError("M must be nonzero")

        threshold = self.rank_tol * s[0]
        numerical_rank = int(np.sum(s > threshold))
        if numerical_rank != self.rank:
            raise ValueError(
                f"declared rank={self.rank}, but numerical rank={numerical_rank} "
                f"using tolerance {self.rank_tol:g} * sigma_max"
            )

        self.singular_values = s[: self.rank].copy()
        root_s = np.sqrt(self.singular_values)
        self.L = U[:, : self.rank] * root_s[None, :]
        self.F = Vh[: self.rank, :].T * root_s[None, :]

        reconstructed = self.L @ self.F.T
        denom = max(np.linalg.norm(self.M), 1.0)
        self.factor_residual = float(np.linalg.norm(reconstructed - self.M) / denom)
        if self.factor_residual > self.factor_tol:
            raise ValueError(
                "rank-r factorization does not reconstruct M accurately: "
                f"relative residual={self.factor_residual:.3e}"
            )

        if support_mode == "auto":
            support_mode = "full" if self.m <= max_full_m else "neighbors"
        self.support_mode = support_mode
        self.n_neighbors = min(int(n_neighbors), max(0, self.m - 1))
        self.max_full_m = int(max_full_m)

        self._r_rows, self._r_cols = self._support_pairs(self.L)
        self._t_rows, self._t_cols = self._support_pairs(self.F)
        self._layout = self._build_layout()
        self._A_eq, self._b_eq = self._build_equalities()
        self._bounds = self._build_bounds()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find(self, k: int) -> PairedMetzlerConeRays:
        if k <= 0:
            raise ValueError("k must be positive")

        rng = np.random.default_rng(self.seed)
        count = max(self.candidate_factor * k, k + 12)
        candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        for _ in range(count):
            objective = rng.standard_normal(self._layout.n_vars)

            # Do not let arbitrary diagonal offsets dominate every objective.
            # The variables remain in the LP; this only balances the probing.
            objective /= max(np.linalg.norm(objective), 1e-12)

            res = linprog(
                c=-objective,
                A_eq=self._A_eq,
                b_eq=self._b_eq,
                bounds=self._bounds,
                method="highs",
            )
            if not res.success:
                continue

            C = res.x[self._layout.c].reshape(
                (self.rank, self.rank), order="C"
            )
            R = self._unpack_supported(
                res.x[self._layout.r], self._r_rows, self._r_cols
            )
            T = self._unpack_supported(
                res.x[self._layout.t], self._t_rows, self._t_cols
            )

            pair_norm = np.sqrt(np.sum(R * R) + np.sum(T * T))
            if pair_norm <= 1e-10:
                continue

            R /= pair_norm
            T /= pair_norm
            C /= pair_norm

            if self._direct_residual(R, T) > 50.0 * self.feasibility_tol:
                continue
            if self._min_offdiag(R) < -self.feasibility_tol:
                continue
            if self._min_offdiag(T) < -self.feasibility_tol:
                continue

            candidates.append((R, T, C))

        if not candidates:
            raise RuntimeError(
                "Could not find a nontrivial paired Metzler direction. "
                "If using neighbor support, increase n_neighbors or use "
                "support_mode='full'."
            )

        selected = self._select_diverse(candidates, k)

        R_batch = np.stack([x[0] for x in selected])
        T_batch = np.stack([x[1] for x in selected])
        C_batch = np.stack([x[2] for x in selected])

        dtype, device = self._torch_dtype_device()
        return PairedMetzlerConeRays(
            R=DenseMatrix(torch.as_tensor(R_batch, dtype=dtype, device=device)),
            T=DenseMatrix(torch.as_tensor(T_batch, dtype=dtype, device=device)),
            C=torch.as_tensor(C_batch, dtype=dtype, device=device),
            singular_values=torch.as_tensor(
                self.singular_values, dtype=dtype, device=device
            ),
            factor_residual=self.factor_residual,
        )

    __call__ = find

    # ------------------------------------------------------------------
    # LP construction
    # ------------------------------------------------------------------

    def _support_pairs(self, embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = self.m
        if self.support_mode == "full" or self.n_neighbors >= m - 1:
            rows = np.repeat(np.arange(m), m)
            cols = np.tile(np.arange(m), m)
            return rows.astype(int), cols.astype(int)

        # Keep the diagonal plus rows whose embedding vectors have the largest
        # absolute cosine similarity.  This is a safe inner approximation: any
        # solution remains an exact Metzler certificate/generator.
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        normalized = embedding / np.maximum(norms, 1e-15)
        similarity = np.abs(normalized @ normalized.T)
        np.fill_diagonal(similarity, -np.inf)

        rows: list[int] = []
        cols: list[int] = []
        for i in range(m):
            rows.append(i)
            cols.append(i)
            nbr = np.argpartition(
                similarity[i], -self.n_neighbors
            )[-self.n_neighbors :]
            for j in np.sort(nbr):
                rows.append(i)
                cols.append(int(j))
        return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)

    def _build_layout(self) -> _Layout:
        r2 = self.rank * self.rank
        p_r = len(self._r_rows)
        p_t = len(self._t_rows)
        c = slice(0, r2)
        r = slice(c.stop, c.stop + p_r)
        t = slice(r.stop, r.stop + p_t)
        return _Layout(c=c, r=r, t=t, n_vars=t.stop)

    def _build_equalities(self) -> tuple[sp.csr_matrix, np.ndarray]:
        """Build R L = L C and T F = -F C^T, plus tr(C)=0."""
        m, r = self.m, self.rank
        n_eq = 2 * m * r + (1 if self.remove_scalar_lineality else 0)
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        # Fast lookup for all supported entries in each row.
        r_by_row: list[list[tuple[int, int]]] = [[] for _ in range(m)]
        for q, (i, j) in enumerate(zip(self._r_rows, self._r_cols)):
            r_by_row[int(i)].append((q, int(j)))

        t_by_row: list[list[tuple[int, int]]] = [[] for _ in range(m)]
        for q, (i, j) in enumerate(zip(self._t_rows, self._t_cols)):
            t_by_row[int(i)].append((q, int(j)))

        eq = 0
        # R L - L C = 0.
        for i in range(m):
            for a in range(r):
                for q, j in r_by_row[i]:
                    v = self.L[j, a]
                    if v != 0.0:
                        rows.append(eq)
                        cols.append(self._layout.r.start + q)
                        vals.append(float(v))
                for b in range(r):
                    v = -self.L[i, b]
                    if v != 0.0:
                        c_idx = b * r + a  # C[b,a], row-major
                        rows.append(eq)
                        cols.append(self._layout.c.start + c_idx)
                        vals.append(float(v))
                eq += 1

        # T F + F C^T = 0.
        for i in range(m):
            for a in range(r):
                for q, j in t_by_row[i]:
                    v = self.F[j, a]
                    if v != 0.0:
                        rows.append(eq)
                        cols.append(self._layout.t.start + q)
                        vals.append(float(v))
                for b in range(r):
                    v = self.F[i, b]
                    if v != 0.0:
                        c_idx = a * r + b  # C[a,b] from C^T[b,a]
                        rows.append(eq)
                        cols.append(self._layout.c.start + c_idx)
                        vals.append(float(v))
                eq += 1

        if self.remove_scalar_lineality:
            for a in range(r):
                c_idx = a * r + a
                rows.append(eq)
                cols.append(self._layout.c.start + c_idx)
                vals.append(1.0)
            eq += 1

        assert eq == n_eq
        A_eq = sp.csr_matrix(
            (vals, (rows, cols)), shape=(n_eq, self._layout.n_vars)
        )
        return A_eq, np.zeros(n_eq, dtype=np.float64)

    def _build_bounds(self) -> list[tuple[float, float]]:
        bounds: list[tuple[float, float]] = []

        # Active generator C is signed.
        bounds.extend([(-1.0, 1.0)] * (self.rank * self.rank))

        # R and T are Metzler: diagonal signed, off-diagonal nonnegative.
        for i, j in zip(self._r_rows, self._r_cols):
            bounds.append((-1.0, 1.0) if i == j else (0.0, 1.0))
        for i, j in zip(self._t_rows, self._t_cols):
            bounds.append((-1.0, 1.0) if i == j else (0.0, 1.0))
        return bounds

    # ------------------------------------------------------------------
    # Verification / diversity
    # ------------------------------------------------------------------

    def _unpack_supported(
        self, values: np.ndarray, rows: np.ndarray, cols: np.ndarray
    ) -> np.ndarray:
        A = np.zeros((self.m, self.m), dtype=np.float64)
        A[rows, cols] = values
        return A

    def _direct_residual(self, R: np.ndarray, T: np.ndarray) -> float:
        E = R @ self.M + self.M @ T.T
        scale = max(np.linalg.norm(self.M), 1.0)
        return float(np.linalg.norm(E) / scale)

    @staticmethod
    def _min_offdiag(A: np.ndarray) -> float:
        if A.shape[0] <= 1:
            return np.inf
        mask = ~np.eye(A.shape[0], dtype=bool)
        return float(A[mask].min())

    def _select_diverse(
        self,
        rays: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        k: int,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        # Runtime geometry is the pair (R,T), so select diversity in that space.
        V = np.stack(
            [np.concatenate([R.ravel(), T.ravel()]) for R, T, _ in rays]
        )
        norms = np.linalg.norm(V, axis=1)
        valid = norms > 1e-12
        V = V[valid]
        rays = [ray for ray, keep in zip(rays, valid) if keep]
        V /= np.linalg.norm(V, axis=1, keepdims=True)

        S = np.clip(V @ V.T, -1.0, 1.0)
        keep: list[int] = []
        for i in range(len(rays)):
            if not keep or np.max(S[i, keep]) < 1.0 - self.cosine_tol:
                keep.append(i)

        rays = [rays[i] for i in keep]
        V = V[keep]
        S = np.clip(V @ V.T, -1.0, 1.0)

        if len(rays) < k:
            raise RuntimeError(
                f"Only found {len(rays)} distinct paired directions; requested {k}. "
                "Increase candidate_factor or use fuller support."
            )
        if k == 1:
            if len(rays) == 1:
                return rays
            Tsim = S.copy()
            np.fill_diagonal(Tsim, -np.inf)
            return [rays[int(np.argmin(Tsim.max(axis=1)))]]

        Tsim = S.copy()
        np.fill_diagonal(Tsim, np.inf)
        i, j = np.unravel_index(np.argmin(Tsim), Tsim.shape)
        selected = [int(i), int(j)]
        available = np.ones(len(rays), dtype=bool)
        available[selected] = False
        max_similarity = S[:, selected].max(axis=1)

        while len(selected) < k:
            cand = np.flatnonzero(available)
            best = cand[np.argmin(max_similarity[cand])]
            selected.append(int(best))
            available[best] = False
            max_similarity = np.maximum(max_similarity, S[:, best])

        return [rays[i] for i in selected]

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_torch_reference(M):
        if isinstance(M, DenseMatrix):
            return M.to_dense()
        if torch.is_tensor(M):
            return M
        return None

    @staticmethod
    def _to_numpy(M) -> np.ndarray:
        if isinstance(M, DenseMatrix):
            M = M.to_dense()
        if torch.is_tensor(M):
            return M.detach().cpu().double().numpy()
        return np.asarray(M, dtype=np.float64)

    def _torch_dtype_device(self):
        if self._torch_ref is None:
            return torch.float64, torch.device("cpu")
        return self._torch_ref.dtype, self._torch_ref.device
