from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import linprog

from rational_factor.models.structured_matrices import DenseMatrix


CertificateMode = Literal["auto", "full", "neighbors"]


@dataclass(frozen=True)
class _VariableLayout:
    c: slice
    r_diag: slice
    r_off: slice
    t_diag: slice
    t_off: slice
    abs_c: slice
    n_vars: int


class LowRankMetzlerConeRayFinder:
    r"""
    Find diverse reduced generator rays C in R^{r x r} with Metzler
    certificates R,T satisfying

        R U       = U C,
        T U_tilde = U_tilde D(C),

    where

        D(C) = -H^T C^T H^{-T}.

    If R and T are Metzler and U,U_tilde are entrywise nonnegative, then

        U exp(C)             = exp(R) U       >= 0,
        U_tilde exp(D(C))    = exp(T) U_tilde >= 0.

    Thus these C rays can be combined with arbitrary nonnegative neural
    coefficients while preserving positivity exactly.

    Parameters
    ----------
    U:
        Nonnegative array/tensor of shape (m, r), full column rank.
    U_tilde:
        Nonnegative array/tensor of shape (m, r), full column rank.
    H:
        Invertible reduced Gram/coupling matrix of shape (r, r). IMPORTANT:
        pass H itself, not H.T. The paired generator is constructed as

            D(C) = -H.T @ C.T @ inv(H).T.

    certificate_mode:
        ``"full"`` uses every off-diagonal entry of R and T as a certificate
        variable. This is exact but uses O(m^2) LP variables.

        ``"neighbors"`` restricts each row of R/T to a fixed set of
        off-diagonal neighbors. Any ray returned is still rigorously feasible;
        this only under-approximates the full cone and reduces the certificate
        variables to O(m * certificate_neighbors).

        ``"auto"`` uses full certificates for m <= max_full_m and neighbor
        certificates otherwise.
    certificate_neighbors:
        Number of off-diagonal row neighbors used in ``"neighbors"`` mode.
        Neighbors are selected by cosine similarity between rows of U (and
        separately U_tilde). Set >= m-1 to recover full support.
    remove_identity:
        Add tr(C)=0 while searching so the trivial scaling line C=c I_r does
        not dominate the random LPs. The identity direction can be added back
        separately at runtime if desired.

    Notes
    -----
    The projected cone in C-space is polyhedral, but explicit extreme-ray
    enumeration is generally unattractive here because the natural Metzler
    certificate formulation contains R/T variables whose count can scale with
    m. Instead this class probes the compact L1 slice of the cone with many
    random LP objectives and returns a cosine-diverse subset of C directions.
    """

    def __init__(
        self,
        U: np.ndarray | torch.Tensor,
        U_tilde: np.ndarray | torch.Tensor,
        H: np.ndarray | torch.Tensor,
        *,
        certificate_mode: CertificateMode = "auto",
        certificate_neighbors: int = 32,
        max_full_m: int = 128,
        candidate_factor: int = 12,
        feasibility_tol: float = 1e-8,
        cosine_tol: float = 1e-6,
        rank_tol: float = 1e-10,
        seed: int = 0,
        remove_identity: bool = True,
    ):
        self._torch_ref = self._pick_torch_reference(U, U_tilde, H)

        self.U = self._to_numpy(U)
        self.U_tilde = self._to_numpy(U_tilde)
        self.H = self._to_numpy(H)

        if self.U.ndim != 2:
            raise ValueError("U must have shape (m, r)")
        if self.U_tilde.shape != self.U.shape:
            raise ValueError(
                "U_tilde must have the same shape as U; got "
                f"{self.U_tilde.shape} and {self.U.shape}"
            )

        self.m, self.r = self.U.shape
        if self.H.shape != (self.r, self.r):
            raise ValueError(
                f"H must have shape ({self.r}, {self.r}); got {self.H.shape}"
            )

        if np.min(self.U) < -feasibility_tol:
            raise ValueError("U must be entrywise nonnegative")
        if np.min(self.U_tilde) < -feasibility_tol:
            raise ValueError("U_tilde must be entrywise nonnegative")

        if np.linalg.matrix_rank(self.U, tol=rank_tol) != self.r:
            raise ValueError("U must have full column rank")
        if np.linalg.matrix_rank(self.U_tilde, tol=rank_tol) != self.r:
            raise ValueError("U_tilde must have full column rank")
        if np.linalg.matrix_rank(self.H, tol=rank_tol) != self.r:
            raise ValueError("H must be invertible")

        if certificate_mode not in ("auto", "full", "neighbors"):
            raise ValueError(
                "certificate_mode must be 'auto', 'full', or 'neighbors'"
            )
        if certificate_neighbors <= 0:
            raise ValueError("certificate_neighbors must be positive")
        if max_full_m <= 0:
            raise ValueError("max_full_m must be positive")
        if candidate_factor <= 0:
            raise ValueError("candidate_factor must be positive")

        if certificate_mode == "auto":
            certificate_mode = "full" if self.m <= max_full_m else "neighbors"

        self.certificate_mode = certificate_mode
        self.certificate_neighbors = min(certificate_neighbors, max(0, self.m - 1))
        self.max_full_m = max_full_m
        self.candidate_factor = candidate_factor
        self.feasibility_tol = feasibility_tol
        self.cosine_tol = cosine_tol
        self.rank_tol = rank_tol
        self.seed = seed
        self.remove_identity = remove_identity

        # H^{-T}; H itself is the input, not H^T.
        self.H_inv_T = np.linalg.inv(self.H).T

        # Linear map vec(D) = L_D vec(C), using row-major vectorization.
        self._D_map = self._build_D_map()

        self._r_pairs = self._certificate_pairs(self.U)
        self._t_pairs = self._certificate_pairs(self.U_tilde)

        self._layout = self._build_layout()
        self._A_eq, self._b_eq = self._build_equalities()
        self._A_ub, self._b_ub = self._build_l1_slice()
        self._bounds = self._build_bounds()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find(self, k: int) -> DenseMatrix:
        """Return ``k`` cosine-diverse feasible C rays as a batched DenseMatrix."""
        if k <= 0:
            raise ValueError("k must be positive")

        rng = np.random.default_rng(self.seed)
        count = max(self.candidate_factor * k, k + 8)
        candidates: list[np.ndarray] = []

        for _ in range(count):
            q = rng.standard_normal((self.r, self.r))
            if self.remove_identity:
                q -= (np.trace(q) / self.r) * np.eye(self.r)

            objective = np.zeros(self._layout.n_vars, dtype=np.float64)
            objective[self._layout.c] = -q.ravel(order="C")

            res = linprog(
                c=objective,
                A_ub=self._A_ub,
                b_ub=self._b_ub,
                A_eq=self._A_eq,
                b_eq=self._b_eq,
                bounds=self._bounds,
                method="highs",
            )

            if not res.success:
                continue

            C = res.x[self._layout.c].reshape((self.r, self.r), order="C")
            norm = np.linalg.norm(C)
            if norm <= 1e-10:
                continue

            # Positive rescaling preserves cone feasibility.
            candidates.append(C / norm)

        if not candidates:
            raise RuntimeError(
                "Could not find a nontrivial reduced cone direction. "
                "If using neighbor certificates, increase certificate_neighbors "
                "or switch to certificate_mode='full'."
            )

        selected = self._select_diverse(candidates, k)
        rays = np.stack(selected)

        return DenseMatrix(
            torch.as_tensor(
                rays,
                dtype=self._torch_dtype,
                device=self._torch_device,
            )
        )

    __call__ = find

    def paired_generator(self, C: np.ndarray | torch.Tensor):
        """
        Compute D(C) = -H^T C^T H^{-T} using the same backend as ``C``.
        """
        if torch.is_tensor(C):
            H = torch.as_tensor(self.H, dtype=C.dtype, device=C.device)
            H_inv_T = torch.as_tensor(self.H_inv_T, dtype=C.dtype, device=C.device)
            return -H.T @ C.T @ H_inv_T

        C_np = np.asarray(C, dtype=np.float64)
        return -self.H.T @ C_np.T @ self.H_inv_T

    @property
    def certificate_support_sizes(self) -> tuple[int, int]:
        """Number of allowed off-diagonal entries in R and T."""
        return len(self._r_pairs), len(self._t_pairs)

    # ------------------------------------------------------------------
    # Input conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_torch_reference(*xs):
        for x in xs:
            if torch.is_tensor(x):
                return x
        return None

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        if torch.is_tensor(x):
            return x.detach().cpu().double().numpy()
        return np.asarray(x, dtype=np.float64)

    @property
    def _torch_dtype(self):
        if self._torch_ref is None:
            return torch.float64
        return self._torch_ref.dtype

    @property
    def _torch_device(self):
        if self._torch_ref is None:
            return torch.device("cpu")
        return self._torch_ref.device

    # ------------------------------------------------------------------
    # D(C) linear map
    # ------------------------------------------------------------------

    def _build_D_map(self) -> np.ndarray:
        """
        Return L_D with row-major vectorization:

            vec_C(D(C)) = L_D @ vec_C(C),
            D(C) = -H^T C^T H^{-T}.

        r is intentionally small, so building this map by basis evaluation is
        simple and avoids Kronecker/vectorization convention mistakes.
        """
        n = self.r * self.r
        L = np.empty((n, n), dtype=np.float64)

        for p in range(n):
            C = np.zeros((self.r, self.r), dtype=np.float64)
            C.ravel(order="C")[p] = 1.0
            D = -self.H.T @ C.T @ self.H_inv_T
            L[:, p] = D.ravel(order="C")

        return L

    # ------------------------------------------------------------------
    # Sparse/full Metzler certificate support
    # ------------------------------------------------------------------

    def _certificate_pairs(self, U: np.ndarray) -> np.ndarray:
        """
        Return allowed directed off-diagonal pairs (i,j) for a Metzler
        certificate. A returned pair means R[i,j] (or T[i,j]) is a nonnegative
        LP variable.

        In full mode all i != j are included. In neighbor mode each row keeps
        the most cosine-similar row directions. Restricting support cannot
        invalidate a returned ray: it only searches a smaller certified cone.
        """
        m = self.m
        if m <= 1:
            return np.empty((0, 2), dtype=np.int64)

        if self.certificate_mode == "full" or self.certificate_neighbors >= m - 1:
            rows = np.repeat(np.arange(m), m - 1)
            cols = np.concatenate(
                [np.concatenate((np.arange(i), np.arange(i + 1, m))) for i in range(m)]
            )
            return np.column_stack((rows, cols)).astype(np.int64, copy=False)

        k = self.certificate_neighbors
        norms = np.linalg.norm(U, axis=1)
        if np.any(norms <= self.rank_tol):
            raise ValueError("U/U_tilde cannot contain zero rows in neighbor mode")
        V = U / norms[:, None]

        pairs = np.empty((m * k, 2), dtype=np.int64)
        pos = 0

        # O(m^2 r) work but only O(m r) storage. This is offline. If m becomes
        # enormous, replace this block with an ANN/kNN implementation.
        for i in range(m):
            similarity = V @ V[i]
            similarity[i] = -np.inf
            js = np.argpartition(similarity, -k)[-k:]
            # Stable ordering is useful for reproducibility/debugging.
            js = js[np.argsort(similarity[js])[::-1]]
            pairs[pos:pos + k, 0] = i
            pairs[pos:pos + k, 1] = js
            pos += k

        return pairs

    def _build_layout(self) -> _VariableLayout:
        n_c = self.r * self.r
        n_roff = len(self._r_pairs)
        n_toff = len(self._t_pairs)

        start = 0
        c = slice(start, start + n_c)
        start = c.stop
        r_diag = slice(start, start + self.m)
        start = r_diag.stop
        r_off = slice(start, start + n_roff)
        start = r_off.stop
        t_diag = slice(start, start + self.m)
        start = t_diag.stop
        t_off = slice(start, start + n_toff)
        start = t_off.stop
        abs_c = slice(start, start + n_c)
        start = abs_c.stop

        return _VariableLayout(c, r_diag, r_off, t_diag, t_off, abs_c, start)

    # ------------------------------------------------------------------
    # Equality constraints: R U = U C and T U_tilde = U_tilde D(C)
    # ------------------------------------------------------------------

    @staticmethod
    def _group_pairs_by_row(pairs: np.ndarray, m: int):
        groups: list[list[tuple[int, int]]] = [[] for _ in range(m)]
        for local_idx, (i, j) in enumerate(pairs):
            groups[int(i)].append((int(j), local_idx))
        return groups

    def _build_equalities(self) -> tuple[sp.csr_matrix, np.ndarray]:
        m, r = self.m, self.r
        layout = self._layout
        n_main = 2 * m * r
        n_trace = 1 if self.remove_identity else 0
        n_rows = n_main + n_trace

        eq_rows: list[int] = []
        eq_cols: list[int] = []
        eq_vals: list[float] = []

        r_groups = self._group_pairs_by_row(self._r_pairs, m)
        t_groups = self._group_pairs_by_row(self._t_pairs, m)

        # --------------------------------------------------------------
        # R U = U C
        # --------------------------------------------------------------
        for i in range(m):
            for a in range(r):
                row = i * r + a

                # (R U)_{i,a}: diagonal certificate entry.
                val = self.U[i, a]
                if val != 0.0:
                    eq_rows.append(row)
                    eq_cols.append(layout.r_diag.start + i)
                    eq_vals.append(val)

                # (R U)_{i,a}: supported nonnegative off-diagonals.
                for j, local_idx in r_groups[i]:
                    val = self.U[j, a]
                    if val != 0.0:
                        eq_rows.append(row)
                        eq_cols.append(layout.r_off.start + local_idx)
                        eq_vals.append(val)

                # -(U C)_{i,a} = -sum_b U[i,b] C[b,a].
                for b in range(r):
                    val = -self.U[i, b]
                    if val != 0.0:
                        eq_rows.append(row)
                        eq_cols.append(layout.c.start + b * r + a)
                        eq_vals.append(val)

        # --------------------------------------------------------------
        # T U_tilde = U_tilde D(C)
        # --------------------------------------------------------------
        # D_map reshaped as D[b,a] coefficients over vec(C).
        D_coeff = self._D_map.reshape((r, r, r * r), order="C")
        second_base = m * r

        for i in range(m):
            # coefficients[a,p] = coefficient of C_p in (U_tilde D)_{i,a}
            coeff = np.einsum("b,bap->ap", self.U_tilde[i], D_coeff)

            for a in range(r):
                row = second_base + i * r + a

                val = self.U_tilde[i, a]
                if val != 0.0:
                    eq_rows.append(row)
                    eq_cols.append(layout.t_diag.start + i)
                    eq_vals.append(val)

                for j, local_idx in t_groups[i]:
                    val = self.U_tilde[j, a]
                    if val != 0.0:
                        eq_rows.append(row)
                        eq_cols.append(layout.t_off.start + local_idx)
                        eq_vals.append(val)

                nz = np.flatnonzero(np.abs(coeff[a]) > 0.0)
                for p in nz:
                    eq_rows.append(row)
                    eq_cols.append(layout.c.start + int(p))
                    eq_vals.append(-float(coeff[a, p]))

        # Remove the trivial C = c I_r lineality while discovering mixing rays.
        if self.remove_identity:
            row = n_main
            for a in range(r):
                eq_rows.append(row)
                eq_cols.append(layout.c.start + a * r + a)
                eq_vals.append(1.0)

        A_eq = sp.csr_matrix(
            (eq_vals, (eq_rows, eq_cols)),
            shape=(n_rows, layout.n_vars),
            dtype=np.float64,
        )
        b_eq = np.zeros(n_rows, dtype=np.float64)
        return A_eq, b_eq

    # ------------------------------------------------------------------
    # Compact L1 slice in C-space
    # ------------------------------------------------------------------

    def _build_l1_slice(self) -> tuple[sp.csr_matrix, np.ndarray]:
        """
        Enforce |C_p| <= s_p and sum_p s_p <= 1. This makes random linear
        objectives bounded without imposing artificial bounds on R/T witnesses.
        """
        n = self.r * self.r
        layout = self._layout

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        # C_p - s_p <= 0.
        for p in range(n):
            rows.extend((p, p))
            cols.extend((layout.c.start + p, layout.abs_c.start + p))
            vals.extend((1.0, -1.0))

        # -C_p - s_p <= 0.
        offset = n
        for p in range(n):
            rows.extend((offset + p, offset + p))
            cols.extend((layout.c.start + p, layout.abs_c.start + p))
            vals.extend((-1.0, -1.0))

        # sum s_p <= 1.
        sum_row = 2 * n
        for p in range(n):
            rows.append(sum_row)
            cols.append(layout.abs_c.start + p)
            vals.append(1.0)

        A_ub = sp.csr_matrix(
            (vals, (rows, cols)),
            shape=(2 * n + 1, layout.n_vars),
            dtype=np.float64,
        )
        b_ub = np.zeros(2 * n + 1, dtype=np.float64)
        b_ub[-1] = 1.0
        return A_ub, b_ub

    def _build_bounds(self):
        layout = self._layout
        bounds: list[tuple[float | None, float | None]] = [
            (None, None)
        ] * layout.n_vars

        # Off-diagonal entries of the Metzler certificates are nonnegative.
        for p in range(layout.r_off.start, layout.r_off.stop):
            bounds[p] = (0.0, None)
        for p in range(layout.t_off.start, layout.t_off.stop):
            bounds[p] = (0.0, None)

        # L1 auxiliary variables are nonnegative.
        for p in range(layout.abs_c.start, layout.abs_c.stop):
            bounds[p] = (0.0, None)

        # C and certificate diagonals are free.
        return bounds

    # ------------------------------------------------------------------
    # Diversity selection in C-space
    # ------------------------------------------------------------------

    def _select_diverse(
        self,
        rays: list[np.ndarray],
        k: int,
    ) -> list[np.ndarray]:
        if len(rays) < k:
            raise RuntimeError(
                f"Only found {len(rays)} candidates; requested {k}. "
                "Increase candidate_factor or certificate_neighbors."
            )

        V = np.stack([C.ravel() for C in rays])
        norms = np.linalg.norm(V, axis=1)
        valid = norms > 1e-12
        V = V[valid]
        rays = [C for C, keep in zip(rays, valid) if keep]
        V /= np.linalg.norm(V, axis=1, keepdims=True)

        S = np.clip(V @ V.T, -1.0, 1.0)

        # Deduplicate positive-scaled copies of the same cone ray.
        keep: list[int] = []
        for i in range(len(rays)):
            if not keep or np.max(S[i, keep]) < 1.0 - self.cosine_tol:
                keep.append(i)

        V = V[keep]
        rays = [rays[i] for i in keep]
        S = np.clip(V @ V.T, -1.0, 1.0)

        if len(rays) < k:
            raise RuntimeError(
                f"Only found {len(rays)} distinct candidates; requested {k}. "
                "Increase candidate_factor or certificate_neighbors."
            )

        if k == 1:
            if len(rays) == 1:
                return rays
            T = S.copy()
            np.fill_diagonal(T, -np.inf)
            return [rays[int(np.argmin(T.max(axis=1)))]]

        T = S.copy()
        np.fill_diagonal(T, np.inf)
        i, j = np.unravel_index(np.argmin(T), T.shape)

        selected = [int(i), int(j)]
        available = np.ones(len(rays), dtype=bool)
        available[selected] = False
        max_similarity = S[:, selected].max(axis=1)

        while len(selected) < k:
            candidates = np.flatnonzero(available)
            best = candidates[np.argmin(max_similarity[candidates])]
            selected.append(int(best))
            available[best] = False
            max_similarity = np.maximum(max_similarity, S[:, best])

        return [rays[i] for i in selected]
