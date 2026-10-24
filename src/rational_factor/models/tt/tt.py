from rational_factor.models.structured_matrices import Matrix
from typing import Sequence
import torch
import math
from rational_factor.models.structured_vectors import Vector, as_vector


class TTVector(Vector):
    r"""Vector stored as a Tensor Train, optionally with leading batch dims.

    Cores have shape (*batch_shape, r_{k-1}, n_k, r_k) with boundary ranks ``r_0 = r_d = 1``.

    The flattened vector dimension is n = prod(n_k) and the public dense shape is (*batch_shape, n).
    """

    def __init__(
        self,
        cores: Sequence[torch.Tensor],
        numerical_tolerance: float = 1e-20,
    ):
        if len(cores) == 0:
            raise ValueError("TTVector requires at least one core")

        cores_t = tuple(torch.as_tensor(c) for c in cores)

        for i, c in enumerate(cores_t):
            if c.dim() < 3:
                raise ValueError(
                    f"TTVector core {i} must have shape "
                    f"(*batch, r_left, n, r_right), got {tuple(c.shape)}"
                )

        # All leading batch dimensions are allowed to broadcast.
        try:
            batch_shape = torch.broadcast_shapes(
                *(c.shape[:-3] for c in cores_t)
            )
        except RuntimeError as exc:
            raise ValueError(
                "TTVector core batch dimensions must be broadcast-compatible, "
                f"got {[tuple(c.shape[:-3]) for c in cores_t]}"
            ) from exc

        cores_t = tuple(
            c.expand(*batch_shape, *c.shape[-3:])
            for c in cores_t
        )

        if cores_t[0].shape[-3] != 1 or cores_t[-1].shape[-1] != 1:
            raise ValueError(
                "TTVector boundary ranks must be 1, got "
                f"r0={cores_t[0].shape[-3]}, "
                f"rd={cores_t[-1].shape[-1]}"
            )

        for i in range(len(cores_t) - 1):
            if cores_t[i].shape[-1] != cores_t[i + 1].shape[-3]:
                raise ValueError(
                    f"TTVector rank mismatch at bond {i}: "
                    f"{cores_t[i].shape[-1]} != "
                    f"{cores_t[i + 1].shape[-3]}"
                )

        self._cores = cores_t
        self._batch_shape = torch.Size(batch_shape)
        self._modes = tuple(int(c.shape[-2]) for c in cores_t)
        self._n = int(math.prod(self._modes))
        self._numerical_tolerance = float(numerical_tolerance)

    @classmethod
    def from_cores(
        cls,
        cores: Sequence[torch.Tensor],
        numerical_tolerance: float = 1e-20,
    ) -> "TTVector":
        return cls(cores, numerical_tolerance=numerical_tolerance)

    def _spawn(self, cores: Sequence[torch.Tensor]) -> "TTVector":
        return TTVector(cores, numerical_tolerance=self.numerical_tolerance)

    @property
    def numerical_tolerance(self) -> float:
        return self._numerical_tolerance

    def with_numerical_tolerance(self, numerical_tolerance: float) -> "TTVector":
        return TTVector(self._cores, numerical_tolerance=numerical_tolerance)

    @property
    def cores(self) -> tuple[torch.Tensor, ...]:
        return self._cores

    @property
    def batch_shape(self) -> torch.Size:
        return self._batch_shape

    @property
    def is_batched(self) -> bool:
        return len(self._batch_shape) > 0

    @property
    def modes(self) -> tuple[int, ...]:
        return self._modes

    @property
    def ranks(self) -> tuple[int, ...]:
        return (1,) + tuple(
            int(c.shape[-1])
            for c in self._cores
        )

    @property
    def shape(self) -> torch.Size:
        return self._batch_shape + (self._n,)

    @property
    def n(self) -> int:
        return self._n

    @property
    def dtype(self) -> torch.dtype:
        return self._cores[0].dtype

    @property
    def device(self) -> torch.device:
        return self._cores[0].device

    @property
    def is_rank_one(self) -> bool:
        """Whether all TT ranks are one."""
        return all(r == 1 for r in self.ranks)

    def elementwise_multiply(self, other: "TTVector") -> "TTVector":
        r"""Return the elementwise / Hadamard product ``self * other``.

        Leading batch dimensions follow normal PyTorch broadcasting.

        If ``self`` has TT ranks ``r_k`` and ``other`` has TT ranks ``s_k``,
        the resulting TT has ranks

            r_k * s_k.

        In particular, if either operand has TT rank 1, the rank structure of
        the other operand is preserved exactly.

        Parameters
        ----------
        other:
            TTVector with the same physical mode sizes.

        Returns
        -------
        TTVector
            Batched TTVector representing the elementwise product.
        """
        if not isinstance(other, TTVector):
            raise TypeError(
                "TTVector.elementwise_multiply requires another TTVector"
            )

        if self.modes != other.modes:
            raise ValueError(
                "TTVector elementwise multiplication requires matching modes, "
                f"got {self.modes} and {other.modes}"
            )

        try:
            batch_shape = torch.broadcast_shapes(
                self.batch_shape,
                other.batch_shape,
            )
        except RuntimeError as exc:
            raise ValueError(
                "TTVector batch shapes are not broadcast-compatible: "
                f"{tuple(self.batch_shape)} and "
                f"{tuple(other.batch_shape)}"
            ) from exc

        out_cores = []

        for a, b in zip(self._cores, other._cores):

            prod = (
                a[..., :, None, :, :, None]
                *
                b[..., None, :, :, None, :]
            )

            ra_left = a.shape[-3]
            n = a.shape[-2]
            ra_right = a.shape[-1]

            rb_left = b.shape[-3]
            rb_right = b.shape[-1]

            # Combine the two left bonds and the two right bonds.
            prod = prod.reshape(
                *batch_shape,
                ra_left * rb_left,
                n,
                ra_right * rb_right,
            )

            out_cores.append(prod)

        return self._spawn(out_cores)


    def __mul__(self, other) -> "TTVector":
        if isinstance(other, TTVector):
            return self.elementwise_multiply(other)

        # Optional convenience: scalar multiplication.
        if isinstance(other, (int, float, torch.Tensor)):
            return self.scale(other)

        return NotImplemented


    def __rmul__(self, other) -> "TTVector":
        if isinstance(other, TTVector):
            return other.elementwise_multiply(self)

        if isinstance(other, (int, float, torch.Tensor)):
            return self.scale(other)

        return NotImplemented

    def to_dense(self) -> torch.Tensor:
        r"""Materialize the flattened dense vectors.
        """

        t = self._cores[0].squeeze(-3)

        for core in self._cores[1:]:
            t = torch.einsum("...ir,...rjs->...ijs", t, core)

            t = t.reshape(*self._batch_shape, -1, core.shape[-1])

        # Final boundary rank is one.
        return t.squeeze(-1)

    def scale(
        self,
        s: torch.Tensor | float,
    ) -> "TTVector":
        r"""Scale the TT.

        ``s`` may be either:

        - a scalar, or
        - a tensor broadcast-compatible with ``batch_shape``.

        A batched scale is absorbed into the first core.
        """
        s = torch.as_tensor(
            s,
            dtype=self.dtype,
            device=self.device,
        )

        try:
            torch.broadcast_shapes(
                self._batch_shape,
                s.shape,
            )
        except RuntimeError as exc:
            raise ValueError(
                f"Scale shape {tuple(s.shape)} is not broadcast-compatible "
                f"with TT batch shape {tuple(self._batch_shape)}"
            ) from exc

        s_core = s.reshape(*s.shape, 1, 1, 1)

        cores = list(self._cores)
        cores[0] = cores[0] * s_core

        return self._spawn(cores)

    def batch_select(self, *index) -> "TTVector":
        r"""Index only the leading batch dimensions.
        """
        if len(index) > len(self._batch_shape):
            raise IndexError(
                f"Received {len(index)} batch indices for TT with "
                f"batch shape {tuple(self._batch_shape)}"
            )

        idx = tuple(index) + (
            slice(None),
        ) * (len(self._batch_shape) - len(index))

        return self._spawn([
            c[idx]
            for c in self._cores
        ])

    def elementwise_divide(
        self,
        other,
    ) -> "TTVector":
        r"""Return the elementwise quotient ``self / other``.

        The denominator ``other`` must be rank one (TT or NestedTT). The
        numerator may have arbitrary TT ranks.  The denominator's
        ``numerical_tolerance`` is added core-wise / factor-wise so call sites
        never need ``q + tol`` (which would raise TT rank).
        """

        if not isinstance(other, TTVector):
            raise TypeError(
                "TTVector.elementwise_divide requires another TTVector "
                f"(or NestedTTVector), got {type(other).__name__}"
            )

        if self.modes != other.modes:
            raise ValueError(
                "TTVector elementwise division requires matching modes, got "
                f"{self.modes} and {other.modes}"
            )

        if not other.is_rank_one:
            raise ValueError(
                "TTVector elementwise division requires a rank-1 denominator, "
                f"got ranks={other.ranks}"
            )

        try:
            torch.broadcast_shapes(
                self.batch_shape,
                other.batch_shape,
            )
        except RuntimeError as exc:
            raise ValueError(
                "TTVector batch shapes are not broadcast-compatible: "
                f"{tuple(self.batch_shape)} and "
                f"{tuple(other.batch_shape)}"
            ) from exc

        tol = float(other.numerical_tolerance)
        return self._spawn([
            a / (b + tol)
            for a, b in zip(self._cores, other._cores)
        ])

    def __truediv__(
        self,
        other,
    ) -> "TTVector":
        return self.elementwise_divide(other)

    def elementwise_add(self, other: "TTVector") -> "TTVector":
        r"""Return the exact elementwise sum ``self + other``.

        Leading batch dimensions follow normal PyTorch broadcasting.

        If ``self`` has TT ranks ``r_k`` and ``other`` has TT ranks ``s_k``,
        the resulting TT has ranks

            r_k + s_k

        at each internal bond.

        No TT rounding/compression is performed.
        """
        if not isinstance(other, TTVector):
            raise TypeError(
                "TTVector.elementwise_add requires another TTVector"
            )

        if self.modes != other.modes:
            raise ValueError(
                "TTVector addition requires matching modes, got "
                f"{self.modes} and {other.modes}"
            )

        if self.device != other.device:
            raise ValueError(
                "TTVector addition requires both operands on the same device, "
                f"got {self.device} and {other.device}"
            )

        try:
            batch_shape = torch.broadcast_shapes(
                self.batch_shape,
                other.batch_shape,
            )
        except RuntimeError as exc:
            raise ValueError(
                "TTVector batch shapes are not broadcast-compatible: "
                f"{tuple(self.batch_shape)} and "
                f"{tuple(other.batch_shape)}"
            ) from exc

        # torch.cat requires a common dtype.
        dtype = torch.promote_types(self.dtype, other.dtype)

        a_cores = [
            c.to(dtype=dtype).expand(*batch_shape, *c.shape[-3:])
            for c in self._cores
        ]
        b_cores = [
            c.to(dtype=dtype).expand(*batch_shape, *c.shape[-3:])
            for c in other._cores
        ]

        d = len(a_cores)

        # Special case: a one-core TT is just a dense physical mode with
        # boundary ranks (1, 1), so addition is direct.
        if d == 1:
            return self._spawn([a_cores[0] + b_cores[0]])

        out_cores = []

        out_cores.append(
            torch.cat(
                [a_cores[0], b_cores[0]],
                dim=-1,
            )
        )

        for a, b in zip(a_cores[1:-1], b_cores[1:-1]):
            ra_left, n, ra_right = a.shape[-3:]
            rb_left, nb, rb_right = b.shape[-3:]

            if n != nb:
                raise ValueError(
                    f"TT mode mismatch during addition: {n} != {nb}"
                )

            c = torch.zeros(
                *batch_shape,
                ra_left + rb_left,
                n,
                ra_right + rb_right,
                dtype=dtype,
                device=self.device,
            )

            c[..., :ra_left, :, :ra_right] = a
            c[..., ra_left:, :, ra_right:] = b

            out_cores.append(c)

        out_cores.append(
            torch.cat(
                [a_cores[-1], b_cores[-1]],
                dim=-3,
            )
        )

        return self._spawn(out_cores)


    def add_scalar(
        self,
        s: torch.Tensor | float | int,
    ) -> "TTVector":
        r"""Add a scalar to every entry of the TTVector.

        ``s`` may be a scalar or a tensor broadcast-compatible with the
        TT batch dimensions.

        This constructs the rank-1 constant tensor

            s * 1 ⊗ 1 ⊗ ... ⊗ 1

        and adds it exactly to ``self``.

        Therefore the internal TT ranks increase by at most one.
        """
        s = torch.as_tensor(
            s,
            dtype=self.dtype,
            device=self.device,
        )

        try:
            batch_shape = torch.broadcast_shapes(
                self.batch_shape,
                s.shape,
            )
        except RuntimeError as exc:
            raise ValueError(
                f"Scalar/batch value shape {tuple(s.shape)} is not "
                f"broadcast-compatible with TT batch shape "
                f"{tuple(self.batch_shape)}"
            ) from exc

        # Broadcast s to the common batch shape.
        s = s.expand(batch_shape)

        constant_cores = []

        for k, n in enumerate(self.modes):
            if k == 0:
                core = s[..., None, None, None].expand(*batch_shape, 1, n, 1)
            else:
                core = torch.ones(*batch_shape, 1, n, 1, dtype=self.dtype, device=self.device)

            constant_cores.append(core)

        return self.elementwise_add(self._spawn(constant_cores))


    def __add__(self, other) -> "TTVector":
        if isinstance(other, TTVector):
            return self.elementwise_add(other)

        if isinstance(other, (int, float, torch.Tensor)):
            return self.add_scalar(other)

        return NotImplemented


    def __radd__(self, other) -> "TTVector":
        # Addition is commutative.
        return self.__add__(other)

    def sum(self) -> torch.Tensor:
        r"""Return the sum of all vector entries using TT contraction.
        This never materializes the dense vector.
        """

        acc = self._cores[0].sum(dim=-2)

        for core in self._cores[1:]:
            core_sum = core.sum(dim=-2)
            acc = torch.matmul(acc, core_sum)
        return acc.squeeze(-1).squeeze(-1)

    def clone(self) -> "TTVector":
        return self._spawn([
            c.clone()
            for c in self._cores
        ])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"shape={tuple(self.shape)}, "
            f"batch_shape={tuple(self._batch_shape)}, "
            f"modes={self._modes}, "
            f"ranks={self.ranks}, "
            f"numerical_tolerance={self.numerical_tolerance}, "
            f"dtype={self.dtype}, "
            f"device={self.device})"
        )


class MPO(Matrix):
    r"""Matrix stored as a Tensor-Train matrix / MPO.

    Cores have shape ``(r_{k-1}, M_k, N_k, r_k)`` with boundary ranks
    ``r_0 = r_d = 1``. The public :class:`Matrix` interface sees the flattened
    shape ``(prod(M_k), prod(N_k))``.

    ``matvec`` with a compatible :class:`~rational_factor.models.structured_vectors.TTVector`
    contracts core-wise and returns a :class:`TTVector` (exact, no rounding).
    Dense vectors densify through :func:`as_vector` and use an implicit TT
    contraction that never materializes the full matrix.
    """

    def __init__(self, cores: Sequence[torch.Tensor]) -> None:
        if len(cores) == 0:
            raise ValueError("TTMatrix requires at least one core")
        cores_t = tuple(torch.as_tensor(c) for c in cores)
        for i, c in enumerate(cores_t):
            if c.dim() != 4:
                raise ValueError(
                    f"TTMatrix core {i} must have shape (r_left, M, N, r_right), "
                    f"got {tuple(c.shape)}"
                )
        if cores_t[0].shape[0] != 1 or cores_t[-1].shape[-1] != 1:
            raise ValueError(
                "TTMatrix boundary ranks must be 1, got "
                f"r0={cores_t[0].shape[0]}, rd={cores_t[-1].shape[-1]}"
            )
        for i in range(len(cores_t) - 1):
            if cores_t[i].shape[-1] != cores_t[i + 1].shape[0]:
                raise ValueError(
                    f"TTMatrix rank mismatch at bond {i}: "
                    f"{cores_t[i].shape[-1]} != {cores_t[i + 1].shape[0]}"
                )
        self._cores = cores_t
        self._row_modes = tuple(int(c.shape[1]) for c in cores_t)
        self._col_modes = tuple(int(c.shape[2]) for c in cores_t)
        self._n_rows = int(math.prod(self._row_modes))
        self._n_cols = int(math.prod(self._col_modes))

    @classmethod
    def from_cores(cls, cores: Sequence[torch.Tensor]) -> 'MPO':
        return cls(cores)

    @property
    def cores(self) -> tuple[torch.Tensor, ...]:
        return self._cores

    @property
    def row_modes(self) -> tuple[int, ...]:
        return self._row_modes

    @property
    def col_modes(self) -> tuple[int, ...]:
        return self._col_modes

    @property
    def ranks(self) -> tuple[int, ...]:
        return (1,) + tuple(int(c.shape[-1]) for c in self._cores)

    @property
    def shape(self) -> torch.Size:
        return torch.Size((self._n_rows, self._n_cols))

    @property
    def dtype(self) -> torch.dtype:
        return self._cores[0].dtype

    @property
    def device(self) -> torch.device:
        return self._cores[0].device

    @property
    def T(self) -> 'MPO':
        return MPO([c.permute(0, 2, 1, 3).contiguous() for c in self._cores])

    def to_dense(self) -> torch.Tensor:
        eye = torch.eye(self._n_cols, dtype=self.dtype, device=self.device)
        return self.matvec(eye)

    def _matvec_dense(self, x: torch.Tensor) -> torch.Tensor:
        """Implicit TT contraction for dense ``(..., n_cols)`` or ``(..., n_cols, k)``."""
        if x.ndim >= 2 and x.shape[-2] == self._n_cols:
            # (..., n_cols, k) -> (..., k, *col_modes)
            x_rhs = x.transpose(-2, -1)
            leading = x_rhs.shape[:-1]
            x_modes = x_rhs.reshape(*leading, *self._col_modes)
            y_modes = self._contract_dense(x_modes)
            return y_modes.reshape(*leading, self._n_rows).transpose(-2, -1)

        if x.shape[-1] == self._n_cols:
            leading = x.shape[:-1]
            x_modes = x.reshape(*leading, *self._col_modes)
            y_modes = self._contract_dense(x_modes)
            return y_modes.reshape(*leading, self._n_rows)

        raise ValueError(
            f"x must have shape (..., {self._n_cols}) or "
            f"(..., {self._n_cols}, k), got {tuple(x.shape)}"
        )

    def _contract_dense(self, x_modes: torch.Tensor) -> torch.Tensor:
        """Contract MPO cores into a tensorized dense vector.

        ``x_modes`` has shape ``(B..., N_1, ..., N_d)``; returns
        ``(B..., M_1, ..., M_d)``.
        """
        d = len(self._cores)
        batch_ndim = x_modes.dim() - d
        result = x_modes.unsqueeze(-1)
        for core in self._cores:
            # Contract current N (always at ``batch_ndim``) and the TT bond.
            result = torch.tensordot(result, core, dims=([batch_ndim, -1], [2, 0]))
        return result.squeeze(-1)

    def _matvec_tt(self, x: TTVector) -> TTVector:
        if x.modes != self._col_modes:
            raise ValueError(
                "TTVector modes do not match TTMatrix column modes: "
                f"{x.modes} != {self._col_modes}"
            )
        out_cores = []
        for Mc, xc in zip(self._cores, x.cores):
            # Mc: (R0, M, N, R1), xc: (*batch, S0, N, S1)
            core = torch.einsum("ijkl,...mkp->...imjlp", Mc, xc)
            batch_shape = core.shape[:-5]
            core = core.reshape(
                *batch_shape,
                Mc.shape[0] * xc.shape[-3],
                Mc.shape[1],
                Mc.shape[3] * xc.shape[-1],
            )
            out_cores.append(core)
        return TTVector(out_cores, numerical_tolerance=x.numerical_tolerance)

    def matvec(self, x: torch.Tensor | Vector) -> torch.Tensor | TTVector:
        x = as_vector(x)
        if isinstance(x, TTVector):
            out = self._matvec_tt(x)
            return out.with_numerical_tolerance(x.numerical_tolerance)
        return self._matvec_dense(
            x.to_dense().to(dtype=self.dtype, device=self.device)
        )

    def sum(self) -> torch.Tensor:
        acc = self._cores[0].sum(dim=(1, 2))  # (1, r1)
        for c in self._cores[1:]:
            acc = acc @ c.sum(dim=(1, 2))
        return acc.reshape(())

    def diag(self) -> torch.Tensor:
        if self._row_modes != self._col_modes:
            return super().diag()
        diag_cores = [
            torch.diagonal(c, dim1=1, dim2=2).permute(0, 2, 1).contiguous()
            for c in self._cores
        ]
        return TTVector(diag_cores).to_dense()

    def matmul(self, other: Matrix) -> Matrix:
        """Exact MPO @ MPO with bond ranks multiplied sitewise."""
        if not isinstance(other, MPO):
            return super().matmul(other)

        if self.col_modes != other.row_modes:
            raise ValueError(
                f"Incompatible MPO modes: {self.col_modes} != {other.row_modes}"
            )

        cores = []
        for a, b in zip(self.cores, other.cores):
            c = torch.einsum("aikb,ckjd->acijbd", a, b)
            cores.append(c.reshape(
                a.shape[0] * b.shape[0],
                a.shape[1],
                b.shape[2],
                a.shape[3] * b.shape[3],
            ))

        return MPO(cores)

    def rmatmul(self, other: Matrix) -> Matrix:
        """Compute other @ self."""
        return (
            other.matmul(self)
            if isinstance(other, MPO)
            else super().rmatmul(other)
        )

    def scale(self, s: torch.Tensor | float) -> Matrix:
        s = torch.as_tensor(s, dtype=self.dtype, device=self.device)
        if s.numel() != 1:
            return super().scale(s)
        cores = list(self._cores)
        cores[0] = cores[0] * s
        return MPO(cores)

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> 'MPO':
        return MPO([
            c.to(device=device if device is not None else c.device,
                 dtype=dtype if dtype is not None else c.dtype)
            for c in self._cores
        ])

    def clone(self) -> 'MPO':
        return MPO([c.clone() for c in self._cores])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"shape={tuple(self.shape)}, "
            f"modes={list(zip(self.row_modes, self.col_modes))}, "
            f"ranks={self.ranks}, "
            f"dtype={self.dtype}, "
            f"device={self.device})"
        )


class LowRankFactorizationMPO(Matrix):
    r"""Ordinary-rank-``r`` matrix ``B = U @ V.T`` with rectangular MPO factors.

    For ``d`` physical sites, both factors have column modes
    ``(1, ..., 1, r)``. The latent index is stored only at the last site;
    their row modes may differ. Only ``r``-element latent vectors are ever
    densified. TT input/output contractions preserve structured storage.
    """

    def __init__(self, U: MPO, V: MPO):
        if not isinstance(U, MPO) or not isinstance(V, MPO):
            raise TypeError("U and V must be MPO instances")
        if len(U.cores) != len(V.cores):
            raise ValueError("U and V must have the same number of sites")
        if U.col_modes != V.col_modes or any(n != 1 for n in U.col_modes[:-1]):
            raise ValueError("both factor column modes must be (1, ..., 1, r)")
        if U.dtype != V.dtype or U.device != V.device:
            raise ValueError("U and V must share a dtype and device")
        self.U, self.V = U, V

    @property
    def r(self) -> int:
        return self.U.col_modes[-1]

    @property
    def shape(self) -> torch.Size:
        return torch.Size((self.U.shape[0], self.V.shape[0]))

    @property
    def dtype(self) -> torch.dtype:
        return self.U.dtype

    @property
    def device(self) -> torch.device:
        return self.U.device

    @property
    def T(self) -> 'LowRankFactorizationMPO':
        return LowRankFactorizationMPO(self.V, self.U)

    def to_dense(self) -> torch.Tensor:
        """Only for debugging small systems; exponentially sized output."""
        return self.U.to_dense() @ self.V.to_dense().T

    def _latent_tt(self, z: torch.Tensor) -> TTVector:
        """Embed (..., r) in a TT with modes (1, ..., 1, r)."""
        z = torch.as_tensor(z, device=self.device, dtype=self.dtype)
        if z.shape[-1] != self.r:
            raise ValueError(f"latent vector must have size {self.r}, got {tuple(z.shape)}")
        batch = z.shape[:-1]
        leading = [z.new_ones(*batch, 1, 1, 1) for _ in range(len(self.U.cores) - 1)]
        return TTVector([*leading, z.reshape(*batch, 1, self.r, 1)])

    def apply_latent(self, z: torch.Tensor) -> TTVector:
        """Compute ``U @ z`` as a TT of ranks bounded by U's MPO ranks."""
        return self.U.matvec(self._latent_tt(z))

    def project_latent(self, x: torch.Tensor | TTVector) -> torch.Tensor:
        """Compute ``V.T @ x``, densifying only the r-dimensional output."""
        if isinstance(x, TTVector):
            out = self.V.T.matvec(x)
            return out.to_dense()
        out = self.V.T._matvec_dense(
            torch.as_tensor(x, device=self.device, dtype=self.dtype)
        )
        return out

    def matvec(self, x: torch.Tensor | Vector) -> TTVector:
        if isinstance(x, Vector) and not isinstance(x, TTVector):
            x = x.to_dense()
        return self.apply_latent(self.project_latent(x))

    def rev_matvec(self, x: torch.Tensor | Vector) -> TTVector:
        return self.T.matvec(x)

    def sum(self) -> torch.Tensor:
        def column_sums(factor: MPO):
            x = TTVector([c.new_ones(1, n, 1) for c, n in zip(factor.cores, factor.row_modes)])
            return factor.T.matvec(x).to_dense()
        return (column_sums(self.U) * column_sums(self.V)).sum()

    @staticmethod
    def _scale_rows(factor: MPO, weights: TTVector) -> MPO:
        if not isinstance(weights, TTVector) or not weights.is_rank_one or weights.is_batched:
            raise ValueError("diagonal scaling requires an unbatched rank-one TTVector")
        if factor.row_modes != weights.modes:
            raise ValueError("diagonal weights must match factor row modes")
        return MPO([core * w[0, :, 0][None, :, None, None]
                    for core, w in zip(factor.cores, weights.cores)])

    def mul_diag_left(self, weights: TTVector) -> 'LowRankFactorizationMPO':
        """Exact diag(weights) @ B, without changing the MPO factor ranks."""
        return LowRankFactorizationMPO(self._scale_rows(self.U, weights), self.V)

    def mul_diag_right(self, weights: TTVector) -> 'LowRankFactorizationMPO':
        """Exact B @ diag(weights), without changing the MPO factor ranks."""
        return LowRankFactorizationMPO(self.U, self._scale_rows(self.V, weights))

    def to(self, *, device=None, dtype=None) -> 'LowRankFactorizationMPO':
        return LowRankFactorizationMPO(self.U.to(device=device, dtype=dtype),
                                       self.V.to(device=device, dtype=dtype))

    def clone(self) -> 'LowRankFactorizationMPO':
        return LowRankFactorizationMPO(self.U.clone(), self.V.clone())

    def latent_kernel(self, omega: MPO, a: TTVector, q: TTVector) -> torch.Tensor:
        r"""Calculate ``K = U.T @ diag(a) @ omega @ diag(q)^-1 @ V``.

        All contractions use only local cores. The result is a dense r x r
        matrix. This version requires rank-one a, q, and omega; it never
        forms an N x r intermediate or a dense N-dimensional vector.
        """
        if not isinstance(omega, MPO) or any(rank != 1 for rank in omega.ranks):
            raise ValueError("omega must be a Kronecker-rank-one MPO")
        if not (isinstance(a, TTVector) and isinstance(q, TTVector)
                and a.is_rank_one and q.is_rank_one
                and not a.is_batched and not q.is_batched):
            raise ValueError("a and q must be unbatched rank-one TT vectors")
        if (a.modes != self.U.row_modes or q.modes != self.V.row_modes
                or omega.row_modes != a.modes or omega.col_modes != q.modes):
            raise ValueError("factor, gram, and vector local modes are incompatible")
        env = self.U.cores[0].new_ones(1, 1)
        for k, (uc, vc, gc, ac, qc) in enumerate(zip(
                self.U.cores, self.V.cores, omega.cores, a.cores, q.cores)):
            av = ac[0, :, 0]
            qv = qc[0, :, 0]
            if bool((qv == 0).any()):
                raise ValueError("q contains a zero local factor; cannot invert diag(q)")
            weighted_g = av[:, None] * gc[0, :, :, 0] / qv[None, :]
            if k == len(self.U.cores) - 1:
                # uc: (rU, n, r, 1), vc: (rV, n, r, 1).
                return torch.einsum('ac,air,ij,cjs->rs', env,
                                    uc[..., 0], weighted_g, vc[..., 0])
            # uc: (rU, n, 1, RU), vc: (rV, n, 1, RV).
            env = torch.einsum('ac,aib,ij,cjd->bd', env,
                               uc[:, :, 0, :], weighted_g, vc[:, :, 0, :])
        raise RuntimeError("empty MPO")

# -----------------------------------------------------------------------------
# Exactly invertible MPO layers (parameter ownership belongs to the caller).
# -----------------------------------------------------------------------------


def _identity_mpo(modes, *, dtype, device):
    return MPO([torch.eye(n, dtype=dtype, device=device).reshape(1, n, n, 1)
                for n in modes])


def _add_mpos(a: MPO, b: MPO) -> MPO:
    """Exact MPO addition, with internal ranks added rather than multiplied."""
    if a.row_modes != b.row_modes or a.col_modes != b.col_modes:
        raise ValueError("MPO addition requires identical physical modes")
    if a.device != b.device:
        raise ValueError("MPO addition requires matching devices")
    dtype = torch.promote_types(a.dtype, b.dtype)
    aa, bb = [c.to(dtype) for c in a.cores], [c.to(dtype) for c in b.cores]
    if len(aa) == 1:
        return MPO([aa[0] + bb[0]])
    cores = [torch.cat((aa[0], bb[0]), dim=-1)]
    for x, y in zip(aa[1:-1], bb[1:-1]):
        rx, m, n, sx = x.shape
        ry, _, _, sy = y.shape
        # Padding rather than assignment keeps autograd connected to both inputs.
        cores.append(
            torch.nn.functional.pad(x, (0, sy, 0, 0, 0, 0, 0, ry))
            + torch.nn.functional.pad(y, (sx, 0, 0, 0, 0, 0, rx, 0))
        )
    cores.append(torch.cat((aa[-1], bb[-1]), dim=0))
    return MPO(cores)


class InvertibleMPO(MPO):
    """An MPO with an exact, structure-aware inverse (never dense inversion)."""

    def inverse(self) -> 'InvertibleMPO':
        raise NotImplementedError

    def matmul(self, other: Matrix) -> Matrix:
        if isinstance(other, InvertibleMPO):
            # Factors are kept in application order: rightmost factor first.
            return InvertibleMPOProduct((other, self))
        return super().matmul(other)


class NilpotentMPO(InvertibleMPO):
    r"""``I + strength * K`` with ``K = |output><input|_control ⊗ B`` and ``K²=0``.

    ``block`` is an arbitrary square MPO on all sites *except* ``control_site``;
    it may couple sites on both sides of the control. No matrix product is needed
    to construct the inverse: change the sign of ``strength``.
    """

    def __init__(
        self,
        modes: Sequence[int],
        block: MPO | None,
        *,
        control_site: int = 0,
        input_mode: int = 0,
        output_mode: int = 1,
        strength: torch.Tensor | float = 1.0,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ):
        modes = tuple(int(n) for n in modes)
        if not modes or any(n < 1 for n in modes):
            raise ValueError("modes must be nonempty positive integers")
        if not 0 <= control_site < len(modes):
            raise ValueError("control_site is out of range")
        if not (0 <= input_mode < modes[control_site] and
                0 <= output_mode < modes[control_site] and input_mode != output_mode):
            raise ValueError("input_mode and output_mode must be distinct valid modes")
        remaining = modes[:control_site] + modes[control_site + 1:]
        if remaining:
            if not isinstance(block, MPO) or block.row_modes != remaining or block.col_modes != remaining:
                raise ValueError(f"block must be a square MPO on modes {remaining}")
            dtype, device = block.dtype, block.device
        elif block is not None:
            raise ValueError("block must be None for a single-site nilpotent MPO")
        device = torch.device('cpu' if device is None else device)
        strength = torch.as_tensor(strength, dtype=dtype, device=device)
        if strength.numel() != 1:
            raise ValueError("strength must be a scalar")

        shift = torch.zeros(modes[control_site], modes[control_site], dtype=dtype, device=device)
        shift[output_mode, input_mode] = 1
        cores = []
        j = 0
        for i, n in enumerate(modes):
            if i != control_site:
                cores.append(block.cores[j])
                j += 1
            else:
                left = block.cores[j - 1].shape[-1] if j else 1
                cores.append(torch.eye(left, dtype=dtype, device=device)[:, None, None, :]
                             * shift[None, :, :, None])
        # The single scalar multiplier only needs to be absorbed into one core.
        cores[0] = cores[0] * strength
        K = MPO(cores)
        super().__init__(_add_mpos(_identity_mpo(modes, dtype=dtype, device=device), K).cores)
        self._modes, self._block = modes, block
        self._control_site = control_site
        self._input_mode, self._output_mode = input_mode, output_mode
        self._strength = strength

    def inverse(self) -> 'NilpotentMPO':
        return NilpotentMPO(
            self._modes, self._block,
            control_site=self._control_site,
            input_mode=self._input_mode,
            output_mode=self._output_mode,
            strength=-self._strength,
            dtype=self.dtype, device=self.device,
        )

    @property
    def T(self) -> 'NilpotentMPO':
        return NilpotentMPO(
            self._modes, self._block.T if self._block is not None else None,
            control_site=self._control_site, input_mode=self._output_mode,
            output_mode=self._input_mode, strength=self._strength,
            dtype=self.dtype, device=self.device,
        )

    def to(self, *, device=None, dtype=None) -> 'NilpotentMPO':
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        return NilpotentMPO(
            self._modes,
            self._block.to(device=device, dtype=dtype) if self._block is not None else None,
            control_site=self._control_site, input_mode=self._input_mode,
            output_mode=self._output_mode,
            strength=self._strength.to(device=device, dtype=dtype),
            dtype=dtype, device=device,
        )

    def clone(self) -> 'NilpotentMPO':
        return NilpotentMPO(
            self._modes, self._block.clone() if self._block is not None else None,
            control_site=self._control_site, input_mode=self._input_mode,
            output_mode=self._output_mode, strength=self._strength.clone(),
            dtype=self.dtype, device=self.device,
        )


def _gate_to_mpo_cores(gate: torch.Tensor, modes: Sequence[int]) -> list[torch.Tensor]:
    """Exact, SVD-free MPO for an arbitrary contiguous local operator.

    Outer site cores route physical row/column indices to a central core
    containing the gate entries. No approximation or rank truncation.
    """
    modes = tuple(modes)
    k = len(modes)
    if gate.shape != (math.prod(modes),) * 2:
        raise ValueError(f"gate must have shape {(math.prod(modes),) * 2}, got {tuple(gate.shape)}")
    axes = [axis for i in range(k) for axis in (i, i + k)]
    pairs = gate.reshape(*modes, *modes).permute(*axes)
    midpoint = k // 2
    pair_sizes = [n * n for n in modes]
    cores = []
    for i, n in enumerate(modes):
        if i < midpoint:
            left = math.prod(pair_sizes[:i])
            right = left * n * n
            core = torch.eye(right, dtype=gate.dtype, device=gate.device).reshape(left, n, n, right)
        elif i == midpoint:
            core = pairs.reshape(math.prod(pair_sizes[:i]), n, n,
                                 math.prod(pair_sizes[i + 1:]))
        else:
            left = math.prod(pair_sizes[i:])
            right = math.prod(pair_sizes[i + 1:])
            core = torch.eye(left, dtype=gate.dtype, device=gate.device).reshape(
                n, n, right, left).permute(3, 0, 1, 2)
        cores.append(core)
    return cores


class GateLayerMPO(InvertibleMPO):
    """One layer of disjoint, *contiguous* invertible k-site gates.

    Args:
        modes: Physical sizes of all sites.
        gates: ``[(sites, Q), ...]`` with each ``sites`` a contiguous tuple of
            zero-based site indices and ``Q`` a square ``prod(modes[sites])`` matrix.
        dtype, device: Used for identity sites if no gates are supplied.

    The layer is a tensor product across gate groups and untouched sites; no
    global MPO multiplication or SVD is used to build it. ``inverse()`` inverts
    only the small gate matrices. The caller is responsible for providing
    invertible gates and any normalization-preserving parameterization.
    """

    def __init__(
        self,
        modes: Sequence[int],
        gates: Sequence[tuple[Sequence[int], torch.Tensor]],
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ):
        modes = tuple(int(n) for n in modes)
        if not modes or any(n < 1 for n in modes):
            raise ValueError("modes must be nonempty positive integers")
        gates = tuple((tuple(int(i) for i in sites), torch.as_tensor(Q)) for sites, Q in gates)
        used = set()
        for sites, Q in gates:
            if not sites or tuple(range(sites[0], sites[0] + len(sites))) != sites:
                raise ValueError("gate sites must be sorted and contiguous")
            if sites[0] < 0 or sites[-1] >= len(modes) or used.intersection(sites):
                raise ValueError("gate sites must be in range and disjoint")
            used.update(sites)
            n = math.prod(modes[i] for i in sites)
            if Q.shape != (n, n):
                raise ValueError(f"gate on {sites} must have shape {(n, n)}, got {tuple(Q.shape)}")
        if gates:
            dtype, device = gates[0][1].dtype, gates[0][1].device
            if any(Q.dtype != dtype or Q.device != device for _, Q in gates):
                raise ValueError("all gate matrices must have matching dtype and device")
        device = torch.device('cpu' if device is None else device)
        cores = [torch.eye(n, dtype=dtype, device=device).reshape(1, n, n, 1)
                 for n in modes]
        for sites, Q in gates:
            for i, core in zip(sites, _gate_to_mpo_cores(Q, [modes[i] for i in sites])):
                cores[i] = core
        super().__init__(cores)
        self._gate_modes, self._gates = modes, gates

    @property
    def gates(self) -> tuple[tuple[tuple[int, ...], torch.Tensor], ...]:
        return self._gates

    def inverse(self) -> 'GateLayerMPO':
        return GateLayerMPO(
            self._gate_modes,
            [(sites, torch.linalg.inv(Q)) for sites, Q in self._gates],
            dtype=self.dtype, device=self.device,
        )

    @property
    def T(self) -> 'GateLayerMPO':
        return GateLayerMPO(
            self._gate_modes, [(sites, Q.T) for sites, Q in self._gates],
            dtype=self.dtype, device=self.device,
        )

    def to(self, *, device=None, dtype=None) -> 'GateLayerMPO':
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        return GateLayerMPO(
            self._gate_modes, [(sites, Q.to(device=device, dtype=dtype))
                               for sites, Q in self._gates],
            dtype=dtype, device=device,
        )

    def clone(self) -> 'GateLayerMPO':
        return GateLayerMPO(
            self._gate_modes, [(sites, Q.clone()) for sites, Q in self._gates],
            dtype=self.dtype, device=self.device,
        )


class InvertibleMPOProduct(InvertibleMPO):
    """Materialized product of exactly invertible MPO layers.

    ``factors`` are in application order: ``(L1, L2)`` means ``L2 @ L1``.
    The factor list is retained so that ``inverse()`` uses each specialized
    inverse in reversed order, rather than attempting dense inversion.
    """

    def __init__(self, factors: Sequence[InvertibleMPO]):
        flat = []
        for f in factors:
            if isinstance(f, InvertibleMPOProduct):
                flat.extend(f.factors)
            elif isinstance(f, InvertibleMPO):
                flat.append(f)
            else:
                raise TypeError("all factors must be InvertibleMPO instances")
        if not flat:
            raise ValueError("at least one invertible MPO factor is required")
        modes = flat[0].row_modes
        if any(f.row_modes != modes or f.col_modes != modes for f in flat):
            raise ValueError("all factors must have identical square site modes")
        product = flat[0]
        for factor in flat[1:]:
            product = MPO.matmul(factor, product)  # Avoid recursive product wrappers.
        super().__init__(product.cores)
        self._factors = tuple(flat)

    @property
    def factors(self) -> tuple[InvertibleMPO, ...]:
        return self._factors

    def inverse(self) -> 'InvertibleMPOProduct':
        return InvertibleMPOProduct([f.inverse() for f in reversed(self._factors)])

    @property
    def T(self) -> 'InvertibleMPOProduct':
        return InvertibleMPOProduct([f.T for f in reversed(self._factors)])

    def to(self, *, device=None, dtype=None) -> 'InvertibleMPOProduct':
        return InvertibleMPOProduct([f.to(device=device, dtype=dtype)
                                     for f in self._factors])

    def clone(self) -> 'InvertibleMPOProduct':
        return InvertibleMPOProduct([f.clone() for f in self._factors])


# A circuit is an ordered product of invertible layers; both names are supported.
InvertibleMPOCircuit = InvertibleMPOProduct
