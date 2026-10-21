import torch
from rational_factor.models.structured_matrices import Matrix, ShiftMatrix
from rational_factor.models.basis_functions import BSpline1DBasis


class Algebra:
    def multiply(self, other: "Algebra") -> "Algebra":
        raise NotImplementedError


class MatrixSetAlgebra(Algebra):
    def __init__(self, matrices: Matrix | list[Matrix]):
        self._matrices = (
            [matrices] if isinstance(matrices, Matrix) else list(matrices)
        )

    @property
    def matrices(self) -> list[Matrix]:
        return self._matrices

    def __len__(self) -> int:
        return len(self.matrices)

    def to_dense(self) -> torch.Tensor:
        if not self.matrices:
            raise ValueError("empty matrix set")
        return torch.stack([m.to_dense() for m in self.matrices]).sum(0)

    def multiply(self, other: "MatrixSetAlgebra") -> "MatrixSetAlgebra":
        raise NotImplementedError


class NormalizedBSplineShiftMatrixSetAlgebra(MatrixSetAlgebra):
    """Canonical algebra

        M = sum_q diag(f_q) @ S_{a_q}

    where ``a_q`` are integer cell shifts and ``f_q`` are nonnegative
    cellwise B-splines satisfying

        sum_q f_q(x) = 1

    with invalid boundary shifts exactly zero.

    ``weights`` must have one batch element per supplied shift. Its cellwise
    coefficients are interpreted as logits and softmax-normalized over shifts.
    """

    def __init__(
        self,
        shifts: torch.Tensor | list[int],
        weights: BSpline1DBasis,
    ):
        shifts = torch.as_tensor(
            shifts,
            dtype=torch.long,
            device=weights.coeffs().device,
        ).reshape(-1)

        if weights.batch_size() != shifts.numel():
            raise ValueError(
                f"weights batch size {weights.batch_size()} must equal "
                f"number of shifts {shifts.numel()}"
            )

        c = weights._cell_coeffs()  # (Q, cells, degree+1)
        valid = self._valid_mask(shifts, weights.n_cells)

        # Drop shifts that are nowhere valid.
        keep = valid.any(dim=1)
        shifts, c, valid = shifts[keep], c[keep], valid[keep]

        if not valid.any(dim=0).all():
            raise ValueError(
                "at least one valid shift is required in every cell"
            )

        # Coefficient-wise normalization works because each cell's Bernstein
        # basis is nonnegative and sums to one.
        c = torch.softmax(
            c.masked_fill(~valid[..., None], -torch.inf),
            dim=0,
        )
        c = torch.where(valid[..., None], c, torch.zeros_like(c))

        shifts, c = self._collect(shifts, c)
        weights = BSpline1DBasis._from_cells(
            c, weights.n_cells, weights.degree
        )

        self._init_canonical(shifts, weights)

    @classmethod
    def _from_canonical(
        cls,
        shifts: torch.Tensor,
        weights: BSpline1DBasis,
    ) -> "NormalizedBSplineShiftMatrixSetAlgebra":
        obj = cls.__new__(cls)
        obj._init_canonical(shifts, weights)
        return obj

    def _init_canonical(
        self,
        shifts: torch.Tensor,
        weights: BSpline1DBasis,
    ) -> None:
        self.shifts = shifts
        self.weights = weights
        super().__init__([])  # matrices are generated lazily

    @staticmethod
    def _valid_mask(shifts: torch.Tensor, n: int) -> torch.Tensor:
        cells = torch.arange(n, device=shifts.device)
        dst = cells[None, :] + shifts[:, None]
        return (dst >= 0) & (dst < n)

    @staticmethod
    def _collect(
        shifts: torch.Tensor,
        coeffs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sum coefficient tensors having the same shift."""
        unique, inverse = torch.unique(
            shifts, sorted=True, return_inverse=True
        )
        index = inverse[:, None, None].expand_as(coeffs)
        out = coeffs.new_zeros(
            unique.numel(), *coeffs.shape[1:]
        ).scatter_add(0, index, coeffs)
        return unique, out

    @property
    def evaluation_points(self) -> torch.Tensor:
        """One matrix state at the center of each spline cell."""
        dtype, device = self.weights.dtype_device()
        n = self.weights.n_cells
        return (
            torch.arange(n, dtype=dtype, device=device) + 0.5
        ) / n

    def weight_values(self) -> torch.Tensor:
        """Return f_q(x_i), shape (Q, n_cells)."""
        B = self.weights.eval(self.evaluation_points)
        return torch.einsum(
            "ib,qb->qi",
            B,
            self.weights.coeffs(),
        )

    @property
    def matrices(self) -> list[Matrix]:
        """Individual diag(f_q) @ S_q atoms."""
        values = self.weight_values()
        dtype, device = self.weights.dtype_device()
        n = self.weights.n_cells

        return [
            ShiftMatrix(
                n,
                int(s),
                dtype=dtype,
                device=device,
            ).mul_diag_left(v)
            for s, v in zip(self.shifts.tolist(), values)
        ]

    def to_dense(self) -> torch.Tensor:
        """Materialize the summed stochastic matrix without Q dense atoms."""
        w = self.weight_values().T  # (n, Q)
        n = self.weights.n_cells

        rows = torch.arange(n, device=w.device)[:, None]
        cols = rows + self.shifts[None, :]
        valid = (cols >= 0) & (cols < n)

        src = torch.where(valid, w, torch.zeros_like(w))
        cols = cols.clamp(0, n - 1)

        return w.new_zeros(n, n).scatter_add(1, cols, src)

    def multiply(
        self,
        other: "NormalizedBSplineShiftMatrixSetAlgebra",
    ) -> "NormalizedBSplineShiftMatrixSetAlgebra":
        """Canonical exact product.

        (D_f S_a)(D_g S_b)
            = D_{f(x) g(x+a)} S_{a+b}.
        """
        if not isinstance(
            other, NormalizedBSplineShiftMatrixSetAlgebra
        ):
            raise TypeError("other must have the same algebra type")

        if self.weights.n_cells != other.weights.n_cells:
            raise ValueError("n_cells must match")

        n = self.weights.n_cells
        left = self.weights._cell_coeffs()

        products = []
        shifts = []

        # Vectorized over every right-hand atom; only left shifts are looped.
        for i, a in enumerate(self.shifts.tolist()):
            f = BSpline1DBasis._from_cells(
                left[i : i + 1],
                n,
                self.weights.degree,
            )

            fg = f.product(other.weights.shift_cells(a))

            products.append(fg._cell_coeffs())
            shifts.append(other.shifts + a)

        coeffs = torch.cat(products, dim=0)
        shifts = torch.cat(shifts)

        shifts, coeffs = self._collect(shifts, coeffs)

        # Exact finite-domain boundary support.
        valid = self._valid_mask(shifts, n)
        coeffs = torch.where(
            valid[..., None],
            coeffs,
            torch.zeros_like(coeffs),
        )

        # FFT multiplication can introduce tiny negative roundoff. The exact
        # algebra has nonnegative coefficients, so project numerical noise out.
        coeffs = coeffs.clamp_min(0)

        # The product of stochastic matrices is stochastic. Enforce the exact
        # Bernstein partition-of-unity numerically.
        z = coeffs.sum(dim=0, keepdim=True)
        if (z <= 0).any():
            raise RuntimeError("product lost all valid mass in a cell")
        coeffs = coeffs / z

        weights = BSpline1DBasis._from_cells(
            coeffs,
            n,
            self.weights.degree + other.weights.degree,
        )

        return type(self)._from_canonical(shifts, weights)