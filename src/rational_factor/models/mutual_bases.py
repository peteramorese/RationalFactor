from __future__ import annotations

import torch
from rational_factor.models.basis_functions import Basis
from rational_factor.models.parameters import Parameters
from rational_factor.models.lrpd import Rank1PlusDiagonal, Rank1PlusDiagonalFactorization


class MutualPairBasis:
    def __init__(self, 
            dim : int, 
            batch_size : int,
            n_basis : int,
            params : tuple[Parameters, ...],
            coeffs : tuple[Parameters, Parameters] = None):

        self._dim = dim
        self._batch_size = batch_size
        self._n_basis = n_basis
        self._params = params
        self._coeffs = coeffs

    def get_basis(self, index: int) -> Basis:
        raise NotImplementedError("get_basis is not implemented for this mutual basis")
    
    def Omega1(self, index : int, lows : torch.Tensor = None, highs : torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError("Omega1 is not implemented for this mutual basis")

    def Omega2(self, lows : torch.Tensor = None, highs : torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError("Omega2 is not implemented for this mutual basis")
    
    def eval(self, y : torch.Tensor, index : int = None):
        raise NotImplementedError("eval is not implemented for this mutual basis")
    
    def dim(self) -> int:
        return self._dim

class MutualPairMemberBasis(Basis):
    def __init__(self, owner : MutualPairBasis, index : int):
        super().__init__(owner.dim())
        self.owner = owner

        assert index in [0, 1], "index must be 0 or 1"
        self.index = index
    
    def _is_mate(self, other : MutualPairMemberBasis | list[MutualPairMemberBasis]) -> bool:
        if isinstance(other, MutualPairMemberBasis):
            return self.owner is other.owner and self.index != other.index
        else:
            same_owner = all(self.owner is basis.owner for basis in other)
            unique_indices = len(set(basis.index for basis in other)) == len(other)
            return same_owner and unique_indices

    def Omega1(self, lows : torch.Tensor = None, highs : torch.Tensor = None):
        return self.owner.Omega1(lows, highs)
    
    def Omega2(self, other : MutualPairMemberBasis, lows : torch.Tensor = None, highs : torch.Tensor = None):
        if not self._is_mate(other):
            raise ValueError("Omega2 is not defined for non-mate bases")
        Omega2 = self.owner.Omega2(other, lows, highs)
        if self.index == 0:
            return Omega2
        elif self.index == 1:
            return torch.transpose(Omega2, -2, -1)
    
    def __call__(self, y : torch.Tensor):
        return self.owner.eval(y, self.index)


class Orthogonal1DPWCBasis(MutualPairBasis):
    def __init__(self, 
            value_params_1 : Parameters, 
            value_params_2 : Parameters, 
            cell_width_params : Parameters, 
            r1pd_factorization : Rank1PlusDiagonalFactorization, 
            zero_mean: bool = True,
            last_alpha_cell_vector : Parameters = None, 
            last_beta_cell_vector : Parameters = None):
        value_params_tensor_1 = value_params_1()
        
        batch_size = value_params_tensor_1.shape[0]
        n_basis = value_params_tensor_1.shape[1]

        if last_alpha_cell_vector is not None or last_beta_cell_vector is not None:
            assert zero_mean, "zero_mean must be True if last_alpha_cell_vector or last_beta_cell_vector is not None"

        assert value_params_2().shape == (batch_size, n_basis), "value_params_2 must have shape (batch_size, n_basis)"
        assert cell_width_params().shape == (batch_size, n_basis), "cell_width_params must have shape (batch_size, n_basis)"

        r1pd_n = r1pd_factorization.d.size()[-1]
        extra_params: tuple[Parameters, ...] = ()
        if zero_mean:
            assert r1pd_n == n_basis - 1, (
                f"with zero_mean, r1pd factors must have trailing size n_basis-1={n_basis - 1}, got {r1pd_n}"
            )
            assert last_alpha_cell_vector is not None and last_beta_cell_vector is not None, (
                "zero_mean=True requires last_alpha_cell_vector and last_beta_cell_vector"
            )
            assert last_alpha_cell_vector().shape == (batch_size, n_basis)
            assert last_beta_cell_vector().shape == (batch_size, n_basis)
            extra_params = (last_alpha_cell_vector, last_beta_cell_vector)
        else:
            assert r1pd_n == n_basis, (
                f"with zero_mean=False, r1pd factors must have trailing size n_basis={n_basis}, got {r1pd_n}"
            )
            assert last_alpha_cell_vector is None and last_beta_cell_vector is None

        super().__init__(1, 1, n_basis, 
            (value_params_1, 
                value_params_2, 
                cell_width_params, 
                r1pd_factorization.d, 
                r1pd_factorization.u, 
                r1pd_factorization.v) + extra_params, 
            (None, None))

        self._zero_mean = zero_mean
        self._r1pd_factorization = r1pd_factorization
        self._last_alpha_cell_vector = last_alpha_cell_vector
        self._last_beta_cell_vector = last_beta_cell_vector
    
    def get_basis(self, index : int) -> MutualPairMemberBasis:
        return MutualPairMemberBasis(self, index)
    
    def Omega1(self, index : int, lows : torch.Tensor = None, highs : torch.Tensor = None):
        pass
    
    def _get_d0(self):
        # \int α_i(x) β_i(x) dx = α_i * β_i * cell_width_i
        return self._params[0]() * self._params[1]() * self._get_normalized_cell_widths()

    def _get_normalized_cell_widths(self) -> torch.Tensor:
        """Cell widths on ``[0, 1]``, shape ``(batch, n_basis)``, summing to 1."""
        return torch.softmax(self._params[2](), dim=-1)

    def _get_AB(self) -> tuple[Rank1PlusDiagonal, Rank1PlusDiagonal]:
        """Return function-vector maps ``A``, ``B`` (R1PD).

        ``A = I - (σ_α / σ_{α,m}) e_mᵀ`` so ``A σ_α = 0``.

        ``B = I_{m-1} - (1/s) (I_{m-1} σ_β) (D₀⁻¹ σ_α)ᵀ`` with
        ``s = σ_αᵀ D₀⁻¹ σ_β``, so ``B σ_β = 0`` and
        ``A D₀ Bᵀ = D₀ I_{m-1}``.
        """
        pre_proj_alpha = self._params[0]()
        pre_proj_beta = self._params[1]()
        ones = torch.ones_like(pre_proj_alpha)
        zeros = torch.zeros_like(pre_proj_alpha)
        if not self._zero_mean:
            I = Rank1PlusDiagonal(ones, zeros, zeros)
            return I, I

        cell_widths = self._get_normalized_cell_widths()
        sigma_alpha = cell_widths * pre_proj_alpha
        sigma_beta = cell_widths * pre_proj_beta
        d0 = self._get_d0()

        u_A = -sigma_alpha / sigma_alpha[..., -1].unsqueeze(-1)
        v_A = torch.zeros_like(pre_proj_alpha)
        v_A[..., -1] = 1.0
        A = Rank1PlusDiagonal(ones, u_A, v_A)

        # v_B = D₀⁻¹ σ_α so D₀ v_B = σ_α and A (D₀ v_B) = A σ_α = 0
        v_B = sigma_alpha / d0
        s = (v_B * sigma_beta).sum(dim=-1)
        u_B = -sigma_beta / s.unsqueeze(-1)
        u_B[..., -1] = 0.0
        d_B = ones.clone()
        d_B[..., -1] = 0.0
        B = Rank1PlusDiagonal(d_B, u_B, v_B)
        return A, B

    def _expand_seq(self, M: Rank1PlusDiagonal, n_data: int) -> Rank1PlusDiagonal:
        """Broadcast a ``(T, k)`` factor product across ``n_data`` independent vectors."""
        return Rank1PlusDiagonal(
            M.d.unsqueeze(0).expand(n_data, -1, -1).contiguous(),
            M.u.unsqueeze(0).expand(n_data, -1, -1).contiguous(),
            M.v.unsqueeze(0).expand(n_data, -1, -1).contiguous(),
        )

    def _r1pd_invT_conj(self, M: Rank1PlusDiagonal, diag: torch.Tensor) -> Rank1PlusDiagonal:
        """``diag(a) M^{-T} diag(a)^{-1}`` as an R1PD product (same factor count)."""
        return M.inverse().T.mul_diag_left(diag).mul_diag_right(diag.reciprocal())

    def _block_last_col_matvec(
        self,
        M: Rank1PlusDiagonal,
        last_col: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply ``[[M, 0], l]`` i.e. last column ``l``, zeros under ``M``.

        ``M`` is ``(T, m-1)``; ``last_col`` is ``(m,)``; ``x`` is ``(N, m)``.
        """
        N = x.shape[0]
        head = self._expand_seq(M, N).sequential_matvec(x[:, :-1], seq_dim=1)
        last = x[:, -1]
        l = last_col.reshape(-1)
        out_head = head + last.unsqueeze(-1) * l[:-1]
        out_last = last * l[-1]
        return torch.cat([out_head, out_last.unsqueeze(-1)], dim=-1)

    def _get_LR(self) -> tuple:
        """Operators ``L``, ``R`` with ``ψ_α = L A α`` and ``ψ_β = R B β``.

        ``zero_mean=False``: ``L`` is the ``m×m`` R1PD product; ``R = D L^{-T} D^{-1}``.

        ``zero_mean=True``: the stored R1PD is ``(m-1)×(m-1)``. With last columns
        ``l``, ``r`` (shape ``(m,)``),

        ``L = [[M, l_{:m-1}], [0, l_{m-1}]]``,
        ``R = [[D_{m-1} M^{-T} D_{m-1}^{-1}, r_{:m-1}], [0, r_{m-1}]]``.

        Each returned object has ``.matvec(x)`` for ``x`` of shape ``(N, m)``.
        """
        M = self._r1pd_factorization()

        if not self._zero_mean:
            d0 = self._get_d0().reshape(-1)
            R = self._r1pd_invT_conj(M, d0)

            class _SeqOp:
                def __init__(self, op, expand):
                    self.op = op
                    self.expand = expand

                def matvec(self, x: torch.Tensor) -> torch.Tensor:
                    return self.expand(self.op, x.shape[0]).sequential_matvec(x, seq_dim=1)

            return _SeqOp(M, self._expand_seq), _SeqOp(R, self._expand_seq)

        l = self._last_alpha_cell_vector()[0]
        r = self._last_beta_cell_vector()[0]
        d_small = self._get_d0().reshape(-1)[:-1]
        N = self._r1pd_invT_conj(M, d_small)
        expand = self._expand_seq
        block = self._block_last_col_matvec

        class _BlockOp:
            def __init__(self, core, last_col):
                self.core = core
                self.last_col = last_col

            def matvec(self, x: torch.Tensor) -> torch.Tensor:
                return block(self.core, self.last_col, x)

        return _BlockOp(M, l), _BlockOp(N, r)

    def _get_alpha_beta(self):
        """Raw disjoint PWC heights (not Aα / Bβ)."""
        return self._params[0](), self._params[1]()

    def cell_edges(self, batch_index: int = 0) -> torch.Tensor:
        """Ordered partition edges on ``[0, 1]``, shape ``(n_basis + 1,)``."""
        widths = self._get_normalized_cell_widths()[batch_index]
        zero = torch.zeros(1, dtype=widths.dtype, device=widths.device)
        return torch.cat([zero, torch.cumsum(widths, dim=0)], dim=0)

    def _cell_index(self, y: torch.Tensor, batch_index: int = 0) -> torch.Tensor:
        """Map ``y ∈ [0, 1]`` to cell indices in ``{0, ..., n_basis - 1}``."""
        y = torch.as_tensor(
            y, dtype=self._params[0]().dtype, device=self._params[0]().device
        ).reshape(-1)
        edges = self.cell_edges(batch_index)
        idx = torch.searchsorted(edges, y, right=True) - 1
        return idx.clamp(0, self._n_basis - 1)

    def eval(self, y: torch.Tensor, index: int | None = None) -> torch.Tensor:
        """Evaluate transformed α / β PWC bases at locations ``y ∈ [0, 1]``.

        At ``y`` in cell ``k`` the raw vector is ``h_k e_k``. Then
        ``ψ_α(y) = L A φ_α(y)`` and ``ψ_β(y) = R B φ_β(y)``.

        Args:
            y: ``(N,)`` or ``(N, 1)`` in ``[0, 1]``.
            index: ``0`` → α, ``1`` → β, ``None`` → both as ``(N, 2, n_basis)``.
        """
        if self._params[0]().shape[0] != 1:
            raise ValueError("eval currently requires parameter batch_size == 1")

        alpha, beta = self._get_alpha_beta()
        A, B = self._get_AB()
        L, R = self._get_LR()

        y = torch.as_tensor(y, dtype=alpha.dtype, device=alpha.device).reshape(-1)
        cell = self._cell_index(y, batch_index=0)
        N = y.shape[0]
        n = self._n_basis

        def _raw_at(heights: torch.Tensor) -> torch.Tensor:
            h = heights.reshape(n)
            raw = torch.zeros(N, n, dtype=h.dtype, device=h.device)
            raw[torch.arange(N, device=y.device), cell] = h[cell]
            return raw

        if index == 0:
            return L.matvec(A.matvec(_raw_at(alpha)))
        if index == 1:
            return R.matvec(B.matvec(_raw_at(beta)))
        if index is None:
            return torch.stack(
                [L.matvec(A.matvec(_raw_at(alpha))), R.matvec(B.matvec(_raw_at(beta)))],
                dim=1,
            )
        raise ValueError("index must be 0, 1, or None")
