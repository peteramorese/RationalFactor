from __future__ import annotations

import torch

from rational_factor.models.density_model import DensityModel, MarkovDecisionProcess
from rational_factor.models.parameters import (
    DenseMatrixParameters,
    FixedParameters,
    NeuralParameters,
    Parameters,
    RowStochasticDenseMatrixParameters,
)
from rational_factor.models.factor_forms import (
    LinearFF,
    LinearRFF,
    QuadraticFF,
    QuadraticRFF,
)
from rational_factor.models.basis_functions import Basis, BetaBasis, GaussianBasis, GaussianKernelBasis
from rational_factor.models.structured_matrices import DenseMatrix, as_matrix
from rational_factor.models.structured_vectors import as_vector


def _batched_action(action: torch.Tensor, action_features: int) -> torch.Tensor:
    """Normalize ``action`` to shape ``(1, action_features)`` for a single MDP step."""
    action = torch.as_tensor(action)
    if action.shape == (action_features,):
        action = action.unsqueeze(0)
    if action.ndim != 2 or action.shape[-1] != action_features:
        raise ValueError(
            f"action must have shape ({action_features},) or (1, {action_features}), "
            f"got {tuple(action.shape)}"
        )
    if action.shape[0] != 1:
        raise ValueError(
            f"propagate currently supports a single action (leading batch 1), "
            f"got {action.shape[0]}"
        )
    return action


def _freeze_basis_at_context(basis: Basis, **context: torch.Tensor) -> Basis:
    """Snap context-dependent basis parameters to FixedParameters (single action)."""
    frozen = []
    for p in basis.params():
        vals = p(**context)
        # Unbatched action yields intrinsic shape only; restore ensemble leading 1.
        if vals.shape == tuple(p.shape()):
            vals = vals.unsqueeze(0)
        if vals.shape[0] != 1:
            raise ValueError(
                f"freeze expects a single context batch, got leading size {vals.shape[0]}"
            )
        frozen.append(FixedParameters(vals.detach().clone()))

    cls = type(basis)
    if cls is GaussianBasis:
        return GaussianBasis(frozen[0], frozen[1])
    if cls is GaussianKernelBasis:
        return GaussianKernelBasis(frozen[0], frozen[1])
    if cls is BetaBasis:
        return BetaBasis(frozen[0], frozen[1], eps=getattr(basis, "eps", 1e-6))
    raise TypeError(
        f"cannot freeze action-conditioned basis of type {cls.__name__}; "
        "expected GaussianBasis, GaussianKernelBasis, or BetaBasis"
    )


class NeuralActionLRFF(MarkovDecisionProcess, LinearRFF):
    """LinearRFF with action-conditioned ``B`` (and typically action-conditioned ``psi``).

    ``a`` / ``phi`` are static. ``psi`` is supplied by the caller (usually a
    GaussianBasis whose parameters are :class:`NeuralParameters`). ``B`` is an
    action MLP. ``log_density`` / ``propagate`` pass ``action`` so ``Omega2``,
    ``psi``, and ``B`` are evaluated at that action while ``a`` / ``phi`` broadcast.
    """

    def __init__(
        self,
        a: Parameters,
        phi: Basis,
        psi: Basis,
        action_features: int,
        numerical_tolerance: float = 1e-20,
        register_modules: bool = True,
        hidden_features: int = 128,
        num_hidden_layers: int = 2,
        activation=torch.nn.Tanh,
        zero_init_last: bool = False,
        positive: bool = False,
    ):
        del positive  # row-stochastic B uses softmax; kept for call-site compat
        batch_size = phi.batch_size()
        n_basis = phi.n_basis_functions()
        out_features = (
            (n_basis, n_basis) if batch_size == 1 else (batch_size, n_basis, n_basis)
        )
        B = RowStochasticDenseMatrixParameters(
            NeuralParameters(
                in_features=action_features,
                out_features=out_features,
                batch_shape=(batch_size,),
                context_key="action",
                hidden_features=hidden_features,
                num_hidden_layers=num_hidden_layers,
                activation=activation,
                zero_init_last=zero_init_last,
                positive=False,
            )
        )
        LinearRFF.__init__(
            self,
            a,
            phi,
            psi,
            B,
            numerical_tolerance=numerical_tolerance,
            register_modules=register_modules,
        )
        self.action_features = action_features

    def log_density(
        self,
        xp: torch.Tensor,
        *,
        conditioner: torch.Tensor,
        action: torch.Tensor,
        **context: torch.Tensor,
    ):
        return LinearRFF.log_density(
            self, xp, conditioner=conditioner, action=action, **context
        )

    def propagate(self, belief: DensityModel, action: torch.Tensor) -> LinearFF:
        r"""One-step LinearFF propagation at a fixed action.

        Instantiates ``B(action)`` and ``Omega2(phi, psi(action))``, updates
        coefficients, and returns a LinearFF that reuses ``a`` / ``phi`` with a
        frozen ``psi`` at ``action``.
        """
        if not isinstance(belief, LinearFF):
            raise TypeError(
                "NeuralActionLRFF.propagate expects a LinearFF belief, got "
                f"{type(belief).__name__}"
            )
        action = _batched_action(action, self.action_features)
        action = action.to(dtype=self.dtype_device()[0], device=self.dtype_device()[1])

        tol = self.numerical_tolerance
        a = as_vector(self.a(), numerical_tolerance=tol)
        B = as_matrix(self.B(action=action))
        omega_current = as_matrix(self.phi.Omega2(belief.psi))
        omega_next = as_matrix(self.phi.Omega2(self.psi, action=action))
        q = as_vector(omega_next.rev_matvec(a), numerical_tolerance=tol)

        c = as_vector(belief.c(), numerical_tolerance=tol)
        # Apply Z only when the belief renormalizes; otherwise ``c`` already
        # carries the intended (possibly unnormalized) scale.
        if belief._renormalize_h:
            Z = torch.as_tensor(
                belief.norm_constant(), dtype=a.dtype, device=a.device
            )
            c = c * Z
        overlap = as_vector(omega_current.matvec(c), numerical_tolerance=tol)
        c_next = as_vector(B.rev_matvec(a * overlap), numerical_tolerance=tol) / q

        psi_next = _freeze_basis_at_context(self.psi, action=action)
        return LinearFF(
            self.a,
            self.phi,
            _fixed_vector_parameters(c_next, modes=getattr(psi_next, "modes", None)),
            psi_next,
            numerical_tolerance=belief.numerical_tolerance,
            renormalize_h=False,
            register_modules=False,
        )


class NeuralActionQRFF(MarkovDecisionProcess, QuadraticRFF):
    """QuadraticRFF with action-conditioned ``LQ0`` (and typically action-conditioned ``psi``).

    ``LA`` / ``phi`` are static. ``psi`` is supplied by the caller; ``LQ0`` is an
    action MLP. Context flows into ``Omega22`` / normalization during
    ``log_density`` and ``propagate``.
    """

    def __init__(
        self,
        LA: Parameters,
        phi: Basis,
        psi: Basis,
        action_features: int,
        numerical_tolerance: float = 1e-20,
        pd_jitter: float = 1e-6,
        register_modules: bool = True,
        hidden_features: int = 128,
        num_hidden_layers: int = 2,
        activation=torch.nn.Tanh,
        zero_init_last: bool = False,
        positive: bool = False,
    ):
        batch_size = phi.batch_size()
        mn = phi.n_basis_functions() * psi.n_basis_functions()
        out_features = (mn, mn) if batch_size == 1 else (batch_size, mn, mn)
        LQ0 = DenseMatrixParameters(
            NeuralParameters(
                in_features=action_features,
                out_features=out_features,
                batch_shape=(batch_size,),
                context_key="action",
                hidden_features=hidden_features,
                num_hidden_layers=num_hidden_layers,
                activation=activation,
                zero_init_last=zero_init_last,
                positive=positive,
            )
        )
        QuadraticRFF.__init__(
            self,
            LA,
            phi,
            psi,
            LQ0,
            numerical_tolerance=numerical_tolerance,
            pd_jitter=pd_jitter,
            register_modules=register_modules,
        )
        self.action_features = action_features

    def log_density(
        self,
        xp: torch.Tensor,
        *,
        conditioner: torch.Tensor,
        action: torch.Tensor,
        **context: torch.Tensor,
    ):
        return QuadraticRFF.log_density(
            self, xp, conditioner=conditioner, action=action, **context
        )

    def propagate(self, belief: DensityModel, action: torch.Tensor) -> QuadraticFF:
        r"""One-step QuadraticFF propagation at a fixed action.

        Uses ``get_normalization(action=...)`` so ``T`` / ``Q0`` match the
        action-conditioned gram, then returns a QuadraticFF with the same
        ``LA`` / ``phi`` and a frozen ``psi`` at ``action``.
        """
        if not isinstance(belief, QuadraticFF):
            raise TypeError(
                "NeuralActionQRFF.propagate expects QuadraticFF, got "
                f"{type(belief).__name__}"
            )
        if belief.LA is not self.LA or belief.phi is not self.phi:
            raise ValueError(
                "belief must share LA and phi with transition (use QuadraticFF.from_rff)"
            )
        action = _batched_action(action, self.action_features)
        action = action.to(dtype=self.dtype_device()[0], device=self.dtype_device()[1])

        _, _, Q0, _, T = self.get_normalization(action=action)
        V = belief.get_V(normalized=True)
        omega = as_matrix(self.phi.Omega22(belief.psi))
        D = DenseMatrix(
            omega.frobenius_inner(DenseMatrix(V.to_dense()[..., None, None, :, :]))
        )
        weighted_D = (T @ D) @ T.T
        m, n = self.phi.n_basis_functions(), self.psi.n_basis_functions()
        qb = Q0.to_dense().reshape(self.phi.batch_size(), m, n, m, n)
        qb = qb.movedim(-3, -2)  # (batch, m, m, n, n)
        Vnext = DenseMatrix(qb).contract_batch_matrix(weighted_D)

        psi_next = _freeze_basis_at_context(self.psi, action=action)
        return QuadraticFF(
            self.LA,
            self.phi,
            None,
            psi_next,
            V_fixed=Vnext.to_dense().detach().clone(),
            numerical_tolerance=belief.numerical_tolerance,
            pd_jitter=self.pd_jitter,
            renormalize_h=False,
            register_modules=False,
        )
