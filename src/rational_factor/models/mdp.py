from __future__ import annotations

import torch

from rational_factor.models.density_model import DensityModel, MarkovDecisionProcess
from rational_factor.models.parameters import DenseMatrixParameters, NeuralParameters, Parameters
from rational_factor.models.factor_forms import LinearRFF, QuadraticRFF
from rational_factor.models.basis_functions import Basis


class NeuralActionLRFF(MarkovDecisionProcess, LinearRFF):
    """LinearRFF with action-conditioned ``B`` (and typically action-conditioned ``psi``).

    ``a`` / ``phi`` are static. ``psi`` is supplied by the caller (usually a
    GaussianBasis whose parameters are :class:`NeuralParameters`). ``B`` is an
    action MLP. ``log_density`` passes ``action`` so ``Omega2``, ``psi``, and
    ``B`` batch with the data while ``a`` / ``phi`` broadcast.
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
        batch_size = phi.batch_size()
        n_basis = phi.n_basis_functions()
        out_features = (
            (n_basis, n_basis) if batch_size == 1 else (batch_size, n_basis, n_basis)
        )
        B = DenseMatrixParameters(
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

    def propagate(self, belief: DensityModel, action: torch.Tensor) -> DensityModel:
        raise NotImplementedError("NeuralActionLRFF.propagate not implemented")


class NeuralActionQRFF(MarkovDecisionProcess, QuadraticRFF):
    """QuadraticRFF with action-conditioned ``LQ0`` (and typically action-conditioned ``psi``).

    ``LA`` / ``phi`` are static. ``psi`` is supplied by the caller; ``LQ0`` is an
    action MLP. Context flows into ``Omega22`` during normalization.
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

    def propagate(self, belief: DensityModel, action: torch.Tensor) -> DensityModel:
        raise NotImplementedError("NeuralActionQRFF.propagate not implemented")
