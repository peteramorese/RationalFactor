from rational_factor.models.parameters import TrainableParameters
from rational_factor.tools.set import Hyperrectangle
import torch


class TrainableController(TrainableParameters):
    """Sequence of controls as trainable (or fixed) parameters.

    When ``action_feature_limits`` is set, trainable values are mapped into
    ``[low, high]`` via a sigmoid; fixed values must already lie in that box.
    Last axis of the parameter tensor is the action-feature dimension.
    """

    def __init__(
        self,
        trainable_init_values: torch.Tensor = None,
        fixed_values: torch.Tensor = None,
        action_feature_limits: Hyperrectangle = None,
    ):
        super().__init__(trainable_init_values=trainable_init_values, fixed_values=fixed_values)
        self.action_feature_limits = action_feature_limits
        if action_feature_limits is not None:
            if action_feature_limits.is_batched:
                raise ValueError(
                    "action_feature_limits must be a single Hyperrectangle, "
                    f"got batched shape {tuple(action_feature_limits.batch_shape)}"
                )
            if self._p.shape[-1] != action_feature_limits.dim:
                raise ValueError(
                    f"last dim of controls must match action_feature_limits.dim "
                    f"({action_feature_limits.dim}), got {self._p.shape[-1]}"
                )
            if not self._trainable:
                low, high = action_feature_limits.low, action_feature_limits.high
                assert torch.all(fixed_values >= low) and torch.all(fixed_values <= high), (
                    "fixed_values must lie within action_feature_limits"
                )

    @classmethod
    def random_init(
        cls,
        action_features: int,
        timesteps: int,
        trainable: bool = True,
        mean: float = 0.0,
        std: float = 1.0,
        action_feature_limits: Hyperrectangle = None,
    ):
        values = torch.randn(timesteps, action_features) * std + mean
        if trainable:
            return cls(
                trainable_init_values=values,
                action_feature_limits=action_feature_limits,
            )
        return cls(fixed_values=values, action_feature_limits=action_feature_limits)

    @classmethod
    def set_init(
        cls,
        timesteps: int,
        action_features: int,
        value: float,
        trainable: bool = True,
        action_feature_limits: Hyperrectangle = None,
    ):
        values = torch.ones(timesteps, action_features) * value
        if trainable:
            return cls(
                trainable_init_values=values,
                action_feature_limits=action_feature_limits,
            )
        return cls(fixed_values=values, action_feature_limits=action_feature_limits)

    @classmethod
    def from_values(
        cls,
        values: torch.Tensor,
        trainable: bool = True,
        action_feature_limits: Hyperrectangle = None,
    ):
        if trainable:
            return cls(
                trainable_init_values=values,
                action_feature_limits=action_feature_limits,
            )
        return cls(
            fixed_values=values.detach().clone(),
            action_feature_limits=action_feature_limits,
        )

    def _bound(self, p: torch.Tensor) -> torch.Tensor:
        limits = self.action_feature_limits
        low = limits.low.to(dtype=p.dtype, device=p.device)
        high = limits.high.to(dtype=p.dtype, device=p.device)
        return low + (high - low) * torch.sigmoid(p)

    def forward(self, timestep: int | None = None, **context) -> torch.Tensor:
        """Return control at ``timestep``, or the full ``(T, action_features)`` sequence."""
        if timestep is None:
            if self._trainable and self.action_feature_limits is not None:
                return self._bound(self._p)
            return self._p
        if self._trainable:
            if self.action_feature_limits is not None:
                return self._bound(self._p[timestep])
            return self._p[timestep]
        return self._p[timestep]

    def __call__(self, timestep: int | None = None, **context) -> torch.Tensor:
        return torch.nn.Module.__call__(self, timestep, **context)

    def n_timesteps(self):
        return self._p.shape[0]

    def freeze_params(self):
        return TrainableController(
            fixed_values=self.forward().detach().clone(),
            action_feature_limits=self.action_feature_limits,
        )

    def with_fixed_values(self, values: torch.Tensor) -> "TrainableController":
        return TrainableController(
            fixed_values=values.detach().clone(),
            action_feature_limits=self.action_feature_limits,
        )

    def get_action_feature_limits(self):
        return self.action_feature_limits
