from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from .density_model import ConditionalDensityModel, DensityModel


PropagateAndUpdateFn = Callable[
    [DensityModel, ConditionalDensityModel, ConditionalDensityModel, list[torch.Tensor | None]],
    tuple[list[DensityModel], list[DensityModel]],
]


class Filter(torch.nn.Module):
    """
    Thin wrapper around a concrete `propagate_and_update` implementation.

    The supplied function is responsible for the filtering algorithm itself
    (e.g. particle filter, closed-form factor filter), while this class provides
    consistent input validation and invocation ergonomics.
    """

    def __init__(
        self,
        transition_model: ConditionalDensityModel,
        observation_model: ConditionalDensityModel,
        prop_and_upd_fn: PropagateAndUpdateFn,
    ):
        super().__init__()
        if not callable(prop_and_upd_fn):
            raise TypeError("prop_and_upd_fn must be callable")
        if transition_model.conditioner_dim != transition_model.dim:
            raise ValueError(
                "transition_model must map previous state to current state "
                "(conditioner_dim == dim)."
            )
        if observation_model.conditioner_dim != transition_model.dim:
            raise ValueError(
                "observation_model.conditioner_dim must match state dimension "
                "of transition_model."
            )

        self.transition_model = transition_model
        self.observation_model = observation_model
        self.prop_and_upd_fn = prop_and_upd_fn

    def filter(
        self,
        initial_belief: DensityModel,
        observations: Sequence[torch.Tensor | None],
        *,
        return_priors: bool = True,
    ) -> tuple[list[DensityModel], list[DensityModel]] | list[DensityModel]:
        """
        Run filtering over a sequence of observations.

        Args:
            initial_belief: belief at timestep 0 over the latent state.
            observations: observations for timesteps 1..T. A value of `None`
                means "predict only" at that timestep.
            return_priors: if True, return (priors, posteriors). If False,
                return posteriors only.
        """
        if initial_belief.dim != self.transition_model.conditioner_dim:
            raise ValueError(
                "initial_belief.dim must match transition_model.conditioner_dim."
            )
        if not isinstance(observations, Sequence):
            raise TypeError("observations must be a sequence of tensors or None")

        obs_list = list(observations)
        priors, posteriors = self.prop_and_upd_fn(
            initial_belief,
            self.transition_model,
            self.observation_model,
            obs_list,
        )
        if return_priors:
            return priors, posteriors
        return posteriors
