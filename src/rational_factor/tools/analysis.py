import torch
from copy import deepcopy
from collections.abc import Sequence

from ..models.density_model import DensityModel, ConditionalDensityModel
from ..models.filter import Filter

def mc_integral_box(f, domain_bounds, n_samples=1000, device=None):
    lows = torch.as_tensor(domain_bounds[0], device=device)
    highs = torch.as_tensor(domain_bounds[1], device=device)


    d = lows.numel()
    x = torch.rand(n_samples, d, device=device) * (highs - lows) + lows
    y = f(x).squeeze(-1)  # if needed

    vol = torch.prod(highs - lows)
    return vol * y.mean()

def avg_log_likelihood(
    belief: DensityModel,
    test_data: torch.Tensor,
    weights: torch.Tensor | None = None,
):
    """
    Average log-density of ``test_data`` rows under ``belief``.

    If ``weights`` is ``None`` (default), returns the arithmetic mean of
    ``log p(x_i)``. If ``weights`` is a length-``n`` tensor aligned with
    ``test_data`` rows, returns ``sum_i w_i log p(x_i)`` after renormalizing
    ``weights`` to sum to 1 (e.g. PF particle weights).
    """
    with torch.no_grad():
        belief.eval()
        logp = belief.log_density(test_data)
        if weights is None:
            return logp.mean()
        w = weights.to(device=logp.device, dtype=logp.dtype).reshape(-1)
        if w.shape[0] != logp.shape[0]:
            raise ValueError(
                f"weights length {w.shape[0]} must match number of test rows {logp.shape[0]}"
            )
        w_sum = w.sum()
        if not torch.isfinite(w_sum) or w_sum <= 0:
            raise ValueError("weights must be finite and sum to a positive value")
        w = w / w_sum
        return (w * logp).sum()

def avg_log_filter_score(
    test_traj_data: Sequence[torch.Tensor],
    test_obs_data: Sequence[torch.Tensor | None],
    filter: Filter,
    initial_belief: DensityModel,
):
    """
    Average per-timestep log-filter scores over many simulated trajectories.

    This evaluates filtering accuracy against latent ground-truth states by:
      1) running the filter independently for each trajectory's observation sequence,
      2) scoring true states under each predicted prior and updated posterior belief,
      3) averaging log-scores across trajectories at each time step.

    Args:
        test_traj_data:
            Sequence of length T+1 where `test_traj_data[k]` has shape
            `(n_trajectories, state_dim)` and stores true latent states at step k.
        test_obs_data:
            Sequence of length T where `test_obs_data[k]` has shape
            `(n_trajectories, obs_dim)` (or None for missing observations).
            Entry k corresponds to the transition from state step k to k+1.
        filter:
            Configured `Filter` instance used for inference.
        initial_belief:
            Belief at k=0 (distribution over the latent state).

    Returns:
        prior_scores:
            Tensor of shape `(T,)` where entry k is
            E_i[log p_prior^{(i)}(x_{k+1}^{(i)})].
        posterior_scores:
            Tensor of shape `(T+1,)` where entry k is
            E_i[log p_post^{(i)}(x_k^{(i)})].
    """
    if len(test_traj_data) == 0:
        raise ValueError("test_traj_data must contain at least one timestep")

    n_trajectories = test_traj_data[0].shape[0]
    state_dim = test_traj_data[0].shape[1]
    n_steps = len(test_traj_data) - 1

    if initial_belief.dim != state_dim:
        raise ValueError(
            "initial_belief.dim must match state dimension in test_traj_data"
        )
    if len(test_obs_data) != n_steps:
        raise ValueError(
            "test_obs_data length must be len(test_traj_data) - 1"
        )
    for k, xk in enumerate(test_traj_data):
        if xk.ndim != 2 or xk.shape[0] != n_trajectories or xk.shape[1] != state_dim:
            raise ValueError(
                f"test_traj_data[{k}] must have shape ({n_trajectories}, {state_dim})"
            )
    for k, ok in enumerate(test_obs_data):
        if ok is None:
            continue
        if ok.ndim != 2 or ok.shape[0] != n_trajectories:
            raise ValueError(
                f"test_obs_data[{k}] must have shape ({n_trajectories}, obs_dim) or be None"
            )

    prior_scores = torch.zeros(n_steps, dtype=test_traj_data[0].dtype, device=test_traj_data[0].device)
    posterior_scores = torch.zeros(n_steps + 1, dtype=test_traj_data[0].dtype, device=test_traj_data[0].device)

    with torch.no_grad():
        # Ensure deterministic behavior from modules with train/eval differences.
        filter.transition_model.eval()
        filter.observation_model.eval()
        initial_belief.eval()

        for i in range(n_trajectories):
            trajectory_obs = []
            for k in range(n_steps):
                ok = test_obs_data[k]
                if ok is None:
                    trajectory_obs.append(None)
                else:
                    trajectory_obs.append(ok[i])

            priors_i, posteriors_i = filter.filter(
                initial_belief=deepcopy(initial_belief),
                observations=trajectory_obs,
                return_priors=True,
            )

            if len(priors_i) != n_steps or len(posteriors_i) != n_steps + 1:
                raise RuntimeError(
                    "Filter returned an unexpected number of priors/posteriors"
                )

            for k in range(n_steps):
                x_true_next = test_traj_data[k + 1][i].unsqueeze(0)
                prior_scores[k] += priors_i[k].log_density(x_true_next).squeeze(0)

            for k in range(n_steps + 1):
                x_true = test_traj_data[k][i].unsqueeze(0)
                posterior_scores[k] += posteriors_i[k].log_density(x_true).squeeze(0)

    prior_scores = prior_scores / n_trajectories
    posterior_scores = posterior_scores / n_trajectories
    return prior_scores, posterior_scores


def avg_log_likelihood_under_particle_belief_reference(
    test_traj_data: Sequence[torch.Tensor],
    test_obs_data: Sequence[torch.Tensor | None],
    filter: Filter,
    initial_belief: DensityModel,
    *,
    reference_filter: Filter | None = None,
    reference_initial_belief_fn=None,
    belief_to_reference_space=None,
):
    """
    Evaluate learned filter beliefs against particle-set reference beliefs.

    Required arguments match ``avg_log_filter_score``. The optional keyword
    arguments provide the particle reference generator and conversion from the
    learned belief space to the particle-reference space.

    Returns:
        prior_scores:
            Tensor of shape ``(T,)`` with averaged prior log-likelihoods under
            per-trajectory particle priors.
        posterior_scores:
            Tensor of shape ``(T+1,)`` with averaged posterior log-likelihoods
            under per-trajectory particle posteriors.
    """
    if len(test_traj_data) == 0:
        raise ValueError("test_traj_data must contain at least one timestep")

    n_trajectories = test_traj_data[0].shape[0]
    state_dim = test_traj_data[0].shape[1]
    n_steps = len(test_traj_data) - 1

    if initial_belief.dim != state_dim:
        raise ValueError(
            "initial_belief.dim must match state dimension in test_traj_data"
        )
    if len(test_obs_data) != n_steps:
        raise ValueError(
            "test_obs_data length must be len(test_traj_data) - 1"
        )
    for k, xk in enumerate(test_traj_data):
        if xk.ndim != 2 or xk.shape[0] != n_trajectories or xk.shape[1] != state_dim:
            raise ValueError(
                f"test_traj_data[{k}] must have shape ({n_trajectories}, {state_dim})"
            )
    for k, ok in enumerate(test_obs_data):
        if ok is None:
            continue
        if ok.ndim != 2 or ok.shape[0] != n_trajectories:
            raise ValueError(
                f"test_obs_data[{k}] must have shape ({n_trajectories}, obs_dim) or be None"
            )

    if reference_filter is None:
        reference_filter = filter
    if reference_initial_belief_fn is None:
        reference_initial_belief_fn = lambda _i: deepcopy(initial_belief)  # noqa: E731
    if belief_to_reference_space is None:
        belief_to_reference_space = lambda b: b  # noqa: E731

    scores_dtype = test_traj_data[0].dtype
    scores_device = test_traj_data[0].device
    prior_scores = torch.zeros(n_steps, dtype=scores_dtype, device=scores_device)
    posterior_scores = torch.zeros(n_steps + 1, dtype=scores_dtype, device=scores_device)

    with torch.no_grad():
        filter.transition_model.eval()
        filter.observation_model.eval()
        initial_belief.eval()
        reference_filter.transition_model.eval()
        reference_filter.observation_model.eval()

        for i in range(n_trajectories):
            trajectory_obs = []
            for k in range(n_steps):
                ok = test_obs_data[k]
                if ok is None:
                    trajectory_obs.append(None)
                else:
                    trajectory_obs.append(ok[i])

            learned_priors, learned_posts = filter.filter(
                initial_belief=deepcopy(initial_belief),
                observations=trajectory_obs,
                return_priors=True,
            )
            reference_priors, reference_posts = reference_filter.filter(
                initial_belief=reference_initial_belief_fn(i),
                observations=trajectory_obs,
                return_priors=True,
            )

            if len(learned_priors) != n_steps or len(learned_posts) != n_steps + 1:
                raise RuntimeError("Filter returned unexpected prior/posterior counts")
            if len(reference_priors) != n_steps or len(reference_posts) != n_steps + 1:
                raise RuntimeError("Reference filter returned unexpected prior/posterior counts")

            for k in range(n_steps):
                particle_belief = reference_priors[k]
                b = belief_to_reference_space(learned_priors[k])
                prior_scores[k] += avg_log_likelihood(
                    b, particle_belief.particles, particle_belief.weights
                )
            for k in range(n_steps + 1):
                particle_belief = reference_posts[k]
                b = belief_to_reference_space(learned_posts[k])
                posterior_scores[k] += avg_log_likelihood(
                    b, particle_belief.particles, particle_belief.weights
                )

    prior_scores = prior_scores / n_trajectories
    posterior_scores = posterior_scores / n_trajectories
    return prior_scores, posterior_scores


def check_pdf_valid(pdf : DensityModel | ConditionalDensityModel, domain_bounds, n_samples=1000, atol=0.2, device=None):
    assert isinstance(pdf, DensityModel)
    integral = mc_integral_box(pdf.forward, domain_bounds, n_samples, device=device)
    err = abs(float(integral) - 1.0)
    if err < atol:
        print(f"   Check PDF:  Integral over x (MC): {integral}")
        return True
    else:
        print(f"   Check PDF:  Integral over x (MC): {integral} (INVALID PDF)")
        return False

def check_conditional_pdf_valid(pdf : ConditionalDensityModel, domain_bounds, conditioner_domain_bounds, n_samples=1000, n_conditioner_samples=10, device=None):
    assert isinstance(pdf, ConditionalDensityModel)
    print("Testing conditional density model...")
    if device is None:
        device = next(pdf.parameters()).device
    domain_lows = torch.as_tensor(domain_bounds[0], device=device)
    domain_highs = torch.as_tensor(domain_bounds[1], device=device)
    domain_bounds_dev = (domain_lows, domain_highs)
    conditioner_lows = torch.as_tensor(conditioner_domain_bounds[0], device=device)
    conditioner_highs = torch.as_tensor(conditioner_domain_bounds[1], device=device)
    conditioner_samples = torch.rand(n_conditioner_samples, conditioner_lows.numel(), device=device, dtype=conditioner_lows.dtype) * (conditioner_highs - conditioner_lows) + conditioner_lows
    print("num samples: ", n_samples)
    print("num conditioner samples: ", n_conditioner_samples)
    integrals = []
    with torch.no_grad():
        pdf.eval()
        for i in range(n_conditioner_samples):
            c = conditioner_samples[i]

            def density_cond(x, conditioner=c):
                y = conditioner.unsqueeze(0).expand(x.shape[0], -1)
                return pdf.forward(x, conditioner=y)

            integral = mc_integral_box(density_cond, domain_bounds_dev, n_samples, device=device)
            integrals.append(integral)

    stacked = torch.stack(integrals)
    print(f"   Check Conditional PDF:  Integral over x (MC) — mean: {stacked.mean().item()}, std: {stacked.std(unbiased=False).item()}")
