import torch
from rational_factor.models.density_model import ConditionalDensityModel
from rational_factor.tools.set import Hyperrectangle

class DiscreteTimeStochasticSystem(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        state_labels: list[str] = None,
        v_dist: torch.distributions.Distribution | None = None,
        support: Hyperrectangle | None = None,
        bounded_support: bool = False,
        wrap_dims: tuple[int, ...] = (),
        boundary_margin_frac: float = 0.1,
        boundary_min_scale: float = 0.25,
        boundary_restitution: float = 0.5,
    ):
        super().__init__()
        self._dim = dim
        if state_labels is not None:
            assert len(state_labels) == dim, "state_labels must be a list of length dim"
            self._state_labels = state_labels
        else:
            self._state_labels = [f"x{i}" for i in range(dim)]

        if v_dist is not None:
            self._v_dist = v_dist
        else:
            self._v_dist = torch.distributions.Uniform(
                low=torch.zeros(self._dim),
                high=torch.ones(self._dim),
            )

        if support is not None:
            if not isinstance(support, Hyperrectangle) or support.is_batched or support.dim != dim:
                raise ValueError(f"support must be a single {dim}D Hyperrectangle")
        if bounded_support and support is None:
            raise ValueError("support must be provided when bounded_support=True")
        self.support = support
        self.bounded_support = bool(bounded_support)
        self._wrap_dims = tuple(int(d) for d in wrap_dims)
        self._boundary_margin_frac = float(boundary_margin_frac)
        self._boundary_min_scale = float(boundary_min_scale)
        self._boundary_restitution = float(boundary_restitution)

    def _apply_support(self, x: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
        """Project ``x_next`` onto ``self.support`` when ``self.bounded_support`` is True.

        Interior transitions are untouched. Near the walls of non-angular
        coordinates, outward steps are damped and any overshoot is reflected
        back into ``self.support`` with damped magnitude. Angular coordinates
        in ``self._wrap_dims`` wrap periodically over their support interval.
        """
        if not self.bounded_support:
            return x_next
        if self.support is None:
            raise ValueError("support must be defined when bounded_support=True")

        lo = self.support.low.to(device=x_next.device, dtype=x_next.dtype)
        hi = self.support.high.to(device=x_next.device, dtype=x_next.dtype)
        span = hi - lo

        x_prev = x.reshape_as(x_next)
        dx = x_next - x_prev
        margin = self._boundary_margin_frac * span

        dist_hi = (hi - x_prev).clamp(min=0.0)
        dist_lo = (x_prev - lo).clamp(min=0.0)
        near_hi = (dx > 0) & (dist_hi < margin)
        near_lo = (dx < 0) & (dist_lo < margin)

        min_scale = self._boundary_min_scale
        scale = torch.ones_like(dx)
        scale = torch.where(near_hi, min_scale + (1.0 - min_scale) * (dist_hi / margin), scale)
        scale = torch.where(near_lo, min_scale + (1.0 - min_scale) * (dist_lo / margin), scale)
        y = x_prev + scale * dx

        restitution = self._boundary_restitution
        over_hi = (y - hi).clamp(min=0.0)
        over_lo = (lo - y).clamp(min=0.0)
        y_refl = torch.where(
            y > hi,
            hi - 0.5 * span * torch.tanh(2.0 * restitution * over_hi / span),
            torch.where(
                y < lo,
                lo + 0.5 * span * torch.tanh(2.0 * restitution * over_lo / span),
                y,
            ),
        )
        y_refl = torch.clamp(y_refl, min=lo, max=hi)

        if len(self._wrap_dims) > 0:
            y_wrap = torch.clamp(lo + torch.remainder(x_next - lo, span), min=lo, max=hi)
            wrap_mask = torch.zeros(self._dim, dtype=torch.bool, device=x_next.device)
            wrap_mask[list(self._wrap_dims)] = True
            return torch.where(wrap_mask, y_wrap, y_refl)

        return y_refl

    def _sample_v(self):
        if isinstance(self._v_dist, torch.distributions.Distribution):
            return self._v_dist.sample()
        if callable(self._v_dist):
            return self._v_dist()
        raise TypeError("v_dist must be a torch.distributions.Distribution or callable")

    def next_state(self, x : torch.Tensor, v : torch.Tensor):
        """
        Args:
            x : current state
            v : realization of the noise parameters
        """
        raise NotImplementedError("next_state not implemented")

    def forward(self, x : torch.Tensor):
        v = self._sample_v()  # Sample a random v to be plugged into the difference function
        return self.next_state(x, v.to(device=x.device, dtype=x.dtype))

    def dim(self):
        return self._dim
    
    def state_label(self, i : int):
        return self._state_labels[i]

class ControllableStochasticSystem(DiscreteTimeStochasticSystem):
    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        state_labels: list[str] = None,
        control_labels: list[str] = None,
        v_dist: torch.distributions.Distribution | None = None,
        support: Hyperrectangle | None = None,
        bounded_support: bool = False,
        wrap_dims: tuple[int, ...] = (),
    ):
        # Call DiscreteTimeStochasticSystem directly: cooperative subclasses also inherit
        # a passive plant (e.g. VanDerPol) whose __init__ signature differs from ours.
        DiscreteTimeStochasticSystem.__init__(
            self,
            state_dim,
            state_labels,
            v_dist,
            support=support,
            bounded_support=bounded_support,
            wrap_dims=wrap_dims,
        )
        self._control_dim = control_dim
        if control_labels is not None:
            assert len(control_labels) == control_dim, "control_labels must be a list of length control_dim"
            self._control_labels = control_labels
        else:
            self._control_labels = [f"u{i}" for i in range(control_dim)]

    def next_state(self, x : torch.Tensor, u : torch.Tensor, v : torch.Tensor):
        """
        Args:
            x : current state
            u : control input
            v : realization of the noise parameters
        """
        raise NotImplementedError("next_state not implemented")

    def forward(self, x : torch.Tensor, u : torch.Tensor):
        v = self._sample_v()  # Sample a random v to be plugged into the difference function
        return self.next_state(x, u, v.to(device=x.device, dtype=x.dtype))

    def control_dim(self):
        return self._control_dim

    def control_label(self, i : int):
        return self._control_labels[i]

class PartiallyObservableSystem(DiscreteTimeStochasticSystem):
    def __init__(
        self,
        state_dim: int,
        observation_dim: int,
        state_labels: list[str] = None,
        observation_labels: list[str] = None,
        v_dist: torch.distributions.Distribution | None = None,
        w_dist: torch.distributions.Distribution | None = None,
        support: Hyperrectangle | None = None,
        bounded_support: bool = False,
        wrap_dims: tuple[int, ...] = (),
    ):
        super().__init__(
            state_dim,
            state_labels,
            v_dist,
            support=support,
            bounded_support=bounded_support,
            wrap_dims=wrap_dims,
        )
        self._observation_dim = observation_dim
        if observation_labels is not None:
            assert len(observation_labels) == observation_dim, "observation_labels must be a list of length observation_dim"
            self._observation_labels = observation_labels
        else:
            self._observation_labels = [f"o{i}" for i in range(observation_dim)]

        if w_dist is not None:
            self._w_dist = w_dist
        else:
            self._w_dist = torch.distributions.Uniform(
                low=torch.zeros(self._observation_dim),
                high=torch.ones(self._observation_dim),
            )

    def _sample_w(self):
        if isinstance(self._w_dist, torch.distributions.Distribution):
            return self._w_dist.sample()
        if callable(self._w_dist):
            return self._w_dist()
        raise TypeError("w_dist must be a torch.distributions.Distribution or callable")

    def observe(self, x : torch.Tensor):
        raise NotImplementedError("observe not implemented")

    def log_observation_likelihood(self, x : torch.Tensor, o : torch.Tensor):
        raise NotImplementedError("log_observation_likelihood not implemented")

    def observation_dim(self):
        return self._observation_dim

    def state_label(self, i : int):
        return self._state_labels[i]

    def observation_label(self, i : int):
        return self._observation_labels[i]

class SystemTransitionDistribution(ConditionalDensityModel):
    """
    Conditional distribution representation of the stochastic system
    """
    def __init__(self, system : DiscreteTimeStochasticSystem):
        super().__init__(system.dim(), system.dim())
        self._system = system

    def sample(self, x : torch.Tensor):
        if x.ndim == 1:
            return self._system(x)
        if x.ndim == 2:
            return torch.stack([self._system(xi) for xi in x], dim=0)
        raise ValueError("x must have shape (dim,) or (n_samples, dim)")

class SystemObservationDistribution(ConditionalDensityModel):
    """
    Conditional distribution representation of the observation model
    """
    def __init__(self, system : PartiallyObservableSystem):
        super().__init__(system.observation_dim(), system.dim())
        self._system = system

    def log_density(self, o: torch.Tensor, *, conditioner: torch.Tensor):
        x = conditioner
        return self._system.log_observation_likelihood(x, o)
    
    def sample(self, conditioner: torch.Tensor):
        x = conditioner
        return self._system.observe(x)


def simulate(system, initial_state_sampler, n_timesteps: int, device : torch.device = None):
    assert isinstance(system, (DiscreteTimeStochasticSystem, PartiallyObservableSystem)), "System must be a DiscreteTimeStochasticSystem or PartiallyObservableSystem"
    true_states = []

    x = initial_state_sampler(1)[0]
    true_states.append(x.clone())

    for _ in range(n_timesteps):
        x = system(x)
        true_states.append(x.clone())
    
    if isinstance(system, PartiallyObservableSystem):
        observations = []
        # (no observation for the initial state x_0).
        for true_state in true_states[1:]:
            observations.append(system.observe(true_state).clone())
        return torch.stack(true_states).to(device=device), torch.stack(observations).to(device=device)
    else:
        return torch.stack(true_states).to(device=device)

def sample_trajectories(system : DiscreteTimeStochasticSystem | ControllableStochasticSystem, initial_state_sampler, n_timesteps : int, n_trajectories : int, controller = None):
    """
    Sample trajectory data from a system under an initial state distribution

    Args:
        system : system model
        initial_state_sampler : callable that takes in an integer n and returns n randomly sampled initial states
        n_timesteps : time horizon of each trajectory
        n_trajectories : number of trajectories to sample
        controller : callable that takes in a time step k and returns the control input u_k applied when advancing x_k -> x_{k+1}, default is None

    Returns:
        traj_data (list) : list of length n_timesteps marginal data sets indexed by time step
    """
    if controller is not None:
        assert isinstance(system, ControllableStochasticSystem), "System must be a ControllableStochasticSystem if controller is not None"

    dim = system.dim()
    traj_data = [torch.zeros((n_trajectories, dim)) for _ in range(n_timesteps + 1)]


    # Sample initial conditions
    traj_data[0] = initial_state_sampler(n_trajectories)

    for k in range(n_timesteps):
        for i in range(n_trajectories):
            xk = traj_data[k][i, :]
            if controller is None:
                xkp1 = system(xk)
            else:
                u_k = controller(k)
                xkp1 = system(xk, u_k)
            traj_data[k + 1][i, :] = xkp1

    return traj_data


def sample_trajectories_without_collisions(
    system: DiscreteTimeStochasticSystem | ControllableStochasticSystem,
    initial_state_sampler,
    obstacles,
    n_timesteps: int,
    n_trajectories: int,
    controller=None,
):
    """Sample trajectories, dropping any particle that enters ``obstacles``.

    At each time index, only particles that do not lie in any hyperrectangle
    are retained and (for non-terminal steps) propagated. Particles that
    collide are removed from all subsequent marginals, so each entry of the
    returned list may have a different leading size.

    Args:
        system : system model
        initial_state_sampler : callable ``n -> (n, dim)`` initial states
        obstacles : ``Hyperrectangle`` (single or batched boxes)
        n_timesteps : time horizon of each trajectory
        n_trajectories : number of trajectories to attempt from the initial sampler
        controller : optional ``k -> u_k`` control law for controllable systems

    Returns:
        traj_data (list) : length ``n_timesteps + 1`` of ``(n_alive_t, dim)`` tensors
    """
    if not isinstance(obstacles, Hyperrectangle):
        raise TypeError(
            f"obstacles must be a Hyperrectangle, got {type(obstacles).__name__}"
        )
    if controller is not None:
        assert isinstance(system, ControllableStochasticSystem), (
            "System must be a ControllableStochasticSystem if controller is not None"
        )

    dim = system.dim()
    x = initial_state_sampler(n_trajectories)
    free = ~obstacles.contains_any(x)
    x = x[free]
    traj_data = [x.clone()]

    for k in range(n_timesteps):
        n_alive = x.shape[0]
        if n_alive == 0:
            traj_data.append(torch.zeros((0, dim), dtype=x.dtype, device=x.device))
            continue

        if controller is None:
            x_next = torch.stack([system(x[i]) for i in range(n_alive)], dim=0)
        else:
            u_k = controller(k)
            x_next = torch.stack([system(x[i], u_k) for i in range(n_alive)], dim=0)

        free = ~obstacles.contains_any(x_next)
        x = x_next[free]
        traj_data.append(x.clone())

    return traj_data

def sample_io_pairs(system : DiscreteTimeStochasticSystem, prev_state_sampler, n_pairs : int):
    """
    Sample input (x, x') pairs from a system model, where the starting state x is sampled from prev_state_sampler

    Args:
        system : system model 
        prev_state_sampler : callable that takes in an integer n and returns n randomly sampled previous states
        n_pairs : number of input (x, x') pairs to sample
    """
    x_data = prev_state_sampler(n_pairs)
    xp_data = torch.zeros_like(x_data)
    for i in range(n_pairs):
        xp_data[i, :] = system(x_data[i, :])
    
    return x_data, xp_data

def sample_uio_tuples(system : ControllableStochasticSystem, prev_state_sampler, control_sampler, n_pairs : int):
    """
    Sample input (u, x, x') tuples from a system model, where the starting state x is sampled from prev_state_sampler
    and the control input u is sampled from control_sampler
    """
    u_data = control_sampler(n_pairs)
    x_data = prev_state_sampler(n_pairs)
    xp_data = torch.zeros_like(x_data)
    for i in range(n_pairs):
        xp_data[i, :] = system(x_data[i, :], u_data[i, :])
    return u_data, x_data, xp_data # (u_k, x_k, x_{k+1}) tuples

def sample_observation_pairs(system : PartiallyObservableSystem, state_sampler, n_pairs : int):
    """
    Sample observation (o, x) pairs from a system model, where the state x is sampled uniformly from a specified region

    Args:
        system : system model
        state_sampler : callable that takes in an integer n and returns n randomly sampled states
        n_pairs : number of observation (o, x) pairs to sample
    """
    x_data = state_sampler(n_pairs)
    o_data = torch.zeros_like(x_data)
    for i in range(n_pairs):
        o_data[i, :] = system.observe(x_data[i, :])
    
    return x_data, o_data

def create_transition_data_matrix(trajectory_data, separate=False):
    """
    Create a data matrix of (x_k, x_{k+1}) pairs from trajectory data.

    Args:
        trajectory_data : list of torch.Tensor
            A list of length k, where each element is a (p x n) array.
            Each array contains p samples from the n-dimensional state distribution at time step k.
        separate : bool
            Specify if the x' data should be separated as two separate return arguments x_k, x_{k+1}

    Returns:
    data_matrix : torch.Tensor
        A ((k-1) * p) x (2n) array where each row is a pair (x_k, x_{k+1}) if separarate is True, 
        else two ((k-1) * p) x (n) arrays for (x_k) and (x_{k+1}) respectively.
    """
    k = len(trajectory_data)
    if k < 2:
        raise ValueError("Need at least two time steps to form transition pairs.")
    
    p, n = trajectory_data[0].shape

    data_matrix = torch.empty(((k - 1) * p, 2 * n))

    for i in range(k - 1):
        x_k = trajectory_data[i]      # shape: (p, n)
        x_kp1 = trajectory_data[i+1]  # shape: (p, n)
        data_matrix[i * p : (i + 1) * p, :] = torch.hstack((x_k, x_kp1))

    if separate:
        return data_matrix[:, :n], data_matrix[:, n:] 
    else:
        return data_matrix 