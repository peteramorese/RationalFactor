import torch
from rational_factor.models.density_model import ConditionalDensityModel

class DiscreteTimeStochasticSystem(torch.nn.Module):
    def __init__(self, dim : int, v_dist : torch.distributions.Distribution | None = None):
        super().__init__()
        self._dim = dim

        if v_dist is not None:
            self._v_dist = v_dist
        else:
            self._v_dist = torch.distributions.Uniform(
                low=torch.zeros(self._dim),
                high=torch.ones(self._dim),
            )

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

class PartiallyObservableSystem(DiscreteTimeStochasticSystem):
    def __init__(self, state_dim : int, observation_dim : int, v_dist : torch.distributions.Distribution | None = None, w_dist : torch.distributions.Distribution | None = None):
        super().__init__(state_dim, v_dist)
        self._observation_dim = observation_dim

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

    def log_density(self, o : torch.Tensor, *, conditioner : torch.Tensor):
        x = conditioner
        return self._system.log_observation_likelihood(x, o)
    
    def sample(self, conditioner : torch.Tensor):
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

def sample_trajectories(system : DiscreteTimeStochasticSystem, initial_state_sampler, n_timesteps : int, n_trajectories : int):
    """
    Sample trajectory data from a system under an initial state distribution

    Args:
        system : system model
        initial_state_sampler : callable that takes in an integer n and returns n randomly sampled initial states
        n_timesteps : time horizon of each trajectory
        n_trajectories : number of trajectories to sample

    Returns:
        traj_data (list) : list of length n_timesteps marginal data sets indexed by time step
    """

    dim = system.dim()
    traj_data = [torch.zeros((n_trajectories, dim)) for _ in range(n_timesteps + 1)]

    # Sample initial conditions
    traj_data[0] = initial_state_sampler(n_trajectories)

    for k in range(n_timesteps):
        for i in range(n_trajectories):
            xk = traj_data[k][i, :]
            xkp1 = system(xk)
            traj_data[k + 1][i, :] = xkp1

    return traj_data

def sample_io_pairs(system : DiscreteTimeStochasticSystem, prev_state_sampler, n_pairs : int):
    """
    Sample input (x', x) pairs from a system model, where the starting state x is sampled uniformly from a specified region

    Args:
        system : system model 
        prev_state_sampler : callable that takes in an integer n and returns n randomly sampled previous states
        n_pairs : number of input (x', x) pairs to sample
    """
    x_data = prev_state_sampler(n_pairs)
    xp_data = torch.zeros_like(x_data)
    for i in range(n_pairs):
        xp_data[i, :] = system(x_data[i, :])
    
    return x_data, xp_data

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