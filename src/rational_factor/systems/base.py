import torch
from abc import ABC, abstractmethod

class DiscreteTimeStochasticSystem(ABC):
    def __init__(self, dim : int, v_dist = None):
        self._dim = dim

        if v_dist is not None:
            self._v_dist = v_dist
        else:
            def uniform():
                return torch.rand(self._dim)
            self._v_dist = uniform

    @abstractmethod
    def next_state(self, x : torch.Tensor, v : torch.Tensor):
        """
        Args:
            x : current state
            v : realization of the noise parameters
        """
        pass

    def __call__(self, x : torch.Tensor):
        v = self._v_dist()  # Sample a random v to be plugged into the difference function
        return self.next_state(x, v)

    def dim(self):
        return self._dim



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
    
    return torch.hstack((x_data, xp_data))

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