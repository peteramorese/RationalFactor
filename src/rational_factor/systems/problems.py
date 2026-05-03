from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from rational_factor.systems import systems, po_systems
from rational_factor.systems.base import DiscreteTimeStochasticSystem, PartiallyObservableSystem, sample_io_pairs, sample_observation_pairs, sample_trajectories
from rational_factor.tools.misc import make_mvnormal_state_sampler, make_uniform_state_sampler


@dataclass
class FullyObservableProblem:
    system: DiscreteTimeStochasticSystem
    initial_state_sampler: Callable[[int], torch.Tensor]
    prev_state_sampler: Callable[[int], torch.Tensor]
    n_timesteps: int
    n_trajectories_test: int
    n_data_tran: int
    n_data_init: int
    seed: int
    numerical_tolerance: float
    plot_bounds_low: torch.Tensor
    plot_bounds_high: torch.Tensor
    plot_marginals_list: list[tuple[int, int]]

    def train_initial_state_data(self):
        with torch.random.fork_rng():
            if self.seed is not None:
                torch.manual_seed(self.seed)
            x0_data = self.initial_state_sampler(self.n_data_init)
            return x0_data

    def train_state_transition_data(self):
        with torch.random.fork_rng():
            if self.seed is not None:
                torch.manual_seed(self.seed)
            x_k_data, x_kp1_data = sample_io_pairs(self.system, self.prev_state_sampler, n_pairs=self.n_data_tran)
            return x_k_data, x_kp1_data    
    
    def test_data(self):
        with torch.random.fork_rng():
            if self.seed is not None:
                torch.manual_seed(self.seed + 1)
            traj_data = sample_trajectories(self.system, self.initial_state_sampler, n_timesteps=self.n_timesteps, n_trajectories=self.n_trajectories_test)
            return traj_data

@dataclass
class PartiallyObservableProblem(FullyObservableProblem):
    obs_state_sampler: Callable[[int], torch.Tensor]
    n_data_obs: int

    def train_obs_data(self):
        with torch.random.fork_rng():
            if self.seed is not None:
                torch.manual_seed(self.seed + 2)
            x_data, o_data = sample_observation_pairs(self.system, self.obs_state_sampler, n_pairs=self.n_data_obs)
            return x_data, o_data    

    def test_data(self):
        with torch.random.fork_rng():
            if self.seed is not None:
                torch.manual_seed(self.seed + 3)
            traj_data = sample_trajectories(self.system, self.obs_state_sampler, n_timesteps=self.n_timesteps, n_trajectories=self.n_trajectories_test)
            obs_data = [self.system.observe(curr_states) for curr_states in traj_data]
            return traj_data, obs_data


# Aircraft training: diagonal covariances are diag(std**2) per axis. The previous-state marginal
# for transition data should be at least as wide as the initial-state marginal (cf. `quadcopter`)
# so the transition model sees a representative envelope, not a tighter subset of the state box.
_AIRCRAFT_INIT_STD = torch.tensor(
    [100.0, 100.0, 32.0, 8.0, 3.5, 5.5, 0.06, 0.08, 0.35, 0.45, 0.45, 0.45],
    dtype=torch.float32,
)
_AIRCRAFT_PREV_STD = torch.tensor(
    [340.0, 340.0, 95.0, 30.0, 20.0, 20.0, 0.55, 0.52, 2.2, 3.0, 3.0, 3.0],
    dtype=torch.float32,
)

# Per-axis SI → dimensionless state: x_phys = x_norm * scale. Keeps typical states near [-10, 10]
# while dynamics, waypoint, and V_ref remain in physical units inside `Aircraft`.
_AIRCRAFT_STATE_SCALE = torch.tensor(
    [140.0, 110.0, 26.0, 4.5, 4.5, 4.0, 1.0, 1.0, 1.0, 0.55, 0.55, 0.55],
    dtype=torch.float32,
)
_AIRCRAFT_INIT_MEAN_PHYS = torch.tensor(
    [0.0, 0.0, 100.0, 21.5, 0.0, -1.2, 0.0, 0.06, 0.45, 0.0, 0.0, 0.0],
    dtype=torch.float32,
)
_AIRCRAFT_PREV_MEAN_PHYS = torch.tensor(
    [150.0, 80.0, 95.0, 20.0, 0.0, -1.0, 0.0, 0.05, 0.4, 0.0, 0.0, 0.0],
    dtype=torch.float32,
)
_AIRCRAFT_INIT_MEAN = _AIRCRAFT_INIT_MEAN_PHYS / _AIRCRAFT_STATE_SCALE
_AIRCRAFT_PREV_MEAN = _AIRCRAFT_PREV_MEAN_PHYS / _AIRCRAFT_STATE_SCALE
_AIRCRAFT_INIT_STD_NORM = _AIRCRAFT_INIT_STD / _AIRCRAFT_STATE_SCALE
_AIRCRAFT_PREV_STD_NORM = _AIRCRAFT_PREV_STD / _AIRCRAFT_STATE_SCALE
_AIRCRAFT_PLOT_LOW_PHYS = torch.tensor(
    [-300.0, -300.0, 10.0, 5.0, -25.0, -25.0, -1.0, -0.75, -3.2, -3.5, -3.5, -3.5],
    dtype=torch.float32,
)
_AIRCRAFT_PLOT_HIGH_PHYS = torch.tensor(
    [1400.0, 1100.0, 260.0, 42.0, 25.0, 25.0, 1.0, 0.75, 3.2, 3.5, 3.5, 3.5],
    dtype=torch.float32,
)
_AIRCRAFT_PLOT_LOW = _AIRCRAFT_PLOT_LOW_PHYS / _AIRCRAFT_STATE_SCALE
_AIRCRAFT_PLOT_HIGH = _AIRCRAFT_PLOT_HIGH_PHYS / _AIRCRAFT_STATE_SCALE

FULLY_OBSERVABLE_PROBLEMS = {
    "scalar_nonlinear_drift": FullyObservableProblem(
        system=systems.ScalarNonlinearDrift(
            sigma=0.1,
        ),
        initial_state_sampler=make_mvnormal_state_sampler(
            mean=torch.tensor([0.0]),
            covariance=torch.diag(torch.tensor([0.5])),
        ),
        #prev_state_sampler=make_uniform_state_sampler(
        #    low=torch.tensor([-4.0]),
        #    high=torch.tensor([4.0]),
        #),
        prev_state_sampler=make_mvnormal_state_sampler(
            mean=torch.tensor([0.0]),
            covariance=torch.diag(torch.tensor([4.0])),
        ),
        n_timesteps=10,
        n_trajectories_test=5000,
        n_data_tran=10000,
        n_data_init=1000,
        seed=42,
        numerical_tolerance=1e-20,
        plot_bounds_low=torch.tensor([-4.0]),
        plot_bounds_high=torch.tensor([4.0]),
        plot_marginals_list=[],
    ),
    "van_der_pol": FullyObservableProblem(
        system=systems.VanDerPol(
            dt=0.3,
            mu = 0.9,
            covariance=0.1 * torch.eye(2),
        ),
        initial_state_sampler=make_mvnormal_state_sampler(
            mean=torch.tensor([0.2, 0.1]),
            covariance=torch.diag(torch.tensor([0.2, 0.2])),
        ),
        prev_state_sampler=make_mvnormal_state_sampler(
            mean=torch.tensor([0.0, 0.0]),
            covariance=torch.diag(2.0 * torch.ones(2)),
        ),
        #prev_state_sampler=make_uniform_state_sampler(
        #    low=torch.tensor([-5.0, -5.0]),
        #    high=torch.tensor([5.0, 5.0]),
        #),
        n_timesteps=10,
        n_trajectories_test=5000,
        n_data_tran=10000,
        n_data_init=1000,
        seed=42,
        numerical_tolerance=1e-20,
        plot_bounds_low=torch.tensor([-5.0, -5.0]),
        plot_bounds_high=torch.tensor([5.0, 5.0]),
        plot_marginals_list=[(0, 1)],
    ),
    "cartpole": FullyObservableProblem(
        system=systems.CartPole(
            dt=0.2,
            m_c=1.0,
            m_p=0.5,
            l=1.5,
            g=9.81,
            covariance=0.1 * torch.eye(4),
        ),
        initial_state_sampler=make_mvnormal_state_sampler(
            mean=torch.tensor([0.0, 0.0, 0.0, 0.0]),
            covariance=torch.diag(torch.tensor([0.2, 0.2, 0.2, 0.2])),
        ),
        prev_state_sampler=make_mvnormal_state_sampler(
            mean=torch.tensor([0.0, 0.0, 0.0, 0.0]),
            covariance=torch.diag(6.0*torch.ones(4, dtype=torch.float32)),
        ),
        n_timesteps=10,
        n_trajectories_test=5000,
        n_data_tran=10000,
        n_data_init=1000,
        seed=42,
        numerical_tolerance=1e-20,
        plot_bounds_low=torch.tensor([-6.0, -6.0, -6.0, -6.0]),
        plot_bounds_high=torch.tensor([6.0, 6.0, 6.0, 6.0]),
        plot_marginals_list=[(0, 1), (2, 3)],
    ),
    "dubins_trailer": FullyObservableProblem(
        system=systems.SecondOrderDubinsTrailer(
            dt=0.3,
            L_t=0.5,
            v_ref=1.0,
            k_v=1.0,
            k_theta=2.2,
            sigma_v=0.12,
            sigma_omega=0.15,
            sigma_speed_state=0.08,
            cov_scale=0.1,
        ),
        initial_state_sampler=make_mvnormal_state_sampler(
            mean=torch.tensor([0.0, 0.0, 0.35, 0.2, -0.85, 1.0]),
            covariance=torch.diag(
                torch.tensor([0.4, 0.4, 0.25, 0.35, 0.6, 0.4], dtype=torch.float32) ** 2
            ),
        ),
        prev_state_sampler=make_mvnormal_state_sampler(
            mean=torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            covariance=torch.diag(torch.tensor([3.0, 3.0, 1.8, 1.8, 1.2, 2.0], dtype=torch.float32)),
        ),
        n_timesteps=15,
        n_trajectories_test=5000,
        n_data_tran=30000,
        n_data_init=3000,
        seed=42,
        numerical_tolerance=1e-10,
        plot_bounds_low=torch.tensor([-5.0, -5.0, -5.0, -5.0, -5.0, -10.0]),
        plot_bounds_high=torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0, 10.0]),
        plot_marginals_list=[(0, 1), (2, 3), (4, 5)],
    ),
    "planar_quadrotor": FullyObservableProblem(
        system=systems.PlanarQuadrotor(
            dt=0.08,
            waypoint=torch.tensor([2.2, 1.4]),
            m=1.0,
            I=0.02,
            ell=0.22,
            covariance=0.1 * torch.eye(6),
        ),
        initial_state_sampler=make_mvnormal_state_sampler(
            mean=torch.tensor([0.15, 0.35, -0.08, 0.0, 0.0, 0.0]),
            covariance=torch.diag(
                torch.tensor([1.0, 1.0, 0.5, 0.35, 0.35, 1.5], dtype=torch.float32) ** 2
            ),
        ),
        prev_state_sampler=make_mvnormal_state_sampler(
            mean=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            covariance=torch.diag(torch.tensor([5.0, 5.0, 3.0, 15.0, 15.0, 20.0], dtype=torch.float32)),
        ),
        n_timesteps=10,
        n_trajectories_test=5000,
        n_data_tran=30000,
        n_data_init=3000,
        seed=42,
        numerical_tolerance=1e-20,
        plot_bounds_low=torch.tensor([-5.0, -5.0, -4.0, -15.0, -15.0, -20.0]),
        plot_bounds_high=torch.tensor([5.0, 5.0, 4.0, 15.0, 15.0, 20.0]),
        #plot_bounds_low=torch.tensor([-5.0, -5.0, -5.0, -5.0, -5.0, -5.0]),
        #plot_bounds_high=torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0, 5.0]),
        plot_marginals_list=[(0, 1), (3, 4), (2, 5)],
    ),
    "quadcopter": FullyObservableProblem(
        system=systems.Quadcopter(
            dt=0.1,
            waypoint=torch.tensor([3.0, -1.5, 2.0]),
            yaw_ref=0.4,
            covariance=0.08 * torch.eye(12),
        ),
        initial_state_sampler=make_mvnormal_state_sampler(
            mean=torch.tensor([0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0]),
            covariance=torch.diag(
                torch.tensor([0.15, 0.15, 0.12, 0.25, 0.25, 0.25, 0.08, 0.08, 0.2, 0.35, 0.35, 0.35], dtype=torch.float32)
                ** 2
            ),
        ),
        prev_state_sampler=make_mvnormal_state_sampler(
            mean=torch.zeros(12),
            covariance=torch.diag(5.0*torch.ones(12, dtype=torch.float32)),
        ),
        n_timesteps=15,
        n_trajectories_test=5000,
        n_data_tran=50000,
        n_data_init=5000,
        seed=42,
        numerical_tolerance=1e-10,
        plot_bounds_low=torch.tensor([-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0]),
        plot_bounds_high=torch.tensor([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
        plot_marginals_list=[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)],
    ),
    "aircraft": FullyObservableProblem(
        system=systems.Aircraft(
            dt=0.1,
            waypoint=torch.tensor([800.0, 400.0, 120.0]),
            V_ref=22.0,
            sigma_thrust=0.04,
            sigma_surface=0.05,
            sigma_CL=0.03,
            sigma_CM=0.04,
            noise_cov_scale=5.0,
            state_scale=_AIRCRAFT_STATE_SCALE,
        ),
        # States are dimensionless (SI / _AIRCRAFT_STATE_SCALE); same closed-loop dynamics in SI inside the system.
        initial_state_sampler=make_mvnormal_state_sampler(
            mean=_AIRCRAFT_INIT_MEAN,
            covariance=torch.diag(_AIRCRAFT_INIT_STD_NORM**2),
        ),
        prev_state_sampler=make_mvnormal_state_sampler(
            mean=_AIRCRAFT_PREV_MEAN,
            covariance=torch.diag(_AIRCRAFT_PREV_STD_NORM**2),
        ),
        n_timesteps=10,
        n_trajectories_test=5000,
        n_data_tran=50000,
        n_data_init=5000,
        seed=42,
        numerical_tolerance=1e-10,
        plot_bounds_low=_AIRCRAFT_PLOT_LOW,
        plot_bounds_high=_AIRCRAFT_PLOT_HIGH,
        plot_marginals_list=[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)],
    ),
}

PARTIALLY_OBSERVABLE_PROBLEMS = {
    "po_van_der_pol": PartiallyObservableProblem(
        system=po_systems.PartiallyObservableVanDerPol(
            dt=0.3,
            mu=0.9,
            process_covariance=0.1 * torch.eye(2),
            observation_covariance=0.1 * torch.eye(2),
        ),
        initial_state_sampler=make_mvnormal_state_sampler(
            mean=torch.tensor([0.2, 0.1]),
            covariance=torch.diag(torch.tensor([0.2, 0.2])),
        ),
        prev_state_sampler=make_mvnormal_state_sampler(
            mean=torch.tensor([0.0, 0.0]),
            covariance=torch.diag(2.0 * torch.ones(2)),
        ),
        obs_state_sampler=make_mvnormal_state_sampler(
            mean=torch.tensor([0.0, 0.0]),
            covariance=torch.diag(2.0 * torch.ones(2)),
        ),
        n_timesteps=10,
        n_trajectories_test=5000,
        n_data_tran=10000,
        n_data_init=1000,
        n_data_obs=10000,
        seed=42,
        numerical_tolerance=1e-20,
        plot_bounds_low=torch.tensor([-5.0, -5.0]),
        plot_bounds_high=torch.tensor([5.0, 5.0]),
        plot_marginals_list=[(0, 1)],
    ),
}