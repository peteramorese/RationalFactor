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
            mu=0.9,
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
            covariance=torch.diag(torch.tensor([0.2, 0.2, 0.2, 0.2])),
        ),
        n_timesteps=10,
        n_trajectories_test=5000,
        n_data_tran=10000,
        n_data_init=1000,
        seed=42,
        numerical_tolerance=1e-20,
        plot_bounds_low=torch.tensor([-5.0, -5.0, -5.0, -5.0]),
        plot_bounds_high=torch.tensor([5.0, 5.0, 5.0, 5.0]),
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
        n_data_tran=15000,
        n_data_init=1500,
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
            covariance=torch.diag(torch.tensor([5.0, 10.0, 3.0, 10.0, 10.0, 15.0], dtype=torch.float32)),
        ),
        n_timesteps=10,
        n_trajectories_test=5000,
        n_data_tran=50000,
        n_data_init=5000,
        seed=42,
        numerical_tolerance=1e-20,
        plot_bounds_low=torch.tensor([-5.0, -5.0, -4.0, -10.0, -10.0, -15.0]),
        plot_bounds_high=torch.tensor([5.0, 5.0, 4.0, 10.0, 10.0, 15.0]),
        #plot_bounds_low=torch.tensor([-5.0, -5.0, -5.0, -5.0, -5.0, -5.0]),
        #plot_bounds_high=torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0, 5.0]),
        plot_marginals_list=[(0, 1), (3, 4), (2, 5)],
    ),
    "quadcopter": FullyObservableProblem(
        system=systems.Quadcopter(
            dt=0.05,
            waypoint=torch.tensor([3.0, -1.5, 2.0]),
            yaw_ref=0.4,
            covariance=0.0008 * torch.eye(12),
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
            covariance=torch.diag(torch.tensor([2.0, 2.0, 1.5, 1.5, 1.5, 1.5, 0.6, 0.6, 1.2, 1.5, 1.5, 1.5], dtype=torch.float32)),
        ),
        n_timesteps=15,
        n_trajectories_test=5000,
        n_data_tran=20000,
        n_data_init=2000,
        seed=42,
        numerical_tolerance=1e-10,
        plot_bounds_low=torch.tensor([-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0]),
        plot_bounds_high=torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]),
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