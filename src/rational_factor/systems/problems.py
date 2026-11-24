from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from rational_factor.systems import systems, po_systems, controlled_systems
from rational_factor.systems.base import (
    ControllableStochasticSystem,
    DiscreteTimeStochasticSystem,
    PartiallyObservableSystem,
    sample_io_pairs,
    sample_uio_tuples,
    sample_observation_pairs,
    sample_trajectories,
    sample_trajectories_without_collisions,
)
from rational_factor.tools.misc import make_mvnormal_state_sampler, make_uniform_state_sampler
from rational_factor.tools.set import Hyperrectangle


def _ambient_box(
    dim: int,
    constraints: dict[int, tuple[float, float]],
    *,
    free_bound: float = 1e6,
) -> Hyperrectangle:
    """Axis-aligned box in ``R^{dim}`` with tight bounds only on selected axes.

    Unconstrained axes use ``[-free_bound, free_bound]`` so containment is
    effectively decided by the listed coordinates (positions, etc.).
    """
    low = torch.full((dim,), -free_bound, dtype=torch.float32)
    high = torch.full((dim,), free_bound, dtype=torch.float32)
    for axis, (lo, hi) in constraints.items():
        if not (0 <= axis < dim):
            raise ValueError(f"axis {axis} out of range for dim={dim}")
        if lo > hi:
            raise ValueError(f"low > high on axis {axis}: {(lo, hi)}")
        low[axis] = float(lo)
        high[axis] = float(hi)
    return Hyperrectangle(low, high)


def _ambient_boxes(
    dim: int,
    boxes: list[dict[int, tuple[float, float]]],
    *,
    free_bound: float = 1e6,
) -> Hyperrectangle:
    """Batched ``Hyperrectangle`` from a list of per-axis constraint dicts."""
    if not boxes:
        raise ValueError("boxes must be non-empty")
    lows, highs = [], []
    for constraints in boxes:
        box = _ambient_box(dim, constraints, free_bound=free_bound)
        lows.append(box.low)
        highs.append(box.high)
    return Hyperrectangle(torch.stack(lows), torch.stack(highs))


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
    
    def test_data_without_collisions(self, set: Hyperrectangle):
        """Test trajectories with particles removed upon entering ``set``.

        ``set`` may be a single box or a batched ``Hyperrectangle`` of obstacles.
        Only non-colliding particles are propagated at each step.
        """
        with torch.random.fork_rng():
            if self.seed is not None:
                torch.manual_seed(self.seed + 1)
            return sample_trajectories_without_collisions(
                self.system,
                self.initial_state_sampler,
                set,
                n_timesteps=self.n_timesteps,
                n_trajectories=self.n_trajectories_test,
            )

@dataclass
class ControlledProblem(FullyObservableProblem):
    system: ControllableStochasticSystem
    goal: Hyperrectangle
    obstacles: Hyperrectangle
    control_limits: Hyperrectangle
    control_sampler: Callable[[int], torch.Tensor]

    def train_state_transition_data(self):
        with torch.random.fork_rng():
            if self.seed is not None:
                torch.manual_seed(self.seed)
            u_data, x_data, x_kp1_data = sample_uio_tuples(self.system, self.prev_state_sampler, self.control_sampler, n_pairs=self.n_data_tran)
            return u_data, x_data, x_kp1_data # (u_k, x_k, x_{k+1}) tuples
    
    def test_data(self, controller):
        with torch.random.fork_rng():
            if self.seed is not None:
                torch.manual_seed(self.seed + 1)
            return sample_trajectories(
                self.system,
                self.initial_state_sampler,
                controller=controller,
                n_timesteps=self.n_timesteps,
                n_trajectories=self.n_trajectories_test,
            )
    
    def test_data_without_collisions(self, controller):
        """Like ``FullyObservableProblem.test_data_without_collisions``, under ``controller``."""
        with torch.random.fork_rng():
            if self.seed is not None:
                torch.manual_seed(self.seed + 1)
            return sample_trajectories_without_collisions(
                self.system,
                self.initial_state_sampler,
                self.obstacles,
                n_timesteps=self.n_timesteps,
                n_trajectories=self.n_trajectories_test,
                controller=controller,
            )

    def test_reach_avoid_probabilities(self, controller) -> torch.Tensor:
        """Per-timestep first-hit goal probabilities under reach-avoid semantics.

        Samples ``test_data(controller)``, then walks the horizon while removing
        any particle that has entered ``obstacles`` or ``goal``. At time ``t``
        the returned entry is
        ``(# still-active particles that lie in the goal) / n_trajectories_test``.
        Particles that land in an obstacle (even if also in the goal) are
        treated as collisions and do not contribute. Summing the returned
        length-``(n_timesteps + 1)`` tensor gives the Monte Carlo reach-avoid
        probability.
        """
        traj_data = self.test_data(controller)
        n = traj_data[0].shape[0]
        if n == 0:
            return torch.zeros(len(traj_data), dtype=torch.float32)

        active = torch.ones(n, dtype=torch.bool, device=traj_data[0].device)
        probs = []
        for x_t in traj_data:
            in_goal = self.goal.contains_any(x_t)
            in_obs = self.obstacles.contains_any(x_t)
            reached = active & in_goal & ~in_obs
            probs.append(reached.to(dtype=torch.float32).sum() / float(n))
            active = active & ~(in_goal | in_obs)
        return torch.stack(probs)

    def random_test_controllers(self, n_controllers: int):
        """Open-loop controllers with actions drawn uniformly in ``control_limits``.

        Each controller is a fixed length-``n_timesteps`` sequence
        ``u[0], ..., u[T-1]`` with every ``u[t]`` in the control box.
        """
        if self.control_limits.is_batched:
            raise ValueError("control_limits must be a single Hyperrectangle")
        if self.control_limits.dim != self.system.control_dim():
            raise ValueError(
                f"control_limits.dim ({self.control_limits.dim}) must match "
                f"control_dim ({self.system.control_dim()})"
            )
        with torch.random.fork_rng():
            if self.seed is not None:
                torch.manual_seed(self.seed + 1)
            low = self.control_limits.low
            high = self.control_limits.high
            controllers = []
            for _ in range(n_controllers):
                u_seq = low + (high - low) * torch.rand(
                    self.n_timesteps, self.system.control_dim()
                )
                controllers.append(lambda k, u_seq=u_seq: u_seq[k])
            return controllers

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
    "kinematic_unicycle": FullyObservableProblem(
        system=systems.KinematicUnicycle(
            dt=0.5,
            speed=1.0,
            angular_displacement=0.12,
            angle_noise_bound=0.2,
            position_noise_std=0.15,
        ),
        initial_state_sampler=make_mvnormal_state_sampler(
            mean=torch.tensor([0.0, 0.0, 0.0]),
            covariance=torch.diag(torch.tensor([0.12, 0.12, 0.08]) ** 2),
        ),
        prev_state_sampler=make_uniform_state_sampler(
            low=torch.tensor([-8.0, -8.0, -3.141593]),
            high=torch.tensor([8.0, 8.0, 3.141593]),
        ),
        n_timesteps=15,
        n_trajectories_test=5000,
        n_data_tran=10000,
        n_data_init=1000,
        seed=42,
        numerical_tolerance=1e-10,
        plot_bounds_low=torch.tensor([-8.0, -8.0, -3.5]),
        plot_bounds_high=torch.tensor([8.0, 8.0, 5.5]),
        plot_marginals_list=[(0, 1), (0, 2), (1, 2)],
    ),
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
            covariance=torch.diag(7.0*torch.ones(4, dtype=torch.float32)),
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
            waypoint=systems.Aircraft.DEFAULT_WAYPOINT,
            V_ref=22.0,
            sigma_thrust=0.04,
            sigma_surface=0.05,
            sigma_CL=0.03,
            sigma_CM=0.04,
            noise_cov_scale=5.0,
            state_scale=systems.Aircraft.STATE_SCALE,
        ),
        initial_state_sampler=make_mvnormal_state_sampler(
            mean=systems.Aircraft.INIT_MEAN,
            covariance=torch.diag(systems.Aircraft.INIT_STD_NORM**2),
        ),
        prev_state_sampler=make_mvnormal_state_sampler(
            mean=systems.Aircraft.PREV_MEAN,
            covariance=torch.diag(systems.Aircraft.PREV_STD_NORM**2),
        ),
        n_timesteps=10,
        n_trajectories_test=5000,
        n_data_tran=50000,
        n_data_init=5000,
        seed=42,
        numerical_tolerance=1e-10,
        plot_bounds_low=systems.Aircraft.PLOT_BOUNDS_LOW,
        plot_bounds_high=systems.Aircraft.PLOT_BOUNDS_HIGH,
        plot_marginals_list=[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)],
    ),
}

OPEN_LOOP_OBSERVABLE_PROBLEMS = {
    "kinematic_unicycle": ControlledProblem(
        system=controlled_systems.ControlledKinematicUnicycle(
            dt=0.5,
            speed=1.0,
            angle_noise_bound=0.2,
            position_noise_std=0.15,
        ),
        initial_state_sampler=make_mvnormal_state_sampler(
            mean=torch.tensor([0.0, 0.0, 0.0]),
            covariance=torch.diag(torch.tensor([0.12, 0.12, 0.08]) ** 2),
        ),
        prev_state_sampler=make_uniform_state_sampler(
            low=torch.tensor([-8.0, -8.0, -3.141593]),
            high=torch.tensor([8.0, 8.0, 3.141593]),
        ),
        # Per-step heading change (rad); matches training sampler support.
        control_limits=Hyperrectangle(torch.tensor([-0.5]), torch.tensor([0.5])),
        control_sampler=make_uniform_state_sampler(
            low=torch.tensor([-0.5]),
            high=torch.tensor([0.5]),
        ),
        # Start near origin; goal in the far quadrant; obstacles block direct paths.
        goal=_ambient_box(3, {0: (3.0, 4.0), 1: (2.0, 3.0)}),
        obstacles=_ambient_boxes(
            3,
            [
                {0: (1.5, 2.0), 1: (-1.0, 1.0)},
                {0: (-1.5, -1.0), 1: (1.5, 3.0)},
            ],
        ),
        n_timesteps=15,
        n_trajectories_test=5000,
        n_data_tran=10000,
        n_data_init=1000,
        seed=42,
        numerical_tolerance=1e-10,
        plot_bounds_low=torch.tensor([-8.0, -8.0, -4.0]),
        plot_bounds_high=torch.tensor([8.0, 8.0, 4.0]),
        plot_marginals_list=[(0, 1), (0, 2), (1, 2)],
    ),
    "icy_road_car": ControlledProblem(
        system=controlled_systems.ControlledIcyRoadCar(
            dt=0.3,
            road_width=6.0,
            road_length=30.0,
        ),
        # Small independent Gaussian initial uncertainties, with the car
        # traveling along +y in the right-hand lane (x approximately 1.5).
        initial_state_sampler=make_mvnormal_state_sampler(
            mean=torch.tensor([1.5, 1.0, 0.0, 6.8, 0.0, 0.0]),
            covariance=torch.diag(
                torch.tensor([0.12, 0.10, 0.025, 0.25, 0.08, 0.035]) ** 2
            ),
        ),
        # Cover the full road in position; remaining states are sampled over
        # a practical range for a small ground vehicle.
        prev_state_sampler=make_uniform_state_sampler(
            low=torch.tensor([0.0, 0.0, -0.55, 1.0, -1.5, -0.65]),
            high=torch.tensor([6.0, 30.0, 0.55, 9.5, 1.5, 0.65]),
        ),
        # Steering (rad) and longitudinal accel (m/s^2); within plant clamps.
        control_limits=Hyperrectangle(
            torch.tensor([-0.30, -3.0]),
            torch.tensor([0.30, 2.0]),
        ),
        control_sampler=make_uniform_state_sampler(
            low=torch.tensor([-0.30, -3.0]),
            high=torch.tensor([0.30, 2.0]),
        ),
        # Goal at the far end of the road; obstacles from the default ice patches.
        goal=_ambient_box(6, {0: (1.0, 2.5), 1: (27.0, 30.0)}),
        obstacles=_ambient_boxes(
            6,
            [
                {0: (0.35, 2.85), 1: (6.5, 11.8)},
                {0: (2.30, 5.70), 1: (13.8, 18.7)},
                {0: (0.65, 3.45), 1: (21.0, 25.7)},
            ],
        ),
        n_timesteps=15,
        n_trajectories_test=2000,
        n_data_tran=30000,
        n_data_init=3000,
        seed=42,
        numerical_tolerance=1e-10,
        plot_bounds_low=torch.tensor([0.0, 0.0, -3.14, -3.0, -5.0, -4.0]),
        plot_bounds_high=torch.tensor([6.0, 30.0, 3.14, 13.0, 5.0, 4.0]),
        plot_marginals_list=[(0, 1), (3, 4), (2, 5)],
    ),
    "scalar_nonlinear_drift": ControlledProblem(
        system=controlled_systems.ControlledScalarNonlinearDrift(
            sigma=0.1,
        ),
        initial_state_sampler=make_mvnormal_state_sampler(
            mean=torch.tensor([0.0]),
            covariance=torch.diag(torch.tensor([0.5])),
        ),
        prev_state_sampler=make_mvnormal_state_sampler(
            mean=torch.tensor([0.0]),
            covariance=torch.diag(torch.tensor([4.0])),
        ),
        control_limits=Hyperrectangle(torch.tensor([-4.0]), torch.tensor([4.0])),
        control_sampler=make_uniform_state_sampler(
            low=torch.tensor([-4.0]),
            high=torch.tensor([4.0]),
        ),
        goal=Hyperrectangle(torch.tensor([2.0]), torch.tensor([3.0])),
        obstacles=Hyperrectangle(torch.tensor([0.6]), torch.tensor([1.2])),
        n_timesteps=10,
        n_trajectories_test=5000,
        n_data_tran=30000,
        n_data_init=3000,
        seed=42,
        numerical_tolerance=1e-20,
        plot_bounds_low=torch.tensor([-3.5]),
        plot_bounds_high=torch.tensor([3.5]),
        plot_marginals_list=[],
    ),
    "van_der_pol": ControlledProblem(
        system=controlled_systems.ControlledVanDerPol(
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
        # Additive forcing on both VdP channels.
        control_limits=Hyperrectangle(
            torch.tensor([-2.0, -2.0]),
            torch.tensor([2.0, 2.0]),
        ),
        control_sampler=make_uniform_state_sampler(
            low=torch.tensor([-2.0, -2.0]),
            high=torch.tensor([2.0, 2.0]),
        ),
        goal=Hyperrectangle(torch.tensor([2.5, 2.5]), torch.tensor([3.5, 3.5])),
        obstacles=_ambient_boxes(
            2,
            [
                {0: (-1.5, -0.5), 1: (0.5, 1.5)},
                {0: (0.8, 1.6), 1: (-1.2, -0.2)},
            ],
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
    "cartpole": ControlledProblem(
        system=controlled_systems.ControlledCartPole(
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
            covariance=torch.diag(6.0 * torch.ones(4, dtype=torch.float32)),
        ),
        # Horizontal cart force (N).
        control_limits=Hyperrectangle(torch.tensor([-15.0]), torch.tensor([15.0])),
        control_sampler=make_uniform_state_sampler(
            low=torch.tensor([-15.0]),
            high=torch.tensor([15.0]),
        ),
        # Reach a cart position with the pole still near upright; avoid rails / tip-over.
        goal=_ambient_box(4, {0: (2.0, 3.0), 2: (-0.35, 0.35)}),
        obstacles=_ambient_boxes(
            4,
            [
                {0: (-6.0, -3.5)},
                {0: (4.5, 6.0)},
                {2: (1.2, 6.0)},
                {2: (-6.0, -1.2)},
            ],
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
    "dubins_trailer": ControlledProblem(
        system=controlled_systems.ControlledSecondOrderDubinsTrailer(
            dt=0.3,
            L_t=0.5,
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
        # Tractor longitudinal accel and yaw rate command.
        control_limits=Hyperrectangle(
            torch.tensor([-4.0, -6.0]),
            torch.tensor([4.0, 6.0]),
        ),
        control_sampler=make_uniform_state_sampler(
            low=torch.tensor([-4.0, -6.0]),
            high=torch.tensor([4.0, 6.0]),
        ),
        goal=_ambient_box(6, {0: (3.0, 4.5), 1: (3.0, 4.5)}),
        obstacles=_ambient_boxes(
            6,
            [
                {0: (1.0, 2.2), 1: (-0.8, 0.8)},
                {0: (-0.8, 0.8), 1: (1.0, 2.2)},
            ],
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
    "planar_quadrotor": ControlledProblem(
        system=controlled_systems.ControlledPlanarQuadrotor(
            dt=0.08,
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
        # Rotor thrusts (N); matches plant thrust_min/max defaults.
        control_limits=Hyperrectangle(
            torch.tensor([0.0, 0.0]),
            torch.tensor([20.0, 20.0]),
        ),
        control_sampler=make_uniform_state_sampler(
            low=torch.tensor([0.0, 0.0]),
            high=torch.tensor([20.0, 20.0]),
        ),
        # Goal in the upper-right of the (px, pz) plane; mid-air obstacles.
        goal=_ambient_box(6, {0: (2.5, 4.0), 1: (2.5, 4.0)}),
        obstacles=_ambient_boxes(
            6,
            [
                {0: (0.8, 1.8), 1: (-0.5, 1.2)},
                {0: (-1.5, -0.3), 1: (1.0, 2.5)},
            ],
        ),
        n_timesteps=10,
        n_trajectories_test=5000,
        n_data_tran=30000,
        n_data_init=3000,
        seed=42,
        numerical_tolerance=1e-20,
        plot_bounds_low=torch.tensor([-5.0, -5.0, -4.0, -15.0, -15.0, -20.0]),
        plot_bounds_high=torch.tensor([5.0, 5.0, 4.0, 15.0, 15.0, 20.0]),
        plot_marginals_list=[(0, 1), (3, 4), (2, 5)],
    ),
    "quadcopter": ControlledProblem(
        system=controlled_systems.ControlledQuadcopter(
            dt=0.1,
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
            covariance=torch.diag(5.0 * torch.ones(12, dtype=torch.float32)),
        ),
        # Collective thrust and body torques.
        control_limits=Hyperrectangle(
            torch.tensor([0.0, -1.0, -1.0, -0.5]),
            torch.tensor([20.0, 1.0, 1.0, 0.5]),
        ),
        control_sampler=make_uniform_state_sampler(
            low=torch.tensor([0.0, -1.0, -1.0, -0.5]),
            high=torch.tensor([20.0, 1.0, 1.0, 0.5]),
        ),
        goal=_ambient_box(12, {0: (3.0, 5.0), 1: (3.0, 5.0), 2: (1.5, 3.5)}),
        obstacles=_ambient_boxes(
            12,
            [
                {0: (1.0, 2.2), 1: (-0.8, 0.8), 2: (0.0, 2.5)},
                {0: (-0.8, 0.8), 1: (1.0, 2.2), 2: (0.0, 2.5)},
            ],
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
    "aircraft": ControlledProblem(
        system=controlled_systems.ControlledAircraft(
            dt=0.1,
            V_ref=22.0,
            sigma_thrust=0.04,
            sigma_surface=0.05,
            sigma_CL=0.03,
            sigma_CM=0.04,
            noise_cov_scale=5.0,
            state_scale=systems.Aircraft.STATE_SCALE,
        ),
        initial_state_sampler=make_mvnormal_state_sampler(
            mean=systems.Aircraft.INIT_MEAN,
            covariance=torch.diag(systems.Aircraft.INIT_STD_NORM**2),
        ),
        prev_state_sampler=make_mvnormal_state_sampler(
            mean=systems.Aircraft.PREV_MEAN,
            covariance=torch.diag(systems.Aircraft.PREV_STD_NORM**2),
        ),
        # Thrust (N) and surface deflections (rad).
        control_limits=Hyperrectangle(
            torch.tensor([50.0, -0.35, -0.35, -0.35]),
            torch.tensor([400.0, 0.35, 0.35, 0.35]),
        ),
        control_sampler=make_uniform_state_sampler(
            low=torch.tensor([50.0, -0.35, -0.35, -0.35]),
            high=torch.tensor([400.0, 0.35, 0.35, 0.35]),
        ),
        # Normalized-state goal near DEFAULT_WAYPOINT; mid-course position obstacles.
        goal=_ambient_box(
            12,
            {
                0: (5.0, 6.5),   # ~700–910 m
                1: (3.0, 4.5),   # ~330–495 m
                2: (3.5, 5.5),   # ~91–143 m
            },
        ),
        obstacles=_ambient_boxes(
            12,
            [
                {0: (1.5, 3.0), 1: (-1.0, 1.0), 2: (2.0, 6.0)},
                {0: (3.5, 4.5), 1: (1.5, 3.0), 2: (2.0, 6.0)},
            ],
        ),
        n_timesteps=10,
        n_trajectories_test=5000,
        n_data_tran=50000,
        n_data_init=5000,
        seed=42,
        numerical_tolerance=1e-10,
        plot_bounds_low=systems.Aircraft.PLOT_BOUNDS_LOW,
        plot_bounds_high=systems.Aircraft.PLOT_BOUNDS_HIGH,
        plot_marginals_list=[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)],
    ),
}

PARTIALLY_OBSERVABLE_PROBLEMS = {
    "van_der_pol": PartiallyObservableProblem(
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
        n_trajectories_test=50,
        n_data_tran=10000,
        n_data_init=1000,
        n_data_obs=10000,
        seed=0,
        numerical_tolerance=1e-20,
        plot_bounds_low=torch.tensor([-5.0, -5.0]),
        plot_bounds_high=torch.tensor([5.0, 5.0]),
        plot_marginals_list=[(0, 1)],
    ),
    "bad_sensor_van_der_pol": PartiallyObservableProblem(
        system=po_systems.BadSensorVanDerPol(
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
        n_trajectories_test=50,
        n_data_tran=10000,
        n_data_init=1000,
        n_data_obs=10000,
        seed=0,
        numerical_tolerance=1e-20,
        plot_bounds_low=torch.tensor([-5.0, -5.0]),
        plot_bounds_high=torch.tensor([5.0, 5.0]),
        plot_marginals_list=[(0, 1)],
    ),
    "cartpole": PartiallyObservableProblem(
        system=po_systems.PartiallyObservableCartPole(
            dt=0.2,
            m_c=1.0,
            m_p=0.5,
            l=1.5,
            g=9.81,
            process_covariance=0.1 * torch.eye(4),
            observation_covariance=0.1 * torch.eye(4),
        ),
        initial_state_sampler=make_mvnormal_state_sampler(
            mean=torch.tensor([0.0, 0.0, 0.0, 0.0]),
            covariance=torch.diag(torch.tensor([0.2, 0.2, 0.2, 0.2])),
        ),
        prev_state_sampler=make_mvnormal_state_sampler(
            mean=torch.tensor([0.0, 0.0, 0.0, 0.0]),
            covariance=torch.diag(6.0 * torch.ones(4, dtype=torch.float32)),
        ),
        obs_state_sampler=make_mvnormal_state_sampler(
            mean=torch.tensor([0.0, 0.0, 0.0, 0.0]),
            covariance=torch.diag(6.0 * torch.ones(4, dtype=torch.float32)),
        ),
        n_timesteps=10,
        n_trajectories_test=50,
        n_data_tran=10000,
        n_data_init=1000,
        n_data_obs=10000,
        seed=0,
        numerical_tolerance=1e-20,
        plot_bounds_low=torch.tensor([-6.0, -6.0, -6.0, -6.0]),
        plot_bounds_high=torch.tensor([6.0, 6.0, 6.0, 6.0]),
        plot_marginals_list=[(0, 1), (2, 3)],
    ),
    "dubins_trailer": PartiallyObservableProblem(
        system=po_systems.PartiallyObservableDubinsTrailer(
            dt=0.3,
            L_t=0.5,
            v_ref=1.0,
            k_v=1.0,
            k_theta=2.2,
            sigma_v=0.12,
            sigma_omega=0.15,
            sigma_speed_state=0.08,
            cov_scale=0.1,
            observation_covariance=0.1 * torch.eye(6),
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
        obs_state_sampler=make_mvnormal_state_sampler(
            mean=torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            covariance=torch.diag(torch.tensor([3.0, 3.0, 1.8, 1.8, 1.2, 2.0], dtype=torch.float32)),
        ),
        n_timesteps=15,
        n_trajectories_test=50,
        n_data_tran=30000,
        n_data_init=3000,
        n_data_obs=30000,
        seed=0,
        numerical_tolerance=1e-10,
        plot_bounds_low=torch.tensor([-5.0, -5.0, -5.0, -5.0, -5.0, -10.0]),
        plot_bounds_high=torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0, 10.0]),
        plot_marginals_list=[(0, 1), (2, 3), (4, 5)],
    ),
    "aircraft": PartiallyObservableProblem(
        system=po_systems.PartiallyObservableAircraft(
            dt=0.1,
            waypoint=systems.Aircraft.DEFAULT_WAYPOINT,
            V_ref=22.0,
            sigma_thrust=0.04,
            sigma_surface=0.05,
            sigma_CL=0.03,
            sigma_CM=0.04,
            noise_cov_scale=5.0,
            state_scale=systems.Aircraft.STATE_SCALE,
            observation_covariance=0.1 * torch.eye(12),
        ),
        initial_state_sampler=make_mvnormal_state_sampler(
            mean=systems.Aircraft.INIT_MEAN,
            covariance=torch.diag(systems.Aircraft.INIT_STD_NORM**2),
        ),
        prev_state_sampler=make_mvnormal_state_sampler(
            mean=systems.Aircraft.PREV_MEAN,
            covariance=torch.diag(systems.Aircraft.PREV_STD_NORM**2),
        ),
        obs_state_sampler=make_mvnormal_state_sampler(
            mean=systems.Aircraft.PREV_MEAN,
            covariance=torch.diag(systems.Aircraft.PREV_STD_NORM**2),
        ),
        n_timesteps=10,
        n_trajectories_test=50,
        n_data_tran=50000,
        n_data_init=5000,
        n_data_obs=50000,
        seed=0,
        numerical_tolerance=1e-10,
        plot_bounds_low=systems.Aircraft.PLOT_BOUNDS_LOW,
        plot_bounds_high=systems.Aircraft.PLOT_BOUNDS_HIGH,
        plot_marginals_list=[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)],
    ),
}