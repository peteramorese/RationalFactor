from rational_factor.systems.controlled_systems import ControlledIcyRoadCar
from rational_factor.systems.vis_sim import plot_icy_road_mc
from rational_factor.systems.problems import OPEN_LOOP_OBSERVABLE_PROBLEMS

import torch

from pathlib import Path


if __name__ == "__main__":
    problem = OPEN_LOOP_OBSERVABLE_PROBLEMS["icy_road_car"]
    controls = torch.tensor([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.1, 0.0],
        [0.1, 0.0],
        [0.0, 0.0],
        [-0.1, 0.0],
        [-0.1, 0.0],
        [-0.1, 0.0],
        [-0.1, 0.0],
        [0.0, 0.0] ])
    n_particles = 100

    out_dir = Path("figures/sim")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "icy_road_car_sim.png"
    
    plot_icy_road_mc(problem, controls, n_particles, save_path=save_path)
