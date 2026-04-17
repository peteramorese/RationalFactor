from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.visualization import plot_problem_test_trajectories


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot marginal test trajectories for a fully observable problem."
    )
    parser.add_argument(
        "problem",
        type=str,
        choices=sorted(FULLY_OBSERVABLE_PROBLEMS.keys()),
        help="Key in FULLY_OBSERVABLE_PROBLEMS.",
    )
    args = parser.parse_args()
    problem = FULLY_OBSERVABLE_PROBLEMS[args.problem]
    plot_problem_test_trajectories(problem, scatter_kwargs={"alpha": 0.5, "s": 4})
    plt.savefig(f"figures/problem_test_trajectories.png")
    print(f"Plotted test trajectories for problem: {args.problem}")
    print(f"Saved figure to: figures/problem_test_trajectories.png")


if __name__ == "__main__":
    main()
