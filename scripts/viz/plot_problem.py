from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.visualization import plot_problem_test_trajectories


def _square_ranges(
    x_range: tuple[float, float], y_range: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_mid = 0.5 * (x_range[0] + x_range[1])
    y_mid = 0.5 * (y_range[0] + y_range[1])
    half_span = 0.5 * max(x_range[1] - x_range[0], y_range[1] - y_range[0])
    return (x_mid - half_span, x_mid + half_span), (y_mid - half_span, y_mid + half_span)


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
    plt.savefig("figures/problem_test_trajectories.png")

    marginals_list = problem.plot_marginals_list
    if len(marginals_list) > 0:
        init_data = problem.train_initial_state_data()
        prev_data, _ = problem.train_state_transition_data()

        fig, axes = plt.subplots(
            2,
            len(marginals_list),
            figsize=(10, 16),
            squeeze=False,
        )
        row_data = [("initial", init_data), ("previous", prev_data)]
        for row_idx, (row_name, row_samples) in enumerate(row_data):
            for col_idx, (x_dim, y_dim) in enumerate(marginals_list):
                ax = axes[row_idx, col_idx]
                ax.scatter(
                    row_samples[:, x_dim].detach().cpu().numpy(),
                    row_samples[:, y_dim].detach().cpu().numpy(),
                    alpha=0.5,
                    s=4,
                )
                ax.set_xlabel(problem.system.state_label(x_dim))
                ax.set_ylabel(problem.system.state_label(y_dim))
                x_range, y_range = _square_ranges(
                    (problem.plot_bounds_low[x_dim].item(), problem.plot_bounds_high[x_dim].item()),
                    (problem.plot_bounds_low[y_dim].item(), problem.plot_bounds_high[y_dim].item()),
                )
                ax.set_xlim(*x_range)
                ax.set_ylim(*y_range)
                ax.set_aspect("equal")
                ax.set_title(
                    f"{row_name}: {problem.system.state_label(x_dim)}-{problem.system.state_label(y_dim)}"
                )

        fig.tight_layout()
        fig.savefig("figures/problem_train_state_data_marginals.png")

    print(f"Plotted test trajectories for problem: {args.problem}")
    print(f"Saved figure to: figures/problem_test_trajectories.png")
    if len(marginals_list) > 0:
        print("Saved figure to: figures/problem_train_state_data_marginals.png")


if __name__ == "__main__":
    main()
