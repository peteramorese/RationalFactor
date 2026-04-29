"""
Benchmark: train conditional + initial NFs, propagate a particle belief
(`particle_filter.propagate`), score held-out states with KDE avg log-likelihood
— same experiment as `scripts/experiments/nf_particle_prop/dub.py`, generalized
to any `FULLY_OBSERVABLE_PROBLEMS` key. Three contexts differ only in
`n_particles` (see `PARTICLE_COUNTS`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import rational_factor.models.loss as loss
import rational_factor.models.train as train
from normalizing_flow.normalizing_flow import ConditionalNormalizingFlow, NormalizingFlow
from particle_filter.particle_set import ParticleSet
from particle_filter.propagate import propagate
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood
from rational_factor.tools.benchmark import Benchmark


def _nf_particle_params(n_particles: int) -> dict:
    return {
        "n_particles": n_particles,
        "batch_size": 256,
        "tran_epochs": 40,
        "init_epochs": 40,
        "lr": 1e-3,
        "num_layers": 5,
        "hidden_features": 128,
        "verbose": True,
    }


PARTICLE_COUNTS = (500, 1000, 5000)

TRIALS = 10
BENCHMARK_ROOT = "benchmark_data"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark NF + particle propagation baseline on a fully observable problem."
    )
    parser.add_argument(
        "problem",
        type=str,
        choices=sorted(FULLY_OBSERVABLE_PROBLEMS.keys()),
        help="Key in FULLY_OBSERVABLE_PROBLEMS.",
    )
    args = parser.parse_args()
    problem_key = args.problem
    problem = FULLY_OBSERVABLE_PROBLEMS[problem_key]

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    # Benchmark does not pass a trial index; offset the problem seed so trials differ.
    _trial = {"i": 0}

    def experiment(
        n_particles: int,
        batch_size: int,
        tran_epochs: int,
        init_epochs: int,
        lr: float,
        num_layers: int,
        hidden_features: int,
        verbose: bool,
    ):
        base_seed = problem.seed if problem.seed is not None else 0
        torch.manual_seed(base_seed + _trial["i"])
        _trial["i"] += 1

        dim = problem.system.dim()

        x0_data = problem.train_initial_state_data()
        x_k_data, x_kp1_data = problem.train_state_transition_data()
        test_traj = problem.test_data()

        x0_loader = DataLoader(
            TensorDataset(x0_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu
        )
        xp_loader = DataLoader(
            TensorDataset(x_kp1_data, x_k_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu
        )

        tran_nf = ConditionalNormalizingFlow(
            dim=dim,
            conditioner_dim=dim,
            num_layers=num_layers,
            hidden_features=hidden_features,
        ).to(device)
        init_nf = NormalizingFlow(
            dim=dim,
            num_layers=num_layers,
            hidden_features=hidden_features,
        ).to(device)

        tran_nf, best_loss_tran, time_tran = train.train(
            tran_nf,
            xp_loader,
            {"mle": loss.conditional_mle_loss},
            torch.optim.Adam(tran_nf.parameters(), lr=lr),
            epochs=tran_epochs,
            verbose=verbose,
            use_best="mle",
        )

        init_nf, best_loss_init, time_init = train.train(
            init_nf,
            x0_loader,
            {"mle": loss.mle_loss},
            torch.optim.Adam(init_nf.parameters(), lr=lr),
            epochs=init_epochs,
            verbose=verbose,
            use_best="mle",
        )

        tran_nf.eval()
        init_nf.eval()

        with torch.no_grad():
            particles0 = init_nf.sample(n_particles).to(device)
        initial_belief = ParticleSet(particles=particles0)

        n_steps = problem.n_timesteps
        belief_seq = propagate(initial_belief, tran_nf, n_steps=n_steps, copy_belief=True)

        ll_per_step: list[float] = []
        for t, belief in enumerate(belief_seq):
            states_t = test_traj[t].to(device)
            ll = avg_log_likelihood(belief, states_t)
            ll_per_step.append(float(ll.item()))

        prop_ll_vector = torch.tensor(ll_per_step, dtype=torch.float32)

        return (
            prop_ll_vector,
            torch.tensor([float(best_loss_tran)], dtype=torch.float32),
            torch.tensor([float(time_tran)], dtype=torch.float32),
            torch.tensor([float(best_loss_init)], dtype=torch.float32),
            torch.tensor([float(time_init)], dtype=torch.float32),
        )

    contexts = [
        {"name": f"{n} particles", "params": _nf_particle_params(n)}
        for n in PARTICLE_COUNTS
    ]

    benchmark = Benchmark(name=Path(__file__).stem + "_" + problem_key)
    benchmark.set_experiment_fn(experiment)
    benchmark.set_contexts(contexts)

    benchmark.set_numerical_result(0, "avg_log_likelihood_per_timestep", json_raw_data=False)
    benchmark.set_numerical_result(1, "best_loss_transition", json_raw_data=False)
    benchmark.set_numerical_result(2, "training_time_transition", json_raw_data=False)
    benchmark.set_numerical_result(3, "best_loss_initial", json_raw_data=False)
    benchmark.set_numerical_result(4, "training_time_initial", json_raw_data=False)

    print(
        f"Running benchmark problem={problem_key} ({len(contexts)} context(s), {TRIALS} trial(s) each)..."
    )
    benchmark.run(trials=TRIALS, verbose=True)
    run_dir = benchmark.process_and_save_results(root_dir=BENCHMARK_ROOT)
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
