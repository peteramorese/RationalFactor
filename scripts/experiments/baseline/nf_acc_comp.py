"""
Quadcopter: train a conditional NF for transitions and an NF for the initial
distribution, propagate a particle belief with `particle_filter.propagate`,
then report average log-likelihood of held-out states under each KDE belief.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from normalizing_flow.composite_model import CompositeConditionalModel, CompositeDensityModel
from particle_filter.particle_set import ParticleSet
from particle_filter.propagate import propagate
import rational_factor.models.loss as loss
from rational_factor.models.distributions import StandardNormalDensity
from rational_factor.models.train import Trainer
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood


def main() -> None:
    problem_name = "quadcopter"
    problem = FULLY_OBSERVABLE_PROBLEMS[problem_name]
    dim = problem.system.dim()
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    n_particles = 4000
    batch_size = 256
    tran_params = {
        "epochs": 40,
        "iterations": 1,
        "lr": 1e-3,
    }
    init_params = {
        "epochs": 40,
        "iterations": 1,
        "lr": 1e-3,
    }
    num_layers = 5
    hidden_features = 128

    torch.manual_seed(problem.seed if problem.seed is not None else 0)

    x0_data = problem.train_initial_state_data()
    x_k_data, x_kp1_data = problem.train_state_transition_data()
    test_traj = problem.test_data()

    x0_loader = DataLoader(
        TensorDataset(x0_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu
    )
    xp_loader = DataLoader(
        TensorDataset(x_kp1_data, x_k_data),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_gpu,
    )

    tran_nf = CompositeConditionalModel(
        StandardNormalDensity(features=dim, batch_shape=()),
        context_features=dim,
        transform="maf",
        num_layers=num_layers,
        hidden_features=hidden_features,
    ).to(device)
    init_nf = CompositeDensityModel(
        StandardNormalDensity(features=dim, batch_shape=()),
        transform="maf",
        num_layers=num_layers,
        hidden_features=hidden_features,
    ).to(device)

    print(f"Device: {device}")
    print("Training conditional transition NF...")
    transition_trainer = Trainer(tran_nf, device=device)
    transition_trainer.add_dataset(xp_loader, "transitions")
    transition_trainer.add_optimizer(
        torch.optim.Adam(tran_nf.parameters(), lr=tran_params["lr"]),
        "nf",
        n_iterations=tran_params["iterations"],
    )
    transition_trainer.add_loss_function(loss.conditional_mle_loss, "mle", "transitions")
    tran_nf, best_loss_tran, time_tran = transition_trainer.train(
        epochs=tran_params["epochs"],
        use_best="mle",
        verbose=True,
    )
    if tran_nf is None:
        raise SystemExit("No stable transition checkpoint was found.")
    print(f"Transition NF best mle loss: {best_loss_tran:.6f}, time: {time_tran:.2f}s\n")

    print("Training initial-state NF...")
    initial_trainer = Trainer(init_nf, device=device)
    initial_trainer.add_dataset(x0_loader, "initial")
    initial_trainer.add_optimizer(
        torch.optim.Adam(init_nf.parameters(), lr=init_params["lr"]),
        "nf",
        n_iterations=init_params["iterations"],
    )
    initial_trainer.add_loss_function(loss.mle_loss, "mle", "initial")
    init_nf, best_loss_init, time_init = initial_trainer.train(
        epochs=init_params["epochs"],
        use_best="mle",
        verbose=True,
    )
    if init_nf is None:
        raise SystemExit("No stable initial-density checkpoint was found.")
    print(f"Initial NF best mle loss: {best_loss_init:.6f}, time: {time_init:.2f}s\n")

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
        ll_per_step.append(max(float(ll.item()), -30))
        print(f"timestep {t}: avg log-likelihood = {ll_per_step[-1]:.6f}")

    out_dir = Path("figures/baseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    ll_path = out_dir / f"{problem_name}_nf__avg_ll.png"
    plt.figure(figsize=(8, 4))
    plt.plot(ll_per_step, marker="o")
    for t, ll in enumerate(ll_per_step):
        plt.annotate(
            f"{ll:.3f}",
            (t, ll),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    plt.xlabel("timestep")
    plt.ylabel("avg log-likelihood per timestep")
    plt.title(
        f"{problem_name} — particle NF avg log-likelihood\n"
        f"tran loss={best_loss_tran:.4f}, init loss={best_loss_init:.4f}"
    )
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(ll_path, dpi=200)
    print(f"Saved plot to {ll_path}")


if __name__ == "__main__":
    main()
