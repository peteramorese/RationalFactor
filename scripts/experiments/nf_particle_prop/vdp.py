"""
Van der Pol oscillator: train a conditional NF for transitions and an NF for
the initial distribution, propagate a particle belief with
`particle_filter.propagate`, then report average log-likelihood of held-out
states under each KDE belief.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from normalizing_flow.normalizing_flow import ConditionalNormalizingFlow, NormalizingFlow
from particle_filter.particle_set import ParticleSet
from particle_filter.propagate import propagate
import rational_factor.models.loss as loss
import rational_factor.models.train as train
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood
from rational_factor.tools.visualization import plot_marginal_trajectory_comparison


def main() -> None:
    problem = FULLY_OBSERVABLE_PROBLEMS["van_der_pol"]
    dim = problem.system.dim()
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    n_particles = 4000
    batch_size = 256
    tran_epochs = 40
    init_epochs = 40
    lr = 1e-3
    num_layers = 5
    hidden_features = 128

    torch.manual_seed(problem.seed if problem.seed is not None else 0)

    x0_data = problem.train_initial_state_data()
    x_k_data, x_kp1_data = problem.train_state_transition_data()
    test_traj = problem.test_data()

    x0_loader = DataLoader(TensorDataset(x0_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    xp_loader = DataLoader(TensorDataset(x_kp1_data, x_k_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

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

    print(f"Device: {device}")
    print("Training conditional transition NF...")
    tran_nf, best_loss_tran, time_tran = train.train(
        tran_nf,
        xp_loader,
        {"mle": loss.conditional_mle_loss},
        torch.optim.Adam(tran_nf.parameters(), lr=lr),
        epochs=tran_epochs,
        verbose=True,
        use_best="mle",
    )
    print(f"Transition NF best mle loss: {best_loss_tran:.6f}, time: {time_tran:.2f}s\n")

    print("Training initial-state NF...")
    init_nf, best_loss_init, time_init = train.train(
        init_nf,
        x0_loader,
        {"mle": loss.mle_loss},
        torch.optim.Adam(init_nf.parameters(), lr=lr),
        epochs=init_epochs,
        verbose=True,
        use_best="mle",
    )
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
        ll_per_step.append(float(ll.item()))
        print(f"timestep {t}: avg log-likelihood = {ll_per_step[-1]:.6f}")

    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    ll_path = out_dir / "van_der_pol_nf__avg_ll.png"
    plt.figure(figsize=(8, 4))
    plt.plot(ll_per_step, marker="o")
    plt.xlabel("timestep")
    plt.ylabel("avg log-likelihood (KDE belief vs test states)")
    plt.title("van_der_pol - particle NF propagation")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(ll_path, dpi=200)
    print(f"Saved plot to {ll_path}")

    comp_out_path = out_dir / "van_der_pol_nf__marginal_comparison.png"
    belief_seq = [belief.to("cpu") for belief in belief_seq]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    fig, _ = plot_marginal_trajectory_comparison(
        problem=problem,
        beliefs=belief_seq,
        scatter_kwargs={"s": 1, "alpha": 0.7},
        n_points=40,
    )
    if fig is not None:
        fig.suptitle("van_der_pol - particle NF marginal comparison", y=1.02)
        fig.savefig(comp_out_path, dpi=200, bbox_inches="tight")
        print(f"Saved marginal comparison plot to {comp_out_path}")


if __name__ == "__main__":
    main()
