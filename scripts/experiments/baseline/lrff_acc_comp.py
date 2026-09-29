from pathlib import Path
from math import ceil, sqrt

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.factor_forms import SumProdRFF, LinearFF
from rational_factor.models.kde import GaussianKDE
from rational_factor.models.parameters import (
    PositiveParameters,
    TrainableParameters,
    param_group_iter,
    DenseMatrixFactorization,
)
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood, check_pdf_valid
import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.tools.propagate as propagate


if __name__ == "__main__":
    problem_name = "van_der_pol"
    problem = FULLY_OBSERVABLE_PROBLEMS[problem_name]

    ###
    use_gpu = torch.cuda.is_available()
    n_basis = 25
    tran_params = {
        "n_epochs_per_group": [3, 3],
        "iterations": 100,
        "lr_basis": 3e-3,
        "lr_weights": 5e-2,
    }
    init_params = {
        "n_epochs_per_group": [10],  # h0 coeffs only
        "iterations": 50,
        "lr_weights": 1e-2,
    }

    batch_size = 256
    n_timesteps_prop = problem.n_timesteps

    ###

    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using GPU: ", use_gpu)
    print("Device: ", device)

    system = problem.system
    dim = system.dim()

    x0 = problem.train_initial_state_data()
    x_k, x_kp1 = problem.train_state_transition_data()
    traj_data = problem.test_data()

    x0_dataloader = DataLoader(
        TensorDataset(x0), batch_size=batch_size, shuffle=True, pin_memory=use_gpu
    )
    xp_dataloader = DataLoader(
        TensorDataset(x_kp1, x_k), batch_size=batch_size, shuffle=True, pin_memory=use_gpu
    )

    box_lows = problem.plot_bounds_low.to(device=device)
    box_highs = problem.plot_bounds_high.to(device=device)

    t = torch.linspace(0.0, 1.0, n_basis, device=device)
    means = box_lows.unsqueeze(-1) + (box_highs - box_lows).unsqueeze(-1) * t
    means = means.unsqueeze(0)  # (1, dim, n_basis)
    mean_jitter = 0.35
    std_init = 0.35
    phi_means = TrainableParameters.from_values(
        means + mean_jitter * torch.randn_like(means)
    ).to(device)
    phi_stds = PositiveParameters.from_values(
        torch.full((1, dim, n_basis), std_init, device=device)
    ).to(device)
    psi_offset = 0.25 * torch.ones(dim, device=device)
    psi_offset[1::2] = -0.25
    psi_means = TrainableParameters.from_values(
        means
        + psi_offset.view(1, dim, 1)
        + mean_jitter * torch.randn_like(means)
    ).to(device)
    psi_stds = PositiveParameters.from_values(
        torch.full((1, dim, n_basis), std_init + 0.1, device=device)
    ).to(device)
    phi_basis = GaussianBasis(phi_means, phi_stds, coeffs=None)
    psi_basis = GaussianBasis(psi_means, psi_stds, coeffs=None)


    g_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0]), epsilon=1e-3
    ).to(device)
    h0_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0])
    ).to(device)

    B_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis, n_basis),
        mean=torch.tensor([1.0]),
        std=torch.tensor([1.0]),
        epsilon=0.0,
    ).to(device)
    B = DenseMatrixFactorization(B_coeffs)

    g_basis = GaussianBasis(phi_means, phi_stds, coeffs=g_coeffs)
    psi_basis = GaussianBasis(psi_means, psi_stds, coeffs=None)

    tran_model = SumProdRFF(g_basis, psi_basis, B, numerical_tolerance=problem.numerical_tolerance)

    print("Training transition model")
    mle_loss_fn = loss.conditional_mle_loss
    optimizers = {
        "basis": torch.optim.Adam(
            [
                {
                    "params": param_group_iter(
                        (phi_means, phi_stds, psi_means, psi_stds)
                    ),
                    "lr": tran_params["lr_basis"],
                },
            ]
        ),
        "weights": torch.optim.Adam(
            param_group_iter((g_coeffs, B_coeffs)), lr=tran_params["lr_weights"]
        ),
    }

    tran_model, best_loss_tran, training_time_tran = train.train_iterate(
        tran_model,
        xp_dataloader,
        {"mle": mle_loss_fn},
        optimizers,
        device=device,
        epochs_per_group=tran_params["n_epochs_per_group"],
        iterations=tran_params["iterations"],
        verbose=True,
        use_best="mle",
    )
    print("Done! \n")

    phi_means.set_requires_grad(False)
    phi_stds.set_requires_grad(False)
    psi_means.set_requires_grad(False)
    psi_stds.set_requires_grad(False)
    g_coeffs.set_requires_grad(False)

    h0_basis = GaussianBasis(psi_means, psi_stds, coeffs=h0_coeffs)
    init_model = LinearFF.from_rff(tran_model, h0_basis).to(device)

    print("Training initial model")
    mle_loss_fn = loss.mle_loss
    optimizers = {
        "weights": torch.optim.Adam(h0_coeffs.parameters(), lr=init_params["lr_weights"]),
    }

    init_model, best_loss_init, training_time_init = train.train_iterate(
        init_model,
        x0_dataloader,
        {"mle": mle_loss_fn},
        optimizers,
        device=device,
        epochs_per_group=init_params["n_epochs_per_group"],
        iterations=init_params["iterations"],
        verbose=True,
        use_best="mle",
    )
    print("Done! \n")

    print(
        f"Transition model loss: {best_loss_tran:.4f}, "
        f"training time: {training_time_tran:.2f} seconds"
    )
    print(
        f"Initial model loss: {best_loss_init:.4f}, "
        f"training time: {training_time_init:.2f} seconds"
    )


    n_slices = n_timesteps_prop + 1

    base_belief_seq = propagate.propagate(
        init_model,
        tran_model,
        n_steps=n_timesteps_prop,
    )
    belief_seq = [belief.eval() for belief in base_belief_seq]

    ll_per_step = []
    for i in range(n_slices):
        data_i = traj_data[i]
        ll = avg_log_likelihood(belief_seq[i], data_i)
        ll_per_step.append(float(ll.detach().cpu()))
        print(f"timestep {i}: avg log-likelihood = {ll_per_step[-1]:.6f}")

    out_dir = Path("figures/baseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    ll_path = out_dir / f"{problem_name}_lrff__avg_ll.png"
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
        f"{problem_name} — LRFF avg log-likelihood\n"
        f"tran loss={best_loss_tran:.4f}, init loss={best_loss_init:.4f}\n"
        f"n_basis={n_basis}"
    )
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(ll_path, dpi=200)
    print(f"Saved plot to {ll_path}")