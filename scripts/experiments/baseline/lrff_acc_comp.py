from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.tools.misc import init_gaussian_basis_params_gmm
from rational_factor.models.factor_forms import SumProdRFF, LinearFF
from rational_factor.models.parameters import (
    PositiveParameters,
    TrainableParameters,
    param_group_iter,
    RowStochasticDenseMatrixParameters,
)
from rational_factor.models.train import Trainer
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood
import rational_factor.models.loss as loss


if __name__ == "__main__":
    problem_name = "quadcopter"
    problem = FULLY_OBSERVABLE_PROBLEMS[problem_name]

    ###
    use_gpu = torch.cuda.is_available()
    dtype = torch.float64
    n_basis = 500
    gaussian_std_epsilon = 1e-3
    tran_params = {
        "epochs": 20,
        "basis_iterations": 3,
        "weights_iterations": 3,
        "lr_basis": 3e-3,
        "lr_weights": 5e-2,
    }
    init_params = {
        "epochs": 50,
        "coeff_iterations": 10,
        "lr_coeffs": 1e-2,
        "basis_iterations": 2,
        "lr_basis": 1e-3,
    }

    batch_size = 256
    n_timesteps_prop = problem.n_timesteps

    ###

    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"device={device}, dtype={dtype}")

    system = problem.system
    dim = system.dim()

    x0 = problem.train_initial_state_data().to(dtype=dtype)
    x_k, x_kp1 = problem.train_state_transition_data()
    x_k, x_kp1 = x_k.to(dtype=dtype), x_kp1.to(dtype=dtype)
    traj_data = [x.to(dtype=dtype) for x in problem.test_data()]

    x0_dataloader = DataLoader(
        TensorDataset(x0), batch_size=batch_size, shuffle=True, pin_memory=use_gpu
    )
    xp_dataloader = DataLoader(
        TensorDataset(x_kp1, x_k), batch_size=batch_size, shuffle=True, pin_memory=use_gpu
    )

    # Fit independent EM dictionaries to each training marginal.
    phi_means, phi_stds = init_gaussian_basis_params_gmm(
        x_k, dim, n_basis, epsilon=0.0, device=device,
    )
    psi_means, psi_stds = init_gaussian_basis_params_gmm(
        x_kp1, dim, n_basis, epsilon=0.0, device=device,
    )
    phi_basis = GaussianBasis(mean_params=phi_means, std_params=phi_stds, coeffs=None)
    psi_basis = GaussianBasis(mean_params=psi_means, std_params=psi_stds, coeffs=None)

    g_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis), mean=1.0, std=1.0, epsilon=1e-3
    ).to(device=device, dtype=dtype)
    h0_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis), mean=1.0, std=1.0, epsilon=1e-3
    ).to(device=device, dtype=dtype)

    # SumProdRFF is normalized iff B @ 1 = 1.
    B_logits = TrainableParameters.random_init(
        shape=(1, n_basis, n_basis),
        mean=0.0,
        std=0.1,
    ).to(device=device, dtype=dtype)
    B = RowStochasticDenseMatrixParameters(B_logits)

    tran_model = SumProdRFF(
        g_coeffs, phi_basis, psi_basis, B, numerical_tolerance=problem.numerical_tolerance
    )

    print("Training transition model")
    transition_trainer = Trainer(tran_model, device=device)
    transition_trainer.add_dataset(xp_dataloader, "transitions")
    transition_trainer.add_optimizer(
        torch.optim.Adam(
            param_group_iter((phi_means, phi_stds, psi_means, psi_stds)),
            lr=tran_params["lr_basis"],
        ),
        "basis",
        n_iterations=tran_params["basis_iterations"],
    )
    transition_trainer.add_optimizer(
        torch.optim.Adam(
            param_group_iter((g_coeffs, B_logits)),
            lr=tran_params["lr_weights"],
        ),
        "weights",
        n_iterations=tran_params["weights_iterations"],
    )
    transition_trainer.add_loss_function(loss.conditional_mle_loss, "mle", "transitions")
    tran_model, best_loss_tran, training_time_tran = transition_trainer.train(
        epochs=tran_params["epochs"],
        use_best="mle",
        verbose=True,
    )
    if tran_model is None:
        raise SystemExit("No stable transition checkpoint was found.")
    print("Done! \n")

    for parameter in tran_model.parameters():
        parameter.requires_grad_(False)

    psi0_means, psi0_stds = init_gaussian_basis_params_gmm(
        x0, dim, n_basis, epsilon=0.0, device=device,
    )
    psi0_basis = GaussianBasis(mean_params=psi0_means, std_params=psi0_stds, coeffs=None)
    init_model = LinearFF.from_rff(tran_model, h0_coeffs, psi0_basis).to(device)

    print("Training initial model")
    initial_trainer = Trainer(init_model, device=device)
    initial_trainer.add_dataset(x0_dataloader, "initial")

    initial_trainer.add_optimizer(
        torch.optim.Adam(
            param_group_iter((h0_coeffs,)),
            lr=init_params["lr_coeffs"],
        ),
        "coeffs",
        n_iterations=init_params["coeff_iterations"],
    )
    initial_trainer.add_optimizer(
        torch.optim.Adam(param_group_iter((psi0_means, psi0_stds)), lr=1e-3),
        "psi0_basis", n_iterations=2,
    )
    initial_trainer.add_loss_function(loss.mle_loss, "mle", "initial")
    init_model, best_loss_init, training_time_init = initial_trainer.train(
        epochs=init_params["epochs"],
        use_best="mle",
        verbose=True,
    )
    if init_model is None:
        raise SystemExit("No stable initial-density checkpoint was found.")
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

    belief_seq = [init_model]
    for _ in range(n_timesteps_prop):
        belief_seq.append(tran_model.propagate(belief_seq[-1]))
    belief_seq = [belief.eval() for belief in belief_seq]

    ll_per_step = []
    with torch.no_grad():
        beliefs = [init_model]
        for _ in range(n_timesteps_prop):
            beliefs.append(tran_model.propagate(beliefs[-1]))
        for i, (belief, samples) in enumerate(zip(beliefs, traj_data)):
            loglik = avg_log_likelihood(belief, samples.to(device=device, dtype=dtype))
            print(f"Avg log-likelihood at time {i}: {float(loglik):.6f}")
            ll_per_step.append(float(loglik.detach().cpu()))

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
