"""Ablation of SumProd basis / SumProd RFF on CartPole belief propagation LL.

Contexts:
  - baseline:       plain Gaussian leaves + LinearRFF (n_leaf = 500)
  - sp_basis:       SumProdBasis (n_leaf ≈ 80, n_output = 500) + LinearRFF
  - sp_model:       plain Gaussian leaves + SumProdRFF (n_leaf = 500)
  - sp_basis_model: SumProdBasis + SumProdRFF (n_leaf ≈ 80, n_output = 500)

Uses e2e Adam with separate basis / weight (/ B,P) learning rates, train/val
split, and validation early stopping so each model can express itself without
overfitting the training MLE.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.tools.propagate as propagate
from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.factor_forms import LinearFF, LinearRFF, SumProdRFF
from rational_factor.models.parameters import PositiveParameters, TrainableParameters, param_group_iter
from rational_factor.models.sum_prod_basis import SumProdBasis
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood
from rational_factor.tools.benchmark import Benchmark
from rational_factor.tools.misc import train_test_split

TRIALS = 5
BENCHMARK_ROOT = "benchmark_data"
PROBLEM_KEY = "cartpole"
BATCH_SIZE = 128
N_OUTPUT_BASIS = 500
N_LEAF_SP = 80
N_LEAF_PLAIN = 500
BP_OFF_DIAG_LOGIT = -20.0

# Shared e2e schedule: enough budget that early stopping (not the iteration cap)
# is the usual stopping reason.
_TRAN_E2E = {
    "n_epochs_per_group": [1],
    "iterations": 250,
    "lr_basis": 5e-3,
    "lr_weights": 5e-2,
    "lr_bp": 5e-3,
    "patience": 40,
}
_INIT_E2E = {
    "n_epochs_per_group": [1],
    "iterations": 400,
    "lr_basis": 5e-3,
    "lr_weights": 5e-2,
    "patience": 60,
}

# Slightly gentler weight LR when matrix coeffs + B/P are both active.
_TRAN_E2E_FULL = {**_TRAN_E2E, "lr_weights": 3e-2, "lr_bp": 3e-3}
_TRAN_E2E_SP_MODEL = {**_TRAN_E2E, "lr_weights": 5e-2, "lr_bp": 5e-3}

CONTEXT_BASELINE = {
    "sp_basis": False,
    "sp_model": False,
    "n_leaf_basis": N_LEAF_PLAIN,
    "n_output_basis": N_OUTPUT_BASIS,
    "tran_params": dict(_TRAN_E2E),
    "init_params": dict(_INIT_E2E),
    "batch_size": BATCH_SIZE,
    "verbose": True,
}

CONTEXT_SP_BASIS = {
    "sp_basis": True,
    "sp_model": False,
    "n_leaf_basis": N_LEAF_SP,
    "n_output_basis": N_OUTPUT_BASIS,
    "tran_params": dict(_TRAN_E2E),
    "init_params": dict(_INIT_E2E),
    "batch_size": BATCH_SIZE,
    "verbose": True,
}

CONTEXT_SP_MODEL = {
    "sp_basis": False,
    "sp_model": True,
    "n_leaf_basis": N_LEAF_PLAIN,
    "n_output_basis": N_OUTPUT_BASIS,
    "tran_params": dict(_TRAN_E2E_SP_MODEL),
    "init_params": dict(_INIT_E2E),
    "batch_size": BATCH_SIZE,
    "verbose": True,
}

CONTEXT_SP_BASIS_MODEL = {
    "sp_basis": True,
    "sp_model": True,
    "n_leaf_basis": N_LEAF_SP,
    "n_output_basis": N_OUTPUT_BASIS,
    "tran_params": dict(_TRAN_E2E_FULL),
    "init_params": dict(_INIT_E2E),
    "batch_size": BATCH_SIZE,
    "verbose": True,
}


def _near_diagonal_raw(n: int, device: torch.device, off: float = BP_OFF_DIAG_LOGIT) -> torch.Tensor:
    eye = torch.eye(n, dtype=torch.float32, device=device)
    return (off + (0.0 - off) * eye).unsqueeze(0)


def main() -> None:
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    problem = FULLY_OBSERVABLE_PROBLEMS[PROBLEM_KEY]
    system = problem.system

    x0_data = problem.train_initial_state_data()
    x_k_data, x_kp1_data = problem.train_state_transition_data()
    test_traj_data = problem.test_data()

    x0_train, x0_val = train_test_split(x0_data, test_size=0.2, seed=0)
    x_k_train, x_k_val, x_kp1_train, x_kp1_val = train_test_split(
        x_k_data, x_kp1_data, test_size=0.2, seed=0
    )
    x0_dataset = TensorDataset(x0_train)
    x0_val_dataset = TensorDataset(x0_val)
    xp_dataset = TensorDataset(x_kp1_train, x_k_train)
    xp_val_dataset = TensorDataset(x_kp1_val, x_k_val)

    def experiment(
        sp_basis: bool,
        sp_model: bool,
        n_leaf_basis: int,
        n_output_basis: int,
        tran_params: dict,
        init_params: dict,
        batch_size: int,
        verbose: bool = True,
    ):
        x0_dataloader = DataLoader(x0_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
        x0_val_dataloader = DataLoader(
            x0_val_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu
        )
        xp_dataloader = DataLoader(xp_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
        xp_val_dataloader = DataLoader(
            xp_val_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu
        )

        d = system.dim()
        phi_means = TrainableParameters.random_init(
            shape=(1, d, n_leaf_basis), mean=torch.tensor([0.0]), std=torch.tensor([5.0])
        ).to(device)
        psi_means = TrainableParameters.random_init(
            shape=(1, d, n_leaf_basis), mean=torch.tensor([0.0]), std=torch.tensor([5.0])
        ).to(device)
        psi0_means = TrainableParameters.random_init(
            shape=(1, d, n_leaf_basis), mean=torch.tensor([0.0]), std=torch.tensor([5.0])
        ).to(device)
        phi_stds = PositiveParameters.random_init(
            shape=(1, d, n_leaf_basis), mean=torch.tensor([5.0]), std=torch.tensor([10.0]), epsilon=1e-1
        ).to(device)
        psi_stds = PositiveParameters.random_init(
            shape=(1, d, n_leaf_basis), mean=torch.tensor([5.0]), std=torch.tensor([10.0]), epsilon=1e-1
        ).to(device)
        psi0_stds = PositiveParameters.random_init(
            shape=(1, d, n_leaf_basis), mean=torch.tensor([5.0]), std=torch.tensor([10.0]), epsilon=1e-1
        ).to(device)

        phi_matrix_coeffs = psi_matrix_coeffs = psi0_matrix_coeffs = None
        B = P = None

        if sp_basis:
            g_coeffs = PositiveParameters.random_init(
                shape=(1, n_output_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0]), epsilon=1e-3
            ).to(device)
            h0_coeffs = PositiveParameters.random_init(
                shape=(1, n_output_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0]), epsilon=1e-3
            ).to(device)
            phi_matrix_coeffs = PositiveParameters.random_init(
                shape=(1, d, n_output_basis, n_leaf_basis),
                mean=torch.tensor([1.0]),
                std=torch.tensor([1.0]),
                epsilon=1e-6,
            ).to(device)
            psi_matrix_coeffs = PositiveParameters.random_init(
                shape=(1, d, n_output_basis, n_leaf_basis),
                mean=torch.tensor([1.0]),
                std=torch.tensor([1.0]),
                epsilon=1e-6,
            ).to(device)
            psi0_matrix_coeffs = PositiveParameters.random_init(
                shape=(1, d, n_output_basis, n_leaf_basis),
                mean=torch.tensor([1.0]),
                std=torch.tensor([1.0]),
                epsilon=1e-6,
            ).to(device)
            g_basis = SumProdBasis(
                n_output_basis, GaussianBasis(phi_means, phi_stds), phi_matrix_coeffs, coeffs=g_coeffs
            )
            psi_basis = SumProdBasis(
                n_output_basis, GaussianBasis(psi_means, psi_stds), psi_matrix_coeffs
            )
            h0_basis = SumProdBasis(
                n_output_basis, GaussianBasis(psi0_means, psi0_stds), psi0_matrix_coeffs, coeffs=h0_coeffs
            )
            n_model_basis = n_output_basis
        else:
            g_coeffs = PositiveParameters.random_init(
                shape=(1, n_leaf_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0]), epsilon=1e-3
            ).to(device)
            h0_coeffs = PositiveParameters.random_init(
                shape=(1, n_leaf_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0]), epsilon=1e-3
            ).to(device)
            g_basis = GaussianBasis(phi_means, phi_stds, coeffs=g_coeffs)
            psi_basis = GaussianBasis(psi_means, psi_stds)
            h0_basis = GaussianBasis(psi0_means, psi0_stds, coeffs=h0_coeffs)
            n_model_basis = n_leaf_basis

        if sp_model:
            raw = _near_diagonal_raw(n_model_basis, device)
            B = PositiveParameters.from_values(values=raw.clone(), trainable=True).to(device)
            P = PositiveParameters.from_values(
                values=raw.clone(), trainable=True, normalization_dim=2
            ).to(device)
            tran_model = SumProdRFF(g_basis, psi_basis, B, P).to(device)
        else:
            tran_model = LinearRFF(g_basis, psi_basis).to(device)

        # ---- transition (e2e, multi-LR) ----
        basis_param_list = [phi_means, phi_stds, psi_means, psi_stds]
        weight_param_list = [g_coeffs]
        if sp_basis:
            weight_param_list.extend([phi_matrix_coeffs, psi_matrix_coeffs])

        adam_groups = [
            {"params": param_group_iter(basis_param_list), "lr": tran_params["lr_basis"]},
            {"params": param_group_iter(weight_param_list), "lr": tran_params["lr_weights"]},
        ]
        if sp_model:
            adam_groups.append(
                {"params": param_group_iter([B, P]), "lr": tran_params["lr_bp"]}
            )

        tran_model, best_loss_tran, training_time_tran = train.train_iterate(
            tran_model,
            xp_dataloader,
            {"mle": loss.conditional_mle_loss},
            {"all": torch.optim.Adam(adam_groups)},
            device=device,
            labeled_validation_loss_fns={"val_mle": loss.conditional_mle_loss},
            validation_data_loader=xp_val_dataloader,
            validation_early_stopping_patience=tran_params["patience"],
            epochs_per_group=tran_params["n_epochs_per_group"],
            iterations=tran_params["iterations"],
            verbose=verbose,
            use_best="val_mle",
        )

        for p in (phi_means, phi_stds, psi_means, psi_stds, g_coeffs):
            p.set_requires_grad(False)
        if sp_basis:
            phi_matrix_coeffs.set_requires_grad(False)
            psi_matrix_coeffs.set_requires_grad(False)
        if sp_model:
            B.set_requires_grad(False)
            P.set_requires_grad(False)

        # ---- initial belief ----
        init_model = LinearFF.from_rff(tran_model, h0_basis).to(device)
        ff_basis_param_list = [psi0_means, psi0_stds]
        ff_weight_param_list = [h0_coeffs] if not sp_basis else [h0_coeffs, psi0_matrix_coeffs]
        init_model, best_loss_init, training_time_init = train.train_iterate(
            init_model,
            x0_dataloader,
            {"mle": loss.mle_loss},
            {
                "all": torch.optim.Adam(
                    [
                        {
                            "params": param_group_iter(ff_basis_param_list),
                            "lr": init_params["lr_basis"],
                        },
                        {
                            "params": param_group_iter(ff_weight_param_list),
                            "lr": init_params["lr_weights"],
                        },
                    ]
                )
            },
            device=device,
            labeled_validation_loss_fns={"val_mle": loss.mle_loss},
            validation_data_loader=x0_val_dataloader,
            validation_early_stopping_patience=init_params["patience"],
            epochs_per_group=init_params["n_epochs_per_group"],
            iterations=init_params["iterations"],
            verbose=verbose,
            use_best="val_mle",
        )

        # ---- propagate & score ----
        init_model = init_model.eval()
        tran_model = tran_model.eval()
        belief_seq = propagate.propagate(init_model, tran_model, n_steps=problem.n_timesteps)
        n_score = min(len(belief_seq), len(test_traj_data))
        ll_per_step = []
        for i in range(n_score):
            data_i = test_traj_data[i].to(device)
            ll = avg_log_likelihood(belief_seq[i].to(device).eval(), data_i)
            ll_per_step.append(ll.detach().cpu().reshape(()).to(dtype=torch.float32))
        prop_ll_vector = torch.stack(ll_per_step)
        mean_ll = prop_ll_vector.mean().reshape(1)

        return (
            prop_ll_vector,
            mean_ll,
            torch.tensor([float(best_loss_tran)], dtype=torch.float32),
            torch.tensor([float(best_loss_init)], dtype=torch.float32),
            torch.tensor([float(training_time_tran)], dtype=torch.float32),
            torch.tensor([float(training_time_init)], dtype=torch.float32),
        )

    contexts = [
        {"name": "baseline", "params": CONTEXT_BASELINE},
        {"name": "sp_basis", "params": CONTEXT_SP_BASIS},
        {"name": "sp_model", "params": CONTEXT_SP_MODEL},
        {"name": "sp_basis_model", "params": CONTEXT_SP_BASIS_MODEL},
    ]

    benchmark = Benchmark(name=Path(__file__).stem)
    benchmark.set_experiment_fn(experiment)
    benchmark.set_contexts(contexts)
    benchmark.set_numerical_result(0, "avg_log_likelihood_per_timestep", json_raw_data=True)
    benchmark.set_numerical_result(1, "mean_avg_log_likelihood", json_raw_data=True)
    benchmark.set_numerical_result(2, "best_loss_transition", json_raw_data=False)
    benchmark.set_numerical_result(3, "best_loss_initial", json_raw_data=False)
    benchmark.set_numerical_result(4, "training_time_transition", json_raw_data=False)
    benchmark.set_numerical_result(5, "training_time_initial", json_raw_data=False)

    print(
        f"Running SP ablation on {PROBLEM_KEY} "
        f"({len(contexts)} contexts, {TRIALS} trial(s) each)..."
    )
    benchmark.run(trials=TRIALS, verbose=True)
    run_dir = benchmark.process_and_save_results(root_dir=BENCHMARK_ROOT)
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
