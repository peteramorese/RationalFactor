from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.tools.propagate as propagate
from rational_factor.models.basis_functions import GaussianKernelBasis
from rational_factor.models.composite_model import CompositeConditionalModel, CompositeDensityModel
from rational_factor.models.density_model import LogisticSigmoid
from rational_factor.models.domain_transformation import MaskedRQSNFTF
from rational_factor.models.factor_forms import LinearFF, LinearRFF
from rational_factor.models.kde import GaussianKDE
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood
from rational_factor.tools.benchmark import Benchmark
from rational_factor.tools.misc import data_bounds, train_test_split

TRIALS = 5
BENCHMARK_ROOT = "benchmark_data"

BANDWIDTH_THRESHOLDS_TRAN = [5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

CONTEXT_TEMPLATE = {
    "dtf_params": {
        "epochs": 20,
        "lr": 1e-3,
    },
    "init_params": {
        "n_epochs_per_group": [1],
        "iterations": 200,
        "lr_weights": 1e-1,
    },
    "batch_size": 1024,
    "ls_temp": 0.1,
    "bw_val_thresh_init": 10.0,
    "bw_val_lr_tran": 1e-3,
    "bw_val_lr_init": 1e-2,
    "verbose": True,
}


def main() -> None:
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using device: ", device)

    problem = FULLY_OBSERVABLE_PROBLEMS["quadcopter"]
    system = problem.system
    x0_data = problem.train_initial_state_data()
    x_k_data, x_kp1_data = problem.train_state_transition_data()
    test_traj_data = problem.test_data()

    def experiment(
        dtf_params: dict,
        init_params: dict,
        batch_size: int,
        ls_temp: float,
        bw_val_thresh_tran: float,
        bw_val_thresh_init: float,
        bw_val_lr_tran: float,
        bw_val_lr_init: float,
        verbose: bool = True,
    ):
        x_dataloader = DataLoader(TensorDataset(x_k_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
        x0_dataloader = DataLoader(TensorDataset(x0_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

        loc, scale = data_bounds(x_k_data, mode="center_lengths")
        loc = loc.to(device)
        scale = scale.to(device)
        base_distribution = LogisticSigmoid(system.dim(), temperature=ls_temp, loc=loc, scale=scale)
        dtf = MaskedRQSNFTF(system.dim(), trainable=True, hidden_features=256, n_layers=5).to(device)
        decorrupter = CompositeDensityModel([dtf], base_distribution).to(device)

        optimizer = torch.optim.Adam(decorrupter.parameters(), lr=dtf_params["lr"], weight_decay=1e-3)
        decorrupter, _, _ = train.train(
            decorrupter,
            x_dataloader,
            {"mle": loss.mle_loss},
            optimizer,
            epochs=dtf_params["epochs"],
            verbose=verbose,
            use_best="mle",
            clip_grad_norm=5.0,
            restore_loss_threshold=50.0,
        )

        dtf_trained = MaskedRQSNFTF.copy_from_trainable(dtf).to(device)

        x_k_data_device = x_k_data.to(device)
        x_kp1_data_device = x_kp1_data.to(device)
        x0_data_device = x0_data.to(device)
        with torch.no_grad():
            x_k_transformed, _ = dtf_trained(x_k_data_device)
            x_kp1_transformed, _ = dtf_trained(x_kp1_data_device)
            x0_transformed, _ = dtf_trained(x0_data_device)

        joint_data = torch.cat([x_k_transformed, x_kp1_transformed], dim=1)
        joint_train_data, joint_val_data = train_test_split(joint_data, test_size=0.2)
        joint_kde = GaussianKDE(joint_train_data)
        best_bandwidth_joint, _ = joint_kde.fit_bandwidth_validation_mle(
            joint_val_data,
            epochs=200,
            threshold=bw_val_thresh_tran,
            lr=bw_val_lr_tran,
            min_step=1e-3,
            verbose=verbose,
            block_size=128,
        )

        init_train_data, init_val_data = train_test_split(x0_transformed, test_size=0.2)
        init_kde = GaussianKDE(init_train_data)
        best_bandwidth_init, _ = init_kde.fit_bandwidth_validation_mle(
            init_val_data,
            epochs=200,
            threshold=bw_val_thresh_init,
            lr=bw_val_lr_init,
            min_step=1e-3,
            verbose=verbose,
            block_size=128,
        )

        phi_basis = GaussianKernelBasis(x_k_transformed, kernel_bandwidth=best_bandwidth_joint, trainable=False)
        psi_basis = GaussianKernelBasis(x_kp1_transformed, kernel_bandwidth=best_bandwidth_joint, trainable=False)
        psi0_basis = GaussianKernelBasis(x0_transformed, kernel_bandwidth=best_bandwidth_init, trainable=True)

        lrff = LinearRFF(phi_basis, psi_basis, numerical_tolerance=problem.numerical_tolerance).to(device)
        tran_model = CompositeConditionalModel([dtf_trained], lrff).to(device)

        ff = LinearFF.from_rff(lrff, psi0_basis).to(device)
        init_model = CompositeDensityModel([dtf_trained], ff).to(device)

        optimizers = {
            "weights": torch.optim.Adam(
                [{"params": init_model.density_model.weight_params(), "lr": init_params["lr_weights"]}]
            )
        }
        init_model, best_loss_init, training_time_init = train.train_iterate(
            init_model,
            x0_dataloader,
            {"mle": loss.mle_loss},
            optimizers,
            epochs_per_group=init_params["n_epochs_per_group"],
            iterations=init_params["iterations"],
            verbose=verbose,
            use_best="mle",
            clip_grad_norm=5.0,
            restore_loss_threshold=50.0,
        )

        cpu = torch.device("cpu")
        init_model.to(cpu)
        tran_model.to(cpu)
        dtf_trained.to(cpu)

        base_belief_seq = propagate.propagate(
            init_model.density_model,
            tran_model.conditional_density_model,
            n_steps=problem.n_timesteps,
            device=cpu,
        )
        belief_seq = [CompositeDensityModel([dtf_trained], belief) for belief in base_belief_seq]

        ll_per_step = []
        for i in range(problem.n_timesteps):
            data_i = test_traj_data[i].to(cpu)
            ll = avg_log_likelihood(belief_seq[i], data_i)
            ll_per_step.append(ll.detach().cpu().reshape(()))
        prop_ll_vector = torch.stack(ll_per_step).to(dtype=torch.float32)

        return (
            prop_ll_vector,
            torch.tensor([float(best_bandwidth_joint)], dtype=torch.float32),
            torch.tensor([float(best_bandwidth_init)], dtype=torch.float32),
            torch.tensor([float(best_loss_init)], dtype=torch.float32),
            torch.tensor([float(training_time_init)], dtype=torch.float32),
        )

    contexts = []
    for threshold in BANDWIDTH_THRESHOLDS_TRAN:
        contexts.append(
            {
                "name": f"bw_val_thresh_tran_{str(threshold).replace('.', 'p')}",
                "params": {**CONTEXT_TEMPLATE, "bw_val_thresh_tran": threshold},
            }
        )

    benchmark = Benchmark(name=Path(__file__).stem)
    benchmark.set_experiment_fn(experiment)
    benchmark.set_contexts(contexts)

    benchmark.set_numerical_result(0, "avg_log_likelihood_per_timestep", json_raw_data=False)
    benchmark.set_numerical_result(1, "best_joint_bandwidth", json_raw_data=False)
    benchmark.set_numerical_result(2, "best_initial_bandwidth", json_raw_data=False)
    benchmark.set_numerical_result(3, "best_loss_initial", json_raw_data=False)
    benchmark.set_numerical_result(4, "training_time_initial", json_raw_data=False)

    print(f"Running benchmark ({len(contexts)} contexts, {TRIALS} trial(s) each)...")
    benchmark.run(trials=TRIALS, verbose=True)
    run_dir = benchmark.process_and_save_results(root_dir=BENCHMARK_ROOT)
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
