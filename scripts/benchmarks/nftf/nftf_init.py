from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.tools.propagate as propagate
from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.composite_model import CompositeConditionalModel, CompositeDensityModel
from rational_factor.models.density_model import LogisticSigmoid
from rational_factor.models.domain_transformation import MaskedAffineNFTF
from rational_factor.models.factor_forms import LinearFF, LinearRFF
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood
from rational_factor.tools.benchmark import Benchmark
from rational_factor.tools.misc import data_bounds

PROBLEM = "dubins_trailer"

CONTEXT_WITH_NFTF = {
    "use_nftf": True,
    "use_nftf_prefit": True,
    "n_basis": 500,
    "tran_params": {
        "n_epochs_per_group": [5, 5],
        "iterations": 50,
        "pre_train_epochs": 10,
        "lr_basis": 5e-3,
        "lr_weights": 1e-3,
        "lr_dtf": 5e-4,
    },
    "init_params": {
        "n_epochs_per_group": [20, 5],
        "iterations": 100,
        "lr_basis": 5e-3,
        "lr_weights": 1e-3,
    },
    "batch_size": 256,
    "ls_temp": 0.1,
    "reg_covar_joint": 1e-3,
    "verbose": True,
}

CONTEXT_WITHOUT_NFTF = {
    "use_nftf": False,
    "use_nftf_prefit": False,
    "n_basis": 500,
    "tran_params": {
        "n_epochs_per_group": [5, 5],
        "iterations": 100,
        "lr_basis": 5e-3,
        "lr_weights": 1e-3,
    },
    "init_params": {
        "n_epochs_per_group": [20, 5],
        "iterations": 100,
        "lr_basis": 5e-3,
        "lr_weights": 1e-3,
    },
    "batch_size": 256,
    "ls_temp": 0.1,
    "reg_covar_joint": 1e-3,
    "verbose": True,
}

CONTEXT_WITH_NFTF_NO_PREFIT = {
    "use_nftf": True,
    "use_nftf_prefit": False,
    "n_basis": 500,
    "tran_params": {
        "n_epochs_per_group": [5, 5],
        "iterations": 50,
        "lr_basis": 5e-3,
        "lr_weights": 1e-3,
        "lr_dtf": 5e-4,
    },
    "init_params": {
        "n_epochs_per_group": [20, 5],
        "iterations": 100,
        "lr_basis": 5e-3,
        "lr_weights": 1e-3,
    },
    "batch_size": 256,
    "ls_temp": 0.1,
    "reg_covar_joint": 1e-3,
    "verbose": True,
}

TRIALS = 5
BENCHMARK_ROOT = "benchmark_data"


def main() -> None:
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    problem = FULLY_OBSERVABLE_PROBLEMS[PROBLEM]
    system = problem.system

    x0_data = problem.train_initial_state_data()
    x_k_data, x_kp1_data = problem.train_state_transition_data()
    test_traj_data = problem.test_data()

    x0_dataset = TensorDataset(x0_data)
    xp_dataset = TensorDataset(x_kp1_data, x_k_data)

    def experiment(
        use_nftf: bool,
        use_nftf_prefit: bool,
        n_basis: int,
        tran_params: dict,
        init_params: dict,
        batch_size: int,
        ls_temp: float,
        reg_covar_joint: float,
        verbose: bool = True,
    ):
        x0_dataloader = DataLoader(x0_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
        xp_dataloader = DataLoader(xp_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
        x_k_dataloader = DataLoader(TensorDataset(x_k_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

        nftf = MaskedAffineNFTF(system.dim(), trainable=True, hidden_features=256, n_layers=8).to(device) if use_nftf else None

        if use_nftf and use_nftf_prefit:
            loc, scale = data_bounds(x_k_data, mode="center_lengths")
            base_distribution = LogisticSigmoid(
                system.dim(),
                temperature=ls_temp,
                loc=loc.to(device),
                scale=scale.to(device),
            )
            decorrupter_density = CompositeDensityModel([nftf], base_distribution).to(device)
            pretrain_optimizer = torch.optim.Adam(
                nftf.parameters(), lr=tran_params["lr_dtf"], weight_decay=tran_params["lr_dtf"]
            )
            decorrupter_density, best_loss_pretrain, training_time_pretrain = train.train(
                decorrupter_density,
                x_k_dataloader,
                {"mle": loss.mle_loss},
                pretrain_optimizer,
                epochs=tran_params["pre_train_epochs"],
                verbose=verbose,
                use_best="mle",
            )

            with torch.no_grad():
                y_k_data, _ = nftf(x_k_data.to(device))
                y_kp1_data, _ = nftf(x_kp1_data.to(device))
                y_k_data = y_k_data.to(torch.device("cpu"))
                y_kp1_data = y_kp1_data.to(torch.device("cpu"))

            y_joint_data = torch.cat([y_k_data, y_kp1_data], dim=1)
            gmm_lf = train.fit_gaussian_lf_em(
                y_joint_data, n_components=n_basis, reg_covar=reg_covar_joint, max_iter=100
            )
            phi_marginal = gmm_lf.basis.marginal(marginal_dims=range(system.dim()))
            phi_means, phi_stds = phi_marginal.means_stds()
            phi_params = torch.stack([phi_means, phi_stds], dim=-1)

            psi_marginal = gmm_lf.basis.marginal(marginal_dims=range(system.dim(), 2 * system.dim()))
            psi_means, psi_stds = psi_marginal.means_stds()
            psi_params = torch.stack([psi_means, psi_stds], dim=-1)

            phi_basis = GaussianBasis(uparams_init=phi_params).to(device)
            psi_basis = GaussianBasis(uparams_init=psi_params).to(device)
            psi0_basis = GaussianBasis.random_init(
                system.dim(),
                n_basis=n_basis,
                offsets=torch.tensor([0.0, 20.0], device=device),
                variance=30.0,
                min_std=1e-4,
            ).to(device)
        else:
            best_loss_pretrain = float("nan")
            training_time_pretrain = 0.0
            phi_basis = GaussianBasis.random_init(
                system.dim(),
                n_basis=n_basis,
                offsets=torch.tensor([0.0, 20.0], device=device),
                variance=30.0,
                min_std=1e-4,
            ).to(device)
            psi_basis = GaussianBasis.random_init(
                system.dim(),
                n_basis=n_basis,
                offsets=torch.tensor([0.0, 20.0], device=device),
                variance=30.0,
                min_std=1e-4,
            ).to(device)
            psi0_basis = GaussianBasis.random_init(
                system.dim(),
                n_basis=n_basis,
                offsets=torch.tensor([0.0, 20.0], device=device),
                variance=30.0,
                min_std=1e-4,
            ).to(device)

        if use_nftf:
            tran_model = CompositeConditionalModel([nftf], LinearRFF(phi_basis, psi_basis)).to(device)
            optimizers = {
                "dtf_and_basis": torch.optim.Adam(
                    [
                        {"params": tran_model.conditional_density_model.basis_params(), "lr": tran_params["lr_basis"]},
                        {"params": tran_model.domain_tfs.parameters(), "lr": tran_params["lr_dtf"]},
                    ]
                ),
                "weights": torch.optim.Adam(
                    tran_model.conditional_density_model.weight_params(), lr=tran_params["lr_weights"]
                ),
            }
        else:
            tran_model = LinearRFF(phi_basis, psi_basis).to(device)
            optimizers = {
                "basis": torch.optim.Adam(tran_model.basis_params(), lr=tran_params["lr_basis"]),
                "weights": torch.optim.Adam(tran_model.weight_params(), lr=tran_params["lr_weights"]),
            }

        tran_model, best_loss_tran, training_time_tran = train.train_iterate(
            tran_model,
            xp_dataloader,
            {"mle": loss.conditional_mle_loss},
            optimizers,
            epochs_per_group=tran_params["n_epochs_per_group"],
            iterations=tran_params["iterations"],
            verbose=verbose,
            use_best="mle",
        )

        if use_nftf:
            trained_nftf = MaskedAffineNFTF.copy_from_trainable(nftf).to(device)
            init_model = CompositeDensityModel(
                [trained_nftf], LinearFF.from_rff(tran_model.conditional_density_model, psi0_basis)
            ).to(device)
            init_optimizers = {
                "basis": torch.optim.Adam(init_model.density_model.basis_params(), lr=init_params["lr_basis"]),
                "weights": torch.optim.Adam(init_model.density_model.weight_params(), lr=init_params["lr_weights"]),
            }
        else:
            trained_nftf = None
            init_model = LinearFF.from_rff(tran_model, psi0_basis).to(device)
            init_optimizers = {
                "basis": torch.optim.Adam(init_model.basis_params(), lr=init_params["lr_basis"]),
                "weights": torch.optim.Adam(init_model.weight_params(), lr=init_params["lr_weights"]),
            }

        init_model, best_loss_init, training_time_init = train.train_iterate(
            init_model,
            x0_dataloader,
            {"mle": loss.mle_loss},
            init_optimizers,
            epochs_per_group=init_params["n_epochs_per_group"],
            iterations=init_params["iterations"],
            verbose=verbose,
            use_best="mle",
        )

        analysis_device = torch.device("cpu")
        if use_nftf:
            init_model = init_model.to(analysis_device).eval()
            tran_model = tran_model.to(analysis_device).eval()
            trained_nftf = trained_nftf.to(analysis_device).eval()
            base_belief_seq = propagate.propagate(
                init_model.density_model, tran_model.conditional_density_model, n_steps=problem.n_timesteps
            )
            belief_seq = [
                CompositeDensityModel([trained_nftf], belief).to(analysis_device).eval()
                for belief in base_belief_seq
            ]
        else:
            init_model = init_model.to(analysis_device).eval()
            tran_model = tran_model.to(analysis_device).eval()
            belief_seq = [
                belief.to(analysis_device).eval()
                for belief in propagate.propagate(init_model, tran_model, n_steps=problem.n_timesteps)
            ]

        ll_per_step = []
        for i in range(problem.n_timesteps):
            data_i = test_traj_data[i].to(analysis_device)
            ll = avg_log_likelihood(belief_seq[i], data_i)
            ll_per_step.append(ll.detach().cpu().reshape(()))
        prop_ll_vector = torch.stack(ll_per_step).to(dtype=torch.float32)

        return (
            prop_ll_vector,
            torch.tensor([float(best_loss_pretrain)], dtype=torch.float32),
            torch.tensor([float(training_time_pretrain)], dtype=torch.float32),
            torch.tensor([float(best_loss_tran)], dtype=torch.float32),
            torch.tensor([float(best_loss_init)], dtype=torch.float32),
            torch.tensor([float(training_time_tran)], dtype=torch.float32),
            torch.tensor([float(training_time_init)], dtype=torch.float32),
        )

    contexts = [
        {"name": "DTF w/ Init", "params": CONTEXT_WITH_NFTF},
        {"name": "DTF w/o Init", "params": CONTEXT_WITH_NFTF_NO_PREFIT},
        {"name": "No DTF", "params": CONTEXT_WITHOUT_NFTF},
    ]

    benchmark = Benchmark(name=Path(__file__).stem + "_" + PROBLEM)
    benchmark.set_experiment_fn(experiment)
    benchmark.set_contexts(contexts)

    benchmark.set_numerical_result(0, "avg_log_likelihood_per_timestep", json_raw_data=False)
    benchmark.set_numerical_result(1, "best_loss_pretrain_decorrupter", json_raw_data=False)
    benchmark.set_numerical_result(2, "training_time_pretrain_decorrupter", json_raw_data=False)
    benchmark.set_numerical_result(3, "best_loss_transition", json_raw_data=False)
    benchmark.set_numerical_result(4, "best_loss_initial", json_raw_data=False)
    benchmark.set_numerical_result(5, "training_time_transition", json_raw_data=False)
    benchmark.set_numerical_result(6, "training_time_initial", json_raw_data=False)

    print(f"Running benchmark ({len(contexts)} contexts, {TRIALS} trial(s) each)...")
    benchmark.run(trials=TRIALS, verbose=True)
    run_dir = benchmark.process_and_save_results(root_dir=BENCHMARK_ROOT)
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
