from __future__ import annotations
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.tools.propagate as propagate
from rational_factor.models.basis_functions import BetaBasis
from rational_factor.models.composite_model import CompositeConditionalModel, CompositeDensityModel
from rational_factor.models.domain_transformation import ErfSeparableTF, MaskedAffineNFTF
from rational_factor.models.factor_forms import LinearFF, LinearRFF
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood
from rational_factor.tools.benchmark import Benchmark

CONTEXT_USE_NFTF_FALSE = {
    "use_nftf": False,
    "n_basis": 1000,
    "tran_params": {
        "n_epochs_per_group": [10, 5],
        "iterations": 20,
        "lr_basis": 1e-2,
        "lr_weights": 1e-2,
        "lr_wrap": 1e-3,
    },
    "init_params": {
        "n_epochs_per_group": [20, 5],
        "iterations": 50,
        "lr_basis": 1e-2,
        "lr_weights": 1e-2,
    },
    "batch_size": 256,
    "var_reg_strength": 1e-3,
    "verbose": True,
}

CONTEXT_USE_NFTF_TRUE = {
    "use_nftf": True,
    "n_basis": 1000,
    "tran_params": {
        "n_epochs_per_group": [10, 5],
        "iterations": 20,
        "lr_basis": 1e-2,
        "lr_weights": 1e-2,
        "lr_dtf": 1e-3,
        "lr_wrap": 1e-3,
    },
    "init_params": {
        "n_epochs_per_group": [20, 5],
        "iterations": 50,
        "lr_basis": 1e-2,
        "lr_weights": 1e-2,
    },
    "batch_size": 256,
    "var_reg_strength": 1e-3,
    "verbose": True,
}

TRIALS = 15
BENCHMARK_ROOT = "benchmark_data"

def main() -> None:

    ######## SETUP ########
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    problem = FULLY_OBSERVABLE_PROBLEMS["van_der_pol"]

    system = problem.system
    x0_data, x_k_data, x_kp1_data = problem.train_data()
    test_traj_data = problem.test_data()

    x0_dataset = TensorDataset(x0_data)
    xp_dataset = TensorDataset(x_kp1_data, x_k_data)

    def experiment(
        use_nftf: bool,
        n_basis: int,
        tran_params: dict,
        init_params: dict,
        batch_size: int,
        var_reg_strength: float,
        verbose: bool = True,
    ):
        x0_dataloader = DataLoader(x0_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
        xp_dataloader = DataLoader(xp_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

        offsets = torch.tensor([10.0, 10.0], device=device)
        phi_basis = BetaBasis.random_init(
            system.dim(),
            n_basis=n_basis,
            offsets=offsets,
            variance=30.0,
            min_concentration=1.0,
        ).to(device)
        psi_basis = BetaBasis.random_init(
            system.dim(),
            n_basis=n_basis,
            offsets=offsets,
            variance=30.0,
            min_concentration=1.0,
        ).to(device)
        psi0_basis = BetaBasis.random_init(
            system.dim(),
            n_basis=n_basis,
            offsets=offsets,
            variance=30.0,
            min_concentration=1.0,
        ).to(device)

        wrap_tf = ErfSeparableTF.from_data(x_k_data, trainable=True).to(device)
        nftf = MaskedAffineNFTF(system.dim(), trainable=True, hidden_features=128, n_layers=5).to(device) if use_nftf else None

        if use_nftf:
            tran_model = CompositeConditionalModel([nftf, wrap_tf], LinearRFF(phi_basis, psi_basis)).to(device)
        else:
            tran_model = CompositeConditionalModel([wrap_tf], LinearRFF(phi_basis, psi_basis)).to(device)

        mle_loss_fn = loss.conditional_mle_loss
        var_reg_loss_fn = lambda model, x, xp: var_reg_strength * (
            loss.beta_basis_concentration_reg_loss(model.conditional_density_model.phi_basis)
            + loss.beta_basis_concentration_reg_loss(model.conditional_density_model.psi_basis)
        )

        if use_nftf:
            optimizers = {
                "dtf_and_basis": torch.optim.Adam(
                    [
                        {"params": tran_model.conditional_density_model.basis_params(), "lr": tran_params["lr_basis"]},
                        {"params": tran_model.domain_tfs[0].parameters(), "lr": tran_params["lr_dtf"]},
                        {"params": tran_model.domain_tfs[1].parameters(), "lr": tran_params["lr_wrap"]},
                    ]
                ),
                "weights": torch.optim.Adam(
                    tran_model.conditional_density_model.weight_params(), lr=tran_params["lr_weights"]
                ),
            }
        else:
            optimizers = {
                "basis": torch.optim.Adam(
                    [
                        {"params": tran_model.conditional_density_model.basis_params(), "lr": tran_params["lr_basis"]},
                        {"params": tran_model.domain_tfs.parameters(), "lr": tran_params["lr_wrap"]},
                    ]
                ),
                "weights": torch.optim.Adam(
                    tran_model.conditional_density_model.weight_params(), lr=tran_params["lr_weights"]
                ),
            }

        tran_model, best_loss_tran, training_time_tran = train.train_iterate(
            tran_model,
            xp_dataloader,
            {"mle": mle_loss_fn, "var_reg": var_reg_loss_fn},
            optimizers,
            epochs_per_group=tran_params["n_epochs_per_group"],
            iterations=tran_params["iterations"],
            verbose=verbose,
            use_best="mle",
        )

        trained_nftf = MaskedAffineNFTF.copy_from_trainable(nftf).to(device) if use_nftf else None
        trained_domain_tf = ErfSeparableTF.copy_from_trainable(wrap_tf).to(device)

        if use_nftf:
            init_model = CompositeDensityModel(
                [trained_nftf, trained_domain_tf], LinearFF.from_rff(tran_model.conditional_density_model, psi0_basis)
            ).to(device)
        else:
            init_model = CompositeDensityModel(
                [trained_domain_tf], LinearFF.from_rff(tran_model.conditional_density_model, psi0_basis)
            ).to(device)

        mle_loss_fn = loss.mle_loss
        var_reg_loss_fn = lambda model, x: var_reg_strength * loss.beta_basis_concentration_reg_loss(
            model.density_model.psi0_basis
        )
        optimizers = {
            "basis": torch.optim.Adam(init_model.density_model.basis_params(), lr=init_params["lr_basis"]),
            "weights": torch.optim.Adam(init_model.density_model.weight_params(), lr=init_params["lr_weights"]),
        }

        init_model, best_loss_init, training_time_init = train.train_iterate(
            init_model,
            x0_dataloader,
            {"mle": mle_loss_fn, "var_reg": var_reg_loss_fn},
            optimizers,
            epochs_per_group=init_params["n_epochs_per_group"],
            iterations=init_params["iterations"],
            verbose=verbose,
            use_best="mle",
        )

        base_belief_seq = propagate.propagate(
            init_model.density_model, tran_model.conditional_density_model, n_steps=problem.n_timesteps
        )
        if use_nftf:
            belief_seq = [CompositeDensityModel([trained_nftf, trained_domain_tf], belief) for belief in base_belief_seq]
        else:
            belief_seq = [CompositeDensityModel([trained_domain_tf], belief) for belief in base_belief_seq]

        ll_per_step = []
        for i in range(problem.n_timesteps):
            data_i = test_traj_data[i].to(device)
            ll = avg_log_likelihood(belief_seq[i], data_i)
            ll_per_step.append(ll.detach().cpu().reshape(()))
        prop_ll_vector = torch.stack(ll_per_step).to(dtype=torch.float32)

        return (
            prop_ll_vector,
            torch.tensor([float(best_loss_tran)], dtype=torch.float32),
            torch.tensor([float(best_loss_init)], dtype=torch.float32),
            torch.tensor([float(training_time_tran)], dtype=torch.float32),
            torch.tensor([float(training_time_init)], dtype=torch.float32),
        )

    contexts = [
        {"name": "wo_nftf", "params": CONTEXT_USE_NFTF_FALSE},
        {"name": "w_nftf", "params": CONTEXT_USE_NFTF_TRUE},
    ]

    benchmark = Benchmark(name=Path(__file__).stem)
    benchmark.set_experiment_fn(experiment)
    benchmark.set_contexts(contexts)

    benchmark.set_numerical_result(0, "avg_log_likelihood_per_timestep", json_raw_data=False)
    benchmark.set_numerical_result(1, "best_loss_transition", json_raw_data=False)
    benchmark.set_numerical_result(2, "best_loss_initial", json_raw_data=False)
    benchmark.set_numerical_result(3, "training_time_transition", json_raw_data=False)
    benchmark.set_numerical_result(4, "training_time_initial", json_raw_data=False)

    print(f"Running benchmark ({len(contexts)} contexts, {TRIALS} trial(s) each)...")
    benchmark.run(trials=TRIALS, verbose=True)
    run_dir = benchmark.process_and_save_results(root_dir=BENCHMARK_ROOT)
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
