from __future__ import annotations
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.tools.propagate as propagate
from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.composite_model import CompositeConditionalModel, CompositeDensityModel
from rational_factor.models.domain_transformation import MaskedRQSNFTF
from rational_factor.models.factor_forms import LinearFF, LinearRFF
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood
from rational_factor.tools.benchmark import Benchmark
from rational_factor.tools.misc import train_test_split

USE_DTF = False

TRAN_PARAMS = {
    "n_epochs_per_group": [10, 5],
    "iterations": 40,
    "lr_basis": 5e-2,
    "lr_weights": 1e-2,
    "lr_dtf": 1e-3,
}
INIT_PARAMS = {
    "n_epochs_per_group": [20, 5],
    "iterations": 50,
    "lr_basis": 1e-2,
    "lr_weights": 1e-2,
}

N_BASIS_VALUES = (200, 500, 1000, 2000, 4000)

TRIALS = 5
BENCHMARK_ROOT = "benchmark_data"

def _context_params(*, use_dtf: bool, n_basis: int) -> dict:
    tran = {
        "n_epochs_per_group": TRAN_PARAMS["n_epochs_per_group"],
        "iterations": TRAN_PARAMS["iterations"],
        "lr_basis": TRAN_PARAMS["lr_basis"],
        "lr_weights": TRAN_PARAMS["lr_weights"],
    }
    if use_dtf:
        tran["lr_dtf"] = TRAN_PARAMS["lr_dtf"]
    return {
        "use_dtf": use_dtf,
        "n_basis": n_basis,
        "tran_params": tran,
        "init_params": dict(INIT_PARAMS),
        "batch_size": 256,
        "var_reg_strength": 5e-1,
        "verbose": True,
    }


def main() -> None:
    ######## SETUP ########
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    problem = FULLY_OBSERVABLE_PROBLEMS["van_der_pol"]

    system = problem.system
    x0_data, x_k_data, x_kp1_data = problem.train_data()
    test_traj_data = problem.test_data()

    x0_train, x0_val = train_test_split(x0_data, test_size=0.2)
    x_k_train, x_k_val, x_kp1_train, x_kp1_val = train_test_split(x_k_data, x_kp1_data, test_size=0.2)
    x0_dataset = TensorDataset(x0_train)
    x0_val_dataset = TensorDataset(x0_val)
    xp_dataset = TensorDataset(x_kp1_train, x_k_train)
    xp_val_dataset = TensorDataset(x_kp1_val, x_k_val)

    def experiment(
        use_dtf: bool,
        n_basis: int,
        tran_params: dict,
        init_params: dict,
        batch_size: int,
        var_reg_strength: float,
        verbose: bool = True,
    ):
        x0_dataloader = DataLoader(x0_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
        xp_dataloader = DataLoader(xp_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
        x0_val_dataloader = DataLoader(x0_val_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
        xp_val_dataloader = DataLoader(xp_val_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

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

        nftf = MaskedRQSNFTF(system.dim(), trainable=True, hidden_features=128, n_layers=5).to(device) if use_dtf else None

        if use_dtf:
            tran_model = CompositeConditionalModel([nftf], LinearRFF(phi_basis, psi_basis)).to(device)
        else:
            tran_model = LinearRFF(phi_basis, psi_basis).to(device)

        mle_loss_fn = loss.conditional_mle_loss
        if use_dtf:
            var_reg_loss_fn = lambda model, x, xp: var_reg_strength * (
                loss.gaussian_basis_var_reg_loss(model.conditional_density_model.phi_basis, mean=True)
                + loss.gaussian_basis_var_reg_loss(model.conditional_density_model.psi_basis, mean=True)
            )
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
            var_reg_loss_fn = lambda model, x, xp: var_reg_strength * (
                loss.gaussian_basis_var_reg_loss(model.phi_basis, mean=True)
                + loss.gaussian_basis_var_reg_loss(model.psi_basis, mean=True)
            )
            optimizers = {
                "basis": torch.optim.Adam(tran_model.basis_params(), lr=tran_params["lr_basis"]),
                "weights": torch.optim.Adam(tran_model.weight_params(), lr=tran_params["lr_weights"]),
            }

        tran_model, best_loss_tran, training_time_tran = train.train_iterate(
            tran_model,
            xp_dataloader,
            {"mle": mle_loss_fn, "var_reg": var_reg_loss_fn},
            optimizers,
            labeled_validation_loss_fns={"val_mle": mle_loss_fn},
            validation_data_loader=xp_val_dataloader,
            epochs_per_group=tran_params["n_epochs_per_group"],
            iterations=tran_params["iterations"],
            verbose=verbose,
            use_best="val_mle",
        )

        trained_nftf = MaskedRQSNFTF.copy_from_trainable(nftf).to(device) if use_dtf else None

        if use_dtf:
            init_model = CompositeDensityModel(
                [trained_nftf], LinearFF.from_rff(tran_model.conditional_density_model, psi0_basis)
            ).to(device)
        else:
            init_model = LinearFF.from_rff(tran_model, psi0_basis).to(device)

        mle_loss_fn = loss.mle_loss
        if use_dtf:
            var_reg_loss_fn = lambda model, x: var_reg_strength * loss.gaussian_basis_var_reg_loss(
                model.density_model.psi0_basis, mean=True
            )
            optimizers = {
                "basis": torch.optim.Adam(init_model.density_model.basis_params(), lr=init_params["lr_basis"]),
                "weights": torch.optim.Adam(init_model.density_model.weight_params(), lr=init_params["lr_weights"]),
            }
        else:
            var_reg_loss_fn = lambda model, x: var_reg_strength * loss.gaussian_basis_var_reg_loss(
                model.psi0_basis, mean=True
            )
            optimizers = {
                "basis": torch.optim.Adam(init_model.basis_params(), lr=init_params["lr_basis"]),
                "weights": torch.optim.Adam(init_model.weight_params(), lr=init_params["lr_weights"]),
            }

        init_model, best_loss_init, training_time_init = train.train_iterate(
            init_model,
            x0_dataloader,
            {"mle": mle_loss_fn, "var_reg": var_reg_loss_fn},
            optimizers,
            labeled_validation_loss_fns={"val_mle": mle_loss_fn},
            validation_data_loader=x0_val_dataloader,
            epochs_per_group=init_params["n_epochs_per_group"],
            iterations=init_params["iterations"],
            verbose=verbose,
            use_best="val_mle",
        )

        if use_dtf:
            base_belief_seq = propagate.propagate(
                init_model.density_model, tran_model.conditional_density_model, n_steps=problem.n_timesteps
            )
            belief_seq = [CompositeDensityModel([trained_nftf], belief) for belief in base_belief_seq]
        else:
            belief_seq = propagate.propagate(init_model, tran_model, n_steps=problem.n_timesteps)

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
        {"name": f"n{n}", "params": _context_params(use_dtf=USE_DTF, n_basis=n)}
        for n in N_BASIS_VALUES
    ]

    bench_name = f"{Path(__file__).stem}_{'dtf' if USE_DTF else 'no_dtf'}"
    benchmark = Benchmark(name=bench_name)
    benchmark.set_experiment_fn(experiment)
    benchmark.set_contexts(contexts)

    benchmark.set_numerical_result(0, "avg_log_likelihood_per_timestep", json_raw_data=False)
    benchmark.set_numerical_result(1, "best_loss_transition", json_raw_data=False)
    benchmark.set_numerical_result(2, "best_loss_initial", json_raw_data=False)
    benchmark.set_numerical_result(3, "training_time_transition", json_raw_data=False)
    benchmark.set_numerical_result(4, "training_time_initial", json_raw_data=False)

    print(
        f"Running benchmark ({len(contexts)} n_basis contexts, use_dtf={USE_DTF}, "
        f"{TRIALS} trial(s) each)..."
    )
    benchmark.run(trials=TRIALS, verbose=True)
    run_dir = benchmark.process_and_save_results(root_dir=BENCHMARK_ROOT)
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
