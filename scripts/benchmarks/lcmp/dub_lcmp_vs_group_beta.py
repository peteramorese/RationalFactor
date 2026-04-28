from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.tools.propagate as propagate
from rational_factor.models.basis_functions import BetaBasis, GaussianBasis
from rational_factor.models.composite_model import CompositeConditionalModel, CompositeDensityModel
from rational_factor.models.density_model import LogisticSigmoid
from rational_factor.models.domain_transformation import (
    ErfSeparableTF,
    MaskedAffineNFTF,
    StackedTF,
    VolumePreservingNFTF,
)
from rational_factor.models.factor_forms import LinearFF, LinearRFF
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood
from rational_factor.tools.benchmark import Benchmark
from rational_factor.tools.misc import data_bounds

TRIALS = 5
BENCHMARK_ROOT = "benchmark_data"

LCMP_CONTEXT = {
    "approach": "lcmp",
    "batch_size": 256,
    "n_basis": 1000,
    "ls_temp": 0.1,
    "reg_covar": 8e-2,
    "decorrupter_params": {
        "epochs": 50,
        "lr": 1e-3,
        "weight_decay": 5e-1,
    },
    "mover_params": {
        "epochs": 50,
        "lr": 5e-4,
        "weight_decay": 5e-1,
    },
    "mover_iterations": 5,
    "init_params": {
        "n_epochs_per_group": [15, 5],
        "iterations": 200,
        "lr_basis": 1e-2,
        "lr_weights": 1e-1,
    },
    "verbose": True,
}

GROUP_BETA_CONTEXT = {
    "approach": "composite_group_beta",
    "batch_size": 256,
    "n_basis": 1000,
    "tran_params": {
        "n_epochs_per_group": [15, 5],
        "iterations": 100,
        "lr_basis": 1e-2,
        "lr_weights": 1e-2,
        "lr_wrap": 1e-3,
    },
    "init_params": {
        "n_epochs_per_group": [20, 5],
        "iterations": 200,
        "lr_basis": 5e-3,
        "lr_weights": 1e-2,
    },
    "verbose": True,
}


def main() -> None:
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    cpu = torch.device("cpu")
    print("Using device: ", device)

    problem = FULLY_OBSERVABLE_PROBLEMS["dubins_trailer"]
    system = problem.system

    x0_data = problem.train_initial_state_data()
    x_k_data, x_kp1_data = problem.train_state_transition_data()
    test_traj_data = problem.test_data()

    x0_dataset = TensorDataset(x0_data)
    xp_dataset = TensorDataset(x_kp1_data, x_k_data)

    def run_lcmp(
        batch_size: int,
        n_basis: int,
        ls_temp: float,
        reg_covar: float,
        decorrupter_params: dict,
        mover_params: dict,
        mover_iterations: int,
        init_params: dict,
        verbose: bool,
    ):
        x_dataloader = DataLoader(TensorDataset(x_k_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

        loc, scale = data_bounds(x_k_data, mode="center_lengths")
        loc = loc.to(device)
        scale = scale.to(device)
        base_distribution = LogisticSigmoid(system.dim(), temperature=ls_temp, loc=loc, scale=scale)
        decorrupter = MaskedAffineNFTF(system.dim(), trainable=True, hidden_features=128, n_layers=5).to(device)
        decorrupter_density = CompositeDensityModel([decorrupter], base_distribution).to(device)

        optimizer = torch.optim.Adam(
            decorrupter.parameters(),
            lr=decorrupter_params["lr"],
            weight_decay=decorrupter_params["weight_decay"],
        )
        dtf_concentration_loss_xk = lambda composite_model, x: loss.dtf_data_concentration_loss(
            composite_model.domain_tfs[0],
            x,
            concentration_point=loc,
            radius=scale,
        )
        decorrupter_density, _, _ = train.train(
            decorrupter_density,
            x_dataloader,
            {"mle": loss.mle_loss, "dtf_conc": dtf_concentration_loss_xk},
            optimizer,
            epochs=decorrupter_params["epochs"],
            verbose=verbose,
            use_best="mle",
            clip_grad_norm=5.0,
            restore_loss_threshold=50.0,
        )
        decorrupter_trained = MaskedAffineNFTF.copy_from_trainable(decorrupter).to(device)

        x_k_data_device = x_k_data.to(device)
        x_kp1_data_device = x_kp1_data.to(device)
        x0_data_device = x0_data.to(device)
        y_k_data, _ = decorrupter_trained(x_k_data_device)
        y_kp1_data, _ = decorrupter_trained(x_kp1_data_device)
        y0_data, _ = decorrupter_trained(x0_data_device)

        mover = VolumePreservingNFTF(system.dim(), trainable=True, hidden_features=256, n_layers=6).to(device)
        y_joint_data = torch.cat([y_k_data, y_kp1_data], dim=1)
        mover_joint = StackedTF([mover, mover])

        z_joint_data = y_joint_data
        best_loss_tran = 0.0
        training_time_tran = 0.0
        for _ in range(mover_iterations):
            gmm_lf = train.fit_gaussian_lf_em(
                z_joint_data.to(cpu),
                n_components=n_basis,
                reg_covar=reg_covar,
                max_iter=100,
            )
            mover_density = CompositeDensityModel([mover_joint], gmm_lf).to(device)
            y_joint_dataloader = DataLoader(
                TensorDataset(y_joint_data.to(cpu)),
                batch_size=batch_size,
                shuffle=True,
                pin_memory=use_gpu,
            )
            optimizer = torch.optim.Adam(
                mover_density.domain_tfs[0].parameters(),
                lr=mover_params["lr"],
                weight_decay=mover_params["weight_decay"],
            )
            mover_density, best_loss_tran, dt = train.train(
                mover_density,
                y_joint_dataloader,
                {"mle": loss.mle_loss},
                optimizer,
                epochs=mover_params["epochs"],
                verbose=verbose,
                use_best="mle",
                clip_grad_norm=5.0,
                restore_loss_threshold=50.0,
            )
            training_time_tran += float(dt)

            z_joint_data, _ = mover_joint(y_joint_data)

        mover_trained = VolumePreservingNFTF.copy_from_trainable(mover).to(device)

        weights = gmm_lf.get_w()
        z_marginal = gmm_lf.basis.marginal(marginal_dims=range(system.dim()))
        z_means, z_stds = z_marginal.means_stds()
        z_params = torch.stack([z_means, z_stds], dim=-1)

        zp_marginal = gmm_lf.basis.marginal(marginal_dims=range(system.dim(), 2 * system.dim()))
        zp_means, zp_stds = zp_marginal.means_stds()
        zp_params = torch.stack([zp_means, zp_stds], dim=-1)

        phi_basis = GaussianBasis(fixed_params=z_params).to(device)
        psi_basis = GaussianBasis(fixed_params=zp_params).to(device)
        psi0_basis = GaussianBasis.random_init(
            system.dim(),
            n_basis=n_basis,
            offsets=torch.tensor([0.0, 20.0], device=device),
            variance=30.0,
            min_std=1e-4,
        ).to(device)

        lrff = LinearRFF(phi_basis, psi_basis, a_fixed=weights.to(device)).to(device)
        ff = LinearFF(lrff.get_a(), phi_basis, psi0_basis).to(device)

        z0_data, _ = mover_trained(y0_data)
        z0_dataloader = DataLoader(TensorDataset(z0_data.to(cpu)), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
        optimizers = {
            "basis": torch.optim.Adam(ff.basis_params(), lr=init_params["lr_basis"]),
            "weights": torch.optim.Adam(ff.weight_params(), lr=init_params["lr_weights"]),
        }
        ff, best_loss_init, training_time_init = train.train_iterate(
            ff,
            z0_dataloader,
            {"mle": loss.mle_loss},
            optimizers,
            epochs_per_group=init_params["n_epochs_per_group"],
            iterations=init_params["iterations"],
            verbose=verbose,
            use_best="mle",
        )

        base_belief_seq = propagate.propagate(ff, lrff, n_steps=problem.n_timesteps)
        belief_seq = [CompositeDensityModel([decorrupter_trained, mover_trained], belief) for belief in base_belief_seq]
        return belief_seq, float(best_loss_tran), float(best_loss_init), float(training_time_tran), float(training_time_init)

    def run_group_beta(
        batch_size: int,
        n_basis: int,
        tran_params: dict,
        init_params: dict,
        verbose: bool,
    ):
        x0_dataloader = DataLoader(x0_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
        xp_dataloader = DataLoader(xp_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

        offsets = torch.tensor([-1.0, -1.0], device=device)
        variance = 1.0
        phi_basis = BetaBasis.random_init(
            system.dim(),
            n_basis=n_basis,
            offsets=offsets,
            variance=variance,
            min_concentration=1.0,
        ).to(device)
        psi_basis = BetaBasis.random_init(
            system.dim(),
            n_basis=n_basis,
            offsets=offsets,
            variance=variance,
            min_concentration=1.0,
        ).to(device)
        psi0_basis = BetaBasis.random_init(
            system.dim(),
            n_basis=n_basis,
            offsets=offsets,
            variance=variance,
            min_concentration=1.0,
        ).to(device)

        wrap_tf = ErfSeparableTF.from_data(torch.cat([x0_data, x_k_data], dim=0), trainable=True).to(device)
        tran_model = CompositeConditionalModel([wrap_tf], LinearRFF(phi_basis, psi_basis)).to(device)
        optimizers = {
            "basis": torch.optim.Adam(
                [
                    {"params": tran_model.conditional_density_model.basis_params(), "lr": tran_params["lr_basis"]},
                    {"params": tran_model.domain_tfs.parameters(), "lr": tran_params["lr_wrap"]},
                ]
            ),
            "weights": torch.optim.Adam(
                tran_model.conditional_density_model.weight_params(),
                lr=tran_params["lr_weights"],
            ),
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

        trained_domain_tf = ErfSeparableTF.copy_from_trainable(wrap_tf).to(device)
        init_model = CompositeDensityModel(
            [trained_domain_tf],
            LinearFF.from_rff(tran_model.conditional_density_model, psi0_basis),
        ).to(device)
        optimizers = {
            "basis": torch.optim.Adam(init_model.density_model.basis_params(), lr=init_params["lr_basis"]),
            "weights": torch.optim.Adam(init_model.density_model.weight_params(), lr=init_params["lr_weights"]),
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
        )

        base_belief_seq = propagate.propagate(
            init_model.density_model,
            tran_model.conditional_density_model,
            n_steps=problem.n_timesteps,
        )
        belief_seq = [CompositeDensityModel([trained_domain_tf], belief) for belief in base_belief_seq]
        return belief_seq, float(best_loss_tran), float(best_loss_init), float(training_time_tran), float(training_time_init)

    def experiment(
        approach: str,
        batch_size: int,
        n_basis: int,
        verbose: bool = True,
        **params,
    ):
        if approach == "lcmp":
            belief_seq, best_loss_tran, best_loss_init, training_time_tran, training_time_init = run_lcmp(
                batch_size=batch_size,
                n_basis=n_basis,
                ls_temp=params["ls_temp"],
                reg_covar=params["reg_covar"],
                decorrupter_params=params["decorrupter_params"],
                mover_params=params["mover_params"],
                mover_iterations=params["mover_iterations"],
                init_params=params["init_params"],
                verbose=verbose,
            )
        elif approach == "composite_group_beta":
            belief_seq, best_loss_tran, best_loss_init, training_time_tran, training_time_init = run_group_beta(
                batch_size=batch_size,
                n_basis=n_basis,
                tran_params=params["tran_params"],
                init_params=params["init_params"],
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unknown approach: {approach}")

        ll_per_step = []
        for i in range(problem.n_timesteps):
            ll = avg_log_likelihood(belief_seq[i], test_traj_data[i].to(device))
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
        {"name": "lcmp", "params": LCMP_CONTEXT},
        {"name": "composite_group_beta", "params": GROUP_BETA_CONTEXT},
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
