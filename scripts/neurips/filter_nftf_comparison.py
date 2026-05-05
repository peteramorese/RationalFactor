from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.tools.propagate as propagate
from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.composite_model import CompositeDensityModel, CompositeRFandR2FF
from rational_factor.models.density_model import LogisticSigmoid
from rational_factor.models.domain_transformation import MaskedAffineNFTF
from rational_factor.models.factor_forms import LinearFF, LinearRFandR2FF
from rational_factor.models.filter import Filter
from particle_filter.particle_set import WeightedParticleSet
from particle_filter.propagate import propagate_and_update as pf_propagate_and_update
from rational_factor.systems.base import SystemObservationDistribution, SystemTransitionDistribution
from rational_factor.systems.problems import PARTIALLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import (
    avg_log_filter_score,
    avg_log_likelihood_under_particle_belief_reference,
)
from rational_factor.tools.benchmark import Benchmark
from rational_factor.tools.misc import data_bounds, train_test_split

N_BASIS = 500
N_OBS_BASIS = 300
TRIALS = 10
BENCHMARK_ROOT = "benchmark_data"
N_PF_PARTICLES = 1000

CONTEXT_WITH_NFTF = {
    "use_nftf": True,
    "use_nftf_prefit": True,
    "n_basis": N_BASIS,
    "n_obs_basis": N_OBS_BASIS,
    "obs_and_tran_params": {
        "n_epochs_per_group": [5, 5],
        "iterations": 50,
        "pre_train_epochs": 5,
        "lr_basis_tran": 5e-3,
        "lr_basis_obs": 5e-3,
        "lr_weights": 1e-3,
        "lr_dtf": 5e-4,
        "dtf_weight_decay": 0.0,
    },
    "init_params": {
        "n_epochs_per_group": [20, 5],
        "iterations": 100,
        "lr_basis": 1e-3,
        "lr_weights": 1e-3,
    },
    "batch_size": 256,
    "ls_temp": 0.1,
    "reg_covar_joint": 1e-1,
    "reg_covar_obs": 1e-1,
    "validation_early_stopping_patience": 30,
    "obs_loss_weight": 1.0,
    "verbose": True,
}

CONTEXT_WITHOUT_NFTF = {
    "use_nftf": False,
    "use_nftf_prefit": False,
    "n_basis": N_BASIS,
    "n_obs_basis": N_OBS_BASIS,
    "obs_and_tran_params": {
        "n_epochs_per_group": [20, 5],
        "iterations": 50,
        "lr_basis_tran": 5e-3,
        "lr_basis_obs": 5e-3,
        "lr_weights": 1e-3,
    },
    "init_params": {
        "n_epochs_per_group": [20, 5],
        "iterations": 100,
        "lr_basis": 1e-3,
        "lr_weights": 1e-3,
    },
    "batch_size": 256,
    "ls_temp": 0.1,
    "reg_covar_joint": 1e-1,
    "reg_covar_obs": 1e-1,
    "validation_early_stopping_patience": 30,
    "obs_loss_weight": 1.0,
    "verbose": True,
}

CONTEXT_WITH_NFTF_NO_PREFIT = {
    "use_nftf": True,
    "use_nftf_prefit": False,
    "n_basis": N_BASIS,
    "n_obs_basis": N_OBS_BASIS,
    "obs_and_tran_params": {
        "n_epochs_per_group": [5, 5],
        "iterations": 50,
        "lr_basis_tran": 5e-3,
        "lr_basis_obs": 5e-3,
        "lr_weights": 1e-3,
        "lr_dtf": 5e-4,
        "dtf_weight_decay": 0.0,
    },
    "init_params": {
        "n_epochs_per_group": [20, 5],
        "iterations": 100,
        "lr_basis": 1e-3,
        "lr_weights": 1e-3,
    },
    "batch_size": 256,
    "ls_temp": 0.1,
    "reg_covar_joint": 1e-1,
    "reg_covar_obs": 1e-1,
    "validation_early_stopping_patience": 30,
    "obs_loss_weight": 1.0,
    "verbose": True,
}



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark filtering with vs without NFTF on a partially observable problem."
    )
    parser.add_argument(
        "problem",
        type=str,
        choices=sorted(PARTIALLY_OBSERVABLE_PROBLEMS.keys()),
        help="Key in PARTIALLY_OBSERVABLE_PROBLEMS.",
    )
    parser.add_argument(
        "--use-pf-reference",
        action="store_true",
        help="Enable particle-filter reference evaluation (disabled by default).",
    )
    args = parser.parse_args()
    problem_key = args.problem
    eval_pf_reference = args.use_pf_reference

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    problem = PARTIALLY_OBSERVABLE_PROBLEMS[problem_key]
    system = problem.system

    x0_data = problem.train_initial_state_data()
    x_k_data, x_kp1_data = problem.train_state_transition_data()
    xo_data, o_data = problem.train_obs_data()
    test_traj_data, test_obs_data_full = problem.test_data()
    test_obs_data = test_obs_data_full[1:]

    x0_train, x0_val = train_test_split(x0_data, test_size=0.2)
    x_k_train, x_k_val, x_kp1_train, x_kp1_val = train_test_split(x_k_data, x_kp1_data, test_size=0.2)
    xo_train, xo_val, o_train, o_val = train_test_split(xo_data, o_data, test_size=0.2)

    x0_dataset = TensorDataset(x0_train)
    x0_val_dataset = TensorDataset(x0_val)
    xp_dataset = TensorDataset(x_kp1_train, x_k_train)
    xp_val_dataset = TensorDataset(x_kp1_val, x_k_val)
    o_dataset = TensorDataset(o_train, xo_train)
    o_val_dataset = TensorDataset(o_val, xo_val)
    x_k_dataset = TensorDataset(x_k_data)

    def experiment(
        use_nftf: bool,
        use_nftf_prefit: bool,
        n_basis: int,
        n_obs_basis: int,
        obs_and_tran_params: dict,
        init_params: dict,
        batch_size: int,
        ls_temp: float,
        reg_covar_joint: float,
        reg_covar_obs: float,
        validation_early_stopping_patience: int,
        obs_loss_weight: float,
        verbose: bool = True,
    ):
        x0_dataloader = DataLoader(x0_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
        x0_val_dataloader = DataLoader(x0_val_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
        xp_dataloader = DataLoader(xp_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
        xp_val_dataloader = DataLoader(xp_val_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
        o_dataloader = DataLoader(o_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
        o_val_dataloader = DataLoader(o_val_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
        x_k_dataloader = DataLoader(x_k_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

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
                nftf.parameters(),
                lr=obs_and_tran_params["lr_dtf"],
                weight_decay=obs_and_tran_params["lr_dtf"],
            )
            _, best_loss_pretrain, training_time_pretrain = train.train(
                decorrupter_density,
                x_k_dataloader,
                {"mle": loss.mle_loss},
                pretrain_optimizer,
                epochs=obs_and_tran_params["pre_train_epochs"],
                verbose=verbose,
                use_best="mle",
            )
        else:
            best_loss_pretrain = float("nan")
            training_time_pretrain = 0.0

        if use_nftf:
            with torch.no_grad():
                y_k_data, _ = nftf(x_k_data.to(device))
                y_kp1_data, _ = nftf(x_kp1_data.to(device))
                yo_data_latent, _ = nftf(xo_data.to(device))
                y_k_data = y_k_data.to(torch.device("cpu"))
                y_kp1_data = y_kp1_data.to(torch.device("cpu"))
                yo_data_latent = yo_data_latent.to(torch.device("cpu"))
        else:
            y_k_data = x_k_data.to(torch.device("cpu"))
            y_kp1_data = x_kp1_data.to(torch.device("cpu"))
            yo_data_latent = xo_data.to(torch.device("cpu"))

        y_joint_data = torch.cat([y_k_data, y_kp1_data], dim=1)
        tran_gmm_lf = train.fit_gaussian_lf_em(
            y_joint_data,
            n_components=n_basis,
            reg_covar=reg_covar_joint,
            max_iter=100,
        )
        phi_marginal = tran_gmm_lf.basis.marginal(marginal_dims=range(system.dim()))
        phi_means, phi_stds = phi_marginal.means_stds()
        phi_params = torch.stack([phi_means, phi_stds], dim=-1)
        psi_marginal = tran_gmm_lf.basis.marginal(marginal_dims=range(system.dim(), 2 * system.dim()))
        psi_means, psi_stds = psi_marginal.means_stds()
        psi_params = torch.stack([psi_means, psi_stds], dim=-1)

        yo_joint_data = torch.cat([yo_data_latent, o_data], dim=1)
        obs_gmm_lf = train.fit_gaussian_lf_em(
            yo_joint_data,
            n_components=n_obs_basis,
            reg_covar=reg_covar_obs,
            max_iter=100,
        )
        xi_marginal = obs_gmm_lf.basis.marginal(marginal_dims=range(system.dim()))
        xi_means, xi_stds = xi_marginal.means_stds()
        xi_params = torch.stack([xi_means, xi_stds], dim=-1)
        zeta_marginal = obs_gmm_lf.basis.marginal(
            marginal_dims=range(system.dim(), system.dim() + system.observation_dim())
        )
        zeta_means, zeta_stds = zeta_marginal.means_stds()
        zeta_params = torch.stack([zeta_means, zeta_stds], dim=-1)

        phi_basis = GaussianBasis(uparams_init=phi_params).to(device)
        psi_basis = GaussianBasis(uparams_init=psi_params).to(device)
        xi_basis = GaussianBasis(uparams_init=xi_params).to(device)
        zeta_basis = GaussianBasis(uparams_init=zeta_params).to(device)

        if use_nftf:
            tran_obs_model = CompositeRFandR2FF(
                [nftf],
                LinearRFandR2FF(xi_basis, zeta_basis, phi_basis, psi_basis),
            ).to(device)
            optimizers = {
                "dtf_and_basis": torch.optim.Adam(
                    [
                        {
                            "params": tran_obs_model.conditional_density_model.basis_params(),
                            "lr": obs_and_tran_params["lr_basis_tran"],
                        },
                        {
                            "params": tran_obs_model.domain_tfs.parameters(),
                            "lr": obs_and_tran_params["lr_dtf"],
                            "weight_decay": obs_and_tran_params["dtf_weight_decay"],
                        },
                    ]
                ),
                "weights": torch.optim.Adam(
                    tran_obs_model.conditional_density_model.weight_params(),
                    lr=obs_and_tran_params["lr_weights"],
                ),
            }
            tran_conditional_mle_loss_fn = lambda model, xp, x: loss.conditional_mle_loss(  # noqa: E731
                model, xp, x, method_name="log_density"
            )
            obs_conditional_mle_loss_fn = lambda model, o, x: obs_loss_weight * loss.conditional_mle_loss(  # noqa: E731
                model, o, x, method_name="log_observation_density"
            )
        else:
            tran_obs_model = LinearRFandR2FF(xi_basis, zeta_basis, phi_basis, psi_basis).to(device)
            optimizers = {
                "basis": torch.optim.Adam(
                    tran_obs_model.basis_params(),
                    lr=obs_and_tran_params["lr_basis_tran"],
                ),
                "weights": torch.optim.Adam(
                    tran_obs_model.weight_params(),
                    lr=obs_and_tran_params["lr_weights"],
                ),
            }
            tran_conditional_mle_loss_fn = lambda model, xp, x: loss.conditional_mle_loss(  # noqa: E731
                model, xp, x, method_name="log_density"
            )
            obs_conditional_mle_loss_fn = lambda model, o, x: obs_loss_weight * loss.conditional_mle_loss(  # noqa: E731
                model, o, x, method_name="log_observation_density"
            )

        tran_obs_model, best_loss_tran_obs, training_time_tran_obs = train.train_iterate_multiset(
            tran_obs_model,
            data_loaders=[xp_dataloader, o_dataloader],
            labeled_loss_fns_list=[
                {"mle": tran_conditional_mle_loss_fn},
                {"obs_mle": obs_conditional_mle_loss_fn},
            ],
            labeled_optimizers=optimizers,
            labeled_validation_loss_fns_list=[
                {"val_mle": tran_conditional_mle_loss_fn},
                {"val_obs_mle": obs_conditional_mle_loss_fn},
            ],
            validation_data_loaders=[xp_val_dataloader, o_val_dataloader],
            validation_early_stopping_patience=validation_early_stopping_patience,
            epochs_per_group=obs_and_tran_params["n_epochs_per_group"],
            iterations=obs_and_tran_params["iterations"],
            verbose=verbose,
            use_best="val_mle",
        )

        psi0_basis = GaussianBasis.random_init(
            system.dim(),
            n_basis=n_basis,
            offsets=torch.tensor([0.0, 20.0], device=device),
            variance=30.0,
            min_std=1e-4,
        ).to(device)
        base_tran_obs_model = tran_obs_model.conditional_density_model if use_nftf else tran_obs_model
        init_model = LinearFF.from_r2ff(base_tran_obs_model, psi0_basis).to(device)
        init_optimizers = {
            "basis": torch.optim.Adam(init_model.basis_params(), lr=init_params["lr_basis"]),
            "weights": torch.optim.Adam(init_model.weight_params(), lr=init_params["lr_weights"]),
        }
        init_model, best_loss_init, training_time_init = train.train_iterate(
            init_model,
            x0_dataloader,
            {"mle": loss.mle_loss},
            init_optimizers,
            labeled_validation_loss_fns={"val_mle": loss.mle_loss},
            validation_data_loader=x0_val_dataloader,
            validation_early_stopping_patience=validation_early_stopping_patience,
            epochs_per_group=init_params["n_epochs_per_group"],
            iterations=init_params["iterations"],
            verbose=verbose,
            use_best="val_mle",
        )

        if use_nftf:
            tran_model = tran_obs_model.conditional_density_model.r2ff().to(device).eval()
            obs_model = tran_obs_model.conditional_density_model.rf().to(device).eval()
        else:
            tran_model = tran_obs_model.r2ff().to(device).eval()
            obs_model = tran_obs_model.rf().to(device).eval()

        init_model = init_model.to(device).eval()
        if use_nftf:
            nftf_eval = MaskedAffineNFTF.copy_from_trainable(nftf).to(device).eval()

            def prop_and_upd_nf(initial_belief, transition_model, observation_model, observations):
                latent_priors, latent_posteriors = propagate.propagate_and_update(
                    initial_belief,
                    transition_model,
                    observation_model,
                    observations,
                )
                priors = [
                    CompositeDensityModel([nftf_eval], belief).to(device).eval()
                    for belief in latent_priors
                ]
                posteriors = [
                    CompositeDensityModel([nftf_eval], belief).to(device).eval()
                    for belief in latent_posteriors
                ]
                return priors, posteriors

            filter_prop_and_upd_fn = prop_and_upd_nf
        else:
            filter_prop_and_upd_fn = propagate.propagate_and_update

        filter_model = Filter(
            transition_model=tran_model,
            observation_model=obs_model,
            prop_and_upd_fn=filter_prop_and_upd_fn,
        )

        test_traj_data_eval = [x_k.to(device) for x_k in test_traj_data]
        test_obs_data_eval = [obs_k.to(device) if obs_k is not None else None for obs_k in test_obs_data]

        print("Evaluating filter with avg_log_filter_score...")
        prior_scores, posterior_scores = avg_log_filter_score(
            test_traj_data=test_traj_data_eval,
            test_obs_data=test_obs_data_eval,
            filter=filter_model,
            initial_belief=init_model,
        )

        if eval_pf_reference:
            t_dist = SystemTransitionDistribution(system).to(device).eval()
            o_dist = SystemObservationDistribution(system).to(device).eval()
            pf_filter = Filter(
                transition_model=t_dist,
                observation_model=o_dist,
                prop_and_upd_fn=pf_propagate_and_update,
            )

            def belief_to_state_space(belief):
                return belief

            pf_ref_dtype = test_obs_data_eval[0].dtype
            ref_particles0 = test_traj_data[0].to(device=device, dtype=pf_ref_dtype)
            ref_weights0 = torch.ones(ref_particles0.shape[0], device=device, dtype=pf_ref_dtype) / ref_particles0.shape[0]
            pf_ref_prior_scores, pf_ref_post_scores = avg_log_likelihood_under_particle_belief_reference(
                test_traj_data=test_traj_data_eval,
                test_obs_data=test_obs_data_eval,
                filter=filter_model,
                initial_belief=init_model,
                reference_filter=pf_filter,
                reference_initial_belief_fn=lambda _i: WeightedParticleSet(
                    particles=ref_particles0.clone(),
                    weights=ref_weights0.clone(),
                ),
                belief_to_reference_space=belief_to_state_space,
            )
        else:
            n_pf_steps = len(test_obs_data_eval)
            pf_ref_dtype = test_obs_data_eval[0].dtype
            pf_ref_prior_scores = torch.full(
                (n_pf_steps,),
                float("nan"),
                device=device,
                dtype=pf_ref_dtype,
            )
            pf_ref_post_scores = torch.full(
                (n_pf_steps + 1,),
                float("nan"),
                device=device,
                dtype=pf_ref_dtype,
            )
        print("Done. \n")

        return (
            prior_scores.detach().cpu().to(dtype=torch.float32),
            posterior_scores.detach().cpu().to(dtype=torch.float32),
            torch.tensor([float(best_loss_pretrain)], dtype=torch.float32),
            torch.tensor([float(training_time_pretrain)], dtype=torch.float32),
            torch.tensor([float(best_loss_tran_obs)], dtype=torch.float32),
            torch.tensor([float(best_loss_init)], dtype=torch.float32),
            torch.tensor([float(training_time_tran_obs)], dtype=torch.float32),
            torch.tensor([float(training_time_init)], dtype=torch.float32),
            pf_ref_prior_scores.detach().cpu().to(dtype=torch.float32),
            pf_ref_post_scores.detach().cpu().to(dtype=torch.float32),
        )

    contexts = [
        {"name": "DTF w/ Init", "params": CONTEXT_WITH_NFTF},
        {"name": "DTF w/o Init", "params": CONTEXT_WITH_NFTF_NO_PREFIT},
        {"name": "No DTF", "params": CONTEXT_WITHOUT_NFTF},
    ]

    benchmark = Benchmark(name=Path(__file__).stem + "_" + problem_key)
    benchmark.set_experiment_fn(experiment)
    benchmark.set_contexts(contexts)

    benchmark.set_numerical_result(0, "avg_log_filter_prior_score_per_timestep", json_raw_data=False)
    benchmark.set_numerical_result(1, "avg_log_filter_posterior_score_per_timestep", json_raw_data=False)
    benchmark.set_numerical_result(2, "best_loss_pretrain_decorrupter", json_raw_data=False)
    benchmark.set_numerical_result(3, "training_time_pretrain_decorrupter", json_raw_data=False)
    benchmark.set_numerical_result(4, "best_loss_transition_observation", json_raw_data=False)
    benchmark.set_numerical_result(5, "best_loss_initial", json_raw_data=False)
    benchmark.set_numerical_result(6, "training_time_transition_observation", json_raw_data=False)
    benchmark.set_numerical_result(7, "training_time_initial", json_raw_data=False)
    benchmark.set_numerical_result(
        8,
        "avg_log_likelihood_prior_under_pf_reference_per_timestep",
        json_raw_data=False,
    )
    benchmark.set_numerical_result(
        9,
        "avg_log_likelihood_posterior_under_pf_reference_per_timestep",
        json_raw_data=False,
    )

    print(
        f"Running benchmark problem={problem_key} ({len(contexts)} contexts, {TRIALS} trial(s) each)..."
    )
    benchmark.run(trials=TRIALS, verbose=True)
    run_dir = benchmark.process_and_save_results(root_dir=BENCHMARK_ROOT)
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
