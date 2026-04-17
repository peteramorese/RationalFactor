from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.systems.truth_models as truth_models
import rational_factor.tools.propagate as propagate
from rational_factor.models.basis_functions import BetaBasis
from rational_factor.models.composite_model import CompositeConditionalModel, CompositeDensityModel
from rational_factor.models.domain_transformation import ErfSeparableTF, MaskedAffineNFTF
from rational_factor.models.factor_forms import LinearFF, LinearRFF
from rational_factor.systems.base import sample_io_pairs, sample_trajectories
from rational_factor.tools.analysis import avg_log_likelihood
from rational_factor.tools.misc import make_mvnormal_state_sampler


def make_quadcopter_benchmark_system() -> truth_models.Quadcopter:
    """
    12D closed-loop quadcopter benchmark system.

    rate_filter_alpha=1.0 keeps the 12D state Markov, matching sample_trajectories / sample_io_pairs.
    """
    return truth_models.Quadcopter(
        dt=0.1,
        waypoint=torch.tensor([2.0, 1.0, 1.35]),
        yaw_ref=0.35,
        m=1.0,
        g=9.81,
        c_v=0.06,
        c_w=0.06,
        thrust_min=0.0,
        thrust_max=24.0,
        torque_limits=torch.tensor([1.2, 1.2, 0.55]),
        covariance=0.012 * torch.eye(12),
        rate_filter_alpha=1.0,
    )


def quadcopter_init_state_sampler():
    mean = torch.tensor(
        [
            0.2,
            -0.15,
            1.0,
            0.45,
            0.3,
            0.18,
            0.08,
            -0.11,
            0.06,
            0.05,
            -0.04,
            0.03,
        ],
        dtype=torch.float32,
    )
    variances = torch.tensor(
        [
            0.12**2,
            0.12**2,
            0.1**2,
            0.22**2,
            0.22**2,
            0.18**2,
            0.1**2,
            0.1**2,
            0.08**2,
            0.08**2,
            0.08**2,
            0.07**2,
        ],
        dtype=torch.float32,
    )
    return make_mvnormal_state_sampler(mean=mean, covariance=torch.diag(variances))


def quadcopter_prev_state_sampler(system: truth_models.Quadcopter):
    mean = torch.zeros(system.dim(), dtype=torch.float32)
    mean[0:3] = torch.tensor([0.8, 0.35, 1.05])
    mean[3:6] = torch.tensor([0.35, 0.2, 0.12])
    mean[6:9] = torch.tensor([0.05, -0.05, 0.12])
    mean[9:12] = torch.tensor([0.03, -0.02, 0.02])

    scales_sq = torch.tensor(
        [
            1.2**2,
            1.2**2,
            0.9**2,
            1.1**2,
            1.1**2,
            0.85**2,
            0.35**2,
            0.35**2,
            0.45**2,
            0.45**2,
            0.45**2,
            0.35**2,
        ],
        dtype=torch.float32,
    )
    cov = torch.diag(scales_sq)

    def _sample(n_samples: int):
        dist = torch.distributions.MultivariateNormal(mean, cov)
        return dist.sample((n_samples,))

    return _sample


def main() -> None:
    ######## USER CONFIG ########
    use_gpu = torch.cuda.is_available()
    use_dtf = False  

    n_basis = 1000
    batch_size = 256
    n_timesteps_prop = 15
    n_trajectories_test = 5000
    n_data_tran = 20000
    n_data_init = 2000
    var_reg_strength = 0.0

    if use_dtf:
        tran_params = {
            "n_epochs_per_group": [15, 5],  # dtf+basis, then weights
            "iterations": 300,
            "lr_basis": 1e-2,
            "lr_weights": 1e-2,
            "lr_dtf": 1e-3,
            "lr_wrap": 1e-3,
        }
        init_params = {
            "n_epochs_per_group": [20, 5],  # basis, then weights
            "iterations": 200,
            "lr_basis": 5e-2,
            "lr_weights": 1e-2,
        }
    else:
        tran_params = {
            "n_epochs_per_group": [15, 5],  # basis, then weights
            "iterations": 300,
            "lr_basis": 1e-2,
            "lr_weights": 1e-2,
            "lr_wrap": 1e-3,
        }
        init_params = {
            "n_epochs_per_group": [20, 5],  # basis, then weights
            "iterations": 200,
            "lr_basis": 5e-2,
            "lr_weights": 1e-2,
        }

    ######## SETUP ########
    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using device: ", device)

    system = make_quadcopter_benchmark_system()
    init_state_sampler = quadcopter_init_state_sampler()
    prev_state_sampler = quadcopter_prev_state_sampler(system)

    test_traj_data = sample_trajectories(
        system,
        init_state_sampler,
        n_timesteps=n_timesteps_prop,
        n_trajectories=n_trajectories_test,
    )

    x0_data = init_state_sampler(n_data_init)
    x_k_data, x_kp1_data = sample_io_pairs(system, prev_state_sampler, n_pairs=n_data_tran)

    x0_dataloader = DataLoader(
        TensorDataset(x0_data),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_gpu,
    )
    xp_dataloader = DataLoader(
        TensorDataset(x_kp1_data, x_k_data),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_gpu,
    )

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

    wrap_tf = ErfSeparableTF.from_data(x_k_data, trainable=True).to(device)
    nftf = (
        MaskedAffineNFTF(system.dim(), trainable=True, hidden_features=256, n_layers=6).to(device)
        if use_dtf
        else None
    )

    ######## TRAIN TRANSITION ########
    if use_dtf:
        tran_model = CompositeConditionalModel([nftf, wrap_tf], LinearRFF(phi_basis, psi_basis)).to(device)
    else:
        tran_model = CompositeConditionalModel([wrap_tf], LinearRFF(phi_basis, psi_basis)).to(device)

    print("Training transition model")
    mle_loss_fn = loss.conditional_mle_loss
    var_reg_loss_fn = lambda model, x, xp: var_reg_strength * (
        loss.beta_basis_concentration_reg_loss(model.conditional_density_model.phi_basis)
        + loss.beta_basis_concentration_reg_loss(model.conditional_density_model.psi_basis)
    )

    if use_dtf:
        optimizers = {
            "dtf_and_basis": torch.optim.Adam(
                [
                    {"params": tran_model.conditional_density_model.basis_params(), "lr": tran_params["lr_basis"]},
                    {"params": tran_model.domain_tfs[0].parameters(), "lr": tran_params["lr_dtf"]},
                    {"params": tran_model.domain_tfs[1].parameters(), "lr": tran_params["lr_wrap"]},
                ]
            ),
            "weights": torch.optim.Adam(tran_model.conditional_density_model.weight_params(), lr=tran_params["lr_weights"]),
        }
    else:
        optimizers = {
            "basis": torch.optim.Adam(
                [
                    {"params": tran_model.conditional_density_model.basis_params(), "lr": tran_params["lr_basis"]},
                    {"params": tran_model.domain_tfs.parameters(), "lr": tran_params["lr_wrap"]},
                ]
            ),
            "weights": torch.optim.Adam(tran_model.conditional_density_model.weight_params(), lr=tran_params["lr_weights"]),
        }

    tran_model, best_loss_tran, training_time_tran = train.train_iterate(
        tran_model,
        xp_dataloader,
        {"mle": mle_loss_fn, "var_reg": var_reg_loss_fn},
        optimizers,
        epochs_per_group=tran_params["n_epochs_per_group"],
        iterations=tran_params["iterations"],
        verbose=True,
        use_best="mle",
    )
    print("Done.\n")
    print("Valid: ", tran_model.valid())

    trained_nftf = MaskedAffineNFTF.copy_from_trainable(nftf).to(device) if use_dtf else None
    trained_domain_tf = ErfSeparableTF.copy_from_trainable(wrap_tf).to(device)

    ######## TRAIN INITIAL ########
    if use_dtf:
        init_model = CompositeDensityModel(
            [trained_nftf, trained_domain_tf],
            LinearFF.from_rff(tran_model.conditional_density_model, psi0_basis),
        ).to(device)
    else:
        init_model = CompositeDensityModel(
            [trained_domain_tf],
            LinearFF.from_rff(tran_model.conditional_density_model, psi0_basis),
        ).to(device)

    print("Training initial model")
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
        verbose=True,
        use_best="mle",
    )
    print("Done.\n")

    print(f"Transition model loss: {best_loss_tran:.6f}, training time: {training_time_tran:.2f}s")
    print(f"Initial model loss: {best_loss_init:.6f}, training time: {training_time_init:.2f}s")

    ######## ANALYSIS ########
    base_belief_seq = propagate.propagate(init_model.density_model, tran_model.conditional_density_model, n_steps=n_timesteps_prop)
    if use_dtf:
        belief_seq = [CompositeDensityModel([trained_nftf, trained_domain_tf], belief) for belief in base_belief_seq]
    else:
        belief_seq = [CompositeDensityModel([trained_domain_tf], belief) for belief in base_belief_seq]

    ll_per_step = []
    for i in range(n_timesteps_prop):
        data_i = test_traj_data[i].to(device)
        ll = avg_log_likelihood(belief_seq[i], data_i)
        ll_per_step.append(ll.detach().cpu().reshape(()))
        print(f"Avg log-likelihood at time {i}: {float(ll_per_step[-1]):.6f}")
    prop_ll_vector = torch.stack(ll_per_step).to(dtype=torch.float32)

    # Save a quick summary figure (LL vs timestep) since 12D beliefs are not directly plottable.
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(__file__).stem}__ll.png"

    plt.figure(figsize=(8, 4))
    plt.plot(prop_ll_vector.numpy(), marker="o")
    plt.xlabel("timestep")
    plt.ylabel("avg log-likelihood")
    plt.title(f"{Path(__file__).stem} (use_dtf={use_dtf}, n_basis={n_basis})")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved LL plot to {out_path}")


if __name__ == "__main__":
    main()
