from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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
from rational_factor.tools.analysis import avg_log_likelihood, check_pdf_valid
from rational_factor.tools.visualization import plot_marginal_trajectory_comparison


def main() -> None:
    problem = FULLY_OBSERVABLE_PROBLEMS["planar_quadrotor"]

    ######## USER CONFIG ########
    use_gpu = torch.cuda.is_available()
    use_dtf = True
    n_basis = 500
    batch_size = 256

    if use_dtf:
        tran_params = {
            "n_epochs_per_group": [15, 15],  # dtf+basis, then weights
            "iterations": 100,
            "lr_basis": 1e-2,
            "lr_weights": 1e-3,
            "lr_dtf": 1e-3,
            "lr_wrap": 1e-3,
        }
        init_params = {
            "n_epochs_per_group": [20, 5],  # basis, then weights
            "iterations": 1000,
            "lr_basis": 5e-2,
            "lr_weights": 1e-2,
        }
    else:
        tran_params = {
            "n_epochs_per_group": [15, 5],  # basis, then weights
            "iterations": 50,
            "lr_basis": 1e-2,
            "lr_weights": 1e-2,
            "lr_wrap": 1e-3,
        }
        init_params = {
            "n_epochs_per_group": [20, 5],  # basis, then weights
            "iterations": 100,
            "lr_basis": 5e-3,
            "lr_weights": 1e-2,
        }

    ######## SETUP ########
    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using device: ", device)

    system = problem.system
    x0_data = problem.train_initial_state_data()
    x_k_data, x_kp1_data = problem.train_state_transition_data()
    test_traj_data = problem.test_data()

    x0_dataloader = DataLoader(TensorDataset(x0_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(TensorDataset(x_kp1_data, x_k_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    offsets = torch.tensor([10.0, 10.0], device=device)
    variance = 30.0
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
    nftf = (
        MaskedAffineNFTF(system.dim(), trainable=True, hidden_features=256, n_layers=6).to(device)
        if use_dtf
        else None
    )

    ######## TRAIN TRANSITION ########
    if use_dtf:
        tran_model = CompositeConditionalModel([nftf, wrap_tf], LinearRFF(phi_basis, psi_basis, numerical_tolerance=problem.numerical_tolerance)).to(device)
    else:
        tran_model = CompositeConditionalModel([wrap_tf], LinearRFF(phi_basis, psi_basis, numerical_tolerance=problem.numerical_tolerance)).to(device)

    print("Training transition model")

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
        {"mle": loss.conditional_mle_loss},
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
        verbose=True,
        use_best="mle",
    )
    print("Done.\n")

    #print("wrap loc scale: ", trained_domain_tf.loc_scale())
    #print("psi0_basis alpha beta: ", init_model.density_model.psi0_basis.alphas_betas())
    #print("psi_basis alpha beta: ", tran_model.conditional_density_model.psi_basis.alphas_betas())
    #print("phi_basis alpha beta: ", tran_model.conditional_density_model.phi_basis.alphas_betas())
    ## Debug a few random physical states through the initial composite log density.
    #n_debug_samples = 5
    #lows = problem.plot_bounds_low.to(device=device, dtype=x0_data.dtype)
    #highs = problem.plot_bounds_high.to(device=device, dtype=x0_data.dtype)
    #x_debug = torch.rand(n_debug_samples, system.dim(), device=device, dtype=x0_data.dtype) * (highs - lows) + lows
    #print("x_debug samples:\n", x_debug)
    #with torch.no_grad():
    #    init_model.eval()
    #    logp_debug = init_model.log_density(x_debug, debug=True)
    #print("init_model.log_density(x_debug):\n", logp_debug)

    print(f"Transition model loss: {best_loss_tran:.6f}, training time: {training_time_tran:.2f}s")
    print(f"Initial model loss: {best_loss_init:.6f}, training time: {training_time_init:.2f}s")

    ######## ANALYSIS ########
    base_belief_seq = propagate.propagate(init_model.density_model, tran_model.conditional_density_model, n_steps=problem.n_timesteps)
    if use_dtf:
        belief_seq = [CompositeDensityModel([trained_nftf, trained_domain_tf], belief) for belief in base_belief_seq]
    else:
        belief_seq = [CompositeDensityModel([trained_domain_tf], belief) for belief in base_belief_seq]

    ll_per_step = []
    for i in range(problem.n_timesteps):
        data_i = test_traj_data[i].to(device)
        ll = avg_log_likelihood(belief_seq[i], data_i)
        check_pdf_valid(belief_seq[i], domain_bounds=(tuple(problem.plot_bounds_low.tolist()), tuple(problem.plot_bounds_high.tolist())), n_samples=10000, device=device)
        ll_per_step.append(ll.detach().cpu().reshape(()))
        print(f"Avg log-likelihood at time {i}: {float(ll_per_step[-1]):.6f}")
    prop_ll_vector = torch.stack(ll_per_step).to(dtype=torch.float32)

    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    ll_out_path = out_dir / f"{Path(__file__).stem}__ll.png"
    plt.figure(figsize=(8, 4))
    plt.plot(prop_ll_vector.numpy(), marker="o")
    plt.xlabel("timestep")
    plt.ylabel("avg log-likelihood")
    plt.title(f"{Path(__file__).stem} (use_dtf={use_dtf}, n_basis={n_basis})")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(ll_out_path, dpi=200)
    print(f"Saved LL plot to {ll_out_path}")

    if not use_dtf:
        comp_out_path = out_dir / f"{Path(__file__).stem}__marginal_comparison.png"
        # Visualization does not need GPU; move beliefs to CPU to reduce peak memory.
        belief_seq = [belief.to("cpu") for belief in belief_seq]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fig, _ = plot_marginal_trajectory_comparison(
            problem=problem,
            beliefs=belief_seq,
            scatter_kwargs={"s": 1, "alpha": 0.7},
            n_points=40,
        )
        if fig is not None:
            fig.suptitle(f"{Path(__file__).stem} marginal comparison", y=1.02)
            fig.savefig(comp_out_path, dpi=200, bbox_inches="tight")
            print(f"Saved marginal comparison plot to {comp_out_path}")


if __name__ == "__main__":
    main()
