from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from rational_factor.models.factor_forms import SumProdRFF, LinearFF
from rational_factor.models.tt.tt_basis import TTBasis
from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.parameters import (
    PositiveParameters,
    TrainableParameters,
    param_group_iter,
)
from rational_factor.models.tt.nested_tt_parameters import (
    NestedTTVectorParameters,
    RowStochasticNestedTTMatrixParameters,
)
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood
import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.tools.misc as misc
import rational_factor.tools.propagate as propagate


if __name__ == "__main__":
    problem = FULLY_OBSERVABLE_PROBLEMS["cartpole"]

    ###
    use_gpu = torch.cuda.is_available()
    dtype = torch.float64
    # Primitive 1-D count; TT modes are (n_primitive,) * dim, so n_basis = n_primitive^dim.
    n_primitive = 100
    rank = 10
    # Nested TT hierarchy depth. depth=1 recovers ordinary TT cores as leaves.
    depth = 2
    # Floor Gaussian bandwidths so MLE cannot form Dirac peaks (loss -> -inf).
    gaussian_std_epsilon = 0.3
    test_fraction = 0.15
    split_seed = 0
    tran_params = {
        "n_epochs_per_group": [5, 5],  # basis, weights
        "iterations": 10,
        "lr_basis": 4 * 3e-3,
        "lr_weights": 1 * 5e-2,
    }
    init_params = {
        "n_epochs_per_group": [10],  # h0 coeffs only
        "iterations": 50,
        "lr_weights": 1e-2,
    }

    batch_size = 512
    n_timesteps_prop = problem.n_timesteps

    ###

    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using GPU: ", use_gpu)
    print("Device: ", device)
    print("Dtype: ", dtype)

    system = problem.system
    dim = system.dim()

    d = dim
    tt_modes = (n_primitive,) * d
    n_basis = n_primitive ** d
    m = n_basis

    x0 = problem.train_initial_state_data().to(dtype=dtype)
    x_k, x_kp1 = problem.train_state_transition_data()
    x_k_train, x_k_test, x_kp1_train, x_kp1_test = misc.train_test_split(
        x_k, x_kp1, test_size=test_fraction, shuffle=True, seed=split_seed
    )
    x_k_train = x_k_train.to(dtype=dtype)
    x_k_test = x_k_test.to(dtype=dtype)
    x_kp1_train = x_kp1_train.to(dtype=dtype)
    x_kp1_test = x_kp1_test.to(dtype=dtype)
    traj_data = [t.to(dtype=dtype) for t in problem.test_data()]

    x0_dataloader = DataLoader(
        TensorDataset(x0), batch_size=batch_size, shuffle=True, pin_memory=use_gpu
    )
    xp_dataloader = DataLoader(
        TensorDataset(x_kp1_train, x_k_train),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_gpu,
    )

    print(
        f"n_basis={m}, n_primitive={n_primitive}, dim={d}, rank={rank}, "
        f"depth={depth}, modes={tt_modes}"
    )

    parameter_shape = (1, d, n_primitive)
    # Means/stds are in the erf-wrapped coordinates (roughly O(1)).
    phi_means = TrainableParameters.random_init(
        shape=parameter_shape, mean=torch.tensor([0.0]), std=torch.tensor([3.5])
    ).to(device, dtype=dtype)
    phi_std = PositiveParameters.random_init(
        shape=parameter_shape,
        mean=torch.tensor([0.8]),
        std=torch.tensor([0.5]),
        epsilon=gaussian_std_epsilon,
    ).to(device, dtype=dtype)
    psi_means = TrainableParameters.random_init(
        shape=parameter_shape, mean=torch.tensor([0.0]), std=torch.tensor([3.5])
    ).to(device, dtype=dtype)
    psi_std = PositiveParameters.random_init(
        shape=parameter_shape,
        mean=torch.tensor([0.8]),
        std=torch.tensor([0.5]),
        epsilon=gaussian_std_epsilon,
    ).to(device, dtype=dtype)
    phi_primitive = GaussianBasis(mean_params=phi_means, std_params=phi_std)
    psi_primitive = GaussianBasis(mean_params=psi_means, std_params=psi_std)

    # Coefficient-free TT bases: tensor-product of 1-D primitives only.
    phi_basis = TTBasis(phi_primitive)
    psi_basis = TTBasis(psi_primitive)

    # Rank-1 NestedTT coefficient vector a (needed for elementwise psi/q division).
    g_coeffs = NestedTTVectorParameters.from_core_spec(
        tt_modes,
        depth=depth,
        ranks=1,
        mean=1.0,
        std=1.0,
        epsilon=1e-3,
        device=device,
        dtype=dtype,
    )
    h0_coeffs = NestedTTVectorParameters.from_core_spec(
        tt_modes,
        depth=depth,
        ranks=1,
        mean=1.0,
        std=1.0,
        device=device,
        dtype=dtype,
    )

    # NestedTT-matrix B on the same mode shape / hierarchy depth.
    B = RowStochasticNestedTTMatrixParameters.from_core_spec(
        tt_modes,
        depth=depth,
        ranks=rank,
        mean=0.0,
        std=1.0,
        device=device,
        dtype=dtype,
    )

    rff = SumProdRFF(g_coeffs, phi_basis, psi_basis, B, numerical_tolerance=1e-10)

    print("Training transition model")
    mle_loss_fn = loss.conditional_mle_loss
    optimizers = {
        "basis": torch.optim.Adam(
            param_group_iter((phi_means, phi_std, psi_means, psi_std)),
            lr=tran_params["lr_basis"],
        ),
        "weights": torch.optim.Adam(
            param_group_iter((*g_coeffs.leaves, *B.leaves)),
            lr=tran_params["lr_weights"],
        ),
    }

    tran_model, best_loss_tran, training_time_tran = train.train_iterate(
        rff,
        xp_dataloader,
        {"mle": mle_loss_fn},
        optimizers,
        device=device,
        epochs_per_group=tran_params["n_epochs_per_group"],
        iterations=tran_params["iterations"],
        verbose=True,
        use_best="mle",
    )
    print("Done! \n")

    tran_model.eval()
    with torch.no_grad():
        test_loss_tran = float(
            loss.conditional_mle_loss(
                tran_model,
                x_kp1_test.to(device),
                x_k_test.to(device),
            )
            .detach()
            .cpu()
        )
    print(
        f"Transition test loss ({x_kp1_test.shape[0]} samples, {100.0 * test_fraction:.0f}% holdout): "
        f"{test_loss_tran:.6f}"
    )
    print(
        f"Transition train best loss: {best_loss_tran:.6f} "
        f"(gap test-train = {test_loss_tran - float(best_loss_tran):.6f})"
    )

    for p in param_group_iter((phi_means, phi_std, psi_means, psi_std, *g_coeffs.leaves)):
        p.requires_grad_(False)

    h0_basis = TTBasis(psi_primitive, coeffs=h0_coeffs)
    init_model = LinearFF.from_rff(rff, h0_basis).to(device=device, dtype=dtype)

    print("Training initial model")
    mle_loss_fn = loss.mle_loss
    optimizers = {
        "weights": torch.optim.Adam(
            param_group_iter(h0_coeffs.leaves),
            lr=init_params["lr_weights"],
        ),
    }

    init_model, best_loss_init, training_time_init = train.train_iterate(
        init_model,
        x0_dataloader,
        {"mle": mle_loss_fn},
        optimizers,
        device=device,
        epochs_per_group=init_params["n_epochs_per_group"],
        iterations=init_params["iterations"],
        verbose=True,
        use_best="mle",
    )
    print("Done! \n")

    print(
        f"Transition model loss: {best_loss_tran:.4f}, "
        f"training time: {training_time_tran:.2f} seconds"
    )
    print(
        f"Initial model loss: {best_loss_init:.4f}, "
        f"training time: {training_time_init:.2f} seconds"
    )
    print(f"Transition test loss: {test_loss_tran:.6f}")

    analysis_device = device
    init_model = init_model.to(device=analysis_device, dtype=dtype).eval()
    tran_model = tran_model.to(device=analysis_device, dtype=dtype).eval()

    output_dir = Path("figures/tt/acc_comp_ntt")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_slices = n_timesteps_prop + 1

    base_belief_seq = propagate.propagate(
        init_model,
        tran_model,
        n_steps=n_timesteps_prop,
    )
    belief_seq = [
        belief.to(device=analysis_device, dtype=dtype).eval()
        for belief in base_belief_seq
    ]

    ll_per_step = []
    for i in range(n_slices):
        data_i = traj_data[i].to(device=analysis_device, dtype=dtype)
        ll = avg_log_likelihood(belief_seq[i], data_i)
        ll_per_step.append(float(ll.detach().cpu()))
        print(f"Avg log-likelihood at time {i}: {ll_per_step[-1]:.6f}")
