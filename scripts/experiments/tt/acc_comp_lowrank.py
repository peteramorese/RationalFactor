"""Cartpole/acc_comp: low-rank SumProdRFF with unwrapped Gaussian TT bases.

No ErfSeparableTransform or mutual basis; dimensions correspond directly to
physical state coordinates. All n_local**dim objects remain TT/MPO-structured.
"""
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.tt.tt_basis import TTBasis
from rational_factor.models.tt.tt_parameters import TTVectorParameters, LowRankFactorizationMPOParameters
from rational_factor.models.parameters import PositiveParameters, TrainableParameters, param_group_iter
from rational_factor.models.factor_forms import SumProdRFF, LinearFF
from rational_factor.models.train import Trainer, LossVisualizer
from rational_factor.tools.propagate import propagate
from rational_factor.tools.analysis import avg_log_likelihood
from rational_factor.tools.misc import train_test_split
import rational_factor.models.loss as loss
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS

def _new_gaussian_basis(d: int, n_local: int, *, dtype: torch.dtype,
                        device: torch.device, bandwidth_floor: float = 0.3):
    shape = (1, d, n_local)
    means = TrainableParameters.random_init(shape, mean=0.0, std=3.5).to(device=device, dtype=dtype)
    stds = PositiveParameters.random_init(
        shape, mean=0.8, std=0.5, epsilon=bandwidth_floor,
    ).to(device=device, dtype=dtype)
    return TTBasis(GaussianBasis(means, stds)), (means, stds)


def build_transition_model(d: int, *, n_local: int, latent_rank: int,
                           mpo_rank: int, dtype: torch.dtype, device: torch.device,
                           numerical_tolerance: float = 1e-10, gaussian_std_floor: float = 0.3):
    phi, phi_modules = _new_gaussian_basis(d, n_local, dtype=dtype, device=device, bandwidth_floor=gaussian_std_floor)
    psi, psi_modules = _new_gaussian_basis(d, n_local, dtype=dtype, device=device, bandwidth_floor=gaussian_std_floor)
    modes = (n_local,) * d
    a = TTVectorParameters.from_core_spec(
        modes, ranks=1, param_cls=PositiveParameters,
        mean=1.0, std=0.15, epsilon=1e-3, dtype=dtype, device=device,
    )
    B = LowRankFactorizationMPOParameters.from_core_spec(
        modes, rank=latent_rank, ranks_u=mpo_rank, ranks_v=mpo_rank,
        row_stochastic=True, mean=0.0, std=0.2, dtype=dtype, device=device,
    )
    rff = SumProdRFF(a, phi, psi, B, numerical_tolerance=numerical_tolerance)
    return (
        rff,
        param_group_iter((*phi_modules, *psi_modules)),
        param_group_iter((*a.parameter_modules(), *B.parameter_modules())),
    )


def build_initial_model(rff: SumProdRFF, *, n_local: int, dtype: torch.dtype,
                        device: torch.device, init_tt_rank: int = 1,
                        gaussian_std_floor: float = 0.01):
    coeffs = TTVectorParameters.from_core_spec(
        (n_local,) * rff.phi.dim(), ranks=init_tt_rank, param_cls=PositiveParameters,
        mean=1.0, std=0.15, epsilon=1e-3, dtype=dtype, device=device,
    )
    # Independent initial psi0, not the transition psi.
    h0_basis, psi0_modules = _new_gaussian_basis(
        rff.phi.dim(), n_local, dtype=dtype, device=device,
        bandwidth_floor=gaussian_std_floor,
    )
    return (LinearFF.from_rff(rff, coeffs, h0_basis).to(device=device, dtype=dtype),
            coeffs, psi0_modules)


if __name__ == "__main__":
    problem = FULLY_OBSERVABLE_PROBLEMS["cartpole"]
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    dtype = torch.float64
    n_local = 6  # 6**4=1296 coefficients, but never materialized
    latent_rank = 10
    mpo_rank = 10
    gaussian_std_epsilon = 0.3
    test_fraction = 0.15
    split_seed = 0
    tran_params = {
        "epochs": 100,
        "basis_iterations": 5,
        "weights_iterations": 2,
        "lr_basis": 1.2e-2,
        "lr_weights": 5e-2,
        "validation_patience": 20,
    }
    init_tt_rank = 4
    init_params = {"epochs": 50, "coeff_iterations": 10, "lr_coeffs": 1e-2,
                   "basis_iterations": 2, "lr_basis": 1e-3, "validation_patience": 20}
    batch_size = 512
    n_timesteps_prop = problem.n_timesteps
    d = problem.system.dim()

    x0 = problem.train_initial_state_data().to(dtype=dtype)
    x_k, x_kp1 = problem.train_state_transition_data()
    x_k, x_kp1 = x_k.to(dtype=dtype), x_kp1.to(dtype=dtype)
    x0_train, x0_val = train_test_split(
        x0, test_size=test_fraction, shuffle=True, seed=split_seed,
    )
    x_k_train, x_k_val, x_kp1_train, x_kp1_val = train_test_split(
        x_k, x_kp1, test_size=test_fraction, shuffle=True, seed=split_seed,
    )
    traj_data = [x.to(dtype=dtype) for x in problem.test_data()]
    train_loader = DataLoader(TensorDataset(x_kp1_train, x_k_train),
                              batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    val_loader = DataLoader(TensorDataset(x_kp1_val, x_k_val),
                            batch_size=batch_size, shuffle=False, pin_memory=use_gpu)
    init_loader = DataLoader(TensorDataset(x0_train), batch_size=batch_size,
                             shuffle=True, pin_memory=use_gpu)
    init_val_loader = DataLoader(TensorDataset(x0_val), batch_size=batch_size,
                                 shuffle=False, pin_memory=use_gpu)
    output_dir = Path("figures/tt/acc_comp")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}, dtype={dtype}, modes={(n_local,) * d}, latent rank={latent_rank}, MPO rank={mpo_rank}")

    rff, basis_params, weight_params = build_transition_model(
        d, n_local=n_local, latent_rank=latent_rank, mpo_rank=mpo_rank,
        dtype=dtype, device=device, numerical_tolerance=1e-10,
        gaussian_std_floor=gaussian_std_epsilon,
    )
    trainer = Trainer(
        rff, device=device,
        validation_early_stopping_patience=tran_params["validation_patience"],
        loss_visualizer=LossVisualizer(output_dir, buffer_size=5, filename="transition_losses.png"),
    )
    trainer.add_dataset(train_loader, "transitions")
    trainer.add_dataset(val_loader, "validation")
    trainer.add_optimizer(
        torch.optim.Adam(basis_params, lr=tran_params["lr_basis"]),
        "basis", n_iterations=tran_params["basis_iterations"],
    )
    trainer.add_optimizer(
        torch.optim.Adam(weight_params, lr=tran_params["lr_weights"]),
        "weights", n_iterations=tran_params["weights_iterations"],
    )
    trainer.add_loss_function(loss.conditional_mle_loss, "mle", "transitions")
    trainer.add_loss_function(loss.conditional_mle_loss, "validation_mle", "validation", validation=True)
    # use_best='mle' avoids full validation-loader evaluation *after every*
    # optimizer step; validation_mle and early stopping still run per epoch.
    rff, best_loss_tran, training_time_tran = trainer.train(
        epochs=tran_params["epochs"], use_best="mle", verbose=True,
    )
    if rff is None:
        raise SystemExit("No stable transition checkpoint was found.")

    rff.eval()

    for parameter in rff.parameters():
        parameter.requires_grad_(False)
    init_model, h0_params, psi0_modules = build_initial_model(
        rff, n_local=n_local, dtype=dtype, device=device,
        init_tt_rank=init_tt_rank, gaussian_std_floor=gaussian_std_epsilon,
    )
    init_trainer = Trainer(
        init_model, device=device,
        validation_early_stopping_patience=init_params["validation_patience"],
        loss_visualizer=LossVisualizer(output_dir, buffer_size=5, filename="initial_losses.png"),
    )
    init_trainer.add_dataset(init_loader, "initial")
    init_trainer.add_dataset(init_val_loader, "validation")
    init_trainer.add_optimizer(
        torch.optim.Adam(param_group_iter(tuple(h0_params.parameter_modules())),
                         lr=init_params["lr_coeffs"]),
        "coeffs", n_iterations=init_params["coeff_iterations"],
    )
    init_trainer.add_optimizer(
        torch.optim.Adam(param_group_iter(psi0_modules), lr=init_params["lr_basis"]),
        "psi0_basis", n_iterations=init_params["basis_iterations"],
    )
    init_trainer.add_loss_function(loss.mle_loss, "mle", "initial")
    init_trainer.add_loss_function(loss.mle_loss, "val_mle", "validation", validation=True)
    init_model, best_loss_init, training_time_init = init_trainer.train(
        epochs=init_params["epochs"], use_best="mle", verbose=True,
    )
    if init_model is None:
        raise SystemExit("No stable initial-density checkpoint was found.")

    init_model.eval()
    with torch.no_grad():
        init_test_loss = loss.mle_loss(init_model, x0_val.to(device))
    print(f"Holdout initial loss: {float(init_test_loss):.6f}")

    print(f"Transition train loss: {best_loss_tran:.5f} ({training_time_tran:.1f}s)")
    print(f"Initial train loss: {best_loss_init:.5f} ({training_time_init:.1f}s)")
    init_model.eval()
    rff.eval()
    with torch.no_grad():
        beliefs = [init_model]
        for _ in range(n_timesteps_prop):
            beliefs.append(rff.propagate(beliefs[-1]))
        for i, (belief, samples) in enumerate(zip(beliefs, traj_data)):
            loglik = avg_log_likelihood(belief, samples.to(device=device, dtype=dtype))
            print(f"Avg log-likelihood at time {i}: {float(loglik):.6f}")
    print(f"Saved loss figures in {output_dir}")
