"""Cartpole/acc_comp: Kronecker-plus-low-rank SumProdRFF with unwrapped Gaussian TT bases.

No ErfSeparableTransform or mutual basis; dimensions correspond directly to
physical state coordinates. All n_local**dim objects remain TT/MPO-structured.
"""
from pathlib import Path
import math
import torch
from torch.utils.data import DataLoader, TensorDataset

from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.tools.misc import init_gaussian_basis_params_gmm
from rational_factor.models.tt.tt_basis import TTBasis
from rational_factor.models.tt.tt_parameters import TTVectorParameters, KroneckerPlusLowRankFactorizationMPOParameters
from rational_factor.models.parameters import PositiveParameters, TrainableParameters, Parameters, param_group_iter
from rational_factor.models.factor_forms import SumProdRFF, LinearFF
from rational_factor.models.train import Trainer, LossVisualizer
from rational_factor.tools.analysis import avg_log_likelihood
from rational_factor.tools.misc import train_test_split
import rational_factor.models.loss as loss
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS

def _new_gaussian_basis(
    data: torch.Tensor, d: int, n_local: int, *,
    dtype: torch.dtype, device: torch.device,
    bandwidth_floor: float = 0.0,
    gmm_reg_covar: float = 1e-5,
    gmm_max_iter: int = 100,
    gmm_seed: int = 0,
):
    """EM-fit Gaussian primitives to the supplied state distribution."""
    means, stds = init_gaussian_basis_params_gmm(
        data, d, n_local,
        reg_covar=gmm_reg_covar, max_iter=gmm_max_iter,
        random_state=gmm_seed, epsilon=bandwidth_floor,
        dtype=dtype, device=device,
    )
    return TTBasis(GaussianBasis(mean_params=means, std_params=stds)), (means, stds)

def _trainable_mixture_weight(init: float, *, dtype: torch.dtype, device: torch.device) -> TrainableParameters:
    """Scalar logit so sigmoid(logit) ≈ init; trained with the MPO weights."""
    eps = 1e-4
    w = min(max(float(init), eps), 1.0 - eps)
    logit = math.log(w / (1.0 - w))
    return TrainableParameters.from_values(
        torch.tensor(logit, dtype=dtype, device=device), trainable=True,
    )


def build_transition_model(x_k: torch.Tensor, x_kp1: torch.Tensor, d: int, *, n_local: int, latent_rank: int,
                           mpo_rank: int, dtype: torch.dtype, device: torch.device,
                           mixture_weight: float | Parameters = 0.5,
                           numerical_tolerance: float = 1e-10, gaussian_std_floor: float = 0.01):
    phi, phi_modules = _new_gaussian_basis(x_k, d, n_local, dtype=dtype, device=device, bandwidth_floor=gaussian_std_floor)
    psi, psi_modules = _new_gaussian_basis(x_kp1, d, n_local, dtype=dtype, device=device, bandwidth_floor=gaussian_std_floor)
    modes = (n_local,) * d
    a = TTVectorParameters.from_core_spec(
        modes, ranks=1, param_cls=PositiveParameters,
        mean=1.0, std=0.15, epsilon=1e-3, dtype=dtype, device=device,
    )
    if not isinstance(mixture_weight, Parameters):
        mixture_weight = _trainable_mixture_weight(mixture_weight, dtype=dtype, device=device)
    B = KroneckerPlusLowRankFactorizationMPOParameters.from_core_spec(
        modes, rank=latent_rank, ranks_u=mpo_rank, ranks_v=mpo_rank,
        row_stochastic=True, mixture_weight=mixture_weight,
        mean=0.0, std=0.2, dtype=dtype, device=device,
    )
    rff = SumProdRFF(a, phi, psi, B, numerical_tolerance=numerical_tolerance)
    return (
        rff,
        B,
        param_group_iter((*phi_modules, *psi_modules)),
        param_group_iter((*a.parameter_modules(), *B.parameter_modules())),
    )


def build_initial_model(rff: SumProdRFF, x0: torch.Tensor, *, n_local: int, dtype: torch.dtype,
                        device: torch.device, init_tt_rank: int = 1,
                        gaussian_std_floor: float = 0.01):
    coeffs = TTVectorParameters.from_core_spec(
        (n_local,) * rff.phi.dim(), ranks=init_tt_rank, param_cls=PositiveParameters,
        mean=1.0, std=0.15, epsilon=1e-3, dtype=dtype, device=device,
    )
    # Independent initial psi0, not the transition psi.
    h0_basis, psi0_modules = _new_gaussian_basis(
        x0, rff.phi.dim(), n_local, dtype=dtype, device=device,
        bandwidth_floor=gaussian_std_floor,
    )
    return (LinearFF.from_rff(rff, coeffs, h0_basis).to(device=device, dtype=dtype),
            coeffs, psi0_modules)


if __name__ == "__main__":
    problem = FULLY_OBSERVABLE_PROBLEMS["quadcopter"]
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    dtype = torch.float64
    n_local = 50
    latent_rank = 200
    mpo_rank = 2
    gaussian_std_epsilon = 0.01
    test_fraction = 0.15
    split_seed = 0
    mixture_weight_init = 0.5
    tran_params = {
        "epochs": 40,
        "basis_iterations": 2,
        "weights_iterations": 5,
        "lr_basis": 5e-2,
        "lr_weights": 5e-1,
        "validation_patience": 50,
    }
    init_tt_rank = 10
    init_params = {"epochs": 100, "coeff_iterations": 10, "lr_coeffs": 1e-2,
                   "basis_iterations": 2, "lr_basis": 1e-3, "validation_patience": 50}
    batch_size = 2048 #512
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
    output_dir = Path("figures/tt/acc_comp_k1plr")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}, dtype={dtype}, modes={(n_local,) * d}, latent rank={latent_rank}, MPO rank={mpo_rank}")

    rff, B_params, basis_params, weight_params = build_transition_model(
        x_k_train, x_kp1_train, d, n_local=n_local, latent_rank=latent_rank, mpo_rank=mpo_rank,
        mixture_weight=mixture_weight_init,
        dtype=dtype, device=device, numerical_tolerance=1e-10,
        gaussian_std_floor=gaussian_std_epsilon,
    )
    print(f"trainable mixture_weight init={mixture_weight_init}")
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
    trainer.add_loss_function(loss.conditional_mle_loss, "val_mle", "validation", validation=True)
    rff, best_loss_tran, training_time_tran = trainer.train(
        epochs=tran_params["epochs"], use_best="mle", verbose=True,
    )
    if rff is None:
        raise SystemExit("No stable transition checkpoint was found.")

    rff.eval()
    learned_mw = float(torch.sigmoid(B_params.mixture_weight().reshape(())))
    print(f"Learned mixture_weight (low-rank share): {learned_mw:.6f}")

    for parameter in rff.parameters():
        parameter.requires_grad_(False)
    init_model, h0_params, psi0_modules = build_initial_model(
        rff, x0, n_local=n_local, dtype=dtype, device=device,
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
            loglik = avg_log_likelihood(belief, samples.to(device=device, dtype=dtype), max_chunks=128)
            print(f"Avg log-likelihood at time {i}: {float(loglik):.6f}")
    print(f"Saved loss figures in {output_dir}")
