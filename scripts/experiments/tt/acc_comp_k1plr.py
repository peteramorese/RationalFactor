"""Cartpole/acc_comp: Kronecker-plus-Kronecker-plus-low-rank SumProdRFF with unwrapped Gaussian TT bases.

No ErfSeparableTransform or mutual basis; dimensions correspond directly to
physical state coordinates. All n_local**dim objects remain TT/MPO-structured.
"""
from pathlib import Path
import math
import torch
from torch.utils.data import DataLoader, TensorDataset

from rational_factor.models.basis_functions import GaussianBasis
try:
    from rational_factor.models.tt.tt_basis import TTBasis
except ModuleNotFoundError as exc:
    if exc.name not in ('rational_factor.models.tt.tt_basis', 'rational_factor.models.tt'):
        raise
    from rational_factor.models.tt_basis import TTBasis
try:
    from rational_factor.models.tt.tt_parameters import TTVectorParameters, KroneckerPlusLowRankFactorizationMPOParameters
except ModuleNotFoundError as exc:
    if exc.name not in ('rational_factor.models.tt.tt_parameters', 'rational_factor.models.tt'):
        raise
    from rational_factor.models.tt_parameters import TTVectorParameters, KroneckerPlusLowRankFactorizationMPOParameters
from rational_factor.models.parameters import PositiveParameters, TrainableParameters, Parameters, param_group_iter
from rational_factor.models.factor_forms import SumProdRFF, LinearFF
from rational_factor.models.train import Trainer, LossVisualizer
# The propagation module may live under models/ or tools/ in repository layouts.
try:
    from rational_factor.models.propagate import propagate
except ModuleNotFoundError as exc:
    if exc.name != 'rational_factor.models.propagate':
        raise
    from rational_factor.tools.propagate import propagate
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood
import rational_factor.models.loss as loss


def _new_gaussian_basis(d: int, n_local: int, *, dtype: torch.dtype,
                        device: torch.device, bandwidth_floor: float = 0.01):
    shape = (1, d, n_local)
    means = TrainableParameters.random_init(shape, mean=0.0, std=3.5).to(device=device, dtype=dtype)
    stds = PositiveParameters.random_init(
        shape, mean=0.8, std=5.5, epsilon=bandwidth_floor,
    ).to(device=device, dtype=dtype)
    return TTBasis(GaussianBasis(means, stds)), (means, stds)


def _trainable_mixture_weight(init: float, *, dtype: torch.dtype, device: torch.device) -> TrainableParameters:
    """Scalar logit so sigmoid(logit) ≈ init; trained with the MPO weights."""
    eps = 1e-4
    w = min(max(float(init), eps), 1.0 - eps)
    logit = math.log(w / (1.0 - w))
    return TrainableParameters.from_values(
        torch.tensor(logit, dtype=dtype, device=device), trainable=True,
    )


def build_transition_model(d: int, *, n_local: int, latent_rank: int,
                           mpo_rank: int, dtype: torch.dtype, device: torch.device,
                           mixture_weight: float | Parameters = 0.5,
                           numerical_tolerance: float = 1e-10, gaussian_std_floor: float = 0.01):
    phi, phi_modules = _new_gaussian_basis(d, n_local, dtype=dtype, device=device, bandwidth_floor=gaussian_std_floor)
    psi, psi_modules = _new_gaussian_basis(d, n_local, dtype=dtype, device=device, bandwidth_floor=gaussian_std_floor)
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


def build_initial_model(rff: SumProdRFF, *, n_local: int, dtype: torch.dtype,
                        device: torch.device):
    coeffs = TTVectorParameters.from_core_spec(
        (n_local,) * rff.phi.dim(), ranks=1, param_cls=PositiveParameters,
        mean=1.0, std=0.15, epsilon=1e-3, dtype=dtype, device=device,
    )
    h0_basis = TTBasis(rff.psi.primitives)
    h0_basis.set_coeffs(coeffs)
    return LinearFF.from_rff(rff, h0_basis).to(device=device, dtype=dtype), coeffs


if __name__ == "__main__":
    problem = FULLY_OBSERVABLE_PROBLEMS["cartpole"]
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    dtype = torch.float64
    n_local = 30  
    latent_rank = 600
    mpo_rank = 20
    gaussian_std_epsilon = 0.3
    test_fraction = 0.15
    split_seed = 0
    # Initial low-rank mixture; trained as a logit (K share = 1 - sigmoid(logit)).
    mixture_weight_init = 0.5
    tran_params = {
        "epochs": 80,
        "basis_iterations": 5,
        "weights_iterations": 5,
        "lr_basis": 5e-3,
        "lr_weights": 2e-2,
        "validation_patience": 5,
    }
    init_params = {"epochs": 100, "coeff_iterations": 10, "lr_coeffs": 1e-2}
    batch_size = 256
    n_timesteps_prop = problem.n_timesteps
    d = problem.system.dim()

    x0 = problem.train_initial_state_data().to(dtype=dtype)
    x_k, x_kp1 = problem.train_state_transition_data()
    x_k, x_kp1 = x_k.to(dtype=dtype), x_kp1.to(dtype=dtype)
    generator = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(x_k.shape[0], generator=generator)
    n_val = max(1, min(x_k.shape[0] - 1, round(test_fraction * x_k.shape[0])))
    if x_k.shape[0] < 2:
        raise ValueError("At least two transition samples are needed for holdout validation")
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    traj_data = [x.to(dtype=dtype) for x in problem.test_data()]
    train_loader = DataLoader(TensorDataset(x_kp1[train_idx], x_k[train_idx]),
                              batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    val_loader = DataLoader(TensorDataset(x_kp1[val_idx], x_k[val_idx]),
                            batch_size=batch_size, shuffle=False, pin_memory=use_gpu)
    init_loader = DataLoader(TensorDataset(x0), batch_size=batch_size,
                             shuffle=True, pin_memory=use_gpu)
    output_dir = Path("figures/tt/acc_comp_k1plr")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}, dtype={dtype}, modes={(n_local,) * d}, latent rank={latent_rank}, MPO rank={mpo_rank}")

    rff, B_params, basis_params, weight_params = build_transition_model(
        d, n_local=n_local, latent_rank=latent_rank, mpo_rank=mpo_rank,
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
    # use_best='mle' avoids full validation-loader evaluation *after every*
    # optimizer step; val_mle and early stopping still run per epoch.
    rff, best_loss_tran, training_time_tran = trainer.train(
<<<<<<< HEAD
        epochs=tran_params["epochs"], use_best="validation_total", verbose=True,
=======
        epochs=tran_params["epochs"], use_best="validation", verbose=True,
>>>>>>> 05baa4a2b2820e85adfa77802eac5c496a99b6d9
    )
    if rff is None:
        raise SystemExit("No stable transition checkpoint was found.")

    rff.eval()
    with torch.no_grad():
        test_loss = loss.conditional_mle_loss(
            rff, x_kp1[val_idx].to(device), x_k[val_idx].to(device)
        )
        learned_mw = float(torch.sigmoid(B_params.mixture_weight().reshape(())))
    print(f"Holdout transition loss: {float(test_loss):.6f}")
    print(f"Learned mixture_weight (low-rank share): {learned_mw:.6f}")

    for parameter in rff.parameters():
        parameter.requires_grad_(False)
    init_model, h0_params = build_initial_model(
        rff, n_local=n_local, dtype=dtype, device=device,
    )
    init_trainer = Trainer(
        init_model, device=device,
        loss_visualizer=LossVisualizer(output_dir, buffer_size=5, filename="initial_losses.png"),
    )
    init_trainer.add_dataset(init_loader, "initial")
    init_trainer.add_optimizer(
        torch.optim.Adam(param_group_iter(tuple(h0_params.parameter_modules())),
                         lr=init_params["lr_coeffs"]),
        "coeffs", n_iterations=init_params["coeff_iterations"],
    )
    init_trainer.add_loss_function(loss.mle_loss, "mle", "initial")
    init_model, best_loss_init, training_time_init = init_trainer.train(
        epochs=init_params["epochs"], use_best="mle", verbose=True,
    )
    if init_model is None:
        raise SystemExit("No stable initial-density checkpoint was found.")

    print(f"Transition train loss: {best_loss_tran:.5f} ({training_time_tran:.1f}s)")
    print(f"Initial train loss: {best_loss_init:.5f} ({training_time_init:.1f}s)")
    init_model.eval()
    rff.eval()
    with torch.no_grad():
        beliefs = propagate(init_model, rff, n_steps=n_timesteps_prop)
        for i, (belief, samples) in enumerate(zip(beliefs, traj_data)):
            loglik = avg_log_likelihood(belief, samples.to(device=device, dtype=dtype))
            print(f"Avg log-likelihood at time {i}: {float(loglik):.6f}")
    print(f"Saved loss figures in {output_dir}")
