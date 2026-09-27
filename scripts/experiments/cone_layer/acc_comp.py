from pathlib import Path
from math import ceil

import torch
from torch.utils.data import DataLoader, TensorDataset

from normalizing_flow.composite_model import CompositeConditionalModel, CompositeDensityModel
from normalizing_flow.transforms import Transforms
from rational_factor.models.autoregressive_basis import AutoregressiveConeLayerBasis
from rational_factor.models.basis_functions import BSpline1DBasis
from rational_factor.models.factor_forms import SumProdRFF, LinearFF
from rational_factor.models.parameters import (
    PositiveParameters,
    LowRankFactorizationParameters,
    DenseMatrixFactorization,
    param_group_iter,
)
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood
import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.tools.propagate as propagate


def _inverse_softplus(y: torch.Tensor) -> torch.Tensor:
    """Map positive targets to PositiveParameters raw values (softplus + eps)."""
    return y + torch.log(-torch.expm1(-y.clamp_min(torch.finfo(y.dtype).tiny)))


def _positive_low_rank(
    d: int,
    m: int,
    r: int,
    *,
    device: torch.device,
    epsilon: float = 1e-6,
) -> LowRankFactorizationParameters:
    """Near-diagonal nonnegative rank-r factors for A0/B0.

    i.i.d. PositiveParameters factors yield dense all-positive A = U Vᵀ whose
    columns are nearly parallel, so α(x) ≈ c(x) v₀ across the domain and the
    RFF collapses to a near-uniform unit-box density (loss ≈ erf Jacobian only).
    Localized factor columns keep A banded and the bases expressive.
    """
    rows = torch.arange(m, device=device, dtype=torch.float32).unsqueeze(1)
    centers = torch.linspace(0.0, m - 1.0, r, device=device).unsqueeze(0)
    width = max(m / r, 1.5)
    envelope = 0.05 + 0.95 * torch.exp(-0.5 * ((rows - centers) / width) ** 2)
    scale = 1.0
    U = scale * envelope * torch.exp(0.15 * torch.randn(d, m, r, device=device))
    V = scale * envelope * torch.exp(0.15 * torch.randn(d, m, r, device=device))
    U_raw = _inverse_softplus((U - epsilon).clamp_min(torch.finfo(U.dtype).tiny))
    V_raw = _inverse_softplus((V - epsilon).clamp_min(torch.finfo(V.dtype).tiny))
    return LowRankFactorizationParameters(
        PositiveParameters.from_values(U_raw, epsilon=epsilon).to(device),
        PositiveParameters.from_values(V_raw, epsilon=epsilon).to(device),
    )


def _jitter_cone_mlp_last_layers(pair: AutoregressiveConeLayerBasis, std: float = 1e-3) -> None:
    """Break exact zero-init on update MLP heads so cone layers get signal."""
    for layer in pair.update_mlps:
        for mlp in layer:
            if mlp.net is None:
                continue
            torch.nn.init.normal_(mlp.net[-1].weight, std=std)
            torch.nn.init.zeros_(mlp.net[-1].bias)


if __name__ == "__main__":
    problem = FULLY_OBSERVABLE_PROBLEMS["cartpole"]

    ###
    use_gpu = torch.cuda.is_available()
    n_basis = 81
    bspline_degree = 3
    n_cone_layers = 4
    n_hidden_features = 64
    tran_params = {
        "n_epochs_per_group": [5, 3],  # wrap + cone layers, weights
        "iterations": 40,
        "lr_basis": 3e-3,
        "lr_weights": 5e-2,
        "lr_wrap": 1e-3,
    }
    init_params = {
        "n_epochs_per_group": [10],  # h0 coeffs only
        "iterations": 50,
        "lr_weights": 1e-2,
    }

    batch_size = 256
    n_timesteps_prop = problem.n_timesteps

    ###

    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using GPU: ", use_gpu)
    print("Device: ", device)

    system = problem.system
    dim = system.dim()

    m = n_basis
    d = dim
    rank = max(1, ceil(m ** (1.0 / d)))

    x0 = problem.train_initial_state_data()
    x_k, x_kp1 = problem.train_state_transition_data()
    traj_data = problem.test_data()

    x0_dataloader = DataLoader(
        TensorDataset(x0), batch_size=batch_size, shuffle=True, pin_memory=use_gpu
    )
    xp_dataloader = DataLoader(
        TensorDataset(x_kp1, x_k), batch_size=batch_size, shuffle=True, pin_memory=use_gpu
    )

    n_cells = n_basis - bspline_degree
    if n_cells < 1:
        raise ValueError(f"n_basis={n_basis} too small for degree={bspline_degree}")

    print(f"n_basis={m}, dim={d}, rank=ceil(m^(1/d))={rank}, n_cone_layers={n_cone_layers}")

    nom_alpha_basis = BSpline1DBasis(
        n_cells=n_cells,
        degree=bspline_degree,
        device=device,
    )
    nom_beta_basis = BSpline1DBasis(
        n_cells=n_cells,
        degree=bspline_degree,
        device=device,
    )
    assert nom_alpha_basis.n_basis_functions() == n_basis

    A0 = _positive_low_rank(d, m, rank, device=device)
    B0 = _positive_low_rank(d, m, rank, device=device)

    phi_psi_mutual = AutoregressiveConeLayerBasis(
        nom_alpha_basis,
        nom_beta_basis,
        A0,
        B0,
        n_layers=n_cone_layers,
        hidden_features=n_hidden_features,
    ).to(device)
    _jitter_cone_mlp_last_layers(phi_psi_mutual)

    g_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0]), epsilon=1e-3
    ).to(device)
    h0_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0])
    ).to(device)

    B_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis, n_basis),
        mean=torch.tensor([1.0]),
        std=torch.tensor([1.0]),
        epsilon=0.0,
    ).to(device)
    B = DenseMatrixFactorization(B_coeffs)

    g_basis = phi_psi_mutual.get_basis(0, coeffs=g_coeffs)
    psi_basis = phi_psi_mutual.get_basis(1)

    # cartpole default tolerance (1e-20) is too tight once basis values are O(1e-6).
    rff = SumProdRFF(g_basis, psi_basis, B, numerical_tolerance=1e-10)
    tran_model = CompositeConditionalModel(
        rff,
        context_features=dim,
        transform="erf",
        x_data=x_k,
        trainable=True,
        transform_conditioner=True,
    ).to(device)

    print("Training transition model")
    mle_loss_fn = loss.conditional_mle_loss
    optimizers = {
        "basis": torch.optim.Adam(
            [
                {"params": phi_psi_mutual.parameters(), "lr": tran_params["lr_basis"]},
                {"params": tran_model.transform.parameters(), "lr": tran_params["lr_wrap"]},
            ]
        ),
        "weights": torch.optim.Adam(
            param_group_iter((g_coeffs, B_coeffs)),
            lr=tran_params["lr_weights"],
        ),
    }

    tran_model, best_loss_tran, training_time_tran = train.train_iterate(
        tran_model,
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

    for p in phi_psi_mutual.parameters():
        p.requires_grad_(False)
    g_coeffs.set_requires_grad(False)
    trained_wrap_tf = Transforms.freeze(tran_model.transform).to(device)

    h0_basis = phi_psi_mutual.get_basis(1, coeffs=h0_coeffs)
    init_model = CompositeDensityModel(
        LinearFF.from_rff(tran_model.base_density, h0_basis),
        transform="stacked",
        transforms=[trained_wrap_tf],
    ).to(device)

    print("Training initial model")
    mle_loss_fn = loss.mle_loss
    optimizers = {
        "weights": torch.optim.Adam(h0_coeffs.parameters(), lr=init_params["lr_weights"]),
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

    analysis_device = device
    init_model = init_model.to(analysis_device).eval()
    tran_model = tran_model.to(analysis_device).eval()
    trained_wrap_tf = trained_wrap_tf.to(analysis_device).eval()

    output_dir = Path("figures/cone_layer/acc_comp")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_slices = n_timesteps_prop + 1

    base_belief_seq = propagate.propagate(
        init_model.base_density,
        tran_model.base_density,
        n_steps=n_timesteps_prop,
    )
    belief_seq = [
        CompositeDensityModel(
            belief,
            transform="stacked",
            transforms=[trained_wrap_tf],
        ).to(analysis_device).eval()
        for belief in base_belief_seq
    ]

    ll_per_step = []
    for i in range(n_slices):
        data_i = traj_data[i].to(analysis_device)
        ll = avg_log_likelihood(belief_seq[i], data_i)
        ll_per_step.append(float(ll.detach().cpu()))
        print(f"Avg log-likelihood at time {i}: {ll_per_step[-1]:.6f}")
