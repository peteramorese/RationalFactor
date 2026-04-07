"""
Train/propagate experiment across basis families and validate normalization.

Run:
  PYTHONPATH=src python test/test_basis.py
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.systems.truth_models as truth_models
import rational_factor.tools.propagate as propagate
from rational_factor.models.basis_functions import BetaBasis, GaussianBasis, QuadraticExpBasis
from rational_factor.models.factor_forms import LinearFF, LinearRFF
from rational_factor.systems.base import create_transition_data_matrix, sample_trajectories
from rational_factor.tools.analysis import check_pdf_valid

# --- test parameters ---------------------------------------------------------

BASIS_NAMES = ["quadratic_exp"] #["gaussian", "quadratic_exp"]
USE_GPU_IF_AVAILABLE = True

N_BASIS = 20
N_EPOCHS = 100
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
N_TIMESTEPS_TRAIN = 10
N_TIMESTEPS_PROP = 10
N_TRAJECTORIES_TRAIN = 2000

SEED = 0
CHECK_DOMAIN_BOUNDS = ((-5.0, -5.0), (5.0, 5.0))
NUM_MC_SAMPLES = 100000

# Basis initialization offsets by family.
GAUSSIAN_OFFSETS = (0.0, 0.5)
BETA_OFFSETS = (0.0, 0.5)
QUADRATIC_EXP_OFFSETS = (0.0, 0.5, 0.0)

# ---------------------------------------------------------------------------



def _make_basis(name: str, d: int, n_basis: int):
    if name == "gaussian":
        offsets = torch.tensor(GAUSSIAN_OFFSETS)
        return GaussianBasis.set_init(d, n_basis=n_basis, offsets=offsets)
    if name == "beta":
        offsets = torch.tensor(BETA_OFFSETS)
        return BetaBasis.set_init(d, n_basis=n_basis, offsets=offsets)
    if name == "quadratic_exp":
        offsets = torch.tensor(QUADRATIC_EXP_OFFSETS)
        return QuadraticExpBasis.set_init(d, n_basis=n_basis, offsets=offsets)
    raise ValueError(f"Unknown basis name: {name}")


def _build_losses_for_transition():
    return {"mle": loss.conditional_mle_loss}


def _build_losses_for_initial():
    return {"mle": loss.mle_loss}


def run_basis_experiment(basis_name: str, device: torch.device, use_gpu: bool) -> None:
    print(f"\n=== Basis: {basis_name} ===")

    system = truth_models.VanDerPol(dt=0.3, mu=0.9, covariance=0.1 * torch.eye(2))
    mean = torch.tensor([0.2, 0.1])
    cov = torch.diag(torch.tensor([0.2, 0.2]))
    init_state_sampler = make_mvnormal_init_sampler(mean, cov)

    traj_data = sample_trajectories(
        system,
        init_state_sampler,
        n_timesteps=N_TIMESTEPS_TRAIN,
        n_trajectories=N_TRAJECTORIES_TRAIN,
    )
    x0_data = TensorDataset(traj_data[0])
    x_k, x_kp1 = create_transition_data_matrix(traj_data, separate=True)
    xp_data = TensorDataset(x_k, x_kp1)

    x0_dataloader = DataLoader(x0_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(xp_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=use_gpu)

    phi_basis = _make_basis(basis_name, system.dim(), N_BASIS)
    psi_basis = _make_basis(basis_name, system.dim(), N_BASIS)
    psi0_basis = _make_basis(basis_name, system.dim(), N_BASIS)

    tran_model = LinearRFF(phi_basis, psi_basis).to(device)
    print("Training transition model")
    tran_model, _, _ = train.train(
        tran_model,
        xp_dataloader,
        _build_losses_for_transition(),
        torch.optim.Adam(tran_model.parameters(), lr=LEARNING_RATE),
        epochs=N_EPOCHS,
        use_best="mle",
    )
    print("Done!")

    init_model = LinearFF.from_rff(tran_model, psi0_basis).to(device)
    print("Training initial model")
    init_model, _, _ = train.train(
        init_model,
        x0_dataloader,
        _build_losses_for_initial(),
        torch.optim.Adam(init_model.parameters(), lr=LEARNING_RATE),
        epochs=N_EPOCHS,
        use_best="mle",
    )
    print("Done!")

    belief_seq = propagate.propagate(init_model, tran_model, n_steps=N_TIMESTEPS_PROP)

    for i, belief in enumerate(belief_seq):
        print(f"Checking normalization for propagated belief {i + 1}/{len(belief_seq)}")
        check_pdf_valid(belief, domain_bounds=CHECK_DOMAIN_BOUNDS, n_samples=NUM_MC_SAMPLES)


def main() -> None:
    if SEED is not None:
        torch.manual_seed(SEED)

    use_gpu = USE_GPU_IF_AVAILABLE and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    for basis_name in BASIS_NAMES:
        run_basis_experiment(basis_name, device=device, use_gpu=use_gpu)

    print("\nAll basis propagation normalization checks completed.")


if __name__ == "__main__":
    main()


