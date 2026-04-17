from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader, TensorDataset

import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.systems.truth_models as truth_models
from normalizing_flow.normalizing_flow import ConditionalNormalizingFlow, NormalizingFlow
from rational_factor.systems.base import sample_io_pairs
from rational_factor.tools.misc import make_mvnormal_init_sampler


SEED = 42
BATCH_SIZE = 512
N_DATA_TRAN = 20_000
N_DATA_INIT = 2_000

TRAIN_PARAMS = {
    "init_epochs": 80,
    "tran_epochs": 80,
    "lr": 1e-3,
    "num_layers": 5,
    "hidden_features": 128,
}


def make_quadcopter_benchmark_system() -> truth_models.Quadcopter:
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
    return make_mvnormal_init_sampler(mean=mean, covariance=torch.diag(variances))


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
    dist = torch.distributions.MultivariateNormal(mean, cov)

    def _sample(n_samples: int):
        return dist.sample((n_samples,))

    return _sample


def evaluate_init_loss(model: NormalizingFlow, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            total += loss.mle_loss(model, x).item()
    model.train()
    return total / len(loader)


def evaluate_transition_loss(
    model: ConditionalNormalizingFlow, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for xp, x in loader:
            xp = xp.to(device)
            x = x.to(device)
            total += loss.conditional_mle_loss(model, xp, x).item()
    model.train()
    return total / len(loader)


def main() -> None:
    torch.manual_seed(SEED)

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"Using device: {device}")

    system = make_quadcopter_benchmark_system()
    init_state_sampler = quadcopter_init_state_sampler()
    prev_state_sampler = quadcopter_prev_state_sampler(system)

    t_data_start = time.time()
    x0_data = init_state_sampler(N_DATA_INIT)
    x_k_data, x_kp1_data = sample_io_pairs(system, prev_state_sampler, n_pairs=N_DATA_TRAN)
    print(f"Generated datasets in {time.time() - t_data_start:.2f}s")
    print(f"Initial-state data shape: {tuple(x0_data.shape)}")
    print(f"Transition data shapes: x_k={tuple(x_k_data.shape)}, x_kp1={tuple(x_kp1_data.shape)}")

    init_loader = DataLoader(
        TensorDataset(x0_data),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=use_gpu,
    )
    tran_loader = DataLoader(
        TensorDataset(x_kp1_data, x_k_data),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=use_gpu,
    )

    init_model = NormalizingFlow(
        dim=system.dim(),
        num_layers=TRAIN_PARAMS["num_layers"],
        hidden_features=TRAIN_PARAMS["hidden_features"],
    ).to(device)
    transition_model = ConditionalNormalizingFlow(
        dim=system.dim(),
        conditioner_dim=system.dim(),
        num_layers=TRAIN_PARAMS["num_layers"],
        hidden_features=TRAIN_PARAMS["hidden_features"],
    ).to(device)

    print("\nTraining conditional transition flow")
    transition_model, best_loss_tran, training_time_tran = train.train(
        transition_model,
        tran_loader,
        {"mle": loss.conditional_mle_loss},
        torch.optim.Adam(transition_model.parameters(), lr=TRAIN_PARAMS["lr"]),
        epochs=TRAIN_PARAMS["tran_epochs"],
        verbose=True,
        use_best="mle",
    )

    print("\nTraining initial-state flow")
    init_model, best_loss_init, training_time_init = train.train(
        init_model,
        init_loader,
        {"mle": loss.mle_loss},
        torch.optim.Adam(init_model.parameters(), lr=TRAIN_PARAMS["lr"]),
        epochs=TRAIN_PARAMS["init_epochs"],
        verbose=True,
        use_best="mle",
    )

    eval_loss_tran = evaluate_transition_loss(transition_model, tran_loader, device)
    eval_loss_init = evaluate_init_loss(init_model, init_loader, device)

    print("\nLoss summary")
    print(f"Transition flow best MLE loss: {best_loss_tran:.6f}")
    print(f"Transition flow eval MLE loss: {eval_loss_tran:.6f}")
    print(f"Transition flow training time: {training_time_tran:.2f}s")
    print(f"Initial-state flow best MLE loss: {best_loss_init:.6f}")
    print(f"Initial-state flow eval MLE loss: {eval_loss_init:.6f}")
    print(f"Initial-state flow training time: {training_time_init:.2f}s")


if __name__ == "__main__":
    main()
