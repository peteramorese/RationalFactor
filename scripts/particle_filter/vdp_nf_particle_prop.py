import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from normalizing_flow.normalizing_flow import NormalizingFlow, ConditionalNormalizingFlow
import rational_factor.systems.po_truth_models as po_truth_models
from rational_factor.systems.base import (
    sample_io_pairs,
    sample_observation_pairs,
    simulate,
    SystemObservationDistribution,
    SystemTransitionDistribution,
)
from rational_factor.models import loss
import rational_factor.models.train as rf_train
from rational_factor.tools.misc import make_mvnormal_init_sampler
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.visualization import plot_particle_belief

from particle_filter.particle_set import WeightedParticleSet
from particle_filter.propagate import propagate_and_update


def main():
    problem = FULLY_OBSERVABLE_PROBLEMS["van_der_pol"]
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"Using device: {device}")

    seed = 42
    torch.manual_seed(seed)

    batch_size = 512
    n_data_init = 5000
    n_data_tran = 40000
    n_data_obs = 40000
    n_timesteps = 10
    n_particles = 4000

    train_params = {
        "init_epochs": 80,
        "tran_epochs": 80,
        "obs_epochs": 80,
        "lr": 1e-3,
        "num_layers": 5,
        "hidden_features": 64,
    }

    system = po_truth_models.PartiallyObservableVanDerPol(
        dt=0.3,
        mu=0.9,
        process_covariance=0.1 * torch.eye(2),
        observation_covariance=0.1 * torch.eye(2),
    )

    init_state_sampler = make_mvnormal_init_sampler(
        mean=torch.tensor([0.2, 0.1]),
        covariance=torch.diag(torch.tensor([0.2, 0.2])),
    )

    def prev_state_sampler(n_samples: int):
        mean = torch.tensor([0.0, 0.0])
        cov = torch.diag(4.0 * torch.ones(system.dim()))
        return torch.distributions.MultivariateNormal(mean, cov).sample((n_samples,))

    x0_data = init_state_sampler(n_data_init)
    x_k_data, x_kp1_data = sample_io_pairs(system, prev_state_sampler, n_pairs=n_data_tran)
    x_data, o_data = sample_observation_pairs(system, init_state_sampler, n_pairs=n_data_obs)

    init_loader = DataLoader(TensorDataset(x0_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    tran_loader = DataLoader(TensorDataset(x_kp1_data, x_k_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    obs_loader = DataLoader(TensorDataset(o_data, x_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    init_model = NormalizingFlow(
        dim=system.dim(),
        num_layers=train_params["num_layers"],
        hidden_features=train_params["hidden_features"],
    )
    transition_model = ConditionalNormalizingFlow(
        dim=system.dim(),
        conditioner_dim=system.dim(),
        num_layers=train_params["num_layers"],
        hidden_features=train_params["hidden_features"],
    )
    observation_model = ConditionalNormalizingFlow(
        dim=system.observation_dim(),
        conditioner_dim=system.dim(),
        num_layers=train_params["num_layers"],
        hidden_features=train_params["hidden_features"],
    )

    labeled_mle = {"mle": loss.mle_loss}
    labeled_cond_mle = {"mle": loss.conditional_mle_loss}

    print("Training initial flow model")
    init_model = init_model.to(device)
    init_opt = torch.optim.Adam(init_model.parameters(), lr=train_params["lr"])
    init_model, best_loss_init, training_time_init = rf_train.train(
        init_model,
        init_loader,
        labeled_mle,
        init_opt,
        epochs=train_params["init_epochs"],
        verbose=True,
        use_best="mle",
    )
    print("Training transition flow model")
    transition_model = transition_model.to(device)
    tran_opt = torch.optim.Adam(transition_model.parameters(), lr=train_params["lr"])
    transition_model, best_loss_tran, training_time_tran = rf_train.train(
        transition_model,
        tran_loader,
        labeled_cond_mle,
        tran_opt,
        epochs=train_params["tran_epochs"],
        verbose=True,
        use_best="mle",
    )
    print("Training observation flow model")
    observation_model = observation_model.to(device)
    obs_opt = torch.optim.Adam(observation_model.parameters(), lr=train_params["lr"])
    observation_model, best_loss_obs, training_time_obs = rf_train.train(
        observation_model,
        obs_loader,
        labeled_cond_mle,
        obs_opt,
        epochs=train_params["obs_epochs"],
        verbose=True,
        use_best="mle",
    )
    print(
        f"Initial model loss: {best_loss_init:.4f}, time: {training_time_init:.2f}s | "
        f"Transition: {best_loss_tran:.4f}, time: {training_time_tran:.2f}s | "
        f"Observation: {best_loss_obs:.4f}, time: {training_time_obs:.2f}s"
    )

    init_model.eval()
    transition_model.eval()
    observation_model.eval()

    true_states, observations = simulate(system, init_state_sampler, n_timesteps=n_timesteps, device=device)

    with torch.no_grad():
        init_particles = init_model.sample(n_particles).to(device)

    flow_pf_initial_belief = WeightedParticleSet(
        particles=init_particles,
        weights=torch.ones(n_particles, device=device) / n_particles,
    )
    flow_pf_priors, flow_pf_posteriors = propagate_and_update(
        belief=flow_pf_initial_belief,
        transition_model=transition_model,
        observation_model=observation_model,
        observations=[obs for obs in observations],
    )
    print(f"NF particle filter steps: {len(flow_pf_priors)} priors, {len(flow_pf_posteriors)} posteriors")

    truth_transition = SystemTransitionDistribution(system).to(device=device)
    truth_observation = SystemObservationDistribution(system).to(device=device)
    truth_initial_belief = WeightedParticleSet(
        particles=init_state_sampler(n_particles).to(device=device),
        weights=torch.ones(n_particles, device=device) / n_particles,
    )
    _, truth_posteriors = propagate_and_update(
        belief=truth_initial_belief,
        transition_model=truth_transition,
        observation_model=truth_observation,
        observations=[obs for obs in observations],
    )

    box_lows = tuple(problem.plot_bounds_low.tolist())
    box_highs = tuple(problem.plot_bounds_high.tolist())
    fig, axes = plt.subplots(n_timesteps + 1, 2, figsize=(10, max(3 * (n_timesteps + 1), 15)))
    fig.suptitle("NF-trained particle filter vs truth-model particle filter")

    for k in range(n_timesteps + 1):
        plot_particle_belief(
            axes[k, 0],
            flow_pf_posteriors[k],
            x_range=(box_lows[0], box_highs[0]),
            y_range=(box_lows[1], box_highs[1]),
        )
        plot_particle_belief(
            axes[k, 1],
            truth_posteriors[k],
            x_range=(box_lows[0], box_highs[0]),
            y_range=(box_lows[1], box_highs[1]),
        )

        axes[k, 0].set_title(f"NF PF posterior k={k}", fontsize=8)
        axes[k, 1].set_title(f"Truth PF posterior k={k}", fontsize=8)
        axes[k, 0].scatter(true_states[k, 0].item(), true_states[k, 1].item(), marker="o", s=10, c="red")
        axes[k, 1].scatter(true_states[k, 0].item(), true_states[k, 1].item(), marker="o", s=10, c="red")

        if k > 0:
            axes[k, 0].scatter(observations[k - 1, 0].item(), observations[k - 1, 1].item(), marker="o", s=10, c="blue")
            axes[k, 1].scatter(observations[k - 1, 0].item(), observations[k - 1, 1].item(), marker="o", s=10, c="blue")

        for j in range(2):
            axes[k, j].grid(alpha=0.25)
            axes[k, j].set_xlabel("")
            axes[k, j].set_ylabel("")
            axes[k, j].tick_params(axis="both", which="both", labelbottom=False, labelleft=False, bottom=False, left=False)

    fig.tight_layout()
    plt.savefig("figures/vdp_nf_particle_posteriors.jpg", dpi=300)


if __name__ == "__main__":
    main()
