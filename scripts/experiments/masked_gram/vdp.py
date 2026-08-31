import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from normalizing_flow.base_distributions import SeparableBeta
from normalizing_flow.vp_flow import VolumePreservingFlow
from rational_factor.models.domain_transformation import MLP
from rational_factor.models.factor_forms import LinearRFF, LinearFF
from rational_factor.models.mutual_bases import (
    Orthogonal1DPWCBasis,
    PositiveMaskedGramMutualBasis,
    VolumePreservingPairBasis,
)
from rational_factor.models.parameters import TrainableParameters, PositiveParameters
from rational_factor.models.qs_matrix import Order1QuasiseparableFactorization
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood, check_pdf_valid
from rational_factor.tools.visualization import plot_belief
import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.tools.propagate as propagate
import matplotlib.pyplot as plt


def _make_qs_factorization(n_basis: int, device: torch.device) -> Order1QuasiseparableFactorization:
    shape = (1, n_basis)
    return Order1QuasiseparableFactorization(
        TrainableParameters.random_init(shape=shape).to(device),
        TrainableParameters.random_init(shape=shape).to(device),
        TrainableParameters.random_init(shape=shape).to(device),
        TrainableParameters.random_init(shape=shape).to(device),
        TrainableParameters.random_init(shape=shape).to(device),
        TrainableParameters.random_init(shape=shape).to(device),
        TrainableParameters.random_init(shape=shape).to(device),
        transition_bound=0.99,
    )


if __name__ == "__main__":
    problem = FULLY_OBSERVABLE_PROBLEMS["van_der_pol"]

    ###
    use_gpu = torch.cuda.is_available()
    n_basis = 200
    sacrificial_index = 0
    conditioner_dim = 8
    flow_n_steps = 4
    flow_hidden = 16
    flow_layers = 2
    tran_params = {
        "n_epochs_per_group": [5, 5],  # basis, weights
        "iterations": 100,
        "lr_basis": 5e-3,
        "lr_weights": 1e-2,
    }
    init_params = {
        "n_epochs_per_group": [20],  # h0 coeffs only
        "iterations": 100,
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
    x0 = problem.train_initial_state_data()
    x_k, x_kp1 = problem.train_state_transition_data()
    traj_data = problem.test_data()

    x0_dataloader = DataLoader(TensorDataset(x0), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(TensorDataset(x_kp1, x_k), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    rest_dim = dim - 1

    # Volume-preserving pair on the non-sacrificial coordinates
    vp_base_dist = SeparableBeta(dim=rest_dim, alpha=2.0, beta=2.0).to(device)
    vp_flow = VolumePreservingFlow(
        dim=rest_dim,
        conditioner_dim=conditioner_dim,
        n_steps=flow_n_steps,
        hidden_features=flow_hidden,
        num_hidden_layers=flow_layers,
        zero_init=True,
    ).to(device)
    vp_embedding = torch.nn.Embedding(n_basis, conditioner_dim).to(device)
    vp_splitter = MLP(
        in_features=rest_dim,
        out_features=n_basis,
        hidden_features=flow_hidden,
        num_hidden_layers=flow_layers,
        zero_init_last=True,
    ).to(device)
    vp_mutual = VolumePreservingPairBasis(vp_flow, vp_base_dist, vp_splitter, vp_embedding)

    # Orthogonal 1D PWC pair on the sacrificial coordinate
    qs_factorization = _make_qs_factorization(n_basis, device)
    gram_diag_params = PositiveParameters.random_init(
        shape=(1, n_basis), mean=torch.tensor([1.0]), std=torch.tensor([0.5]), epsilon=1e-3
    ).to(device)
    orth_pwc_mutual = Orthogonal1DPWCBasis(qs_factorization, gram_diag_params=gram_diag_params)

    phi_psi_mutual = PositiveMaskedGramMutualBasis(
        orth_pwc_mutual,
        vp_mutual,
        sacrificial_index=sacrificial_index,
        domain_lows=problem.plot_bounds_low,
        domain_highs=problem.plot_bounds_high,
    ).to(device)

    g_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0]), epsilon=10.0
    ).to(device)
    h0_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0])
    ).to(device)

    g_basis = phi_psi_mutual.get_basis(0, coeffs=g_coeffs)
    psi_basis = phi_psi_mutual.get_basis(1)

    # Create and train the transition model
    tran_model = LinearRFF(g_basis, psi_basis)
    print("Training transition model")
    mle_loss_fn = loss.conditional_mle_loss

    rff_basis_params = phi_psi_mutual.parameters()
    rff_weight_params = g_coeffs.parameters()
    optimizers = {
        "basis": torch.optim.Adam(rff_basis_params, lr=tran_params["lr_basis"]),
        "weights": torch.optim.Adam(rff_weight_params, lr=tran_params["lr_weights"]),
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

    # Freeze the shared pair (g and psi shapes) and g coeffs; reuse psi for h0
    for p in phi_psi_mutual.parameters():
        p.requires_grad_(False)
    g_coeffs.set_requires_grad(False)

    h0_basis = phi_psi_mutual.get_basis(1, coeffs=h0_coeffs)
    init_model = LinearFF.from_rff(tran_model, h0_basis).to(device)

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

    print(f"Transition model loss: {best_loss_tran:.4f}, training time: {training_time_tran:.2f} seconds")
    print(f"Initial model loss: {best_loss_init:.4f}, training time: {training_time_init:.2f} seconds")

    analysis_device = device
    init_model = init_model.to(analysis_device).eval()
    tran_model = tran_model.to(analysis_device).eval()

    box_lows = tuple(problem.plot_bounds_low.tolist())
    box_highs = tuple(problem.plot_bounds_high.tolist())

    belief_seq = [
        belief.to(analysis_device).eval()
        for belief in propagate.propagate(init_model, tran_model, n_steps=n_timesteps_prop)
    ]

    ll_per_step = []
    for i in range(n_timesteps_prop):
        data_i = traj_data[i].to(analysis_device)
        ll = avg_log_likelihood(belief_seq[i], data_i)
        ll_per_step.append(float(ll.detach().cpu()))
        print(f"Avg log-likelihood at time {i}: {ll_per_step[-1]:.6f}")

    fig, axes = plt.subplots(2, n_timesteps_prop, figsize=(20, 10))
    fig.suptitle("Beliefs at each time step")
    for i in range(n_timesteps_prop):
        check_pdf_valid(belief_seq[i], (box_lows, box_highs), device=analysis_device)
        plot_belief(axes[1, i], belief_seq[i], x_range=(box_lows[0], box_highs[0]), y_range=(box_lows[1], box_highs[1]))
        axes[0, i].scatter(traj_data[i][:, 0], traj_data[i][:, 1], s=1)
        axes[0, i].set_aspect("equal")
        axes[0, i].set_xlim(box_lows[0], box_highs[0])
        axes[0, i].set_ylim(box_lows[1], box_highs[1])

    output_dir = Path("figures/masked_gram/vdp")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "beliefs.png", dpi=1000)
    print(f"Saved beliefs to {output_dir / 'beliefs.png'}")
