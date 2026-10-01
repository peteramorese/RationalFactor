from pathlib import Path
from math import ceil, sqrt

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.factor_forms import SumProdRFF, LinearFF
from rational_factor.models.kde import GaussianKDE
from rational_factor.models.parameters import (
    PositiveParameters,
    TrainableParameters,
    param_group_iter,
    DenseMatrixFactorization,
)
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood, check_pdf_valid
from rational_factor.tools.visualization import plot_belief
import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.tools.propagate as propagate

from rational_factor.models.gram_preserving_basis import DeepGramPreservingBasis
from rational_factor.models.index_embedding_model import IndexEmbeddingTransform
from normalizing_flow.transforms import Transforms
from rational_factor.models.space_splitter import LatentReflectionSpaceSplitter
from rational_factor.models.mlp import MLP



if __name__ == "__main__":
    problem = FULLY_OBSERVABLE_PROBLEMS["cartpole"]

    ###
    use_gpu = torch.cuda.is_available()
    n_basis = 200
    n_hidden_features = 64
    n_hidden_layers = 5
    n_gp_layers = 3
    embedding_dim = 2
    tran_params = {
        "n_epochs_per_group": [5, 3],
        "iterations": 30,
        "lr_basis": 3e-3,
        "lr_weights": 5e-2,
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

    x0 = problem.train_initial_state_data()
    x_k, x_kp1 = problem.train_state_transition_data()
    traj_data = problem.test_data()

    x0_dataloader = DataLoader(
        TensorDataset(x0), batch_size=batch_size, shuffle=True, pin_memory=use_gpu
    )
    xp_dataloader = DataLoader(
        TensorDataset(x_kp1, x_k), batch_size=batch_size, shuffle=True, pin_memory=use_gpu
    )

    box_lows = problem.plot_bounds_low.to(device=device)
    box_highs = problem.plot_bounds_high.to(device=device)
    plot_x_range = (float(box_lows[0]), float(box_highs[0]))
    plot_y_range = (float(box_lows[1]), float(box_highs[1]))

    t = torch.linspace(0.0, 1.0, n_basis, device=device)
    means = box_lows.unsqueeze(-1) + (box_highs - box_lows).unsqueeze(-1) * t
    means = means.unsqueeze(0)  # (1, dim, n_basis)
    mean_jitter = 0.35
    std_init = 0.35
    phi_means = TrainableParameters.from_values(
        means + mean_jitter * torch.randn_like(means)
    ).to(device)
    phi_stds = PositiveParameters.from_values(
        torch.full((1, dim, n_basis), std_init, device=device)
    ).to(device)
    psi_offset = 0.25 * torch.ones(dim, device=device)
    psi_offset[1::2] = -0.25
    psi_means = TrainableParameters.from_values(
        means
        + psi_offset.view(1, dim, 1)
        + mean_jitter * torch.randn_like(means)
    ).to(device)
    psi_stds = PositiveParameters.from_values(
        torch.full((1, dim, n_basis), std_init + 0.1, device=device)
    ).to(device)
    phi_basis = GaussianBasis(phi_means, phi_stds, coeffs=None)
    psi_basis = GaussianBasis(psi_means, psi_stds, coeffs=None)

    embedding = torch.nn.Embedding(n_gp_layers, embedding_dim).to(device)
    tf = Transforms.make_transform("maf", features=dim, context_features=embedding_dim, init_identity=True).to(device)
    idx_tf = IndexEmbeddingTransform(tf, embedding)
    space_splitter = LatentReflectionSpaceSplitter(idx_tf, reflection_axis=0).to(device)
    deformer = MLP(
        in_features=dim,
        out_features=n_gp_layers,
        hidden_features=n_hidden_features,
        num_hidden_layers=n_hidden_layers,
        zero_init_last=True,
    ).to(device)
    # sigmoid(0)=0.5 puts s midway in [L,U]; with large U that yields strong
    # deformations at init and O(eps) negativity. Bias toward s≈L (near 0).
    with torch.no_grad():
        deformer.net[-1].bias.fill_(-4.0)

    phi_psi_mutual = DeepGramPreservingBasis(
        phi_basis,
        psi_basis,
        space_splitter,
        deformer,
        fixed_base_basis=False,
        eps=1e-4,
    ).to(device)

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

    phi_basis = phi_psi_mutual.get_basis(0)
    psi_basis = phi_psi_mutual.get_basis(1)

    # cartpole default tolerance (1e-20) is too tight once basis values are O(1e-6).
    tran_model = SumProdRFF(g_coeffs, phi_basis, psi_basis, B, numerical_tolerance=1e-10)

    print("Training transition model")
    mle_loss_fn = loss.conditional_mle_loss

    splitter_params = list(phi_psi_mutual._space_splitter.parameters())
    splitter_ids = {id(p) for p in splitter_params}
    deformer_params = [
        p for p in phi_psi_mutual.parameters() if id(p) not in splitter_ids
    ]
    optimizers = {
        "basis": torch.optim.Adam(
            [
                {
                    "params": deformer_params,
                    "lr": tran_params["lr_basis"],
                },
                {
                    "params": splitter_params,
                    "lr": tran_params["lr_basis"] * 0.02,
                },
                {
                    "params": param_group_iter(
                        (phi_means, phi_stds, psi_means, psi_stds)
                    ),
                    "lr": tran_params["lr_basis"],
                },
            ]
        ),
        "weights": torch.optim.Adam(
            param_group_iter((g_coeffs, B_coeffs)), lr=tran_params["lr_weights"]
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
    phi_means.set_requires_grad(False)
    phi_stds.set_requires_grad(False)
    psi_means.set_requires_grad(False)
    psi_stds.set_requires_grad(False)
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

    output_dir = Path("figures/gram_preserving/vdp")
    output_dir.mkdir(parents=True, exist_ok=True)


    box_lows_tuple = tuple(box_lows.tolist())
    box_highs_tuple = tuple(box_highs.tolist())


    n_slices = n_timesteps_prop + 1

    base_belief_seq = propagate.propagate(
        init_model,
        tran_model,
        n_steps=n_timesteps_prop,
    )
    belief_seq = [belief.to(analysis_device).eval() for belief in base_belief_seq]

    ll_per_step = []
    for i in range(n_slices):
        data_i = traj_data[i].to(analysis_device)
        ll = avg_log_likelihood(belief_seq[i], data_i)
        ll_per_step.append(float(ll.detach().cpu()))
        print(f"Avg log-likelihood at time {i}: {ll_per_step[-1]:.6f}")
        #check_pdf_valid(belief_seq[i], (box_lows_tuple, box_highs_tuple), device=analysis_device)

