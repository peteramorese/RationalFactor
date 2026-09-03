from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from torch.utils.data import DataLoader, TensorDataset

from normalizing_flow.vp_flow import VolumePreservingFlow
from rational_factor.models.basis_functions import BetaBasis
from rational_factor.models.composite_model import CompositeConditionalModel, CompositeDensityModel
from rational_factor.models.domain_transformation import ErfSeparableTF, MLP
from rational_factor.models.factor_forms import SumProdRFF, LinearFF
from rational_factor.models.mutual_bases import (
    Orthogonal1DPWCBasis,
    PositiveMaskedGramMutualBasis,
    VolumePreservingPairBasis,
)
from rational_factor.models.parameters import (
    Order1QuasiseparableFactorization,
    PositiveParameters,
    Rank1PlusDiagonalFactorization,
    SequentialRank1PlusDiagonalFactorization,
    TrainableParameters,
    param_group_iter,
)
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood, check_pdf_valid
import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.tools.propagate as propagate


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


def _plot_snd_conditional_true_vs_learned(
    tran_model: CompositeConditionalModel,
    system,
    out_path: Path,
    x_cond_train: torch.Tensor,
    xp_train: torch.Tensor,
    *,
    problem,
    n_grid: int = 200,
    title: str = "",
) -> None:
    """Compare true and learned p(x'|x) on a 2D grid (conditioner vs next state)."""
    lo = float(problem.plot_bounds_low[0].item())
    hi = float(problem.plot_bounds_high[0].item())

    param = next(iter(tran_model.parameters()), None)
    buffer = next(iter(tran_model.buffers()), None)
    dev = param.device if param is not None else (buffer.device if buffer is not None else torch.device("cpu"))
    dt = param.dtype if param is not None else (buffer.dtype if buffer is not None else torch.float32)

    nx = n_grid
    nxp = n_grid
    x_1d = torch.linspace(lo, hi, nx, device=dev, dtype=dt)
    xp_1d = torch.linspace(lo, hi, nxp, device=dev, dtype=dt)
    x_cond = x_1d.unsqueeze(1).expand(nx, nxp).reshape(-1, 1)
    xp_flat = xp_1d.unsqueeze(0).expand(nx, nxp).reshape(-1, 1)

    with torch.no_grad():
        tran_model.eval()
        log_learned = tran_model.log_density(xp_flat, conditioner=x_cond)
        learned = log_learned.exp().detach().cpu().numpy().reshape(nx, nxp)

    log_true = system.log_transition_density(
        x_cond.detach().cpu().float(),
        xp_flat.detach().cpu().float(),
    )
    true_pdf = log_true.exp().detach().numpy().reshape(nx, nxp)

    x_cond_np = x_cond_train[:, 0].detach().cpu().numpy()
    xp_train_np = xp_train[:, 0].detach().cpu().numpy()
    x_np = x_1d.detach().cpu().numpy()
    xp_np = xp_1d.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), squeeze=False)
    cmap = "viridis"
    for ax, Z, subt in zip(
        axes[0, :2],
        [true_pdf.T, learned.T],
        ["true p(x'|x)", "learned p(x'|x)"],
    ):
        zmax = float(np.max(Z))
        if zmax <= 0:
            zmax = 1.0
        norm = Normalize(vmin=0.0, vmax=zmax)
        levels = np.linspace(0.0, zmax, 51)
        cf = ax.contourf(x_np, xp_np, Z, levels=levels, cmap=cmap, norm=norm)
        ax.set_aspect("auto")
        ax.set_xlabel("x (conditioner)")
        ax.set_ylabel("x'")
        ax.set_title(subt)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

    ax_scatter = axes[0, 2]
    ax_scatter.scatter(
        x_cond_np,
        xp_train_np,
        s=4,
        alpha=0.25,
        c="C1",
        edgecolors="none",
        rasterized=True,
    )
    ax_scatter.set_xlim(lo, hi)
    ax_scatter.set_ylim(lo, hi)
    ax_scatter.set_aspect("auto")
    ax_scatter.set_xlabel("x (conditioner)")
    ax_scatter.set_ylabel("x'")
    ax_scatter.set_title("training samples scatter")
    ax_scatter.grid(True, alpha=0.2)

    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_snd_1d_belief_trajectory(
    problem,
    beliefs: list[CompositeDensityModel],
    test_traj_data: list[torch.Tensor],
    out_path: Path,
    *,
    n_grid: int = 400,
    title: str = "",
) -> None:
    """Plot propagated 1D beliefs against test-trajectory histograms for each timestep."""
    n_slices = problem.n_timesteps + 1
    if len(beliefs) != n_slices:
        raise ValueError(f"Expected {n_slices} beliefs, got {len(beliefs)}")

    lo = float(problem.plot_bounds_low[0].item())
    hi = float(problem.plot_bounds_high[0].item())
    x_grid = torch.linspace(lo, hi, n_grid).unsqueeze(1)

    param = next(iter(beliefs[0].parameters()), None)
    buffer = next(iter(beliefs[0].buffers()), None)
    dev = param.device if param is not None else (buffer.device if buffer is not None else torch.device("cpu"))
    dt = param.dtype if param is not None else (buffer.dtype if buffer is not None else torch.float32)
    x_grid = x_grid.to(device=dev, dtype=dt)

    fig, axes = plt.subplots(n_slices, 1, figsize=(9, 2.4 * n_slices), squeeze=False)
    for t in range(n_slices):
        ax = axes[t, 0]
        samples = test_traj_data[t][:, 0].detach().cpu().numpy()
        with torch.no_grad():
            beliefs[t].eval()
            pdf = beliefs[t](x_grid).squeeze(-1).detach().cpu().numpy()
        x_np = x_grid.squeeze(-1).detach().cpu().numpy()
        ax.plot(x_np, pdf, color="C0", lw=2.0, label="propagated belief")
        ax.hist(samples, bins=50, density=True, alpha=0.35, color="C1", label="test traj. (hist)")
        ax.set_xlim(lo, hi)
        ax.set_ylabel("density")
        ax.set_title(f"t = {t}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1, 0].set_xlabel(problem.system.state_label(0))
    if title:
        fig.suptitle(title, y=1.002)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    problem = FULLY_OBSERVABLE_PROBLEMS["scalar_nonlinear_drift"]

    ###
    use_gpu = torch.cuda.is_available()
    n_basis = 200
    sacrificial_index = 0
    embedding_dim = 4
    splitter_hidden = 16
    splitter_layers = 2
    use_vp_flow = False
    flow_n_steps = 4
    flow_hidden = 16
    flow_layers = 2
    B_rank = 10
    P_rank = 10
    tran_params = {
        "n_epochs_per_group": [5, 5],  # basis+wrap, weights
        "iterations": 4,
        "lr_basis": 1e-2,
        "lr_weights": 1e-2,
        "lr_wrap": 1e-3,
    }
    init_params = {
        "n_epochs_per_group": [20],  # h0 coeffs only
        "iterations": 30,
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
    if dim != 1:
        raise ValueError(f"SND script expects a 1D system, got dim={dim}")

    x0 = problem.train_initial_state_data()
    x_k, x_kp1 = problem.train_state_transition_data()
    traj_data = problem.test_data()

    x0_dataloader = DataLoader(TensorDataset(x0), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(TensorDataset(x_kp1, x_k), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    rest_dim = dim - 1

    # Rest pair is 0-d (identically 1) when the only coordinate is sacrificial.
    # Splitter is σ(x, e_i); the VP flow is optional and unused for rest_dim < 2.
    base_dist_alpha = PositiveParameters.set_init((1, rest_dim, n_basis), 0.0, epsilon=1.0).to(device)
    base_dist_beta = PositiveParameters.set_init((1, rest_dim, n_basis), 0.0, epsilon=1.0).to(device)
    vp_base_dist = BetaBasis(
        base_dist_alpha,
        base_dist_beta,
    )
    vp_embedding = torch.nn.Embedding(n_basis, embedding_dim).to(device)
    vp_splitter = MLP(
        in_features=rest_dim + embedding_dim,
        out_features=1,
        hidden_features=splitter_hidden,
        num_hidden_layers=splitter_layers,
        zero_init_last=True,
    ).to(device)
    vp_flow = None
    if use_vp_flow:
        vp_flow = VolumePreservingFlow(
            dim=rest_dim,
            conditioner_dim=embedding_dim,
            n_steps=flow_n_steps,
            hidden_features=flow_hidden,
            num_hidden_layers=flow_layers,
            zero_init=True,
        ).to(device)
    vp_mutual = VolumePreservingPairBasis(vp_base_dist, vp_splitter, vp_embedding, vp_flow)

    # Orthogonal 1D PWC pair on the sacrificial coordinate
    qs_factorization = _make_qs_factorization(n_basis, device)
    gram_diag_params = PositiveParameters.random_init(
        shape=(1, n_basis), mean=torch.tensor([1.0]), std=torch.tensor([0.5]), epsilon=1e-3
    ).to(device)
    orth_pwc_mutual = Orthogonal1DPWCBasis(qs_factorization, gram_diag_params=gram_diag_params)

    phi_psi_mutual = PositiveMaskedGramMutualBasis(
        orth_pwc_mutual,
        sacrificial_index,
        vp_mutual,
    ).to(device)

    g_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0]), epsilon=0.1
    ).to(device)
    h0_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0])
    ).to(device)

    # Shape is (model batch, product sequence, basis).  The product wrapper
    # consumes dimension 1, so SumProdRFF sees one (n_basis, n_basis) matrix.
    B_shape = (1, B_rank, n_basis)
    B_u = PositiveParameters.random_init(shape=B_shape, mean=-5.0, std=0.1, epsilon=1e-6).to(device)
    B_v = PositiveParameters.random_init(shape=B_shape, mean=-5.0, std=0.1, epsilon=1e-6).to(device)
    B_d = PositiveParameters.random_init(shape=B_shape, mean=0.54, std=0.01, epsilon=1e-3).to(device)
    B_factors = Rank1PlusDiagonalFactorization(B_u, B_v, B_d)
    B = SequentialRank1PlusDiagonalFactorization(B_factors, seq_dim=1)

    P_shape = (1, P_rank, n_basis)
    P_u = TrainableParameters.random_init(shape=P_shape, mean=0.0, std=0.1).to(device)
    P_v = TrainableParameters.random_init(shape=P_shape, mean=-5.0, std=0.1).to(device)
    P_factors = Rank1PlusDiagonalFactorization(P_u, P_v, normalization_dim=0)
    P = SequentialRank1PlusDiagonalFactorization(P_factors, seq_dim=1)

    g_basis = phi_psi_mutual.get_basis(0, coeffs=g_coeffs)
    psi_basis = phi_psi_mutual.get_basis(1)

    wrap_tf = ErfSeparableTF.from_data(x_k, trainable=True).to(device)
    rff = SumProdRFF(g_basis, psi_basis, B, P, numerical_tolerance=problem.numerical_tolerance)
    tran_model = CompositeConditionalModel([wrap_tf], rff).to(device)

    print("Training transition model")
    mle_loss_fn = loss.conditional_mle_loss
    optimizers = {
        "basis": torch.optim.Adam(
            [
                {"params": phi_psi_mutual.parameters(), "lr": tran_params["lr_basis"]},
                {"params": wrap_tf.parameters(), "lr": tran_params["lr_wrap"]},
            ]
        ),
        "weights": torch.optim.Adam(
            param_group_iter((g_coeffs, B_u, B_v, B_d, P_u, P_v)),
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

    # Freeze the shared pair, g coeffs, and wrap; reuse psi for h0
    for p in phi_psi_mutual.parameters():
        p.requires_grad_(False)
    g_coeffs.set_requires_grad(False)
    trained_wrap_tf = ErfSeparableTF.copy_from_trainable(wrap_tf).to(device)

    h0_basis = phi_psi_mutual.get_basis(1, coeffs=h0_coeffs)
    init_model = CompositeDensityModel(
        [trained_wrap_tf],
        LinearFF.from_rff(tran_model.conditional_density_model, h0_basis),
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

    print(f"Transition model loss: {best_loss_tran:.4f}, training time: {training_time_tran:.2f} seconds")
    print(f"Initial model loss: {best_loss_init:.4f}, training time: {training_time_init:.2f} seconds")

    analysis_device = device
    init_model = init_model.to(analysis_device).eval()
    tran_model = tran_model.to(analysis_device).eval()
    trained_wrap_tf = trained_wrap_tf.to(analysis_device).eval()

    box_lows = tuple(problem.plot_bounds_low.tolist())
    box_highs = tuple(problem.plot_bounds_high.tolist())
    n_slices = n_timesteps_prop + 1

    base_belief_seq = propagate.propagate(
        init_model.density_model,
        tran_model.conditional_density_model,
        n_steps=n_timesteps_prop,
    )
    belief_seq = [
        CompositeDensityModel([trained_wrap_tf], belief).to(analysis_device).eval()
        for belief in base_belief_seq
    ]

    ll_per_step = []
    for i in range(n_slices):
        data_i = traj_data[i].to(analysis_device)
        ll = avg_log_likelihood(belief_seq[i], data_i)
        ll_per_step.append(float(ll.detach().cpu()))
        print(f"Avg log-likelihood at time {i}: {ll_per_step[-1]:.6f}")
        check_pdf_valid(belief_seq[i], (box_lows, box_highs), device=analysis_device)

    output_dir = Path("figures/masked_gram/snd")
    output_dir.mkdir(parents=True, exist_ok=True)

    beliefs_out_path = output_dir / "beliefs.png"
    _plot_snd_1d_belief_trajectory(
        problem,
        belief_seq,
        traj_data,
        beliefs_out_path,
        n_grid=400,
        title="SND PWC: propagated beliefs vs test trajectories",
    )
    print(f"Saved beliefs to {beliefs_out_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tran_model_cpu = tran_model.cpu().eval()

    cond_out_path = output_dir / "conditional_rff.png"
    _plot_snd_conditional_true_vs_learned(
        tran_model_cpu,
        system,
        cond_out_path,
        x_k,
        x_kp1,
        problem=problem,
        n_grid=200,
        title="SND PWC: true vs learned p(x'|x)",
    )
    print(f"Saved conditional RFF to {cond_out_path}")
