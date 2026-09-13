from pathlib import Path
from math import ceil, sqrt

import copy

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from normalizing_flow.conditional_base_distributions import ConditionalBernstein1D, ConditionalBSpline1D
from rational_factor.models.composite_model import CompositeConditionalModel, CompositeDensityModel
from rational_factor.models.domain_transformation import ErfSeparableTF, IdentityTF, MaskedRQSNFTF, MLP, StackedTF
from rational_factor.models.factor_forms import SumProdRFF, LinearFF
from rational_factor.models.mutual_bases import (
    LocalBSplineMutualBasis,
    NormalizedProductPairBasis,
    PositiveMaskedGramMutualBasis,
)
from rational_factor.models.parameters import (
    PositiveParameters,
    QuasiseparableFactorization,
    DenseMatrixFactorization,
    param_group_iter,
)
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood, check_pdf_valid
from rational_factor.tools.visualization import plot_belief
from rational_factor.models.kde import GaussianKDE
import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.tools.propagate as propagate
from nflows.transforms import CompositeTransform


def _plot_conditional_slices_model_vs_data(
    tran_model: CompositeConditionalModel,
    x_k_data: torch.Tensor,
    x_kp1_data: torch.Tensor,
    out_path: Path,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    n_random_points: int = 10,
    n_grid: int = 120,
    min_points_per_slice: int = 40,
    bin_width: float = 0.4,
    random_seed: int = 0,
    title: str = "",
) -> None:
    """Compare trained p(x'|x_i) to binned empirical samples and a conditional KDE."""
    if x_k_data.shape[1] != 2 or x_kp1_data.shape[1] != 2:
        raise ValueError("_plot_conditional_slices_model_vs_data expects 2D state data")

    param = next(iter(tran_model.parameters()), None)
    buffer = next(iter(tran_model.buffers()), None)
    dev = (
        param.device
        if param is not None
        else (buffer.device if buffer is not None else torch.device("cpu"))
    )
    dt = (
        param.dtype
        if param is not None
        else (buffer.dtype if buffer is not None else torch.float32)
    )

    xk_cpu = x_k_data.detach().cpu()
    xkp1_cpu = x_kp1_data.detach().cpu()
    lo = torch.tensor([x_range[0], y_range[0]], dtype=xk_cpu.dtype)
    hi = torch.tensor([x_range[1], y_range[1]], dtype=xk_cpu.dtype)

    x_lin = torch.linspace(float(lo[0]), float(hi[0]), n_grid)
    y_lin = torch.linspace(float(lo[1]), float(hi[1]), n_grid)
    X, Y = torch.meshgrid(x_lin, y_lin, indexing="xy")
    xp_grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).to(device=dev, dtype=dt)

    n_data = xk_cpu.shape[0]
    if n_data == 0:
        raise ValueError("x_k_data is empty")
    n_random_points = max(1, min(n_random_points, n_data))
    rng = torch.Generator(device="cpu")
    rng.manual_seed(random_seed)
    centers = xk_cpu[torch.randperm(n_data, generator=rng)[:n_random_points]]
    half_width = torch.full((2,), 0.5 * float(bin_width), dtype=xk_cpu.dtype)

    fig, axes = plt.subplots(
        n_random_points, 3, figsize=(15, 3.2 * n_random_points), squeeze=False
    )
    cmap = "viridis"
    eps = torch.finfo(dt).eps

    xk_dev = xk_cpu.to(device=dev, dtype=dt)
    xkp1_dev = xkp1_cpu.to(device=dev, dtype=dt)
    bw_x = GaussianKDE.scott_bandwidth(xk_dev).clamp_min(eps)
    bw_xp = GaussianKDE.scott_bandwidth(xkp1_dev).clamp_min(eps)

    with torch.no_grad():
        tran_model.eval()
        for row, center in enumerate(centers):
            lo_box = center - half_width
            hi_box = center + half_width
            mask = (
                (xk_cpu[:, 0] >= lo_box[0])
                & (xk_cpu[:, 0] < hi_box[0])
                & (xk_cpu[:, 1] >= lo_box[1])
                & (xk_cpu[:, 1] < hi_box[1])
            )
            xp_slice = xkp1_cpu[mask]

            ax_emp = axes[row, 0]
            if xp_slice.shape[0] >= min_points_per_slice:
                ax_emp.scatter(
                    xp_slice[:, 0].numpy(),
                    xp_slice[:, 1].numpy(),
                    s=5,
                    alpha=1.0,
                    c="orange",
                    edgecolors="none",
                    rasterized=True,
                )
            else:
                ax_emp.text(
                    0.5,
                    0.5,
                    f"Too few samples\nn={xp_slice.shape[0]}",
                    ha="center",
                    va="center",
                    transform=ax_emp.transAxes,
                )
            ax_emp.set_title(
                "empirical samples x' | x in bin\n"
                f"n={xp_slice.shape[0]}, "
                f"x=({float(center[0]):.3f}, {float(center[1]):.3f})"
            )
            ax_emp.set_xlim(float(lo[0]), float(hi[0]))
            ax_emp.set_ylim(float(lo[1]), float(hi[1]))
            ax_emp.set_aspect("equal")
            ax_emp.set_ylabel("x'_2")

            center_dev = center.to(device=dev, dtype=dt)
            cond = center_dev.unsqueeze(0).expand(xp_grid.shape[0], -1)
            model_pdf = (
                tran_model.log_density(xp_grid, conditioner=cond)
                .exp()
                .reshape(n_grid, n_grid)
                .detach()
                .cpu()
                .numpy()
            )
            ax_model = axes[row, 1]
            cf = ax_model.contourf(X.numpy(), Y.numpy(), model_pdf, levels=40, cmap=cmap)
            fig.colorbar(cf, ax=ax_model, fraction=0.046, pad=0.04)
            ax_model.set_title(
                "model p(x'|x_i)\n"
                f"bin_w=({float(2 * half_width[0]):.3f}, {float(2 * half_width[1]):.3f})"
            )
            ax_model.set_xlim(float(lo[0]), float(hi[0]))
            ax_model.set_ylim(float(lo[1]), float(hi[1]))
            ax_model.set_aspect("equal")
            if xp_slice.shape[0] > 0:
                ax_model.scatter(
                    xp_slice[:, 0].numpy(),
                    xp_slice[:, 1].numpy(),
                    s=5,
                    alpha=1.0,
                    c="orange",
                    edgecolors="none",
                    rasterized=True,
                )

            # Conditional KDE at fixed x_i: w(x) ∝ N(x_i, h_x^2), then mix N(x'_k, h_xp^2).
            diff_x = xk_dev - center_dev.unsqueeze(0)
            w_x = torch.exp(-0.5 * diff_x.square().sum(dim=1) / (bw_x * bw_x))
            w_sum = w_x.sum().clamp_min(eps)
            kde_vals = []
            block = 1024
            for start in range(0, xp_grid.shape[0], block):
                end = min(start + block, xp_grid.shape[0])
                xp_blk = xp_grid[start:end]
                diff_xp = xp_blk[:, None, :] - xkp1_dev[None, :, :]
                k_xp = torch.exp(-0.5 * diff_xp.square().sum(dim=2) / (bw_xp * bw_xp))
                kde_vals.append((k_xp * w_x.unsqueeze(0)).sum(dim=1) / w_sum)
            kde_pdf = (
                torch.cat(kde_vals, dim=0).reshape(n_grid, n_grid).detach().cpu().numpy()
            )

            ax_kde = axes[row, 2]
            cf_kde = ax_kde.contourf(X.numpy(), Y.numpy(), kde_pdf, levels=40, cmap=cmap)
            fig.colorbar(cf_kde, ax=ax_kde, fraction=0.046, pad=0.04)
            ax_kde.set_title("conditional KDE p(x'|x_i)")
            ax_kde.set_xlim(float(lo[0]), float(hi[0]))
            ax_kde.set_ylim(float(lo[1]), float(hi[1]))
            ax_kde.set_aspect("equal")
            if xp_slice.shape[0] > 0:
                ax_kde.scatter(
                    xp_slice[:, 0].numpy(),
                    xp_slice[:, 1].numpy(),
                    s=5,
                    alpha=1.0,
                    c="orange",
                    edgecolors="none",
                    rasterized=True,
                )

    for col in range(3):
        axes[-1, col].set_xlabel("x'_1")
    if title:
        fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_separable_phi_psi_grid(
    pair_basis,
    out_path: Path,
    *,
    title: str,
    n_grid: int = 256,
) -> None:
    """Plot every 1D phi/psi factor of a separable pair on a square subplot grid.

    Domain is the unit interval (post-wrap coordinates). Subplots are filled in
    index order left-to-right, top-to-bottom; unused cells are hidden.
    """
    dtype, device = pair_basis.dtype_device()
    dim = pair_basis.dim()
    if dim != 1:
        raise ValueError(
            f"_plot_separable_phi_psi_grid expects a 1D pair basis, got dim={dim}"
        )

    x = torch.linspace(0.0, 1.0, n_grid, device=device, dtype=dtype).reshape(-1, 1)
    with torch.no_grad():
        if isinstance(pair_basis, torch.nn.Module):
            torch.nn.Module.eval(pair_basis)
        phi = pair_basis.eval(x, 0).detach().cpu()
        psi = pair_basis.eval(x, 1).detach().cpu()

    while phi.ndim > 2 and phi.shape[0] == 1:
        phi = phi.squeeze(0)
    while psi.ndim > 2 and psi.shape[0] == 1:
        psi = psi.squeeze(0)
    if phi.ndim != 2 or psi.ndim != 2:
        raise ValueError(
            f"Expected phi/psi shapes (n_grid, n_basis), got "
            f"{tuple(phi.shape)} and {tuple(psi.shape)}"
        )

    x_np = x.squeeze(-1).detach().cpu().numpy()
    phi_np = phi.numpy()
    psi_np = psi.numpy()
    n_basis = phi_np.shape[-1]
    if psi_np.shape[-1] != n_basis:
        raise ValueError(
            f"phi/psi n_basis mismatch: {phi_np.shape[-1]} vs {psi_np.shape[-1]}"
        )

    n_cols = max(1, ceil(sqrt(n_basis)))
    n_rows = max(1, ceil(n_basis / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.35 * n_cols, 1.15 * n_rows),
        sharex=True,
        squeeze=False,
    )
    if title:
        fig.suptitle(title, y=1.01)

    for i in range(n_basis):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        ax.plot(x_np, phi_np[:, i], color="C0", lw=1.0, label="phi" if i == 0 else None)
        ax.plot(x_np, psi_np[:, i], color="C1", lw=1.0, label="psi" if i == 0 else None)
        ax.set_title(str(i), fontsize=7, pad=1)
        ax.set_xlim(0.0, 1.0)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    for j in range(n_basis, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _make_qs_B(n_basis: int, order: int, device: torch.device) -> QuasiseparableFactorization:
    """Order-``order`` quasiseparable ``B = L D U`` with nonnegative factors.

    Zero LDU generators are a critical point (off-diagonal grads vanish), and
    signed generators make ``B`` indefinite — both break SumProdRFF. Use small
    positive random generators; ``transition_bound`` keeps ``a, b ∈ (0, 1)``.
    """
    gen_shape = (1, n_basis, order)
    diag_shape = (1, n_basis)
    pos_off = lambda: PositiveParameters.random_init(gen_shape, mean=-2.0, std=0.3, epsilon=1e-4).to(device)
    pos_trans = lambda: PositiveParameters.random_init(gen_shape, mean=0.0, std=0.3, epsilon=1e-4).to(device)
    return QuasiseparableFactorization(
        pos_off(),
        pos_trans(),
        pos_off(),
        PositiveParameters.random_init(diag_shape, mean=1.0, std=0.1, epsilon=1e-4).to(device),
        pos_off(),
        pos_trans(),
        pos_off(),
        transition_bound=0.99,
    )


if __name__ == "__main__":
    problem = FULLY_OBSERVABLE_PROBLEMS["van_der_pol"]

    ###
    use_gpu = torch.cuda.is_available()
    n_basis = 50
    sacrificial_index = 0
    embedding_dim = 10
    k_alpha = 3
    k_beta = 5
    trainable_beta = False
    B_order = 10
    flow_hidden = 8
    flow_layers = 2
    tran_params = {
        "n_epochs_per_group": [3, 5, 3],  # domain_tf+wrap, embedding+base_mlp, weights
        "iterations": 10,
        "lr_domain_tf": 1e-3,
        "lr_base": 1e-2,
        "lr_weights": 5e-2,
        "lr_wrap": 1e-3,
    }
    init_params = {
        "n_epochs_per_group": [10],  # h0 coeffs only
        "iterations": 10,
        "lr_weights": 1e-2,
    }

    batch_size = 128
    n_timesteps_prop = problem.n_timesteps
    ###

    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using GPU: ", use_gpu)
    print("Device: ", device)

    system = problem.system
    dim = system.dim()
    if dim != 2:
        raise ValueError(f"VDP script expects a 2D system, got dim={dim}")

    x0 = problem.train_initial_state_data()
    x_k, x_kp1 = problem.train_state_transition_data()
    traj_data = problem.test_data()

    x0_dataloader = DataLoader(TensorDataset(x0), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(TensorDataset(x_kp1, x_k), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    rest_dim = dim - 1

    # Local B-spline pair on the sacrificial coordinate (unit interval after Erf wrap).
    masking = LocalBSplineMutualBasis(
        n_basis=n_basis,
        k_alpha=k_alpha,
        k_beta=k_beta,
        trainable_beta=trainable_beta,
        device=device,
    ).to(device)


    embedding = torch.nn.Embedding(n_basis, embedding_dim).to(device)
    domain_tf = MaskedRQSNFTF(
        dim=rest_dim,
        context_features=embedding_dim,
        n_layers=flow_layers,
        hidden_features=flow_hidden,
        tails=None,  # free coords are already on the unit box after Erf wrap
        num_bins=4
    ).to(device)

    b_spline_params = 15
    base_mlp = MLP(
        in_features=embedding_dim,
        out_features=b_spline_params,
        hidden_features=flow_hidden,
        zero_init_last=False,
    ).to(device)
    base = ConditionalBSpline1D(
        dim=rest_dim,
        conditioner_dim=embedding_dim,
        n_basis=b_spline_params,
        mlp=base_mlp,
        degree=3,
    ).to(device)
    #bernstein_deg = 50
    #base_mlp = MLP(
    #    in_features=embedding_dim,
    #    out_features=bernstein_deg + 1,
    #    hidden_features=flow_hidden,
    #    zero_init_last=False,
    #).to(device)
    #base = ConditionalBernstein1D(
    #    dim=rest_dim,
    #    conditioner_dim=embedding_dim,
    #    degree=bernstein_deg,
    #    mlp=base_mlp,
    #).to(device)

    free_basis = NormalizedProductPairBasis(base, domain_tf, embedding).to(device)


    phi_psi_mutual = PositiveMaskedGramMutualBasis(
        masking,
        sacrificial_index,
        free_basis,
        swap_alpha_beta=True,
    ).to(device)

    g_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0]), epsilon=10.0
    ).to(device)
    h0_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0])
    ).to(device)

    #B = _make_qs_B(n_basis, B_order, device)
    B_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis, n_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0]), epsilon=10.0).to(device)
    B = DenseMatrixFactorization(B_coeffs)

    g_basis = phi_psi_mutual.get_basis(0, coeffs=g_coeffs)
    psi_basis = phi_psi_mutual.get_basis(1)

    wrap_tf = ErfSeparableTF.from_data(x_k, trainable=True).to(device)
    rff = SumProdRFF(g_basis, psi_basis, B, numerical_tolerance=problem.numerical_tolerance)
    tran_model = CompositeConditionalModel([wrap_tf], rff).to(device)

    print("Training transition model")
    mle_loss_fn = loss.conditional_mle_loss
    optimizers = {
        "domain_tf": torch.optim.Adam(
            [
                {"params": domain_tf.parameters(), "lr": tran_params["lr_domain_tf"]},
                {"params": wrap_tf.parameters(), "lr": tran_params["lr_wrap"]},
            ]
        ),
        "base": torch.optim.Adam(
            [
                {"params": embedding.parameters(), "lr": tran_params["lr_base"]},
                {"params": base_mlp.parameters(), "lr": tran_params["lr_base"]},
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

    # Freeze the shared pair, g coeffs, and wrap; reuse psi for h0
    for p in phi_psi_mutual.parameters():
        p.requires_grad_(False)
    g_coeffs.set_requires_grad(False)
    #trained_wrap_tf = wrap_tf.
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

    output_dir = Path("figures/masked_gram/vdp_lbs")
    output_dir.mkdir(parents=True, exist_ok=True)

    masking_basis_out = output_dir / "masking_basis_phi_psi.png"
    _plot_separable_phi_psi_grid(
        masking,
        masking_basis_out,
        title="VDP LBS: masking basis (sacrificial dim) — phi / psi by index",
    )
    print(f"Saved masking basis grid to {masking_basis_out}")

    free_basis_out = output_dir / "free_basis_phi_psi.png"
    _plot_separable_phi_psi_grid(
        free_basis,
        free_basis_out,
        title="VDP LBS: free basis (rest dim) — phi / psi by index",
    )
    print(f"Saved free basis grid to {free_basis_out}")

    box_lows = tuple(problem.plot_bounds_low.tolist())
    box_highs = tuple(problem.plot_bounds_high.tolist())

    cond_slice_out_path = output_dir / "conditional_slices_model_vs_data.png"
    _plot_conditional_slices_model_vs_data(
        tran_model,
        x_k,
        x_kp1,
        cond_slice_out_path,
        x_range=(box_lows[0], box_highs[0]),
        y_range=(box_lows[1], box_highs[1]),
        n_random_points=10,
        n_grid=120,
        min_points_per_slice=20,
        bin_width=0.4,
        random_seed=0,
        title="VDP LBS: random conditional bins — empirical vs model vs KDE",
    )
    print(f"Saved conditional slice comparison to {cond_slice_out_path}")

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

    n_plot = min(n_timesteps_prop, len(belief_seq))
    fig, axes = plt.subplots(2, n_plot, figsize=(3.2 * n_plot, 6.5), squeeze=False)
    fig.suptitle("VDP LBS: beliefs at each time step")
    for i in range(n_plot):
        plot_belief(
            axes[1, i],
            belief_seq[i],
            x_range=(box_lows[0], box_highs[0]),
            y_range=(box_lows[1], box_highs[1]),
        )
        axes[0, i].scatter(traj_data[i][:, 0], traj_data[i][:, 1], s=1)
        axes[0, i].set_aspect("equal")
        axes[0, i].set_xlim(box_lows[0], box_highs[0])
        axes[0, i].set_ylim(box_lows[1], box_highs[1])
        axes[0, i].set_title(f"t = {i}")

    beliefs_out_path = output_dir / "beliefs.png"
    fig.tight_layout()
    fig.savefig(beliefs_out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved beliefs to {beliefs_out_path}")
