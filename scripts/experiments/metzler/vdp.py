from pathlib import Path
from math import ceil, sqrt

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from rational_factor.models.basis_functions import BSpline1DBasis
from rational_factor.models.composite_model import CompositeConditionalModel, CompositeDensityModel
from rational_factor.models.density_model import ConditionalDensityModel
from rational_factor.models.domain_transformation import ErfSeparableTF, MaskedRQSNFTF
from rational_factor.models.factor_forms import SumProdRFF, LinearFF
from rational_factor.models.index_embedding_basis import NormalizedIndexEmbeddingBasis
from rational_factor.models.mlp import MetzlerConeMLP
from rational_factor.models.preorthogonal_basis import PreorthogonalMutualBasis
from rational_factor.models.parameters import (
    PositiveParameters,
    DenseMatrixFactorization,
    param_group_iter,
)
from rational_factor.models.structured_matrices import DenseMatrix
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood, check_pdf_valid
from rational_factor.tools.metzler_cone_rays import MetzlerConeRayFinder
from rational_factor.tools.visualization import plot_belief
from rational_factor.models.kde import GaussianKDE
import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.tools.propagate as propagate


class ConditionalRQSNF(ConditionalDensityModel):
    """Conditional RQ-NSF density on ``[0, 1]^d`` (``tails=None``) with Uniform base.

    With Uniform base density 1, ``log p(x | c) = log |det DT(x | c)|``.
    """

    def __init__(
        self,
        dim: int,
        conditioner_dim: int,
        n_layers: int = 2,
        hidden_features: int = 16,
        num_bins: int = 4,
    ):
        super().__init__(dim=dim, conditioner_dim=conditioner_dim)
        self.transform = MaskedRQSNFTF(
            dim=dim,
            context_features=conditioner_dim,
            n_layers=n_layers,
            hidden_features=hidden_features,
            tails=None,
            num_bins=num_bins,
        )

    def log_density(self, x: torch.Tensor, *, conditioner: torch.Tensor, **contexts):
        _, ladj = self.transform.forward(x, context=conditioner)
        return self._clip_log_density(ladj)

    def dtype_device(self):
        p = next(self.parameters())
        return p.dtype, p.device


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
    """Compare trained p(x'|x) to binned empirical samples and a conditional KDE.

    Conditioners share one randomly chosen x1 and span different x2 values so
    rows can be compared for conditional dependence on x2.
    """
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
    half_width = torch.full((2,), 0.5 * float(bin_width), dtype=xk_cpu.dtype)

    # Shared x1, varied x2: pick a random reference x1, then take points near that
    # x1 whose x2 values span the local support (so rows compare x2 dependence).
    ref_idx = int(torch.randint(0, n_data, (1,), generator=rng).item())
    x1_fixed = 0.0 #float(xk_cpu[ref_idx, 0])
    near_x1 = (xk_cpu[:, 0] - x1_fixed).abs() <= half_width[0]
    near_pts = xk_cpu[near_x1]
    if near_pts.shape[0] < n_random_points:
        # Fall back to all points ordered by x2 if the x1 strip is too thin.
        near_pts = xk_cpu
        x1_fixed = float(near_pts[ref_idx, 0])
    order = torch.argsort(near_pts[:, 1])
    near_sorted = near_pts[order]
    if near_sorted.shape[0] == n_random_points:
        pick = torch.arange(n_random_points)
    else:
        pick = torch.linspace(
            0, near_sorted.shape[0] - 1, n_random_points
        ).round().long()
        # Deduplicate if rounding collapses indices; fill from neighbors if needed.
        pick = torch.unique(pick)
        if pick.numel() < n_random_points:
            need = n_random_points - pick.numel()
            all_idx = torch.arange(near_sorted.shape[0])
            mask = torch.ones(near_sorted.shape[0], dtype=torch.bool)
            mask[pick] = False
            extra = all_idx[mask][:need]
            pick = torch.sort(torch.cat([pick, extra]))[0]
    centers = near_sorted[pick]
    # Force exact shared x1 so model conditioners differ only in x2.
    centers = centers.clone()
    centers[:, 0] = x1_fixed

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
                f"x1={float(center[0]):.3f}, x2={float(center[1]):.3f}"
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
                "model p(x'|x1 fixed, x2)\n"
                f"x2={float(center[1]):.3f}, "
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
            ax_kde.set_title(
                f"conditional KDE p(x'|x)\n"
                f"x1={float(center[0]):.3f}, x2={float(center[1]):.3f}"
            )
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


def _plot_2d_values_grid(
    vals: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
    out_path: Path,
    *,
    title: str,
) -> None:
    """Plot (n_pts, n_basis) values on the unit-square mesh as a subplot grid."""
    while vals.ndim > 2 and vals.shape[0] == 1:
        vals = vals.squeeze(0)
    if vals.ndim != 2:
        raise ValueError(
            f"Expected values shape (n_grid*n_grid, n_basis), got {tuple(vals.shape)}"
        )

    n_grid = int(X.shape[0])
    if X.shape != (n_grid, n_grid) or Y.shape != (n_grid, n_grid):
        raise ValueError("X and Y must be square meshes of equal shape")
    if vals.shape[0] != n_grid * n_grid:
        raise ValueError(
            f"values leading dim {vals.shape[0]} != n_grid*n_grid={n_grid * n_grid}"
        )

    vals_np = vals.detach().cpu().numpy().reshape(n_grid, n_grid, -1)
    n_basis = vals_np.shape[-1]
    x_np = X.detach().cpu().numpy()
    y_np = Y.detach().cpu().numpy()

    n_cols = max(1, ceil(sqrt(n_basis)))
    n_rows = max(1, ceil(n_basis / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.6 * n_cols, 1.45 * n_rows),
        squeeze=False,
    )
    if title:
        fig.suptitle(title, y=1.01)

    cmap = "viridis"
    for i in range(n_basis):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        ax.contourf(x_np, y_np, vals_np[:, :, i], levels=40, cmap=cmap)
        ax.set_title(str(i), fontsize=7, pad=1)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=5)
        if r < n_rows - 1:
            ax.set_xticklabels([])
        if c > 0:
            ax.set_yticklabels([])

    for j in range(n_basis, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_metzler_cone_mlp_entries(
    mc_mlp: MetzlerConeMLP,
    out_path: Path,
    *,
    n_grid: int = 256,
    title: str = "",
) -> None:
    """Plot every entry of ``M(z) = mc_mlp(z)`` vs 1D input ``z ∈ [0, 1]``.

    Produces an ``m × m`` subplot grid: row ``i``, column ``j`` shows
    ``M(z)_{ij}`` as a function of the free (rest) coordinate.
    """
    if mc_mlp.in_features() != 1:
        raise ValueError(
            f"_plot_metzler_cone_mlp_entries expects 1D input, got in_features={mc_mlp.in_features()}"
        )

    m = mc_mlp.out_features()
    dtype, device = mc_mlp.K.dtype, mc_mlp.K.device
    z = torch.linspace(0.0, 1.0, n_grid, device=device, dtype=dtype).unsqueeze(-1)

    with torch.no_grad():
        mc_mlp.eval()
        M = mc_mlp(z).to_dense()  # (n_grid, m, m)
    if M.shape != (n_grid, m, m):
        raise ValueError(f"expected mc_mlp output shape {(n_grid, m, m)}, got {tuple(M.shape)}")

    M_np = M.detach().cpu().numpy()
    z_np = z.squeeze(-1).detach().cpu().numpy()

    fig, axes = plt.subplots(
        m,
        m,
        figsize=(1.15 * m, 1.05 * m),
        sharex=True,
        squeeze=False,
    )
    if title:
        fig.suptitle(title, y=1.01)

    for i in range(m):
        for j in range(m):
            ax = axes[i, j]
            ax.plot(z_np, M_np[:, i, j], color="C0", lw=0.9)
            ax.axhline(0.0, color="0.6", lw=0.4, zorder=0)
            ax.set_xlim(0.0, 1.0)
            ax.tick_params(labelsize=4, length=2)
            if i == 0:
                ax.set_title(f"j={j}", fontsize=6, pad=1)
            if j == 0:
                ax.set_ylabel(f"i={i}", fontsize=6)
            if i < m - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("z", fontsize=5)
            if j > 0:
                ax.set_yticklabels([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _unit_square_mesh(
    pair_basis,
    *,
    n_grid: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(X, Y, y_flat)`` on the unit square in the basis dtype/device."""
    dtype, device = pair_basis.dtype_device()
    lin = torch.linspace(0.0, 1.0, n_grid, device=device, dtype=dtype)
    X, Y = torch.meshgrid(lin, lin, indexing="xy")
    y = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    return X, Y, y


def _plot_2d_pair_member_grid(
    pair_basis,
    out_path: Path,
    *,
    index: int,
    title: str,
    n_grid: int = 64,
) -> None:
    """Plot every 2D phi (index=0) or psi (index=1) basis on a square subplot grid.

    Domain is the unit square (post-wrap coordinates). Each subplot is a filled
    contour of one basis function; unused cells are hidden.
    """
    if index not in (0, 1):
        raise ValueError(f"index must be 0 (phi) or 1 (psi), got {index}")

    dim = pair_basis.dim()
    if dim != 2:
        raise ValueError(
            f"_plot_2d_pair_member_grid expects a 2D pair basis, got dim={dim}"
        )

    X, Y, y = _unit_square_mesh(pair_basis, n_grid=n_grid)
    with torch.no_grad():
        if isinstance(pair_basis, torch.nn.Module):
            torch.nn.Module.eval(pair_basis)
        vals = pair_basis.eval(y, index).detach().cpu()

    _plot_2d_values_grid(vals, X, Y, out_path, title=title)


if __name__ == "__main__":
    problem = FULLY_OBSERVABLE_PROBLEMS["van_der_pol"]

    ###
    use_gpu = torch.cuda.is_available()
    n_basis = 30
    sacrificial_index = 0
    embedding_dim = 8
    bspline_degree = 20
    n_rays = 3
    tf_flow_hidden = 16
    tf_flow_layers = 2
    mc_mlp_hidden = 64
    mc_mlp_layers = 2
    tran_params = {
        "n_epochs_per_group": [10, 3],  # wrap + product +mc_mlp, weights
        "iterations": 5,
        "lr_product": 1e-3,
        "lr_mc_mlp": 1e-3,
        "lr_weights": 5e-2,
        "lr_wrap": 1e-3,
    }
    init_params = {
        "n_epochs_per_group": [10],  # h0 coeffs only
        "iterations": 1,
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
    if dim != 2:
        raise ValueError(f"VDP script expects a 2D system, got dim={dim}")

    x0 = problem.train_initial_state_data()
    x_k, x_kp1 = problem.train_state_transition_data()
    traj_data = problem.test_data()

    x0_dataloader = DataLoader(TensorDataset(x0), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(TensorDataset(x_kp1, x_k), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    rest_dim = dim - 1
    n_cells = n_basis - bspline_degree
    if n_cells < 1:
        raise ValueError(f"n_basis={n_basis} too small for degree={bspline_degree}")

    # Nominal 1D B-spline bases on the sacrificial coordinate (unit interval after Erf wrap).
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
    assert nom_beta_basis.n_basis_functions() == n_basis

    # Dense Metzler cone rays for gram G = Omega2(nom_alpha, nom_beta)^T (finder uses transpose=True).
    gram = nom_alpha_basis.Omega2(nom_beta_basis)
    gram_dense = gram.to_dense()
    if gram_dense.dim() == 3:
        gram_dense = gram_dense.squeeze(0)
    print(f"Computing {n_rays} dense Metzler cone rays for gram shape {tuple(gram_dense.shape)} ...")
    ray_finder = MetzlerConeRayFinder(
        DenseMatrix(gram_dense.detach().cpu()),
        transpose=True,
        n_constraint_cols=min(64, n_basis),
        n_verify_cols=min(128, n_basis),
        seed=0,
    )
    K = ray_finder.find(n_rays, ray_type="dense")
    K = DenseMatrix(K.to_dense().to(device=device, dtype=gram_dense.dtype))
    print(f"Ray matrix K shape: {tuple(K.shape)}")
    print(f"Ray matrix K: {K.to_dense()}")
    input("Press Enter to continue...")

    mc_mlp = MetzlerConeMLP(
        in_features=rest_dim,
        K=K,
        hidden_features=mc_mlp_hidden,
        num_hidden_layers=mc_mlp_layers,
        zero_init_last=True,
        coeff_scale=1.0,
        max_coeff=5.0,
        bias_init=0.0,
    ).to(device)

    # Product basis on free coords (rest_dim=1): unit-box conditional RQ-NSF + index embeddings.
    embedding = torch.nn.Embedding(n_basis, embedding_dim).to(device)
    product_model = ConditionalRQSNF(
        dim=rest_dim,
        conditioner_dim=embedding_dim,
        n_layers=tf_flow_layers,
        hidden_features=tf_flow_hidden,
        num_bins=4,
    ).to(device)
    product_basis = NormalizedIndexEmbeddingBasis(
        product_model,
        n_basis=n_basis,
        embedding=embedding,
    ).to(device)

    phi_psi_mutual = PreorthogonalMutualBasis(
        nom_alpha_basis,
        nom_beta_basis,
        mc_mlp,
        product_basis,
        sacrificial_index=sacrificial_index,
    ).to(device)

    g_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0]), epsilon=1e-3
    ).to(device)
    h0_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0])
    ).to(device)

    B_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis, n_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0]), epsilon=0.0
    ).to(device)
    B = DenseMatrixFactorization(B_coeffs)

    g_basis = phi_psi_mutual.get_basis(0, coeffs=g_coeffs)
    psi_basis = phi_psi_mutual.get_basis(1)

    wrap_tf = ErfSeparableTF.from_data(x_k, trainable=True).to(device)
    rff = SumProdRFF(g_basis, psi_basis, B, numerical_tolerance=problem.numerical_tolerance)
    tran_model = CompositeConditionalModel([wrap_tf], rff).to(device)

    print("Training transition model")
    mle_loss_fn = loss.conditional_mle_loss
    optimizers = {
        "basis": torch.optim.Adam(
            [
                {"params": mc_mlp.parameters(), "lr": tran_params["lr_mc_mlp"]},
                {"params": wrap_tf.parameters(), "lr": tran_params["lr_wrap"]},
                {"params": embedding.parameters(), "lr": tran_params["lr_product"]},
                {"params": product_model.parameters(), "lr": tran_params["lr_product"]},
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

    output_dir = Path("figures/metzler/vdp")
    output_dir.mkdir(parents=True, exist_ok=True)

    mc_mlp_out = output_dir / "metzler_cone_mlp_entries.png"
    _plot_metzler_cone_mlp_entries(
        mc_mlp,
        mc_mlp_out,
        title="VDP Metzler: trained mc_mlp(z) entries M(z)_{ij} vs free coord z",
    )
    print(f"Saved MetzlerConeMLP entry grid to {mc_mlp_out}")

    mutual_phi_out = output_dir / "mutual_basis_phi_2d.png"
    _plot_2d_pair_member_grid(
        phi_psi_mutual,
        mutual_phi_out,
        index=0,
        title="VDP Metzler: 2D mutual phi basis by index",
    )
    print(f"Saved 2D mutual phi grid to {mutual_phi_out}")

    mutual_psi_out = output_dir / "mutual_basis_psi_2d.png"
    _plot_2d_pair_member_grid(
        phi_psi_mutual,
        mutual_psi_out,
        index=1,
        title="VDP Metzler: 2D mutual psi basis by index",
    )
    print(f"Saved 2D mutual psi grid to {mutual_psi_out}")

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
        title=(
            "VDP Metzler: fixed x1, varied x2 conditional bins — "
            "empirical vs model vs KDE"
        ),
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
    fig.suptitle("VDP Metzler: beliefs at each time step")
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
