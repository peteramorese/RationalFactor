from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from normalizing_flow.composite_model import CompositeConditionalModel, CompositeDensityModel
from normalizing_flow.transforms import Transforms
from rational_factor.models.big_polynomial import BigBernstein, ConditionalBigLegendre
from rational_factor.models.kde import GaussianKDE
from rational_factor.models.parameters import PositiveParameters, TrainableParameters, param_group_iter
from rational_factor.models.train import LossVisualizer, Trainer
from rational_factor.models.tt.tt_parameters import MPOParameters, TTVectorParameters
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
import rational_factor.models.loss as loss


def _plot_conditional_slices_model_vs_data(
    tran_model,
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

    ref_idx = int(torch.randint(0, n_data, (1,), generator=rng).item())
    x1_fixed = 0.0
    near_x1 = (xk_cpu[:, 0] - x1_fixed).abs() <= half_width[0]
    near_pts = xk_cpu[near_x1]
    if near_pts.shape[0] < n_random_points:
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
        pick = torch.unique(pick)
        if pick.numel() < n_random_points:
            need = n_random_points - pick.numel()
            all_idx = torch.arange(near_sorted.shape[0])
            mask = torch.ones(near_sorted.shape[0], dtype=torch.bool)
            mask[pick] = False
            extra = all_idx[mask][:need]
            pick = torch.sort(torch.cat([pick, extra]))[0]
    centers = near_sorted[pick]
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


def _cbl_positivity_loss(model) -> torch.Tensor:
    """Positivity penalty for a wrapped ``ConditionalBigLegendre``.

    The ``Trainer`` passes the full composite (wrapped) transition model, but
    ``loss.cbl_positivity_loss`` requires the bare ``ConditionalBigLegendre``,
    so unwrap ``base_density`` here. ``method='cores'`` is the default: zero
    loss certifies nonnegative Bernstein coefficients of every MPO core.
    """
    return loss.cbl_positivity_loss(model.base_density)


if __name__ == "__main__":
    problem = FULLY_OBSERVABLE_PROBLEMS["van_der_pol"]

    ###
    use_gpu = torch.cuda.is_available()
    # Local (per-dimension) polynomial size: degree = local_basis_size - 1.
    # Both big-polynomial bases require equal local mode sizes, so the TT/MPO
    # modes are (local_basis_size,) * dim.
    local_basis_size = 8
    tran_rank = 5
    init_rank = 5
    tran_params = {
        "epochs": 50,
        "wrap_iterations": 1,  # full data passes per epoch for the wrap optimizer
        "matrix_iterations": 1,  # full data passes per epoch for the MPO optimizer
        "lr_matrix": 1e-2,
        "lr_wrap": 1e-3,
    }
    init_params = {
        "epochs": 50,
        "coeff_iterations": 1,  # full data passes per epoch for the TT-coeff optimizer
        "lr_coeffs": 1e-2,
    }

    batch_size = 256

    ###

    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using GPU: ", use_gpu)
    print("Device: ", device)

    system = problem.system
    dim = system.dim()
    if dim != 2:
        raise ValueError(f"VDP script expects a 2D system, got dim={dim}")
    d = dim
    modes = (local_basis_size,) * d

    x0 = problem.train_initial_state_data()
    x_k, x_kp1 = problem.train_state_transition_data()

    x0_dataloader = DataLoader(
        TensorDataset(x0), batch_size=batch_size, shuffle=True, pin_memory=use_gpu
    )
    xp_dataloader = DataLoader(
        TensorDataset(x_kp1, x_k), batch_size=batch_size, shuffle=True, pin_memory=use_gpu
    )

    print(
        f"modes={modes}, dim={d}, tran_rank={tran_rank}, init_rank={init_rank}"
    )

    # Shared by both Trainer runs; the second run re-initializes (and
    # overwrites) figures/loss_progress.png, which is fine.
    loss_visualizer = LossVisualizer(
        Path("figures"), buffer_size=1, filename="loss_progress.png"
    )

    # ------------------------------------------------------------------
    # Transition model: ConditionalBigLegendre behind a trainable erf wrap.
    #
    # The wrap maps raw states to the unit box the Legendre kernel lives on.
    # Both x' and the conditioner x are wrapped (transform_conditioner=True),
    # and the wrap is trained jointly with the transition model. Its
    # parameters form their own optimizer group below.
    # ------------------------------------------------------------------
    matrix_params = MPOParameters.from_core_spec(
        modes,
        ranks=tran_rank,
        param_cls=TrainableParameters,
        mean=0.0,
        std=0.5,
        device=device,
    )
    tran_base = ConditionalBigLegendre(matrix_params)
    tran_model = CompositeConditionalModel(
        tran_base,
        context_features=d,
        transform="erf",
        x_data=x_k,
        trainable=True,
        transform_conditioner=True,
    ).to(device)
    wrap_tf = tran_model.transform

    print("Training transition model")
    tran_trainer = Trainer(tran_model, device=device, loss_visualizer=loss_visualizer)
    tran_trainer.add_dataset(xp_dataloader, "transitions")
    tran_trainer.add_optimizer(
        torch.optim.Adam(wrap_tf.parameters(), lr=tran_params["lr_wrap"]),
        "wrap",
        n_iterations=tran_params["wrap_iterations"],
    )
    tran_trainer.add_optimizer(
        torch.optim.Adam(
            param_group_iter(tuple(matrix_params.parameter_modules())),
            lr=tran_params["lr_matrix"],
        ),
        "matrix",
        n_iterations=tran_params["matrix_iterations"],
    )
    tran_trainer.add_loss_function(
        loss.conditional_mle_loss, "mle", dataset_label="transitions"
    )
    # Feasibility loss: still differentiated and included in the total, but a
    # checkpoint is only saved when this re-evaluates to exactly zero after
    # an epoch (see Trainer). If no epoch is feasible, train() returns None.
    tran_trainer.add_loss_function(
        _cbl_positivity_loss, "positivity", dataset_label=None, feasibility=True
    )
    tran_model, best_loss_tran, training_time_tran = tran_trainer.train(
        epochs=tran_params["epochs"],
        use_best="mle",
        verbose=True,
    )
    if tran_model is None:
        raise SystemExit(
            "Transition training found no feasible checkpoint: "
            "cbl_positivity_loss never reached exactly zero at an epoch "
            "boundary, so no transition model was saved. Increase the number "
            "of epochs or the matrix learning rate and rerun."
        )
    print("Done! \n")

    # Freeze the trained wrap: the initial model reuses exactly this map into
    # the unit box, but does not train it (as in metzler/vdp.py).
    trained_wrap_tf = Transforms.freeze(wrap_tf).to(device)

    # ------------------------------------------------------------------
    # Initial model: BigBernstein behind the frozen wrap. Independent of the
    # transition model (no shared bases/coeffs), so nothing else to freeze.
    # ------------------------------------------------------------------
    init_coeff_params = TTVectorParameters.from_core_spec(
        modes,
        ranks=init_rank,
        param_cls=PositiveParameters,
        mean=1.0,
        std=1.0,
        epsilon=1e-3,
        device=device,
    )
    init_base = BigBernstein(init_coeff_params)
    init_model = CompositeDensityModel(
        init_base,
        transform="stacked",
        transforms=[trained_wrap_tf],
    ).to(device)

    print("Training initial model")
    init_trainer = Trainer(init_model, device=device, loss_visualizer=loss_visualizer)
    init_trainer.add_dataset(x0_dataloader, "initial")
    init_trainer.add_optimizer(
        torch.optim.Adam(
            param_group_iter(tuple(init_coeff_params.parameter_modules())),
            lr=init_params["lr_coeffs"],
        ),
        "coeffs",
        n_iterations=init_params["coeff_iterations"],
    )
    init_trainer.add_loss_function(loss.mle_loss, "mle", dataset_label="initial")
    init_model, best_loss_init, training_time_init = init_trainer.train(
        epochs=init_params["epochs"],
        use_best="mle",
        verbose=True,
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

    tran_model = tran_model.to(device).eval()
    init_model = init_model.to(device).eval()

    output_dir = Path("figures/big_polynomial/vdp")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_x_range = (
        float(problem.plot_bounds_low[0]),
        float(problem.plot_bounds_high[0]),
    )
    plot_y_range = (
        float(problem.plot_bounds_low[1]),
        float(problem.plot_bounds_high[1]),
    )

    cond_slice_out_path = output_dir / "conditional_slices_model_vs_data.png"
    _plot_conditional_slices_model_vs_data(
        tran_model,
        x_k,
        x_kp1,
        cond_slice_out_path,
        x_range=plot_x_range,
        y_range=plot_y_range,
        n_random_points=10,
        n_grid=120,
        min_points_per_slice=20,
        bin_width=0.4,
        random_seed=0,
        title=(
            "VDP BigPolynomial: fixed x1, varied x2 conditional bins — "
            "empirical vs model vs KDE"
        ),
    )
    print(f"Saved conditional slice comparison to {cond_slice_out_path}")
