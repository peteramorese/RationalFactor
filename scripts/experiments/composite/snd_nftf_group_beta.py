from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.tools.propagate as propagate
from rational_factor.models.basis_functions import BetaBasis
from rational_factor.models.composite_model import CompositeConditionalModel, CompositeDensityModel
from rational_factor.models.domain_transformation import ErfSeparableTF, MaskedRQSNFTF
from rational_factor.models.factor_forms import LinearFF, LinearRFF
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood, check_pdf_valid


def _plot_snd_1d_marginal_trajectory_comparison(
    problem,
    beliefs: list[CompositeDensityModel],
    test_traj_data: list[torch.Tensor],
    out_path: Path,
    *,
    n_grid: int = 400,
    title: str = "",
) -> None:
    """1D-only: learned marginal pdf vs normalized histogram of test trajectories per timestep."""
    n_slices = problem.n_timesteps + 1
    assert len(beliefs) == n_slices
    lo = float(problem.plot_bounds_low[0].item())
    hi = float(problem.plot_bounds_high[0].item())
    x_grid = torch.linspace(lo, hi, n_grid).unsqueeze(1)

    param = next(iter(beliefs[0].parameters()), None)
    buffer = next(iter(beliefs[0].buffers()), None)
    dev = param.device if param is not None else (buffer.device if buffer is not None else torch.device("cpu"))
    dt = param.dtype if param is not None else (buffer.dtype if buffer is not None else torch.float32)
    x_grid = x_grid.to(device=dev, dtype=dt)

    fig, axes = plt.subplots(n_slices, 1, figsize=(8, 2.2 * n_slices), squeeze=False)
    for t in range(n_slices):
        ax = axes[t, 0]
        samples = test_traj_data[t][:, 0].detach().cpu().numpy()
        with torch.no_grad():
            beliefs[t].eval()
            pdf = beliefs[t](x_grid).squeeze(-1).detach().cpu().numpy()
        x_np = x_grid.squeeze(-1).detach().cpu().numpy()
        ax.plot(x_np, pdf, color="C0", lw=2.0, label="learned marginal")
        ax.hist(samples, bins=50, density=True, alpha=0.35, color="C1", label="test traj. (hist)")
        ax.set_xlim(lo, hi)
        ax.set_ylabel("density")
        ax.set_title(f"t = {t}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel(problem.system.state_label(0))
    if title:
        fig.suptitle(title, y=1.002)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_snd_conditional_true_vs_learned(
    tran_model: CompositeConditionalModel,
    system,
    out_path: Path,
    *,
    problem,
    n_grid: int = 200,
    title: str = "",
) -> None:
    """
    2D heatmaps of p(x'|x): horizontal axis = conditioner x, vertical axis = next state x'.
    Left: true (Gaussian around drift); right: learned LinearRFF (+ domain transforms).
    """
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

    # Shared color scale for direct comparison
    vmax = float(max(true_pdf.max(), learned.max()))
    vmin = 0.0
    if vmax <= 0:
        vmax = 1.0

    x_np = x_1d.detach().cpu().numpy()
    xp_np = xp_1d.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    cmap = "viridis"
    for ax, Z, subt in zip(
        axes[0],
        [true_pdf.T, learned.T],
        ["true p(x'|x)", "learned p(x'|x)"],
    ):
        # contourf expects Z with shape (len(yp), len(xp)) for values at (x, y) = (columns, rows)
        cf = ax.contourf(x_np, xp_np, Z, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_aspect("auto")
        ax.set_xlabel("x (conditioner)")
        ax.set_ylabel("x'")
        ax.set_title(subt)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    problem = FULLY_OBSERVABLE_PROBLEMS["scalar_nonlinear_drift"]

    ######## USER CONFIG ########
    use_gpu = torch.cuda.is_available()
    use_dtf = False
    n_basis = 1000
    batch_size = 256

    if use_dtf:
        tran_params = {
            "n_epochs_per_group": [15, 5],  # dtf+basis, then weights
            "iterations": 50,
            "lr_basis": 1e-2,
            "lr_weights": 1e-2,
            "lr_dtf": 1e-3,
            "lr_wrap": 1e-3,
        }
        init_params = {
            "n_epochs_per_group": [20, 5],  # basis, then weights
            "iterations": 100,
            "lr_basis": 5e-2,
            "lr_weights": 1e-2,
        }
    else:
        tran_params = {
            "n_epochs_per_group": [15, 5],  # basis, then weights
            "iterations": 100,
            "lr_basis": 1e-2,
            "lr_weights": 1e-2,
            "lr_wrap": 1e-5,
        }
        init_params = {
            "n_epochs_per_group": [20, 5],  # basis, then weights
            "iterations": 100,
            "lr_basis": 5e-2,
            "lr_weights": 1e-2,
        }

    ######## SETUP ########
    device = torch.device("cuda" if use_gpu else "cpu")
    print("Using device: ", device)

    system = problem.system
    x0_data = problem.train_initial_state_data()
    x_k_data, x_kp1_data = problem.train_state_transition_data()
    test_traj_data = problem.test_data()

    x0_dataloader = DataLoader(TensorDataset(x0_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(TensorDataset(x_kp1_data, x_k_data), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    offsets = torch.tensor([10.0, 10.0], device=device)
    variance = 30.0
    phi_basis = BetaBasis.random_init(
        system.dim(),
        n_basis=n_basis,
        offsets=offsets,
        variance=variance,
        min_concentration=1.0,
    ).to(device)
    psi_basis = BetaBasis.random_init(
        system.dim(),
        n_basis=n_basis,
        offsets=offsets,
        variance=variance,
        min_concentration=1.0,
    ).to(device)
    psi0_basis = BetaBasis.random_init(
        system.dim(),
        n_basis=n_basis,
        offsets=offsets,
        variance=variance,
        min_concentration=1.0,
    ).to(device)

    wrap_tf = ErfSeparableTF.from_data(torch.cat([x0_data, x_k_data], dim=0), trainable=False).to(device)
    #wrap_tf = ErfSeparableTF.from_data(torch.cat([x0_data, x_k_data], dim=0), trainable=True).to(device)
    nftf = (
        MaskedRQSNFTF(system.dim(), trainable=True, hidden_features=256, n_layers=6).to(device)
        if use_dtf
        else None
    )

    ######## TRAIN TRANSITION ########
    if use_dtf:
        tran_model = CompositeConditionalModel([nftf, wrap_tf], LinearRFF(phi_basis, psi_basis, numerical_tolerance=problem.numerical_tolerance)).to(device)
    else:
        tran_model = CompositeConditionalModel([wrap_tf], LinearRFF(phi_basis, psi_basis, numerical_tolerance=problem.numerical_tolerance)).to(device)

    print("Training transition model")

    if use_dtf:
        optimizers = {
            "dtf_and_basis": torch.optim.Adam(
                [
                    {"params": tran_model.conditional_density_model.basis_params(), "lr": tran_params["lr_basis"]},
                    {"params": tran_model.domain_tfs[0].parameters(), "lr": tran_params["lr_dtf"]},
                    #{"params": tran_model.domain_tfs[1].parameters(), "lr": tran_params["lr_wrap"]},
                ]
            ),
            "weights": torch.optim.Adam(tran_model.conditional_density_model.weight_params(), lr=tran_params["lr_weights"]),
        }
    else:
        optimizers = {
            "basis": torch.optim.Adam(
                [
                    {"params": tran_model.conditional_density_model.basis_params(), "lr": tran_params["lr_basis"]},
                    {"params": tran_model.domain_tfs.parameters(), "lr": tran_params["lr_wrap"]},
                ]
            ),
            "weights": torch.optim.Adam(tran_model.conditional_density_model.weight_params(), lr=tran_params["lr_weights"]),
        }

    tran_model, best_loss_tran, training_time_tran = train.train_iterate(
        tran_model,
        xp_dataloader,
        {"mle": loss.conditional_mle_loss},
        optimizers,
        epochs_per_group=tran_params["n_epochs_per_group"],
        iterations=tran_params["iterations"],
        verbose=True,
        use_best="mle",
    )
    print("Done.\n")
    print("Valid: ", tran_model.valid())

    trained_nftf = MaskedRQSNFTF.copy_from_trainable(nftf).to(device) if use_dtf else None
    trained_domain_tf = ErfSeparableTF.copy_from_trainable(wrap_tf).to(device)

    print("Wrap loc/scale: ", wrap_tf.loc_scale())

    ######## TRAIN INITIAL ########
    if use_dtf:
        init_model = CompositeDensityModel(
            [trained_nftf, trained_domain_tf],
            LinearFF.from_rff(tran_model.conditional_density_model, psi0_basis),
        ).to(device)
    else:
        init_model = CompositeDensityModel(
            [trained_domain_tf],
            LinearFF.from_rff(tran_model.conditional_density_model, psi0_basis),
        ).to(device)

    print("Training initial model")
    optimizers = {
        "basis": torch.optim.Adam(init_model.density_model.basis_params(), lr=init_params["lr_basis"]),
        "weights": torch.optim.Adam(init_model.density_model.weight_params(), lr=init_params["lr_weights"]),
    }

    init_model, best_loss_init, training_time_init = train.train_iterate(
        init_model,
        x0_dataloader,
        {"mle": loss.mle_loss},
        optimizers,
        epochs_per_group=init_params["n_epochs_per_group"],
        iterations=init_params["iterations"],
        verbose=True,
        use_best="mle",
    )
    print("Done.\n")

    print(f"Transition model loss: {best_loss_tran:.6f}, training time: {training_time_tran:.2f}s")
    print(f"Initial model loss: {best_loss_init:.6f}, training time: {training_time_init:.2f}s")

    init_model.density_model.debug = True
    n_debug_samples = 5
    #lows = problem.plot_bounds_low.to(device=device, dtype=x0_data.dtype)
    #highs = problem.plot_bounds_high.to(device=device, dtype=x0_data.dtype)
    #x_debug = torch.rand(n_debug_samples, system.dim(), device=device, dtype=x0_data.dtype) * (highs - lows) + lows
    #x_debug = torch.tensor([
    #    [0.0, 0.0, 0.0, 0.0],
    #    [0.5, 0.0, 0.0, 0.0],
    #    [0.0, 0.5, 0.0, 0.0],
    #    [0.0, 0.0, 0.5, 0.0],
    #    [0.0, 0.0, 0.0, 0.5],
    #    ], device=device, dtype=x0_data.dtype)
    #print("x_debug samples:\n", x_debug)
    #with torch.no_grad():
    #    init_model.eval()
    #    logp_debug = init_model.log_density(x_debug, debug=True)
    #print("init_model.log_density(x_debug):\n", logp_debug)
    #x_debug_marginal = torch.tensor([
    #    [0.0, 0.0],
    #    [0.5, 0.0],
    #    [0.0, 0.5],
    #    [0.0, 0.0],
    #    [0.0, 0.0],
    #    ], device=device, dtype=x0_data.dtype)
    #test_marginal = init_model.marginal((0, 1))
    #test_marginal.debug = True
    #print("test_marginal.log_density(x_debug_marginal):\n", test_marginal.log_density(x_debug_marginal, debug=True))

    #input("Press Enter to continue...")

    ######## ANALYSIS ########
    base_belief_seq = propagate.propagate(init_model.density_model, tran_model.conditional_density_model, n_steps=problem.n_timesteps)
    if use_dtf:
        belief_seq = [CompositeDensityModel([trained_nftf, trained_domain_tf], belief) for belief in base_belief_seq]
    else:
        belief_seq = [CompositeDensityModel([trained_domain_tf], belief) for belief in base_belief_seq]

    ll_per_step = []
    for i in range(problem.n_timesteps):
        data_i = test_traj_data[i].to(device)
        ll = avg_log_likelihood(belief_seq[i], data_i)
        check_pdf_valid(belief_seq[i], domain_bounds=(tuple(problem.plot_bounds_low.tolist()), tuple(problem.plot_bounds_high.tolist())), n_samples=10000, device=device)
        ll_per_step.append(ll.detach().cpu().reshape(()))
        print(f"Avg log-likelihood at time {i}: {float(ll_per_step[-1]):.6f}")
    prop_ll_vector = torch.stack(ll_per_step).to(dtype=torch.float32)

    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    ll_out_path = out_dir / f"{Path(__file__).stem}__ll.png"
    plt.figure(figsize=(8, 4))
    plt.plot(prop_ll_vector.numpy(), marker="o")
    plt.xlabel("timestep")
    plt.ylabel("avg log-likelihood")
    plt.title(f"{Path(__file__).stem} (use_dtf={use_dtf}, n_basis={n_basis})")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(ll_out_path, dpi=200)
    print(f"Saved LL plot to {ll_out_path}")

    # 1D-only figures: marginal pdfs + true vs learned conditional densities (CPU for plotting).
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    belief_seq_cpu = [belief.to("cpu").eval() for belief in belief_seq]
    tran_model_cpu = tran_model.cpu().eval()

    stem = Path(__file__).stem
    marg_out_path = out_dir / f"{stem}__marginal_comparison.png"
    _plot_snd_1d_marginal_trajectory_comparison(
        problem,
        belief_seq_cpu,
        test_traj_data,
        marg_out_path,
        n_grid=400,
        title=f"{stem} marginal (1D): learned vs test samples",
    )
    print(f"Saved 1D marginal comparison plot to {marg_out_path}")

    cond_out_path = out_dir / f"{stem}__conditional_comparison.png"
    _plot_snd_conditional_true_vs_learned(
        tran_model_cpu,
        system,
        cond_out_path,
        problem=problem,
        n_grid=200,
        title=f"{stem}: true vs learned conditional density (2D)",
    )
    print(f"Saved conditional density comparison plot to {cond_out_path}")


if __name__ == "__main__":
    main()
