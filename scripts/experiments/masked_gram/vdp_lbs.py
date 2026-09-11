from pathlib import Path

import copy

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from normalizing_flow.normalizing_flow import ConditionalNSFNormalizingFlow
from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.composite_model import CompositeConditionalModel, CompositeDensityModel
from rational_factor.models.domain_transformation import ErfSeparableTF, IdentityTF, StackedTF
from rational_factor.models.factor_forms import SumProdRFF, LinearFF
from rational_factor.models.mutual_bases import (
    LocalBSplineMutualBasis,
    NormalizedProductPairBasis,
    PositiveMaskedGramMutualBasis,
)
from rational_factor.models.parameters import (
    PositiveParameters,
    QuasiseparableFactorization,
    TrainableParameters,
    param_group_iter,
)
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood, check_pdf_valid
from rational_factor.tools.visualization import plot_belief
import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.tools.propagate as propagate


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


def _make_wrap_tf(
    x_data: torch.Tensor,
    *,
    dim: int,
    sacrificial_index: int,
    trainable: bool = True,
) -> StackedTF | ErfSeparableTF:
    """Erf on the sacrificial coordinate; identity on the free ``R^{d-1}`` coords."""
    if not (0 <= sacrificial_index < dim):
        raise ValueError(f"sacrificial_index must be in [0, {dim}), got {sacrificial_index}")

    parts = []
    if sacrificial_index > 0:
        parts.append(IdentityTF(sacrificial_index))

    x_s = x_data[:, sacrificial_index : sacrificial_index + 1]
    parts.append(ErfSeparableTF.from_data(x_s, trainable=trainable))

    n_after = dim - sacrificial_index - 1
    if n_after > 0:
        parts.append(IdentityTF(n_after))

    if len(parts) == 1:
        return parts[0]
    return StackedTF(parts)


def _freeze_wrap_tf(wrap: StackedTF | ErfSeparableTF) -> StackedTF | ErfSeparableTF:
    """Detach a trained wrap so initial-state / belief models stay fixed."""
    if isinstance(wrap, ErfSeparableTF):
        return ErfSeparableTF.copy_from_trainable(wrap)

    frozen = []
    for tf in wrap.tfs:
        if isinstance(tf, ErfSeparableTF):
            frozen.append(ErfSeparableTF.copy_from_trainable(tf))
        elif isinstance(tf, IdentityTF):
            frozen.append(IdentityTF(tf.dim))
        else:
            part = copy.deepcopy(tf)
            for p in part.parameters():
                p.requires_grad_(False)
            frozen.append(part)
    return StackedTF(frozen)


def _make_free_basis(
    *,
    rest_dim: int,
    n_basis: int,
    embedding_dim: int,
    flow_layers: int,
    flow_hidden: int,
    device: torch.device,
) -> NormalizedProductPairBasis:
    """Free pair on ``R^{rest_dim}``: conditional NSF product + GaussianBasis splitter.

    Additive VP splitters are unavailable in 1-D, and MAF is similarly restricted,
    so the product is a conditional NSF and the splitter is a trainable Gaussian
    PDF basis (one component per free-basis index).
    """
    if rest_dim < 1:
        raise ValueError(f"VDP free basis requires rest_dim >= 1, got {rest_dim}")

    embedding = torch.nn.Embedding(n_basis, embedding_dim).to(device)
    product = ConditionalNSFNormalizingFlow(
        dim=rest_dim,
        conditioner_dim=embedding_dim,
        num_layers=flow_layers,
        hidden_features=flow_hidden,
    ).to(device)

    shape = (1, rest_dim, n_basis)
    means = TrainableParameters.random_init(shape=shape, mean=0.0, std=2.0).to(device)
    stds = PositiveParameters.random_init(
        shape=shape, mean=1.0, std=0.5, epsilon=1e-2
    ).to(device)
    splitter = GaussianBasis(means, stds)

    # PositiveMaskedGram needs a finite free-beta bound; NSF/Gaussian has no
    # closed form, so use a multiple of the Gaussian PDF peak as a soft bound.
    with torch.no_grad():
        gauss_peak = float(splitter.supremum_bound().amax().item())
    beta_supremum = max(10.0, 20.0 * gauss_peak)

    return NormalizedProductPairBasis(
        product,
        splitter,
        embedding,
        beta_supremum=beta_supremum,
    ).to(device)


if __name__ == "__main__":
    problem = FULLY_OBSERVABLE_PROBLEMS["van_der_pol"]

    ###
    use_gpu = torch.cuda.is_available()
    n_basis = 100
    sacrificial_index = 0
    embedding_dim = 4
    k_alpha = 3
    k_beta = 5
    trainable_beta = True
    B_order = 10
    flow_hidden = 32
    flow_layers = 3
    tran_params = {
        "n_epochs_per_group": [1, 5],  # basis+wrap, weights
        "iterations": 5,
        "lr_basis": 1e-3,
        "lr_weights": 5e-2,
        "lr_wrap": 1e-3,
    }
    init_params = {
        "n_epochs_per_group": [20],  # h0 coeffs only
        "iterations": 10,
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

    # Local B-spline pair on the sacrificial coordinate (unit interval after Erf wrap).
    masking = LocalBSplineMutualBasis(
        n_basis=n_basis,
        k_alpha=k_alpha,
        k_beta=k_beta,
        trainable_beta=trainable_beta,
        device=device,
    ).to(device)

    # Free pair on R^{d-1}: NSF product + GaussianBasis splitter.
    free_basis = _make_free_basis(
        rest_dim=rest_dim,
        n_basis=n_basis,
        embedding_dim=embedding_dim,
        flow_layers=flow_layers,
        flow_hidden=flow_hidden,
        device=device,
    )

    phi_psi_mutual = PositiveMaskedGramMutualBasis(
        masking,
        sacrificial_index,
        free_basis,
    ).to(device)

    g_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0]), epsilon=10.0
    ).to(device)
    h0_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0])
    ).to(device)

    B = _make_qs_B(n_basis, B_order, device)

    g_basis = phi_psi_mutual.get_basis(0, coeffs=g_coeffs)
    psi_basis = phi_psi_mutual.get_basis(1)

    wrap_tf = _make_wrap_tf(
        x_k, dim=dim, sacrificial_index=sacrificial_index, trainable=True
    ).to(device)
    rff = SumProdRFF(g_basis, psi_basis, B, numerical_tolerance=problem.numerical_tolerance)
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
            param_group_iter((g_coeffs, *B.parameters)),
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
    trained_wrap_tf = _freeze_wrap_tf(wrap_tf).to(device)

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

    output_dir = Path("figures/masked_gram/vdp_lbs")
    output_dir.mkdir(parents=True, exist_ok=True)

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
