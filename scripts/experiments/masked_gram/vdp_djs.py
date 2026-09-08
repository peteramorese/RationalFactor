import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from normalizing_flow.normalizing_flow import ConditionalNSFNormalizingFlow
from rational_factor.models.composite_model import CompositeConditionalModel, CompositeDensityModel
from rational_factor.models.domain_transformation import ErfSeparableTF, MLP
from rational_factor.models.factor_forms import SumProdRFF, LinearFF
from rational_factor.models.mutual_bases import (
    NFPairBasis,
    DisjointSupport1DPWCBasis,
    MaskedGramMutualBasis,
)
from rational_factor.models.parameters import PositiveParameters, TrainableParameters, R1PDFactorizationParameters, param_group_iter, QuasiseparableFactorization
from rational_factor.systems.problems import FULLY_OBSERVABLE_PROBLEMS
from rational_factor.tools.analysis import avg_log_likelihood, check_pdf_valid
from rational_factor.tools.visualization import plot_belief
import rational_factor.models.loss as loss
import rational_factor.models.train as train
import rational_factor.tools.propagate as propagate
import matplotlib.pyplot as plt

def _make_qs_B(n_basis: int, order: int, device: torch.device) -> QuasiseparableFactorization:
    """Order-``order`` quasiseparable ``B = L D U`` with nonnegative factors.

    Zero LDU generators are a critical point (off-diagonal grads vanish), and
    signed generators make ``B`` indefinite — both break SumProdRFF. Use small
    positive random generators; ``transition_bound`` keeps ``a, b ∈ (0, 1)``.
    """
    gen_shape = (1, n_basis, order)
    diag_shape = (1, n_basis)
    # softplus(N(-2, 0.3)) ≈ small positive off-diagonals; transitions use
    # softplus then tanh·bound so products along the chain stay stable.
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
    n_basis = 200
    sacrificial_index = 0
    embedding_dim = 4
    splitter_hidden = 16
    splitter_layers = 2
    B_order = 20
    flow_hidden = 16
    flow_layers = 2
    tran_params = {
        "n_epochs_per_group": [5, 5],  # weights, basis+wrap 
        "iterations": 10,
        "lr_basis": 1e-3,
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
    x0 = problem.train_initial_state_data()
    x_k, x_kp1 = problem.train_state_transition_data()
    traj_data = problem.test_data()

    x0_dataloader = DataLoader(TensorDataset(x0), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(TensorDataset(x_kp1, x_k), batch_size=batch_size, shuffle=True, pin_memory=use_gpu)

    rest_dim = dim - 1

    # Create nf mutual basis
    nf = ConditionalNSFNormalizingFlow(
        dim=rest_dim,
        conditioner_dim=embedding_dim,
        num_layers=flow_layers,
        hidden_features=flow_hidden,
    ).to(device)
    nf_wrapper = ErfSeparableTF.from_data(x_k, trainable=True).marginal((d for d in range(dim) if d != sacrificial_index)).to(device)
    nf_wrapped = CompositeConditionalModel([nf_wrapper], nf).to(device)

    nf_embedding = torch.nn.Embedding(n_basis, embedding_dim).to(device)

    nf_splitter = MLP(
        in_features=rest_dim + embedding_dim,
        out_features=1,
        hidden_features=splitter_hidden,
        num_hidden_layers=splitter_layers,
        zero_init_last=True,
    ).to(device)

    nf_mutual = NFPairBasis(nf_wrapped, nf_splitter, nf_embedding)

    # Orthogonal 1D PWC pair on the sacrificial coordinate
    #cell_widths = PositiveParameters.random_init(shape=(1, n_basis), mean=torch.tensor([1.0]), std=torch.tensor([0.5]), normalization_dim=1).to(device)
    #alpha_params = PositiveParameters.random_init(shape=(1, n_basis), mean=torch.tensor([1.0]), std=torch.tensor([0.5])).to(device)
    cell_widths = PositiveParameters.set_init(shape=(1, n_basis), value=torch.tensor([1.0]), normalization_dim=1).to(device)
    alpha_params = PositiveParameters.set_init(shape=(1, n_basis), value=torch.tensor([1.0])).to(device)
    orth_pwc_mutual = DisjointSupport1DPWCBasis(cell_widths, alpha_params)

    phi_psi_mutual = MaskedGramMutualBasis(orth_pwc_mutual, sacrificial_index, nf_mutual).to(device)

    g_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0]), epsilon=10.0
    ).to(device)
    h0_coeffs = PositiveParameters.random_init(
        shape=(1, n_basis), mean=torch.tensor([1.0]), std=torch.tensor([1.0])
    ).to(device)

    B = _make_qs_B(n_basis, B_order, device)

    g_basis = phi_psi_mutual.get_basis(0, coeffs=g_coeffs)
    psi_basis = phi_psi_mutual.get_basis(1)

    wrap_tf = ErfSeparableTF.from_data(x_k, trainable=True).to(device)
    rff = SumProdRFF(g_basis, psi_basis, B, numerical_tolerance=problem.numerical_tolerance)
    tran_model = CompositeConditionalModel([wrap_tf], rff).to(device)

    print("Training transition model")
    mle_loss_fn = loss.conditional_mle_loss
    optimizers = {
        "weights": torch.optim.Adam(
            param_group_iter((g_coeffs, *B.parameters)),
            lr=tran_params["lr_weights"],
        ),
        "basis": torch.optim.Adam(
            [
                {"params": phi_psi_mutual.parameters(), "lr": tran_params["lr_basis"]},
                {"params": wrap_tf.parameters(), "lr": tran_params["lr_wrap"]},
                {"params": nf_wrapper.parameters(), "lr": tran_params["lr_wrap"]},
            ]
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
