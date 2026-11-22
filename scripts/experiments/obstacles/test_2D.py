"""Learn nonoverlapping rectangular latent obstacle approximations in 2D.

From the repository root (with dependencies installed, e.g. `pip install -e .`):
    python scripts/experiments/obstacle_space/test_2D.py

Each obstacle is a lumpy star-shaped region with a smooth, non-rectangular
boundary. Samples are drawn uniformly from the entire workspace and rejected
unless they lie inside at least one obstacle. The flow is fitted *only* on the
retained colliding samples (no ground-truth labels in the training losses).
"""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from torch.utils.data import DataLoader, TensorDataset


DOMAIN_LOW = -3.0
DOMAIN_HIGH = 3.0


def sample_collisions(
    n_samples: int,
    candidate_batch: int,
    dimension: int,
    label_fn,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Uniformly sample the full domain, retaining *only* colliding states.

    ``label_fn(points)`` returns the first intersected obstacle index or -1.
    Using rejection sampling (rather than sampling inside known obstacles) keeps
    the collision sample distribution uniform over their union.
    """
    collected, labels = [], []
    accepted, proposed = 0, 0
    while accepted < n_samples:
        if proposed > 200 * candidate_batch:
            raise RuntimeError("Very few collision samples; enlarge the obstacles or candidate batch")
        candidates = DOMAIN_LOW + (DOMAIN_HIGH - DOMAIN_LOW) * torch.rand(
            candidate_batch, dimension, generator=generator
        )
        which = label_fn(candidates)
        is_collision = which >= 0
        hit = candidates[is_collision]
        lab = which[is_collision]
        collected.append(hit)
        labels.append(lab)
        accepted += hit.shape[0]
        proposed += candidate_batch
    return torch.cat(collected)[:n_samples], torch.cat(labels)[:n_samples], proposed


def figure_directory() -> Path:
    directory = Path(__file__).resolve().parents[3] / "figures" / "obstacle_space"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def random_nonoverlapping_boxes(
    n_boxes: int,
    dimension: int,
    *,
    domain_low: float = -2.0,
    domain_high: float = 2.0,
    min_width: float = 0.4,
    max_width: float = 1.0,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
    max_tries: int = 20_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random axis-aligned boxes that do not overlap."""
    span = domain_high - domain_low
    if max_width > span or min_width <= 0 or min_width > max_width:
        raise ValueError("invalid box width / domain settings")
    lowers: list[torch.Tensor] = []
    uppers: list[torch.Tensor] = []
    tries = 0
    while len(lowers) < n_boxes:
        tries += 1
        if tries > max_tries:
            raise RuntimeError("could not place enough nonoverlapping boxes; relax sizes or domain")
        width = min_width + (max_width - min_width) * torch.rand(
            dimension, generator=generator, dtype=dtype,
        )
        lower = domain_low + (span - width) * torch.rand(
            dimension, generator=generator, dtype=dtype,
        )
        upper = lower + width
        overlaps = False
        for lo, hi in zip(lowers, uppers):
            if torch.all(torch.minimum(upper, hi) - torch.maximum(lower, lo) > 0):
                overlaps = True
                break
        if not overlaps:
            lowers.append(lower)
            uppers.append(upper)
    return torch.stack(lowers), torch.stack(uppers)


def fit_rectangular_flow(
    samples: torch.Tensor,
    n_components: int,
    cfg: SimpleNamespace,
    figure_dir: Path,
    figure_prefix: str,
    generator: torch.Generator | None = None,
):
    """Train MLE + sigma + geometric overlap via the repository's Trainer.

    Training is unsupervised: MLE on the colliding point cloud plus model-only
    regularizers. Base boxes are initialized as random nonoverlapping rectangles.
    """
    from normalizing_flow.composite_model import CompositeDensityModel
    from normalizing_flow.transforms import Transforms
    from rational_factor.models.distributions import SmoothBoxMixture
    from rational_factor.models.loss import (
        mle_loss, smooth_box_overlap_loss, smooth_box_sigma_loss,
    )
    from rational_factor.models.train import LossVisualizer, Trainer

    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    device = torch.device(cfg.device)
    data = samples.to(dtype=dtype)
    dimension = data.shape[-1]
    if cfg.identity_transform:
        transform = Transforms.make_transform("identity", features=dimension)
    else:
        transform = Transforms.make_transform(
            cfg.transform,
            features=dimension,
            num_layers=cfg.num_layers,
            hidden_features=cfg.hidden_features,
            init_identity=True,  # Supported by MAF; ignored by NSF.
        )
    transform = transform.to(device=device, dtype=dtype)

    lower, upper = random_nonoverlapping_boxes(
        n_components, dimension, generator=generator, dtype=dtype,
    )
    base = SmoothBoxMixture(
        features=dimension,
        n_components=n_components,
        lower=lower,
        upper=upper,
        sigma=cfg.initial_sigma,
        batch_shape=(),  # log_density: (n_particles,), rather than (n_particles, 1)
    )
    model = CompositeDensityModel(base, transform=transform).to(device=device, dtype=dtype)

    loader = DataLoader(
        TensorDataset(data), batch_size=cfg.batch_size, shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    trainer = Trainer(
        model, device=device,
        loss_visualizer=LossVisualizer(
            figure_dir, buffer_size=10,
            filename=f"{figure_prefix}_training_losses.png",
        ),
    )
    trainer.add_dataset(loader, "collisions")
    # Identity mode tunes only the SmoothBoxMixture; otherwise train the full flow.
    trainable = (
        model.base_density.parameters() if cfg.identity_transform else model.parameters()
    )
    trainer.add_optimizer(torch.optim.Adam(trainable, lr=cfg.lr), "adam")
    trainer.add_loss_function(mle_loss, "mle", "collisions")
    trainer.add_loss_function(
        lambda m: cfg.sigma_weight * smooth_box_sigma_loss(m.base_density),
        "sigma", None,
    )
    trainer.add_loss_function(
        lambda m: cfg.overlap_weight * smooth_box_overlap_loss(
            m.base_density, margin=cfg.overlap_margin
        ),
        "overlap", None,
    )
    model, best_loss, elapsed = trainer.train(
        epochs=cfg.epochs, use_best="total", verbose=cfg.verbose,
        restore_loss_threshold=None,
    )
    if model is None:
        raise RuntimeError("Trainer could not find a finite, valid checkpoint")

    model.eval()
    with torch.no_grad():
        # Preserve the original sample order for the label/color correspondence.
        latent = torch.cat(
            [model.transform(chunk.to(device))[0].cpu() for chunk in data.split(2048)], dim=0
        )
        lower, upper = (v.detach().cpu() for v in model.base_density.bounds())
        sigmas = model.base_density.sigmas().detach().cpu()
        weights = model.base_density.weights().detach().cpu()

    xyz = latent.numpy()
    lo, hi = lower.numpy(), upper.numpy()
    covered = ((xyz[:, None, :] >= lo[None, :, :]) & (xyz[:, None, :] <= hi[None, :, :])).all(axis=-1).any(axis=-1)
    overlapping_pairs = 0
    for i in range(n_components):
        for j in range(i + 1, n_components):
            if np.all(np.minimum(hi[i], hi[j]) - np.maximum(lo[i], lo[j]) > 0):
                overlapping_pairs += 1
    print(
        f"Training complete: best loss={best_loss:.5f}; time={elapsed:.1f}s; "
        f"particles in boxes={covered.mean():.1%}; "
        f"overlapping box pairs={overlapping_pairs}; "
        f"mean sigma={float(sigmas.mean()):.5f}"
    )
    print(f"Learned mixture weights: {weights.numpy().round(3)}")
    return xyz, lo, hi, weights.numpy()


def set_padded_limits_2d(ax, coordinates: np.ndarray) -> None:
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    pad = np.maximum(0.08 * (upper - lower), 0.12)
    ax.set_xlim(lower[0] - pad[0], upper[0] + pad[0])
    ax.set_ylim(lower[1] - pad[1], upper[1] + pad[1])
    ax.set_aspect("equal", adjustable="box")


def radial_distance(angle: torch.Tensor, phase: torch.Tensor, scale: torch.Tensor):
    return scale * (
        1
        + 0.23 * torch.sin(3 * angle + phase)
        + 0.13 * torch.cos(5 * angle - 0.7 * phase)
        + 0.075 * torch.sin(8 * angle + 1.3 * phase)
    )


def obstacle_labels(points: torch.Tensor, centers, phases, radii) -> torch.Tensor:
    displacement = points[:, None, :] - centers[None, :, :]
    angle = torch.atan2(displacement[..., 1], displacement[..., 0])
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    is_inside = distance <= radial_distance(angle, phases[None, :], radii[None, :])
    any_inside = is_inside.any(dim=1)
    # argmax returns the first true index; -1 encodes a non-colliding point.
    return torch.where(any_inside, is_inside.to(torch.int64).argmax(dim=1), -1)


def draw_results(samples, labels, centers, phases, radii, latent, lower, upper, weights, path):
    obstacle_colors = plt.get_cmap("tab10" if len(centers) <= 10 else "tab20").colors
    box_color = "#4a4a4a"
    fig, (ax_x, ax_z) = plt.subplots(1, 2, figsize=(13.5, 6.2), constrained_layout=True)

    angle = torch.linspace(-torch.pi, torch.pi, 601)
    for k, (center, phase, scale) in enumerate(zip(centers, phases, radii)):
        color = obstacle_colors[k % len(obstacle_colors)]
        rr = radial_distance(angle, phase, scale)
        boundary = (center[None, :] + torch.stack((rr * angle.cos(), rr * angle.sin()), dim=-1)).numpy()
        ax_x.add_patch(Polygon(boundary, closed=True, facecolor=color, edgecolor=color, alpha=0.24, linewidth=1.8))
        mask = (labels == k).numpy()
        ax_x.scatter(samples[mask, 0], samples[mask, 1], s=3.0, alpha=0.55, color=color, rasterized=True)
        ax_z.scatter(latent[mask, 0], latent[mask, 1], s=3.0, alpha=0.55, color=color, rasterized=True)

    span = np.maximum(upper.max(axis=0) - lower.min(axis=0), 1e-3)
    label_offset = 0.02 * span
    for k in range(len(lower)):
        ax_z.add_patch(Rectangle(
            (lower[k, 0], lower[k, 1]), *(upper[k] - lower[k]),
            fill=True, facecolor=box_color, edgecolor=box_color, linewidth=2.0, alpha=0.12,
        ))
        # Redraw the outline separately so it is legible above the points.
        ax_z.add_patch(Rectangle(
            (lower[k, 0], lower[k, 1]), *(upper[k] - lower[k]),
            fill=False, edgecolor=box_color, linewidth=2.2,
        ))
        ax_z.text(
            upper[k, 0] + label_offset[0], upper[k, 1] + label_offset[1],
            f"{weights[k]:.2f}", color=box_color, fontsize=8, ha="left", va="bottom",
        )

    ax_x.set(xlabel=r"$x_1$", ylabel=r"$x_2$", title="Original space: obstacles and colliding samples")
    ax_x.set_xlim(DOMAIN_LOW, DOMAIN_HIGH)
    ax_x.set_ylim(DOMAIN_LOW, DOMAIN_HIGH)
    ax_x.set_aspect("equal", adjustable="box")
    ax_z.set(xlabel=r"$z_1$", ylabel=r"$z_2$", title="Latent space: learned rectangles and mapped samples")
    set_padded_limits_2d(ax_z, np.concatenate([latent, lower, upper], axis=0))
    for ax in (ax_x, ax_z):
        ax.grid(alpha=0.14)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    ######## USER CONFIG ########
    n_collisions = 4000
    candidate_batch = 25000
    n_components = 20
    epochs = 500
    batch_size = 256
    num_layers = 5
    hidden_features = 64
    transform = "maf"  # "maf" or "nsf"; ignored when identity_transform is True
    identity_transform = True  # if True, fix flow as identity and train only the base boxes
    lr = 5e-3
    initial_sigma = 0.2
    sigma_weight = 10.0
    overlap_weight = 25.0
    overlap_margin = 0.02
    seed = 7
    dtype = "float64"  # "float32" or "float64"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    verbose = True

    # Obstacle locations: centers (x, y), radial phase, and base radius per obstacle.
    obstacle_centers = torch.tensor([
        [1.50181425, 0.38347614],
        [-1.5347614, 1.00181425],
        [-1.50181425, -0.38347614],
        [0.38347614, -1.50181425],
    ], dtype=torch.float32)
    obstacle_phases = torch.tensor([0.3, 2.01, 3.72, 5.43], dtype=torch.float32)
    obstacle_radii = torch.tensor([0.53, 0.56, 0.59, 0.53], dtype=torch.float32)

    ######## SETUP ########
    n_obstacles = int(obstacle_centers.shape[0])
    if n_obstacles < 1 or n_components < 1 or n_collisions < n_obstacles or candidate_batch < 1:
        raise ValueError(
            "Require n_obstacles >= 1, n_components >= 1, "
            "n_collisions >= n_obstacles and candidate_batch >= 1"
        )
    if not (
        obstacle_phases.shape[0] == n_obstacles and obstacle_radii.shape[0] == n_obstacles
        and obstacle_centers.shape[1] == 2
    ):
        raise ValueError("obstacle_centers/phases/radii must describe the same set of 2D obstacles")

    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator().manual_seed(seed)

    cfg = SimpleNamespace(
        epochs=epochs,
        batch_size=batch_size,
        num_layers=num_layers,
        hidden_features=hidden_features,
        transform=transform,
        identity_transform=identity_transform,
        lr=lr,
        initial_sigma=initial_sigma,
        sigma_weight=sigma_weight,
        overlap_weight=overlap_weight,
        overlap_margin=overlap_margin,
        dtype=dtype,
        device=device,
        verbose=verbose,
    )

    samples, labels, proposed = sample_collisions(
        n_collisions, candidate_batch, 2,
        lambda pts: obstacle_labels(pts, obstacle_centers, obstacle_phases, obstacle_radii),
        generator,
    )
    print(f"2D: {len(samples)} colliding points retained from {proposed} uniform workspace samples")
    fig_dir = figure_directory()
    latent, lower, upper, weights = fit_rectangular_flow(
        samples, n_components, cfg, fig_dir, "test_2D",
        generator=torch.Generator().manual_seed(seed),
    )
    draw_results(
        samples, labels, obstacle_centers, obstacle_phases, obstacle_radii,
        latent, lower, upper, weights, fig_dir / "test_2D.png",
    )


if __name__ == "__main__":
    main()
