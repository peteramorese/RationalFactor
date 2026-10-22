"""Configurable training for torch models with multiple datasets and optimizers.

A training epoch consists of ``n_iterations`` full passes through the training
loaders for each optimizer, in registration order. Multiple training loaders
are zipped (the shortest determines the number of gradient steps). Validation
loaders are evaluated separately, once per outer epoch, without gradients.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from collections.abc import Mapping
import sys

import torch


class TrainingTimer:
    """Estimate remaining time for repeated groups of full training passes."""

    def __init__(self, n_groups: int, iterations: int, epochs_per_group: int | list[int]):
        self.n_groups = n_groups
        self.iterations = iterations
        self.epochs_per_group = (
            list(epochs_per_group)
            if isinstance(epochs_per_group, list)
            else [epochs_per_group] * n_groups
        )
        if len(self.epochs_per_group) != n_groups:
            raise ValueError("epochs_per_group must contain one entry per optimizer")
        self.total_epochs = iterations * sum(self.epochs_per_group)

    def initialize(self):
        self.times = [0.0] * self.n_groups
        self.completed_epochs = [0] * self.n_groups
        self.curr_date_time = time.time()
        self.curr_iter = self.curr_group = self.curr_epoch = 0

    def update_epoch(self, time_taken: float | None = None):
        self.times[self.curr_group] += (
            time.time() - self.curr_date_time if time_taken is None else time_taken
        )
        self.completed_epochs[self.curr_group] += 1
        self.curr_epoch += 1
        self.curr_date_time = time.time()

    def update_group(self):
        self.curr_group += 1
        self.curr_epoch = 0

    def update_iteration(self):
        self.curr_iter += 1
        self.curr_group = self.curr_epoch = 0

    def get_predicted_time(self):
        remaining = sum(
            (self.iterations * n - count) * (elapsed / count)
            for n, count, elapsed in zip(
                self.epochs_per_group, self.completed_epochs, self.times
            )
            if count > 0
        )
        remaining = max(0.0, remaining)
        return timedelta(seconds=remaining), datetime.now() + timedelta(seconds=remaining)

    def get_predicted_etr_str(self):
        remaining, _ = self.get_predicted_time()
        hours, rest = divmod(int(remaining.total_seconds()), 3600)
        minutes, seconds = divmod(rest, 60)
        return f"{hours}:{minutes:02d}:{seconds:02d}"

    def total_since_start(self):
        return sum(self.times)


class LossVisualizer:
    """Periodically save loss-versus-epoch plots to a PNG file.

    ``update`` caches observations; the figure is written only after
    ``buffer_size`` new epochs. ``flush`` writes any incomplete final buffer,
    including when training stops early. The output file is replaced atomically
    to avoid leaving a partially written figure.

    The first panel contains every training loss and ``total``. If validation
    losses are registered, a second panel contains every validation loss and
    ``validation_total``. Plotting uses matplotlib's non-interactive Agg canvas.
    """

    def __init__(
        self,
        directory: str | Path,
        buffer_size: int = 1,
        *,
        filename: str = "losses.png",
        dpi: int = 140,
    ) -> None:
        if not isinstance(buffer_size, int) or isinstance(buffer_size, bool) or buffer_size < 1:
            raise ValueError("buffer_size must be a positive integer")
        if not isinstance(dpi, int) or dpi < 1:
            raise ValueError("dpi must be a positive integer")
        if not isinstance(filename, str) or not filename or Path(filename).name != filename or not filename.lower().endswith(".png"):
            raise ValueError("filename must be a PNG filename without directories")
        self.directory = Path(directory)
        self.path = self.directory / filename
        self.buffer_size = buffer_size
        self.dpi = dpi
        self.history: list[dict[str, float]] = []
        self._buffer: list[dict[str, float]] = []
        self.training_labels: tuple[str, ...] = ()
        self.validation_labels: tuple[str, ...] = ()
        self.figure = None
        self.axes: tuple = ()
        self.save_count = 0
        self._initialized = False

    def initialize(self, training_labels: Iterable[str], validation_labels: Iterable[str] = ()) -> None:
        """Start a fresh run after all Trainer loss functions are registered."""
        self.training_labels = tuple(training_labels)
        self.validation_labels = tuple(validation_labels)
        if not self.training_labels:
            raise ValueError("At least one training loss is required")
        self.history = []
        self._buffer = []
        self.save_count = 0
        self.figure = None
        self.axes = ()
        self.directory.mkdir(parents=True, exist_ok=True)
        # Do not leave an outdated plot from a previous run masquerading as the
        # current run's trajectory while the first buffer is accumulating.
        if self.path.exists():
            self.path.unlink()
        self._initialized = True

    @property
    def buffered_epochs(self) -> int:
        return len(self._buffer)

    def update(self, epoch: int, metrics: Mapping[str, float]) -> None:
        """Record one completed epoch, saving only when the buffer fills."""
        if not self._initialized:
            raise RuntimeError("Call initialize() before update()")
        labels = (*self.training_labels, "total")
        if self.validation_labels:
            labels += (*self.validation_labels, "validation_total")
        missing = [label for label in labels if label not in metrics]
        if missing:
            raise KeyError(f"Missing loss metrics: {missing}")
        if (self._buffer or self.history) and epoch <= (
            self._buffer[-1]["epoch"] if self._buffer else self.history[-1]["epoch"]
        ):
            raise ValueError("epoch must be strictly increasing")
        entry = {"epoch": int(epoch)}
        entry.update({label: float(metrics[label]) for label in labels})
        self._buffer.append(entry)
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Write the pending epochs, if any (also called after early stopping)."""
        if not self._buffer:
            return
        self.history.extend(self._buffer)
        self._buffer.clear()
        self._save_plot()

    def _save_plot(self) -> None:
        # Lazy imports: Trainer itself does not require matplotlib unless a
        # visualizer is used. FigureCanvasAgg works in headless environments.
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.ticker import MaxNLocator

        has_validation = bool(self.validation_labels)
        fig = Figure(figsize=(12, 4.8) if has_validation else (7, 4.8), dpi=self.dpi)
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, 2 if has_validation else 1, squeeze=False)[0]
        epochs = [entry["epoch"] for entry in self.history]

        def panel(ax, labels, title, total_label):
            for label in (*labels, total_label):
                values = [entry[label] for entry in self.history]
                ax.plot(epochs, values, label=label,
                        linewidth=2.2 if label == total_label else 1.35,
                        linestyle="--" if label == total_label else "-")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=0.22)
            ax.legend(loc="best", fontsize="small")

        panel(axes[0], self.training_labels, "Training losses", "total")
        if has_validation:
            panel(axes[1], self.validation_labels, "Validation losses", "validation_total")
        fig.tight_layout()
        temporary = self.path.with_name(f".{self.path.stem}.tmp.png")
        try:
            fig.savefig(temporary, format="png", dpi=self.dpi)
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
        self.figure = fig
        self.axes = tuple(axes)
        self.save_count += 1


class TrainingProgressBar:
    """Terminal progress indicator backed by TrainingTimer's measured ETA.

    Completion counts *optimizer passes* instead of outer epochs, so
    n_iterations for each optimizer is accounted for. In an interactive
    terminal it refreshes after each pass; elsewhere it renders in the
    epoch-level log line to avoid flooding captured logs.
    """

    def __init__(self, width: int = 24, stream=None) -> None:
        if not isinstance(width, int) or width < 1:
            raise ValueError("width must be a positive integer")
        self.width = width
        self.stream = stream
        self._last_length = 0

    def _stream(self):
        return sys.stdout if self.stream is None else self.stream

    def render(self, timer: TrainingTimer) -> str:
        completed = sum(timer.completed_epochs)
        total = timer.total_epochs
        fraction = min(1.0, completed / total) if total else 1.0
        filled = min(self.width, int(fraction * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        # Before each optimizer has finished one pass, estimates ignore the
        # unmeasured groups. Don't present that partial estimate as a real ETA.
        etr = (timer.get_predicted_etr_str()
               if all(n > 0 for n in timer.completed_epochs) else "--:--:--")
        return f"[{bar}] {fraction * 100:5.1f}% ({completed}/{total} passes) ETR {etr}"

    def update(self, timer: TrainingTimer) -> None:
        stream = self._stream()
        if not getattr(stream, "isatty", lambda: False)():
            return
        msg = self.render(timer)
        stream.write("\r" + msg + " " * max(0, self._last_length - len(msg)))
        stream.flush()
        self._last_length = len(msg)

    def log_epoch(self, message: str, timer: TrainingTimer) -> None:
        stream = self._stream()
        msg = f"{message} | {self.render(timer)}"
        if getattr(stream, "isatty", lambda: False)():
            stream.write("\r" + msg + " " * max(0, self._last_length - len(msg)) + "\n")
        else:
            stream.write(msg + "\n")
        stream.flush()
        self._last_length = 0


@dataclass(frozen=True)
class _OptimizerSpec:
    optimizer: torch.optim.Optimizer
    n_iterations: int


@dataclass(frozen=True)
class _LossSpec:
    fn: Callable
    dataset_label: str | None
    validation: bool
    feasibility: bool = False


def _to_device(batch, device: torch.device):
    if isinstance(batch, dict):
        return {key: _to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [_to_device(value, device) for value in batch]
    return batch.to(device) if hasattr(batch, "to") else batch


def _apply_loss(fn: Callable, model: torch.nn.Module, batch):
    if batch is None:
        loss = fn(model)
    elif isinstance(batch, dict):
        loss = fn(model, **batch)
    elif isinstance(batch, (tuple, list)):
        loss = fn(model, *batch)
    else:
        loss = fn(model, batch)
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise ValueError("Each loss function must return a scalar torch.Tensor")
    return loss.reshape(())


class Trainer:
    """Train a model using labeled datasets, optimizers, and additive losses.

    ``add_loss_function(fn, label, dataset_label)`` expects a callable of the
    form ``fn(model, *batch)`` (or ``fn(model, **batch)`` for dict batches).
    ``dataset_label=None`` allows data-independent regularization terms of the
    form ``fn(model)``; these are evaluated once per gradient step.

    ``feasibility=True`` marks a model-only training penalty as a hard
    checkpoint constraint. It is still differentiated and included in the
    total training loss. A checkpoint is eligible only when *every* feasibility
    loss, reevaluated on the post-epoch model, is finite and exactly zero.
    An infeasible checkpoint is never saved, even when its training score is
    better. If none is found, ``train`` returns ``(None, inf, elapsed)`` and
    prints a warning. Without feasibility losses the original behavior applies.

    Training datasets are zipped at each gradient step and the shortest loader
    determines the number of steps in a pass. Each optimizer's ``n_iterations``
    is the number of complete passes it makes per outer epoch. Training losses
    are added before backward; validation losses are evaluated at the end of
    each outer epoch, and never participate in backward.

    History stores one dictionary per *outer* epoch. ``total`` is the mean sum
    of training losses across gradient steps; ``validation_total`` is the sum
    of per-validation-loss means. Early stopping always monitors
    ``validation_total`` regardless of the ``use_best`` checkpoint metric.
    An optional LossVisualizer saves loss plots every ``buffer_size`` epochs.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        validation_early_stopping_patience: int | None = None,
        *,
        device: torch.device | str | None = None,
        loss_visualizer: LossVisualizer | None = None,
    ):
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        self.model = model
        self.device = torch.device(device) if device is not None else None
        self.loss_visualizer: LossVisualizer | None = None
        self.set_loss_visualizer(loss_visualizer)
        self.datasets: dict[str, Iterable] = {}
        self.optimizers: dict[str, _OptimizerSpec] = {}
        self.loss_functions: dict[str, _LossSpec] = {}
        self.history: list[dict[str, object]] = []
        self.subiteration_history: list[dict] = []
        self.best_loss = float("inf")
        self.best_state: dict | None = None
        self.best_epoch: int | None = None
        self.early_stopped = False
        self.training_timer: TrainingTimer | None = None
        self.validation_early_stopping_patience = validation_early_stopping_patience

    def set_loss_visualizer(self, visualizer: LossVisualizer | None) -> "Trainer":
        if visualizer is not None and not isinstance(visualizer, LossVisualizer):
            raise TypeError("loss_visualizer must be a LossVisualizer or None")
        self.loss_visualizer = visualizer
        return self

    @property
    def validation_early_stopping_patience(self) -> int | None:
        return self._validation_early_stopping_patience

    @validation_early_stopping_patience.setter
    def validation_early_stopping_patience(self, patience: int | None):
        if patience is not None and (not isinstance(patience, int) or patience < 1):
            raise ValueError("validation_early_stopping_patience must be positive or None")
        self._validation_early_stopping_patience = patience

    def add_dataset(self, data_loader: Iterable, label: str) -> "Trainer":
        if label in self.datasets:
            raise ValueError(f"Dataset label {label!r} is already registered")
        if not isinstance(label, str) or not label:
            raise ValueError("Dataset label must be a nonempty string")
        self.datasets[label] = data_loader
        return self

    def add_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        label: str,
        n_iterations: int = 1,
    ) -> "Trainer":
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer")
        if not isinstance(n_iterations, int) or n_iterations < 1:
            raise ValueError("n_iterations must be a positive integer")
        if not isinstance(label, str) or not label or label in self.optimizers:
            raise ValueError(f"Invalid or duplicate optimizer label {label!r}")
        self.optimizers[label] = _OptimizerSpec(optimizer, n_iterations)
        return self

    def add_loss_function(
        self,
        fn: Callable,
        label: str,
        dataset_label: str | None,
        validation: bool = False,
        feasibility: bool = False,
    ) -> "Trainer":
        """Register a training or validation loss.

        Feasibility losses must be model-only (``dataset_label=None``) and
        non-validation. They contribute to the differentiated training total,
        but a saved checkpoint additionally requires each feasibility loss to
        be finite and exactly zero when reevaluated after the epoch.
        """
        if not callable(fn):
            raise TypeError("fn must be callable")
        if not isinstance(label, str) or not label or label in self.loss_functions:
            raise ValueError(f"Invalid or duplicate loss label {label!r}")
        if label in {"total", "validation_total"}:
            raise ValueError("'total' and 'validation_total' are reserved loss labels")
        if feasibility and dataset_label is not None:
            raise ValueError("Feasibility losses must have dataset_label=None")
        if feasibility and validation:
            raise ValueError("Feasibility losses are training losses, not validation losses")
        if dataset_label is not None and dataset_label not in self.datasets:
            raise KeyError(f"Unknown dataset {dataset_label!r}; call add_dataset first")
        self.loss_functions[label] = _LossSpec(fn, dataset_label, bool(validation), bool(feasibility))
        return self

    @torch.no_grad()
    def _check_feasibility(self, constraints: dict[str, _LossSpec]) -> tuple[bool, dict[str, float]]:
        """Check the actual post-update state, not epoch-average loss values.

        Require each scalar feasibility loss to be finite and exactly zero:
        summed losses could cancel, and losses averaged during optimization
        describe earlier parameter states rather than the checkpoint state.
        """
        values = {
            label: float(_apply_loss(spec.fn, self.model, None))
            for label, spec in constraints.items()
        }
        return all(math.isfinite(value) and value == 0.0 for value in values.values()), values

    def _model_device(self) -> torch.device:
        if self.device is not None:
            return self.device
        tensor = next(self.model.parameters(), None)
        if tensor is None:
            tensor = next(self.model.buffers(), None)
        return tensor.device if tensor is not None else torch.device("cpu")

    def _validate_configuration(self, use_best: str):
        if not self.optimizers:
            raise ValueError("Register at least one optimizer")
        if not any(not spec.validation for spec in self.loss_functions.values()):
            raise ValueError("Register at least one non-validation loss")
        if self.validation_early_stopping_patience is not None and not any(
            spec.validation for spec in self.loss_functions.values()
        ):
            raise ValueError("Early stopping requires at least one validation loss")
        selectable = {"total", *self.loss_functions}
        if any(spec.validation for spec in self.loss_functions.values()):
            selectable.add("validation_total")
        if use_best not in selectable:
            raise ValueError(f"use_best must be one of {sorted(selectable)}")

    def _train_batches(self):
        labels = list(dict.fromkeys(
            spec.dataset_label for spec in self.loss_functions.values()
            if not spec.validation and spec.dataset_label is not None
        ))
        if not labels:
            yield {}
            return
        for batches in zip(*(self.datasets[label] for label in labels)):
            yield {
                label: _to_device(batch, self._model_device())
                for label, batch in zip(labels, batches)
            }

    def _train_pass(
        self,
        optimizer: torch.optim.Optimizer,
        train_losses: dict[str, _LossSpec],
        clip_grad_norm: float | None,
    ) -> tuple[dict[str, float], int]:
        sums = {label: 0.0 for label in train_losses}
        sums["total"] = 0.0
        steps = 0
        for batches in self._train_batches():
            # Avoid gradient accumulation in parameters owned by other optimizers.
            self.model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            components = {
                label: _apply_loss(spec.fn, self.model, batches.get(spec.dataset_label))
                for label, spec in train_losses.items()
            }
            total = torch.stack(list(components.values())).sum()
            if not bool(torch.isfinite(total).item()):
                # Do not take a gradient step with a non-finite loss.
                for label, loss in components.items():
                    sums[label] += float(loss.detach())
                sums["total"] += float(total.detach())
                steps += 1
                continue
            if total.requires_grad:
                total.backward()
                if clip_grad_norm is not None:
                    active_params = [
                        p for group in optimizer.param_groups for p in group["params"]
                        if p.requires_grad and p.grad is not None
                    ]
                    if active_params:
                        torch.nn.utils.clip_grad_norm_(active_params, max_norm=clip_grad_norm)
                optimizer.step()
            else:
                raise RuntimeError("The summed training loss is not connected to trainable parameters")
            for label, loss in components.items():
                sums[label] += float(loss.detach())
            sums["total"] += float(total.detach())
            steps += 1
        if steps == 0:
            raise RuntimeError("Training datasets produced no batches (zip stops at the shortest loader)")
        return sums, steps

    @torch.no_grad()
    def _validate(self, val_losses: dict[str, _LossSpec]) -> dict[str, float]:
        if not val_losses:
            return {}
        sums = {}
        # Only the datasets used by validation losses are evaluated. Each is
        # exhausted independently, so loaders of different lengths work.
        for dataset_label in dict.fromkeys(spec.dataset_label for spec in val_losses.values()):
            selected = {
                label: spec for label, spec in val_losses.items()
                if spec.dataset_label == dataset_label
            }
            if dataset_label is None:
                sums.update({
                    label: float(_apply_loss(spec.fn, self.model, None))
                    for label, spec in selected.items()
                })
                continue
            accum = {label: 0.0 for label in selected}
            n_batches = 0
            for batch in self.datasets[dataset_label]:
                batch = _to_device(batch, self._model_device())
                for label, spec in selected.items():
                    accum[label] += float(_apply_loss(spec.fn, self.model, batch))
                n_batches += 1
            if not n_batches:
                raise RuntimeError(f"Validation dataset {dataset_label!r} is empty")
            sums.update({label: value / n_batches for label, value in accum.items()})
        sums["validation_total"] = sum(sums.values())
        return sums

    def train(
        self,
        epochs: int,
        use_best: str = "total",
        verbose: bool = True,
        clip_grad_norm: float | None = 5.0,
        restore_loss_threshold: float | None = 50.0,
    ) -> tuple[torch.nn.Module | None, float, float]:
        """Train and restore the best eligible checkpoint according to ``use_best``.

        Feasibility losses are included in the optimization objective, but
        checkpoint eligibility requires that every such loss evaluate to
        exactly zero *after* the epoch's optimizer updates. If no feasible
        checkpoint was saved, return ``None`` in place of the model.

        Early-stopping patience monitors ``validation_total``. Non-finite or
        above-threshold training loss restores the best eligible checkpoint
        when available.
        """
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if clip_grad_norm is not None and clip_grad_norm <= 0:
            raise ValueError("clip_grad_norm must be positive or None")
        self._validate_configuration(use_best)
        train_losses = {k: v for k, v in self.loss_functions.items() if not v.validation}
        val_losses = {k: v for k, v in self.loss_functions.items() if v.validation}
        feasibility_losses = {k: v for k, v in train_losses.items() if v.feasibility}
        if self.loss_visualizer is not None:
            self.loss_visualizer.initialize(train_losses, val_losses)
        progress_bar = TrainingProgressBar() if verbose else None
        self.history = []
        self.subiteration_history = []
        self.best_loss = float("inf")
        self.best_state = None
        self.best_epoch = None
        self.early_stopped = False
        best_validation = float("inf")
        no_improvement = 0
        start = time.monotonic()
        previous_training_mode = self.model.training
        original_grad_flags = {p: p.requires_grad for p in self.model.parameters()}
        self.training_timer = TrainingTimer(
            n_groups=len(self.optimizers),
            iterations=epochs,
            epochs_per_group=[s.n_iterations for s in self.optimizers.values()],
        )
        self.training_timer.initialize()
        self.model.train()

        try:
            for epoch in range(epochs):
                epoch_start = time.monotonic()
                sums = {label: 0.0 for label in train_losses}
                sums["total"] = 0.0
                steps = 0
                for opt_label, opt_spec in self.optimizers.items():
                    optimizer = opt_spec.optimizer
                    opt_params = {p for g in optimizer.param_groups for p in g["params"]}
                    for param, initially_trainable in original_grad_flags.items():
                        param.requires_grad_(initially_trainable and param in opt_params)
                    for sub_iteration in range(opt_spec.n_iterations):
                        pass_start = time.monotonic()
                        part, n_steps = self._train_pass(optimizer, train_losses, clip_grad_norm)
                        self.subiteration_history.append({
                            "epoch": epoch + 1,
                            "optimizer": opt_label,
                            "sub_iteration": sub_iteration + 1,
                            **{label: value / n_steps for label, value in part.items()},
                        })
                        for label, value in part.items():
                            sums[label] += value
                        steps += n_steps
                        self.training_timer.update_epoch(time.monotonic() - pass_start)
                        if progress_bar is not None:
                            progress_bar.update(self.training_timer)
                    self.training_timer.update_group()

                metrics = {label: value / steps for label, value in sums.items()}
                for param, flag in original_grad_flags.items():
                    param.requires_grad_(flag)
                if val_losses:
                    self.model.eval()
                    try:
                        metrics.update(self._validate(val_losses))
                    finally:
                        self.model.train()

                # A training-pass average is not the loss at the model state
                # being checkpointed. Recheck all feasibility constraints now.
                feasible, feasibility_values = self._check_feasibility(feasibility_losses)
                model_valid = bool(self.model.valid()) if hasattr(self.model, "valid") else True
                valid = model_valid and feasible
                score = metrics[use_best]
                unstable = not math.isfinite(metrics["total"]) or (
                    restore_loss_threshold is not None
                    and metrics["total"] > restore_loss_threshold
                )
                if valid and not unstable and math.isfinite(score) and score < self.best_loss:
                    self.best_loss = score
                    self.best_state = deepcopy(self.model.state_dict())
                    self.best_epoch = epoch + 1

                if val_losses:
                    validation_score = metrics["validation_total"]
                    if math.isfinite(validation_score) and validation_score < best_validation:
                        best_validation = validation_score
                        no_improvement = 0
                    else:
                        no_improvement += 1

                if unstable and self.best_state is not None:
                    self.model.load_state_dict(self.best_state)

                metrics["epoch"] = epoch + 1
                metrics["valid"] = valid
                if feasibility_losses:
                    metrics["feasible"] = feasible
                    # Make the values used for checkpoint eligibility visible
                    # separately from epoch-averaged training loss trajectories.
                    metrics["feasibility_values"] = feasibility_values
                self.history.append(metrics)
                if self.loss_visualizer is not None:
                    self.loss_visualizer.update(epoch + 1, metrics)
                if verbose:
                    details = ", ".join(
                        f"{label}:{metrics[label]:.4f}" for label in self.loss_functions
                    )
                    validation = (
                        f", validation_total:{metrics['validation_total']:.4f}"
                        if val_losses else ""
                    )
                    status = " <invalid>" if not valid else ""
                    if unstable:
                        status += " <restored>" if self.best_state is not None else " <unstable>"
                    progress_bar.log_epoch(
                        f"Epoch {epoch+1}/{epochs}, total:{metrics['total']:.4f}, "
                        f"{details}{validation}{status}, time:{time.monotonic()-epoch_start:.2f}s",
                        self.training_timer,
                    )
                self.training_timer.update_iteration()
                if (self.validation_early_stopping_patience is not None
                        and no_improvement >= self.validation_early_stopping_patience):
                    self.early_stopped = True
                    if verbose:
                        print(f"Early stopping: validation_total did not improve for {no_improvement} epochs")
                    break
        finally:
            # Always restore model state even if writing the visualization fails.
            for param, flag in original_grad_flags.items():
                param.requires_grad_(flag)
            if self.best_state is not None:
                self.model.load_state_dict(self.best_state)
            self.model.train(previous_training_mode)
            if self.loss_visualizer is not None:
                self.loss_visualizer.flush()

        elapsed = time.monotonic() - start
        if feasibility_losses and self.best_state is None:
            print(
                "WARNING: No feasible checkpoint was found: every feasibility "
                "loss must equal exactly zero at an epoch boundary (and the "
                "model must otherwise be valid). Returning no model."
            )
            if verbose:
                print(f"Completed in {elapsed:.2f}s; no feasible model")
            return None, self.best_loss, elapsed
        if verbose:
            print(f"Completed in {elapsed:.2f}s; best {use_best}: {self.best_loss:.6g}")
        return self.model, self.best_loss, elapsed


@torch.no_grad()
def fit_gaussian_lf_em(
    z: torch.Tensor,
    n_components: int,
    reg_covar: float = 1e-5,
    max_iter: int = 100,
    warm_start: bool = False,
):
    """Fit a diagonal Gaussian mixture and wrap it in the existing LinearForm API.

    This independent utility is retained for compatibility; sklearn and the
    Gaussian/LinearForm dependencies are imported only when it is called.
    """
    from sklearn.mixture import GaussianMixture

    from .basis_functions import GaussianBasis
    from .factor_forms import LinearForm
    from .parameters import FixedParameters, PositiveParameters

    if z.ndim != 2 or not z.is_floating_point():
        raise ValueError("z must be a floating-point tensor of shape (N, D)")
    if n_components < 1 or reg_covar < 0 or max_iter < 1:
        raise ValueError("Require n_components >= 1, reg_covar >= 0, max_iter >= 1")
    mixture = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        reg_covar=reg_covar,
        max_iter=max_iter,
        warm_start=warm_start,
    )
    mixture.fit(z.detach().cpu().numpy())
    means = torch.as_tensor(mixture.means_, dtype=z.dtype, device=z.device).T[None]
    stds = torch.as_tensor(mixture.covariances_, dtype=z.dtype, device=z.device).sqrt().T[None]
    weights = torch.as_tensor(mixture.weights_, dtype=z.dtype, device=z.device)[None]
    basis = GaussianBasis(
        mean_params=FixedParameters(means),
        std_params=PositiveParameters.from_values(stds, trainable=False),
        coeffs=PositiveParameters.from_values(weights, trainable=False),
    )
    return LinearForm(w=basis)
