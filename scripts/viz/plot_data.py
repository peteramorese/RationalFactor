
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def _context_sort_key(name: str) -> tuple[int, str]:
    m = re.match(r"^context_(\d+)$", name)
    if m:
        return (0, int(m.group(1)))
    return (1, name)


def _ordered_context_keys(contexts_raw: dict) -> list[str]:
    """Preserve JSON order; sort only legacy ``context_<i>`` keys by index."""
    keys = list(contexts_raw.keys())
    if keys and all(re.fullmatch(r"context_\d+", k) is not None for k in keys):
        return sorted(keys, key=_context_sort_key)
    return keys


def _context_display_label(entry: dict, key: str) -> str:
    n = entry.get("name")
    if isinstance(n, str) and n.strip():
        return n
    return key


def _load_processed(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _as_float_list(x) -> list[float]:
    if x is None:
        return []
    if isinstance(x, (int, float)):
        return [float(x)]
    if isinstance(x, list):
        return [float(v) for v in x]
    raise TypeError(f"expected number or list, got {type(x)}")


def _collect_field_names(contexts: dict) -> list[str]:
    names: set[str] = set()
    for data in contexts.values():
        res = data.get("results") or {}
        names.update(res.keys())
    return sorted(names)


def _vector_plot_length(
    contexts_raw: dict, context_keys: list[str], field: str
) -> int | None:
    n_plot: int | None = None
    for ckey in context_keys:
        res = (contexts_raw[ckey].get("results") or {}).get(field)
        if res is None:
            continue
        mean = _as_float_list(res.get("mean"))
        if len(mean) <= 1:
            continue
        n_plot = len(mean) if n_plot is None else min(n_plot, len(mean))
    if n_plot is None or n_plot <= 1:
        return None
    return n_plot


def _parse_index_range(range_spec: str, n_points: int) -> np.ndarray:
    """
    Parse an index selector for vector plots.

    Supports:
      - python-slice style "start:end" (end exclusive), with either side optional
      - single index "i"
    """
    spec = range_spec.strip()
    if not spec:
        raise ValueError("index range cannot be empty")

    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError("index range must use at most one ':' (e.g. 0:9)")
        start_str, end_str = parts
        start = int(start_str) if start_str.strip() else None
        end = int(end_str) if end_str.strip() else None
        idx = np.arange(n_points)[slice(start, end)]
    else:
        i = int(spec)
        idx = np.arange(n_points)[slice(i, i + 1)]

    if idx.size == 0:
        raise ValueError(
            f"index range '{range_spec}' selects no points for vector length {n_points}"
        )
    return idx


def _plot_vector_on_ax(
    ax: plt.Axes,
    contexts_raw: dict,
    context_keys: list[str],
    context_labels: list[str],
    field: str,
    colors: list,
    idx: np.ndarray,
    *,
    legend: bool,
) -> None:
    x = idx
    for i, ckey in enumerate(context_keys):
        res = (contexts_raw[ckey].get("results") or {}).get(field)
        if res is None:
            continue
        mean = _as_float_list(res.get("mean"))
        if len(mean) <= 1:
            continue
        y_full = np.asarray(mean, dtype=float)
        if np.max(idx) >= y_full.size:
            continue
        y = y_full[idx]
        color = colors[i]
        std = res.get("std")
        if std is not None:
            yerr_full = np.asarray(_as_float_list(std), dtype=float)
            if np.max(idx) < yerr_full.size:
                yerr = yerr_full[idx]
            else:
                yerr = np.asarray([])
            if yerr.shape == y.shape:
                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    marker="o",
                    markersize=3,
                    capsize=2,
                    linestyle="-",
                    color=color,
                    label=context_labels[i] if legend else None,
                )
            else:
                ax.plot(
                    x,
                    y,
                    marker="o",
                    markersize=3,
                    linestyle="-",
                    color=color,
                    label=context_labels[i] if legend else None,
                )
        else:
            ax.plot(
                x,
                y,
                marker="o",
                markersize=3,
                linestyle="-",
                color=color,
                label=context_labels[i] if legend else None,
            )

    ax.set_xlabel("index")
    ax.set_ylabel("mean")
    ax.set_title(field)
    if legend:
        ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_scalar_on_ax(
    ax: plt.Axes,
    contexts_raw: dict,
    context_keys: list[str],
    context_labels: list[str],
    field: str,
    colors: list,
    *,
    legend: bool,
) -> None:
    n_ctx = len(context_keys)
    heights = np.full(n_ctx, np.nan)
    yerr = np.full(n_ctx, np.nan)
    has_err = np.zeros(n_ctx, dtype=bool)

    for i, ckey in enumerate(context_keys):
        res = (contexts_raw[ckey].get("results") or {}).get(field)
        if res is None:
            continue
        mean = _as_float_list(res.get("mean"))
        if not mean:
            continue
        heights[i] = mean[0]
        std = res.get("std")
        if std is not None:
            s = _as_float_list(std)
            if s:
                yerr[i] = s[0]
                has_err[i] = True

    x = np.arange(n_ctx)
    yerr_plot = np.ma.masked_where(~has_err, yerr)
    use_err = bool(np.any(has_err))
    for i in range(n_ctx):
        if np.isnan(heights[i]):
            continue
        err_i = yerr_plot[i] if use_err and has_err[i] else None
        ax.bar(
            x[i],
            heights[i],
            width=0.75,
            color=colors[i],
            yerr=err_i,
            capsize=2 if err_i is not None else 0,
            label=context_labels[i] if legend else None,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(context_labels, rotation=15, ha="right")
    ax.set_ylabel("mean")
    ax.set_title(field)
    if legend:
        ax.legend()
    ax.grid(True, axis="y", alpha=0.3)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot mean (and optional std) from a benchmark processed_results.json."
    )
    parser.add_argument(
        "benchmark_folder",
        help="Folder name under benchmark_data/ (e.g. nftf_comparison_y2026-m04-d09_H21-M11-S32)",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=None,
        help="Root directory containing benchmark_data/ (default: repo root inferred from script)",
    )
    parser.add_argument(
        "--vector-field",
        default=None,
        help=(
            "Plot only this vector-valued numerical field "
            "(e.g. avg_log_likelihood_per_timestep)"
        ),
    )
    parser.add_argument(
        "--index-range",
        default=None,
        help=(
            "Subset vector indices to plot; python-slice style start:end (end exclusive) "
            "or single index, e.g. 0:9 or 5"
        ),
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    benchmark_root = args.benchmark_root or repo_root
    data_dir = benchmark_root / "benchmark_data" / args.benchmark_folder
    json_path = data_dir / "processed_results.json"
    if not json_path.is_file():
        print(f"error: missing {json_path}", file=sys.stderr)
        return 1

    payload = _load_processed(json_path)
    contexts_raw = payload.get("contexts") or {}
    if not contexts_raw:
        print("error: no contexts in processed_results.json", file=sys.stderr)
        return 1

    context_keys = _ordered_context_keys(contexts_raw)
    context_labels = [_context_display_label(contexts_raw[k], k) for k in context_keys]
    n_ctx = len(context_keys)

    figures_dir = repo_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    safe_slug = re.sub(r"[^\w.\-]+", "_", args.benchmark_folder)

    field_names = _collect_field_names(contexts_raw)

    vector_fields: list[str] = []
    scalar_fields: list[str] = []

    for field in field_names:
        lens = []
        for ckey in context_keys:
            res = (contexts_raw[ckey].get("results") or {}).get(field)
            if res is None:
                continue
            mean = _as_float_list(res.get("mean"))
            lens.append(len(mean))
        if not lens:
            continue
        n_min = min(lens)
        if n_min > 1:
            vector_fields.append(field)
        else:
            scalar_fields.append(field)

    subplot_specs: list[tuple[str, str]] = []
    if args.vector_field is not None:
        if args.vector_field not in vector_fields:
            available = ", ".join(vector_fields) if vector_fields else "<none>"
            print(
                f"error: vector field '{args.vector_field}' not found. "
                f"Available vector fields: {available}",
                file=sys.stderr,
            )
            return 1
        if _vector_plot_length(contexts_raw, context_keys, args.vector_field) is not None:
            subplot_specs.append(("vector", args.vector_field))
    else:
        for f in vector_fields:
            if _vector_plot_length(contexts_raw, context_keys, f) is not None:
                subplot_specs.append(("vector", f))
    for f in scalar_fields:
        subplot_specs.append(("scalar", f))

    if not subplot_specs:
        print("error: no plottable metrics", file=sys.stderr)
        return 1

    n_sub = len(subplot_specs)
    ncols = min(3, n_sub) if n_sub > 1 else 1
    nrows = math.ceil(n_sub / ncols)

    cycle = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    colors = [cycle[i % len(cycle)] for i in range(n_ctx)]

    fig_w = 5.2 * ncols
    fig_h = 3.4 * nrows + 1.0
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    bench_title = str(payload.get("name", "benchmark"))
    fig.suptitle(f"{bench_title} ({args.benchmark_folder})", fontsize=12, y=1.02)

    for idx, (kind, field) in enumerate(subplot_specs):
        ax = axes_flat[idx]
        if kind == "vector":
            n_plot = _vector_plot_length(contexts_raw, context_keys, field)
            assert n_plot is not None
            if args.index_range is not None:
                try:
                    idx = _parse_index_range(args.index_range, n_plot)
                except ValueError as e:
                    print(f"error: {e}", file=sys.stderr)
                    return 1
            else:
                idx = np.arange(n_plot)
            _plot_vector_on_ax(
                ax,
                contexts_raw,
                context_keys,
                context_labels,
                field,
                colors,
                idx,
                legend=False,
            )
        else:
            _plot_scalar_on_ax(
                ax,
                contexts_raw,
                context_keys,
                context_labels,
                field,
                colors,
                legend=False,
            )

    for j in range(n_sub, len(axes_flat)):
        axes_flat[j].set_visible(False)

    legend_handles = [
        Patch(facecolor=colors[i], edgecolor="0.35", linewidth=0.8, label=context_labels[i])
        for i in range(n_ctx)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(n_ctx, 6),
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    out_path = figures_dir / f"{safe_slug}__summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
