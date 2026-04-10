"""
Runnable benchmark demo for numerical + visual outputs.

Run:
  PYTHONPATH=src python test/test_benchmark.py
"""

from __future__ import annotations

import json
import os

import matplotlib
import torch

from rational_factor.tools.benchmark import Benchmark


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def toy_experiment(scale: float, bias: float):
    value_a = torch.tensor([scale + bias, scale - bias], dtype=torch.float32)
    value_b = torch.tensor([[scale * bias, scale + 1.0]], dtype=torch.float32)
    value_c = torch.tensor([scale**2 + bias, scale**3 + bias, scale**4 + bias], dtype=torch.float32)

    fig, ax = plt.subplots()
    x = torch.linspace(0.0, 1.0, 25)
    y = scale * x + bias
    ax.plot(x.tolist(), y.tolist())
    ax.set_title(f"scale={scale}, bias={bias}")

    return value_a, value_b, value_c, fig


def main() -> None:
    save_first_trial_only = True

    benchmark = Benchmark(name="toy_benchmark")
    benchmark.set_experiment_fn(toy_experiment)

    contexts = [
        {"name": "low_scale", "params": {"scale": 1.0, "bias": 0.5}},
        {"name": "high_scale", "params": {"scale": 2.0, "bias": -0.25}},
    ]
    benchmark.set_contexts(contexts)

    benchmark.set_numerical_result(return_idx=0, tag="vector_metric")
    benchmark.set_numerical_result(return_idx=1, tag="matrix_metric")
    benchmark.set_numerical_result(
        return_idx=2,
        tag="trial_raw_only_metric",
        calc_mean=False,
        calc_std=False,
        json_raw_data=True,
    )
    benchmark.set_visual_result(
        return_idx=3,
        tag="line_plot",
        save_fig=True,
        save_first_trial_only=save_first_trial_only,
        file_extension="png",
    )

    trials = 3
    print("Running benchmark...")
    benchmark.run(trials=trials, verbose=False)
    run_dir = benchmark.process_and_save_results(root_dir="benchmark_data")
    print(f"Saved benchmark outputs to: {run_dir}")

    raw_data_path = os.path.join(run_dir, "raw_data.pt")
    processed_results_path = os.path.join(run_dir, "processed_results.json")
    assert os.path.isfile(raw_data_path)
    assert os.path.isfile(processed_results_path)

    raw_data = torch.load(raw_data_path)
    with open(processed_results_path, "r", encoding="utf-8") as f:
        processed = json.load(f)

    processed_ctx = processed["contexts"]
    for ctx in contexts:
        context_key = ctx["name"]
        assert context_key in raw_data
        assert context_key in processed_ctx
        assert processed_ctx[context_key]["name"] == context_key
        assert processed_ctx[context_key]["params"] == ctx["params"]
        assert raw_data[context_key]["vector_metric"].shape == (trials, 2)
        assert raw_data[context_key]["matrix_metric"].shape == (trials, 1, 2)
        assert raw_data[context_key]["trial_raw_only_metric"].shape == (trials, 3)

        raw_only_json = processed_ctx[context_key]["results"]["trial_raw_only_metric"]
        assert "raw_data" in raw_only_json
        assert len(raw_only_json["raw_data"]) == trials
        assert "mean" not in raw_only_json
        assert "std" not in raw_only_json

    figures_root = os.path.join(run_dir, "figures")
    assert os.path.isdir(figures_root)
    expected_figures_per_context = 1 if save_first_trial_only else trials
    for ctx in contexts:
        context_fig_dir = os.path.join(figures_root, ctx["name"])
        assert os.path.isdir(context_fig_dir)
        files = os.listdir(context_fig_dir)
        assert len(files) == expected_figures_per_context
        assert all(file_name.endswith(".png") for file_name in files)

    print("Benchmark demo completed successfully.")
    plt.close("all")


if __name__ == "__main__":
    main()
