"""Summarize cartpole_sp_ablation benchmark results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main(run_dir: str) -> None:
    p = Path(run_dir)
    data = json.loads((p / "processed_results.json").read_text())
    names = list(data["contexts"].keys())
    print("run_dir:", p)
    print("contexts:", names)

    rows = []
    for cname in names:
        ctx = data["contexts"][cname]
        res = ctx["results"]
        ll_mean = np.asarray(res["avg_log_likelihood_per_timestep"]["mean"], dtype=float)
        ll_std = np.asarray(res["avg_log_likelihood_per_timestep"]["std"], dtype=float)
        mean_ll = float(res["mean_avg_log_likelihood"]["mean"][0])
        mean_ll_std = float(res["mean_avg_log_likelihood"]["std"][0])
        tran = float(res["best_loss_transition"]["mean"][0])
        init = float(res["best_loss_initial"]["mean"][0])
        t_tran = float(res["training_time_transition"]["mean"][0])
        t_init = float(res["training_time_initial"]["mean"][0])
        rows.append(
            {
                "name": cname,
                "n_valid": ctx.get("n_trials_valid"),
                "ll_t": ll_mean,
                "ll_t_std": ll_std,
                "mean_ll": mean_ll,
                "mean_ll_std": mean_ll_std,
                "tran_loss": tran,
                "init_loss": init,
                "t_tran": t_tran,
                "t_init": t_init,
                "params": ctx["params"],
            }
        )
        print(f"\n=== {cname} (n_valid={ctx.get('n_trials_valid')}) ===")
        print(
            f"  sp_basis={ctx['params']['sp_basis']} sp_model={ctx['params']['sp_model']} "
            f"n_leaf={ctx['params']['n_leaf_basis']} n_out={ctx['params']['n_output_basis']}"
        )
        print(f"  mean belief LL: {mean_ll:.4f} ± {mean_ll_std:.4f}")
        print(f"  LL@t0: {ll_mean[0]:.4f}  LL@t-1: {ll_mean[-1]:.4f}")
        print(f"  per-t mean: " + " ".join(f"{v:7.3f}" for v in ll_mean))
        print(f"  tran/init val-MLE: {tran:.4f} / {init:.4f}")
        print(f"  train time tran/init (s): {t_tran:.1f} / {t_init:.1f}")

    base = next(r for r in rows if r["name"] == "baseline")
    print("\n=== deltas vs baseline (mean LL, higher better) ===")
    for r in rows:
        d = r["mean_ll"] - base["mean_ll"]
        print(f"  {r['name']:16s}: {d:+.4f}  ({100*d/abs(base['mean_ll']):+.2f}% of |baseline|)")

    print("\n=== per-timestep delta vs baseline ===")
    header = "t".ljust(4) + "".join(f"{r['name']:>14s}" for r in rows)
    print(header)
    for t in range(len(base["ll_t"])):
        line = f"{t:<4d}"
        for r in rows:
            line += f"{r['ll_t'][t] - base['ll_t'][t]:+14.4f}"
        print(line)

    # dump compact json for canvas
    out = {
        "run_dir": str(p),
        "contexts": [
            {
                "name": r["name"],
                "mean_ll": r["mean_ll"],
                "mean_ll_std": r["mean_ll_std"],
                "ll_t": r["ll_t"].tolist(),
                "ll_t_std": r["ll_t_std"].tolist(),
                "tran_loss": r["tran_loss"],
                "init_loss": r["init_loss"],
                "t_tran": r["t_tran"],
                "t_init": r["t_init"],
                "sp_basis": r["params"]["sp_basis"],
                "sp_model": r["params"]["sp_model"],
                "n_leaf_basis": r["params"]["n_leaf_basis"],
                "n_output_basis": r["params"]["n_output_basis"],
            }
            for r in rows
        ],
    }
    out_path = p / "summary_for_analysis.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "benchmark_data/cartpole_sp_ablation_y2026-m08-d02_H01-M37-S51")
