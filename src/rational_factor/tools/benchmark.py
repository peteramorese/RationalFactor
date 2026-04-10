import re
import torch
import enum
import os
import json
import traceback
from datetime import datetime


def _safe_figures_dir_segment(name: str) -> str:
    """Filesystem-safe subdirectory name under run_dir/figures/."""
    s = re.sub(r"[^\w.\-]+", "_", name.strip())
    return s if s else "context"

class ResultType(enum.Enum):
    NUMERICAL = "numerical"
    STRING = "string"
    BOOLEAN = "boolean"
    VISUAL = "visual"

class Benchmark:
    def __init__(self, name : str):
        self.name = name
        self.result_info = list()
        self.__ran = False
        self.__n_trials = 0
        self.trial_errors = []

    def set_experiment_fn(self, experiment_fn):
        self.experiment_fn = experiment_fn

    def set_contexts(self, contexts):
        assert isinstance(contexts, list)
        assert not self.__ran, "Cannot set contexts after benchmark has been run"
        seen_names: set[str] = set()
        for context in contexts:
            assert isinstance(context, dict)
            assert "name" in context, 'each context must have a "name" string'
            assert "params" in context, 'each context must have a "params" dict'
            assert isinstance(context["name"], str), "context name must be a string"
            assert isinstance(context["params"], dict), "context params must be a dict"
            n = context["name"]
            assert n not in seen_names, f"duplicate context name: {n!r}"
            seen_names.add(n)

        self.contexts = contexts
        # One list per declared return index, each storing flat trial outputs
        # in context-major order: [c0t0, c0t1, ..., c1t0, ...].
        self.results = []
        # Parallel to each (context, trial): None if success, else {"error": str, "traceback": str}
        self.trial_errors = []
    
    def _fit_results_to_size(self, return_idx : int):
        assert not self.__ran, "Cannot set results after benchmark has been run"
        while len(self.result_info) <= return_idx:
            self.result_info.append(dict())
            self.results.append(list())
    
    def set_numerical_result(self, return_idx : int, tag : str, calc_mean : bool = True, calc_std : bool = True, json_raw_data : bool = False):
        self._fit_results_to_size(return_idx)

        self.result_info[return_idx] = {
            "type": ResultType.NUMERICAL,
            "tag": tag,
            "calc_mean": calc_mean,
            "calc_std": calc_std,
            "json_raw_data": json_raw_data,
        }
    
    def set_string_result(self, return_idx : int, tag : str):
        self._fit_results_to_size(return_idx)

        self.result_info[return_idx] = {
            "type": ResultType.STRING,
            "tag": tag,
        }
    
    def set_boolean_result(self, return_idx : int, tag : str):
        self._fit_results_to_size(return_idx)

        self.result_info[return_idx] = {
            "type": ResultType.BOOLEAN,
            "tag": tag,
        }
    
    def set_visual_result(self, return_idx : int, tag : str, save_fig : bool = True, save_first_trial_only : bool = True, file_extension : str = "jpg"):
        self._fit_results_to_size(return_idx)

        self.result_info[return_idx] = {
            "type": ResultType.VISUAL,
            "tag": tag,
            "save_fig": save_fig,
            "save_first_trial_only": save_first_trial_only,
            "file_extension": file_extension,
        }

    def run(self, trials : int = 1, verbose : bool = True):
        for context_idx, context in enumerate(self.contexts):
            ctx_name = context["name"]
            if verbose:
                print(
                    f" ===== Running experiment {self.name} ({ctx_name!r}), "
                    f"context {context_idx + 1} of {len(self.contexts)} ===== "
                )
            for trial in range(trials):
                if verbose:
                    print(f"Running trial {trial + 1} of {trials}")
                try:
                    result = self.experiment_fn(**context["params"])
                    for return_idx, r in enumerate(result):
                        self.results[return_idx].append(r)
                    self.trial_errors.append(None)
                except Exception as e:
                    tb = traceback.format_exc()
                    print(
                        f"Error in experiment {self.name} ({ctx_name!r}), "
                        f"context {context_idx + 1} of {len(self.contexts)}: {e}"
                    )
                    self.trial_errors.append({"error": repr(e), "traceback": tb})
                    for return_idx in range(len(self.results)):
                        self.results[return_idx].append(None)
        self.__ran = True
        self.__n_trials += trials
    
    def process_and_save_results(self, root_dir : str = "benchmark_data"):
        assert self.__ran, "Benchmark has not been run"
        timestamp = datetime.now().strftime("y%Y-m%m-d%d_H%H-M%M-S%S")
        run_dir = os.path.join(root_dir, f"{self.name}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        n_contexts = len(self.contexts)
        n_trials = self.__n_trials
        expected_flat_len = n_contexts * n_trials

        processed_results = {
            "name": self.name,
            "contexts": {
                self.contexts[context_idx]["name"]: {
                    "name": self.contexts[context_idx]["name"],
                    "params": self.contexts[context_idx]["params"],
                    "n_trials_total": n_trials,
                    "n_trials_valid": 0,
                    "results": {},
                }
                for context_idx in range(n_contexts)
            },
        }

        raw_numerical = {self.contexts[i]["name"]: {} for i in range(n_contexts)}

        figures_root = os.path.join(run_dir, "figures")
        has_figures = False

        assert len(self.trial_errors) == expected_flat_len, (
            f"trial_errors has {len(self.trial_errors)} items, expected {expected_flat_len}"
        )

        for context_idx in range(n_contexts):
            ctx_key = self.contexts[context_idx]["name"]
            start = context_idx * n_trials
            end = start + n_trials
            context_err_chunk = self.trial_errors[start:end]
            if len(self.results) > 0:
                reference_data = self.results[0]
                assert len(reference_data) == expected_flat_len, (
                    f"Result index 0 has {len(reference_data)} items, expected {expected_flat_len}"
                )
                context_ref_data = reference_data[start:end]
                # Prefer explicit error log: success iff no recorded exception for that trial.
                n_ok_from_errors = sum(1 for e in context_err_chunk if e is None)
                n_ok_from_results = sum(1 for d in context_ref_data if d is not None)
                assert n_ok_from_errors == n_ok_from_results, (
                    "trial_errors and results disagree on success count"
                )
                processed_results["contexts"][ctx_key]["n_trials_valid"] = n_ok_from_errors
            else:
                processed_results["contexts"][ctx_key]["n_trials_valid"] = sum(
                    1 for e in context_err_chunk if e is None
                )

            trial_error_entries = []
            for err in context_err_chunk:
                if err is None:
                    trial_error_entries.append(None)
                else:
                    trial_error_entries.append(
                        {"error": err["error"], "traceback": err["traceback"]}
                    )
            if any(e is not None for e in trial_error_entries):
                processed_results["contexts"][ctx_key]["trial_errors"] = trial_error_entries

        for return_idx, info in enumerate(self.result_info):
            if "type" not in info:
                continue

            tag = info.get("tag", f"result_{return_idx}")
            data = self.results[return_idx]
            assert len(data) == expected_flat_len, (
                f"Result index {return_idx} has {len(data)} items, expected {expected_flat_len}"
            )

            for context_idx in range(n_contexts):
                ctx_key = self.contexts[context_idx]["name"]
                start = context_idx * n_trials
                end = start + n_trials
                context_data = data[start:end]

                if info["type"] == ResultType.NUMERICAL:
                    valid_data = [d for d in context_data if d is not None]

                    for d in valid_data:
                        assert isinstance(d, torch.Tensor), (
                            f"Numerical result '{tag}' must return torch.Tensor or None"
                        )
                        assert d.ndim > 0, (
                            f"Numerical result '{tag}' must be an ndim tensor (ndim > 0)"
                        )

                    context_payload = {}

                    if len(valid_data) > 0:
                        data_stack = torch.stack(valid_data, dim=0)
                        raw_numerical[ctx_key][tag] = data_stack

                        if info.get("calc_mean", True):
                            mean = data_stack.mean(dim=0)
                            context_payload["mean"] = mean.tolist()
                        if info.get("calc_std", True):
                            std = data_stack.std(dim=0, unbiased=False)
                            context_payload["std"] = std.tolist()
                    else:
                        empty_tensor = torch.empty((0,), dtype=torch.float32)
                        raw_numerical[ctx_key][tag] = empty_tensor
                        if info.get("calc_mean", True):
                            context_payload["mean"] = None
                        if info.get("calc_std", True):
                            context_payload["std"] = None

                    if info.get("json_raw_data", False):
                        context_payload["raw_data"] = [
                            d.tolist() if d is not None else None for d in context_data
                        ]

                    if len(context_payload) > 0:
                        processed_results["contexts"][ctx_key]["results"][tag] = context_payload

                elif info["type"] == ResultType.VISUAL:
                    if not info.get("save_fig", True):
                        continue

                    trials_to_save = [0] if info.get("save_first_trial_only", True) else list(range(n_trials))
                    file_extension = info.get("file_extension", "jpg")

                    fig_seg = _safe_figures_dir_segment(ctx_key)
                    context_fig_dir = os.path.join(figures_root, fig_seg)
                    saved_count = 0

                    for trial_idx in trials_to_save:
                        fig = context_data[trial_idx]
                        if fig is None:
                            continue

                        assert hasattr(fig, "savefig"), (
                            f"Visual result '{tag}' must be a matplotlib Figure-like object"
                        )
                        os.makedirs(context_fig_dir, exist_ok=True)
                        fig_path = os.path.join(
                            context_fig_dir,
                            f"{tag}_trial_{trial_idx}.{file_extension}",
                        )
                        fig.savefig(fig_path)
                        saved_count += 1

                    if saved_count > 0:
                        has_figures = True

                elif info["type"] == ResultType.STRING:
                    pass
                elif info["type"] == ResultType.BOOLEAN:
                    pass

        raw_data_path = os.path.join(run_dir, "raw_data.pt")
        torch.save(raw_numerical, raw_data_path)

        if not has_figures and os.path.isdir(figures_root):
            # Keep output directory clean when no figures were produced.
            try:
                os.rmdir(figures_root)
            except OSError:
                pass

        processed_path = os.path.join(run_dir, "processed_results.json")
        with open(processed_path, "w", encoding="utf-8") as f:
            json.dump(processed_results, f, indent=2)

        printed_any = False
        for context_key, payload in processed_results["contexts"].items():
            trial_errs = payload.get("trial_errors")
            if not trial_errs:
                continue
            for trial_idx, entry in enumerate(trial_errs):
                if entry is None:
                    continue
                printed_any = True
                print(
                    f"\n--- Benchmark error: {context_key} trial {trial_idx} ---\n"
                    f"{entry['error']}\n{entry['traceback']}"
                )
        if printed_any:
            print(f"\n(Full error records are in {processed_path})")

        return run_dir