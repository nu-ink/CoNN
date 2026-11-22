"""Scores, the pre-registered contrast, log contracts, and the run report."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from conn_exp.schema import (
    CONTRAST_COLUMNS,
    DISTANCE_DEFINITION,
    GRADIENT_DEFINITION,
    PRIMARY_CONTROL,
    PRIMARY_CONTROL_NAME,
    PRIMARY_METRIC,
    PRIMARY_TASK,
    PRIMARY_TREATMENT,
    SCORE_COLUMNS,
    condition_by_id,
    condition_role,
    make_row,
)

# Two-sided 95% Student-t critical values (0.975 quantile).
_T_975 = {
    1: 12.706204736432095,
    2: 4.302652729911275,
    3: 3.182446305284263,
    4: 2.7764451051977987,
    5: 2.5705818366147395,
    6: 2.4469118511449692,
    7: 2.364624251592784,
    8: 2.3060041350333704,
    9: 2.2621571627409915,
    10: 2.2281388519862724,
    11: 2.200985160082949,
    12: 2.1788128296634177,
    13: 2.1603686564627917,
    14: 2.1447866879169273,
    15: 2.131449545559323,
    16: 2.1199052992210112,
    17: 2.1098155778331806,
    18: 2.1009220402410415,
    19: 2.093024054408263,
    20: 2.0859634472658364,
    21: 2.079613844727163,
    22: 2.0738730679040146,
    23: 2.0686576104190406,
    24: 2.0638985616280205,
    25: 2.059538552753294,
    26: 2.055529438642871,
    27: 2.0518305164802833,
    28: 2.048407141795244,
    29: 2.045229642132703,
    30: 2.0422724563012373,
}


def t_critical_975(df: int) -> float:
    if df not in _T_975:
        raise ValueError(f"No stored 95% t critical value for df={df}. Extend the table rather than approximating.")
    return _T_975[df]


def paired_summary(deltas: np.ndarray) -> dict[str, object]:
    values = np.asarray(list(deltas), dtype=np.float64)
    values = values[np.isfinite(values)]
    n = int(values.size)
    empty = {
        "n": n,
        "mean": None,
        "sd": None,
        "t_crit": None,
        "ci_low": None,
        "ci_high": None,
        "spans_zero": None,
    }
    if n == 0:
        return empty
    mean = float(values.mean())
    if n < 2:
        empty["mean"] = mean
        return empty
    sd = float(values.std(ddof=1))
    standard_error = sd / math.sqrt(n)
    critical = t_critical_975(n - 1)
    low = mean - critical * standard_error
    high = mean + critical * standard_error
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "t_crit": critical,
        "ci_low": low,
        "ci_high": high,
        "spans_zero": bool(low <= 0.0 <= high),
    }


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    text = series.astype(str).str.strip().str.lower()
    return text.isin(["true", "1", "1.0"])


def coerce_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = frame.copy()
    metrics["boundary"] = metrics["boundary"].fillna("").astype(str).replace({"nan": ""})
    metrics["exact_dense"] = _as_bool(metrics["exact_dense"])
    metrics["tau_floored"] = _as_bool(metrics["tau_floored"])
    metrics["non_finite"] = _as_bool(metrics["non_finite"])
    return metrics


def _gate_view(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "seed",
        "condition_id",
        "step_index",
        "C_ih",
        "C_hh",
        "row_energy_mean_ih",
        "row_energy_std_ih",
        "exact_dense",
        "phase",
        "tau_requested",
    ]
    return metrics[columns].drop_duplicates(["seed", "condition_id", "step_index"])


def _max_abs(left: pd.Series, right: pd.Series) -> float:
    if left.empty:
        raise AssertionError("Invariant comparison had no overlapping rows.")
    return float(np.max(np.abs(left.to_numpy(dtype=float) - right.to_numpy(dtype=float))))


def audit_invariants(metrics: pd.DataFrame, row_energy: pd.DataFrame) -> dict[str, float]:
    """Correctness checks. These do not test whether D outpredicts B'."""
    view = _gate_view(metrics)
    report: dict[str, float] = {}
    for other in ("B_prime", "A_prime", "C"):
        merged = view[view.condition_id == "D"].merge(
            view[view.condition_id == other],
            on=["seed", "step_index"],
            suffixes=("_d", "_o"),
        )
        energy_gap = max(_max_abs(merged["C_ih_d"], merged["C_ih_o"]), _max_abs(merged["C_hh_d"], merged["C_hh_o"]))
        report[f"{other}_vs_D_layer_energy_max_abs"] = energy_gap
        if energy_gap > 1e-4:
            raise AssertionError(f"{other} layer energy diverged from D by {energy_gap}")
    local = view[(view.condition_id == "D") & (view.phase == "local_to_global")]
    if local.empty or float(local["row_energy_std_ih"].max()) <= 1e-4:
        raise AssertionError("D did not show unequal per-row gate energy during the local phase.")
    scalar = view[(view.condition_id == "A_prime") & (view.step_index.isin(set(local.step_index.tolist())))]
    scalar_std = float(scalar["row_energy_std_ih"].max())
    report["A_prime_local_row_energy_std_max"] = scalar_std
    if scalar_std > 1e-5:
        raise AssertionError("A' per-row gate energy is not uniform; it would be matching more than layer energy.")
    compared = row_energy[row_energy.condition_id.isin(["D", "B_prime"])]
    merged_rows = compared[compared.condition_id == "D"].merge(
        compared[compared.condition_id == "B_prime"],
        on=["seed", "step_index", "measure_timing", "layer", "row"],
        suffixes=("_d", "_b"),
    )
    row_gap = _max_abs(merged_rows["energy_d"], merged_rows["energy_b"])
    report["B_prime_vs_D_row_energy_max_abs"] = row_gap
    if row_gap > 1e-3:
        raise AssertionError(f"B' per-row energy diverged from D by {row_gap}")
    return report


def assert_log_contracts(
    metrics: pd.DataFrame,
    gradients: pd.DataFrame,
    ablation: pd.DataFrame,
    probes: pd.DataFrame,
    protocol: dict[str, object],
) -> None:
    if protocol["primary_treatment"] != PRIMARY_TREATMENT or protocol["primary_control"] != PRIMARY_CONTROL:
        raise AssertionError("Primary contrast drifted away from D versus B_prime.")
    if protocol["primary_control_name"] != PRIMARY_CONTROL_NAME:
        raise AssertionError("Primary control is no longer the within-row shuffle.")
    total_steps = int(protocol["total_steps"])
    release_step = int(protocol["release_step"])
    tau_start = float(protocol["tau_start"])
    if set(metrics["optimizer"].unique()) != {"sgd"}:
        raise AssertionError("A logged row does not use SGD.")
    if not np.allclose(metrics["momentum"].astype(float), 0.0) or not np.allclose(metrics["weight_decay"].astype(float), 0.0):
        raise AssertionError("Momentum or weight decay was not zero.")
    if set(metrics["distance_definition"].unique()) != {DISTANCE_DEFINITION}:
        raise AssertionError("Metric rows are not labeled with the base-geometry distance.")
    for _, frame in metrics.groupby("seed"):
        if frame["data_fingerprint"].nunique() != 1 or frame["init_fingerprint"].nunique() != 1:
            raise AssertionError("A seed does not share one dataset and one initialization across conditions.")
    test_rows = metrics[metrics["split"] == "test"]
    if test_rows.empty:
        raise AssertionError("The final test evaluation is missing.")
    if set(test_rows["optimizer_steps_completed"].astype(int).unique()) != {total_steps}:
        raise AssertionError("Test rows exist before the final optimizer step.")
    if set(test_rows["measure_timing"].unique()) != {"post_step"}:
        raise AssertionError("Test rows are not post-step measurements.")
    if gradients.empty or not (gradients["measure_timing"] == "pre_step").all():
        raise AssertionError("Gradient rows must be pre-step so they are not confused with post-step values.")
    if not gradients["gradient_definition"].astype(str).str.contains("gate-attenuated").all():
        raise AssertionError("Gradient rows do not identify dL/dW as gate-attenuated.")
    if not gradients["gradient_definition"].astype(str).str.contains("dL/dW_eff").all():
        raise AssertionError("Gradient rows do not identify dL/dW_eff.")
    if set(gradients["distance_definition"].unique()) != {DISTANCE_DEFINITION}:
        raise AssertionError("Gradient bins are not labeled as task-coordinate bins.")
    if ablation.empty or set(ablation["distance_definition"].unique()) != {DISTANCE_DEFINITION}:
        raise AssertionError("Ablation distance definition is missing or wrong.")
    if not ablation["delta_definition"].astype(str).str.contains("shuffled-mask").all():
        raise AssertionError("Ablation rows do not say that far is independent of the shuffled mask.")
    if set(probes["fit_split"].unique()) != {"train"} or set(probes["eval_split"].unique()) != {"val"}:
        raise AssertionError("Probes were not restricted to train fit and validation score.")
    if set(probes["target"].unique()) - {"motif_a", "motif_b"}:
        raise AssertionError("Probe targets drifted.")

    def boundary_rows(condition: str, flag: str, timing: str) -> pd.DataFrame:
        subset = metrics[(metrics.condition_id == condition) & (metrics.measure_timing == timing)]
        return subset[subset["boundary"].astype(str).str.contains(flag)]

    pre = boundary_rows("D", "pre_release", "post_step")
    post = boundary_rows("D", "post_release", "pre_step")
    if pre.empty or post.empty:
        raise AssertionError("D is missing the paired 70% release snapshots.")
    if bool(pre["exact_dense"].any()) or not bool(post["exact_dense"].all()):
        raise AssertionError("D release snapshots do not flip from Gaussian to exact dense.")
    if set(pre["optimizer_steps_completed"].astype(int).unique()) != {release_step}:
        raise AssertionError("The pre-release snapshot is not at the release weight count.")
    if set(post["optimizer_steps_completed"].astype(int).unique()) != {release_step}:
        raise AssertionError("The post-release snapshot is not at the same weight count.")
    final_d = metrics[
        (metrics.condition_id == "D")
        & (metrics.measure_timing == "post_step")
        & (metrics.optimizer_steps_completed.astype(int) == total_steps)
    ]
    if final_d.empty or not bool(final_d["exact_dense"].all()):
        raise AssertionError("D's final phase is not exact dense.")
    static = metrics[metrics.condition_id == "S"]
    if bool(static["exact_dense"].any()):
        raise AssertionError("S entered an exact dense phase.")
    if float(np.nanmax(np.abs(static["tau_requested"].astype(float) - tau_start))) > 1e-8:
        raise AssertionError("S did not hold tau_start.")
    dense = metrics[metrics.condition_id == "A"]
    if not bool(dense["exact_dense"].all()):
        raise AssertionError("A was not exact dense throughout.")
    local_e = metrics[metrics.boundary.astype(str).str.contains("E_local_extreme")]
    if local_e.empty or bool(local_e["exact_dense"].any()):
        raise AssertionError("E has no observable local phase.")
    if float(np.nanmax(np.abs(local_e["tau_requested"].astype(float) - tau_start))) > 1e-6:
        raise AssertionError("E's local extreme is not tau_start.")


def scores_from_metrics(metrics: pd.DataFrame, total_steps: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = metrics.groupby(["seed", "condition_id"], sort=True)
    for (seed, condition_id), frame in grouped:
        condition = condition_by_id(str(condition_id))
        def final(split: str) -> pd.Series | None:
            subset = frame[
                (frame["split"] == split)
                & (frame["measure_timing"] == "post_step")
                & (frame["optimizer_steps_completed"].astype(int) == total_steps)
            ]
            if subset.empty:
                return None
            if len(subset) != 1:
                raise AssertionError(f"Expected one final {split} row for {condition_id} seed {seed}.")
            return subset.iloc[0]

        train = final("train")
        val = final("val")
        test = final("test")
        non_finite = bool(frame["non_finite"].any()) or test is None
        test_evals = int(((frame["split"] == "test")).sum())
        sample = frame.iloc[0]
        rows.append(
            make_row(
                SCORE_COLUMNS,
                run_id=sample["run_id"],
                seed=int(seed),
                condition_id=condition.id,
                condition_name=condition.name,
                task=sample["task"],
                role=condition_role(condition.id),
                optimizer_steps=total_steps,
                final_train_loss=None if train is None else float(train["loss"]),
                final_train_accuracy=None if train is None else float(train["accuracy"]),
                final_val_loss=None if val is None else float(val["loss"]),
                final_val_accuracy=None if val is None else float(val["accuracy"]),
                final_test_loss=None if test is None else float(test["loss"]),
                final_test_accuracy=None if test is None else float(test["accuracy"]),
                lr=float(sample["lr"]),
                momentum=float(sample["momentum"]),
                weight_decay=float(sample["weight_decay"]),
                optimizer=sample["optimizer"],
                non_finite=non_finite,
                data_fingerprint=sample["data_fingerprint"],
                init_fingerprint=sample["init_fingerprint"],
                test_evaluations=test_evals,
            )
        )
    return pd.DataFrame(rows, columns=list(SCORE_COLUMNS))


def contrast_from_scores(scores: pd.DataFrame, run_id: str, task: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed, frame in scores.groupby("seed"):
        treatment = frame[frame.condition_id == PRIMARY_TREATMENT]
        control = frame[frame.condition_id == PRIMARY_CONTROL]
        if treatment.empty or control.empty:
            raise AssertionError("Primary contrast conditions are missing from scores.")
        treat = treatment.iloc[0]
        ctrl = control.iloc[0]
        usable = (not bool(treat["non_finite"])) and (not bool(ctrl["non_finite"]))
        value_t = treat["final_test_accuracy"]
        value_c = ctrl["final_test_accuracy"]
        delta = None
        if usable and value_t is not None and value_c is not None and np.isfinite([value_t, value_c]).all():
            delta = float(value_t) - float(value_c)
        else:
            usable = False
        rows.append(
            make_row(
                CONTRAST_COLUMNS,
                run_id=run_id,
                task=task,
                metric=PRIMARY_METRIC,
                treatment=PRIMARY_TREATMENT,
                control=PRIMARY_CONTROL,
                control_name=PRIMARY_CONTROL_NAME,
                seed=int(seed),
                treatment_value=None if value_t is None else float(value_t),
                control_value=None if value_c is None else float(value_c),
                delta=delta,
                used_in_aggregate=usable,
            )
        )
    return pd.DataFrame(rows, columns=list(CONTRAST_COLUMNS))


def summarize_contrast(contrast: pd.DataFrame) -> dict[str, object]:
    used = contrast[contrast["used_in_aggregate"] == True]  # noqa: E712
    summary = paired_summary(used["delta"].to_numpy() if not used.empty else np.array([]))
    summary["metric"] = PRIMARY_METRIC
    summary["treatment"] = PRIMARY_TREATMENT
    summary["control"] = PRIMARY_CONTROL
    summary["control_name"] = PRIMARY_CONTROL_NAME
    summary["task"] = None if contrast.empty else contrast.iloc[0]["task"]
    return summary


def _fmt(value: object) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_report(run_dir: Path, protocol: dict[str, object], summary: dict[str, object], audit: dict[str, float]) -> None:
    scores = pd.read_csv(run_dir / "scores.csv")
    contrast = pd.read_csv(run_dir / "paired_contrast.csv")
    n_seeds = int(contrast["seed"].nunique()) if not contrast.empty else 0
    pre_registered = protocol.get("task") == PRIMARY_TASK and not protocol.get("smoke")
    lines = [
        "# CoNN Experiment 01 report",
        "",
        f"Run `{protocol['run_id']}`. Config SHA-256 `{protocol['config_sha256']}`.",
        f"Device `{protocol['device']}`. Optimizer recorded as SGD, momentum 0, weight decay 0, foreach false.",
        "",
        "## Protocol",
        "",
        "Primary metric: held-out test accuracy after the pre-specified optimizer-step budget.",
        "The test split is evaluated once, on the post-update weights of the final step. It is not used to pick a checkpoint, to place coordinates, or to fit probes.",
        f"Primary contrast: {PRIMARY_TREATMENT} minus {PRIMARY_CONTROL} (`{PRIMARY_CONTROL_NAME}`) on {PRIMARY_TASK}.",
        f"This file's task is `{protocol['task']}` with seeds {protocol['seed_list']}.",
        "Learning rate: one shared initial value for every condition. "
        f"The predefined equal grid {protocol['lr_grid']} was not executed (`lr_grid_executed={protocol['lr_grid_executed']}`). "
        "These numbers cannot support a claim of superiority.",
        f"Schedule: release_step={protocol['release_step']} of {protocol['total_steps']}, tau_start={protocol['tau_start']}, tau_end={protocol['tau_end']}.",
        "S stays at tau_start through the final step. E is a three-phase path with the same final exact-dense endpoint as D, not a pure reversal.",
        "",
        "## Invariant audit",
        "",
        "These checks confirm the controls still match the quantities they are supposed to match. They are not outcome results.",
        "",
    ]
    for key, value in audit.items():
        lines.append(f"- `{key}`: {value:.6e}")
    lines.extend(
        [
            "",
            "## Measured primary contrast",
            "",
            "Values below were read from the logged scores. Empty cells were not computed.",
            "",
            "| seed | D test accuracy | B' test accuracy | delta | used in aggregate |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in contrast.sort_values("seed").iterrows():
        lines.append(
            f"| {int(row['seed'])} | {_fmt(row['treatment_value'])} | {_fmt(row['control_value'])} | "
            f"{_fmt(row['delta'])} | {row['used_in_aggregate']} |"
        )
    lines.extend(
        [
            "",
            f"Paired n = {summary['n']}. Mean delta = {_fmt(summary['mean'])}. Sample SD = {_fmt(summary['sd'])}.",
            f"95% interval using t critical {_fmt(summary['t_crit'])}: [{_fmt(summary['ci_low'])}, {_fmt(summary['ci_high'])}].",
            "",
        ]
    )
    if not pre_registered or n_seeds != 5:
        lines.append(
            "This run is not the five-seed AND pilot, or it is a smoke/pipeline run. Do not read it as evidence for or against the curriculum."
        )
    else:
        lines.append(
            "Five seeds are pilot/exploratory. This interval is not a confirmatory significance test, and no multiplicity correction is applied because no confirmatory claim is made."
        )
    if summary.get("spans_zero") is True:
        lines.append("The interval contains zero. No effect is claimed.")
    elif summary.get("spans_zero") is False:
        lines.append(
            "In this sample the interval excludes zero. That is still not a confirmatory finding at five seeds."
        )
    lines.extend(
        [
            "",
            "D minus B' does not isolate geometry by itself: within-row shuffles preserve each row's gate multiset and energy, and they change which sources receive those values.",
            "A' matches layerwise mean square gate energy, not the per-row distribution. E shares D's dense endpoint and changes the path. S is still local at the final step.",
            "",
            "## All conditions, final test accuracy",
            "",
            "| seed | condition | role | test accuracy | test loss | non_finite | test evaluations |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in scores.sort_values(["seed", "condition_id"]).iterrows():
        lines.append(
            f"| {int(row['seed'])} | {row['condition_id']} | {row['role']} | {_fmt(row['final_test_accuracy'])} | "
            f"{_fmt(row['final_test_loss'])} | {row['non_finite']} | {int(row['test_evaluations'])} |"
        )
    lines.extend(
        [
            "",
            "## Expectations are not measurements",
            "",
            "A positive D minus B' delta would be consistent with a benefit of spatial alignment under this protocol. A negative delta, or an interval that contains zero, is a failure of that prediction in this run. The sentence stating the prediction is not a result.",
            "",
            "## Test-set usage",
            "",
            protocol["test_protocol_text"],
            "",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def rebuild_tables(run_dir: Path) -> dict[str, object]:
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    metrics = coerce_metrics(pd.read_csv(run_dir / "metrics.csv"))
    gradients = pd.read_csv(run_dir / "gradients.csv")
    ablation = pd.read_csv(run_dir / "ablation.csv")
    probes = pd.read_csv(run_dir / "probes.csv")
    row_energy = pd.read_csv(run_dir / "row_energy.csv")
    assert_log_contracts(metrics, gradients, ablation, probes, protocol)
    audit = audit_invariants(metrics, row_energy)
    scores = scores_from_metrics(metrics, int(protocol["total_steps"]))
    contrast = contrast_from_scores(scores, str(protocol["run_id"]), str(protocol["task"]))
    summary = summarize_contrast(contrast)
    summary["pre_registered_and_pilot"] = (not protocol.get("smoke")) and protocol.get("task") == PRIMARY_TASK and int(summary["n"]) == 5
    scores.to_csv(run_dir / "scores.csv", index=False)
    contrast.to_csv(run_dir / "paired_contrast.csv", index=False)
    (run_dir / "summary.json").write_text(json.dumps({"contrast": summary, "audit": audit}, indent=2), encoding="utf-8")
    (run_dir / "invariant_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_report(run_dir, protocol, summary, audit)
    return summary
