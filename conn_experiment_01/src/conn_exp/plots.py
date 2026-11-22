"""Figures drawn only from saved tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {
    "A": "#4C78A8",
    "A_prime": "#72B7B2",
    "B_prime": "#F58518",
    "C": "#E45756",
    "D": "#54A24B",
    "E": "#B279A2",
    "S": "#FF9DA6",
}


def _require(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{name} is missing columns {missing}")


def _color(condition: str) -> str:
    return COLORS.get(condition, "#333333")


def write_all_plots(run_dir: Path) -> list[Path]:
    metrics = pd.read_csv(run_dir / "metrics.csv")
    gradients = pd.read_csv(run_dir / "gradients.csv")
    ablation = pd.read_csv(run_dir / "ablation.csv")
    contrast = pd.read_csv(run_dir / "paired_contrast.csv")
    _require(
        metrics,
        [
            "condition_id",
            "seed",
            "split",
            "measure_timing",
            "optimizer_steps_completed",
            "step_index",
            "loss",
            "accuracy",
            "C_ih",
            "C_hh",
            "tau_used",
            "d50_exact",
            "exact_dense",
            "row_energy_mean_ih",
            "row_energy_std_ih",
        ],
        "metrics.csv",
    )
    _require(
        gradients,
        ["condition_id", "step_index", "layer", "bin_id", "bin_lo", "bin_hi", "mean_abs_dL_dW", "mean_abs_dL_dW_eff"],
        "gradients.csv",
    )
    _require(ablation, ["condition_id", "optimizer_steps_completed", "val_loss_delta"], "ablation.csv")
    _require(contrast, ["seed", "delta"], "paired_contrast.csv")
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        _plot_loss_accuracy(metrics, plot_dir / "loss_accuracy.png"),
        _plot_gate_energy(metrics, plot_dir / "gate_energy.png"),
        _plot_schedule(metrics, plot_dir / "tau_d50_schedule.png"),
        _plot_row_energy(metrics, plot_dir / "row_energy.png"),
        _plot_gradients(gradients, plot_dir / "gradient_distance.png"),
        _plot_ablation(ablation, plot_dir / "ablation_delta.png"),
        _plot_scatter(contrast, plot_dir / "seed_scatter_D_minus_B_prime.png"),
    ]
    plt.close("all")
    return paths


def _mean_lines(frame: pd.DataFrame, x_col: str, y_col: str, axis: plt.Axes, style: str) -> None:
    for condition, group in frame.groupby("condition_id"):
        averaged = group.groupby(x_col, as_index=False)[y_col].mean()
        axis.plot(
            averaged[x_col],
            averaged[y_col],
            style,
            color=_color(str(condition)),
            label=str(condition),
            linewidth=1.5,
        )


def _plot_loss_accuracy(metrics: pd.DataFrame, path: Path) -> Path:
    curve = metrics[
        metrics["split"].isin(["train", "val"])
        & (
            (metrics["measure_timing"] == "post_step")
            | ((metrics["measure_timing"] == "pre_step") & (metrics["optimizer_steps_completed"] == 0))
        )
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for split, style in (("val", "-"), ("train", "--")):
        _mean_lines(curve[curve["split"] == split], "optimizer_steps_completed", "loss", axes[0], style)
        _mean_lines(curve[curve["split"] == split], "optimizer_steps_completed", "accuracy", axes[1], style)
    axes[0].set_title("Loss vs optimizer steps")
    axes[1].set_title("Accuracy vs optimizer steps")
    for axis, ylabel in zip(axes, ("loss", "accuracy")):
        axis.set_xlabel("optimizer steps completed")
        axis.set_ylabel(ylabel)
    axes[1].legend(ncol=2, fontsize=8)
    figure.suptitle("Train (dashed) and validation (solid). Test is not on this figure.")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    return path


def _plot_gate_energy(metrics: pd.DataFrame, path: Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    deduped = metrics.drop_duplicates(["seed", "condition_id", "step_index"])
    _mean_lines(deduped, "step_index", "C_ih", axes[0], "-")
    _mean_lines(deduped, "step_index", "C_hh", axes[1], "-")
    axes[0].set_title("Input to hidden, mean G^2")
    axes[1].set_title("Hidden to hidden, mean G^2")
    for axis in axes:
        axis.set_xlabel("schedule step index")
        axis.set_ylabel("C(t)")
    axes[1].legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    return path


def _plot_schedule(metrics: pd.DataFrame, path: Path) -> Path:
    deduped = metrics.drop_duplicates(["seed", "condition_id", "step_index"])
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    _mean_lines(deduped.dropna(subset=["tau_used"]), "step_index", "tau_used", axes[0], "-")
    _mean_lines(deduped.dropna(subset=["d50_exact"]), "step_index", "d50_exact", axes[1], "-")
    exact = deduped.copy()
    if exact["exact_dense"].dtype == bool:
        exact["exact_flag"] = exact["exact_dense"].astype(float)
    else:
        exact["exact_flag"] = exact["exact_dense"].astype(str).str.lower().isin(["true", "1", "1.0"]).astype(float)
    _mean_lines(exact, "step_index", "exact_flag", axes[2], "-")
    axes[0].set_title("tau")
    axes[1].set_title("d50")
    axes[2].set_title("Exact dense phase")
    for axis in axes:
        axis.set_xlabel("schedule step index")
    axes[0].set_ylabel("tau")
    axes[1].set_ylabel("d50")
    axes[2].set_ylabel("exact G = 1")
    axes[2].legend(ncol=2, fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    return path


def _plot_row_energy(metrics: pd.DataFrame, path: Path) -> Path:
    deduped = metrics.drop_duplicates(["seed", "condition_id", "step_index"])
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    _mean_lines(deduped, "step_index", "row_energy_mean_ih", axis, "-")
    axis.set_xlabel("schedule step index")
    axis.set_ylabel("mean over rows of sum of squared gates")
    axis.set_title("Input-layer per-row gate energy")
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    return path


def _plot_gradients(gradients: pd.DataFrame, path: Path) -> Path:
    subset = gradients[gradients["layer"] == "ih"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for bin_id, group in subset.groupby("bin_id"):
        averaged_w = group.groupby("step_index", as_index=False)["mean_abs_dL_dW"].mean()
        averaged_we = group.groupby("step_index", as_index=False)["mean_abs_dL_dW_eff"].mean()
        low = float(group["bin_lo"].iloc[0])
        high = float(group["bin_hi"].iloc[0])
        label = f"[{low:.1f}, {high:.1f})"
        axes[0].plot(averaged_w["step_index"], averaged_w["mean_abs_dL_dW"], label=label)
        axes[1].plot(averaged_we["step_index"], averaged_we["mean_abs_dL_dW_eff"], label=label)
    axes[0].set_title("mean |dL/dW| (gate-attenuated)")
    axes[1].set_title("mean |dL/dW_eff| (data gradient)")
    for axis in axes:
        axis.set_xlabel("schedule step index")
    axes[1].legend(fontsize=7, title="task-coordinate bin")
    figure.suptitle("Input layer, averaged over seeds and conditions. Bins are not shuffled-mask distances.")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    return path


def _plot_ablation(ablation: pd.DataFrame, path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    _mean_lines(ablation, "optimizer_steps_completed", "val_loss_delta", axis, "-")
    axis.axhline(0.0, color="#666666", linewidth=0.8)
    axis.set_xlabel("optimizer steps completed")
    axis.set_ylabel("suppressed validation loss − original")
    axis.set_title("Far-connection ablation on validation")
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    return path


def _plot_scatter(contrast: pd.DataFrame, path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    finite = contrast.dropna(subset=["delta"])
    axis.axhline(0.0, color="#666666", linewidth=0.8)
    if not finite.empty:
        axis.scatter(finite["seed"], finite["delta"], color=COLORS["D"], zorder=3)
        axis.plot(finite["seed"], finite["delta"], color=COLORS["D"], linewidth=1.0)
    axis.set_xlabel("paired seed")
    axis.set_ylabel("test accuracy D − B'")
    axis.set_title("Primary contrast, final-step test accuracy")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    return path
