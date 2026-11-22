"""Rebuild Experiment 1B statistics and figures exclusively from recorded files."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from conn_exp.evaluation import paired_summary
from conn_exp.plots import COLORS
from conn_exp.schema import CONDITIONS


def summarize(deltas):
    values = np.asarray(deltas, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Missing/nonfinite paired scores cannot be silently excluded")
    result = paired_summary(values)
    result["cohen_dz"] = float(values.mean() / values.std(ddof=1)) if values.std(ddof=1) > 0 else None
    result["mean_percentage_points"] = 100 * result["mean"]
    result["positive_seeds"] = int((values > 0).sum())
    result["tied_seeds"] = int((values == 0).sum())
    result["negative_seeds"] = int((values < 0).sum())
    return result


def read_jobs(phase, scores, name):
    return pd.concat([pd.read_csv(phase / row.job / f"{name}.csv") for row in scores.itertuples()], ignore_index=True)


def save(fig, folder, name):
    fig.tight_layout()
    fig.savefig(folder / name, dpi=150)
    plt.close(fig)


def lines(axis, frame, x, y, *, style="-", uncertainty=False):
    for cid, group in frame.groupby("condition_id"):
        series = group.groupby(x)[y].agg(["mean", "std", "count"]).sort_index()
        axis.plot(series.index, series["mean"], style, color=COLORS[cid], label=cid, lw=1.5)
        if uncertainty:
            se = series["std"] / np.sqrt(series["count"])
            axis.fill_between(series.index, series["mean"]-se, series["mean"]+se, color=COLORS[cid], alpha=.10)
    axis.set_xlabel("optimizer steps completed")
    axis.set_ylabel(y)


def phase_plots(phase, scores, label):
    folder = phase / "plots"
    folder.mkdir(exist_ok=True)
    metrics = read_jobs(phase, scores, "metrics")
    curve = metrics[((metrics.measure_timing == "post_step") | (metrics.optimizer_steps_completed == 0)) & metrics.split.isin(["train", "val"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for split, style in (("val", "-"), ("train", "--")):
        for axis, quantity in zip(axes, ["loss", "accuracy"]):
            lines(axis, curve[curve.split == split], "optimizer_steps_completed", quantity, style=style, uncertainty=split == "val")
    axes[1].legend(ncol=2, fontsize=8)
    fig.suptitle(label + ": validation solid (±1 SE), training dashed; no test curve")
    save(fig, folder, "learning_curves.png")
    unique = metrics[(metrics.split == "val") & (metrics.measure_timing == "pre_step")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for layer, row in zip(["ih", "hh"], axes):
        for quantity, axis in zip(["mean_abs_W_eff_", "norm_W_eff_"], row):
            lines(axis, unique, "optimizer_steps_completed", quantity+layer)
    axes[0, 1].legend(ncol=2, fontsize=8)
    fig.suptitle(label + ": effective weight magnitude (pre-update)")
    save(fig, folder, "effective_weights.png")
    energy = read_jobs(phase, scores, "row_energy")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for layer, row in zip(["ih", "hh"], axes):
        for cid, axis in zip(["D", "B_prime"], row):
            e = energy[(energy.condition_id == cid) & (energy.layer == layer) & (energy.measure_timing == "pre_step")]
            pivot = e.pivot_table(index="row", columns="optimizer_steps_completed", values="energy")
            im = axis.imshow(pivot, aspect="auto", origin="lower", interpolation="nearest")
            axis.set_title(f"{cid} {layer}: sum G² per unit")
            ticks = np.linspace(0, len(pivot.columns)-1, 5).astype(int)
            axis.set_xticks(ticks, [str(pivot.columns[t]) for t in ticks])
            axis.set_xlabel("optimizer steps completed")
            axis.set_ylabel("destination unit")
            fig.colorbar(im, ax=axis)
    fig.suptitle(label + ": per-unit gate energy (seed mean)")
    save(fig, folder, "per_unit_gate_energy.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for layer, axis in zip(["ih", "hh"], axes):
        lines(axis, unique, "optimizer_steps_completed", "row_energy_std_"+layer)
    axes[1].legend(ncol=2, fontsize=8)
    fig.suptitle(label + ": across-unit dispersion of gate energy, all conditions")
    save(fig, folder, "gate_energy_conditions.png")
    grads = read_jobs(phase, scores, "gradients")
    for layer in ["ih", "hh"]:
        fig, axes = plt.subplots(2, 7, figsize=(23, 7), sharex=True, sharey="row")
        for cid, column in zip([c.id for c in CONDITIONS], axes.T):
            frame = grads[(grads.condition_id == cid) & (grads.layer == layer)]
            for quantity, axis in zip(["mean_abs_dL_dW", "mean_abs_dL_dW_eff"], column):
                for bin_id, group in frame.groupby("bin_id"):
                    avg = group.groupby("optimizer_steps_completed")[quantity].mean()
                    axis.plot(avg.index, avg, label=f"bin {bin_id}", lw=1)
                axis.set_title(cid + " " + quantity.replace("mean_abs_", ""), fontsize=9)
                axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                axis.set_xlabel("steps")
        axes[0, -1].legend(fontsize=7)
        fig.suptitle(label + f": {layer} gradients by task distance (bins 0–4: [0,.2), …, [.8,1])")
        save(fig, folder, f"gradients_{layer}.png")
    ablation = read_jobs(phase, scores, "checkpoint_ablation")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    names = ["gate_pre_release.pt", "gate_post_release.pt", "final.pt"]
    for intervention, axis in zip(["ih", "hh", "both"], axes):
        frame = ablation[ablation.intervention == intervention].copy()
        frame["snapshot"] = frame.checkpoint.map(lambda p: names.index(Path(p).name))
        for cid, group in frame.groupby("condition_id"):
            avg = group.groupby("snapshot").val_loss_delta.agg(["mean", "sem"])
            axis.errorbar(avg.index, avg["mean"], yerr=avg["sem"], label=cid, color=COLORS[cid], marker="o", capsize=2)
        axis.axhline(0, color="gray", lw=.7)
        axis.set_xticks([0, 1, 2], ["pre release", "post release", "final"])
        axis.set_title("Suppress far: " + intervention)
        axis.set_ylabel("validation loss suppressed − original (±1 SE)")
    axes[-1].legend(ncol=2, fontsize=8)
    fig.suptitle(label + ": checkpoint reload ablations; no retraining")
    save(fig, folder, "checkpoint_ablation.png")
    return metrics, ablation


def build_report(root):
    root = Path(root)
    protocol = json.loads((root / "protocol.frozen.json").read_text())
    frozen = json.loads((root / "selection.frozen.json").read_text())
    tuning = pd.read_csv(root / "tuning/scores.csv")
    replication = pd.read_csv(root / "replication/scores.csv")
    assert "test_accuracy" not in tuning.columns
    assert (tuning.test_evaluations == 0).all()
    assert (replication.test_evaluations == 1).all()
    assert len(tuning) == 420 and len(replication) == 140
    assert set(tuning.seed).isdisjoint(replication.seed)
    pivot = replication.pivot(index="seed", columns="condition_id", values="test_accuracy")
    assert pivot.shape == (20, 7) and not pivot.isna().any().any()
    paired = pivot.copy()
    paired["D_minus_B_prime"] = pivot.D - pivot.B_prime
    paired.to_csv(root / "replication/individual_seed_results.csv")
    primary = summarize(paired.D_minus_B_prime)
    comparisons = []
    for cid in pivot.columns:
        if cid != "D":
            comparisons.append(dict(treatment="D", control=cid, primary=cid == "B_prime", **summarize(pivot.D-pivot[cid])))
    pd.DataFrame(comparisons).to_csv(root / "replication/paired_comparisons.csv", index=False)
    condition_summary = replication.groupby("condition_id").agg(
        mean_test_accuracy=("test_accuracy", "mean"), sd_test_accuracy=("test_accuracy", "std"),
        mean_test_loss=("test_loss", "mean"), sd_test_loss=("test_loss", "std"))
    condition_summary.to_csv(root / "replication/condition_summary.csv")
    (root / "replication/summary.json").write_text(json.dumps(primary, indent=2, allow_nan=False)+"\n")
    plot = root / "plots"
    plot.mkdir(exist_ok=True)
    best = pd.read_csv(root / "tuning/selected_lr_by_duration.csv")
    fig, axis = plt.subplots(figsize=(9, 5))
    for cid, group in best.groupby("condition_id"):
        axis.plot(group.total_steps, group.val_accuracy, marker="o", color=COLORS[cid], label=cid)
    axis.set_xticks(protocol["duration_grid"])
    axis.set_xlabel("independent training budget (schedule scales with duration)")
    axis.set_ylabel("mean final validation accuracy, selected LR per duration")
    axis.set_title("Exploratory duration analysis, seeds 5–9; no test evaluation")
    axis.legend(ncol=2)
    save(fig, plot, "duration_comparison.png")
    grid = pd.read_csv(root / "tuning/grid_validation.csv")
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    for steps, axis in zip(protocol["duration_grid"], axes):
        frame = grid[grid.total_steps == steps].pivot(index="condition_id", columns="lr", values="val_accuracy")
        im = axis.imshow(frame, vmin=grid.val_accuracy.min(), vmax=grid.val_accuracy.max(), aspect="auto")
        axis.set_xticks(range(3), [str(x) for x in frame.columns])
        axis.set_yticks(range(7), frame.index)
        for i in range(7):
            for j in range(3):
                axis.text(j, i, f"{frame.iloc[i,j]:.3f}", ha="center", va="center", fontsize=8, color="white" if frame.iloc[i,j] < .75 else "black")
        axis.set_title(f"{steps} steps")
        axis.set_xlabel("learning rate")
    fig.suptitle("Equal validation grid: five seeds per cell; all seven conditions")
    save(fig, plot, "validation_grid.png")
    for steps in protocol["duration_grid"]:
        selected = best[best.total_steps == steps][["condition_id", "total_steps", "lr"]]
        subset = tuning.merge(selected, on=["condition_id", "total_steps", "lr"])
        folder = root / "tuning" / f"duration_{steps}"
        folder.mkdir(exist_ok=True)
        # Paths in scores remain relative to tuning; plots are placed in each duration folder.
        phase_plots_at(root / "tuning", folder, subset, f"Exploratory tuning: {steps} steps, validation-selected LRs")
    metrics, ablation = phase_plots(root / "replication", replication,
        f"Independent replication: {frozen['total_steps']} steps, frozen LRs, n=20")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(pivot.B_prime, pivot.D, c=COLORS["D"])
    bounds = [float(pivot[["D", "B_prime"]].min().min())-.01, float(pivot[["D", "B_prime"]].max().max())+.01]
    axes[0].plot(bounds, bounds, color="gray", ls="--")
    axes[0].set_xlabel("B′ final test accuracy")
    axes[0].set_ylabel("D final test accuracy")
    axes[1].bar(paired.index.astype(str), 100*paired.D_minus_B_prime, color=COLORS["D"])
    axes[1].axhline(0, color="gray", lw=.8)
    axes[1].tick_params(axis="x", rotation=90)
    axes[1].set_xlabel("independent paired seed")
    axes[1].set_ylabel("D − B′ accuracy (percentage points)")
    fig.suptitle("Independent replication: frozen validation-selected configurations")
    save(fig, plot, "replication_paired.png")
    final_ablation = ablation[ablation.checkpoint.str.endswith("final.pt")]
    final_ablation.groupby(["condition_id", "intervention"])[["val_loss_delta", "val_accuracy_delta"]].agg(["mean", "std"]).to_csv(root / "replication/checkpoint_ablation_summary.csv")
    diagnostics = metrics[(metrics.split == "val") & (metrics.measure_timing == "post_step") & (metrics.optimizer_steps_completed == frozen["total_steps"])]
    diagnostic_cols = ["mean_abs_W_eff_ih", "mean_abs_W_eff_hh", "norm_W_eff_ih", "norm_W_eff_hh", "act_var_h1", "act_var_h2"]
    diagnostics.groupby("condition_id")[diagnostic_cols].agg(["mean", "std"]).to_csv(root / "replication/diagnostic_summary.csv")
    direction = ("The structured advantage did not persist as a clearly positive paired mean after fair tuning: the interval includes zero."
        if primary["spans_zero"] else
        "The structured condition retained a positive paired advantage after fair tuning in this independent replication."
        if primary["mean"] > 0 else
        "The structured advantage did not persist: the independent replication favored the shuffled control.")
    lines_out = ["# CoNN Experiment 1B — Optimization and Replication", "",
        direction, "",
        f"Measured primary result, 20 independent paired seeds: mean D − B′ = {primary['mean']:.8f} ({primary['mean_percentage_points']:+.4f} percentage points); sample SD = {primary['sd']:.8f}; two-sided 95% Student-t CI [{primary['ci_low']:.8f}, {primary['ci_high']:.8f}] (df=19). Cohen’s paired dz = {primary['cohen_dz']:.6f}.",
        f"Positive/tied/negative seed differences: {primary['positive_seeds']}/{primary['tied_seeds']}/{primary['negative_seeds']}.", "",
        "## Preserved pilot and separation", "",
        "The five-seed pilot at 20261017T235839Z_pilot remains exploratory. Its measured +0.021875 difference motivated this follow-up and is not pooled with replication. Pilot files and exact configuration are preserved; before/after SHA-256 manifests verify this.",
        "Tuning used new seeds 5–9; replication used seeds 100–119. Neither set overlaps pilot seeds 0–4. Tuning performed zero test evaluations and stored no test measurements. Replication test accuracy was measured once per model after the final optimizer update.", "",
        "## Fixed protocol and model selection", "",
        "The unchanged synthetic AND task, split sizes, hidden widths, SGD without momentum/decay, batch order, geometry and gate rules come from configs/pilot.yaml. Seven conditions were retained. Each received the predefined learning-rate grid [0.01, 0.05, 0.1] at 400, 800, 1600 and 3200 steps: 12 candidates × five tuning seeds = 60 fits per condition (420 total). Every condition received 90,000 tuning optimizer steps.",
        "At each duration, each condition’s LR maximized mean final validation accuracy; ties used lower mean validation loss, then lower LR. The common duration maximized macro-average validation accuracy across all seven selected conditions; ties used lower macro validation loss, then shorter duration. The D − B′ contrast was never the selection objective.",
        f"Frozen common duration: {frozen['total_steps']} optimizer steps. Frozen learning rates: " + ", ".join(f"{c.id}={frozen['lr_by_condition'][c.id]:g}" for c in CONDITIONS) + ".",
        "The exclusive-create selection.frozen.json and SHA-256 seal were written before replication.started.json and before replication models or data were created. Replication retained the frozen choices for all 20 seeds; no test-selected checkpoint, replacement seed, or repeated model selection was used.",
        "Duration experiments restart from the paired initialization and scale the curriculum to each duration. They compare complete training budgets, not intermediate checkpoints on a common 3200-step schedule. Diagnostic measurements use 20 equally spaced intervals plus schedule boundaries.", "",
        "## Exploratory validation duration comparison", "",
        "These are measured tuning validation means after per-condition LR selection, and have selection optimism. They are not held-out test results.", "",
        "| Steps | " + " | ".join(c.id for c in CONDITIONS) + " |", "|---|" + "---|"*7]
    for steps in protocol["duration_grid"]:
        frame = best[best.total_steps == steps].set_index("condition_id")
        lines_out.append(f"| {steps} | " + " | ".join(f"{frame.loc[c.id, 'val_accuracy']:.6f} (lr {frame.loc[c.id, 'lr']:g})" for c in CONDITIONS) + " |")
    lines_out += ["", "## Independent replication: all seed results", "",
        "| Seed | " + " | ".join(c.id for c in CONDITIONS) + " | D − B′ |", "|---|" + "---|"*8]
    for seed, row in paired.iterrows():
        lines_out.append(f"| {seed} | " + " | ".join(f"{row[c.id]:.6f}" for c in CONDITIONS) + f" | {row.D_minus_B_prime:+.6f} |")
    lines_out += ["", "## All-condition replication means", "",
        "| Condition | Mean test accuracy | Sample SD | Mean test loss |", "|---|---|---|---|"]
    for cid, row in condition_summary.iterrows():
        lines_out.append(f"| {cid} | {row.mean_test_accuracy:.6f} | {row.sd_test_accuracy:.6f} | {row.mean_test_loss:.6f} |")
    static_difference = float((pivot.D - pivot.S).mean())
    lines_out += ["",
        f"The static-local condition S averaged {100*pivot.S.mean():.4f}% accuracy versus D’s {100*pivot.D.mean():.4f}%; D − S = {100*static_difference:+.4f} percentage points. This is a secondary, exploratory comparison. The primary D − B′ advantage does not establish that widening to global connectivity is necessary or optimal; S keeps local gating throughout and achieved the higher observed mean here.", ""]
    lines_out += ["", "## Diagnostics and ablations", "",
        "Raw metrics record effective-weight mean absolute magnitude and L2 norm, activation variance, per-destination-unit sum of squared gates, and fixed-minibatch pre-update dL/dW and dL/dW_eff by base task distance. Gradient charts separate conditions and layers; they do not average the structured model together with controls. Train-fit motif probes are scored only on validation.",
        "Each model’s saved pre-release, post-release and final checkpoints were reloaded independently. The release pair shares identical weights and differs in gate state. Far connections are defined by distance at or above each base geometry matrix’s linear 75th percentile. Input-layer, hidden-layer and joint suppression measure validation loss and accuracy changes without retraining or changing stored weights. Positive loss delta means dependence on the removed connections; it does not establish that those connections caused a training advantage.",
        "See replication/checkpoint_ablation_summary.csv and replication/diagnostic_summary.csv for measured endpoint means and SDs; job-level tables preserve every measurement and checkpoint hash. Invariant audits check layer energy, per-unit D/B′ energy matching, and paired data/initialization fingerprints for every grid cell.", "",
        "## Interpretation and limitations", "",
        "The expectation was a positive D − B′ difference; the measured result above determines whether that expectation survived this protocol. The t interval describes seed-to-seed variation with fresh finite synthetic data splits, not robustness across tasks or data distributions. Twenty seeds, 128 test examples per seed, a narrow predefined SGD LR grid, and a single synthetic AND task limit generalization. The grid boundary may be optimal within the tested grid without being globally optimal. Hyperparameter selection on five validation seeds has uncertainty. Other condition contrasts in paired_comparisons.csv are exploratory and their intervals are unadjusted for multiplicity; D versus B′ is the sole primary comparison.",
        "Condition-specific LRs make the primary comparison one between equally tuned training procedures. A′ matches layer energy, while B′ matches each row’s gate multiset and energy. A within-row shuffle changes source alignment and higher-order mask topology together; this does not isolate geometry alone. S remains local at its endpoint; E is a path control with the shared dense endpoint. Longer budgets also stretch the curriculum, so duration and curriculum speed are coupled.",
        "These experiments do not establish a new architecture or prove the geometry hypothesis. A repeatable advantage here would support task-aligned gating under these fixed settings only.", "",
        "## Reproducibility", "",
        "protocol.frozen.json records the full plan before training; config.resolved.yaml and hashes capture study and per-fit settings. environment.json, requirements.freeze.txt, frozen source snapshot and source_manifest.json identify the implementation. No Git repository was present, so git_revision is null and the content-addressed source snapshot is the code revision. Existing independent data/init/order/mask streams and frozen row permutations were reused. All scores, curves, gradients, unit energies, probes, checkpoint ablations and weights are retained in versioned phase/job directories. completion.json records successful preservation and freeze checks.", ""]
    (root / "research_summary.md").write_text("\n".join(lines_out), encoding="utf-8")
    (root / "replication/report.md").write_text("\n".join(lines_out), encoding="utf-8")
    (root / "tuning/report.md").write_text("# Exploratory validation tuning\n\n420 completed fits, zero test evaluations. See grid_validation.csv, selected_lr_by_duration.csv, duration_validation.csv, duration plots and ../research_summary.md. These measurements selected the frozen replication configuration and are not independent evidence.\n", encoding="utf-8")


def phase_plots_at(source, target, scores, label):
    """Reuse the phase plotting routine while directing output to a duration folder."""
    # A lightweight path wrapper would obscure artifact paths: build plots in the phase,
    # then move the freshly generated figures to the explicit duration directory.
    phase_plots(source, scores, label)
    (target / "plots").mkdir(exist_ok=True)
    for p in (source / "plots").glob("*.png"):
        p.replace(target / "plots" / p.name)


if __name__ == "__main__":
    import sys
    build_report(Path(sys.argv[1]))
