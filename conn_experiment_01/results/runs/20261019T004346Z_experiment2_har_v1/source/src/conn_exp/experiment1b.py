"""Experiment 1B: validation-only factorial tuning, freeze, independent replication.

Run with python -m conn_exp.experiment1b. No existing run is overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from conn_exp.config import config_sha256, load_config
from conn_exp.data import generate_dataset, assert_dataset_invariants
from conn_exp.diagnostics import far_suppression
from conn_exp.evaluation import audit_invariants
from conn_exp.gates import GateBuilder, draw_masks
from conn_exp.geometry import make_task_geometry
from conn_exp.models import CoNNMLP, build_optimizer, initial_state_dict, fingerprint_state
from conn_exp.runner import _save_splits, _tensor
from conn_exp.runtime import configure_runtime, resolve_device
from conn_exp.schema import CONDITIONS
from conn_exp.train import HeldOutSplit, evaluate_split, minibatch_order, train_condition

REPO = Path(__file__).resolve().parents[2]
PILOT = REPO / "results/runs/20261017T235839Z_pilot"
STEPS = (400, 800, 1600, 3200)
TUNING_SEEDS = tuple(range(5, 10))
REPLICATION_SEEDS = tuple(range(100, 120))


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def tree_manifest(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def seal(path, payload):
    """Exclusive creation: a frozen decision can never be replaced by this runner."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    path.with_suffix(path.suffix + ".sha256").write_text(sha(path) + "\n", encoding="utf-8")


class ValidationOnly:
    evaluations = 0

    def evaluate(self, *args, **kwargs):
        raise AssertionError("Test access forbidden during model selection")


def select_validation(scores):
    """No test columns enter selection. All candidates must be complete and finite."""
    columns = ["condition_id", "total_steps", "lr", "seed", "val_accuracy", "val_loss"]
    values = scores[columns].copy()
    if values.duplicated(columns[:4]).any() or not np.isfinite(values[["val_accuracy", "val_loss"]]).all().all():
        raise ValueError("Duplicate or nonfinite validation candidates")
    seed_sets = values.groupby(columns[:3]).seed.agg(lambda s: tuple(sorted(s)))
    if seed_sets.nunique() != 1:
        raise ValueError("Unequal tuning seed budgets")
    expected = pd.MultiIndex.from_product(
        [values.condition_id.unique(), values.total_steps.unique(), values.lr.unique()], names=columns[:3])
    if len(seed_sets) != len(expected) or not seed_sets.index.sort_values().equals(expected.sort_values()):
        raise ValueError("Incomplete factorial grid")
    grid = values.groupby(columns[:3], as_index=False).agg(
        val_accuracy=("val_accuracy", "mean"), val_loss=("val_loss", "mean"), n=("seed", "size"))
    best = grid.sort_values(
        ["condition_id", "total_steps", "val_accuracy", "val_loss", "lr"],
        ascending=[True, True, False, True, True]).drop_duplicates(["condition_id", "total_steps"])
    duration = best.groupby("total_steps", as_index=False).agg(
        macro_val_accuracy=("val_accuracy", "mean"), macro_val_loss=("val_loss", "mean"))
    chosen_steps = int(duration.sort_values(
        ["macro_val_accuracy", "macro_val_loss", "total_steps"], ascending=[False, True, True]).iloc[0].total_steps)
    chosen_lr = {str(r.condition_id): float(r.lr) for r in best[best.total_steps == chosen_steps].itertuples()}
    return grid, best, duration, chosen_steps, chosen_lr


def checkpoint_ablation(job, config, geometry, x_val, y_val):
    """Reload stored weights and gates; use validation only; never resume training."""
    rows = []
    for path in sorted((job / "checkpoints").rglob("*.pt")):
        state = torch.load(path, map_location=x_val.device, weights_only=False)
        model = CoNNMLP(config.n_features, config.hidden_sizes).to(x_val.device)
        model.load_state_dict(state["model"])
        before = fingerprint_state(model.state_dict())
        ih, hh = state["gate_ih"], state["gate_hh"]
        original_loss, original_acc = evaluate_split(model, x_val, y_val, ih, hh)
        suppressed = far_suppression(ih, hh, geometry.dist_ih.to(x_val.device), geometry.dist_hh.to(x_val.device), config.far_quantile)
        # Joint and layer-specific interventions distinguish where dependence lies.
        for intervention, gi, gh in (
            ("both", suppressed["gate_ih"], suppressed["gate_hh"]),
            ("ih", suppressed["gate_ih"], hh), ("hh", ih, suppressed["gate_hh"])):
            loss, acc = evaluate_split(model, x_val, y_val, gi, gh)
            rows.append(dict(checkpoint=path.relative_to(job).as_posix(), checkpoint_sha256=sha(path),
                seed=state["seed"], condition_id=state["condition_id"],
                optimizer_steps_completed=state["optimizer_steps_completed"],
                measure_timing=state["measure_timing"], phase=state["phase"], intervention=intervention,
                far_quantile=config.far_quantile, threshold_ih=suppressed["threshold_ih"],
                threshold_hh=suppressed["threshold_hh"], n_suppressed_ih=suppressed["n_suppressed_ih"] if intervention != "hh" else 0,
                n_suppressed_hh=suppressed["n_suppressed_hh"] if intervention != "ih" else 0,
                val_loss=original_loss, val_accuracy=original_acc,
                suppressed_val_loss=loss, suppressed_val_accuracy=acc,
                val_loss_delta=loss-original_loss, val_accuracy_delta=acc-original_acc,
                distance_definition="base task geometry, distance >= linear 0.75 quantile", split="val"))
        assert fingerprint_state(model.state_dict()) == before
    pd.DataFrame(rows).to_csv(job / "checkpoint_ablation.csv", index=False)


def run_phase(root, base, seeds, candidates, replication=False):
    root.mkdir()
    device = resolve_device(base.device)
    scores = []
    for seed in seeds:
        dataset = generate_dataset(base, seed)
        assert_dataset_invariants(dataset, base)
        if replication:
            _save_splits(root / "data" / f"seed_{seed}.npz", dataset)
        else:
            (root / "data").mkdir(exist_ok=True)
            np.savez_compressed(root / "data" / f"seed_{seed}.npz",
                x_train=dataset.train.x, y_train=dataset.train.y, x_val=dataset.val.x, y_val=dataset.val.y)
        masks = draw_masks(base.n_features, *base.hidden_sizes, seed)
        (root / "masks").mkdir(exist_ok=True)
        torch.save(vars(masks), root / "masks" / f"seed_{seed}.pt")
        init = initial_state_dict(base.n_features, base.hidden_sizes, seed)
        geometry = make_task_geometry(base.n_features, base.hidden_sizes)
        order = minibatch_order(base.n_train, seed)
        xtr, ytr = _tensor(dataset.train.x, device), _tensor(dataset.train.y, device)
        xv, yv = _tensor(dataset.val.x, device), _tensor(dataset.val.y, device)
        for total_steps, lr, cid in candidates:
            config = base.override(seed_list=(seed,), total_steps=total_steps, lr=lr,
                eval_interval=max(20, total_steps // 20))
            job = root / "jobs" / f"steps_{total_steps}_lr_{lr:g}" / f"seed_{seed}" / cid
            job.mkdir(parents=True)
            resolved = config.to_resolved_dict()
            # The legacy config remains unchanged; this overlay states the 1B protocol.
            resolved.update(lr_selection="validation_grid_1b", lr_grid_executed=True,
                phase="replication" if replication else "exploratory_tuning",
                test_protocol="final_step_once" if replication else "forbidden")
            (job / "config.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
            (job / "config_sha256.txt").write_text(sha(job / "config.resolved.yaml") + "\n", encoding="utf-8")
            model = CoNNMLP(base.n_features, base.hidden_sizes).to(device)
            model.load_state_dict(init)
            optimizer, hparams = build_optimizer(model, lr)
            access = HeldOutSplit(_tensor(dataset.test.x, device), _tensor(dataset.test.y, device)) if replication else ValidationOnly()
            result = train_condition(model=model, optimizer=optimizer, optimizer_hparams=hparams,
                config=config, gate_builder=GateBuilder(geometry, masks, cid, config.tau_floor), order=order,
                x_train=xtr, y_train=ytr, x_val=xv, y_val=yv,
                motif_a_train=dataset.train.motif_a, motif_b_train=dataset.train.motif_b,
                motif_a_val=dataset.val.motif_a, motif_b_val=dataset.val.motif_b,
                test_access=access, base_dist_ih=geometry.dist_ih, base_dist_hh=geometry.dist_hh,
                seed=seed, condition_id=cid, run_id=root.name, data_fingerprint=dataset.fingerprint(),
                init_fingerprint=fingerprint_state(init), artifact_dir=job, evaluate_test=replication)
            if result.non_finite or result.test_evaluations != int(replication):
                raise RuntimeError(f"Failed run {job}; no imputation or replacement seed allowed")
            for name, rows in (("metrics", result.metrics), ("gradients", result.gradients),
                ("ablation", result.ablation), ("probes", result.probes), ("row_energy", result.row_energy)):
                frame = pd.DataFrame(rows)
                frame["total_steps"], frame["candidate_lr"] = total_steps, lr
                frame.to_csv(job / f"{name}.csv", index=False)
            checkpoint_ablation(job, config, geometry, xv, yv)
            final = [r for r in result.metrics if r["measure_timing"] == "post_step" and r["optimizer_steps_completed"] == total_steps]
            score = dict(seed=seed, condition_id=cid, total_steps=total_steps, lr=lr,
                test_evaluations=result.test_evaluations, data_fingerprint=dataset.fingerprint(),
                init_fingerprint=fingerprint_state(init), job=job.relative_to(root).as_posix())
            for r in final:
                score[f"{r['split']}_accuracy"] = r["accuracy"]
                score[f"{r['split']}_loss"] = r["loss"]
            if not np.isfinite([score["val_accuracy"], score["val_loss"], score["train_loss"]]).all():
                raise RuntimeError(f"Nonfinite endpoint {job}")
            scores.append(score)
            pd.DataFrame(scores).to_csv(root / "scores.csv", index=False)
            print(f"{root.name}: {len(scores)}/{len(seeds)*len(candidates)} seed={seed} {cid} steps={total_steps} lr={lr} val={score['val_accuracy']:.4f}", flush=True)
    # Audit every duration/rate cell, including each unit, rather than averaging energies.
    audits = []
    for steps in sorted({c[0] for c in candidates}):
        rates = [None] if replication else sorted({c[1] for c in candidates})
        for rate in rates:
            subset = [s for s in scores if s["total_steps"] == steps and (rate is None or s["lr"] == rate)]
            frames = [pd.read_csv(root / s["job"] / "metrics.csv") for s in subset]
            energies = [pd.read_csv(root / s["job"] / "row_energy.csv") for s in subset]
            metrics = pd.concat(frames, ignore_index=True)
            for _, group in metrics.groupby("seed"):
                assert group.data_fingerprint.nunique() == group.init_fingerprint.nunique() == 1
            audit = audit_invariants(metrics, pd.concat(energies, ignore_index=True))
            audits.append(dict(total_steps=steps, lr=rate, **audit))
    dump(root / "invariant_audit.json", audits)
    return pd.DataFrame(scores)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPO / "results/runs")
    args = parser.parse_args(argv)
    root = args.output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_experiment1b_v1")
    root.mkdir(parents=True, exist_ok=False)
    flags = configure_runtime()
    base = load_config(REPO / "configs/pilot.yaml")
    pilot_manifest = tree_manifest(PILOT)
    dump(root / "pilot_manifest.before.json", pilot_manifest)
    source_files = [p for folder in ("src", "tests", "configs") for p in (REPO / folder).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts]
    source_files += [REPO / x for x in ("pyproject.toml", "README.md", "DESIGN_DECISIONS.md")]
    source_manifest = {}
    for path in sorted(source_files):
        relative = path.relative_to(REPO)
        target = root / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_manifest[relative.as_posix()] = sha(path)
    dump(root / "source_manifest.json", source_manifest)
    source_digest = sha(root / "source_manifest.json")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    environment = dict(created_utc=utc(), python=sys.version, executable=sys.executable,
        platform=platform.platform(), torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
        runtime=flags, device=base.device, git_revision=git.stdout.strip() if git.returncode == 0 else None,
        thread_environment={key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        git_limitation=None if git.returncode == 0 else "Source directory is not a Git repository; content-addressed source snapshot is the revision.",
        source_manifest_sha256=source_digest, command=sys.argv)
    dump(root / "environment.json", environment)
    packages = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (root / "requirements.freeze.txt").write_text(packages.stdout, encoding="utf-8")
    protocol = dict(version="1B-v1", created_utc=utc(), base_config=base.to_resolved_dict(),
        base_config_sha256=config_sha256(base), pilot=str(PILOT), pilot_config_sha256=sha(PILOT / "config.resolved.yaml"),
        lr_grid=list(base.lr_grid), duration_grid=list(STEPS), conditions=[c.id for c in CONDITIONS],
        tuning_seeds=list(TUNING_SEEDS), replication_seeds=list(REPLICATION_SEEDS),
        candidates_per_condition=12, tuning_runs_per_condition=60,
        selection="per-condition per-duration mean validation accuracy descending, mean validation loss ascending, lr ascending; common duration by macro mean across seven selected conditions, accuracy descending, loss ascending, steps ascending",
        test_usage="zero tuning evaluations; once per replication model, final step only; ablations validation only",
        primary="paired final test accuracy D minus B_prime, independent replication only",
        interval="two-sided Student t 95% on seed differences, df=19; sole primary contrast",
        effect_size="Cohen dz = mean paired difference / sample SD of differences; raw mean also in percentage points",
        diagnostics="20 equally spaced measurement intervals plus schedule boundaries; train/val curves, effective weights, per-unit row gate energy, fixed-batch pre-step gradients, train-fit motif probes",
        checkpoint_ablation="reload pre-release and post-release checkpoints at identical weights, plus final; far=base distance >= linear quantile .75; suppress ih, hh, both; validation loss/accuracy changes; no retraining",
        duration_interpretation="fresh training at each duration with schedule scaled to that budget, not checkpoints from one 3200-step trajectory",
        source_manifest_sha256=source_digest)
    seal(root / "protocol.frozen.json", protocol)
    (root / "config.resolved.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    (root / "config_sha256.txt").write_text(sha(root / "config.resolved.yaml") + "\n", encoding="utf-8")
    print(f"Experiment root: {root}", flush=True)
    tuning_candidates = [(steps, lr, c.id) for steps in STEPS for lr in base.lr_grid for c in CONDITIONS]
    tuning = run_phase(root / "tuning", base, TUNING_SEEDS, tuning_candidates)
    grid, best, duration, chosen_steps, chosen_lr = select_validation(tuning)
    grid.to_csv(root / "tuning/grid_validation.csv", index=False)
    best.to_csv(root / "tuning/selected_lr_by_duration.csv", index=False)
    duration.to_csv(root / "tuning/duration_validation.csv", index=False)
    freeze = dict(frozen_utc=utc(), total_steps=chosen_steps, lr_by_condition=chosen_lr,
        replication_seeds=list(REPLICATION_SEEDS), tuning_scores_sha256=sha(root / "tuning/scores.csv"),
        protocol_sha256=sha(root / "protocol.frozen.json"), source_manifest_sha256=source_digest,
        primary=protocol["primary"], selection=protocol["selection"],
        replication_test_data_evaluated=False)
    seal(root / "selection.frozen.json", freeze)
    print(f"FROZEN BEFORE REPLICATION: {freeze}", flush=True)
    dump(root / "replication.started.json", dict(started_utc=utc(), selection_sha256=sha(root / "selection.frozen.json")))
    replication = run_phase(root / "replication", base, REPLICATION_SEEDS,
        [(chosen_steps, chosen_lr[c.id], c.id) for c in CONDITIONS], replication=True)
    from conn_exp.experiment1b_report import build_report
    build_report(root)
    after = tree_manifest(PILOT)
    dump(root / "pilot_manifest.after.json", after)
    assert after == pilot_manifest, "Pilot artifacts changed"
    assert all(sha(REPO / p) == digest for p, digest in source_manifest.items()), "Source changed during study"
    assert sha(root / "selection.frozen.json") == (root / "selection.frozen.json.sha256").read_text().strip()
    dump(root / "completion.json", dict(completed_utc=utc(), tuning_models=len(tuning), replication_models=len(replication),
        pilot_unchanged=True, source_unchanged=True, frozen_selection_verified=True))
    print(f"COMPLETE: {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
