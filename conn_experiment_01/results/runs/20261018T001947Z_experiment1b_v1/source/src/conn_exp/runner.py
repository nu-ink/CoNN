"""Command line entry for a paired run and for rebuilding tables from its logs."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from conn_exp import __version__
from conn_exp.config import ExperimentConfig, config_sha256, load_config
from conn_exp.data import assert_dataset_invariants, generate_dataset
from conn_exp.evaluation import rebuild_tables
from conn_exp.gates import GateBuilder, draw_masks
from conn_exp.geometry import COORDINATE_CONVENTION, make_task_geometry
from conn_exp.models import CoNNMLP, build_optimizer, fingerprint_state, initial_state_dict
from conn_exp.plots import write_all_plots
from conn_exp.runtime import configure_runtime, resolve_device
from conn_exp.schema import (
    ABLATION_COLUMNS,
    CONDITIONS,
    GRADIENT_COLUMNS,
    METRIC_COLUMNS,
    PRIMARY_CONTROL,
    PRIMARY_CONTROL_NAME,
    PRIMARY_METRIC,
    PRIMARY_TASK,
    PRIMARY_TREATMENT,
    PROBE_COLUMNS,
    ROW_ENERGY_COLUMNS,
)
from conn_exp.train import HeldOutSplit, minibatch_order, train_condition

TEST_PROTOCOL_TEXT = (
    "Coordinates are functions of feature count and hidden width only. "
    "The test split is not an argument to geometry, schedules, masks, probes, or checkpoint selection. "
    "Probes fit on training motif labels and score validation. "
    "Test accuracy is computed once, after total_steps optimizer updates, and that value is the recorded score."
)


def _write_frame(rows: list[dict[str, object]], columns: tuple[str, ...], path: Path) -> None:
    pd.DataFrame(rows, columns=list(columns)).to_csv(path, index=False)


def _tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array)).to(device=device, dtype=torch.float32)


def _save_splits(path: Path, data: object) -> None:
    payload: dict[str, np.ndarray] = {}
    for name in ("train", "val", "test"):
        split = getattr(data, name)
        payload[f"x_{name}"] = split.x
        payload[f"y_{name}"] = split.y
        payload[f"motif_a_{name}"] = split.motif_a
        payload[f"motif_b_{name}"] = split.motif_b
        payload[f"motif_a_pos_{name}"] = split.motif_a_pos
        payload[f"motif_b_pos_{name}"] = split.motif_b_pos
        payload[f"distractor_pos_{name}"] = split.distractor_pos
        payload[f"cell_{name}"] = split.cell.astype(str)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def execute(config: ExperimentConfig, run_dir: Path, command: str, smoke: bool) -> int:
    flags = configure_runtime()
    device = resolve_device(config.device)
    run_dir.mkdir(parents=True, exist_ok=True)
    digest = config_sha256(config)
    resolved = config.to_resolved_dict()
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    (run_dir / "config_sha256.txt").write_text(digest + "\n", encoding="utf-8")
    run_id = run_dir.name
    protocol = {
        "run_id": run_id,
        "experiment_version": __version__,
        "smoke": smoke,
        "task": config.task,
        "seed_list": list(config.seed_list),
        "total_steps": config.total_steps,
        "release_step": config.release_step,
        "release_fraction": config.release_fraction,
        "tau_start": config.tau_start,
        "tau_end": config.tau_end,
        "d50_start": config.d50_start,
        "d50_end": config.d50_end,
        "lr": config.lr,
        "lr_grid": list(config.lr_grid),
        "lr_selection": config.lr_selection,
        "lr_grid_executed": False,
        "optimizer": "sgd",
        "momentum": 0.0,
        "weight_decay": 0.0,
        "optimizer_foreach": False,
        "device": str(device),
        "primary_treatment": PRIMARY_TREATMENT,
        "primary_control": PRIMARY_CONTROL,
        "primary_control_name": PRIMARY_CONTROL_NAME,
        "primary_metric": PRIMARY_METRIC,
        "primary_task": PRIMARY_TASK,
        "config_sha256": digest,
        "coordinate_convention": COORDINATE_CONVENTION,
        "permutation_rule": "within_row_column_permutation_frozen",
        "test_protocol_text": TEST_PROTOCOL_TEXT,
        "distance_bins_use": "unpermuted task coordinates for every condition, including B_prime and C",
        "runtime_flags": flags,
    }
    environment = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "command": command,
        "config_sha256": digest,
        "runtime_flags": flags,
    }
    (run_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    metric_rows: list[dict[str, object]] = []
    gradient_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    energy_rows: list[dict[str, object]] = []
    blocked = False
    for seed in config.seed_list:
        dataset = generate_dataset(config, int(seed))
        assert_dataset_invariants(dataset, config)
        _save_splits(run_dir / "data" / f"seed_{seed}.npz", dataset)
        masks = draw_masks(config.n_features, config.hidden_sizes[0], config.hidden_sizes[1], int(seed))
        mask_path = run_dir / "masks" / f"seed_{seed}.pt"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "input_perm": masks.input_perm,
                "row_perm_ih": masks.row_perm_ih,
                "row_perm_hh": masks.row_perm_hh,
                "permutation_type": masks.permutation_type,
            },
            mask_path,
        )
        geometry = make_task_geometry(config.n_features, config.hidden_sizes)
        init_state = initial_state_dict(config.n_features, config.hidden_sizes, int(seed))
        data_fingerprint = dataset.fingerprint()
        init_fingerprint = fingerprint_state(init_state)
        order = minibatch_order(config.n_train, int(seed))
        x_train = _tensor(dataset.train.x, device)
        y_train = _tensor(dataset.train.y, device)
        x_val = _tensor(dataset.val.x, device)
        y_val = _tensor(dataset.val.y, device)
        x_test = _tensor(dataset.test.x, device)
        y_test = _tensor(dataset.test.y, device)
        for condition in CONDITIONS:
            model = CoNNMLP(config.n_features, config.hidden_sizes)
            model.load_state_dict(init_state)
            model.to(device)
            optimizer, hparams = build_optimizer(model, config.lr)
            access = HeldOutSplit(x_test, y_test)
            result = train_condition(
                model=model,
                optimizer=optimizer,
                optimizer_hparams=hparams,
                config=config,
                gate_builder=GateBuilder(geometry, masks, condition.id, config.tau_floor),
                order=order,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                motif_a_train=dataset.train.motif_a,
                motif_b_train=dataset.train.motif_b,
                motif_a_val=dataset.val.motif_a,
                motif_b_val=dataset.val.motif_b,
                test_access=access,
                base_dist_ih=geometry.dist_ih,
                base_dist_hh=geometry.dist_hh,
                seed=int(seed),
                condition_id=condition.id,
                run_id=run_id,
                data_fingerprint=data_fingerprint,
                init_fingerprint=init_fingerprint,
                artifact_dir=run_dir,
            )
            if result.non_finite or result.test_evaluations != 1:
                blocked = True
            metric_rows.extend(result.metrics)
            gradient_rows.extend(result.gradients)
            ablation_rows.extend(result.ablation)
            probe_rows.extend(result.probes)
            energy_rows.extend(result.row_energy)
            print(
                f"seed {seed} {condition.id}: non_finite={result.non_finite} test_evaluations={result.test_evaluations}",
                flush=True,
            )
    _write_frame(metric_rows, METRIC_COLUMNS, run_dir / "metrics.csv")
    _write_frame(gradient_rows, GRADIENT_COLUMNS, run_dir / "gradients.csv")
    _write_frame(ablation_rows, ABLATION_COLUMNS, run_dir / "ablation.csv")
    _write_frame(probe_rows, PROBE_COLUMNS, run_dir / "probes.csv")
    _write_frame(energy_rows, ROW_ENERGY_COLUMNS, run_dir / "row_energy.csv")
    if blocked:
        (run_dir / "report.md").write_text(
            "# CoNN Experiment 01 report\n\n"
            "This run is blocked. At least one condition produced a non-finite loss or did not evaluate "
            "the test split exactly once at the final step. No primary contrast was aggregated, and no "
            "missing score was imputed.\n",
            encoding="utf-8",
        )
        (run_dir / "summary.json").write_text(
            json.dumps({"blocked": True, "reason": "non-finite loss or test-evaluation contract failure"}, indent=2),
            encoding="utf-8",
        )
        print(f"Blocked run directory: {run_dir}", flush=True)
        return 1
    try:
        rebuild_tables(run_dir)
        write_all_plots(run_dir)
    except Exception:
        (run_dir / "contract_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    print(f"Run directory: {run_dir}", flush=True)
    return 0


def regenerate(run_dir: Path) -> int:
    rebuild_tables(run_dir)
    write_all_plots(run_dir)
    print(f"Regenerated tables and plots in {run_dir}", flush=True)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CoNN Experiment 01. Paired SGD comparison of a structured Gaussian curriculum "
            "against controls that keep gate energy but break spatial alignment. "
            "The pre-registered contrast is D minus B_prime (within-row shuffle) on AND."
        ),
        epilog=(
            "Examples:\n"
            "  python -m conn_exp.runner --config configs/pilot.yaml --smoke\n"
            "  python -m conn_exp.runner --config configs/pilot.yaml --seeds 5\n"
            "  python -m conn_exp.runner --regenerate --run-dir results/runs/<run_id>\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, help="Path to the YAML config.")
    parser.add_argument("--smoke", action="store_true", help="Apply the config's smoke overrides and train every condition once.")
    parser.add_argument("--seeds", type=int, default=None, help="Use this many seeds from the config list.")
    parser.add_argument("--output-root", type=Path, default=Path("results/runs"))
    parser.add_argument("--run-dir", type=Path, default=None, help="Exact output directory, or the directory to regenerate.")
    parser.add_argument("--regenerate", action="store_true", help="Rebuild tables, plots, and the report from logged metrics.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.regenerate:
        if args.run_dir is None:
            raise SystemExit("--regenerate requires --run-dir")
        return regenerate(args.run_dir)
    if args.config is None:
        raise SystemExit("--config is required")
    if args.smoke and args.seeds is not None:
        print("Ignoring --seeds because --smoke fixes the seed list in the config.", file=sys.stderr)
    config = load_config(args.config, smoke=args.smoke)
    if args.seeds is not None and not args.smoke:
        if args.seeds < 1:
            raise SystemExit("--seeds must be positive")
        chosen = tuple(config.seed_list[: args.seeds])
        if len(chosen) < args.seeds:
            chosen = tuple(range(args.seeds))
        print(f"Using seeds {list(chosen)}", flush=True)
        config = config.override(seed_list=chosen)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    kind = "smoke" if args.smoke else ("pilot" if len(config.seed_list) == 5 and config.task == "and" else "run")
    run_dir = args.run_dir or (args.output_root / f"{stamp}_{kind}")
    command = " ".join(sys.argv if argv is None else ["python -m conn_exp.runner", *argv])
    return execute(config, run_dir, command, args.smoke)


if __name__ == "__main__":
    raise SystemExit(main())
