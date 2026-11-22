"""Read-only scientific audit; creates a unique evidence run, never trains models.

Run with the existing Experiment 2 Python environment from this directory.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OLD = ROOT.parent / "conn_experiment_01"
OLD_RUN = OLD / "results/runs/20261019T004346Z_experiment2_har_v1"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def manifest(paths):
    return {str(p): {"sha256": sha(p), "bytes": p.stat().st_size,
                     "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted(set(paths)) if p.is_file()}


def main():
    started = time.perf_counter()
    run = ROOT / "results/runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_audit_recovery_v1")
    # Collect pre-existing evidence before creating the run or importing project modules.
    original = [p for p in ROOT.rglob("*") if p.is_file() and "runs" not in p.relative_to(ROOT).parts]
    old_files = [p for p in OLD_RUN.rglob("*") if p.is_file()]
    old_files += [p for folder in ("src", "tests", "configs", "scripts") for p in (OLD / folder).rglob("*") if p.is_file()]
    old_files += [OLD / "data/raw/uci_har_original.zip"]
    run.mkdir(parents=True, exist_ok=False)

    def write(name, data):
        (run / name).write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")

    before = manifest(old_files)
    write("experiment2.before.json", before)
    initial = manifest(original)
    write("associative.before.json", initial)
    for path in original:
        target = run / "original_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write("initial_evidence.json", {
        "classification": "Not implemented (training, context, memory); Phase 0/1 implementation and inspection exist",
        "historical_inspection_is_not_training_log": True,
        "prior_pytest_nodeids": json.loads((ROOT / ".pytest_cache/v/cache/nodeids").read_text()),
        "prior_pytest_lastfailed": json.loads((ROOT / ".pytest_cache/v/cache/lastfailed").read_text()),
        "process_inventory_limitation": "PowerShell Get-CimInstance Win32_Process returned Access denied; Get-Process python* returned no rows before audit",
        "inspection_result_models_trained_field": "literal zero in source, not an independently maintained training registry",
        "snapshot_note": "audit_recovery.py is newly added audit tooling; all other snapshot files predate this investigation",
    })
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    commands = []

    def command(name, args, cwd=ROOT):
        t = time.perf_counter()
        result = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, timeout=180)
        (run / (name + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
        commands.append({"name": name, "argv": args, "cwd": str(cwd), "exit_code": result.returncode,
                         "elapsed_seconds": time.perf_counter() - t})
        write("commands.json", commands)
        return result.returncode

    command("git_workspace", ["git", "status", "--short"], ROOT.parent)
    command("git_experiment", ["git", "log", "-6", "--oneline"])
    command("pip_check", [sys.executable, "-m", "pip", "check"])
    command("requirements", [sys.executable, "-m", "pip", "freeze"])
    gpu = shutil.which("nvidia-smi")
    if gpu:
        command("nvidia_smi", [gpu])
    import torch
    import numpy as np
    from conn_assoc import data, gate, geometry, inspect, protocol
    from conn_exp.gates import gaussian_gate as reference_gate
    write("environment.json", {"python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda, "nvidia_smi": gpu,
        "imports": {m: importlib.metadata.version(m) for m in ("torch", "numpy", "PyYAML", "pytest", "matplotlib")},
        "conn_assoc": str(data.__file__), "original_gate": str(sys.modules[reference_gate.__module__].__file__)})
    syntax = []
    for path in (ROOT / "src").rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        syntax.append(str(path))
    write("syntax_imports.json", {"status": "passed", "parsed_files": syntax})
    test_rc = command("pytest", [sys.executable, "-m", "pytest", "-v", "-rs", "--durations=10",
        "-o", f"cache_dir={run / 'pytest_cache'}", f"--junitxml={run / 'pytest.xml'}"])
    # Call the pure function, not main(), which would overwrite audit/inspection.json.
    t = time.perf_counter()
    inspected = inspect.run_inspection()
    write("inspection.json", inspected)
    write("inspection_timing.json", {"seconds": time.perf_counter() - t, "entry": "conn_assoc.inspect.run_inspection()", "original_inspection_overwritten": False})
    torch.manual_seed(104729)
    torch.set_num_threads(1)
    distance = geometry.pairwise_abs_distance(geometry.uniform_coordinates(64), geometry.temporal_input_coordinates())
    rows = []
    for tau in (1e-8, 1e-6, .06794574402304153, .21233045007200477, .8493218002880191):
        g, floor = gate.gaussian_gate(distance, tau)
        reference, ref_floor = reference_gate(distance, tau)
        assert torch.equal(g, reference) and floor == ref_floor
        rows.append({"tau": tau, "floored": floor, "min": float(g.min()), "max": float(g.max()),
                     "mean": float(g.mean()), "row_energy_mean": float(g.square().sum(1).mean()),
                     "original_max_abs_difference": float((g-reference).abs().max())})
    with (run / "gate_measurements.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    # CPU primitive smoke only: actual HAR development rows, no optimizer or model training.
    sample = torch.from_numpy(data.load_prepared("temporal", "train", "x", "development")[:8])
    weight = torch.nn.Parameter(torch.randn(64, 1152) * .01)
    g, _ = gate.gaussian_gate(distance, .06794574402304153)
    out = torch.nn.functional.linear(sample, gate.training_effective_weight(weight, g))
    objective = out.square().mean()
    objective.backward()
    assert torch.isfinite(objective) and weight.grad is not None and torch.isfinite(weight.grad).all()
    analytic = ((2 * out.detach() / out.numel()).T @ sample) * g
    assert torch.allclose(weight.grad, analytic, atol=1e-7, rtol=1e-5)
    write("cpu_smoke.json", {"status": "passed", "scope": "gate/linear/autograd primitive on eight real fit-split windows; no optimizer or model training",
        "input_shape": list(sample.shape), "output_shape": list(out.shape), "finite_quadratic_objective": float(objective.detach()),
        "gradient_norm": float(weight.grad.norm()), "gradient_max_abs_formula_error": float((weight.grad-analytic).abs().max()),
        "model_weights_updated": False, "checkpoints_written": False, "classification_metrics": None})
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    d = torch.linspace(0, 1, 201)
    for tau in (.06794574402304153, .21233045007200477, .8493218002880191):
        ax.plot(d.numpy(), gate.gaussian_gate(d, tau)[0].numpy(), label=f"tau={tau:.5f}")
    ax.set(xlabel="Absolute coordinate distance", ylabel="Gaussian gate magnitude", title="Phase 1 gate arithmetic (not model performance)")
    ax.legend(); fig.tight_layout(); fig.savefig(run / "gaussian_gate_diagnostic.png", dpi=160); plt.close(fig)
    # Historical results are copied and labelled, not recomputed from new test predictions.
    for name in ("primary_summary.json", "paired_statistical_summary.csv", "subject_level_summary.csv", "condition_summary.csv"):
        source = OLD_RUN / "replication" / name
        if source.exists():
            dest = run / "historical_experiment2" / name; dest.parent.mkdir(exist_ok=True); shutil.copy2(source, dest)
    history = OLD_RUN / "replication/paired_statistical_summary.csv"
    if history.exists():
        historical = list(csv.DictReader(history.open(encoding="utf-8")))
        write("historical_schema.json", {"columns": list(historical[0]) if historical else [], "rows": len(historical)})
    after = manifest(old_files)
    write("experiment2.after.json", after)
    changed_old = [p for p in before if before[p] != after[p]]
    changed_associative = [p for p in initial if manifest([Path(p)])[p] != initial[p]]
    write("preservation.json", {"experiment2_files_checked": len(before), "experiment2_changed": changed_old,
        "preexisting_associative_files_checked": len(initial), "preexisting_associative_changed": changed_associative,
        "protocol_sha256": protocol.protocol_sha256(), "matches_frozen_manifest": protocol.protocol_sha256() == protocol.manifest()["protocol_sha256"]})
    write("completion.json", {"audit_completed_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter()-started,
        "pytest_exit_code": test_rc, "new_models_trained": 0, "new_final_test_model_evaluations": 0,
        "training_resume": "not authorized in phase_0_and_1 scope; training entry point does not exist",
        "gpu_smoke": "not run: CPU-only PyTorch" if not torch.cuda.is_available() else "not run: phase-limited audit",
        "context_tests": "not implemented; not run", "memory_tests": "not implemented; not run"})
    assert not changed_old and not changed_associative
    print(run)
    return test_rc


if __name__ == "__main__":
    raise SystemExit(main())
