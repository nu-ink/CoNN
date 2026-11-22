"""Archive CoNN experiments into ORGANIZED_EXPERIMENTS without altering originals.

Copies historical source, configuration, and small summary artifacts.
References large generated trees instead of duplicating them.
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
from pathlib import Path

ROOT = Path(r"E:\CoNN")
DEST = ROOT / "ORGANIZED_EXPERIMENTS"
SKIP_DIR_NAMES = {
    "AUDIT_20261020",
    "ORGANIZED_EXPERIMENTS",
    ".git",
    ".venv",
    ".venv_cuda23",
    ".pip-cache23",
    ".pytest_cache",
    "__pycache__",
    "site-packages",
}

records: list[dict[str, str]] = []
copied_destinations: set[Path] = set()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(src: Path, rel: str, experiment: str, reason: str) -> None:
    src = src.resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    target = DEST / rel
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    original_hash = sha256(src)
    size = src.stat().st_size
    shutil.copy2(src, target)
    copy_hash = sha256(target)
    copied_destinations.add(target.resolve())
    records.append(
        {
            "original_absolute_path": str(src),
            "new_relative_path": rel.replace("\\", "/"),
            "experiment": experiment,
            "size_bytes": str(size),
            "sha256": original_hash,
            "sha256_copy": copy_hash,
            "match": str(original_hash == copy_hash),
            "git_commit": "unavailable_no_git_repository",
            "action": "copied",
            "reason": reason,
        }
    )


def copy_tree(src_dir: Path, rel_dir: str, experiment: str, reason: str, exclude_names: set[str] | None = None) -> None:
    exclude_names = exclude_names or set()
    for path in src_dir.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.name in exclude_names:
            continue
        rel = Path(rel_dir) / path.relative_to(src_dir)
        copy_file(path, str(rel), experiment, reason)


def reference(src: Path, rel_note: str, experiment: str, reason: str) -> None:
    src = src.resolve()
    note = DEST / rel_note
    if note.exists():
        raise FileExistsError(note)
    note.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        size = src.stat().st_size
        count = 1
    else:
        files = [p for p in src.rglob("*") if p.is_file()]
        size = sum(p.stat().st_size for p in files)
        count = len(files)
    text = (
        f"Reference only. This organized tree does not duplicate the original artifact.\n"
        f"Original: {src}\n"
        f"Exists: {src.exists()}\n"
        f"Files: {count}\n"
        f"Bytes: {size}\n"
        f"Reason: {reason}\n"
        f"Per-file SHA-256 was not computed for this referenced tree.\n"
    )
    note.write_text(text, encoding="utf-8")
    records.append(
        {
            "original_absolute_path": str(src),
            "new_relative_path": rel_note.replace("\\", "/"),
            "experiment": experiment,
            "size_bytes": str(size),
            "sha256": "DIRECTORY_OR_LARGE_FILE_NOT_HASHED",
            "sha256_copy": "",
            "match": "reference_only",
            "git_commit": "unavailable_no_git_repository",
            "action": "referenced",
            "reason": f"{reason} | note={rel_note} | files={count}",
        }
    )


def write_text(rel: str, text: str, experiment: str, reason: str) -> None:
    target = DEST / rel
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")
    records.append(
        {
            "original_absolute_path": "",
            "new_relative_path": rel.replace("\\", "/"),
            "experiment": experiment,
            "size_bytes": str(target.stat().st_size),
            "sha256": "",
            "sha256_copy": sha256(target),
            "match": "organization_note",
            "git_commit": "unavailable_no_git_repository",
            "action": "created_note",
            "reason": reason,
        }
    )


def main() -> None:
    exp1 = ROOT / "conn_experiment_01"
    assoc = ROOT / "CoNN_Associative_Relationships"
    snap1b = exp1 / "results/runs/20261018T001947Z_experiment1b_v1/source"
    snap1c = exp1 / "results/runs/20261018T024146Z_experiment1c_v1/source"
    snap2 = exp1 / "results/runs/20261019T004346Z_experiment2_har_v1/source"
    phase_run = assoc / "results/runs/20261019T030442340025Z_phase23_v1"
    phase_source = phase_run / "source"
    run3b = assoc / "phase3b_context_diagnostics/results/20261020_phase3b_v3_run01"

    # Experiment 01: earliest frozen tree, without the 1B-only modules.
    copy_tree(
        snap1b,
        "EXPERIMENT_01/source",
        "EXPERIMENT_01",
        "Earliest frozen source tree, taken at the start of Experiment 1B. Experiment 1 has no separate snapshot. 1B-only modules are excluded from this copy.",
        exclude_names={"experiment1b.py", "experiment1b_report.py", "test_experiment1b.py"},
    )
    copy_file(exp1 / "RESULTS.md", "EXPERIMENT_01/documentation/RESULTS.md", "EXPERIMENT_01", "Pilot and smoke results narrative in the current experiment-01 tree.")
    copy_file(exp1 / "DESIGN_DECISIONS.md", "EXPERIMENT_01/documentation/DESIGN_DECISIONS.md", "EXPERIMENT_01", "Design record copied again beside the snapshot so the experiment directory is readable on its own. The snapshot also contains this file if it was present at freeze time.")
    brief = ROOT / "CoNN_Cursor_Experiment_01"
    copy_file(brief / "CURSOR_BUILD_BRIEF.md", "EXPERIMENT_01/documentation/CURSOR_BUILD_BRIEF.md", "EXPERIMENT_01", "Original build brief for the seven-condition pilot.")
    copy_file(brief / "START_IN_CURSOR.md", "EXPERIMENT_01/documentation/START_IN_CURSOR.md", "EXPERIMENT_01", "Cursor start note accompanying the build brief.")
    for run_name, label in (
        ("20261017T235744Z_smoke", "smoke"),
        ("20261017T235839Z_pilot", "pilot"),
    ):
        run = exp1 / "results/runs" / run_name
        for name in (
            "report.md",
            "scores.csv",
            "summary.json",
            "protocol.json",
            "paired_contrast.csv",
            "config.resolved.yaml",
            "environment.json",
            "invariant_audit.json",
        ):
            copy_file(run / name, f"EXPERIMENT_01/results/{label}/{name}", "EXPERIMENT_01", f"Small summary artifact from the {label} run. The rest of the run is referenced.")
        reference(run, f"EXPERIMENT_01/results/{label}/FULL_RUN_REFERENCE.txt", "EXPERIMENT_01", "Full smoke or pilot run, including checkpoints, curves, and plots.")
    reference(exp1 / "configs/pilot.yaml", "EXPERIMENT_01/configs/ORIGINAL_CONFIG_REFERENCE.txt", "EXPERIMENT_01", "Authoritative pilot config. A frozen copy is inside EXPERIMENT_01/source/configs/pilot.yaml from the 1B snapshot.")
    reference(snap1b, "EXPERIMENT_01/documentation/SOURCE_SNAPSHOT_REFERENCE.txt", "EXPERIMENT_01", "Full 1B-era snapshot from which the Experiment 01 source copy was taken, excluding 1B-only filenames.")

    # Experiment 1B
    copy_tree(snap1b, "EXPERIMENT_01B/source", "EXPERIMENT_01B", "Source snapshot frozen with the Experiment 1B run.")
    run = exp1 / "results/runs/20261018T001947Z_experiment1b_v1"
    for name in (
        "research_summary.md",
        "protocol.frozen.json",
        "selection.frozen.json",
        "config.resolved.yaml",
        "environment.json",
        "requirements.freeze.txt",
        "completion.json",
        "replication.started.json",
        "source_manifest.json",
    ):
        copy_file(run / name, f"EXPERIMENT_01B/documentation/{name}", "EXPERIMENT_01B", "Experiment 1B protocol or summary file.")
    copy_file(run / "replication/scores.csv", "EXPERIMENT_01B/results/replication_scores.csv", "EXPERIMENT_01B", "Replication score table used for the published D minus B prime mean.")
    copy_file(run / "replication/report.md", "EXPERIMENT_01B/results/replication_report.md", "EXPERIMENT_01B", "Replication report.")
    copy_file(run / "tuning/report.md", "EXPERIMENT_01B/results/tuning_report.md", "EXPERIMENT_01B", "Validation-only tuning report.")
    copy_file(exp1 / "results/experiment1b_execution.log", "EXPERIMENT_01B/logs/experiment1b_execution.log", "EXPERIMENT_01B", "Execution log stored beside the runs directory.")
    reference(run, "EXPERIMENT_01B/results/FULL_RUN_REFERENCE.txt", "EXPERIMENT_01B", "Full 1B run, including per-job checkpoints and curves.")
    reference(run / "replication", "EXPERIMENT_01B/checkpoints/REPLICATION_REFERENCE.txt", "EXPERIMENT_01B", "Replication job directory, including checkpoints.")
    reference(run / "tuning", "EXPERIMENT_01B/results/TUNING_REFERENCE.txt", "EXPERIMENT_01B", "Tuning job directory. No test evaluations.")

    # Experiment 1C
    copy_tree(snap1c, "EXPERIMENT_01C/source", "EXPERIMENT_01C", "Source snapshot frozen with the Experiment 1C run.")
    run = exp1 / "results/runs/20261018T024146Z_experiment1c_v1"
    for name in (
        "research_report.md",
        "measured_findings.md",
        "protocol.frozen.json",
        "selection.frozen.json",
        "config.resolved.yaml",
        "environment.json",
        "requirements.freeze.txt",
        "completion.json",
        "interpret_results.py",
        "verify_outputs.py",
    ):
        copy_file(run / name, f"EXPERIMENT_01C/documentation/{name}", "EXPERIMENT_01C", "Experiment 1C report, protocol, or report-tool script from the run root.")
    copy_file(run / "replication/scores.csv", "EXPERIMENT_01C/results/replication_scores.csv", "EXPERIMENT_01C", "1C replication scores.")
    copy_file(exp1 / "results/experiment1c_execution.log", "EXPERIMENT_01C/logs/experiment1c_execution.log", "EXPERIMENT_01C", "Execution log.")
    reference(run, "EXPERIMENT_01C/results/FULL_RUN_REFERENCE.txt", "EXPERIMENT_01C", "Full 1C run.")
    reference(run / "audit", "EXPERIMENT_01C/documentation/PHASE1_AUDIT_REFERENCE.txt", "EXPERIMENT_01C", "Pre-training audit of Experiment 1B checkpoints. Not a separate experiment.")

    # Experiment 2
    copy_tree(snap2, "EXPERIMENT_02/source", "EXPERIMENT_02", "Source snapshot frozen with the Experiment 2 run.")
    run = exp1 / "results/runs/20261019T004346Z_experiment2_har_v1"
    for name in (
        "research_report.md",
        "protocol.frozen.json",
        "selection.frozen.json",
        "config.resolved.yaml",
        "environment.json",
        "requirements.freeze.txt",
        "completion.json",
        "subject_split.frozen.json",
        "prepared_manifest.json",
        "verify_outputs.py",
    ):
        copy_file(run / name, f"EXPERIMENT_02/documentation/{name}", "EXPERIMENT_02", "Experiment 2 report or freeze file.")
    copy_file(exp1 / "configs/har.yaml", "EXPERIMENT_02/configs/har.yaml", "EXPERIMENT_02", "HAR configuration in the experiment package. The run also stored config.resolved.yaml.")
    copy_file(run / "replication/scores.csv", "EXPERIMENT_02/results/replication_scores.csv", "EXPERIMENT_02", "Experiment 2 replication scores, both representations.")
    copy_file(exp1 / "results/experiment2_execution.log", "EXPERIMENT_02/logs/experiment2_execution.log", "EXPERIMENT_02", "Execution log.")
    reference(run, "EXPERIMENT_02/results/FULL_RUN_REFERENCE.txt", "EXPERIMENT_02", "Full Experiment 2 run.")
    reference(exp1 / "data/raw/uci_har_original.zip", "EXPERIMENT_02/datasets/UCI_HAR_ZIP_REFERENCE.txt", "EXPERIMENT_02", "Official UCI HAR archive. Not duplicated.")
    reference(run / "prepared", "EXPERIMENT_02/datasets/PREPARED_ARRAYS_REFERENCE.txt", "EXPERIMENT_02", "Prepared arrays and scalers derived from the archive.")
    reference(run / "replication", "EXPERIMENT_02/checkpoints/REPLICATION_REFERENCE.txt", "EXPERIMENT_02", "Final-evaluation jobs and checkpoints.")

    # Associative phase 0/1. Earliest snapshot already contains phase-2 files; those names are skipped.
    pre = phase_source / "phase2_pretraining"
    copy_tree(
        pre,
        "EXPERIMENT_03_PHASE01/source",
        "EXPERIMENT_03_PHASE01",
        "Earliest associative snapshot. Phase-2 module filenames are excluded so this directory is not presented as the phase-2 trainer.",
        exclude_names={"phase23_core.py", "phase23.py", "test_phase2.py", "phase23_preflight.py", "PROTOCOL_PHASE23_v1.md", "phase23_active_run.txt"},
    )
    prior = phase_run / "prior_snapshot"
    for rel in (
        "PROTOCOL_v2.md",
        "README.md",
        "MANIFEST.json",
        "config/phase01.yaml",
        "audit/inspection.json",
        "reports/EXPERIMENT_AUDIT_AND_RESULTS.md",
    ):
        copy_file(prior / rel, f"EXPERIMENT_03_PHASE01/documentation/{Path(rel).name}", "EXPERIMENT_03_PHASE01", "File from the pre-training prior snapshot of the associative project.")
    copy_file(assoc / "audit/normalization_history.md", "EXPERIMENT_03_PHASE01/documentation/normalization_history.md", "EXPERIMENT_03_PHASE01", "Normalization limits recorded for the associative protocol.")
    reference(prior, "EXPERIMENT_03_PHASE01/documentation/PRIOR_SNAPSHOT_REFERENCE.txt", "EXPERIMENT_03_PHASE01", "Complete prior snapshot directory.")
    reference(pre, "EXPERIMENT_03_PHASE01/documentation/EARLIEST_CODE_SNAPSHOT_REFERENCE.txt", "EXPERIMENT_03_PHASE01", "Earliest code snapshot. It also contains phase-2 files that were not copied into this experiment.")

    # Phase 2 and Phase 3, one authorized grid, several code stages.
    for stage in ("phase2_pretraining", "phase2_ready", "phase3_ready", "recovery_io_retry", "analysis_ready", "delivery"):
        copy_tree(
            phase_source / stage,
            f"EXPERIMENT_03_PHASE23/source/stages/{stage}",
            "EXPERIMENT_03_PHASE23",
            f"Historical source stage {stage} stored inside the phase 2/3 run. These stages are not interchangeable.",
        )
    copy_file(assoc / "PROTOCOL_PHASE23_v1.md", "EXPERIMENT_03_PHASE23/documentation/PROTOCOL_PHASE23_v1.md", "EXPERIMENT_03_PHASE23", "Phase 2/3 amendment in the associative project root.")
    copy_file(assoc / "README_PHASE23.md", "EXPERIMENT_03_PHASE23/documentation/README_PHASE23.md", "EXPERIMENT_03_PHASE23", "Phase 2/3 readme.")
    copy_file(assoc / "reports/PHASE_2_3_IMPLEMENTATION_AND_RESULTS.md", "EXPERIMENT_03_PHASE23/documentation/PHASE_2_3_IMPLEMENTATION_AND_RESULTS.md", "EXPERIMENT_03_PHASE23", "Development results report for the 1050-fit grid.")
    analysis = phase_run / "analysis"
    for name in (
        "condition_summary.csv",
        "paired_contrasts.csv",
        "selected_seed_scores.csv",
        "selected_seed_context_scores.csv",
        "individual_seed_differences.csv",
    ):
        copy_file(analysis / name, f"EXPERIMENT_03_PHASE23/results/{name}", "EXPERIMENT_03_PHASE23", "Small summary table from the phase 2/3 analysis directory.")
    reference(phase_run, "EXPERIMENT_03_PHASE23/results/FULL_RUN_REFERENCE.txt", "EXPERIMENT_03_PHASE23", "Full phase 2/3 run, including checkpoints and curves.")
    reference(phase_run / "jobs", "EXPERIMENT_03_PHASE23/checkpoints/JOBS_REFERENCE.txt", "EXPERIMENT_03_PHASE23", "Per-fit checkpoints and job logs.")
    reference(assoc / "results/runs", "EXPERIMENT_03_PHASE23/logs/EARLIER_RECOVERY_RUNS_REFERENCE.txt", "EXPERIMENT_03_PHASE23", "Directory containing three audit_recovery runs that preceded the phase 2/3 grid. They are not separate scientific experiments.")

    # Phase 3B
    copy_tree(run3b / "source", "EXPERIMENT_03B/source/frozen_run_snapshot", "EXPERIMENT_03B", "Source snapshot stored inside the Phase 3B run.")
    for name in (
        "PROTOCOL_PHASE3B_v3.md",
        "PROTOCOL_PHASE3B_v2.md",
        "PROTOCOL_PHASE3B_v2_reference.md",
        "README.md",
    ):
        copy_file(assoc / "phase3b_context_diagnostics" / name, f"EXPERIMENT_03B/documentation/{name}", "EXPERIMENT_03B", "Phase 3B protocol or readme from the diagnostics project.")
    reports = assoc / "phase3b_context_diagnostics/reports"
    for path in reports.glob("*.md"):
        copy_file(path, f"EXPERIMENT_03B/documentation/reports/{path.name}", "EXPERIMENT_03B", "Phase 3B report markdown.")
    copy_file(ROOT / "CoNN_Experiment_3B_Independent_Scientific_Review.md", "EXPERIMENT_03B/documentation/CoNN_Experiment_3B_Independent_Scientific_Review.md", "EXPERIMENT_03B", "Independent review stored at the repository root. It is a review of Phase 3B, not a new experiment.")
    analysis = run3b / "analysis"
    for name in (
        "paired_contrasts.csv",
        "condition_scores.csv",
        "seed_differences.csv",
        "subject_differences.csv",
        "omitted_fits.csv",
    ):
        copy_file(analysis / name, f"EXPERIMENT_03B/results/{name}", "EXPERIMENT_03B", "Small Phase 3B summary table.")
    for name in ("protocol_seal.json", "config.frozen.json", "completion.json", "environment.json"):
        copy_file(run3b / name, f"EXPERIMENT_03B/documentation/{name}", "EXPERIMENT_03B", "Phase 3B freeze or completion file from the run root.")
    reference(run3b, "EXPERIMENT_03B/results/FULL_RUN_REFERENCE.txt", "EXPERIMENT_03B", "Full Phase 3B run, including grid checkpoints.")
    reference(run3b / "grid", "EXPERIMENT_03B/checkpoints/GRID_REFERENCE.txt", "EXPERIMENT_03B", "Phase 3B grid checkpoints.")
    reference(assoc / "phase3b_context_diagnostics/protocol_versions", "EXPERIMENT_03B/documentation/PROTOCOL_VERSIONS_REFERENCE.txt", "EXPERIMENT_03B", "Sealed protocol-version directory. Not duplicated.")
    reference(ROOT / "Collapsible Neural Network Infographic.png", "UNASSIGNED_FILES/INFOGRAPHIC_REFERENCE.txt", "UNASSIGNED", "Promotional infographic. It is not a protocol for any numbered experiment.")

    write_shared_docs(snap1b, snap1c, snap2, pre, phase_source / "delivery", run3b / "source")
    write_readmes()
    excluded = write_exclusion_list()
    write_tables()
    write_index_and_report(excluded)
    mismatches = [row for row in records if row["action"] == "copied" and row["match"] != "True"]
    print(f"copied={sum(1 for r in records if r['action']=='copied')} referenced={sum(1 for r in records if r['action']=='referenced')} mismatches={len(mismatches)} excluded={excluded}")
    if mismatches:
        raise SystemExit("checksum mismatch")


def write_shared_docs(*snapshots: Path) -> None:
    lines = [
        "# Shared components",
        "",
        "No module in this directory is a substitute for an experiment snapshot.",
        "Each experiment directory keeps the historical files copied from that experiment's own snapshot.",
        "There is no Git repository, so versions are identified by snapshot path and SHA-256.",
        "",
        "Dependency direction, from import statements and path helpers:",
        "",
        "- Experiments 01, 01B, 01C, and 02 use the `conn_exp` package in their own snapshots.",
        "- Associative Phase 2/3 imports Experiment 2's `conn_exp` package from `conn_experiment_01/src` through `conn_assoc.paths.experiment2_root`.",
        "- Phase 3B imports `conn_assoc.phase3_context` and `conn_assoc.phase23_core` from the associative project, which in turn import Experiment 2.",
        "- Replacing any of those with a later copy would change the historical experiment.",
        "",
        "## Exact-duplicate check across snapshots",
        "",
        "Files with the same relative name and the same SHA-256 are exact copies. A shared name with different hashes is a historical change.",
        "",
    ]
    groups: dict[str, list[tuple[str, str]]] = {}
    interesting = {
        "gates.py",
        "geometry.py",
        "models.py",
        "schedules.py",
        "gate.py",
        "data.py",
        "train.py",
        "runner.py",
    }
    for snap in snapshots:
        label = str(snap.relative_to(ROOT))
        for path in snap.rglob("*"):
            if path.is_file() and path.name in interesting and "__pycache__" not in path.parts:
                groups.setdefault(path.name, []).append((label, sha256(path)))
    for name, items in sorted(groups.items()):
        lines.append(f"### {name}")
        lines.append("")
        unique = {digest for _, digest in items}
        lines.append(f"Distinct hashes: {len(unique)}.")
        lines.append("")
        for label, digest in items:
            lines.append(f"- `{digest}` — `{label}`")
        lines.append("")
    write_text(
        "SHARED_COMPONENTS/documentation/VERSION_MAP.md",
        "\n".join(lines) + "\n",
        "SHARED_COMPONENTS",
        "Cross-snapshot hash map. Not a replacement implementation.",
    )
    write_text(
        "SHARED_COMPONENTS/models/README.md",
        "No model implementation is stored here.\n\nHistorical models remain inside each experiment's `source` directory so a later module cannot be mistaken for an earlier one.\n",
        "SHARED_COMPONENTS",
        "Placeholder explaining why shared model code was not centralized.",
    )
    write_text(
        "SHARED_COMPONENTS/utilities/README.md",
        "No utility implementation is stored here.\n\nGate, geometry, seeding, and path helpers differ across snapshots or are imported across experiment directories. See `documentation/VERSION_MAP.md`.\n",
        "SHARED_COMPONENTS",
        "Placeholder explaining why utilities were not centralized.",
    )
    write_text(
        "SHARED_COMPONENTS/datasets/README.md",
        "No dataset bytes are stored here.\n\nSynthetic data are generated by each experiment's own `data.py`. The UCI HAR archive is referenced from `EXPERIMENT_02/datasets`.\n",
        "SHARED_COMPONENTS",
        "Placeholder explaining why datasets were not centralized.",
    )


def write_readmes() -> None:
    write_text("EXPERIMENT_01/README.md", EXPERIMENT_01_README, "EXPERIMENT_01", "Experiment readme written during organization.")
    write_text("EXPERIMENT_01B/README.md", EXPERIMENT_01B_README, "EXPERIMENT_01B", "Experiment readme written during organization.")
    write_text("EXPERIMENT_01C/README.md", EXPERIMENT_01C_README, "EXPERIMENT_01C", "Experiment readme written during organization.")
    write_text("EXPERIMENT_02/README.md", EXPERIMENT_02_README, "EXPERIMENT_02", "Experiment readme written during organization.")
    write_text("EXPERIMENT_03_PHASE01/README.md", EXPERIMENT_03_PHASE01_README, "EXPERIMENT_03_PHASE01", "Experiment readme written during organization.")
    write_text("EXPERIMENT_03_PHASE23/README.md", EXPERIMENT_03_PHASE23_README, "EXPERIMENT_03_PHASE23", "Experiment readme written during organization.")
    write_text("EXPERIMENT_03B/README.md", EXPERIMENT_03B_README, "EXPERIMENT_03B", "Experiment readme written during organization.")
    for experiment, folder in (
        ("EXPERIMENT_01", "EXPERIMENT_01"),
        ("EXPERIMENT_01B", "EXPERIMENT_01B"),
        ("EXPERIMENT_01C", "EXPERIMENT_01C"),
        ("EXPERIMENT_02", "EXPERIMENT_02"),
        ("EXPERIMENT_03_PHASE01", "EXPERIMENT_03_PHASE01"),
        ("EXPERIMENT_03_PHASE23", "EXPERIMENT_03_PHASE23"),
        ("EXPERIMENT_03B", "EXPERIMENT_03B"),
    ):
        for sub, text in (
            ("configs/README.md", "Configuration files for this experiment are in `source` or `documentation` when a separate frozen config was copied. This file marks the directory.\n"),
            ("datasets/README.md", "Dataset bytes are referenced, not duplicated, when they are large or generated. See any `*_REFERENCE.txt` file in this directory.\n"),
            ("checkpoints/README.md", "Checkpoints remain in the original run directory. See `*_REFERENCE.txt`.\n"),
            ("plots/README.md", "Plots remain in the original run directory and were not duplicated.\n"),
            ("logs/README.md", "Small execution logs are copied when they were stored as single files. Per-job logs remain in the referenced run.\n"),
        ):
            write_text(f"{folder}/{sub}", text, experiment, "Directory marker.")
    write_text(
        "UNASSIGNED_FILES/README.md",
        "\n".join(
            [
                "# Unassigned files",
                "",
                "The infographic at the repository root is referenced here because it is not the protocol of a numbered experiment.",
                "",
                "The following names were looked for and were not found as executed studies: Experiment 3A, Experiment 3C, and Experiment 3D.",
                "Experiment 3D exists only as a proposal inside `AUDIT_20261020`, which was excluded from this organization.",
                "Phase 2 and Phase 3 were one authorized grid and are kept together as `EXPERIMENT_03_PHASE23`. They were not renamed to Experiment 3A.",
                "",
                "Three directories named `audit_recovery_v1` sit beside the phase 2/3 run. They are referenced from that experiment as engineering recovery runs, not as separate scientific experiments. That assignment is a judgment based on their names and timestamps.",
                "",
            ]
        ),
        "UNASSIGNED",
        "Explanation of files and names that were not given experiment directories.",
    )


def write_exclusion_list() -> int:
    copied_sources = {row["original_absolute_path"] for row in records if row["action"] == "copied"}
    path = DEST / "EXCLUDED_FILES.csv"
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["original_absolute_path", "size_bytes", "reason"])
        writer.writeheader()
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [name for name in dirnames if name not in SKIP_DIR_NAMES]
            for name in filenames:
                src = Path(dirpath) / name
                resolved = str(src.resolve())
                if resolved in copied_sources:
                    continue
                try:
                    size = src.stat().st_size
                except OSError:
                    size = -1
                reason = exclusion_reason(src)
                writer.writerow(
                    {
                        "original_absolute_path": resolved,
                        "size_bytes": size,
                        "reason": reason,
                    }
                )
                count += 1
    return count


def exclusion_reason(path: Path) -> str:
    text = str(path)
    if "\\results\\" in text or text.endswith("\\results"):
        return "generated_output_not_duplicated"
    if "\\data\\raw\\" in text:
        return "large_dataset_referenced_not_copied"
    if path.suffix.lower() == ".png" and path.parent == ROOT:
        return "unassigned_image_referenced_not_copied"
    if "protocol_versions" in path.parts or "protocol_reviews" in path.parts:
        return "duplicate_protocol_tree_referenced_not_copied"
    if "execution_extensions" in path.parts:
        return "phase3b_reporting_extension_not_duplicated"
    if path.suffix.lower() == ".pt":
        return "checkpoint_referenced_not_copied"
    return "not_selected_for_experiment_packet"


def write_tables() -> None:
    provenance_fields = [
        "original_absolute_path",
        "new_relative_path",
        "experiment",
        "size_bytes",
        "sha256",
        "sha256_copy",
        "match",
        "git_commit",
        "action",
        "reason",
    ]
    with (DEST / "FILE_PROVENANCE.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=provenance_fields)
        writer.writeheader()
        writer.writerows(records)
    with (DEST / "RELOCATION_MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["original_absolute_path", "new_relative_path", "action", "sha256_original", "sha256_copy", "match"],
        )
        writer.writeheader()
        for row in records:
            if row["action"] == "created_note":
                continue
            writer.writerow(
                {
                    "original_absolute_path": row["original_absolute_path"],
                    "new_relative_path": row["new_relative_path"],
                    "action": row["action"],
                    "sha256_original": row["sha256"],
                    "sha256_copy": row["sha256_copy"],
                    "match": row["match"],
                }
            )


def write_index_and_report(excluded_count: int) -> None:
    copied = [row for row in records if row["action"] == "copied"]
    referenced = [row for row in records if row["action"] == "referenced"]
    mismatches = [row for row in records if row["action"] == "copied" and row["match"] != "True"]
    by_hash: dict[str, list[str]] = {}
    for row in copied:
        by_hash.setdefault(row["sha256"], []).append(row["new_relative_path"])
    duplicate_groups = {digest: paths for digest, paths in by_hash.items() if len(paths) > 1}
    (DEST / "EXPERIMENT_INDEX.md").write_text(experiment_index(), encoding="utf-8", newline="\n")
    report = f"""# Organization report

Date: 2026-10-20. Destination: `E:\\CoNN\\ORGANIZED_EXPERIMENTS`. Original files were read and copied. None were edited, moved, or deleted by this script.

## Counts

| Item | Value |
| --- | --- |
| Experiment directories created | 7 |
| Experiments looked for and not found | Experiment 3A, Experiment 3C, Experiment 3D |
| Files copied | {len(copied)} |
| Original artifacts referenced rather than copied | {len(referenced)} |
| Organization notes created | {sum(1 for row in records if row['action']=='created_note')} |
| Copied-file checksum mismatches | {len(mismatches)} |
| Exact duplicate groups among copied files | {len(duplicate_groups)} |
| Discovered files not copied | {excluded_count} |
| Git commit available | no |

Checksum rule: every copied file has `SHA256(original) == SHA256(copy)` recorded in `FILE_PROVENANCE.csv`. A mismatch would have stopped the script. Referenced checkpoint trees and prepared arrays were not hashed file by file. Their total bytes are recorded on the reference rows. The combined result trees are about 33 GB, which is why they were not duplicated.

## Experiments

Organizational status describes whether the packet contains the historical source, the configuration, and a pointer to the results. It does not mean the organized directory can be launched on its own.

| Directory | Status | Why |
| --- | --- | --- |
| EXPERIMENT_01 | PARTIAL | Smoke and pilot summaries are present. The executing source was not snapshotted separately. The copy comes from the later 1B snapshot with 1B-only filenames removed. That exclusion is an inference. |
| EXPERIMENT_01B | COMPLETE | Run snapshot, protocol, selection file, replication scores, and a reference to the full run are present. |
| EXPERIMENT_01C | COMPLETE | Same standard as 1B. The pre-training audit directory is referenced, not merged into 1B. |
| EXPERIMENT_02 | COMPLETE | Snapshot, HAR config, scores, subject split, and references to the archive and prepared arrays are present. |
| EXPERIMENT_03_PHASE01 | AMBIGUOUS | The protocol and audit report are present. The earliest code snapshot already mixes in phase-2 files, which were left out of this directory by filename. |
| EXPERIMENT_03_PHASE23 | COMPLETE | All six source stages are copied separately. Summary tables are copied. The 1050-fit checkpoint tree is referenced. |
| EXPERIMENT_03B | COMPLETE | Frozen run snapshot, protocols, reports, and summary tables are copied. Grid checkpoints are referenced. |

Runtime reproduction was not attempted. Every organized Python tree still imports packages and data through the original paths. Treat `E:\\CoNN` as the runnable source of truth.

## Shared dependencies

`SHARED_COMPONENTS` contains a hash map, not a merged library. Experiment 2's `conn_exp` package is imported by the associative phase 2/3 code. Phase 3B imports that associative code. Gate filenames recur with more than one hash. The version map lists them.

## Static reproduction blockers

- No Git revision.
- Organized copies are not installed packages. Entry points in the readmes use the original directories.
- Experiment 01's exact pre-1B bytes are not isolated.
- Phase 0/1 and phase 2 code share one early snapshot.
- Phase 2/3 has six source stages. The run report identifies stage snapshots, but this organization does not assign each of the 1050 fits to a single stage directory.
- Checkpoint and prepared-array hashes were not recomputed.
- Virtual environments were excluded from discovery: `conn_experiment_01\\.venv`, `CoNN_Associative_Relationships\\.venv_cuda23`, and `.pip-cache23`.

## Duplicates

Copied files that share a SHA-256 are exact duplicates, usually the same protocol or module frozen into more than one snapshot. Count of such groups: {len(duplicate_groups)}. Paths are grouped in `FILE_PROVENANCE.csv` by the `sha256` column. Historical files with the same name and different hashes were both kept.

## Excluded files

The complete path list is `EXCLUDED_FILES.csv` ({excluded_count} rows). Discovery skipped `AUDIT_20261020`, `ORGANIZED_EXPERIMENTS`, virtual environments, pip caches, pytest caches, and `__pycache__`. Those skipped directories are not rows in the exclusion list.

Main exclusion reasons inside the discovered tree:

- `generated_output_not_duplicated`: run directories, job checkpoints, curves, and plots.
- `large_dataset_referenced_not_copied`: `uci_har_original.zip`.
- `duplicate_protocol_tree_referenced_not_copied`: Phase 3B `protocol_versions` and `protocol_reviews`.
- `phase3b_reporting_extension_not_duplicated`: `execution_extensions` under the Phase 3B run.
- `unassigned_image_referenced_not_copied`: the root infographic.
- `not_selected_for_experiment_packet`: current working copies of modules that are already represented by a frozen snapshot, egg-info, and similar files.

## Recommended future cleanup

Do this only under a separate authorization. Do not do it by editing the organized packets or the historical runs.

1. Put the original tree under Git and record the snapshot hashes already stored in the runs.
2. Keep the current `conn_exp` package from being the only copy of Experiments 01 through 02. The snapshots are the historical versions.
3. Leave checkpoint directories where they are and cite them. Moving them would break the path helpers in `conn_assoc.paths`.
4. Update the associative root README only if a later edit is authorized. This organization did not change it.
"""
    (DEST / "ORGANIZATION_REPORT.md").write_text(report, encoding="utf-8", newline="\n")


def experiment_index() -> str:
    return """# Experiment index

Discovery excluded `AUDIT_20261020`, `ORGANIZED_EXPERIMENTS`, virtual environments, caches, and Git storage. There is no Git repository.

| Directory | What it is | Objective, as stated by its own documents | Organizational status |
| --- | --- | --- | --- |
| `EXPERIMENT_01` | Smoke run `20261017T235744Z_smoke` and five-seed pilot `20261017T235839Z_pilot` | Compare task-aligned local-to-global Gaussian gating with magnitude-matched controls on a synthetic AND task. Primary contrast: final test accuracy of D minus B prime. | PARTIAL |
| `EXPERIMENT_01B` | Run `20261018T001947Z_experiment1b_v1` | Repeat that contrast after a validation-only learning-rate and duration grid, on seeds 100–119. | COMPLETE |
| `EXPERIMENT_01C` | Run `20261018T024146Z_experiment1c_v1` | Compare locality and dense release. Primary contrast: AND partial-width P minus full-release D, seeds 200–219. | COMPLETE |
| `EXPERIMENT_02` | Run `20261019T004346Z_experiment2_har_v1` | Test the scheduled gate on UCI HAR. Primary contrast: engineered-feature D minus B macro-F1, seeds 300–319. | COMPLETE |
| `EXPERIMENT_03_PHASE01` | Associative protocol v2, Phase 0 and Phase 1 | Freeze the associative questions, data audit, and gate definition before training. | AMBIGUOUS |
| `EXPERIMENT_03_PHASE23` | Run `20261019T030442340025Z_phase23_v1` | Development grid for temporal HAR Phase 2 conditions and Phase 3 context conditions. One amendment, one run, six source stages. | COMPLETE |
| `EXPERIMENT_03B` | Run `20261020_phase3b_v3_run01` | Diagnostic study of context-factor symmetry, saturation, and precision. Primary contrast: Gaussian C4 minus Gaussian masked. | COMPLETE |

Not created, because the source repository does not contain them as executed experiments:

- Experiment 3A
- Experiment 3C
- Experiment 3D

Phase 2 and Phase 3 were not split into 3A and an unnamed 3. The source treats them as one authorized grid with distinct condition names. Splitting the checkpoint tree by condition would have been a new derivation, not a copy of an existing boundary.

The root infographic is referenced from `UNASSIGNED_FILES`.
"""


EXPERIMENT_01_README = """# Experiment 01 — Synthetic AND pilot

## 1. Identifier and title

Experiment 01. Smoke run `20261017T235744Z_smoke` and five-seed pilot `20261017T235839Z_pilot`.

Documented name in the build brief: CoNN Experiment 01.

## 2. Research question

Documented in `documentation/CURSOR_BUILD_BRIEF.md`: does structured local-to-global Gaussian weight gating improve learning relative to controls that preserve gate magnitude or per-row energy and disrupt spatial alignment?

## 3. Hypothesis

Documented primary prediction: final held-out test accuracy of condition D minus condition B prime on the AND task. Five seeds are marked exploratory in the brief.

## 4. Mathematical model

Documented and implemented in the copied `source/src/conn_exp` modules: coordinates on `[0, 1]`, distance the absolute difference, `G = exp(-d^2 / (2 tau^2))`, `W_eff = W * G` with `G` detached. `tau` comes from `d50 / sqrt(2 ln 2)`. This description is taken from those modules and the design note. It was not re-derived during organization.

## 5. Dataset and preprocessing

Documented in `source/configs/pilot.yaml` and `source/src/conn_exp/data.py`. Synthetic 100-feature motifs, splits 512/128/128, noise 0.15, distractor amplitude 0.5. Data are generated at runtime. They are not stored as a canonical raw file for the pilot. Generated arrays inside the run directory are referenced, not copied.

## 6. Conditions and controls

Documented identifiers: A dense, A prime scalar energy match, B prime within-row column permutation, C permuted input coordinates, D task-aligned schedule, E reverse path, S static local.

## 7. Training configuration

Documented in the pilot config: two hidden layers of width 32, plain SGD, momentum 0, weight decay 0, learning rate 0.1, 400 steps, release fraction 0.7. The smoke override is in the same YAML under `smoke`.

## 8. Original entry point

From `conn_experiment_01`:

`python -m conn_exp.runner --config configs/pilot.yaml --smoke`

`python -m conn_exp.runner --config configs/pilot.yaml --seeds 5`

Run those commands in the original package, not in this organized directory.

## 9. Random seeds

Documented: smoke seed 0. Pilot seeds 0, 1, 2, 3, 4. Stream offsets are in `DESIGN_DECISIONS.md`.

## 10. Available results

Copied summaries: `results/smoke` and `results/pilot`, including `scores.csv` and `report.md`. Full runs, checkpoints, and plots remain at the paths in `FULL_RUN_REFERENCE.txt`.

## 11. Known limitations

Documented by the original results note: the pilot is exploratory, the shared learning rate was not grid-selected, and D minus B prime does not isolate geometry by itself.

Organization limit: this source copy is the 1B-era snapshot with `experiment1b.py`, `experiment1b_report.py`, and `test_experiment1b.py` removed. Removal was an organizational inference. No snapshot labeled as the pre-1B tree was found.

## 12. Shared components

Uses its own copied `conn_exp` modules. Later experiments import similar filenames. Do not replace these files with a later snapshot. See `../SHARED_COMPONENTS/documentation/VERSION_MAP.md`.

## 13. Original locations

- Code snapshot: `conn_experiment_01/results/runs/20261018T001947Z_experiment1b_v1/source`
- Runs: `conn_experiment_01/results/runs/20261017T235744Z_smoke` and `20261017T235839Z_pilot`
- Brief: `CoNN_Cursor_Experiment_01`

## 14. Reproduction instructions

Use the original `conn_experiment_01` environment and the commands in section 8. This organized copy is an archive. Imports and paths were not rewritten.

## 15. Verification status

Organizational status: PARTIAL. Checksums of copied files are in `../FILE_PROVENANCE.csv`. Runtime reproduction was not executed. The exact bytes that produced the pilot were not isolated from the later snapshot.
"""

EXPERIMENT_01B_README = """# Experiment 1B — Optimization and replication

## 1. Identifier and title

Experiment 1B. Run `20261018T001947Z_experiment1b_v1`.

## 2. Research question

Documented in `documentation/research_summary.md`: whether the pilot's D minus B prime advantage remains after equal validation tuning and new seeds.

## 3. Hypothesis

Documented primary contrast remains final AND test accuracy of D minus B prime. The summary states the expectation of a positive difference and then reports the measurement.

## 4. Mathematical model

The copied snapshot uses the same Gaussian gate, detached product, and schedules as Experiment 01. `source/src/conn_exp/experiment1b.py` is the 1B runner.

## 5. Dataset and preprocessing

Same synthetic AND generator and pilot YAML sizes. New seeds draw new splits. Generated arrays stay in the original run.

## 6. Conditions and controls

The same seven conditions as Experiment 01. Documented frozen learning rates: A, A prime, B prime, D, E, and S at 0.1; C at 0.05. Common duration 3200 steps.

## 7. Training configuration

Documented grid: learning rates 0.01, 0.05, 0.1 and durations 400, 800, 1600, 3200 on tuning seeds, then the frozen choice on replication seeds. SGD with momentum 0 and weight decay 0.

## 8. Original entry point

From the original package, as recorded in `conn_experiment_01/README.md`:

`python -m conn_exp.experiment1b`

Regenerate the summary with:

`python -m conn_exp.experiment1b_report results/runs/20261018T001947Z_experiment1b_v1`

## 9. Random seeds

Documented: tuning seeds 5–9. Replication seeds 100–119. Pilot seeds 0–4 are not reused.

## 10. Available results

Copied: `documentation/research_summary.md`, `results/replication_scores.csv`, tuning and replication reports. The published replication mean is in the summary. The full job tree is referenced.

## 11. Known limitations

Documented in the summary: one synthetic task, 128 test rows, a narrow learning-rate grid, and condition-specific learning rates. S is a secondary comparison because its terminal gate differs.

## 12. Shared components

Self-contained snapshot of `conn_exp` at freeze time. Later 1C and Experiment 2 snapshots are separate copies.

## 13. Original locations

`conn_experiment_01/results/runs/20261018T001947Z_experiment1b_v1`

## 14. Reproduction instructions

Run the entry point in section 8 from the original repository. Do not point it at this organized directory.

## 15. Verification status

Organizational status: COMPLETE for source, protocol, scores, and a resolvable reference to checkpoints. Runtime reproduction was not executed. Copied checksums match. Checkpoint bytes were not rehashed.
"""

EXPERIMENT_01C_README = """# Experiment 1C — Locality versus collapse

## 1. Identifier and title

Experiment 1C. Run `20261018T024146Z_experiment1c_v1`.

## 2. Research question

Documented in `documentation/research_report.md`: whether partial widening differs from full dense release, and how alignment interacts with expansion.

## 3. Hypothesis

Documented sole primary contrast: AND P minus D. Other contrasts in the report are labeled secondary.

## 4. Mathematical model

The snapshot adds terminal policies in `experiment1c_core.py`. The Gaussian definition remains the Experiment 01 gate. Policies named in the report: S fixed d50 0.08, F fixed 0.25, P smooth 0.08 to 0.25 then hold, W smooth 0.08 to 1 then hold, H fixed d50 1, D the original release path, plus row-shuffled controls and dense A.

## 5. Dataset and preprocessing

Documented: original AND generator, and the existing XOR generator. Splits 512/128/128. AND difficulty was not retuned.

## 6. Conditions and controls

See section 4. Alignment degradation at input-coordinate fractions 0, 0.5, and 1 is part of this run and is not a separate experiment.

## 7. Training configuration

Documented: 3200 updates, SGD, shared learning rate selected per task. Frozen rates in the report: AND 0.1, XOR 0.1.

## 8. Original entry point

Documented in the original README:

`python -m conn_exp.experiment1c_audit`

`python -m conn_exp.experiment1c --run-dir results/runs/<timestamp>_experiment1c_v1 --workers 6`

`python -m conn_exp.experiment1c_report results/runs/20261018T024146Z_experiment1c_v1`

## 9. Random seeds

Documented: tuning seeds 10–14. Replication seeds 200–219.

## 10. Available results

Copied: research report, measured findings, replication scores. The phase-1 audit of 1B checkpoints is referenced and was not merged into Experiment 1B.

## 11. Known limitations

Documented: the primary interval includes zero, secondary intervals are unadjusted, and XOR uses the same geometry with a different label rule.

## 12. Shared components

Own snapshot. It includes earlier `conn_exp` modules as they stood for this run.

## 13. Original locations

`conn_experiment_01/results/runs/20261018T024146Z_experiment1c_v1`

## 14. Reproduction instructions

Use the original commands in section 8. This directory is an archive.

## 15. Verification status

Organizational status: COMPLETE by the same standard as 1B. Runtime was not re-executed. Checkpoint hashes were not recomputed.
"""

EXPERIMENT_02_README = """# Experiment 2 — UCI HAR

## 1. Identifier and title

Experiment 2. Run `20261019T004346Z_experiment2_har_v1`.

## 2. Research question

Documented in `documentation/research_report.md`: whether the scheduled alignment result transfers to UCI HAR. The sole primary contrast is engineered-feature D minus B in final test macro-F1.

## 3. Hypothesis

A positive engineered D minus B result would be the transfer prediction stated in the report. The report also states what a negative result does not establish.

## 4. Mathematical model

Same Gaussian and release rule as Experiment 01, applied to two coordinate systems. Engineered features use metadata order `j/560`. Temporal windows colocate nine channels at time `t/127`. Hidden width is 64/64. The report states that engineered coordinates are file order, not a measured physical distance.

## 5. Dataset and preprocessing

Official archive referenced from `datasets/UCI_HAR_ZIP_REFERENCE.txt`. Prepared arrays referenced from `datasets/PREPARED_ARRAYS_REFERENCE.txt`. Subject split is the copied `subject_split.frozen.json`. Fit-only normalization is described in the research report. The archive was not copied.

## 6. Conditions and controls

Documented per representation: A dense, S08, S25, D scheduled, B row-shuffled schedule, C permuted input coordinates.

## 7. Training configuration

Documented: 3200 updates, batch 128, SGD, learning-rate grid 0.01, 0.05, 0.1. Selected rates are in the research report and differ by condition. That fact is documented, not an inference.

## 8. Original entry point

`python -m conn_exp.har_data`

`python -m conn_exp.experiment2 --run-dir results/runs/<timestamp>_experiment2_har_v1 --workers 6`

`python -m conn_exp.experiment2_report results/runs/20261019T004346Z_experiment2_har_v1`

## 9. Random seeds

Documented: tuning seeds 30–34. Final seeds 300–319. Subject split seed 104729.

## 10. Available results

Copied: research report and `results/replication_scores.csv`. Checkpoints and prepared NumPy arrays remain at the referenced paths.

## 11. Known limitations

Documented: one subject partition, optimization seeds rather than new cohorts, and a file-order coordinate system for the primary contrast.

## 12. Shared components

This snapshot is the `conn_exp` tree later imported by associative Phase 2/3. Do not replace it with the current working tree.

## 13. Original locations

Run: `conn_experiment_01/results/runs/20261019T004346Z_experiment2_har_v1`

Archive: `conn_experiment_01/data/raw/uci_har_original.zip`

## 14. Reproduction instructions

Use the original commands. The organized copy does not contain the archive or the prepared arrays.

## 15. Verification status

Organizational status: COMPLETE for documents, scores, and existing references. The archive and prepared arrays were confirmed present by the reference writer and were not byte-hashed here. Training was not rerun.
"""

EXPERIMENT_03_PHASE01_README = """# Associative Phase 0 and Phase 1

## 1. Identifier and title

Not named Experiment 3A in the source. Directory name `EXPERIMENT_03_PHASE01` means the associative protocol's Phase 0 and Phase 1.

## 2. Research question

Documented in `documentation/PROTOCOL_v2.md`: whether Gaussian-gated connectivity improves representation relationships, context sensitivity, and associative retrieval beyond conventional networks, kNN, and memory-augmented controls.

## 3. Hypothesis

The protocol separates classification from retrieval. Phases 2 through 6 are specified and, in that frozen file, described as not yet implemented. Later phases were authorized by a separate amendment and are stored under `EXPERIMENT_03_PHASE23` and `EXPERIMENT_03B`.

## 4. Mathematical model

The protocol records `G_ij(tau) = exp(-d_ij^2 / (2 tau^2))` and `W_eff = W odot G`, with the Experiment 2 detach behavior. Copied modules `gate.py` and `geometry.py` are the Phase 0/1 implementation of that statement, taken from the earliest snapshot.

## 5. Dataset and preprocessing

Read-only use of Experiment 2's prepared temporal arrays and the UCI archive. `documentation/normalization_history.md` records vendor scaling that cannot be undone. No new dataset was generated.

## 6. Conditions and controls

No training conditions were executed in this phase. The protocol names later controls. They were not run here.

## 7. Training configuration

None. The protocol forbids training until later gates are accepted.

## 8. Original entry point

From `CoNN_Associative_Relationships`, with `PYTHONPATH=src` and the Experiment 2 interpreter, as written in the original README:

`python -m conn_assoc.inspect`

`python -m pytest`

## 9. Random seeds

No training seeds. Inspection uses the frozen Experiment 2 subject partition.

## 10. Available results

Copied: protocol, manifest, inspection JSON, and `EXPERIMENT_AUDIT_AND_RESULTS.md`. No checkpoints belong to this phase.

## 11. Known limitations

The earliest code snapshot already contains phase-2 modules. Those filenames were not copied into this directory. That filter is an organizational decision documented in the organization report. A pure pre-phase-2 code snapshot was not found.

The current associative README still says later phases are absent. That sentence is historically the Phase 0/1 status. It is not a description of the repository after Phase 2/3.

## 12. Shared components

`paths.py` points at `../conn_experiment_01` and the Experiment 2 run. Those paths only work from the original associative directory.

## 13. Original locations

Prior snapshot: `CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/prior_snapshot`

Earliest code: `.../source/phase2_pretraining`

## 14. Reproduction instructions

Run `conn_assoc.inspect` from the original associative directory. Do not expect this filtered copy to be a complete package.

## 15. Verification status

Organizational status: AMBIGUOUS, because the code boundary was inferred by filename. Copied checksums match. Inspection was not re-run.
"""

EXPERIMENT_03_PHASE23_README = """# Associative Phase 2 and Phase 3

## 1. Identifier and title

One development study. Run `20261019T030442340025Z_phase23_v1`. Authorization text in the amendment: implement Phases 2 and 3. This directory is not named Experiment 3A in the source.

Phase 2 and Phase 3 share the amendment, the run, and the seeds. Their condition names differ. Both are kept here, with source stages stored separately.

## 2. Research question

Documented in `documentation/PROTOCOL_PHASE23_v1.md` and the results report: on temporal UCI HAR cross-validation, how the original Gaussian schedule compares with dense and shuffled controls, and whether a bounded context factor changes masked-sensor performance.

## 3. Hypothesis

Documented primary development contrast: P2 gaussian minus P2 dense. Required geometry comparison: P2 gaussian minus P2 row-shuffle. Context contrasts are specified separately. The amendment says these intervals are descriptive.

## 4. Mathematical model

Gaussian schedule and detach behavior are the Experiment 2 definitions. Context factor, documented in the amendment:

`C(c) = 1 - (0.95/2) sigmoid(U[c]) sigmoid(V[c]).T`

`W_eff(c) = W * detach(G) * C(c)`

Rank is frozen at 2. U starts at -6 and V at 0.

## 5. Dataset and preprocessing

Experiment 2 temporal windows, official training subjects only, seven folds defined by the v2 protocol. Official test subjects are not an evaluation target of this run. Arrays were not copied.

## 6. Conditions and controls

Phase 2: P2_dense, P2_gaussian, P2_rowshuffle, P2_coordperm, P2_fixed08, P2_fixed25.

Phase 3: P3_dense_masked, P3_gaussian_masked, P3_dense_context, P3_gaussian_context.

## 7. Training configuration

Documented: hidden 64/64, SGD, momentum 0, weight decay 0, batch 128, 3200 updates, learning rates 0.01, 0.05, 0.1, seeds 400–404, seven folds.

## 8. Original entry point

`scripts/run_phase23.py` in the associative project, after the preflight script named in `README_PHASE23.md`. The runnable tree is the original project, not a single stage folder here.

## 9. Random seeds

Documented: 400–404, with the stream offsets in the amendment. Fold index enters the batch and context streams.

## 10. Available results

Copied summary tables in `results/`. The narrative report is `documentation/PHASE_2_3_IMPLEMENTATION_AND_RESULTS.md`. Checkpoints are referenced. Six code stages are under `source/stages`. They are historical stages, not six results.

## 11. Known limitations

Documented: overlapping folds, learning-rate selection on the same subjects, context models have more parameters than masked baselines, and no official-test evaluation.

Organization limit: this packet does not assign each fit to one source stage. The run stored the stages. Which stage executed which fit is described in the results report's implementation narrative and was not recomputed.

## 12. Shared components

Imports Experiment 2's `conn_exp` from the original `conn_experiment_01` tree. A copied stage does not include that dependency.

## 13. Original locations

`CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1`

## 14. Reproduction instructions

Follow `documentation/README_PHASE23.md` in the original associative directory. Do not launch the grid from this archive.

## 15. Verification status

Organizational status: COMPLETE for staged source, protocol, summary tables, and a reference to the existing run. Runtime was not re-executed. The multi-gigabyte job tree was not hashed file by file.
"""

EXPERIMENT_03B_README = """# Phase 3B — Context diagnostics

## 1. Identifier and title

Phase 3B v3. Run `20261020_phase3b_v3_run01`. The root review file uses the name Experiment 3B. There is no executed Experiment 3A or 3C beside it.

## 2. Research question

Documented in `documentation/reports/PHASE3B_FINAL_REPORT.md`: why historical contextual attenuation barely learned, and whether symmetry breaking, less saturation, and master-parameter precision yield useful external-context behavior.

## 3. Hypothesis

Documented primary contrast: Gaussian C4 minus Gaussian masked subject macro-F1, with a practical threshold of +0.005 and a positive bootstrap lower bound. The report states that this threshold was not met.

## 4. Mathematical model

The report keeps the original Gaussian schedule. Context cells C0–C4 change initialization, symmetry noise, and storage dtype of the same factor family used in Phase 3. The frozen formula in the report is `C = 1 - 0.475 sigmoid(U) sigmoid(V)^T`.

## 5. Dataset and preprocessing

Documented: existing Experiment 2 temporal HAR development data. No official test evaluation. No new synthetic task.

## 6. Conditions and controls

Documented grid includes dense and Gaussian C0, C1, C2, and C4, masked baselines, and Gaussian constant, shuffled, and frozen controls. C3 grid fits were omitted by the pilot futility rule. `results/omitted_fits.csv` records that omission.

## 7. Training configuration

Documented: SGD learning rate 0.1, batch 128, 3200 steps, hidden 64/64, seeds 400–404, seven historical folds. No validation learning-rate search.

## 8. Original entry point

`CoNN_Associative_Relationships/phase3b_context_diagnostics/scripts/run.py`

The run snapshot copied here includes `scripts/run.py`. Later reporting scripts live in the original `scripts` directory and in `execution_extensions`, which are referenced rather than fully duplicated.

## 9. Random seeds

Documented base seeds 400–404. Factor seed 30031. Bootstrap seed 7000031, as written in `src/reporting.py` of the diagnostics project and in the v3 protocol. Phase 2's bootstrap seed 230319 is a different study.

## 10. Available results

Copied: final report, executive summary, the other report markdown files, and the small contrast tables. The grid checkpoints are referenced from `checkpoints/GRID_REFERENCE.txt`.

## 11. Known limitations

Documented in the final report: development subjects, overlapping folds, omitted C3, weak second singular component, and no associative-memory claim.

The independent review in `documentation/CoNN_Experiment_3B_Independent_Scientific_Review.md` is a review. It is not an additional result file produced by the trainer.

## 12. Shared components

The frozen snapshot imports associative Phase 2/3 modules from the original project tree. Those modules are not re-copied here as if they were the 3B trainer.

## 13. Original locations

Project: `CoNN_Associative_Relationships/phase3b_context_diagnostics`

Run: `.../results/20261020_phase3b_v3_run01`

## 14. Reproduction instructions

Use `scripts/run.py` from the original diagnostics project, as its README describes. Verification commands there rescore saved predictions. They were not run during organization.

## 15. Verification status

Organizational status: COMPLETE for the frozen snapshot, protocols, reports, and summary CSV files. Runtime and checkpoint hashes were not recomputed. Reporting extensions under the run were referenced, not copied.
"""


if __name__ == "__main__":
    main()
