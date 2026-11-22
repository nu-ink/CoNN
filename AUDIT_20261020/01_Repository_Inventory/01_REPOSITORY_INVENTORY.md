# 01 — Repository inventory

Audit date: 2026-10-20. Root: `E:\CoNN`. This inventory excludes `AUDIT_20261020`, virtual environments, pip caches, and `__pycache__` from the research-file counts. Those excluded trees were seen and were not treated as scientific sources.

## Scope actually inspected

| Location | Role | Research files counted |
| --- | --- | --- |
| `conn_experiment_01` | Experiments 1 (pilot/smoke), 1B, 1C, and 2 | 23,474 |
| `CoNN_Associative_Relationships` | Protocol v2, Phases 2–3, Phase 3B | 21,339 |
| `CoNN_Cursor_Experiment_01` | Original build brief | 2 |
| Repository root | Infographic and one prior 3B review | 2 |

Extension counts in that scope include 14,202 `.pt` checkpoints, 12,645 JSON files, 11,960 CSV files, 2,256 `.npz` files, 397 Python files, 125 Markdown files, and 1 `.zip` dataset archive. No `.ipynb` notebooks are present. No path in this scope was unreadable.

The associative CUDA environment (`.venv_cuda23`, about 29,378 files) and `.pip-cache23` are installation artifacts. They were not line-audited.

## What is absent

- There is no `.git` directory. Commit history, authorship, and branch structure are **UNVERIFIABLE**. Content hashes and source snapshots inside run directories are the available revision record.
- The bandwidth formula `tau(q) = sigma / r(q)` does not occur in the Python, YAML, Markdown, or JSON sources searched. It is not an implemented CoNN equation.
- Phase 4 associative memory, a memory bank, and retrieval evaluation are specified in `PROTOCOL_v2.md` and are not implemented.
- Official UCI HAR test subjects are measured in Experiment 2. Later associative phases state that they do not score those subjects again.

## Primary source map

| Artifact | Path |
| --- | --- |
| Build brief | `CoNN_Cursor_Experiment_01/CURSOR_BUILD_BRIEF.md` |
| Pilot design record | `conn_experiment_01/DESIGN_DECISIONS.md` |
| Pilot results narrative | `conn_experiment_01/RESULTS.md` |
| Pilot scores | `conn_experiment_01/results/runs/20261017T235839Z_pilot/scores.csv` |
| Experiment 1B scores and summary | `conn_experiment_01/results/runs/20261018T001947Z_experiment1b_v1/` |
| Experiment 1C scores and report | `conn_experiment_01/results/runs/20261018T024146Z_experiment1c_v1/` |
| Experiment 2 scores and report | `conn_experiment_01/results/runs/20261019T004346Z_experiment2_har_v1/` |
| UCI archive | `conn_experiment_01/data/raw/uci_har_original.zip` (61,005,872 bytes) |
| Gate, geometry, model | `conn_experiment_01/src/conn_exp/gates.py`, `geometry.py`, `models.py`, `schedules.py` |
| Frozen associative protocol | `CoNN_Associative_Relationships/PROTOCOL_v2.md` |
| Phase 2/3 amendment and report | `PROTOCOL_PHASE23_v1.md`, `reports/PHASE_2_3_IMPLEMENTATION_AND_RESULTS.md` |
| Phase 2/3 run | `CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/` |
| Phase 3B protocol and report | `phase3b_context_diagnostics/PROTOCOL_PHASE3B_v3.md`, `reports/PHASE3B_FINAL_REPORT.md` |
| Phase 3B tables | `phase3b_context_diagnostics/results/20261020_phase3b_v3_run01/analysis/` |
| Promotional diagram | `Collapsible Neural Network Infographic.png` |

Three copies of `PHASE3B_FINAL_REPORT.md` share SHA-256 `d197cde3114cd2742e9fd20215e8eddc418bdb284958791d8908e228d5bfa39b`: the reports directory, the run `reports` directory, and `research_report.md`.

## Binary and checkpoint policy

Checkpoints were not deserialized. PyTorch `.pt` files in this repository are ordinary `torch.save` archives and would require pickle loading. Their scientific content was taken from the CSV and JSON tables the runners wrote beside them. The largest single research file is `analysis/diagnostics.csv` in the Phase 2/3 run (322,731,046 bytes).

## Duplicate and superseded material

Protocol and report text is copied under `protocol_versions/`, `protocol_reviews/`, `source/` snapshots, and `execution_extensions/`. The associative `README.md` still describes Phase 0/1 only. Later training lives in sibling files and was audited from those files. Historical run directories for smoke, pilot, 1B, 1C, and Experiment 2 are all present and were not overwritten by later runners, according to the later reports' preservation statements. This audit did not recompute those preservation hashes.

## Independent calculation artifact

`Supporting_Evidence/independent_recalculation.py` reads saved CSV tables only. Its output is `Supporting_Evidence/recalculated_statistics.json`. It does not import the research packages or load weights.

## Finding

**INV-01.** The repository is a multi-experiment research workspace without Git. Provenance is file-system snapshots and SHA-256 manifests. Confidence: **CONFIRMED**. Impact: exact code-to-result linkage depends on those snapshots remaining intact. Action: keep the audit read-only and cite run directories rather than a commit id.
