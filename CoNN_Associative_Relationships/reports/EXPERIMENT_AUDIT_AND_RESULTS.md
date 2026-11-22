# CoNN associative relationships: experiment audit and results

Audit completed 2026-10-19 UTC (2026-10-18 America/Chicago). No new scientific model training or final test evaluation occurred.

## Executive Summary

**Initial classification: Not implemented, for model-training/context/memory experiments.** Phase 0/1 data and Gaussian-gate work was implemented and previously inspected. This is a deliberate scope boundary, not evidence of an interrupted training run. `README.md:3,32` explicitly defers model training; `PROTOCOL_v2.md:3` freezes only Phase 0/1. The initial result folders held only `.gitkeep` files. No associative training entry point, model class, optimizer loop, checkpoint, prediction file, or performance metric was present.

A separate, reproducible implementation defect was found in the direct inspection wrapper: `python scripts/inspect.py` shadowed Python's standard-library `inspect` module and failed during NumPy import. That path is repaired. The documented `python -m conn_assoc.inspect` path does not have that name collision. There is no historical traceback establishing that this defect caused an earlier stop; the failure was reproduced during this investigation.

Fresh verification now passes **21 tests, zero failures and zero skips**, the repaired inspection command, dataset checks, and a CPU gate/linear/autograd smoke test. These validate infrastructure, not the associative research hypothesis. No completed associative classification or retrieval experiment exists to interpret.

Original Experiment 2 is a separate completed experiment with a negative primary result. Its existing artifacts were read, not retrained or overwritten. The frozen associative protocol, manifest, configuration, historical inspection and normalization history remain preserved.

Evidence runs (paths below are relative to this experiment):

- **A0:** `results/runs/20261019T024525616341Z_audit_recovery_v1/` preserves the original files, metadata and reproduced wrapper failure in `inspect_entrypoint.before.log` (exit code 1).
- **A1:** `results/runs/20261019T024619338011Z_audit_recovery_v1/` preserves an intermediate audit attempt: 21 tests passed, repaired wrapper and CPU smoke passed, but the newly added audit plotting code requested Tk and encountered missing `init.tcl`. This was an audit-tool failure, not a historical experiment failure.
- **A2:** `results/runs/20261019T024656620209Z_audit_recovery_v1/` is the completed audit. `commands.json` records exact commands, working directories, exit codes and durations; `completion.json` records the bounded scope. Snapshots and SHA-256 inventories accompany each run. The launcher log is alongside these directories.

## Technical Audit

### Repository and execution evidence

Workspace: `E:\CoNN`. Associative project: `E:\CoNN\CoNN_Associative_Relationships`. Original Experiment 2: `E:\CoNN\conn_experiment_01\results\runs\20261019T004346Z_experiment2_har_v1`.

`git status --short` at the workspace and `git log -6 --oneline` at the associative project both return exit 128, “not a git repository” (A2 `git_workspace.log`, `git_experiment.log`). No tracked/untracked status, commit ID or history can be supplied. The byte-level protocol freeze is SHA-256 `e97d35bab3e684cd4ce09fe8dc8e8bafa22371b6f9c0c4801b1fd84eb0f01c48`, matching `MANIFEST.json`.

The original inspection timestamp is `2026-10-19T02:31:39.233541+00:00`. Original directory/source timestamps and sizes are preserved in A0 `associative.before.json`. The original pytest cache lists 19 test node IDs and an empty `lastfailed` object. Those establish prior collection/inspection activity but do not substitute for an archived test execution transcript. The inspection's `models_trained: 0` is a literal field in source, not an independently maintained execution registry. Its numerical data checks and timestamps are meaningful; the field alone cannot prove that no program ever ran elsewhere.

The source contains only data access, positional geometry, Gaussian arithmetic, protocol/path utilities and inspection. The only pre-existing executable wrapper is `scripts/inspect.py`. Initial results/checkpoints, metrics, plots and relationship_maps contain placeholders only; no partial checkpoint or failure log was present there. A2 `workspace_candidate_paths.txt` records a workspace filename search for associative names and logs, excluding virtual-environment files. No separate associative training output was identified. This search cannot rule out deleted artifacts or output outside the workspace. Detailed process command-line enumeration via `Get-CimInstance Win32_Process` was denied; `Get-Process python*` returned no rows before audit execution. No historical evidence supports an OOM, CUDA error, timeout or failed model evaluation.

### Environment and configuration

The intended environment was reused unchanged: `E:\CoNN\conn_experiment_01\.venv\Scripts\python.exe`. A2 `environment.json`, `requirements.log`, `pip_check.log` show Python 3.11.1, PyTorch 2.14.1+cpu, NumPy 2.4.6, PyYAML 6.0.3, pytest 9.1.1 and matplotlib 3.11.2. `pip check` passes. Imports resolve the local associative package and the original `conn_exp.gates` source. No packages were installed or environments replaced.

`nvidia-smi` succeeds and lists two RTX 3070 Ti cards with 8192 MiB each (A2 `nvidia_smi.log`). PyTorch has no CUDA runtime and reports CUDA unavailable. GPU smoke testing was therefore not run; no GPU installation was necessary for Phase 0/1.

`config/phase01.yaml` agrees with the frozen phase, data paths, temporal representation and tau floor. It is descriptive: current path helpers and gate defaults are defined in Python, rather than driven by a general YAML training runner. Changing this YAML alone would not configure a future experiment. This is a documented implementation limitation, not a demonstrated parsing failure. New output creation and report persistence succeeded inside this project. The new explicit-output CLI rejects existing files and paths outside this project.

### Dataset and normalization

The exact vendor archive is `E:\CoNN\conn_experiment_01\data\raw\uci_har_original.zip`, containing `UCI HAR Dataset.zip`. Its SHA-256 is `c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`. Identity/source are recorded in the local publisher documentation and Experiment 2 audit: UCI Human Activity Recognition Using Smartphones, DOI 10.24432/C54S4K. Archive and every prepared file hash pass verification against the frozen manifest. No substitute data were created.

| Property | Verified finding |
|---|---|
| Official train / test | 7352 / 2947 windows; 21 / 9 disjoint subjects |
| Original Experiment 2 fit / validation | 5796 / 1556 windows; 16 / 5 disjoint subjects |
| Temporal data | Nine channels, 128 ordered samples, flattened time-major to 1152 columns |
| Engineered data | 561 indexed columns; duplicate feature names retained, not merged |
| Labels | Six activities: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING; vendor 1–6, prepared 0–5 |
| Channels | body_acc_x/y/z, body_gyro_x/y/z, total_acc_x/y/z |
| Official test subjects | 2, 4, 9, 10, 12, 13, 18, 20, 24 |
| Experiment 2 validation subjects | 7, 8, 11, 14, 15; fixed label-independent split seed 104729 |
| Acquisition | Vendor reports 50 Hz, 2.56-second windows and 50% overlap |
| Provenance unavailable | No session/timestamp/interval key identifying overlap between different rows |

Subject sets and all fit subjects are enumerated in A2 `inspection.json`. Raw sensor windows remain inside the archive and were successfully loaded for integrity inspection, including test schema inspection under the explicitly permitted `audit` purpose. No test model predictions or test-guided choices were made. Body acceleration is derived from total acceleration and gravity separation; six acceleration channels are not six independent physical sensors.

Temporal coordinates are sample time `t/127`, shared across the nine channels. Engineered coordinates are indexed file order `j/560`, not measured physical distance. Feature names and coordinate conventions are retained in the original Experiment 2 prepared files/audit. No new layout was selected.

The frozen Experiment 2 z-score uses only its 16 fit subjects, per channel for temporal data and per column for engineered data. There are zero constant dimensions. New verification measured maximum reconstruction discrepancies of 9.5367431640625e-7 (temporal) and 1.9073486328125e-6 (engineered); inversion to vendor fit rows differs by at most 3.3049143066676834e-7 and 1.1727061921540383e-7, respectively. Vendor filtering and engineered [-1,1] normalization are not reversible. The supplier's engineered scaling fit population is unknown.

**Important accepted limitation:** `PROTOCOL_v2.md:185` freezes this scaler for seven-fold subject CV across all 21 development subjects. Thus some CV validation people contributed to the fixed scaler. Subject-disjoint optimizer training does not remove this preprocessing leakage. It is explicitly accepted in the frozen design; the present audit does not silently refit scalers. A fold-fitted sensitivity analysis would need a separately declared protocol amendment. Official test subjects contributed to neither the scaler nor development folds. Row adjacency is never used to invent overlap exclusions.

### Gaussian mathematics

Implemented arithmetic is `G = exp(-d²/(2 tau²))`, `W_eff = W * G`, with gate shape `[destination, source]` and absolute coordinate distances. Tau and its floor must be finite and positive; positive tau below 1e-6 is floored. Gates lie in [0,1], equal 1 at zero distance and grow monotonically with tau. Numerical underflow to zero is retained. Increasing width is connectivity expansion; the future schedule's exact dense release is a separate gate policy, not merely finite-width Gaussian arithmetic.

`tests/test_gate.py` imports original `conn_exp.gates.gaussian_gate` and verifies bitwise equality on the shared positive-tau domain, including floor flags. A2 `gate_measurements.csv` additionally checks full 64×1152 temporal gates: maximum difference from original is zero for every tested width. `effective_weight` permits gate differentiation mathematically; `training_effective_weight` detaches the gate to match original `GatedLinear`. Weight gradients propagate. A CPU analytical gradient comparison has zero maximum error (A2 `cpu_smoke.json`). This does not validate future learnable context gradients.

The original implementation floors nonpositive tau; the associative implementation rejects it. This is a deliberate, documented frozen-domain restriction, not an undisclosed mathematical change. No gate, schedule, optimization budget, split or model definition was altered by this audit.

### Architecture and root-cause analysis

| Planned condition/component | Actual associative implementation | Scientific checks |
|---|---|---|
| Dense baseline; original Gaussian CoNN | No model/trainer; original Experiment 2 classes exist read-only elsewhere | No new classification run |
| Dense + Context; CoNN + Context | Not implemented | External context, dimensions, initialization and gradients not tested |
| Dense + Memory; CoNN + Memory | Not implemented | No bank, self-exclusion, temperature or refresh behavior tested |
| Dense + Context + Memory; CoNN + Context + Memory | Not implemented | Combined behavior not tested |
| Raw-input and dense-embedding kNN | Specified, not implemented here | No retrieval baseline measured |
| B/C/S08/S25 controls | Specified for later phases; no associative trainer | No new matched-control measurement |

The proposed external context is supplied sensor availability (both/accelerometer/gyroscope), not labels, subject identity or an inferred code. The protocol requires a positive low-rank factor initialized at one, bounded above by one. Proposed memory uses encoder keys, non-label values, a zero-initialized residual into h2, separate retrieval temperature and Gaussian width, detached cached keys, declared refreshes, exact-row self-exclusion and separated support/query pools. None of those requirements has an implementation to inspect for correctness. Missing rank/capacity/refresh/noise details must be concretized and frozen before their later experiments; this audit does not supply post-hoc defaults.

| Symptom/category | Evidence and root cause | Repair/disposition; scientific risk |
|---|---|---|
| No model results / missing components | README and protocol explicitly stop at Phase 0/1; no model/trainer source | No model “repair” invented. Later implementation requires acceptance and a scope extension; no historical model result changed |
| Direct wrapper import failure / implementation bug | A0 traceback: local scripts/inspect.py imported as stdlib inspect during NumPy initialization | Remove scripts directory from import search before dependency import; regression tests and full CLI inspection pass; no numerical changes |
| Default inspection replaces historical report / logging risk | Original main writes fixed audit/inspection.json | Add optional --output, reject existing explicit output and paths outside project; use new run destination; original default retained for compatibility |
| GPU test unavailable / environment limitation | Hardware visible but CPU-only installed torch | Use CPU, retain environment; no dependency replacement |
| CV preprocessing overlap / experimental-design limitation | Frozen protocol explicitly retains original 16-subject scaler | Disclose, propose sensitivity amendment; changing it now would change the frozen representation |
| Intermediate audit plot failed / new audit-tool defect | A1 reached plotting after successful tests/inspection; Tk init.tcl absent | Explicit Agg backend in new audit utility; A2 PNGs generated; no historical training failure inferred |
| No Git revision history / provenance limitation | Both Git queries fail | Preserve source snapshots, hashes and patch evidence; do not invent a commit |

Changed pre-existing source files: `scripts/inspect.py` (import path), `src/conn_assoc/inspect.py` (optional protected report destination). Added `tests/test_entrypoint.py` and `scripts/audit_recovery.py`. Scientific gate/data/protocol source and frozen configuration remain unchanged. `repair_changes.patch` and `investigation_changes.json` in A2 enumerate differences from the A0 snapshot. No model/checkpoint or scientific-result repair was necessary.

## Verification Results

| Check | Status | Exact evidence |
|---|---|---|
| Syntax and imports | PASS | A2 syntax_imports.json, environment.json; all source files parsed |
| Required dependency consistency | PASS | A2 pip_check.log; exit 0 |
| Full current unit suite | PASS: 21 passed in 7.21 s, no skips | A2 pytest.log and pytest.xml; argv in commands.json |
| Original gate equivalence, geometry, floor/range/monotonicity, detach behavior | PASS | tests/test_gate.py, A2 pytest.log and gate_measurements.csv |
| Archive/prepared hashes, raw windows, inverse normalization | PASS | tests/test_data.py and A2 inspection.json |
| Subject separation, seven-fold coverage, official-test embargo | PASS within documented scaler limitation | tests/test_data.py; final_evaluation deliberately rejected |
| Direct wrapper failure reproduction | EXPECTED FAILURE, before repair | A0 inspect_entrypoint.before.log; exit 1 |
| Direct wrapper after repair | PASS | A2 inspect_entrypoint_after.log, inspection_cli.json; exit 0 |
| Overwrite/outside-project output rejection | PASS | tests/test_entrypoint.py; existing inspection bytes preserved |
| CPU linear/gate/autograd primitive on real fit data | PASS | A2 cpu_smoke.json; input 8×1152, output 8×64 |
| Model optimizer update, classification loss, model checkpoint round trip | NOT RUN: no authorized model implementation | cpu_smoke.json explicitly reports no updates/checkpoints; its quadratic objective is not classification loss |
| Memory self-retrieval exclusion, support/query separation and key refresh | NOT IMPLEMENTED / NOT RUN | Source inventory; not counted as test passes |
| Contextual-gating behavior and gradients | NOT IMPLEMENTED / NOT RUN | Source inventory; not counted as test passes |
| GPU smoke | NOT RUN | CPU-only torch; hardware inventory retained |
| One-seed / multi-seed associative baseline training | NOT RUN | README review boundary; protocol Phase 0/1; no training command exists |
| Original Experiment 2 preservation | PASS | A2 experiment2.before.json, experiment2.after.json and preservation.json |

The CPU primitive's finite quadratic objective is 0.0005134638049639761 and gradient norm is 0.013985566794872284. No optimizer was instantiated, weights were not updated, and no model checkpoint was written. Persisted JSON/CSV/PNG output demonstrates audit I/O, not training checkpoint functionality. Audit wall time is in A2 completion.json and command-level durations in commands.json. New model runtime, model memory usage and inference cost are unavailable because no such model ran.

## Experimental Results

### New associative experiment

All classification accuracy, macro-F1, per-class recall, confusion matrices, train/validation cross-entropy, per-seed performance, model parameter counts and model memory/runtime measurements are **not measured**. All retrieval Recall@1/Recall@5/MRR, noise/masking/context effects, kNN comparisons, random-key/shuffled-value controls and subject uncertainty estimates are **not measured**. There is no valid associative checkpoint comparison or learning curve. A missing result is not zero performance and not a negative finding.

![Gate arithmetic diagnostic](../results/runs/20261019T024656620209Z_audit_recovery_v1/gaussian_gate_diagnostic.png)

This chart evaluates the existing gate formula; it is not a model learning curve. CSV measurements include magnitude and per-row squared energy.

### Historical Experiment 2, separately labelled

Original `research_report.md`, `final_verification.json`, `replication/primary_summary.json`, `paired_statistical_summary.csv` and `subject_level_summary.csv` establish a completed HAR experiment: 180 tuning fits, 240 final fits, 20 replication seeds and 1260 checkpoint files verified by its historical verifier. These counts describe its saved verification, not a fresh replay of that full verifier. The present audit verifies file preservation and prepared-data integrity; it does not reload every checkpoint or rescore every prediction.

| Historical comparison | n | Mean macro-F1 difference | SD | 95% CI | Paired dz |
|---|---|---|---|---|---|
| Engineered D−B, original primary | 20 seeds | -0.0050103812 | 0.0078942819 | [-0.0087050188, -0.0013157435] | -0.634685 |
| Temporal D−B, original secondary | 20 seeds | +0.00186144 | 0.007821 (rounded) | [-0.00179889, +0.00552177] | +0.2380 |
| Engineered D−B, historical subject analysis | 9 subject means over 20 models | -0.0051518842 | 0.0062149593 | [-0.0099291248, -0.0003746436] | -0.828949 |

The seed intervals are Student-t intervals conditional on the fixed subject partition, not independent subject-population intervals. The historical subject row is also the saved Student-t analysis, not the subject bootstrap planned for future associative evaluation. No new bootstrap was substituted. Engineered primary differences were positive for 3 seeds and negative for 17. Historical engineered accuracy means: dense A 0.929623, D 0.925501, shuffled B 0.930353. These are not associative or contextual results.

![Historical comparisons only](../results/runs/20261019T024656620209Z_audit_recovery_v1/historical_experiment2_comparisons.png)

A2 `historical_experiment2/` contains copies of the original primary/paired/subject summaries. Original per-seed tables, subject scores, per-class metrics, confusion files, training curves, runtime records and checkpoints remain at the original run. This report does not relabel their old test evidence as new independent evidence.

## Scientific Interpretation

The data and positive-tau Gaussian implementation behave as specified in the executed checks. That supports the integrity of Phase 0/1 infrastructure. It provides no estimate of an associative-model benefit. External contextual relationships and associative retrieval have not been tested; no correct memory behavior can be inferred from the absence of a memory implementation.

Historical Experiment 2 did not demonstrate transfer of the primary structured advantage to engineered HAR features: its matched shuffled control performed better. Its temporal D−B interval includes zero. Those valid negative/unresolved findings should be retained, not “fixed” by changing layouts, schedules or test selection. Possible representation/schedule/width/optimization interactions remain hypotheses; none explains a missing associative training run without implementation and measurements. Neither this audit nor the historical findings prove geometry or establish a new architecture.

## Limitations

- Phase 0/1 only; missing models and controls prevent classification or retrieval conclusions.
- Historical inspection/cache evidence is incomplete as an execution history; no global or deleted-log audit is possible. Detailed process enumeration was denied.
- Official test subjects already appear in the published Experiment 2 research record; they are held out from development rows but not unseen to the researchers. No new test-driven redesign was performed.
- Fixed-scaler CV includes validation-subject preprocessing contribution by design. Vendor engineered scaling provenance is unknown.
- No continuous-session metadata supports excluding different overlapping windows; exact-row exclusions in a future bank remain mandatory and untested.
- Five engineering seeds are proposed, not a confirmatory replication; no engineering seed ran here. Future context/memory implementation details and compute costs remain to be frozen.
- CPU-only torch prevents GPU verification despite available hardware. No new training/inference memory consumption was measured.
- No Git history; SHA-256 inventories and source snapshots provide reproducibility evidence instead.

## Next Steps

1. Review and explicitly accept Phase 0/1, then authorize the next implementation scope. Preserve the current protocol bytes. A later amendment must lift the phase boundary; final_evaluation requires its own explicit authorization (`PROTOCOL_v2.md:214`). There is no existing training command to resume, so an invented command is not supplied.
2. Implement only Dense and original Gaussian CoNN first, on temporal HAR, with the frozen seven subject folds, independent streams, seeds 400–404, equal LR grid [.01,.05,.1], 64/64 hidden units, plain SGD, batch 128, 3200 steps and existing schedule. Select using development validation only; do not reuse Experiment 2's selected LRs. Report the fixed-scaler limitation. Declare any fold-fitted scaler sensitivity analysis separately before execution.
3. Verify finite classification loss, actual optimizer updates, deterministic pairing, checkpoint round trips, persisted fold metrics and bounded run budgets. Then add B/C/fixed-width controls. Freeze concrete context ranks and memory capacity/refresh/noise/support policies before subsequent implementation. Test context interventions, exact-row exclusion, separate retrieval temperature, storage accounting and negative controls before combined training.
4. Before future final scoring, freeze selection and the final-evaluation amendment. Use the declared subject-level uncertainty analysis and report all negative results. Do not infer success from execution alone.

Reproduce this bounded audit (creates a fresh unique evidence directory):

```powershell
Set-Location E:\CoNN\CoNN_Associative_Relationships
$env:PYTHONDONTWRITEBYTECODE = '1'
..\conn_experiment_01\.venv\Scripts\python.exe scripts/audit_recovery.py
```

Run the repaired inspection alone without replacing historical evidence (choose a new, unused filename):

```powershell
..\conn_experiment_01\.venv\Scripts\python.exe scripts/inspect.py --output results/inspection_review_02.json
```

Exact verification commands and resolved output locations used for this report are recorded in A2 `commands.json`. No scientific protocol amendment has been applied.
