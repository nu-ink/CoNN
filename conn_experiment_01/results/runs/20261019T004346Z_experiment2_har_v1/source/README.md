# CoNN Experiment 01

Paired PyTorch comparison of a structured local-to-global Gaussian gate against controls that keep gate magnitude or per-row energy and break spatial alignment. The pre-registered contrast is final-step test accuracy of **D minus B′** (`row_shuffle`) on the synthetic AND task.

This repository is a controlled experiment. It is not tuned to make the structured curriculum win. A shared SGD learning rate cannot support a superiority claim. Five seeds are exploratory.

Read `DESIGN_DECISIONS.md` before changing coordinates, the shuffle, the schedule, or the optimizer.

## Experiment 1B: optimization and independent replication

```powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
.venv/Scripts/python.exe -m conn_exp.experiment1b
```

This separate runner preserves the original pilot and executes the predefined
learning-rate grid at 400/800/1600/3200 steps for every condition on tuning seeds
5–9 (420 fits, no test evaluations). It selects LRs per condition and one common
duration using validation only, seals the configuration, then executes all seven
conditions on replication seeds 100–119 (140 fits). Each invocation creates a new
versioned directory and saves a source snapshot, protocol, raw logs, checkpoints,
checkpoint reload ablations, plots and `research_summary.md`. The original pilot
runner and its exact configuration remain unchanged.

Rebuild Experiment 1B figures and summary from a completed run:

```powershell
.venv/Scripts/python.exe -m conn_exp.experiment1b_report results/runs/<timestamp>_experiment1b_v1
```

## Experiment 1C: locality versus collapse

First audit existing 1B measurements and checkpoint gate interventions, without
training or new test evaluations:

```powershell
.venv/Scripts/python.exe -m conn_exp.experiment1c_audit
```

Then use the new audit directory printed by that command:

```powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
.venv/Scripts/python.exe -u -m conn_exp.experiment1c --run-dir results/runs/<timestamp>_experiment1c_v1 --workers 6
```

The runner freezes six explicit terminal policies, their row-energy-matched
shuffles, a dense baseline, XOR generalization and AND input-alignment degradation.
`TerminalPolicy` in `experiment1c_core.py` represents a configurable fixed or smooth
Gaussian terminal width and optional exact dense release. The versioned protocol
records the specific policies used in this experiment.

All 32 task/variant combinations receive three learning rates and five tuning
seeds, with a common validation-selected LR per task to preserve shared pre-release
trajectories. Replication uses 20 new seeds per variant. No prior run is modified.
Regenerate the completed report and figures with:

```powershell
.venv/Scripts/python.exe -m conn_exp.experiment1c_report results/runs/<timestamp>_experiment1c_v1
```

## Experiment 2: UCI HAR real-world validation

Download the original official UCI release into `data/raw/uci_har_original.zip`
from the URL recorded in `conn_exp.har_data.DATASET_URL`, then run the audit:

```powershell
.venv/Scripts/python.exe -m conn_exp.har_data
```

Use the newly printed audit directory for the frozen benchmark:

```powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
.venv/Scripts/python.exe -u -m conn_exp.experiment2 --run-dir results/runs/<timestamp>_experiment2_har_v1 --workers 6
```

The audit retains all 561 indexed engineered features, including repeated names,
and the 9-channel × 128-sample temporal windows. Subject IDs select disjoint
16/5/9-person fit/validation/test partitions. Only fit subjects determine added
normalization. Engineered coordinates follow the metadata column order; temporal
coordinates follow within-window time, colocating simultaneous channels.

`configs/har.yaml` defines dense, two fixed-local widths, scheduled structure,
row-shuffled and input-coordinate-permuted controls. All receive the same SGD
learning-rate grid and 3200-update budget. Selection uses validation macro-F1 on
five tuning seeds, then freezes before 20 final initialization/order seeds per
condition. Report seed intervals as conditional on the fixed subject partition;
separate subject-level summaries are included.

```powershell
.venv/Scripts/python.exe -m conn_exp.experiment2_report results/runs/<timestamp>_experiment2_har_v1
```

## Setup

Windows PowerShell, from `conn_experiment_01`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
python -m conn_exp.runner --config configs/pilot.yaml --smoke
python -m conn_exp.runner --config configs/pilot.yaml --seeds 5
```

Linux or macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
python -m conn_exp.runner --config configs/pilot.yaml --smoke
python -m conn_exp.runner --config configs/pilot.yaml --seeds 5
```

Rebuild tables, plots, and `report.md` from a finished log without training again:

```bash
python -m conn_exp.runner --regenerate --run-dir results/runs/<run_id>
```

## Results directory

Each invocation writes `results/runs/<UTC timestamp>_<smoke|pilot|run>/`:

```text
config.resolved.yaml
config_sha256.txt
environment.json
protocol.json
metrics.csv
gradients.csv
ablation.csv
probes.csv
row_energy.csv
scores.csv
paired_contrast.csv
summary.json
invariant_audit.json
report.md
data/seed_<seed>.npz
masks/seed_<seed>.pt
checkpoints/seed_<seed>/<condition_id>/gate_pre_release.pt
checkpoints/seed_<seed>/<condition_id>/gate_post_release.pt
checkpoints/seed_<seed>/<condition_id>/final.pt
plots/loss_accuracy.png
plots/gate_energy.png
plots/tau_d50_schedule.png
plots/row_energy.png
plots/gradient_distance.png
plots/ablation_delta.png
plots/seed_scatter_D_minus_B_prime.png
```

`gate_pre_release.pt` and `gate_post_release.pt` share the weight state at the 70% boundary and differ in the gate. `final.pt` is the model after the step budget. Test metrics are the final post-step rows in `scores.csv`, not a selected checkpoint.

## Conditions

| ID | Name | Gate |
| --- | --- | --- |
| A | dense | exact ones |
| A_prime | scalar | uniform `sqrt(mean(G_D^2))` per layer |
| B_prime | row_shuffle | D's gate with a frozen within-row column permutation |
| C | permuted_input | input coordinates permuted once; hidden-to-hidden gate matches D |
| D | structured | task-aligned local to global, then exact ones |
| E | reverse | exact ones, then local, then exact ones by 70% and through the end |
| S | static_local | `tau_start` for every step, including the last |

B′ does not use a global shuffle and does not redraw its permutation. Optimizer is SGD with momentum 0 and weight decay 0.
