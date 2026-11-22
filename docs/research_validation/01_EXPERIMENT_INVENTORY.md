# Experiment Inventory

The evidence below is based on read-only inspection of the live source trees, organized copies, run reports and CSV/JSON result artifacts. It does not treat organized copies as independent reruns.

| Study | Status | Implementation / entry point | Primary question | Evidence location |
|---|---|---|---|---|
| Experiment 01 smoke | Completed, pipeline-only | `python -m conn_exp.runner --smoke` | Verify implementation contract | [pilot smoke](../../conn_experiment_01/results/runs/20261017T235744Z_smoke/report.md) |
| Experiment 01 pilot | Completed, exploratory | `conn_exp.runner`, seeds 0–4 | D − B′ on synthetic AND | [pilot report](../../conn_experiment_01/results/runs/20261017T235839Z_pilot/report.md) |
| Experiment 1B | Completed replication | `conn_exp.experiment1b`, run `20261018T001947Z_experiment1b_v1` | Does tuned AND advantage replicate? | [1B report](../../conn_experiment_01/results/runs/20261018T001947Z_experiment1b_v1/research_summary.md) |
| Experiment 1C | Completed replication | `conn_exp.experiment1c` | Partial widening / release / alignment | [1C report](../../conn_experiment_01/results/runs/20261018T024146Z_experiment1c_v1/research_report.md) |
| Experiment 02 | Completed replication | `conn_exp.experiment2` | UCI HAR transfer | [HAR report](../../ORGANIZED_EXPERIMENTS/EXPERIMENT_02/documentation/research_report.md) |
| Associative Phase 0/1 | Protocol/inspection, no training | `conn_assoc.inspect` | Protocol feasibility and frozen access | [Phase 01 README](../../ORGANIZED_EXPERIMENTS/EXPERIMENT_03_PHASE01/README.md) |
| Associative Phase 2/3 | Completed development runs | `scripts/run_phase23.py` | Gaussian schedule and context factor | [Phase23 report](../../ORGANIZED_EXPERIMENTS/EXPERIMENT_03_PHASE23/README.md) |
| Phase 3B v3 | Completed development grid | `phase3b_context_diagnostics/scripts/run.py` | Context factor precision/symmetry/predictive benefit | [3B report](../../CoNN_Associative_Relationships/phase3b_context_diagnostics/results/20261020_phase3b_v3_run01/reports/PHASE3B_FINAL_REPORT.md) |
| Experiment 04 | Proposed only | None | Meaningful versus scrambled relationship geometry | Not implemented |

## Actual source locations

- Synthetic and HAR implementation: [conn_experiment_01/src/conn_exp](../../conn_experiment_01/src/conn_exp).
- Context diagnostics: [CoNN_Associative_Relationships](../../CoNN_Associative_Relationships).
- Historical copies: [ORGANIZED_EXPERIMENTS](../../ORGANIZED_EXPERIMENTS).
- Prior architecture-review material: [architecture_review](../architecture_review).

## Separation of evidence

The 1B pilot, tuning set and 20-seed replication are distinct samples. Pilot and tuning scores must not be pooled with replication. The 1C and HAR studies are additional, different task/protocol samples. Phase 2/3 and 3B use HAR development subjects, overlapping folds and no official test conclusion.

