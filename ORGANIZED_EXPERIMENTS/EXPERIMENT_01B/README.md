# Experiment 1B — Optimization and replication

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
