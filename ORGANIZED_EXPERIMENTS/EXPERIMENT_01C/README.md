# Experiment 1C — Locality versus collapse

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
