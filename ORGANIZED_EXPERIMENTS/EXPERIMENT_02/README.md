# Experiment 2 — UCI HAR

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
