# Associative Phase 2 and Phase 3

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
