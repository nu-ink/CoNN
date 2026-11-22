# Phase 3B — Context diagnostics

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
