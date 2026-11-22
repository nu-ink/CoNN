# 06 — Statistical verification and reproducibility

Independent program: `Supporting_Evidence/independent_recalculation.py`. It uses the Experiment 1 virtual environment's NumPy only to read CSV files. Student-t quantiles were computed with a continued-fraction regularized incomplete beta. Checks against reference values: df=4 critical value 2.776445105 (matches the pilot's 2.776445 to the printed digits); df=19 critical value 2.093024054.

## Experiment 1B primary number

Published statement: mean D minus B′ test accuracy = 0.04843750 on 20 seeds.

Recalculation from `replication/scores.csv`, pairing on `seed`, metric `test_accuracy`:

| Quantity | Recalculated | Published |
| --- | --- | --- |
| n | 20 | 20 |
| mean | 0.0484375 | 0.04843750 |
| sample SD | 0.0317193374 | 0.03171934 |
| 95% t interval | [0.033592393, 0.063282607] | [0.03359239, 0.06328261] |
| paired dz | 1.52706532 | 1.527065 |
| positive / negative | 19 / 1 | 19 / 0 / 1 |

**CONFIRMED.** The figure is the arithmetic mean of twenty paired accuracy differences. It is not a pooled accuracy on a single shared test set. Each seed has its own 128-row test split. The interval describes seed-to-seed variation of that difference under the frozen 1B procedure.

What the number establishes: in this procedure, task-aligned gates scored higher than a within-row shuffle by about 4.84 accuracy points on average. What it does not establish: a collapse mechanism, a benefit over staying local, or transfer beyond AND. In the same file, S minus D is positive and its interval excludes zero in S's favor.

Pairing is appropriate. Data and initialization fingerprints do not vary across conditions inside a seed (0 disagreements in 20 seeds). `test_evaluations` is 1 on every 1B, 1C, and Experiment 2 replication row that was checked. Tuning rows for Experiment 2 have `test_evaluations` 0.

## Other confirmed reproductions

| Contrast | Source | Recalculated mean | Matches published value |
| --- | --- | --- | --- |
| Pilot D−B′ | pilot `scores.csv` | 0.021875 | yes |
| 1C AND P−D | 1C replication scores | 0.018359375 | yes, interval includes 0 |
| 1C AND D−shuffle | same | 0.025390625 | yes |
| 1C XOR D−shuffle | same | −0.025390625 | yes |
| Exp2 engineered D−B macro-F1 | replication scores | −0.005010381 | yes |
| Exp2 temporal D−B macro-F1 | same | +0.001861441 | yes, interval includes 0 |
| Phase 2 Gaussian−row-shuffle | `individual_seed_differences.csv` | +0.011375365 | yes |
| Phase 3B C4−masked | `subject_differences.csv` | +0.002133983 | yes |

## Phase 3B interval

The code in `phase3b_context_diagnostics/src/reporting.py` lines 8–12 draws 10,000 means with `numpy.random.default_rng(7000031)` and `rng.integers`, then takes `np.quantile` at 0.025 and 0.975. Repeating that procedure on the saved 21 subject differences reproduces the published endpoints `0.00011565544771546819` and `0.004446269568044705` exactly.

The Phase 2 analysis uses a different seed, `230319` (`phase23_analysis.py` line 118). Applying 230319 to the Phase 3B subject vector does **not** reproduce the published 3B interval. The 3B protocol text that names seed `7000031` matches the code. An earlier amendment's seed must not be substituted when checking 3B.

## Dependence and multiplicity

- Five pilot seeds and twenty replication seeds are different samples. The reports do not pool them. That choice is sound.
- 1C declares one primary contrast. The alignment and XOR intervals are secondary and unadjusted. Using them as a family of confirmatory tests would overstate the evidence. The reports label them as unadjusted. The scientific problem is when a narrative promotes a secondary winner (fixed d50=0.25) to a mechanism conclusion. The 1C text mostly avoids that promotion. The ranking remains descriptive.
- Experiment 2's primary comparison is one contrast. Condition-specific learning rates mean the engineered comparison is not a pure gate comparison. The matched-rate validation check, reported in this audit's JSON, keeps the negative sign at all three rates.
- Phase 2 and Phase 3B resample subjects whose training sets overlap, after learning rates or analysis choices were informed by the same development pool. A bootstrap that excludes zero is a description of that pool. Phase 3B's own practical threshold is the operative success rule, and it fails.
- The Phase 3B seed-level t interval and the subject-level t interval both include zero. See `06` JSON keys `seed_mean_t_interval` and `subject_mean_t_interval`. The bootstrap result is method-sensitive.
- Phase 2 seed differences are extremely stable (Gaussian minus shuffle dz about 5.4 on five seeds). Overlapping folds make seeds share most of their data. That t interval is a poor model of independent replication. The authors' subject-bootstrap caveat is the relevant one.
- No multiple-comparison correction is required for a single primary test. It is required before counting every secondary exclusion of zero as a separate discovery. This audit does not count them that way.

## Reproducibility limits

| Item | State |
| --- | --- |
| Git revision | Absent |
| Source snapshots and SHA-256 manifests inside runs | Present; not re-hashed in this audit |
| Score arithmetic | Reproduced |
| Checkpoint bytes | Not reloaded |
| Unit tests | Not re-run |
| Phase 3B autograd fixture at 1.39e-17 | Not re-run |
| Randomness | Separate data, init, order, and mask streams are coded and documented |

**STAT-01.** The 0.0484375 figure is a real paired mean in the 1B score file. It is evidence of an AND alignment gap under a matched learning rate. It is not evidence that release or collapse caused the gap. Class: **CONFIRMED** for the number; the mechanistic reading is **CONTRADICTED** by S beating D in the same file.

**STAT-02.** Phase 3B's statement that the bootstrap interval excludes zero is true for the frozen procedure and false for a t interval on the same subject means and for a t interval on the five seeds. Class: **CONFIRMED**.
