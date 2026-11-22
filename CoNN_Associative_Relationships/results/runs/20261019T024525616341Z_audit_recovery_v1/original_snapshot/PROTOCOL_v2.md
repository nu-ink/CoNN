# Protocol v2 — CoNN associative relationships

Status: frozen for Phase 0 and Phase 1. Phases 2–6 are specified here and are not implemented. No model has been trained under this protocol. The official test split has not been used for model selection or final evaluation.

The protocol hash is the SHA-256 of this file, recorded in `MANIFEST.json`. This directory is not a Git repository. If the work is later placed under Git, this file must be committed by itself before any final test evaluation.

## Research question

Can Gaussian-gated connectivity improve learned representation relationships, context sensitivity, and associative retrieval beyond what conventional neural networks, kNN, and memory-augmented controls achieve?

The question is answered as two separate tracks. A classification gain is not evidence of associative retrieval. Agreement with activity labels is not evidence of an unlabeled relationship.

## Confirmed data sources

Experiment 2 is the run `conn_experiment_01/results/runs/20261019T004346Z_experiment2_har_v1`. Its code, prepared arrays, checkpoints, and reports are read-only.

| Item | Location |
|---|---|
| Vendor archive | `conn_experiment_01/data/raw/uci_har_original.zip` |
| Nested dataset | `UCI HAR Dataset.zip` inside that archive |
| Prepared arrays | `<run>/prepared/{engineered,temporal}/` |
| Scalers | `scaler.npz` in each prepared directory |
| Coordinates | `feature_coordinates.csv` in each prepared directory |
| Subject partition seal | `<run>/subject_split.frozen.json` |
| Experiment 2 training protocol | `<run>/protocol.frozen.json` |
| Dataset audit | `<run>/audit/dataset_report.md` and `dataset_audit.json` |

The archive SHA-256 is `c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`. The publisher page is the UCI HAR dataset, DOI 10.24432/C54S4K. The current UCI page lists CC BY 4.0. The bundled 2013 README contains older noncommercial wording. Both notices stay with the source files. This experiment does not redistribute a modified dataset.

### Windows

Raw 128-step windows are in the archive and are not stored as loose arrays. For each official split `train` and `test`, nine files under `Inertial Signals/` have one row per window and 128 samples per row:

`body_acc_{x,y,z}`, `body_gyro_{x,y,z}`, `total_acc_{x,y,z}`

Inspection measured official training windows `(7352, 128, 9)` and official test windows `(2947, 128, 9)`. Sampling is 50 Hz. The vendor README states a 2.56 s window and a 50% overlap. Experiment 2 flattened each window time-major into 1152 columns, with all nine channels at sample `t` sharing coordinate `t/127`.

### Labels and subjects

Activity labels in the vendor files are 1–6: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING. Experiment 2 stores them as 0–5. Subject IDs are metadata and are not model inputs.

Official training subjects (21), from `subject_train.txt`:

`1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30`

Official test subjects (9), from `subject_test.txt`:

`2, 4, 9, 10, 12, 13, 18, 20, 24`

These two sets are disjoint. Window counts are 7352 and 2947. Every class occurs in both official splits.

Experiment 2 then split the official training subjects with NumPy `default_rng(104729)`, taking the first five IDs of a permutation as validation and leaving sixteen to fit the scaler and the training rows. That assignment did not use labels or feature values.

- Fit subjects: `1, 3, 5, 6, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30` (5796 windows)
- Experiment 2 validation subjects: `7, 8, 11, 14, 15` (1556 windows)
- Test subjects: the official nine (2947 windows)

Prepared `train` means the 16 fit subjects. Prepared `val` means the five Experiment 2 validation subjects. Prepared `test` means the official test subjects. Relative order inside each of those arrays matches the vendor file order. This protocol rebuilds the 7352 official training rows by interleaving those two arrays with `subject_train.txt`. That reconstruction restores file order. It does not establish recording time.

### Provenance limits

Same-row alignment across the nine inertial files is part of the released schema: one row is one window. Inspection confirmed equal row counts. That alignment is the only instance key used for paired sensor views.

Row order is not recording continuity. The release has no session id, timestamp, or overlap index. The stated 50% overlap cannot be turned into a list of neighboring rows. Phase 4 may exclude a query's own row. It may not drop "adjacent" rows under an assumption that they overlap.

The 561 engineered columns contain mixed accelerometer, gyroscope, jerk, magnitude, frequency, and angle summaries. Eighty-four feature names are duplicated; Experiment 2 kept every column by integer id. Their coordinate is `j/560` in file order. That coordinate is a semantic prior, not a measured physical distance. Engineered columns are not a clean sensor-availability context, because one column can mix sensors.

`body_acc` is total acceleration after the vendor gravity filter. It is not a third physical sensor.

Experiment 2 already published test metrics for these nine people. They remain the held-out people relative to the training rows. They are not unseen to the research record. This protocol still bars them from development, tuning, and memory writes.

## Normalization that cannot be undone

Details are in `audit/normalization_history.md`.

1. Vendor filtering, windowing, gravity separation, and the [-1, 1] engineered scaling stay in the numbers. The publisher does not identify the subjects used for that scaling.
2. Experiment 2 z-scored engineered columns, and temporal channels, using only the 16 fit subjects. No prepared dimension was constant. Validation and test used those frozen statistics.
3. Inverting the Experiment 2 z-score on fit rows recovers vendor inertial windows and vendor engineered rows within absolute error 1e-5. The inverse does not recover pre-filter signals or pre-scaling features.

Primary modeling uses the Experiment 2 temporal representation so that sensor channels remain separable. The engineered representation is available for a secondary classification comparison and is not the context or retrieval representation.

## Gaussian gate, geometry, and tau schedule

The preserved gate, for strictly positive tau, is

```
G_ij(tau) = exp(-d_ij^2 / (2 tau^2))
W_eff = W ⊙ G(tau)
```

`d_ij` is the absolute difference of fixed unit-interval coordinates. Gate rows are destination units and columns are source units. Hidden units use `linspace(0, 1, width)`. A requested tau below the floor is replaced by the floor. The default floor is `1e-6`, and it must be configured as a strictly positive number. The exponential may underflow to 0. Values are clamped to [0, 1]. At distance 0 the gate is 1. Increasing tau does not decrease the gate at a fixed distance.

On this positive-tau domain the local implementation must match `conn_exp.gates.gaussian_gate` exactly, including the floor flag. Non-positive tau is rejected here. Experiment 2 would have replaced it with the floor; that silent substitution is not kept.

During training the original `GatedLinear` detaches `G` before `W ⊙ G`. The gate is a function of distance and tau, not of `W`. The output head is dense.

`d50` is the distance at which the gate equals 1/2:

```
d50 = tau * sqrt(2 ln 2)
```

Experiment 2's HAR configuration, which Original CoNN keeps:

| Setting | Value |
|---|---|
| Hidden widths | 64, 64 |
| Optimizer | plain SGD, momentum 0, weight decay 0, CPU |
| Batch size | 128 |
| Updates | 3200 |
| Evaluation interval | 160 |
| Learning-rate grid | 0.01, 0.05, 0.1 |
| Release fraction | 0.7, so release step 2240 |
| Schedule endpoints | d50 0.08 then 1.0 |

Tau for a scheduled step `s` in `0 ... 2239` interpolates from `tau_start = 0.08 / sqrt(2 ln 2)` to `tau_end = 1.0 / sqrt(2 ln 2)` with the Experiment 2 smoothstep: `u = s / 2239`, then `u^2 (3 - 2u)`. Steps `2240 ... 3199` use an exact gate of ones for conditions D, B, and C.

Condition roles copied from Experiment 2:

- A, Dense: exact ones at every step. The forward path skips gate multiplication.
- D, Original CoNN: the scheduled Gaussian gate, then exact ones after the release step.
- S08 and S25: fixed d50 0.08 and 0.25 at every step, including after the release step. These are gate-width ablations, not the scheduled comparison.
- B: D's gate with a frozen within-row column permutation. Row energy matches D.
- C: D's input-side gate with a frozen permutation of input coordinates. The hidden-to-hidden gate matches D.

S08, S25, B, and C are specificity and width controls. They are outside the eight-cell context/memory matrix. A claim that distance geometry, rather than a multiplicative mask, caused a result requires the B comparison. A claim about the training schedule requires the result to be labeled as scheduled, fixed-tau, or post-training. Those three are not interchangeable.

Experiment 2's selected learning rates are historical (`engineered` mostly 0.1 or 0.05, `temporal` mostly 0.01). This protocol does not reuse them. Development chooses a learning rate later, with the same grid and the same budget for every condition, using grouped validation only.

## Hypotheses

H1, Track A. With the temporal input, encoder, optimizer, batch size, and 3200-step budget held fixed, Original CoNN changes official-test subject-level macro-F1 relative to Dense.

H2, geometry. Any Track A improvement assigned to the coordinate system also appears against the energy-matched row-shuffled gate B, under the same schedule.

H3, context. An externally supplied sensor-availability code changes predictions when the contextual factor is free, relative to the same weights with that factor fixed at 1. Activity labels and subject identity are not context variables.

H4, retrieval. On declared paired views of the same window, similarity retrieval ranks the paired view above shuffled-value and randomized-key controls. Supervised class similarity is not the success measure.

H1–H4 are separate. Failing one does not get absorbed into a composite score.

## Controls

The factorial matrix, implemented only in a later phase:

| Gaussian gate | Context gate | Memory | Condition |
|---|---|---|---|
| No | No | No | Dense |
| Yes | No | No | Original CoNN |
| No | Yes | No | Dense + Context |
| Yes | Yes | No | CoNN + Context |
| No | No | Yes | Dense + Memory |
| Yes | No | Yes | CoNN + Memory |
| No | Yes | Yes | Dense + Context + Memory |
| Yes | Yes | Yes | CoNN + Context + Memory |

Additional controls, also later:

- kNN on the same scaled input vectors the network receives. Primary k is 5. k = 1 is secondary.
- kNN on frozen Dense penultimate activations.
- Specificity gates B and C, and fixed-width gates S08 and S25.
- Memory ablations: as-designed memory, randomized keys, shuffled values, zero or disabled memory, kNN retrieval without CoNN, and a capacity-matched comparison.

Trainable parameter counts and stored key/value bytes are reported separately. A cached activation bank is storage, not a parameter, unless a later design explicitly makes the bank trainable. The first memory design does not.

## Context and memory rules for later phases

Context codes are `both`, `accelerometer`, and `gyroscope`.

- Accelerometer channels: `body_acc_{x,y,z}` and `total_acc_{x,y,z}`.
- Gyroscope channels: `body_gyro_{x,y,z}`.

The code `c` is supplied with the example. It is not inferred from `x`. The primary contextual factor cannot amplify the Gaussian gate: `C(c)` is positive, initialized at 1, and bounded above by 1, so `W_eff = W ⊙ G(tau) ⊙ C(c)` stays at or below `G(tau)`. An ablation that allows `C(c) > 1` must be reported as a change to that prior. The factor is low-rank. It is not a full per-example hypernetwork matrix. Fixed-representation interventions, where `x` is unchanged and only `c` changes, are reported separately from actual missing-sensor inputs.

The first memory bank uses encoder-derived keys and non-label values. Retrieved values enter by a residual into the penultimate representation `h2`, before the classification head. The residual projection is initialized at zero so the model starts at the no-memory baseline. Tau stays independent of the retrieval temperature `T_mem`.

Training retrieval refreshes detached keys at declared synchronization points. Gradients do not flow into stale cached keys. Ordinary retrieval excludes the query row. It does not exclude inferred neighbors, because that graph does not exist. It does not retrieve the query label as a value.

A labeled correspondence episode may retrieve a different augmented view of the same window, and may not retrieve the identical entry. Those support and query partitions are separate from the classification training memory. At a future final evaluation, episodic support is subject-disjoint, frozen, and not used for gradient updates.

## Splitting

The official nine test subjects are the untouched final test set. Development uses 7-fold subject-grouped cross-validation on the 21 official training subjects. Fold assignment is `numpy.random.default_rng(20261018).permutation` of the sorted training IDs, cut into contiguous groups of three. The groups are listed in the frozen constants below. Each training subject occurs in one fold. No test subject occurs in a fold.

The Experiment 2 five-subject validation cut remains a historical scaler partition. It is not the model-selection split for this protocol. The z-score statistics stay the published fit-subject statistics, including when a development fold contains one of those sixteen people or one of the five. Refitting the scaler inside each fold would change the input representation relative to Experiment 2 and is not part of this protocol. The limitation is accepted: some development folds evaluate people who contributed to the published scaler, and the five who did not are evaluated with a scaler fit on other training people. Official test people contributed to neither the scaler nor the folds.

## Metrics and success criteria

These criteria are fixed before final test evaluation. Five training seeds, reserved as `400, 401, 402, 403, 404`, are an engineering run. They are disjoint from Experiment 2 tuning seeds 30–34 and replication seeds 300–319. Five seeds do not meet the success bar.

### Track A — classification

Report accuracy, macro-F1, per-class recall, subject-level scores, the confusion matrix, parameter count, stored-memory bytes, training time, and inference cost. No unnamed composite is the decision metric.

Primary contrast: temporal Original CoNN minus temporal Dense, official-test macro-F1. Average each subject first, then compare the nine subject means. Uncertainty is a subject bootstrap percentile interval of that paired difference, plus the subject-level effect size. Window-level tests are not the primary inference.

Success for an improvement claim, in a later confirmatory run rather than the five-seed check: the subject-clustered interval for CoNN minus Dense lies entirely above zero, and the predeclared comparison with row-shuffled condition B is reported beside it. If B matches D, the result is a multiplicative-mask result, not a geometry result.

The eight-cell matrix, both kNN baselines, per-class recall, and the engineered representation are secondary. Gate-width sweeps and post-training tau edits are reported as interventions, separate from the scheduled training comparison.

### Track B — associative retrieval

Candidate pools and support/query splits are identical across models. The correspondence is a declared second view of the same released window: the complementary sensor group, or a declared augmentation. It is not "another window of the same activity."

Report Recall@1, Recall@5, and mean reciprocal rank, including under the declared noise and masking levels and across the three sensor contexts. Shuffled values, randomized keys, and disabled memory are the negative controls.

Primary retrieval contrast: Recall@1 of the Gaussian memory model minus Recall@1 of the shuffled-value control, on the same pools, summarized by subject. A second required comparison is kNN on identical pools. Success for an associative-retrieval claim requires the subject-level interval against the shuffled-value control to lie above zero and the kNN comparison to be reported on those same pools. Beating class-majority similarity is not sufficient.

Static activities (sitting, standing, laying) and dynamic activities (the three walking labels), and known sitting/standing confusions, may be described only as label-based structural diagnostics.

## Evaluation rules

- Development code loads official test features only with purpose `audit`, and audit code does not fit a model.
- Purpose `final_evaluation` is rejected until a later protocol amendment authorizes it.
- The same seeds, folds, and step budget are used for every cell that is compared.
- Subject-clustered summaries and subject bootstrap intervals are the inferential summaries. Effect sizes are reported with the interval.
- Five seeds remain an engineering check after they are run.
- No biological memory, unsupervised discovery, or unique benefit of representational collapse is claimed from this design alone.

## Phase gates

Gate 0, this document: sources, splits, provenance limits, hypotheses, controls, and success criteria are fixed.

Gate 1: data hashes, subject separation, normalization inversion, embargo behavior, and Gaussian equivalence tests pass.

Gate 2, later: Dense and Original CoNN run and repeat before context or memory is added.

Gate 3, later: context dependence, gradients, and the non-Gaussian context control.

Gate 4, later: row exclusion, detached key refresh, retrieval, and memory ablations, before combined training.

Gates 5 and 6, later: the two evaluation tracks and the report that separates verified mathematics, measured classification, measured retrieval, ablation findings, and unverified interpretation.

## Frozen constants

```frozen-json
{
  "protocol_version": "v2",
  "phase_scope": "phase_0_and_1",
  "experiment2": {
    "run_id": "20261019T004346Z_experiment2_har_v1",
    "archive_sha256": "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031",
    "prepared_manifest_sha256": "44840bb09ccaf8af79e6f2e4ed7c02713880895fe0a63e34e637b7ffb5d65517",
    "subject_split_sha256": "d941639621a5dcdd94623507c0e29c8a2b27886dc30813b6b3e207cd88adf698",
    "dataset_audit_sha256": "22aeab98f6f8de894e4c5079d6bfd8a62dd8ca2e7739b04779a4aadeacd18b27",
    "experiment2_protocol_sha256": "7c469aea89cc73fa8de8373493755186a962bfb4bc200802882f8ae65b81cf83"
  },
  "labels": {
    "0": "WALKING",
    "1": "WALKING_UPSTAIRS",
    "2": "WALKING_DOWNSTAIRS",
    "3": "SITTING",
    "4": "STANDING",
    "5": "LAYING"
  },
  "official_train_subjects": [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30],
  "official_test_subjects": [2, 4, 9, 10, 12, 13, 18, 20, 24],
  "experiment2_fit_subjects": [1, 3, 5, 6, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30],
  "experiment2_validation_subjects": [7, 8, 11, 14, 15],
  "window_counts": {"official_train": 7352, "fit": 5796, "experiment2_validation": 1556, "official_test": 2947},
  "development_cv": {
    "generator": "numpy.random.default_rng",
    "seed": 20261018,
    "group_size": 3,
    "folds": [[14, 21, 29], [6, 7, 17], [22, 23, 28], [8, 11, 26], [5, 15, 16], [3, 25, 30], [1, 19, 27]]
  },
  "channels": ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z", "total_acc_x", "total_acc_y", "total_acc_z"],
  "accelerometer_channels": ["body_acc_x", "body_acc_y", "body_acc_z", "total_acc_x", "total_acc_y", "total_acc_z"],
  "gyroscope_channels": ["body_gyro_x", "body_gyro_y", "body_gyro_z"],
  "window_samples": 128,
  "sampling_hz": 50,
  "vendor_overlap_fraction_documented": 0.5,
  "overlap_graph_available": false,
  "tau_floor_default": 1e-6,
  "schedule": {
    "hidden_sizes": [64, 64],
    "total_steps": 3200,
    "batch_size": 128,
    "eval_interval": 160,
    "release_fraction": 0.7,
    "release_step": 2240,
    "d50_start": 0.08,
    "d50_end": 1.0,
    "tau_start": 0.06794574402304153,
    "tau_end": 0.8493218002880191,
    "s08_d50": 0.08,
    "s25_d50": 0.25,
    "s08_tau": 0.06794574402304153,
    "s25_tau": 0.21233045007200477,
    "learning_rate_grid": [0.01, 0.05, 0.1],
    "optimizer": "sgd",
    "momentum": 0.0,
    "weight_decay": 0.0,
    "engineering_seeds": [400, 401, 402, 403, 404]
  }
}
```
