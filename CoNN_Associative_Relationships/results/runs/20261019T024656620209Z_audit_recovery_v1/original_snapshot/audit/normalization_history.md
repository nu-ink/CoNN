# HAR normalization history

This note records preprocessing that later code cannot undo. It is an inspection finding from Phase 0, checked again by the Phase 1 loaders.

## Vendor release

The source is the UCI Human Activity Recognition Using Smartphones archive, DOI 10.24432/C54S4K. The local file is `conn_experiment_01/data/raw/uci_har_original.zip`, SHA-256 `c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`. The archive contains a nested `UCI HAR Dataset.zip`. Experiment 2 did not extract that nested archive into a second raw tree; the inertial windows remain inside the zip.

The bundled README states:

- Signals were noise-filtered, then cut into 2.56 s windows at 50 Hz with 50% overlap (128 samples).
- Total acceleration was separated into body acceleration and gravity with a Butterworth low-pass filter at 0.3 Hz.
- Each released window has total acceleration, body acceleration, and gyroscope angular velocity, each on X/Y/Z.
- The 561 engineered features are already normalized and bounded to [-1, 1]. The README does not say which subjects were used to fit that normalization.

Those vendor steps are not reversible from the released files. Inverting Experiment 2's later z-score returns the released window or the released 561-vector. It does not return an unfiltered recording, and it does not return engineered features from before the vendor [-1, 1] scaling.

## Experiment 2 z-score

Experiment 2 fit a second normalization on 16 of the 21 official training subjects:

- Engineered representation: one mean and one standard deviation per column, across those subjects' windows.
- Temporal representation: one mean and one standard deviation per sensor channel, across those subjects' windows and the 128 samples.
- A scale below 1e-8 would have been replaced by 1. No channel or engineered column met that test.
- Validation and official test windows were transformed with the frozen fit-subject statistics.
- Prepared arrays are float32. On the fit rows, inverting that cast z-score recovers the vendor values within absolute error 1e-5.

The fit-subject lists, scaler files, and prepared arrays live under `conn_experiment_01/results/runs/20261019T004346Z_experiment2_har_v1/prepared/`. This experiment reads them and does not rewrite them.

## What remains unavailable

There is no session identifier, timestamp, or window-index field that links two different rows as overlapping samples of one continuous recording. The 50% overlap is a vendor acquisition parameter, not a row-to-row key. Adjacent rows are therefore not labeled as neighbors.

`body_acc_*` is the vendor's gravity-removed acceleration, not an independent third sensor. An accelerometer-only condition that keeps both `body_acc` and `total_acc` still uses that derived stream.
