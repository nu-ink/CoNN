# Experiment 2: UCI HAR dataset audit

Source: [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), DOI 10.24432/C54S4K. Archive SHA-256 `c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`. Metadata and archive inventory are retained alongside this report.

Measured structure: 7352 official training windows, 2947 official test windows, 30 subjects, six activity labels, 561 engineered columns and nine 128-sample sensor channels. Duplicate engineered names: 84; all columns are retained by integer ID.
Activities are walking, walking upstairs, walking downstairs, sitting, standing and laying. Subject IDs are metadata, never input features. Train/test official subject sets are disjoint. Features describe acceleration, gravity/body motion, gyroscope, jerk, magnitude, frequency statistics and angles; original feature names and all sensor channel names are saved.

## Predetermined subject-disjoint partitions

Fit subjects: [1, 3, 5, 6, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30].
Validation subjects: [7, 8, 11, 14, 15].
Test subjects: [2, 4, 9, 10, 12, 13, 18, 20, 24].
Window counts: {'train': 5796, 'val': 1556, 'test': 2947}.
The fixed split seed selects validation subjects from official training IDs only. No class balance search, feature inspection or test outcomes select the split. Per-subject/class counts are structural audit measurements; test performance remains unmeasured.

## Coordinates and leakage controls

Primary representation: the released 561 engineered columns, in their original indexed order, with coordinate j/560. This is a semantic file-order prior, not a measured physical distance or a temporal sample axis. No covariance, labels or model outcomes define the layout.
Secondary representation: nine inertial channels flattened time-major, coordinate t/127 for all nine channels at time sample t. Channels at the same instant are colocated; the X/Y/Z axes are not assigned invented physical distances. Hidden-unit coordinates are fixed uniform unit-interval points. This encodes within-window temporal proximity only.
All added scaling is fitted once on the 16 training subjects: per engineered column, or per raw sensor channel across training windows/time. Validation/test receive those frozen training statistics. No feature removal, missing-value imputation, label-informed coordinate learning or split-specific renormalization is used. Overlapping windows cannot cross subject partitions.
561 features are supplied normalized to [-1,1]; README does not identify which split fitted original normalization. Our normalization is train-subject-only and cannot undo unknown upstream choices. Temporal signals are also vendor-filtered windows, not untouched continuous recordings.
No complete session/timestamp metadata: within-window temporal order is available, continuous ordering across rows is not assumed.

The primary and secondary layouts are specified before training and are never selected using test accuracy. Native feature values are not permuted in the coordinate control; only the assignment of input coordinates changes.

## Scope and source limitations

Current UCI page lists CC BY 4.0; bundled 2013 README contains older noncommercial wording. Both source records are retained; this run performs research analysis without redistributing a modified dataset.
This audit is descriptive, including test schema and counts. It performs no model evaluation. Independent seed repetitions on this one fixed subject partition will quantify initialization/order variation, not variability across new subject partitions.
