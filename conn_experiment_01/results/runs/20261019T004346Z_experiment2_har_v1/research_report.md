# CoNN Experiment 2 — UCI HAR real-world validation

The primary comparison favored the row-shuffled control; the scheduled synthetic-data advantage did not transfer under this engineered-feature protocol.

Sole primary result: engineered-feature D − B in final test macro-F1, n=20 paired initialization/order seeds: mean -0.00501038 (-0.5010 points), sample SD 0.00789428, two-sided 95% Student-t CI [-0.00870502, -0.00131574], df=19, paired Cohen dz -0.6347. Positive/tied/negative seed differences: 3/0/17.

## Transfer findings and their scope

Against the conventional dense baseline, engineered D − A macro-F1 was -0.4822 points, 95% CI [-0.7217, -0.2427]; fixed S08 − A was -1.4607 points, 95% CI [-1.8772, -1.0443]; fixed S25 − A was -0.2586 points, 95% CI [-0.4965, -0.0208]. These are distinct from the primary alignment contrast. Beating a shuffled curriculum would not by itself show a useful improvement over dense training.
The predeclared temporal-window secondary D − B contrast was +0.1861 points, 95% CI [-0.1799, +0.5522]; D − A was -0.1951 points, 95% CI [-0.5247, +0.1344]. Coordinate-permutation comparisons were engineered D − C -0.1918 points, 95% CI [-0.5616, +0.1780] and temporal D − C +0.4866 points, 95% CI [+0.1711, +0.8020]. These secondary intervals are exploratory and unadjusted.
The table below reports both positive and negative comparisons. No method, width, layout, duration or learning rate was redesigned after observing test results. An isolated favorable secondary comparison does not overturn an unresolved or negative primary result.

## Dataset audit and predefined coordinates

Source: [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), DOI 10.24432/C54S4K. Official archive SHA-256 `c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`. The audit retained source documentation, activity labels, feature names, subject IDs, class/subject counts and archive inventory before any model training.
The primary input is the released 561-feature vector, in original indexed metadata order with coordinate j/560. Its neighborhood structure is semantic file order, not measured physical geometry. Repeated names (84 duplicated occurrences) are retained as distinct indexed columns. The secondary input is the nine-channel 128-sample inertial window flattened time-major, with all channels at time t assigned t/127. The axes are colocated at each time; no invented physical sensor distances are used. Hidden coordinates are fixed uniformly on the same unit interval. Neither layout uses labels, test values, correlations, learned embeddings or test-accuracy search.
Fit subjects: [1, 3, 5, 6, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]. Validation subjects: [7, 8, 11, 14, 15]. Test subjects: [2, 4, 9, 10, 12, 13, 18, 20, 24]. Window counts: {'train': 5796, 'val': 1556, 'test': 2947}. The official test subjects were retained; five of the 21 original training subjects were selected for validation using fixed split seed 104729, without labels or feature values.
Every added normalization statistic uses only the 16 fit subjects: per feature for engineered vectors; per sensor channel across fit windows/time for temporal inputs. Validation/test are transformed with those frozen statistics. Subject IDs and activity labels are excluded from model inputs. Overlapping windows cannot cross subject partitions. No validation subjects were merged back into training after selection.
561 features are supplied normalized to [-1,1]; README does not identify which split fitted original normalization. Our normalization is train-subject-only and cannot undo unknown upstream choices. Temporal signals are also vendor-filtered windows, not untouched continuous recordings.
No complete session/timestamp metadata: within-window temporal order is available, continuous ordering across rows is not assumed.
The audit inspected test schema, labels and counts for integrity; no model predictions or performance-based choices used test before the recorded freeze. See audit/dataset_report.md for the full audit.

## Comparable training and validation-only selection

Every condition uses two 64-unit ReLU hidden layers, a six-class dense output head, and the original plain SGD family (no momentum/weight decay), batch size 128, unweighted cross-entropy and 3200 updates. A is conventional dense and bypasses gate multiplication. S08 and S25 hold Gaussian d50 at .08 or .25. D smoothly widens .08→1 over the first 2240 updates, then uses exact dense gates. B applies frozen independent within-row permutations to D’s gates in both layers. C permutes input coordinate assignments only; its hidden gate follows D. B/C preserve D’s row gate multisets and energy. Feature values and labels are unchanged by either control.
Initialization, training row permutations, source splits and scaler are paired across all conditions and LRs for each seed. Separate initialization/order/mask/execution-order streams prevent one condition from consuming another’s randomness. Parameter counts are identical within each representation; input dimensions differ between representations, so cross-representation rankings also reflect input/parameter differences.
Each representation/condition received LR [.01,.05,.1] × five tuning seeds 30–34, for 180 fits total and 48,000 tuning optimizer updates per condition per representation. Mean final validation macro-F1 selected LR; ties used lower validation cross-entropy then lower LR. The final fixed-budget checkpoint is the scoring checkpoint. No early stopping, test-selected checkpoint or test-based layout choice was used.
protocol.frozen.json declared the primary and secondary contrasts before training. selection.frozen.json sealed selected LRs before final_evaluation.started.json and before final models loaded test arrays. Final evaluation used seeds 300–319, 240 models total. Each model computed native final test predictions once; pooled, subject and class metrics reuse those predictions.

| Representation | Condition | Selected LR | Parameter count |
|---|---|---|---|
| engineered | A: conventional_dense | 0.1 | 40518 |
| engineered | S08: fixed_local_d50_0.08 | 0.1 | 40518 |
| engineered | S25: fixed_local_d50_0.25 | 0.1 | 40518 |
| engineered | D: structured_local_to_global | 0.1 | 40518 |
| engineered | B: row_shuffled_schedule | 0.05 | 40518 |
| engineered | C: permuted_input_coordinates_schedule | 0.05 | 40518 |
| temporal | A: conventional_dense | 0.01 | 78342 |
| temporal | S08: fixed_local_d50_0.08 | 0.05 | 78342 |
| temporal | S25: fixed_local_d50_0.25 | 0.01 | 78342 |
| temporal | D: structured_local_to_global | 0.01 | 78342 |
| temporal | B: row_shuffled_schedule | 0.01 | 78342 |
| temporal | C: permuted_input_coordinates_schedule | 0.01 | 78342 |

## Final test performance and generalization gaps

| Representation | Condition | Accuracy mean ± SD | Macro-F1 mean ± SD | Train−test accuracy gap | Train−test F1 gap |
|---|---|---|---|---|---|
| engineered | A | 0.929623 ± 0.003241 | 0.928713 ± 0.003079 | +0.070342 | +0.071255 |
| engineered | S08 | 0.915898 ± 0.008756 | 0.914106 ± 0.009162 | +0.074803 | +0.077448 |
| engineered | S25 | 0.928096 ± 0.005726 | 0.926127 ± 0.005534 | +0.068410 | +0.070704 |
| engineered | D | 0.925501 ± 0.005158 | 0.923892 ± 0.005212 | +0.071920 | +0.073730 |
| engineered | B | 0.930353 ± 0.006581 | 0.928902 ± 0.006407 | +0.064635 | +0.066552 |
| engineered | C | 0.927248 ± 0.007057 | 0.925809 ± 0.006867 | +0.067378 | +0.069313 |
| temporal | A | 0.881337 ± 0.003927 | 0.880731 ± 0.004062 | +0.072071 | +0.076949 |
| temporal | S08 | 0.878351 ± 0.008284 | 0.877778 ± 0.008479 | +0.075799 | +0.080566 |
| temporal | S25 | 0.869393 ± 0.010125 | 0.868555 ± 0.010450 | +0.070695 | +0.075459 |
| temporal | D | 0.879403 ± 0.005762 | 0.878780 ± 0.006105 | +0.072538 | +0.077481 |
| temporal | B | 0.877774 ± 0.005319 | 0.876918 ± 0.005566 | +0.074926 | +0.079996 |
| temporal | C | 0.874856 ± 0.006339 | 0.873914 ± 0.006592 | +0.077206 | +0.082432 |

Accuracy is pooled across test windows. Macro-F1 is the arithmetic mean of six activity F1 scores, with zero contribution for an undefined class denominator. Accuracy and F1 gaps are training minus test; positive gaps indicate poorer held-out performance. Full train/validation/test losses, gap SDs and val−test differences are in condition_summary.csv and raw scores.csv.

## Predeclared paired differences

Only engineered D − B macro-F1 is primary. All other comparisons use unadjusted secondary intervals. Intervals below quantify seed-to-seed optimization variation conditional on this one fixed subject partition, not independent population sampling of windows or subjects.

| Representation | Contrast | Metric | Mean difference | Paired SD | 95% t CI | dz |
|---|---|---|---|---|---|---|
| engineered | D_minus_B | accuracy | -0.004852 | 0.007875 | [-0.008538, -0.001167] | -0.6162 |
| engineered | D_minus_A | accuracy | -0.004123 | 0.005172 | [-0.006544, -0.001702] | -0.7971 |
| engineered | S08_minus_A | accuracy | -0.013726 | 0.008549 | [-0.017727, -0.009725] | -1.6055 |
| engineered | S25_minus_A | accuracy | -0.001527 | 0.005267 | [-0.003992, +0.000938] | -0.2899 |
| engineered | D_minus_C | accuracy | -0.001748 | 0.007747 | [-0.005373, +0.001878] | -0.2256 |
| engineered | S25_minus_D | accuracy | +0.002596 | 0.007153 | [-0.000752, +0.005944] | 0.3629 |
| engineered | S08_minus_D | accuracy | -0.009603 | 0.008188 | [-0.013435, -0.005771] | -1.1729 |
| engineered | D_minus_B | macro_f1 | -0.005010 | 0.007894 | [-0.008705, -0.001316] | -0.6347 |
| engineered | D_minus_A | macro_f1 | -0.004822 | 0.005118 | [-0.007217, -0.002427] | -0.9422 |
| engineered | S08_minus_A | macro_f1 | -0.014607 | 0.008897 | [-0.018772, -0.010443] | -1.6418 |
| engineered | S25_minus_A | macro_f1 | -0.002586 | 0.005083 | [-0.004965, -0.000208] | -0.5089 |
| engineered | D_minus_C | macro_f1 | -0.001918 | 0.007901 | [-0.005616, +0.001780] | -0.2427 |
| engineered | S25_minus_D | macro_f1 | +0.002235 | 0.006992 | [-0.001037, +0.005508] | 0.3197 |
| engineered | S08_minus_D | macro_f1 | -0.009786 | 0.008726 | [-0.013870, -0.005702] | -1.1214 |
| temporal | D_minus_B | accuracy | +0.001629 | 0.007427 | [-0.001847, +0.005105] | 0.2193 |
| temporal | D_minus_A | accuracy | -0.001934 | 0.006701 | [-0.005070, +0.001202] | -0.2886 |
| temporal | S08_minus_A | accuracy | -0.002986 | 0.008337 | [-0.006888, +0.000916] | -0.3582 |
| temporal | S25_minus_A | accuracy | -0.011944 | 0.010197 | [-0.016717, -0.007172] | -1.1714 |
| temporal | D_minus_C | accuracy | +0.004547 | 0.006552 | [+0.001481, +0.007613] | 0.6940 |
| temporal | S25_minus_D | accuracy | -0.010010 | 0.009263 | [-0.014345, -0.005675] | -1.0807 |
| temporal | S08_minus_D | accuracy | -0.001052 | 0.007722 | [-0.004666, +0.002562] | -0.1362 |
| temporal | D_minus_B | macro_f1 | +0.001861 | 0.007821 | [-0.001799, +0.005522] | 0.2380 |
| temporal | D_minus_A | macro_f1 | -0.001951 | 0.007042 | [-0.005247, +0.001344] | -0.2771 |
| temporal | S08_minus_A | macro_f1 | -0.002952 | 0.008557 | [-0.006957, +0.001052] | -0.3450 |
| temporal | S25_minus_A | macro_f1 | -0.012176 | 0.010484 | [-0.017082, -0.007269] | -1.1613 |
| temporal | D_minus_C | macro_f1 | +0.004866 | 0.006740 | [+0.001711, +0.008020] | 0.7219 |
| temporal | S25_minus_D | macro_f1 | -0.010224 | 0.009398 | [-0.014623, -0.005826] | -1.0879 |
| temporal | S08_minus_D | macro_f1 | -0.001001 | 0.007875 | [-0.004687, +0.002685] | -0.1271 |

Individual seed scores and paired differences are in replication/*individual_seed*.csv. Per-class F1 and confusion matrices are retained per model.

## Subject-level sensitivity

The same nine held-out people are used by all 20 models per condition. Treating 20×9 subject/model cells or overlapping windows as independent would exaggerate precision. The following descriptive intervals average D−B over model seeds within each subject, then use nine subject means (df=8). They describe subject heterogeneity within this small fixed held-out cohort and do not replace the primary seed analysis.

| Representation | Metric | Mean subject difference | Subject SD | Descriptive 95% t CI |
|---|---|---|---|---|
| engineered | accuracy | -0.005132 | 0.006993 | [-0.010507, +0.000243] |
| engineered | macro_f1 | -0.005152 | 0.006215 | [-0.009929, -0.000375] |
| temporal | accuracy | +0.001773 | 0.006674 | [-0.003357, +0.006903] |
| temporal | macro_f1 | +0.001897 | 0.006873 | [-0.003387, +0.007180] |

## Training time and diagnostic measurements

Updates include minibatch gathering, gate construction/application, loss, backpropagation and the SGD step. Update process CPU time and accumulated wall time exclude diagnostic evaluations. Job wall time includes diagnostics, checkpoint saves and result writing. Six one-thread CPU workers run concurrently; independent condition order randomization reduces ordering bias, but cache/contention/host load remain. These measurements do not establish isolated inference latency or sparse-compute acceleration.

| Representation | Condition | Update process CPU seconds, mean ± SD | Update wall seconds, mean | Total job wall seconds, mean |
|---|---|---|---|---|
| engineered | A | 3.888 ± 0.134 | 3.941 | 4.981 |
| engineered | S08 | 7.399 ± 0.214 | 7.493 | 9.984 |
| engineered | S25 | 4.378 ± 0.320 | 4.405 | 5.475 |
| engineered | D | 4.977 ± 0.269 | 5.008 | 6.253 |
| engineered | B | 6.052 ± 0.312 | 6.082 | 7.575 |
| engineered | C | 5.289 ± 0.309 | 5.326 | 6.479 |
| temporal | A | 5.320 ± 0.264 | 5.377 | 6.897 |
| temporal | S08 | 11.812 ± 0.288 | 11.910 | 16.325 |
| temporal | S25 | 5.890 ± 0.605 | 5.970 | 7.542 |
| temporal | D | 6.987 ± 0.446 | 7.052 | 8.862 |
| temporal | B | 8.716 ± 0.834 | 8.764 | 10.964 |
| temporal | C | 7.602 ± 0.505 | 7.636 | 9.374 |

Learning curves use full training and validation metrics every 160 updates plus a same-weight release pair and final. Test is not drawn as a learning curve. Raw diagnostics distinguish gate-attenuated dL/dW and dL/dW_eff on a fixed paired training batch; plots separate conditions and layers by native distance bins. Effective-weight magnitude and per-destination-unit gate energy are retained. Checkpoint reloads compare native/dense gates on validation only; distant-quartile ablations measure validation dependence without retraining. No counterfactual test gates were evaluated.
Positive gate-ablation loss deltas show dependence at those weights, not a necessary beneficial role during training. A static-local null far ablation can follow from already negligible far gates. The fixed-local conditions lack their own static shuffled controls in this benchmark; their dense-baseline comparisons do not isolate geometry from gate magnitude. The scheduled D/B contrast supplies the energy-matched primary control.

## Measured observations versus hypotheses and limits

The primary comparison favored the row-shuffled control; the scheduled synthetic-data advantage did not transfer under this engineered-feature protocol.
A positive engineered D/B result would support transfer of the scheduled alignment effect under this specific metadata layout; a null or negative result does not establish a real-world CoNN advantage. The temporal analysis is a separate predeclared test of a more direct time-neighborhood prior, with different inputs and parameter count. Neither a file-order effect nor a temporal-window result proves a general geometry hypothesis or a new architecture.
This is one sensor dataset and one fixed 16/5/9-subject partition. Twenty seeds are optimization repetitions, not 20 independent subject cohorts. Windows overlap and class/subject frequencies vary. The LR grid is narrow, only one network width and duration are tested, and no claim is made against tuned state-of-the-art HAR classifiers. Fixed local widths were transferred from synthetic studies, and secondary comparisons are numerous. Negative results bound this protocol rather than all possible coordinate systems, optimizers, network capacities or sensor tasks. Upstream preprocessing provenance limits any stronger end-to-end leakage claim.
All favorable and unfavorable results are retained; there was no final-test-driven redesign, seed replacement, split search or spatial-layout selection. Existing synthetic results are preserved byte-for-byte and are not pooled with real-world results.

## Reproducibility

The versioned directory contains the dataset audit, original-document copies, archive SHA-256, subject split seal, coordinate tables, frozen train-only scalers, prepared-data hashes, frozen protocol and LR selection, environment/package lock, exact source snapshot, paired initial states and batch indices, masks, all raw curves and diagnostics, native checkpoints, test predictions, CSV summaries and plots. No Git repository was present; source_manifest.json identifies the content-addressed code revision. Before/after manifests verify all prior runs remain unchanged. Regenerate this report with python -m conn_exp.experiment2_report <run_directory>.
