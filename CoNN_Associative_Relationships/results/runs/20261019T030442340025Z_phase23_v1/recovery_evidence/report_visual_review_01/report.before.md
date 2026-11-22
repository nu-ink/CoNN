# Phase 2/3 implementation and development results

## Executive summary

Completed **1050/1050 development fits**, 3,360,000 optimizer updates: six Phase 2 conditions and four Phase 3 conditions, each with seven subject folds, five engineering seeds and three learning rates. Every fit completed 3200 updates. Models, exact-resume checkpoints, curves, validation predictions, class/subject metrics, diagnostics and manifests are saved under `20261019T030442340025Z_phase23_v1`. No official-test model evaluation occurred. No Phase 4 component or associative retrieval result exists.

The primary development Gaussian−Dense macro-F1 difference is +0.5700 percentage points (descriptive subject-bootstrap 95% interval [+0.0743, +1.1963]). The required Gaussian−row-shuffled comparison is +1.1375 percentage points (descriptive subject-bootstrap 95% interval [+0.4350, +1.9800]). These are selected-LR development results, not independent replication or final-test confirmation.

Context increments are Gaussian+Context−masked Gaussian -0.0418 percentage points (descriptive subject-bootstrap 95% interval [-0.2606, +0.1696]), and Dense+Context−masked Dense -0.0093 percentage points (descriptive subject-bootstrap 95% interval [-0.1735, +0.1567]). Interpret them alongside actual modulation magnitudes, same-weight interventions, extra capacity and the numerical limitations below; they do not establish a benefit from associative memory.

The original CPU environment and all checked historical artifacts remain unchanged. An isolated CUDA environment works on both cards. Sequential warmed-up benchmarking favored CPU for this small MLP. The grid began with six one-thread CPU workers and resumed with eight; execution plans label these contention conditions. Historical import/Tk problems did not recur. One initial test assertion that every U parameter must change on a single minibatch failed: the test exposed real float32 update rounding, not detached gradients. A Windows checkpoint-replacement error also paused the grid; evidence was preserved, a bounded I/O retry was added, and the interrupted trajectory was replay-verified before resuming. All setup/test/failure attempts remain saved.

## Implementation inventory

All implementation files below are new; no pre-existing Phase 0/1 source, protocol, configuration, report or result was modified:

- `PROTOCOL_PHASE23_v1.md`
- `README_PHASE23.md`
- `scripts/phase23_preflight.py`
- `scripts/run_phase23.py`
- `scripts/phase23_report.py`
- `scripts/phase23_finish.ps1`
- `scripts/phase23_finalize.py`
- `src/conn_assoc/phase23_core.py`
- `src/conn_assoc/phase23.py`
- `src/conn_assoc/phase3_context.py`
- `src/conn_assoc/phase23_analysis.py`
- `src/conn_assoc/phase23_diagnostics.py`
- `src/conn_assoc/phase23_verify.py`
- `tests/test_phase2.py`
- `tests/test_phase2_cuda.py`
- `tests/test_phase3.py`
- `tests/test_phase23_integrity.py`
- `tests/test_phase23_io.py`
- `config/phase23_active_run.txt`

Generated environments/artifacts: `.venv_cuda23/`, `.pip-cache23/`, this report, and the unique run directory. Source snapshots before each implementation stage and source manifests are under `../results/runs/20261019T030442340025Z_phase23_v1/source/`; original protocol/readme/audit snapshots are under `prior_snapshot/`. Original Experiment 2 `conn_exp` model/gate utilities are imported read-only from their existing source and hash-verified. There is no Git repository/commit; source snapshots and SHA-256 seals provide provenance.

### Frozen design and authorization

The user explicitly authorized this separate Phase 2/3 extension. `PROTOCOL_v2.md` is unchanged, hash `e97d35bab3e684cd4ce09fe8dc8e8bafa22371b6f9c0c4801b1fd84eb0f01c48`. The separately sealed amendment hash is `8ed6dcdbefaa0c22b02fb9ec2c449d45e66e0e7e5ce8bf4b4ce7e783af0eab30`; resolved configuration hash `43f847d243b4105a8d617361cd7ed79dbe7bad114bc7733f5f61b5d94b20fad5`. Seal time `2026-10-19T03:04:47.652990+00:00` predates training. The amendment changes only scope and explicitly defined new context/evaluation implementation; it does not lift the official-test embargo.

New names map to Experiment 2 A/D/B/C/S08/S25 as specified in the amendment. `P2_gaussian` is original D, not historical B. All use 1152 time-major inputs, 64/64 ReLU, six-class dense head, paired uniform initialization, plain SGD without momentum/decay, batch 128, LR [.01,.05,.1] and seeds 400–404. Original Gaussian geometry/detach behavior/smoothstep are retained; release step 2240 uses exact ones. Fixed d50 .08/.25 never release. Row permutations preserve row energy; coordinate permutation changes gate assignments, not input values.

The seven validation folds are the unchanged v2 groups of three subjects. Only the 7352 official-training rows are used. Initialization, batch order, mask permutations and context/execution orders have independent streams. The first 42 Dense/Gaussian seed-400 fits were completed/reviewed before context implementation; they count toward the grid. `first_seed_review.json` and `first_seed_summary.csv` preserve that progression without selecting LR early.

Every condition receives 105 fits and 336000 updates. LR is selected solely by final validation subject-macro-F1 averaged equally over 21 subjects, five seeds and, for Phase 3, three contexts; ties use validation loss then lower LR. `selection.frozen.json` records all candidates. No best-curve checkpoint selection or test-guided redesign occurred.

### Context mechanism and masking

External codes are both / accelerometer / gyroscope. Both body and total acceleration derive from the accelerometer group; the gyroscope supplies three angular-velocity channels. Unavailable channels become zero after frozen scaling. All four Phase 3 conditions see identical balanced context/mask sequences, batches and training budgets; only contextual models receive the code as a modulation selector. Comparisons within Phase 3 hold the masking curriculum fixed. Comparing Phase 2 full-sensor training directly to Phase 3 mixed-sensor training would confound that curriculum.

For each hidden layer, `C(c)=1-(.95/2)*sigmoid(U[c])@sigmoid(V[c]).T`, with nominal attenuation rank 2, U initialized -6 and V zero. C is positive, bounded [.05,1], initially approximately .9988255. This near-one initialization is explicitly amended from v2 exact-one so gradients can exist at initialization. Gaussian factors are detached; C is differentiable and cannot amplify G. Context models add 8064 parameters (86406 total versus 78342). They match each other in capacity, not their smaller baselines. There is no full per-sample hypernetwork matrix.

## CUDA report

Verified CPU: Intel(R) Core(TM) i9-10900KF CPU @ 3.70GHz; 20 logical CPUs; 127.920 GiB RAM. Original executable: `E:\CoNN\conn_experiment_01\.venv\Scripts\python.exe`, Python 3.11.1 / PyTorch 2.14.1+cpu. Original package freeze is unchanged.

Isolated executable: `.venv_cuda23/Scripts/python.exe`, PyTorch 2.14.1+cu130 with bundled CUDA 13.0. `nvidia-smi` records driver 617.42 and supported CUDA UMD 13.4. Both RTX 3070 Ti cards expose approximately 8 GiB independently. Tensor allocation, matmul, backward gradients and CPU transfer passed on each GPU. GPU checkpoint/optimizer smoke and CPU/GPU inference checks passed (atol 2e-5, rtol 2e-4). No system toolkit, model sharding or distributed training was installed.

Installation used the [official CUDA 13.0 PyTorch index](https://download.pytorch.org/whl/cu130/torch/) and [official installation guidance](https://pytorch.org/get-started/locally/). The sandbox network initially failed; a network-enabled retry found the wheel. Old pip dependency metadata/build resolution failed; upgrading pip only inside the new environment resolved it. Logs retain all attempts. `cuda_requirements.freeze.txt` and `cuda_pip_check.log` record the resulting environment. The initial mid-install import probe failed before support packages were ready; it was not treated as a CUDA execution result.

| device | condition | batches_per_second | samples_per_second | peak_gpu_allocated_bytes |
| --- | --- | --- | --- | --- |
| cpu | P2_dense | 1010.911782 | 129396.708075 | 0 |
| cpu | P2_gaussian | 491.438406 | 62904.115944 | 0 |
| cuda:0 | P2_dense | 626.692069 | 80216.584778 | 69616640 |
| cuda:0 | P2_gaussian | 483.164612 | 61845.070367 | 70779392 |

Timing used 20 warm-up and 200 measured updates per model, sequential CPU then GPU 0, float32/TF32 off, GPU synchronization. Device selection used throughput only. Two-GPU seed parallelism was not pursued because these measurements favored CPU. No GPU acceleration improvement is claimed. The subsequent CPU grid reuses identical deterministic Gaussian tensors to avoid repeated exponentiation; this numerical cache is not associative memory or a learned bank. Unit tests retain exact original-gate equivalence. Grid timings below include parallel contention and differ from isolated benchmark latency.

## Validation and execution

| Verification | Result | Evidence in run |
|---|---|---|
| Phase 2 tests | 24 passed | pytest_phase2_retry.log |
| Phase 2 including CUDA | 26 passed | pytest_phase2_cuda.log |
| Phase 3 including CUDA | 30 passed | pytest_phase3_cuda.log |
| Final CPU suite | 28 passed, 2 explicitly NOT RUN (CPU-only torch) | pytest_final_cpu.log |
| Additional integrity/I/O guards | 3 passed | pytest_recovery.log |
| Final delivery CUDA suite | 33 passed, including CUDA checks | pytest_delivery_cuda.log |
| Gate/control schedule equivalence | Bitwise equal at representative endpoints/interior/release | tests/test_phase2.py |
| Actual updates, masking, differentiability, bounds | Passed | tests/test_phase2.py, tests/test_phase3.py |
| Exact interrupted/resumed weights | Passed for Gaussian and context fits | verification/ resume fixtures |
| Independent final verification | 1050 fits; 5250 checkpoint hashes | final_verification.json |
| Stored predictions independently rescored | 13895280 rows, all confusion/recall/F1/accuracy match | final_verification.json |
| Same-weight pre/post release | 1050 identical weight pairs | final_verification.json |
| Historical preservation | 5529 original Experiment 2 and 193 prior associative files unchanged | final_verification.json |

Initial, pre-release, post-release, final and latest checkpoints store model/optimizer/RNG/batch/context histories and explicit next-step schedule. No stochastic LR scheduler exists. Latest checkpoints are atomic every 160 updates. A failed/interrupted process resumes from that boundary; completed fits are reused and never overwritten. Both uninterrupted-vs-resumed tests verify exact weights. Every completed fit checks checkpoint-reloaded predictions and actual weight changes.

Setup/test failures were preserved: an initial PowerShell JUnit argument invocation collected no tests; the corrected invocation passed. A Phase 3 assertion on U updates failed because float32 rounded tiny updates away while nonzero gradients and V updates remained. The revised test explicitly measures both gradients and actual V updates, and stores precision evidence; it does not claim U updated when it did not.

The first full-grid invocation paused with WinError 5 while replacing `P2_rowshuffle_seed403_fold5_lr0.1/latest.pt`. The last committed checkpoint was at update 640; the intact temporary checkpoint was at 800. Replaying those 160 updates from preserved model/optimizer/RNG state matched the temporary weights bitwise (`recovery_evidence/checkpoint_replace_01/replay_verification.json`). The OS did not identify the lock holder; a transient sharing/access issue is the supported diagnosis, not an asserted antivirus cause. Bounded replacement retry (at most ten attempts, short delays) preserves the last good checkpoint and leaves the temp file on persistent failure. The I/O regression test passed, original failure/checkpoints were copied before resume, and completed fits were reused. The 3,360,000-update total describes completed logical trajectories; failed/replayed work adds physical computation and is excluded from tuning outcomes.

The resumed completed jobs record 7 transient replacement retry events in their own `io_retries.jsonl` files, demonstrating recovery during actual training as well as the synthetic file-lock regression test.

## Phase 2 results

| condition | lr | subject_accuracy_mean | subject_macro_f1_mean | subject_macro_f1_seed_sd | parameters |
| --- | --- | --- | --- | --- | --- |
| P2_dense | 0.010000 | 0.887935 | 0.884649 | 0.003140 | 78342 |
| P2_gaussian | 0.010000 | 0.893032 | 0.890350 | 0.001682 | 78342 |
| P2_rowshuffle | 0.010000 | 0.882861 | 0.878974 | 0.001015 | 78342 |
| P2_coordperm | 0.010000 | 0.885722 | 0.881831 | 0.002274 | 78342 |
| P2_fixed08 | 0.050000 | 0.895167 | 0.893866 | 0.004115 | 78342 |
| P2_fixed25 | 0.050000 | 0.900882 | 0.899641 | 0.003201 | 78342 |

Accuracy/F1 here average subjects equally, then seeds. Seed SD quantifies five engineering initializations, not five independent populations. Full per-seed and per-subject values are in `analysis/selected_seed_scores.csv` and `selected_validation_subjects.csv`; class recalls/confusions and pooled-window OOF scores are separately labelled. Final train/validation loss, gaps and fold averages are in `condition_summary.csv`.

## Phase 3 results

| condition | lr | subject_accuracy_mean | subject_macro_f1_mean | subject_macro_f1_seed_sd | parameters |
| --- | --- | --- | --- | --- | --- |
| P3_dense_masked | 0.100000 | 0.806577 | 0.805753 | 0.004224 | 78342 |
| P3_gaussian_masked | 0.100000 | 0.816279 | 0.816141 | 0.005308 | 78342 |
| P3_dense_context | 0.100000 | 0.806446 | 0.805660 | 0.004759 | 86406 |
| P3_gaussian_context | 0.100000 | 0.815905 | 0.815723 | 0.006973 | 86406 |

Realistic sensor-specific performance (0 both, 1 accelerometer, 2 gyroscope):

| condition | context | accuracy | macro_f1 |
| --- | --- | --- | --- |
| P3_dense_context | 0 | 0.889121 | 0.885316 |
| P3_dense_context | 1 | 0.862624 | 0.854357 |
| P3_dense_context | 2 | 0.667591 | 0.677308 |
| P3_dense_masked | 0 | 0.888454 | 0.884384 |
| P3_dense_masked | 1 | 0.862848 | 0.854628 |
| P3_dense_masked | 2 | 0.668430 | 0.678246 |
| P3_gaussian_context | 0 | 0.897371 | 0.895023 |
| P3_gaussian_context | 1 | 0.876991 | 0.871033 |
| P3_gaussian_context | 2 | 0.673352 | 0.681113 |
| P3_gaussian_masked | 0 | 0.897947 | 0.895501 |
| P3_gaussian_masked | 1 | 0.877595 | 0.871444 |
| P3_gaussian_masked | 2 | 0.673295 | 0.681477 |

Paired comparisons and uncertainty:

| contrast | metric | mean | seed_sd | ci_low | ci_high | subject_dz |
| --- | --- | --- | --- | --- | --- | --- |
| P2_gaussian minus P2_dense | macro_f1 | 0.005700 | 0.002817 | 0.000743 | 0.011963 | 0.420010 |
| P2_gaussian minus P2_rowshuffle | macro_f1 | 0.011375 | 0.002093 | 0.004350 | 0.019800 | 0.611605 |
| P2_gaussian minus P2_coordperm | macro_f1 | 0.008519 | 0.002914 | 0.004431 | 0.013458 | 0.781602 |
| P2_fixed08 minus P2_gaussian | macro_f1 | 0.003516 | 0.004686 | -0.003837 | 0.011758 | 0.185913 |
| P2_fixed25 minus P2_gaussian | macro_f1 | 0.009291 | 0.003454 | 0.002632 | 0.016374 | 0.560774 |
| P3_dense_context minus P3_dense_masked | macro_f1 | -0.000093 | 0.001770 | -0.001735 | 0.001567 | -0.023477 |
| P3_gaussian_context minus P3_gaussian_masked | macro_f1 | -0.000418 | 0.002853 | -0.002606 | 0.001696 | -0.080241 |
| P3_gaussian_context minus P3_dense_context | macro_f1 | 0.010063 | 0.006266 | 0.003757 | 0.017011 | 0.639784 |
| P3_gaussian_masked minus P3_dense_masked | macro_f1 | 0.010388 | 0.005620 | 0.004675 | 0.016409 | 0.743353 |
| Gaussian × context interaction | macro_f1 | -0.000325 | 0.003403 | -0.002484 | 0.001788 | -0.063356 |

`individual_seed_differences.csv` supplies all seed-level paired differences; accuracy contrasts are in `paired_contrasts.csv`. Bootstrap intervals use 10000 subject resamples, averaging seeds first. They are descriptive post-selection development intervals: CV training sets overlap and the same folds selected LR. They do not establish confirmatory success, and no window-independent significance test is used.

### Actual modulation and same-weight interventions

Across selected context checkpoints, maximum absolute C departure from its near-one initialization ranges from 1.1920929e-07 to 2.50339508e-06; observed C range is [0.998822987, 0.998826385]. Individual U updates can round away in float32; `context_precision_and_symmetry.csv` records U/V changed-element counts and deltas. Both rank columns remain exactly symmetric across these checkpoints: **True**. Where symmetric, the nominal rank-two deficit operates at rank at most one, an initialization-induced limitation.

Measured total gradient norms, averaged over recorded steps/folds/seeds (base and context parameter families kept separate):

| condition | base | context | context_to_base_norm_ratio |
| --- | --- | --- | --- |
| P3_dense_context | 0.625896 | 0.000060 | 0.000097 |
| P3_gaussian_context | 0.627168 | 0.000068 | 0.000109 |

With weights fixed, realistic validation input masks unchanged, set C=1 (native minus disabled; fold/context averages):

| condition | accuracy_native_minus_disabled | macro_f1_native_minus_disabled | loss_native_minus_disabled |
| --- | --- | --- | --- |
| P3_dense_context | -0.000047 | -0.000054 | -0.001117 |
| P3_gaussian_context | -0.000031 | -0.000028 | -0.001019 |

Fixed-h1 interventions vary only the hidden context factor, holding the same learned representation fixed. They are diagnostics, not realistic missing-sensor predictions:

| condition | context | logit_rms_change |
| --- | --- | --- |
| P3_dense_context | 0 | 0.000000 |
| P3_dense_context | 1 | 0.000003 |
| P3_dense_context | 2 | 0.000003 |
| P3_gaussian_context | 0 | 0.000000 |
| P3_gaussian_context | 1 | 0.000005 |
| P3_gaussian_context | 2 | 0.000005 |

Additional capacity, near-constant attenuation and training-path changes remain alternative explanations for between-model differences. These interventions isolate immediate gate effects at the saved weights; they cannot retrospectively undo optimization or supply a capacity-matched training control. No causal claim that context caused a gain is justified solely by a positive contextual-model comparison.

## Curves and diagnostic plots

All figures derive from computed frozen gates or measured checkpoints/metrics. Representative weight/modulation plots use fixed seed 400, fold 0, never the best-performing fit.

![Frozen schedule gates](../results/runs/20261019T030442340025Z_phase23_v1/plots/gaussian_expansion.png)

![Effective trained weights](../results/runs/20261019T030442340025Z_phase23_v1/plots/effective_weight_patterns.png)

![Phase 2 measured curves](../results/runs/20261019T030442340025Z_phase23_v1/plots/P2_learning_curves.png)

![Phase 3 measured curves](../results/runs/20261019T030442340025Z_phase23_v1/plots/P3_learning_curves.png)

![Model comparison](../results/runs/20261019T030442340025Z_phase23_v1/plots/model_comparison.png)

![Paired development contrasts](../results/runs/20261019T030442340025Z_phase23_v1/plots/paired_contrasts.png)

![Sensor contexts](../results/runs/20261019T030442340025Z_phase23_v1/plots/sensor_context_effects.png)

![Dense modulation](../results/runs/20261019T030442340025Z_phase23_v1/plots/P3_dense_context_modulation.png)

![Gaussian modulation](../results/runs/20261019T030442340025Z_phase23_v1/plots/P3_gaussian_context_modulation.png)

![Gradient diagnostics](../results/runs/20261019T030442340025Z_phase23_v1/plots/gradients.png)

![Context versus base gradients](../results/runs/20261019T030442340025Z_phase23_v1/plots/context_gradients.png)

![Context modulation trajectories](../results/runs/20261019T030442340025Z_phase23_v1/plots/context_modulation_curves.png)

![Throughput](../results/runs/20261019T030442340025Z_phase23_v1/plots/cpu_cuda_throughput.png)

`release_and_final_comparisons.csv` distinguishes immediate release from later optimization: pre/post release checkpoints have identical weights, so their metric difference is a gate intervention. Final-minus-post-release additionally includes 960 optimizer updates. `representative_distant_connection_ablation.csv` measures final development checkpoint far-connection suppression at distance quantile .75, separately labelled post-selection diagnostics; it did not influence model selection or redesign.

Mean fold/context validation changes (positive accuracy/F1 is better; positive loss is worse):

| condition | macro_f1_immediate_gate_switch | macro_f1_further_960_updates | loss_immediate_gate_switch | loss_further_960_updates |
| --- | --- | --- | --- | --- |
| P2_coordperm | -0.000818 | 0.012500 | -0.000028 | -0.015878 |
| P2_dense | 0.000000 | 0.002212 | 0.000000 | 0.011108 |
| P2_fixed08 | 0.000000 | 0.002485 | 0.000000 | 0.017862 |
| P2_fixed25 | 0.000000 | -0.000819 | 0.000000 | 0.034111 |
| P2_gaussian | -0.003796 | 0.011178 | 0.006591 | -0.009913 |
| P2_rowshuffle | 0.000195 | 0.014973 | 0.000432 | -0.027556 |
| P3_dense_context | 0.000000 | 0.001705 | 0.000000 | 0.066574 |
| P3_dense_masked | 0.000000 | 0.002275 | 0.000000 | 0.071599 |
| P3_gaussian_context | -0.004619 | 0.006539 | 0.031379 | 0.054361 |
| P3_gaussian_masked | -0.004794 | 0.007808 | 0.031464 | 0.018578 |

![Release versus later optimization](../results/runs/20261019T030442340025Z_phase23_v1/plots/release_vs_optimization.png)

### Runtime and storage

| condition | fits | mean_job_seconds | sd_job_seconds | mean_update_seconds | parameters |
| --- | --- | --- | --- | --- | --- |
| P2_coordperm | 105 | 10.364918 | 1.545116 | 7.532837 | 78342 |
| P2_dense | 105 | 7.506402 | 1.187017 | 5.767248 | 78342 |
| P2_fixed08 | 105 | 17.322094 | 1.465200 | 13.254146 | 78342 |
| P2_fixed25 | 105 | 9.194395 | 1.217573 | 7.246189 | 78342 |
| P2_gaussian | 105 | 9.433813 | 1.221337 | 7.291586 | 78342 |
| P2_rowshuffle | 105 | 11.205554 | 1.471717 | 8.022048 | 78342 |
| P3_dense_context | 105 | 16.977538 | 2.140371 | 11.486948 | 86406 |
| P3_dense_masked | 105 | 12.455083 | 1.517840 | 7.625171 | 78342 |
| P3_gaussian_context | 105 | 18.580025 | 2.199343 | 12.591830 | 86406 |
| P3_gaussian_masked | 105 | 14.205314 | 1.982512 | 8.882853 | 78342 |

Job wall times include evaluation, checkpointing and contention. Update time excludes most gate construction/evaluation and GPU asynchronous timing is not used for the CPU grid. GPU peak allocation belongs to the isolated benchmark/smokes, not CPU training. `python_process_memory_during_grid.json` records observed process working sets/peaks, including deterministic gate caches; it is not a per-model incremental RAM estimate. Base/context parameter tensors require 313368/345624 float32 bytes, excluding activations, optimizer/RNG/data and cached tensors. Every checkpoint and stored array contributes separate disk storage, not trainable parameters.

## Scientific interpretation

The executed development contrast is Gaussian−Dense +0.5700 percentage points (descriptive subject-bootstrap 95% interval [+0.0743, +1.1963]), alongside required Gaussian−rowshuffle +1.1375 percentage points (descriptive subject-bootstrap 95% interval [+0.4350, +1.9800]). These measured values should be read with their intervals, individual seeds and subject dependence; they do not supersede historical Experiment 2's negative engineered-feature primary or unresolved temporal secondary results. Task, folds, scaler, schedule and test embargo were preserved.

The development comparison favors scheduled Gaussian CoNN over Dense; this is exploratory support within the frozen CV design. The development comparison favors structured Gaussian CoNN over the energy-matched shuffled control; this is exploratory support within the frozen CV design. Neither comparison meets v2's later independent-confirmation rule.

The fixed d50=.25 control has the highest Phase 2 validation mean. Its paired macro-F1 difference from the scheduled model is +0.9291 percentage points (descriptive subject-bootstrap 95% interval [+0.2632, +1.6374]). Fixed locality deserves further controlled study. This does not prove that widening is harmful: width, regularization, optimization and representation can interact. The same-weight release diagnostic isolates only the immediate gate change; the subsequent 960 updates are a different intervention.

The development advantage of Gaussian + Context over masked Gaussian is unresolved: this descriptive interval includes zero. The development advantage of Dense + Context over masked Dense is unresolved: this descriptive interval includes zero.

The contextual comparisons measure the frozen mechanism under a matched sensor-mask curriculum. Immediate intervention and modulation measurements are essential: tiny modulation changes and initialization symmetry restrict what this implementation tests about useful context conditioning. A better contextual-model score alone cannot distinguish explicit context use from extra parameters, nearly constant rescaling or optimization effects. Positive/negative differences are retained equally; no final-test redesign follows from them.

No memory bank, retrieval support/query evaluation, Recall@k or MRR was implemented. Neither classification gains nor successful software execution demonstrate associative memory, prove the geometry hypothesis or establish a new architecture.

## Limitations

- Fixed Experiment 2 scaling uses 16 of the 21 development subjects; some validation-fold subjects contributed to preprocessing. This accepted leakage limitation is not silently corrected. A fold-fitted sensitivity study requires a separately declared amendment.
- The release lacks continuous session/timestamp/overlap keys. Within-window temporal alignment is valid; adjacent rows are not asserted to be overlapping neighbors.
- Five engineering seeds and selected-LR CV are not independent confirmation. Bootstrap intervals do not remove overlapping training sets or selection bias. Test subjects have appeared in historical research but were not loaded by this development runner.
- Context has 8064 extra parameters; no parameter-matched noncontext training condition was added to the frozen ten-condition grid. Symmetric factor initialization and float32 rounding limit effective context learning.
- Balanced homogeneous context per minibatch is the frozen policy. It does not test arbitrary mixed-code examples in the same matrix operation.
- CPU/GPU agreement was checked within declared tolerances; full models were trained on CPU only. No cross-backend full-grid equivalence or GPU speedup is claimed.
- Representative far-connection and fixed-hidden interventions are post-selection development diagnostics, not predeclared primary confirmatory comparisons.
- Git history is unavailable. Full source snapshots, environment freezes, protocol/config hashes and file verification provide reproducibility records.

## Next research stage

Do not treat these classification measurements as justification that associative memory will work. Before Phase 4, review Phase 3's actual modulation/precision evidence and, if context remains nearly inert, predeclare a separate context-parameterization sensitivity experiment using development data only. Preserve this implementation and its results as measured, including negative findings. A future capacity-matched control and separately frozen independent protocol would improve attribution.

Phase 4 remains unimplemented and requires separate authorization/design: bank capacity, key refresh, temperature, exact-row exclusion, non-label values, support/query separation and matched negative controls. Final-test evaluation also remains separately unauthorized. No new protocol or test evaluation is silently initiated by this recommendation.

Reproducible commands are in `README_PHASE23.md`; the exact completed run is `E:\CoNN\CoNN_Associative_Relationships\results\runs\20261019T030442340025Z_phase23_v1`. To validate/resume existing work without changing its design:

```powershell
Set-Location E:\CoNN\CoNN_Associative_Relationships
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = 'src;..\conn_experiment_01\src'
$run = (Get-Content config/phase23_active_run.txt).Trim()
..\conn_experiment_01\.venv\Scripts\python.exe scripts/run_phase23.py grid --stage all --workers 6 --devices cpu
..\conn_experiment_01\.venv\Scripts\python.exe -m conn_assoc.phase23_verify --run $run
```
