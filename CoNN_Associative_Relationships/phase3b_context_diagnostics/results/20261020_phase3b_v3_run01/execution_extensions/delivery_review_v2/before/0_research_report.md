# Phase 3B v3 — Final measured research report

## 1. Executive summary

Completed **455 development grid fits and 10 pilots**, all 3200 updates: **1,488,000 training optimizer updates**. CPU was selected by throughput; all paired performance fits used CPU. Seventy C3 grid fits were omitted by the predeclared numerical futility rule. No official HAR test subjects were evaluated and no Phase 4 was implemented.

Primary Gaussian C4−Gaussian masked subject macro-F1: **+0.00213398**, descriptive95% subject-bootstrapCI **[+0.00011566,+0.00444627]**. The frozen +.005-and-positive-lower-bound predictive criterion was **not met**. This is exploratory development evidence.

Gaussian C4 met Level 1 in 35/35 fits. Level 2 operational pair/layer criteria passed in 49/175 comparisons; no undeclared study-wide aggregation turns this into universal learned context differentiation. See section8. Numerical repair, functional sensitivity and predictive benefit are separate outcomes.

## 2. Research question and frozen design

Why did historical contextual attenuation barely learn, and do symmetry breaking, reduced saturation and master-parameter precision yield useful external-context behavior? The primary contrast, .005 practical threshold, bootstrap, three secondary contrasts,15 conditions, common SGD 0.1, batch128,3200 steps,64/64 hidden widths, five base seeds 400–404 and seven historical folds were fixed before pilots. The raw v3 amendment and supplied v2 historical-reference synopsis are preserved. V3 superseded the 14-condition v2 matrix by adding frozen C4.

Inputs are128×9 time-major temporal HAR windows. IH context factors control64×1152 weights, HH 64×64 weights; categorical contexts 0 both,1 accel,2 gyro select three rank-two factor rows. Unavailable scaled channels are zeroed. Independent streams govern base initialization, factor noise(seed 30031), batches, balanced mask curriculum, shuffled codes, evaluation randomization and bootstrap. No validation LR search or stronger-initialization substitution occurred. The original Gaussian geometry and release schedule were unchanged.

## 3. Historical Phase 3 comparison and prospective precision

Before new performance training,5762 historical source/data manifest entries and all 1050 historical fit artifacts were verified. C0 checkpoint-loaded logits and saved predictions matched exactly in six fixtures. The end-of-run preservation audit passed separately.

| Historical contrast | Mean | Subject SD | Seed SD | New descriptive95% CI | Approx. zero-exclusion effect |
| --- | --- | --- | --- | --- | --- |
| dense context−masked | -0.00009255 | 0.00394236 | 0.00176987 | [-0.00172922,+0.00158275] | 0.00168618 |
| gaussian context−masked | -0.00041791 | 0.00520815 | 0.00285288 | [-0.00263591,+0.00172669] | 0.00222756 |

Fresh same-seed C0/masked full-fit checks: 140/140 exact final tensor states; 140/140 exact saved predictions. These are reproducibility checks, not independent replications. The.001 macro-F1 statistical/cross-backend tolerance was not substituted for exact checkpoint replay. Historical Monte Carlo intervals can have slightly different endpoints under the new frozen bootstrap seed.

All21 subjects have all 6 ground-truth classes. Historical six-class macro-F1 and supported-class sensitivity therefore coincide. False positives remain in the confusion matrices; present classes with no predicted positives receive zero precision. The historical precision estimates do not guarantee power: only 21 subject clusters, overlapping CV training sets and previous development inspection limit interpretation.

## 4. Mathematical gradient findings

C=1−.475 sigmoid(U)sigmoid(V)ᵀ. Local derivatives are not complete loss gradients. Analytical loss-gradient contractions matched autograd to1.39e−17 in the fixed float64 fixture; central finite differences passed at h=1e−5, atol=1e−8, rtol=1e−5. The original Gaussian gate was bitwise equivalent at tested schedule points.

The42-fixture gradient sweep used identical initial weights and training minibatches across seven initializations, two architectures and three masks, without validation classification scores. G0 context/base gradient ratio ranged1.0745282e-05–3.108699e-05; G2(U−4.5)ranged4.7248914e-05–0.00013756262. G2 reached the 1e−3 engineering reference in 0/6 fixtures and had zero maximum one-step float32 gate change. C2/C4 nevertheless retained−4.5 as frozen. Stronger candidates are diagnostic follow-up proposals only.

## 5. Symmetry and attenuation rank

SVD uses float64 A=1−C, with numerical rank threshold sigma1×1e−10. Identical component columns constrain attenuation to rank≤1. C1/C4 perturbations broke that exact symmetry, but their second component was initially weak. The prescribed perturbation was not increased.

| Cell | Initial sigma2/sigma1 range(all layers/contexts) | Final Gaussian grid range | Initial .01 target |
| --- | --- | --- | --- |
| C0 | 1.01574e-15–1.8029e-14 | 5.16269e-15–1.61155e-14 | below .01 |
| C1 | 0.00291041–0.00374616 | 0.00291031–0.00374627 | below .01 |
| C2 | 1.12781e-15–1.7679e-14 | 9.6066e-16–4.07194e-15 | below .01 |
| C3 | 1.01574e-15–1.8029e-14 | pilot only | below .01 |
| C4 | 0.00288579–0.00371387 | 0.00288537–0.00371395 | below .01 |

Rank-two allocation is not proof of two useful learned factors, semantic relationships or predictive gain. Initial random context differences must be separated from learned displacement.

## 6. Numerical precision findings

The following are full3200-step seed 400/fold0 pilots. Reference changes are evaluated in float64; forward changes use actual float32 modulation on live support.

| Pilot | Median context/base gradient | Max forwardΔC | Max referenceΔC | Max changed fraction | Level 1 | Grid decision |
| --- | --- | --- | --- | --- | --- | --- |
| dense_C0 | 0.00010479 | 4.76837158e-07 | 4.73925988e-07 | 1.0 | False | retained |
| dense_C1 | 0.00010053 | 4.76837158e-07 | 4.60504993e-07 | 0.9958881578947368 | False | retained |
| dense_C2 | 0.00044210 | 9.35792923e-06 | 9.36437402e-06 | 1.0 | True | retained |
| dense_C3 | 0.00010467 | 4.76837158e-07 | 4.78042598e-07 | 1.0 | False | omitted |
| dense_C4 | 0.00044018 | 9.11951065e-06 | 9.10689393e-06 | 1.0 | True | retained |
| gaussian_C0 | 0.00010206 | 8.34465027e-07 | 8.48384604e-07 | 0.9991776315789473 | False | retained |
| gaussian_C1 | 0.00010830 | 8.34465027e-07 | 8.53742999e-07 | 0.9901315789473685 | False | retained |
| gaussian_C2 | 0.00044695 | 1.65104866e-05 | 1.64748224e-05 | 1.0 | True | retained |
| gaussian_C3 | 0.00010205 | 8.34465027e-07 | 8.46731703e-07 | 1.0 | False | omitted |
| gaussian_C4 | 0.00045176 | 1.67489052e-05 | 1.67260991e-05 | 1.0 | True | retained |

C3 preserved float64 master updates but failed the 1e−6 forward threshold in both pilot architectures. Its70 grid fits were omitted and retained as explicit omitted_fits.csv entries, with no replacement. This is a pilot-specific engineering futility decision, not proof precision-only interventions cannot help elsewhere. C2/C4 produced changes that survived forward conversion.

| Condition | Factor | Mean initial→final changed fraction | Mean max parameter displacement |
| --- | --- | --- | --- |
| gaussian_C0 | context_ih.u | 0.94687500 | 0.00019933 |
| gaussian_C0 | context_ih.v | 0.66666667 | 0.00001059 |
| gaussian_C0 | context_hh.u | 0.72872024 | 0.00041368 |
| gaussian_C0 | context_hh.v | 1.00000000 | 0.00010487 |
| gaussian_C2 | context_ih.u | 0.99642857 | 0.00154585 |
| gaussian_C2 | context_ih.v | 0.66666667 | 0.00005442 |
| gaussian_C2 | context_hh.u | 0.92633929 | 0.00256290 |
| gaussian_C2 | context_hh.v | 1.00000000 | 0.00078305 |
| gaussian_C4 | context_ih.u | 1.00000000 | 0.00099623 |
| gaussian_C4 | context_ih.v | 0.66666667 | 0.00004970 |
| gaussian_C4 | context_hh.u | 1.00000000 | 0.00195107 |
| gaussian_C4 | context_hh.v | 1.00000000 | 0.00049621 |

These final displacements are not reconstructed stepwise fractions. Sampled aggregate stored-update fractions and gradient norms are raw measurements; per-factor stepwise stored-update fractions were not saved. Scalar-conversion warnings in diagnostic formatting were preserved; those scalar reads do not enter losses or optimization.

## 7. Controlled intervention results

| Condition | Completed grid fits | Mean subject macro-F1 | Mean subject accuracy | Level 1 fits | Allocated/trainable parameters |
| --- | --- | --- | --- | --- | --- |
| dense_C0 | 35 | 0.80566021 | 0.80644563 | 0/35 | 86406/86406 |
| dense_C1 | 35 | 0.80399815 | 0.80480480 | 0/35 | 86406/86406 |
| dense_C2 | 35 | 0.80612108 | 0.80695322 | 35/35 | 86406/86406 |
| dense_C3 | 0 | omitted | not measured | control / omitted | 86406/86406 (pilot) |
| dense_C4 | 35 | 0.80473136 | 0.80546495 | 35/35 | 86406/86406 |
| gaussian_C0 | 35 | 0.81572300 | 0.81590469 | 2/35 | 86406/86406 |
| gaussian_C1 | 35 | 0.81664449 | 0.81671925 | 5/35 | 86406/86406 |
| gaussian_C2 | 35 | 0.81378087 | 0.81388487 | 35/35 | 86406/86406 |
| gaussian_C3 | 0 | omitted | not measured | control / omitted | 86406/86406 (pilot) |
| gaussian_C4 | 35 | 0.81827489 | 0.81805155 | 35/35 | 86406/86406 |
| dense_masked | 35 | 0.80575277 | 0.80657734 | control / omitted | 78342/78342 |
| gaussian_masked | 35 | 0.81614090 | 0.81627894 | control / omitted | 78342/78342 |
| gaussian_constant | 35 | 0.81494531 | 0.81522125 | control / omitted | 86406/86406 |
| gaussian_shuffled | 35 | 0.81552920 | 0.81568715 | control / omitted | 86406/86406 |
| gaussian_frozen | 35 | 0.81561115 | 0.81610652 | control / omitted | 86406/78342 |

C0/C1 remain valid comparison cells even when they fail the forward-learning threshold. Masked, constant-code and frozen-gate controls have different expected numerical-learning behavior; a frozen gate is not an implementation failure. C3 classification performance is not extrapolated from one pilot. All available class metrics and per-seed/context results are preserved.

## 8. Context-specific functional differences

IH comparisons use source-support intersections. Accel/gyro intersection is empty and undefined; nonintersection differences are recorded separately. HH uses all potentially active hidden sources. A Gaussian shuffled-training null supplies35 matched engineering seed/fold values per nonempty layer/pair; its95 thpercentile is a diagnostic reference, not a significance test.

| Layer | Codes | Passes | Fits | Mean final normalizedΔA | Mean initialΔA | Mean learned contrast displacement | Mean null95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ih | (0, 1) | 6 | 35 | 0.13737150 | 0.13738332 | 0.00034103 | 0.13740576 |
| ih | (0, 2) | 1 | 35 | 0.13973515 | 0.13976918 | 0.00039256 | 0.13980142 |
| ih | (1, 2) | undefined | 0 | — | — | — | — |
| hh | (0, 1) | 24 | 35 | 0.12266167 | 0.12264720 | 0.00041427 | 0.12264828 |
| hh | (0, 2) | 8 | 35 | 0.12102245 | 0.12103100 | 0.00053079 | 0.12106218 |
| hh | (1, 2) | 10 | 35 | 0.12878328 | 0.12877864 | 0.00032646 | 0.12880963 |

The operational checks additionally require final normalized difference>1e−4, corresponding null95 exceedance, live forward learning and same-weight output change>1e−6. Final/initial differences and learned displacement are all shown because random initialization alone produces code sensitivity. Dense has no separately trained matched shuffled null; its functional diagnostics are descriptive. Pair-level results do not establish consistent study-wide learned differentiation.

Gaussian C4 fixed-h1 logit RMS difference mean0.0079705654, maximum recorded absolute logit change0.060890198. These frozen probes use the first 128 validation windows per fold, not the whole population. Separate IH-only interventions fix the HHcode0; disabled-Cand all five nonidentity permutations keep actual sensor masks and weights fixed.

## 9. Classification, convergence and generalization

| Condition | Train loss | Validation loss | Window macro-F1 gap | Window accuracy gap |
| --- | --- | --- | --- | --- |
| dense_masked | 0.20074748 | 0.79702308 | 0.10624455 | 0.10126798 |
| dense_C4 | 0.20092606 | 0.80281329 | 0.10729887 | 0.10228371 |
| gaussian_masked | 0.20762485 | 0.70244664 | 0.09380057 | 0.08897389 |
| gaussian_C4 | 0.20540441 | 0.71836076 | 0.09226814 | 0.08782323 |
| gaussian_frozen | 0.21171569 | 0.70748127 | 0.09195421 | 0.08716599 |

These gaps and the full learning curves use pooled windows within each split; the primary endpoint gives equal weight to subjects, contexts and seeds. Convergence.csv includes observed curve area and first recorded step crossing.8 window macro-F1. No interpolation fills missing measurements.

| Condition | Pre-release accuracy | Post-release accuracy | Final accuracy | Same-weight gate-switchΔ | Further-optimizationΔ |
| --- | --- | --- | --- | --- | --- |
| gaussian_C0 | 0.81120089 | 0.80671236 | 0.81445001 | -0.00448852 | 0.00773765 |
| gaussian_C2 | 0.81056952 | 0.80593388 | 0.81240328 | -0.00463564 | 0.00646939 |
| gaussian_C4 | 0.81016749 | 0.80640096 | 0.81679961 | -0.00376653 | 0.01039865 |
| dense_C4 | 0.80271474 | 0.80271474 | 0.80388126 | 0.00000000 | 0.00116653 |

All455 grid pre/post checkpoint pairs were independently confirmed bitwise identical in model weights at optimizer step 2240. Gate switching and further optimization are separate measured changes. This does not prove widening is universally helpful or harmful.

## 10. Primary comparison and uncertainty

| Contrast | Mean paired effect | Subject SD | Seed SD | Descriptive95% CI | Paired subject dz |
| --- | --- | --- | --- | --- | --- |
| gaussian_C4−gaussian_masked | +0.00213398 | 0.005159161971100433 | 0.002926468286497447 | [0.00011565544771546819,0.004446269568044705] | 0.4136298179153211 |

| Engineering seed | Paired subject macro-F1 difference |
| --- | --- |
| 400 | +0.00527059 |
| 401 | +0.00234289 |
| 402 | -0.00018842 |
| 403 | +0.00466035 |
| 404 | -0.00141550 |

The bootstrap averages five seeds within each subject after equal-context averaging, then applies10000 paired subject resamples, using identical subject draws for both conditions. Percentile95% intervals are descriptive post-selection development uncertainty. The +.005 practical threshold is retained regardless precision; no confirmatory significance claim is made.

Level 3 predictive criterion: **not met**. Numerical/functional evidence is separate; attributing any gain specifically to learned external context also requires valid Level 2 evidence.

## 11. Capacity and information controls

| Family | Contrast | Mean | Descriptive95% CI |
| --- | --- | --- | --- |
| secondary | gaussian_C4−gaussian_constant | +0.00332958 | [0.00019395090899479458,0.0070209081379799575] |
| secondary | gaussian_C4−gaussian_shuffled | +0.00274569 | [-0.0006054111959727092,0.006911917322823885] |
| secondary | dense_C4−dense_masked | -0.00102141 | [-0.00289157523621723,0.0005092487395759418] |
| exploratory | gaussian_C4−gaussian_frozen | +0.00266374 | [0.00018686656355502607,0.0053024720856095844] |

The three registered secondary contrasts are C4−constant,C4−shuffled,and Dense C4−Dense masked. No formal p-values are reported, so no uncorrected significance claims or post-hoc Holm family are introduced. Other comparisons are exploratory. Frozen C4 keeps its actual nonidentity initialization and trains base weights; it is not C=1. Constant-code allocation matches C4 but at most2688 context parameters are actively exposed, versus 8064 allocated; actual displacement and byte-storage counts are saved.

C4's incremental effect over frozen attenuation has a positive lower descriptive interval. This supports an incremental learning benefit within this development comparison. Both information-control contrasts have positive lower descriptive bounds: False. These are descriptive secondary comparisons, not independent confirmation or uniquely identified mechanisms.

Shuffled training codes preserve frequencies and are independent of true masks. Evaluations include true codes and independently randomized balanced per-window codes, explicitly different joint code/mask distributions from training. Capacity and fixed attenuation comparisons help interpret the primary result but do not uniquely identify every mechanism.

## 12. CPU/CUDA execution

Measured mean C4 throughput: CPU329.525 updates/s; CUDA0245.347 updates/s. CPU was selected. CUDA forward maximum logit discrepancy was2.2351742e-08 within frozen tensor tolerances. Benchmark parameters reset before each one-step fixture; these are not extra training trajectories.

Training: 1,456,000 grid+32,000 pilot=1,488,000 logical updates. Additional one-step optimizer calls:880 benchmark+42 sweep+18 unit tests=940. Interrupted replay updates:0. Maximum declared execution cap21600 seconds was not exceeded. The execution status gives measured training wall time; per-fit times include evaluation/IO and shared CPU contention.

Four worker processes used disjoint future seed blocks on the same CPU backend under a separate source-hashed execution extension. The original frozen trainer/model/config files were unchanged. GPU memory was not pooled. Full CUDA training/replay and cross-backend macro-F1 reproduction were not run. Environment records include Python/PyTorch/CUDA/cuDNN/package versions, GPU driver query, deterministic flags, TF32 off and CUBLAS workspace settings.

## 13. Failures, omissions and repairs

Nineteen distinct unit tests passed in each of two runs; zero test failures. Analytical gradients, finite differences, original-gate equivalence, masking/live support, permutation separation, serialization and exact-resume checks passed. The initial v2 reference was unavailable; no performance training occurred until the user supplied it and explicit choices were reconciled under v3.

An analysis-only JSON serialization defect was repaired losslessly by converting one NumPy boolean to a Python boolean. Training, endpoint calculations, bootstrap draws and CSV statistics were unchanged. Downstream report guards stopped; all failure evidence was retained. No scientific redesign occurred. Preserved exceptions/warnings and omission evidence are documented in REPAIR_AND_DEVIATION_LOG.md. C1 rank-target and G2 gradient-reference misses were reported without changing perturbation or initialization. C3 futility omitted70 fits. Extra execution/reporting scripts have separate snapshots and hashes; old research artifacts remain unchanged.

## 14. Scientific interpretation

**H1:** supported as a mathematical restriction and numerical symmetry diagnosis. Breaking symmetry removes exact rank-one restriction, but the second component remains initially weak and predictive benefit does not follow. **H2:** supported by the fixed-batch gradient sweep; reduced saturation strengthens gradients, while frozen−4.5 still misses the engineering reference. **H3:** master precision preserves stored updates, but C3 failed the declared forward-level pilot criterion. Storage precision alone was insufficient in those pilots.

**H4/H5:** unresolved. Shared weights may compensate attenuation, and availability may be inferable from zero-masked inputs. Null classification results cannot prove either explanation, particularly with remaining weak gradients and weak second-component strength. Positive code-switch sensitivity can originate partly from initialization and does not itself establish useful context information. No semantic, biological or associative-memory relationship has been demonstrated.

## 15. Limitations

This is development-only evidence on previously examined subjects and engineering seeds. Overlapping training sets limit bootstrap independence. The historical16-subject scaler was preserved and includes contributions from some subsequent CVvalidation subjects. No official test result is available. Fixed factor seed 30031 limits factor-noise generalization. C3 futility uses one seed/fold per architecture. The correction cells do not identify every interaction among saturation,symmetry and precision.

Only35 matched Gaussian shuffled null fits define each diagnostic quantile; Layer2 evidence remains operational and pair-specific. Probe sensitivity uses128 windows/fold. Per-factor stepwise update fractions are missing; initial/final displacement is not their reconstruction. Pooled-window curves differ from the equal-subject primary endpoint. Detailed qualifications are preserved in PILOT_SCOPE_AND_MEASUREMENT_LIMITS.md.

## 16. Recommendation regarding Phase 4

Do not interpret this study as demonstrating associative memory or a new architecture. A functioning contextual gate alone does not justify Phase 4. Any stronger-initialization or information-identification follow-up should use a separate frozen protocol and must not replace this study retrospectively. The present results answer the registered recovery comparison; broader mechanism and independent generalization questions remain open.

## 17. Exact reproducibility and artifact inventory

From E:\CoNN, set PYTHONDONTWRITEBYTECODE=1 and use `CoNN_Associative_Relationships\.venv_cuda23\Scripts\python.exe CoNN_Associative_Relationships\phase3b_context_diagnostics\scripts\run.py verify` to rescore saved predictions independently. `execute` is single-owner launch/resume; do not start a second owner. READMEcontains preflight,execute,verify,report and final-analysis commands.

Frozen protocol SHA256:`901e57c4d5dcc9bf63b05eb297c75948698db232eec15361248f8c791157e72c`. Frozen config SHA256:`044b0ad52720d3b7c234beec28d4bc416c212deca94b2b6fa657cb9a376310cf`. Frozen training-source-manifest SHA256:`96eff6c7314032940eca43356ab4092f748ce220d1143e9fcefc47f2468900d7`. There is no workspace Git repository; source SHA256 snapshots identify exact revisions.

Run inventory: config/protocol seals; original request reference; source and execution-extension snapshots; preflight and final preservation audits; environment/tests;455 grid+10 pilot checkpoint/prediction/step-log directories; raw and verified subject/class CSVs; paired/seed/subject effects; all code permutations and disabled-Cablations; attenuation snapshots/spectra; learning/gradient/precision curves; per-unit energy; release-vs-optimization and isolated-layer probes; CPU/CUDA charts; required research summaries and final delivery hashes. Earlier automatic research reports are preserved in this analysis extension. No missing fit is fabricated or interpolated.


## Forward-dtype rank sensitivity

The reference rank in section5 is computed from factors in float64. Actual forward C is float32; converting it to float64 does not remove its prior rounding. The additional saved-checkpoint diagnostic reports both ranks at the same tolerance. Tiny rounding-induced singular values are not additional independent learned factors. See FORWARD_RANK_SENSITIVITY.md and analysis/forward_vs_reference_rank.csv. The frozen endpoint, uncertainty analysis and Level2 computation are unchanged.


## Final integrity and loss check

All465completed pilot/grid final model states were checked finite, and each checkpoint contains3200actual recorded optimizer steps. Frozen training-source/config hashes remained unchanged. All455grid pre/post-release weight pairs were bitwise identical. LOSS_AND_RELEASE_DIAGNOSTICS.md and plots/train_validation_loss.png separate the measured gate-switch loss change from subsequent optimization. The final preservation audit passed.
