# Phase 2/3 development amendment v1

Authorization: user request “Implement Phases 2 and 3”, received 2026-10-18.
This separate amendment accepts verified Phase 0/1 and authorizes bounded development
training only. PROTOCOL_v2.md and its MANIFEST remain unchanged. No Phase 4 memory,
retrieval claim, official-test model evaluation, or final-evaluation authorization.

## Inherited definitions

Temporal UCI HAR: exact Experiment 2 prepared arrays, frozen original scaler,
1152 time-major inputs; hidden 64/64 ReLU; six-class dense head; original uniform
initialization and plain SGD, zero momentum/decay; 128 rows/update; 3200 updates;
evaluation every 160 updates. LR grid [.01,.05,.1], seeds 400–404 and all seven
subject folds exactly as frozen in v2. Restore official training row order;
official test subjects must never be loaded by the training/evaluation runner.
Known CV scaler leakage remains disclosed; no fold-refit sensitivity study here.

New unambiguous conditions and source mapping:

| New name | Phase | Source |
|---|---|---|
| P2_dense | 2 | Experiment 2 A, dense bypass |
| P2_gaussian | 2 | Experiment 2 D, original smoothstep then exact dense |
| P2_rowshuffle | 2 | Experiment 2 B, frozen independent within-row permutations |
| P2_coordperm | 2 | Experiment 2 C, input coordinate permutation, native hidden geometry |
| P2_fixed08 | 2 | Experiment 2 S08, fixed d50=.08 |
| P2_fixed25 | 2 | Experiment 2 S25, fixed d50=.25 |
| P3_dense_masked | 3 | A with same external sensor-mask exposure as context conditions |
| P3_gaussian_masked | 3 | D with same external sensor-mask exposure as context conditions |
| P3_dense_context | 3 | A plus bounded context modulation below |
| P3_gaussian_context | 3 | D plus same bounded context modulation |

Gaussian arithmetic, geometry, detach behavior, tau floor and release remain v2.
At step s<2240, u=s/2239 and tau=tau_start+(tau_end-tau_start)u²(3−2u).
Step indices 2240–3199 use exact Gaussian gates of ones for D/B/C.
The context factor remains active after Gaussian release and is NOT detached.
Output weights are always dense. Rowshuffle preserves each row's gate multiset.

## Random streams and pairing

Original Experiment 2 init/order/mask offsets 1000003/2000003/3000003 are kept.
Fold training order uses seed+2000003+10007*fold, independently of masks/init.
Initialization and masks for a seed match across folds, conditions and LRs.
Context RNG seed+5000003+10007*fold creates consecutive independently shuffled
cycles [both,accelerometer,gyroscope]; one homogeneous context per update.
Thus exposure is balanced within one update, independent of features/labels.
All four Phase 3 conditions use identical context/order streams. Phase 2 uses both.
Execution ordering uses an independent seed; never consumes the above streams.

## Context definition and exact parameterization

Externally supplied integer codes: 0 both, 1 accelerometer only, 2 gyroscope only.
Accelerometer includes body_acc x/y/z AND total_acc x/y/z; gyroscope includes
body_gyro x/y/z. Body acceleration is derived, not another independent sensor.
Unavailable channels are replaced by zero AFTER frozen scaling, at every time;
there is no unmasked bypass. Baselines get the same masked values, without the code.
Realistic performance always pairs a code with its matching input mask.

For each hidden connection layer, learned U[c,destination,rank] and
V[c,source,rank], rank r=2, define

    C(c) = 1 - (0.95/r) sigmoid(U[c]) @ sigmoid(V[c]).T
    W_eff(c) = W * detach(G) * C(c)

The attenuation deficit has rank at most two; C includes the rank-one all-ones
matrix. Bounds are .05<=C<=1 with strict positivity even at sigmoid saturation.
U starts at -6, V at 0, giving C=1-.475*sigmoid(-6), approximately .9988255.
This explicitly replaces v2's exact-one context initialization with the user's
authorized near-one initialization: a differentiable bounded smooth function at
its exact maximum cannot have nonzero derivatives. No parameter/rank/bound search.
All contexts start identically. The factor cannot amplify/override G. Only one
shared matrix per layer/context batch is built; no per-sample hypernetwork tensor.
Base parameters 78342; contextual models add 8064, total 86406. Context models are
capacity-matched with each other, NOT with the smaller baselines. Context-vs-base
gains alone cannot identify capacity causally. Report same-weight C=1 interventions
and fixed-input context switches separately from realistic missing-sensor metrics.

## Evaluation and selection

Every condition gets 7 folds × 5 seeds × 3 LRs × 3200 updates: 105 fits and
336000 optimizer steps; ten conditions give 1050 fits/3360000 updates. First run
all 42 Phase 2 Dense/Gaussian fits for seed 400 (7×3×2), inspect, then add Phase 3.
Those fits count toward the grid, not an extra tuning budget. CPU/GPU smoke and
throughput fixtures are engineering diagnostics, excluded from selection.
Choose LR separately per condition using final validation subject-macro-F1 averaged
equally across 21 subjects and five seeds. For Phase 3, average each subject's
three context-specific F1s equally. Ties: lower mean validation cross-entropy,
then lower LR. Never choose checkpoints or conditions using official test data.
Evaluate final checkpoint only for selection, not best observed curve point.

Save train/validation curves, class recalls/confusions, subject scores, all seeds,
gradient norms (base and context), Gaussian row energies, effective-weight and
context summaries, update/job runtimes and GPU memory. Final validation predictions
identify vendor row and subject, enabling independently rescored summaries.
Primary development contrast P2_gaussian−P2_dense, required geometry comparison
P2_gaussian−P2_rowshuffle. Secondary P2_gaussian−P2_coordperm/fixed widths;
P3_dense_context−P3_dense_masked; P3_gaussian_context−P3_gaussian_masked;
P3_gaussian_context−P3_dense_context; factorial interaction of those context gains.
Report mean paired seed differences/SDs and 95% percentile subject bootstrap
(10000 resamples, RNG 230319) after averaging seeds, plus paired subject dz.
CV training overlaps and selection reuse make these descriptive post-selection
development intervals, NOT confirmatory population intervals. Five engineering
seeds do not satisfy the original success criterion. No hypothesis redesign.

## Execution, checkpointing and compute boundary

Original CPU environment is read-only. An isolated .venv_cuda23 may install official
CUDA PyTorch wheels, with no system CUDA toolkit. Driver/hardware and tensor/matmul/
backprop/memory/CPU transfer must be measured. Benchmark CPU then GPU sequentially,
warm up, synchronize GPU timing, use float32, TF32 off, deterministic operations.
Choose the faster measured backend without considering validation accuracy.
Independent CPU workers (one Torch thread) or one process per explicit GPU are
allowed; no DDP/sharding. Timing under parallel contention is labelled separately.
Bound this execution to the finite 1050-fit grid and at most six hours of grid
wall time; if interrupted/time-limited, save clean resumable state and report
unfinished cells. Never mark incomplete grids as negative findings.

Checkpoint step means updates completed; gate for evaluation is next-step gate,
capped at 3199 at completion. Save initial, pre-release (2240 updates with gate
2239), immediate post-release (same weights, gate 2240), final and resumable latest
at each 160-update boundary. Resume restores model, optimizer, next-step schedule,
Python/NumPy/Torch CPU/CUDA RNG, batch/context plans, histories and immutable job
hashes. No stochastic layers or LR scheduler; schedule state is explicit step.
Atomic file replacement is allowed only for current run's latest checkpoint;
historical results and completed fits are never overwritten.

Freeze amendment bytes/config hash before any new model training. Snapshot source
before each stage, preserving earlier snapshots; log implementation corrections.
Verify original artifacts before and after. Use Agg plotting. No final test loading
or final model-selection-to-test evaluation until a separately authorized amendment.
