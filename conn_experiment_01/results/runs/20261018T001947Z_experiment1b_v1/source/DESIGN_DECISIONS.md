# Design decisions recorded before they are used

These are the choices the build brief left open, plus clarifications where a phrase could be implemented in more than one way. They are not permission to change a frozen rule. The runner rejects a different optimizer, a different primary control, test-set checkpoint selection, and a silent CUDA fallback.

## Not deviations

- **Python 3.11 venv.** `python` on PATH is 3.10.1. The brief requires 3.11+. The project venv is created with `py -3.11`.
- **CPU is the executed device.** CUDA is not selected unless `device: cuda` and CUDA is actually available. There is no silent fallback to CPU.
- **Shared initial learning rate.** Smoke and the five-seed pilot use one SGD learning rate for every condition. An equal validation grid is written in `configs/pilot.yaml` and is not executed. The report states that this cannot support a superiority claim. Running that grid is a later task.
- **Adam is not in this run.** The brief puts Adam after the SGD pilot.
- **Primary metric.** Final-step held-out test accuracy after a fixed optimizer-step budget. Validation is logged during training. The test split is evaluated once, after the last update. No checkpoint is chosen by test performance.
- **Primary contrast.** `D (structured) − B_prime (row_shuffle)` on AND. The code rejects a config that renames or replaces that control.

## Coordinate and gate layout

- Coordinates live in `[0, 1]`. Index `i` among `n > 1` positions is `i / (n - 1)`. A single unit would sit at `0.5`.
- Distance is the absolute difference on that line. `d50 = tau * sqrt(2 ln 2)` is the distance at which the Gaussian gate is one half. The brief's recorder constant `1.17741` is also stored. The gate itself uses `sqrt(2 ln 2)`, not the rounded constant.
- `tau_start` is set from `d50_start = 0.08` (several nearest neighbors stay connected). `tau_end` is set from `d50_end = 1` (global relative to the unit interval). Exact `G = 1` still begins at 70% of steps; `tau_end` is not a substitute for that exact phase.
- Gate matrices match `nn.Linear.weight`: shape `[n_destination, n_source]`. Rows are destinations. Biases are not gated.
- `W_eff = W * G` is computed in the forward pass. `W` is not overwritten.

## B' is a frozen within-row column permutation

The brief requires an independent within-row permutation of columns, preserving each row's gate multiset and sum of squared gates, and it forbids replacing that with a global shuffle or redrawing it.

That permutation scrambles **sources** attached to each destination. The same sentence also says "destinations." The row-multiset sentence and the words "within-row permutation of its columns" are the invariant that is tested. A row permutation would break it. A single shared column permutation is condition C's input remap, not B'.

Masks are drawn once from the mask stream and reused every step.

## Condition C

Input features keep their data columns. Their coordinates are permuted once per seed. Hidden coordinates and the hidden-to-hidden gate stay identical to D. This is the brief's preferred way to isolate input alignment. Diagnostic distance bins still use the unpermuted task coordinates for every condition, including B' and C, so a shuffled mask does not define the bins.

## Schedule path for E

E is not a monotone reverse of D. Over the first 70% of steps it has three equal-length phases:

1. exact `G = 1`
2. Gaussian `tau` moving from `tau_end` down to `tau_start`
3. Gaussian `tau` moving from `tau_start` back to `tau_end`

At the 70% index, and for the last 30% of steps, `G` is exact ones. The equal split is a choice inside the required three-phase path. Leaving phase 1 creates a jump from exact ones to a wide Gaussian. That jump is labeled `E_leave_exact`. The 70% jump is logged twice at the **same** weight state: the post-update row of the last non-dense step, and the pre-update row of the first exact-dense step.

D, B', C, and A' share one smoothstep (`u^2 * (3 - 2u)`) from `tau_start` to `tau_end` on steps `[0, release)`, then exact ones. A' takes `sqrt(mean(G_D^2))` per layer from that schedule, including the exact phase, and does not follow E.

S holds `tau_start` for every step, including the final step. It is not released to dense.

`release_step = floor(0.7 * total_steps + 0.5)`.

## Task difficulty, set from a dense-only check

The first draft used noise 0.25, distractor amplitude 0.75, and learning rate 0.05. Condition A, and no other condition, reached validation accuracy 0.586 after 400 steps on seed 0. That is barely above chance, so the motifs were not easily recognizable to the frozen MLP.

Before any other condition was trained, the same dense model was rerun on seed 0:

| Noise | Distractor amplitude | Learning rate | Validation accuracy |
| --- | --- | --- | --- |
| 0.15 | 0.5 | 0.05 | 0.648 |
| 0.15 | 0.5 | 0.1 | 0.828 |
| 0.10 | 0.4 | 0.1 | 0.867 |

The pilot uses noise 0.15, distractor amplitude 0.5, and shared learning rate 0.1. On that draw, the best single-feature threshold accuracy was 0.609 and the motif-A indicator agreed with the AND label on 0.832 of training rows. The learning rate is one value for every condition. It was not chosen by comparing conditions, and the predefined grid is still not executed.

## Data

- 100 features. Motif A occupies indices `[8, 32)`. Motif B occupies `[68, 92)`. The gap is 36 features.
- Templates are fixed, length 6, with both signs. Jitter is uniform inside the region. Two weaker random distractors and Gaussian noise are added.
- AND is balanced by taking half the examples as both-motifs and splitting the rest across A-only, B-only, and neither. XOR uses four cells as evenly as possible and is not the pilot task.
- For AND, absence of either motif implies label 0 by definition. The non-triviality checks are that presence of one motif does not imply label 1, that all four cells exist, and that no single feature is a perfect classifier. XOR is the task on which each motif is marginally uninformative.
- One NumPy stream generates train, then validation, then test. The resulting arrays are reused for every condition. Coordinates never read labels or the test split.
- Splits are not rejected against each other, because rejection would let the test set influence training examples.

## Paired randomness

Each paired seed feeds four streams. The offsets below are stream labels, not separate experimental seeds, and they do not depend on condition:

| Stream | Offset |
| --- | --- |
| data | 0 |
| init | 1000003 |
| order | 2000003 |
| mask | 3000003 |

Initialization is kaiming-uniform style with bound `1/sqrt(fan_in)`, matching the usual `nn.Linear` default (`a = sqrt(5)`), drawn from the init generator. The minibatch at step `t` is a pure function of the order permutation and `t`. It is not redrawn inside the step, and conditions do not consume a shared global RNG.

## Optimizer and numerics

- `torch.optim.SGD`, momentum 0, dampening 0, weight decay 0, nesterov false, foreach false, fused false.
- `foreach=false` is recorded so the update is the simple SGD loop. It is not a change of momentum or penalty.
- Loss is `BCEWithLogits`. Accuracy treats a logit of 0 as class 0.
- Runtime flags recorded in each run: one Torch thread, cuDNN deterministic, TF32 off when those switches exist.
- Tau below `1e-6` is floored and flagged. The planned schedule never enters that regime. The floor exists so a tiny tau cannot emit NaN.

## Diagnostics

- Logged train and validation losses at measurement points are full-split means, not the minibatch. `batch_loss` is the minibatch objective that SGD stepped on, and only on post-step rows.
- `dL/dW` is the gate-attenuated parameter gradient. `dL/dW_eff` is the data gradient on the effective weight. Rows that store gradients are pre-step only.
- The diagnostic batch is the first minibatch of the paired order, reused at every diagnostic step.
- Far connections are those with base-geometry distance at or above the linear quantile 0.75 of that layer's base distance matrix. Ablation is a second forward. It does not write weights or optimizer state. `val_loss_delta = suppressed − original`.
- Motif probes are ridge linear classifiers fit on training hidden states and scored on validation. They are not fit on test labels. They are not a pre-registered endpoint.

## What a finished pilot may say

Five seeds are exploratory. The report states a paired mean, sample SD, and 95% interval with a Student-t critical value, and it does not call that interval a confirmatory test. If the interval contains 0, no effect is claimed. D beating B' would still not be a pure geometry isolation: column structure and higher-order mask topology differ. A' matches layer energy, not the per-row gate distribution. E is a path control with a common dense endpoint. S does not end dense.
