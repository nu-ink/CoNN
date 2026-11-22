# 04 — Architecture and code audit

Static review only. Repository tests were not re-executed. Checkpoints were not loaded.

## Implemented network

`CoNNMLP` (`models.py` lines 41–58) is a two-hidden-layer ReLU MLP.

| Piece | Synthetic pilot | HAR experiments |
| --- | --- | --- |
| Input | 100 | 561 engineered or 1152 temporal |
| Hidden | 32, 32 | 64, 64 |
| Output | 1 logit, BCE-with-logits | 6 logits, cross-entropy |
| Gated matrices | input→hidden, hidden→hidden | same |
| Ungated | both biases and the output head | same |
| Trainable count | 4,321 | 40,518 engineered; 78,342 temporal |

Count check for the synthetic net: `32*100+32 + 32*32+32 + 32+1 = 4321`. Every Experiment 1 condition uses that count. Gates are buffers computed each step, not parameters. A parameter-count explanation of D versus B′ is unavailable: the counts match, and 1B records identical initialization fingerprints across conditions within each seed (20/20 seeds, recalculated).

Initialization is uniform on `[-1/sqrt(fan_in), 1/sqrt(fan_in)]` from a generator seeded by `paired_seed(seed, "init")` (`models.py` lines 61–90). Optimizer is SGD with momentum 0, dampening 0, weight decay 0 (`models.py` lines 102–123).

## Forward and backward

The forward pass multiplies a detached gate into a copy of `W` and does not write the parameter (`models.py` lines 26–28). The comment forbids in-place `mul_` on the parameter. That pattern is present.

Loss for the synthetic task is `BCEWithLogits`. Accuracy treats a logit of exactly 0 as class 0 (`train.py` line 70: `logits > 0`). `DESIGN_DECISIONS.md` records this boundary. It affects only ties at zero.

Diagnostic gradients (`diagnostics.py` lines 14–36) are computed on a fixed minibatch, then the model gradients are cleared. They are not the optimizer step. The logged `dL/dW` is the attenuated parameter gradient. Plots of `dL/dW` shrinking on far bins are partly implied by multiplication with a small `G`. The code also logs `dL/dW_eff`. A claim that uses only `dL/dW` would be circular. The reports that pair the two quantities follow the brief.

## Condition mechanics

| ID | What the code does | What a comparison can isolate |
| --- | --- | --- |
| A | Gate of exact ones, or a dense bypass in the HAR path | Ordinary dense optimization |
| A′ | One scalar per layer, `sqrt(mean(G_D^2))` (`gates.py` lines 39–43) | Layer energy, not spatial pattern or per-row energy |
| B′ | Frozen within-row column permutation of D's gate (`gates.py` lines 31–36, 167–169) | Source assignment at matched per-row gate multisets |
| C | One frozen permutation of input-gate columns; hidden gate copied from D | Input-coordinate alignment, with hidden geometry intact |
| D | Task-line Gaussian, smoothstep, then exact ones | The full curriculum |
| E | Exact, then narrow, then wide, then exact | A different path with the same dense endpoint |
| S | `tau_start` forever | Local mask at train and test time |

The build brief's phrase "scrambling strong-connection destinations" (`CURSOR_BUILD_BRIEF.md` line 24) disagrees with a column permutation, which scrambles sources. `DESIGN_DECISIONS.md` lines 22–26 records the ambiguity and adopts the column permutation because that is what preserves each row's multiset. The code follows the design note. Reading the brief alone would describe a different control.

## Schedules and a redundant assignment

`phase23_core.py` lines 109–118 compute `tau` from interpolated `d50`, then overwrite `tau` with interpolation in `tau`. Because `d50 = tau * sqrt(2 log 2)` is linear, the two expressions are algebraically the same. Endpoint steps are then forced to the exact start and end. This is a dead first assignment, not a second schedule. Floating-point order can differ before the endpoint override. The comment says the second form matches Experiment 2.

CPU Gaussian tensors are cached by `tau` (`phase23_core.py` lines 119–124). Later conditions index that tensor; they do not write into it in the lines reviewed. An in-place consumer of a cached gate would corrupt later conditions. No such write was found in `build`.

## Context architecture

`ContextHARMLP` adds `U` and `V` for three context codes, rank 2, on both gated layers (`phase3_context.py` lines 25–50). The amendment's count is 78,342 base parameters plus 8,064 context parameters = 86,406. Context models match each other and exceed the masked baselines. A gain of a context model over a masked model therefore mixes capacity, optimization, and the factor `C`.

Phase 3B `Model.run` (`phase3b_context_diagnostics/src/models.py` lines 47–55) multiplies `gates[0]` without an extra detach. Those gates are schedule tensors, not parameters, so they still contribute no gate gradient. Context parameters remain differentiable except in `gaussian_frozen`, where `requires_grad` is false (line 45).

C3 stores factors in float64 and is dropped from the grid when a one-seed, one-fold pilot misses a forward-change threshold (`PHASE3B_FINAL_REPORT.md` section 6). The omission is recorded in `omitted_fits.csv`. A success report that ignored those rows would be misleading. The final report counts 455 grid fits plus 10 pilots and states that 70 C3 fits were omitted. That accounting is visible. It is still a reduced test of the precision-only cell.

## Release behavior

Pre-release and post-release checkpoints are specified to share weights and differ in the gate. Phase 3B reports that all 455 grid pairs at step 2240 were bitwise identical in the weights. This audit did not reload the tensors. The CSV and report statement is the evidence. If it is accurate, the immediate metric change at release is a gate intervention at fixed `W`, and later change includes 960 further updates. Those two quantities are separated in the 1C and Phase 2/3 tables.

Forcing `G=1` on a network trained at `d50=0.08` changes the function, because about 42% of the early gates are numerically negligible. The 1C audit's statement that dense inference on final S weights scored about 0.771 validation accuracy against a native 0.944 is a measurement of that dependence. It was not recomputed from checkpoints here. Class: **UNVERIFIABLE** by reload, **SUPPORTED** by the saved audit report.

## Defects and risks

| ID | Location | Issue | Class | Impact |
| --- | --- | --- | --- | --- |
| CODE-01 | `models.py` 26–28 | Gate is detached. Correct for a frozen schedule. Any future learned `tau` would receive no gradient until this line changes | CONFIRMED | High if someone treats `tau` as trained |
| CODE-02 | `CURSOR_BUILD_BRIEF.md` line 24 vs `gates.py` 167–169 | Brief says destinations are scrambled; code permutes columns (sources) | CONFIRMED wording conflict, resolved in `DESIGN_DECISIONS.md` | High if the brief is cited as the control definition |
| CODE-03 | `phase23_core.py` 109–118 | First `tau` expression is overwritten by an algebraically equal one | CONFIRMED | Low |
| CODE-04 | `phase23_core.py` 119–124 | Shared CPU gate cache | SUPPORTED risk, no observed in-place writer | Medium if later code mutates gates |
| CODE-05 | `CoNN_Associative_Relationships/README.md` lines 3–5 | Says no training and no context gate exist | CONFIRMED stale status | Medium for a reader who stops at the README |
| CODE-06 | `train.py` line 70 | Logit 0 maps to class 0 | CONFIRMED, documented | Low |
| CODE-07 | No Git repository | Snapshots are the revision system | CONFIRMED | Medium for external replication |
| CODE-08 | `.pt` checkpoints | Not loaded, by policy | UNVERIFIABLE weights | Does not affect CSV score checks |

No evidence was found that a failed job is rewritten as a successful accuracy. Phase 3B's futility path writes omitted rows instead of imputed scores. The Phase 2/3 report describes a Windows checkpoint-replace failure, a bitwise replay of the interrupted segment, and a resume. This audit did not re-verify that replay.

## Names

"Collapsible Neural Network" and the infographic describe expansion, pruning, and a return to a smaller net. The trained object is a fixed-width MLP with a prespecified nonnegative mask. After release, D's mask is identically one. The name describes a mechanism the training loop does not perform.
