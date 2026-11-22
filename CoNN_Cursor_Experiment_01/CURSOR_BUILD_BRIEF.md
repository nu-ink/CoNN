# CoNN Experiment 01 — Cursor Build Brief

## Mission
Implement a reproducible PyTorch experiment testing whether **structured local-to-global Gaussian weight gating** improves learning compared with controls that preserve gate magnitude and per-neuron energy but disrupt spatial alignment. This is a *training-curriculum* hypothesis, not a claim that a Gaussian kernel or a Hoberman mechanism is new.

**Scientific priority:** Do not optimize for CoNN winning. Preserve fair comparisons, reveal failures, and log enough diagnostics to explain results.

## Frozen decisions
- Language/framework: Python 3.11+; PyTorch, NumPy, pandas, matplotlib, pytest; CPU default, CUDA optional. No network calls during training.
- Deterministic synthetic data: 1-D signal, 100 features, two easily recognizable motifs in distant regions. Train/evaluate on disjoint sampled examples; split data once per seed and reuse it across all models. No information from evaluation data may define model coordinates.
- Pilot first: AND target (both motifs present); balanced labels via stratified generation; jitter motif positions within separated regions, controlled distractors and noise. Ensure neither a single motif nor a single feature trivially determines the label. Add XOR as a second, harder task, **not** the pilot's only task. Include a geometry-misaligned negative control.
- Architecture: explicit input and hidden 1-D coordinates in [0,1], set **before** training; hidden units fixed on a uniform grid. Two hidden layers with ReLU; dense final output head. Apply gates only on input→hidden and hidden→hidden weight matrices; biases ungated. Document coordinate conventions and distance shape. The synthetic data's spatial ordering is known, not learned from labels.
- Weights: `W_eff = W * G` (elementwise), `G_ij(t) = exp(-D_ij**2/(2*tau(t)**2))`. Use numerical safeguards for tiny tau; log actual zeros or near-zero gates. Gating happens in forward pass, not by destructively mutating parameters.
- Optimizer: plain SGD, momentum=0, weight_decay=0, no batch/layer normalization; fixed mini-batch ordering per paired seed. Scheduler changes by **optimizer step**, not epoch, and is recorded. Follow with an Adam sensitivity run only after the SGD pilot.
- Fairness: same architecture, trainable parameter count, base initial weights, dataset split, data order and total optimizer steps per paired seed across conditions; separate seeded randomness for mask shuffling and input coordinate permutations.
- Hyperparameter selection: predefine modest **equal** LR tuning budget for every model on validation data; never tune on test data. Initial smoke tests may use one shared LR, but cannot support claims of superiority.

## Seven conditions (stable identifiers)

A `dense`: G=1 throughout.

A_prime `scalar`: at each step every gated layer uses a uniform scalar `G=sqrt(C_D(t))`, where `C_D(t)=mean(G_D(t)^2)` **for that layer**. This matches layerwise gate energy, not per-row distribution; label conclusions accordingly.

B_prime `row_shuffle`: compute D's Gaussian gate and fix an independent **within-row permutation** of its columns at initialization. Apply the same frozen permutation to D's gate at every step. This preserves each row's gate multiset and sum of squared gates at every step, while scrambling strong-connection destinations. Do **not** redraw permutations each batch.

C `permuted_input`: randomly permute the input feature-to-coordinate mapping **once per seed** while keeping the coordinate multiset and all hidden coordinates unchanged. Its first-layer gate row multiset remains identical to D. For the hidden→hidden layer, either keep D's gate identical (preferred for isolating input alignment) or clearly label any additional hidden permutation as a separate ablation. Do not randomly redraw coordinates.

D `structured`: task-aligned input and hidden coordinates, scheduled local→global.

E `reverse`: task-aligned coordinates, deliberately global→local→dense with exact `G=1` in the final phase. **This is not a pure reversed monotone schedule**, so interpret D−E as a schedule-path ablation with common endpoint, not an isolated direction effect.

S `static_local`: constant early-stage small tau and hence locality throughout training. Its end is not dense; compare final predictive performance but explicitly disclose unequal final gate states. If a same-endpoint comparison is needed, add separately a static-local→dense release condition; do not conflate it with S.

## Schedule
- The schedule is fixed, deterministic and stepped on optimizer updates, not learned.
- Use a smooth monotonic increase of tau from `tau_start` to `tau_end` over the first 70% of steps for D/B_prime/C and use **exact G=1** for the final 30%.
- Make tau relative to coordinate units (all coordinates within [0,1]). Pick the initial `tau_start` to maintain nonzero near-neighbor connectivity; avoid almost-total disconnection. Record `tau_start`, `tau_end`, and `d50=1.17741*tau`.
- E begins with G=1, transitions to a local Gaussian state, and returns to exact G=1 by the 70% mark; explicitly implement and plot its three phases.
- Calculate `C(t)=mean(G^2)` separately for each gated layer. A_prime's scalar matches D on C(t), not E's trajectory.
- At 70% log measurements immediately **before and after** exact-gate release to expose any discontinuity.

## Outcomes and logging
**Pre-registered primary metric:** held-out test accuracy at a fixed optimizer-step budget (after one checkpoint is selected by validation **without inspecting the test set**; preferably report final-step test accuracy to avoid selecting a favorable checkpoint). The primary planned contrast is `D - B_prime` on AND. Report paired per-seed deltas, mean, SD, 95% CI with small-sample t critical, and raw outcomes. At five seeds mark all results *pilot/exploratory*, not statistically conclusive. Apply multiplicity correction if asserting significance for multiple comparisons.

Log per step or evaluation interval: train/validation loss and accuracy, tau and phase, layerwise C(t), per-row gate energy summaries, mean absolute W_eff, W and W_eff norm, fraction of gates below thresholds (e.g. 1e-6 and float32 zero), activation variance, and gradient diagnostics.

To avoid tautological 'far gradients grew' claims, record **both** `dL/dW` and `dL/dW_eff` across predeclared distance bins. Use `torch.autograd.grad` or `retain_grad` safely for W_eff; report data-gradient and gate-attenuated gradients separately. Sample diagnostic batches consistently, and do not confuse post-step values with pre-step values.

Causal diagnostic: at fixed checkpoints evaluate a copy of the model with **far connections temporarily suppressed** and report validation-loss delta, without changing training weights or optimizer state. Define 'far' by a fixed distance quantile derived from the base geometry, not gate values. A shuffled baseline's distance bins are **task-coordinate bins**, not its shuffled mask distances; document this.

Local representation diagnostic: evaluate linear probes trained on frozen hidden representations to predict motif A and motif B **using training examples only**, and evaluate probe accuracy on held-out validation. This may be a separate follow-up module if it slows the initial pilot; keep labels from the generator and avoid training probes on test data.

## Suggested repository structure
```
conn_experiment_01/
  README.md
  pyproject.toml
  configs/pilot.yaml
  src/conn_exp/__init__.py
  src/conn_exp/data.py
  src/conn_exp/geometry.py
  src/conn_exp/gates.py
  src/conn_exp/schedules.py
  src/conn_exp/models.py
  src/conn_exp/train.py
  src/conn_exp/diagnostics.py
  src/conn_exp/evaluation.py
  src/conn_exp/plots.py
  src/conn_exp/runner.py
  tests/test_data.py
  tests/test_gates.py
  tests/test_schedules.py
  tests/test_models.py
  tests/test_reproducibility.py
  results/.gitkeep
```

## Work tasks (execute in order)

### T01 — Scaffold and environment
- [ ] Create the above repository, `pyproject.toml`, type hints, CLI help, README and `configs/pilot.yaml`; select reasonable explicit defaults and seed list.
- [ ] CLI examples: `python -m conn_exp.runner --config configs/pilot.yaml --smoke` and `python -m conn_exp.runner --config configs/pilot.yaml --seeds 5` (after editable install).
- **Done when:** package imports, `pytest` runs, and smoke CLI writes a run directory with config + environment metadata.

### T02 — Synthetic task generator
- [ ] Implement reproducible AND plus optional XOR examples; controlled noise, distractors, motif jitter, label balance, motif metadata; separate train/validation/test.
- [ ] Include simple sanity baseline verifying learnability and that labels are not leaked in features or split logic.
- **Done when:** tests validate balance, locality ranges, split isolation and deterministic seeds; a quick dense sanity model can beat chance on AND.

### T03 — Geometry, gates and invariants
- [ ] Explicit input/hidden positions, distance matrices and masks; row-preserving shuffled mask; input-position permutation; scalar energy matched mask.
- [ ] Verify B'_row multiset/energy equals D at multiple tau values; verify C first-layer row multiset equals D while assignments differ.
- **Done when:** exact or floating-tolerance invariant tests pass, including finite values and gates within [0,1].

### T04 — Schedules and endpoints
- [ ] Implement D/B'/C local→global→exact dense, E global→local→exact dense, S static and A/A' policies. Boundary logs at 70%.
- **Done when:** schedule unit tests establish start/mid/end values and all intended endpoints. E's intermediate locality must be observable.

### T05 — Models and training loop
- [ ] Implement gated linear layers (do not in-place change W); baseline with same layer shapes; SGD loop and paired seed initialization/data orders.
- [ ] Validate optimizer-step control, checkpoint save/resume, floating-point robustness, CPU and optional CUDA.
- **Done when:** smoke experiment trains all seven conditions, produces non-NaN metrics and reproducible repeated runs.

### T06 — Scientific diagnostics
- [ ] Add gate energy, per-row stats, effective weight magnitude, activation variance, distance-binned pre-gate and post-gate gradients, far-connection ablation.
- [ ] Add motif linear probes if feasible; otherwise create a clearly marked follow-up and do not claim motif-learning order.
- **Done when:** every chart metric is backed by a CSV/Parquet column and assertions rule out mislabeled quantities.

### T07 — Runner and reporting
- [ ] Run paired seeds with fixed split/init/order within each seed, separate random mask RNG, long-format CSV and individual checkpoints.
- [ ] Plots: loss/accuracy vs steps, gate energy vs steps, d50/tau schedule, per-row energy, distance-gradient trajectories (pre/post gate), ablation loss delta, seed scatter for D−B'.
- [ ] Generate a report with raw per-seed scores, descriptive mean/SD, paired CI, and explicit pilot limitations; include test-set usage protocol and configuration hash.
- **Done when:** one command regenerates all plots and tables without manual notebook steps.

### T08 — Validation and next experiments
- [ ] Add XOR difficulty tier; misaligned or shuffled-input geometry task; train-data size sweep; LR tuning fairness; 20+ seeds only after pilot feasibility.
- [ ] Add Adam, LR warmup/annealed DropConnect baselines, potentially convolutional setting **after** primary protocol is functioning.
- **Done when:** controls and negative cases can falsify the geometry/curriculum hypothesis, not just support it.

## Interpretation rules
- D>B'_row does NOT perfectly isolate geometry: columns and higher-order mask topology differ, though row energy and value multisets are matched.
- Gradient-distance trends in `dL/dW` alone are partly guaranteed by gating. Pair with `dL/dW_eff`, causal ablation, and probes.
- E is an equal-endpoint *path* control, not a clean one-variable reversal; S intentionally retains locality at inference.
- Different initial effective capacity remains a concern when comparing with A; A' helps but does not match per-row forward statistics.
- Never claim an architectural invention, statistical significance from five seeds, or an effect when uncertainty spans zero.

## Deliverable requirements
1. Working code, not pseudocode.
2. Reproducibility tests and invariants before lengthy runs.
3. Run commands for Windows PowerShell and Linux/macOS shells.
4. A brief `RESULTS.md` that explicitly distinguishes measured findings from expectations.
5. At each task, report which files changed, commands executed, tests passed, and unresolved issues; proceed task-by-task, do not silently change the scientific design.
