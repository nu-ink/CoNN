# CoNN Phase 3B v2 — Contextual Gate Failure Analysis and Controlled Recovery Experiment

> **Reference specification for VS Code Agent.** This reproduces the complete Phase 3B v2 agent specification supplied in the conversation. It is a historical design reference, not a protocol claimed to have been frozen or executed. Subsequent v3 amendments supersede conflicting v2 provisions. Do not hash/freeze a merged protocol until conflicts and underspecified choices are resolved explicitly.

## Mission

Implement and execute a separate, controlled Phase 3B experiment to determine why the Phase 3 contextual gate barely learned and whether correcting its implementation produces useful context-dependent behavior.

**Workspace:** `E:\CoNN`

**Existing project:** `E:\CoNN\CoNN_Associative_Relationships`

**New directory:** `E:\CoNN\CoNN_Associative_Relationships\phase3b_context_diagnostics`

Use the existing Experiment 2 temporal Human Activity Recognition (HAR) dataset.

Preserve all historical research, checkpoint files, protocols, and results.

Do not modify the original Gaussian gate, tau schedule, geometry, or established training procedure.

Do not implement Phase 4 associative memory or evaluate official HAR test subjects.

Execute the work in the following stages, producing a final technical and scientific report.

---

## Stage 1 — Inspect and reproduce the historical failure

Read:

- `PROTOCOL_v2.md`
- `PROTOCOL_PHASE23_v1.md`
- `README_PHASE23.md`
- `reports/PHASE_2_3_IMPLEMENTATION_AND_RESULTS.md`
- Existing Phase 3 contextual-gating implementation
- Phase 2/3 configurations, diagnostics, training code, and saved checkpoints

Locate the original completed Phase 2/3 run:

`results/runs/20261019T030442340025Z_phase23_v1`

Verify source hashes and data integrity before proceeding.

Historical findings:

- C initialized near 0.9988255.
- Maximum departure from initialization was approximately 2.5e-6.
- Context gradient norms were approximately 0.0001 of base gradient norms.
- Low-rank factors remained symmetric.
- Some U updates were rounded away in float32.
- Context-enabled models did not meaningfully outperform their matched masked controls.

Reproduce the C0 historical gate using the new harness.

For a deterministic checkpoint replay, require checkpoint-loaded predictions and diagnostic outputs to agree within established numerical tolerances. For any fresh training replay, compare measured development metrics with historical scores using a predeclared tolerance of 0.001 macro-F1; distinguish numerical equivalence from statistical reproducibility.

Do not silently change training settings to force agreement.

### Competing hypotheses

H1 — Symmetric initialization prevents the two low-rank components from learning distinct factors.

H2 — Saturated sigmoid factors and bilinear coupling produce weak effective gradients.

H3 — float32 parameter-update rounding prevents representable changes.

H4 — The chosen architecture makes context modulation functionally redundant, even when gradients are adequate.

H5 — The available sensor context provides little additional predictive information beyond the masked inputs themselves.

Describe predictions that would distinguish these hypotheses.

---

## Stage 2 — Mathematical diagnostics

The original Gaussian gate remains:

\[
G_{ij}(\tau)=\exp(-d_{ij}^{2}/(2\tau^{2}))
\]

Contextual modulation remains:

\[
W_{\mathrm{eff}}(c)=W\odot G(\tau)\odot C(c)
\]

The historical context gate is:

\[
C(c)=1-\frac{0.95}{2}\sigma(U_c)\sigma(V_c)^T
\]

with factorization rank 2.

Define attenuation:

\[
A(c)=1-C(c)
\]

Measure effective rank using A rather than C, because the all-ones contribution to C can obscure the factorization rank.

### Analytical gradient checks

For each factor component k:

\[
\frac{\partial A_{ij}}{\partial U_{ik}}=0.475\,\sigma'(U_{ik})\sigma(V_{jk})
\]

\[
\frac{\partial A_{ij}}{\partial V_{jk}}=0.475\,\sigma(U_{ik})\sigma'(V_{jk})
\]

At U = −6, V = 0, verify approximately:

- σ(−6) = 0.002473
- σ(0) = 0.5
- ∂A/∂U contribution ≈ 0.000586
- ∂A/∂V contribution ≈ 0.000294

These are local derivative magnitudes, not the complete loss gradients.

Compare analytical expressions, PyTorch autograd, and finite differences.

Use appropriate finite-difference step sizes and float64 reference calculations.

Stop if gradients do not agree within documented tolerances.

### Precision diagnostics

Distinguish:

1. Whether a gradient exists.
2. Whether an optimizer update changes the stored parameter.
3. Whether the updated parameter changes C.
4. Whether the resulting C change survives conversion into the forward-pass dtype.
5. Whether the changed C affects predictions.

Measure all five independently.

A successful float64 parameter update does not establish a meaningful float32 forward effect.

---

## Stage 3 — Freeze the experimental protocol

Before performance training, create and hash:

`phase3b_context_diagnostics/PROTOCOL_PHASE3B_v2.md`

Record the complete experimental design, dataset, splits, mathematical definitions, initialization seeds, optimizer settings, primary endpoint, ablations, evaluation rules, and compute cap.

The original Phase 2/3 protocol and results remain unchanged.

### Context representation

Use three externally supplied categorical context codes:

- 0 = Both accelerometer and gyroscope available
- 1 = Accelerometer available
- 2 = Gyroscope available

Use the existing three-entry categorical lookup for factor selection. Do not replace it with a newly learned sensor encoder.

Apply the same masking policy as Phase 3.

Retain contextual modulation at the same hidden layers and locations as the historical implementation. Record exact tensor dimensions.

### Fixed experimental cells

**C0:** Historical U = −6, V = 0, float32.

**C1:** Symmetry-breaking only. Preserve historical parameterization and optimizer, but initialize distinct low-rank columns using a deterministic generator with fixed seed 30031 and elementwise uniform perturbations in [−0.2, +0.2] around historical U and V.

**C2:** Less-saturated initialization only. Use U = −4.5, V = 0, float32. Do not break component symmetry.

**C3:** Precision only. Preserve historical U and V initialization but use float64 master context parameters and float64 context-optimizer updates. Convert modulation to model dtype at the declared forward boundary. Keep the base network float32.

**C4:** Combined changes. Use U = −4.5, V = 0, the same deterministic symmetry-breaking perturbation method, and float64 context master parameters.

For C1, require the initial mean of C to remain within 0.0005 of C0, and document any discrepancy before training.

For C2/C4, acknowledge that initial C is intentionally farther from 1 than C0. Do not claim identical initial forward computations.

Preserve the attenuation bounds and original Gaussian gate.

### Optimization

Use the established:

- Plain SGD
- Learning rate 0.1
- Batch size 128
- 3,200 optimizer updates
- Hidden dimensions 64/64
- Seeds 400–404
- Seven subject-grouped development folds

Use no weight decay, momentum, or gradient clipping unless the original frozen implementation requires them. Do not introduce a context-specific LR multiplier in this experiment.

The common LR is fixed for the diagnostic comparison. No Phase 3B validation-based LR search is permitted.

This avoids confounding numerical precision with an independently changed optimization rate.

The historical LR selection remains historical evidence, not a new search.

Do not select new initialization candidates using validation accuracy.

### Diagnostic decision rule

Conduct numerical tests and a small training pilot before the full development grid.

C4 is run as the predeclared combined condition, not selected because a particular earlier condition scored well.

If a condition fails gradient correctness, numerical stability, or checkpoint integrity, stop that condition and preserve evidence.

If C3 produces representable master-parameter updates but fails to change forward modulation beyond numerical tolerance in its pilot, classify precision-only intervention as functionally futile. Do not allocate further full-grid training to C3 unless required for a specifically documented diagnostic comparison.

Do not replace the failed cell with a new parameterization.

---

## Stage 4 — Required controls

Run C0–C4 for both Gaussian and dense architectures wherever required by the frozen comparison matrix.

Also include:

1. Gaussian masked, no context.
2. Dense masked, no context.
3. Gaussian with C4 parameters but one fixed context code during training and evaluation.
4. Gaussian C4 with training-time context codes shuffled independently of the true availability masks.

For shuffled-code training, use a deterministic permutation preserving overall code frequencies, with independent random streams from masks and examples. Use a separately defined context-code randomization rule at evaluation and report the resulting distribution shift.

For the constant-code control, retain the same contextual parameter tensors and their storage cost while assessing their effective use.

These controls help separate additional capacity from usable contextual information. The constant-code model is parameter-count matched at allocation but may have fewer actively trained parameters; report that limitation.

The required planned comparison matrix therefore contains:

- 10 architecture × gate-variant cells
- 2 noncontext masked controls
- 2 information/capacity controls

**Total: 14 training conditions.**

Do not introduce further conditions without a versioned protocol amendment.

---

## Stage 5 — Primary endpoint and success criteria

### Primary comparison

Gaussian C4 versus Gaussian masked without learned context.

Use identical sensor-masking curricula and subject folds.

### Primary endpoint

Mean subject-level validation macro-F1, averaging the three prescribed sensor-availability contexts equally, then aggregating according to the frozen paired analysis.

### Minimum effect of interest

+0.005 absolute macro-F1.

This is a practical research threshold, not an independently validated superiority margin.

### Statistical procedure

Use paired subject-clustered bootstrap intervals with 10,000 resamples, averaging over the five engineering seeds before subject resampling.

Report individual seed effects and subject-level differences.

Because seeds, folds, and development data have been examined previously, treat these intervals as descriptive, post-selection development uncertainty.

Do not use them for confirmatory significance claims.

The primary comparison is evaluated once under the frozen protocol.

All additional model comparisons, individual sensor contexts, accuracy metrics, and ablations are secondary or exploratory.

Do not apply uncorrected significance claims across multiple secondary comparisons. If formal tests are reported for a predeclared secondary family, apply Holm correction within that family; otherwise give effect sizes and intervals without significance claims.

### Three separate success levels

**Level 1 — Numerical learning:** context gradients and updates are finite, correct, and representable.

**Level 2 — Functional context differentiation:** learned context-specific attenuation matrices differ beyond fixed numerical tolerance and produce measurable same-weight output effects.

**Level 3 — Predictive benefit:** the primary paired macro-F1 improvement exceeds the frozen minimum effect of interest and the uncertainty analysis supports a practically meaningful advantage.

Record these independently.

Do not claim Level 3 automatically implies Level 2.

Do not claim the network learned semantic or biological relationships solely because contextual matrices differ.

---

## Stage 6 — Bounded execution

Use the existing CPU environment and isolated CUDA environment where available.

Benchmark each on representative corrected contextual models.

Select one execution backend for the entire performance grid based on measured throughput and verified numerical correctness.

Do not mix CPU and CUDA results within paired development comparisons.

GPU memory is not pooled across devices.

### Compute budget

Maximum planned full training:

14 conditions × 7 folds × 5 seeds = 490 fits.

490 × 3,200 = 1,568,000 logical optimizer updates.

Allow up to 10 additional bounded pilot fits, for a maximum of 500 fit trajectories and 1,600,000 logical optimizer updates, excluding documented interrupted-step replay overhead.

Before the full grid, benchmark representative fit runtime and estimate the complete wall-clock budget from measurements.

Set an explicit maximum wall-clock duration in the run manifest based on the available execution budget.

Do not assume the historical Phase 2/3 throughput will apply unchanged.

Execute:

1. Unit tests.
2. Analytical and numerical gradient checks.
3. One-batch updates.
4. Bounded pilot fits.
5. Pilot verification.
6. Predeclared development grid.
7. Independent result verification.

Preserve unfinished progress if execution is interrupted.

Never describe an unfinished grid as completed.

---

## Stage 7 — Context-learning measurements

For every completed condition, calculate:

- Context gradient norms
- Update representability
- Parameter-change fractions
- Initial and final attenuation matrices
- Effective rank and singular values of A = 1−C
- Pairwise context-matrix differences
- Context-dependent output changes
- Context-code permutation effects
- Modulation-disabled ablation effects
- Class and subject-level prediction metrics

Use float64 for reductions and diagnostic comparisons where appropriate.

### Functional interventions

Evaluate the same frozen hidden representation under all three context codes.

Keep weights and hidden inputs identical.

Measure logit RMS differences, classification margins, and prediction changes.

Clearly label these as same-representation interventions, not realistic sensor-availability predictions.

### Realistic masking

For ordinary validation evaluation, apply input masks corresponding to each external context.

Report the contexts separately and combined.

Do not compare full-sensor Phase 2 training directly to mixed-sensor Phase 3B training as though the data curriculum were equivalent.

---

## Stage 8 — Verification, visualizations, and results

Create the following plots from executed measurements:

1. Historical versus revised modulation
2. Initial and final attenuation matrices
3. Attenuation singular-value spectra
4. Gradient-norm trajectories
5. Parameter-update representability
6. float32 versus float64 forward differences
7. Pairwise external-context relationships
8. Same-hidden-state context interventions
9. Development macro-F1 by model
10. Sensor-specific performance
11. Primary paired effect and bootstrap interval
12. Capacity and shuffled-code control comparisons
13. CPU versus CUDA training throughput

Use a headless-compatible plotting backend.

Independently verify reported accuracy and F1 by loading saved predictions and recalculating confusion matrices, precision, recall, and macro-F1 in a verification script separate from the training implementation.

Check model checkpoint hashes, source snapshots, experiment manifest, actual optimizer-step counts, subject-group separation, and protocol hash.

Preserve all original results.

---

## Stage 9 — Explicit failure behavior

Stop the affected stage and preserve evidence when:

- Source or dataset hashes unexpectedly differ.
- An analytical/numerical gradient check fails.
- Subject leakage is detected.
- Context masking exposes unavailable sensor information.
- Checkpoint replay cannot reproduce the declared state.
- The experiment exceeds the recorded execution budget.
- The selected device produces nonfinite values or unsupported operations.

Do not silently repair scientific-design changes.

Implementation defects may be repaired with source snapshots, test evidence, and a recorded changelog.

For protocol changes, create an amendment and determine whether previous results remain comparable.

Never delete failed-run evidence merely because a later retry succeeds.

---

## Stage 10 — Final deliverables

Create the following files under `phase3b_context_diagnostics/`:

- `README.md`
- `PROTOCOL_PHASE3B_v2.md`
- `config/experiment.yaml`
- `src/` containing models, training, diagnostics, and evaluation
- `tests/` containing numerical, masking, and reproducibility tests
- `scripts/` containing launch, resume, verification, and reporting commands
- `results/` containing run-specific checkpoints, metrics, predictions, and plots
- `reports/PHASE3B_FINAL_REPORT.md`
- `reports/PHASE3B_EXECUTIVE_SUMMARY.md`
- `reports/REPAIR_AND_DEVIATION_LOG.md`

### Final report structure

1. Executive summary
2. Research question and frozen design
3. Historical Phase 3 comparison
4. Mathematical gradient findings
5. Symmetry and effective-rank findings
6. Numerical precision findings
7. Controlled intervention results
8. Context-specific functional differences
9. Classification and predictive results
10. Primary comparison with uncertainty
11. Capacity and information-control results
12. CPU/CUDA execution report
13. Failures and repairs
14. Scientific interpretation
15. Limitations and unresolved hypotheses
16. Recommendation regarding Phase 4 associative memory
17. Exact reproducibility commands and artifact inventory

### Final agent response

Summarize:

- Implementation status
- Exact number of successful fits
- Optimizer updates
- Tests passed, failed, and not run
- Selected CPU/GPU backend
- Historical versus revised context behavior
- Primary macro-F1 effect
- Whether Level 1 was achieved
- Whether Level 2 was achieved
- Whether Level 3 was achieved
- Whether any root-cause hypothesis was supported
- Whether results warrant further associative-memory research
- Report locations

**Scientific honesty requirements**

Never invent measurements.

Never equate a functioning contextual gate with demonstrated associative memory.

Never report a development-level result as independent confirmation.

Never claim successful training from a passing unit test.

Never reinterpret the historical Phase 2/3 scores as new results.

Never change the frozen protocol based on observed validation scores without explicitly versioning that change.

### Final instruction

Perform the repository inspection first.

Implement and execute Phase 3B within the specified isolated directory.

Work through all verification gates, bounded diagnostics, training, independent analysis, and reporting.

Do not start Phase 4.

Do not evaluate official HAR test subjects.

If the full experiment cannot complete within the verified execution budget, preserve the partial results and produce an accurate status report rather than claiming completion.
