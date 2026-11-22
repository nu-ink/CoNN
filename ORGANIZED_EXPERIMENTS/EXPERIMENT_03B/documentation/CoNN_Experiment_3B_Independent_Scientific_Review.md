# Independent Scientific Review: Collapsible Neural Network (CoNN), Experiment 3B

**Prepared:** 20 October 2026  
**Scope:** Eleven supplied Markdown research documents, including the Phase 3B final report, executive summary, five diagnostic/interpretive supplements, scope note, protocol reconciliation, prospective v3.1 amendment, and repair/deviation log.  
**Status:** Independent document-based critical assessment, **not** a reproduction of code, trained checkpoints, CSVs, manifest hashes, or raw data.  
**Intended audience:** External scientific or technical review committee.

## 1. Executive verdict

**Experiment 3B does not satisfy its predeclared practical predictive-benefit criterion.** Its primary development-set comparison, Gaussian C4 minus Gaussian masked, is **+0.00213398 subject-averaged macro-F1** (about **0.213 percentage points**), with a *descriptive* 95% subject-bootstrap interval **[+0.00011566, +0.00444627]**. The prespecified practical bar was **+0.005 together with a positive lower interval bound**. Although the descriptive interval is wholly positive, the mean misses the required magnitude. The result should be described as a small, exploratory, positive development-set difference, not a successful confirmatory result. [PHASE3B_FINAL_REPORT.md §§1,10; MEASURED_CONTRASTS_AND_IDENTIFICATION_LIMITS.md]

The study **does support** a more constrained mechanistic conclusion: the initial factorization suffered an exact rank-one restriction when its component columns were identical; perturbing the components removes that strict symmetry; reducing sigmoid saturation increases context-gradient strength; and master-parameter precision alone does not guarantee sufficiently large live forward gate changes. Gaussian C4 met the frozen Level 1 numerical-learning criterion in **35/35** completed fits. Its contextual Level 2 operational tests passed **49/175** eligible layer/pair comparisons, showing detectable but inconsistent functional differentiation. These observations do **not** establish useful learned associative memory, a geometric generalization advantage, or an independently replicated gain. [PHASE3B_FINAL_REPORT.md §§4–8,14; FORWARD_RANK_SENSITIVITY.md]

**Review recommendation:** Preserve Phase 3B, label it accurately, and authorize a *new*, inexpensive, independently frozen Experiment 3D only after non-training mathematical/forward/gradient sanity checks. Experiment 3D should directly distinguish (i) whether context is necessary for predicting the label, (ii) whether the Gaussian attenuation pathway is capable of context-conditioned computation, and (iii) whether *learning and deploying that pathway* adds anything beyond ordinary contextual conditioning.

## 2. Evidence reviewed and limits of this review

The supplied materials are:

1. `PHASE3B_FINAL_REPORT.md` — primary design, results, measurements, claims, limitations.
2. `PHASE3B_EXECUTIVE_SUMMARY.md` — high-level interpretation.
3. `MEASURED_CONTRASTS_AND_IDENTIFICATION_LIMITS.md` — comparisons and attribution limits.
4. `GRADIENT_INITIALIZATION_SWEEP.md` — seven initializations, fixed-batch gradient diagnostics.
5. `FORWARD_RANK_SENSITIVITY.md` — reference versus actual-forward attenuation rank.
6. `LOSS_AND_RELEASE_DIAGNOSTICS.md` — paired pre/post-release and subsequent training loss.
7. `HISTORICAL_PRECISION_ESTIMATE.md` — descriptive uncertainty/power approximations.
8. `PILOT_SCOPE_AND_MEASUREMENT_LIMITS.md` — measurement limitations.
9. `PROTOCOL_RECONCILIATION_20261020.md` — original v2/v3 conflict reconciliation.
10. `PROTOCOL_AMENDMENT_V3_1_20261020.md` — prospective A1–A3 clarification.
11. `REPAIR_AND_DEVIATION_LOG.md` — operational provenance, deviations and reporting repairs.

These are **reports of an experiment**, not the complete reproducibility bundle. This review can scrutinize reported arithmetic, consistency, interpretation, controls and experimental design. It cannot independently certify code correctness, the claimed 5,762 file-manifest checks, all 1,050 historical artifacts, the 465 saved trajectories, raw predictions, class-wise confusion calculations, SHA-256 evidence, checkpoint identity, or empirical bootstrap resampling without the underlying artifacts. Statements such as “verified” below mean **reported as verified within the submitted documents** unless explicitly qualified.

## 3. Study design and provenance

The Phase 3B research question was whether symmetry breaking, reduced sigmoid saturation, and master-parameter precision could repair a context-dependent attenuation mechanism that previously displayed negligible change, and whether the repaired mechanism yielded useful external-context behavior. The reported matrix has 15 conditions; Gaussian and dense variants, mask/constant/shuffled/frozen controls, and C0–C4 mechanisms (C3 subject to pilot futility). The frozen settings included SGD learning rate 0.1, batch size 128, 3,200 optimizer steps, two width-64 hidden layers, five base seeds (400–404), seven historical folds, and the predeclared practical endpoint. [PHASE3B_FINAL_REPORT.md §§2,11]

The reported completed workload was **455 grid fits + 10 pilots**, **1,488,000 optimizer updates**; 70 C3 grid fits were omitted under a predeclared pilot rule. The arithmetic is internally coherent: (455 × 3,200) + (10 × 3,200) = 1,488,000. The report states that no official HAR test subjects were evaluated. [PHASE3B_FINAL_REPORT.md §§1,12]

An important chronology distinction must remain visible: the root v3 execution was completed and preserved; a subsequent v2/v3 reconciliation and **v3.1 amendment** clarified A1–A3 *prospectively after the outcomes were known*. Those clarifications cannot be treated as preregistration of the already-completed run or as retrospective alteration of its Level 2 claims. The amendment itself states that no new fitting or reassessment occurred. [PROTOCOL_RECONCILIATION_20261020.md; PROTOCOL_AMENDMENT_V3_1_20261020.md]

The deviation log reports separate source snapshots and hashes for scheduling/reporting fixes, preservation of original training files, and raw contrast CSV identity after serialization repairs. This is positive provenance documentation. Nonetheless, a committee should request and execute the published verifier and compare archived hashes; the documents alone are not a substitute for that audit. [REPAIR_AND_DEVIATION_LOG.md; PHASE3B_FINAL_REPORT.md §17]

## 4. Findings supported by the submitted evidence

### 4.1 Mathematical symmetry problem is well identified

The model uses the attenuation factorization

\[
C = 1 - 0.475\,\sigma(U)\sigma(V)^\top.
\]

When the columns of the component factors are identical, the attenuation matrix \(A=1-C\) is effectively an outer product and has algebraic rank at most one, regardless of an allocation nominally intended for two components. Symmetry-breaking perturbations removed this **exact** restriction, but the second singular component remained small: C4 initial \(\sigma_2/\sigma_1\) ranged approximately **0.002886–0.003714**, below the prespecified **0.01** reference. The report does not establish that the second component became meaningfully useful. [PHASE3B_FINAL_REPORT.md §5]

The forward-rank sensitivity supplement adds an essential caution: matrices rounded in float32 can produce tiny additional singular values and numerical ranks as large as 64 under the extremely strict reference tolerance. Those rounding artifacts **must not** be described as additional independent learned factors. The correct interpretation is the rank constraint of the unrounded factorization, not a raw count of float32 SVD singular values. [FORWARD_RANK_SENSITIVITY.md]

### 4.2 Saturated initialization suppresses context learning

The fixed-batch gradient sweep reports original G0 \(U=-6\) context/base gradient ratios from **1.07e−5 to 3.11e−5**. The frozen G2/C4 setting \(U=-4.5\) gave ratios **4.72e−5 to 1.38e−4**, failing the diagnostic \(10^{-3}\) reference in all six fixtures. Stronger initialization \(U=-1\) reached that reference in five of six fixtures, but this was diagnostic only and was **not** selected for performance training. The sweep is evidence of a gradient mechanism, not evidence that the stronger setting would improve generalization. [GRADIENT_INITIALIZATION_SWEEP.md; PHASE3B_FINAL_REPORT.md §4]

The mathematical gradient-check claims are unusually concrete: analytic contractions reportedly matched autograd to **1.39e−17** in a float64 fixture and finite differences passed stated tolerances. That supports local derivative implementation in tested fixtures, not the quality of end-to-end generalization. [PHASE3B_FINAL_REPORT.md §4]

### 4.3 Precision improvements were necessary for some changes, but not sufficient alone

A float64 master-parameter implementation allows small optimizer updates to be retained. Nevertheless, the precision-only C3 pilot reportedly produced maximum live float32 forward \(\Delta C\) around **8.34e−7** in Gaussian and **4.77e−7** in dense, under the predeclared **1e−6** threshold. Its 70 grid fits were therefore omitted. C2/C4 pilots crossed that threshold, with Gaussian forward maxima approximately **1.65e−5 and 1.67e−5**, respectively. These statements are specific to the sampled checkpoints and pilot configurations, not all seeds or every transient update. [PHASE3B_FINAL_REPORT.md §6; PILOT_SCOPE_AND_MEASUREMENT_LIMITS.md]

### 4.4 Functional sensitivity is detectable but incompletely attributed

Gaussian C4 passed Level 1 numerical learning in **35/35** full fits. Level 2 operational checks passed **49/175** eligible layer/context-pair evaluations. Isolated layer interventions and same-weight code interventions reportedly moved logits; the fixed-h1 C4 logit RMS difference was approximately **0.00797**. But initially random different context factors already generated gate differences, and Level 2 null testing used a modest **35-fit** pooled shuffled-code reference. These observations do not establish a uniform learned context mechanism across layers, sensor contrasts or future samples. The input-to-hidden accelerometer-versus-gyroscope overlap comparison is undefined because the true live supports do not intersect. [PHASE3B_FINAL_REPORT.md §8; MEASURED_CONTRASTS_AND_IDENTIFICATION_LIMITS.md]

The prospective v3.1 protocol improves future attribution: pooled Gaussian-only 95th-percentile nulls by layer/pair; layer-specific live forward change at least 1e−6 for one compared code; isolated IH intervention with HH=0 and HH intervention with cached h1; and sampled pilot-max futility separately per architecture. These are **future protocol definitions**, not proof that the already-completed 3B result met them. [PROTOCOL_AMENDMENT_V3_1_20261020.md]

### 4.5 Primary predictive result is small and below the frozen bar

| Comparison | Mean subject macro-F1 difference | Descriptive 95% interval | Scientific interpretation |
|---|---:|---|---|
| Gaussian C4 − Gaussian masked (**primary**) | +0.00213398 | [+0.00011566, +0.00444627] | Small positive development signal; primary +0.005 bar not met |
| Gaussian C4 − historical Gaussian C0 | +0.00255189 | [−0.00044630, +0.00598871] | Exploratory; interval crosses zero |
| Gaussian C4 − Gaussian frozen C4 | +0.00266374 | [+0.00018687, +0.00530247] | Descriptive increment beyond frozen initial attenuation; nonconfirmatory |
| Gaussian C4 − Gaussian constant-code | +0.00332958 | [+0.00019395, +0.00702091] | Descriptive positive contrast, but unequal active lookup exposure |
| Gaussian C4 − Gaussian shuffled-code | +0.00274569 | [−0.00060541, +0.00691192] | Uncertain; fails to cleanly isolate true-code information |
| Dense C4 − dense masked | −0.00102141 | [−0.00289158, +0.00050925] | No demonstrated positive dense benefit |

Sources: `PHASE3B_FINAL_REPORT.md` §§10–11 and `MEASURED_CONTRASTS_AND_IDENTIFICATION_LIMITS.md`.

Reported mean subject macro-F1 values for Gaussian C4 and Gaussian masked were **0.81827489** and **0.81614090**. The seed-level primary differences were approximately **+0.00527, +0.00234, −0.00019, +0.00466, −0.00142**. Thus only three of five base-seed averages were positive; this pattern cautions against overstatements of reliability. No claim of zero effect is warranted either. [PHASE3B_FINAL_REPORT.md §§7,10]

### 4.6 Release schedule has an observable immediate cost

Pre/post release was evaluated using bitwise-identical weights at step 2,240. For Gaussian C4, mean pooled-window accuracy reportedly declined from **0.81017 to 0.80640**, an immediate **−0.00377**, and the corresponding pooled loss increased from **0.65619 to 0.68869**, an immediate **+0.03249**. Later optimization improved accuracy and further raised loss versus the post-release checkpoint in the reported window aggregate. These patterns separate a same-weight gate-switch effect from subsequent weight adaptation but do not independently establish that gating or release is beneficial. [PHASE3B_FINAL_REPORT.md §9; LOSS_AND_RELEASE_DIAGNOSTICS.md]

## 5. Major threats to scientific interpretation

**A. Development-data reuse and independence.** All predictive outcomes are from development subjects rather than the untouched official test set. The 21 subject clusters, overlapping cross-validation training sets, five engineered seeds, and historical selection constrain uncertainty interpretation. The report also acknowledges a historical scaler fitted to 16 subjects with some validation-subject contribution: a possible leakage mechanism that prevents treating the current results as clean independent generalization. That preprocessing issue should be repaired *prospectively* for any new real-data study, not rewritten into Phase 3B. [PILOT_SCOPE_AND_MEASUREMENT_LIMITS.md; PHASE3B_FINAL_REPORT.md §§3,15]

**B. Bootstrap intervals are descriptive, not confirmatory.** A 10,000-draw paired subject bootstrap of 21 clusters after averaging seeds does not make five seeds independent datasets or remove leakage, model-development selection, fold dependence, or unknown external-population differences. A positive descriptive lower bound cannot be relabeled a confirmatory rejection of the null, particularly across a family of examined comparisons. [HISTORICAL_PRECISION_ESTIMATE.md; PHASE3B_FINAL_REPORT.md §10]

**C. The mechanism's learned contribution is not cleanly identified.** Three different quantities must be separated: (1) variation due to random initial context factors; (2) change of those factors over training; (3) out-of-sample performance causally dependent on the **learned gate changes**. Passing a live-output intervention with different contexts demonstrates sensitivity, but not necessarily (2) or (3). The frozen/constant/shuffled controls are informative but do not fully isolate information, representational capacity, active-row exposure, or optimization. [MEASURED_CONTRASTS_AND_IDENTIFICATION_LIMITS.md]

**D. External context may be redundant in HAR.** If zero masking reveals which sensors are missing, a flexible network can infer availability without reading the context code. The supplied documents explicitly leave H4 (base-weight/architecture compensation) and H5 (limited additional context information) unresolved. This is a plausible explanation, **not an established finding**. An additional empirical mutual-information or inferability test on the original inputs has not been provided here. [PHASE3B_FINAL_REPORT.md §14]

**E. The rank-two repair is weak.** The prescribed perturbation falls below its own 0.01 second-component target, and factor noise is fixed across the five base seeds. Therefore reported seed spread does not assess robustness to independent factor initializations. [PILOT_SCOPE_AND_MEASUREMENT_LIMITS.md; PHASE3B_FINAL_REPORT.md §§5,15]

**F. Architecture-versus-regularizer ambiguity.** The Gaussian model may benefit from its broader geometry, training schedule, effective regularization or optimization trajectory rather than context-conditioned learning itself. The positive C4-versus-masked contrast cannot isolate these alternatives. A clean new study needs capacity-aware controls and a frozen-gate ablation that tests whether *trained* attenuation produces additional context-dependent prediction changes. [PHASE3B_FINAL_REPORT.md §§11,14]

**G. Checkpoint and compute provenance is strong on paper, not independently inspected here.** The detailed count, hash, repair and numerical-gradient documentation improves auditability. But no external reviewer should call it independently reproduced without running the verifier, reading raw predictions, checking evaluation grouping, and inspecting source/data manifests. [REPAIR_AND_DEVIATION_LOG.md]

## 6. What is demonstrated, what is not, and what remains open

| Proposition | Current judgment |
|---|---|
| Identical component columns mathematically restrict attenuation rank | **Supported mathematically** by stated parameterization; implementation fixtures reported |
| Prescribed symmetry-breaking removes exact rank-one restriction | **Supported**, but weak second component |
| Less-saturated initialization increases diagnostic context gradients | **Supported within fixed-batch sweep** |
| Precision-only C3 failed frozen live-change threshold | **Supported for recorded two pilots**, not universal |
| Gaussian C4 produced stored/live numerical attenuation changes | **Supported within reported 35 grid fits** |
| Different context codes can affect fixed-weight outputs | **Supported on recorded probes**, initial differentiation confounds learning attribution |
| Context-specific learned functional effects generalize uniformly | **Not established** (49/175 operational checks) |
| C4 meets the primary practical improvement target | **No** (+0.00213 < +0.005) |
| C4 has a small positive development-set effect | **Descriptively supported**, inference limited |
| C4 improves reliably on independent subjects/datasets | **Not established** (no official-test assessment) |
| Context is redundant because zero masks reveal it | **Plausible but untested H5** |
| Gaussian geometry enables a unique learned associative-memory mechanism | **Not established** |
| v3.1 retrospectively fixes or validates 3B Level 2 analyses | **No**; it is a prospective amendment |

## 7. Experiment 3D: proposed research question and falsifiable hypotheses

**Primary research question:** When a context code changes the correct response for an identical, fully observed input, can learned Gaussian attenuation implement context-dependent mappings **through the gate itself**, rather than merely through a conventional context-input pathway or fortuitous random gates?

**H3D-1, necessity:** The identical inputs with mutually conflicting labels under balanced contexts should force a context-blind deterministic classifier to the predetermined chance limit (binary 50%). An ordinary context-aware model should learn to high held-out accuracy; otherwise the task/pipeline is defective.

**H3D-2, computational feasibility:** The exact implemented gate architecture must be able to represent the task when context acts *only* through its Gaussian attenuation factors, using shared base weights and no alternate context bypass. Verify with a small analytic/overfit fixture before any performance claim. If the architecture is provably nonexpressive on the chosen task, its failure is not evidence of a broken optimizer; choose a mathematically feasible target task **before freezing**.

**H3D-3, learnability:** Under frozen initialization and optimizer settings, learned-gate CoNN should learn the held-out context-dependent mapping across fresh paired seeds and independently varied factor seeds, with useful update magnitudes at the actual forward dtype.

**H3D-4, causal reliance:** Replacing a correctly matched test context with a wrong one, or replacing the final gate with initial/frozen/constant gates **while keeping all trained base weights fixed**, should produce a prespecified deterioration or task-predicted label reversal. The key causal contrast is trained gate versus the same model's pre-training/frozen/incorrect-context gate under identical weights and inputs; avoid treating mere initial context sensitivity as proof of learning.

**H3D-5, incremental value:** If CoNN succeeds, compare its performance, robustness and efficiency with matched dense contextual conditioning and matched random/frozen gate baselines. No novelty or superiority claim is justified by learning the synthetic task alone.

## 8. Suggested synthetic task (minimum viable design)

Generate balanced input samples \(x=(x_1,x_2)\) from a distribution independent of context \(c\in\{0,1\}\). Construct a deterministic context-dependent rule:

\[
 b(x)=\mathbb{1}[x_1x_2>0],\qquad y=b(x)\oplus c.
\]

Because the **same** sampled \(x\) is evaluated under both codes, \(y(x,0)\ne y(x,1)\) exactly (apart from the measure-zero boundary); context cannot be inferred from input. Use independently generated train, validation and test base inputs, with both contexts paired per base input; ensure all splitting happens by **base input**, never splitting two context versions of the same input across train and test. Fix input distributions and record deterministic generation hashes. Do not expose labels, index, mask flags, sorting information, or augmentation seeds that leak context.

**Important expressivity caution:** The XOR-like sign-based map may itself challenge a narrow architecture. Before freezing, conduct deterministic oracle/unit checks that the implemented gated model can represent both context-conditioned rules. An alternative predeclared simpler linearly separable base rule \(b(x)=\mathbb{1}[x_1>0]\) with context label inversion may isolate conditioning from nonlinear representation, provided gate architecture can realize the inversion. The final protocol should freeze **one primary task**, not choose whichever gives a positive result later. A secondary harder task may be preregistered without redefining primary success.

**Models/ablation cells** (use matched or transparently reported parameter counts and identical base-input splits):

1. **Dense without context**: intentional 50% task-information floor.
2. **Dense with explicit context**: positive feasibility benchmark; expected to succeed.
3. **CoNN gate-only context**: original mechanism with context entering exclusively through the attenuation lookup (no direct concatenated code, per-context classifier heads, or context-dependent normalization/statistics).
4. **CoNN fixed/random gate**: retain initialized distinct code gates but do not optimize gate parameters; distinguish random context feature conditioning from learned gating.
5. **CoNN constant gate/code**: no usable context; should remain at the context-blind information floor.
6. **CoNN shuffled-code training**: shuffle training codes independent of the true paired label rule while preserving frequency; evaluate prespecified matching/mismatching policies, with train/test distributions explicit.
7. **CoNN trained-gate ablations**: same trained weights with gates reset to initial, frozen reference, swapped, constant or disabled as defined in the protocol; assess both predictions and margins/logits.

Keep context balanced for every input. For a deterministic model with identical input and no context, maximum expected balanced accuracy is 50% over the paired evaluation by construction. Report accuracy **per base-input pair**, including the proportion for which both context predictions are simultaneously correct; this ensures “95% overall” cannot conceal failure in one context. For balanced binary pairs, chance-level accuracy is 50%; always report per-code and paired success.

## 9. Pre-training sanity gates and stopping logic

**Gate 0 — specification/audit:** Freeze code snapshot, config, independent train/validation/test generation, pair-group splits, randomization streams, architecture diagram showing the **only** context path, resource/time budget, sample count, all endpoints/thresholds and stopping rules. Include checked mathematical gradients and live support definitions.

**Gate 1 — forward dependence (zero optimizer updates):** For identical fixed \(x\), compare \(C(x,c=0)\) and \(C(x,c=1)\), effective weights, logits and class decisions. Check both reference float64 and *actual* forward dtype; differentiate random initial code sensitivity from trained behavior. Test the same input across contexts without masks. If the context channel or Gaussian multiplier is disconnected, stop as implementation failure.

**Gate 2 — gradient/expressivity (tiny diagnostic fixtures only):** Confirm nonzero, stable finite gate gradients and finite-difference agreement; verify the selected task is representable by the exact implemented gate. Compare at least the historical \(U=-4.5\) reference and a *prospectively selected* less-saturated candidate based on gradient diagnostics only, never validation accuracy. A failed expressivity check necessitates redesign/re-freeze; a failed gradient/forward check necessitates numerical redesign/re-freeze. Neither is a 3D negative performance result.

**Gate 3 — trained feasibility, fixed seeds and budget:** Train all frozen model cells on fresh paired base seeds, with **independently varied factor initialization** across seeds, matching optimizer steps and stopping policy. Do not search seeds or select the best split/model using test outcomes. Prespecify validation-only checkpoint choice or final-step evaluation. Hold final test untouched until all cells and rules are frozen.

**Gate 4 — mechanism and decision:** First establish a strong dense+context positive control and context-blind/constant/shuffled negative controls. Then require both successful CoNN context-conditioned prediction and a causal trained-gate reliance effect. If CoNN cannot learn while dense can, diagnose separately expressivity, attenuation, gradient flow and optimization; it is a negative result for the **frozen implementation**, not a mathematical disproof of all collapsible networks. If CoNN learns but gate-reset ablations do not impair predictions, declare predictive feasibility **without evidence of learned gate reliance**. If it learns and relies on trained gates, declare synthetic mechanistic feasibility only; do not claim practical generalization or biological association.

## 10. Example preregistered thresholds to debate and freeze

These numbers are **reviewer recommendations, not existing Phase 3B protocol commitments**. Before execution, power/feasibility-check the architecture and freeze precise criteria:

- Dense-context positive control: >=95% held-out balanced accuracy, >=90% paired-both-correct, per fresh seed or prespecified proportion of seeds.
- Context-blind and constant-code controls: within a prespecified practical tolerance of 50%, with paired evaluation used to diagnose any deviation and distribution imbalance.
- Gate-only CoNN: >=95% held-out balanced accuracy **and** >=90% paired-both-correct, across a prespecified high proportion of independently initialized seeds (suggest at least 9/10), *not* just the best seed.
- Gate reliance: a substantial preregistered **paired** held-out accuracy loss (e.g. >=20 percentage points) or context-predicted reversal under wrong-code/reset intervention, with paired-seed uncertainty; distinguish gate-specific causal ablation from out-of-distribution artifact.
- Numerical criterion: live forward attenuation change and output sensitivity above measured numeric noise in each tested layer; threshold calibrated during untrained fixtures and frozen before training, not merely taken from old HAR magnitude.
- Secondary efficiency claims: use matched computations and report train/inference time, capacity, calibration and generalization, with no superiority claim without a separately powered contrast.

A mechanism that produces correct code-swapped outputs **from randomly initialized fixed gates** may already be exploiting a context-specific random feature map; a trained-gate increment must still be shown to justify the claim of *learned* relationships. Conversely, a gate-reset ablation can introduce distribution shift; compare with trained frozen/random counterparts and report both the same-weight intervention and trained-model endpoint contrasts.

## 11. Statistical and reproducibility improvements for 3D

Predefine a **single primary endpoint** (e.g. mean paired-both-correct on fresh base-input test groups), one primary contrast and one hierarchical success rule. Treat base inputs as paired groups; do not count the two context versions as independent observations. Aggregate by seed and input group appropriately; report confidence intervals that respect data pairing and seed variability rather than treating every synthetic sample as an independent replication of training. If factor initializations are independently sampled, that is a genuine source of robustness information absent from 3B. Report *all* attempts, convergence failures, task IDs, environment versions, optimizer updates, and planned exclusions.

Separate a small **diagnostic feasibility set** (for unit and oracle checks) from the frozen **evaluation test generator**. A deterministic held-out synthetic generator does not substitute for real-world external validity, but it eliminates the HAR-specific input-mask ambiguity. Lock labels and data generation before model comparison. Store source/data/config hashes, checkpoints, per-code confusion matrices, gate-state snapshots, before/after intervention logits, and trained vs initial gate displacements. Make a one-command verification runner capable of recomputing endpoints from saved predictions independently of the trainer.

## 12. Conditions before moving to a real dataset

Progress to Experiment 3E or a real-world Phase 4 **only** if all of the following hold in a frozen test:

1. Task design makes context *necessary* rather than visually or statistically inferable from other inputs.
2. An ordinary context-aware dense benchmark learns the mapping.
3. Gate-only CoNN robustly learns the mapping with independently varied factor seeds.
4. Same-weight gate interventions establish learned gating as a causally necessary path, beyond fixed random lookup effects.
5. Reproducibility artifacts and paired estimates withstand outside verification.

Then choose a real multi-task or regime-switching problem with experimentally controlled context: identical or deliberately overlapping covariates can imply different targets under a supplied regime code, and the regime is not recoverable from censoring/missingness. Freeze its splits and preprocessing *within each training fold* and reserve a genuinely untouched final evaluation. Compare to ordinary conditioning and matched-compute alternatives. Success on synthetic data alone is **necessary diagnostic evidence, not proof of competitive utility**.

## 13. Committee-facing questions and requested evidence

1. Can the authors make the full exact frozen v3 code/data/snapshot/verifier inventory accessible, and can a reviewer recompute the subject-level macro-F1 primary contrast independently?
2. Can they show that the historical scaler leakage is contained to the stated development setup, and specify a leakage-free prospective pipeline?
3. Are the v3.1 changes clearly quarantined from every completed-v3 inference and report heading?
4. Can they demonstrate functional effects caused by **learned displacement**, not just random initial per-code gates, using same-weight initial/final and fixed/random trained baselines?
5. How sensitive are conclusions to independent factor seed, stronger non-saturated gate initialization, and more substantial second singular component? (Do **not** tune 3B retrospectively.)
6. Can a diagnostic classify sensor-availability context from masked HAR input? Its success or failure should be reported separately from CoNN training.
7. Is the 3D task *expressible* by the exact gate-only architecture? Which mathematically testable property supports that assertion?
8. How are raw 3D primary endpoints, seed failures and stopping/selection rules fixed before tests are viewed?

## 14. Final review decision

**For Experiment 3B:** Accept as a carefully documented **development-stage negative decision on the frozen practical-benefit criterion**, accompanied by useful diagnostic evidence for symmetry, gradient saturation, forward precision and limited gate sensitivity. Do **not** present it as an independently confirmed predictive win, an associative-memory demonstration, or a general failure of the larger CoNN concept.

**For Experiment 3D:** Conditional go: authorize one low-cost, frozen, context-essential synthetic feasibility experiment preceded by forward/gradient/expressivity checks. Stop at a negative or ambiguous result; do not initiate broad tuning or new HAR variations. Advance to real data only if *both* the predictive and learned-gate causal criteria succeed reproducibly.

**Reviewer confidence:** High in the *logical interpretation and internal documentary consistency* of the reported findings; **not an independent verification** of trained artifacts, numeric calculations or reproducibility because the underlying executable research bundle was not among these 11 documents.

---

### Source index

All findings and quotations are sourced from the 11 files supplied on 20 October 2026, named in Section 2. No external literature search was needed or used for this bounded document-based review. Proposed Experiment 3D hypotheses, tasks, thresholds and judgments are reviewer recommendations, not experimental observations.
