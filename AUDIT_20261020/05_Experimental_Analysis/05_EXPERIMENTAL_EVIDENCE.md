# 05 — Experimental evidence

Numbers in the recalculation column come from `Supporting_Evidence/recalculated_statistics.json`, computed from the saved score CSV files. Published intervals that match to the printed precision are marked confirmed.

## Experiment 1 pilot

**Question.** Does task-aligned D beat row-shuffled B′ on final AND test accuracy?

**Data.** Synthetic 100-feature signals. Motif A in features `[8, 32)`, motif B in `[68, 92)`, length-6 templates, jitter inside the region, two distractors, noise 0.15. Splits 512/128/128. AND label is 1 only when both motifs are present. One generator stream builds train, then validation, then test. The same arrays are reused across conditions.

**Protocol.** Seven conditions, width 32/32, 400 SGD steps, shared learning rate 0.1, seeds 0–4. Test evaluated once at the end. The learning-rate grid in `configs/pilot.yaml` was not run.

**Result.** D minus B′ mean +0.021875, sample SD 0.011588, 95% t interval [0.007487, 0.036263], 5/5 seeds positive. Recalculation matches. Smoke accuracy 0.500 is a pipeline result.

**What it isolates.** B′ keeps each row's gate values and moves them across sources. A positive gap is consistent with source alignment mattering on this layout. It does not isolate alignment from higher-order topology. Dense A was already able to learn the task after the difficulty was adjusted on seed 0 (`DESIGN_DECISIONS.md` lines 50–62). The mechanism is not required to reach above-chance accuracy.

## Experiment 1B

**Question.** Does the pilot gap remain after a shared validation grid and new seeds?

**Selection.** Seeds 5–9, learning rates `{0.01, 0.05, 0.1}`, durations `{400, 800, 1600, 3200}`, no test labels. Per condition, pick the learning rate by mean final validation accuracy. Pick one duration by the macro-average of the seven selected conditions. Frozen result: 3200 steps; D and B′ both at learning rate 0.1; C at 0.05.

**Replication.** Seeds 100–119, 140 models, one test evaluation each. Initialization and data fingerprints agree across conditions within every seed.

**Primary result.** Mean D minus B′ = 0.0484375 exactly. Sample SD 0.03171934. Student t critical value for df=19 is 2.093024. Interval [0.03359239, 0.06328261]. Paired dz 1.527065. Signs: 19 positive, 1 negative (seed 110, delta −0.015625). Recalculation matches the published summary to the printed digits. The value is a mean of per-seed accuracies on 128-example tests, so the quantum of each accuracy is `1/128 = 0.0078125`. The mean of twenty such paired differences can be an integer multiple of `0.0078125/20 = 0.000390625`. `0.0484375 / 0.000390625 = 124`, so the reported mean sits on that lattice.

**Condition means, test accuracy.** A 0.864453, A′ 0.863281, B′ 0.862891, C 0.860156, D 0.911328, E 0.869922, S 0.937891. D minus S = −0.0265625, interval [−0.044293, −0.008832], 15 of 20 seeds favor S.

**Interpretation bound.** The primary contrast supports a repeatable AND alignment gap under this optimizer and this geometry. S's higher mean is a secondary result under a different terminal gate. It is evidence that exact dense release was not necessary for the best observed accuracy in this sample. Duration selection used all seven conditions. On the published rounded validation table, the seven-condition macro is highest at 3200 (0.8859). Dropping S, the macro is still highest at 3200 (0.8755 versus 0.8732 at 1600). S widened the margin and did not change the argmax of that rounded table.

## Experiment 1C

**Question.** Is partial widening better than full dense release, and does alignment interact with expansion?

**Primary.** AND, P (smooth d50 0.08→0.25, then hold) minus D (original release). Seeds 200–219, shared learning rate 0.1. Mean +0.018359, interval [−0.000181, +0.036900]. The interval includes zero. Recalculation matches. The prespecified primary comparison is unresolved.

**Secondary observations, same seeds.**

| Contrast | Mean | 95% t interval | Interval vs zero |
| --- | --- | --- | --- |
| AND D aligned minus shuffle | +0.025391 | [+0.011976, +0.038805] | excludes zero, positive |
| AND S minus D | −0.019141 | [−0.052578, +0.014296] | includes zero |
| AND fixed d50=0.25 mean accuracy | 0.948438 | descriptive mean | highest listed core mean |
| XOR D aligned minus shuffle | −0.025391 | [−0.045727, −0.005054] | excludes zero, negative |

The 1B alignment effect reappears on new AND seeds at about half the 1B magnitude (+2.54 points versus +4.84). The XOR effect has the opposite sign. S no longer beats D once the seeds and the shared learning rate change. Static-local superiority is seed- and protocol-sensitive.

W minus D, the same pre-release weights with a wide Gaussian held instead of exact ones, is +0.004297 with an interval that includes zero in the published table. Exact release is not a resolved final-accuracy benefit.

## Experiment 2

**Question.** Does engineered-feature D beat row-shuffled B on held-out subject macro-F1?

**Data.** Official UCI HAR archive, SHA-256 recorded in the report. Fit subjects 16, validation 5, test 9, split seed 104729. Coordinates for the primary input are file order. Temporal windows colocate channels in time. Normalization added by Experiment 2 uses fit subjects only. Vendor scaling of the 561 features remains upstream and unexplained.

**Selection.** Learning rates differ by condition. Engineered D uses 0.1 and B uses 0.05. Temporal D and B both use 0.01.

**Primary test result.** Engineered D minus B macro-F1 mean −0.00501038, SD 0.00789428, interval [−0.008705, −0.001316], dz −0.635, signs 3/0/17. Recalculation matches. D is also below dense A by 0.004822 macro-F1.

**Matched learning-rate check on tuning validation, five seeds, no test labels.** Engineered D minus B is negative at 0.01 (−0.00157), 0.05 (−0.00751), and 0.1 (−0.00373). The negative sign is present at every grid learning rate, including the two rates later selected. The test comparison still uses different selected rates, so it is a comparison of tuned procedures. The tuning table does not show a hidden positive alignment effect that selection then discarded.

**Temporal test.** D minus B macro-F1 +0.001861, interval [−0.001799, +0.005522]. Unresolved. At the shared selected rate 0.01, tuning-validation D minus B is −0.004742 with an interval that includes zero.

## Phase 2 and Phase 3 development

**Question.** On temporal HAR cross-validation, does the Gaussian schedule beat dense and row-shuffle, and does the context factor beat the same mask without the code?

**Design.** Seven folds, seeds 400–404, 3200 steps, validation subject-macro-F1. Official test subjects are not scored. Intervals are subject bootstraps after the learning rate is chosen on the same subjects. Training sets overlap. These are development intervals.

**Selected-LR seed means, macro-F1, recalculated from `selected_seed_scores.csv`.** Gaussian 0.890350, dense 0.884649, row-shuffle 0.878974, fixed d50=0.25 0.899641, fixed d50=0.08 0.893866. Gaussian minus row-shuffle seed-level mean +0.011375, and all five seed differences are positive. Fixed d50=0.25 minus Gaussian is +0.009291, all five seeds positive.

**Context.** Gaussian context minus Gaussian masked seed-level mean −0.000418. The report's same-weight `C=1` intervention changes macro-F1 by about −0.00003. Observed `C` stays inside approximately `[0.998823, 0.998826]`. The forward factor is effectively a constant near one. A classification comparison at that operating point cannot demonstrate contextual routing.

## Phase 3B

**Question.** Do symmetry breaking, a less saturated initialization, and float64 factor storage produce a useful context effect?

**Primary.** Gaussian C4 minus Gaussian masked subject-mean macro-F1 +0.00213398. Frozen bootstrap, 10,000 resamples, `numpy` Generator seed `7000031`, integer indexing, linear 2.5/97.5 percentiles: interval [0.00011566, 0.00444627]. Reproduced exactly from `subject_differences.csv`. The +0.005 criterion is not met.

**Robustness of "the interval excludes zero".** A Student-t interval on the same 21 subject means is [−0.000214, +0.004482] and includes zero. A Student-t interval on the five seed means is [−0.001500, +0.005768] and includes zero. Two of five seeds are negative. The positive lower bound is a property of the frozen subject bootstrap, not of every reasonable summary of the same table.

**Level 2.** Operational pair checks pass 49 of 175 eligible comparisons. The report states that initial random differences already create code sensitivity. Learned displacement of the normalized attenuation difference is on the order of `0.0003` to `0.0005`, while the initial difference is about 0.12 to 0.14. The final pattern is dominated by initialization.

**Shuffled-code secondary contrast.** C4 minus shuffled has bootstrap interval [−0.000605, +0.006912], which includes zero (`paired_contrasts.csv`). The primary gain is not separated from a shuffled-code control by that interval.

**C3.** Both architecture pilots fail the forward threshold. Seventy grid fits are omitted. Precision-only training on the full grid is **NOT TESTED**.

## Dataset suitability

The AND task can be solved by dense A to about 86–88% test accuracy. Motifs occupy known distant intervals, so a local coordinate prior matches the labeling rule. A shuffle of those coordinates is a real destruction of that prior. The experiment can detect an alignment effect. It cannot show that collapse is required, because dense and static-local models also perform well.

XOR uses the same geometry with a labeling rule that makes each motif marginally uninformative. Alignment then hurts. The prior is task-specific.

Engineered HAR does not contain the geometry the Gaussian assumes. A negative D minus B result bounds this file-order prior. It does not by itself bound every future coordinate system. Temporal coordinates are a weaker, explicit time prior. Experiment 2 does not show a resolved test-set gain. Phase 2's validation gain for the schedule over a shuffle coexists with a larger gain for a fixed medium width over the schedule.

A conventional dense model solves every classification task in this repository at a level close to the gated models. Distinguishing a collapse mechanism from mask-biased optimization requires a measurement of connectivity change at fixed weights and a comparison against fixed masks of equal terminal width. Experiment 1C already contains that design. Its primary result did not resolve a benefit of partial expansion over release, and its best observed AND mean was a fixed mask.

## Finding

**EXP-01.** The only contrast that is positive, same-learning-rate, and repeated on held-out synthetic seeds is AND aligned-minus-row-shuffle. That contrast reverses on XOR and on engineered HAR. Confidence: **CONFIRMED**.
