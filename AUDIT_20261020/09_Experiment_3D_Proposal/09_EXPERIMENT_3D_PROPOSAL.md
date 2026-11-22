# 09 — Experiment 3D proposal

This proposal is not implemented and was not run.

## 1. Primary hypothesis

On the existing synthetic AND generator, a scheduled change in Gaussian bandwidth produces a higher final test accuracy than every constant-bandwidth control, and the final weights remain sensitive to that terminal bandwidth when the weights are held fixed.

Both clauses are required. An accuracy gain with a null same-weight swap means the schedule acted as an optimization path and left a dense, or otherwise interchangeable, predictor. A same-weight swap without an accuracy gain over the best fixed mask means the forward mask matters and the schedule is not the useful choice.

## 2. Mechanism under test

The unresolved mechanism is the **change** in `G` during training. Experiment 1B already shows that placing large gate values on the true neighbors beats shuffling those values. Experiment 1C did not show that moving the bandwidth, or releasing it, beats a well-chosen fixed bandwidth. Phase 3 did not show a useful learned `C`.

3D does not retest associative memory, HAR, or a new formula for `tau`.

## 3. Dataset

Reuse the frozen AND generator from `configs/pilot.yaml`: 100 features, motifs in `[8,32)` and `[68,92)`, noise 0.15, distractor amplitude 0.5, splits 512/128/128. Do not retune noise or learning rate after seeing 3D outcomes. XOR is a secondary negative-control task using the same generator and the same frozen hyperparameters, scored only after the AND decision rule is applied. It is not used to pick a model.

## 4. Baselines

All arms share width 32/32, 4,321 parameters, plain SGD, momentum 0, weight decay 0, 3200 updates, batch 32, and one learning rate chosen only from AND validation on a separate tuning seed block.

| Arm | Gate |
| --- | --- |
| Dense | exact ones |
| Fixed local | constant `d50=0.08` |
| Fixed medium | constant `d50=0.25` |
| Fixed wide | constant `d50=1` |
| Schedule | smoothstep `d50` from 0.08 to 0.25 over the first 70% of steps, then **hold** 0.25. No exact-ones release |
| Shuffle | the schedule arm's gate with one frozen within-row column permutation |

The schedule's terminal width equals the fixed-medium arm, so a final accuracy gap is not an endpoint-width gap. This is the comparison 1C's P versus F already approximated. 3D makes it primary and adds the fixed-weight measurement as a co-primary clause.

## 5. Negative controls

- Shuffle arm: same gate multiset per row, wrong sources.
- Dense and fixed-wide arms: if they match the schedule, a local prior is unnecessary.
- XOR, same arms, no retuning: the AND prior is expected to fail or reverse, as in 1C. A method that wins on XOR by the same margin would indicate a generic optimization effect.
- Scalar energy match of the fixed-medium gate, one extra arm if compute remains after the six above: a single scale per layer. It separates spatial pattern from global step-size scaling. It is secondary.

## 6. Mechanism measurement

At the final checkpoint, copy the weights and evaluate validation accuracy under three gates without any further update:

1. the arm's native terminal gate
2. exact ones
3. the fixed-`d50=0.08` gate

The co-primary mechanistic statistic is native minus exact-ones validation accuracy on the schedule arm, paired by seed. The same three evaluations are recorded for every arm so the schedule is not singled out after seeing the result.

Report the fraction of gates below `1e-6` and mean `G^2` for the native gate. Do not infer collapse from `tau` alone.

## 7. Success criterion

All three, on AND replication seeds, two-sided paired t intervals:

1. Schedule minus the better of fixed-local and fixed-medium has mean test accuracy greater than +0.03, and the interval lies entirely above +0.01.
2. Schedule minus shuffle has an interval entirely above zero.
3. On the schedule arm, the absolute same-weight change between the native gate and exact ones has a mean validation-accuracy magnitude of at least 0.02, and the interval for that signed change excludes zero.

The winning fixed arm must be designated from the tuning block before replication seeds are drawn.

## 8. Failure or futility criterion

Stop and record a negative result if any one of these holds on the replication seeds:

- The better fixed-width arm has mean test accuracy greater than or equal to the schedule.
- The schedule-minus-shuffle interval contains zero or is negative.
- The schedule arm's same-weight native-versus-ones validation change has absolute mean below 0.01.

A negative result means this bandwidth schedule does not implement a functional connectivity change beyond a constant mask on the only task where alignment has been reliable. It does not ban every future mask.

## 9. Statistical design

Use 24 paired replication seeds, new integers that overlap none of 0–4, 5–9, 10–14, 100–119, or 200–219. Tuning uses 5 other fresh seeds and the existing learning-rate grid `{0.01, 0.05, 0.1}`. One learning rate is shared by every arm. Selection maximizes macro validation accuracy across the six AND arms. Ties break to lower validation loss, then lower learning rate.

Primary family: the three success clauses. If clause 1 fails, clauses 2 and 3 are descriptive. Do not add HAR, context codes, or rank claims to this experiment.

Assumed paired SD of 0.04 from the 1B D-minus-S sample SD (0.038). A true 3-point effect then has a paired-t power near 90% at n=24 and two-sided alpha 0.05. If the true effect resembles 1C's unresolved P-minus-D mean of 1.8 points, this design is likely to hit the failure rule. That is an acceptable outcome.

Each test split has 128 rows. Accuracy is discrete at `1/128`. The +0.03 margin is about 3.8 test examples per seed, larger than one example of noise.

## 10. Compute

One hundred forty-four replication fits and ninety tuning fits, each 3200 updates of a 4,321-parameter MLP on 512 rows. Experiment 1B ran a larger grid of the same model on CPU. Expect on the order of a few CPU-hours, not a GPU study. This estimate is extrapolated from the size of 1B and was not benchmarked in the audit. No official dataset download is required.

## 11. Confounds to lock before training

- Shared learning rate, shared step budget, shared initialization and batch order within a seed.
- Terminal `d50` of the schedule equals fixed medium.
- Shuffle permutation drawn once from the mask stream.
- Test labels unread until replication checkpoints are frozen.
- No checkpoint picked by validation. Score the final update only.
- Same-weight probes use validation, not test.
- Write the protocol hash before the first replication fit.

## 12. What each result would mean

| Outcome | Establishes |
| --- | --- |
| All three success clauses | On AND, changing bandwidth to a held medium width beats the best matching constant width, the placement of the mass matters, and the final function still uses the narrow-or-medium mask. This is the first result in this line that would tie accuracy to a retained connectivity change |
| Accuracy win, null same-weight swap | The schedule changed optimization and left weights that no longer depend on the mask. The deployed model is not a collapsible network |
| Fixed medium wins or ties | A constant spatial prior explains the AND results. Scheduling is unnecessary. This is the outcome most consistent with 1B and 1C |
| Shuffle matches the schedule | The AND alignment result failed to replicate under the held-medium schedule. The 1B finding would be procedure-specific |
| XOR produces the same ranking as AND | The gain is a generic optimizer interaction. The geometric story would be unsupported |
| Same-weight swap is large and accuracy still favors a fixed mask | Forward connectivity matters, and the particular schedule is the wrong trajectory |

Do not redefine success after seeing which arm won. Do not interpret a fixed-mask victory as a failure of the Gaussian formula. The formula can be correct while the schedule is unnecessary.
