# 08 — Hypothesis assessment

Labels: **CONFIRMED**, **SUPPORTED**, **UNCERTAIN**, **CONTRADICTED**, **NOT TESTED**, **UNVERIFIABLE**.

## Question A — What has CoNN demonstrated?

On the synthetic AND task, a prespecified Gaussian mask aligned to the feature axis beats a within-row permutation of that mask.

| Sample | Mean aligned minus shuffle | Interval excludes zero |
| --- | --- | --- |
| Pilot, 5 seeds, 400 steps | +0.021875 | yes |
| 1B, 20 seeds, 3200 steps, matched learning rate 0.1 | +0.0484375 | yes |
| 1C, 20 new seeds, secondary contrast | +0.025391 | yes |

The implementation matches `G = exp(-d^2/(2 tau^2))`, `W_eff = W⊙G`, and the detached gradient identity. Phase 3B shows that identical factor columns force rank at most 1, that moving the sigmoid center from −6 to −4.5 increases context-gradient scale, and that float64 storage without that center shift failed the authors' forward-change pilot.

## Question B — What has CoNN failed to demonstrate?

- A benefit of exact dense release over staying local or over a fixed medium width. 1B S exceeds D. 1C primary P minus D includes zero. Fixed d50=0.25 has the highest 1C AND mean and beats the Phase 2 schedule on validation.
- Transfer of the alignment gain to XOR. The difference reverses.
- Transfer to engineered HAR. The primary difference reverses, and D is below dense A.
- A resolved temporal-HAR test-set alignment gain in Experiment 2.
- Useful learned context. Phase 3 factors stay near 0.9988. Phase 3B's primary gain is +0.00213 macro-F1 and misses +0.005. Shuffled-code and seed-level intervals include zero.
- Associative retrieval, memory, or a biological collapse. Phase 4 was not run. Width never changes.

## Question C — Is the formulation internally consistent?

**CONFIRMED** for the equations that are actually in the code. Dimensions match. The half-distance formula matches the Gaussian. The product rule matches the detached forward pass.

**CONTRADICTED** as a description of the infographic. **NOT TESTED** for `tau(q)=sigma/r(q)`.

The original context parameterization is consistent and almost inert: a smooth factor initialized where its derivative is tiny, with duplicate columns.

## Question D — Collapse, expansion, or rescaling?

The schedule increases `tau`, which raises every gate. That is expansion of a soft neighborhood. Exact release sets every gate to 1. At inference, D is a dense MLP. S and the fixed-d50 policies keep a static spatial scale.

Early training is not a pure rescaling of a dense step. About 42% of input-layer gates at `d50=0.08` are below `1e-6`, so those edges barely move. A single global scale, which is what A′ applies, does not reproduce that pattern. A′'s 1B accuracy (0.863) sits with dense A and with B′, not with D.

Changing `tau` at fixed `W` changes the function. That is a real structural change of effective connectivity for the forward pass. It is a change of a prespecified mask, not a learned rewiring. After release, the structural change is gone and only the weight values remain.

## Question E — Learned relational reorganization?

**NOT TESTED** as a learned graph. Coordinates are fixed before training. `tau` is a clock. The data can change `W` only where the current gate lets the gradient through. That is curriculum-biased learning of weights, not estimation of a relation structure.

Phase 3B's learned change in context factors is small beside the random initialization of those factors. Level 2 passes in a minority of pair comparisons. **CONTRADICTED** as a demonstration of learned relational reorganization. **SUPPORTED** as a demonstration that initialization can mimic code sensitivity.

## Question F — Contextual or associative memory?

Context codes select a low-rank multiplier and zero some sensor channels. That is an external mask plus a small learned gain. No addressable memory, no retrieval loss, and no held-out associative query were implemented. **NOT TESTED** for associative memory. **CONTRADICTED** for the claim that the implemented context factor is a useful learned context mechanism, by the authors' own +0.005 rule and by the near-constant Phase 3 gates.

## Question G — Could regularization, initialization, masking, or optimization explain the AND gap?

Yes, several of those are viable, and the controls separate some of them.

| Alternative | What the evidence says |
| --- | --- |
| More parameters | Counts match. Does not explain D versus B′ |
| Different initialization of `W` | Fingerprints match within seed. Does not explain the gap |
| Different data order | Paired. Does not explain the gap |
| Layer-wide gate energy | A′ matches layer energy and does not match D's accuracy in 1B |
| Per-row gate energy | B′ matches it and still loses on AND. Energy matching is doing its job |
| Per-edge learning-rate schedule `G^2` | Still fully available as an explanation. B′ has the same multiset of scales and a different assignment. The gap is about which edges receive those scales |
| Dense release itself | S and fixed medium width do not need it to score well. Release is a weak explanation of the best AND numbers |
| Task-aligned placement of the large scales | The remaining difference between D and B′. Supported on AND. Reversed on XOR and engineered HAR |

The AND result is therefore best read as evidence for a useful fixed spatial prior on that synthetic layout, implemented as a multiplicative mask. It is a weak read as evidence for a distinct collapse principle.

## Question H — Strongest evidence against the current hypothesis

If the current hypothesis is that scheduled collapse or expansion of Gaussian connectivity is a generally useful learning mechanism, the strongest contrary evidence is the joint pattern:

1. In 1B the model that never expands, S, outscores the model that expands and releases, D, by 2.66 points, with an interval that excludes zero.
2. In 1C and Phase 2, a fixed intermediate width outscores the scheduled release path.
3. On XOR the aligned schedule is worse than its shuffle.
4. On engineered HAR the aligned schedule is worse than its shuffle and worse than dense training.
5. At the end of D there is no remaining mask to inspect. The supposed structure has been replaced by a dense matrix.
6. The learned context factor does not clear its own practical threshold, and a shuffled-code control is not separated from it.

Item 1 is the most direct internal counterexample, because it uses the same task on which the alignment result looks strongest.

## Question I — Which conclusions remain defensible?

1. The code implements a detached Gaussian mask on a fixed line, with the documented controls.
2. On AND, alignment of that mask beats a row shuffle under the 1B procedure, by 0.0484375 mean test accuracy.
3. The effect size is protocol-sensitive: 1C's new seeds give 0.0254 under a shared learning rate.
4. Exact release is not an established benefit.
5. The effect does not generalize to XOR or to file-order HAR features in these experiments.
6. The Phase 3 context factor, as initialized, barely changes the forward pass.
7. Phase 3B's numerical repairs were measured. They do not meet the predictive criterion the protocol set.
8. No associative-memory result exists.

## Question J — Smallest decisive experiment

See `09_EXPERIMENT_3D_PROPOSAL.md`. The decision this repository still cannot make is whether a **change** in bandwidth during training does anything that a **fixed** mask of the best constant width does not do, on a task where the neighborhood is known and a dense model is not already sufficient. Experiment 1C asked a version of this and left the primary interval on zero. The next experiment should be smaller, should make the mask change the object of measurement at fixed weights, and should be built so that a dense model and a fixed mask are both expected to fail if the schedule is doing real work.

## Summary table

| Claim | Class |
| --- | --- |
| Gaussian formula and detached product rule | CONFIRMED |
| AND alignment beats row shuffle in 1B | CONFIRMED |
| The 1B gap is caused by collapse or release | CONTRADICTED by S and by 1C |
| Alignment is a general inductive bias | CONTRADICTED by XOR and engineered HAR |
| Terminal D is structurally different from a dense net | CONTRADICTED by `G=1` |
| Learned reorganization of relations | NOT TESTED |
| Associative memory | NOT TESTED |
| Useful Phase 3B context gain at the +0.005 rule | CONTRADICTED by the primary estimate |
| `tau(q)=sigma/r(q)` | NOT TESTED |
