# Experiment 04 Proposal

**PROPOSED — NOT IMPLEMENTED. Do not implement without explicit approval.**

## Research objective

Determine whether known ground-truth relationship geometry improves performance compared with:

- a conventional ungated MLP;
- uniform energy-matched gating;
- scrambled relationship geometry;
- fixed and scheduled variants that can separate schedule from alignment.

This is a proposal. No existing architecture or task should be claimed as Experiment 04 until a preregistration freeze is accepted.

## Proposed conditions

| Condition | Description |
|---|---|
| A | Conventional dense MLP |
| B | Gaussian gate from known ground-truth relationships |
| C | Scrambled gate values preserving allowed energy invariants |
| D | Learned geometry, future extension only |

Condition D is explicitly deferred.

## Dataset

Use a controlled synthetic classification or association task with:

- known relationship graph or coordinate geometry;
- features generated from latent relationship structure;
- train/validation/test partitions sealed before tuning;
- independent streams for initialization, data, minibatch order and mask permutation;
- at least task-appropriate distribution shifts;
- fixed noise and distractor levels;
- no test-set-dependent coordinate or mask construction.

The final test split should be accessed once after configuration freeze and model fitting.

## Statistical plan

The primary outcome is a preregistered paired metric. Candidate metrics include held-out accuracy or macro-F1, depending on class balance. Each model pair should share initialization, data and minibatch order.

Use:

- paired mean differences and sample standard deviations;
- two-sided 95% intervals;
- Cohen dz;
- preregistered success/failure/inconclusive thresholds;
- multiplicity control for secondary contrasts;
- robust sensitivity checks for graph perturbation and gate energy.

No test-split tuning is permitted.

