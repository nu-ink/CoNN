# Experiment 04 Statistical Plan

**PROPOSED — NOT IMPLEMENTED.**

## Primary outcome

Paired final held-out metric for aligned geometry minus scrambled geometry, using the same seed, split and weight initialization.

## Success criteria

- One positive paired difference with CI excluding zero and a prespecified minimum effect;
- repeated replication with no test-based seed replacement;
- invariant audits passing;
- no architecture or task-specific conclusion beyond the preregistered domain.

## Failure criteria

- Pair difference at zero or negative under the prespecified interval;
- invariant mismatch;
- test-based hyperparameter or checkpoint selection;
- no independent replication.

## Inconclusive criteria

- CI spans the minimum effect threshold;
- high seed heterogeneity;
- missing or nonfinite fits;
- energy/control confounding.

## Multiple comparisons

Declare one primary contrast. Treat all dense baseline, schedule, ablation and distribution-shift comparisons as secondary unless preregistered multiplicity control is included.

