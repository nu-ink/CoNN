# Statistical analysis

[Confirmatory run](../../../experiments/experiment_04/results/run_20261121T031743_424624Z_7ac51c80/), [analysis JSON](../../../experiments/experiment_04/results/run_20261121T031743_424624Z_7ac51c80/analysis.json), [raw endpoint table](../../../experiments/experiment_04/results/run_20261121T031743_424624Z_7ac51c80/endpoint_metrics.csv).

Independent unit: paired seed (n=10), not individual examples or minibatches.
The primary contrast was frozen before fitting and test access.
Aligned minus scrambled final structured test accuracy: mean **1.1328125 pp**,
sample SD **1.2771871 pp**, two-sided 95% t CI **[0.2191679, 2.0464571] pp**,
Cohen dz **0.8869589**. Exact one-sided sign-flip p=**0.0146484375**;
paired t-test sensitivity p=0.01027525595.

There are 1,024 possible sign assignments for ten differences. The exact test
enumerates them and compares their mean with the observed mean, including ties.
This relies on a symmetric null distribution of seed differences, stronger than
an unrestricted mean-zero null. The t interval assumes an approximately well-behaved
independent seed mean; ten replications cannot validate those assumptions strongly.
The preregistered test rejects at .05, but practical support required the lower
95% bound to exceed 2 pp. That criterion is **not met**. The point estimate is also
below 2 pp. This is not a demonstration of equivalence or absence of a useful effect.

| Seed | Aligned − scrambled (pp) |
| --- | --- |
| 4000 | 2.8320 |
| 4001 | -0.7812 |
| 4002 | 2.5879 |
| 4003 | 1.6113 |
| 4004 | 2.0996 |
| 4005 | 2.0996 |
| 4006 | 0.2441 |
| 4007 | -0.0488 |
| 4008 | 0.8301 |
| 4009 | -0.1465 |


Six two-sided secondary sign-flip tests form one Holm-adjusted family:

| Contrast | Mean effect (pp) | 95% t interval (pp), unadjusted | Exact two-sided p | Holm p |
| --- | --- | --- | --- | --- |
| structured_final_aligned_minus_mlp | 1.177 | [0.623, 1.730] | 0.001953 | 0.011719 |
| structured_final_aligned_minus_uniform | 0.845 | [0.066, 1.623] | 0.039062 | 0.117188 |
| structured_pre_release_aligned_minus_scrambled | 1.396 | [0.383, 2.410] | 0.021484 | 0.085938 |
| structured_immediate_release_aligned_minus_scrambled | 0.537 | [-0.178, 1.252] | 0.125000 | 0.250000 |
| negative_final_aligned_minus_scrambled | -0.024 | [-0.808, 0.759] | 0.957031 | 0.957031 |
| structured_minus_negative_alignment_effect | 1.157 | [0.254, 2.060] | 0.009766 | 0.048828 |


Secondary intervals are descriptive and unadjusted. The adjusted evidence supports
the aligned-minus-MLP contrast and the structured-minus-negative interaction at .05.
The aligned-minus-uniform contrast does not survive adjustment; gate magnitude alone
has therefore not been decisively excluded by that control. The negative-control
null is not rejected, which does not prove that its methods are equivalent.

F1, BCE, convergence, runtime, memory and weight-distance bins are descriptive,
without additional superiority tests. No seeds were dropped or replaced, no
hyperparameters selected from tests, and no post-hoc success threshold introduced.
