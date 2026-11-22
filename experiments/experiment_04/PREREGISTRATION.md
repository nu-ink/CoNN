# Experiment 04 preregistration

Frozen before the first fit on 2026-11-20 (America/Chicago). Machine timestamps use UTC.
This local record is not an externally timestamped registry. User authorized implementation
and then explicitly waived the Git checkout requirement. No existing source may be edited.
The earlier Experiment 04 proposal is retained unchanged; this specification resolves its
unspecified task, thresholds, controls and endpoints. Here D means uniform control, not learned geometry.

## Claim and decision

One primary contrast: structured-task final independent test accuracy, aligned minus
energy-matched row-scrambled, across seeds 4000–4009. H0: expected paired difference = 0;
H1: expected difference > 0. Predetermined practical difference: 0.02 (two percentage points).
Use an exact one-sided sign-flip test of the mean difference, alpha .05; report a two-sided
95% Student t confidence interval and paired Cohen dz. A statistically positive result
requires p < .05 and positive mean. Practical support additionally requires the lower
95% bound to exceed .02. Otherwise report inconclusive practical support or a disadvantage
as the evidence warrants. A non-rejection is not equivalence. Exact sign-flip inference
assumes symmetric independent seed differences under H0; t intervals assume a sufficiently
well-behaved seed mean. With n=10, both assumptions and low power limit inference.

## Task fixed before training

For every independent example and group g=0,...,4, draw independent a_g,b_g ~ N(0,1).
Each group has 10 independent noisy measurements of a_g, followed by 10 of b_g:
x_(20g+k)=a_g+0.3 epsilon_(20g+k) for k<10, otherwise b_g+0.3 epsilon,
epsilon~N(0,1). Score S=sum_g (|a_g+b_g|-|a_g-b_g|).
Clean label = 1[S>0]; independently flip it with probability .05.
The score encodes within-group sign agreement and magnitude, and never uses a gate.
Simultaneously changing all a signs negates S, giving balanced classes in expectation.
No balancing by resampling or test-set adjustment is performed.

Structured inputs retain group order; assigned coordinates are i/99. Hidden coordinates
are j/31, fixed before data generation. These coordinates are meaningful because nearby
indices measure the same latent or the interacting pair in one group. Hidden locations
impose a local computation prior; they do not represent known learned semantics.
Negative-control inputs use a seed-specific random column permutation, frozen for all
splits, with the same assigned grid and identical labels/examples. Thus true relationships
are dispersed; the label-free geometry is unrelated in construction to the permuted layout.
Residual chance alignment is possible. No train-test distribution shift is claimed.

Train/validation/test sizes: 512/256/2048, independent named RNG streams. Seed controls
independent data draws, initialization, ordering, negative permutation, and scrambling.
All four conditions within seed/task share exactly the same datasets, weights and batches.

## Models and optimization

100–32–32–1, ReLU, BCE-with-logits, 4,321 trainable parameters. Biases and output head
are ungated. Conditions: mlp (ones), aligned Gaussian, scrambled Gaussian (fixed independent
within-row column permutations in both layers), uniform (sqrt of the Gaussian layer mean
square). No learned coordinates. Uniform matches layer energy, not individual row energy.
Unmodified historical CoNNMLP, initializer, SGD builder, Gaussian, scalar, distance and
schedule functions are reused. Initial raw weights match; effective initial weights differ
by treatment. Gates therefore affect both initialization scale and optimization; mechanism
claims must acknowledge this. The ones baseline uses the same gated forward path, so runtime
comparison measures control generation and permutation overhead over a matched implementation,
not a fully optimized nn.Linear baseline. All conditions remain dense tensor operations.

SGD lr=.1, no momentum/weight decay, batch 32, fixed shuffled order cycled, 1,600 updates.
Smoothstep bandwidth expansion d50=.08 to 1.0 (tau=d50/sqrt(2 ln 2)); release index 1120.
Updates 0–1119 use Gaussian/control gates; 1120–1599 use exact ones. Evaluation after
update 1120 uses last active gate as the pre-release endpoint. An immediate-release endpoint
uses these exact same weights with ones. Primary final endpoint uses weights after all
1600 updates and ones. No checkpoint selection, early stopping or hyperparameter search.
Validation is monitored only; all planned checkpoints are fixed. No post-release gain is
attributed to surviving inference gates. CPU, one thread, deterministic PyTorch algorithms.

## Execution and stopping

Unit tests first; smoke seed 3900, 64/64/128 examples, 40 steps. Pilot seeds 3910,3911,
512/256/2048, 400 steps; monitor training/validation only (pilot test remains unread).
Confirmatory seeds 4000–4009 are disjoint. Complete all 80 confirmatory fits before a separate
evaluation command generates held-out test samples and evaluates three predefined endpoints.
No adjustment to configuration based on smoke/pilot test scores. Technical failure stops
the stage and records its exception; no selective seed replacement. Interrupted stages
remain partial. Resource constraint: local CPU only, no installations, cloud or paid jobs.
Ten seeds is the fixed target; any shorter execution is preliminary, not a replacement study.

Six secondary accuracy contrasts, each exact two-sided sign-flip and Holm-adjusted as one
family: structured final aligned-minus-mlp; structured final aligned-minus-uniform;
structured pre-release aligned-minus-scrambled; structured immediate-release aligned-minus-
scrambled; negative final aligned-minus-scrambled; structured-minus-negative final alignment
effect. Their 95% t intervals are descriptive and unadjusted. F1, BCE, learning curves,
initial metrics, weight-distance bins, runtime and sampled process RSS are descriptive.
No other formal superiority tests. Runtime practical limit: mean paired optimization time
ratio <=1.5 versus mlp on this machine. RSS is sampled process memory, not an isolated peak.

## Competing explanations and limits

Uniform removes a layer energy explanation; scrambling preserves row multisets exactly.
Scrambling still changes topology as well as alignment. Negative task tests specificity,
but finite seeds permit accidental alignment. Shared raw initialization is controlled,
effective initial scale is a treatment effect; diagnostics cannot establish its causal role.
One favorable DGP is not external validity. No result establishes associative memory,
learned geometry, general intelligence, consciousness or replacement of transformers.
Advance to learned associations only after practically supported, task-specific predictive
evidence survives the primary and magnitude controls; otherwise first investigate fixed
geometry with separately preregistered experiments.
