# Experiment design

Status: executed. [Confirmatory run](../../../experiments/experiment_04/results/run_20261121T031743_424624Z_7ac51c80/), [analysis JSON](../../../experiments/experiment_04/results/run_20261121T031743_424624Z_7ac51c80/analysis.json), [raw endpoint table](../../../experiments/experiment_04/results/run_20261121T031743_424624Z_7ac51c80/endpoint_metrics.csv).

The complete [preregistration](../../../experiments/experiment_04/PREREGISTRATION.md) was written before the first fit.
It specifies the DGP, endpoints, 2 pp threshold, one-sided primary test, stopping rule,
six secondary contrasts and runtime criterion. Earlier Experiment 04 proposal files
remain historical proposals; this directory documents the authorized implementation.

One manipulation is source assignment of Gaussian gate values, preserving every
scrambled row's multiset. A layer-uniform control matches mean squared magnitude.
All four conditions share raw initial weights, 4,321 parameters, architecture,
SGD configuration, batches, data and 1,600 updates. Uniform matches layer energy,
whereas scrambled matches row energy. Effective initial weights differ by treatment.

Structured features have a known latent-interaction grouping. The negative control
permutes observed columns independent of targets and keeps coordinates fixed. Its
geometry is unrelated by construction, with chance residual alignment possible.
The target never references Gaussian gates. Hidden coordinates encode a prior,
not identified semantic neurons. No geometry is learned.

Smoke: one independent seed, eight fits and three saved evaluation endpoints per fit.
Pilot: two disjoint seeds, 16 fits; training and validation only, no test evaluation.
Confirmatory: ten new seeds (4000–4009), 80 fits; 240 fixed endpoint measurements.
Train/validation/test sizes are 512/256/2048 for pilot/main, 64/64/128 for smoke.
No validation-based hyperparameter change or model selection occurred.

The primary endpoint is final independent test accuracy after 480 dense updates.
Pre-release and immediate-release endpoints are secondary and share identical weights.
All main models were trained before the separate test-evaluation command ran.
Each seed independently samples its latent data, initialization and mask.

No Git checkout or commit exists; the user explicitly waived checkout after the
preflight blocker. Original files are protected by a hash manifest and source snapshots.
No packages were installed or upgraded. All work used the existing Python 3.10 environment.
