# CoNN Experiment 1B — Optimization and Replication

The structured condition retained a positive paired advantage after fair tuning in this independent replication.

Measured primary result, 20 independent paired seeds: mean D − B′ = 0.04843750 (+4.8438 percentage points); sample SD = 0.03171934; two-sided 95% Student-t CI [0.03359239, 0.06328261] (df=19). Cohen’s paired dz = 1.527065.
Positive/tied/negative seed differences: 19/0/1.

## Preserved pilot and separation

The five-seed pilot at 20261017T235839Z_pilot remains exploratory. Its measured +0.021875 difference motivated this follow-up and is not pooled with replication. Pilot files and exact configuration are preserved; before/after SHA-256 manifests verify this.
Tuning used new seeds 5–9; replication used seeds 100–119. Neither set overlaps pilot seeds 0–4. Tuning performed zero test evaluations and stored no test measurements. Replication test accuracy was measured once per model after the final optimizer update.

## Fixed protocol and model selection

The unchanged synthetic AND task, split sizes, hidden widths, SGD without momentum/decay, batch order, geometry and gate rules come from configs/pilot.yaml. Seven conditions were retained. Each received the predefined learning-rate grid [0.01, 0.05, 0.1] at 400, 800, 1600 and 3200 steps: 12 candidates × five tuning seeds = 60 fits per condition (420 total). Every condition received 90,000 tuning optimizer steps.
At each duration, each condition’s LR maximized mean final validation accuracy; ties used lower mean validation loss, then lower LR. The common duration maximized macro-average validation accuracy across all seven selected conditions; ties used lower macro validation loss, then shorter duration. The D − B′ contrast was never the selection objective.
Frozen common duration: 3200 optimizer steps. Frozen learning rates: A=0.1, A_prime=0.1, B_prime=0.1, C=0.05, D=0.1, E=0.1, S=0.1.
The exclusive-create selection.frozen.json and SHA-256 seal were written before replication.started.json and before replication models or data were created. Replication retained the frozen choices for all 20 seeds; no test-selected checkpoint, replacement seed, or repeated model selection was used.
Duration experiments restart from the paired initialization and scale the curriculum to each duration. They compare complete training budgets, not intermediate checkpoints on a common 3200-step schedule. Diagnostic measurements use 20 equally spaced intervals plus schedule boundaries.

## Exploratory validation duration comparison

These are measured tuning validation means after per-condition LR selection, and have selection optimism. They are not held-out test results.

| Steps | A | A_prime | B_prime | C | D | E | S |
|---|---|---|---|---|---|---|---|
| 400 | 0.832812 (lr 0.1) | 0.759375 (lr 0.1) | 0.767188 (lr 0.1) | 0.753125 (lr 0.1) | 0.753125 (lr 0.1) | 0.795312 (lr 0.1) | 0.603125 (lr 0.1) |
| 800 | 0.853125 (lr 0.1) | 0.850000 (lr 0.1) | 0.850000 (lr 0.1) | 0.846875 (lr 0.1) | 0.875000 (lr 0.1) | 0.885938 (lr 0.1) | 0.654687 (lr 0.1) |
| 1600 | 0.856250 (lr 0.1) | 0.848437 (lr 0.1) | 0.876563 (lr 0.1) | 0.868750 (lr 0.1) | 0.904687 (lr 0.1) | 0.884375 (lr 0.05) | 0.756250 (lr 0.1) |
| 3200 | 0.857812 (lr 0.1) | 0.865625 (lr 0.1) | 0.876563 (lr 0.1) | 0.865625 (lr 0.05) | 0.925000 (lr 0.1) | 0.862500 (lr 0.1) | 0.948438 (lr 0.1) |

## Independent replication: all seed results

| Seed | A | A_prime | B_prime | C | D | E | S | D − B′ |
|---|---|---|---|---|---|---|---|---|
| 100 | 0.929688 | 0.921875 | 0.898438 | 0.937500 | 0.945312 | 0.929688 | 0.945312 | +0.046875 |
| 101 | 0.851562 | 0.804688 | 0.859375 | 0.804688 | 0.906250 | 0.812500 | 0.968750 | +0.046875 |
| 102 | 0.851562 | 0.859375 | 0.851562 | 0.875000 | 0.906250 | 0.890625 | 0.937500 | +0.054688 |
| 103 | 0.867188 | 0.875000 | 0.820312 | 0.851562 | 0.960938 | 0.835938 | 0.976562 | +0.140625 |
| 104 | 0.867188 | 0.898438 | 0.882812 | 0.882812 | 0.929688 | 0.890625 | 0.960938 | +0.046875 |
| 105 | 0.882812 | 0.890625 | 0.898438 | 0.906250 | 0.945312 | 0.890625 | 0.953125 | +0.046875 |
| 106 | 0.867188 | 0.859375 | 0.875000 | 0.867188 | 0.898438 | 0.867188 | 0.976562 | +0.023438 |
| 107 | 0.812500 | 0.812500 | 0.835938 | 0.781250 | 0.867188 | 0.812500 | 0.820312 | +0.031250 |
| 108 | 0.906250 | 0.890625 | 0.914062 | 0.906250 | 0.937500 | 0.890625 | 0.945312 | +0.023438 |
| 109 | 0.859375 | 0.914062 | 0.875000 | 0.890625 | 0.937500 | 0.898438 | 0.953125 | +0.062500 |
| 110 | 0.820312 | 0.835938 | 0.812500 | 0.773438 | 0.796875 | 0.820312 | 0.898438 | -0.015625 |
| 111 | 0.812500 | 0.843750 | 0.828125 | 0.820312 | 0.906250 | 0.851562 | 0.968750 | +0.078125 |
| 112 | 0.867188 | 0.812500 | 0.828125 | 0.835938 | 0.914062 | 0.867188 | 0.921875 | +0.085938 |
| 113 | 0.804688 | 0.804688 | 0.820312 | 0.828125 | 0.843750 | 0.804688 | 0.929688 | +0.023438 |
| 114 | 0.820312 | 0.820312 | 0.835938 | 0.828125 | 0.867188 | 0.828125 | 0.914062 | +0.031250 |
| 115 | 0.882812 | 0.867188 | 0.882812 | 0.890625 | 0.945312 | 0.945312 | 0.945312 | +0.062500 |
| 116 | 0.914062 | 0.898438 | 0.867188 | 0.875000 | 0.929688 | 0.914062 | 0.906250 | +0.062500 |
| 117 | 0.921875 | 0.898438 | 0.906250 | 0.906250 | 0.960938 | 0.890625 | 0.992188 | +0.054688 |
| 118 | 0.906250 | 0.906250 | 0.937500 | 0.914062 | 0.953125 | 0.906250 | 0.937500 | +0.015625 |
| 119 | 0.843750 | 0.851562 | 0.828125 | 0.828125 | 0.875000 | 0.851562 | 0.906250 | +0.046875 |

## All-condition replication means

| Condition | Mean test accuracy | Sample SD | Mean test loss |
|---|---|---|---|
| A | 0.864453 | 0.038296 | 0.455297 |
| A_prime | 0.863281 | 0.038649 | 0.513357 |
| B_prime | 0.862891 | 0.036334 | 0.465494 |
| C | 0.860156 | 0.046038 | 0.421625 |
| D | 0.911328 | 0.043111 | 0.293324 |
| E | 0.869922 | 0.041283 | 0.812168 |
| S | 0.937891 | 0.037892 | 0.161319 |

The static-local condition S averaged 93.7891% accuracy versus D’s 91.1328%; D − S = -2.6562 percentage points. This is a secondary, exploratory comparison. The primary D − B′ advantage does not establish that widening to global connectivity is necessary or optimal; S keeps local gating throughout and achieved the higher observed mean here.


## Diagnostics and ablations

Raw metrics record effective-weight mean absolute magnitude and L2 norm, activation variance, per-destination-unit sum of squared gates, and fixed-minibatch pre-update dL/dW and dL/dW_eff by base task distance. Gradient charts separate conditions and layers; they do not average the structured model together with controls. Train-fit motif probes are scored only on validation.
Each model’s saved pre-release, post-release and final checkpoints were reloaded independently. The release pair shares identical weights and differs in gate state. Far connections are defined by distance at or above each base geometry matrix’s linear 75th percentile. Input-layer, hidden-layer and joint suppression measure validation loss and accuracy changes without retraining or changing stored weights. Positive loss delta means dependence on the removed connections; it does not establish that those connections caused a training advantage.
See replication/checkpoint_ablation_summary.csv and replication/diagnostic_summary.csv for measured endpoint means and SDs; job-level tables preserve every measurement and checkpoint hash. Invariant audits check layer energy, per-unit D/B′ energy matching, and paired data/initialization fingerprints for every grid cell.

## Interpretation and limitations

The expectation was a positive D − B′ difference; the measured result above determines whether that expectation survived this protocol. The t interval describes seed-to-seed variation with fresh finite synthetic data splits, not robustness across tasks or data distributions. Twenty seeds, 128 test examples per seed, a narrow predefined SGD LR grid, and a single synthetic AND task limit generalization. The grid boundary may be optimal within the tested grid without being globally optimal. Hyperparameter selection on five validation seeds has uncertainty. Other condition contrasts in paired_comparisons.csv are exploratory and their intervals are unadjusted for multiplicity; D versus B′ is the sole primary comparison.
Condition-specific LRs make the primary comparison one between equally tuned training procedures. A′ matches layer energy, while B′ matches each row’s gate multiset and energy. A within-row shuffle changes source alignment and higher-order mask topology together; this does not isolate geometry alone. S remains local at its endpoint; E is a path control with the shared dense endpoint. Longer budgets also stretch the curriculum, so duration and curriculum speed are coupled.
These experiments do not establish a new architecture or prove the geometry hypothesis. A repeatable advantage here would support task-aligned gating under these fixed settings only.

## Reproducibility

protocol.frozen.json records the full plan before training; config.resolved.yaml and hashes capture study and per-fit settings. environment.json, requirements.freeze.txt, frozen source snapshot and source_manifest.json identify the implementation. No Git repository was present, so git_revision is null and the content-addressed source snapshot is the code revision. Existing independent data/init/order/mask streams and frozen row permutations were reused. All scores, curves, gradients, unit energies, probes, checkpoint ablations and weights are retained in versioned phase/job directories. completion.json records successful preservation and freeze checks.
