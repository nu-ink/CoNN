# Experiment index

Discovery excluded `AUDIT_20261020`, `ORGANIZED_EXPERIMENTS`, virtual environments, caches, and Git storage. There is no Git repository.

| Directory | What it is | Objective, as stated by its own documents | Organizational status |
| --- | --- | --- | --- |
| `EXPERIMENT_01` | Smoke run `20261017T235744Z_smoke` and five-seed pilot `20261017T235839Z_pilot` | Compare task-aligned local-to-global Gaussian gating with magnitude-matched controls on a synthetic AND task. Primary contrast: final test accuracy of D minus B prime. | PARTIAL |
| `EXPERIMENT_01B` | Run `20261018T001947Z_experiment1b_v1` | Repeat that contrast after a validation-only learning-rate and duration grid, on seeds 100–119. | COMPLETE |
| `EXPERIMENT_01C` | Run `20261018T024146Z_experiment1c_v1` | Compare locality and dense release. Primary contrast: AND partial-width P minus full-release D, seeds 200–219. | COMPLETE |
| `EXPERIMENT_02` | Run `20261019T004346Z_experiment2_har_v1` | Test the scheduled gate on UCI HAR. Primary contrast: engineered-feature D minus B macro-F1, seeds 300–319. | COMPLETE |
| `EXPERIMENT_03_PHASE01` | Associative protocol v2, Phase 0 and Phase 1 | Freeze the associative questions, data audit, and gate definition before training. | AMBIGUOUS |
| `EXPERIMENT_03_PHASE23` | Run `20261019T030442340025Z_phase23_v1` | Development grid for temporal HAR Phase 2 conditions and Phase 3 context conditions. One amendment, one run, six source stages. | COMPLETE |
| `EXPERIMENT_03B` | Run `20261020_phase3b_v3_run01` | Diagnostic study of context-factor symmetry, saturation, and precision. Primary contrast: Gaussian C4 minus Gaussian masked. | COMPLETE |

Not created, because the source repository does not contain them as executed experiments:

- Experiment 3A
- Experiment 3C
- Experiment 3D

Phase 2 and Phase 3 were not split into 3A and an unnamed 3. The source treats them as one authorized grid with distinct condition names. Splitting the checkpoint tree by condition would have been a new derivation, not a copy of an existing boundary.

The root infographic is referenced from `UNASSIGNED_FILES`.
