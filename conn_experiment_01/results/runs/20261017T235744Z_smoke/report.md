# CoNN Experiment 01 report

Run `20261017T235744Z_smoke`. Config SHA-256 `add24a38c491191b3d930a62ed45a83ead1b2959bc92a060be00899575199ab3`.
Device `cpu`. Optimizer recorded as SGD, momentum 0, weight decay 0, foreach false.

## Protocol

Primary metric: held-out test accuracy after the pre-specified optimizer-step budget.
The test split is evaluated once, on the post-update weights of the final step. It is not used to pick a checkpoint, to place coordinates, or to fit probes.
Primary contrast: D minus B_prime (`row_shuffle`) on and.
This file's task is `and` with seeds [0].
Learning rate: one shared initial value for every condition. The predefined equal grid [0.01, 0.05, 0.1] was not executed (`lr_grid_executed=False`). These numbers cannot support a claim of superiority.
Schedule: release_step=21 of 30, tau_start=0.06794574402304153, tau_end=0.8493218002880191.
S stays at tau_start through the final step. E is a three-phase path with the same final exact-dense endpoint as D, not a pure reversal.

## Invariant audit

These checks confirm the controls still match the quantities they are supposed to match. They are not outcome results.

- `B_prime_vs_D_layer_energy_max_abs`: 1.192093e-07
- `A_prime_vs_D_layer_energy_max_abs`: 2.384186e-07
- `C_vs_D_layer_energy_max_abs`: 5.960464e-08
- `A_prime_local_row_energy_std_max`: 7.629395e-06
- `B_prime_vs_D_row_energy_max_abs`: 7.629395e-06

## Measured primary contrast

Values below were read from the logged scores. Empty cells were not computed.

| seed | D test accuracy | B' test accuracy | delta | used in aggregate |
| --- | --- | --- | --- | --- |
| 0 | 0.500000 | 0.500000 | 0.000000 | True |

Paired n = 1. Mean delta = 0.000000. Sample SD = .
95% interval using t critical : [, ].

This run is not the five-seed AND pilot, or it is a smoke/pipeline run. Do not read it as evidence for or against the curriculum.

D minus B' does not isolate geometry by itself: within-row shuffles preserve each row's gate multiset and energy, and they change which sources receive those values.
A' matches layerwise mean square gate energy, not the per-row distribution. E shares D's dense endpoint and changes the path. S is still local at the final step.

## All conditions, final test accuracy

| seed | condition | role | test accuracy | test loss | non_finite | test evaluations |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | A | secondary | 0.500000 | 0.692254 | False | 1 |
| 0 | A_prime | secondary | 0.500000 | 0.692703 | False | 1 |
| 0 | B_prime | primary_control | 0.500000 | 0.692771 | False | 1 |
| 0 | C | secondary | 0.500000 | 0.692731 | False | 1 |
| 0 | D | primary_treatment | 0.500000 | 0.692712 | False | 1 |
| 0 | E | secondary | 0.500000 | 0.692529 | False | 1 |
| 0 | S | secondary | 0.500000 | 0.693505 | False | 1 |

## Expectations are not measurements

A positive D minus B' delta would be consistent with a benefit of spatial alignment under this protocol. A negative delta, or an interval that contains zero, is a failure of that prediction in this run. The sentence stating the prediction is not a result.

## Test-set usage

Coordinates are functions of feature count and hidden width only. The test split is not an argument to geometry, schedules, masks, probes, or checkpoint selection. Probes fit on training motif labels and score validation. Test accuracy is computed once, after total_steps optimizer updates, and that value is the recorded score.

