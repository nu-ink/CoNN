# CoNN Experiment 01 report

Run `20261017T235839Z_pilot`. Config SHA-256 `d0c1381a689bf3a675dfba4f01af1add76ab9fc3550c5df012d40790176bae14`.
Device `cpu`. Optimizer recorded as SGD, momentum 0, weight decay 0, foreach false.

## Protocol

Primary metric: held-out test accuracy after the pre-specified optimizer-step budget.
The test split is evaluated once, on the post-update weights of the final step. It is not used to pick a checkpoint, to place coordinates, or to fit probes.
Primary contrast: D minus B_prime (`row_shuffle`) on and.
This file's task is `and` with seeds [0, 1, 2, 3, 4].
Learning rate: one shared initial value for every condition. The predefined equal grid [0.01, 0.05, 0.1] was not executed (`lr_grid_executed=False`). These numbers cannot support a claim of superiority.
Schedule: release_step=280 of 400, tau_start=0.06794574402304153, tau_end=0.8493218002880191.
S stays at tau_start through the final step. E is a three-phase path with the same final exact-dense endpoint as D, not a pure reversal.

## Invariant audit

These checks confirm the controls still match the quantities they are supposed to match. They are not outcome results.

- `B_prime_vs_D_layer_energy_max_abs`: 1.788139e-07
- `A_prime_vs_D_layer_energy_max_abs`: 1.788139e-07
- `C_vs_D_layer_energy_max_abs`: 1.192093e-07
- `A_prime_local_row_energy_std_max`: 7.629395e-06
- `B_prime_vs_D_row_energy_max_abs`: 1.525879e-05

## Measured primary contrast

Values below were read from the logged scores. Empty cells were not computed.

| seed | D test accuracy | B' test accuracy | delta | used in aggregate |
| --- | --- | --- | --- | --- |
| 0 | 0.789062 | 0.781250 | 0.007812 | True |
| 1 | 0.742188 | 0.703125 | 0.039062 | True |
| 2 | 0.781250 | 0.757812 | 0.023438 | True |
| 3 | 0.679688 | 0.664062 | 0.015625 | True |
| 4 | 0.757812 | 0.734375 | 0.023438 | True |

Paired n = 5. Mean delta = 0.021875. Sample SD = 0.011588.
95% interval using t critical 2.776445: [0.007487, 0.036263].

Five seeds are pilot/exploratory. This interval is not a confirmatory significance test, and no multiplicity correction is applied because no confirmatory claim is made.
In this sample the interval excludes zero. That is still not a confirmatory finding at five seeds.

D minus B' does not isolate geometry by itself: within-row shuffles preserve each row's gate multiset and energy, and they change which sources receive those values.
A' matches layerwise mean square gate energy, not the per-row distribution. E shares D's dense endpoint and changes the path. S is still local at the final step.

## All conditions, final test accuracy

| seed | condition | role | test accuracy | test loss | non_finite | test evaluations |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | A | secondary | 0.835938 | 0.340781 | False | 1 |
| 0 | A_prime | secondary | 0.781250 | 0.554734 | False | 1 |
| 0 | B_prime | primary_control | 0.781250 | 0.577881 | False | 1 |
| 0 | C | secondary | 0.789062 | 0.549725 | False | 1 |
| 0 | D | primary_treatment | 0.789062 | 0.553523 | False | 1 |
| 0 | E | secondary | 0.781250 | 0.487471 | False | 1 |
| 0 | S | secondary | 0.500000 | 0.690810 | False | 1 |
| 1 | A | secondary | 0.851562 | 0.358472 | False | 1 |
| 1 | A_prime | secondary | 0.726562 | 0.610835 | False | 1 |
| 1 | B_prime | primary_control | 0.703125 | 0.617022 | False | 1 |
| 1 | C | secondary | 0.718750 | 0.629220 | False | 1 |
| 1 | D | primary_treatment | 0.742188 | 0.610747 | False | 1 |
| 1 | E | secondary | 0.750000 | 0.521049 | False | 1 |
| 1 | S | secondary | 0.500000 | 0.692689 | False | 1 |
| 2 | A | secondary | 0.765625 | 0.458084 | False | 1 |
| 2 | A_prime | secondary | 0.750000 | 0.602732 | False | 1 |
| 2 | B_prime | primary_control | 0.757812 | 0.606914 | False | 1 |
| 2 | C | secondary | 0.773438 | 0.616118 | False | 1 |
| 2 | D | primary_treatment | 0.781250 | 0.575086 | False | 1 |
| 2 | E | secondary | 0.781250 | 0.524605 | False | 1 |
| 2 | S | secondary | 0.539062 | 0.692404 | False | 1 |
| 3 | A | secondary | 0.835938 | 0.370518 | False | 1 |
| 3 | A_prime | secondary | 0.679688 | 0.560687 | False | 1 |
| 3 | B_prime | primary_control | 0.664062 | 0.561200 | False | 1 |
| 3 | C | secondary | 0.679688 | 0.579424 | False | 1 |
| 3 | D | primary_treatment | 0.679688 | 0.578213 | False | 1 |
| 3 | E | secondary | 0.710938 | 0.531215 | False | 1 |
| 3 | S | secondary | 0.531250 | 0.692192 | False | 1 |
| 4 | A | secondary | 0.828125 | 0.364343 | False | 1 |
| 4 | A_prime | secondary | 0.742188 | 0.489943 | False | 1 |
| 4 | B_prime | primary_control | 0.734375 | 0.516261 | False | 1 |
| 4 | C | secondary | 0.726562 | 0.502166 | False | 1 |
| 4 | D | primary_treatment | 0.757812 | 0.479996 | False | 1 |
| 4 | E | secondary | 0.796875 | 0.429805 | False | 1 |
| 4 | S | secondary | 0.531250 | 0.691645 | False | 1 |

## Expectations are not measurements

A positive D minus B' delta would be consistent with a benefit of spatial alignment under this protocol. A negative delta, or an interval that contains zero, is a failure of that prediction in this run. The sentence stating the prediction is not a result.

## Test-set usage

Coordinates are functions of feature count and hidden width only. The test split is not an argument to geometry, schedules, masks, probes, or checkpoint selection. Probes fit on training motif labels and score validation. Test accuracy is computed once, after total_steps optimizer updates, and that value is the recorded score.

