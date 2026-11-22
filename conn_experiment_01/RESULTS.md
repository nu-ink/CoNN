# Results

Measured numbers below were copied from the run logs. Expectations are marked as such. This file does not add scores that the runner did not write.

## Smoke, pipeline only

Directory: `results/runs/20261017T235744Z_smoke`

Command: `python -m conn_exp.runner --config configs/pilot.yaml --smoke`

One seed, 30 optimizer steps, hidden width 16. Every condition finished with finite loss and exactly one test evaluation. Final test accuracy was 0.500 for all seven conditions. The paired D minus B′ delta was 0 with n = 1, so no interval was computed. This run checks that the logger, release checkpoints, and plots are produced. It is not evidence about the curriculum.

## Five-seed AND pilot

Directory: `results/runs/20261017T235839Z_pilot`

Command: `python -m conn_exp.runner --config configs/pilot.yaml --seeds 5`

Config SHA-256 `d0c1381a689bf3a675dfba4f01af1add76ab9fc3550c5df012d40790176bae14`. CPU, PyTorch 2.14.1+cpu, Python 3.11.1. SGD, momentum 0, weight decay 0, learning rate 0.1 for every condition. 400 optimizer steps. Release at step 280. The predefined learning-rate grid was not run.

All 35 condition-by-seed runs were finite, and each evaluated the test split once at the final step.

Primary contrast, final test accuracy, D minus B′ (`row_shuffle`):

| Seed | D | B′ | Delta |
| --- | --- | --- | --- |
| 0 | 0.789062 | 0.781250 | 0.007812 |
| 1 | 0.742188 | 0.703125 | 0.039062 |
| 2 | 0.781250 | 0.757812 | 0.023438 |
| 3 | 0.679688 | 0.664062 | 0.015625 |
| 4 | 0.757812 | 0.734375 | 0.023438 |

n = 5. Mean delta = 0.021875. Sample SD = 0.011588. Student-t 95% interval with critical value 2.776445: [0.007487, 0.036263].

The interval excludes zero in this sample. Five seeds are still exploratory. This is not a confirmatory test, and no multiplicity correction was applied. A higher D than B′ would not, by itself, isolate geometry: B′ keeps each row's gate values and energy and changes which sources receive them. The shared learning rate was not selected by a fair grid.

Raw final test accuracies for the other conditions are in that run's `report.md` and `scores.csv`. Static-local (S) stayed near chance (0.500 to 0.539) and does not end in the dense gate. Dense (A) is a different final gate state from the curriculum conditions.

Invariant checks on the same logs: B′, A′, and C matched D's layer energy to within 2e-7. B′ matched D's per-row gate energy to within 1.6e-5. A′'s per-row energy standard deviation during the local phase stayed under 1e-5. Those checks are about the controls, not about accuracy.

## Expectation, not a result

A positive D minus B′ delta is what the curriculum hypothesis predicts under this protocol. The measured interval is reported above. It is not a claim that a Gaussian gate or a Hoberman mechanism is a new architecture, and it is not a claim of statistical confirmation.
