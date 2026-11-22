# Existing Experimental Results

All figures in this section are measured from preserved run reports and summaries.

## Experiment 01

- Smoke: seven conditions each scored 0.500 at 30 steps on seed 0. This validated execution, not research performance.
- Five-seed pilot: D − B′ mean 0.021875, sample SD 0.011588, t 95% CI [0.007487, 0.036263].
- The predefined LR grid was not executed. The pilot is exploratory.

Source: [Experiment 01 results](../../conn_experiment_01/RESULTS.md).

## Experiment 1B

After validation-only tuning, seeds 5–9 were exploratory tuning seeds and seeds 100–119 were the independent replication.

- D − B′ mean: 0.0484375 accuracy, or 4.84375 percentage points.
- Paired sample SD: 0.03171934.
- 95% t CI: [0.03359239, 0.06328261].
- Cohen dz: 1.527065.
- Positive/tied/negative: 19/0/1.
- Frozen common duration: 3200 steps.
- Frozen LRs: A/A′/B′/D/E/S = 0.1; C = 0.05.

The 1B result is positive under its single synthetic AND task and fixed protocol. Static local S averaged 93.7891% versus D 91.1328%, so D − B′ should not be interpreted as a benefit over every condition.

Source: [1B research summary](../../conn_experiment_01/results/runs/20261018T001947Z_experiment1b_v1/research_summary.md).

## Experiment 1C

- Sole primary AND P − D: mean 0.01835938, paired SD 0.03961574, 95% CI [-0.00018136, 0.03690011], dz 0.463436.
- The interval includes zero, so the partial-widening advantage is unresolved.
- AND S/P/D aligned-minus-shuffle contrasts were positive in the observed means.
- XOR D − shuffle was negative at -2.5391 points.

Source: [1C research report](../../conn_experiment_01/results/runs/20261018T024146Z_experiment1c_v1/research_report.md).

## Experiment 02 — UCI HAR

- Primary engineered D − B macro-F1: mean -0.00501038, 95% CI [-0.00870502, -0.00131574], dz -0.6347.
- The row-shuffled control outperformed D in the engineered-feature primary result.
- Temporal D − B macro-F1 was +0.001861, CI [-0.001799, 0.005522].
- These are secondary or primary estimates under different coordinate representations and fixed subject splits.

Source: [HAR research report](../../ORGANIZED_EXPERIMENTS/EXPERIMENT_02/documentation/research_report.md).

## Associative Phase 3B

- Primary Gaussian C4 − Gaussian masked subject macro-F1: +0.00213398.
- Descriptive subject-bootstrap CI: [0.00011566, 0.00444627].
- The predeclared +0.005 practical criterion and positive lower bound were not met.
- C4 exact report: 455 development grid fits plus 10 pilots in the summary, 3200 updates per fit.

Source: [Phase 3B final report](../../CoNN_Associative_Relationships/phase3b_context_diagnostics/results/20261020_phase3b_v3_run01/reports/PHASE3B_FINAL_REPORT.md).

