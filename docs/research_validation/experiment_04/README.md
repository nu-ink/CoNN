# Experiment 04 — Relationship Geometry Validation

**COMPLETED**: implementation, mathematical tests, paired smoke, two-seed pilot,
ten-seed confirmatory experiment and held-out evaluation.

Aligned final accuracy: **77.54%**, scrambled: **76.41%**; paired effect **+1.13 pp**,
95% CI **[0.22, 2.05] pp**, preregistered one-sided p=**.01465**.
The 2 pp practical-support criterion is unmet. The uniform contrast does not survive
Holm adjustment. Recommendation: strengthen fixed-geometry evidence before learned associations.

- [Design](EXPERIMENT_DESIGN.md)
- [Mathematics](MATHEMATICAL_SPECIFICATION.md)
- [Actual results and six research answers](RESULTS.md)
- [Paired statistics](STATISTICAL_ANALYSIS.md)
- [Tests and integrity](TEST_REPORT.md)
- [Limitations and recommendation](LIMITATIONS.md)
- [Reproduction](REPRODUCTION.md)
- [Preregistration](../../../experiments/experiment_04/PREREGISTRATION.md)

## Diagrams and verified sources

- [Full experimental architecture](diagrams/01_architecture.mmd)
- [Synthetic data generation](diagrams/02_dataset.mmd)
- [Gaussian and control calculations](diagrams/03_gating.mmd)
- [Aligned versus scrambled geometry](diagrams/04_alignment.mmd)
- [Actual training execution flow](diagrams/05_training.mmd)
- [Experiment lifecycle statechart](diagrams/06_lifecycle.mmd)
- [Held-out evaluation and reporting](diagrams/07_evaluation.mmd)
- [Comparison with previous CoNN experiments](diagrams/08_history.mmd)

Source: [core.py](../../../experiments/experiment_04/src/conn_exp04/core.py), [runner.py](../../../experiments/experiment_04/src/conn_exp04/runner.py),
[analysis.py](../../../experiments/experiment_04/src/conn_exp04/analysis.py), and linked unmodified historical modules.
All diagrams were parsed; Python names were verified. No formal model checking.

[Confirmatory run](../../../experiments/experiment_04/results/run_20261121T031743_424624Z_7ac51c80/), [analysis JSON](../../../experiments/experiment_04/results/run_20261121T031743_424624Z_7ac51c80/analysis.json), [raw endpoint table](../../../experiments/experiment_04/results/run_20261121T031743_424624Z_7ac51c80/endpoint_metrics.csv).
