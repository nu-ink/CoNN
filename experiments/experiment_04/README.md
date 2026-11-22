# Experiment 04 — Relationship Geometry Validation

Isolated implementation of a preregistered four-condition comparison on a known
latent-interaction task and a negative geometry control. No original CoNN source changed.

**Completed:** 29 new tests + 32 historical regressions passed; smoke 1 seed / 8 fits;
pilot 2 seeds / 16 fits (test unread); confirmatory 10 seeds / 80 fits / 240 test endpoints.

Aligned versus scrambled final test accuracy: 77.54% versus 76.41%; paired +1.13 pp,
95% CI [0.22,2.05] pp, one-sided exact p=.01465. Practical 2 pp criterion unmet.
Uniform secondary contrast: Holm p=.1172. No learned associations were implemented.

[Full report](../../docs/research_validation/experiment_04/README.md),
[preregistration](PREREGISTRATION.md), [technical log](TECHNICAL_LOG.md).

Structure: `config/` frozen JSON; `src/conn_exp04/` DGP/controls/runner/statistics;
`tests/` automated checks; `scripts/` preparation, figures/report, diagrams and verification;
`results/` unique run directories with frozen config, provenance, checkpoints and measurements.
Use the [reproduction instructions](../../docs/research_validation/experiment_04/REPRODUCTION.md).
The user waived Git checkout; all original source/document/config hashes remain unchanged.
