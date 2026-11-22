# Experiment 04 Implementation Plan

**PROPOSED — NOT IMPLEMENTED. No implementation is authorized by this document.**

## Proposed module layout

- `src/conn_exp04/data.py`: synthetic task and leakage guards;
- `src/conn_exp04/geometry.py`: ground-truth relationship coordinates;
- `src/conn_exp04/gates.py`: aligned, uniform and scrambled gates;
- `src/conn_exp04/models.py`: dense MLP and gated MLP;
- `src/conn_exp04/train.py`: paired SGD loop;
- `src/conn_exp04/evaluation.py`: validation-only selection and frozen test evaluation;
- `src/conn_exp04/diagnostics.py`: gradients, effective weights and ablations;
- `src/conn_exp04/experiment04.py`: sealed runner;
- `tests/`: invariants, replay, energy matching and test-evaluation counts.

## Required freezes

Before any fit:

- task and split generator;
- seed streams;
- condition definitions;
- geometry derivation;
- LR/duration grid;
- test protocol;
- report schema;
- success criteria.

## Required gates

No test access during tuning. Independent replication only after LR/duration freeze. Save protocol, data, masks, initial states, optimizer state, curves and checkpoints for each completed fit.

