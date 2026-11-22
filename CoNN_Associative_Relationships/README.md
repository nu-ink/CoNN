# CoNN associative relationships

Phase 0 and Phase 1 only. `PROTOCOL_v2.md` is the frozen plan. No model training, context gate, or memory bank is implemented. Experiment 2 under `conn_experiment_01` is read-only.

The protocol hash is in `MANIFEST.json`. There is no Git repository, so that hash is the freeze record.

## Commands

From this directory, using the existing Experiment 2 environment:

```powershell
$env:PYTHONPATH = "src"
..\conn_experiment_01\.venv\Scripts\python.exe -m conn_assoc.inspect
..\conn_experiment_01\.venv\Scripts\python.exe -m pytest
```

`python -m conn_assoc.inspect` rechecks the archive, the prepared-file hashes, the subject partition, and the fit-row normalization, then writes `audit/inspection.json`.

`python -m pytest` runs the gate, data, and protocol tests. A failure is a failed test, not a skip.

## Layout

- `PROTOCOL_v2.md` — frozen questions, controls, splits, and success criteria
- `config/phase01.yaml` — paths and the tau floor for this phase
- `src/conn_assoc/` — read-only data access and the Gaussian gate
- `tests/` — equivalence, leakage, embargo, and hash checks
- `audit/normalization_history.md` — vendor and Experiment 2 preprocessing that cannot be undone
- `results/` — empty until a later phase writes checkpoints, metrics, plots, or relationship maps

## Next commands, after review

Do not run these until Phase 0 and Phase 1 have been accepted. They are not implemented.

1. Implement Dense and Original CoNN only, on the temporal representation, using the seven subject folds and seeds 400–404.
2. Add the row-shuffled and fixed-tau controls after those two conditions repeat.
3. Add context, then memory, each with the tests named in Gates 3 and 4 of the protocol.
