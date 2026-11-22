# Shared components

No module in this directory is a substitute for an experiment snapshot.
Each experiment directory keeps the historical files copied from that experiment's own snapshot.
There is no Git repository, so versions are identified by snapshot path and SHA-256.

Dependency direction, from import statements and path helpers:

- Experiments 01, 01B, 01C, and 02 use the `conn_exp` package in their own snapshots.
- Associative Phase 2/3 imports Experiment 2's `conn_exp` package from `conn_experiment_01/src` through `conn_assoc.paths.experiment2_root`.
- Phase 3B imports `conn_assoc.phase3_context` and `conn_assoc.phase23_core` from the associative project, which in turn import Experiment 2.
- Replacing any of those with a later copy would change the historical experiment.

## Exact-duplicate check across snapshots

Files with the same relative name and the same SHA-256 are exact copies. A shared name with different hashes is a historical change.

### data.py

Distinct hashes: 2.

- `3994c240b0040146c553eeb2c7d36140b9dc7bdff501986712a7a34e7719b557` — `conn_experiment_01\results\runs\20261018T001947Z_experiment1b_v1\source`
- `3994c240b0040146c553eeb2c7d36140b9dc7bdff501986712a7a34e7719b557` — `conn_experiment_01\results\runs\20261018T024146Z_experiment1c_v1\source`
- `3994c240b0040146c553eeb2c7d36140b9dc7bdff501986712a7a34e7719b557` — `conn_experiment_01\results\runs\20261019T004346Z_experiment2_har_v1\source`
- `8c9d7514387df28331095a5863a849a97b3fd8ae1a68ea64cc67e110806ff7eb` — `CoNN_Associative_Relationships\results\runs\20261019T030442340025Z_phase23_v1\source\phase2_pretraining`
- `8c9d7514387df28331095a5863a849a97b3fd8ae1a68ea64cc67e110806ff7eb` — `CoNN_Associative_Relationships\results\runs\20261019T030442340025Z_phase23_v1\source\delivery`

### gate.py

Distinct hashes: 1.

- `8e04a81cf316234a0790f39b07caff07a73c19ac87485a184fecafc9be49e91f` — `CoNN_Associative_Relationships\results\runs\20261019T030442340025Z_phase23_v1\source\phase2_pretraining`
- `8e04a81cf316234a0790f39b07caff07a73c19ac87485a184fecafc9be49e91f` — `CoNN_Associative_Relationships\results\runs\20261019T030442340025Z_phase23_v1\source\delivery`

### gates.py

Distinct hashes: 1.

- `86f3740c1ccc9bc0df1843e4a629621ae66fdd93bf6dafd92bdfe03250c476a7` — `conn_experiment_01\results\runs\20261018T001947Z_experiment1b_v1\source`
- `86f3740c1ccc9bc0df1843e4a629621ae66fdd93bf6dafd92bdfe03250c476a7` — `conn_experiment_01\results\runs\20261018T024146Z_experiment1c_v1\source`
- `86f3740c1ccc9bc0df1843e4a629621ae66fdd93bf6dafd92bdfe03250c476a7` — `conn_experiment_01\results\runs\20261019T004346Z_experiment2_har_v1\source`

### geometry.py

Distinct hashes: 2.

- `9ffe85afe8eca79eeccf3ae2505e16cef38d2fca8eb17f46cf95c3ff97501bba` — `conn_experiment_01\results\runs\20261018T001947Z_experiment1b_v1\source`
- `9ffe85afe8eca79eeccf3ae2505e16cef38d2fca8eb17f46cf95c3ff97501bba` — `conn_experiment_01\results\runs\20261018T024146Z_experiment1c_v1\source`
- `9ffe85afe8eca79eeccf3ae2505e16cef38d2fca8eb17f46cf95c3ff97501bba` — `conn_experiment_01\results\runs\20261019T004346Z_experiment2_har_v1\source`
- `b8de2664ffb61aa29a9d4d5e66e7038fb9d463d140da3f337772c98411394c17` — `CoNN_Associative_Relationships\results\runs\20261019T030442340025Z_phase23_v1\source\phase2_pretraining`
- `b8de2664ffb61aa29a9d4d5e66e7038fb9d463d140da3f337772c98411394c17` — `CoNN_Associative_Relationships\results\runs\20261019T030442340025Z_phase23_v1\source\delivery`

### models.py

Distinct hashes: 2.

- `1cf26d227a61a600e96d5ecb13a1f785b0507d19bfe52edaeba59c1f5f3f47a3` — `conn_experiment_01\results\runs\20261018T001947Z_experiment1b_v1\source`
- `1cf26d227a61a600e96d5ecb13a1f785b0507d19bfe52edaeba59c1f5f3f47a3` — `conn_experiment_01\results\runs\20261018T024146Z_experiment1c_v1\source`
- `1cf26d227a61a600e96d5ecb13a1f785b0507d19bfe52edaeba59c1f5f3f47a3` — `conn_experiment_01\results\runs\20261019T004346Z_experiment2_har_v1\source`
- `f5f00ef1930a91f280ba263f5721bdd1437f49d21472f883269f6d456fd775f9` — `CoNN_Associative_Relationships\phase3b_context_diagnostics\results\20261020_phase3b_v3_run01\source`

### runner.py

Distinct hashes: 1.

- `ae387c61ae7f020f448a43bb4f555d8a94465be0610a0e86575b22c7fe07b34f` — `conn_experiment_01\results\runs\20261018T001947Z_experiment1b_v1\source`
- `ae387c61ae7f020f448a43bb4f555d8a94465be0610a0e86575b22c7fe07b34f` — `conn_experiment_01\results\runs\20261018T024146Z_experiment1c_v1\source`
- `ae387c61ae7f020f448a43bb4f555d8a94465be0610a0e86575b22c7fe07b34f` — `conn_experiment_01\results\runs\20261019T004346Z_experiment2_har_v1\source`

### schedules.py

Distinct hashes: 1.

- `1f1070256510d9b3da3d69b69e9a8ae24b6566f378922c9708d293fdbc7004f9` — `conn_experiment_01\results\runs\20261018T001947Z_experiment1b_v1\source`
- `1f1070256510d9b3da3d69b69e9a8ae24b6566f378922c9708d293fdbc7004f9` — `conn_experiment_01\results\runs\20261018T024146Z_experiment1c_v1\source`
- `1f1070256510d9b3da3d69b69e9a8ae24b6566f378922c9708d293fdbc7004f9` — `conn_experiment_01\results\runs\20261019T004346Z_experiment2_har_v1\source`

### train.py

Distinct hashes: 1.

- `e2e23a97bac1d06254d3ea4b70c1179e8d7248d236adcfe8f006dc05d326ee38` — `conn_experiment_01\results\runs\20261018T001947Z_experiment1b_v1\source`
- `e2e23a97bac1d06254d3ea4b70c1179e8d7248d236adcfe8f006dc05d326ee38` — `conn_experiment_01\results\runs\20261018T024146Z_experiment1c_v1\source`
- `e2e23a97bac1d06254d3ea4b70c1179e8d7248d236adcfe8f006dc05d326ee38` — `conn_experiment_01\results\runs\20261019T004346Z_experiment2_har_v1\source`

