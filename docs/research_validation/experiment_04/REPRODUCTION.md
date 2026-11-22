# Reproduction

Use the existing environment, from `E:\CoNN`. No packages were installed.
Recorded environment: Python 3.10.1, PyTorch 2.5.1+cu121, NumPy 1.26.4,
SciPy 1.15.2, pytest 8.3.5; CPU execution despite available CUDA.
`conn_experiment_01/.venv` has no SciPy, and the separate Python 3.11 launcher has no Torch.
The default `python` executable recorded in every run is the environment used.
Hardware, system, full version provenance and imported source hashes appear in
each run's `environment.json`. Bitwise replay is tested within this environment;
cross-version or cross-machine bitwise identity is not promised.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='E:\CoNN\experiments\experiment_04\src'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m pytest experiments/experiment_04/tests -q -p no:cacheprovider
python -m conn_exp04.runner train --config experiments/experiment_04/config/smoke.json
# Use the unique RUN path printed above:
python -m conn_exp04.runner evaluate --run <SMOKE_RUN>
python -m conn_exp04.runner train --config experiments/experiment_04/config/pilot.json
# Pilot is for training/validation checks; do not evaluate its test split.
python -m conn_exp04.runner train --config experiments/experiment_04/config/confirmatory.json
python -m conn_exp04.runner evaluate --run <CONFIRMATORY_RUN>
python -m conn_exp04.analysis --run <CONFIRMATORY_RUN>
python experiments/experiment_04/scripts/verify_artifacts.py <CONFIRMATORY_RUN>
```

Config loader validates all supplied fields; no hidden tuning phase or seed replacement exists.
Each train command creates a timestamp-plus-UUID directory. Do not rerun `prepare.py`;
it intentionally refuses to overwrite initial configurations. Evaluation refuses an
already evaluated run. Analysis refuses an existing analysis file. Reproduction uses
a new run directory, preserving original measurements. Failed runs remain failed;
there is no silent checkpoint resume.

Frozen main config SHA-256:
`ff17dd4e5ba36c4eec8e63f11b9db4489611fa0ee1317203d280f5613d67ff42`.
The immutable run config, original source snapshot, initial states, batches, gate
permutations/distances, train/validation samples, checkpoints, optimizer state,
test labels and logits allow audit without retraining. Seeds for data/model/order/
mask/training are recorded separately in each fit's provenance. Test RNG seeds
and sample hashes appear in endpoint rows. Saved checkpoints belong to this
locally generated run; no untrusted historical checkpoint was loaded.

To validate diagrams again, `python experiments/experiment_04/scripts/validate_diagrams.py`.
It uses existing Playwright and local Chrome plus a pinned Mermaid browser script
from jsDelivr. It installs no packages, uploads no diagram content, and blocks network
requests after the renderer loads. The statechart is parsed without a pstack render.
The diagram generator and documentation report intentionally create files exclusively;
rebuild them into a new destination or preserve outputs before a deliberate regeneration.
