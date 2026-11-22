# Phase 2/3 development implementation

The historical README, protocol v2 and audit are preserved unchanged. The user
authorized this extension in `PROTOCOL_PHASE23_v1.md`, sealed before training in
the new run's `protocol_seal.json`. Phase 4 and official-test evaluation remain
outside this implementation. No memory bank or retrieval metrics exist.

The active unique run is named in `config/phase23_active_run.txt`.

```powershell
Set-Location E:\CoNN\CoNN_Associative_Relationships
$run = (Get-Content config/phase23_active_run.txt).Trim()
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = 'src;..\conn_experiment_01\src'
```

Resume the bounded full development grid using the original CPU environment:

```powershell
..\conn_experiment_01\.venv\Scripts\python.exe scripts/run_phase23.py grid --stage all --workers 6 --devices cpu
```

Completed jobs are verified against their configuration and reused. Partial jobs
restore their latest saved model/optimizer/RNG/step/batch/context state. Latest
is saved every 160 updates; an abrupt process failure may repeat the unsaved
updates, reproducing them from the restored state. Do not launch a second runner
against the same run concurrently. New research protocols should use a new
versioned preflight, not overwrite this run or reuse its selection data as an
independent replication. The preflight deliberately fails if protected original
Experiment 2 hashes have changed.

After the complete 1050-fit grid:

```powershell
..\conn_experiment_01\.venv\Scripts\python.exe -m conn_assoc.phase23_analysis full --run $run
..\conn_experiment_01\.venv\Scripts\python.exe -m conn_assoc.phase23_diagnostics --run $run
..\conn_experiment_01\.venv\Scripts\python.exe -m conn_assoc.phase23_verify --run $run
```

Tests in either environment:

```powershell
..\conn_experiment_01\.venv\Scripts\python.exe -m pytest -v -rs -o "cache_dir=$run/pytest_review_cpu"
.\.venv_cuda23\Scripts\python.exe -m pytest -v -rs -o "cache_dir=$run/pytest_review_cuda"
```

The CPU-only environment explicitly skips the two CUDA checks as NOT RUN; the
isolated CUDA environment executes them. GPU availability is not sufficient:
`cuda_execution.json` records tensor/matmul/backprop/transfer tests on both cards.
Benchmarking is sequential and warmed up. CPU was faster for this small MLP;
the grid began with six one-thread CPU workers and resumed with eight after a
checkpoint replacement error, with contention-labelled runtime. The checkpoint
repair retries transient replacement failures without deleting the last good file.
Two cards are independent devices, never a combined 16 GB resource.

All development evaluation loads only the 7352 official training rows. The fixed
Experiment 2 scaler is retained; some CV validation subjects contributed to it.
LR selection reuses the same development folds later summarized, so reported
bootstrap intervals are descriptive and not confirmatory. Context models have
8064 extra parameters and near-one modulation; inspect actual changes and
same-weight interventions before attributing any gain to external context.
