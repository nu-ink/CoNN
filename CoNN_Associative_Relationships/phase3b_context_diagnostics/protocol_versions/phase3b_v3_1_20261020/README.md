# Phase 3B v3.1 prospective amendment

The operative design in this version directory is PROTOCOL_PHASE3B_v3.md, accompanied by protocol_config.json, the user's exact A1–A3 decision text, reference snapshots, a checksum and PROTOCOL_SEAL.json. It consolidates v2, v3 and the user-approved prospective resolutions. The older root v3 file and completed run remain unchanged.

This freezes the design for a future versioned experiment. It does not reclassify any earlier result and is not independent confirmation. No model training or trained-checkpoint reanalysis was performed while preparing it.

Before execution, implement the prospective rules in versioned source and verify numerical correctness, pair/layer-specific live support, isolated interventions, all required persistent pilot checkpoints, reproducibility and source/data integrity. Measure backend and execution feasibility, record the budget and seal source/config/environment/data before performance training; seal the full-grid execution manifest after complete pilots and the independent architecture-specific C3 decisions. The historical root launcher is not an implementation of this amendment. No official HAR test evaluation or Phase 4 is permitted.

Verify the immutable design seal without loading ML libraries:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
& E:\CoNN\CoNN_Associative_Relationships\.venv_cuda23\Scripts\python.exe -X utf8 E:\CoNN\CoNN_Associative_Relationships\phase3b_context_diagnostics\protocol_versions\phase3b_v3_1_20261020\verify_and_seal.py
```

The script checks document/configuration consistency and selected historical hashes, and refuses to replace a mismatched existing seal. It does not train, evaluate predictions or certify a future implementation.
