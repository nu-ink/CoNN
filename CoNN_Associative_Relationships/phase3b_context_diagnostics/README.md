# CoNN Phase3B v3

Isolated contextual-gate diagnostics and bounded exploratory development experiment. Historical files are read-only. No official HAR test evaluation and no Phase4.

From PowerShell in `E:\CoNN`:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$py='E:\CoNN\CoNN_Associative_Relationships\.venv_cuda23\Scripts\python.exe'
$entry='E:\CoNN\CoNN_Associative_Relationships\phase3b_context_diagnostics\scripts\run.py'
& $py $entry preflight
& $py $entry execute
& $py $entry verify
& $py $entry report
```

`execute` also resumes the same run, verifying source seals and existing completed fits. Default run identifier `20261020_phase3b_v3_run01` uses the user-provided local date, not the host timestamp. Pass `--run NAME` to create a separately versioned run; never reuse an existing identifier for a changed design. Run `preflight` first. Once sealed, source/config changes stop execution rather than being incorporated silently. Do not rerun a completed preflight to overwrite its evidence; use a new run for changed diagnostic definitions.

The protocol explicitly resolves v2/v3 differences and implementation choices. `PROTOCOL_PHASE3B_v2.md` is a labeled synopsis of the user-supplied historical design reference, not a claim of an earlier frozen v2 run. `PROTOCOL_PHASE3B_v3.md` is operative. The original v3 request is archived under each run's references directory.

Execution extensions preserve the original training-source seal. `scripts/parallel_tail.py` has a separate hashed manifest and computes disjoint future seed blocks on CPU; it is not a second serial launcher. It queues only while the original runner remains in seed400 and leaves the complete seed401 block as a separation barrier. `scripts/finalize_analysis.py` runs once after the full grid and original `report` stage, adding class/storage/release diagnostics, isolated IH interventions, common-scale attenuation plots, full historical reproduction comparisons and a final preservation audit. Its source and manifest are snapshotted separately before that analysis. No extension changes models, optimizer settings, data, endpoints or training budgets.

Final analysis command:

```powershell
& $py 'E:\CoNN\CoNN_Associative_Relationships\phase3b_context_diagnostics\scripts\finalize_analysis.py'
```

Only one original serial runner should execute a run; launch/resume is a single-owner operation. Completed results can be independently checked with `verify` without re-executing training or optimizer tests.

After automatic finalization, `scripts/measured_summary.py` builds the complete 17-section measured research report from verified CSVs and independently confirms pre/post-release checkpoint weight identity. It preserves the earlier automatic summaries in its own hashed analysis-extension directory before writing the final synthesis. This adds reporting and integrity checks, with no additional optimizer updates or changes to the frozen endpoint. Its exact command is:

```powershell
& $py 'E:\CoNN\CoNN_Associative_Relationships\phase3b_context_diagnostics\scripts\measured_summary.py'
```

Reports distinguish completed fits from incomplete work. Primary and null comparisons require complete paired cells. Diagnostic plots use measured records only. The independent verifier never imports trainer metrics.

The original frozen report writer hit a NumPy-boolean JSON serialization defect after all training and independent verification completed. Its source and all stopped-stage evidence are preserved. Reporting was recovered with `scripts/recover_reporting.py`, which snapshots a one-conversion reporting copy and verifies that contrast, subject and seed CSVs remain byte-identical. It then runs the remaining diagnostics under separately versioned recovery stages. It performs no optimizer updates. The original `run.py report` remains preserved historical source; use the recovery entry point for this run's reporting, and `run.py verify` for independent metric checks:

```powershell
& $py 'E:\CoNN\CoNN_Associative_Relationships\phase3b_context_diagnostics\scripts\recover_reporting.py'
```

Once recovery is complete, that command returns the recorded completion without rerunning training or altering results. See `execution_extensions/recovery_analysis_v1/` for the before/after CSV hashes, five repair checks, preserved failed-report outputs, exact corrected code and acknowledgment of prior guard failures.

A second reporting-only issue was the legacy Windows cp1252 console rejecting the Unicode minus sign after valid UTF-8 files had been written. Both earlier failures remain preserved. The final recovery command explicitly selects UTF-8 console output and records six repair checks under `execution_extensions/recovery_analysis_v2/`:

```powershell
& $py 'E:\CoNN\CoNN_Associative_Relationships\phase3b_context_diagnostics\scripts\recover_reporting_v2.py'
```

Use this v2 recovery command for the final completed run; the v1 recovery entry and failed guard stages are retained as evidence. Both recovery attempts introduced zero optimizer updates. The v2 before/after contrast, subject and seed CSV hashes match exactly.
