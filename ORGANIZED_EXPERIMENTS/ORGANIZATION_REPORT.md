# Organization report

Date: 2026-10-20. Destination: `E:\CoNN\ORGANIZED_EXPERIMENTS`. Original files were read and copied. None were edited, moved, or deleted by this script.

## Counts

| Item | Value |
| --- | --- |
| Experiment directories created | 7 |
| Experiments looked for and not found | Experiment 3A, Experiment 3C, Experiment 3D |
| Files copied | 458 |
| Original artifacts referenced rather than copied | 22 |
| Organization notes created | 47 |
| Copied-file checksum mismatches | 0 |
| Exact duplicate groups among copied files | 77 |
| Discovered files not copied | 44408 |
| Git commit available | no |

Checksum rule: every copied file has `SHA256(original) == SHA256(copy)` recorded in `FILE_PROVENANCE.csv`. A mismatch would have stopped the script. Referenced checkpoint trees and prepared arrays were not hashed file by file. Their total bytes are recorded on the reference rows. The combined result trees are about 33 GB, which is why they were not duplicated.

## Experiments

Organizational status describes whether the packet contains the historical source, the configuration, and a pointer to the results. It does not mean the organized directory can be launched on its own.

| Directory | Status | Why |
| --- | --- | --- |
| EXPERIMENT_01 | PARTIAL | Smoke and pilot summaries are present. The executing source was not snapshotted separately. The copy comes from the later 1B snapshot with 1B-only filenames removed. That exclusion is an inference. |
| EXPERIMENT_01B | COMPLETE | Run snapshot, protocol, selection file, replication scores, and a reference to the full run are present. |
| EXPERIMENT_01C | COMPLETE | Same standard as 1B. The pre-training audit directory is referenced, not merged into 1B. |
| EXPERIMENT_02 | COMPLETE | Snapshot, HAR config, scores, subject split, and references to the archive and prepared arrays are present. |
| EXPERIMENT_03_PHASE01 | AMBIGUOUS | The protocol and audit report are present. The earliest code snapshot already mixes in phase-2 files, which were left out of this directory by filename. |
| EXPERIMENT_03_PHASE23 | COMPLETE | All six source stages are copied separately. Summary tables are copied. The 1050-fit checkpoint tree is referenced. |
| EXPERIMENT_03B | COMPLETE | Frozen run snapshot, protocols, reports, and summary tables are copied. Grid checkpoints are referenced. |

Runtime reproduction was not attempted. Every organized Python tree still imports packages and data through the original paths. Treat `E:\CoNN` as the runnable source of truth.

## Shared dependencies

`SHARED_COMPONENTS` contains a hash map, not a merged library. Experiment 2's `conn_exp` package is imported by the associative phase 2/3 code. Phase 3B imports that associative code. Gate filenames recur with more than one hash. The version map lists them.

## Static reproduction blockers

- No Git revision.
- Organized copies are not installed packages. Entry points in the readmes use the original directories.
- Experiment 01's exact pre-1B bytes are not isolated.
- Phase 0/1 and phase 2 code share one early snapshot.
- Phase 2/3 has six source stages. The run report identifies stage snapshots, but this organization does not assign each of the 1050 fits to a single stage directory.
- Checkpoint and prepared-array hashes were not recomputed.
- Virtual environments were excluded from discovery: `conn_experiment_01\.venv`, `CoNN_Associative_Relationships\.venv_cuda23`, and `.pip-cache23`.

## Duplicates

Copied files that share a SHA-256 are exact duplicates, usually the same protocol or module frozen into more than one snapshot. Count of such groups: 77. Paths are grouped in `FILE_PROVENANCE.csv` by the `sha256` column. Historical files with the same name and different hashes were both kept.

## Excluded files

The complete path list is `EXCLUDED_FILES.csv` (44408 rows). Discovery skipped `AUDIT_20261020`, `ORGANIZED_EXPERIMENTS`, virtual environments, pip caches, pytest caches, and `__pycache__`. Those skipped directories are not rows in the exclusion list.

Main exclusion reasons inside the discovered tree:

- `generated_output_not_duplicated`: run directories, job checkpoints, curves, and plots.
- `large_dataset_referenced_not_copied`: `uci_har_original.zip`.
- `duplicate_protocol_tree_referenced_not_copied`: Phase 3B `protocol_versions` and `protocol_reviews`.
- `phase3b_reporting_extension_not_duplicated`: `execution_extensions` under the Phase 3B run.
- `unassigned_image_referenced_not_copied`: the root infographic.
- `not_selected_for_experiment_packet`: current working copies of modules that are already represented by a frozen snapshot, egg-info, and similar files.

## Recommended future cleanup

Do this only under a separate authorization. Do not do it by editing the organized packets or the historical runs.

1. Put the original tree under Git and record the snapshot hashes already stored in the runs.
2. Keep the current `conn_exp` package from being the only copy of Experiments 01 through 02. The snapshots are the historical versions.
3. Leave checkpoint directories where they are and cite them. Moving them would break the path helpers in `conn_assoc.paths`.
4. Update the associative root README only if a later edit is authorized. This organization did not change it.
