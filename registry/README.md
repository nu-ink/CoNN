# CoNN registry maintenance

Verified against local workspace files on **2026-11-22**, on branch `initial-repository`. This registry adds navigation while preserving existing research locations. It records what the available evidence establishes and leaves missing facts unresolved.

## Files and coverage

| File | Contents |
| --- | --- |
| [experiments.yaml](experiments.yaml) | Eight requested experiment families, purposes, recorded statuses, existing implementations, source dependencies, dataset IDs and evidence |
| [dependencies.yaml](dependencies.yaml) | 24 verified static dependency edges, six dataset/provenance records and unresolved dependency questions |
| [runs.csv](runs.csv) | 32 records: 14 primary study/audit runs, three execution attempts, 14 execution extensions and one failure event |
| [Root README](../README.md) | Research objectives, history, current implementation scope and navigation |

Paths inside YAML/CSV are workspace-root-relative, use forward slashes, and retain original names. Markdown links are relative to the Markdown file containing them. These conventions differ deliberately: do not resolve YAML paths relative to `registry/`.

Primary run discovery covered direct study/audit directories under:

- `conn_experiment_01/results/runs`
- `CoNN_Associative_Relationships/results/runs`
- `CoNN_Associative_Relationships/phase3b_context_diagnostics/results`
- `experiments/experiment_04/results`

Nested Phase 2/3 `execution` attempts, Phase 3B `execution_extensions`, and the separate Phase 3B failure ledger are indexed with distinct record kinds. Per-fit seed/fold/condition directories, checkpoints, tuning/replication partitions, unit-test temporary directories, CUDA/benchmark checks, recovery evidence bundles, and organized copies are not counted as separate study runs. Date-named protocol reviews are review artifacts, not executed studies. This is a study/attempt registry, not a row for every optimizer fit or a complete filesystem inventory.

## Evidence and status rules

A **verified static finding** means the cited local file was inspected or a source statement/path was checked. It does not imply that code executed correctly, that a saved scientific claim was recomputed, or that all artifacts were rehashed. SHA-256 values identify the cited evidence files as read during registry creation; they do not authenticate an author or provide an external timestamp.

A **completed study as recorded** has a local completion/status record supporting that state. Completion is separate from scientific success, test evaluation, independent reproduction, or production readiness. Do not replace an explicit failure or unevaluated state with the final parent's completion state.

A **historical source snapshot** belongs to a particular run or stage. An **existing implementation** is code presently available at the documented location. Neither designation establishes one authoritative implementation for all studies or shows that a current implementation matches a historical executing version.

A **proposal or post-outcome review** is specification/review material. It is not an executed experiment or a retroactive preregistration. Preserve this distinction when a proposal later gains a separate implementation.

An **assumption** is an unverified interpretation and must be labeled as such. Each experiment has an `assumptions` list; empty lists mean no unsupported assumptions were intentionally recorded, not that every historical detail is known. Missing or conflicting facts belong in `limitations_and_unresolved` or an explicit unresolved status.

`independent_reproduction: not_established` and CSV `independent_reproduction_status=not_established` mean this registry does not establish independent reproduction. They do not prove that no such reproduction exists elsewhere. New seeds, local verification scripts, document titles containing 'independent', and repeated local training do not alone establish external independent reproduction.

## Experiment and dependency schemas

Both YAML files declare `schema_version`, `verified_on` and `path_base`. The experiment file records the branch observed during verification. These are capture metadata, not assertions about the branch or filesystem at all future dates.

Experiment IDs are stable registry identifiers. Each experiment records its purpose, status and basis; implementation path and entry point; source dependency paths; dataset IDs; historical source paths; cited evidence with SHA-256; verification limits; assumptions; and unresolved issues. Shared implementation paths are expected for Experiments 01 through 02 and the associative phases.

Dependency edges have unique IDs and directed `from_path`/`to_path` values: the consumer points to its dependency. `relation` distinguishes imports, source-path insertion, raw/prepared inputs, historical-run references and configuration pointers. Each edge includes a literal source/document excerpt, its one-based line number and the evidence file hash. An import was verified statically; runtime module resolution was not executed. The edge list covers important cross-directory relationships and is explicitly not every intra-package import.

Dataset IDs are referenced by experiments. `parent_dataset_ids` describe documented provenance, not a transformation performed for this registry. Prepared HAR arrays retain their original scalers, coordinates, manifests and subject partition. Experiment 04's synthetic data is distinct from HAR. Shared data or paired control variants must not be described as independent datasets merely because they occupy different directories.

## CSV contract

CSV is UTF-8 with a header and ordinary CSV quoting. Read it with a CSV parser. `original_relative_path` is the unique record key; `run_id` is the original directory basename or failure-event filename, not a promised globally unique identifier. `parent_run_id` identifies the enclosing primary study for indexed execution attempts/extensions. A failure event with an unresolved launch identity intentionally has no assigned parent.

| Column | Meaning |
| --- | --- |
| `run_id` | Original basename; do not invent a timestamp from it |
| `experiment_id` | Foreign key into `experiments.yaml` |
| `record_kind` | `study_run`, `execution_attempt`, `execution_extension`, or `failure_event` |
| `parent_run_id` | Enclosing indexed study run where established; empty otherwise |
| `original_relative_path` | Preserved directory path, or failure-event file path |
| `status` | Explicit recorded or unresolved state |
| `status_basis` | Why that status is supported |
| `test_evaluation_status` | Test use as recorded, independently of completion |
| `independent_reproduction_status` | Evidence level for independent reproduction |
| `evidence_path` | Primary status evidence file |
| `evidence_sha256` | SHA-256 of that file at registry capture |
| `supporting_evidence_paths` | Additional original paths separated by `|`; empty is allowed |
| `notes` | Scope, exceptions and interpretation limits |

Current statuses include `completed_as_recorded`, `development_completed_as_recorded`, `diagnostic_study_completed_as_recorded`, `audit_completed_as_recorded`, `outputs_present_completion_unresolved`, `audit_attempt_completion_unresolved`, `failed_as_recorded`, `trained_as_recorded`, `evaluated_as_recorded`, `failure_recorded_completion_unresolved`, and `failure_recorded`. Future unknown/conflicting evidence should receive an explicit unresolved status rather than a guessed successful state.

The first Experiment 04 smoke is failed as recorded. Its pilot remains trained because test evaluation was deliberately withheld. Older smoke/pilot directories contain summaries without explicit completion markers, so their completion remains unresolved. The early Phase 0/1 audit attempts also lack final completion markers. Some Phase 3B failure files contain a non-JSON prefix despite a `.json` extension; preserve their bytes and treat them as text evidence rather than silently repairing them.

## Updating the registry

1. Inspect the original protocol/configuration, source imports/path helpers, run status/completion and supporting evidence. Use the prior registry as a discovery aid, not as the source of new facts.
2. Retain stable IDs and original relative paths. Add a new study/run record for genuinely new work; do not turn an old proposal, omitted fit or failed attempt into an executed study by renaming its record.
3. Cite the evidence and update capture hashes only after reviewing an actual change. A hash mismatch requires investigation; do not automatically refresh it. Historical evidence stays untouched.
4. Record source, dataset, preprocessing/split and historical-run dependencies separately. Identify the exact source revision/snapshot where supported, and leave per-fit stage assignments unresolved where evidence is insufficient.
5. Keep completion, evaluation policy and reproduction level separate. Promote an independent-reproduction claim only with attributable executor/environment, exact source and data identity, procedure, saved outputs and a supported comparison against the original result.
6. Validate syntax, IDs, relationships, hashes, original-path existence and the new Markdown links. Review the five-file diff. Do not run project commands merely because their name contains 'verify': some write artifacts or execute optimizer tests.
7. Do not stage, commit, push, install software, rerun experiments, edit evidence or reorganize directories without the relevant explicit authorization. Registry maintenance itself does not authorize migration.

## Validation and preservation for this creation

Validation uses the already installed YAML parser and Python standard-library CSV/path/hash utilities, without importing research packages. Checks cover YAML round trips; unique experiment, dataset and dependency IDs; CSV row width and unique original paths; experiment/dataset/parent-run foreign keys; local path existence; evidence hashes and dependency excerpts; and local Markdown destinations in the two new READMEs. No external links or historical-document link repair is included. No experiments, research tests, checkpoint loading or environment changes are part of these checks.

A pre-write in-memory baseline covers existing research/support file names, sizes and modification times, content hashes for existing source/document/configuration files, and the Git index/staged-diff checksums. The five new files are excluded from the preservation comparison. Existing staged and untracked work is preserved; the registry files are not automatically staged.

## Questions still unresolved

- Exact original executing bytes for the first smoke/pilot and the clean Phase 0/1 boundary.
- Exact Phase 2/3 source-stage lineage for every fit, especially resumed execution.
- Portable, pinned transitive environments for independent replay of every study.
- Externally attributable independent reproduction of the scientific findings.
- The Phase 3B future-freeze definitions documented in the [protocol reconciliation](../CoNN_Associative_Relationships/phase3b_context_diagnostics/reports/PROTOCOL_RECONCILIATION_20261020.md).

See also the [recorded Experiment 04 reproduction procedure](../docs/research_validation/experiment_04/REPRODUCTION.md), [historical organization report](../ORGANIZED_EXPERIMENTS/ORGANIZATION_REPORT.md), and [prior audit](../AUDIT_20261020/FINAL_AUDIT_SUMMARY.md). Their commands and historical claims were not executed or independently reproduced for this registry.
