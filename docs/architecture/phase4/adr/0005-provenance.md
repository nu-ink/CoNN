# ADR-0005: Experiment and artifact provenance

Status: **Proposed**. Design-freeze candidate **DF-2026-11-22-v1**. Prepared 2026-11-22 against branch initial-repository, HEAD 7a89cd51b1dcfb3526fd9212b0b395cd91317be7. No ADR is accepted and no runtime design is implemented by this documentation.

Labels: **V** verified current file/source fact; **S** static inference; **P** proposed requirement; **U** unresolved evidence or approval. Requirements below are normative only if explicitly approved. Historical provenance is never filled by inference.

Design documents: [architecture decisions](../ARCHITECTURE_DECISIONS.md) | [interface contracts](../INTERFACE_CONTRACTS.md) | [behavioral version matrix](../BEHAVIORAL_VERSION_MATRIX.md) | [reproducibility contract](../REPRODUCIBILITY_CONTRACT.md) | [repository architecture](../REPOSITORY_ARCHITECTURE.md) | [github workflow](../GITHUB_WORKFLOW.md) | [migration and validation plan](../MIGRATION_AND_VALIDATION_PLAN.md).

## Context and problem

Existing registry indexes studies/attempts and many local seals exist, but source-to-fit lineage and full transitive closure are uneven. Analysis repairs have separate sources.

## Available alternatives

1. Use Git commit alone: insufficient for dirty/local dependencies and artifacts.
2. Use directory timestamps or final delivery snapshots: insufficient for per-attempt execution.
3. Immutable source/data/config/artifact manifests plus append-only execution events: selected.

## Recommended decision

Require REPRODUCIBILITY_CONTRACT.md fields for every future run, fit and attempt. SourceClosure includes explicit transitive project sources, clean commit for confirmatory runs and per-file hashes. Link attempts and analysis repairs to parents; store immutable terminal events and derive summaries. Registry remains reviewed navigation, with future additive manifest references rather than silent schema mutation.

## Supporting evidence

- [registry/README.md](../../../../registry/README.md)
- [registry/runs.csv](../../../../registry/runs.csv)
- [load](../../../../CoNN_Associative_Relationships/phase3b_context_diagnostics/scripts/recover_reporting_v2.py#L20)
- [docs/architecture/PHASE3_RISK_ASSESSMENT.md](../../../../docs/architecture/PHASE3_RISK_ASSESSMENT.md)

## Expected benefits

Separates execution success, verification and scientific interpretation while preserving failure lineage.

## Tradeoffs and risks

Storage and metadata overhead; artifact registration needs atomicity and concurrency control. A missing provenance field blocks a new conforming run rather than being guessed.

## Compatibility implications

C08.strict-config.v2 and C14.explicit-artifact.v2 define separate digest namespaces. Old unknown fields remain explicitly unresolved in derived catalog annotations; originals are untouched.

## Validation requirements

T-P01 mandatory fields and event-chain validation; T-P02 incomplete/repair/resume cases; T-M01 byte preservation. No checkpoint execution is needed for schema-only checks.

## Rollback strategy

Retain immutable records; publish a new corrected projection with predecessor references. Disable a faulty registrar and resume from the last valid event; never edit the old ledger to hide a failure.

## Approval and supersession

Status: **Proposed**. No acceptance date, approver or runtime implementation exists. Explicit user approval is required before this decision becomes operative. A later accepted or superseding ADR must identify its predecessor; it must not rewrite a historical scientific protocol.
