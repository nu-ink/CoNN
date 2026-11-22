# ADR-0001: Historical isolation and future development

Status: **Proposed**. Design-freeze candidate **DF-2026-11-22-v1**. Prepared 2026-11-22 against branch initial-repository, HEAD 7a89cd51b1dcfb3526fd9212b0b395cd91317be7. No ADR is accepted and no runtime design is implemented by this documentation.

Labels: **V** verified current file/source fact; **S** static inference; **P** proposed requirement; **U** unresolved evidence or approval. Requirements below are normative only if explicitly approved. Historical provenance is never filled by inference.

Design documents: [architecture decisions](../ARCHITECTURE_DECISIONS.md) | [interface contracts](../INTERFACE_CONTRACTS.md) | [behavioral version matrix](../BEHAVIORAL_VERSION_MATRIX.md) | [reproducibility contract](../REPRODUCIBILITY_CONTRACT.md) | [repository architecture](../REPOSITORY_ARCHITECTURE.md) | [github workflow](../GITHUB_WORKFLOW.md) | [migration and validation plan](../MIGRATION_AND_VALIDATION_PLAN.md).

## Context and problem

Historical snapshots, failed attempts and working imports overlap. Copied packets are not portable execution roots. A cleaner framework must not rewrite the evidence it intends to support.

## Available alternatives

1. Refactor original trees in place: few files but loses direct byte-reference boundaries.
2. Copy all experiments into a new canonical tree: preserves copies but duplicates evidence and leaves path-coupling unresolved.
3. Keep originals at original paths and develop a separate opt-in future package: selected.

## Recommended decision

Preserve all existing research paths and bytes. Proposed future src/conn and tests/ infrastructure is additive; historical scripts are never redirected or monkeypatched to the new package. Permit legacy artifact readers only through explicit, versioned adapters. No historical launch becomes a supported replay merely by import success.

## Supporting evidence

- [ORGANIZED_EXPERIMENTS/ORGANIZATION_REPORT.md](../../../../ORGANIZED_EXPERIMENTS/ORGANIZATION_REPORT.md)
- [experiment2_root](../../../../CoNN_Associative_Relationships/src/conn_assoc/paths.py#L13)
- [docs/architecture/PHASE3_RISK_ASSESSMENT.md](../../../../docs/architecture/PHASE3_RISK_ASSESSMENT.md)

## Expected benefits

Preserves research traceability and makes rollback a return to unchanged original callers.

## Tradeoffs and risks

Two code generations coexist. Duplicate-looking source stays intentionally preserved. Maintenance support must be declared per profile.

## Compatibility implications

C07.explicit-data-lineage.v2 and C14.explicit-artifact.v2 describe future contracts. Historical reads must keep original data/scaler/source identities and unresolved fields; no future profile is automatically assigned to old runs.

## Validation requirements

T-M01 preservation baseline, T-R01 output/import isolation and T-C07/T-C14. Manual review of all source/data paths before an adapter is enabled.

## Rollback strategy

Disable the new adapter/package for callers; preserve any new outputs as separate attempts. Do not restore history by editing or deleting research files.

## Approval and supersession

Status: **Proposed**. No acceptance date, approver or runtime implementation exists. Explicit user approval is required before this decision becomes operative. A later accepted or superseding ADR must identify its predecessor; it must not rewrite a historical scientific protocol.
