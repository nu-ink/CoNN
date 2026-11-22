# ADR-0004: Data ingestion and preprocessing contracts

Status: **Proposed**. Design-freeze candidate **DF-2026-11-22-v1**. Prepared 2026-11-22 against branch initial-repository, HEAD 7a89cd51b1dcfb3526fd9212b0b395cd91317be7. No ADR is accepted and no runtime design is implemented by this documentation.

Labels: **V** verified current file/source fact; **S** static inference; **P** proposed requirement; **U** unresolved evidence or approval. Requirements below are normative only if explicitly approved. Historical provenance is never filled by inference.

Design documents: [architecture decisions](../ARCHITECTURE_DECISIONS.md) | [interface contracts](../INTERFACE_CONTRACTS.md) | [behavioral version matrix](../BEHAVIORAL_VERSION_MATRIX.md) | [reproducibility contract](../REPRODUCIBILITY_CONTRACT.md) | [repository architecture](../REPOSITORY_ARCHITECTURE.md) | [github workflow](../GITHUB_WORKFLOW.md) | [migration and validation plan](../MIGRATION_AND_VALIDATION_PLAN.md).

## Context and problem

HAR has frozen splits/scalers and coordinate conventions; later CV reuses scaling with known overlap. Synthetic families have different generators and paired variants. Files of equal shape are not equivalent datasets.

## Available alternatives

1. A generic array loader with implicit preprocessing: rejected.
2. Continue hardcoded historical paths: insufficient for future portability.
3. Read-only registered DatasetDescriptor, explicit SplitDescriptor/TransformSpec and access grants: selected.

## Recommended decision

Register original and derived data by byte identity and logical IDs. Require row/subject/feature-order/split/scaler fit-population lineage. Loading never fits, downloads, shuffles or rewrites data. A separately authorized preprocessing operation consumes an explicit fit split and writes a new derived artifact. Test audit access is distinct from model-evaluation permission.

## Supporting evidence

- [fit_scaler](../../../../conn_experiment_01/src/conn_exp/har_data.py#L32)
- [load_prepared](../../../../CoNN_Associative_Relationships/src/conn_assoc/data.py#L105)
- [stitch_official_train](../../../../CoNN_Associative_Relationships/src/conn_assoc/data.py#L132)
- [CoNN_Associative_Relationships/audit/normalization_history.md](../../../../CoNN_Associative_Relationships/audit/normalization_history.md)

## Expected benefits

Makes data access, leakage controls and derived versions inspectable.

## Tradeoffs and risks

Descriptors are more detailed than arrays. Some vendor history cannot be reconstructed and remains unknown.

## Compatibility implications

C01.explicit-layout.v2 and C07.explicit-data-lineage.v2 preserve original scalers/coordinates for comparison. Future fold-refit data is a new protocol/transform, not a correction of historical evidence.

## Validation requirements

T-C01/T-C07; exact split/scaler/layout hash fixtures; reject test grants of the wrong scope and transformations fitted outside authorized rows.

## Rollback strategy

Select the prior descriptor/transform for a new attempt; keep both versions and original datasets. No in-place array or scaler replacement.

## Approval and supersession

Status: **Proposed**. No acceptance date, approver or runtime implementation exists. Explicit user approval is required before this decision becomes operative. A later accepted or superseding ADR must identify its predecessor; it must not rewrite a historical scientific protocol.
