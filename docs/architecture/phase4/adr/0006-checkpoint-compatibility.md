# ADR-0006: Checkpoint and model compatibility

Status: **Proposed**. Design-freeze candidate **DF-2026-11-22-v1**. Prepared 2026-11-22 against branch initial-repository, HEAD 7a89cd51b1dcfb3526fd9212b0b395cd91317be7. No ADR is accepted and no runtime design is implemented by this documentation.

Labels: **V** verified current file/source fact; **S** static inference; **P** proposed requirement; **U** unresolved evidence or approval. Requirements below are normative only if explicitly approved. Historical provenance is never filled by inference.

Design documents: [architecture decisions](../ARCHITECTURE_DECISIONS.md) | [interface contracts](../INTERFACE_CONTRACTS.md) | [behavioral version matrix](../BEHAVIORAL_VERSION_MATRIX.md) | [reproducibility contract](../REPRODUCIBILITY_CONTRACT.md) | [repository architecture](../REPOSITORY_ARCHITECTURE.md) | [github workflow](../GITHUB_WORKFLOW.md) | [migration and validation plan](../MIGRATION_AND_VALIDATION_PLAN.md).

## Context and problem

Binary/multiclass/context heads, parameter dtypes and optimizer groups differ. Endpoint checkpoints are not necessarily complete resume records; loader modes and RNG schemas differ.

## Available alternatives

1. One loader that guesses format from keys: rejected.
2. Convert every historical checkpoint into a new format: unnecessary evidence risk.
3. New explicit checkpoint envelope and opt-in schema-specific legacy readers: selected.

## Recommended decision

Define conn-checkpoint-1 as manifest metadata plus separately hashed payloads. Distinguish endpoint_only and resumable. Validate model behavior/head/state keys/shapes/dtypes, optimizer groups, all required RNG/iterator/context state, source closure and update position before deserialization/restore. Select readers from a trusted fixed schema map; no class name import from artifact text. Conversions, if later approved, are new artifacts with original parent hashes.

## Supporting evidence

- [_save_checkpoint](../../../../conn_experiment_01/src/conn_exp/train.py#L139)
- [rng_state](../../../../CoNN_Associative_Relationships/src/conn_assoc/phase23_core.py#L193)
- [save_checkpoint](../../../../experiments/experiment_04/src/conn_exp04/runner.py#L73)
- [docs/architecture/PHASE3_BEHAVIORAL_COMPATIBILITY.md](../../../../docs/architecture/PHASE3_BEHAVIORAL_COMPATIBILITY.md)

## Expected benefits

Makes unsupported resume fail explicitly and protects model/schema identity.

## Tradeoffs and risks

Readers and payload codec choices need separate validation; tensor-only historical loading is not assumed possible.

## Compatibility implications

C03.explicit-application.v2, C11.explicit-initialization.v2, C12.explicit-context.v2 and C14.explicit-artifact.v2 bind model semantics to checkpoint identity. No checkpoint conversion or deserialization happens in this phase.

## Validation requirements

T-C03/C11/C12/C14; schema mismatch negatives; separately authorized roundtrip and interrupted/resume fixtures. Default verifier checks metadata and bytes only.

## Rollback strategy

Keep the old reader and original payload; disable the new reader. Never overwrite a historical checkpoint or falsely label endpoint_only as resumable.

## Approval and supersession

Status: **Proposed**. No acceptance date, approver or runtime implementation exists. Explicit user approval is required before this decision becomes operative. A later accepted or superseding ADR must identify its predecessor; it must not rewrite a historical scientific protocol.
