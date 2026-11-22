# ADR-0002: Versioned mathematical and behavioral components

Status: **Proposed**. Design-freeze candidate **DF-2026-11-22-v1**. Prepared 2026-11-22 against branch initial-repository, HEAD 7a89cd51b1dcfb3526fd9212b0b395cd91317be7. No ADR is accepted and no runtime design is implemented by this documentation.

Labels: **V** verified current file/source fact; **S** static inference; **P** proposed requirement; **U** unresolved evidence or approval. Requirements below are normative only if explicitly approved. Historical provenance is never filled by inference.

Design documents: [architecture decisions](../ARCHITECTURE_DECISIONS.md) | [interface contracts](../INTERFACE_CONTRACTS.md) | [behavioral version matrix](../BEHAVIORAL_VERSION_MATRIX.md) | [reproducibility contract](../REPRODUCIBILITY_CONTRACT.md) | [repository architecture](../REPOSITORY_ARCHITECTURE.md) | [github workflow](../GITHUB_WORKFLOW.md) | [migration and validation plan](../MIGRATION_AND_VALIDATION_PLAN.md).

## Context and problem

Similar formulas conceal differences in domain validation, detach/cast, RNG, precision, statistics and failure policy. Semantic API versions do not establish scientific compatibility.

## Available alternatives

1. One shared implementation and package SemVer only: rejected.
2. Fork every experiment indefinitely: preserves differences but prevents deliberate reuse.
3. Explicit behavioral IDs and complete typed selections, separate from package/schema versions: selected.

## Recommended decision

Adopt the C01-C14 catalog in BEHAVIORAL_VERSION_MATRIX.md. Each future run selects one v2 policy per family and all required variant parameters. IDs are immutable descriptions, never import strings. Bind ID+parameter digest to implementation source digest and reference-suite version; a changed scientific rule gets a new ID/version even with an unchanged signature.

## Supporting evidence

- [resolve_tau](../../../../conn_experiment_01/src/conn_exp/gates.py#L14)
- [resolve_tau](../../../../CoNN_Associative_Relationships/src/conn_assoc/gate.py#L19)
- [Factor.forward](../../../../CoNN_Associative_Relationships/phase3b_context_diagnostics/src/models.py#L19)
- [docs/architecture/PHASE3_BEHAVIORAL_COMPATIBILITY.md](../../../../docs/architecture/PHASE3_BEHAVIORAL_COMPATIBILITY.md)

## Expected benefits

Makes differences reviewable and prevents silent substitutions during extraction.

## Tradeoffs and risks

More metadata and fixtures; some policy variants remain unsupported until validation. No global default may hide a scientific choice.

## Compatibility implications

All 14 future IDs are Proposed and unimplemented. Historical v1 IDs are source-derived descriptors only. API compatibility, source identity, numerical agreement, bitwise replay and scientific replication are different claims.

## Validation requirements

T-C01 through T-C14 with explicit fixtures, environment and acceptance level. No equivalence classification is promoted by static similarity.

## Rollback strategy

Pin consumers to a previously approved behavior/implementation pair. Withdraw an unsuitable new profile without retagging it or changing historical code.

## Approval and supersession

Status: **Proposed**. No acceptance date, approver or runtime implementation exists. Explicit user approval is required before this decision becomes operative. A later accepted or superseding ADR must identify its predecessor; it must not rewrite a historical scientific protocol.
