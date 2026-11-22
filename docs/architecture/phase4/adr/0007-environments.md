# ADR-0007: Dependency and environment management

Status: **Proposed**. Design-freeze candidate **DF-2026-11-22-v1**. Prepared 2026-11-22 against branch initial-repository, HEAD 7a89cd51b1dcfb3526fd9212b0b395cd91317be7. No ADR is accepted and no runtime design is implemented by this documentation.

Labels: **V** verified current file/source fact; **S** static inference; **P** proposed requirement; **U** unresolved evidence or approval. Requirements below are normative only if explicitly approved. Historical provenance is never filled by inference.

Design documents: [architecture decisions](../ARCHITECTURE_DECISIONS.md) | [interface contracts](../INTERFACE_CONTRACTS.md) | [behavioral version matrix](../BEHAVIORAL_VERSION_MATRIX.md) | [reproducibility contract](../REPRODUCIBILITY_CONTRACT.md) | [repository architecture](../REPOSITORY_ARCHITECTURE.md) | [github workflow](../GITHUB_WORKFLOW.md) | [migration and validation plan](../MIGRATION_AND_VALIDATION_PLAN.md).

## Context and problem

Source-path insertion, bare Phase 3B imports and undeclared transitive packages undermine portability. Historical Python/toolchain profiles differ.

## Available alternatives

1. One current global environment: hides compatibility differences.
2. Retroactively reinstall all historical environments: changes evidence and is prohibited.
3. One future distribution with optional analysis/rendering groups and pinned per-profile environments: selected.

## Recommended decision

Future modules use package-qualified imports and explicit object/data references. Disallow sys.path mutation, workspace sibling discovery and module globals patched by callers. Declare direct dependencies by layer; core imports never render reports, initialize CUDA or mutate process settings. Initial future interpreter target is Python 3.11, with exact dependency/build/runtime versions chosen and pinned at stage M2 after source/profile verification. Historical Python 3.10/3.11 environments remain unchanged.

## Supporting evidence

- [conn_experiment_01/pyproject.toml](../../../../conn_experiment_01/pyproject.toml)
- [CoNN_Associative_Relationships/pyproject.toml](../../../../CoNN_Associative_Relationships/pyproject.toml)
- [docs/research_validation/experiment_04/REPRODUCTION.md](../../../../docs/research_validation/experiment_04/REPRODUCTION.md)
- [configure](../../../../CoNN_Associative_Relationships/src/conn_assoc/phase23_core.py#L73)

## Expected benefits

Explicit closure and optional groups reduce hidden dependencies and improve isolated testing.

## Tradeoffs and risks

More than one environment may remain necessary. The chosen interpreter target is a design proposal, not a compatibility claim or completed installation.

## Compatibility implications

C13.explicit-runtime.v2 records backend/device/thread/determinism policy. No historical profile is upgraded by a package release; cross-environment bitwise identity is not promised.

## Validation requirements

T-E01 isolated import/dependency checks, T-C13 runtime fixtures and environment hash capture. No install/update is performed until separately authorized.

## Rollback strategy

Use the previously approved lock/profile for a new attempt. Keep old environments untouched; rollback is profile selection, not dependency downgrades in historical environments.

## Approval and supersession

Status: **Proposed**. No acceptance date, approver or runtime implementation exists. Explicit user approval is required before this decision becomes operative. A later accepted or superseding ADR must identify its predecessor; it must not rewrite a historical scientific protocol.
