# ADR-0008: GitHub development and release practices

Status: **Proposed**. Design-freeze candidate **DF-2026-11-22-v1**. Prepared 2026-11-22 against branch initial-repository, HEAD 7a89cd51b1dcfb3526fd9212b0b395cd91317be7. No ADR is accepted and no runtime design is implemented by this documentation.

Labels: **V** verified current file/source fact; **S** static inference; **P** proposed requirement; **U** unresolved evidence or approval. Requirements below are normative only if explicitly approved. Historical provenance is never filled by inference.

Design documents: [architecture decisions](../ARCHITECTURE_DECISIONS.md) | [interface contracts](../INTERFACE_CONTRACTS.md) | [behavioral version matrix](../BEHAVIORAL_VERSION_MATRIX.md) | [reproducibility contract](../REPRODUCIBILITY_CONTRACT.md) | [repository architecture](../REPOSITORY_ARCHITECTURE.md) | [github workflow](../GITHUB_WORKFLOW.md) | [migration and validation plan](../MIGRATION_AND_VALIDATION_PLAN.md).

## Context and problem

The branch initial-repository and conservative attributes/ignore rules now exist. New development needs review and release provenance without rewriting the initial research capture.

## Available alternatives

1. Direct changes on initial-repository: rejected for new runtime development.
2. Automatically rename branches or enable protections now: outside this phase.
3. Propose reviewed main, scoped feature branches, scientific review and immutable release identities: selected.

## Recommended decision

After separate authorization, use main for reviewed future framework/docs and retain initial-repository as the baseline branch. Follow GITHUB_WORKFLOW.md: scoped branches, reviewed PRs, no direct protected-branch changes, unique required checks, declared reviewers, source/environment/provenance manifests and immutable package/scientific tags. Confirm remote capabilities and owner assignments before configuring rules. Keep large artifacts local/archival by current policy unless storage and publication are explicitly approved.

## Supporting evidence

- [.gitattributes](../../../../.gitattributes)
- [.gitignore](../../../../.gitignore)
- [registry/README.md](../../../../registry/README.md)
- [docs/architecture/PHASE3_RISK_ASSESSMENT.md](../../../../docs/architecture/PHASE3_RISK_ASSESSMENT.md)

## Expected benefits

Makes scientific and engineering changes reviewable and associates releases with evidence.

## Tradeoffs and risks

Review availability and private-repository feature entitlements are unresolved. Git attributes can change checkout bytes; raw evidence requires separate preserved-byte identities.

## Compatibility implications

Package tags do not guarantee C01-C14 behavior compatibility. No history migration, renormalization, LFS conversion, remote setting change, staging, commit, push or branch switch is authorized here.

## Validation requirements

T-G01 proposed CI/PR/release checks; T-M01 preservation; manual repository-owner and scientific approval. Current Git branch/HEAD/index checks remain read-only.

## Rollback strategy

Revert only new reviewed development changes via later approved corrective commits; do not force-update tags/history. Disable a faulty future release and issue a new version with explicit withdrawal/correction notes.

## Approval and supersession

Status: **Proposed**. No acceptance date, approver or runtime implementation exists. Explicit user approval is required before this decision becomes operative. A later accepted or superseding ADR must identify its predecessor; it must not rewrite a historical scientific protocol.
