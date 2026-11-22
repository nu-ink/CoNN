# Phase 4 architecture decisions and freeze candidate

Status: **Proposed**. Design-freeze candidate **DF-2026-11-22-v1**. Prepared 2026-11-22 against branch initial-repository, HEAD 7a89cd51b1dcfb3526fd9212b0b395cd91317be7. No ADR is accepted and no runtime design is implemented by this documentation.

Labels: **V** verified current file/source fact; **S** static inference; **P** proposed requirement; **U** unresolved evidence or approval. Requirements below are normative only if explicitly approved. Historical provenance is never filled by inference.

Design documents: [architecture decisions](ARCHITECTURE_DECISIONS.md) | [interface contracts](INTERFACE_CONTRACTS.md) | [behavioral version matrix](BEHAVIORAL_VERSION_MATRIX.md) | [reproducibility contract](REPRODUCIBILITY_CONTRACT.md) | [repository architecture](REPOSITORY_ARCHITECTURE.md) | [github workflow](GITHUB_WORKFLOW.md) | [migration and validation plan](MIGRATION_AND_VALIDATION_PLAN.md).

## Design scope and authority

This package is an implementation-ready **proposal**, not an implementation authorization. The freeze candidate fixes a vocabulary, dependency direction, contracts and acceptance gates for review. All eight ADRs remain Proposed. No runtime source, configuration, registry, dataset, checkpoint, result, manifest, Git policy or branch is changed. This architecture Phase 4 is distinct from the unimplemented associative-memory Phase 4 described by historical protocols.

## Decision register

| ADR | Proposed decision | Status |
| --- | --- | --- |
| [ADR-0001](adr/0001-historical-isolation.md) | Historical isolation and future development | Proposed |
| [ADR-0002](adr/0002-behavioral-versioning.md) | Versioned mathematical and behavioral components | Proposed |
| [ADR-0003](adr/0003-runner-adapters.md) | Experiment runner and adapter architecture | Proposed |
| [ADR-0004](adr/0004-data-contracts.md) | Data ingestion and preprocessing contracts | Proposed |
| [ADR-0005](adr/0005-provenance.md) | Experiment and artifact provenance | Proposed |
| [ADR-0006](adr/0006-checkpoint-compatibility.md) | Checkpoint and model compatibility | Proposed |
| [ADR-0007](adr/0007-environments.md) | Dependency and environment management | Proposed |
| [ADR-0008](adr/0008-development-releases.md) | GitHub development and release practices | Proposed |

## Freeze rules

- Behavioral catalog C01-C14 is authoritative only for the proposed design; v1 descriptors identify inspected policies, not authenticated historical executions. Future v2 IDs remain unimplemented until an approved implementation passes its reference suite.
- Package/API, behavioral, protocol/configuration, checkpoint/artifact and digest versions are distinct. Any scientifically significant change requires a new behavior selection and reviewed protocol implications even when SemVer remains API-compatible.
- INTERFACE_CONTRACTS.md owns interface names/types. BEHAVIORAL_VERSION_MATRIX.md owns behavior IDs and T-C01..T-C14 suites. REPRODUCIBILITY_CONTRACT.md owns run/attempt/artifact fields and state semantics. MIGRATION_AND_VALIDATION_PLAN.md owns M0..M7 acceptance gates. Other documents must not silently redefine them.
- No default silently selects detach policy, model head, batch ordering, context precision, statistical unit or resume eligibility. Unknown profile IDs fail before output allocation or dataset test access.
- All future work is additive. Historical files and paths remain the comparison reference; unknown source-to-fit mapping is never filled from a final snapshot.

## Required material read and captured

All five Phase 3 reports were read in full, including their large source/import tables. The prior compatibility definitions and source assertions were checked against existing code where decisions rely on them. The table records current file-byte identities, not a claim of historical authorship or independent reproduction.

| Existing required file | SHA-256 |
| --- | --- |
| [docs/architecture/PHASE3_SOURCE_INVENTORY.md](../../../docs/architecture/PHASE3_SOURCE_INVENTORY.md) | 2d00578c9458aa4c705c03993007908e074b256b187dc7bd0587b03192b1e1a7 |
| [docs/architecture/PHASE3_DEPENDENCY_MAP.md](../../../docs/architecture/PHASE3_DEPENDENCY_MAP.md) | 70ace9059b470a8f6b29f93289633e797fe7f52709a35080366b68abd486194c |
| [docs/architecture/PHASE3_BEHAVIORAL_COMPATIBILITY.md](../../../docs/architecture/PHASE3_BEHAVIORAL_COMPATIBILITY.md) | d7d1383cb3232736b44e3ad637f89c74fac6c1a87dcf08b68a99820431d78984 |
| [docs/architecture/PHASE3_PROPOSED_ARCHITECTURE.md](../../../docs/architecture/PHASE3_PROPOSED_ARCHITECTURE.md) | f93a5bc328c2b275c832f6b84666bea149be99f370ff020b15ce4d0c6d01d585 |
| [docs/architecture/PHASE3_RISK_ASSESSMENT.md](../../../docs/architecture/PHASE3_RISK_ASSESSMENT.md) | 1cefd9179daaa8cab45d425de678b0752b7248aceb04dfa6dc54e53119f037e6 |
| [registry/experiments.yaml](../../../registry/experiments.yaml) | 4262f04d6cb0a40edc80e311684355b1f184263230cf3665046b054fb8491b54 |
| [registry/dependencies.yaml](../../../registry/dependencies.yaml) | b06cafa0184c6429154940ad56f1e6d330386b80d87787f981f48b54e4fe6837 |
| [registry/runs.csv](../../../registry/runs.csv) | 38d1a5a6d7c808e24aacb6583adddc76af328200411e9bace2b610395ab8ea10 |
| [registry/README.md](../../../registry/README.md) | d78f421a9cd3b101d24da2d39c89a0e1869478069c4e9c90741cba4f89928329 |
| [README.md](../../../README.md) | 8de999564625e42dc36fd9fc0bffebcfbe16f9c070497da32e8043e9f486d40e |
| [.gitattributes](../../../.gitattributes) | 06941d398740a0312b5a6ade781959c0427ebc0bdcac8f6e440c8e8dd7bcff70 |
| [.gitignore](../../../.gitignore) | 455bf8659fa7b7cf5b28e63d3e104b5dc9d0c1cc3d8981b2f8befbf2370c9369 |

## Unresolved decisions and gates

| ID | Open item | Proposed disposition; no historical inference |
| --- | --- | --- |
| U01 | Required replay level and test budgets | Baseline proposal requires artifact identity and declared numerical fixtures in a pinned CPU environment; full training or cross-device bitwise replay requires separate approval and evidence. |
| U02 | Early executing bytes and Phase 2/3 per-fit source stages | Leave unresolved. Reconstruct only from attributable saved evidence, never directory timestamps or final source alone. |
| U03 | Repository owners, independent scientific reviewers and remote protection entitlements | Assign identities and inspect remote capabilities before configuration; none inferred from local remote URL. |
| U04 | Exact Python 3.11 dependency/build lock and optional profiles | Resolve at M2 in a separately authorized new environment; historical environments remain untouched. |
| U05 | Phase 3B null population, layer attribution and C3 futility definitions | Block a new confirmatory context adapter until a new approved protocol resolves them. |
| U06 | Durable artifact storage, license/publication permissions and retention ownership | Preserve local originals; no upload/LFS/history conversion without a separate approved storage decision. |
| U07 | Reference fixture inventory and numerical tolerances | Default exact checks only for discrete identities; tolerances must be justified per operation/environment and approved before tests. |
| U08 | Release publisher, tag-signing identity and distribution access | Resolve before M7; no account/key/ownership assumptions. |

## Acceptance of the design

Review all eight ADRs together with the ten interfaces, fourteen behavior families and provenance/lifecycle contract. Approval must identify the accepted documents/version and any exceptions. Partial approval permits only the explicitly scoped future stage; it does not accept every ADR or authorize experiments. Until explicit authorization arrives, stop at documentation and validation. Proposed architecture diagrams are design views, not evidence of implemented behavior.
