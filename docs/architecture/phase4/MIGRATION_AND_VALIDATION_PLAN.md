# Incremental migration and validation plan

Status: **Proposed**. Design-freeze candidate **DF-2026-11-22-v1**. Prepared 2026-11-22 against branch initial-repository, HEAD 7a89cd51b1dcfb3526fd9212b0b395cd91317be7. No ADR is accepted and no runtime design is implemented by this documentation.

Labels: **V** verified current file/source fact; **S** static inference; **P** proposed requirement; **U** unresolved evidence or approval. Requirements below are normative only if explicitly approved. Historical provenance is never filled by inference.

Design documents: [architecture decisions](ARCHITECTURE_DECISIONS.md) | [interface contracts](INTERFACE_CONTRACTS.md) | [behavioral version matrix](BEHAVIORAL_VERSION_MATRIX.md) | [reproducibility contract](REPRODUCIBILITY_CONTRACT.md) | [repository architecture](REPOSITORY_ARCHITECTURE.md) | [github workflow](GITHUB_WORKFLOW.md) | [migration and validation plan](MIGRATION_AND_VALIDATION_PLAN.md).

## Governing rule and current state

**P:** Implement only through additive, explicitly approved stages alongside existing research. Every stage below is a future plan. Phase 4 produces documents and static documentation validation only; M0 design acceptance and M1..M7 execution are not completed by this report. No original implementation, configuration, dataset, checkpoint, result, manifest, registry file or repository policy is modified.

Authorization is scoped: approval of an ADR is not permission to train/evaluate, install dependencies, change settings or publish. Approval of Phase 5 coding is not blanket approval of scientific experiments or history migration. Resolve relevant U01..U08 items from ARCHITECTURE_DECISIONS.md before the stage that depends on them. A failing or unresolved gate blocks progression; elapsed time is not approval.

## M0 — Review and accept design scope

### Prerequisites

All fifteen Phase 4 documents exist; the owner has reviewed Phase 3 findings and understands the distinction between architecture Phase 4 and historical memory Phase 4.

### Scope and expected changes

Documentation/decision review only. Proposed expected change is a later explicit acceptance record identifying ADRs, behavior catalog, interfaces, provenance contract and permitted next stage; do not silently edit Proposed into Accepted.

### Compatibility risks

Unclear authority, invented historical provenance or approval inferred from document creation.

### Automated validation

Static ADR sections/status, internal links, all C01-C14 IDs, interface annotation/name checks, source references, diagram structure and baseline branch/HEAD/index comparison.

### Manual review requirements

Owner decisions for U01 replay target/budgets, U03 reviewers, U04 environment approach, U06 storage/publication and U07 fixtures/tolerances; U05 scientific definitions may remain blocked for a context adapter.

### Acceptance criteria

Explicit approval cites DF-2026-11-22-v1 and the exact accepted subset/exceptions. Every ADR remains Proposed until that action; source/registry bytes are unchanged.

### Rollback procedure

Revise only new design documents through reviewed superseding proposals. No runtime or Git rollback is needed; existing research remains unchanged.

## M1 — Add schema and static infrastructure in new paths

### Prerequisites

M0 approval and separately authorized Phase 5 implementation scope; preservation baseline and new output root approved.

### Scope and expected changes

Add future contracts/errors/config schema, fixed behavior metadata allowlist and static dependency checks under new src/conn and tests/future. Do not call old launchers or modify old tests/configs. No numerical kernel, model evaluation or dependency installation required for this stage.

### Compatibility risks

Import-time effects, schema mismatch, implicit defaults, accidental source/registry edits, incompatible metadata treated as a supported behavior.

### Automated validation

T-M01, static portions of T-C08/T-C14, T-P01 and T-R01. Reject unknown IDs, duplicate/invalid config fields, incomplete profile selections and forbidden import/path patterns. Assert only approved new files appear in the diff.

### Manual review requirements

Review all ten interfaces, field ownership and allowed dependency graph; identify side effects explicitly and inspect old-file preservation.

### Acceptance criteria

Schemas and metadata can be validated without importing research code; every new required field has an error path; old-file hashes match; no historical run is reassigned provenance.

### Rollback procedure

Stop using the new metadata package. Revert only authorized new development files in a later corrective change; preserve any generated evidence rather than deleting old material.

## M2 — Freeze reference profiles, fixtures and environment specifications

### Prerequisites

M1 accepted; U01/U04/U07 resolved for the chosen profile; separate approval for any new environment installation or reference execution.

### Scope and expected changes

Define exact environment/codec/dependency locks, profile parameters and T-C01..T-C14 reference fixture manifests. Historical profiles are source-derived descriptors until executing evidence establishes more. New environments are separate from historical venvs.

### Compatibility risks

Golden data generated by the same new implementation can hide bugs; missing historical lineage, package/kernel drift or outcome-driven tolerance changes can manufacture compatibility.

### Automated validation

T-E01 dependency/import/closure checks in the approved new environment; fixture schema/hash/source attribution checks; exact discrete identities and justified numerical comparisons only when authorized. Missing executed evidence remains not_run.

### Manual review requirements

Scientific review approves fixture origin, independence of oracle, estimand, RNG/draw order, precision/cast and threshold rationale before results. Environment owner approves storage/installation scope.

### Acceptance criteria

Versioned profile/fixture/environment manifests pin exact source and all variants. Reference cases cover positive/negative boundaries and a credible oracle; unsupported combinations are explicitly blocked. No claim of full historical replay from a source fixture.

### Rollback procedure

Keep earlier approved lock/profile choices; do not downgrade or reinstall original environments. Withdraw a new unsupported profile and retain failed fixture evidence.

## M3 — Implement isolated mathematical/model components

### Prerequisites

M2 manifests and numerical execution scope approved; source/behavior bindings are fixed and output isolation available.

### Scope and expected changes

Add C01-C06/C11-C13 primitives and model variants in the future package. Keep policy-specific validation, detach/cast, RNG recipes, schedule timing and context precision. No imports redirect historical code.

### Compatibility risks

A mathematically similar expression can change floating arithmetic, seeds, gradients, invalid-input errors or parameter groups.

### Automated validation

Approved relevant T-C suites, with exact shape/error/discrete checks and environment-bound forward/gradient/initial-state comparisons. Keep pure kernels separate from process runtime flags. T-M01/T-R01 remain mandatory.

### Manual review requirements

Review equations, axes/domains, initialization order, gradient paths, context rank interpretation and scientific version change. Inspect evidence for each promoted claim.

### Acceptance criteria

Each enabled implementation binds to a named v2 selection, parameter digest and passed reference-suite version; all other profiles remain unavailable. Agreement is scoped to fixtures, not blanket equivalence.

### Rollback procedure

Disable/select an earlier new profile. Original components and callers remain intact. Failed comparisons are retained; no historical test expectations are rewritten.

## M4 — Add data/artifact services and one non-executing adapter plan

### Prerequisites

M3 accepted for selected profile; U06 storage/location/permission and data lineage resolved; no scientific test use authorized by default.

### Scope and expected changes

Add registered read-only dataset views, explicit derived-transform service, ArtifactService, provenance append and runner plan/create interfaces. One future study adapter may produce a read-only execution plan; it must not fit or evaluate yet.

### Compatibility risks

Test access leakage, transformed data overwrites, path traversal/symlink escapes, races or a plan operation executing arbitrary code.

### Automated validation

T-C07/T-C08, T-P01/T-P02 and T-R01/T-R02 in new temporary/output roots under approved test scope. Test denied grants, occupied destinations, partial writes, stale predecessors and missing evidence.

### Manual review requirements

Review data/split/scaler/layout identity, output roots, access reservation and effect boundaries. Confirm no report or preflight command is run as a shortcut.

### Acceptance criteria

plan is read-only; create writes only a new authorized run root; dataset loading is read-only; event append and artifacts refuse collisions; no automatic old-registry mutation.

### Rollback procedure

Disable the new service/adapter plan. Preserve new partial/failed records and keep original paths. Do not move/delete original data or replace its manifests.

## M5 — Add trainer/evaluator and checkpoint readers under fixture scope

### Prerequisites

M4 accepted; explicit approval for model forward/optimizer/checkpoint/metric fixture execution, exact environment and budgets; codec/schema choices reviewed.

### Scope and expected changes

Implement declared training/evaluation interfaces, C09/C14 checkpoint state and optional legacy readers. Exercise only bounded new fixtures, not a historical rerun or broad HAR test evaluation.

### Compatibility risks

Incomplete resume, changed endpoint/count semantics, mixed model heads/dtypes, loader incompatibility or implicit extra test access.

### Automated validation

T-C03/C05/C06/C09/C11/C12/C14, T-R02 and T-P02. Test interruptions at boundaries, full restore cursor/RNG/plans, endpoint_only resume refusal, header/group mismatch and cumulative test reservations. Explicitly check Python optimization does not remove protocol guards.

### Manual review requirements

Review state completeness, observed numerical level, endpoint interpretation, loader allowlist and test-use accounting. Distinguish serialized model inspection from independent reproduction.

### Acceptance criteria

No enabled reader guesses schema; resumable requires complete matching state; authorized uninterrupted/resumed fixture comparison passes at its declared level; failed/partial attempts are preserved and outputs stay new.

### Rollback procedure

Disable new runner/checkpoint profile. Return new consumers to the prior approved version; retain all original and new failure artifacts. No checkpoint conversion in place.

## M6 — Validate one new scientific pilot through the adapter

### Prerequisites

M5 passed; a new protocol, frozen seeds/control/selection/endpoints/analysis plan and precise compute/test-access budget explicitly approved. Resolve U05 before any new context study.

### Scope and expected changes

Run one bounded newly specified study in runs/future with the complete conn-run-1 contract. Synthetic or other approved data only; do not reuse old test outcomes as new confirmation. No expansion to every historical study by default.

### Compatibility risks

Overclaiming scientific equivalence, low-power results, reused test data, changed practical threshold, leakage or undeclared protocol deviations.

### Automated validation

All applicable T-C/T-P/T-R checks, plus run/fit/attempt/artifact completeness, declared pairing/estimands and independently recomputable saved measurements. Training/metrics execution occurs only under that approval, never in Phase 4.

### Manual review requirements

Scientific reviewer evaluates protocol compliance, failures/omissions, statistical assumptions, deviations and claim scope; engineering reviewer checks operability and source/data closure.

### Acceptance criteria

All approved fits/endpoints accounted for; policy-complete pilots may remain unevaluated; results and failure evidence are attributable. A successful run does not establish independent replication or general superiority.

### Rollback procedure

Stop new execution; mark the new study/attempt outcome without rewriting it. Preserve evidence and propose a new protocol for redesign rather than editing old outcomes.

## M7 — Review development/release workflow and publication

### Prerequisites

Applicable M1..M6 gates accepted for the released scope; U03/U06/U08 resolved; separate owner approval for branch/protection/workflow/tag/storage/publication actions.

### Scope and expected changes

Adopt reviewed main/feature workflow and optional CODEOWNERS/CI/protections; release only supported package/profile/schema combinations and permitted artifact references. No history rewrite or automatic upload of old binaries.

### Compatibility risks

Unavailable remote features, incomplete artifact access, leaked credentials, normalized historical bytes mistaken for originals or API tags mistaken for scientific compatibility.

### Automated validation

T-G01 plus source/build/environment/artifact manifests and applicable suites; verify required checks exist and run on the actual candidate; scan/review approved publication inputs. No real secrets or private data in CI fixtures.

### Manual review requirements

Owner confirms permissions/features/reviewers and publisher identity; scientific reviewer approves claim wording; storage/license reviewer approves every upload and retention/restore plan.

### Acceptance criteria

Approved release references exact source, behavior, protocol, schemas, environment and validation scopes; tags/artifacts treated immutable. Source checkout is not falsely presented as the full artifact archive.

### Rollback procedure

Withdraw/supersede the new release with an attributable correction. Do not retag, force-push/reset history, replace old artifacts or delete failed scientific evidence.

## Validation suite register

| Suite | Purpose | Effect tier and gate |
| --- | --- | --- |
| T-M01 | Preserve original file inventory and byte hashes, source/config/docs/registry/policy identities, branch/HEAD/index; check any backup restore under an approved new destination | Static comparisons now; full restore drills future approval |
| T-C01..T-C14 | All fourteen mathematical/behavioral families in BEHAVIORAL_VERSION_MATRIX.md | Static description now; numerical/autograd/optimizer/checkpoint execution only with M2/M3/M5 permission |
| T-P01 | Required conn-run-1/conn-experiment-1 fields, digest namespaces, source/data closure, policy and schema relationships | Schema/fixture validation; no arbitrary model deserialization |
| T-P02 | Failed/interrupted/resumed/repaired/partial artifacts, event predecessors, collision/race and missing-state negatives | Future isolated IO fixtures; model replay only under M5 approval |
| T-R01 | No old-path writes, hidden imports, output-root escapes or automatic Git operations | Static graph plus future isolated side-effect fixtures |
| T-R02 | Lifecycle transitions, immutable terminal attempts, operation authorization, evaluation access reservation/read budgets and resume capability | Future controlled fixtures; no real test split by default |
| T-E01 | Declared direct/transitive dependencies, new environment profile, package import purity and supported device/runtime combinations | No current installation; future profile-specific checks only |
| T-G01 | Reviewed PR/release scope, applicable checks, source/tag/build manifest, publication permissions and repository feature availability | Current policy design only; future remote actions require owner approval |

## Acceptance evidence and claims

Every gate produces a reviewed evidence record listing source/profile/environment, fixture and artifact hashes, exact commands/procedure, effects/budgets, pass/fail/not_run/not_applicable and unresolved limitations. Do not count an unexecuted required test as passed. Static formula/source identity, fixture equality, statistical agreement, bitwise replay and independent scientific replication are different levels and must be named explicitly. Approval cannot make missing historical provenance true.

Reference oracles should use independently specified mathematical invariants and attributable saved fixtures where possible. Do not generate both expected and observed values from the same new implementation and call that compatibility evidence. Preserve negative results and discovered bugs; fixes receive their own source/behavior/protocol impact review. Any future test that writes beneath an original active-run tree is out of scope until redesigned for isolated outputs or separately authorized.

## Phase 4 checks performed versus future tests

This phase reads required source reports/registry/policies, verifies relevant source statements, creates fifteen Markdown files only, and checks document existence/sections, links/source lines, all fourteen families, interface syntax/name consistency, cross-document IDs and Mermaid structure. It compares existing-file metadata and source/document/configuration/registry/policy byte-hash baselines and Git branch/HEAD/index/staged state before/after writing. This is not a new full hash audit of every binary artifact, a backup restore drill, Mermaid renderer/parser certification, model evaluation or behavioral test run. No source module, test, config, lockfile, CI file, dataset or validation artifact is created.

## Stop condition

After documentation validation and the completion notification, stop. Phase 5 implementation, future tests/environments, scientific runs and repository settings/publication remain separately gated. No stage starts merely because its plan is implementation-ready.
