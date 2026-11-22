# Proposed GitHub development and release workflow

Status: **Proposed**. Design-freeze candidate **DF-2026-11-22-v1**. Prepared 2026-11-22 against branch initial-repository, HEAD 7a89cd51b1dcfb3526fd9212b0b395cd91317be7. No ADR is accepted and no runtime design is implemented by this documentation.

Labels: **V** verified current file/source fact; **S** static inference; **P** proposed requirement; **U** unresolved evidence or approval. Requirements below are normative only if explicitly approved. Historical provenance is never filled by inference.

Design documents: [architecture decisions](ARCHITECTURE_DECISIONS.md) | [interface contracts](INTERFACE_CONTRACTS.md) | [behavioral version matrix](BEHAVIORAL_VERSION_MATRIX.md) | [reproducibility contract](REPRODUCIBILITY_CONTRACT.md) | [repository architecture](REPOSITORY_ARCHITECTURE.md) | [github workflow](GITHUB_WORKFLOW.md) | [migration and validation plan](MIGRATION_AND_VALIDATION_PLAN.md).

## Observed local state versus proposed remote policy

**V:** Local branch is initial-repository, HEAD 7a89cd51b1dcfb3526fd9212b0b395cd91317be7. The configured repository remote is nu-ink/frank_h_torralva-conn. No local .github directory was present during inspection. Existing [.gitattributes](../../../.gitattributes) and [.gitignore](../../../.gitignore) were read and preserved. Remote default branch, visibility/plan, protections, permissions, review ownership, Actions and secret-scanning settings were **not inspected**; their configuration is unresolved. No branch, setting, history, stage/index, commit or push is changed by this phase.

**P:** The practices below become operational only after separate owner approval, capability checks and implementation. They do not assert that GitHub has been configured to enforce them.

## Branch purposes and naming

| Branch/tag class | Proposed purpose and rule |
| --- | --- |
| main | Reviewed, reproducible future framework and documentation; merge only after applicable engineering/scientific and provenance checks. Creating/selecting it or changing the default branch requires authorization. |
| initial-repository | Preserve the initial research capture and its existing history. Do not repurpose it for future runtime work, delete it, force-push it or rewrite prior commits. Owner may approve lock/protection later. |
| feature/phase5-<issue>-<slug> | Scoped additive implementation against an accepted ADR and M-stage; no historical edits. |
| fix/<issue>-<slug> | Explicit defect scope, affected behavior IDs, scientific impact and new-version decision. |
| research/<experiment>-<protocol>-<slug> | New approved scientific definition and run plan; not changes to old outcomes. |
| docs/<issue>-<slug> | Documentation/ADR changes, including explicit proposed/accepted/superseded status. |
| release/<package-version> | Optional short-lived review branch for an approved release candidate; no special authority to bypass tests or scientific review. |

Branch names are intent labels, not provenance. Record the full executing revision and file closure in every run. No branch is created, renamed, checked out, merged or deleted now.

## Pull request requirements

Each future PR must include: problem and resulting behavior; accepted ADRs and migration stage; exact file/path scope; affected C01-C14 selections and protocol/config/schema versions; existing-data/source preservation evidence; validation performed and deliberately not run; checkpoint/environment/data implications; rollback; and unresolved limitations. Numerical or scientific changes must say whether they change the hypothesis, sample/order, control, preprocessing, endpoint, statistic or success criterion. Do not bury a behavior change under a refactor title.

Review the complete changed-file list, including untracked additions and generated artifacts. Never bulk-stage historical outputs or use a broad add operation as a substitute for review. PRs cannot silently reformat/renormalize evidence, change old manifests, refresh hashes to hide drift, install into historical environments, or introduce sibling-path imports. New fixtures/configurations/outputs have their own reviewed identities and licenses.

## Owners and scientific review

Propose ownership roles for core/models, data/preprocessing, execution/checkpoints, statistics, provenance, documentation and release/security. Actual people/teams and availability remain U03/U08; do not invent a CODEOWNERS identity. Engineering review checks API/dependency/error/operability concerns. Scientific review checks experimental definition, equivalence scope, data leakage, RNG/pairing, precision, endpoints, uncertainty and test reuse. Changes to C02-C07/C09-C12/C14 need explicit scientific impact review as well as engineering review; other families can also require it when their consequences affect experiments.

A single maintainer may record both role-specific reviews only under an explicit approved exception; this is not independent scientific reproduction. No automated test or code-owner approval alone establishes a scientific claim. The owner must choose reviewers before enforcing a policy that would otherwise make approval impossible.

## Proposed CI and repository protections

**P:** Require reviewed PRs, resolved discussions and uniquely named applicable checks on the candidate commit before merging to main; block unreviewed direct/force pushes and deletion where available. Consider stale-approval dismissal after relevant changes and protect release tags/baseline references. Feature support depends on repository type/plan and must be checked before configuration. GitHub supports required review/status checks through branch protection; this is a proposed configuration, not the observed state. [GitHub protected-branch documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).

| Proposed check | Scope and pass condition |
| --- | --- |
| conn-docs-static | Required docs/ADR/schema sections, internal links, behavioral IDs and interface signature consistency. |
| conn-source-static | New source AST/import graph, forbidden path/dynamic-import effects, explicit dependency declarations and no historical file changes. |
| conn-contracts | Approved T-C01..T-C14 fixtures in an exact pinned profile, only after future numerical/test execution is authorized. Missing required fixture evidence blocks the affected profile. |
| conn-provenance-lifecycle | T-P01/T-P02/T-R01/T-R02 with isolated new output roots; no real historical-run writes. |
| conn-environment | T-E01 import/build/profile support checks in a new environment, no modification of old venvs. |
| conn-release-review | Accepted behavior/schemas, release source/environment manifests, evidence publication permissions and owner signoff. |

Do not require a check name before an applicable workflow actually exists and has been exercised. The future planner should distinguish applicable passed, failed, not_run and explicitly not_applicable checks; a silently skipped scientific suite cannot count as behavioral validation. Default PR CI must never launch full experiments, model evaluations or paid/GPU jobs. Such work requires separate approved protocols, budgets, trusted execution context and recorded artifacts. No workflow file or CI check is created now.

## Secrets and workflow security

**P:** Propose secret scanning and push protection where the repository has the required capability, plus review of scanner coverage and documented handling of exceptions. Scanning is a guard, not proof that all secrets are absent. Remote entitlement/configuration remains unknown. If a real leak is detected later, rotate/revoke it and follow an explicitly approved remediation process; do not automatically rewrite research history. GitHub describes supported scope and push blocking in its [secret-scanning scope](https://docs.github.com/en/code-security/reference/secret-security/secret-scanning-scope) and [push-protection guidance](https://docs.github.com/en/code-security/how-tos/secure-your-secrets/prevent-future-leaks/enable-push-protection).

Future Actions should default to read-only repository token permissions, with narrowly elevated permissions only for explicitly approved jobs; avoid privileged execution of untrusted PR code, pass untrusted values as data rather than shell syntax, and pin reviewed third-party actions to full commit SHAs. Keep secrets out of manifests, artifacts and logs; capture dependency versions without environment secret values. No security setting or workflow is enabled in this phase. [GitHub Actions secure-use reference](https://docs.github.com/en/actions/reference/security/secure-use).

## Source-code and scientific versioning

Use package release tags such as conn-v0.1.0 for API/distribution versions. Separately record C01-C14 behavior selections, conn-experiment-1 configuration schema, conn-run-1 provenance schema, conn-checkpoint-1 capability and digest format versions. A compatible API patch can still require a new behavior ID and protocol review. Use separate study tags such as study-<experiment>-<protocol>-<run-id> only after authorized reviewed publication; tags identify commits, while run manifests identify external data/output bytes and environments.

Treat published tags as immutable project policy. Record full source commit, release artifact hashes, build environment and signer/publisher identity; do not assume a tag name or Git timestamp authenticates an old executing source. Prefer signed annotated tags when an approved identity/key exists. A failed release is withdrawn/superseded with a new version and correction note; never retarget an old tag silently. Package publication and study-result publication are separate approvals. No tag, release or upload is created now.

## Large datasets, artifacts and byte preservation

**V/P:** Current ignore rules exclude *.pt/*.pth/*.ckpt, numerical arrays and archives while retaining general CSV/JSON/Markdown/log evidence. One oversized historical diagnostics CSV is explicitly ignored. These tracking rules do not delete local data and are not a complete archive or access-control policy. Review existing tracked files separately; adding an ignore pattern does not untrack anything already committed.

For future large artifacts, keep original local bytes and track explicit size/hash/provenance/location records. Choose durable storage, retention, access, licensing and backup/restore ownership at U06 before upload. Git LFS is an optional future choice, not enabled here: it stores pointer references in Git while the content is stored separately, so a source checkout alone is not proof the artifact is available. Do not migrate existing history to LFS, remove local data or upload datasets/checkpoints without separate approval. [GitHub Git LFS documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).

Current attributes specify LF for most text, CRLF for Windows scripts and binary treatment for selected artifact extensions. New Phase 4 Markdown uses UTF-8/LF. Do not run renormalization or bulk formatting over historical files. A Git blob or checkout may have different text bytes from an original working-file capture; preserve and label original file-byte hashes independently. If distribution requires exact historical bytes, use an approved byte-preserving archive with its own manifest rather than assuming a normalized checkout recreates them.

## Release acceptance and rollback

M7 requires: accepted ADRs; complete implementation/profile bindings; applicable tests passed with explicit scope; reviewed scientific claims; provenance and artifact availability; approved license/publication/security decisions; clean release source; exact environment/build records; and an attributable publisher. It must not claim independent reproduction merely because CI passed. Manual review verifies release manifest references and that a consumer can locate permitted artifacts without altering originals.

Rollback selects the prior approved package/profile for new consumers or uses a later corrective/revert commit after approval. Preserve failed release/run evidence and original tags; do not force-reset published history, replace artifacts or modify old research. This document is a proposal only; current branches, settings, Git history and index remain unchanged.
