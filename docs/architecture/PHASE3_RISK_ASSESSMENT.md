# Phase 3 architecture risk assessment

Verified 2026-11-22 against local workspace E:/CoNN; repository remote nu-ink/frank_h_torralva-conn, branch initial-repository, HEAD d06568ac87bf370eba627858a12997d88cf30208.

Evidence labels: **V** = verified source/file fact; **S** = static-analysis inference, with limits; **P** = architectural proposal; **U** = unresolved. No project module was imported, no checkpoint loaded, and no training, evaluation, research test, package installation or runtime framework creation was performed. Byte identity and AST similarity are not behavioral-equivalence claims.

Reports: [inventory](PHASE3_SOURCE_INVENTORY.md) | [dependencies](PHASE3_DEPENDENCY_MAP.md) | [compatibility](PHASE3_BEHAVIORAL_COMPATIBILITY.md) | [proposal](PHASE3_PROPOSED_ARCHITECTURE.md) | [risks](PHASE3_RISK_ASSESSMENT.md). Navigation foundation: [registry guide](../../registry/README.md).

## Overall assessment

**S:** The project can evolve into a coherent experimental framework, but a broad consolidation would currently carry high scientific-compatibility risk. Existing safeguards include separate run directories, frozen protocols/source/configuration, saved arrays and plans, hash manifests, checkpoint reload checks and retained failed/recovered attempts. These safeguards are valuable and should be preserved. The registry adds navigation but does not make existing implementations portable or behaviorally interchangeable.

Severity describes the consequence **if the stated change/trigger occurs**, not proof that historical results are invalid. Critical means loss or corruption of irreplaceable evidence; high means changed scientific meaning, replay or checkpoint interpretation; medium means significant reliability/portability/maintenance impact; low means primarily navigation friction. Each finding identifies its evidence class. Suggested validation is future work requiring appropriate authorization, not execution performed here.

## Findings and actions

| ID | Severity | Consequence area | Trigger | Evidence class |
| --- | --- | --- | --- | --- |
| R01 | Critical | Historical experiment integrity | Overwrite or merge original snapshots, configurations, manifests or failed attempts during cleanup. | V/S |
| R02 | High | Reproducibility and portability | Move a project, reorder sys.path or substitute a snapshot without capturing transitive imports. | V/S |
| R03 | High | Numerical behavior and scientific meaning | Unify Gaussian domain policy, detach behavior or dense baseline path. | V/S |
| R04 | High | Randomization, pairing and reproducibility | Replace fixed cycling with reshuffled epochs or standardize all seed derivation. | V/S |
| R05 | High | Context numerical behavior and checkpoint interpretation | Normalize all context factors to one dtype, initialization or code/mask policy. | V/S |
| R06 | High | Data lineage and evaluation validity | Refit scaling, change flattened feature/channel order, or generalize loaders without preserving test policy. | V/S |
| R07 | High | Source provenance and historical replay | Assign one delivery snapshot to every Phase 2/3 fit without proof. | V/U |
| R08 | High | Checkpoint compatibility and replay | Use one generic checkpoint loader or change model heads, keys, dtypes or resume expectations. | V/S |
| R09 | High | Statistical validity and verification | Consolidate metrics or inference without preserving the estimand. | V/S |
| R10 | High | Environment portability and reproducibility | Assume broad package minimums or current Git HEAD reproduce earlier runs. | V/S |
| R11 | Medium | Performance and caching | Generalize fixed-geometry caches or compare timings from different baseline/diagnostic scopes. | V/S |
| R12 | High | Historical integrity and operability | Run inspect/verify/preflight/report or selected tests assuming they are read-only. | V/S |
| R13 | Medium | Configuration, hash identity and traceability | Re-serialize a frozen config or use one digest namespace for files, arrays and model state. | V/S |
| R14 | Medium | Maintainability and repair provenance | Fold generated or patched reporting recovery into an unversioned main script. | V/S |
| R15 | High | Integrity checks and production assumptions | Run an integrity-critical path under optimized Python or broaden it without explicit validations. | V/S |
| R16 | Medium | Long-term maintainability and documentation | Treat the registry or copied packet as a complete execution specification. | V/S/U |

### R01 — Historical experiment integrity (Critical)

**Trigger:** Overwrite or merge original snapshots, configurations, manifests or failed attempts during cleanup.

**Evidence [V/S]:** [ORGANIZED_EXPERIMENTS/ORGANIZATION_REPORT.md](../../ORGANIZED_EXPERIMENTS/ORGANIZATION_REPORT.md); [CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/source](../../CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/source).

**Consequence:** Original Experiment 01 reconstruction and six distinct Phase 2/3 stages show that directory similarity does not identify one executing version. Evidence loss would make later disagreement impossible to investigate.

**Recommended action [P]:** Preserve originals and paths; extraction must be additive. Keep failed and recovered attempts separate.

**Validation/rollback [P]:** Before/after hashes and a restore drill for any future authorized change; rollback selects the unchanged original tree.

### R02 — Reproducibility and portability (High)

**Trigger:** Move a project, reorder sys.path or substitute a snapshot without capturing transitive imports.

**Evidence [V/S]:** [experiment2_root](../../CoNN_Associative_Relationships/src/conn_assoc/paths.py#L13); [experiments/experiment_04/src/conn_exp04/__init__.py](../../experiments/experiment_04/src/conn_exp04/__init__.py#L7).

**Consequence:** Historical arrays or live modules can be selected from a different tree; copied packets can retain correct bytes while seeking nonexistent sibling roots.

**Recommended action [P]:** Use explicit source/data dependency closure and opt-in location mapping for future code; do not redirect old imports silently.

**Validation/rollback [P]:** Static reference resolution followed by separately authorized isolated import/replay checks; keep original launchers.

### R03 — Numerical behavior and scientific meaning (High)

**Trigger:** Unify Gaussian domain policy, detach behavior or dense baseline path.

**Evidence [V/S]:** [resolve_tau](../../conn_experiment_01/src/conn_exp/gates.py#L14); [resolve_tau](../../CoNN_Associative_Relationships/src/conn_assoc/gate.py#L19); [GatedLinear.forward](../../conn_experiment_01/src/conn_exp/models.py#L23).

**Consequence:** Invalid bandwidths are floored in one package and rejected in another; gradients to a future learned gate depend on detach; baseline overhead differs.

**Recommended action [P]:** Name and version domain/gradient/control policies; retain historical adapters.

**Validation/rollback [P]:** Future domain/error and forward/gradient fixtures with explicit dtype/device and boundary cases; no behavioral equivalence is assumed.

### R04 — Randomization, pairing and reproducibility (High)

**Trigger:** Replace fixed cycling with reshuffled epochs or standardize all seed derivation.

**Evidence [V/S]:** [batch_indices](../../conn_experiment_01/src/conn_exp/train.py#L39); [paired_batches](../../conn_experiment_01/src/conn_exp/har_core.py#L57); [stream](../../experiments/experiment_04/src/conn_exp04/core.py#L25).

**Consequence:** Same seed labels can produce different data, initializations, masks and batch order. This can break paired contrasts while leaving shapes and budgets unchanged.

**Recommended action [P]:** Record algorithm IDs, composed effective seeds, draw order and plans; keep sampling strategies separate.

**Validation/rollback [P]:** Future byte comparisons of saved plans and initial-state identities before any new training; retain original stream implementations.

### R05 — Context numerical behavior and checkpoint interpretation (High)

**Trigger:** Normalize all context factors to one dtype, initialization or code/mask policy.

**Evidence [V/S]:** [ContextFactor](../../CoNN_Associative_Relationships/src/conn_assoc/phase3_context.py#L25); [Model.__init__](../../CoNN_Associative_Relationships/phase3b_context_diagnostics/src/models.py#L25); [Factor.forward](../../CoNN_Associative_Relationships/phase3b_context_diagnostics/src/models.py#L19).

**Consequence:** Rank symmetry, saturation, jitter and float64 storage/float32 forward casts are the diagnostic treatments themselves; standardization could erase or change a contrast.

**Recommended action [P]:** Keep C0-C4 and shuffled/constant/frozen variants explicitly versioned, including parameter eligibility and draw order.

**Validation/rollback [P]:** Future factor, effective-weight, gradient and replay comparisons by variant; never replace historical factors in place.

### R06 — Data lineage and evaluation validity (High)

**Trigger:** Refit scaling, change flattened feature/channel order, or generalize loaders without preserving test policy.

**Evidence [V/S]:** [fit_scaler](../../conn_experiment_01/src/conn_exp/har_data.py#L32); [_check_purpose](../../CoNN_Associative_Relationships/src/conn_assoc/data.py#L96); [CoNN_Associative_Relationships/PROTOCOL_PHASE23_v1.md](../../CoNN_Associative_Relationships/PROTOCOL_PHASE23_v1.md).

**Consequence:** Changed preprocessing/layout changes the treatment and data distribution. Existing development CV reuses a scaler with disclosed overlap; a later fold-refit is a different experiment. Test-policy weakening changes evidence validity.

**Recommended action [P]:** Preserve arrays, splits, coordinates and scalers; require explicit access policy and new protocol for new preprocessing.

**Validation/rollback [P]:** Hash/source/split lineage and future transform fixtures; retain original data and treat alternatives as new derived datasets.

### R07 — Source provenance and historical replay (High)

**Trigger:** Assign one delivery snapshot to every Phase 2/3 fit without proof.

**Evidence [V/U]:** [CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/source](../../CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/source); [CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/jobs](../../CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/jobs); [registry/experiments.yaml](../../registry/experiments.yaml).

**Consequence:** Six stages exist, while sampled fit configuration/completion records lack a direct executing-source-stage identity. Resume can introduce more than one executing source version.

**Recommended action [P]:** Mark mapping unresolved unless logs/manifests establish it. Future runs need source closure per fit and per resume attempt.

**Validation/rollback [P]:** Reconcile source/configuration/execution logs as evidence, without inventing timestamps or stage assignments; no historical records are rewritten.

### R08 — Checkpoint compatibility and replay (High)

**Trigger:** Use one generic checkpoint loader or change model heads, keys, dtypes or resume expectations.

**Evidence [V/S]:** [_save_checkpoint](../../conn_experiment_01/src/conn_exp/train.py#L139); [train_job](../../CoNN_Associative_Relationships/src/conn_assoc/phase23_core.py#L203); [save_checkpoint](../../experiments/experiment_04/src/conn_exp04/runner.py#L73).

**Consequence:** Binary/multiclass/context models and checkpoint schemas differ. Original endpoint checkpoints are not complete general resume state. weights_only handling also differs.

**Recommended action [P]:** Version checkpoint formats/readers; preserve architecture, optimizer groups, plans, RNG policy and source identity.

**Validation/rollback [P]:** Future schema/roundtrip checks only with authorized loading; conversions write new artifacts with parent hashes. Originals remain unchanged.

### R09 — Statistical validity and verification (High)

**Trigger:** Consolidate metrics or inference without preserving the estimand.

**Evidence [V/S]:** [metric](../../CoNN_Associative_Relationships/phase3b_context_diagnostics/src/evaluation.py#L6); [paired_summary](../../conn_experiment_01/src/conn_exp/evaluation.py#L68); [paired](../../experiments/experiment_04/src/conn_exp04/analysis.py#L13).

**Consequence:** Binary F1, six-class macro and supported-class macro differ. Subject versus window weighting, seed versus subject inference, nonfinite filtering versus rejection, and multiplicity rules change conclusions.

**Recommended action [P]:** Make class universe, aggregation, pairing, sample unit, resampling and nonfinite rules explicit and versioned.

**Validation/rollback [P]:** Future arithmetic comparisons on saved tables with known edge cases; do not promote exploratory development summaries to confirmation.

### R10 — Environment portability and reproducibility (High)

**Trigger:** Assume broad package minimums or current Git HEAD reproduce earlier runs.

**Evidence [V/S]:** [conn_experiment_01/pyproject.toml](../../conn_experiment_01/pyproject.toml); [CoNN_Associative_Relationships/pyproject.toml](../../CoNN_Associative_Relationships/pyproject.toml); [docs/research_validation/experiment_04/REPRODUCTION.md](../../docs/research_validation/experiment_04/REPRODUCTION.md).

**Consequence:** Imported undeclared project/analysis dependencies and Python 3.10 recorded execution versus >=3.11 metadata prevent a single supported-environment claim. Git added later does not identify historical execution retroactively.

**Recommended action [P]:** Capture exact environment and transitive source closure per behavior profile; preserve working environments.

**Validation/rollback [P]:** Future isolated replay in an explicitly approved environment; installation/reconstruction was not attempted here.

### R11 — Performance and caching (Medium)

**Trigger:** Generalize fixed-geometry caches or compare timings from different baseline/diagnostic scopes.

**Evidence [V/S]:** [Gates.build](../../CoNN_Associative_Relationships/src/conn_assoc/phase23_core.py#L105); [HARMLP.forward](../../conn_experiment_01/src/conn_exp/har_core.py#L33); [benchmark](../../CoNN_Associative_Relationships/phase3b_context_diagnostics/src/training.py#L43).

**Consequence:** The CPU cache keys only tau under a fixed geometry. A generic cache could return incorrect tensors for another shape/dtype/layout. Dense bypass versus ones multiplication and diagnostic overhead make timings noncomparable.

**Recommended action [P]:** Use semantic cache identities and explicit measurement scopes; avoid performance claims from architectural shape alone.

**Validation/rollback [P]:** Future matched warm-up/device synchronization and per-scope timings; keep a correct uncached path. No speedup was measured in this phase.

### R12 — Historical integrity and operability (High)

**Trigger:** Run inspect/verify/preflight/report or selected tests assuming they are read-only.

**Evidence [V/S]:** [main](../../experiments/experiment_04/scripts/verify_artifacts.py#L14); [execute](../../CoNN_Associative_Relationships/phase3b_context_diagnostics/src/training.py#L208); [CoNN_Associative_Relationships/tests/test_phase2_cuda.py](../../CoNN_Associative_Relationships/tests/test_phase2_cuda.py).

**Consequence:** Some checks load checkpoints, execute optimizer/benchmark tests or create new artifacts beneath a historical run. A name cannot establish side-effect safety.

**Recommended action [P]:** Separate check-only results from explicit writers and execution-capable verification; document effects and output destinations.

**Validation/rollback [P]:** Static side-effect audit before future execution; keep all new validation outputs in authorized new locations. None of these checks was invoked here.

### R13 — Configuration, hash identity and traceability (Medium)

**Trigger:** Re-serialize a frozen config or use one digest namespace for files, arrays and model state.

**Evidence [V/S]:** [config_sha256](../../conn_experiment_01/src/conn_exp/config.py#L143); [digest_json](../../experiments/experiment_04/src/conn_exp04/core.py#L21); [fingerprint_state](../../conn_experiment_01/src/conn_exp/models.py#L93).

**Consequence:** Byte hashes and compact object digests differ. Tensor fingerprints omit explicit shape/dtype tags while another array digest includes them. Equivalent-looking JSON is not the same frozen byte artifact.

**Recommended action [P]:** Preserve original encodings; define new schema/digest versions and parent references for derived records.

**Validation/rollback [P]:** Future canonicalization and type/shape ambiguity fixtures; never refresh a historical seal merely to accommodate formatting.

### R14 — Maintainability and repair provenance (Medium)

**Trigger:** Fold generated or patched reporting recovery into an unversioned main script.

**Evidence [V/S]:** [load](../../CoNN_Associative_Relationships/phase3b_context_diagnostics/scripts/recover_reporting_v2.py#L20); [CoNN_Associative_Relationships/phase3b_context_diagnostics/results/20261020_phase3b_v3_run01/execution_extensions/recovery_analysis_v2](../../CoNN_Associative_Relationships/phase3b_context_diagnostics/results/20261020_phase3b_v3_run01/execution_extensions/recovery_analysis_v2).

**Consequence:** Dynamic file loading, ROOT reassignment, source string replacement and separately hashed repairs are part of the result-production lineage. Training and analysis provenance are distinct.

**Recommended action [P]:** Retain repair source and attempt evidence. Future analysis pipelines should use explicit dependency inputs and separate result versions.

**Validation/rollback [P]:** Future compare saved table identities and repaired-report contracts; retain original failure/source/report versions. No recovery script was executed here.

### R15 — Integrity checks and production assumptions (High)

**Trigger:** Run an integrity-critical path under optimized Python or broaden it without explicit validations.

**Evidence [V/S]:** [check_seal](../../CoNN_Associative_Relationships/phase3b_context_diagnostics/src/training.py#L33); [train_job](../../CoNN_Associative_Relationships/src/conn_assoc/phase23_core.py#L203).

**Consequence:** Several scientific/configuration/resume integrity guards use assert. Python optimization can remove asserts. This is a conditional implementation risk, not evidence that historical execution used optimization.

**Recommended action [P]:** For a future maintained framework, use explicit exceptions for external/protocol/state checks and record interpreter flags; preserve historical implementations.

**Validation/rollback [P]:** Future injected mismatch fixtures and interpreter-mode checks when authorized; do not declare production readiness from source presence.

### R16 — Long-term maintainability and documentation (Medium)

**Trigger:** Treat the registry or copied packet as a complete execution specification.

**Evidence [V/S/U]:** [registry/README.md](../../registry/README.md); [ORGANIZED_EXPERIMENTS/ORGANIZATION_REPORT.md](../../ORGANIZED_EXPERIMENTS/ORGANIZATION_REPORT.md).

**Consequence:** Registry scope is intentionally study/attempt navigation, not every fit or environment dependency. Old README/proposal text has historical scope. Current module availability does not settle maintenance authority.

**Recommended action [P]:** Keep current navigation and dated evidence separate; add future richer manifests only through reviewed schema changes.

**Validation/rollback [P]:** Static link/hash/schema checks and explicit owner/replay-level decisions; leave current registry files untouched in this assessment.

## Preserved strengths and material limits

**V:** Source snapshots, protocol/configuration seals, prepared-data manifests, paired initial-state/batch/mask records, saved predictions, recorded verification outputs and separate failed/recovered attempts exist. The registry explicitly records completion separately from evaluation and independent reproduction. These provide a stronger starting point than a source-only collection and should constrain any future redesign.

**U:** No fresh test execution, checkpoint deserialization, autograd run, metric evaluation or end-to-end replay was performed. All 708 Python files were parsed/read and hashed, but scientific semantics were examined most closely in the 50 working modules and 14 compatibility families. Installed environments and third-party code are not the basis of architectural conclusions. External independent reproduction, per-fit execution source lineage, cross-environment numeric identity and future memory mechanisms remain unresolved. Historical verification claims are cited as records, not adopted as new verification results.

## Prioritized next decisions

1. **Protect evidence and decide replay targets.** Agree whether future support means artifact inspection, statistical recomputation, numerical replay or bitwise training replay. Preserve original paths and failures in all cases.
2. **Approve behavior contracts before extraction.** Resolve C01-C14 policies, especially bandwidth validation, RNG/batch order, context dtype, endpoint timing and statistical units.
3. **Design explicit dependency/provenance records.** Specify per-fit/per-attempt source closure, dataset/scaler/layout identity and checkpoint/digest versions, using the existing registry for navigation.
4. **Only after implementation approval, add small isolated components and one future study adapter.** Future fixtures and replay must use authorized new destinations; old source remains selectable.
5. **Treat performance, portability and expanded scientific claims as separate validation work.** Do not infer improvements from a cleaner directory tree or shared API.

## Validation performed for these reports

**V:** Read-only source discovery, byte hashing and AST parsing; symbol/line/path checks for generated evidence links; registry evidence-hash/excerpt checks; complete source inventory and distinct-content import coverage; static comparison of historical source stages; local Markdown/line-fragment and structural Mermaid checks; and pre/post preservation checks. Diagram rendering/parser execution, runtime import resolution and behavioral tests are not implied by structural diagram checks. Detailed final validation counts are reported in the completion notification. No new script, configuration, registry entry, source module or validation artifact is created beyond the five requested Markdown reports.

**P:** The recommended next step is review and approval of the architecture/contracts. Implementation and any further reorganization remain outside this phase.
