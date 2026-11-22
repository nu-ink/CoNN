# Core interface contracts

Status: **Proposed**. Design-freeze candidate **DF-2026-11-22-v1**. Prepared 2026-11-22 against branch initial-repository, HEAD 7a89cd51b1dcfb3526fd9212b0b395cd91317be7. No ADR is accepted and no runtime design is implemented by this documentation.

Labels: **V** verified current file/source fact; **S** static inference; **P** proposed requirement; **U** unresolved evidence or approval. Requirements below are normative only if explicitly approved. Historical provenance is never filled by inference.

Design documents: [architecture decisions](ARCHITECTURE_DECISIONS.md) | [interface contracts](INTERFACE_CONTRACTS.md) | [behavioral version matrix](BEHAVIORAL_VERSION_MATRIX.md) | [reproducibility contract](REPRODUCIBILITY_CONTRACT.md) | [repository architecture](REPOSITORY_ARCHITECTURE.md) | [github workflow](GITHUB_WORKFLOW.md) | [migration and validation plan](MIGRATION_AND_VALIDATION_PLAN.md).

## Authority and terminology

**P:** These ten interfaces are the proposed public boundary. Sketches are documentation-only type declarations, never imported or executed here. Names and references below are consistent with the reproducibility and behavioral contracts. Immutable means a deep immutable validated value; frozen dataclasses alone do not make nested mappings immutable. Runtime implementations must copy/freeze nested data and bind canonical digests before use.

## Shared type vocabulary and signatures

The complete sketch uses Python 3.11 notation as the proposed future target. Mapping/tuple fields have closed schemas specified below and in REPRODUCIBILITY_CONTRACT.md; they are not arbitrary plugin execution or unvalidated configuration dictionaries. TensorValue describes data, not a new tensor implementation. Runtime object selection is by a trusted fixed catalog, never eval, dotted-name import or sys.path mutation.

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Mapping, Sequence, Iterable, Literal

JsonValue = str | int | float | bool | None | Sequence["JsonValue"] | Mapping[str, "JsonValue"]

@dataclass(frozen=True)
class Digest:
    namespace: str
    algorithm: Literal["sha256"]
    value: str

@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    digest: Digest
    size_bytes: int
    media_type: str

@dataclass(frozen=True)
class BehaviorSelection:
    family_id: str
    behavior_id: str
    parameters: Mapping[str, JsonValue]
    parameters_digest: Digest
    implementation_digest: Digest
    reference_suite_version: str

@dataclass(frozen=True)
class ProtocolRef:
    protocol_id: str
    digest: Digest
    frozen_status: Literal["draft", "frozen"]
    frozen_at_utc: str | None

@dataclass(frozen=True)
class DatasetRef:
    dataset_id: str
    descriptor_digest: Digest

@dataclass(frozen=True)
class SplitDescriptor:
    split_id: str
    role: Literal["train", "validation", "test", "audit"]
    row_ids_digest: Digest
    subject_ids_digest: Digest | None
    schema: Mapping[str, JsonValue]

@dataclass(frozen=True)
class DatasetDescriptor:
    ref: DatasetRef
    sources: tuple[ArtifactRef, ...]
    parent_datasets: tuple[DatasetRef, ...]
    splits: tuple[SplitDescriptor, ...]
    representation: Mapping[str, JsonValue]
    transform_artifacts: tuple[ArtifactRef, ...]
    lineage: Mapping[str, JsonValue]

@dataclass(frozen=True)
class ModelSpec:
    architecture_id: str
    component_ids: tuple[str, ...]
    head: Literal["binary_logits", "multiclass_logits"]
    state_schema: Mapping[str, JsonValue]
    parameters: Mapping[str, JsonValue]

@dataclass(frozen=True)
class TrainingSpec:
    objective: Literal["bce_logits", "cross_entropy"]
    updates: int
    batch_size: int
    optimizer: Mapping[str, JsonValue]
    sampling_variant: str
    resume_policy: Literal["forbidden", "complete_state_only"]
    finite_check_policy: Mapping[str, JsonValue]

@dataclass(frozen=True)
class EndpointSpec:
    endpoint_id: str
    split_id: str
    completed_updates: int
    gate_step: int
    gate_mode: str
    context_policy: Mapping[str, JsonValue]

@dataclass(frozen=True)
class EvaluationPolicy:
    allowed_endpoints: tuple[EndpointSpec, ...]
    read_budgets: Mapping[str, int]
    selection_policy: Mapping[str, JsonValue]

@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    schema_id: Literal["conn-experiment-1"]
    protocol: ProtocolRef
    behavior: tuple[BehaviorSelection, ...]
    datasets: tuple[DatasetRef, ...]
    model: ModelSpec
    training: TrainingSpec
    evaluation: EvaluationPolicy
    analysis_plan: ArtifactRef
    seed_plan: ArtifactRef
    runtime_policy: ArtifactRef
    configuration_digest: Digest
    stage: Literal["smoke", "pilot", "development", "confirmatory"]

@dataclass(frozen=True)
class SourceClosure:
    closure_digest: Digest
    revision: str
    dirty: bool
    source_files: Mapping[str, Digest]
    environment: ArtifactRef

@dataclass(frozen=True)
class RunContext:
    run_id: str
    attempt_id: str
    output_root: str
    spec_digest: Digest
    source: SourceClosure
    parent_attempt_id: str | None

@dataclass(frozen=True)
class AccessGrant:
    run_id: str
    attempt_id: str
    dataset_id: str
    split_id: str
    purpose: Literal["development", "audit", "final_evaluation"]
    endpoint_ids: tuple[str, ...]
    policy_digest: Digest

class TensorValue(Protocol):
    shape: tuple[int, ...]
    dtype: str
    device: str

@dataclass(frozen=True)
class GateState:
    behavior_id: str
    value: TensorValue
    requested_bandwidth: float | None
    used_bandwidth: float | None
    floored: bool
    exact_dense: bool

@dataclass(frozen=True)
class ScheduleState:
    behavior_id: str
    update_index: int
    total_updates: int
    release_update: int
    phase: str
    boundary: str
    requested_bandwidth: float | None
    exact_dense: bool

@dataclass(frozen=True)
class Batch:
    features: TensorValue
    labels: TensorValue
    row_ids: tuple[str, ...]

class DatasetView(Protocol):
    descriptor: DatasetDescriptor
    split: SplitDescriptor
    grant: AccessGrant
    def rows(self, row_ids: Sequence[str]) -> Batch: ...

@dataclass(frozen=True)
class BatchPlan:
    indices: ArtifactRef
    effective_seeds: Mapping[str, int]
    cursor: int
    digest: Digest

@dataclass(frozen=True)
class CheckpointRef:
    envelope: ArtifactRef
    capability: Literal["endpoint_only", "resumable"]

class ModelHandle(Protocol):
    spec: ModelSpec
    initial_state_digest: Digest
    def forward(self, batch: Batch, gate_state: Mapping[str, TensorValue]) -> TensorValue: ...

@dataclass(frozen=True)
class FitResult:
    fit_id: str
    completed_updates: int
    checkpoints: tuple[CheckpointRef, ...]
    artifacts: tuple[ArtifactRef, ...]

@dataclass(frozen=True)
class PredictionTable:
    artifact: ArtifactRef
    endpoint: EndpointSpec
    pairing_keys: tuple[str, ...]

@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    class_universe: tuple[int, ...]
    reduction: str
    missing_policy: Literal["reject", "explicit_legacy_filter"]
    parameters: Mapping[str, JsonValue]

@dataclass(frozen=True)
class MetricTable:
    artifact: ArtifactRef
    metric_spec_digest: Digest
    pairing_keys: tuple[str, ...]

@dataclass(frozen=True)
class AnalysisPlan:
    plan_id: str
    digest: Digest
    inference_unit: str
    pairing_keys: tuple[str, ...]
    parameters: Mapping[str, JsonValue]

@dataclass(frozen=True)
class ContrastResult:
    artifact: ArtifactRef
    plan_digest: Digest
    interpretation: Literal["descriptive", "confirmatory"]

@dataclass(frozen=True)
class NewArtifactSpec:
    logical_path: str
    media_type: str
    role: str
    schema_id: str

@dataclass(frozen=True)
class CheckReport:
    scope: str
    status: Literal["passed", "failed", "not_run", "unresolved"]
    checks: Mapping[str, JsonValue]

@dataclass(frozen=True)
class ProvenanceEvent:
    schema_id: Literal["conn-run-1"]
    event_id: str
    previous_event_digest: Digest | None
    run_id: str
    attempt_id: str
    payload: Mapping[str, JsonValue]

@dataclass(frozen=True)
class ExecutionRequest:
    operation: Literal["train", "evaluate", "analyze", "verify"]
    endpoints: tuple[str, ...]
    authorization_ref: str

# I01: configuration
class ConfigurationService(Protocol):
    def validate(self, raw: bytes, format: Literal["json", "yaml"]) -> ExperimentSpec: ...

# I02: dataset registration and loading
class DatasetService(Protocol):
    def register(self, descriptor: DatasetDescriptor, context: RunContext) -> DatasetRef: ...
    def open(self, ref: DatasetRef, split_id: str, grant: AccessGrant) -> DatasetView: ...

# I03: mathematical component selection
class ComponentCatalog(Protocol):
    def validate(self, selections: tuple[BehaviorSelection, ...]) -> CheckReport: ...
    def geometry(self, selection: BehaviorSelection, layout: Mapping[str, JsonValue]) -> Mapping[str, TensorValue]: ...
    def gate(self, selection: BehaviorSelection, distance: TensorValue, bandwidth: float | None, exact_dense: bool) -> GateState: ...
    def schedule(self, selection: BehaviorSelection, update_index: int, total_updates: int) -> ScheduleState: ...

# I04: model construction
class ModelFactory(Protocol):
    def build(self, spec: ModelSpec, behavior: tuple[BehaviorSelection, ...], initial_state: ArtifactRef, context: RunContext) -> ModelHandle: ...

# I05: training
class Trainer(Protocol):
    def fit(self, spec: ExperimentSpec, data: DatasetView, model: ModelHandle, batches: BatchPlan,
            context: RunContext, resume: CheckpointRef | None) -> FitResult: ...

# I06: evaluation
class Evaluator(Protocol):
    def evaluate(self, checkpoint: CheckpointRef, data: DatasetView, endpoint: EndpointSpec,
                 policy: EvaluationPolicy, context: RunContext) -> PredictionTable: ...

# I07: metrics and statistical analysis
class MetricsService(Protocol):
    def compute(self, predictions: PredictionTable, spec: MetricSpec, context: RunContext) -> MetricTable: ...
    def estimate(self, measurements: MetricTable, plan: AnalysisPlan, context: RunContext) -> ContrastResult: ...

# I08: artifact management
class ArtifactService(Protocol):
    def write(self, spec: NewArtifactSpec, chunks: Iterable[bytes], context: RunContext) -> ArtifactRef: ...
    def read(self, ref: ArtifactRef) -> Iterable[bytes]: ...
    def check(self, ref: ArtifactRef) -> CheckReport: ...

# I09: provenance recording
class ProvenanceService(Protocol):
    def append(self, event: ProvenanceEvent, context: RunContext) -> ArtifactRef: ...
    def check(self, context: RunContext) -> CheckReport: ...

# I10: experiment execution through a registered adapter
class ExperimentAdapter(Protocol):
    adapter_id: str
    def execute(self, spec: ExperimentSpec, context: RunContext,
                request: ExecutionRequest) -> tuple[ArtifactRef, ...]: ...

class ExperimentRunner(Protocol):
    def plan(self, spec: ExperimentSpec, source: SourceClosure, datasets: tuple[DatasetDescriptor, ...]) -> CheckReport: ...
    def create(self, spec: ExperimentSpec, source: SourceClosure, datasets: tuple[DatasetDescriptor, ...], authorization_ref: str) -> RunContext: ...
    def execute(self, adapter: ExperimentAdapter, spec: ExperimentSpec,
                context: RunContext, request: ExecutionRequest) -> tuple[ArtifactRef, ...]: ...
```

## Interface-by-interface contracts

### I01 — ConfigurationService

**Inputs/outputs:** Raw JSON/YAML bytes with declared format -> validated ExperimentSpec; retain original bytes and normalized conn-json-1 digest.

**Invariants:** Require exactly C01-C14 once each, matching family/behavior ID, complete variant parameters, frozen protocol for execution, closed field schemas, valid split/endpoint IDs and positive finite budgets. Reject duplicate mapping keys; booleans are not numeric counts.

**Failure conditions:** Reject unknown/ambiguous fields or behavior IDs, NaN/Inf, implicit defaults affecting science, invalid references, incompatible head/objective and missing analysis/seed/runtime plans. Parsing never imports a plugin or allocates a run.

**Responsibilities and effects:** Own validation and normalized identities, not data access, dependency installation or scientific selection.

**Required reference validation:** T-C08 and T-P01; ADR-0002/0003.

### I02 — DatasetService

**Inputs/outputs:** DatasetDescriptor + new RunContext -> DatasetRef registration artifact; DatasetRef + split/grant -> read-only DatasetView.

**Invariants:** Bind byte identities, parent data, row/subject membership, feature/channel order, dtype/shape, representation and transforms. Synthetic descriptors include generator/source, effective seed recipes and realized split hashes before model training. Grant binds run/attempt/split/operation/endpoints.

**Failure conditions:** Missing bytes or inconsistent hashes/schema, unauthorized test access, mismatched split/grant, mutable input locations or unresolved new-run lineage fail closed. No implicit download, normalization, shuffling or fit.

**Responsibilities and effects:** Register descriptors in new-run artifacts; proposed catalog registration is explicit and must not edit existing registry files automatically. Preprocessing is a separately authorized derived-data operation.

**Required reference validation:** T-C01/C07, T-R01 and T-P01; ADR-0004/0005.

### I03 — ComponentCatalog

**Inputs/outputs:** Complete BehaviorSelection tuple -> CheckReport; explicit geometry/gate/schedule arguments -> typed values.

**Invariants:** Select only trusted implementation bindings with recorded source digest. No Python dotted-name loading from ID. All variants, arithmetic domains, detach/cast, axes, RNG algorithms and exact-dense state are explicit. bandwidth=None is valid only for exact_dense=True.

**Failure conditions:** Unknown/unvalidated binding, parameter mismatch, wrong axes/dtype/device, invalid bandwidth/domain or schedule index fails before training. Preserve policy-specific legacy errors only under explicitly selected legacy adapters.

**Responsibilities and effects:** Pure math/component lookup, no IO, dataset introspection, environment mutation or registry writes. GateState records requested/used bandwidth, floor and exact-dense state; ScheduleState identifies the about-to-execute update and requested bandwidth. The adapter passes GateState.value to the model without discarding its provenance.

**Required reference validation:** T-C01..T-C06 and T-C12/C13; ADR-0002.

### I04 — ModelFactory

**Inputs/outputs:** ModelSpec + complete BehaviorSelection tuple + explicit initialized-state artifact + RunContext -> ModelHandle.

**Invariants:** State schema includes named parameters, shapes/dtypes and requires_grad groups; binary [N] versus multiclass [N,K] head agrees with objective. Context variant and cast points declared. Constructor allocation is not initialization.

**Failure conditions:** Missing/unverified state, key/head/shape/dtype/group mismatch or incompatible behavior set fails before optimizer construction. No automatic reinitialization or partial-key acceptance.

**Responsibilities and effects:** Model assembly with named initialization; callers cannot change scientific architecture through an inferred checkpoint shape.

**Required reference validation:** T-C03/C11/C12/C14; ADR-0002/0006.

### I05 — Trainer

**Inputs/outputs:** ExperimentSpec (including TrainingSpec, complete behavior and endpoint plans), authorized training DatasetView, ModelHandle, frozen BatchPlan, RunContext and optional checkpoint -> FitResult.

**Invariants:** Use declared loss/update count/batch and diagnostic cadence; finite checks and mode transitions are frozen. No test DatasetView accepted. Resume requires resumable schema, exact spec/profile/plan/source lineage and recorded next-update cursor.

**Failure conditions:** Invalid view, mismatch, nonfinite values per policy, exhausted budget, bad cursor or incomplete restore state raises a typed failure and retains committed artifacts. No silent restart or fallback.

**Responsibilities and effects:** Perform updates only for an authorized operation; use injected artifact/provenance services bound at composition, write only within context. Protocol-specific logic belongs to the adapter.

**Required reference validation:** T-C05/C06/C09/C11/C14 and T-R02; ADR-0003/0006.

### I06 — Evaluator

**Inputs/outputs:** CheckpointRef + authorized DatasetView + EndpointSpec + EvaluationPolicy + RunContext -> PredictionTable.

**Invariants:** Verify model/checkpoint IDs, planned weight/update/gate step, context/mask, split role and access grant. Enforce event-level test budget before reading test tensors, reserve evaluation attempt durably, and count failed attempts as access where data was read.

**Failure conditions:** Undeclared/repeated endpoint beyond budget, unauthorized purpose, mismatched split/checkpoint or partial trained state fails closed. No tuning, checkpoint selection or extra test endpoint creation.

**Responsibilities and effects:** Separate evaluation from fitting; save predictions and complete pairing keys with hashes so metrics can be recalculated without retraining. A deliberate pilot may end without evaluation.

**Required reference validation:** T-C05/C07/C09/C14 and T-R02; ADR-0003/0004.

### I07 — MetricsService

**Inputs/outputs:** PredictionTable + MetricSpec -> MetricTable; matched measurements + frozen AnalysisPlan -> ContrastResult.

**Invariants:** Compute binary/multiclass metrics with explicit class universe, absent-class/zero-division/reduction rules. Freeze seed/subject inference unit, weights, pairing, missing/nonfinite policy, RNG, sidedness, ties and multiplicity before analysis.

**Failure conditions:** Missing/duplicate pairing keys, changed plan, nonfinite values or insufficient sampling units fail rather than silently filter unless a named legacy policy explicitly requires filtering.

**Responsibilities and effects:** No model fitting/evaluation; raw arithmetic can be pure, but this service persists new versioned outputs through ArtifactService. Pure reference kernels must be separable from writing.

**Required reference validation:** T-C10 and T-P02; ADR-0002/0005.

### I08 — ArtifactService

**Inputs/outputs:** NewArtifactSpec + byte chunks + RunContext -> immutable ArtifactRef; ref -> verified byte stream or CheckReport.

**Invariants:** Logical paths are normalized root-relative names; reject absolute paths, traversal, symlink escapes and existing destinations. Exclusive creation, fsync where supported, atomic final publication and file-byte SHA-256. Hash before marking complete.

**Failure conditions:** Collision, incomplete stream, failed replace, bytes/hash mismatch or write outside context yields explicit failure. Keep partial data separately identified; never replace an original artifact or report pass on missing data.

**Responsibilities and effects:** Own storage mechanics, no interpretation of model pickle or training semantics. A local resolver maps identity to explicit approved locations; it never searches sibling run folders.

**Required reference validation:** T-M01, T-R01 and T-P02; ADR-0001/0005/0006.

### I09 — ProvenanceService

**Inputs/outputs:** ProvenanceEvent + RunContext -> immutable event ArtifactRef; context -> integrity CheckReport.

**Invariants:** Use conn-run-1 fields and a per-run event sequence with predecessor hash; validate source/data/config references and attempt parentage. Append has compare-and-swap on expected predecessor and single-writer ownership; no stale writer wins.

**Failure conditions:** Missing required new-run provenance, invalid transition, changed predecessor, duplicate terminal event, unknown parent or unresolved required closure fails registration. Local registration is not external timestamping.

**Responsibilities and effects:** Store events in new output roots. Current registry is a manually reviewed index; explicit later registration can reference manifests without retroactively filling historical gaps.

**Required reference validation:** T-P01/T-P02 and T-R02; ADR-0005.

### I10 — ExperimentRunner and ExperimentAdapter

**Inputs/outputs:** Validated ExperimentSpec + SourceClosure + explicit DatasetDescriptor tuple -> CheckReport plan; authorized create with the same inputs -> unique RunContext; explicit adapter/request -> new artifacts.

**Invariants:** plan is read-only and does not import/execute arbitrary adapter code. Confirmatory source must be clean and frozen. Allocate timestamp+random run ID with collision refusal. Every operation requires authorization_ref, declared endpoints and matching adapter/profile/spec.

**Failure conditions:** Unknown/unapproved adapter, planning failure, changed seal, missing access/budget/environment or occupied output root stops execution before effects. Metadata validation success never authorizes optimizer/evaluation calls.

**Responsibilities and effects:** Runner owns lifecycle and services; adapter owns scientific definition. No shell command strings from config, hidden path imports or automatic staging/commit/push.

**Required reference validation:** T-R01/T-R02/T-P01/T-E01; ADR-0003/0007/0008.

## Cross-interface composition rules

- Type names above are the canonical vocabulary. ArtifactRef is byte identity, CheckpointRef adds a schema/capability envelope, DatasetRef identifies a descriptor, and SourceClosure identifies exact transitive code/environment. None is interchangeable with a path or a Git commit alone.
- Scientific kernels return values without effects. Dataset registration, metric-table persistence, artifact writes, provenance append and runner execution are explicitly effectful services bound to RunContext. Artifact and provenance services do not import trainers; dependency injection is done by the runner composition root.
- A DatasetView never grants more access than its AccessGrant. An audit grant permits declared structural inspection only and is rejected by Evaluator. A final-evaluation grant does not permit fitting or selection. Development examples and model state remain separate from the protected test-access ledger.
- Planning receives dataset descriptors directly and validates their identities read-only; it does not require registration into a not-yet-created run. create revalidates the same spec/source/descriptors before allocation. DatasetService.register then writes a run-local copy of the descriptor. For generated data, the frozen descriptor identifies the generator/recipe commitment; each realized split is a new immutable child artifact linked to that commitment, not an in-place descriptor/spec edit. Actual train/validation hashes are sealed before fitting, and test realization follows the reproducibility contract.
- The runner freezes the complete planned endpoints, pairing and analysis policy before fitting. Changing seeds/controls/scaler/head/precision/statistical unit creates a new protocol/run rather than an unnoticed retry.
- Error taxonomy for future implementations: ContractError, UnknownBehaviorError, AccessDeniedError, IdentityMismatchError, ArtifactCollisionError, IncompleteStateError, BudgetExceededError and ConcurrentWriteError. Protocol/integrity guards use explicit exceptions, not optimization-removable asserts. Every effectful failure produces an attributable attempt event when durable storage is available; if storage is unavailable it remains unregistered, never inferred complete.
- Cancellation/interrupt records the last committed checkpoint and ledger position; resumability requires the complete schema. Evaluation access reservation happens before data access so interrupted evaluations cannot evade the budget. Budget changes require a new approved policy, not deletion of the reservation.
- Legacy adapters are opt-in readers and profiles, never compatibility promises. Unknown historical provenance is represented in a separate evidence annotation, not fabricated SourceClosure required for a new conforming run.

## Validation boundary

These type sketches will be AST-parsed and annotation names cross-checked as documentation. They are not type-checked against a tensor library or executed, and do not establish numerical behavior. Future validation uses the T-C01..T-C14 suites and M0..M7 gates. All thresholds/environment-specific tolerances must be agreed before running those suites; Phase 4 does not run them.
