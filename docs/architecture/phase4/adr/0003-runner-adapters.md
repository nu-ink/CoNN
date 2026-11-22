# ADR-0003: Experiment runner and adapter architecture

Status: **Proposed**. Design-freeze candidate **DF-2026-11-22-v1**. Prepared 2026-11-22 against branch initial-repository, HEAD 7a89cd51b1dcfb3526fd9212b0b395cd91317be7. No ADR is accepted and no runtime design is implemented by this documentation.

Labels: **V** verified current file/source fact; **S** static inference; **P** proposed requirement; **U** unresolved evidence or approval. Requirements below are normative only if explicitly approved. Historical provenance is never filled by inference.

Design documents: [architecture decisions](../ARCHITECTURE_DECISIONS.md) | [interface contracts](../INTERFACE_CONTRACTS.md) | [behavioral version matrix](../BEHAVIORAL_VERSION_MATRIX.md) | [reproducibility contract](../REPRODUCIBILITY_CONTRACT.md) | [repository architecture](../REPOSITORY_ARCHITECTURE.md) | [github workflow](../GITHUB_WORKFLOW.md) | [migration and validation plan](../MIGRATION_AND_VALIDATION_PLAN.md).

## Context and problem

Current runners embed study-specific objectives, batching, gates, selection, reporting and resume behavior. A universal loop would silently standardize differences.

## Available alternatives

1. One configurable training loop for all studies: rejected pending evidence.
2. Every adapter owns all IO/provenance: duplicates unreliable orchestration.
3. Runner coordinates declared adapter operations and artifact/provenance services: selected.

## Recommended decision

The runner validates/freeze-checks a complete ExperimentSpec, allocates a new RunContext, and invokes adapter train/evaluate operations under policy. Adapter declares model/objective, sampling, endpoint timing, finite-check cadence, test budget and resume eligibility. Statistics and reporting are separate operations. No adapter may discover historical inputs through sibling paths.

## Supporting evidence

- [train](../../../../conn_experiment_01/src/conn_exp/har_core.py#L138)
- [train_job](../../../../CoNN_Associative_Relationships/src/conn_assoc/phase23_core.py#L203)
- [evaluate_run](../../../../experiments/experiment_04/src/conn_exp04/runner.py#L194)

## Expected benefits

Centralizes safe output/lifecycle rules while preserving scientific study boundaries.

## Tradeoffs and risks

Adapters may share fewer loops initially. Event ordering and protocol selection remain explicit rather than inferred.

## Compatibility implications

C05.explicit-endpoint.v2, C06.explicit-plan.v2 and C09.explicit-lifecycle.v2 require no hidden test reads or checkpoint selection. Full resume is allowed only when the schema and adapter declare complete state.

## Validation requirements

T-R01 isolation, T-R02 state transitions, T-C05/C06/C09/C14. Manual scientific review of objective/control/endpoints before any authorized execution.

## Rollback strategy

Disable the new runner or select a previous adapter version for new attempts. Preserve terminal events and failed outputs; never overwrite a parent failure with a later success.

## Approval and supersession

Status: **Proposed**. No acceptance date, approver or runtime implementation exists. Explicit user approval is required before this decision becomes operative. A later accepted or superseding ADR must identify its predecessor; it must not rewrite a historical scientific protocol.
