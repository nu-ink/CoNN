# CoNN Research Validation

This read-only documentation package assesses the existing CoNN implementation and recorded experiments. It does not change application code or experimental data.

## Status

**Overall assessment:** INCONCLUSIVE / PARTIALLY SUPPORTED.

Task-aligned gating has a repeatable positive result in the controlled synthetic AND 1B replication, but this result is task-specific and did not transfer consistently to UCI HAR. Partial-widening evidence remains unresolved. Context-modulation results were exploratory and did not meet the Phase 3B predictive criterion. No implementation demonstrates associative memory or learned relationship geometry.

## Documents

1. [Experiment inventory](01_EXPERIMENT_INVENTORY.md)
2. [Mathematical validation](02_MATHEMATICAL_VALIDATION.md)
3. [Existing experimental results](03_EXISTING_RESULTS.md)
4. [Hypothesis assessment](04_HYPOTHESIS_ASSESSMENT.md)
5. [Experiment 04 proposal](05_EXPERIMENT_04_PROPOSAL.md)
6. [Experiment 04 mathematics](06_EXPERIMENT_04_MATHEMATICS.md)
7. [Experiment 04 statistical plan](07_EXPERIMENT_04_STATISTICAL_PLAN.md)
8. [Experiment 04 implementation plan](08_EXPERIMENT_04_IMPLEMENTATION_PLAN.md)
9. [Unresolved questions](09_UNRESOLVED_QUESTIONS.md)
10. [Evidence index](10_EVIDENCE_INDEX.md)

## Diagrams

- [Master experiment map](diagrams/01_master_experiment_map.mmd)
- [Experiment 01 flow](diagrams/02_experiment01_flow.mmd)
- [Experiment 1B/1C comparison](diagrams/03_experiment1b_1c_comparison.mmd)
- [Experiment 02 HAR flow](diagrams/04_experiment02_har_flow.mmd)
- [Phase 2 schedule and architecture](diagrams/05_phase2_architecture.mmd)
- [Phase 3/3B context modulation](diagrams/06_phase3_context_modulation.mmd)
- [HAR pipeline](diagrams/07_har_pipeline.mmd)
- [Experiment 04 proposed architecture](diagrams/08_experiment04_proposed_architecture.mmd)
- [Experiment 04 conditions](diagrams/09_experiment04_conditions.mmd)
- [Experiment 04 lifecycle](diagrams/10_experiment04_lifecycle.mmd)

The diagram sources are Mermaid; implementation diagrams describe verified code, while Experiment 04 sources are prominently labeled **PROPOSED — NOT IMPLEMENTED**.

