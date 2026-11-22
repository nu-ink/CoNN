# Central Hypothesis Assessment

## Hypothesis

> Meaningfully aligned relationship geometry can improve learning or generalization compared with arbitrary relationship geometry under comparable training conditions.

## Claim-by-claim assessment

| Claim | Assessment | Rationale |
|---|---|---|
| Fixed task-aligned gating improves synthetic AND versus row shuffle | **PARTIALLY SUPPORTED** | Pilot and 1B replication show positive D − B′; 1C alignment contrasts are task-dependent |
| The advantage persists after fair tuning | **SUPPORTED WITHIN 1B** | 1B used predefined validation tuning and independent seeds |
| The advantage transfers to real-world HAR | **CONTRADICTED** for engineered primary | Engineered D − B macro-F1 was negative |
| Partial widening is universally useful | **INCONCLUSIVE** | 1C P − D interval included zero |
| Local fixed gates outperform wider gates | **PARTIALLY SUPPORTED/SEED-SENSITIVE** | 1B S was higher than D in means; 1C S/D rankings changed |
| Context factors improve external-context performance | **NOT SUPPORTED** | 3B missed predictive threshold |
| Relationships are learned | **NOT TESTED / FALSE FOR BASE GATING** | Base gate parameters are prescribed and gates are detached |
| Associative-memory capability exists | **NOT TESTED / NOT IMPLEMENTED** | No memory bank or retrieval API was found |
| Geometry is proven as a mechanism | **NOT SUPPORTED** | Shuffled controls alter alignment and topology together; many results are task-specific |

## Unverified or incomplete claims

- No experiment establishes that gate geometry alone is causal.
- No result establishes superiority across tasks or distributions.
- No formal verification of the research behavior was performed.
- No implementation learns relationship geometry.
- HAR and Phase 3B use fixed subject partitions and cannot establish population-wide robustness.
- Multiple secondary contrasts are exploratory and unadjusted.

