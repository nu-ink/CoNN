# 12 — Priority findings

Severity is scientific or engineering impact. Confidence uses the audit labels.

| Rank | ID | Severity | Confidence | Finding | Evidence | Next action |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | PF-01 | Critical | CONFIRMED | The implemented schedule expands a mask and then deletes it. It does not collapse network size. At the end of condition D the model is a dense MLP. The strongest AND accuracy in 1B belongs to the arm that never expands | `schedules.py`; 1B means S 0.937891, D 0.911328; infographic | Stop describing release as the mechanism. Run Experiment 3D before any further architecture claim |
| 2 | PF-02 | Critical | CONFIRMED | The repeated positive result is AND alignment versus a row shuffle, magnitude 0.0484375 in 1B and 0.025391 in 1C. The same contrast is negative on XOR and on engineered HAR | Score CSV recalculation | Treat alignment as a task-matched prior with demonstrated bounds, not as a general CoNN effect |
| 3 | PF-03 | High | CONFIRMED | A fixed medium bandwidth beats the scheduled release path on 1C AND (mean 0.948438) and on Phase 2 temporal validation (fixed d50=0.25 minus Gaussian +0.009291, 5/5 seeds) | 1C and Phase 2 tables | Make fixed-width the primary control in any follow-up. The shuffle alone no longer answers the open question |
| 4 | PF-04 | High | CONFIRMED | Phase 3 context factors stay near 0.9988. Phase 3B's best prespecified gain is +0.002134 macro-F1 and misses +0.005. The shuffled-code interval includes zero. Seed-level and subject-level t intervals on the primary contrast include zero | `paired_contrasts.csv`; `reporting.py` seed 7000031; this audit's t intervals | Do not start Phase 4 on this evidence. Do not cite the bootstrap lower bound without the seed table |
| 5 | PF-05 | High | CONFIRMED | `tau` and coordinates receive no gradient. Relational structure is not a learned object | `models.py` 26–28; `schedules.py` | Keep claims inside multiplicative curricula. A trainable gate would be a different model and would require removing the detach |
| 6 | PF-06 | High | CONFIRMED | Original context columns are identical, so the rank-2 factor starts at rank at most 1. After a small perturbation the second component remains near 0.003 of the first, below the authors' 0.01 reference | `phase3_context.py` 29–30; Phase 3B section 5 | Any new factor needs a prespecified initialization with a real second component, or an explicit rank-1 model |
| 7 | PF-07 | Medium | CONFIRMED | Engineered HAR D and B were selected at different learning rates (0.1 and 0.05). The negative sign also appears at matched rates on tuning validation, so selection did not manufacture it, but the published test contrast is still not a pure gate contrast | Experiment 2 scores and tuning scores | Report matched-rate test contrasts if this result is cited as mechanistic |
| 8 | PF-08 | Medium | CONFIRMED | The build brief says B′ scrambles destinations. The code permutes sources. The design note records the choice | Brief line 24; `DESIGN_DECISIONS.md` 22–26; `gates.py` 167–169 | Cite the design note and the code, not the destinations sentence |
| 9 | PF-09 | Medium | CONFIRMED | The associative README says no training has occurred. Phases 2, 3, and 3B trained thousands of fits. `PROTOCOL_v2.md` is a frozen file that still says later phases are unimplemented | README lines 3–5; run directories | A new reader should start from the phase reports. This audit did not edit the README |
| 10 | PF-10 | Medium | CONFIRMED | There is no Git repository. Fourteen thousand checkpoint files were not deserialized | Inventory | External replication depends on snapshots and CSV tables. Weight-level claims remain unchecked by this audit |
| 11 | PF-11 | Medium | NOT TESTED | `tau(q)=sigma/r(q)` is not in the project | Source search | Do not attribute that law to these experiments |
| 12 | PF-12 | Low | CONFIRMED | `phase23_core.py` computes `tau` twice with algebraically equal expressions. The second one is used | lines 109–118 | No scientific correction required |
| 13 | PF-13 | Low | CONFIRMED | Seventy C3 fits were omitted by a pilot futility rule and are labeled omitted | Phase 3B sections 6 and 13 | Precision-only training on the full grid is unresolved, not a negative grid result |
| 14 | PF-14 | Low | SUPPORTED | The CPU Gaussian cache returns shared tensors. Reviewed call sites do not mutate them | `phase23_core.py` 119–124 | Avoid in-place edits of returned gates in any later code |

## Recommended order of work

1. Accept the bounded AND-alignment result and the negative transfer results as the current evidence.
2. Do not add parameters, datasets, or a memory module to repair the context result.
3. If one more experiment is authorized, run the written 3D protocol unchanged, including its failure rule.
4. Leave original research files untouched.
