# 07 — Cross-document contradictions

Class labels: confirmed contradiction, possible contradiction, documentation ambiguity, missing evidence, resolved historical change.

## CON-01. Infographic versus implementation

**Confirmed contradiction.** `Collapsible Neural Network Infographic.png` describes a network that adds neurons, prunes and merges units, and returns to a smaller graph. Every trained model has fixed hidden widths. The scheduled conditions end at a dense gate or, for S and the fixed-d50 policies, at a static mask of the original width. No merge operator exists.

The build brief is closer to the code than the infographic is. The infographic is not a protocol.

## CON-02. "Collapse" versus the schedule

**Resolved historical change in wording, with a remaining ambiguity.** The brief's prediction is local-to-global expansion, then exact ones. Experiment 1C names the scientific question "locality versus collapse" and uses collapse for the dense-release endpoint. Both usages appear in the repository. They describe different events. Equations implement expansion of `tau` and a jump to `G=1`. They do not implement a return to a smaller active graph.

## CON-03. B′ scrambles destinations versus sources

**Resolved historical change.** Brief line 24 says a within-row column permutation and also says destinations are scrambled. Those cannot both be true. `DESIGN_DECISIONS.md` lines 22–26 chooses column permutation. `gates.py` implements that choice. Citations of the brief's "destinations" sentence are wrong relative to the data.

## CON-04. Bandwidth law `tau(q)=sigma/r(q)`

**Missing evidence.** The formula is an audit target supplied to this review. It is not in the source tree. The implemented law is `tau = d50 / sqrt(2 ln 2)` with `d50` moved by smoothstep from 0.08 to 1, then a hard ones-gate.

## CON-05. Associative README versus later training

**Confirmed contradiction as a status statement.** `CoNN_Associative_Relationships/README.md` lines 3–5 say Phase 0/1 only, and that no model, context gate, or memory bank has been trained. `PROTOCOL_PHASE23_v1.md`, the 1050-fit run, and Phase 3B contradict the training sentence. The memory-bank sentence remains true. The README was not updated when the amendment authorized training. `PROTOCOL_v2.md` still says Phases 2–6 are not implemented; the amendment explicitly leaves that file unchanged. That is a frozen historical document, not a description of the directory after 2026-10-19.

## CON-06. Experiment 2 temporal result versus Phase 2

**Possible contradiction if the two are treated as the same estimand.** Experiment 2 temporal test D minus B macro-F1 is +0.00186 with an interval that includes zero, on one 16/5/9 subject split. Phase 2 Gaussian minus row-shuffle is +0.0114 on seven-fold validation inside the official training subjects, with a selected learning rate and overlapping folds. Both numbers reproduce. They answer different questions. A sentence that says the HAR alignment effect both transferred and failed, without naming the split, would be a contradiction. The detailed reports keep the scopes apart. The executive risk is real because both are "temporal HAR, D versus shuffle."

## CON-07. Static local is better, and also is not

**Resolved historical change, seed-sensitive result.** 1B: S mean 0.9379, D mean 0.9113, interval for D minus S excludes zero. 1C: S mean 0.8914, D mean 0.9105, interval for S minus D includes zero. The reports do not pool the samples and 1C says the earlier ranking was seed-sensitive. That is an accurate description. It removes any standing claim that static locality dominates release.

## CON-08. Alignment helps, and also hurts

**Confirmed task dependence, not a bookkeeping error.** AND D minus shuffle is positive in 1B and 1C. XOR D minus shuffle is −0.0254. Engineered HAR D minus B is −0.0050. All three are in the score files. The hypothesis of a general geometry benefit is contradicted by the later tasks. The AND measurement is not contradicted.

## CON-09. Bootstrap seed 230319 versus 7000031

**Resolved if each phase is read with its own protocol.** Phase 2 code uses `230319`. Phase 3B code and `PROTOCOL_PHASE3B_v3.md` use `7000031`. The 3B report says a new seed can move historical interval endpoints. Substituting the Phase 2 seed into the 3B vector moves the interval and still excludes zero, so the qualitative bootstrap conclusion is stable to that particular swap. The exact published endpoints require `7000031`.

## CON-10. "Interval excludes zero" versus other intervals on the same 3B table

**Confirmed contradiction if the bootstrap is described as the unique uncertainty.** Subject bootstrap lower bound is positive. Subject-level t lower bound is −0.000214. Seed-level t lower bound is −0.001500. The report calls the bootstrap descriptive and says the practical criterion failed. It does not claim a confirmatory p-value. The residual contradiction is between a positive lower bound and the seed table, where seeds 402 and 404 are negative.

## CON-11. Context rank

**Resolved by measurement.** The amendment allocates rank 2. Identical initial columns make the attenuation rank at most 1. Phase 3B states this and measures `sigma2/sigma1` near 0.003 after a small perturbation, still below its 0.01 reference. The allocation and the achieved rank differ. The report does not hide that.

## CON-12. Capacity-matched context claims

**Documentation ambiguity.** Early Phase 3 comparisons of context versus masked models differ by 8,064 parameters. Phase 3B adds frozen, constant, and shuffled controls. The primary contrast remains C4 (86,406 trainable parameters) versus masked (78,342). The report says capacity is not identified by that contrast alone. A reader of only the primary line can still miss the imbalance. The shuffled contrast, which matches the C4 allocation, has an interval that includes zero.

## CON-13. d50 recorder constant

**Resolved.** Brief stores `d50 = 1.17741 * tau`. Code uses `sqrt(2 ln 2)` for the gate and stores the rounded factor beside it. The numerical difference is about `1.0e-8` relative. It does not move the schedule in any way that this audit can tie to an accuracy change.

## CON-14. Pilot `RESULTS.md` versus later experiments

**Resolved historical change.** `RESULTS.md` covers smoke and the five-seed pilot only. Later claims live in run-level `research_summary.md` and `research_report.md`. The root results file is not a summary of 1B, 1C, or HAR. Using it as the project result would omit the failures.

## Missing evidence

- No Git history.
- No deserialized confirmation of bitwise release pairs, despite extensive CSV claims.
- No Phase 4 retrieval metric.
- No experiment whose coordinates were learned from data.
- No preregistered test of `tau(q)=sigma/r(q)`.
