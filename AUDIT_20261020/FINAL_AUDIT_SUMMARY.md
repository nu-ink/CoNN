# Final audit summary

Date: 2026-10-20. Scope: `E:\CoNN`, read only. New files are confined to `AUDIT_20261020`. No training run, checkpoint load, package install, or edit of the original research was performed.

Detailed reports:

- `01_Repository_Inventory/01_REPOSITORY_INVENTORY.md`
- `02_Research_History/02_RESEARCH_TIMELINE.md`
- `03_Mathematical_Review/03_MATHEMATICAL_AUDIT.md`
- `04_Code_Architecture/04_ARCHITECTURE_AND_CODE_AUDIT.md`
- `05_Experimental_Analysis/05_EXPERIMENTAL_EVIDENCE.md`
- `06_Statistical_Verification/06_STATISTICAL_REPRODUCIBILITY.md`
- `07_Document_Contradictions/07_DOCUMENT_CONTRADICTIONS.md`
- `08_Hypothesis_Assessment/08_HYPOTHESIS_ASSESSMENT.md`
- `09_Experiment_3D_Proposal/09_EXPERIMENT_3D_PROPOSAL.md`
- `10_Independent_Review/10_INDEPENDENT_REVIEW_COMMITTEE.md`
- `11_Evidence_Traceability/11_EVIDENCE_TRACEABILITY.md`
- `12_Priority_Findings/12_PRIORITY_FINDINGS.md`

Independent arithmetic is in `Supporting_Evidence/recalculated_statistics.json`.

## What CoNN has demonstrated

The training code implements a fixed-width MLP whose hidden weight matrices are multiplied by a detached Gaussian of absolute distance on a prescribed line:

```
G(d; tau) = exp(-d^2 / (2 tau^2)),    W_eff = W ⊙ G
```

with `tau = d50 / sqrt(2 ln 2)` and a smoothstep schedule of `d50` from 0.08 to 1, then a hard switch to `G = 1` in the released conditions. Because `G` does not depend on `W`, the parameter gradient is `G ⊙ ∂L/∂W_eff`. That identity was derived and matches the code. It was not rechecked by rerunning autograd.

On the synthetic AND task, assigning the large gate values to the true neighboring features beats a within-row shuffle of those same values. The historical 20-seed Experiment 1B figure is real: mean test-accuracy difference **0.0484375**, sample SD 0.03171934, 95% paired t interval [0.03359239, 0.06328261], 19 seeds positive and 1 negative. A later 20-seed sample (Experiment 1C, secondary contrast, shared learning rate) finds **+0.025391** with an interval that still excludes zero. Initialization and data fingerprints match across conditions inside each 1B seed. Parameter counts match. This is a repeated alignment effect under a planted one-dimensional layout.

## What remains unproven

- That widening or exact dense release is why the AND effect appears. In the same 1B file, static local gating (S) averages **0.937891** against D at **0.911328**. The paired interval favors S.
- That any schedule beats the best constant width. Experiment 1C's primary comparison, partial widening versus full release, has a 95% interval that includes zero. Fixed `d50=0.25` has the highest observed AND mean in that study, 0.948438. On temporal HAR cross-validation, fixed `d50=0.25` also beats the scheduled Gaussian, by 0.009291 macro-F1, on all five development seeds.
- That the effect is a general geometric principle. XOR reverses it (−0.025391). Engineered UCI HAR reverses it (−0.005010 macro-F1, 17 of 20 seeds). Temporal HAR's held-out test contrast includes zero (+0.001861).
- That the network learns a relation, a context map, or a memory. `tau` and the coordinates are not parameters. The Phase 3 factor stays near 0.9988. Phase 3B's primary gain is **+0.002134** macro-F1. The protocol required +0.005 and a positive lower bound. The gain misses the practical bar. A t interval on the five seeds includes zero, as does a t interval on the 21 subject means. The frozen subject bootstrap excludes zero only under its own resampling rule, which this audit reproduced exactly with NumPy seed `7000031`. The shuffled-code control's interval includes zero. Phase 4 was not built.
- The bandwidth law `tau(q) = sigma / r(q)`. It does not occur in the repository.

## Confirmed implementation and reasoning errors

1. The public infographic describes neurons being added, merged, and retired. The code never changes width.
2. The build brief says the row-shuffle control scrambles destinations. The implementation permutes columns, which scrambles sources. A design note records the choice. The code and the note agree with each other. The brief sentence does not.
3. "Collapse" is used both for expansion of `tau` and for the later jump to a dense gate. The mathematics does the second after doing the first. Neither operation reduces the graph to a smaller learned subnetwork.
4. The original context factor is initialized with identical rank columns, so its attenuation has rank at most one, and it is initialized where the sigmoid slope is about 0.0025. Later measurements show the second component staying near three thousandths of the first.
5. The associative README still says that no model has been trained. The Phase 2/3 and Phase 3B directories contain the training.

No score-file arithmetic error was found in the headline contrasts listed above.

## Contradictions across experiments

These are different outcomes, not failed transcriptions. The numbers were recomputed.

| Comparison | Result |
| --- | --- |
| AND, aligned versus shuffle | Positive in the pilot, 1B, and 1C |
| AND, static local versus scheduled release | S better in 1B; unresolved and directionally reversed in 1C |
| XOR, aligned versus shuffle | Negative |
| Engineered HAR, D versus B | Negative, and D also below dense |
| Temporal HAR test, D versus B | Unresolved, interval includes zero |
| Temporal HAR cross-validation, Gaussian versus shuffle | Positive on the development folds |
| Temporal HAR cross-validation, fixed d50=0.25 versus schedule | Fixed width better |
| Context versus masked input | Approximately zero in Phase 3; +0.002 in Phase 3B, below the success rule |

Phase 2 and Experiment 2 do not measure the same split. Treating them as one HAR verdict would be a mistake. Both can stand: a development-fold gain for alignment on time coordinates, and no resolved gain on the official test subjects.

## Is the proposed collapse observable?

A change in forward connectivity is observable when `G` changes at fixed weights. At `d50=0.08`, about 42% of the synthetic input-layer gates are below `1e-6`, so the early map is approximately local. At `d50=1`, the smallest gate on that layer is 1/2. Exact release sets every gate to 1. Experiment 1C and the Phase reports separate that instantaneous swap from the updates that follow it.

What is not observable is a learned collapse. After release, the distinctive mask is gone. The deployed D network is a dense matrix with a particular training history. The arm that keeps a mask, S, is the one that posted the best 1B accuracy. Gate energy and `tau` moving on a schedule are not, by themselves, evidence that the trained function uses a collapsing structure.

The context mechanism is observable and small. Same-weight removal of `C` in Phase 3 changes macro-F1 by a few hundred-thousandths. Phase 3B's larger initialization still leaves learned factor movement far below the difference that random initialization already created between context codes.

## The most important unresolved question

Does a **change** in bandwidth during training do anything that the best **constant** bandwidth does not do, on the AND task, when the final weights are probed under both gates?

Alignment versus shuffle is no longer the open question. Release versus a fixed mask is. Experiment 1C asked it and did not resolve it. Experiment 3D in this directory is a smaller, precommitted version of that question, with a same-weight gate swap as a required clause and a written failure rule. It was not started.

## Recommendations for Experiment 3D

Hold the existing AND generator fixed. Compare one held-medium schedule with dense, fixed `d50` in {0.08, 0.25, 1}, and a row shuffle, at one shared learning rate. Require a final-accuracy margin over the best fixed width, a win over the shuffle, and a same-weight accuracy change when the terminal gate is replaced by exact ones. Failure if the best fixed mask ties or wins, if the shuffle interval contains zero, or if the same-weight swap is smaller than one accuracy point. Twenty-four new seeds. No HAR, no context module, and no change to the hypothesis after the results.

## Prioritized corrections and investigations

1. Describe the positive evidence as a task-aligned fixed mask effect on AND, with measured failures on XOR and engineered HAR.
2. Treat exact release and learned context as unsupported by the success rules those experiments set.
3. Do not cite `0.0484375` as evidence for collapse. Cite it as evidence against the shuffle, and cite S in the same table as evidence against release.
4. If a bootstrap interval is quoted for Phase 3B, quote the five seed signs with it. Two are negative.
5. Point new readers to the phase reports rather than the stale associative README or the infographic.
6. Leave the original files unchanged. Any new fit belongs under a new protocol, not a revision of these results.

## Incomplete sections of this audit

- No `.pt` checkpoint was unpickled. Claims that live only inside weight files, including bitwise identity of release pairs and the empirical range of `C` on saved tensors, were not re-derived from the arrays.
- Unit tests and the Phase 3B float64 autograd fixture were not re-executed.
- Preservation manifests that say earlier runs were unmodified were not re-hashed.
- Singular-value tables were accepted from the Phase 3B report rather than recomputed from factor matrices.
- There is no Git history. The timeline uses directory timestamps and document text.
- Virtual environments and the pip cache were excluded from the line review.
- Secondary Phase 3B bootstrap intervals were read from `paired_contrasts.csv`. Only the primary interval was redrawn from the subject table. That redraw matched the published endpoints exactly.

Those gaps do not sit under the headline score arithmetic. The 1B mean, the 1C primary interval, the HAR primary mean, the Phase 2 seed means, and the Phase 3B primary mean were computed from the CSV records.
