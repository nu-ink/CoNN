# 02 — Research timeline

Dates below are the UTC timestamps embedded in run directory names and the dates written in protocol headers. There is no Git history to confirm them independently.

## Chronology

| When | Event | Status of the claim at that time |
| --- | --- | --- |
| Before any run | `CURSOR_BUILD_BRIEF.md` defines a training-curriculum test: task-aligned local-to-global Gaussian gating versus magnitude-matched controls. Primary contrast is final test accuracy of D minus B′ on a synthetic AND task. The brief states that a Gaussian kernel or a Hoberman mechanism is not being claimed as new. | Original hypothesis and pre-test prediction |
| Design record | `DESIGN_DECISIONS.md` fixes coordinates, the within-row column permutation, the three-phase path for E, and SGD. It also records that noise, distractor amplitude, and learning rate were adjusted after a dense-only seed-0 validation check, before the other conditions were trained. | Implementation choice, with a documented pilot-design adaptation |
| 2026-10-17 23:57Z | Smoke run, 30 steps, width 16. All seven conditions finish at test accuracy 0.500. | Pipeline check. Not evidence about the curriculum |
| 2026-10-17 23:58Z | Five-seed pilot, 400 steps, shared learning rate 0.1. D minus B′ mean +0.021875, 95% t interval [0.007487, 0.036263]. | Exploratory observation. `RESULTS.md` withholds a confirmatory claim |
| 2026-10-18 00:19Z | Experiment 1B. Validation-only grid on seeds 5–9, then replication seeds 100–119 at 3200 steps. D and B′ both use learning rate 0.1. Mean D minus B′ = +0.0484375. Static-local S mean accuracy 0.937891 exceeds D at 0.911328. | Replication of the alignment contrast. Secondary observation that staying local scored higher than release |
| 2026-10-18 02:41Z | Experiment 1C, seeds 200–219. Primary partial-width P minus full-release D has a 95% interval that includes zero. Fixed d50=0.25 has the highest observed AND mean (0.948438). XOR reverses the alignment contrast. | The release-versus-local question is unresolved on the primary contrast. Task dependence is measured |
| 2026-10-19 00:43Z | Experiment 2, UCI HAR, seeds 300–319. Primary engineered D minus B macro-F1 = −0.005010, interval excludes zero in the negative direction. Temporal D minus B interval includes zero. | The synthetic alignment advantage does not transfer to the engineered-feature protocol |
| Frozen v2 | `PROTOCOL_v2.md` asks whether Gaussian connectivity improves relationships, context sensitivity, and associative retrieval. It says Phases 2–6 are not implemented and bars official-test evaluation. | New hypothesis, broader than Experiment 1. Later phases amend the implementation status without editing this file |
| 2026-10-19 03:04Z | Phase 2/3 development grid, 1050 fits, temporal HAR, seven subject folds, seeds 400–404, validation only. Gaussian minus row-shuffle subject-macro-F1 +0.011375. Context minus masked is about −0.0004. Measured context factors stay near 0.9988. | Development observation on overlapping folds. Context learning is numerically tiny. No retrieval experiment |
| 2026-10-20 | Phase 3B changes context initialization, symmetry, and storage precision. Primary Gaussian C4 minus Gaussian masked = +0.002134. The frozen +0.005 practical criterion is not met. Level 2 pair checks pass 49/175. C3's 70 grid fits are omitted after a pilot futility rule. | Numerical repairs are separated from predictive benefit. Associative memory remains untested |

## What each layer of the record is

**Original hypotheses.** Structured local-to-global gating would improve AND learning relative to a row-energy-matched shuffle (`CURSOR_BUILD_BRIEF.md`). Later, Gaussian connectivity would improve context sensitivity and associative retrieval (`PROTOCOL_v2.md`, research question).

**Predictions recorded before the corresponding test.** D minus B′ positive on AND. Experiment 1C primary was P minus D, not a restatement of D minus B′. Experiment 2 primary was engineered D minus B macro-F1. Phase 3B primary was a +0.005 macro-F1 gain of C4 over the masked Gaussian with a positive bootstrap lower bound.

**Observations measured.** The AND alignment difference is positive in the pilot, in 1B, and again as a 1C secondary contrast, with different magnitudes. S exceeds D in 1B and does not in 1C. XOR alignment difference is negative. Engineered HAR alignment difference is negative. Temporal HAR test alignment difference is small. Phase 3 context weights barely move under the original initialization. Phase 3B C4 moves more, and its predictive gain stays near two thousandths of macro-F1.

**Interpretations added after measurement.** Experiment 1C's `measured_findings.md` section is generated from saved scores and is labeled post-training interpretation. Phase 3B hypotheses H4 and H5, about shared weights compensating for attenuation and about mask information already present in zeroed inputs, are left unresolved by the authors. The infographic's story of neurons growing, merging, and returning to a small idle network is not the experimental hypothesis in the build brief.

**Conclusions that the measurements support on their own.** Under the synthetic AND geometry, assigning large gate values to nearby task coordinates beats a within-row reassignment of those same values. Releasing the gate to exact ones is not required for the best observed AND accuracy in these protocols. The same alignment advantage is absent on XOR and on the engineered HAR layout. A trainable context factor of the form implemented here did not produce the prespecified practical gain, and no associative-retrieval metric exists.

## Abandoned or narrowed claims

- The infographic's variable neuron count, merge operation, and return to a smaller network were never the training code.
- A claim that widening or exact dense release is the useful event is in tension with S in 1B and with fixed d50=0.25 in 1C and Phase 2.
- Phase 4 memory was not started. The Phase 3B report recommends against treating the contextual gate as a reason to start it.
- The associative `README.md` still says no model has been trained. That sentence describes the Phase 0/1 freeze. It is stale as a description of the directory after 2026-10-19.

## Finding

**HIST-01.** The research question changed from "does a prespecified local-to-global mask curriculum beat a row shuffle on AND?" to "does a learned low-rank context factor create useful external-context behavior?" The second question was tested. Retrieval was not. Confidence: **CONFIRMED**.
