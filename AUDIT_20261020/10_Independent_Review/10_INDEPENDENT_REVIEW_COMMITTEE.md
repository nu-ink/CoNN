# 10 — Independent review committee

The six roles below reviewed the same record. They are not external people. They are the review lenses requested for this audit. Agreement is stated only where the artifacts support it.

## Machine-learning research scientist

The line of work is a careful ablation of a fixed spatial mask, followed by an attempt to add an external context gain. The AND result is a real inductive-bias result. It is the kind of result a local-connectivity paper can report for a toy task with planted neighborhoods. It is not a result about learned architecture search, dynamic neuron allocation, or memory. The later HAR and XOR results are the correct stress tests, and they bound the bias. I would not describe the project as having a transferable architecture.

## Applied mathematician

The Gaussian, the half-width identity, and the detached chain rule are internally consistent. The model at a fixed positive gate is a dense layer with a per-edge learning-rate scale `G_ij`. Underflow and very small gates approximate a sparse layer. The exact-ones phase deletes that structure. There is no degree of freedom that learns a graph. The context factor is a low-rank residual on the multiplier, initialized at a point of tiny slope and, originally, at rank one. The formula supplied as `tau(q)=sigma/r(q)` is not part of this mathematics.

## PyTorch and Python engineer

The training code is unusually explicit for a research prototype: paired streams, frozen masks, gate detachment called out in comments, test-evaluation counters, and condition identifiers that the runner checks. The main engineering hazards are a stale top-level README, a build-brief sentence that describes the wrong shuffle, a shared gate cache that must not be mutated, and the absence of Git. I did not find a path that turns a crashed fit into a successful score. Phase 3B's omitted C3 rows are labeled as omitted. I did not reload any `.pt` file.

## Experimental-design and inference specialist

1B's primary contrast is paired, same learning rate, new seeds, one test read, and the arithmetic checks out. That is a valid estimate of an AND alignment effect. It is not a valid estimate of a release effect, because S wins in the same table. 1C preregistered the release question and got an interval that includes zero. Experiment 2 preregistered engineered HAR and got a negative result. Phase 3B preregistered a 0.005 gain and did not reach it. The uncomfortable pattern is not hidden failures. The uncomfortable pattern is how much of the positive story sits on secondary rankings and on a subject bootstrap that a seed-level t interval does not support.

Overlapping cross-validation and selection on the same subjects mean Phase 2 and Phase 3B intervals are descriptive. Five engineering seeds are a small sample. The discrete 128-row synthetic test sets make one-example swings equal to 0.78 points.

## Reproducibility and integrity auditor

Score tables reproduce the headline means, including 0.0484375 and the Phase 3B bootstrap endpoints under seed 7000031. Protocols distinguish exploratory from confirmatory language more carefully than the infographic does. Gaps: no version control, checkpoints not re-audited byte for byte in this review, unit tests not re-run, and duplicate protocol trees that a later editor could diverge. The three copies of the Phase 3B final report that were hashed are identical. The associative README is not identical to the later record.

I see no evidence of test-set checkpoint picking in the runners that were read. I see documented, early use of a dense seed-0 validation run to make the AND task learnable. That is a design degree of freedom, disclosed before the multi-condition pilot.

## Skeptical peer reviewer

I would not recommend this body of work for publication as a new neural architecture or as a demonstration of collapsible, contextual, or associative computation.

I would consider a narrow methods note only if it led with the bounds: the mask helps when the coordinates are the coordinates of the labeling rule; a fixed mask can beat the schedule; the effect reverses when the labeling rule or the coordinate meaning changes; the context module did not meet its own criterion. The present infographic and the name "collapsible" would have to be removed or explicitly marked as non-descriptive.

The manuscript would need one decisive comparison that is still missing: schedule versus the best fixed width, with a same-weight gate swap as a required endpoint. Experiment 3D in this audit is that comparison. Until it is run, the defensible claim stops at "a fixed, task-aligned multiplicative mask can help a small MLP on a planted one-dimensional AND task."

## Joint assessment

| Question | Committee position |
| --- | --- |
| Is the algebra implemented? | Yes |
| Is there a repeated AND alignment effect? | Yes, with unstable magnitude |
| Is collapse observed as a change in network size or as a necessary release step? | No |
| Is the context mechanism useful under the authors' rule? | No |
| Is associative memory shown? | No |
| Is the work a valid negative-and-bounded positive record? | Yes, if the secondary wins are not promoted |
| Publication as a new architecture | Not supported |
