# Phase 3B v2 historical reference

The complete v2 reference was supplied by the user in the conversation on 2026-10-20, in response to the missing-specification question. It is a historical design reference, not a protocol previously frozen or executed. The operative design is PROTOCOL_PHASE3B_v3.md and the original v3 attachment archived with each run.

Inherited definitions: C0 historical float32 U=-6,V=0; C1 adds independent elementwise Uniform[-.2,.2] perturbations using generator seed 30031 to U and V; C2 float32 U=-4.5,V=0; C3 float64 context master parameters and SGD updates, historical initialization, modulation cast to float32 before effective-weight multiplication; C4 combines C2 initialization, C1 perturbations and C3 precision. No context LR multiplier. The categorical lookup, original Gaussian schedule, two 64-unit hidden layers, batch 128, plain SGD .1, 3200 steps, seeds 400–404 and seven historical subject folds remain unchanged. All remaining inherited requirements are explicitly incorporated in v3 below.

This file is a labeled synopsis, not a verbatim transcript or a substitute for the supplied conversation reference. No v2 experiment is claimed.
