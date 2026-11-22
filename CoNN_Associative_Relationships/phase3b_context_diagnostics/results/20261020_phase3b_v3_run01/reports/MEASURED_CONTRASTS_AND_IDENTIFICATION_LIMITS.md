# Measured contrasts and identification limits

These results come from the completed, independently verified development grid. The serialization-only reporting recovery preserved the contrast, subject and seed CSVs byte for byte.

| Contrast | Mean paired subject macro-F1 difference | Descriptive 95% subject-bootstrap interval |
| --- | --- | --- |
| Gaussian C4 − Gaussian masked (primary) | +0.00213398 | [+0.00011566, +0.00444627] |
| Gaussian C4 − historical C0 (exploratory) | +0.00255189 | [−0.00044630, +0.00598871] |
| Gaussian C4 − frozen C4 (exploratory) | +0.00266374 | [+0.00018687, +0.00530247] |
| Gaussian C4 − constant-code C4 (secondary) | +0.00332958 | [+0.00019395, +0.00702091] |
| Gaussian C4 − shuffled-code C4 (secondary) | +0.00274569 | [−0.00060541, +0.00691192] |
| Dense C4 − Dense masked (secondary) | −0.00102141 | [−0.00289158, +0.00050925] |

The primary mean is positive and its descriptive interval has a positive lower endpoint. It nevertheless misses the prespecified +0.005 practical threshold. Report this as a modest exploratory development improvement, not as meeting Level 3, independent confirmation, or proof of general superiority. Do not describe it as demonstrating no effect either.

The corrected model's exploratory comparison with historical C0 is less clearly separated: its interval spans zero. The numerical recovery should therefore not be equated with a confidently established predictive improvement over the original contextual implementation.

The frozen-gate comparison provides descriptive evidence of an incremental benefit beyond C4's initial attenuation in this study. The constant-code comparison also has a positive lower interval, but constant-code training exposes fewer active lookup rows. The shuffled-code contrast spans zero. Together these controls do not consistently isolate additional useful sensor-availability information from capacity, parameterization, optimization, or distribution-shift effects. None is an equivalence test.

Gaussian C4 passed Level 1 in all 35 grid fits. Forty-nine of 175 eligible layer/context-pair diagnostic checks passed the frozen Level 2 criteria. These are pair-level outcomes, with initial random differentiation and the shuffled-code null recorded separately. They do not establish uniform, population-wide learned contextual differentiation. First-layer accel-versus-gyro intersection comparisons remain undefined.

Weak gradients and representability problems were measured, and symmetry breaking removed an exact mathematical restriction. Float64 precision alone failed the recorded forward-level pilot threshold. Those observations narrow the numerical diagnosis, but H4 architectural redundancy and H5 limited additional context information remain unresolved. Remaining weak gradients, weak second-component strength, the fixed factor-noise seed, overlapping development folds and historical preprocessing limit causal identification. No official-test evaluation or associative-memory experiment was performed.
