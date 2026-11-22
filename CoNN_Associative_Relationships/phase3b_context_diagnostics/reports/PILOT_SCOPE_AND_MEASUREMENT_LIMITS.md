# Scope and measurement limits

These qualifications describe the frozen design and recorded fields, independently of the final primary outcome.

- Each architecture's C3 futility decision uses one seed400/fold0 pilot. It is an operational allocation rule, not evidence that precision-only learning is impossible for every seed or fold. Both full pilot trajectories and their measurements are retained.
- C1's deterministic perturbations broke symmetry but did not meet the initial sigma2/sigma1 target of .01. A mathematically rank-two gate is not necessarily a strong test of useful two-component capacity.
- Factor seed30031 is fixed across base seeds400–404. Reported seed uncertainty does not cover alternative factor-noise realizations.
- Level2 is assessed per layer/context pair against the Gaussian shuffled-training null. No study-wide aggregation rule converts a few passing pairs into universal context learning. Initial random differences and learned displacements are distinct. Dense has no separately trained matched shuffled-code null in this design.
- IH accel-versus-gyro live-support intersection is empty and undefined. Full-matrix row energies describe allocated gates, including connections neutralized by true input masks; they are not estimates of available-signal flow. Live-support comparisons are the relevant contextual differentiation measurements.
- Fixed-representation interventions use the first128validation windows per fold. They establish sensitivity on that probe, not population-wide utility or realistic changes in sensor availability.
- Per-factor initial-to-final displacements, sampled gradient norms and aggregate active-row step-update fractions are saved. Per-factor stepwise stored-update fractions were not saved and must not be reconstructed from scalar norms.
- Learning curves and training gaps use pooled-window metrics within each split. The primary endpoint uses equal-subject, equal-context, equal-seed averaging; the two summaries answer different questions.
- Per-fit times include evaluation, file operations and shared CPU contention. They are not clean architecture-latency comparisons. Backend throughput is measured separately using disposable one-step fixtures.
- Full CUDA training and cross-backend macro-F1 reproduction are not run because CPU was selected for the complete paired grid. CUDA forward correctness and one-step throughput were measured separately.
- The historical16-subject scaler is retained for comparability. Some later cross-validation validation subjects contributed to this historical preprocessing; the study does not establish clean independently tested generalization. Official HAR test subjects remain embargoed.
- Five engineering seeds and overlapping CV training sets do not establish adequate power or independent confirmation. Null predictive effects do not prove H4 or H5, and functioning gates do not demonstrate associative memory.

Scalar-conversion warnings occurred while reading diagnostic tensors into Python numbers. These expressions do not feed subsequent losses or optimizer updates; the warnings are preserved rather than hidden. The frozen training source was not changed to silence them.

Pilot gate-change maxima are maxima over saved diagnostic checkpoints (every160updates, plus release checks and final), not a reconstruction of every transient optimizer-step gate value. The C3 allocation decision uses those recorded pilot measurements. The five correction cells do not identify all interactions among symmetry, saturation and precision.
