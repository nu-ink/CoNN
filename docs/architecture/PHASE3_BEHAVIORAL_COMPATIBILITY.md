# Phase 3 mathematical and behavioral compatibility

Verified 2026-11-22 against local workspace E:/CoNN; repository remote nu-ink/frank_h_torralva-conn, branch initial-repository, HEAD d06568ac87bf370eba627858a12997d88cf30208.

Evidence labels: **V** = verified source/file fact; **S** = static-analysis inference, with limits; **P** = architectural proposal; **U** = unresolved. No project module was imported, no checkpoint loaded, and no training, evaluation, research test, package installation or runtime framework creation was performed. Byte identity and AST similarity are not behavioral-equivalence claims.

Reports: [inventory](PHASE3_SOURCE_INVENTORY.md) | [dependencies](PHASE3_DEPENDENCY_MAP.md) | [compatibility](PHASE3_BEHAVIORAL_COMPATIBILITY.md) | [proposal](PHASE3_PROPOSED_ARCHITECTURE.md) | [risks](PHASE3_RISK_ASSESSMENT.md). Navigation foundation: [registry guide](../../registry/README.md).

## Conclusion and compatibility standard

**S:** CoNN has reusable mathematical ingredients, but no evidence supports a single drop-in replacement for its experiment pipelines. Extraction should preserve named behavioral versions and experiment adapters. A matching name, formula, source hash or AST does not establish equal RNG draws, data order, dtype/device behavior, losses, endpoints or saved artifacts. No numerical fixture, autograd check, training or evaluation was executed in this phase.

**P:** A candidate becomes reusable only after its domain, shape/axis conventions, scalar/tensor types, output schema, configuration identity, seed mapping/draw order, arithmetic order, device, gradient policy and failure behavior are explicit. The contracts below cover all eight requested review dimensions. Validation suggestions are future approval gates, not completed tests.

## Mathematical invariants and their limits

**V:** [pairwise_abs_distance](../../conn_experiment_01/src/conn_exp/geometry.py#L26) defines D[j,i] = abs(q_dest[j] - q_src[i]); rows are destination units and columns are sources. [gaussian_gate](../../conn_experiment_01/src/conn_exp/gates.py#L20) implements G = clamp(exp(-D squared / (2 tau_used squared)), 0, 1), and [_point_from_tau](../../conn_experiment_01/src/conn_exp/schedules.py#L62) records d50 = tau_used * sqrt(2 ln 2). For finite positive tau, D=d50 gives exp(-ln 2)=1/2 by algebra. Exact dense release is a separate operation G=1, not the same as using finite d50=1 on the unit interval.

**S, symbolic deduction:** If G is independent of W, W_eff=W elementwise-multiplied by G gives dL/dW=G elementwise-multiplied by dL/dW_eff. Detaching G intentionally removes that branch of gradients to G; a generic product preserves it. If a context C is trainable, W_eff=W*G*C still allows gradients to C when G is detached. These are chain-rule statements, not freshly checked autograd results. Relevant source: [GatedLinear.forward](../../conn_experiment_01/src/conn_exp/models.py#L23), [effective_weight](../../CoNN_Associative_Relationships/src/conn_assoc/gate.py#L44), [training_effective_weight](../../CoNN_Associative_Relationships/src/conn_assoc/gate.py#L51).

**V/S:** [ContextFactor](../../CoNN_Associative_Relationships/src/conn_assoc/phase3_context.py#L25) defines C(c)=1-(0.95/2)*sigmoid(U_c)@sigmoid(V_c).T with rank fixed at two. The attenuation 1-C has algebraic rank at most two; that does not imply C itself has rank at most two. Equal initialized columns can reduce attenuation rank further. The interval bound and rank statement depend on this formula/parameterization; they do not demonstrate learned memory or retrieval.

## C01 — Coordinates and distances

**Evidence [V]:** [uniform_coordinates](../../conn_experiment_01/src/conn_exp/geometry.py#L18); [temporal_input_coordinates](../../CoNN_Associative_Relationships/src/conn_assoc/geometry.py#L26); [feature_layout](../../conn_experiment_01/src/conn_exp/har_data.py#L58).

- **Definition and assumptions:** Uniform coordinates use i/(n-1), with singleton 0.5; pairwise input vectors produce a destination-by-source matrix. Temporal HAR instead repeats t/127 for all nine channels at an instant. Engineered HAR uses metadata order, not measured physical distance.
- **Input/output contract:** Inputs are position count/dtype or coordinate vectors; distance output is a tensor [n_dest,n_src]. The temporal helper assumes at least two time samples; representation shape/order is part of the contract.
- **Configuration, RNG and errors:** Counts/layout come from configuration or frozen representation. No RNG enters geometry. uniform_coordinates explicitly rejects n<=0 and special-cases n=1; pairwise_abs_distance relies on tensor indexing/broadcasting rather than a complete rank/type check.
- **Precision, device and historical differences:** Torch float32 is default; feature-layout tables use NumPy values before downstream float32 conversion. Device movement is performed by callers. No learned coordinates exist in these helpers.
- **Consolidation judgment [S/P]:** The same uniform coordinate logic appears in both packages, but HAR temporal and synthetic feature grids are semantically different. **Risk:** replacing temporal repeated coordinates with a uniform 1152-point grid changes the treatment. Preserve a versioned layout ID and original feature/channel order.

## C02 — Gaussian bandwidth policy

**Evidence [V]:** [resolve_tau](../../conn_experiment_01/src/conn_exp/gates.py#L14); [resolve_tau](../../CoNN_Associative_Relationships/src/conn_assoc/gate.py#L19); [gaussian_gate](../../CoNN_Associative_Relationships/src/conn_assoc/gate.py#L33).

- **Definition and assumptions:** Both gate functions use exp(-d*d/(2*tau_used*tau_used)), followed by clamping, on their common finite-positive domain. Floors are numerical policy, not a learned bandwidth.
- **Input/output contract:** Input distance tensor and scalar tau/floor; output (gate tensor, floored bool). Shapes follow distance. No RNG. Output uses input tensor arithmetic dtype/device.
- **Configuration, RNG and errors:** conn_exp floors any tau<tau_floor with no explicit finite/positive validation; tau=0 or negative is floored when the floor is positive. conn_assoc first requires finite tau/floor, strictly positive floor and tau, then floors positive sub-floor values.
- **Precision, device and historical differences:** NaN/Inf and invalid floors therefore have different failure/propagation behavior. Exponential underflow to zero is retained, not evidence of discrete sparse execution.
- **Consolidation judgment [S/P]:** **Risk: high.** Do not replace one policy with the other for historical replay. Name permissive_legacy and finite_positive policies separately. Matching positive-domain formulas is not a bitwise replay test across devices/versions.

## C03 — Gate application, model heads and gradients

**Evidence [V]:** [GatedLinear.forward](../../conn_experiment_01/src/conn_exp/models.py#L23); [effective_weight](../../CoNN_Associative_Relationships/src/conn_assoc/gate.py#L44); [HARMLP.forward](../../conn_experiment_01/src/conn_exp/har_core.py#L33); [Model.run](../../CoNN_Associative_Relationships/phase3b_context_diagnostics/src/models.py#L47).

- **Definition and assumptions:** GatedLinear returns (linear output, W_eff), detaches G, casts it to W dtype/device and enforces exact weight shape. Mathematical effective_weight checks shape but does not detach/cast. Phase 3B Model.run directly multiplies G and C.
- **Input/output contract:** Binary CoNNMLP produces logits [N] and a ForwardCache for BCE-with-logits. HARMLP produces [N,6] by default for cross-entropy and can bypass gates when gate_ih is None. Context models add a different forward/run interface.
- **Configuration, RNG and errors:** Architecture/head dimensions and condition flags are part of checkpoint/config identity. Model initialization is separate from constructors. RNG must be supplied through the historical initialization stream.
- **Precision, device and historical differences:** Direct context paths do not uniformly apply the GatedLinear cast/shape guard. Current gates are generated constants; a future differentiable gate changes the significance of detach. A ones multiplication and the HAR dense bypass have different performance scope.
- **Consolidation judgment [S/P]:** **Risk: high.** Preserve detach, cast, shape, head, cache and baseline-path policies. An adapter can share the mathematical primitive; it must not silently change a study into a learnable-geometry experiment.

## C04 — Shuffle and energy controls

**Evidence [V]:** [apply_row_permutation](../../conn_experiment_01/src/conn_exp/gates.py#L31); [scalar_gate](../../conn_experiment_01/src/conn_exp/gates.py#L39); [draw_masks](../../conn_experiment_01/src/conn_exp/gates.py#L65); [Gates.audit](../../experiments/experiment_04/src/conn_exp04/core.py#L149).

- **Definition and assumptions:** Row shuffle permutes columns independently per destination row; it preserves each row multiset mathematically. Uniform control uses sqrt(mean(G squared)) for the whole layer, preserving layer mean square, not every individual row energy.
- **Input/output contract:** Permutation shape must equal gate shape; actual bijection/range checks are not performed by apply_row_permutation itself. draw_masks uses an explicit Torch generator and ordered input/row draws.
- **Configuration, RNG and errors:** Seed passes through paired_seed(mask), including when Experiment 04 first derives its named hash stream. Moving/reordering draws changes all later permutations despite the same base seed.
- **Precision, device and historical differences:** Float32 reduction sums can vary under a permutation even when row multisets match. Experiment 04 audit uses float64 row reductions with 1e-12 tolerance, while its uniform gate is still created in float32 with a 2e-7 energy tolerance.
- **Consolidation judgment [S/P]:** **Risk: high.** Preserve random draw order and reduction/tolerance policy. The float64 audit correction in [experiments/experiment_04/TECHNICAL_LOG.md](../../experiments/experiment_04/TECHNICAL_LOG.md) changed diagnostic behavior without changing the specified gate treatment; that distinction must survive consolidation.

## C05 — Schedules and measurement endpoints

**Evidence [V]:** [release_step_index](../../conn_experiment_01/src/conn_exp/schedules.py#L24); [_tau_between](../../conn_experiment_01/src/conn_exp/schedules.py#L31); [TerminalPolicy.at](../../conn_experiment_01/src/conn_exp/experiment1c_core.py#L36); [Gates.build](../../CoNN_Associative_Relationships/src/conn_assoc/phase23_core.py#L105).

- **Definition and assumptions:** release_step_index is floor(fraction*T+0.5). Smoothstep interpolation includes both pre-release endpoints using span-1; at the release index some policies switch to exact ones. TerminalPolicy can retain a fixed or terminal Gaussian without release.
- **Input/output contract:** A zero-based about-to-execute update index differs from completed optimizer updates. Return schemas differ: SchedulePoint, GatePack, (tau,exact), gate tuple or tensors.
- **Configuration, RNG and errors:** Original schedule validates ranges and phase lengths; Phase 2/3 Gates embeds release=2240 and denominator=2239 for a 3200-step protocol. Experiment 04 normally uses 1600 updates and release=1120 through the original schedule.
- **Precision, device and historical differences:** Phase 2/3 comments deliberately preserve tau interpolation arithmetic order. Some curve measurements use the next gate index after an update; Experiment 04 records the gate used at the last update, and uses pre_release weights plus ones for immediate_release.
- **Consolidation judgment [S/P]:** **Risk: high.** A generic schedule must not reinterpret event timing or fixed constants. Store (completed_updates, gate_step, endpoint, gate_mode) explicitly and retain protocol-versioned adapters.

## C06 — Seed streams and sample order

**Evidence [V]:** [paired_seed](../../conn_experiment_01/src/conn_exp/seeding.py#L18); [minibatch_order](../../conn_experiment_01/src/conn_exp/train.py#L34); [batch_indices](../../conn_experiment_01/src/conn_exp/train.py#L39); [paired_batches](../../conn_experiment_01/src/conn_exp/har_core.py#L57); [stream](../../experiments/experiment_04/src/conn_exp04/core.py#L25).

- **Definition and assumptions:** Original streams add offsets: data 0, init 1000003, order 2000003, mask 3000003. Experiment 04 first hashes exp04:seed:name with SHA-256 and takes four little-endian bytes; reused initializers/mask builders then add their own offsets.
- **Input/output contract:** Synthetic batching cycles one fixed permutation. HAR paired_batches concatenates independently redrawn full permutations from one NumPy generator, truncates to steps*batch and reshapes. Both return int64 indices but have different sequences.
- **Configuration, RNG and errors:** Phase 2/3 adds 10007*fold before HAR batching; external-context plans use another offset and three-code blocks. Phase 3B shuffles the context-code sequence separately from input masks.
- **Precision, device and historical differences:** Same integer seed is insufficient across algorithm, stream composition, draw dtype, RNG implementation and draw order. Original dataset train/val/test generation consumes one stream sequentially; Experiment 04 uses split-named streams.
- **Consolidation judgment [S/P]:** **Risk: high.** Preserve seed algorithm IDs, effective seeds, draw order and saved batches/masks. Do not standardize all studies to one new shuffle engine or claim pairing from seed labels alone.

## C07 — Data, preprocessing and access policy

**Evidence [V]:** [generate_dataset](../../conn_experiment_01/src/conn_exp/data.py#L151); [fit_scaler](../../conn_experiment_01/src/conn_exp/har_data.py#L32); [transform](../../conn_experiment_01/src/conn_exp/har_data.py#L46); [load_prepared](../../CoNN_Associative_Relationships/src/conn_assoc/data.py#L105); [stitch_official_train](../../CoNN_Associative_Relationships/src/conn_assoc/data.py#L132); [generate_split](../../experiments/experiment_04/src/conn_exp04/core.py#L102).

- **Definition and assumptions:** Synthetic motif AND/XOR and Experiment 04 latent-interaction data are different DGPs. HAR uses 561 engineered features or 128x9 time-major flattened windows; labels are six zero-based classes. Mean/std fits use float64 then contiguous float32 outputs; temporal scaling is per channel over windows/time.
- **Input/output contract:** Data contracts need x/y dtype and shape, row/subject identities, split roles, layout ID, scaler fit-subject IDs, source archive hash and transform version. A same-shaped array does not establish lineage.
- **Configuration, RNG and errors:** Original motif splits share sequential RNG consumption; Experiment 04 split streams and a fixed negative permutation pair variants. HAR subject partition is frozen, and associative stitch restores official training row order without claiming adjacent rows are continuous recordings.
- **Precision, device and historical differences:** Associative purpose checking rejects final_evaluation and restricts official-test feature access to audit purpose. This gate must not be dropped in a common loader. Phase 2/3 reuses original fitted scaling over later CV folds, with acknowledged overlap.
- **Consolidation judgment [S/P]:** **Risk: high.** Preserve original prepared arrays/scalers and distinguish future fold-fitted preprocessing as a new scientific protocol. Inverse z-scoring recovers vendor values, not untouched raw sensor recordings.

## C08 — Configuration and serialization identity

**Evidence [V]:** [load_config](../../conn_experiment_01/src/conn_exp/config.py#L288); [_build](../../conn_experiment_01/src/conn_exp/config.py#L167); [load_config](../../conn_experiment_01/src/conn_exp/experiment2.py#L25); [Config.validate](../../experiments/experiment_04/src/conn_exp04/core.py#L62); [digest_json](../../experiments/experiment_04/src/conn_exp04/core.py#L21).

- **Definition and assumptions:** The original ExperimentConfig derives tau/release values, rejects unknown/missing YAML keys and freezes task/optimizer constraints. HAR uses a separate dict validator; Phase 2/3/3B use frozen JSON/YAML plus seals. Experiment 04 uses a dataclass with finite/type checks and tuple conversions.
- **Input/output contract:** Payload types, defaults, allowed overrides and derived fields differ. A common permissive schema would expand accepted inputs and can alter study definitions. Parser error types and missing-field handling are part of adapter behavior.
- **Configuration, RNG and errors:** Configuration does not itself seed every RNG; downstream seed mapping is separate. Dtype/device choices can be implicit in model/configuration code, rather than serialized in every configuration.
- **Precision, device and historical differences:** Byte hashes of pretty-printed files differ from canonical sorted compact JSON object digests. Experiment 04 digest_json rejects NaN; several historical serializers have other options. Paths and NumPy scalar types can affect serialization.
- **Consolidation judgment [S/P]:** **Risk: high.** Preserve original bytes and digest algorithms. New schema migrations must create derived versioned configs with parent hashes, never rewrite a frozen config to match a new serializer.

## C09 — Training, evaluation and failure boundaries

**Evidence [V]:** [HeldOutSplit](../../conn_experiment_01/src/conn_exp/train.py#L108); [train](../../conn_experiment_01/src/conn_exp/har_core.py#L138); [train_job](../../CoNN_Associative_Relationships/src/conn_assoc/phase23_core.py#L203); [train_job](../../CoNN_Associative_Relationships/phase3b_context_diagnostics/src/training.py#L87); [fit_condition](../../experiments/experiment_04/src/conn_exp04/runner.py#L81); [evaluate_run](../../experiments/experiment_04/src/conn_exp04/runner.py#L194).

- **Definition and assumptions:** Binary pipelines use BCE-with-logits; HAR/context pipelines use cross-entropy. Evaluators use logits>0 versus argmax and differ in loss batching/return schemas. Eval/train mode transitions and no_grad/inference_mode are not uniform.
- **Input/output contract:** Original HeldOutSplit counts test reads; HAR final_test is explicit; associative development excludes official-test model evaluation; Experiment 04 generates tests in a separate command after training and has three fixed endpoints.
- **Configuration, RNG and errors:** Batch order, fixed diagnostic batches, intervention evaluation and logging are part of the experiment definition. Common training must not introduce early stopping, checkpoint selection, extra test reads or a different update budget.
- **Precision, device and historical differences:** Phase 2/3/3B check gradient finiteness on sampled logging steps; Experiment 04 checks every update. Phase 3B has a wall deadline/resume path; Experiment 04 preserves a failed/interrupted state and does not silently resume.
- **Consolidation judgment [S/P]:** **Risk: high.** Use experiment-specific adapters over explicit primitives. A unified loop is a future scientific/engineering change until pairing, endpoints, failure semantics and replay are validated.

## C10 — Classification metrics and statistical estimands

**Evidence [V]:** [classification_metrics](../../conn_experiment_01/src/conn_exp/har_core.py#L100); [metrics](../../CoNN_Associative_Relationships/src/conn_assoc/phase23_core.py#L160); [metric](../../CoNN_Associative_Relationships/phase3b_context_diagnostics/src/evaluation.py#L6); [independent_scores](../../CoNN_Associative_Relationships/phase3b_context_diagnostics/src/verify.py#L6); [paired_summary](../../conn_experiment_01/src/conn_exp/evaluation.py#L68); [paired](../../experiments/experiment_04/src/conn_exp04/analysis.py#L13).

- **Definition and assumptions:** Binary F1 and six-class macro-F1 are different estimands. Six-class macro includes absent classes as zero; Phase 3B also exposes supported_macro_f1 over present truth classes. Subject-averaged F1 differs from pooled-window F1 because F1 is nonlinear.
- **Input/output contract:** Historical paired_summary filters nonfinite deltas, allows n<2 summaries and uses a stored t-critical table. Experiment 04 paired rejects fewer than two/nonfinite values, enumerates sign flips for n<=20, and computes a SciPy t interval.
- **Configuration, RNG and errors:** Phase 3B bootstrap resets RNG seed 7000031 for 10000 subject resamples per call. Phase 2/3 analysis reuses a generator across contrasts; changing contrast order can change later draws. Experiment 04 has one primary exact test and six Holm-adjusted secondary p-values.
- **Precision, device and historical differences:** Uncertainty units differ: fixed-partition initialization/order seeds, subject means with reused development folds, and independent DGP/training seeds. Reported descriptive post-selection intervals must not be relabeled confirmatory.
- **Consolidation judgment [S/P]:** **Risk: high.** Reuse a metric only with explicit class universe, zero-division, weighting, nonfinite, inference-unit, multiplicity and RNG policies. Never silently swap a subject bootstrap for a seed test.

## C11 — Initializers and optimizer semantics

**Evidence [V]:** [initialize_model](../../conn_experiment_01/src/conn_exp/models.py#L74); [initial_state_dict](../../conn_experiment_01/src/conn_exp/models.py#L83); [build_optimizer](../../conn_experiment_01/src/conn_exp/models.py#L102); [Model.__init__](../../CoNN_Associative_Relationships/phase3b_context_diagnostics/src/models.py#L25).

- **Definition and assumptions:** Original initialization draws each weight and bias in a fixed order from a dedicated Torch generator, uniform with bound 1/sqrt(fan_in). Constructors alone allocate empty tensors; they are not initialized runnable models.
- **Input/output contract:** Binary and HAR head shapes differ. Context variants allocate additional factors and use their own initialization. load_state_dict compatibility requires architecture, dtype, condition and requires_grad policy, not merely matching base key names.
- **Configuration, RNG and errors:** build_optimizer introspects installed SGD options and explicitly disables supported foreach/fused/differentiable flags; Phase 2/3 and 3B construct their own SGD calls. Phase 3B filters frozen parameters, unlike blindly passing all model parameters.
- **Precision, device and historical differences:** Configuring momentum=0 does not erase every backend/option/version distinction. Explicit float64 context factors change parameter storage and gradient arithmetic even when the forward output is cast to float32.
- **Consolidation judgment [S/P]:** **Risk: high.** Preserve initializer algorithm/draw order, full optimizer option record and parameter grouping. Deterministic replay is environment-bounded and untested here.

## C12 — Context modulation and numeric precision

**Evidence [V]:** [mask_input](../../CoNN_Associative_Relationships/src/conn_assoc/phase3_context.py#L15); [ContextFactor](../../CoNN_Associative_Relationships/src/conn_assoc/phase3_context.py#L25); [Factor.forward](../../CoNN_Associative_Relationships/phase3b_context_diagnostics/src/models.py#L19); [Model.__init__](../../CoNN_Associative_Relationships/phase3b_context_diagnostics/src/models.py#L25).

- **Definition and assumptions:** mask_input expects flattened time-major [N,1152] and code in {0,1,2}; context zero returns the original input tensor, other codes allocate a masked result. Unavailable channels are zero in scaled space.
- **Input/output contract:** Historical factors use equal rank columns and saturated initialization. C0 retains the historical class. C1/C4 use canonical float64 jitter draws cast to storage type; C2/C4 alter initialization center; C3/C4 use float64 parameter storage and Factor casts forward C to float32.
- **Configuration, RNG and errors:** The random factor generator seed, layer/u/v draw order and code/mask relation are frozen. gaussian_shuffled, gaussian_constant and gaussian_frozen change code assignment or gradient eligibility, not the DGP.
- **Precision, device and historical differences:** Identical base weights do not imply identical effective initial weights. A dtype cast or symmetric reinitialization can remove the intended diagnostic contrast. mask_input(0) aliasing means a generic caller must not mutate its return in place.
- **Consolidation judgment [S/P]:** **Risk: high.** Keep named factor variants and cast points as versioned behavioral policies. Do not promote this to a generic learned-memory abstraction.

## C13 — Runtime/device and caching

**Evidence [V]:** [configure_runtime](../../conn_experiment_01/src/conn_exp/runtime.py#L8); [resolve_device](../../conn_experiment_01/src/conn_exp/runtime.py#L25); [configure](../../CoNN_Associative_Relationships/src/conn_assoc/phase23_core.py#L73); [Gates.build](../../CoNN_Associative_Relationships/src/conn_assoc/phase23_core.py#L105); [configure](../../experiments/experiment_04/src/conn_exp04/runner.py#L39).

- **Definition and assumptions:** Original runtime fixes one thread and cuDNN/TF32 flags; original device resolver accepts cpu/cuda and refuses silent fallback. Associative configure additionally requires deterministic algorithms and accepts indexed CUDA devices. Experiment 04 disables MKLDNN and sets configured threads.
- **Input/output contract:** Global Torch flags, import-time environment state, device ownership and warm-up/synchronization scope are process-level contracts. They are not pure math functions.
- **Configuration, RNG and errors:** Phase 2/3 CPU Gaussian cache is keyed only by tau under a fixed 64x64/1152 geometry. This is adequate only under its fixed-shape implementation assumptions; a generalized geometry/dtype cache must use a broader key.
- **Precision, device and historical differences:** Some diagnostics use float64 while forward tensors remain float32; timings include different diagnostic overhead and dense bypass/ones-multiplication paths. No memory/runtime improvement was benchmarked here.
- **Consolidation judgment [S/P]:** **Risk: medium/high.** Future runtime should record policies explicitly, avoid global mutations on core import and never reuse cache entries across different semantic tensor identities.

## C14 — Checkpoint, fingerprint and provenance formats

**Evidence [V]:** [_save_checkpoint](../../conn_experiment_01/src/conn_exp/train.py#L139); [rng_state](../../CoNN_Associative_Relationships/src/conn_assoc/phase23_core.py#L193); [train_job](../../CoNN_Associative_Relationships/src/conn_assoc/phase23_core.py#L203); [train_job](../../CoNN_Associative_Relationships/phase3b_context_diagnostics/src/training.py#L87); [save_checkpoint](../../experiments/experiment_04/src/conn_exp04/runner.py#L73); [fingerprint_state](../../conn_experiment_01/src/conn_exp/models.py#L93); [arrays_hash](../../experiments/experiment_04/src/conn_exp04/core.py#L30).

- **Definition and assumptions:** Original checkpoints store weights/optimizer/gates and measurement metadata; they do not claim complete RNG/iterator resume. Phase 2/3 adds next_step, Python/NumPy/Torch/CUDA RNG, batches, contexts and histories. Phase 3B stores plans/codes/diagnostics but no general RNG state; its fixed regenerated plans support a narrower replay assumption.
- **Input/output contract:** Experiment 04 stores Torch RNG and endpoint state, and has no resume command. Loading a checkpoint is not a generic restore operation: architecture, key schema, dtype, optimizer grouping, source version and policy must match.
- **Configuration, RNG and errors:** Historical loaders often use weights_only=False; Experiment 04 uses weights_only=True for its own newly generated checkpoints. A blanket loader switch can reject non-tensor historical state. No checkpoint was loaded for this review.
- **Precision, device and historical differences:** fingerprint_state hashes sorted keys and tensor bytes without explicit dtype/shape tags; arrays_hash prefixes shape/dtype. File-byte hashes, canonical config hashes and model fingerprints are distinct namespaces. Their equality/inequality cannot be interchanged.
- **Consolidation judgment [S/P]:** **Risk: high.** Version schemas and digest encodings; write separate readers/adapters for historical formats. Any future conversion produces a new artifact with parent hash and explicit equivalence limits, never replaces an original checkpoint.

## Historical implementation comparisons

**V:** The table compares stored source bytes and AST structure for Phase 2/3 stage files. AST digests include constants and docstrings and omit source positions; they are not semantic hashes. Identical ASTs still depend on imports, configuration and environment.

| Stage/file | SHA-256 | AST SHA-256 |
| --- | --- | --- |
| [CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/source/analysis_ready/src/conn_assoc/phase23_core.py](../../CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/source/analysis_ready/src/conn_assoc/phase23_core.py) | 4e5c05965ac8669b66f77a596b79fed659d197f32646de835bc5edfd3a5ad172 | 4a787c797f2844dd5a3c800c7926d064cc71c6bed3a068c2c4baa098ef603cc6 |
| [CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/source/delivery/src/conn_assoc/phase23_core.py](../../CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/source/delivery/src/conn_assoc/phase23_core.py) | 4e5c05965ac8669b66f77a596b79fed659d197f32646de835bc5edfd3a5ad172 | 4a787c797f2844dd5a3c800c7926d064cc71c6bed3a068c2c4baa098ef603cc6 |
| [CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/source/phase2_pretraining/src/conn_assoc/phase23_core.py](../../CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/source/phase2_pretraining/src/conn_assoc/phase23_core.py) | 89124eb6c92c994d0cff34317ad59d4252d7d36038b5d09a061bf1fe431c14de | c7c4c9aca1807099e24d51282fe0ff896c3cda54c274c126842438d637dcf665 |
| [CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/source/phase2_ready/src/conn_assoc/phase23_core.py](../../CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/source/phase2_ready/src/conn_assoc/phase23_core.py) | 497e0ba7becac3270e036202404ec8225776cdc161d5e0e1b59aeb5df84cd20b | a06ce67bbd9e9bd40d89ffb17a57ab1da8cc1ae5a23e2da3da40278d6c9707be |
| [CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/source/phase3_ready/src/conn_assoc/phase23_core.py](../../CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/source/phase3_ready/src/conn_assoc/phase23_core.py) | 55dc364413c3afb96608f8a049277a9265abf07114c779d3c8ef4c6edf02f5e2 | c23b41401dcec696c57656b8a61ec18c79d55d13f380bbef5e7629673cc6a6db |
| [CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/source/recovery_io_retry/src/conn_assoc/phase23_core.py](../../CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/source/recovery_io_retry/src/conn_assoc/phase23_core.py) | 4e5c05965ac8669b66f77a596b79fed659d197f32646de835bc5edfd3a5ad172 | 4a787c797f2844dd5a3c800c7926d064cc71c6bed3a068c2c4baa098ef603cc6 |
| [CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/source/analysis_ready/src/conn_assoc/phase23_analysis.py](../../CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/source/analysis_ready/src/conn_assoc/phase23_analysis.py) | 2b89f185d2a643959f1c97b193b181aae3d726f4895fe923668afcb49c91e224 | ea4f169dddb309d48ed9659c3bb579111ea24361583d8e5a6f37ad2224656a6a |
| [CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/source/delivery/src/conn_assoc/phase23_analysis.py](../../CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/source/delivery/src/conn_assoc/phase23_analysis.py) | 2b89f185d2a643959f1c97b193b181aae3d726f4895fe923668afcb49c91e224 | ea4f169dddb309d48ed9659c3bb579111ea24361583d8e5a6f37ad2224656a6a |
| [CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/source/phase3_ready/src/conn_assoc/phase23_analysis.py](../../CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/source/phase3_ready/src/conn_assoc/phase23_analysis.py) | f16b538c88255e8432d8ba83e6e98a9178ffe73b1ee39fc039fad7d82df1361b | 0d5cd02a9a2734181cedadd15f29c61f1fbd57b8219cdf67fe6738d74ba37d59 |
| [CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/source/recovery_io_retry/src/conn_assoc/phase23_analysis.py](../../CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/source/recovery_io_retry/src/conn_assoc/phase23_analysis.py) | 64a9fdaaefdea9c6929f538d6be6fc7042d7f3d8215b37f2e3d565790a128d22 | 72db16dfed9e9702955bb0912cddcd86772d2beb5f62d833cc99dade7412630b |

**V/S:** Core stage differences include deterministic CUDA environment setup between phase2_pretraining and phase2_ready, context-related model/diagnostic additions by phase3_ready, and atomic replacement retries in recovery_io_retry. The retry code changes IO failure handling while attempting to preserve training state; this is a narrower statement than certifying identical execution histories. Analysis stages also differ; later reports must cite the postprocessing implementation separately from the training implementation.

## Candidate disposition and future validation gates

**P:** Lowest-coupling candidates are coordinate/distance primitives, a named positive-domain Gaussian policy, explicit schedule evaluators, fixed stream derivation and metric kernels with declared class semantics. They can form a future core only after contracts are frozen. Model initialization, context factors, data transforms, sampling, checkpoint readers, statistical plans and training loops need study-specific adapters. Hash/manifest helpers can be shared only with explicit byte/canonical/tensor digest namespaces and exclusive-write policies.

For each approved extraction, future validation should compare invalid-input behavior, tensor shapes/dtypes/devices, boundary steps, RNG draws and saved sample order, forward and gradient results, checkpoint schema/roundtrip, and protocol-defined metrics/aggregation on fixed saved fixtures. Use exact equality only where the contract demands it; tolerances need numerical justification and must be recorded. Such checks may require imports, evaluation or optimizer steps and are **not authorized or performed in this assessment**. Historical reference bytes remain unchanged; incompatible changes receive new behavior IDs and new runs.

**U:** No cross-version/device equivalence, complete independent replay, per-fit source-stage assignment or general benefit from scheduled release has been established here. The strongest supported architectural conclusion is that reusable ingredients exist while scientific behavior remains versioned at the experiment boundary.
