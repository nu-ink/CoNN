# Implemented architecture

CoNN investigates whether associative relationships can guide computation and learning. Existing experiments study fixed relationship priors through gated multilayer perceptrons, followed by externally conditioned modulation. Learned associative retrieval, memory banks, dynamic neuron growth and a unified future runner are not established implemented features.

## Source and execution boundaries

| Existing implementation | Responsibility | Dependencies |
| --- | --- | --- |
| [conn_exp](../conn_experiment_01/src/conn_exp) | Synthetic AND/XOR studies 01, 1B, 1C; engineered and temporal HAR study 02 | Separate entry points share gates, schedules, models and training utilities. Current source includes later work; historical attribution requires run-specific snapshots. |
| [conn_assoc](../CoNN_Associative_Relationships/src/conn_assoc) | Phase 0/1 inspection and Phase 2/3 HAR development CV; external availability context | Reuses conn_exp and Experiment 02’s prepared temporal arrays/scaler. Historical path helpers and sys.path changes couple the trees. |
| [Phase 3B modules](../CoNN_Associative_Relationships/phase3b_context_diagnostics/src) | Context initialization, symmetry, precision, diagnostics and execution | Flat modules reuse earlier code and HAR development data. Recovery and execution extensions remain separate evidence. |
| [conn_exp04](../experiments/experiment_04/src/conn_exp04) | Experiment 04 synthetic geometry generation, training, evaluation and analysis | Reuses selected conn_exp geometry/gates/models/schedules; does not use HAR. Own configurations and run snapshots define the study. |

The [dependency registry](../registry/dependencies.yaml) records 24 static edges and six dataset relationships, not a complete executed import graph. Detailed listings remain in the [source inventory](architecture/PHASE3_SOURCE_INVENTORY.md) and [dependency inventory](architecture/PHASE3_DEPENDENCY_MAP.md).

Runners generally freeze protocol/configuration and selection, generate or prepare data, train paired conditions, save states/predictions, and compute study-specific analyses. Some studies deliberately omit final-test evaluation. Recorded completion does not imply a successful hypothesis.

## Mathematical definitions

For fixed source coordinate s_i and destination coordinate d_j on the unit interval:

    D_ji = abs(d_j - s_i)
    G_ji = clamp(exp(-D_ji² / (2 tau_used²)), 0, 1)
    W_eff = W ⊙ G
    d50 = tau * sqrt(2 ln 2)

Coordinates are prescribed, not learned. Uniform singleton coordinates are 0.5. Temporal HAR assigns the same time coordinate t/127 to nine channels per time step; an equal-sized uniform index layout is a different prior.

See [geometry](../conn_experiment_01/src/conn_exp/geometry.py), [gates](../conn_experiment_01/src/conn_exp/gates.py) and [associative gate functions](../CoNN_Associative_Relationships/src/conn_assoc/gate.py). conn_exp floors tau below tau_floor; conn_assoc first requires strictly positive finite tau and floor. Their invalid-input contracts differ.

Detached training application uses W * stopgrad(G) with the required cast, giving dL/dW = G ⊙ dL/dW_eff. The generic effective_weight product retains gate gradients. Equal forward formulas do not establish equivalent training. Binary BCE logits and six-class HAR cross-entropy heads also differ. See [models](../conn_experiment_01/src/conn_exp/models.py) and [contextual forward](../CoNN_Associative_Relationships/src/conn_assoc/phase3_context.py).

Controls preserve different properties:

- Frozen within-row permutations preserve each row’s gate multiset and mathematical row energy.
- A uniform gate sqrt(mean(G²)) matches whole-layer mean-square energy, not every row.
- Input-coordinate permutations change the input prior. Dense bypass and explicit ones multiplication are distinct paths.

Gate tensors remain dense. Locality or zeros alone do not establish sparse acceleration.

Scheduled conditions use smoothstep u²(3−2u). The first dense update index is floor(release_fraction*T + 0.5); pre-release interpolation uses denominator R−1. Historical 3200-update studies release at index 2240; Experiment 04 uses 1600 updates and index 1120. Fixed local, fixed medium, partial widening, held-wide Gaussian and exact dense release are distinct interventions. Completed updates, next gate index and same-weight release measurements must remain separate. See [schedules](../conn_experiment_01/src/conn_exp/schedules.py).

Phase 3 uses externally supplied sensor-availability context c:

    C(c) = 1 - (0.95 / 2) * sigmoid(U_c) @ sigmoid(V_c).T
    contextual hidden weight = W ⊙ G ⊙ C(c)

Rank two is allocated to factors; in real arithmetic rank(1−C) <= 2, not necessarily rank(C) <= 2. Original equal-column initialization U=−6, V=0 produces modulation near one. Phase 3B changes symmetry, center/jitter and factor storage precision. Float64 storage can still produce float32 forward factors. Rounding-induced numerical rank, parameter movement, functional sensitivity and predictive benefit are separate questions. See [Phase 3B models](../CoNN_Associative_Relationships/phase3b_context_diagnostics/src/models.py) and [rank diagnosis](../CoNN_Associative_Relationships/phase3b_context_diagnostics/reports/FORWARD_RANK_SENSITIVITY.md). This mechanism does not supply an internal associative memory.

Experiment 04 uses repeated noisy measurements of five independent latent pairs (a_g,b_g). Labels depend on the sign of sum_g(abs(a_g+b_g)−abs(a_g−b_g)), with 5% label flips; the negative task permutes coordinates independently of labels. Its exact DGP/network/control equations remain in the [mathematical specification](research_validation/experiment_04/MATHEMATICAL_SPECIFICATION.md).

## Behavioral distinctions

| Family | Preserve explicitly |
| --- | --- |
| C01 — geometry | Coordinate order, singleton policy, temporal colocation, destination-by-source axes. |
| C02 — bandwidth | Positive finite validation versus permissive flooring; underflow and invalid-input behavior. |
| C03 — application | Detachment, casting, shape checks, model head and dense path. |
| C04 — controls | Row/layer energy scope, permutation identity, draw order and reduction precision. |
| C05 — schedules | Release index, interpolation endpoint, held/exact-dense state and measurement cursor. |
| C06 — RNG/order | Seed recipes, fold/context streams, fixed cycling versus epoch permutations. |
| C07 — data | DGP, fit population, split identities, scaler reuse and test-access policy. |
| C08 — configuration | Schemas/defaults; file-byte SHA versus normalized-object digest. |
| C09 — execution | Loss, finite-check cadence, evaluation timing, failure/recovery and test-use rules. |
| C10 — inference | Binary/fixed-six/supported-class F1; seed/subject unit, pooling, bootstrap, sign-flip and multiplicity. |
| C11 — initialization | Generator/draw order, parameter groups and optimizer options. |
| C12 — context | C0–C4 variants, symmetry, storage/forward dtype and rank threshold. |
| C13 — runtime | Device, threads, determinism/TF32/MKLDNN and geometry-dependent cache assumptions. |
| C14 — checkpoints | Endpoint/full-resume state, schema/dtypes/shapes, RNG/cursor, loader mode and digest definition. |

The [source compatibility assessment](architecture/PHASE3_BEHAVIORAL_COMPATIBILITY.md) and [proposed version matrix](architecture/phase4/BEHAVIORAL_VERSION_MATRIX.md) retain detailed distinctions. Policy IDs do not certify equivalence. No behavioral reference suite was executed during consolidation.

## Continuing development

Keep future code separate from frozen research. Use explicit configuration/data lineage/behavioral selections and new output paths. Establish independently checked fixtures and justified numerical tolerances before replacing historical operations.

The [design proposals](architecture/phase4/ARCHITECTURE_DECISIONS.md) cover historical isolation, component behavior, adapters, data, provenance, checkpoints, environments and development practices. All eight ADRs remain **Proposed**. [M0](architecture/phase5/M0_IMPLEMENTATION_REPORT.md) completed a review/reporting package; acceptance remains blocked/pending. No future package, migration stage or verification phase is implemented or approved by this guide.
