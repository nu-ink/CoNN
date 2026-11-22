# Reproducibility

Start with [study evidence](EXPERIMENTS.md) and select a specific protocol/run. Current source, copied packets, saved test passes and independent reproduction are different evidence categories. Consolidation ran no research tests, statistical recalculations, training, evaluation or checkpoint loading.

## Inspecting an existing study

1. Locate the study in [experiments.yaml](../registry/experiments.yaml) and original run path in [runs.csv](../registry/runs.csv). Distinguish studies, failed attempts, extensions and repairs.
2. Read its frozen protocol/configuration, selection, completion/status and deviations. Identify the statistical unit, class universe, endpoint and test-access restrictions.
3. Compare raw SHA-256 values with source/data/artifact manifests using each manifest’s own path base. Matching hashes establish identity against recorded values, not authorship, freeze time or scientific correctness.
4. Identify executing source, shared imports, preprocessing, RNG streams, batches and runtime flags. A final snapshot does not identify every earlier fit.
5. Inspect raw predictions/endpoint tables and analysis code. Preserve pairing, aggregation, nonfinite policy, resampling unit, tie rule and multiplicity. Reading saved numbers is not recalculation.
6. Before any new computation or replay, establish separate outputs and explicit scope. Historical scripts can write into original trees; do not run report generators in place.

## Reconstruction limits

| Study | Available evidence | Remaining gap |
| --- | --- | --- |
| Pilot 01 | Configuration, reports and summaries | Executing source unresolved; organized source reconstructed from later 1B is not an authenticated pilot snapshot. |
| 1B / 1C | Source snapshots, frozen protocol/selection, seeds, scores, states, completion | Shared imports, exact environment and test timing need per-run reconstruction. New seeds are local replication, not external reproduction. |
| 02 | HAR archive, prepared arrays/scalers/layouts, split manifest and source | Vendor preprocessing cannot reconstruct original continuous recordings. Seeds do not create independent subject populations. |
| Phase 0/1 | Protocol, inspection, recovery completion and snapshot | Clean early snapshot boundary unresolved; completion records no new training. |
| Phase 2/3 | Six source stages, seals, data, predictions, environments and recovery records | Per-fit/stage/resume mapping incomplete; fixed-scaler overlap and selected-LR reuse limit confirmation. Official test remains embargoed. |
| Phase 3B | Source/config, diagnostics, checkpoints, environment and deviation extensions | Post-outcome definitions, null population, layer attribution and C3 futility scope unresolved. Saved CPU replay does not prove cross-device/external reproduction. |
| 04 | Local preregistration, source, configs, named seeds, data/batches, predictions, artifacts and tests | Within-DGP study; no universal replay environment. Failed first smoke and deliberately unevaluated pilot remain lineage. |

## Data and environments

HAR’s [prepared-file manifest](../conn_experiment_01/results/runs/20261019T004346Z_experiment2_har_v1/prepared_manifest.json) binds engineered/temporal arrays, coordinates and scalers. [Subject identities](../conn_experiment_01/results/runs/20261019T004346Z_experiment2_har_v1/subject_split.frozen.json) separate 16 fit, five validation and nine test subjects. Phase 2/3 and 3B reuse temporal data with development CV and the original scaler. Refitting per fold changes the study; it does not silently restore old behavior. Experiment 04 uses a separate synthetic generator.

Start from each run’s environment capture. [Phase 2/3’s CUDA freeze](../CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/cuda_requirements.freeze.txt) and CPU records describe distinct profiles. Package/device availability does not prove replay.

Experiment 04 records Python 3.10.1, Torch 2.5.1+cu121, NumPy 1.26.4, SciPy 1.15.2 and pytest 8.3.5 with CPU execution. Its [original reproduction instructions](research_validation/experiment_04/REPRODUCTION.md) retain exact commands and output behavior. Inspect before use: training/evaluation/report generation write files; original generators use exclusive creation. No packages or environments changed during consolidation.

## Verification before replacement or replay

Preserve the fourteen distinctions in [architecture](ARCHITECTURE.md). Required checks depend on the proposed change:

- **Mathematics/numerical reference:** asymmetric geometry; valid/invalid tau; detached/differentiable gradients; row/layer energy; release indices; symmetric/perturbed context factors. Use analytical identities where available and justify floating tolerances by operation, dtype, backend and independent reference.
- **Saved-statistic recalculation:** paired seed means/SD/t intervals for 1B/1C/02; selected-development subject bootstrap separately from seed inference for Phase 2/3 and 3B; exact 1024-assignment sign-flip and fixed six-test Holm family for 04. Check missing classes, pairs and nonfinite values explicitly.
- **Determinism/checkpoints:** compare initial-state, permutation, batch and RNG identities in a frozen environment. Validate schemas/hashes before trusted loading. Endpoint weights alone cannot resume: optimizer, RNG, iterator and completed-update state must be supported.
- **End-to-end/future regression:** establish source closure, environment, protocol, test access, resources and separate output location first. New executions remain separate records. Scientific equivalence requires justified successful checks, not matching function names or API versions.

These are requirements, not newly executed tests or approval of a next phase. Old numerical thresholds are scoped historical definitions, not universal equivalence tolerances.

## Known integrity and attribution exceptions

**Scientific-review checksum (D-M0-01).** The root [Phase 3B review](../CoNN_Experiment_3B_Independent_Scientific_Review.md) SHA-256 is:

    c1bf15886c0b83615084e9d895979c3dd311b9dc177e329a8b2e9f5b64aab03c

The [provenance table](../ORGANIZED_EXPERIMENTS/FILE_PROVENANCE.csv) records the following, still matched by the [organized copy](../ORGANIZED_EXPERIMENTS/EXPERIMENT_03B/documentation/CoNN_Experiment_3B_Independent_Scientific_Review.md):

    1988440eedf0f5345601e8c4cba521616e304464c86025dca1a144491661ba7e

Root lines 3–5 each lack two trailing spaces retained by the copy. This is not CRLF conversion and can change Markdown hard breaks. The discrepancy predates cleanup; cause/time remain unknown. Neither file nor manifest was repaired or normalized.

**Bootstrap attribution.** The [audit summary](../AUDIT_20261020/FINAL_AUDIT_SUMMARY.md) attributes an exact Phase 3B redraw to seed 7000031; its saved [script](../AUDIT_20261020/Supporting_Evidence/independent_recalculation.py) calls seed 230319 and its [JSON](../AUDIT_20261020/Supporting_Evidence/recalculated_statistics.json) stores different redraw endpoints from the published interval. Another unsaved redraw may exist. Attribution remains unresolved; no bootstrap was rerun or scientific finding declared contradicted during cleanup.

## Records for new research

Record experiment/protocol identity and frozen status; source revision/hashes; behavioral variants; dataset/split/preprocessing identity; configuration and RNG streams; dependencies/device/precision; endpoint/evaluation policy; checkpoint capabilities; artifact hashes; execution and verification status. Leave historical missing fields unknown.

Failed/interrupted attempts remain evidence. Resume requires compatible complete state and a child attempt with parent/cursor. Repairs need explicit deviations and before/after identities; never overwrite failures or silently label a repaired run independently verified. The older [provenance contract](architecture/phase4/REPRODUCIBILITY_CONTRACT.md) remains a **Proposed** design, not a retroactive promise.
