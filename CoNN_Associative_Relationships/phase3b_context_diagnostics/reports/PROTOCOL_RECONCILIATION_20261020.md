# Complete v2 reference and v3 amendment reconciliation

The complete PROTOCOL_PHASE3B_v2_reference.md is now in the Phase 3B directory and has been read in full alongside the archived original v3 amendment. The root PROTOCOL_PHASE3B_v2.md remains its explicitly labeled historical synopsis. The reference file is historical specification material, not evidence of an executed/frozen v2 experiment.

The single consolidated review draft is [PROTOCOL_PHASE3B_v3.md](../protocol_reviews/20261020_reference_reconciliation/PROTOCOL_PHASE3B_v3.md). It contains the explicit fifteen-condition matrix, unchanged C0–C4 definitions and SGD settings, deterministic ten-fit pilot plan inherited from the existing implementation, numerical thresholds, complete training bounds, wall cap, statistical plan, verification gates, plots and reporting requirements. Existing implementation choices are distinguished from requirements explicitly present in v2/v3.

The already frozen root v3 protocol and completed run were preserved. The root protocol SHA256 remains 901e57c4d5dcc9bf63b05eb297c75948698db232eec15361248f8c791157e72c. The completed run already contains 455 grid fits plus ten pilots. This reconciliation occurred after those outcomes were available; it cannot preregister or retroactively change that run. The existing scientific reports and statistics were not rewritten.

Explicit v3 overrides resolve the fourteen-versus-fifteen condition count, fit budget, success criteria, exact-replay rule, context permutations and secondary comparison family. Three choices still prevent certification for a new freeze:

1. The null population: old protocol/config wording suggests seed/fold matching, but reporting pools all 35 Gaussian shuffled fits per layer/context pair. A matched seed/fold has only one shuffled fit and does not itself define a percentile distribution.
2. Layer attribution: the existing main IH intervention changes both contextual gates, and a fit-wide forward-learning criterion can qualify a layer comparison. Layer-specific evidence must be distinguished from a fit-wide effect. Existing supplemental isolated-layer measurements do not retrospectively fix the original criterion.
3. C3 pilot futility: specify final-only versus maximum recorded forward change and separate versus joint architecture decisions. The existing implementation used maxima over sampled checkpoints and separate architecture decisions; unrecorded trajectories are unknown.

The review records these ambiguities instead of inventing replacements. It has a content SHA256 and a machine-readable REVIEW_MANIFEST.json in the review directory, but is explicitly NOT an operative frozen training protocol. Hashing a draft does not resolve missing scientific definitions. Resolve A1–A3 in a subsequent versioned revision and verify protocol/config/source agreement before another performance experiment.

The verification script checks the complete matrix, budget arithmetic, explicit open items, root/run/source protocol hash agreement, the original seal/completion references and unchanged selected historical input hashes during verification. This is document-level verification, not a claim of a fresh full artifact audit or a semantic proof of consistency. No new fits, optimizer updates, official HAR test evaluation or Phase 4 work were performed in this review.

Reproduce the document checks with the existing isolated Python environment:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
& E:\CoNN\CoNN_Associative_Relationships\.venv_cuda23\Scripts\python.exe -X utf8 E:\CoNN\CoNN_Associative_Relationships\phase3b_context_diagnostics\protocol_reviews\20261020_reference_reconciliation\verify_review.py
```
