# Phase 3B v3 final research report

Status: completed development grid. Verified 455 grid fits and 10 pilots; 1,488,000 completed-fit optimizer updates. Backend: cpu. Official-test model evaluations:0. Passing numerical tests do not establish successful training.

## Research question and design

Can repairing weak context-factor learning yield useful external-context behavior? Fixed LR.1, architecture1152–64–64–6,3200 SGD updates, seeds400–404, seven subject-group folds; 15 conditions and up to10 pilots. Read the frozen protocol/config/source and execution manifests for exact definitions. No validation search. C3 omissions are frozen after pilot numerical futility, never substituted.

Execution manifest: `{'backend': 'cpu', 'omitted_conditions': ['dense_C3', 'gaussian_C3'], 'omitted_fits': 70, 'omission_reason': 'predeclared C3 forward futility after3200 pilot steps', 'full_grid_fits': 455, 'pilot_fits': 10, 'source_manifest_sha256': '96eff6c7314032940eca43356ab4092f748ce220d1143e9fcefc47f2468900d7', 'protocol_sha256': '901e57c4d5dcc9bf63b05eb297c75948698db232eec15361248f8c791157e72c', 'maximum_wall_seconds': 21600}`. Tests: ...................                                                      [100%]
============================== warnings summary ===============================
phase3b_context_diagnostics/tests/test_phase3b.py::test_analytic_and_finite_difference
  E:\CoNN\CoNN_Associative_Relationships\phase3b_context_diagnostics\src\preflight.py:94: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
  Consider using tensor.detach() first. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\torch\csrc\autograd\generated\python_variable_methods.cpp:821.)
    return {'analytic_autograd_max_error':max(float((gu-au).abs().max()),float((gv-av).abs().max())),

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
19 passed, 1 warning in 4.61s.

## Historical comparison and precision

All1050 historical fits and5762 historical source/data checks passed at preflight. C0 checkpoint replays exactly matched logits and saved predictions in six architecture/context fixtures. The original Gaussian gate was bitwise equivalent on tested schedule positions. The earlier Phase23 verifier imported trainer metric routines; this new verifier does not. Earlier results remain historical, not new replications. Historical Gaussian context effect −.00041791, subject SD .00520815; the new historical bootstrap uses its own frozen random seed, so interval endpoints need not reproduce the earlier Monte Carlo realization. All21 subjects contain all6 classes, making supported-class and six-class endpoints identical. See HISTORICAL_PRECISION_ESTIMATE.md.

## Mathematical, symmetry and precision findings

Analytic loss-gradient contractions agree with autograd to1.39e-17 in the fixed float64 fixture; central differences passed. The42-fixture gradient-only sweep used no validation scores. U=-4.5 failed the1e-3 engineering ratio in6/6 fixtures and had zero maximum one-step float32 ΔC. U=-1 reached the ratio in5/6 fixtures, but did not replace the frozen initialization. C1 broke symmetry and produced numerical rank2; its sigma2/sigma1 remained below.01 at initialization, a limitation retained as required. See initial_rank.csv, gradient sweep report and measured trajectories.

Gaussian C4 Level1 passed in 35/35 completed grid fits. Level2 pair/layer criteria passed in 49/175 eligible comparisons against the fully available shuffled-code null; detailed results include undefined empty IH intersections. Initial random gate differences are reported separately from learned changes. Pair-level evidence does not imply all contexts learned distinct useful relationships.

## Classification and paired uncertainty

Primary GaussianC4−Gaussianmasked: +0.00213398 macro-F1; subject SD 0.00515916, seed SD 0.00292647; descriptive95% subject-bootstrapCI [+0.00011566, +0.00444627]; paired subject dz 0.41363. +.005 threshold reached: False. Level3 predictive threshold achieved: False. Contextual attribution also requires valid Level2; random initialization, fixed attenuation and capacity controls limit interpretation.

secondary: gaussian_C4−gaussian_constant +0.00332958, descriptive95%CI [+0.00019395,+0.00702091].

secondary: gaussian_C4−gaussian_shuffled +0.00274569, descriptive95%CI [-0.00060541,+0.00691192].

secondary: dense_C4−dense_masked -0.00102141, descriptive95%CI [-0.00289158,+0.00050925].

exploratory: gaussian_C4−gaussian_C0 +0.00255189, descriptive95%CI [-0.00044630,+0.00598871].

exploratory: gaussian_C4−gaussian_C1 +0.00163039, descriptive95%CI [-0.00126370,+0.00516054].

exploratory: gaussian_C4−gaussian_C2 +0.00449402, descriptive95%CI [+0.00183077,+0.00755585].

exploratory: gaussian_C4−gaussian_frozen +0.00266374, descriptive95%CI [+0.00018687,+0.00530247].

No formal hypothesis tests or uncorrected significance claims. Secondary family and exploratory contrasts remain separate. See individual seed and subject CSVs, supported-class scores, confusion verification and sensor-specific results. Intervals resample21subjects together across all5seeds after equal-context averaging. Overlapping CV training sets violate simple independent-fit interpretations; all outcomes are exploratory development evidence.

## Capacity, interventions and convergence

Frozen C4 retains nonidentity attenuation; it isolates initial weighting from learned gate changes. Constant-code preserves allocated capacity but trains fewer lookup rows. Allocation and trainable counts are separate. All five nonidentity code permutations retain true sensor masks. Random-code evaluation of the shuffled-training control uses independent balanced per-window codes and is a distribution shift; correctly matched evaluation differs from its training code distribution. Disabled-C, fixed-h1, same-masked-input, release pre/post and full learning curves are saved. Same-weight interventions are diagnostic, not realistic changed sensor availability. Training/validation losses and final gaps are available in raw curves and scores; gate-switch pre/post uses identical weights, separating gate release from further optimization.

## Scientific interpretation and limitations

H1 is supported mathematically: identical component columns constrain A to rank≤1; perturbations break this restriction but the weak second singular component may remain practically small. H2 is supported by the fixed-batch saturation sweep, without establishing a predictive remedy. H3 must be assessed at each level: gradient, stored update, reference C, float32 C and logits; float64 storage alone does not establish forward learning. H4 compensation and H5 context recoverability remain unresolved explanations, even if predictive effects are null. Positive fixed-code or frozen-gate effects would also limit information-specific claims.

The same development subjects, engineering seeds and historical scaler have been examined before. The16-subject scaler contributes to preprocessing of some later CV validation subjects; it was preserved for historical comparability and is a limitation. Only21 subject clusters constrain precision; five seeds do not guarantee power. No official test evaluation, independent confirmation, semantic relationship learning, new architecture or associative memory is demonstrated. Phase4 is not implemented; further associative-memory research requires a separate question and protocol, rather than being warranted solely by functioning gates.

## Execution, integrity and reproducibility

Backend benchmark: {'device': 'cpu', 'mean_updates_per_second': {'cpu': 329.5245396894343, 'cuda:0': 245.34734602693902}, 'conservative_estimated_total_seconds': 15586.092631645904, 'maximum_wall_seconds': 21600, 'benchmark_one_step_diagnostic_updates': 880, 'benchmark_classification_scores_examined': False}. Completed fits are independently rescored and checkpoint/file hashes verified. Actual checkpoint next_step, contiguous step logs, base-init/batch/mask pairing and subject disjointness are checked. Same-backend checkpoint/resume tests require exact tensors and predictions. Cross-backend tolerance checks are separately labeled. Fixed-work benchmark one-step fixtures reset parameters each time and are not full training trajectories; their actual optimizer calls are recorded separately.

Commands (from E:\CoNN, PYTHONDONTWRITEBYTECODE=1): use the isolated `.venv_cuda23\Scripts\python.exe` with `phase3b_context_diagnostics\scripts\run.py preflight`, `execute` (also resume), `verify`, `report`. Read README.md for exact full paths. Source snapshots, SHA manifests, frozen configs, environment, raw prediction NPZ, subject/class CSV, checkpoints, curves, all code permutations and plots are run-specific. Repairs and omissions are documented in REPAIR_AND_DEVIATION_LOG.md.
