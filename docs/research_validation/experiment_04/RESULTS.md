# Experiment 04 results

Status: **COMPLETED**. [Confirmatory run](../../../experiments/experiment_04/results/run_20261121T031743_424624Z_7ac51c80/), [analysis JSON](../../../experiments/experiment_04/results/run_20261121T031743_424624Z_7ac51c80/analysis.json), [raw endpoint table](../../../experiments/experiment_04/results/run_20261121T031743_424624Z_7ac51c80/endpoint_metrics.csv).

Ten independent paired confirmatory seeds completed all 80 fits and 240 predefined
test-endpoint evaluations. The positive primary difference is statistically supported
under the preregistered test assumptions, but the 2 pp practical-support criterion is unmet.

| Task | Condition | Accuracy mean ± SD (%) | Binary F1 mean ± SD | BCE mean ± SD |
| --- | --- | --- | --- | --- |
| structured | mlp | 76.362 ± 1.569 | 0.7605 ± 0.0197 | 1.2390 ± 0.0976 |
| structured | aligned | 77.539 ± 1.251 | 0.7723 ± 0.0144 | 1.1926 ± 0.0957 |
| structured | scrambled | 76.406 ± 1.654 | 0.7619 ± 0.0185 | 1.2524 ± 0.1209 |
| structured | uniform | 76.694 ± 1.647 | 0.7643 ± 0.0207 | 1.2812 ± 0.1146 |
| negative | mlp | 76.479 ± 1.504 | 0.7623 ± 0.0168 | 1.2533 ± 0.1008 |
| negative | aligned | 76.108 ± 1.003 | 0.7588 ± 0.0126 | 1.2644 ± 0.1015 |
| negative | scrambled | 76.133 ± 1.859 | 0.7591 ± 0.0219 | 1.2331 ± 0.1087 |
| negative | uniform | 76.440 ± 1.384 | 0.7624 ± 0.0171 | 1.2927 ± 0.1058 |


![Paired test results and validation convergence](../../../experiments/experiment_04/results/run_20261121T031743_424624Z_7ac51c80/experiment04_summary.png)

## Answers to the six research questions

1. **Aligned versus scrambled:** on the specified structured task, 77.539% versus
   76.406%, mean paired +1.133 pp, 95% CI [0.219, 2.046], one-sided p=.01465.
   Seven of ten seed differences are positive; three are negative. The practical threshold is unmet.
2. **Independent generalization:** yes, this measured difference is on 2,048 new examples
   per seed generated only after fitting all models. It establishes within-DGP generalization,
   not transfer to other tasks or distributions.
3. **Alignment versus magnitude:** the primary row-energy-matched comparison is positive.
   Aligned minus uniform is +0.845 pp, but Holm p=.1172. The structured-minus-negative
   effect is +1.157 pp, Holm p=.04883. These patterns suggest task alignment matters,
   while the uniform control prevents a decisive claim that magnitude is fully ruled out.
4. **After release:** the final primary advantage exists after 480 updates using exact ones.
   Immediate release at fixed weights gives +0.537 pp, CI [-0.178,1.252], Holm p=.25.
   Pre-release gives +1.396 pp, Holm p=.08594. Immediate persistence is uncertain;
   final persistence is observed under the primary protocol. No further tests were added.
5. **Overhead:** aligned mean paired optimization ratio is 1.091 on structured data and
   1.051 on negative data, below the preregistered 1.5 limit. This measures a matched dense
   gating path, not an optimized nn.Linear benchmark, and does not establish deployment cost.
6. **Original hypothesis:** limited task-specific statistical support for fixed, known
   relationship alignment. Practical importance and the uniform contrast remain unresolved.
   This does not justify moving directly to learned associations.

## Dynamics and diagnostics

| Task | Condition | Final train accuracy (%) | Final validation accuracy (%) | Train BCE | Validation BCE |
| --- | --- | --- | --- | --- | --- |
| structured | mlp | 100.00 | 76.68 | 0.0014 | 1.3022 |
| structured | aligned | 100.00 | 75.90 | 0.0018 | 1.2782 |
| structured | scrambled | 100.00 | 76.09 | 0.0020 | 1.3532 |
| structured | uniform | 100.00 | 76.99 | 0.0015 | 1.3493 |
| negative | mlp | 100.00 | 76.91 | 0.0015 | 1.2868 |
| negative | aligned | 100.00 | 76.21 | 0.0020 | 1.3080 |
| negative | scrambled | 100.00 | 75.90 | 0.0020 | 1.2841 |
| negative | uniform | 100.00 | 76.37 | 0.0015 | 1.3794 |


Training fits the small noisy training set very strongly while validation/test BCE is
much larger. This is consistent with overfitting and overconfident errors under the fixed
budget. It is an observation, not a proven cause of any between-condition difference.
No early stopping was retroactively applied. Initial train/validation metrics and complete
curves are retained for every condition; effective initial scaling differs by design.
Final weight-versus-assigned-distance bins are saved in each fit's `weight_distance.json`.
At the final endpoint all gates are one, so effective and raw final weights coincide.
Those bins describe learned weight patterns; they cannot by themselves prove a geometric mechanism.

The negative-control final aligned-minus-scrambled effect is -0.024 pp, CI [-0.808,0.759],
Holm p=.9570. This null result is reported with the same replication count and methods.

## Measured cost

| Task | Condition | Optimization seconds/fit | Mean paired ratio to mlp | Max sampled process RSS (MiB) |
| --- | --- | --- | --- | --- |
| structured | mlp | 1.369 | 1.000 | 434.0 |
| structured | aligned | 1.491 | 1.091 | 434.0 |
| structured | scrambled | 1.643 | 1.201 | 434.0 |
| structured | uniform | 1.620 | 1.182 | 434.0 |
| negative | mlp | 1.370 | 1.000 | 434.0 |
| negative | aligned | 1.436 | 1.051 | 434.0 |
| negative | scrambled | 1.586 | 1.161 | 434.0 |
| negative | uniform | 1.583 | 1.158 | 434.0 |


Total confirmatory training wall time was 141.79 s including per-seed audits and I/O.
Optimization timings include gate generation, finite checks, forward/backward and updates;
they exclude evaluation/checkpoint logging. Condition order rotates by seed. One process
serves all conditions; RSS is sampled at logging points and includes shared interpreter,
allocator and dataset state. It is not a per-model peak allocation estimate. Timing is
one local execution without independent hardware benchmarking or repeated timing trials.
