# Experiment 1C phase 1: audit before new training

Historical 1B D − B′ replication remains +0.04843750, independently measured on seeds 100–119. No original artifacts or scores are overwritten, pooled, or reclassified.

Read: frozen protocol and selection, diagnostic and checkpoint-ablation summaries, raw curves and saved checkpoints. Existing measurements are labeled separately from newly performed inference on existing weights. No new model was trained in this audit. New gate interventions use training and validation only; no new test evaluation is performed.

## Existing validation checkpoints (20 seeds)

| Condition | Snapshot | Accuracy | Loss |
|---|---|---|---|
| D | pre_release | 0.916016 | 0.229767 |
| D | post_release | 0.902344 | 0.277548 |
| D | final | 0.909375 | 0.270180 |
| S | pre_release | 0.872266 | 0.282033 |
| S | post_release | 0.872266 | 0.282033 |
| S | final | 0.944141 | 0.152799 |
| B_prime | pre_release | 0.866797 | 0.419534 |
| B_prime | post_release | 0.859375 | 0.488472 |
| B_prime | final | 0.866016 | 0.474104 |

## Gate switch versus subsequent optimization

- D, gate_switch_only: mean validation accuracy change -0.01367188; mean loss change +0.04778136.
- D, further_optimization_960_steps: mean validation accuracy change +0.00703125; mean loss change -0.00736887.
- S, gate_switch_only: mean validation accuracy change +0.00000000; mean loss change +0.00000000.
- S, further_optimization_960_steps: mean validation accuracy change +0.07187500; mean loss change -0.12923375.
- B_prime, gate_switch_only: mean validation accuracy change -0.00742187; mean loss change +0.06893817.
- B_prime, further_optimization_960_steps: mean validation accuracy change +0.00664062; mean loss change -0.01436816.

The pre/post release snapshots contain bitwise-identical weights at 2240 updates. Their difference is a gate intervention. Post-release to final spans 960 actual optimization updates under a fixed terminal gate. S has no release gate change; its boundary difference is zero. These quantities must not be merged into one “collapse effect.”

## Native versus dense gates on the same weights

| Condition | Snapshot | Native val accuracy | Dense val accuracy | Native val loss | Dense val loss |
|---|---|---|---|---|---|
| D | pre_release | 0.916016 | 0.902344 | 0.229767 | 0.277548 |
| D | post_release | 0.902344 | 0.902344 | 0.277548 | 0.277548 |
| D | final | 0.909375 | 0.909375 | 0.270180 | 0.270180 |
| S | pre_release | 0.872266 | 0.756641 | 0.282033 | 0.892402 |
| S | post_release | 0.872266 | 0.756641 | 0.282033 | 0.892402 |
| S | final | 0.944141 | 0.770703 | 0.152799 | 1.398203 |

## Diagnostic context and missing measurements

The saved 1B final joint far-ablation validation loss deltas were D +0.07288950, B′ +0.88018548, S 0.0. S already has near-zero gates at far base distances, so zero additional effect does not mean that widening is intrinsically harmful. Effective weight magnitude differs substantially under S’s native gates. The copied diagnostic tables contain all layer norms and activation variances.
Missing in 1B: held-out test metrics before release, test scores under counterfactual dense gates, pre-2240 saved weights beyond the three retained snapshots, and a branch trained after release while retaining the wide Gaussian instead of switching to exact dense. These are not reconstructed. Existing curve points are measurements, not interpolated checkpoint states. Counterfactual dense inference is newly measured and is not a missing historical measurement.
Static-local superiority is observational across training procedures. Same-weight gate interventions assess immediate sensitivity, not the outcome of training with that gate. Whether persistent locality, gradual widening, exact release, or subsequent adaptation explains the difference requires the controlled follow-up.

All phase-1 analyses are exploratory because the 1B outcomes motivated them. The 1B primary result is preserved as historical independent evidence, not reused as independent 1C replication.
