# Experiment 01 — Synthetic AND pilot

## 1. Identifier and title

Experiment 01. Smoke run `20261017T235744Z_smoke` and five-seed pilot `20261017T235839Z_pilot`.

Documented name in the build brief: CoNN Experiment 01.

## 2. Research question

Documented in `documentation/CURSOR_BUILD_BRIEF.md`: does structured local-to-global Gaussian weight gating improve learning relative to controls that preserve gate magnitude or per-row energy and disrupt spatial alignment?

## 3. Hypothesis

Documented primary prediction: final held-out test accuracy of condition D minus condition B prime on the AND task. Five seeds are marked exploratory in the brief.

## 4. Mathematical model

Documented and implemented in the copied `source/src/conn_exp` modules: coordinates on `[0, 1]`, distance the absolute difference, `G = exp(-d^2 / (2 tau^2))`, `W_eff = W * G` with `G` detached. `tau` comes from `d50 / sqrt(2 ln 2)`. This description is taken from those modules and the design note. It was not re-derived during organization.

## 5. Dataset and preprocessing

Documented in `source/configs/pilot.yaml` and `source/src/conn_exp/data.py`. Synthetic 100-feature motifs, splits 512/128/128, noise 0.15, distractor amplitude 0.5. Data are generated at runtime. They are not stored as a canonical raw file for the pilot. Generated arrays inside the run directory are referenced, not copied.

## 6. Conditions and controls

Documented identifiers: A dense, A prime scalar energy match, B prime within-row column permutation, C permuted input coordinates, D task-aligned schedule, E reverse path, S static local.

## 7. Training configuration

Documented in the pilot config: two hidden layers of width 32, plain SGD, momentum 0, weight decay 0, learning rate 0.1, 400 steps, release fraction 0.7. The smoke override is in the same YAML under `smoke`.

## 8. Original entry point

From `conn_experiment_01`:

`python -m conn_exp.runner --config configs/pilot.yaml --smoke`

`python -m conn_exp.runner --config configs/pilot.yaml --seeds 5`

Run those commands in the original package, not in this organized directory.

## 9. Random seeds

Documented: smoke seed 0. Pilot seeds 0, 1, 2, 3, 4. Stream offsets are in `DESIGN_DECISIONS.md`.

## 10. Available results

Copied summaries: `results/smoke` and `results/pilot`, including `scores.csv` and `report.md`. Full runs, checkpoints, and plots remain at the paths in `FULL_RUN_REFERENCE.txt`.

## 11. Known limitations

Documented by the original results note: the pilot is exploratory, the shared learning rate was not grid-selected, and D minus B prime does not isolate geometry by itself.

Organization limit: this source copy is the 1B-era snapshot with `experiment1b.py`, `experiment1b_report.py`, and `test_experiment1b.py` removed. Removal was an organizational inference. No snapshot labeled as the pre-1B tree was found.

## 12. Shared components

Uses its own copied `conn_exp` modules. Later experiments import similar filenames. Do not replace these files with a later snapshot. See `../SHARED_COMPONENTS/documentation/VERSION_MAP.md`.

## 13. Original locations

- Code snapshot: `conn_experiment_01/results/runs/20261018T001947Z_experiment1b_v1/source`
- Runs: `conn_experiment_01/results/runs/20261017T235744Z_smoke` and `20261017T235839Z_pilot`
- Brief: `CoNN_Cursor_Experiment_01`

## 14. Reproduction instructions

Use the original `conn_experiment_01` environment and the commands in section 8. This organized copy is an archive. Imports and paths were not rewritten.

## 15. Verification status

Organizational status: PARTIAL. Checksums of copied files are in `../FILE_PROVENANCE.csv`. Runtime reproduction was not executed. The exact bytes that produced the pilot were not isolated from the later snapshot.
