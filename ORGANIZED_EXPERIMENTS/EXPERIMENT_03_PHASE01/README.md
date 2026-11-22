# Associative Phase 0 and Phase 1

## 1. Identifier and title

Not named Experiment 3A in the source. Directory name `EXPERIMENT_03_PHASE01` means the associative protocol's Phase 0 and Phase 1.

## 2. Research question

Documented in `documentation/PROTOCOL_v2.md`: whether Gaussian-gated connectivity improves representation relationships, context sensitivity, and associative retrieval beyond conventional networks, kNN, and memory-augmented controls.

## 3. Hypothesis

The protocol separates classification from retrieval. Phases 2 through 6 are specified and, in that frozen file, described as not yet implemented. Later phases were authorized by a separate amendment and are stored under `EXPERIMENT_03_PHASE23` and `EXPERIMENT_03B`.

## 4. Mathematical model

The protocol records `G_ij(tau) = exp(-d_ij^2 / (2 tau^2))` and `W_eff = W odot G`, with the Experiment 2 detach behavior. Copied modules `gate.py` and `geometry.py` are the Phase 0/1 implementation of that statement, taken from the earliest snapshot.

## 5. Dataset and preprocessing

Read-only use of Experiment 2's prepared temporal arrays and the UCI archive. `documentation/normalization_history.md` records vendor scaling that cannot be undone. No new dataset was generated.

## 6. Conditions and controls

No training conditions were executed in this phase. The protocol names later controls. They were not run here.

## 7. Training configuration

None. The protocol forbids training until later gates are accepted.

## 8. Original entry point

From `CoNN_Associative_Relationships`, with `PYTHONPATH=src` and the Experiment 2 interpreter, as written in the original README:

`python -m conn_assoc.inspect`

`python -m pytest`

## 9. Random seeds

No training seeds. Inspection uses the frozen Experiment 2 subject partition.

## 10. Available results

Copied: protocol, manifest, inspection JSON, and `EXPERIMENT_AUDIT_AND_RESULTS.md`. No checkpoints belong to this phase.

## 11. Known limitations

The earliest code snapshot already contains phase-2 modules. Those filenames were not copied into this directory. That filter is an organizational decision documented in the organization report. A pure pre-phase-2 code snapshot was not found.

The current associative README still says later phases are absent. That sentence is historically the Phase 0/1 status. It is not a description of the repository after Phase 2/3.

## 12. Shared components

`paths.py` points at `../conn_experiment_01` and the Experiment 2 run. Those paths only work from the original associative directory.

## 13. Original locations

Prior snapshot: `CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/prior_snapshot`

Earliest code: `.../source/phase2_pretraining`

## 14. Reproduction instructions

Run `conn_assoc.inspect` from the original associative directory. Do not expect this filtered copy to be a complete package.

## 15. Verification status

Organizational status: AMBIGUOUS, because the code boundary was inferred by filename. Copied checksums match. Inspection was not re-run.
