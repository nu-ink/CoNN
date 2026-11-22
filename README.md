# CoNN research project

CoNN investigates whether structured Gaussian gating of neural-network connectivity improves learning and generalization, when compared with controls for gate magnitude, alignment, fixed locality and dense connectivity. Later work investigates externally supplied context modulation and fixed relationship geometry. The broader research asks whether these mechanisms could support associative relationships and retrieval; classification results alone do not establish associative memory.

This is a research workspace. Existing implementations use fixed-width gated multilayer perceptrons. Scheduled bandwidth changes and exact dense release are implemented; learned coordinates, a Phase 4 associative memory bank and dynamic neuron creation/merging/retirement are not established by the inspected implementation. No implementation is designated authoritative for every historical run.

The registry was verified against local files on **2026-11-22**. Verification was static: source and recorded artifacts were read; no experiments or tests were run. Completion and scientific success are separate claims. Independent reproduction is **not established** by this registry.

## Start here

| Resource | Purpose |
| --- | --- |
| [Registry guide](registry/README.md) | Definitions, evidence rules, coverage and maintenance |
| [Experiment registry](registry/experiments.yaml) | Eight experiment families, purposes, implementation paths and evidence |
| [Run index](registry/runs.csv) | 14 primary study/audit runs and 18 execution/extension/failure records |
| [Dependency registry](registry/dependencies.yaml) | 24 verified static dependency edges and six dataset/provenance records |
| [Recorded Experiment 04 results](docs/research_validation/experiment_04/RESULTS.md) | Most recent recorded study results and their limitations |

## Research history and implementation status

These are scientific study boundaries, not a proposed filesystem migration. Status means what the cited local records support, without re-execution.

| Study | Purpose and recorded status | Existing implementation and evidence |
| --- | --- | --- |
| Experiment 01 | Synthetic AND gating smoke/pilot; output summaries exist. Exact original executing source and explicit run completion markers remain unresolved. | [Shared conn_exp source](conn_experiment_01/src/conn_exp/), [pilot summary](conn_experiment_01/results/runs/20261017T235839Z_pilot/summary.json) |
| Experiment 1B | Validation-only optimization followed by replication on seeds 100-119; completed as recorded. | [Runner](conn_experiment_01/src/conn_exp/experiment1b.py), [frozen protocol](conn_experiment_01/results/runs/20261018T001947Z_experiment1b_v1/protocol.frozen.json), [completion](conn_experiment_01/results/runs/20261018T001947Z_experiment1b_v1/completion.json) |
| Experiment 1C | Fixed locality, partial widening and dense release, including XOR/alignment controls; completed as recorded. | [Runner](conn_experiment_01/src/conn_exp/experiment1c.py), [report](conn_experiment_01/results/runs/20261018T024146Z_experiment1c_v1/research_report.md) |
| Experiment 02 | Engineered and temporal UCI HAR with subject-disjoint final evaluation; completed as recorded. | [Runner](conn_experiment_01/src/conn_exp/experiment2.py), [frozen protocol](conn_experiment_01/results/runs/20261019T004346Z_experiment2_har_v1/protocol.frozen.json), [prepared-data manifest](conn_experiment_01/results/runs/20261019T004346Z_experiment2_har_v1/prepared_manifest.json) |
| Associative Phase 0/1 | Protocol, data provenance and gate audit; final audit completed as recorded, with earlier attempts retained. | [Protocol v2](CoNN_Associative_Relationships/PROTOCOL_v2.md), [inspection](CoNN_Associative_Relationships/audit/inspection.json), [audit completion](CoNN_Associative_Relationships/results/runs/20261019T024656620209Z_audit_recovery_v1/completion.json) |
| Associative Phase 2/3 | Temporal HAR development and context grid; 1,050 fits recorded complete, with zero official-test model evaluations recorded. | [Phase README](CoNN_Associative_Relationships/README_PHASE23.md), [development amendment](CoNN_Associative_Relationships/PROTOCOL_PHASE23_v1.md), [completion](CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/completion.json) |
| Phase 3B | Context symmetry, initialization and precision diagnostics; 455 grid fits and 10 pilots complete, 70 grid fits omitted as recorded. | [Study README](CoNN_Associative_Relationships/phase3b_context_diagnostics/README.md), [completion](CoNN_Associative_Relationships/phase3b_context_diagnostics/results/20261020_phase3b_v3_run01/completion.json), [protocol reconciliation](CoNN_Associative_Relationships/phase3b_context_diagnostics/reports/PROTOCOL_RECONCILIATION_20261020.md) |
| Experiment 04 | Fixed latent-interaction geometry and negative-control study; successful smoke, deliberately unevaluated pilot and evaluated confirmatory run recorded. Failed first smoke retained. | [Study README](experiments/experiment_04/README.md), [local preregistration](experiments/experiment_04/PREREGISTRATION.md), [stage manifest](experiments/experiment_04/results/stage_manifest.json), [results](docs/research_validation/experiment_04/RESULTS.md) |

Experiments 01, 1B, 1C and 02 share the existing `conn_experiment_01` project. Phase 2/3 and Phase 3B reuse its code and prepared HAR data. Experiment 04 reuses selected original `conn_exp` functions while generating a separate synthetic dataset. These studies must not be merged merely because code or names overlap.

## Reading the evidence

- Historical frozen protocols and run snapshots describe specific versions. Current working source is a different evidence category, even when bytes match.
- Phase 2/3 is development CV with disclosed scaler overlap and selection reuse. Its completion does not establish independent confirmation.
- Phase 3B's completion record says its predictive threshold was not met. Post-outcome reconciliation documents remaining definitions; it cannot preregister the completed run.
- Experiment 04's results report a positive primary statistical contrast under its assumptions, while its practical-support criterion is unmet. Its local preregistration is not an externally timestamped registration.
- Saved verification/test reports are evidence of recorded local checks. They do not mean this registry reran those checks or reproduced training independently.

See the [prior scientific audit](AUDIT_20261020/FINAL_AUDIT_SUMMARY.md), [architecture review](docs/architecture_review/README.md), [research assessment and proposals](docs/research_validation/README.md), and [Experiment 04 limitations](docs/research_validation/experiment_04/LIMITATIONS.md). These documents have their own dates and scopes; older proposal/status text is preserved historical evidence.

## Original locations and preservation

| Location | Role |
| --- | --- |
| [conn_experiment_01](conn_experiment_01/) | Shared synthetic/HAR package and original run trees |
| [CoNN_Associative_Relationships](CoNN_Associative_Relationships/) | Historical Phase 0/1 plus later Phase 2/3 and nested Phase 3B |
| [experiments/experiment_04](experiments/experiment_04/) | Separate Experiment 04 implementation and run tree |
| [ORGANIZED_EXPERIMENTS](ORGANIZED_EXPERIMENTS/ORGANIZATION_REPORT.md) | Historical copied evidence packets and original-artifact references; not independently runnable packages |
| [AUDIT_20261020](AUDIT_20261020/) | Dated audit and supporting calculations |
| [docs](docs/) | Architecture, research reviews, proposals and study documentation |
| [CoNN_Cursor_Experiment_01](CoNN_Cursor_Experiment_01/) | Historical build instructions |

All original paths remain in place. Source imports, absolute run pointers and report links can depend on this layout; the registry does not make the workspace portable or authorize relocation. Existing environments are preserved, and no environment is designated a universal replay environment.

Git now exists on branch `initial-repository`. Earlier documents saying no Git repository existed refer to their historical capture context; historical runs do not acquire a source revision retroactively. This registry creation did not stage, commit or push any files and did not alter pre-existing staged work.

## Open research and provenance questions

- Which exact original source bytes executed the initial smoke/pilot and clean Phase 0/1 boundary?
- Which Phase 2/3 source stage executed each fit, including resumed fits?
- Which environment and transitive dependency versions suffice for independent replay of each study?
- How should a future Phase 3B freeze resolve the documented null-population, layer-attribution and C3 futility definitions?
- Does a separately specified study establish a benefit over the best fixed-width control, transfer beyond favorable tasks, or learned associations? Existing completion records do not settle these questions.

This change creates navigation and registry files only. Further reorganization requires explicit approval.
