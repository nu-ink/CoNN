# 11 — Evidence traceability

Each row connects a claim to the equation or rule, the code, the configuration or table, and the report sentence. "Audit check" is this review's independent recalculation unless noted as source inspection only.

| ID | Claim | Equation or rule | Code | Data or config | Report | Audit check |
| --- | --- | --- | --- | --- | --- | --- |
| T-01 | Gate is a unit-height Gaussian | `G=exp(-d^2/(2 tau^2))` | `conn_exp/gates.py` 20–28; `conn_assoc/gate.py` 33–41 | `pilot.yaml` `d50_start` 0.08, `d50_end` 1 | Brief schedule section; `PROTOCOL_v2.md` lines 85–101 | Formula inspected. Float32 profile in `recalculated_statistics.json` |
| T-02 | Half-distance | `d50=tau*sqrt(2 ln 2)` | `schedules.py` 11; `config.py` 172–173 | same | `DESIGN_DECISIONS.md` lines 16–18 | Algebra repeated in the math report. Rounded factor 1.17741 is stored separately in `schema.py` 27 |
| T-03 | Effective weight | `W_eff=W⊙G` | `models.py` 26–28 | all gated runs | Brief line 13 | Inspected. `G` is detached |
| T-04 | Parameter gradient | `∂L/∂W=G⊙∂L/∂W_eff` | `schema.py` 13–17; `diagnostics.py` 28–31 | gradient CSV columns | Brief diagnostic section | Chain rule under detachment. Autograd fixture not rerun |
| T-05 | No learning of `tau` | `tau` is a step function of the clock | `schedules.py` `schedule_at` | release fraction 0.7 | Protocols | Inspected. `tau` is not a parameter |
| T-06 | B′ preserves row multisets | Within-row column permutation | `gates.py` 31–36, 167–169 | mask stream offset 3000003 | `DESIGN_DECISIONS.md` 22–28 | Inspected. Brief "destinations" wording is the conflicting text |
| T-07 | Pilot D−B′ = 0.021875 | Paired mean, df=4 t interval | runner scores | `results/runs/20261017T235839Z_pilot/scores.csv` | `RESULTS.md` lines 23–33 | Recalculated; match |
| T-08 | 1B D−B′ = 0.0484375 | Paired mean, df=19 | 1B replication | `.../experiment1b_v1/replication/scores.csv` | `research_summary.md` line 5 | Recalculated; exact match |
| T-09 | 1B S exceeds D | Secondary paired contrast | same file | same | `research_summary.md` lines 59–69 | Recalculated mean D−S = −0.0265625, interval excludes zero |
| T-10 | 1C primary unresolved | P−D | 1C scores, variant ids | `experiment1c_v1/replication/scores.csv` | `research_report.md` lines 5–6 | Recalculated; interval includes zero |
| T-11 | XOR alignment reverses | D−shuffle on XOR | same | same | `research_report.md` line 20 | Recalculated mean −0.025390625 |
| T-12 | HAR engineered primary is negative | D−B macro-F1 | Experiment 2 | `experiment2_har_v1/replication/scores.csv` | `research_report.md` line 5 | Recalculated mean −0.00501038 |
| T-13 | Different HAR learning rates | Validation selection | selection file | engineered D at 0.1, B at 0.05 | `research_report.md` table lines 30–43 | Confirmed in scores. Matched-LR validation still negative at 0.01, 0.05, and 0.1 |
| T-14 | Phase 2 schedule beats shuffle on CV | Subject-macro-F1 | `phase23` | `individual_seed_differences.csv` | Phase 2/3 report line 7 | Seed-mean +0.011375365 recalculated |
| T-15 | Fixed d50=0.25 beats the Phase 2 schedule | same | same | `selected_seed_scores.csv` | report table lines 100–106 | Seed-mean difference +0.009291, 5/5 positive |
| T-16 | Phase 3 context is nearly constant | `C=1-0.475 sigmoid(U)sigmoid(V)^T`, U=−6 | `phase3_context.py` 32–34 | modulation CSV cited by the report | Phase 2/3 report lines 156–157 | Algebra and initialization inspected. The 1e-6 empirical range was not recomputed from checkpoints |
| T-17 | Phase 3B primary +0.00213398 | Subject bootstrap, seed 7000031 | `reporting.py` 8–12 | `subject_differences.csv`, `paired_contrasts.csv` | `PHASE3B_FINAL_REPORT.md` section 10 | Mean and both endpoints reproduced exactly |
| T-18 | +0.005 criterion fails | Prespecified threshold | protocol | same | executive summary line 5 | 0.00213 < 0.005. Confirmed |
| T-19 | Seed-level uncertainty includes zero | t interval on 5 seed means | not the authors' primary interval | `seed_differences.csv` | section 10 lists the five seed values, including two negatives | Recalculated interval [−0.001500, +0.005768] |
| T-20 | Shuffled codes are not separated | C4−shuffled bootstrap | `reporting.py` | `paired_contrasts.csv` row 4 | final report section 11 | Published interval includes zero. Not re-drawn; endpoints taken from the CSV that the bootstrap writer produced |
| T-21 | Rank-2 allocation can be rank-1 | Identical columns | `phase3_context.py` 29–30 | Phase 3B spectra table | final report section 5 | Algebra confirmed. Singular-value ranges not recomputed from factors |
| T-22 | Terminal D is dense | `G=1` after release | `schedules.py` 163–174; `phase23_core.py` 107 | release step 2240 of 3200 | design notes lines 45–48 | Inspected |
| T-23 | Early gate is approximately sparse | `d50=0.08` on the unit interval | formula evaluation only | 32×100 and 64×1152 profiles | — | 42.3% and 41.7% of gates below 1e-6 in float32 |
| T-24 | Associative memory was shown | none | no Phase 4 module | — | Phase 3B section 16 says it was not shown | Confirmed absent |
| T-25 | `tau(q)=sigma/r(q)` | absent | absent | absent | absent | Search of source texts. Not in the infographic's implemented mathematics |
| T-26 | Neuron merge / variable width | infographic | fixed `hidden_sizes` | `pilot.yaml` 20; HAR width 64 | infographic versus `models.py` 44–50 | Confirmed mismatch |

## Provenance gaps

Checkpoint tensors behind T-16's empirical range and behind the bitwise release-pair claims were not opened. Those claims stay at the level of the project reports. Score claims T-07 through T-15 and T-17 were checked from text tables.
