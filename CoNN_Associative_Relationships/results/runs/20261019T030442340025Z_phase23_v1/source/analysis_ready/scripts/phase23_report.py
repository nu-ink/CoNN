"""Render the research report only from completed, verified development measurements."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:]=[p for p in sys.path if Path(p or '.').resolve()!=ROOT/'scripts']
sys.path[:0]=[str(ROOT/'src'),str(ROOT.parent/'conn_experiment_01/src')]
import numpy as np
import pandas as pd


def table(frame):
    def fmt(x):
        if isinstance(x,(float,np.floating)):return f'{x:.6f}'
        return str(x)
    cols=list(frame.columns)
    return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+
        ['| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)])


def main():
    run=Path((ROOT/'config/phase23_active_run.txt').read_text().strip())
    out=run/'analysis'
    verified=json.loads((run/'final_verification.json').read_text())
    assert verified['fits']==1050
    summary=pd.read_csv(out/'condition_summary.csv')
    paired=pd.read_csv(out/'paired_contrasts.csv')
    context=pd.read_csv(out/'selected_seed_context_scores.csv')
    runtime=pd.read_csv(out/'runtime.csv')
    precision=pd.read_csv(out/'context_precision_and_symmetry.csv')
    context_gradients=pd.read_csv(out/'context_gradient_totals.csv')
    release=pd.read_csv(out/'release_effect_differences.csv')
    interventions=pd.read_csv(out/'context_interventions.csv')
    seal=json.loads((run/'protocol_seal.json').read_text())
    bench=json.loads((run/'benchmark.json').read_text())
    machine=json.loads((run/'machine_hardware.json').read_text())
    cuda=json.loads((run/'cuda_execution.json').read_text())
    io_events=sum(len(p.read_text().splitlines()) for p in (run/'jobs').glob('*/io_retries.jsonl'))
    rpath='../results/runs/'+run.name
    basecols=['condition','lr','subject_accuracy_mean','subject_macro_f1_mean','subject_macro_f1_seed_sd','parameters']
    contrastcols=['contrast','metric','mean','seed_sd','ci_low','ci_high','subject_dz']
    ctx=context.groupby(['condition','context'])[['accuracy','macro_f1']].mean().reset_index()
    ctx=ctx[ctx.condition.str.startswith('P3_')]
    c1=paired[(paired.contrast=='P2_gaussian minus P2_dense')&(paired.metric=='macro_f1')].iloc[0]
    c2=paired[(paired.contrast=='P2_gaussian minus P2_rowshuffle')&(paired.metric=='macro_f1')].iloc[0]
    c3=paired[(paired.contrast=='P3_gaussian_context minus P3_gaussian_masked')&(paired.metric=='macro_f1')].iloc[0]
    c4=paired[(paired.contrast=='P3_dense_context minus P3_dense_masked')&(paired.metric=='macro_f1')].iloc[0]
    def describe(r):return f"{100*r['mean']:+.4f} percentage points (descriptive subject-bootstrap 95% interval [{100*r['ci_low']:+.4f}, {100*r['ci_high']:+.4f}])"
    def direction(r,a,b):
        if r['ci_low']>0:return f'The development comparison favors {a} over {b}; this is exploratory support within the frozen CV design.'
        if r['ci_high']<0:return f'The development comparison favors {b} over {a}; it does not support the proposed {a} advantage in this setting.'
        return f'The development advantage of {a} over {b} is unresolved: this descriptive interval includes zero.'
    epochs=runtime.groupby('condition')[['elapsed_seconds','update_seconds']].agg(['mean','std'])
    timing=runtime.groupby('condition').agg(fits=('condition','size'),mean_job_seconds=('elapsed_seconds','mean'),
        sd_job_seconds=('elapsed_seconds','std'),mean_update_seconds=('update_seconds','mean'),parameters=('parameter_count','first')).reset_index()
    measured=interventions[interventions.kind=='same_weights_context_factor_set_to_one'].groupby('condition')[['accuracy_native_minus_disabled','macro_f1_native_minus_disabled','loss_native_minus_disabled']].mean().reset_index()
    fixed=interventions[interventions.kind=='fixed_h1_hidden_gate_context_switch_diagnostic_only'].groupby(['condition','context']).logit_rms_change.mean().reset_index()
    grad_summary=context_gradients.groupby(['condition','family']).total_l2_gradient_norm.mean().unstack('family').reset_index()
    grad_summary['context_to_base_norm_ratio']=grad_summary['context']/grad_summary['base']
    rank_symmetric=bool((precision.u_rank_columns_max_abs_difference==0).all() and (precision.v_rank_columns_max_abs_difference==0).all())
    record_names=['PROTOCOL_PHASE23_v1.md','README_PHASE23.md','scripts/phase23_preflight.py','scripts/run_phase23.py',
        'scripts/phase23_report.py','scripts/phase23_finish.ps1','scripts/phase23_finalize.py','src/conn_assoc/phase23_core.py','src/conn_assoc/phase23.py',
        'src/conn_assoc/phase3_context.py','src/conn_assoc/phase23_analysis.py',
        'src/conn_assoc/phase23_diagnostics.py','src/conn_assoc/phase23_verify.py',
        'tests/test_phase2.py','tests/test_phase2_cuda.py','tests/test_phase3.py',
        'tests/test_phase23_integrity.py','tests/test_phase23_io.py','config/phase23_active_run.txt']
    inventory='\n'.join('- `'+p+'`' for p in record_names)
    results=summary.copy()
    text=f'''# Phase 2/3 implementation and development results

## Executive summary

Completed **1050/1050 development fits**, 3,360,000 optimizer updates: six Phase 2 conditions and four Phase 3 conditions, each with seven subject folds, five engineering seeds and three learning rates. Every fit completed 3200 updates. Models, exact-resume checkpoints, curves, validation predictions, class/subject metrics, diagnostics and manifests are saved under `{run.name}`. No official-test model evaluation occurred. No Phase 4 component or associative retrieval result exists.

The primary development Gaussian−Dense macro-F1 difference is {describe(c1)}. The required Gaussian−row-shuffled comparison is {describe(c2)}. These are selected-LR development results, not independent replication or final-test confirmation.

Context increments are Gaussian+Context−masked Gaussian {describe(c3)}, and Dense+Context−masked Dense {describe(c4)}. Interpret them alongside actual modulation magnitudes, same-weight interventions, extra capacity and the numerical limitations below; they do not establish a benefit from associative memory.

The original CPU environment and all checked historical artifacts remain unchanged. An isolated CUDA environment works on both cards. Sequential warmed-up benchmarking favored CPU for this small MLP. The grid began with six one-thread CPU workers and resumed with eight; execution plans label these contention conditions. Historical import/Tk problems did not recur. One initial test assertion that every U parameter must change on a single minibatch failed: the test exposed real float32 update rounding, not detached gradients. A Windows checkpoint-replacement error also paused the grid; evidence was preserved, a bounded I/O retry was added, and the interrupted trajectory was replay-verified before resuming. All setup/test/failure attempts remain saved.

## Implementation inventory

All implementation files below are new; no pre-existing Phase 0/1 source, protocol, configuration, report or result was modified:

{inventory}

Generated environments/artifacts: `.venv_cuda23/`, `.pip-cache23/`, this report, and the unique run directory. Source snapshots before each implementation stage and source manifests are under `{rpath}/source/`; original protocol/readme/audit snapshots are under `prior_snapshot/`. Original Experiment 2 `conn_exp` model/gate utilities are imported read-only from their existing source and hash-verified. There is no Git repository/commit; source snapshots and SHA-256 seals provide provenance.

### Frozen design and authorization

The user explicitly authorized this separate Phase 2/3 extension. `PROTOCOL_v2.md` is unchanged, hash `{seal['original_protocol_sha256']}`. The separately sealed amendment hash is `{seal['amendment_sha256']}`; resolved configuration hash `{seal['config_sha256']}`. Seal time `{seal['sealed_utc']}` predates training. The amendment changes only scope and explicitly defined new context/evaluation implementation; it does not lift the official-test embargo.

New names map to Experiment 2 A/D/B/C/S08/S25 as specified in the amendment. `P2_gaussian` is original D, not historical B. All use 1152 time-major inputs, 64/64 ReLU, six-class dense head, paired uniform initialization, plain SGD without momentum/decay, batch 128, LR [.01,.05,.1] and seeds 400–404. Original Gaussian geometry/detach behavior/smoothstep are retained; release step 2240 uses exact ones. Fixed d50 .08/.25 never release. Row permutations preserve row energy; coordinate permutation changes gate assignments, not input values.

The seven validation folds are the unchanged v2 groups of three subjects. Only the 7352 official-training rows are used. Initialization, batch order, mask permutations and context/execution orders have independent streams. The first 42 Dense/Gaussian seed-400 fits were completed/reviewed before context implementation; they count toward the grid. `first_seed_review.json` and `first_seed_summary.csv` preserve that progression without selecting LR early.

Every condition receives 105 fits and 336000 updates. LR is selected solely by final validation subject-macro-F1 averaged equally over 21 subjects, five seeds and, for Phase 3, three contexts; ties use validation loss then lower LR. `selection.frozen.json` records all candidates. No best-curve checkpoint selection or test-guided redesign occurred.

### Context mechanism and masking

External codes are both / accelerometer / gyroscope. Both body and total acceleration derive from the accelerometer group; the gyroscope supplies three angular-velocity channels. Unavailable channels become zero after frozen scaling. All four Phase 3 conditions see identical balanced context/mask sequences, batches and training budgets; only contextual models receive the code as a modulation selector. Comparisons within Phase 3 hold the masking curriculum fixed. Comparing Phase 2 full-sensor training directly to Phase 3 mixed-sensor training would confound that curriculum.

For each hidden layer, `C(c)=1-(.95/2)*sigmoid(U[c])@sigmoid(V[c]).T`, with nominal attenuation rank 2, U initialized -6 and V zero. C is positive, bounded [.05,1], initially approximately .9988255. This near-one initialization is explicitly amended from v2 exact-one so gradients can exist at initialization. Gaussian factors are detached; C is differentiable and cannot amplify G. Context models add 8064 parameters (86406 total versus 78342). They match each other in capacity, not their smaller baselines. There is no full per-sample hypernetwork matrix.

## CUDA report

Verified CPU: {machine['cpu']}; {machine['logical_cpu_count']} logical CPUs; {machine['total_ram_bytes']/2**30:.3f} GiB RAM. Original executable: `E:\\CoNN\\conn_experiment_01\\.venv\\Scripts\\python.exe`, Python 3.11.1 / PyTorch 2.14.1+cpu. Original package freeze is unchanged.

Isolated executable: `.venv_cuda23/Scripts/python.exe`, PyTorch {cuda['torch']} with bundled CUDA {cuda['torch_cuda']}. `nvidia-smi` records driver 617.42 and supported CUDA UMD 13.4. Both RTX 3070 Ti cards expose approximately 8 GiB independently. Tensor allocation, matmul, backward gradients and CPU transfer passed on each GPU. GPU checkpoint/optimizer smoke and CPU/GPU inference checks passed (atol 2e-5, rtol 2e-4). No system toolkit, model sharding or distributed training was installed.

Installation used the [official CUDA 13.0 PyTorch index](https://download.pytorch.org/whl/cu130/torch/) and [official installation guidance](https://pytorch.org/get-started/locally/). The sandbox network initially failed; a network-enabled retry found the wheel. Old pip dependency metadata/build resolution failed; upgrading pip only inside the new environment resolved it. Logs retain all attempts. `cuda_requirements.freeze.txt` and `cuda_pip_check.log` record the resulting environment. The initial mid-install import probe failed before support packages were ready; it was not treated as a CUDA execution result.

{table(pd.DataFrame(bench['measurements'])[['device','condition','batches_per_second','samples_per_second','peak_gpu_allocated_bytes']])}

Timing used 20 warm-up and 200 measured updates per model, sequential CPU then GPU 0, float32/TF32 off, GPU synchronization. Device selection used throughput only. Two-GPU seed parallelism was not pursued because these measurements favored CPU. No GPU acceleration improvement is claimed. The subsequent CPU grid reuses identical deterministic Gaussian tensors to avoid repeated exponentiation; this numerical cache is not associative memory or a learned bank. Unit tests retain exact original-gate equivalence. Grid timings below include parallel contention and differ from isolated benchmark latency.

## Validation and execution

| Verification | Result | Evidence in run |
|---|---|---|
| Phase 2 tests | 24 passed | pytest_phase2_retry.log |
| Phase 2 including CUDA | 26 passed | pytest_phase2_cuda.log |
| Phase 3 including CUDA | 30 passed | pytest_phase3_cuda.log |
| Final CPU suite | 28 passed, 2 explicitly NOT RUN (CPU-only torch) | pytest_final_cpu.log |
| Additional integrity/I/O guards | 3 passed | pytest_recovery.log |
| Final delivery CUDA suite | 33 passed, including CUDA checks | pytest_delivery_cuda.log |
| Gate/control schedule equivalence | Bitwise equal at representative endpoints/interior/release | tests/test_phase2.py |
| Actual updates, masking, differentiability, bounds | Passed | tests/test_phase2.py, tests/test_phase3.py |
| Exact interrupted/resumed weights | Passed for Gaussian and context fits | verification/ resume fixtures |
| Independent final verification | {verified['fits']} fits; {verified['checkpoint_files']} checkpoint hashes | final_verification.json |
| Stored predictions independently rescored | {verified['prediction_rows_rescored']} rows, all confusion/recall/F1/accuracy match | final_verification.json |
| Same-weight pre/post release | {verified['pre_post_release_weight_pairs_identical']} identical weight pairs | final_verification.json |
| Historical preservation | {verified['original_experiment2_files_unchanged']} original Experiment 2 and {verified['prior_associative_files_unchanged']} prior associative files unchanged | final_verification.json |

Initial, pre-release, post-release, final and latest checkpoints store model/optimizer/RNG/batch/context histories and explicit next-step schedule. No stochastic LR scheduler exists. Latest checkpoints are atomic every 160 updates. A failed/interrupted process resumes from that boundary; completed fits are reused and never overwritten. Both uninterrupted-vs-resumed tests verify exact weights. Every completed fit checks checkpoint-reloaded predictions and actual weight changes.

Setup/test failures were preserved: an initial PowerShell JUnit argument invocation collected no tests; the corrected invocation passed. A Phase 3 assertion on U updates failed because float32 rounded tiny updates away while nonzero gradients and V updates remained. The revised test explicitly measures both gradients and actual V updates, and stores precision evidence; it does not claim U updated when it did not.

The first full-grid invocation paused with WinError 5 while replacing `P2_rowshuffle_seed403_fold5_lr0.1/latest.pt`. The last committed checkpoint was at update 640; the intact temporary checkpoint was at 800. Replaying those 160 updates from preserved model/optimizer/RNG state matched the temporary weights bitwise (`recovery_evidence/checkpoint_replace_01/replay_verification.json`). The OS did not identify the lock holder; a transient sharing/access issue is the supported diagnosis, not an asserted antivirus cause. Bounded replacement retry (at most ten attempts, short delays) preserves the last good checkpoint and leaves the temp file on persistent failure. The I/O regression test passed, original failure/checkpoints were copied before resume, and completed fits were reused. The 3,360,000-update total describes completed logical trajectories; failed/replayed work adds physical computation and is excluded from tuning outcomes.

The resumed completed jobs record {io_events} transient replacement retry events in their own `io_retries.jsonl` files, demonstrating recovery during actual training as well as the synthetic file-lock regression test.

## Phase 2 results

{table(summary[summary.condition.str.startswith('P2_')][basecols])}

Accuracy/F1 here average subjects equally, then seeds. Seed SD quantifies five engineering initializations, not five independent populations. Full per-seed and per-subject values are in `analysis/selected_seed_scores.csv` and `selected_validation_subjects.csv`; class recalls/confusions and pooled-window OOF scores are separately labelled. Final train/validation loss, gaps and fold averages are in `condition_summary.csv`.

## Phase 3 results

{table(summary[summary.condition.str.startswith('P3_')][basecols])}

Realistic sensor-specific performance (0 both, 1 accelerometer, 2 gyroscope):

{table(ctx)}

Paired comparisons and uncertainty:

{table(paired[paired.metric=='macro_f1'][contrastcols])}

`individual_seed_differences.csv` supplies all seed-level paired differences; accuracy contrasts are in `paired_contrasts.csv`. Bootstrap intervals use 10000 subject resamples, averaging seeds first. They are descriptive post-selection development intervals: CV training sets overlap and the same folds selected LR. They do not establish confirmatory success, and no window-independent significance test is used.

### Actual modulation and same-weight interventions

Across selected context checkpoints, maximum absolute C departure from its near-one initialization ranges from {precision.max_abs_change_from_initial_C.min():.9g} to {precision.max_abs_change_from_initial_C.max():.9g}; observed C range is [{precision.modulation_min.min():.9g}, {precision.modulation_max.max():.9g}]. Individual U updates can round away in float32; `context_precision_and_symmetry.csv` records U/V changed-element counts and deltas. Both rank columns remain exactly symmetric across these checkpoints: **{rank_symmetric}**. Where symmetric, the nominal rank-two deficit operates at rank at most one, an initialization-induced limitation.

Measured total gradient norms, averaged over recorded steps/folds/seeds (base and context parameter families kept separate):

{table(grad_summary)}

With weights fixed, realistic validation input masks unchanged, set C=1 (native minus disabled; fold/context averages):

{table(measured)}

Fixed-h1 interventions vary only the hidden context factor, holding the same learned representation fixed. They are diagnostics, not realistic missing-sensor predictions:

{table(fixed)}

Additional capacity, near-constant attenuation and training-path changes remain alternative explanations for between-model differences. These interventions isolate immediate gate effects at the saved weights; they cannot retrospectively undo optimization or supply a capacity-matched training control. No causal claim that context caused a gain is justified solely by a positive contextual-model comparison.

## Curves and diagnostic plots

All figures derive from computed frozen gates or measured checkpoints/metrics. Representative weight/modulation plots use fixed seed 400, fold 0, never the best-performing fit.

![Frozen schedule gates]({rpath}/plots/gaussian_expansion.png)

![Effective trained weights]({rpath}/plots/effective_weight_patterns.png)

![Phase 2 measured curves]({rpath}/plots/P2_learning_curves.png)

![Phase 3 measured curves]({rpath}/plots/P3_learning_curves.png)

![Model comparison]({rpath}/plots/model_comparison.png)

![Paired development contrasts]({rpath}/plots/paired_contrasts.png)

![Sensor contexts]({rpath}/plots/sensor_context_effects.png)

![Dense modulation]({rpath}/plots/P3_dense_context_modulation.png)

![Gaussian modulation]({rpath}/plots/P3_gaussian_context_modulation.png)

![Gradient diagnostics]({rpath}/plots/gradients.png)

![Context versus base gradients]({rpath}/plots/context_gradients.png)

![Context modulation trajectories]({rpath}/plots/context_modulation_curves.png)

![Throughput]({rpath}/plots/cpu_cuda_throughput.png)

`release_and_final_comparisons.csv` distinguishes immediate release from later optimization: pre/post release checkpoints have identical weights, so their metric difference is a gate intervention. Final-minus-post-release additionally includes 960 optimizer updates. `representative_distant_connection_ablation.csv` measures final development checkpoint far-connection suppression at distance quantile .75, separately labelled post-selection diagnostics; it did not influence model selection or redesign.

Mean fold/context validation changes (positive accuracy/F1 is better; positive loss is worse):

{table(release.groupby('condition')[['macro_f1_immediate_gate_switch','macro_f1_further_960_updates','loss_immediate_gate_switch','loss_further_960_updates']].mean().reset_index())}

![Release versus later optimization]({rpath}/plots/release_vs_optimization.png)

### Runtime and storage

{table(timing)}

Job wall times include evaluation, checkpointing and contention. Update time excludes most gate construction/evaluation and GPU asynchronous timing is not used for the CPU grid. GPU peak allocation belongs to the isolated benchmark/smokes, not CPU training. `python_process_memory_during_grid.json` records observed process working sets/peaks, including deterministic gate caches; it is not a per-model incremental RAM estimate. Base/context parameter tensors require 313368/345624 float32 bytes, excluding activations, optimizer/RNG/data and cached tensors. Every checkpoint and stored array contributes separate disk storage, not trainable parameters.

## Scientific interpretation

The executed development contrast is Gaussian−Dense {describe(c1)}, alongside required Gaussian−rowshuffle {describe(c2)}. These measured values should be read with their intervals, individual seeds and subject dependence; they do not supersede historical Experiment 2's negative engineered-feature primary or unresolved temporal secondary results. Task, folds, scaler, schedule and test embargo were preserved.

{direction(c1,'scheduled Gaussian CoNN','Dense')} {direction(c2,'structured Gaussian CoNN','the energy-matched shuffled control')} Neither comparison meets v2's later independent-confirmation rule.

{direction(c3,'Gaussian + Context','masked Gaussian')} {direction(c4,'Dense + Context','masked Dense')}

The contextual comparisons measure the frozen mechanism under a matched sensor-mask curriculum. Immediate intervention and modulation measurements are essential: tiny modulation changes and initialization symmetry restrict what this implementation tests about useful context conditioning. A better contextual-model score alone cannot distinguish explicit context use from extra parameters, nearly constant rescaling or optimization effects. Positive/negative differences are retained equally; no final-test redesign follows from them.

No memory bank, retrieval support/query evaluation, Recall@k or MRR was implemented. Neither classification gains nor successful software execution demonstrate associative memory, prove the geometry hypothesis or establish a new architecture.

## Limitations

- Fixed Experiment 2 scaling uses 16 of the 21 development subjects; some validation-fold subjects contributed to preprocessing. This accepted leakage limitation is not silently corrected. A fold-fitted sensitivity study requires a separately declared amendment.
- The release lacks continuous session/timestamp/overlap keys. Within-window temporal alignment is valid; adjacent rows are not asserted to be overlapping neighbors.
- Five engineering seeds and selected-LR CV are not independent confirmation. Bootstrap intervals do not remove overlapping training sets or selection bias. Test subjects have appeared in historical research but were not loaded by this development runner.
- Context has 8064 extra parameters; no parameter-matched noncontext training condition was added to the frozen ten-condition grid. Symmetric factor initialization and float32 rounding limit effective context learning.
- Balanced homogeneous context per minibatch is the frozen policy. It does not test arbitrary mixed-code examples in the same matrix operation.
- CPU/GPU agreement was checked within declared tolerances; full models were trained on CPU only. No cross-backend full-grid equivalence or GPU speedup is claimed.
- Representative far-connection and fixed-hidden interventions are post-selection development diagnostics, not predeclared primary confirmatory comparisons.
- Git history is unavailable. Full source snapshots, environment freezes, protocol/config hashes and file verification provide reproducibility records.

## Next research stage

Do not treat these classification measurements as justification that associative memory will work. Before Phase 4, review Phase 3's actual modulation/precision evidence and, if context remains nearly inert, predeclare a separate context-parameterization sensitivity experiment using development data only. Preserve this implementation and its results as measured, including negative findings. A future capacity-matched control and separately frozen independent protocol would improve attribution.

Phase 4 remains unimplemented and requires separate authorization/design: bank capacity, key refresh, temperature, exact-row exclusion, non-label values, support/query separation and matched negative controls. Final-test evaluation also remains separately unauthorized. No new protocol or test evaluation is silently initiated by this recommendation.

Reproducible commands are in `README_PHASE23.md`; the exact completed run is `{run}`. To validate/resume existing work without changing its design:

```powershell
Set-Location E:\\CoNN\\CoNN_Associative_Relationships
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = 'src;..\\conn_experiment_01\\src'
$run = (Get-Content config/phase23_active_run.txt).Trim()
..\\conn_experiment_01\\.venv\\Scripts\\python.exe scripts/run_phase23.py grid --stage all --workers 6 --devices cpu
..\\conn_experiment_01\\.venv\\Scripts\\python.exe -m conn_assoc.phase23_verify --run $run
```
'''
    path=ROOT/'reports/PHASE_2_3_IMPLEMENTATION_AND_RESULTS.md'
    if path.exists():raise FileExistsError('Preserve the prior report; use a versioned report path')
    path.write_text(text,encoding='utf-8')
    (run/'research_report.md').write_text(text.replace(rpath+'/', ''),encoding='utf-8')
    (run/'report_manifest.json').write_text(json.dumps({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'implementation_inventory':record_names},indent=2)+'\n')
    print(path)


if __name__=='__main__':main()
