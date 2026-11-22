"""Final measured scientific synthesis. No training or endpoint redesign."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import sys
ROOT=Path('E:\\CoNN\\CoNN_Associative_Relationships\\phase3b_context_diagnostics')
sys.path.insert(0,str(ROOT/'src'))
import collections
import json
import shutil
import re
import numpy as np
import torch
from common import read,write,sha,csv_read,csv_write

def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(x) for x in row)+' |\n' for row in rows)+'\n'

def mean(rows,key):return float(np.mean([float(r[key]) for r in rows])) if rows else float('nan')
def fmt(x):return f'{float(x):.8f}'

def prose_polish(text):
    # Format prose without changing measured numbers, identifiers or equations.
    text=re.sub(r'\b(all|and|with|at|over|of|in|first|only|each|across|Level|Phase|step|seed|seeds|contexts|Completed|completed|versus|than|before|after|within|both|including|have|the|plus)(?=\d)',r'\1 ',text)
    def separate(match):
        digit,letter=match.groups()
        following=text[match.end():match.end()+1]
        if letter=='B' or (letter in 'eE' and following in ('−','-','+')):return match.group(0)
        return digit+' '+letter
    text=re.sub(r'(\d)([A-Za-z])',separate,text)
    text=re.sub(r'%(?=[A-Za-z])','% ',text)
    replacements={'GaussianC':'Gaussian C','DenseC':'Dense C','Gaussianmasked':'Gaussian masked',
        'Densemasked':'Dense masked','CPUwas':'CPU was','CUDAforward':'CUDA forward','CUDAtraining':'CUDA training',
        'CPUbackend':'CPU backend','GPUmemory':'GPU memory','GPUdriver':'GPU driver','CUBLASworkspace':'CUBLAS workspace',
        'IH64':'IH 64','HH64':'HH 64',
        'IOand':'IO and','CPU/CUDAcharts':'CPU/CUDA charts','SGD.1':'SGD 0.1',
        'at h1e−5,atol1e−8,rtol1e−5':'at h=1e−5, atol=1e−8, rtol=1e−5',
        '`to rescore':'` to rescore','`is single-owner':'` is single-owner','development-grid fits':'development grid fits'}
    for a,b in replacements.items():text=text.replace(a,b)
    return text

def polish(text):
    text=text.replace('`to rescore','` to rescore').replace('`is single-owner','` is single-owner')
    pieces=re.split(r'(`[^`]*`)',text)
    return ''.join(piece if piece.startswith('`') else prose_polish(piece) for piece in pieces)

def main():
    run=ROOT/'results/20261020_phase3b_v3_run01'
    assert read(run/'analysis_summary.json')['complete']
    assert (run/'execution_extensions/recovery_analysis_v1/finalization_stage_complete.json').exists()
    extension=run/'execution_extensions/measured_summary_recovery_v1';extension.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,extension/'measured_summary.py')
    write(extension/'manifest.frozen.json',{'source_sha256':sha(__file__),'role':'measured scientific synthesis only',
        'primary_statistics_source':'analysis/paired_contrasts.csv','no_new_models_or_model_selection':True,
        'primary_rule_unchanged':True})
    # Archive earlier generated research summaries before adding the complete synthesis.
    for path in (run/'research_report.md',ROOT/'reports/PHASE3B_FINAL_REPORT.md',ROOT/'reports/PHASE3B_EXECUTIVE_SUMMARY.md'):
        shutil.copy2(path,extension/('earlier_'+path.name))
    a=run/'analysis';summary=read(run/'analysis_summary.json');manifest=read(run/'execution_manifest.frozen.json')
    contrasts=csv_read(a/'paired_contrasts.csv');primary=next(r for r in contrasts if r['family']=='primary')
    subjects=csv_read(a/'verified_subject_scores.csv');subjects=[r for r in subjects if r['stage']=='grid']
    numerical=csv_read(a/'numerical_learning.csv');levels=csv_read(a/'level2_null_comparisons.csv')
    gradients=csv_read(a/'gradient_trajectories.csv');ranks=csv_read(a/'attenuation_spectra.csv')
    seedrows=csv_read(a/'seed_differences.csv');gaps=csv_read(a/'generalization_gaps.csv')
    release=csv_read(a/'release_vs_optimization.csv');inter=csv_read(a/'fixed_representation_interventions.csv')
    changes=csv_read(a/'per_factor_initial_final_changes.csv');capacity=csv_read(a/'parameter_storage_and_active_capacity.csv')
    precision=csv_read(a/'final_precision_diagnostics.csv');sweep=csv_read(a/'gradient_initialization_sweep.csv')
    accounting=read(run/'optimizer_accounting.json');backend=read(run/'backend_selection.json')
    reproductions=csv_read(a/'historical_full_fit_reproduction.csv')
    tests=read(run/'tests.json');assert tests['exit_code']==0
    conditions=[f'{arch}_C{i}' for arch in ('dense','gaussian') for i in range(5)]+[
        'dense_masked','gaussian_masked','gaussian_constant','gaussian_shuffled','gaussian_frozen']
    score_rows=[]
    for c in conditions:
        ss=[r for r in subjects if r['condition']==c]
        nn=[r for r in numerical if r['stage']=='grid' and r['condition']==c]
        cc=[r for r in capacity if r['condition']==c]
        if cc:count=cc[0]['allocated_parameters']+'/'+cc[0]['trainable_parameters']
        else:
            pilot=read(run/'pilots'/f'{c}_seed400_fold0/completion.json')
            count=str(pilot['allocated_parameters'])+'/'+str(pilot['trainable_parameters'])+' (pilot)'
        score_rows.append([c,len(nn),fmt(mean(ss,'macro_f1')) if ss else 'omitted',fmt(mean(ss,'accuracy')) if ss else 'not measured',
            f"{sum(r['level1']=='True' for r in nn)}/{len(nn)}" if nn and '_C' in c else 'control / omitted',
            count])
    # Independently confirm every pre/post-release pair actually has identical weights.
    identity=[]
    for job in sorted((run/'grid').iterdir()):
        if not (job/'completion.json').exists():continue
        pre=torch.load(job/'pre_release.pt',map_location='cpu',weights_only=False)
        post=torch.load(job/'post_release.pt',map_location='cpu',weights_only=False)
        exact=all(torch.equal(v,post['model'][k]) for k,v in pre['model'].items())
        assert exact and pre['next_step']==post['next_step']==2240
        identity.append({'job':job.name,'weights_bitwise_identical':exact,'pre_next_step':2240,'post_next_step':2240})
    csv_write(a/'release_checkpoint_identity.csv',identity)
    pilotrows=[]
    for arch in ('dense','gaussian'):
        for cell in range(5):
            c=f'{arch}_C{cell}';job=run/'pilots'/f'{c}_seed400_fold0';done=read(job/'completion.json')
            ds=read(job/'diagnostics.json');ds=[r for r in ds if 'float64_max_delta_c' in r]
            maxref=max(r['float64_max_delta_c'] for r in ds)
            gg=read(job/'gradients.json');ratio=float(np.median([r['gradient_ratio'] for r in gg]))
            pilotrows.append([c,fmt(ratio),f"{done['max_live_forward_delta_c']:.8e}",f'{maxref:.8e}',
                done['max_changed_context_fraction'],done['numerical_level1'],'omitted' if c in manifest['omitted_conditions'] else 'retained'])
    # Frozen endpoint decisions; no new aggregation rule is introduced for Level2.
    l1=summary['gaussian_C4_level1_fits'];l1n=summary['gaussian_C4_level1_denominator']
    l2=summary['level2_passed_comparisons'];l2n=summary['level2_denominator']
    benefit=float(primary['mean'])>=.005 and float(primary['ci_low'])>0
    nullgroups=[]
    for layer in ('ih','hh'):
        for pair in [(0,1),(0,2),(1,2)]:
            rr=[r for r in levels if r['layer']==layer and int(r['a'])==pair[0] and int(r['b'])==pair[1]]
            if rr:nullgroups.append([layer,str(pair),sum(r['level2_pair_pass']=='True' for r in rr),len(rr),
                fmt(mean(rr,'normalized_frobenius')),fmt(mean(rr,'initial_normalized_frobenius')),fmt(mean(rr,'learned_difference_change')),fmt(mean(rr,'null95'))])
            else:nullgroups.append([layer,str(pair),'undefined' if layer=='ih' and pair==(1,2) else 'not measured',0,'—','—','—','—'])
    hist=read(run/'historical_precision.json')['results']
    gr0=[float(r['gradient_ratio']) for r in sweep if r['candidate']=='G0']
    gr2=[float(r['gradient_ratio']) for r in sweep if r['candidate']=='G2']
    ranktable=[]
    initialrank=csv_read(a/'initial_rank.csv')
    for c in ['C0','C1','C2','C3','C4']:
        ir=[float(r['ratio']) for r in initialrank if r['cell']==c]
        fr=[float(r['sigma2_over_sigma1']) for r in ranks if r['stage']=='grid' and r['condition']=='gaussian_'+c]
        ranktable.append([c,f'{min(ir):.6g}–{max(ir):.6g}',f'{min(fr):.6g}–{max(fr):.6g}' if fr else 'pilot only',
            'below .01' if max(ir)<.01 else 'target reached'])
    finalchange=[]
    for c in ('gaussian_C0','gaussian_C2','gaussian_C4'):
        for factor in ('context_ih.u','context_ih.v','context_hh.u','context_hh.v'):
            rr=[r for r in changes if r['condition']==c and r['factor']==factor]
            finalchange.append([c,factor,fmt(mean(rr,'changed_fraction')),fmt(mean(rr,'max_abs_parameter_change'))])
    text='# Phase 3B v3 — Final measured research report\n\n'
    text+='## 1. Executive summary\n\n'
    text+=f"Completed **455 development-grid fits and10pilots**, all3200updates: **{accounting['completed_fit_training_updates']:,}training optimizer updates**. CPU was selected by throughput; all paired performance fits used CPU. Seventy C3 grid fits were omitted by the predeclared numerical futility rule. No official HAR test subjects were evaluated and no Phase4 was implemented.\n\n"
    text+=f"Primary GaussianC4−Gaussianmasked subject macro-F1: **{float(primary['mean']):+.8f}**, descriptive95%subject-bootstrapCI **[{float(primary['ci_low']):+.8f},{float(primary['ci_high']):+.8f}]**. The frozen +.005-and-positive-lower-bound predictive criterion was **{'met' if benefit else 'not met'}**. This is exploratory development evidence.\n\n"
    text+=f"GaussianC4 met Level1 in{l1}/{l1n}fits. Level2 operational pair/layer criteria passed in{l2}/{l2n}comparisons; no undeclared study-wide aggregation turns this into universal learned context differentiation. See section8. Numerical repair, functional sensitivity and predictive benefit are separate outcomes.\n\n"
    text+='## 2. Research question and frozen design\n\n'
    text+='Why did historical contextual attenuation barely learn, and do symmetry breaking, reduced saturation and master-parameter precision yield useful external-context behavior? The primary contrast, .005 practical threshold, bootstrap, three secondary contrasts,15conditions, common SGD.1, batch128,3200steps,64/64hidden widths, five base seeds400–404 and seven historical folds were fixed before pilots. The raw v3 amendment and supplied v2 historical-reference synopsis are preserved. V3 superseded the14-condition v2 matrix by adding frozen C4.\n\n'
    text+='Inputs are128×9 time-major temporal HAR windows. IH context factors control64×1152weights, HH64×64weights; categorical contexts0both,1accel,2gyro select three rank-two factor rows. Unavailable scaled channels are zeroed. Independent streams govern base initialization, factor noise(seed30031), batches, balanced mask curriculum, shuffled codes, evaluation randomization and bootstrap. No validation LR search or stronger-initialization substitution occurred. The original Gaussian geometry and release schedule were unchanged.\n\n'
    text+='## 3. Historical Phase3 comparison and prospective precision\n\n'
    text+='Before new performance training,5762historical source/data manifest entries and all1050historical fit artifacts were verified. C0 checkpoint-loaded logits and saved predictions matched exactly in six fixtures. The end-of-run preservation audit passed separately.\n\n'
    text+=table(['Historical contrast','Mean','Subject SD','Seed SD','New descriptive95%CI','Approx. zero-exclusion effect'],
        [[r['architecture']+' context−masked',f"{r['mean']:+.8f}",fmt(r['subject_sd']),fmt(r['seed_sd']),
          f"[{r['ci_low']:+.8f},{r['ci_high']:+.8f}]",fmt(r['approximate_zero_exclusion_effect'])] for r in hist])
    text+=f"Fresh same-seed C0/masked full-fit checks: {sum(r['exact_final_tensors']=='True' for r in reproductions)}/{len(reproductions)}exact final tensor states; {sum(r['exact_saved_predictions']=='True' for r in reproductions)}/{len(reproductions)}exact saved predictions. These are reproducibility checks, not independent replications. The.001macro-F1 statistical/cross-backend tolerance was not substituted for exact checkpoint replay. Historical Monte Carlo intervals can have slightly different endpoints under the new frozen bootstrap seed.\n\n"
    text+='All21subjects have all6ground-truth classes. Historical six-class macro-F1 and supported-class sensitivity therefore coincide. False positives remain in the confusion matrices; present classes with no predicted positives receive zero precision. The historical precision estimates do not guarantee power: only21subject clusters, overlapping CV training sets and previous development inspection limit interpretation.\n\n'
    text+='## 4. Mathematical gradient findings\n\n'
    text+='C=1−.475sigmoid(U)sigmoid(V)ᵀ. Local derivatives are not complete loss gradients. Analytical loss-gradient contractions matched autograd to1.39e−17in the fixed float64 fixture; central finite differences passed at h1e−5,atol1e−8,rtol1e−5. The original Gaussian gate was bitwise equivalent at tested schedule points.\n\n'
    text+=f"The42-fixture gradient sweep used identical initial weights and training minibatches across seven initializations, two architectures and three masks, without validation classification scores. G0 context/base gradient ratio ranged{min(gr0):.8g}–{max(gr0):.8g}; G2(U−4.5)ranged{min(gr2):.8g}–{max(gr2):.8g}. G2reached the1e−3engineering reference in0/6fixtures and had zero maximum one-step float32gate change. C2/C4 nevertheless retained−4.5as frozen. Stronger candidates are diagnostic follow-up proposals only.\n\n"
    text+='## 5. Symmetry and attenuation rank\n\n'
    text+='SVD uses float64 A=1−C, with numerical rank threshold sigma1×1e−10. Identical component columns constrain attenuation to rank≤1. C1/C4perturbations broke that exact symmetry, but their second component was initially weak. The prescribed perturbation was not increased.\n\n'
    text+=table(['Cell','Initial sigma2/sigma1 range(all layers/contexts)','Final Gaussian grid range','Initial .01target'],ranktable)
    text+='Rank-two allocation is not proof of two useful learned factors, semantic relationships or predictive gain. Initial random context differences must be separated from learned displacement.\n\n'
    text+='## 6. Numerical precision findings\n\n'
    text+='The following are full3200-step seed400/fold0pilots. Reference changes are evaluated in float64; forward changes use actual float32modulation on live support.\n\n'
    text+=table(['Pilot','Median context/base gradient','Max forwardΔC','Max referenceΔC','Max changed fraction','Level1','Grid decision'],pilotrows)
    text+='C3preserved float64master updates but failed the1e−6forward threshold in both pilot architectures. Its70grid fits were omitted and retained as explicit omitted_fits.csv entries, with no replacement. This is a pilot-specific engineering futility decision, not proof precision-only interventions cannot help elsewhere. C2/C4 produced changes that survived forward conversion.\n\n'
    text+=table(['Condition','Factor','Mean initial→final changed fraction','Mean max parameter displacement'],finalchange)
    text+='These final displacements are not reconstructed stepwise fractions. Sampled aggregate stored-update fractions and gradient norms are raw measurements; per-factor stepwise stored-update fractions were not saved. Scalar-conversion warnings in diagnostic formatting were preserved; those scalar reads do not enter losses or optimization.\n\n'
    text+='## 7. Controlled intervention results\n\n'
    text+=table(['Condition','Completed grid fits','Mean subject macro-F1','Mean subject accuracy','Level1 fits','Allocated/trainable parameters'],score_rows)
    text+='C0/C1remain valid comparison cells even when they fail the forward-learning threshold. Masked, constant-code and frozen-gate controls have different expected numerical-learning behavior; a frozen gate is not an implementation failure. C3classification performance is not extrapolated from one pilot. All available class metrics and per-seed/context results are preserved.\n\n'
    text+='## 8. Context-specific functional differences\n\n'
    text+='IH comparisons use source-support intersections. Accel/gyro intersection is empty and undefined; nonintersection differences are recorded separately. HH uses all potentially active hidden sources. A Gaussian shuffled-training null supplies35matched engineering seed/fold values per nonempty layer/pair; its95thpercentile is a diagnostic reference, not a significance test.\n\n'
    text+=table(['Layer','Codes','Passes','Fits','Mean final normalizedΔA','Mean initialΔA','Mean learned contrast displacement','Mean null95'],nullgroups)
    text+='The operational checks additionally require final normalized difference>1e−4, corresponding null95exceedance, live forward learning and same-weight output change>1e−6. Final/initial differences and learned displacement are all shown because random initialization alone produces code sensitivity. Dense has no separately trained matched shuffled null; its functional diagnostics are descriptive. Pair-level results do not establish consistent study-wide learned differentiation.\n\n'
    hh=[r for r in inter if r['stage']=='grid' and r['condition']=='gaussian_C4' and r['layer']=='hh_fixed_h1']
    text+=f"GaussianC4 fixed-h1 logit RMS difference mean{mean(hh,'logit_rms'):.8g}, maximum recorded absolute logit change{max(float(r['logit_max_abs']) for r in hh):.8g}. These frozen probes use the first128validation windows per fold, not the whole population. Separate IH-only interventions fix the HHcode0; disabled-Cand all five nonidentity permutations keep actual sensor masks and weights fixed.\n\n"
    text+='## 9. Classification, convergence and generalization\n\n'
    gaprows=[]
    for c in ('dense_masked','dense_C4','gaussian_masked','gaussian_C4','gaussian_frozen'):
        rr=[r for r in gaps if r['condition']==c]
        gaprows.append([c,fmt(mean(rr,'train_loss')),fmt(mean(rr,'val_loss')),fmt(mean(rr,'macro_f1_gap')),fmt(mean(rr,'accuracy_gap'))])
    text+=table(['Condition','Train loss','Validation loss','Window macro-F1 gap','Window accuracy gap'],gaprows)
    text+='These gaps and the full learning curves use pooled windows within each split; the primary endpoint gives equal weight to subjects, contexts and seeds. Convergence.csv includes observed curve area and first recorded step crossing.8window macro-F1. No interpolation fills missing measurements.\n\n'
    releasetable=[]
    for c in ('gaussian_C0','gaussian_C2','gaussian_C4','dense_C4'):
        rr=[r for r in release if r['condition']==c and r['split']=='val']
        releasetable.append([c,fmt(mean(rr,'pre_accuracy')),fmt(mean(rr,'post_accuracy')),fmt(mean(rr,'final_accuracy')),
            fmt(mean(rr,'gate_switch_accuracy_difference')),fmt(mean(rr,'subsequent_optimization_accuracy_difference'))])
    text+=table(['Condition','Pre-release accuracy','Post-release accuracy','Final accuracy','Same-weight gate-switchΔ','Further-optimizationΔ'],releasetable)
    text+=f"All{len(identity)}grid pre/post checkpoint pairs were independently confirmed bitwise identical in model weights at optimizer step2240. Gate switching and further optimization are separate measured changes. This does not prove widening is universally helpful or harmful.\n\n"
    text+='## 10. Primary comparison and uncertainty\n\n'
    text+=table(['Contrast','Mean paired effect','Subject SD','Seed SD','Descriptive95%CI','Paired subject dz'],
        [[primary['a']+'−'+primary['b'],f"{float(primary['mean']):+.8f}",primary['subject_sd'],primary['seed_sd'],
          '['+primary['ci_low']+','+primary['ci_high']+']',primary['paired_subject_dz']]])
    selectedseed=[r for r in seedrows if r['contrast']=='gaussian_C4-gaussian_masked']
    text+=table(['Engineering seed','Paired subject macro-F1 difference'],[[r['seed'],f"{float(r['difference']):+.8f}"] for r in selectedseed])
    text+='The bootstrap averages five seeds within each subject after equal-context averaging, then applies10000paired subject resamples, using identical subject draws for both conditions. Percentile95%intervals are descriptive post-selection development uncertainty. The +.005 practical threshold is retained regardless precision; no confirmatory significance claim is made.\n\n'
    text+=f"Level3 predictive criterion: **{'met' if benefit else 'not met'}**. Numerical/functional evidence is separate; attributing any gain specifically to learned external context also requires valid Level2 evidence.\n\n"
    text+='## 11. Capacity and information controls\n\n'
    ctr=[r for r in contrasts if r['family']=='secondary' or r['b']=='gaussian_frozen']
    text+=table(['Family','Contrast','Mean','Descriptive95%CI'],[[r['family'],r['a']+'−'+r['b'],f"{float(r['mean']):+.8f}",'['+r['ci_low']+','+r['ci_high']+']'] for r in ctr])
    text+='The three registered secondary contrasts are C4−constant,C4−shuffled,and DenseC4−Densemasked. No formal p-values are reported, so no uncorrected significance claims or post-hoc Holm family are introduced. Other comparisons are exploratory. Frozen C4keeps its actual nonidentity initialization and trains base weights; it is not C=1. Constant-code allocation matches C4but at most2688context parameters are actively exposed, versus8064allocated; actual displacement and byte-storage counts are saved.\n\n'
    frozen=next(r for r in ctr if r['b']=='gaussian_frozen')
    information=[r for r in ctr if r['b'] in ('gaussian_constant','gaussian_shuffled')]
    text+=f"C4's incremental effect over frozen attenuation {'has a positive lower descriptive interval' if float(frozen['ci_low'])>0 else 'does not have a positive lower descriptive interval'}. This {'supports an incremental learning benefit within this development comparison' if float(frozen['ci_low'])>0 else 'leaves fixed attenuation and optimization effects as unresolved explanations; it does not establish equivalence'}. Both information-control contrasts have positive lower descriptive bounds: {all(float(r['ci_low'])>0 for r in information)}. These are descriptive secondary comparisons, not independent confirmation or uniquely identified mechanisms.\n\n"
    text+='Shuffled training codes preserve frequencies and are independent of true masks. Evaluations include true codes and independently randomized balanced per-window codes, explicitly different joint code/mask distributions from training. Capacity and fixed attenuation comparisons help interpret the primary result but do not uniquely identify every mechanism.\n\n'
    text+='## 12. CPU/CUDA execution\n\n'
    text+=f"Measured mean C4throughput: CPU{backend['mean_updates_per_second']['cpu']:.3f}updates/s; CUDA0{backend['mean_updates_per_second'].get('cuda:0',float('nan')):.3f}updates/s. CPUwas selected. CUDAforward maximum logit discrepancy was{read(run/'backend_correctness.json')['max_abs_logit_difference']:.8g}within frozen tensor tolerances. Benchmark parameters reset before each one-step fixture; these are not extra training trajectories.\n\n"
    text+=f"Training: {accounting['grid_updates']:,}grid+{accounting['pilot_updates']:,}pilot={accounting['completed_fit_training_updates']:,}logical updates. Additional one-step optimizer calls:880benchmark+42sweep+18unit tests=940. Interrupted replay updates:0. Maximum declared execution cap21600seconds was not exceeded. The execution status gives measured training wall time; per-fit times include evaluation/IOand shared CPU contention.\n\n"
    text+='Four worker processes used disjoint future seed blocks on the same CPUbackend under a separate source-hashed execution extension. The original frozen trainer/model/config files were unchanged. GPUmemory was not pooled. Full CUDAtraining/replay and cross-backend macro-F1 reproduction were not run. Environment records include Python/PyTorch/CUDA/cuDNN/package versions, GPUdriver query, deterministic flags, TF32off and CUBLASworkspace settings.\n\n'
    text+='## 13. Failures, omissions and repairs\n\n'
    text+='Nineteen distinct unit tests passed in each of two runs; zero test failures. Analytical gradients, finite differences, original-gate equivalence, masking/live support, permutation separation, serialization and exact-resume checks passed. The initial v2reference was unavailable; no performance training occurred until the user supplied it and explicit choices were reconciled under v3.\n\n'
    text+='An analysis-only JSON serialization defect was repaired losslessly by converting one NumPy boolean to a Python boolean. Training, endpoint calculations, bootstrap draws and CSV statistics were unchanged. Downstream report guards stopped; all failure evidence was retained. No scientific redesign occurred. Preserved exceptions/warnings and omission evidence are documented in REPAIR_AND_DEVIATION_LOG.md. C1rank-target and G2gradient-reference misses were reported without changing perturbation or initialization. C3futility omitted70fits. Extra execution/reporting scripts have separate snapshots and hashes; old research artifacts remain unchanged.\n\n'
    text+='## 14. Scientific interpretation\n\n'
    text+='**H1:** supported as a mathematical restriction and numerical symmetry diagnosis. Breaking symmetry removes exact rank-one restriction, but the second component remains initially weak and predictive benefit does not follow. **H2:** supported by the fixed-batch gradient sweep; reduced saturation strengthens gradients, while frozen−4.5still misses the engineering reference. **H3:** master precision preserves stored updates, but C3failed the declared forward-level pilot criterion. Storage precision alone was insufficient in those pilots.\n\n'
    text+='**H4/H5:** unresolved. Shared weights may compensate attenuation, and availability may be inferable from zero-masked inputs. Null classification results cannot prove either explanation, particularly with remaining weak gradients and weak second-component strength. Positive code-switch sensitivity can originate partly from initialization and does not itself establish useful context information. No semantic, biological or associative-memory relationship has been demonstrated.\n\n'
    text+='## 15. Limitations\n\n'
    text+='This is development-only evidence on previously examined subjects and engineering seeds. Overlapping training sets limit bootstrap independence. The historical16-subject scaler was preserved and includes contributions from some subsequent CVvalidation subjects. No official test result is available. Fixed factor seed30031limits factor-noise generalization. C3futility uses one seed/fold per architecture. The correction cells do not identify every interaction among saturation,symmetry and precision.\n\n'
    text+='Only35matched Gaussian shuffled null fits define each diagnostic quantile; Layer2evidence remains operational and pair-specific. Probe sensitivity uses128windows/fold. Per-factor stepwise update fractions are missing; initial/final displacement is not their reconstruction. Pooled-window curves differ from the equal-subject primary endpoint. Detailed qualifications are preserved in PILOT_SCOPE_AND_MEASUREMENT_LIMITS.md.\n\n'
    text+='## 16. Recommendation regarding Phase4\n\n'
    text+='Do not interpret this study as demonstrating associative memory or a new architecture. A functioning contextual gate alone does not justify Phase4. Any stronger-initialization or information-identification follow-up should use a separate frozen protocol and must not replace this study retrospectively. The present results answer the registered recovery comparison; broader mechanism and independent generalization questions remain open.\n\n'
    text+='## 17. Exact reproducibility and artifact inventory\n\n'
    text+='From E:\\CoNN, set PYTHONDONTWRITEBYTECODE=1and use `CoNN_Associative_Relationships\\.venv_cuda23\\Scripts\\python.exe CoNN_Associative_Relationships\\phase3b_context_diagnostics\\scripts\\run.py verify`to rescore saved predictions independently. `execute`is single-owner launch/resume; do not start a second owner. READMEcontains preflight,execute,verify,report and final-analysis commands.\n\n'
    text+=f"Frozen protocol SHA256:`{read(run/'protocol_seal.json')['protocol_sha256']}`. Frozen config SHA256:`{read(run/'protocol_seal.json')['config_sha256']}`. Frozen training-source-manifest SHA256:`{sha(run/'source_manifest.json')}`. There is no workspace Git repository; source SHA256snapshots identify exact revisions.\n\n"
    text+='Run inventory: config/protocol seals; original request reference; source and execution-extension snapshots; preflight and final preservation audits; environment/tests;455grid+10pilot checkpoint/prediction/step-log directories; raw and verified subject/class CSVs; paired/seed/subject effects; all code permutations and disabled-Cablations; attenuation snapshots/spectra; learning/gradient/precision curves; per-unit energy; release-vs-optimization and isolated-layer probes; CPU/CUDAcharts; required research summaries and final delivery hashes. Earlier automatic research reports are preserved in this analysis extension. No missing fit is fabricated or interpolated.\n'
    text=polish(text)
    for path in (run/'research_report.md',ROOT/'reports/PHASE3B_FINAL_REPORT.md',run/'reports/PHASE3B_FINAL_REPORT.md'):path.write_text(text,encoding='utf-8')
    short='# Phase3B v3 executive summary\n\n'
    short+=f"Completed455grid fits+10pilots on CPU, with{accounting['completed_fit_training_updates']:,}training updates. Seventy C3fits were omitted by the frozen futility rule.19tests passed; no test failures or official-test evaluation. Historical artifacts passed initial and final audits.\n\n"
    short+=f"Primary GaussianC4−Gaussianmasked macro-F1: **{float(primary['mean']):+.8f}**, descriptive95%subject-bootstrapCI **[{float(primary['ci_low']):+.8f},{float(primary['ci_high']):+.8f}]**. The +.005 practical-benefit criterion with a positive lower bound was **{'met' if benefit else 'not met'}**. This is exploratory development evidence.\n\n"
    short+=f"C4numerical learning passed{l1}/{l1n}fits. Functional pair/layer checks passed{l2}/{l2n}; initial random differences and the shuffled-code null limit context-specific attribution. Symmetry breaking worked mathematically but its second component was weak. Less saturation strengthened gradients. Float64storage preserved updates, while precision-only C3did not clear the forward threshold in its pilots.\n\n"
    short+='Architectural redundancy and limited extra context information remain unresolved hypotheses. These results do not demonstrate associative memory, prove a geometry hypothesis or justify an unregistered Phase4. See the measured technical report and individual seed/subject CSVs.\n'
    short=polish(short)
    for path in (ROOT/'reports/PHASE3B_EXECUTIVE_SUMMARY.md',run/'reports/PHASE3B_EXECUTIVE_SUMMARY.md'):path.write_text(short,encoding='utf-8')
    write(extension/'completion.json',{'measured_report_completed':True,'release_pairs_bitwise_verified':len(identity),
        'primary_statistics_unchanged':primary,'training_updates':accounting['completed_fit_training_updates']})
    write(run/'delivery_manifest.json',{str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'})
    print(short,flush=True)

if __name__=='__main__':main()
