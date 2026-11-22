"""Measured-only development analysis and headless plots."""
from pathlib import Path
import math
import numpy as np
from common import read,write,csv_write,csv_read,sha
from models import ROOT,CONDITIONS

def bootstrap(values):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(7000031)
    means=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
    return np.quantile(means,[.025,.975])

def report(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from verify import verify_run
    run=Path(run);verification=verify_run(run)
    analysis=run/'analysis';plots=run/'plots';plots.mkdir(exist_ok=True)
    rows=csv_read(analysis/'verified_subject_scores.csv')
    grid=[r for r in rows if r['stage']=='grid']
    jobs=list((run/'grid').glob('*/completion.json')) if (run/'grid').exists() else []
    pilots=list((run/'pilots').glob('*/completion.json')) if (run/'pilots').exists() else []
    manifest=read(run/'execution_manifest.frozen.json') if (run/'execution_manifest.frozen.json').exists() else {}
    complete=len(jobs)==manifest.get('full_grid_fits',-1)
    scores={};subjectseed={};subjectcontext={}
    for row in grid:
        key=(row['condition'],int(row['subject']),int(row['seed']))
        subjectseed.setdefault(key,[]).append(float(row['macro_f1']))
        subjectcontext.setdefault((row['condition'],int(row['context'])),[]).append(float(row['macro_f1']))
    means={}
    for condition in CONDITIONS:
        vals=[np.mean(v) for (c,s,seed),v in subjectseed.items() if c==condition]
        if vals:means[condition]=float(np.mean(vals))
    contrast_specs=[('primary','gaussian_C4','gaussian_masked'),('secondary','gaussian_C4','gaussian_constant'),
        ('secondary','gaussian_C4','gaussian_shuffled'),('secondary','dense_C4','dense_masked')]
    contrast_specs += [('exploratory','gaussian_C4',f'gaussian_C{i}') for i in range(4)]
    contrast_specs += [('exploratory','gaussian_C4','gaussian_frozen')]
    contrasts=[];individual=[];seedrows=[]
    for family,a,b in contrast_specs:
        keys=sorted((s,seed) for c,s,seed in subjectseed if c==a and (b,s,seed) in subjectseed)
        if len(keys)!=105:continue # No partial-grid endpoint estimates.
        seed_effects=[];subject_effects=[]
        for s in sorted(set(s for s,seed in keys)):
            deltas=[np.mean(subjectseed[a,s,seed])-np.mean(subjectseed[b,s,seed]) for seed in range(400,405)]
            d=float(np.mean(deltas));subject_effects.append(d)
            individual.append({'contrast':a+'-'+b,'subject':s,'difference':d})
        for seed in range(400,405):
            d=float(np.mean([np.mean(subjectseed[a,s,seed])-np.mean(subjectseed[b,s,seed]) for s in sorted(set(s for s,se in keys))]))
            seed_effects.append(d);seedrows.append({'contrast':a+'-'+b,'seed':seed,'difference':d})
        lo,hi=bootstrap(subject_effects);sd=float(np.std(subject_effects,ddof=1));mean=float(np.mean(subject_effects))
        contrasts.append(dict(family=family,a=a,b=b,mean=mean,subject_sd=sd,seed_sd=float(np.std(seed_effects,ddof=1)),
            ci_low=float(lo),ci_high=float(hi),paired_subject_dz=mean/sd if sd else None,
            practical_threshold_reached=mean>=.005,descriptive_ci_excludes_zero_positive=lo>0))
    csv_write(analysis/'paired_contrasts.csv',contrasts);csv_write(analysis/'subject_differences.csv',individual);csv_write(analysis/'seed_differences.csv',seedrows)
    csv_write(analysis/'condition_scores.csv',[{'condition':c,'mean_subject_macro_f1':v} for c,v in means.items()])
    curves=[];grad=[];pair=[];inter=[];ranks=[];numerical=[];timings=[];capacity=[];perm=[]
    for cp in pilots+jobs:
        job=cp.parent;done=read(cp);meta={k:done[k] for k in ('condition','seed','fold')};meta['stage']=job.parent.name
        numerical.append(dict(**meta,level1=done['numerical_level1'],max_live_delta_c=done['max_live_forward_delta_c'],
            changed_fraction=done['max_changed_context_fraction']))
        timings.append(dict(**meta,seconds=done['elapsed_seconds']))
        capacity.append(dict(**meta,allocated=done['allocated_parameters'],trainable=done['trainable_parameters']))
        curves.extend(dict(**meta,**r) for r in read(job/'history.json'))
        for r in read(job/'gradients.json'):grad.append(dict(**meta,**{k:v for k,v in r.items() if k!='norms'}))
        d=read(job/'context_diagnostics.json')
        for r in d.get('pairs',[]):pair.append(dict(**meta,**r))
        for r in d.get('interventions',[]):inter.append(dict(**meta,**r))
        for r in d.get('rank',[]):
            r=dict(r);sv=r.pop('singular_values')
            ranks.append(dict(**meta,**r,sigma1=sv[0],sigma2=sv[1],sigma3=sv[2]))
        for item in read(job/'interventions.json'):
            for r in item['scores']:
                perm.append(dict(**meta,kind=item['kind'],permutation=str(item.get('permutation','')),**{k:v for k,v in r.items() if k!='confusion'}))
    for name,data in [('learning_curves',curves),('gradient_trajectories',grad),('live_support_pairs',pair),('fixed_representation_interventions',inter),
        ('attenuation_spectra',ranks),('numerical_learning',numerical),('training_time',timings),('capacity',capacity),('code_and_disable_interventions',perm)]:
        csv_write(analysis/(name+'.csv'),data)
    null={}
    for row in pair:
        if row['stage']=='grid' and row['condition']=='gaussian_shuffled' and row['normalized_frobenius'] is not None:
            null.setdefault((row['layer'],row['a'],row['b']),[]).append(row['normalized_frobenius'])
    levels=[]
    for row in pair:
        if row['stage']!='grid' or row['condition']!='gaussian_C4':continue
        key=(row['layer'],row['a'],row['b']);values=null.get(key,[])
        # Require all35 null fits before the frozen quantile is evaluated.
        if len(values)!=35:continue
        q=float(np.quantile(values,.95));n=row['normalized_frobenius']
        relevant=[r for r in inter if r['stage']=='grid' and r['condition']=='gaussian_C4' and r['seed']==row['seed'] and r['fold']==row['fold']
            and r['a']==row['a'] and r['b']==row['b'] and (r['layer']=='hh_fixed_h1' if row['layer']=='hh' else r['layer']=='same_mask_input_both_layers')]
        maximum=max([r['logit_max_abs'] for r in relevant]+[0])
        livechanged=next(r['max_live_delta_c'] for r in numerical if r['stage']=='grid' and r['condition']=='gaussian_C4' and r['seed']==row['seed'] and r['fold']==row['fold'])>=1e-6
        passed=n is not None and n>1e-4 and n>q and maximum>1e-6 and livechanged
        levels.append(dict(**row,null95=q,output_max_difference=maximum,live_forward_learning=livechanged,level2_pair_pass=passed))
    csv_write(analysis/'level2_null_comparisons.csv',levels)
    def save(name):
        plt.tight_layout();plt.savefig(plots/(name+'.png'),dpi=150);plt.close()
    sw=csv_read(analysis/'gradient_initialization_sweep.csv')
    plt.figure(figsize=(9,4))
    for arch in ('dense','gaussian'):
        for c in range(3):
            rr=[r for r in sw if r['architecture']==arch and int(r['context'])==c]
            plt.semilogy([float(r['u']) for r in rr],[float(r['gradient_ratio']) for r in rr],'-o',label=f'{arch} context{c}')
    plt.axhline(.001,color='black',ls='--',label='engineering reference');plt.xlabel('Initial U');plt.ylabel('Context/base gradient norm');plt.legend(fontsize=7);save('gradient_initialization_sweep')
    if curves:
        plt.figure(figsize=(11,5))
        source='grid' if jobs else 'pilots'
        for condition in CONDITIONS:
            rr=[r for r in curves if r['stage']==source and r['condition']==condition and r['split']=='val']
            if not rr:continue
            steps=sorted(set(r['completed_steps'] for r in rr))
            plt.plot(steps,[np.mean([r['macro_f1'] for r in rr if r['completed_steps']==s]) for s in steps],label=condition)
        plt.xlabel('Optimizer updates');plt.ylabel('Validation window macro-F1 (curve, not primary)');plt.legend(fontsize=6,ncol=3);save('learning_curves')
    if means:
        plt.figure(figsize=(11,5));names=list(means);plt.bar(names,[means[n] for n in names]);plt.xticks(rotation=65,ha='right',fontsize=8)
        plt.ylabel('Subject macro-F1');plt.title('Development grid' if complete else 'Incomplete grid: available fits only');save('development_performance')
        plt.figure(figsize=(9,4))
        for c in range(3):
            yy=[np.mean(subjectcontext.get((n,c),[np.nan])) for n in names]
            plt.plot(range(len(names)),yy,'o-',label=f'context{c}')
        plt.xticks(range(len(names)),names,rotation=65,ha='right',fontsize=7);plt.ylabel('Subject macro-F1');plt.legend();save('sensor_performance')
        selected=[n for n in names if n.startswith('gaussian') and (n.endswith('C4') or n.endswith(('masked','constant','shuffled','frozen')))]
        plt.figure(figsize=(8,4));plt.bar(selected,[means[n] for n in selected]);plt.xticks(rotation=30,ha='right');plt.ylabel('Subject macro-F1');save('capacity_information_controls')
    if contrasts:
        plt.figure(figsize=(9,5))
        for i,r in enumerate(contrasts):plt.errorbar(r['mean'],i,xerr=[[r['mean']-r['ci_low']],[r['ci_high']-r['mean']]],fmt='o')
        plt.yticks(range(len(contrasts)),[r['a']+' − '+r['b'] for r in contrasts],fontsize=8);plt.axvline(0,color='grey');plt.axvline(.005,color='green',ls='--');plt.xlabel('Paired subject macro-F1 difference; descriptive95% CI');save('paired_effect_intervals')
    if grad:
        plt.figure(figsize=(9,4))
        for condition in [f'gaussian_C{i}' for i in range(5)]:
            rr=[r for r in grad if r['condition']==condition and r['seed']==400 and r['fold']==0 and r['stage']=='pilots']
            if rr:plt.semilogy([r['step'] for r in rr],[max(r['gradient_ratio'],1e-16) for r in rr],label=condition)
        plt.xlabel('Step');plt.ylabel('Context/base gradient norm');plt.legend();save('gradient_norm_trajectories')
        plt.figure(figsize=(9,4))
        for condition in [f'gaussian_C{i}' for i in range(5)]:
            rr=[r for r in grad if r['condition']==condition and r['seed']==400 and r['fold']==0 and r['stage']=='pilots']
            if rr:plt.plot([r['step'] for r in rr],[r['active_update_fraction'] for r in rr],label=condition)
        plt.xlabel('Step');plt.ylabel('Active factor stored-update fraction');plt.legend();save('parameter_update_representability')
    if ranks:
        plt.figure(figsize=(9,4))
        for condition in [f'gaussian_C{i}' for i in range(5)]:
            rr=[r for r in ranks if r['condition']==condition and r['layer']=='ih' and r['stage']=='pilots']
            if rr:plt.semilogy([1,2,3],np.mean([[r['sigma1'],r['sigma2'],r['sigma3']] for r in rr],axis=0),'-o',label=condition)
        plt.xlabel('Singular-value index (float64 A=1−C)');plt.ylabel('Singular value');plt.legend();save('attenuation_singular_values')
    if inter:
        plt.figure(figsize=(9,4))
        rr=[r for r in inter if r['stage']=='pilots' and r['layer']=='hh_fixed_h1']
        names=sorted(set(r['condition'] for r in rr))
        plt.bar(names,[np.mean([r['logit_rms'] for r in rr if r['condition']==n]) for n in names]);plt.xticks(rotation=40,ha='right');plt.ylabel('Fixed-h1 code-switch logit RMS');save('same_hidden_state_interventions')
    if pair:
        plt.figure(figsize=(10,4))
        for layer in ('ih','hh'):
            rr=[r for r in pair if r['stage']=='pilots' and r['layer']==layer and r['normalized_frobenius'] is not None]
            names=sorted(set(r['condition'] for r in rr))
            plt.plot(names,[np.mean([r['normalized_frobenius'] for r in rr if r['condition']==n]) for n in names],'o-',label=layer)
        plt.xticks(rotation=40,ha='right');plt.ylabel('Live-intersection normalized A difference');plt.legend();save('external_context_relationships')
    benchmarks=csv_read(analysis/'backend_benchmark.csv') if (analysis/'backend_benchmark.csv').exists() else []
    if benchmarks:
        plt.figure(figsize=(7,4));plt.bar([r['device']+' '+r['condition'] for r in benchmarks],[float(r['updates_per_second']) for r in benchmarks]);plt.xticks(rotation=30,ha='right');plt.ylabel('One-step diagnostic updates/s');save('cpu_cuda_throughput')
    if pilots:
        fig,axes=plt.subplots(2,3,figsize=(11,5))
        for i,cell in enumerate(('C0','C2','C4')):
            job=run/'pilots'/f'gaussian_{cell}_seed400_fold0'
            if (job/'attenuation.npz').exists():
                with np.load(job/'attenuation.npz') as p:
                    for j,phase in enumerate(('initial','final')):
                        im=axes[j,i].imshow(p[f'ih_{phase}_0'][:,:128],aspect='auto');axes[j,i].set_title(f'{cell} {phase} A, first128 sources');fig.colorbar(im,ax=axes[j,i])
        save('historical_revised_attenuation')
        plt.figure(figsize=(9,4))
        for cell in ('C0','C1','C2','C3','C4'):
            job=run/'pilots'/f'gaussian_{cell}_seed400_fold0'
            if not (job/'diagnostics.json').exists():continue
            rr=[r for r in read(job/'diagnostics.json') if 'float64_max_delta_c' in r]
            xx=sorted(set(r['completed_steps'] for r in rr))
            plt.plot(xx,[max(r['float64_max_delta_c'] for r in rr if r['completed_steps']==s) for s in xx],label=cell+' f64 ref')
            plt.plot(xx,[max(r['live_max_delta_c'] for r in rr if r['completed_steps']==s) for s in xx],ls='--',label=cell+' forward')
        plt.axhline(1e-6,color='black',ls=':');plt.xlabel('Steps');plt.ylabel('Max live ΔC from init');plt.legend(fontsize=6,ncol=2);save('float32_float64_forward_changes')
    primary=next((r for r in contrasts if r['family']=='primary'),None)
    c4levels=[r for r in numerical if r['stage']=='grid' and r['condition']=='gaussian_C4']
    l1=sum(r['level1'] for r in c4levels)
    l2=sum(r['level2_pair_pass'] for r in levels)
    level3=bool(primary and primary['mean']>=.005 and primary['ci_low']>0)
    backend=read(run/'backend_selection.json') if (run/'backend_selection.json').exists() else {}
    tests=read(run/'tests.json') if (run/'tests.json').exists() else {}
    text='# Phase 3B v3 final research report\n\n'
    text+=f"Status: {'completed development grid' if complete else 'partial; grid incomplete'}. Verified {len(jobs)} grid fits and {len(pilots)} pilots; {verification['verified_logical_updates']:,} completed-fit optimizer updates. Backend: {backend.get('device','not selected')}. Official-test model evaluations:0. Passing numerical tests do not establish successful training.\n\n"
    text+='## Research question and design\n\nCan repairing weak context-factor learning yield useful external-context behavior? Fixed LR.1, architecture1152–64–64–6,3200 SGD updates, seeds400–404, seven subject-group folds; 15 conditions and up to10 pilots. Read the frozen protocol/config/source and execution manifests for exact definitions. No validation search. C3 omissions are frozen after pilot numerical futility, never substituted.\n\n'
    text+=f"Execution manifest: `{manifest}`. Tests: {tests.get('stdout','not run').strip()}.\n\n"
    text+='## Historical comparison and precision\n\nAll1050 historical fits and5762 historical source/data checks passed at preflight. C0 checkpoint replays exactly matched logits and saved predictions in six architecture/context fixtures. The original Gaussian gate was bitwise equivalent on tested schedule positions. The earlier Phase23 verifier imported trainer metric routines; this new verifier does not. Earlier results remain historical, not new replications. Historical Gaussian context effect −.00041791, subject SD .00520815; the new historical bootstrap uses its own frozen random seed, so interval endpoints need not reproduce the earlier Monte Carlo realization. All21 subjects contain all6 classes, making supported-class and six-class endpoints identical. See HISTORICAL_PRECISION_ESTIMATE.md.\n\n'
    text+='## Mathematical, symmetry and precision findings\n\nAnalytic loss-gradient contractions agree with autograd to1.39e-17 in the fixed float64 fixture; central differences passed. The42-fixture gradient-only sweep used no validation scores. U=-4.5 failed the1e-3 engineering ratio in6/6 fixtures and had zero maximum one-step float32 ΔC. U=-1 reached the ratio in5/6 fixtures, but did not replace the frozen initialization. C1 broke symmetry and produced numerical rank2; its sigma2/sigma1 remained below.01 at initialization, a limitation retained as required. See initial_rank.csv, gradient sweep report and measured trajectories.\n\n'
    text+=f"Gaussian C4 Level1 passed in {l1}/{len(c4levels)} completed grid fits. Level2 pair/layer criteria passed in {l2}/{len(levels)} eligible comparisons against the fully available shuffled-code null; detailed results include undefined empty IH intersections. Initial random gate differences are reported separately from learned changes. Pair-level evidence does not imply all contexts learned distinct useful relationships.\n\n"
    text+='## Classification and paired uncertainty\n\n'
    if primary:
        text+=f"Primary GaussianC4−Gaussianmasked: {primary['mean']:+.8f} macro-F1; subject SD {primary['subject_sd']:.8f}, seed SD {primary['seed_sd']:.8f}; descriptive95% subject-bootstrapCI [{primary['ci_low']:+.8f}, {primary['ci_high']:+.8f}]; paired subject dz {primary['paired_subject_dz']:.5f}. +.005 threshold reached: {primary['practical_threshold_reached']}. Level3 predictive threshold achieved: {level3}. Contextual attribution also requires valid Level2; random initialization, fixed attenuation and capacity controls limit interpretation.\n\n"
    else:text+='Primary endpoint not estimated: all paired subject/seed fits are required. Missing fits are not interpolated.\n\n'
    for r in contrasts:
        if r['family']=='primary':continue
        text+=f"{r['family']}: {r['a']}−{r['b']} {r['mean']:+.8f}, descriptive95%CI [{r['ci_low']:+.8f},{r['ci_high']:+.8f}].\n\n"
    text+='No formal hypothesis tests or uncorrected significance claims. Secondary family and exploratory contrasts remain separate. See individual seed and subject CSVs, supported-class scores, confusion verification and sensor-specific results. Intervals resample21subjects together across all5seeds after equal-context averaging. Overlapping CV training sets violate simple independent-fit interpretations; all outcomes are exploratory development evidence.\n\n'
    text+='## Capacity, interventions and convergence\n\nFrozen C4 retains nonidentity attenuation; it isolates initial weighting from learned gate changes. Constant-code preserves allocated capacity but trains fewer lookup rows. Allocation and trainable counts are separate. All five nonidentity code permutations retain true sensor masks. Random-code evaluation of the shuffled-training control uses independent balanced per-window codes and is a distribution shift; correctly matched evaluation differs from its training code distribution. Disabled-C, fixed-h1, same-masked-input, release pre/post and full learning curves are saved. Same-weight interventions are diagnostic, not realistic changed sensor availability. Training/validation losses and final gaps are available in raw curves and scores; gate-switch pre/post uses identical weights, separating gate release from further optimization.\n\n'
    text+='## Scientific interpretation and limitations\n\nH1 is supported mathematically: identical component columns constrain A to rank≤1; perturbations break this restriction but the weak second singular component may remain practically small. H2 is supported by the fixed-batch saturation sweep, without establishing a predictive remedy. H3 must be assessed at each level: gradient, stored update, reference C, float32 C and logits; float64 storage alone does not establish forward learning. H4 compensation and H5 context recoverability remain unresolved explanations, even if predictive effects are null. Positive fixed-code or frozen-gate effects would also limit information-specific claims.\n\n'
    text+='The same development subjects, engineering seeds and historical scaler have been examined before. The16-subject scaler contributes to preprocessing of some later CV validation subjects; it was preserved for historical comparability and is a limitation. Only21 subject clusters constrain precision; five seeds do not guarantee power. No official test evaluation, independent confirmation, semantic relationship learning, new architecture or associative memory is demonstrated. Phase4 is not implemented; further associative-memory research requires a separate question and protocol, rather than being warranted solely by functioning gates.\n\n'
    text+='## Execution, integrity and reproducibility\n\n'
    text+=f"Backend benchmark: {backend}. Completed fits are independently rescored and checkpoint/file hashes verified. Actual checkpoint next_step, contiguous step logs, base-init/batch/mask pairing and subject disjointness are checked. Same-backend checkpoint/resume tests require exact tensors and predictions. Cross-backend tolerance checks are separately labeled. Fixed-work benchmark one-step fixtures reset parameters each time and are not full training trajectories; their actual optimizer calls are recorded separately.\n\n"
    text+='Commands (from E:\\CoNN, PYTHONDONTWRITEBYTECODE=1): use the isolated `.venv_cuda23\\Scripts\\python.exe` with `phase3b_context_diagnostics\\scripts\\run.py preflight`, `execute` (also resume), `verify`, `report`. Read README.md for exact full paths. Source snapshots, SHA manifests, frozen configs, environment, raw prediction NPZ, subject/class CSV, checkpoints, curves, all code permutations and plots are run-specific. Repairs and omissions are documented in REPAIR_AND_DEVIATION_LOG.md.\n'
    for path in (run/'research_report.md',ROOT/'reports/PHASE3B_FINAL_REPORT.md'):path.write_text(text,encoding='utf-8')
    short=f"# Phase3B v3 executive summary\n\n{'Completed' if complete else 'Incomplete'}: {len(jobs)} grid fits and {len(pilots)} pilot fits, {verification['verified_logical_updates']:,} verified completed-fit updates on {backend.get('device','unselected')}. No official test subjects were evaluated.\n\n"
    short+=(f"GaussianC4−masked primary macro-F1 difference {primary['mean']:+.8f}, descriptive95%CI [{primary['ci_low']:+.8f},{primary['ci_high']:+.8f}]. Practical +.005 threshold and positive lower interval: {level3}.\n\n" if primary else 'No complete primary comparison is available.\n\n')
    short+=f"C4 numerical learning: {l1}/{len(c4levels)} fits; functional pair/layer tests: {l2}/{len(levels)} passed. Symmetry breaking worked numerically, but the added rank component was initially below the diagnostic target. Less saturation strengthened fixed-batch gradients; float64 storage must still pass the float32 forward criterion.\n\nThese development measurements do not establish associative memory or prove architectural redundancy or limited contextual information. Historical experiments were preserved. See the final report, exact fit records and diagnostic CSVs.\n"
    (ROOT/'reports/PHASE3B_EXECUTIVE_SUMMARY.md').write_text(short,encoding='utf-8')
    write(run/'report_manifest.json',{str(p.relative_to(run)):sha(p) for folder in (analysis,plots) for p in folder.iterdir() if p.is_file()})
    write(run/'analysis_summary.json',{'complete':complete,'grid_fits':len(jobs),'pilot_fits':len(pilots),'verification':verification,
        'primary':primary,'gaussian_C4_level1_fits':l1,'gaussian_C4_level1_denominator':len(c4levels),
        'level2_passed_comparisons':l2,'level2_denominator':len(levels),'level3_predictive_criterion':level3})
    print(short,flush=True)
