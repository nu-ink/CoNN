"""Additive post-training diagnostics; never selects or changes any model.

Snapshot and hash this analysis extension before execution. Model training and
primary statistics remain in the original sealed source. This adds class,
capacity, convergence and isolated first-layer interventions and improves plot
comparability without modifying frozen trainer/model files.
"""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import itertools
import json
import shutil
import numpy as np
import torch
from torch.nn import functional as F
from common import read,write,sha,csv_write,csv_read
from models import Model,gates_for,configure,support
from conn_assoc.phase23_core import fold_arrays
from conn_assoc.phase3_context import mask_input
from verify import independent_scores

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=ROOT/'results/20261020_phase3b_v3_run01'
    manifest=read(run/'execution_manifest.frozen.json')
    completions=sorted((run/'grid').glob('*/completion.json'))
    assert len(completions)==manifest['full_grid_fits'],'No incomplete-grid endpoint analysis'
    extension=run/'execution_extensions/final_analysis_v1';extension.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,extension/'finalize_analysis.py')
    write(extension/'manifest.frozen.json',{'source_sha256':sha(__file__),'role':'additive post-training diagnostic reporting',
        'no_model_selection':True,'training_source_manifest_sha256':sha(run/'source_manifest.json'),
        'primary_analysis_unchanged':True,'plot_policy':'same initial/final color range within cell; separate delta panel'})
    configure('cpu')
    import platform,subprocess
    freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout
    (extension/'environment_packages.txt').write_text(freeze,encoding='utf-8')
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],capture_output=True,text=True)
    write(extension/'environment_details.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
        'torch':torch.__version__,'torch_build':torch.__config__.show(),'cuda_runtime':torch.version.cuda,
        'cudnn_version':torch.backends.cudnn.version(),'gpu_driver_query':gpu.stdout,'gpu_query_exit_code':gpu.returncode,
        'execution_backend':'cpu','torch_threads':torch.get_num_threads(),'deterministic':torch.are_deterministic_algorithms_enabled(),
        'git_revision':None,'git_reason':'workspace and project are not Git repositories; SHA256 source snapshots are authoritative'})
    cfg=read(run/'config.frozen.json');arrays={};classes=[];gaps=[];capacity=[];release=[];convergence=[];isolated=[];units=[];numerics=[];factorchanges=[];reproduction=[]
    omitted=[dict(condition=c,seed=s,fold=f,steps_not_executed=3200,reason='predeclared C3 pilot forward futility')
        for c in manifest['omitted_conditions'] for s in cfg['seeds'] for f in range(7)]
    csv_write(run/'analysis/omitted_fits.csv',omitted)
    for cp in completions:
        job=cp.parent;done=read(cp);meta={k:done[k] for k in ('condition','seed','fold')}
        scores=read(job/'final_metrics.json')['scores']
        with np.load(job/'predictions.npz') as p:
            for split in ('train','val'):
                for context in range(3):
                    m=independent_scores(p[split+'_truth'],p[f'{split}_context{context}'])
                    cm=np.asarray(m['confusion']);tp=cm.diagonal();actual=cm.sum(1);predicted=cm.sum(0)
                    for label in range(6):
                        precision=float(tp[label]/predicted[label]) if predicted[label] else 0.
                        recall=m['recall'][label]
                        classes.append(dict(**meta,split=split,context=context,label=label,truth_count=int(actual[label]),
                            predicted_count=int(predicted[label]),true_positive=int(tp[label]),precision=precision,recall=recall,
                            f1=2*precision*recall/(precision+recall) if precision+recall else 0.))
        for c in range(3):
            a=scores['train'][c];b=scores['val'][c]
            gaps.append(dict(**meta,context=c,train_accuracy=a['accuracy'],val_accuracy=b['accuracy'],
                accuracy_gap=a['accuracy']-b['accuracy'],train_macro_f1=a['macro_f1'],val_macro_f1=b['macro_f1'],
                macro_f1_gap=a['macro_f1']-b['macro_f1'],train_loss=a['loss'],val_loss=b['loss'],
                loss_gap=b['loss']-a['loss'],metric_unit='pooled windows within split, not primary subject endpoint'))
        history=read(job/'history.json')
        for c in range(3):
            rr=[r for r in history if r['split']=='val' and r['context']==c and r['kind'] in ('curve','initial','final')]
            xx=np.array([r['completed_steps'] for r in rr]);yy=np.array([r['macro_f1'] for r in rr])
            convergence.append(dict(**meta,context=c,normalized_curve_auc=float(np.trapezoid(yy,xx)/3200),
                first_curve_step_above_point8=next((r['completed_steps'] for r in rr if r['macro_f1']>=.8),None),
                final_curve_macro_f1=float(yy[-1])))
            for split in ('train','val'):
                pre=next(r for r in history if r['kind']=='pre_release' and r['context']==c and r['split']==split)
                post=next(r for r in history if r['kind']=='post_release' and r['context']==c and r['split']==split)
                final=next(r for r in history if r['kind']=='final' and r['context']==c and r['split']==split)
                release.append(dict(**meta,split=split,context=c,pre_accuracy=pre['accuracy'],post_accuracy=post['accuracy'],final_accuracy=final['accuracy'],
                    pre_loss=pre['loss'],post_loss=post['loss'],final_loss=final['loss'],
                    gate_switch_accuracy_difference=post['accuracy']-pre['accuracy'],
                    subsequent_optimization_accuracy_difference=final['accuracy']-post['accuracy'],
                    same_weights_for_pre_post=True))
        st=torch.load(job/'final.pt',map_location='cpu',weights_only=False)
        initial=torch.load(job/'initial.pt',map_location='cpu',weights_only=False)
        if done['condition'] in ('dense_C0','gaussian_C0','dense_masked','gaussian_masked'):
            from preflight import OLD
            arch='dense' if done['condition'].startswith('dense') else 'gaussian'
            priorcond=f'P3_{arch}_'+('context' if done['condition'].endswith('C0') else 'masked')
            priorjob=OLD/'jobs'/f"{priorcond}_seed{done['seed']}_fold{done['fold']}_lr0.1"
            prior=torch.load(priorjob/'final.pt',map_location='cpu',weights_only=False)
            tensors_equal=all(torch.equal(v,prior['model'][k]) for k,v in st['model'].items())
            with np.load(priorjob/'predictions.npz') as oldpred,np.load(job/'predictions.npz') as newpred:
                predictions_equal=all(np.array_equal(newpred[k],oldpred[k]) for k in newpred.files)
            prior_metrics=read(priorjob/'final_metrics.json')['scores']
            maxmetric=max(abs(scores[split][c]['macro_f1']-prior_metrics[split][c]['macro_f1']) for split in ('train','val') for c in range(3))
            reproduction.append(dict(**meta,historical_condition=priorcond,exact_final_tensors=tensors_equal,
                exact_saved_predictions=predictions_equal,max_macro_f1_difference=maxmetric,
                statistical_reproduction_within_point001=maxmetric<=.001,
                independent_confirmation=False))
        model=Model(done['condition'],done['seed']);model.load_state_dict(st['model'])
        active=0;allocated_bytes=0;trainable_bytes=0
        for name,p in model.named_parameters():
            allocated_bytes+=p.numel()*p.element_size()
            if p.requires_grad:trainable_bytes+=p.numel()*p.element_size()
            active+=int((p.detach()!=initial['model'][name]).sum())
            if name.startswith('context_'):
                old=initial['model'][name]
                for context in range(3):
                    delta=p[context].detach().double()-old[context].double()
                    stored_delta=p[context].detach().float()-old[context].float()
                    factorchanges.append(dict(**meta,factor=name,context=context,dtype=str(p.dtype),
                        changed_fraction=float((delta!=0).double().mean()),max_abs_parameter_change=float(delta.abs().max()),
                        update_l2=float(delta.norm()),fraction_changed_after_float32_parameter_cast=float((stored_delta!=0).double().mean()),
                        note='initial-to-final displacement; not a reconstruction of missing per-factor stepwise update fractions'))
        capacity.append(dict(**meta,allocated_parameters=done['allocated_parameters'],trainable_parameters=done['trainable_parameters'],
            allocated_parameter_bytes=allocated_bytes,trainable_parameter_bytes=trainable_bytes,
            parameters_changed_from_initial=active,constant_code_max_potential_active_parameters=81030 if done['condition']=='gaussian_constant' else None))
        d=read(job/'context_diagnostics.json')
        for row in d.get('changes',[]):numerics.append(dict(**meta,**row))
        if not hasattr(model,'context_ih'):continue
        if done['fold'] not in arrays:arrays[done['fold']]=fold_arrays(done['fold'],cfg['folds'])
        x=torch.from_numpy(arrays[done['fold']]['val']['x'][:128])
        gg=gates_for(done['condition'],done['seed'],'cpu').build(3199)
        with torch.no_grad():
            for layer,weight,gate,factor in [('ih',model.fc1.weight,gg[0],model.context_ih),('hh',model.fc2.weight,gg[1],model.context_hh)]:
                for c in range(3):
                    effective=weight*gate*factor(c)
                    for unit in range(64):units.append(dict(**meta,layer=layer,context=c,unit=unit,
                        gate_energy=float((gate[unit]*factor(c)[unit]).double().square().sum()),
                        effective_weight_l2=float(effective[unit].double().norm()),effective_weight_mean_abs=float(effective[unit].double().abs().mean())))
            for mask in range(3):
                input_fixed=mask_input(x,mask);logits=[];hiddens=[]
                for code in range(3):
                    h1=F.relu(F.linear(input_fixed,model.fc1.weight*gg[0]*model.context_ih(code),model.fc1.bias))
                    h2=F.relu(F.linear(h1,model.fc2.weight*gg[1]*model.context_hh(0),model.fc2.bias))
                    hiddens.append(h1);logits.append(F.linear(h2,model.out_weight,model.out_bias))
                for a,b in itertools.combinations(range(3),2):
                    delta=logits[a]-logits[b];hd=hiddens[a]-hiddens[b]
                    isolated.append(dict(**meta,true_mask=mask,a=a,b=b,layer='IH_only_HH_code_fixed0',
                        h1_max_abs_change=float(hd.abs().max()),h1_rms_change=float(hd.double().square().mean().sqrt()),
                        logit_max_abs_change=float(delta.abs().max()),logit_rms_change=float(delta.double().square().mean().sqrt()),
                        prediction_change_fraction=float((logits[a].argmax(1)!=logits[b].argmax(1)).double().mean())))
    for name,rows in [('class_metrics',classes),('generalization_gaps',gaps),('parameter_storage_and_active_capacity',capacity),
        ('release_vs_optimization',release),('convergence',convergence),('isolated_IH_interventions',isolated),
        ('per_unit_effective_gate_energy',units),('final_precision_diagnostics',numerics),('per_factor_initial_final_changes',factorchanges),
        ('historical_full_fit_reproduction',reproduction)]:csv_write(run/'analysis'/(name+'.csv'),rows)
    # Diagnostic plots with common scale per cell and explicit delta magnitude.
    fig,axes=plt.subplots(3,3,figsize=(13,8),layout='constrained')
    for col,cell in enumerate(('C0','C2','C4')):
        job=run/'pilots'/f'gaussian_{cell}_seed400_fold0'
        with np.load(job/'attenuation.npz') as p:
            a=p['ih_initial_0'][:,:128];b=p['ih_final_0'][:,:128]
            lo=min(a.min(),b.min());hi=max(a.max(),b.max())
            for row,data,name in [(0,a,'initial A'),(1,b,'final A')]:
                im=axes[row,col].imshow(data,aspect='auto',vmin=lo,vmax=hi)
                axes[row,col].set_title(cell+' '+name);fig.colorbar(im,ax=axes[row,col],format='%.6f')
            delta=b-a;lim=max(float(np.abs(delta).max()),1e-20)
            im=axes[2,col].imshow(delta,aspect='auto',vmin=-lim,vmax=lim,cmap='coolwarm')
            axes[2,col].set_title(cell+' ΔA, independent delta scale');fig.colorbar(im,ax=axes[2,col],format='%.1e')
    fig.suptitle('Gaussian pilot IH, context0, first128 sources; initial/final share scale within cell')
    fig.savefig(run/'plots/historical_revised_attenuation_comparable.png',dpi=150);plt.close(fig)
    # Save exact training-vs-diagnostic update accounting.
    accounting={'grid_fits':len(completions),'grid_updates':len(completions)*3200,'pilot_fits':10,'pilot_updates':32000,
        'completed_fit_training_updates':len(completions)*3200+32000,'benchmark_one_step_updates':880,
        'gradient_sweep_one_step_updates':42,'unit_test_optimizer_updates':18,
        'unit_test_accounting':'two full 19-test runs; three resume cases with three updates each per run',
        'interrupted_training_steps':0,'official_test_evaluations':0}
    write(run/'optimizer_accounting.json',accounting)
    # Preserve original preflight audit and add a separate end-of-run audit.
    from preflight import audit
    final_audit=run/'final_preservation_audit';final_audit.mkdir(exist_ok=True)
    audit(final_audit)
    tests=read(run/'tests.json');assert tests['exit_code']==0
    primary=read(run/'analysis_summary.json')['primary']
    add='\n\n## Additive final diagnostic verification\n\n'
    add+=f"All{len(completions)}grid fits completed; paired training counts are{accounting['completed_fit_training_updates']:,}optimizer updates including ten pilots. Additional one-step diagnostic calls:880backend+42sweep+18unit-test=940; no interrupted training replay. All earlier historical manifests and1050historical fit artifacts passed the separate end-of-run preservation audit. Source has no workspace Git revision available; frozen SHA256 snapshots identify the exact code.\n\n"
    add+='Additional CSVs independently record all class precision/recall/F1, train–validation gaps, validation curve area, same-weight release vs further optimization, actual parameter byte storage and changed parameter counts, final forward precision and per-unit contextual gate energy. Isolated IH interventions hold input, base weights and HH code0 fixed; HH fixed-h1 interventions remain separately recorded. The original independently autoscaled attenuation figure is retained as an early rendering; use historical_revised_attenuation_comparable.png for shared initial/final scales and explicit delta panels.\n\n'
    add+='Parallelization was an execution-only extension under a separately hashed driver, with all frozen training files unchanged and all comparisons on CPU. No additional models, selections, fits or updates were introduced. The independent final verifier checks checkpoint hash, actual next_step, step logs, exact pairing signatures, subject separation and all saved prediction metrics. Statistical intervals remain descriptive; no development comparison constitutes independent confirmation.\n'
    add+=f"\nHistorical full-fit reproduction across{len(reproduction)}new C0/masked fits: exact final tensors in{sum(r['exact_final_tensors'] for r in reproduction)}, exact saved predictions in{sum(r['exact_saved_predictions'] for r in reproduction)}, within the predeclared.001 macro-F1 statistical-reproduction tolerance in{sum(r['statistical_reproduction_within_point001'] for r in reproduction)}. These reuse the historical seeds and are reproducibility checks, not independent trials. Missing per-factor stepwise stored-update fractions cannot be reconstructed from scalar norms; only actual saved aggregate fractions and per-factor initial-to-final displacements are reported.\n"
    add+='\nLevel2 null comparisons are reported for Gaussian C4 because the predefined shuffled-training null is Gaussian. Dense functional diagnostics remain descriptive, rather than importing a mismatched-architecture null. The fixed-representation output diagnostics use the frozen first128validation windows per fold; they demonstrate probe sensitivity, not population-wide causal information utility. All70omitted C3 seed/fold entries are explicit in omitted_fits.csv.\n'
    add+='\nNineteen distinct unit tests passed in each of two runs, with zero failures. Full CUDA training/replay and cross-backend macro-F1 reproduction were not run: CPU was the selected full-grid backend. CUDA received the measured one-step throughput benchmark and a tensor-tolerance forward correctness check. Official-test evaluation was not run by design. Factor-initialization seed30031 is fixed across the five base-initialization seeds, so uncertainty does not average over alternative factor-noise realizations. Per-fit training times include evaluation/IO and varying shared CPU contention; they are measured wall times, not controlled architecture-latency comparisons.\n'
    for path in (run/'research_report.md',ROOT/'reports/PHASE3B_FINAL_REPORT.md'):
        with path.open('a',encoding='utf-8') as f:f.write(add)
    reportdir=run/'reports';reportdir.mkdir(exist_ok=True)
    for path in (ROOT/'reports').glob('*.md'):shutil.copy2(path,reportdir/path.name)
    for path in (ROOT/'README.md',ROOT/'PROTOCOL_PHASE3B_v3.md',ROOT/'PROTOCOL_PHASE3B_v2.md'):
        shutil.copy2(path,run/path.name)
    write(extension/'completion.json',{'analyses_generated':True,'optimizer_accounting':accounting,'historical_preservation_passed':True,
        'primary_analysis_unchanged':primary})
    # Final artifact manifest includes sources, plots, configs and reports, not its own hash.
    manifest_paths=[p for p in run.rglob('*') if p.is_file() and p.name!='delivery_manifest.json']
    write(run/'delivery_manifest.json',{str(p.relative_to(run)):sha(p) for p in manifest_paths})
    print(accounting,flush=True)

if __name__=='__main__':main()
