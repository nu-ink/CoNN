import json
import math
import platform
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from common import read,write,sha,csv_write
from models import ROOT,PROJECT,Model,gates_for,configure,support,parameter_groups,factor_reference,PERMUTATIONS
from conn_assoc.phase23_core import fold_arrays,paired_batches,context_plan
from conn_assoc.data import verify_archive,verify_prepared_manifest
from verify import independent_scores

OLD=PROJECT/'results/runs/20261019T030442340025Z_phase23_v1'

def audit(run):
    checks=[]
    for name,base in [('delivery_source_manifest.json',PROJECT),('prior_associative_manifest.json',PROJECT),('experiment2.before.json',None)]:
        for rel,record in read(OLD/name).items():
            path=Path(rel) if base is None else base/rel
            expected=record['sha256'] if isinstance(record,dict) else record
            if sha(path)!=expected: raise RuntimeError(f'Historical hash mismatch: {path}')
            checks.append({'path':str(path),'sha256':expected})
    verify_archive();verify_prepared_manifest()
    # Every historical fit artifact is verified against its original completion.
    jobs=0;files=0
    for cp in OLD.glob('jobs/*/completion.json'):
        for name,digest in read(cp)['files'].items():
            if sha(cp.parent/name)!=digest: raise RuntimeError(f'Historical fit changed: {cp.parent/name}')
            files+=1
        jobs+=1
    write(run/'historical_integrity.json',{'checks':checks,'historical_fits':jobs,'historical_job_files':files,'archive_sha256':verify_archive(),'passed':True})
    return len(checks),jobs

def historical_precision(run):
    conds=['P3_dense_context','P3_dense_masked','P3_gaussian_context','P3_gaussian_masked']
    scores=[];classsets={};recomputed=0
    for condition in conds:
        for seed in range(400,405):
            for fold in range(7):
                job=OLD/'jobs'/f'{condition}_seed{seed}_fold{fold}_lr0.1'
                with np.load(job/'predictions.npz') as p:
                    for subject in np.unique(p['val_subject']):
                        ix=p['val_subject']==subject;truth=p['val_truth'][ix]
                        classsets[int(subject)]=np.unique(truth).tolist()
                        for c in range(3):
                            m=independent_scores(truth,p[f'val_context{c}'][ix]);recomputed+=len(truth)
                            scores.append(dict(condition=condition,seed=seed,fold=fold,subject=int(subject),context=c,
                                macro_f1=m['macro_f1'],supported_macro_f1=m['supported_macro_f1']))
    csv_write(run/'analysis/historical_subject_scores.csv',scores)
    results=[]
    for arch in ('dense','gaussian'):
        grouped={}
        for row in scores:
            if row['condition'].startswith('P3_'+arch):
                k=(row['subject'],row['seed'],row['condition']);grouped.setdefault(k,[]).append(row['macro_f1'])
        values=[];seed_values=[]
        subjects=sorted(classsets)
        for s in subjects:
            values.append(np.mean([np.mean(grouped[s,seed,f'P3_{arch}_context'])-np.mean(grouped[s,seed,f'P3_{arch}_masked']) for seed in range(400,405)]))
        for seed in range(400,405):
            seed_values.append(np.mean([np.mean(grouped[s,seed,f'P3_{arch}_context'])-np.mean(grouped[s,seed,f'P3_{arch}_masked']) for s in subjects]))
        values=np.array(values);rng=np.random.default_rng(7000031)
        boot=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
        lo,hi=np.quantile(boot,[.025,.975]);sd=values.std(ddof=1)
        results.append(dict(architecture=arch,mean=float(values.mean()),subject_sd=float(sd),seed_sd=float(np.std(seed_values,ddof=1)),
            ci_low=float(lo),ci_high=float(hi),bootstrap_width=float(hi-lo),
            approximate_zero_exclusion_effect=float(1.96*sd/np.sqrt(21)),
            approximate_80percent_normal_power_effect=float((1.96+.841621)*sd/np.sqrt(21))))
    write(run/'historical_precision.json',{'results':results,'subject_supported_classes':classsets,'rescored_prediction_rows':recomputed})
    text='# Historical precision estimate\n\nExisting Phase3 predictions only; no new training or validation candidate selection. Independent confusion calculations were used.\n\n'
    for r in results: text+=f"{r['architecture']}: mean {r['mean']:+.8f}; paired subject SD {r['subject_sd']:.8f}; seed SD {r['seed_sd']:.8f}; descriptive bootstrap 95% CI [{r['ci_low']:+.8f}, {r['ci_high']:+.8f}], width {r['bootstrap_width']:.8f}. Approximate normal zero-exclusion threshold {r['approximate_zero_exclusion_effect']:.8f}; rough 80% normal-power effect {r['approximate_80percent_normal_power_effect']:.8f}. These are conditional engineering estimates, not a prospective power guarantee.\n\n"
    text+=f'All 21 subjects have all six ground-truth classes: {all(len(v)==6 for v in classsets.values())}. Thus supported-class and historical six-class macro-F1 coincide here. The +.005 endpoint is retained. Five seeds do not create additional independent subjects; overlapping CV training sets, historical selection and unknown future variance limit precision.\n'
    (ROOT/'reports/HISTORICAL_PRECISION_ESTIMATE.md').write_text(text,encoding='utf-8')

def mathematical_tests():
    g=torch.Generator().manual_seed(112233)
    u=torch.randn(3,2,generator=g,dtype=torch.float64,requires_grad=True)
    v=torch.randn(4,2,generator=g,dtype=torch.float64,requires_grad=True)
    h=torch.randn(3,4,generator=g,dtype=torch.float64)
    fn=lambda a,b: 1-.475*(a.sigmoid()@b.sigmoid().T)
    loss=(fn(u,v)*h).sum();gu,gv=torch.autograd.grad(loss,(u,v))
    p=u.sigmoid();q=v.sigmoid()
    au=-.475*(h@q)*p*(1-p);av=-.475*(h.T@p)*q*(1-q)
    assert torch.allclose(gu,au,rtol=1e-5,atol=1e-8)
    assert torch.allclose(gv,av,rtol=1e-5,atol=1e-8)
    assert torch.autograd.gradcheck(fn,(u,v),eps=1e-5,atol=1e-8,rtol=1e-5)
    assert not (support(1)&support(2)).any()
    assert (support(0)&support(1)).sum()==768
    assert (support(0)&support(2)).sum()==384
    assert len(set(PERMUTATIONS))==5 and all(sorted(x)==[0,1,2] for x in PERMUTATIONS)
    return {'analytic_autograd_max_error':max(float((gu-au).abs().max()),float((gv-av).abs().max())),
        'central_finite_difference_passed':True,'live_support_passed':True,'permutations_passed':True}

def sweep(run,arrays):
    rows=[];configure('cpu')
    ix=paired_batches(len(arrays['train']['y']),3200,128,400)[0]
    x=torch.from_numpy(arrays['train']['x'][ix]);y=torch.tensor(arrays['train']['y'][ix],dtype=torch.long)
    for arch in ('dense','gaussian'):
        gates=gates_for(arch+'_C0',400,'cpu').build(0)
        for gi,uvalue in enumerate([-6.,-5.,-4.5,-4.,-3.,-2.,-1.]):
            for context in range(3):
                model=Model(arch+'_C0')
                with torch.no_grad():
                    for factor in (model.context_ih,model.context_hh): factor.u.fill_(uvalue)
                initial={n:p.detach().clone() for n,p in model.named_parameters()}
                cs={layer:getattr(model,'context_'+layer)(context).detach().clone() for layer in ('ih','hh')}
                before=model.run(x,gates,context).detach()
                F.cross_entropy(model.run(x,gates,context),y).backward()
                base,ctx=parameter_groups(model)
                bn=math.sqrt(sum(float(p.grad.double().square().sum()) for p in base))
                cn=math.sqrt(sum(float(p.grad.double().square().sum()) for p in ctx))
                ideal=math.sqrt(sum(float((p.grad.double()*.1).square().sum()) for p in ctx))
                torch.optim.SGD(model.parameters(),lr=.1,foreach=False).step()
                changed=sum(int((p!=initial[n]).sum()) for n,p in model.named_parameters() if n.startswith('context_'))
                # Count active selected context only: unused lookup rows intentionally unchanged.
                active=sum(p[context].numel() for p in ctx)
                maxforward=0;maxref=0
                for layer in ('ih','hh'):
                    fac=getattr(model,'context_'+layer);live=support(context,layer)
                    maxforward=max(maxforward,float((fac(context)-cs[layer])[:,live].abs().max()))
                    oldc=1-.475*(initial[f'context_{layer}.u'][context].double().sigmoid()@initial[f'context_{layer}.v'][context].double().sigmoid().T)
                    maxref=max(maxref,float((factor_reference(fac,context)-oldc)[:,live].abs().max()))
                sig=1/(1+math.exp(-uvalue))
                after=model.run(x,gates,context).detach()
                # Separate gate effect from simultaneous base optimization.
                for n,p in model.named_parameters():
                    if not n.startswith('context_'):
                        with torch.no_grad(): p.copy_(initial[n])
                gateonly=model.run(x,gates,context).detach()
                rows.append(dict(candidate=f'G{gi}',architecture=arch,context=context,u=uvalue,attenuation=.475*sig,mean_c=1-.475*sig,
                    local_dA_dU=.475*sig*(1-sig)*.5,local_dA_dV=.475*sig*.25,context_gradient_l2=cn,base_gradient_l2=bn,
                    gradient_ratio=cn/bn,ratio_reference_reached=cn/bn>=1e-3,ideal_context_update_l2=ideal,
                    active_parameter_update_fraction=changed/active,forward_max_delta_c=maxforward,float64_max_delta_c=maxref,
                    gate_only_logit_max_delta=float((gateonly-before).abs().max()),combined_step_logit_max_delta=float((after-before).abs().max())))
    csv_write(run/'analysis/gradient_initialization_sweep.csv',rows)
    text='# Gradient-only initialization sweep\n\nFixed seed400, fold0, first historical paired training batch; initial weights and batches identical across candidates. Dense/Gaussian step0 gates, all contexts. No validation scores used. Local derivatives differ from loss gradients. Parameter fractions use the active lookup row; inactive contexts are not counted as rounding failures. Raw CSV contains all42 measurements.\n\n'
    for name in [f'G{i}' for i in range(7)]:
        rr=[r for r in rows if r['candidate']==name]
        text+=f"{name} U={rr[0]['u']}: gradient ratio range {min(r['gradient_ratio'] for r in rr):.6g}–{max(r['gradient_ratio'] for r in rr):.6g}; reaches1e-3 in {sum(r['ratio_reference_reached'] for r in rr)}/6 fixtures; live forward ΔC maximum {max(r['forward_max_delta_c'] for r in rr):.6g}.\n\n"
    text+='C2/C4 remain U=-4.5. No candidate is selected from this sweep. One-step logit effects distinguish gate-only changes from concurrent weight optimization. Stronger initializations, if desired, require a separate amendment.\n'
    (ROOT/'reports/GRADIENT_INITIALIZATION_SWEEP.md').write_text(text,encoding='utf-8')

def gates_and_replay(run,arrays):
    from conn_assoc.phase23_core import make_model,forward
    from conn_exp.gates import gaussian_gate
    testg=gates_for('gaussian_C0',400,'cpu')
    import math
    for step in (0,123,2239):
        u=step/2239;start=.08/math.sqrt(2*math.log(2));end=1/math.sqrt(2*math.log(2))
        tau=start+(end-start)*(u*u*(3-2*u))
        if step==0:tau=start
        if step==2239:tau=end
        a,b=testg.build(step)
        assert torch.equal(a,gaussian_gate(testg.di,tau)[0])
        assert torch.equal(b,gaussian_gate(testg.dh,tau)[0])
    x=torch.from_numpy(arrays['val']['x'][:512])
    errors=[]
    for arch in ('dense','gaussian'):
        oldcondition=f'P3_{arch}_context'
        job=OLD/'jobs'/f'{oldcondition}_seed400_fold0_lr0.1'
        historical=make_model(oldcondition,400)
        state=torch.load(job/'final.pt',map_location='cpu',weights_only=False)
        historical.load_state_dict(state['model'])
        new=Model(arch+'_C0');new.load_state_dict(state['model'])
        g=gates_for(arch+'_C0',400,'cpu');g.current=g.build(3199)
        for c in range(3):
            with torch.no_grad(): a=forward(historical,x,g,c);b=new.run(x,g.current,c)
            assert torch.equal(a,b),'Historical C0 logits differ'
            with np.load(job/'predictions.npz') as pred: assert np.array_equal(b.argmax(1).numpy(),pred[f'val_context{c}'][:len(x)])
            errors.append({'architecture':arch,'context':c,'exact_logits':True,'exact_saved_predictions':True})
    write(run/'historical_checkpoint_replay.json',errors)
    return {'original_gaussian_equivalence':True,'historical_C0_replays':len(errors)}

def run_preflight(run):
    run=Path(run);run.mkdir(parents=True,exist_ok=True);(ROOT/'reports').mkdir(exist_ok=True)
    started=time.monotonic()
    write(run/'stage.json',{'stage':'audit','status':'running'})
    counts=audit(run)
    historical_precision(run)
    configure('cpu');arrays=fold_arrays(0,read(OLD/'config.frozen.json')['folds'])
    tests=mathematical_tests();tests.update(gates_and_replay(run,arrays))
    sweep(run,arrays)
    ranks=[]
    for cell in range(5):
        model=Model(f'gaussian_C{cell}')
        for layer in ('ih','hh'):
            factor=getattr(model,'context_'+layer)
            for c in range(3):
                a=1-factor_reference(factor,c);sv=torch.linalg.svdvals(a.detach())
                ranks.append({'cell':f'C{cell}','layer':layer,'context':c,'mean_c':float((1-a).mean()),
                    'sigma1':float(sv[0]),'sigma2':float(sv[1]),'ratio':float(sv[1]/sv[0]),'rank':int((sv>sv[0]*1e-10).sum())})
    csv_write(run/'analysis/initial_rank.csv',ranks)
    c0=next(r['mean_c'] for r in ranks if r['cell']=='C0')
    assert all(abs(r['mean_c']-c0)<.0005 for r in ranks if r['cell']=='C1')
    tests.update({'C1_mean_constraint_passed':True,'initial_rank_measured':True,'historical_hash_checks':counts[0],
        'historical_fits_verified':counts[1],'elapsed_seconds':time.monotonic()-started})
    write(run/'preflight.json',tests)
    write(run/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),'torch':torch.__version__,
        'numpy':np.__version__,'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
        'devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'threads':torch.get_num_threads(),
        'tf32':False,'cublas_workspace_config':':4096:8','date_basis':'user context 2026-10-20 America/Chicago'})
    write(run/'stage.json',{'stage':'preflight','status':'passed'})
    return tests
