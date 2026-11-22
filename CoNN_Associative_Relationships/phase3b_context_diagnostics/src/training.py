import copy
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch.nn import functional as F
import yaml
from common import sha,write,read,replace,csv_write
from models import ROOT,PROJECT,CONDITIONS,PERMUTATIONS,Model,gates_for,configure,parameter_groups
from conn_assoc.phase23_core import fold_arrays,paired_batches,context_plan,fingerprint
from diagnostics import snapshot,changes,final_diagnostics
from evaluation import evaluate

def checkpoint(path,state):
    temp=path.with_suffix('.tmp');torch.save(state,temp);replace(temp,path)

def seal_sources(run):
    files=[ROOT/'PROTOCOL_PHASE3B_v3.md',ROOT/'PROTOCOL_PHASE3B_v2.md',ROOT/'config/experiment.yaml']
    for folder in ('src','scripts','tests'):files.extend(sorted((ROOT/folder).glob('*.py')))
    manifest={str(p.relative_to(ROOT)):sha(p) for p in files}
    for p in files:
        dst=run/'source'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    write(run/'source_manifest.json',manifest)
    cfg=yaml.safe_load((ROOT/'config/experiment.yaml').read_text())
    write(run/'config.frozen.json',cfg)
    write(run/'protocol_seal.json',{'protocol_sha256':sha(ROOT/'PROTOCOL_PHASE3B_v3.md'),
        'config_sha256':sha(run/'config.frozen.json'),'source_manifest_sha256':sha(run/'source_manifest.json')})

def check_seal(run):
    for rel,digest in read(run/'source_manifest.json').items():
        if sha(ROOT/rel)!=digest:raise RuntimeError(f'Source changed after seal: {rel}')
    s=read(run/'protocol_seal.json')
    assert sha(run/'config.frozen.json')==s['config_sha256']
    assert sha(run/'source_manifest.json')==s['source_manifest_sha256']

def tensors(arrays,device):
    return {key:dict(a,xt=torch.from_numpy(a['x']).to(device),yt=torch.tensor(a['y'],dtype=torch.long,device=device)) for key,a in arrays.items()}

def benchmark(run,cfg):
    arrays=fold_arrays(0,cfg['folds']);rows=[]
    devices=['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])
    for device in devices:
        configure(device);data=tensors(arrays,device)
        for condition in ('dense_C4','gaussian_C4'):
            model=Model(condition,400,device);g=gates_for(condition,400,device)
            optimizer=torch.optim.SGD(model.parameters(),lr=.1,foreach=False)
            initial_state=copy.deepcopy(model.state_dict())
            batch=paired_batches(len(data['train']['y']),220,128,400)
            measured=0.
            for step in range(220):
                # Independent one-step fixtures, never an extra trained trajectory.
                model.load_state_dict(initial_state)
                if device!='cpu':torch.cuda.synchronize()
                tick=time.perf_counter()
                ix=torch.as_tensor(batch[step],device=device)
                optimizer.zero_grad(set_to_none=True)
                loss=F.cross_entropy(model.run(data['train']['xt'][ix],g.build(step),step%3),data['train']['yt'][ix])
                loss.backward();optimizer.step()
                if device!='cpu':torch.cuda.synchronize()
                if step>=20:measured+=time.perf_counter()-tick
            sec=measured
            # Benchmark is fixed-work disposable throughput, not a pilot trajectory/score.
            rows.append({'device':device,'condition':condition,'measured_updates':200,'warmup_updates':20,'seconds':sec,'updates_per_second':200/sec})
    csv_write(run/'analysis/backend_benchmark.csv',rows)
    speed={d:np.mean([r['updates_per_second'] for r in rows if r['device']==d]) for d in devices}
    device=max(speed,key=speed.get)
    # Cross-backend forward correctness on identical C4 model, no classification selection.
    if len(devices)>1:
        cpu=Model('gaussian_C4',400);gpu=Model('gaussian_C4',400,'cuda:0')
        x=torch.from_numpy(arrays['train']['x'][:128]);a=gates_for('gaussian_C4',400,'cpu').build(0)
        b=tuple(v.cuda() for v in a)
        with torch.no_grad():p=cpu.run(x,a,0);q=gpu.run(x.cuda(),b,0).cpu()
        assert torch.allclose(p,q,rtol=1e-5,atol=1e-6),'Cross backend forward mismatch'
        write(run/'backend_correctness.json',{'max_abs_logit_difference':float((p-q).abs().max()),'atol':1e-6,'rtol':1e-5})
    # Estimate includes a deliberately conservative evaluation/IO multiplier.
    estimate=1712000/speed[device]*3
    write(run/'backend_selection.json',{'device':device,'mean_updates_per_second':speed,'conservative_estimated_total_seconds':estimate,
        'maximum_wall_seconds':cfg['maximum_wall_seconds'],'benchmark_one_step_diagnostic_updates':len(rows)*220,
        'benchmark_classification_scores_examined':False})
    if estimate>cfg['maximum_wall_seconds']:raise RuntimeError('Measured runtime estimate exceeds execution cap')
    return device

def train_job(run,stage,condition,seed,fold,device,cfg,deadline,stop_at=None):
    configure(device);check_seal(run)
    job=run/stage/f'{condition}_seed{seed}_fold{fold}';job.mkdir(parents=True,exist_ok=True)
    if (job/'completion.json').exists():
        done=read(job/'completion.json')
        for name,digest in done['files'].items():assert sha(job/name)==digest
        return done
    arrays=tensors(fold_arrays(fold,cfg['folds']),device)
    model=Model(condition,seed,device);initial=snapshot(model)
    base_initial=fingerprint({n:p for n,p in model.state_dict().items() if not n.startswith('context_')})
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.1,foreach=False)
    gates=gates_for(condition,seed,device)
    batches=paired_batches(len(arrays['train']['y']),3200,128,seed+10007*fold)
    masks=context_plan(seed,fold)
    codes=masks.copy();np.random.default_rng(seed+6000011+10007*fold).shuffle(codes)
    assert np.array_equal(np.sort(masks),np.sort(codes))
    history=[];gradients=[];diags=[];executed=[];start=0;elapsed=0
    spec={'condition':condition,'seed':seed,'fold':fold,'stage':stage,'backend':device,'steps':3200,
        'protocol_sha256':sha(ROOT/'PROTOCOL_PHASE3B_v3.md'),'source_manifest_sha256':sha(run/'source_manifest.json')}
    if (job/'latest.pt').exists():
        st=torch.load(job/'latest.pt',map_location=device,weights_only=False)
        assert st['spec']==spec
        model.load_state_dict(st['model']);optimizer.load_state_dict(st['optimizer'])
        assert np.array_equal(st['batches'],batches) and np.array_equal(st['masks'],masks) and np.array_equal(st['codes'],codes)
        start=st['next_step'];history=st['history'];gradients=st['gradients'];diags=st['diagnostics'];executed=st['executed'];elapsed=st['elapsed_seconds']
        # A replay of committed state must preserve all model tensors exactly.
        clone=Model(condition,seed,device);clone.load_state_dict(st['model'])
        with torch.no_grad():
            probe=arrays['train']['xt'][:16];gg=gates.build(min(start,3199))
            assert torch.equal(model.run(probe,gg,0),clone.run(probe,gg,0))
    began=time.perf_counter()
    def save(name,step):
        checkpoint(job/(name+'.pt'),{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'spec':spec,'next_step':step,
            'batches':batches,'masks':masks,'codes':codes,'history':history,'gradients':gradients,'diagnostics':diags,
            'initial_context':initial,'executed':executed,'elapsed_seconds':elapsed+time.perf_counter()-began})
        write(job/'steps.json',executed)
    def measure(completed,gate_step,kind):
        gg=gates.build(gate_step)
        for split in ('train','val'):
            scores,_,_=evaluate(model,arrays[split],gg)
            for row in scores:history.append(dict(completed_steps=completed,gate_step=gate_step,kind=kind,split=split,
                **{k:v for k,v in row.items() if k!='confusion'}))
        for row in changes(model,initial):diags.append(dict(completed_steps=completed,kind=kind,**row))
        for layer,weight,g in [('ih',model.fc1.weight,gg[0]),('hh',model.fc2.weight,gg[1])]:
            effective=weight.detach()*g
            diags.append(dict(completed_steps=completed,kind=kind,layer=layer,gate_row_energy=g.double().square().sum(1).cpu().tolist(),
                effective_weight_norm=float(effective.double().norm()),effective_weight_mean_abs=float(effective.double().abs().mean())))
    if start==0:
        write(job/'config.json',spec);measure(0,0,'initial');save('initial',0)
    limit=min(3200,stop_at) if stop_at is not None else 3200
    for step in range(start,limit):
        if time.monotonic()>deadline:save('latest',step);raise TimeoutError('Frozen execution wall cap reached')
        if step==2240:
            measure(2240,2239,'pre_release');save('pre_release',2240)
            measure(2240,2240,'post_release');save('post_release',2240)
        mask=int(masks[step]);code=int(codes[step]) if condition=='gaussian_shuffled' else mask
        gg=gates.build(step);ix=torch.as_tensor(batches[step],device=device)
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model.run(arrays['train']['xt'][ix],gg,mask,code),arrays['train']['yt'][ix])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward()
        sampled=step%160==0 or step==3199
        if sampled:
            norms={n:float(p.grad.double().norm()) for n,p in model.named_parameters() if p.grad is not None}
            if not all(math.isfinite(v) for v in norms.values()):raise FloatingPointError('Nonfinite gradient')
            before={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith('context_')}
            base=math.sqrt(sum(v*v for n,v in norms.items() if not n.startswith('context_')))
            ctx=math.sqrt(sum(v*v for n,v in norms.items() if n.startswith('context_')))
            record=dict(step=step,mask=mask,code=code,loss=float(loss.detach()),norms=norms,base_gradient_l2=base,
                context_gradient_l2=ctx,gradient_ratio=ctx/base if base else 0.)
        optimizer.step();executed.append(step)
        if sampled:
            active=[(n,p) for n,p in model.named_parameters() if n.startswith('context_') and p.requires_grad]
            record['active_update_fraction']=sum(int((p[code]!=before[n][code]).sum()) for n,p in active)/max(1,sum(p[code].numel() for n,p in active))
            record['stored_update_l2']=math.sqrt(sum(float((p-before[n]).double().square().sum()) for n,p in active))
            gradients.append(record)
        if (step+1)%160==0 or step+1==limit:
            measure(step+1,min(step+1,3199),'final' if step+1==3200 else 'curve');save('latest',step+1)
    if limit<3200:return {'partial':True,'completed_steps':limit}
    final={};subjectrows=[];preds={};gg=gates.build(3199)
    for split in ('train','val'):
        score,subjects,pred=evaluate(model,arrays[split],gg);final[split]=score
        subjectrows.extend([dict(split=split,**s) for s in subjects])
        preds.update({split+'_'+c:p for c,p in pred.items()})
        for key in ('y','subject','row'):preds[split+('_truth' if key=='y' else '_'+key)]=arrays[split][key]
    write(job/'final_metrics.json',{'scores':final,'subject_scores':subjectrows})
    np.savez_compressed(job/'predictions.npz',**preds)
    d,matrices=final_diagnostics(model,initial,arrays['val']['xt'][:128],gg)
    write(job/'context_diagnostics.json',d)
    if matrices:np.savez_compressed(job/'attenuation.npz',**matrices)
    interventions=[]
    if initial:
        disabled,_,_=evaluate(model,arrays['val'],gg,disabled=True)
        interventions.append({'kind':'context_disabled','scores':disabled})
        for perm in PERMUTATIONS:
            scores,_,_=evaluate(model,arrays['val'],gg,codes=perm)
            interventions.append({'kind':'code_permutation','permutation':perm,'scores':scores})
        if condition=='gaussian_shuffled':
            scores,_,_=evaluate(model,arrays['val'],gg,random_seed=8000029+seed+10007*fold)
            interventions.append({'kind':'balanced_random_codes_distribution_shift','scores':scores})
    write(job/'interventions.json',interventions)
    write(job/'history.json',history);write(job/'gradients.json',gradients);write(job/'diagnostics.json',diags)
    save('final',3200)
    restored=Model(condition,seed,device);restored.load_state_dict(torch.load(job/'final.pt',map_location=device,weights_only=False)['model'])
    with torch.no_grad():
        for c in range(3):assert torch.equal(model.run(arrays['val']['xt'][:128],gg,c),restored.run(arrays['val']['xt'][:128],gg,c))
    maxchange=max([r['live_max_delta_c'] for r in diags if 'live_max_delta_c' in r]+[0])
    changed=max([r['changed_parameter_fraction'] for r in diags if 'changed_parameter_fraction' in r]+[0])
    bounds=all(r['bounds_valid'] for r in diags if 'bounds_valid' in r)
    assert bounds,'Context bounds violated'
    done={'condition':condition,'seed':seed,'fold':fold,'backend':device,'completed_steps':3200,
        'allocated_parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'base_initial_fingerprint':base_initial,'batch_sha256':__import__('hashlib').sha256(batches.tobytes()).hexdigest(),
        'mask_sha256':__import__('hashlib').sha256(masks.tobytes()).hexdigest(),
        'checkpoint_reload_exact':True,'bounds_valid':bounds,'max_live_forward_delta_c':maxchange,
        'max_changed_context_fraction':changed,'numerical_level1':bool(changed>0 and maxchange>=1e-6 and bounds),
        'elapsed_seconds':elapsed+time.perf_counter()-began,'official_test_evaluations':0,
        'files':{p.name:sha(p) for p in job.iterdir() if p.is_file() and p.name not in ('completion.json',)}}
    write(job/'completion.json',done)
    return done

def execute(run):
    run=Path(run)
    assert (run/'preflight.json').exists(),'Preflight required'
    if not (run/'source_manifest.json').exists():seal_sources(run)
    check_seal(run);cfg=read(run/'config.frozen.json')
    started=time.monotonic()
    spent=read(run/'execution_status.json').get('elapsed_seconds',0) if (run/'execution_status.json').exists() else read(run/'preflight.json')['elapsed_seconds']
    deadline=started+cfg['maximum_wall_seconds']-spent
    if not (run/'backend_selection.json').exists():device=benchmark(run,cfg)
    else:device=read(run/'backend_selection.json')['device']
    configure(device)
    # Unit tests are required independently of analytical preflight gates.
    import subprocess,sys
    test=subprocess.run([sys.executable,'-m','pytest',str(ROOT/'tests'),'-q','-p','no:cacheprovider'],capture_output=True,text=True,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    write(run/'tests.json',{'exit_code':test.returncode,'stdout':test.stdout,'stderr':test.stderr})
    if test.returncode:raise RuntimeError('Unit tests failed; see tests.json')
    for condition in cfg['pilot_conditions']:
        print('pilot',condition,flush=True)
        train_job(run,'pilots',condition,400,0,device,cfg,deadline)
        write(run/'execution_status.json',{'stage':'pilots','last':condition,'elapsed_seconds':spent+time.monotonic()-started})
    from verify import verify_run
    verify_run(run)
    pilot_elapsed=[read(p/'completion.json')['elapsed_seconds'] for p in (run/'pilots').iterdir() if (p/'completion.json').exists()]
    write(run/'pilot_runtime_projection.json',{'pilot_mean_fit_seconds':float(np.mean(pilot_elapsed)),
        'estimated_525_grid_fit_seconds':float(np.mean(pilot_elapsed)*525),
        'elapsed_budget_seconds':spent+time.monotonic()-started})
    if np.mean(pilot_elapsed)*525+spent+time.monotonic()-started>cfg['maximum_wall_seconds']:
        raise RuntimeError('Pilot runtime projection exceeds frozen wall cap; grid not launched')
    omitted=[]
    for arch in ('dense','gaussian'):
        done=read(run/'pilots'/f'{arch}_C3_seed400_fold0'/'completion.json')
        if done['max_changed_context_fraction']>0 and done['max_live_forward_delta_c']<1e-6:omitted.append(arch+'_C3')
    if not (run/'execution_manifest.frozen.json').exists():
        write(run/'execution_manifest.frozen.json',{'backend':device,'omitted_conditions':omitted,
            'omitted_fits':35*len(omitted),'omission_reason':'predeclared C3 forward futility after3200 pilot steps',
            'full_grid_fits':35*(15-len(omitted)),'pilot_fits':10,'source_manifest_sha256':sha(run/'source_manifest.json'),
            'protocol_sha256':sha(ROOT/'PROTOCOL_PHASE3B_v3.md'),'maximum_wall_seconds':cfg['maximum_wall_seconds']})
        write(run/'execution_manifest.sha256',{'sha256':sha(run/'execution_manifest.frozen.json')})
    else:
        assert read(run/'execution_manifest.frozen.json')['omitted_conditions']==omitted
        assert read(run/'execution_manifest.sha256')['sha256']==sha(run/'execution_manifest.frozen.json')
    for seed in cfg['seeds']:
        for fold in range(7):
            for condition in CONDITIONS:
                if condition in omitted:continue
                print('grid',condition,seed,fold,flush=True)
                train_job(run,'grid',condition,seed,fold,device,cfg,deadline)
                write(run/'execution_status.json',{'stage':'grid','last':[condition,seed,fold],
                    'completed_grid_fits':len(list((run/'grid').glob('*/completion.json'))),
                    'elapsed_seconds':spent+time.monotonic()-started})
    result=verify_run(run)
    assert result['verified_fits']==10+35*(15-len(omitted))
    write(run/'execution_status.json',{'stage':'completed','elapsed_seconds':spent+time.monotonic()-started,**result})
    from reporting import report
    report(run)
