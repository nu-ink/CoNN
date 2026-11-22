"""Phase 2 development primitives. No official-test access or memory components."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from conn_assoc.data import stitch_official_train
from conn_assoc.gate import gaussian_gate
from conn_assoc.geometry import pairwise_abs_distance, temporal_input_coordinates, uniform_coordinates
from conn_assoc.paths import EXPERIMENT_ROOT, experiment2_root

sys.path.insert(0, str(experiment2_root() / 'src'))
from conn_exp.har_core import HARMLP, initial_state, paired_batches
from conn_exp.gates import draw_masks

CONDITIONS = {'P2_dense':'A','P2_gaussian':'D','P2_rowshuffle':'B','P2_coordperm':'C',
              'P2_fixed08':'S08','P2_fixed25':'S25'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def json_write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def fingerprint(state):
    h=hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode());h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def configure(device='cpu'):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if device!='cpu':torch.cuda.set_device(device)


def fold_arrays(fold, folds):
    arrays=stitch_official_train('temporal')
    validation=np.isin(arrays['subject'],folds[fold])
    fit=~validation
    assert set(arrays['subject'][fit]).isdisjoint(set(arrays['subject'][validation]))
    assert len(set(arrays['subject']))==21
    return {key:{'x':arrays['x'][m], 'y':arrays['y'][m], 'subject':arrays['subject'][m],
                 'row':np.arange(len(m))[m]} for key,m in [('train',fit),('val',validation)]}


class Gates:
    def __init__(self, condition, seed, device='cpu'):
        self.condition=CONDITIONS[condition]
        self.device=device
        h=uniform_coordinates(64)
        self.di=pairwise_abs_distance(h,temporal_input_coordinates()).to(device)
        self.dh=pairwise_abs_distance(h,h).to(device)
        mask=draw_masks(1152,64,64,seed)
        self.ip=mask.input_perm.to(device)
        self.pi=mask.row_perm_ih.to(device);self.ph=mask.row_perm_hh.to(device)
        self.ones=(torch.ones_like(self.di),torch.ones_like(self.dh))
        self.static=None

    def build(self,step):
        cid=self.condition
        if cid=='A' or (cid in ('D','B','C') and step>=2240):return self.ones
        if self.static is not None:return self.static
        if cid in ('S08','S25'):tau=(.08 if cid=='S08' else .25)/math.sqrt(2*math.log(2))
        else:
            u=min(1.,max(0.,step/2239))
            tau=(.08+.92*u*u*(3-2*u))/math.sqrt(2*math.log(2))
        # Match the original arithmetic order, including tau interpolation.
        if cid not in ('S08','S25'):
            start=.08/math.sqrt(2*math.log(2));end=1/math.sqrt(2*math.log(2))
            tau=start+(end-start)*(u*u*(3-2*u))
            if step==0:tau=start
            if step==2239:tau=end
        gi=gaussian_gate(self.di,tau)[0];gh=gaussian_gate(self.dh,tau)[0]
        if cid=='B':gi=torch.gather(gi,1,self.pi);gh=torch.gather(gh,1,self.ph)
        elif cid=='C':gi=gi[:,self.ip]
        result=(gi,gh)
        if cid in ('S08','S25'):self.static=result
        return result


def make_model(condition,seed,device='cpu'):
    if condition not in CONDITIONS:raise ValueError(condition)
    model=HARMLP(1152,[64,64]);model.load_state_dict(initial_state(1152,[64,64],seed))
    return model.to(device)


def forward(model,x,gates,context=0,disable_context=False):
    return model(x,None if gates.condition=='A' else gates.current[0],
                 None if gates.condition=='A' else gates.current[1])[0]


def context_plan(seed,fold,steps=3200):
    rng=np.random.default_rng(seed+5000003+10007*fold)
    return np.concatenate([rng.permutation(3) for _ in range((steps+2)//3)])[:steps]


def metrics(y,pred):
    cm=np.bincount(6*y.astype(int)+pred.astype(int),minlength=36).reshape(6,6)
    tp=np.diag(cm);den=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*tp,den,out=np.zeros(6),where=den>0)
    recall=np.divide(tp,cm.sum(1),out=np.zeros(6),where=cm.sum(1)>0)
    return {'accuracy':float(tp.sum()/len(y)),'macro_f1':float(f1.mean()),
            'recall':recall.tolist(),'confusion':cm.tolist()}


def evaluate(model,arrays,gates,contexts=(0,),disable_context=False,save_predictions=False):
    model.eval();results=[];subject_scores=[];predictions={}
    with torch.no_grad():
        for context in contexts:
            pred=[];loss=0.
            for start in range(0,len(arrays['y']),512):
                x=arrays['xt'][start:start+512];y=arrays['yt'][start:start+512]
                logits=forward(model,x,gates,context,disable_context)
                loss+=float(F.cross_entropy(logits,y,reduction='sum'))
                pred.append(logits.argmax(1).cpu().numpy())
            pred=np.concatenate(pred);m=metrics(arrays['y'],pred)
            m.update(context=int(context),loss=loss/len(pred));results.append(m)
            for subject in np.unique(arrays['subject']):
                mask=arrays['subject']==subject
                sub=metrics(arrays['y'][mask],pred[mask]);sub.update(subject=int(subject),context=int(context),windows=int(mask.sum()))
                sub.pop('confusion');sub.pop('recall');subject_scores.append(sub)
            if save_predictions:predictions[str(context)]=pred
    return results,subject_scores,predictions


def atomic_checkpoint(path,state):
    path=Path(path);temp=path.with_suffix('.tmp');torch.save(state,temp);temp.replace(path)


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def train_job(run,condition,seed,fold,lr,device='cpu',steps=3200,stop_at=None,diagnostics=True):
    """Exact resume of one finite development fit. Partial runs never get completion.json."""
    configure(device)
    run=Path(run).resolve()
    if not run.is_relative_to(EXPERIMENT_ROOT):raise ValueError('Run must be inside this experiment')
    config=json.loads((run/'config.frozen.json').read_text())
    seal=json.loads((run/'protocol_seal.json').read_text())
    if sha(EXPERIMENT_ROOT/'PROTOCOL_PHASE23_v1.md')!=seal['amendment_sha256']:raise RuntimeError('Amendment seal changed')
    if sha(run/'config.frozen.json')!=seal['config_sha256']:raise RuntimeError('Config seal changed')
    job=run/'jobs'/f'{condition}_seed{seed}_fold{fold}_lr{lr:g}'
    job.mkdir(parents=True,exist_ok=True)
    spec={'condition':condition,'seed':seed,'fold':fold,'lr':lr,'steps':steps,'device':device,
          'config_sha256':seal['config_sha256'],'amendment_sha256':seal['amendment_sha256']}
    spec_hash=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    if (job/'completion.json').exists():
        done=json.loads((job/'completion.json').read_text())
        if done['spec_hash']!=spec_hash:raise RuntimeError('Completed fit has different specification')
        return str(job)
    json_write(job/'config.json',spec)
    arrays=fold_arrays(fold,config['folds'])
    for a in arrays.values():a['xt']=torch.from_numpy(a['x']).to(device);a['yt']=torch.tensor(a['y'],dtype=torch.long,device=device)
    model=make_model(condition,seed,device);initial_fingerprint=fingerprint(model.state_dict())
    optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=0,weight_decay=0,foreach=False)
    gates=Gates(condition,seed,device)
    batches=paired_batches(len(arrays['train']['y']),steps,128,seed+10007*fold)
    contexts=context_plan(seed,fold,steps)
    evaluate_contexts=(0,1,2) if condition.startswith('P3_') else (0,)
    history=[];gradients=[];diag=[];start_step=0;elapsed_before=0.;update_seconds=0.
    if device!='cpu':torch.cuda.reset_peak_memory_stats(device)
    if (job/'latest.pt').exists():
        state=torch.load(job/'latest.pt',map_location=device,weights_only=False)
        if state['spec_hash']!=spec_hash:raise RuntimeError('Resume specification mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        assert np.array_equal(state['batches'],batches) and np.array_equal(state['contexts'],contexts)
        restore_rng(state['rng']);start_step=state['next_step'];history=state['history'];gradients=state['gradients'];diag=state['diagnostics']
        elapsed_before=state['elapsed_seconds'];update_seconds=state['update_seconds']
    began=time.perf_counter()

    def save(name,next_step):
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'next_step':next_step,
            'schedule':{'next_step':next_step,'release_step':2240,'lr_scheduler':None},'rng':rng_state(),
            'batches':batches,'contexts':contexts,'history':history,'gradients':gradients,'diagnostics':diag,
            'spec_hash':spec_hash,'spec':spec,'initial_fingerprint':initial_fingerprint,
            'elapsed_seconds':elapsed_before+time.perf_counter()-began,'update_seconds':update_seconds}
        atomic_checkpoint(job/(name+'.pt'),state)

    def measure(completed,gate_step,kind):
        gates.current=gates.build(gate_step)
        for split in ('train','val'):
            scores,_,_=evaluate(model,arrays[split],gates,evaluate_contexts)
            for row in scores:
                history.append({'completed_steps':completed,'gate_step':gate_step,'kind':kind,'split':split,
                                **{k:v for k,v in row.items() if k not in ('recall','confusion')}})
        # Record per-unit energy for both layers, effective-weight magnitude and context summaries.
        if diagnostics:
            for name,g,weight in [('ih',gates.current[0],model.fc1.weight),('hh',gates.current[1],model.fc2.weight)]:
                eff=weight.detach()*g
                diag.append({'completed_steps':completed,'kind':kind,'layer':name,
                    'gate_row_energy':g.square().sum(1).cpu().tolist(),'effective_weight_norm':float(eff.norm()),
                    'effective_weight_mean_abs':float(eff.abs().mean())})
        model.train()

    if start_step==0:
        measure(0,0,'initial');save('initial',0)
    limit=min(steps,stop_at) if stop_at is not None else steps
    for step in range(start_step,limit):
        if step==2240:
            measure(step,2239,'pre_release');save('pre_release',step)
            measure(step,2240,'post_release');save('post_release',step)
        gates.current=gates.build(step)
        c=int(contexts[step]) if condition.startswith('P3_') else 0
        index=torch.as_tensor(batches[step],device=device)
        optimizer.zero_grad(set_to_none=True)
        t=time.perf_counter()
        logits=forward(model,arrays['train']['xt'][index],gates,c)
        loss=F.cross_entropy(logits,arrays['train']['yt'][index])
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss {job} step {step}')
        loss.backward()
        if step%160==0 or step==steps-1:
            norms={name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None}
            if not all(math.isfinite(v) for v in norms.values()):raise FloatingPointError('Nonfinite gradient')
            gradients.append({'step':step,'context':c,'loss':float(loss.detach()),'norms':norms})
        optimizer.step();update_seconds+=time.perf_counter()-t
        completed=step+1
        if completed%160==0 or completed==limit:
            measure(completed,min(completed,3199),'final' if completed==steps else 'curve')
            save('latest',completed)
    if limit<steps:return str(job)
    gates.current=gates.build(steps-1)
    final={};subject_rows=[];preds={}
    for split in ('train','val'):
        score,subjects,pred=evaluate(model,arrays[split],gates,evaluate_contexts,save_predictions=True)
        final[split]=score
        subject_rows.extend([dict(split=split,**s) for s in subjects])
        preds.update({split+'_context'+c:p for c,p in pred.items()})
        preds[split+'_truth']=arrays[split]['y'];preds[split+'_subject']=arrays[split]['subject'];preds[split+'_row']=arrays[split]['row']
    json_write(job/'final_metrics.json',{'scores':final,'subject_scores':subject_rows})
    json_write(job/'history.json',history);json_write(job/'gradients.json',gradients);json_write(job/'diagnostics.json',diag)
    np.savez_compressed(job/'predictions.npz',**preds)
    save('final',steps)
    assert initial_fingerprint!=fingerprint(model.state_dict()),'Weights did not update'
    cloned=make_model(condition,seed,device)
    cloned.load_state_dict(torch.load(job/'final.pt',map_location=device,weights_only=False)['model'])
    with torch.no_grad():
        probe=arrays['val']['xt'][:16]
        assert torch.equal(forward(model,probe,gates),forward(cloned,probe,gates))
    json_write(job/'completion.json',{'spec_hash':spec_hash,'completed_steps':steps,'parameter_count':sum(p.numel() for p in model.parameters()),
      'initial_fingerprint':initial_fingerprint,'final_fingerprint':fingerprint(model.state_dict()),
      'elapsed_seconds':elapsed_before+time.perf_counter()-began,'update_seconds':update_seconds,
      'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(device) if device!='cpu' else 0,
      'official_test_evaluations':0,'checkpoint_reload_exact':True,
      'files':{p.name:sha(p) for p in job.iterdir() if p.is_file() and p.name!='completion.json'}})
    return str(job)
