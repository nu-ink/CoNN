"""Multiclass HAR models using the existing Gaussian gates and paired streams."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from conn_exp.diagnostics import bin_mean_abs, far_suppression
from conn_exp.gates import apply_row_permutation, gaussian_gate, summarize_gate
from conn_exp.geometry import Geometry, pairwise_abs_distance, uniform_coordinates
from conn_exp.models import GatedLinear, ForwardCache, build_optimizer, fingerprint_state, initialize_model
from conn_exp.schedules import EXACT_FACTOR, _tau_between, release_step_index
from conn_exp.seeding import paired_seed

CONDITION_NAMES=dict(A='conventional_dense',S08='fixed_local_d50_0.08',S25='fixed_local_d50_0.25',
    D='structured_local_to_global',B='row_shuffled_schedule',C='permuted_input_coordinates_schedule')


class HARMLP(nn.Module):
    def __init__(self,n_features,hidden_sizes,n_classes=6):
        super().__init__()
        h1,h2=hidden_sizes
        self.fc1=GatedLinear(n_features,h1);self.fc2=GatedLinear(h1,h2)
        self.out_weight=nn.Parameter(torch.empty(n_classes,h2))
        self.out_bias=nn.Parameter(torch.empty(n_classes))

    def forward(self,x,gate_ih=None,gate_hh=None):
        if gate_ih is None:
            # Conventional dense baseline: no gate multiplication overhead.
            w1,w2=self.fc1.weight,self.fc2.weight
            h1=F.relu(F.linear(x,w1,self.fc1.bias));h2=F.relu(F.linear(h1,w2,self.fc2.bias))
        else:
            p1,w1=self.fc1(x,gate_ih);h1=F.relu(p1)
            p2,w2=self.fc2(h1,gate_hh);h2=F.relu(p2)
        logits=F.linear(h2,self.out_weight,self.out_bias)
        return logits,ForwardCache(h1,h2,w1,w2,logits)


def initial_state(n_features,hidden_sizes,seed):
    model=HARMLP(n_features,hidden_sizes)
    initialize_model(model,torch.Generator().manual_seed(paired_seed(seed,'init')))
    return {k:v.detach().clone() for k,v in model.state_dict().items()}


def geometry_from_coordinates(coordinates,hidden_sizes):
    x=torch.as_tensor(coordinates,dtype=torch.float32)
    h1,h2=[uniform_coordinates(n) for n in hidden_sizes]
    return Geometry(x,(h1,h2),pairwise_abs_distance(h1,x),pairwise_abs_distance(h2,h1))


def paired_batches(n,steps,batch_size,seed):
    generator=np.random.default_rng(paired_seed(seed,'order'))
    required=steps*batch_size
    order=np.concatenate([generator.permutation(n) for _ in range((required+n-1)//n)])[:required]
    return order.reshape(steps,batch_size).astype(np.int64)


class HARGates:
    def __init__(self,geometry,masks,condition,config):
        if condition not in CONDITION_NAMES:
            raise ValueError('Unknown condition')
        self.geometry,self.masks,self.condition,self.config=geometry,masks,condition,config
        self.cached={}

    def build(self,step):
        cfg=self.config;cid=self.condition
        release=release_step_index(cfg['total_steps'],cfg['release_fraction'])
        exact=cid=='A' or (cid in ['D','B','C'] and step>=release)
        if exact:
            tau=None
        elif cid in ['S08','S25']:
            tau=(.08 if cid=='S08' else .25)/EXACT_FACTOR
        else:
            tau=_tau_between(step,0,release,cfg['d50_start']/EXACT_FACTOR,cfg['d50_end']/EXACT_FACTOR)
        static=exact or cid in ['S08','S25']
        key=(tau,exact)
        if static and key in self.cached:
            return self.cached[key]
        if exact:
            ih,hh=torch.ones_like(self.geometry.dist_ih),torch.ones_like(self.geometry.dist_hh)
        else:
            ih,_=gaussian_gate(self.geometry.dist_ih,tau)
            hh,_=gaussian_gate(self.geometry.dist_hh,tau)
        if cid=='B':
            ih=apply_row_permutation(ih,self.masks.row_perm_ih)
            hh=apply_row_permutation(hh,self.masks.row_perm_hh)
        elif cid=='C':
            ih=ih[:,self.masks.input_perm]
        if static:
            self.cached[key]=(ih,hh)
        return ih,hh


def classification_metrics(y,pred,n_classes=6):
    y=np.asarray(y,dtype=np.int64);pred=np.asarray(pred,dtype=np.int64)
    if len(y)==0 or not np.isin(y,np.arange(n_classes)).all() or not np.isin(pred,np.arange(n_classes)).all():
        raise ValueError('Invalid classification labels')
    confusion=np.bincount(n_classes*y+pred,minlength=n_classes*n_classes).reshape(n_classes,n_classes)
    tp=np.diag(confusion);den=confusion.sum(1)+confusion.sum(0)
    f1=np.divide(2*tp,den,out=np.zeros(n_classes,dtype=float),where=den!=0)
    return dict(accuracy=float(tp.sum()/len(y)),macro_f1=float(f1.mean()),
        per_class_f1=f1,confusion=confusion)


def evaluate(model,x,y,ih,hh,dense=False,batch_size=512,predictions=False):
    model.eval();loss=0.;outputs=[];probabilities=[]
    with torch.no_grad():
        for start in range(0,len(y),batch_size):
            logits,_=model(x[start:start+batch_size],None if dense else ih,None if dense else hh)
            loss+=float(F.cross_entropy(logits,y[start:start+batch_size],reduction='sum'))
            outputs.append(logits.argmax(1).cpu().numpy())
            if predictions:
                probabilities.append(logits.softmax(1).cpu().numpy())
    pred=np.concatenate(outputs)
    result=classification_metrics(y.cpu().numpy(),pred)
    result['loss']=loss/len(y)
    if predictions:
        result.update(prediction=pred,probability=np.concatenate(probabilities))
    return result


def gradient_measurement(model,x,y,ih,hh,dense):
    model.zero_grad(set_to_none=True)
    logits,cache=model(x,None if dense else ih,None if dense else hh)
    loss=F.cross_entropy(logits,y)
    gradients=torch.autograd.grad(loss,[model.fc1.weight,cache.w_eff_ih,model.fc2.weight,cache.w_eff_hh])
    model.zero_grad(set_to_none=True)
    return dict(ih=dict(dL_dW=gradients[0].detach(),dL_dW_eff=gradients[1].detach()),
        hh=dict(dL_dW=gradients[2].detach(),dL_dW_eff=gradients[3].detach()))


def train(config,representation,condition,seed,lr,arrays,init,masks,geometry,batches,job,final_test=False):
    """No test array is required unless the caller explicitly enables final evaluation."""
    job=Path(job);job.mkdir(parents=True,exist_ok=False)
    body=json.dumps(dict(**config,representation=representation,condition=condition,seed=seed,lr=lr,
        test_enabled=final_test),indent=2,sort_keys=True)+'\n'
    (job/'config.resolved.json').write_text(body,encoding='utf-8',newline='\n')
    digest=hashlib.sha256(body.encode()).hexdigest()
    (job/'config_sha256.txt').write_text(digest+'\n')
    model=HARMLP(arrays['x_train'].shape[1],config['hidden_sizes'])
    model.load_state_dict(init)
    optimizer,hparams=build_optimizer(model,lr)
    gates=HARGates(geometry,masks,condition,config)
    xtr=torch.as_tensor(arrays['x_train']);ytr=torch.as_tensor(arrays['y_train'],dtype=torch.long)
    xv=torch.as_tensor(arrays['x_val']);yv=torch.as_tensor(arrays['y_val'],dtype=torch.long)
    diag_index=torch.as_tensor(batches[0]);xd,yd=xtr[diag_index],ytr[diag_index]
    release=release_step_index(config['total_steps'],config['release_fraction'])
    base=dict(representation=representation,condition=condition,seed=seed,lr=lr)
    tables={name:[] for name in ['metrics','gradients','row_energy','ablation','checkpoint_gate_comparisons']}
    saved={};test_evaluations=0;update_wall=0.;update_cpu=0.
    total_start=time.perf_counter()

    def measure(step,completed,kind,checkpoint=False,test=False):
        nonlocal test_evaluations
        ih,hh=gates.build(step);dense=condition=='A'
        common=dict(**base,step_index=step,optimizer_steps_completed=completed,kind=kind)
        diag={}
        for layer,gate,weight in [('ih',ih,model.fc1.weight),('hh',hh,model.fc2.weight)]:
            stats=summarize_gate(gate)
            eff=weight.detach()*gate
            diag.update({f'gate_energy_{layer}':stats.energy,f'norm_W_eff_{layer}':float(eff.norm()),
                f'mean_abs_W_eff_{layer}':float(eff.abs().mean())})
            for unit,energy in enumerate(stats.row_energy.tolist()):
                tables['row_energy'].append(dict(**common,layer=layer,unit=unit,energy=energy))
        for name,x,y in [('train',xtr,ytr),('val',xv,yv)]:
            result=evaluate(model,x,y,ih,hh,dense)
            if not np.isfinite([result[k] for k in ['loss','accuracy','macro_f1']]).all():
                raise RuntimeError(f'Nonfinite measured metrics {job}')
            tables['metrics'].append(dict(**common,**diag,split=name,
                **{k:result[k] for k in ['loss','accuracy','macro_f1']}))
        gradients=gradient_measurement(model,xd,yd,ih,hh,dense)
        for layer,distance in [('ih',geometry.dist_ih),('hh',geometry.dist_hh)]:
            for quantity,values in gradients[layer].items():
                for b in bin_mean_abs(values,distance,[0,.2,.4,.6,.8,1.]):
                    tables['gradients'].append(dict(**common,layer=layer,quantity=quantity,**b,
                        distance='frozen_native_coordinates',batch='fixed_first_paired_training_minibatch'))
        val=evaluate(model,xv,yv,ih,hh,dense)
        suppressed=far_suppression(ih,hh,geometry.dist_ih,geometry.dist_hh,.75)
        for intervention,gi,gh in [('ih',suppressed['gate_ih'],hh),('hh',ih,suppressed['gate_hh']),
                ('both',suppressed['gate_ih'],suppressed['gate_hh'])]:
            result=evaluate(model,xv,yv,gi,gh,False)
            tables['ablation'].append(dict(**common,split='val',intervention=intervention,
                val_loss_delta=result['loss']-val['loss'],val_accuracy_delta=result['accuracy']-val['accuracy'],
                val_macro_f1_delta=result['macro_f1']-val['macro_f1'],far_quantile=.75,
                threshold_ih=suppressed['threshold_ih'],threshold_hh=suppressed['threshold_hh']))
        if checkpoint:
            state=dict(model={k:v.detach().clone() for k,v in model.state_dict().items()},
                optimizer=optimizer.state_dict(),optimizer_hparams=hparams,gate_ih=ih,gate_hh=hh,
                **common,config_sha256=digest)
            path=job/(kind+'.pt');torch.save(state,path)
            saved[kind]=fingerprint_state(state['model'])
            clone=HARMLP(xtr.shape[1],config['hidden_sizes'])
            clone.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
            before=fingerprint_state(clone.state_dict())
            for mode,gi,gh in [('native',ih,hh),('exact_dense',torch.ones_like(ih),torch.ones_like(hh))]:
                result=evaluate(clone,xv,yv,gi,gh,dense or mode=='exact_dense')
                tables['checkpoint_gate_comparisons'].append(dict(**common,gate_mode=mode,split='val',
                    loss=result['loss'],accuracy=result['accuracy'],macro_f1=result['macro_f1'],
                    weight_fingerprint=before,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            assert fingerprint_state(clone.state_dict())==before
        if test:
            if not final_test:
                raise AssertionError('Test evaluation forbidden')
            xt=torch.as_tensor(arrays['x_test']);yt=torch.as_tensor(arrays['y_test'],dtype=torch.long)
            result=evaluate(model,xt,yt,ih,hh,dense,predictions=True)
            test_evaluations+=1
            tables['metrics'].append(dict(**common,**diag,split='test',
                **{k:result[k] for k in ['loss','accuracy','macro_f1']}))
            np.savez_compressed(job/'test_predictions.npz',truth=arrays['y_test'],subject=arrays['subject_test'],
                prediction=result['prediction'],probability=result['probability'])
            pd.DataFrame(result['confusion']).to_csv(job/'test_confusion.csv',index=False)
            pd.DataFrame(dict(label=np.arange(1,7),f1=result['per_class_f1'])).to_csv(job/'test_per_class_f1.csv',index=False)
            subject_rows=[]
            for subject in np.unique(arrays['subject_test']):
                chosen=arrays['subject_test']==subject
                r=classification_metrics(arrays['y_test'][chosen],result['prediction'][chosen])
                subject_rows.append(dict(**base,subject=int(subject),windows=int(chosen.sum()),accuracy=r['accuracy'],macro_f1=r['macro_f1']))
            pd.DataFrame(subject_rows).to_csv(job/'test_subject_scores.csv',index=False)

    measure(0,0,'initial')
    for step in range(config['total_steps']):
        if step==release:
            measure(step,step,'post_release',True)
        elif step>0 and step%config['eval_interval']==0:
            measure(step,step,'curve')
        wall_start=time.perf_counter();cpu_start=time.process_time()
        ih,hh=gates.build(step)
        optimizer.zero_grad(set_to_none=True);model.train()
        index=torch.as_tensor(batches[step])
        logits,_=model(xtr[index],None if condition=='A' else ih,None if condition=='A' else hh)
        loss=F.cross_entropy(logits,ytr[index])
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite training loss {job}, step {step}')
        loss.backward();optimizer.step()
        update_cpu+=time.process_time()-cpu_start;update_wall+=time.perf_counter()-wall_start
        if step+1==release:
            measure(step,step+1,'pre_release',True)
    measure(config['total_steps']-1,config['total_steps'],'final',True,final_test)
    assert saved['pre_release']==saved['post_release']
    elapsed=time.perf_counter()-total_start
    for name,rows in tables.items():
        pd.DataFrame(rows).to_csv(job/(name+'.csv'),index=False)
    score=dict(**base,condition_name=CONDITION_NAMES[condition],parameter_count=sum(p.numel() for p in model.parameters()),
        total_steps=config['total_steps'],training_examples_seen=config['total_steps']*config['batch_size'],
        update_wall_seconds=update_wall,update_process_cpu_seconds=update_cpu,
        fit_elapsed_wall_seconds_with_diagnostics=elapsed,test_evaluations=test_evaluations,
        init_fingerprint=fingerprint_state(init),batch_order_sha256=hashlib.sha256(batches.tobytes()).hexdigest(),
        config_sha256=digest)
    for r in tables['metrics']:
        if r['kind']=='final':
            for name in ['loss','accuracy','macro_f1']:
                score[r['split']+'_'+name]=r[name]
    if final_test:
        for name in ['accuracy','macro_f1']:
            score['train_test_gap_'+name]=score['train_'+name]-score['test_'+name]
            score['train_val_gap_'+name]=score['train_'+name]-score['val_'+name]
            score['val_test_gap_'+name]=score['val_'+name]-score['test_'+name]
    assert test_evaluations==int(final_test)
    return score
