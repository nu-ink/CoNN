"""Bounded Phase 2/3 development CLI; never evaluates the official test split."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from conn_assoc.paths import EXPERIMENT_ROOT
from conn_assoc.phase23_core import (Gates, configure, context_plan, fold_arrays, forward,
    json_write, make_model, sha, train_job)


def active_run():return Path((EXPERIMENT_ROOT/'config/phase23_active_run.txt').read_text().strip())


def snapshot(run,tag):
    dest=run/'source'/tag
    if dest.exists():raise FileExistsError(dest)
    files=[p for folder in ('src','tests','scripts','config') for p in (EXPERIMENT_ROOT/folder).rglob('*')
           if p.is_file() and '__pycache__' not in p.parts]
    files += [EXPERIMENT_ROOT/'PROTOCOL_PHASE23_v1.md',EXPERIMENT_ROOT/'pyproject.toml']
    for p in files:
        q=dest/p.relative_to(EXPERIMENT_ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    json_write(dest/'source_manifest.json',{str(p.relative_to(EXPERIMENT_ROOT)):sha(p) for p in files})


def hardware(run):
    info={'executable':sys.executable,'python':sys.version,'torch':torch.__version__,'torch_cuda':torch.version.cuda,
          'available':torch.cuda.is_available(),'device_count':torch.cuda.device_count(),'devices':[]}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            configure(f'cuda:{i}');torch.cuda.reset_peak_memory_stats(i)
            a=torch.arange(64,dtype=torch.float32,device=f'cuda:{i}').reshape(8,8).requires_grad_()
            product=a@a.T;product.sum().backward();torch.cuda.synchronize(i)
            expected=torch.arange(64,dtype=torch.float32).reshape(8,8)
            assert torch.allclose(product.detach().cpu(),expected@expected.T)
            assert torch.isfinite(a.grad).all() and a.grad.abs().sum()>0
            prop=torch.cuda.get_device_properties(i)
            info['devices'].append({'id':i,'name':prop.name,'total_memory_bytes':prop.total_memory,
                'allocated_bytes':torch.cuda.memory_allocated(i),'peak_allocated_bytes':torch.cuda.max_memory_allocated(i),
                'allocation_matmul_backward_transfer':'passed'})
    json_write(run/'cuda_execution.json',info)
    return info


def benchmark(run):
    info=hardware(run)
    config=json.loads((run/'config.frozen.json').read_text())
    arrays=fold_arrays(0,config['folds'])
    results=[]
    devices=['cpu']+(['cuda:0'] if info['available'] else [])
    for device in devices:
        configure(device)
        x=torch.from_numpy(arrays['train']['x'][:128]).to(device)
        y=torch.tensor(arrays['train']['y'][:128],dtype=torch.long,device=device)
        for condition in ('P2_dense','P2_gaussian'):
            model=make_model(condition,400,device);gates=Gates(condition,400,device)
            opt=torch.optim.SGD(model.parameters(),lr=.01,foreach=False)
            for step in range(20+200):
                if step==20:
                    if device!='cpu':torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
                    started=time.perf_counter()
                gates.current=gates.build(step)
                opt.zero_grad(set_to_none=True)
                loss=F.cross_entropy(forward(model,x,gates),y)
                assert torch.isfinite(loss)
                loss.backward();opt.step()
            if device!='cpu':torch.cuda.synchronize(device)
            elapsed=time.perf_counter()-started
            results.append({'device':device,'condition':condition,'updates':200,'warmup_updates':20,
                'seconds':elapsed,'batches_per_second':200/elapsed,'samples_per_second':25600/elapsed,
                'parameter_count':sum(p.numel() for p in model.parameters()),
                'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(device) if device!='cpu' else 0,
                'cpu_fallback':not info['available'],'timing':'sequential isolated warmed-up synchronized, float32 TF32 off'})
    scores={dev:np.mean([r['batches_per_second'] for r in results if r['device']==dev]) for dev in devices}
    chosen=max(scores,key=scores.get)
    json_write(run/'benchmark.json',{'measurements':results,'chosen_device':chosen,'selection_rule':'highest mean training batches/s over dense and Gaussian, no validation scores'})
    print(json.dumps({'chosen_device':chosen,'throughput':scores}),flush=True)


def worker(task):
    run,condition,seed,fold,lr,device=task
    try:
        return {'task':list(task[1:]),'job':train_job(run,condition,seed,fold,lr,device),'status':'complete'}
    except BaseException:
        return {'task':list(task[1:]),'status':'failed','traceback':traceback.format_exc()}


def grid(run,stage,workers,devices):
    config=json.loads((run/'config.frozen.json').read_text())
    conditions=list(config['conditions'])
    if stage=='first_seed':conditions=['P2_dense','P2_gaussian'];seeds=[400]
    elif stage=='phase2':conditions=[c for c in conditions if c.startswith('P2_')];seeds=config['seeds']
    else:seeds=config['seeds']
    tasks=[(str(run),c,s,f,lr,devices[0]) for s in seeds for f in range(7)
           for lr in config['learning_rates'] for c in conditions]
    # Shuffle condition execution independently; first-seed work remains paired.
    rng=np.random.default_rng(230019);rng.shuffle(tasks)
    pending=[];already=0
    for t in tasks:
        _,c,s,f,lr,_=t
        if (run/'jobs'/f'{c}_seed{s}_fold{f}_lr{lr:g}'/'completion.json').exists():already+=1
        else:pending.append(t)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    schedule=run/'execution'/stamp;schedule.mkdir(parents=True)
    json_write(schedule/'plan.json',{'stage':stage,'tasks':tasks,'workers':workers,'devices':devices,'already_completed':already})
    started=time.perf_counter();done=0;failed=0
    # Independent one-thread CPU workers. With CUDA, one single-worker pool per GPU.
    pools=[ProcessPoolExecutor(max_workers=workers)] if devices==['cpu'] else [ProcessPoolExecutor(max_workers=1) for _ in devices]
    futures={}
    try:
        for i,task in enumerate(pending):
            pool=pools[0] if devices==['cpu'] else pools[i%len(pools)]
            task=(*task[:-1],devices[i%len(devices)])
            futures[pool.submit(worker,task)]=task
        for future in as_completed(futures):
            result=future.result();done+=result['status']=='complete';failed+=result['status']!='complete'
            with (schedule/'events.jsonl').open('a',encoding='utf-8') as handle:handle.write(json.dumps(result)+'\n')
            status={'stage':stage,'done':done,'failed':failed,'already_completed':already,'pending_total':len(pending),
                    'elapsed_seconds':time.perf_counter()-started,'last':result}
            json_write(schedule/'progress.json',status)
            print(f'{stage}: {already+done}/{len(tasks)} completed; failed={failed}; seconds={status["elapsed_seconds"]:.1f}',flush=True)
            if failed:raise RuntimeError(result)
            if time.perf_counter()-started>config['maximum_grid_wall_seconds']:
                for queued in futures:queued.cancel()
                raise TimeoutError('Bounded grid wall-time reached; running finite jobs finish at saved boundaries')
    finally:
        for pool in pools:pool.shutdown(wait=True,cancel_futures=True)
    json_write(schedule/'completion.json',{'stage':stage,'completed':already+done,'expected':len(tasks),
        'failed':failed,'elapsed_seconds':time.perf_counter()-started})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['snapshot','benchmark','grid'])
    parser.add_argument('--run',type=Path,default=active_run())
    parser.add_argument('--tag',default='phase2_pretraining')
    parser.add_argument('--stage',choices=['first_seed','phase2','all'],default='first_seed')
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--devices',default='cpu')
    args=parser.parse_args()
    if args.command=='snapshot':snapshot(args.run,args.tag)
    if args.command=='benchmark':benchmark(args.run)
    if args.command=='grid':grid(args.run,args.stage,args.workers,args.devices.split(','))


if __name__=='__main__':main()
