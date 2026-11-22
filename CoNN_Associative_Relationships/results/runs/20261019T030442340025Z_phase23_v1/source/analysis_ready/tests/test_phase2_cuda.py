"""Explicit CUDA checks: skipped means NOT RUN, never counted as a pass."""
import json
from pathlib import Path
import shutil
import uuid

import pytest
import torch

from conn_assoc.paths import EXPERIMENT_ROOT
from conn_assoc.phase23_core import Gates,configure,forward,make_model,train_job


def require_cuda():
    if not torch.cuda.is_available():pytest.skip('NOT RUN: current PyTorch cannot execute CUDA')


def test_cpu_gpu_inference_declared_tolerance():
    require_cuda();configure('cuda:0')
    rng=torch.Generator().manual_seed(311)
    x=torch.randn(16,1152,generator=rng)
    for condition in ('P2_dense','P2_gaussian','P2_rowshuffle','P2_coordperm','P2_fixed08','P2_fixed25',
                      'P3_dense_masked','P3_gaussian_masked','P3_dense_context','P3_gaussian_context'):
        cpu=make_model(condition,400);gpu=make_model(condition,400,'cuda:0')
        a=Gates(condition,400);b=Gates(condition,400,'cuda:0')
        for step in (0,1600,2240):
            a.current=a.build(step);b.current=b.build(step)
            for context in ((0,1,2) if condition.startswith('P3_') else (0,)):
                with torch.no_grad():
                    y=forward(cpu,x,a,context);z=forward(gpu,x.cuda(),b,context).cpu()
                assert torch.allclose(y,z,atol=2e-5,rtol=2e-4),(condition,step,context,(y-z).abs().max())


def test_cuda_actual_optimizer_checkpoint_smoke():
    require_cuda()
    active=Path((EXPERIMENT_ROOT/'config/phase23_active_run.txt').read_text().strip())
    run=active/'verification'/('cuda_smoke_'+uuid.uuid4().hex);run.mkdir(parents=True)
    for name in ('config.frozen.json','protocol_seal.json'):shutil.copy2(active/name,run/name)
    path=Path(train_job(run,'P2_gaussian',400,0,.01,'cuda:0',steps=8,diagnostics=False))
    done=json.loads((path/'completion.json').read_text())
    assert done['completed_steps']==8 and done['checkpoint_reload_exact']
    assert done['peak_gpu_allocated_bytes']>0 and done['official_test_evaluations']==0
