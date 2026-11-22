"""External context isolation, bounded differentiability, controls and resume."""
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import torch
import pytest

from conn_assoc.paths import EXPERIMENT_ROOT
from conn_assoc.phase23_core import Gates,configure,context_plan,forward,make_model,train_job
from conn_assoc.phase3_context import ContextFactor,mask_input


def test_masking_excludes_all_unavailable_channels_and_context_is_external():
    x=torch.arange(1152,dtype=torch.float32).reshape(1,-1)
    for context,allowed in [(1,[0,1,2,6,7,8]),(2,[3,4,5])]:
        masked=mask_input(x,context).reshape(128,9)
        for ch in range(9):
            assert torch.equal(masked[:,ch],x.reshape(128,9)[:,ch]) if ch in allowed else torch.count_nonzero(masked[:,ch])==0
        changed=x.reshape(128,9).clone()
        changed[:,[i for i in range(9) if i not in allowed]]=1e9
        assert torch.equal(mask_input(x,context),mask_input(changed.reshape(1,-1),context))
    with pytest.raises(ValueError):mask_input(x,3)
    plan=context_plan(400,0)
    assert np.array_equal(plan,context_plan(400,0))
    assert np.bincount(plan).max()-np.bincount(plan).min()<=1


def test_near_identity_bounds_parameter_counts_and_context_only_gradients():
    configure()
    model=make_model('P3_gaussian_context',400)
    assert sum(p.numel() for p in model.parameters())==86406
    assert sum(p.numel() for p in make_model('P3_gaussian_masked',400).parameters())==78342
    factor=model.context_ih
    expected=1.-.475*torch.sigmoid(torch.tensor(-6.))
    assert torch.allclose(factor(0),torch.full((64,1152),expected),atol=1e-7)
    before=[factor(c).detach().clone() for c in range(3)]
    opt=torch.optim.SGD(factor.parameters(),lr=1.)
    factor(1).sum().backward()
    assert factor.u.grad[1].abs().sum()>0 and factor.v.grad[1].abs().sum()>0
    assert factor.u.grad[0].count_nonzero()==0 and factor.v.grad[2].count_nonzero()==0
    opt.step()
    assert torch.equal(before[0],factor(0)) and not torch.equal(before[1],factor(1))
    for extreme in (-1000.,1000.):
        with torch.no_grad():factor.u.fill_(extreme);factor.v.fill_(extreme)
        c=factor(0)
        assert (c>0).all() and (c<=1).all() and (c>=.05-1e-6).all()


def test_context_gradients_not_detached_gaussian_cannot_be_amplified():
    configure();model=make_model('P3_gaussian_context',400)
    base=make_model('P3_gaussian_masked',400)
    for key,value in base.state_dict().items():assert torch.equal(value,model.state_dict()[key])
    gates=Gates('P3_gaussian_context',400);gates.current=gates.build(0)
    gates.current=tuple(g.clone().requires_grad_() for g in gates.current)
    x=torch.randn(16,1152);y=torch.arange(16)%6
    opt=torch.optim.SGD(model.parameters(),lr=.05)
    before=model.context_ih.u.detach().clone()
    before_v=model.context_ih.v.detach().clone()
    loss=torch.nn.functional.cross_entropy(forward(model,x,gates,2),y)
    loss.backward()
    assert model.context_ih.u.grad.abs().sum()>0 and model.context_ih.v.grad.abs().sum()>0
    opt.step()
    # At U=-6 a tiny float32 SGD change can round away; V starts at zero and
    # must actually update. Record this numerical limitation instead of assuming
    # every parameter changes on every minibatch.
    assert not torch.equal(before_v,model.context_ih.v)
    active=Path((EXPERIMENT_ROOT/'config/phase23_active_run.txt').read_text().strip())
    evidence=active/'verification'/('context_precision_'+uuid.uuid4().hex+'.json')
    evidence.parent.mkdir(exist_ok=True)
    evidence.write_text(json.dumps({'u_gradient_norm':float(model.context_ih.u.grad.norm()),
        'v_gradient_norm':float(model.context_ih.v.grad.norm()),
        'u_updated':not torch.equal(before,model.context_ih.u),'v_updated':not torch.equal(before_v,model.context_ih.v),
        'note':'Individual U updates can round away near -6 in float32; frozen design retained.'},indent=2))
    assert all(g.grad is None for g in gates.current)
    assert torch.all(gates.current[0].detach()*model.context_ih(2)<=gates.current[0].detach())
    # Alter unavailable acceleration values: no change to gyroscope-only prediction.
    changed=x.reshape(-1,128,9).clone();changed[:,:,[0,1,2,6,7,8]]=100000.
    assert torch.equal(forward(model,x,gates,2),forward(model,changed.reshape(-1,1152),gates,2))


def test_context_checkpoint_exact_resume_and_intervention_persistence():
    active=Path((EXPERIMENT_ROOT/'config/phase23_active_run.txt').read_text().strip())
    testroot=active/'verification'/('context_resume_'+uuid.uuid4().hex)
    for tag in ('continuous','resumed'):
        run=testroot/tag;run.mkdir(parents=True)
        for name in ('config.frozen.json','protocol_seal.json'):shutil.copy2(active/name,run/name)
    a=Path(train_job(testroot/'continuous','P3_gaussian_context',400,0,.01,steps=9,diagnostics=False))
    train_job(testroot/'resumed','P3_gaussian_context',400,0,.01,steps=9,stop_at=4,diagnostics=False)
    b=Path(train_job(testroot/'resumed','P3_gaussian_context',400,0,.01,steps=9,diagnostics=False))
    sa=torch.load(a/'final.pt',weights_only=False);sb=torch.load(b/'final.pt',weights_only=False)
    assert all(torch.equal(sa['model'][k],sb['model'][k]) for k in sa['model'])
    assert (a/'context_interventions.json').is_file()
