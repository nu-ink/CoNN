from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
import pytest
from models import Model,gates_for,support,PERMUTATIONS,configure,parameter_groups
from diagnostics import snapshot,changes
from preflight import mathematical_tests
from conn_assoc.phase23_core import context_plan,paired_batches
from conn_assoc.phase3_context import mask_input,ContextHARMLP
from verify import independent_scores
from evaluation import metric

@pytest.fixture(autouse=True)
def deterministic():configure('cpu')

def test_analytic_and_finite_difference():assert mathematical_tests()['central_finite_difference_passed']

@pytest.mark.parametrize('context',[0,1,2])
def test_live_mask(context):
    x=torch.ones(2,1152);masked=mask_input(x,context)
    assert torch.equal(masked[0].bool(),support(context))

@pytest.mark.parametrize('condition',['dense_C0','gaussian_C1','gaussian_C2','gaussian_C3','gaussian_C4','gaussian_frozen'])
def test_pairing_dtype_and_bounds(condition):
    model=Model(condition);baseline=Model('dense_masked')
    for a,b in zip(parameter_groups(model)[0],parameter_groups(baseline)[0]):assert torch.equal(a,b)
    assert model.context_ih.u.dtype==(torch.float64 if model.variant in ('C3','C4') else torch.float32)
    assert bool((model.context_ih(0)>=.05).all() and (model.context_ih(0)<=1).all())

def test_base_initialization_independent():
    base=Model('dense_masked').state_dict()
    for c in ('dense_C0','gaussian_C1','gaussian_C3','gaussian_C4'):
        model=Model(c)
        for n,p in base.items():assert torch.equal(p,model.state_dict()[n])
        for factor in (model.context_ih,model.context_hh):
            for ctx in range(3): assert bool((factor(ctx)>=.05).all() and (factor(ctx)<=1).all())

def test_historical_C0_identical():
    old=ContextHARMLP();new=Model('dense_C0');old.load_state_dict(new.state_dict())
    x=torch.linspace(-1,1,2*1152).reshape(2,1152)
    g=gates_for('dense_C0',400,'cpu').build(0)
    for c in range(3):assert torch.equal(old.contextual(mask_input(x,c),None,None,c),new.run(x,g,c))

def test_frozen_nonidentity_and_parameter_counts():
    model=Model('gaussian_frozen');base,ctx=parameter_groups(model)
    assert len(ctx)==4 and all(not p.requires_grad for p in ctx)
    assert sum(p.numel() for p in model.parameters())==86406
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==78342
    assert not torch.equal(model.context_ih(0),torch.ones(64,1152))

def test_factor_initialization_canonical():
    a=Model('gaussian_C4',400);b=Model('dense_C4',404)
    assert a.context_ih.u.dtype==torch.float64
    assert torch.equal(a.context_ih.u,b.context_ih.u)
    assert not torch.equal(a.context_ih.u[:,:,0],a.context_ih.u[:,:,1])
    c=Model('gaussian_C1');d=Model('gaussian_C0')
    assert abs(float(c.context_ih(0).double().mean()-d.context_ih(0).double().mean()))<.0005

def test_code_and_mask_independence():
    model=Model('gaussian_C4');g=gates_for('gaussian_C4',400,'cpu').build(0)
    x=torch.ones(2,1152);damaged=x.clone();damaged[:,~support(1)]=12345
    for perm in PERMUTATIONS:
        assert torch.equal(model.run(x,g,1,perm[1]),model.run(damaged,g,1,perm[1]))
    masks=context_plan(400,0);codes=masks.copy();np.random.default_rng(6000411).shuffle(codes)
    assert np.array_equal(np.sort(codes),np.sort(masks))
    assert not np.array_equal(codes,masks)

def test_independent_metric_edge_cases():
    truth=np.array([0,0,1,1]);pred=np.array([0,2,2,2])
    a=independent_scores(truth,pred);b=metric(truth,pred)
    assert a['macro_f1']==pytest.approx(b['macro_f1'])
    assert a['supported_macro_f1']==pytest.approx(1/3)
    assert a['recall'][1]==0

@pytest.mark.parametrize('condition',['gaussian_C0','gaussian_C3','gaussian_C4'])
def test_exact_checkpoint_optimizer_resume(tmp_path,condition):
    model=Model(condition);other=Model(condition)
    g=gates_for(condition,400,'cpu').build(0)
    x=torch.linspace(-1,1,128*1152).reshape(128,1152);y=torch.arange(128)%6
    opt=torch.optim.SGD(model.parameters(),lr=.1,foreach=False)
    def step(m,o):
        o.zero_grad(set_to_none=True);torch.nn.functional.cross_entropy(m.run(x,g,0),y).backward();o.step()
    step(model,opt)
    path=tmp_path/'checkpoint.pt';torch.save({'model':model.state_dict(),'optimizer':opt.state_dict()},path)
    st=torch.load(path,weights_only=False);other.load_state_dict(st['model'])
    oo=torch.optim.SGD(other.parameters(),lr=.1,foreach=False);oo.load_state_dict(st['optimizer'])
    step(model,opt);step(other,oo)
    for n,p in model.state_dict().items():assert torch.equal(p,other.state_dict()[n])
    assert torch.equal(model.run(x,g,0),other.run(x,g,0))
