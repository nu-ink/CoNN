from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from conn_exp04.core import (Config, Gates, CONDITIONS, TASKS, generate_split, target_score,
    stream, arrays_hash, metrics, CoNNMLP, initial_state_dict, fingerprint_state, trainable_count)
from conn_exp04 import runner
from conn_exp04.analysis import paired, holm, analyze
from conn_exp.gates import gaussian_gate
from conn_exp.models import GatedLinear


@pytest.fixture
def cfg():
    return Config(stage='smoke', seeds=(9000,), n_train=32, n_val=16, n_test=32,
                  total_steps=12, eval_interval=6).validate()


def test_distance_and_gaussian(cfg):
    for d in Gates(cfg, 1).distances:
        assert (d >= 0).all() and torch.isfinite(d).all()
    assert [tuple(d.shape) for d in Gates(cfg, 1).distances] == [(32,100),(32,32)]
    g, _ = gaussian_gate(torch.tensor([[0., .1, .2, .5, 1.]]), .2)
    assert g[0,0] == 1 and ((g >= 0) & (g <= 1)).all()
    assert (g[:,1:] < g[:,:-1]).all()


@pytest.mark.parametrize('seed', [1,2,3])
def test_gate_invariants_all_steps(cfg, seed):
    gates = Gates(cfg, seed)
    assert gates.audit()['row_multisets_exact']
    for c in CONDITIONS:
        assert all(torch.equal(g, torch.ones_like(g)) for g in gates.at(c, cfg.release_step))
    for a,b in zip(gates.at('scrambled',0), Gates(cfg,seed).at('scrambled',0)):
        assert torch.equal(a,b)
    assert any(not torch.equal(a,b) for a,b in zip(gates.at('aligned',0),gates.at('scrambled',0)))


def test_schedule_endpoints(cfg):
    gates = Gates(cfg, 1)
    from conn_exp.schedules import EXACT_FACTOR
    for t, d50 in ((0,cfg.d50_start),(cfg.release_step-1,cfg.d50_end)):
        for d, g in zip(gates.distances, gates.at('aligned',t)):
            assert torch.equal(g, gaussian_gate(d,d50/EXACT_FACTOR)[0])
    with pytest.raises(ValueError): gates.at('bad',0)
    with pytest.raises(ValueError): gates.at('aligned',cfg.total_steps)


def test_smoke_audit_float32_reduction_roundoff():
    # Seed/schedule of the preserved first smoke attempt, which exposed a
    # float32 reduction tolerance failure before any model was fitted.
    c=Config(stage='smoke',seeds=(3900,),total_steps=40)
    result=Gates(c,3900).audit()
    assert result['max_row_energy_abs_error'] < 1e-12
    assert result['max_uniform_energy_abs_error'] < 2e-7


def test_effective_weight_and_finite_difference():
    layer = GatedLinear(3,2).double()
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[.2,-.1,.3],[-.4,.5,.1]]))
        layer.bias.zero_()
    x = torch.tensor([[.3,.7,-.2],[-.6,.4,.9]],dtype=torch.double)
    g = torch.tensor([[1.,.5,.1],[.2,.4,.8]],dtype=torch.double,requires_grad=True)
    out, eff = layer(x,g)
    assert torch.equal(eff, layer.weight*g.detach())
    assert torch.equal(out,F.linear(x,layer.weight*g.detach(),layer.bias))
    loss = out.square().mean()
    loss.backward()
    assert g.grad is None
    numerical = torch.empty_like(layer.weight)
    eps = 1e-6
    for j in range(2):
        for i in range(3):
            with torch.no_grad(): layer.weight[j,i] += eps
            plus = float(layer(x,g)[0].square().mean())
            with torch.no_grad(): layer.weight[j,i] -= 2*eps
            minus = float(layer(x,g)[0].square().mean())
            with torch.no_grad(): layer.weight[j,i] += eps
            numerical[j,i] = (plus-minus)/(2*eps)
    assert torch.allclose(layer.weight.grad,numerical,atol=1e-9,rtol=1e-6)


@pytest.mark.parametrize('kwargs', [dict(n_features=99),dict(groups=0),dict(hidden=(32,)),
    dict(lr=0),dict(lr=float('nan')),dict(feature_noise=-1),dict(label_flip=.5),
    dict(total_steps=5),dict(seeds=(1,1)),dict(release_fraction=1),dict(d50_end=.01),dict(batch_size=0)])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError): replace(Config(),**kwargs).validate()


@pytest.mark.parametrize('task', TASKS)
def test_data_reproducibility_and_independence(cfg,task):
    a = generate_split(cfg,1,'train',task)
    b = generate_split(cfg,1,'train',task)
    c = generate_split(cfg,1,'test',task)
    assert a['x'].shape == (cfg.n_train,100)
    assert arrays_hash(a['x'],a['y']) == arrays_hash(b['x'],b['y'])
    assert not np.array_equal(a['x'],c['x'])
    assert not set(map(bytes,a['x'])).intersection(map(bytes,c['x']))
    assert np.array_equal(a['clean'],target_score(a['z']) > 0)
    assert np.array_equal(a['y'],np.logical_xor(a['clean'],a['flipped']))


def test_known_relationship_and_noise(cfg):
    noiseless=replace(cfg,feature_noise=0,label_flip=0,n_train=10000)
    a=generate_split(noiseless,1,'train','structured')
    reshaped=a['x'].reshape(-1,5,2,10)
    assert np.allclose(reshaped[:,:,:,0],a['z'])
    assert np.array_equal(reshaped[:,:,:,0],reshaped[:,:,:,9])
    assert .47 < a['y'].mean() < .53
    assert not a['flipped'].any()
    noisy=generate_split(replace(noiseless,feature_noise=.3,label_flip=.2),1,'train','structured')
    assert np.array_equal(a['z'],noisy['z'])
    assert .18 < noisy['flipped'].mean() < .22
    assert .28 < np.std(noisy['x']-a['x']) < .32
    # Independent latent coordinates within a group: interaction, not a shared label shortcut.
    assert abs(np.corrcoef(a['z'][:,0,0],a['z'][:,0,1])[0,1]) < .04
    # Sign inversion of a alone flips the score; this is an actual interaction.
    z=a['z'].copy();z[:,:,0]*=-1
    assert np.allclose(target_score(z),-target_score(a['z']))


def test_negative_control_and_no_label_leakage(cfg):
    a=generate_split(cfg,7,'train','structured')
    b=generate_split(cfg,7,'train','negative')
    c=generate_split(cfg,7,'val','negative')
    assert np.array_equal(a['y'],b['y'])
    assert np.array_equal(a['x'][:,b['permutation']],b['x'])
    assert np.array_equal(b['permutation'],c['permutation'])
    d=generate_split(replace(cfg,label_flip=.49),7,'train','negative')
    assert np.array_equal(b['x'],d['x'])
    assert np.array_equal(b['permutation'],d['permutation'])
    # Adjacent repeated measurements are dispersed by the negative control.
    groups=(b['permutation']//20)
    assert np.mean(groups[1:]==groups[:-1]) < .4


def test_metrics_known_counts():
    result=metrics(torch.tensor([1.,1.,-1.,-1.]),torch.tensor([1.,0.,1.,0.]))
    assert result['accuracy']==.5 and result['f1']==.5
    assert metrics(torch.zeros(2),torch.zeros(2))['accuracy']==1


def test_initialization_and_counts(cfg):
    hashes=[]
    initial=initial_state_dict(100,(32,32),stream(1,'init'))
    for _ in CONDITIONS:
        model=CoNNMLP(100,(32,32));model.load_state_dict(initial)
        assert trainable_count(model)==4321
        hashes.append(fingerprint_state(model.state_dict()))
    assert len(set(hashes))==1


def test_statistics_known_values():
    result=paired([.1]*10,'greater')
    assert result['exact_signflip_p']==1/1024
    assert paired([0.]*10,'greater')['exact_signflip_p']==1
    assert paired([-.1]*10,'greater')['exact_signflip_p']==1
    assert holm([.01,.04,.03])==pytest.approx([.03,.06,.06])
    assert paired([1.,2.,3.])['mean_difference']==2
    with pytest.raises(ValueError): paired([1.])


def test_complete_run_and_replay(cfg,tmp_path):
    run=runner.train_run(cfg,tmp_path)
    assert json.loads((run/'status.json').read_text())['state']=='trained'
    assert not (run/'evaluation').exists()
    for task in TASKS:
        records=[json.loads((run/f'seed_9000'/task/c/'provenance.json').read_text()) for c in CONDITIONS]
        assert len({r['batch_sha256'] for r in records})==1
        assert len({r['initialization_sha256'] for r in records})==1
        assert all(r['data_hashes']==records[0]['data_hashes'] for r in records)
        assert all(r['optimizer']==records[0]['optimizer'] for r in records)
        for c in CONDITIONS:
            curves=json.loads((run/'seed_9000'/task/c/'curves.json').read_text())
            assert all(np.isfinite(r['val']['loss']) for r in curves)
    rows=runner.evaluate_run(run)
    assert len(rows)==24
    with pytest.raises(ValueError): runner.evaluate_run(run)
    other=runner.train_run(cfg,tmp_path)
    assert run!=other
    for task in TASKS:
        for c in CONDITIONS:
            a=torch.load(run/'seed_9000'/task/c/'final.pt',weights_only=True)
            b=torch.load(other/'seed_9000'/task/c/'final.pt',weights_only=True)
            assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])


@pytest.mark.parametrize('exception,expected',[(RuntimeError('injected failure'),'failed'),(KeyboardInterrupt(),'interrupted')])
def test_failure_and_interruption_recorded(cfg,tmp_path,monkeypatch,exception,expected):
    def fail(*args,**kwargs): raise exception
    monkeypatch.setattr(runner,'evaluate',fail)
    with pytest.raises(type(exception)):
        runner.train_run(cfg,tmp_path)
    run=next(tmp_path.iterdir())
    assert json.loads((run/'status.json').read_text())['state']==expected
    statuses=[json.loads(p.read_text()) for p in run.rglob('status.json')]
    assert sum(s['state']==expected for s in statuses)==2
    assert not list(run.rglob('final.pt'))
