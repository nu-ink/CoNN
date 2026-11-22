from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from conn_exp.experiment1b import sha
from conn_exp.experiment2 import select_validation
from conn_exp.gates import draw_masks
from conn_exp.har_core import HARMLP,HARGates,classification_metrics,geometry_from_coordinates,gradient_measurement,initial_state,paired_batches,train
from conn_exp.har_data import fit_scaler,transform,subject_partition,feature_layout
from conn_exp.runtime import configure_runtime


def config():
    return dict(hidden_sizes=[8,8],total_steps=12,eval_interval=4,batch_size=8,
        release_fraction=.7,d50_start=.08,d50_end=1.,conditions=['A','S08','S25','D','B','C'],
        representations=['engineered','temporal'],learning_rates=[.01,.05,.1],tuning_seeds=[30,31])


def test_scaler_train_only_and_temporal_channel_statistics():
    train_values=np.array([[1.,2.,4.],[3.,6.,4.]])
    mean,scale,constant=fit_scaler(train_values,'engineered')
    assert np.allclose(mean,[2,4,4]) and constant.tolist()==[False,False,True]
    assert np.allclose(transform(train_values,mean,scale,'engineered').mean(0),0)
    validation=np.array([[100.,200.,4.]])
    transformed=transform(validation,mean,scale,'engineered')
    assert transformed[0,0]==98 and np.allclose(mean,[2,4,4])
    raw=np.random.default_rng(1).normal(size=(4,128*9))
    m,s,c=fit_scaler(raw,'temporal')
    output=transform(raw,m,s,'temporal').reshape(4,128,9)
    assert np.allclose(output.mean((0,1)),0,atol=1e-6)
    assert np.allclose(output.std((0,1)),1,atol=1e-6)


def test_subjects_disjoint_and_coordinates_use_metadata_only():
    fit,val,test=subject_partition(np.arange(1,22),np.arange(22,31))
    assert len(fit)==16 and len(val)==5 and len(test)==9
    assert not (set(fit)&set(val) or set(fit)&set(test) or set(val)&set(test))
    with pytest.raises(ValueError):
        subject_partition(np.arange(10),np.arange(5,15))
    engineered=feature_layout(['duplicate','duplicate','other'],'engineered')
    assert len(engineered)==3 and engineered.coordinate.tolist()==[0,.5,1]
    temporal=feature_layout([],'temporal')
    assert len(temporal)==1152 and temporal.groupby('sample').coordinate.nunique().eq(1).all()
    assert temporal.groupby('sample').size().eq(9).all()


def test_macro_f1_and_dense_forward_equivalence():
    result=classification_metrics([0,0,1,1,2,2],[0,1,1,1,0,2],3)
    assert result['accuracy']==pytest.approx(4/6)
    assert result['macro_f1']==pytest.approx((.5+.8+2/3)/3)
    model=HARMLP(12,[8,8]);model.load_state_dict(initial_state(12,[8,8],30))
    x=torch.randn(5,12)
    a,_=model(x)
    b,_=model(x,torch.ones(8,12),torch.ones(8,8))
    assert a.shape==(5,6) and torch.equal(a,b)


def test_gate_controls_energy_and_multiclass_gradient_chain_rule():
    configure_runtime();cfg=config();geo=geometry_from_coordinates(np.linspace(0,1,12),[8,8])
    masks=draw_masks(12,8,8,30)
    gates={c:HARGates(geo,masks,c,cfg) for c in cfg['conditions']}
    for step in range(12):
        d=gates['D'].build(step)
        for cid in ['B','C']:
            for native,control in zip(d,gates[cid].build(step)):
                assert torch.allclose(native.square().sum(1),control.square().sum(1),atol=1e-5)
    model=HARMLP(12,[8,8]);model.load_state_dict(initial_state(12,[8,8],30))
    ih,hh=gates['D'].build(0)
    gradients=gradient_measurement(model,torch.randn(6,12),torch.arange(6),ih,hh,False)
    for layer,gate in [('ih',ih),('hh',hh)]:
        assert torch.allclose(gradients[layer]['dL_dW'],gradients[layer]['dL_dW_eff']*gate,atol=1e-7)


@pytest.mark.parametrize('cid',['A','S08','S25','D','B','C'])
def test_training_needs_no_test_arrays_until_enabled(tmp_path,cid):
    configure_runtime();cfg=config();rng=np.random.default_rng(3)
    arrays=dict(x_train=rng.normal(size=(24,12)).astype('float32'),y_train=np.arange(24)%6,
        x_val=rng.normal(size=(12,12)).astype('float32'),y_val=np.arange(12)%6)
    init=initial_state(12,[8,8],30);masks=draw_masks(12,8,8,30)
    geo=geometry_from_coordinates(np.linspace(0,1,12),[8,8]);batches=paired_batches(24,12,8,30)
    job=tmp_path/cid
    score=train(cfg,'unit_fixture',cid,30,.05,arrays,init,masks,geo,batches,job)
    assert score['test_evaluations']==0 and 'test_accuracy' not in score
    assert score['config_sha256']==sha(job/'config.resolved.json')
    pre=torch.load(job/'pre_release.pt',weights_only=False);post=torch.load(job/'post_release.pt',weights_only=False)
    assert all(torch.equal(v,post['model'][k]) for k,v in pre['model'].items())
    assert np.isfinite(pd.read_csv(job/'gradients.csv').mean_abs).all()


def test_final_test_metrics_reuse_predictions_and_pairing(tmp_path):
    configure_runtime();cfg=config();rng=np.random.default_rng(4)
    arrays=dict(x_train=rng.normal(size=(24,12)).astype('float32'),y_train=np.arange(24)%6,
        x_val=rng.normal(size=(12,12)).astype('float32'),y_val=np.arange(12)%6,
        x_test=rng.normal(size=(12,12)).astype('float32'),y_test=np.arange(12)%6,subject_test=np.repeat([90,91],6))
    init=initial_state(12,[8,8],30);masks=draw_masks(12,8,8,30)
    batches=paired_batches(24,12,8,30)
    assert np.array_equal(batches,paired_batches(24,12,8,30))
    score=train(cfg,'unit_fixture','A',30,.05,arrays,init,masks,geometry_from_coordinates(np.linspace(0,1,12),[8,8]),batches,tmp_path/'a',True)
    pred=np.load(tmp_path/'a/test_predictions.npz')
    expected=classification_metrics(pred['truth'],pred['prediction'])
    assert score['test_evaluations']==1
    assert score['test_macro_f1']==expected['macro_f1'] and score['test_accuracy']==expected['accuracy']
    assert score['train_test_gap_accuracy']==pytest.approx(score['train_accuracy']-score['test_accuracy'])


def test_equal_selection_budgets_and_test_independence():
    cfg=config()
    rows=[dict(representation=r,condition=c,lr=lr,seed=s,val_macro_f1=.8 if lr==.05 else .7,
        val_loss=.4,test_accuracy=1 if lr==.1 else 0) for r in cfg['representations'] for c in cfg['conditions']
        for lr in cfg['learning_rates'] for s in cfg['tuning_seeds']]
    frame=pd.DataFrame(rows)
    _,rates=select_validation(frame,cfg)
    assert all(lr==.05 for r in rates.values() for lr in r.values())
    frame.test_accuracy=np.nan
    assert select_validation(frame,cfg)[1]==rates
    with pytest.raises(ValueError,match='Incomplete'):
        select_validation(frame.iloc[1:],cfg)
