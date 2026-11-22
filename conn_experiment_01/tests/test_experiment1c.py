import json
import numpy as np
import pandas as pd
import pytest
import torch

from conn_exp.data import generate_dataset
from conn_exp.experiment1b import ValidationOnly, sha
from conn_exp.experiment1c import select_shared_lr
from conn_exp.experiment1c_core import POLICIES, PolicyGates, Variant, alignment_permutation, train_variant, variants
from conn_exp.gates import GateBuilder, draw_masks
from conn_exp.geometry import make_task_geometry
from conn_exp.models import CoNNMLP, initial_state_dict, build_optimizer
from conn_exp.runtime import configure_runtime
from conn_exp.train import train_condition, minibatch_order
from tests.helpers import tiny_config


def test_policy_endpoints_and_exact_release_are_distinct():
    cfg = tiny_config()
    for step in range(cfg.total_steps):
        assert POLICIES['S'].at(step, cfg) == (cfg.tau_start, False)
        assert POLICIES['F'].at(step, cfg)[1] is False
    for p in ['P', 'W', 'D']:
        assert POLICIES[p].at(0, cfg) == (cfg.tau_start, False)
    assert POLICIES['W'].at(cfg.release_step-1, cfg) == POLICIES['W'].at(cfg.release_step, cfg)
    assert POLICIES['D'].at(cfg.release_step, cfg) == (None, True)
    assert POLICIES['P'].at(cfg.release_step-1, cfg) == POLICIES['P'].at(cfg.total_steps-1, cfg)


def test_degradation_moves_declared_fraction_and_preserves_each_row_energy():
    cfg = tiny_config()
    geo = make_task_geometry(100, cfg.hidden_sizes)
    masks = draw_masks(100, *cfg.hidden_sizes, 10)
    for q in [0, .5, 1]:
        perm = alignment_permutation(100, 10, q)
        assert int((perm != torch.arange(100)).sum()) == int(100*q)
        for policy in POLICIES:
            native = PolicyGates(geo, masks, Variant(policy), 10)
            shifted = PolicyGates(geo, masks, Variant(policy, False, q), 10)
            shuffled = PolicyGates(geo, masks, Variant(policy, True), 10)
            for step in [0, cfg.release_step-1, cfg.release_step, cfg.total_steps-1]:
                a, b, c = native.build(step, cfg), shifted.build(step, cfg), shuffled.build(step, cfg)
                for gi, gj, gk in zip(a, b, c):
                    assert torch.allclose(gi.square().sum(1), gj.square().sum(1), atol=1e-4)
                    assert torch.allclose(gi.square().sum(1), gk.square().sum(1), atol=1e-4)
    assert torch.equal(alignment_permutation(100, 10, .5), alignment_permutation(100, 10, .5))


@pytest.mark.parametrize('legacy,variant', [('D',Variant('D')), ('S',Variant('S')), ('B_prime',Variant('D',True))])
def test_new_training_matches_legacy_updates_exactly(tmp_path, legacy, variant):
    configure_runtime()
    cfg, seed = tiny_config(), 10
    data = generate_dataset(cfg, seed)
    init = initial_state_dict(100, cfg.hidden_sizes, seed)
    masks = draw_masks(100, *cfg.hidden_sizes, seed)
    geo = make_task_geometry(100, cfg.hidden_sizes)
    order = minibatch_order(cfg.n_train, seed)
    model = CoNNMLP(100, cfg.hidden_sizes)
    model.load_state_dict(init)
    optimizer, hparams = build_optimizer(model, cfg.lr)
    train_condition(model=model, optimizer=optimizer, optimizer_hparams=hparams, config=cfg,
        gate_builder=GateBuilder(geo,masks,legacy,cfg.tau_floor), order=order,
        x_train=torch.from_numpy(data.train.x),y_train=torch.from_numpy(data.train.y),
        x_val=torch.from_numpy(data.val.x),y_val=torch.from_numpy(data.val.y),
        motif_a_train=data.train.motif_a,motif_b_train=data.train.motif_b,
        motif_a_val=data.val.motif_a,motif_b_val=data.val.motif_b,test_access=ValidationOnly(),
        base_dist_ih=geo.dist_ih,base_dist_hh=geo.dist_hh,seed=seed,condition_id=legacy,
        run_id='unit',data_fingerprint='',init_fingerprint='',diagnostics=False,evaluate_test=False)
    job = tmp_path/variant.id
    score, actual = train_variant(cfg,variant,seed,init,masks,geo,order,data,job,diagnostics=False)
    assert score['test_evaluations'] == 0 and 'test_accuracy' not in score
    for key, expected in model.state_dict().items():
        assert torch.equal(expected,actual[key]), key
    assert score['config_sha256'] == sha(job/'config.resolved.json')
    a = torch.load(job/'pre_release.pt',weights_only=False)
    b = torch.load(job/'post_release.pt',weights_only=False)
    assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())


def test_shared_selection_equal_budget_and_test_independence():
    rows = [dict(task=task,variant_id=v.id,lr=lr,seed=seed,alignment_fraction=v.alignment_fraction,
        val_accuracy=.8 if lr==.05 else .7,val_loss=.4,test_accuracy=1 if lr==.1 else 0)
        for task in ['and','xor'] for v in variants(task) for lr in [.01,.05,.1] for seed in [10,11]]
    frame = pd.DataFrame(rows)
    assert select_shared_lr(frame,seeds=[10,11])[1] == {'and':.05,'xor':.05}
    frame.test_accuracy = np.nan
    assert select_shared_lr(frame,seeds=[10,11])[1] == {'and':.05,'xor':.05}
    with pytest.raises(ValueError,match='Incomplete'):
        select_shared_lr(frame.iloc[1:],seeds=[10,11])


def test_diagnostic_gradients_and_terminal_checkpoint_inference(tmp_path):
    configure_runtime()
    cfg, seed = tiny_config(), 11
    data = generate_dataset(cfg,seed)
    init = initial_state_dict(100,cfg.hidden_sizes,seed)
    geo = make_task_geometry(100,cfg.hidden_sizes)
    masks = draw_masks(100,*cfg.hidden_sizes,seed)
    score, _ = train_variant(cfg,Variant('P'),seed,init,masks,geo,minibatch_order(cfg.n_train,seed),data,tmp_path/'p')
    grads = pd.read_csv(tmp_path/'p/gradients.csv')
    assert set(grads.quantity) == {'dL_dW','dL_dW_eff'}
    assert np.isfinite(grads.mean_abs).all()
    comparisons = pd.read_csv(tmp_path/'p/checkpoint_comparisons.csv')
    assert len(comparisons)==6 and set(comparisons.split)=={'val'}
    assert comparisons.groupby('kind').weight_fingerprint.nunique().eq(1).all()
    assert not score['test_evaluations']
