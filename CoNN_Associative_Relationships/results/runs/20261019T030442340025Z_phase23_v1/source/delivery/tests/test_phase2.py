"""Original-policy equivalence and genuine fit/resume verification on HAR development rows."""
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import torch

from conn_assoc.phase23_core import Gates,configure,fold_arrays,forward,make_model,train_job
from conn_assoc.geometry import temporal_input_coordinates
from conn_assoc.paths import EXPERIMENT_ROOT
from conn_exp.har_core import HARGates,geometry_from_coordinates
from conn_exp.gates import draw_masks


def test_phase2_all_controls_match_original_schedule_exactly():
    configure()
    geometry=geometry_from_coordinates(temporal_input_coordinates(),[64,64])
    masks=draw_masks(1152,64,64,400)
    config={'total_steps':3200,'release_fraction':.7,'d50_start':.08,'d50_end':1.}
    for cid,source in [('P2_dense','A'),('P2_gaussian','D'),('P2_rowshuffle','B'),('P2_coordperm','C'),('P2_fixed08','S08'),('P2_fixed25','S25')]:
        original=HARGates(geometry,masks,source,config);new=Gates(cid,400)
        for step in (0,1,159,160,1000,2238,2239,2240,3199):
            for actual,expected in zip(new.build(step),original.build(step)):
                assert torch.equal(actual,expected),(cid,step,(actual-expected).abs().max())


def test_phase2_paired_init_mask_energy_and_optimizer_update():
    configure()
    dense=make_model('P2_dense',400);structured=make_model('P2_gaussian',400)
    for a,b in zip(dense.parameters(),structured.parameters()):assert torch.equal(a,b)
    d=Gates('P2_gaussian',400);b=Gates('P2_rowshuffle',400)
    for step in (0,1600,2239):
        for a,c in zip(d.build(step),b.build(step)):
            assert torch.allclose(a.square().sum(1),c.square().sum(1),atol=.001,rtol=1e-6)
    arrays=fold_arrays(0,[[14,21,29]])
    gates=Gates('P2_gaussian',400);gates.current=gates.build(0)
    before=structured.fc1.weight.detach().clone()
    optimizer=torch.optim.SGD(structured.parameters(),lr=.01)
    x=torch.from_numpy(arrays['train']['x'][:128]);y=torch.tensor(arrays['train']['y'][:128],dtype=torch.long)
    logits=forward(structured,x,gates);assert logits.shape==(128,6)
    loss=torch.nn.functional.cross_entropy(logits,y);loss.backward();optimizer.step()
    assert torch.isfinite(loss) and not torch.equal(before,structured.fc1.weight)


def test_phase2_checkpoint_exact_resume(tmp_path):
    active=Path((EXPERIMENT_ROOT/'config/phase23_active_run.txt').read_text().strip())
    testroot=active/'verification'/('resume_test_'+uuid.uuid4().hex)
    for tag in ('continuous','resumed'):
        run=testroot/tag;run.mkdir(parents=True)
        for name in ('config.frozen.json','protocol_seal.json'):shutil.copy2(active/name,run/name)
    job1=Path(train_job(testroot/'continuous','P2_gaussian',400,0,.01,steps=8,diagnostics=False))
    train_job(testroot/'resumed','P2_gaussian',400,0,.01,steps=8,stop_at=3,diagnostics=False)
    job2=Path(train_job(testroot/'resumed','P2_gaussian',400,0,.01,steps=8,diagnostics=False))
    a=torch.load(job1/'final.pt',weights_only=False);b=torch.load(job2/'final.pt',weights_only=False)
    for key in a['model']:assert torch.equal(a['model'][key],b['model'][key])
    assert a['next_step']==b['next_step']==8 and a['schedule']==b['schedule']
    assert np.array_equal(a['batches'],b['batches'])
    assert json.loads((job1/'completion.json').read_text())['checkpoint_reload_exact']
