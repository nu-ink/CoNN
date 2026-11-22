import json
from pathlib import Path
import torch
from conn_assoc.phase23_core import configure,fold_arrays,make_model,Gates,forward,restore_rng,fingerprint
root=Path.cwd();run=Path((root/'config/phase23_active_run.txt').read_text().strip())
e=run/'recovery_evidence/checkpoint_replace_01'
a=torch.load(e/'latest.before.pt',map_location='cpu',weights_only=False)
b=torch.load(e/'latest.uncommitted.tmp',map_location='cpu',weights_only=False)
configure();model=make_model('P2_rowshuffle',403);model.load_state_dict(a['model'])
opt=torch.optim.SGD(model.parameters(),lr=.1,foreach=False);opt.load_state_dict(a['optimizer']);restore_rng(a['rng'])
folds=json.loads((run/'config.frozen.json').read_text())['folds'];data=fold_arrays(5,folds)['train']
x=torch.from_numpy(data['x']);y=torch.tensor(data['y'],dtype=torch.long);g=Gates('P2_rowshuffle',403)
for step in range(a['next_step'],b['next_step']):
    g.current=g.build(step);index=torch.tensor(a['batches'][step]);opt.zero_grad(set_to_none=True)
    loss=torch.nn.functional.cross_entropy(forward(model,x[index],g),y[index]);loss.backward();opt.step()
exact=all(torch.equal(model.state_dict()[k],b['model'][k]) for k in b['model'])
assert exact
(e/'replay_verification.json').write_text(json.dumps({'committed_step':a['next_step'],'uncommitted_step':b['next_step'],'replayed_optimizer_updates':b['next_step']-a['next_step'],'replayed_weights_bitwise_equal_to_preserved_uncommitted_checkpoint':exact,'scope':'recovery verification excluded from tuning selection'},indent=2)+'\n')
print('Recovery replay exact:',a['next_step'],'to',b['next_step'])
