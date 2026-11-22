"""Independent metrics: intentionally imports no trainer/evaluation routines."""
import numpy as np
from pathlib import Path
from common import read,sha,write,csv_write

def independent_scores(truth,prediction):
    table=np.zeros((6,6),dtype=np.int64)
    for actual,guessed in zip(truth.tolist(),prediction.tolist()): table[int(actual),int(guessed)]+=1
    f1=[];recall=[]
    for label in range(6):
        tp=int(table[label,label]);fp=int(table[:,label].sum())-tp;fn=int(table[label,:].sum())-tp
        precision=tp/(tp+fp) if tp+fp else 0.
        r=tp/(tp+fn) if tp+fn else 0.
        recall.append(r);f1.append(2*precision*r/(precision+r) if precision+r else 0.)
    supported=np.unique(truth)
    return {'accuracy':float(np.trace(table)/len(truth)),'macro_f1':float(np.mean(f1)),
        'supported_macro_f1':float(np.mean(np.array(f1)[supported])),
        'confusion':table.tolist(),'recall':recall}

def verify_run(run):
    import torch
    run=Path(run);rows=[];n=0;updates=0;pairing={}
    for completion in sorted(run.glob('*/**/completion.json')):
        job=completion.parent
        if not (job/'predictions.npz').exists(): continue
        done=read(completion)
        for name,digest in done['files'].items():
            assert sha(job/name)==digest,(job,name,'hash mismatch')
        logs=read(job/'steps.json')
        assert logs==list(range(done['completed_steps'])),(job,'optimizer log mismatch')
        state=torch.load(job/'final.pt',map_location='cpu',weights_only=False)
        assert state['next_step']==done['completed_steps'] and state['executed']==logs
        key=(job.parent.name,done['seed'],done['fold'])
        signature=(done['base_initial_fingerprint'],done['batch_sha256'],done['mask_sha256'])
        if key in pairing:assert pairing[key]==signature,(job,'pairing mismatch')
        else:pairing[key]=signature
        saved=read(job/'final_metrics.json')
        with np.load(job/'predictions.npz') as p:
            assert set(p['train_subject']).isdisjoint(p['val_subject'])
            for split in ('train','val'):
                for c in range(3):
                    actual=independent_scores(p[split+'_truth'],p[f'{split}_context{c}'])
                    expected=saved['scores'][split][c]
                    for key in ('accuracy','macro_f1','supported_macro_f1'):
                        assert abs(actual[key]-expected[key])<1e-12,(job,key)
                    assert actual['confusion']==expected['confusion']
                    if split=='val':
                        for s in np.unique(p['val_subject']):
                            ix=p['val_subject']==s
                            m=independent_scores(p['val_truth'][ix],p[f'val_context{c}'][ix])
                            rows.append(dict(stage=job.parent.name,job=job.name,condition=done['condition'],seed=done['seed'],fold=done['fold'],
                                context=c,subject=int(s),**{k:v for k,v in m.items() if k not in ('confusion','recall')}))
        n+=1;updates+=done['completed_steps']
    csv_write(run/'analysis'/'verified_subject_scores.csv',rows)
    result={'verified_fits':n,'verified_logical_updates':updates,'independent_metrics':True,'official_test_model_evaluations':0}
    write(run/'verification.json',result)
    return result
