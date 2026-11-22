"""Trainer-side summaries; independent verification lives in verify.py."""
import numpy as np
import torch
from torch.nn import functional as F

def metric(y,p):
    cm=np.bincount(6*y.astype(int)+p.astype(int),minlength=36).reshape(6,6)
    tp=np.diag(cm); den=cm.sum(0)+cm.sum(1)
    f=np.divide(2*tp,den,out=np.zeros(6),where=den>0)
    present=cm.sum(1)>0
    return {'accuracy':float(tp.sum()/len(y)),'macro_f1':float(f.mean()),
        'supported_macro_f1':float(f[present].mean()),'confusion':cm.tolist()}

def evaluate(model,arrays,gates,codes=None,disabled=False,random_seed=None):
    results=[];subjects=[];preds={}
    with torch.no_grad():
        for mask in range(3):
            predictions=[];loss=0
            randomized=None
            if random_seed is not None:
                randomized=np.resize(np.arange(3),len(arrays['y']))
                np.random.default_rng(random_seed+mask).shuffle(randomized)
            for start in range(0,len(arrays['y']),512):
                x=arrays['xt'][start:start+512];y=arrays['yt'][start:start+512]
                code=mask if codes is None else codes[mask]
                if randomized is None: logits=model.run(x,gates,mask,code,disabled)
                else:
                    logits=x.new_empty((len(x),6))
                    for c in range(3):
                        ix=np.where(randomized[start:start+len(x)]==c)[0]
                        if len(ix): logits[ix]=model.run(x[ix],gates,mask,c,disabled)
                loss+=float(F.cross_entropy(logits,y,reduction='sum'))
                predictions.append(logits.argmax(1).cpu().numpy())
            pred=np.concatenate(predictions);preds[f'context{mask}']=pred
            results.append(dict(context=mask,loss=loss/len(pred),**metric(arrays['y'],pred)))
            for s in np.unique(arrays['subject']):
                ix=arrays['subject']==s
                subjects.append(dict(context=mask,subject=int(s),windows=int(ix.sum()),**metric(arrays['y'][ix],pred[ix])))
    return results,subjects,preds
