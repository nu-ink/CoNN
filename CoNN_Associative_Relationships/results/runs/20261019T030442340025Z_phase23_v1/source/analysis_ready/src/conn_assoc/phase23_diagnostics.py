"""Measured checkpoint interventions and figures on development folds only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch

from conn_assoc.phase23_core import Gates,configure,evaluate,fingerprint,fold_arrays,json_write,make_model


def run_diagnostics(run):
    configure()
    config=json.loads((run/'config.frozen.json').read_text())
    selected=json.loads((run/'selection.frozen.json').read_text())['selection']
    selected={r['condition']:r['selected_lr'] for r in selected}
    out=run/'plots';out.mkdir(exist_ok=True)
    output=run/'analysis';output.mkdir(exist_ok=True)
    release=[];context_rows=[];interventions=[];precision=[]
    for condition,lr in selected.items():
        for seed in config['seeds']:
            for fold in range(7):
                job=run/'jobs'/f'{condition}_seed{seed}_fold{fold}_lr{lr:g}'
                if condition.endswith('context'):
                    for r in json.loads((job/'context_interventions.json').read_text()):
                        flat={k:v for k,v in r.items() if k not in ('native','disabled')}
                        if 'native' in r:
                            for metric in ('accuracy','macro_f1','loss'):
                                flat['native_'+metric]=r['native'][metric]
                                flat['disabled_'+metric]=r['disabled'][metric]
                                flat[metric+'_native_minus_disabled']=r['native'][metric]-r['disabled'][metric]
                        interventions.append(dict(condition=condition,seed=seed,fold=fold,**flat))
                # Reuse measured pre/post rows; these share exactly the same saved weights.
                history=json.loads((job/'history.json').read_text())
                for r in history:
                    if r['kind'] in ('pre_release','post_release','final'):
                        release.append(dict(condition=condition,seed=seed,fold=fold,**r))
                if condition.endswith('context'):
                    state=torch.load(job/'final.pt',map_location='cpu',weights_only=False)['model']
                    initial=torch.load(job/'initial.pt',map_location='cpu',weights_only=False)['model']
                    model=make_model(condition,seed);model.load_state_dict(state)
                    for layer in ('ih','hh'):
                        factor=getattr(model,'context_'+layer)
                        for c in range(3):
                            u=state['context_'+layer+'.u'][c];v=state['context_'+layer+'.v'][c]
                            u0=initial['context_'+layer+'.u'][c];v0=initial['context_'+layer+'.v'][c]
                            with torch.no_grad():
                                matrix=factor(c).numpy();reference=1.-.475*torch.sigmoid(torch.tensor(-6.)).item()
                            precision.append({'condition':condition,'seed':seed,'fold':fold,'layer':layer,'context':c,
                                'u_changed_elements':int((u!=u0).sum()),'v_changed_elements':int((v!=v0).sum()),
                                'u_delta_norm':float((u-u0).norm()),'v_delta_norm':float((v-v0).norm()),
                                'u_rank_columns_max_abs_difference':float((u[:,0]-u[:,1]).abs().max()),
                                'v_rank_columns_max_abs_difference':float((v[:,0]-v[:,1]).abs().max()),
                                'modulation_min':float(matrix.min()),'modulation_max':float(matrix.max()),
                                'modulation_std':float(matrix.std()),'max_abs_change_from_initial_C':float(abs(matrix-reference).max())})
    release_frame=pd.DataFrame(release)
    release_frame.to_csv(output/'release_and_final_comparisons.csv',index=False)
    valid=release_frame[release_frame.split=='val']
    wide=valid.pivot_table(index=['condition','seed','fold','context'],columns='kind',values=['loss','accuracy','macro_f1'])
    changes=[]
    for keys,row in wide.iterrows():
        condition,seed,fold,context=keys
        change={'condition':condition,'seed':seed,'fold':fold,'context':context}
        for metric in ('loss','accuracy','macro_f1'):
            change[metric+'_immediate_gate_switch']=row[(metric,'post_release')]-row[(metric,'pre_release')]
            change[metric+'_further_960_updates']=row[(metric,'final')]-row[(metric,'post_release')]
        changes.append(change)
    changes=pd.DataFrame(changes)
    changes.to_csv(output/'release_effect_differences.csv',index=False)
    plotted=changes.groupby('condition')[['macro_f1_immediate_gate_switch','macro_f1_further_960_updates']].mean()
    fig,ax=plt.subplots(figsize=(10,5));(100*plotted).plot.barh(ax=ax)
    ax.set(xlabel='Validation macro-F1 difference (percentage points)',title='Gate switch at unchanged weights vs subsequent optimization')
    ax.legend(['Immediate release (same weights)','Further 960 updates'],fontsize=7)
    fig.tight_layout();fig.savefig(out/'release_vs_optimization.png',dpi=160);plt.close(fig)
    pd.DataFrame(interventions).to_csv(output/'context_interventions.csv',index=False)
    pd.DataFrame(precision).to_csv(output/'context_precision_and_symmetry.csv',index=False)
    # Representative images are explicitly fixed seed 400/fold 0, never chosen by performance.
    conditions=['P2_dense','P2_gaussian','P2_rowshuffle','P2_coordperm','P2_fixed08','P2_fixed25']
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    weights=[]
    for condition in conditions:
        job=run/'jobs'/f'{condition}_seed400_fold0_lr{selected[condition]:g}'
        state=torch.load(job/'final.pt',map_location='cpu',weights_only=False)['model']
        gates=Gates(condition,400);gi,_=gates.build(3199)
        weights.append((state['fc1.weight']*gi).abs().reshape(64,128,9).mean(2).numpy())
    vmax=max(float(w.max()) for w in weights)
    for ax,c,w in zip(axes.flat,conditions,weights):
        image=ax.imshow(w,aspect='auto',vmin=0,vmax=vmax,cmap='magma');ax.set(title=c,xlabel='Time sample',ylabel='Hidden unit')
    fig.colorbar(image,ax=axes.ravel().tolist(),label='Mean |effective weight| over nine channels',shrink=.7)
    fig.suptitle('Measured final weights: representative seed 400 / fold 0, native gates')
    fig.savefig(out/'effective_weight_patterns.png',dpi=160);plt.close(fig)
    gates=Gates('P2_gaussian',400)
    fig,axes=plt.subplots(1,4,figsize=(14,3))
    for ax,step in zip(axes,(0,1119,2239,2240)):
        g=gates.build(step)[0].reshape(64,128,9).mean(2).numpy()
        image=ax.imshow(g,aspect='auto',vmin=0,vmax=1);ax.set(title=f'Gate step {step}',xlabel='Time sample')
    fig.suptitle('Computed frozen schedule gates; 2240 is exact dense release')
    fig.tight_layout();fig.savefig(out/'gaussian_expansion.png',dpi=160);plt.close(fig)
    for condition in ('P3_dense_context','P3_gaussian_context'):
        job=run/'jobs'/f'{condition}_seed400_fold0_lr{selected[condition]:g}'
        model=make_model(condition,400);model.load_state_dict(torch.load(job/'final.pt',map_location='cpu',weights_only=False)['model'])
        fig,axes=plt.subplots(1,3,figsize=(12,3))
        matrices=[model.context_ih(c).detach().reshape(64,128,9).mean(2).numpy() for c in range(3)]
        lo=min(float(m.min()) for m in matrices);hi=max(float(m.max()) for m in matrices)
        for ax,name,m in zip(axes,('both','accelerometer','gyroscope'),matrices):
            image=ax.imshow(m,aspect='auto',vmin=lo,vmax=hi);ax.set(title=name,xlabel='Time sample')
        fig.suptitle(f'Measured {condition} C; seed 400 / fold 0; scale [{lo:.8f}, {hi:.8f}]')
        fig.tight_layout();fig.savefig(out/(condition+'_modulation.png'),dpi=160);plt.close(fig)
    # Validation-only far-connection ablation on final selected representative checkpoints.
    ablation=[]
    arrays=fold_arrays(0,config['folds'])['val']
    arrays['xt']=torch.from_numpy(arrays['x']);arrays['yt']=torch.tensor(arrays['y'],dtype=torch.long)
    for condition in conditions:
        job=run/'jobs'/f'{condition}_seed400_fold0_lr{selected[condition]:g}'
        model=make_model(condition,400);model.load_state_dict(torch.load(job/'final.pt',map_location='cpu',weights_only=False)['model'])
        gates=Gates(condition,400);native=gates.build(3199);gates.current=native
        # Dense bypass must be disabled for the ablation so the explicit mask is applied.
        before=fingerprint(model.state_dict())
        baseline=evaluate(model,arrays,gates)[0][0]
        gates.condition='intervention'
        masks=[(g*(d<=torch.quantile(d,.75)).float()) for g,d in zip(native,(gates.di,gates.dh))]
        gates.current=tuple(masks)
        changed=evaluate(model,arrays,gates)[0][0]
        assert fingerprint(model.state_dict())==before
        ablation.append({'condition':condition,'seed':400,'fold':0,'distance_quantile':.75,
            'diagnostic_only':True,'post_selection_not_used_for_redesign':True,
            **{metric+'_delta':changed[metric]-baseline[metric] for metric in ('accuracy','macro_f1','loss')}})
    pd.DataFrame(ablation).to_csv(output/'representative_distant_connection_ablation.csv',index=False)
    json_write(output/'diagnostic_completion.json',{'selected_jobs_inspected':350,'official_test_evaluations':0,
        'representative_plots':'predeclared seed 400 fold 0; not performance selected','weights_modified':False,
        'ablation_scope':'post-selection development checkpoint diagnostic only'})
    print('Checkpoint diagnostics completed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run_diagnostics(a.run)
