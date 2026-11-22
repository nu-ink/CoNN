"""Development summaries. Selection is allowed only after the complete equal-budget grid."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from conn_assoc.phase23_core import json_write,metrics,sha


def collect(run,destination='analysis'):
    scores=[];subjects=[];classes=[];curves=[];timing=[];gradient=[];diagnostics=[]
    for job in sorted((run/'jobs').iterdir()):
        if not (job/'completion.json').exists():continue
        base=json.loads((job/'config.json').read_text())
        base={k:base[k] for k in ('condition','seed','fold','lr')}
        done=json.loads((job/'completion.json').read_text())
        result=json.loads((job/'final_metrics.json').read_text())
        for split,rows in result['scores'].items():
            for row in rows:
                scores.append(dict(**base,split=split,**{k:v for k,v in row.items() if k not in ('recall','confusion')}))
                for label,recall in enumerate(row['recall']):
                    classes.append(dict(**base,split=split,context=row['context'],label=label,recall=recall))
        subjects.extend([dict(**base,**row) for row in result['subject_scores']])
        curves.extend([dict(**base,**row) for row in json.loads((job/'history.json').read_text())])
        timing.append(dict(**base,**{k:done[k] for k in ('elapsed_seconds','update_seconds','parameter_count','peak_gpu_allocated_bytes')}))
        for row in json.loads((job/'gradients.json').read_text()):
            for parameter,norm in row['norms'].items():
                gradient.append(dict(**base,step=row['step'],context=row['context'],parameter=parameter,norm=norm,loss=row['loss']))
        for row in json.loads((job/'diagnostics.json').read_text()):
            if 'gate_row_energy' in row:
                for unit,energy in enumerate(row['gate_row_energy']):
                    diagnostics.append(dict(**base,**{k:v for k,v in row.items() if k!='gate_row_energy'},unit=unit,row_energy=energy))
            else:diagnostics.append(dict(**base,**row))
    output=run/destination;output.mkdir(exist_ok=True)
    tables={name:pd.DataFrame(rows) for name,rows in [('scores',scores),('subjects',subjects),('class_recall',classes),
        ('curves',curves),('runtime',timing),('gradients',gradient),('diagnostics',diagnostics)]}
    for name,table in tables.items():table.to_csv(output/(name+'.csv'),index=False)
    return tables


def inspect_first_seed(run):
    tables=collect(run,'first_seed_analysis')
    selected=tables['runtime'].query('seed==400 and condition in ["P2_dense","P2_gaussian"]')
    assert len(selected)==42,len(selected)
    scores=tables['scores'].query('split=="val" and seed==400')
    assert np.isfinite(scores[['loss','accuracy','macro_f1']].values).all()
    summary=scores.groupby(['condition','lr'])[['accuracy','macro_f1','loss']].mean().reset_index()
    summary.to_csv(run/'first_seed_summary.csv',index=False)
    json_write(run/'first_seed_review.json',{'status':'accepted for implementation progression','fits_completed':42,
        'seed':400,'steps_per_fit':3200,'finite_validation_metrics':True,'gradient_norms_finite':bool(np.isfinite(tables['gradients']['norm']).all()),
        'official_test_evaluations':0,'note':'Development engineering evidence only. No LR selected from this partial grid.'})
    print(summary.to_string(index=False))


def full_analysis(run):
    config=json.loads((run/'config.frozen.json').read_text())
    tables=collect(run);out=run/'analysis'
    runtime=tables['runtime'];subject=tables['subjects'];scores=tables['scores']
    assert len(runtime)==1050,f'Incomplete grid: {len(runtime)}/1050; no selection permitted'
    keys=runtime[['condition','seed','fold','lr']]
    assert not keys.duplicated().any()
    for cond in config['conditions']:
        assert len(runtime[runtime.condition==cond])==105
    selection=[]
    for condition in config['conditions']:
        candidates=[]
        for lr in config['learning_rates']:
            s=subject[(subject.condition==condition)&(subject.lr==lr)&(subject.split=='val')]
            assert len(s)==105*(3 if condition.startswith('P3_') else 1)
            f1=float(s.groupby(['subject','seed']).macro_f1.mean().mean())
            loss=float(scores[(scores.condition==condition)&(scores.lr==lr)&(scores.split=='val')].loss.mean())
            candidates.append({'lr':lr,'subject_macro_f1':f1,'loss':loss})
        best=sorted(candidates,key=lambda r:(-r['subject_macro_f1'],r['loss'],r['lr']))[0]
        selection.append({'condition':condition,'selected_lr':best['lr'],'candidates':candidates})
    selection_path=run/'selection.frozen.json'
    if selection_path.exists():
        assert json.loads(selection_path.read_text())['selection']==selection,'Existing frozen selection differs; preserve and investigate'
    else:
        json_write(selection_path,{'rule':'validation only; final checkpoint, equal subject/context/seed weighting',
            'frozen_utc':datetime.now(timezone.utc).isoformat(),'official_test_evaluations':0,
            'final_test_authorized':False,'development_only':True,'selection':selection})
    digest=sha(selection_path)
    (run/'selection.frozen.json.sha256').write_text(digest+'\n',encoding='utf-8')
    lrmap={r['condition']:r['selected_lr'] for r in selection}
    def chosen(frame):return frame[np.array([lrmap[c]==lr for c,lr in zip(frame.condition,frame.lr)])]
    sv=chosen(subject[subject.split=='val']).copy()
    sc=chosen(scores);sv.to_csv(out/'selected_validation_subjects.csv',index=False)
    seed_table=sv.groupby(['condition','seed','context'])[['accuracy','macro_f1']].mean().reset_index()
    seed_table.to_csv(out/'selected_seed_context_scores.csv',index=False)
    seed_overall=sv.groupby(['condition','seed'])[['accuracy','macro_f1']].mean().reset_index()
    seed_overall.to_csv(out/'selected_seed_scores.csv',index=False)
    summary=[]
    for condition in config['conditions']:
        a=seed_overall[seed_overall.condition==condition]
        tr=sc[(sc.condition==condition)&(sc.split=='train')]
        val=sc[(sc.condition==condition)&(sc.split=='val')]
        summary.append({'condition':condition,'lr':lrmap[condition],
            'subject_accuracy_mean':a.accuracy.mean(),'subject_accuracy_seed_sd':a.accuracy.std(ddof=1),
            'subject_macro_f1_mean':a.macro_f1.mean(),'subject_macro_f1_seed_sd':a.macro_f1.std(ddof=1),
            'fold_train_accuracy':tr.accuracy.mean(),'fold_val_accuracy':val.accuracy.mean(),
            'fold_train_macro_f1':tr.macro_f1.mean(),'fold_val_macro_f1':val.macro_f1.mean(),
            'fold_train_loss':tr.loss.mean(),'fold_val_loss':val.loss.mean(),
            'train_val_accuracy_gap':tr.accuracy.mean()-val.accuracy.mean(),
            'parameters':int(runtime[runtime.condition==condition].parameter_count.iloc[0])})
    pd.DataFrame(summary).to_csv(out/'condition_summary.csv',index=False)
    contrasts=[('P2_gaussian','P2_dense'),('P2_gaussian','P2_rowshuffle'),('P2_gaussian','P2_coordperm'),
               ('P2_fixed08','P2_gaussian'),('P2_fixed25','P2_gaussian'),
               ('P3_dense_context','P3_dense_masked'),('P3_gaussian_context','P3_gaussian_masked'),
               ('P3_gaussian_context','P3_dense_context'),('P3_gaussian_masked','P3_dense_masked')]
    intervals=[];individual=[]
    rng=np.random.default_rng(230319)
    for a,b in contrasts:
        sa=sv[sv.condition==a].groupby(['subject','seed'])[['accuracy','macro_f1']].mean()
        sb=sv[sv.condition==b].groupby(['subject','seed'])[['accuracy','macro_f1']].mean()
        diff=sa-sb
        for metric in ('accuracy','macro_f1'):
            seeds=diff[metric].groupby('seed').mean()
            for seed,value in seeds.items():individual.append({'contrast':a+' minus '+b,'metric':metric,'seed':int(seed),'difference':float(value)})
            values=diff[metric].groupby('subject').mean().values
            boots=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
            lo,hi=np.quantile(boots,[.025,.975]);sd=values.std(ddof=1)
            intervals.append({'contrast':a+' minus '+b,'metric':metric,'mean':values.mean(),'seed_sd':seeds.std(ddof=1),
                'subject_sd':sd,'subject_dz':values.mean()/sd if sd else None,'ci_low':lo,'ci_high':hi,
                'n_subjects':len(values),'n_seeds':5,'interval':'descriptive post-selection subject bootstrap; overlapping CV training'})
    # Difference in context gains: contextual Gaussian interaction beyond contextual dense gain.
    pivot=sv.groupby(['condition','subject','seed']).macro_f1.mean().unstack('condition')
    interaction=(pivot.P3_gaussian_context-pivot.P3_gaussian_masked)-(pivot.P3_dense_context-pivot.P3_dense_masked)
    vals=interaction.groupby('subject').mean().values
    lo,hi=np.quantile(vals[rng.integers(0,len(vals),(10000,len(vals)))].mean(1),[.025,.975])
    intervals.append({'contrast':'Gaussian × context interaction','metric':'macro_f1','mean':vals.mean(),
        'seed_sd':interaction.groupby('seed').mean().std(ddof=1),'subject_sd':vals.std(ddof=1),'subject_dz':vals.mean()/vals.std(ddof=1) if vals.std(ddof=1)>0 else None,
        'ci_low':lo,'ci_high':hi,'n_subjects':21,'n_seeds':5,'interval':'descriptive post-selection subject bootstrap'})
    pd.DataFrame(intervals).to_csv(out/'paired_contrasts.csv',index=False)
    pd.DataFrame(individual).to_csv(out/'individual_seed_differences.csv',index=False)
    # Independently score saved out-of-fold validation predictions; no model/test load.
    rescored=[]
    confusion=[]
    for condition in config['conditions']:
        contexts=(0,1,2) if condition.startswith('P3_') else (0,)
        for seed in config['seeds']:
            truth=[];pred={c:[] for c in contexts}
            for fold in range(7):
                job=run/'jobs'/f'{condition}_seed{seed}_fold{fold}_lr{lrmap[condition]:g}'
                with np.load(job/'predictions.npz') as z:
                    truth.append(z['val_truth'])
                    for c in contexts:pred[c].append(z['val_context'+str(c)])
            y=np.concatenate(truth);assert len(y)==7352
            for c in contexts:
                m=metrics(y,np.concatenate(pred[c]))
                rescored.append({'condition':condition,'seed':seed,'context':c,'pooled_accuracy':m['accuracy'],'pooled_macro_f1':m['macro_f1']})
                for i,row in enumerate(m['confusion']):
                    for j,count in enumerate(row):confusion.append({'condition':condition,'seed':seed,'context':c,'true':i,'predicted':j,'count':count})
    pd.DataFrame(rescored).to_csv(out/'pooled_oof_scores.csv',index=False)
    pd.DataFrame(confusion).to_csv(out/'pooled_oof_confusions.csv',index=False)
    plot(run,tables,summary,intervals,chosen)
    json_write(out/'completion.json',{'fits':1050,'updates':3360000,'all_conditions_equal_budget':True,
        'official_test_evaluations':0,'seed_count':5,'subject_count':21,'selection_post_cv':True})
    print(pd.DataFrame(summary).to_string(index=False))


def plot(run,tables,summary,intervals,chosen):
    out=run/'plots';out.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(10,5))
    rows=pd.DataFrame(summary)
    ax.barh(rows.condition,100*rows.subject_macro_f1_mean)
    ax.set(xlabel='Validation subject macro-F1 (%)',title='Selected-LR development CV; five engineering seeds')
    fig.tight_layout();fig.savefig(out/'model_comparison.png',dpi=160);plt.close(fig)
    curves=chosen(tables['curves']);curves=curves[curves.kind!='pre_release']
    for phase in ('P2_','P3_'):
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for condition in rows.condition[rows.condition.str.startswith(phase)]:
            for ax,metric in zip(axes,('loss','accuracy','macro_f1')):
                for split,style in [('train','--'),('val','-')]:
                    r=curves[(curves.condition==condition)&(curves.split==split)].groupby('completed_steps')[metric].mean()
                    ax.plot(r.index,r.values,style,label=condition+' '+split)
                ax.set(xlabel='Optimizer updates',ylabel=metric)
        axes[-1].legend(fontsize=5);fig.suptitle('Measured selected-LR fold/context-averaged learning curves')
        fig.tight_layout();fig.savefig(out/(phase+'learning_curves.png'),dpi=160);plt.close(fig)
    subj=chosen(tables['subjects']);subj=subj[(subj.split=='val')&subj.condition.str.startswith('P3_')]
    means=subj.groupby(['condition','context']).macro_f1.mean().unstack('context')
    fig,ax=plt.subplots(figsize=(9,5));means.plot.bar(ax=ax)
    ax.set(ylabel='Validation subject macro-F1',title='Realistic input mask agrees with supplied sensor context')
    ax.legend(['both','accelerometer','gyroscope']);fig.tight_layout();fig.savefig(out/'sensor_context_effects.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,6))
    rows=[r for r in intervals if r['metric']=='macro_f1']
    for i,r in enumerate(rows):ax.errorbar(100*r['mean'],i,xerr=[[100*(r['mean']-r['ci_low'])],[100*(r['ci_high']-r['mean'])]],fmt='o',capsize=3)
    ax.set_yticks(range(len(rows)),[r['contrast'] for r in rows]);ax.axvline(0,color='grey',ls='--')
    ax.set(xlabel='Macro-F1 difference (percentage points)',title='Descriptive subject bootstrap; selection and overlapping CV limit inference')
    fig.tight_layout();fig.savefig(out/'paired_contrasts.png',dpi=160);plt.close(fig)
    bench=json.loads((run/'benchmark.json').read_text())['measurements']
    fig,ax=plt.subplots(figsize=(7,4));ax.bar([r['condition']+'\n'+r['device'] for r in bench],[r['batches_per_second'] for r in bench])
    ax.set(ylabel='Updates per second',title='Sequential warmed-up training throughput; no parallel contention')
    fig.tight_layout();fig.savefig(out/'cpu_cuda_throughput.png',dpi=160);plt.close(fig)
    gradient=chosen(tables['gradients'])
    fig,ax=plt.subplots(figsize=(10,4))
    for condition in rows_condition(tables):
        g=gradient[gradient.condition==condition].groupby('step').norm.mean()
        ax.plot(g.index,g.values,label=condition)
    ax.set(yscale='log',xlabel='Optimizer update',ylabel='Mean parameter gradient norm',title='Measured gradient diagnostics')
    ax.legend(fontsize=6,ncol=2);fig.tight_layout();fig.savefig(out/'gradients.png',dpi=160);plt.close(fig)
    context_gradient=gradient[gradient.condition.str.endswith('_context')].copy()
    context_gradient['family']=np.where(context_gradient.parameter.str.startswith('context_'),'context','base')
    totals=context_gradient.groupby(['condition','seed','fold','lr','step','family'])['norm'].agg(lambda x:float(np.sqrt((x*x).sum()))).reset_index(name='total_l2_gradient_norm')
    totals.to_csv(run/'analysis/context_gradient_totals.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4))
    for (condition,family),frame in totals.groupby(['condition','family']):
        means=frame.groupby('step').total_l2_gradient_norm.mean()
        ax.plot(means.index,means.values,label=condition+' '+family)
    ax.set(yscale='log',xlabel='Optimizer update',ylabel='Total L2 gradient norm',title='Base versus contextual-parameter gradients, selected LR')
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/'context_gradients.png',dpi=160);plt.close(fig)
    modulation=chosen(tables['diagnostics'])
    modulation=modulation[modulation.modulation_mean.notna()]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,(condition,frame) in zip(axes,modulation.groupby('condition')):
        means=frame.groupby('completed_steps').modulation_mean.mean()
        lo=frame.groupby('completed_steps').modulation_min.min()
        hi=frame.groupby('completed_steps').modulation_max.max()
        ax.plot(means.index,means.values,label='Mean C')
        ax.fill_between(lo.index,lo.values,hi.values,alpha=.25,label='Observed min–max')
        ax.set(title=condition,xlabel='Optimizer updates',ylabel='C modulation magnitude');ax.legend(fontsize=7)
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    fig.suptitle('Measured bounded modulation; narrow axis range is explicit')
    fig.tight_layout();fig.savefig(out/'context_modulation_curves.png',dpi=160);plt.close(fig)


def rows_condition(tables):return sorted(tables['runtime'].condition.unique())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['first-seed','full']);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='first-seed':inspect_first_seed(args.run)
    else:full_analysis(args.run)


if __name__=='__main__':main()
