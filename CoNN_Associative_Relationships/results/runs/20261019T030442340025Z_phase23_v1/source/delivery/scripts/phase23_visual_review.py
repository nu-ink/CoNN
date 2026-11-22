"""Preserved visual review: clarify policies and quantify diagnostic reduction precision."""
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:]=[p for p in sys.path if Path(p or '.').resolve()!=ROOT/'scripts']
sys.path[:0]=[str(ROOT/'src'),str(ROOT.parent/'conn_experiment_01/src')]
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
import torch
from conn_assoc.phase23_core import configure,make_model,sha


def review_learning_curves(run):
    summary=pd.read_csv(run/'analysis/condition_summary.csv')
    selection=dict(zip(summary.condition,summary.lr))
    curves=pd.read_csv(run/'analysis/curves.csv')
    curves=curves[[selection[c]==lr for c,lr in zip(curves.condition,curves.lr)]]
    curves=curves[curves.kind!='pre_release']
    for phase in ('P2_','P3_'):
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        conditions=[c for c in summary.condition if c.startswith(phase)]
        for i,condition in enumerate(conditions):
            color=plt.get_cmap('tab10')(i)
            for ax,metric in zip(axes,('loss','accuracy','macro_f1')):
                for split,style in [('train','--'),('val','-')]:
                    frame=curves[(curves.condition==condition)&(curves.split==split)].groupby('completed_steps')[metric].mean()
                    ax.plot(frame.index,frame.values,style,color=color,label=condition if split=='val' else None)
                ax.set(xlabel='Optimizer updates',ylabel=metric)
        axes[-1].legend(fontsize=6)
        fig.suptitle('Measured selected-LR fold/context means; dashed=train, solid=validation; color=condition')
        fig.tight_layout();fig.savefig(run/'plots'/(phase+'learning_curves_reviewed.png'),dpi=160);plt.close(fig)
    review=run/'recovery_evidence/report_visual_review_01'
    report=ROOT/'reports/PHASE_2_3_IMPLEMENTATION_AND_RESULTS.md'
    shutil.copy2(report,review/'report.before_curve_style_review.md')
    for p in (report,run/'research_report.md'):
        text=p.read_text(encoding='utf-8')
        for phase in ('P2_','P3_'):text=text.replace(phase+'learning_curves.png',phase+'learning_curves_reviewed.png')
        p.write_text(text,encoding='utf-8')
    metadata=json.loads((run/'report_manifest.json').read_text());metadata['sha256']=sha(report)
    metadata['curve_style_review']='Unique color per condition; line style identifies train versus validation; original plots retained.'
    (run/'report_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')


def main():
    configure()
    run=Path((ROOT/'config/phase23_active_run.txt').read_text().strip())
    review=run/'recovery_evidence/report_visual_review_01';review.mkdir(parents=True,exist_ok=False)
    report=ROOT/'reports/PHASE_2_3_IMPLEMENTATION_AND_RESULTS.md'
    shutil.copy2(report,review/'report.before.md');shutil.copy2(run/'research_report.md',review/'run_report.before.md')
    selection={r['condition']:r['selected_lr'] for r in json.loads((run/'selection.frozen.json').read_text())['selection']}
    rows=[]
    for condition in ('P3_dense_context','P3_gaussian_context'):
        job=run/'jobs'/f'{condition}_seed400_fold0_lr{selection[condition]:g}'
        recorded=json.loads((job/'diagnostics.json').read_text())
        for checkpoint in ('initial','final'):
            state=torch.load(job/(checkpoint+'.pt'),map_location='cpu',weights_only=False)['model']
            model=make_model(condition,400);model.load_state_dict(state)
            for layer in ('ih','hh'):
                for context in range(3):
                    c=getattr(model,'context_'+layer)(context).detach().double()
                    row=next(r for r in recorded if r.get('kind')==checkpoint and r.get('layer')==layer and r.get('context')==context and 'modulation_mean' in r)
                    rows.append({'condition':condition,'seed':400,'fold':0,'checkpoint':checkpoint,'layer':layer,'context':context,
                        'mean_float64':float(c.mean()),'sample_std_float64':float(c.std()),'min':float(c.min()),'max':float(c.max()),
                        'recorded_float32_mean':row['modulation_mean'],'recorded_float32_std':row['modulation_std'],
                        'recorded_mean_error':row['modulation_mean']-float(c.mean())})
    precise=pd.DataFrame(rows);precise.to_csv(run/'analysis/representative_modulation_float64_review.csv',index=False)
    summary=pd.read_csv(run/'analysis/condition_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,prefix,title in zip(axes,('P2_','P3_'),('Phase 2: both sensors available','Phase 3: mean across three sensor contexts')):
        frame=summary[summary.condition.str.startswith(prefix)]
        ax.barh(frame.condition,100*frame.subject_macro_f1_mean)
        ax.set(xlabel='Validation subject macro-F1 (%)',title=title)
    fig.suptitle('Selected-LR development CV; phase-specific evaluation policies')
    fig.tight_layout();fig.savefig(run/'plots/model_comparison_reviewed.png',dpi=160);plt.close(fig)
    frame=pd.read_csv(run/'analysis/diagnostics.csv',usecols=['condition','lr','completed_steps','modulation_mean','modulation_min','modulation_max']).dropna(subset=['modulation_mean'])
    frame=frame[[selection[c]==lr for c,lr in zip(frame.condition,frame.lr)]]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,(condition,group) in zip(axes,frame.groupby('condition')):
        means=group.groupby('completed_steps').modulation_mean.mean()
        lo=group.groupby('completed_steps').modulation_min.min();hi=group.groupby('completed_steps').modulation_max.max()
        ax.plot(means.index,means.values,label='Recorded float32 mean reduction')
        ax.fill_between(lo.index,lo.values,hi.values,alpha=.25,label='Pointwise observed min–max')
        ax.set(title=condition,xlabel='Optimizer updates',ylabel='Recorded C summary')
        ax.ticklabel_format(axis='y',style='plain',useOffset=False);ax.legend(fontsize=6)
    fig.suptitle('Float32 reduction error is visible at this narrow scale; see checkpoint float64 review')
    fig.tight_layout();fig.savefig(run/'plots/context_modulation_curves_reviewed.png',dpi=160);plt.close(fig)
    review_learning_curves(run)
    note=f'''\n\n## Postprocessing precision and visual review\n\nThe first report and original plots are preserved. Reviewed model-comparison panels explicitly separate Phase 2 full-sensor scores from Phase 3 scores averaged across three sensor contexts; these are different evaluation policies. No measurements or LR choices were changed.\n\nThe recorded modulation means/standard deviations use float32 reductions. At changes near 1e-7, rounding can place a recorded mean outside the pointwise min–max and give an approximately nonzero standard deviation for a constant matrix. This is diagnostic reduction error, not evidence that C violates its bounds. Recomputing summaries in float64 from the **saved initial and final checkpoints** for the predeclared representative seed 400/fold 0 gives a maximum absolute recorded-mean error of {precise.recorded_mean_error.abs().max():.9g}. These are new checkpoint measurements, not reconstructed intermediate states. The full intermediate curves retain their actual float32 records and are now explicitly labelled. `analysis/representative_modulation_float64_review.csv` stores all 24 comparisons. Classification predictions, paired effects, weights, protocol and selection remain unchanged.\n\nAdditional implementation file: `scripts/phase23_visual_review.py`. Review evidence and both pre-review reports are in `recovery_evidence/report_visual_review_01/`.\n'''
    for path in (report,run/'research_report.md'):
        text=path.read_text(encoding='utf-8').replace('plots/model_comparison.png','plots/model_comparison_reviewed.png').replace('plots/context_modulation_curves.png','plots/context_modulation_curves_reviewed.png')
        path.write_text(text+note,encoding='utf-8')
    metadata=json.loads((run/'report_manifest.json').read_text());metadata['sha256']=sha(report)
    metadata['implementation_inventory'].append('scripts/phase23_visual_review.py')
    metadata['review']={'preserved_prior_report_sha256':sha(review/'report.before.md'),'precision_comparisons':len(precise),'classification_measurements_changed':False}
    (run/'report_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Visual/precision review complete; all original measurements retained.')


if __name__=='__main__':main()
