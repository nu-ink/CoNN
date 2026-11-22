"""Measured HAR reports; no training, selection or extra test inference."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from conn_exp.experiment1b import dump
from conn_exp.experiment1b_report import summarize
from conn_exp.har_core import CONDITION_NAMES

COLORS=dict(A='#777777',S08='#e377c2',S25='#9467bd',D='#2ca02c',B='#ff7f0e',C='#17becf')
ORDER=['A','S08','S25','D','B','C']
ACTIVITIES=['WALK','UPSTAIRS','DOWNSTAIRS','SIT','STAND','LAY']
CONTRASTS=[('D_minus_B','D','B'),('D_minus_A','D','A'),('S08_minus_A','S08','A'),
    ('S25_minus_A','S25','A'),('D_minus_C','D','C'),('S25_minus_D','S25','D'),('S08_minus_D','S08','D')]


def fmt(value):
    return 'undefined' if value is None else f'{value:.4f}'


def save(fig,folder,name):
    fig.tight_layout();fig.savefig(folder/name,dpi=145);plt.close(fig)


def read_jobs(phase,scores,name):
    return pd.concat([pd.read_csv(phase/r.job/(name+'.csv')) for r in scores.itertuples()],ignore_index=True)


def mean_curve(axis,frame,metric,cid,style,label):
    group=frame.groupby(['optimizer_steps_completed','kind'])[metric].agg(['mean','sem']).reset_index()
    group['kind_rank']=group.kind.map(dict(initial=0,curve=1,pre_release=2,post_release=3,final=4))
    group=group.sort_values(['optimizer_steps_completed','kind_rank'])
    x=group.optimizer_steps_completed.to_numpy();y=group['mean'].to_numpy()
    axis.plot(x,y,style,color=COLORS[cid],label=label,lw=1.4)
    if style=='-':
        se=group['sem'].to_numpy();axis.fill_between(x,y-se,y+se,color=COLORS[cid],alpha=.08)


def build_report(root):
    root=Path(root);phase=root/'replication'
    protocol=json.loads((root/'protocol.frozen.json').read_text())
    selection=json.loads((root/'selection.frozen.json').read_text())
    audit=json.loads((root/'audit/dataset_audit.json').read_text())
    config=protocol['configuration'];reps=config['representations']
    scores=pd.read_csv(phase/'scores.csv');tuning=pd.read_csv(root/'tuning/scores.csv')
    assert len(scores)==protocol['final_models'] and len(tuning)==protocol['tuning_models']
    assert scores.test_evaluations.eq(1).all() and tuning.test_evaluations.eq(0).all()
    assert 'test_accuracy' not in tuning and 'test_macro_f1' not in tuning
    assert set(scores.seed).isdisjoint(tuning.seed)
    for rep in reps:
        for cid in ORDER:
            f=scores[(scores.representation==rep)&(scores.condition==cid)]
            assert len(f)==20 and set(f.seed)==set(config['replication_seeds'])
            assert f.lr.eq(selection['learning_rates'][rep][cid]).all()
    numeric=['train_accuracy','val_accuracy','test_accuracy','train_macro_f1','val_macro_f1','test_macro_f1',
        'train_loss','val_loss','test_loss','train_test_gap_accuracy','train_test_gap_macro_f1',
        'train_val_gap_accuracy','train_val_gap_macro_f1','val_test_gap_accuracy','val_test_gap_macro_f1',
        'update_wall_seconds','update_process_cpu_seconds','job_elapsed_wall_seconds']
    assert np.isfinite(scores[numeric]).all().all()
    grouped=scores.groupby(['representation','condition'])[numeric].agg(['mean','std','median'])
    grouped.to_csv(phase/'condition_summary.csv')
    means=scores.groupby(['representation','condition'])[numeric].mean()
    differences=[];individual=[]
    for rep in reps:
        data=scores[scores.representation==rep]
        for metric in ['accuracy','macro_f1']:
            pivot=data.pivot(index='seed',columns='condition',values='test_'+metric)
            pivot.to_csv(phase/f'{rep}_individual_seed_{metric}.csv')
            for name,a,b in CONTRASTS:
                delta=pivot[a]-pivot[b]
                primary=rep=='engineered' and name=='D_minus_B' and metric=='macro_f1'
                differences.append(dict(representation=rep,contrast=name,metric=metric,primary=primary,**summarize(delta)))
                individual.extend(dict(representation=rep,contrast=name,metric=metric,primary=primary,seed=int(s),delta=float(v)) for s,v in delta.items())
    statistics=pd.DataFrame(differences)
    statistics.to_csv(phase/'paired_statistical_summary.csv',index=False)
    pd.DataFrame(individual).to_csv(phase/'individual_seed_differences.csv',index=False)
    primary=[r for r in differences if r['primary']][0]
    dump(phase/'primary_summary.json',primary)
    subjects=read_jobs(phase,scores,'test_subject_scores')
    subjects.to_csv(phase/'subject_scores.csv',index=False)
    subject_stats=[];subject_differences=[]
    for rep in reps:
        for metric in ['accuracy','macro_f1']:
            pivot=subjects[subjects.representation==rep].pivot(index=['seed','subject'],columns='condition',values=metric)
            delta=(pivot.D-pivot.B).rename('delta').reset_index()
            delta['representation']=rep;delta['metric']=metric
            subject_differences.append(delta)
            per_subject=delta.groupby('subject').delta.mean()
            subject_stats.append(dict(representation=rep,contrast='D_minus_B',metric=metric,
                unit='test_subject_mean_over_20_models',**summarize(per_subject)))
    subject_diff=pd.concat(subject_differences,ignore_index=True)
    subject_diff.to_csv(phase/'paired_subject_differences.csv',index=False)
    pd.DataFrame(subject_stats).to_csv(phase/'subject_level_summary.csv',index=False)
    metrics=read_jobs(phase,scores,'metrics');metrics.to_csv(phase/'training_curves.csv',index=False)
    checkpoint=read_jobs(phase,scores,'checkpoint_gate_comparisons')
    checkpoint.to_csv(phase/'checkpoint_gate_comparisons.csv',index=False)
    checkpoint.groupby(['representation','condition','kind','gate_mode'])[['accuracy','macro_f1','loss']].agg(['mean','std']).to_csv(phase/'checkpoint_summary.csv')
    ablation=read_jobs(phase,scores,'ablation')
    ablation[ablation.kind=='final'].groupby(['representation','condition','intervention'])[[
        'val_accuracy_delta','val_macro_f1_delta','val_loss_delta']].agg(['mean','std']).to_csv(phase/'final_distant_ablation_summary.csv')
    plots=root/'plots';plots.mkdir(exist_ok=True)
    for rep in reps:
        f=metrics[(metrics.representation==rep)&metrics.split.isin(['train','val'])]
        fig,axes=plt.subplots(1,3,figsize=(17,4.5))
        for cid in ORDER:
            for split,style in [('val','-'),('train','--')]:
                for metric,axis in zip(['accuracy','macro_f1','loss'],axes):
                    mean_curve(axis,f[(f.condition==cid)&(f.split==split)],metric,cid,style,cid if split=='val' else '_nolegend_')
        for metric,axis in zip(['accuracy','macro_f1','loss'],axes):
            axis.set_xlabel('optimizer updates');axis.set_ylabel(metric);axis.axvline(2240,color='gray',ls=':',lw=.8)
        axes[-1].legend(ncol=2,fontsize=8);fig.suptitle(f'HAR {rep}: validation solid ±1 SE, training dashed; no test curve')
        save(fig,plots,f'{rep}_learning_curves.png')
        fig,axes=plt.subplots(2,3,figsize=(16,9))
        for cid,axis in zip(ORDER,axes.flat):
            files=scores[(scores.representation==rep)&(scores.condition==cid)]
            confusion=sum(pd.read_csv(phase/r.job/'test_confusion.csv').to_numpy() for r in files.itertuples())
            normalized=confusion/confusion.sum(1,keepdims=True)
            axis.imshow(normalized,vmin=0,vmax=1,cmap='Blues')
            for i in range(6):
                for j in range(6):
                    axis.text(j,i,f'{normalized[i,j]:.2f}',ha='center',va='center',fontsize=8,
                        color='white' if normalized[i,j]>.5 else 'black')
            axis.set_xticks(range(6),ACTIVITIES,rotation=45,ha='right',fontsize=7)
            axis.set_yticks(range(6),ACTIVITIES,fontsize=7);axis.set_title(cid)
            axis.set_xlabel('predicted');axis.set_ylabel('true activity')
        fig.suptitle(f'{rep}: test confusion matrices, row normalized across 20 model repetitions')
        save(fig,plots,f'{rep}_test_confusion.png')
        chosen=scores[(scores.representation==rep)&scores.condition.isin(['S25','D','B','C'])]
        grads=read_jobs(phase,chosen,'gradients')
        for layer in ['ih','hh']:
            fig,axes=plt.subplots(2,4,figsize=(17,7),sharex=True,sharey='row')
            for cid,column in zip(['S25','D','B','C'],axes.T):
                for quantity,axis in zip(['dL_dW','dL_dW_eff'],column):
                    frame=grads[(grads.condition==cid)&(grads.quantity==quantity)&(grads.layer==layer)]
                    for bin_id,g in frame.groupby('bin_id'):
                        avg=g.groupby('optimizer_steps_completed').mean_abs.mean()
                        axis.plot(avg.index,avg,label=f'bin {bin_id}',lw=1)
                    axis.set_title(cid+' '+quantity);axis.set_xlabel('updates')
                    axis.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            axes[0,-1].legend(fontsize=7)
            fig.suptitle(f'{rep} {layer}: fixed-minibatch gradients by native coordinate distance')
            save(fig,plots,f'{rep}_{layer}_gradient_distance.png')
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for layer,axis in zip(['ih','hh'],axes):
            for cid in ORDER:
                frame=f[(f.condition==cid)&(f.split=='val')]
                mean_curve(axis,frame,'norm_W_eff_'+layer,cid,'-',cid)
            axis.set_xlabel('updates');axis.set_ylabel('effective weight L2 norm '+layer)
        axes[1].legend(ncol=2,fontsize=8)
        fig.suptitle(rep+': effective weight magnitudes under native gates')
        save(fig,plots,f'{rep}_effective_weights.png')
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        frame=checkpoint[(checkpoint.representation==rep)&checkpoint.condition.isin(['S08','S25','D'])]
        for cid in ['S08','S25','D']:
            for mode,style in [('native','-'),('exact_dense','--')]:
                values=frame[(frame.condition==cid)&(frame.gate_mode==mode)].groupby('kind')[['macro_f1','loss']].mean().reindex(['pre_release','post_release','final'])
                for metric,axis in zip(['macro_f1','loss'],axes):
                    axis.plot(range(3),values[metric],style,marker='o',color=COLORS[cid],label=f'{cid} {mode}')
                    axis.set_xticks(range(3),['pre release','post release','final']);axis.set_ylabel('validation '+metric)
        axes[1].legend(fontsize=8,ncol=2)
        fig.suptitle(rep+': native/dense validation gate interventions on reloaded weights')
        save(fig,plots,f'{rep}_checkpoint_gate_comparisons.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for rep,axis in zip(reps,axes):
        subset=scores[scores.representation==rep]
        for metric,offset,label,color in [('test_accuracy',-.18,'accuracy','#4c78a8'),('test_macro_f1',.18,'macro-F1','#f58518')]:
            g=subset.groupby('condition')[metric].agg(['mean','sem']).reindex(ORDER)
            axis.bar(np.arange(6)+offset,g['mean'],width=.35,yerr=g['sem'],capsize=3,label=label,color=color)
        axis.set_xticks(range(6),ORDER);axis.set_ylim(0,1);axis.set_title(rep)
        axis.set_ylabel('test metric (mean ±1 SE)');axis.legend()
    save(fig,plots,'final_metric_comparison.png')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for rep,axis in zip(reps,axes):
        data=statistics[(statistics.representation==rep)&(statistics.metric=='macro_f1')].copy()
        axis.errorbar(100*data['mean'],range(len(data)),xerr=[100*(data['mean']-data.ci_low),100*(data.ci_high-data['mean'])],fmt='o',capsize=3)
        axis.axvline(0,color='gray',lw=.8);axis.set_yticks(range(len(data)),data.contrast)
        axis.set_xlabel('paired macro-F1 difference, points (95% t interval)');axis.set_title(rep)
    save(fig,plots,'paired_macro_f1_comparisons.png')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    vmax=max(abs(subject_diff.delta.min()),abs(subject_diff.delta.max()))
    for rep,axis in zip(reps,axes):
        grid=subject_diff[(subject_diff.representation==rep)&(subject_diff.metric=='macro_f1')].pivot(index='seed',columns='subject',values='delta')
        im=axis.imshow(grid,aspect='auto',vmin=-vmax,vmax=vmax,cmap='RdBu')
        axis.set_xticks(range(len(grid.columns)),grid.columns);axis.set_yticks(range(len(grid)),grid.index,fontsize=6)
        axis.set_xlabel('held-out subject');axis.set_ylabel('model seed');axis.set_title(rep+' D − B macro-F1')
        fig.colorbar(im,ax=axis)
    save(fig,plots,'subject_seed_heterogeneity.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for rep,axis in zip(reps,axes):
        data=scores[scores.representation==rep]
        g=data.groupby('condition')[['train_accuracy','val_accuracy','test_accuracy']].mean().reindex(ORDER)
        g.plot.bar(ax=axis,ylim=(0,1),rot=0);axis.set_title(rep);axis.set_ylabel('mean accuracy')
    save(fig,plots,'generalization_gaps.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for rep,axis in zip(reps,axes):
        g=scores[scores.representation==rep].groupby('condition').update_process_cpu_seconds.agg(['mean','std']).reindex(ORDER)
        axis.bar(range(6),g['mean'],yerr=g['std'],capsize=3)
        axis.set_xticks(range(6),ORDER);axis.set_ylabel('update process CPU seconds (mean ±SD)');axis.set_title(rep)
    fig.suptitle('Fixed update budget; excludes diagnostics; parallel contention limits timing comparisons')
    save(fig,plots,'training_time.png')
    grid=pd.read_csv(root/'tuning/validation_grid.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for rep,axis in zip(reps,axes):
        p=grid[grid.representation==rep].pivot(index='condition',columns='lr',values='mean_val_macro_f1').reindex(ORDER)
        im=axis.imshow(p,aspect='auto',vmin=0,vmax=1)
        axis.set_xticks(range(3),p.columns);axis.set_yticks(range(6),p.index);axis.set_title(rep)
        axis.set_xlabel('learning rate');fig.colorbar(im,ax=axis)
        for i in range(6):
            for j in range(3):
                axis.text(j,i,f'{p.iloc[i,j]:.3f}',ha='center',va='center',fontsize=8,color='white' if p.iloc[i,j]<.7 else 'black')
    fig.suptitle('Validation-only tuning: five seeds per cell, equal budgets')
    save(fig,plots,'validation_grid.png')
    # Build a concrete interpretation from measured results, keeping primary scope.
    if primary['spans_zero']:
        outcome='The primary scheduled-structure versus shuffled-control comparison is unresolved: its interval includes zero.'
    elif primary['mean']>0:
        outcome='The primary scheduled-structure versus shuffled-control comparison was positive on this fixed subject-disjoint engineered-feature benchmark.'
    else:
        outcome='The primary comparison favored the row-shuffled control; the scheduled synthetic-data advantage did not transfer under this engineered-feature protocol.'
    by={(r['representation'],r['contrast'],r['metric']):r for r in differences}
    def effect(rep,name,metric='macro_f1'):
        r=by[(rep,name,metric)]
        return f"{100*r['mean']:+.4f} points, 95% CI [{100*r['ci_low']:+.4f}, {100*r['ci_high']:+.4f}]"
    lines=['# CoNN Experiment 2 — UCI HAR real-world validation','',outcome,'',
        f"Sole primary result: engineered-feature D − B in final test macro-F1, n=20 paired initialization/order seeds: mean {primary['mean']:+.8f} ({100*primary['mean']:+.4f} points), sample SD {primary['sd']:.8f}, two-sided 95% Student-t CI [{primary['ci_low']:.8f}, {primary['ci_high']:.8f}], df=19, paired Cohen dz {fmt(primary['cohen_dz'])}. Positive/tied/negative seed differences: {primary['positive_seeds']}/{primary['tied_seeds']}/{primary['negative_seeds']}.",'',
        '## Transfer findings and their scope','',
        f"Against the conventional dense baseline, engineered D − A macro-F1 was {effect('engineered','D_minus_A')}; fixed S08 − A was {effect('engineered','S08_minus_A')}; fixed S25 − A was {effect('engineered','S25_minus_A')}. These are distinct from the primary alignment contrast. Beating a shuffled curriculum would not by itself show a useful improvement over dense training.",
        f"The predeclared temporal-window secondary D − B contrast was {effect('temporal','D_minus_B')}; D − A was {effect('temporal','D_minus_A')}. Coordinate-permutation comparisons were engineered D − C {effect('engineered','D_minus_C')} and temporal D − C {effect('temporal','D_minus_C')}. These secondary intervals are exploratory and unadjusted.",
        'The table below reports both positive and negative comparisons. No method, width, layout, duration or learning rate was redesigned after observing test results. An isolated favorable secondary comparison does not overturn an unresolved or negative primary result.','',
        '## Dataset audit and predefined coordinates','',
        f"Source: [UCI HAR]({audit['dataset_page']}), DOI 10.24432/C54S4K. Official archive SHA-256 `{audit['archive_sha256']}`. The audit retained source documentation, activity labels, feature names, subject IDs, class/subject counts and archive inventory before any model training.",
        'The primary input is the released 561-feature vector, in original indexed metadata order with coordinate j/560. Its neighborhood structure is semantic file order, not measured physical geometry. Repeated names (84 duplicated occurrences) are retained as distinct indexed columns. The secondary input is the nine-channel 128-sample inertial window flattened time-major, with all channels at time t assigned t/127. The axes are colocated at each time; no invented physical sensor distances are used. Hidden coordinates are fixed uniformly on the same unit interval. Neither layout uses labels, test values, correlations, learned embeddings or test-accuracy search.',
        f"Fit subjects: {protocol['subject_partition']['fit_subjects']}. Validation subjects: {protocol['subject_partition']['validation_subjects']}. Test subjects: {protocol['subject_partition']['test_subjects']}. Window counts: {audit['split_sizes']}. The official test subjects were retained; five of the 21 original training subjects were selected for validation using fixed split seed 104729, without labels or feature values.",
        'Every added normalization statistic uses only the 16 fit subjects: per feature for engineered vectors; per sensor channel across fit windows/time for temporal inputs. Validation/test are transformed with those frozen statistics. Subject IDs and activity labels are excluded from model inputs. Overlapping windows cannot cross subject partitions. No validation subjects were merged back into training after selection.',
        audit['vendor_preprocessing_limitation'],audit['timing_limitation'],
        'The audit inspected test schema, labels and counts for integrity; no model predictions or performance-based choices used test before the recorded freeze. See audit/dataset_report.md for the full audit.','',
        '## Comparable training and validation-only selection','',
        'Every condition uses two 64-unit ReLU hidden layers, a six-class dense output head, and the original plain SGD family (no momentum/weight decay), batch size 128, unweighted cross-entropy and 3200 updates. A is conventional dense and bypasses gate multiplication. S08 and S25 hold Gaussian d50 at .08 or .25. D smoothly widens .08→1 over the first 2240 updates, then uses exact dense gates. B applies frozen independent within-row permutations to D’s gates in both layers. C permutes input coordinate assignments only; its hidden gate follows D. B/C preserve D’s row gate multisets and energy. Feature values and labels are unchanged by either control.',
        'Initialization, training row permutations, source splits and scaler are paired across all conditions and LRs for each seed. Separate initialization/order/mask/execution-order streams prevent one condition from consuming another’s randomness. Parameter counts are identical within each representation; input dimensions differ between representations, so cross-representation rankings also reflect input/parameter differences.',
        'Each representation/condition received LR [.01,.05,.1] × five tuning seeds 30–34, for 180 fits total and 48,000 tuning optimizer updates per condition per representation. Mean final validation macro-F1 selected LR; ties used lower validation cross-entropy then lower LR. The final fixed-budget checkpoint is the scoring checkpoint. No early stopping, test-selected checkpoint or test-based layout choice was used.',
        'protocol.frozen.json declared the primary and secondary contrasts before training. selection.frozen.json sealed selected LRs before final_evaluation.started.json and before final models loaded test arrays. Final evaluation used seeds 300–319, 240 models total. Each model computed native final test predictions once; pooled, subject and class metrics reuse those predictions.','',
        '| Representation | Condition | Selected LR | Parameter count |','|---|---|---|---|']
    for rep in reps:
        for cid in ORDER:
            params=int(scores[(scores.representation==rep)&(scores.condition==cid)].parameter_count.iloc[0])
            lines.append(f"| {rep} | {cid}: {CONDITION_NAMES[cid]} | {selection['learning_rates'][rep][cid]:g} | {params} |")
    lines+=['','## Final test performance and generalization gaps','',
        '| Representation | Condition | Accuracy mean ± SD | Macro-F1 mean ± SD | Train−test accuracy gap | Train−test F1 gap |',
        '|---|---|---|---|---|---|']
    for rep in reps:
        for cid in ORDER:
            group=scores[(scores.representation==rep)&(scores.condition==cid)]
            lines.append(f"| {rep} | {cid} | {group.test_accuracy.mean():.6f} ± {group.test_accuracy.std(ddof=1):.6f} | {group.test_macro_f1.mean():.6f} ± {group.test_macro_f1.std(ddof=1):.6f} | {group.train_test_gap_accuracy.mean():+.6f} | {group.train_test_gap_macro_f1.mean():+.6f} |")
    lines+=['','Accuracy is pooled across test windows. Macro-F1 is the arithmetic mean of six activity F1 scores, with zero contribution for an undefined class denominator. Accuracy and F1 gaps are training minus test; positive gaps indicate poorer held-out performance. Full train/validation/test losses, gap SDs and val−test differences are in condition_summary.csv and raw scores.csv.','',
        '## Predeclared paired differences','',
        'Only engineered D − B macro-F1 is primary. All other comparisons use unadjusted secondary intervals. Intervals below quantify seed-to-seed optimization variation conditional on this one fixed subject partition, not independent population sampling of windows or subjects.','',
        '| Representation | Contrast | Metric | Mean difference | Paired SD | 95% t CI | dz |',
        '|---|---|---|---|---|---|---|']
    for r in differences:
        lines.append(f"| {r['representation']} | {r['contrast']} | {r['metric']} | {r['mean']:+.6f} | {r['sd']:.6f} | [{r['ci_low']:+.6f}, {r['ci_high']:+.6f}] | {fmt(r['cohen_dz'])} |")
    lines+=['','Individual seed scores and paired differences are in replication/*individual_seed*.csv. Per-class F1 and confusion matrices are retained per model.','',
        '## Subject-level sensitivity','',
        'The same nine held-out people are used by all 20 models per condition. Treating 20×9 subject/model cells or overlapping windows as independent would exaggerate precision. The following descriptive intervals average D−B over model seeds within each subject, then use nine subject means (df=8). They describe subject heterogeneity within this small fixed held-out cohort and do not replace the primary seed analysis.','',
        '| Representation | Metric | Mean subject difference | Subject SD | Descriptive 95% t CI |','|---|---|---|---|---|']
    for r in subject_stats:
        lines.append(f"| {r['representation']} | {r['metric']} | {r['mean']:+.6f} | {r['sd']:.6f} | [{r['ci_low']:+.6f}, {r['ci_high']:+.6f}] |")
    lines+=['','## Training time and diagnostic measurements','',
        'Updates include minibatch gathering, gate construction/application, loss, backpropagation and the SGD step. Update process CPU time and accumulated wall time exclude diagnostic evaluations. Job wall time includes diagnostics, checkpoint saves and result writing. Six one-thread CPU workers run concurrently; independent condition order randomization reduces ordering bias, but cache/contention/host load remain. These measurements do not establish isolated inference latency or sparse-compute acceleration.','',
        '| Representation | Condition | Update process CPU seconds, mean ± SD | Update wall seconds, mean | Total job wall seconds, mean |','|---|---|---|---|---|']
    for rep in reps:
        for cid in ORDER:
            f=scores[(scores.representation==rep)&(scores.condition==cid)]
            lines.append(f'| {rep} | {cid} | {f.update_process_cpu_seconds.mean():.3f} ± {f.update_process_cpu_seconds.std(ddof=1):.3f} | {f.update_wall_seconds.mean():.3f} | {f.job_elapsed_wall_seconds.mean():.3f} |')
    lines+=['',
        'Learning curves use full training and validation metrics every 160 updates plus a same-weight release pair and final. Test is not drawn as a learning curve. Raw diagnostics distinguish gate-attenuated dL/dW and dL/dW_eff on a fixed paired training batch; plots separate conditions and layers by native distance bins. Effective-weight magnitude and per-destination-unit gate energy are retained. Checkpoint reloads compare native/dense gates on validation only; distant-quartile ablations measure validation dependence without retraining. No counterfactual test gates were evaluated.',
        'Positive gate-ablation loss deltas show dependence at those weights, not a necessary beneficial role during training. A static-local null far ablation can follow from already negligible far gates. The fixed-local conditions lack their own static shuffled controls in this benchmark; their dense-baseline comparisons do not isolate geometry from gate magnitude. The scheduled D/B contrast supplies the energy-matched primary control.','',
        '## Measured observations versus hypotheses and limits','',outcome,
        'A positive engineered D/B result would support transfer of the scheduled alignment effect under this specific metadata layout; a null or negative result does not establish a real-world CoNN advantage. The temporal analysis is a separate predeclared test of a more direct time-neighborhood prior, with different inputs and parameter count. Neither a file-order effect nor a temporal-window result proves a general geometry hypothesis or a new architecture.',
        'This is one sensor dataset and one fixed 16/5/9-subject partition. Twenty seeds are optimization repetitions, not 20 independent subject cohorts. Windows overlap and class/subject frequencies vary. The LR grid is narrow, only one network width and duration are tested, and no claim is made against tuned state-of-the-art HAR classifiers. Fixed local widths were transferred from synthetic studies, and secondary comparisons are numerous. Negative results bound this protocol rather than all possible coordinate systems, optimizers, network capacities or sensor tasks. Upstream preprocessing provenance limits any stronger end-to-end leakage claim.',
        'All favorable and unfavorable results are retained; there was no final-test-driven redesign, seed replacement, split search or spatial-layout selection. Existing synthetic results are preserved byte-for-byte and are not pooled with real-world results.','',
        '## Reproducibility','',
        'The versioned directory contains the dataset audit, original-document copies, archive SHA-256, subject split seal, coordinate tables, frozen train-only scalers, prepared-data hashes, frozen protocol and LR selection, environment/package lock, exact source snapshot, paired initial states and batch indices, masks, all raw curves and diagnostics, native checkpoints, test predictions, CSV summaries and plots. No Git repository was present; source_manifest.json identifies the content-addressed code revision. Before/after manifests verify all prior runs remain unchanged. Regenerate this report with python -m conn_exp.experiment2_report <run_directory>.','']
    (root/'research_report.md').write_text('\n'.join(lines),encoding='utf-8')
    dump(root/'report.generated.json',dict(primary=primary,figures=len(list(plots.glob('*.png'))),
        tuning_test_model_evaluations=0,final_test_model_evaluations=int(scores.test_evaluations.sum())))


if __name__=='__main__':
    import sys
    build_report(Path(sys.argv[1]))
