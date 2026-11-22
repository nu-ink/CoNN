"""Regenerable 1C analysis: measured statistics, curves and mechanism diagnostics."""
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
from conn_exp.experiment1c_core import POLICIES, variants

COLORS = dict(A='#777777', S='#e377c2', F='#9467bd', P='#17becf', W='#bcbd22', H='#8c564b', D='#2ca02c')
KINDS = ['initial', 'curve', 'pre_release', 'post_release', 'final']
SNAPSHOTS = ['pre_release', 'post_release', 'final']


def fmt(value, digits=6):
    return 'undefined' if value is None else f'{value:.{digits}f}'


def save(fig, folder, name):
    fig.tight_layout(); fig.savefig(folder/name, dpi=140); plt.close(fig)


def read_jobs(phase, scores, filename):
    return pd.concat([pd.read_csv(phase/r.job/(filename+'.csv')) for r in scores.itertuples()], ignore_index=True)


def contrast_values(pivot, terms):
    result = np.zeros(len(pivot), dtype=float)
    for variant_id, coefficient in terms.items():
        result += coefficient*pivot[variant_id].to_numpy()
    return result


def definitions(task):
    p = lambda name: name+'_aligned_q0'
    result = [
        ('P_minus_D', {p('P'):1, p('D'):-1}),
        ('W_minus_D', {p('W'):1, p('D'):-1}),
        ('S_minus_D', {p('S'):1, p('D'):-1}),
        ('P_minus_F_same_terminal_width', {p('P'):1, p('F'):-1}),
        ('W_minus_H_same_terminal_width', {p('W'):1, p('H'):-1}),
        ('alignment_by_expansion_interaction', {p('P'):1, 'P_shuffle_q0':-1, p('S'):-1, 'S_shuffle_q0':1}),
    ]
    result += [(policy+'_aligned_minus_shuffle', {p(policy):1, policy+'_shuffle_q0':-1}) for policy in POLICIES]
    result += [(policy+'_minus_dense_baseline', {p(policy):1, 'A':-1}) for policy in ['S','P','D']]
    if task == 'and':
        for policy in ['S','P','D']:
            for q in [.5,1.]:
                result.append((f'{policy}_q{q:g}_minus_q0', {f'{policy}_aligned_q{q:g}':1, p(policy):-1}))
                result.append((f'{policy}_q{q:g}_minus_dense_baseline', {f'{policy}_aligned_q{q:g}':1, 'A':-1}))
    return result


def curve_line(axis, frame, metric, policy, style, label):
    g = frame.groupby(['optimizer_steps_completed','kind'], as_index=False)[metric].agg(['mean','sem']).reset_index()
    g['kind_order'] = g.kind.map({kind:i for i,kind in enumerate(KINDS)})
    g = g.sort_values(['optimizer_steps_completed','kind_order'])
    x = g.optimizer_steps_completed.to_numpy()
    y = g['mean'].to_numpy()
    axis.plot(x,y,style,color=COLORS[policy],lw=1.4,label=label)
    if style == '-':
        se = g['sem'].to_numpy()
        axis.fill_between(x,y-se,y+se,color=COLORS[policy],alpha=.08)


def curve_plots(metrics, folder, task):
    for shuffled, label in [(False,'aligned'),(True,'row_shuffled')]:
        fig, axes = plt.subplots(1,2,figsize=(12,4.5))
        frame = metrics[(metrics.task == task) & (metrics.alignment_fraction == 0)]
        for policy in ['A']+list(POLICIES):
            selected = frame[(frame.policy == policy) & ((frame.shuffled == shuffled) | (frame.policy == 'A'))]
            for split,style in [('val','-'),('train','--')]:
                for metric,axis in zip(['accuracy','loss'],axes):
                    curve_line(axis,selected[selected.split == split],metric,policy,style,policy if split=='val' else '_nolegend_')
        for metric,axis in zip(['accuracy','loss'],axes):
            axis.axvline(2240,color='gray',ls=':',lw=.8)
            axis.set_xlabel('optimizer steps completed');axis.set_ylabel(metric)
        axes[1].legend(ncol=2,fontsize=8)
        fig.suptitle(f'{task.upper()} {label}: validation solid ±1 SE, training dashed; no test curve')
        save(fig,folder,f'{task}_{label}_learning_curves.png')


def diagnostic_plots(phase, scores, metrics, folder, task):
    frame = metrics[(metrics.task == task) & (metrics.alignment_fraction == 0) & ~metrics.shuffled & (metrics.split == 'val')]
    fig, axes = plt.subplots(2,2,figsize=(12,8))
    for row,layer in zip(axes,['ih','hh']):
        for axis,metric in zip(row,['norm_W_eff_'+layer,'row_energy_std_'+layer]):
            for policy,g in frame.groupby('policy'):
                curve_line(axis,g,metric,policy,'-',policy)
            axis.set_ylabel(metric);axis.set_xlabel('optimizer steps completed')
    axes[0,1].legend(ncol=2,fontsize=8)
    fig.suptitle(task.upper()+': effective weight norms and gate-energy dispersion')
    save(fig,folder,f'{task}_weight_energy_diagnostics.png')
    chosen = scores[(scores.task == task) & (scores.alignment_fraction == 0) & ~scores.shuffled & scores.policy.isin(['S','P','W','D'])]
    gradients = read_jobs(phase,chosen,'gradients')
    for layer in ['ih','hh']:
        fig, axes = plt.subplots(2,4,figsize=(17,7),sharex=True,sharey='row')
        for column,policy in zip(axes.T,['S','P','W','D']):
            for axis,quantity in zip(column,['dL_dW','dL_dW_eff']):
                f = gradients[(gradients.policy==policy)&(gradients.layer==layer)&(gradients.quantity==quantity)].copy()
                for bid,g in f.groupby('bin_id'):
                    mean = g.groupby('optimizer_steps_completed').mean_abs.mean()
                    axis.plot(mean.index,mean,label=f'bin {bid}',lw=1.1)
                axis.set_title(policy+' '+quantity);axis.set_xlabel('steps')
                axis.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        axes[0,-1].legend(fontsize=7)
        fig.suptitle(f'{task.upper()} aligned {layer}: fixed-minibatch gradients by base task distance bin')
        save(fig,folder,f'{task}_{layer}_gradient_distance.png')
    # Structured q=0 gate energies do not depend on seed; show one declared seed.
    chosen = chosen[chosen.seed == chosen.seed.min()]
    energy = read_jobs(phase,chosen,'row_energy')
    fig, axes = plt.subplots(2,4,figsize=(17,7))
    for row,layer in zip(axes,['ih','hh']):
        for axis,policy in zip(row,['S','P','W','D']):
            f = energy[(energy.layer==layer)&(energy.policy==policy)].copy()
            f['point'] = f.optimizer_steps_completed.astype(str)+' '+f.kind
            order = f[['point','optimizer_steps_completed','kind']].drop_duplicates()
            order['kind_order'] = order.kind.map({kind:i for i,kind in enumerate(KINDS)})
            order = order.sort_values(['optimizer_steps_completed','kind_order'])
            pivot = f.pivot(index='unit',columns='point',values='energy')[order.point]
            im=axis.imshow(pivot,aspect='auto',origin='lower',interpolation='nearest')
            axis.set_title(policy+' '+layer+' sum G² per unit')
            ticks=np.linspace(0,len(pivot.columns)-1,4).astype(int)
            axis.set_xticks(ticks,[str(order.optimizer_steps_completed.iloc[i]) for i in ticks])
            axis.set_xlabel('optimizer steps');axis.set_ylabel('unit');fig.colorbar(im,ax=axis)
    fig.suptitle(f'{task.upper()}: per-unit aligned gate energy (identical across seeds by construction)')
    save(fig,folder,f'{task}_per_unit_gate_energy.png')


def build_report(root):
    root=Path(root)
    protocol=json.loads((root/'protocol.frozen.json').read_text())
    frozen=json.loads((root/'selection.frozen.json').read_text())
    tuning=pd.read_csv(root/'tuning/scores.csv')
    scores=pd.read_csv(root/'replication/scores.csv')
    assert len(tuning)==480 and len(scores)==640
    assert tuning.test_evaluations.eq(0).all() and 'test_accuracy' not in tuning
    assert scores.test_evaluations.eq(1).all()
    assert set(tuning.seed).isdisjoint(scores.seed)
    for task in ['and','xor']:
        f=scores[scores.task==task]
        assert len(f)==len(variants(task))*20
        assert set(f.seed)==set(protocol['replication_seeds'])
        assert f.lr.eq(frozen['learning_rate_by_task'][task]).all()
        assert not f[['test_accuracy','test_loss']].isna().any().any()
    summary=scores.groupby(['task','variant_id','policy','shuffled','alignment_fraction']).agg(
        n=('seed','size'),mean_accuracy=('test_accuracy','mean'),sd_accuracy=('test_accuracy','std'),
        mean_loss=('test_loss','mean'),sd_loss=('test_loss','std')).reset_index()
    summary.to_csv(root/'replication/condition_summary.csv',index=False)
    contrasts, individual=[],[]
    for task in ['and','xor']:
        data=scores[scores.task==task]
        wide=data.pivot(index='seed',columns='variant_id',values='test_accuracy')
        wide.to_csv(root/f'replication/{task}_individual_seed_accuracy.csv')
        data.pivot(index='seed',columns='variant_id',values='test_loss').to_csv(root/f'replication/{task}_individual_seed_loss.csv')
        for name,terms in definitions(task):
            values=contrast_values(wide,terms)
            primary=task=='and' and name=='P_minus_D'
            result=dict(task=task,contrast=name,primary=primary,metric='native_final_test_accuracy',**summarize(values))
            contrasts.append(result)
            individual.extend(dict(task=task,contrast=name,seed=int(seed),delta=float(value),primary=primary)
                for seed,value in zip(wide.index,values))
    statistics=pd.DataFrame(contrasts)
    statistics.to_csv(root/'replication/paired_statistical_summary.csv',index=False)
    pd.DataFrame(individual).to_csv(root/'replication/individual_seed_contrasts.csv',index=False)
    primary=[r for r in contrasts if r['primary']][0]
    dump(root/'replication/primary_summary.json',primary)
    phase=root/'replication'
    metrics=read_jobs(phase,scores,'metrics')
    metrics.to_csv(phase/'training_curves.csv',index=False)
    checkpoints=read_jobs(phase,scores,'checkpoint_comparisons')
    checkpoints.to_csv(phase/'checkpoint_comparisons.csv',index=False)
    checkpoints.groupby(['task','variant_id','kind','gate_mode'])[['accuracy','loss']].agg(['mean','std']).to_csv(phase/'checkpoint_summary.csv')
    ablations=read_jobs(phase,scores,'ablation')
    final_ablation=ablations[ablations.kind=='final']
    final_ablation.groupby(['task','variant_id','intervention'])[['val_loss_delta','val_accuracy_delta']].agg(['mean','std']).to_csv(phase/'final_ablation_summary.csv')
    changes=[]
    for (task,seed,vid),group in metrics[(metrics.split=='val')&metrics.kind.isin(SNAPSHOTS)].groupby(['task','seed','variant_id']):
        f=group.set_index('kind')
        for metric in ['accuracy','loss']:
            for effect,a,b in [('gate_only','post_release','pre_release'),('optimization_after_release','final','post_release')]:
                changes.append(dict(task=task,seed=seed,variant_id=vid,metric=metric,effect=effect,delta=float(f.loc[a,metric]-f.loc[b,metric])))
    changes=pd.DataFrame(changes)
    changes.to_csv(phase/'separated_gate_optimization_changes.csv',index=False)
    changes.groupby(['task','variant_id','metric','effect']).delta.agg(['mean','std']).to_csv(phase/'separated_change_summary.csv')
    convergence=[]
    for (task,seed,vid),group in metrics[metrics.split=='val'].groupby(['task','seed','variant_id']):
        for threshold in [.8,.9,.95]:
            crossed=group[group.accuracy>=threshold]
            convergence.append(dict(task=task,seed=seed,variant_id=vid,threshold=threshold,
                first_measured_step=None if crossed.empty else int(crossed.optimizer_steps_completed.min()),
                reached=not crossed.empty))
    convergence=pd.DataFrame(convergence)
    convergence.to_csv(phase/'convergence.csv',index=False)
    convergence.groupby(['task','variant_id','threshold']).agg(reached_seeds=('reached','sum'),
        mean_first_measured_step_among_reached=('first_measured_step','mean')).to_csv(phase/'convergence_summary.csv')
    plots=root/'plots';plots.mkdir(exist_ok=True)
    for task in ['and','xor']:
        curve_plots(metrics,plots,task)
        diagnostic_plots(phase,scores,metrics,plots,task)
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        f=checkpoints[(checkpoints.task==task)&(checkpoints.alignment_fraction==0)&~checkpoints.shuffled]
        for policy in ['S','P','W','D']:
            for mode,style in [('native','-'),('exact_dense','--')]:
                g=f[(f.policy==policy)&(f.gate_mode==mode)].groupby('kind')[['accuracy','loss']].mean().reindex(SNAPSHOTS)
                for metric,axis in zip(['accuracy','loss'],axes):
                    axis.plot(range(3),g[metric],style,marker='o',color=COLORS[policy],label=f'{policy} {mode}')
                    axis.set_xticks(range(3),['pre release','post release','final']);axis.set_ylabel('validation '+metric)
        axes[1].legend(fontsize=7,ncol=2);fig.suptitle(task.upper()+': gate interventions on reloaded checkpoint weights')
        save(fig,plots,f'{task}_checkpoint_native_dense.png')
        fig,axes=plt.subplots(1,3,figsize=(16,4.5))
        f=final_ablation[(final_ablation.task==task)&(final_ablation.alignment_fraction==0)&final_ablation.policy.isin(['S','P','W','D'])]
        for intervention,axis in zip(['ih','hh','both'],axes):
            g=f[f.intervention==intervention].groupby('variant_id').val_loss_delta.agg(['mean','sem'])
            axis.bar(range(len(g)),g['mean'],yerr=g['sem'],capsize=2)
            axis.set_xticks(range(len(g)),g.index,rotation=90,fontsize=7)
            axis.axhline(0,color='gray',lw=.7);axis.set_title('Suppress far '+intervention)
            axis.set_ylabel('validation loss delta ±1 SE')
        fig.suptitle(task.upper()+': final dependence on distant base-geometry connections')
        save(fig,plots,f'{task}_final_distant_ablation.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for policy in ['S','P','D']:
        f=scores[(scores.task=='and')&(scores.policy==policy)&~scores.shuffled]
        for quantity,axis in zip(['test_accuracy','test_loss'],axes):
            g=f.groupby('alignment_fraction')[quantity].agg(['mean','sem'])
            axis.errorbar(g.index,g['mean'],yerr=g['sem'],label=policy,color=COLORS[policy],marker='o',capsize=3)
            baseline=scores[(scores.task=='and')&(scores.policy=='A')][quantity].mean()
            axis.axhline(baseline,color='gray',ls=':',lw=.8)
            axis.set_xlabel('fraction of input coordinates permuted');axis.set_ylabel(quantity)
            axis.set_xticks([0,.5,1])
    axes[1].legend();fig.suptitle('AND alignment degradation; gray dotted = dense baseline; ±1 SE')
    save(fig,plots,'and_alignment_degradation.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for task,axis in zip(['and','xor'],axes):
        f=scores[(scores.task==task)&(scores.alignment_fraction==0)]
        g=f.groupby('variant_id').test_accuracy.agg(['mean','sem'])
        axis.bar(range(len(g)),g['mean'],yerr=g['sem'],capsize=2)
        axis.set_xticks(range(len(g)),g.index,rotation=90,fontsize=7)
        axis.set_ylabel('mean native final test accuracy ±1 SE');axis.set_title(task.upper())
    save(fig,plots,'core_condition_comparison.png')
    fig,axis=plt.subplots(figsize=(10,4.5))
    values=pd.DataFrame(individual);values=values[values.primary]
    axis.bar(values.seed.astype(str),100*values.delta,color=COLORS['P']);axis.axhline(0,color='gray',lw=.8)
    axis.set_ylabel('P − D test accuracy (percentage points)');axis.set_xlabel('independent paired seed')
    axis.tick_params(axis='x',rotation=90);axis.set_title('Sole primary contrast: partial widening versus dense release on AND')
    save(fig,plots,'primary_paired_seed_differences.png')
    validation=tuning.groupby(['task','variant_id','lr']).agg(mean_val_accuracy=('val_accuracy','mean'),mean_val_loss=('val_loss','mean')).reset_index()
    validation.to_csv(root/'tuning/validation_grid.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,7))
    for task,axis in zip(['and','xor'],axes):
        p=validation[validation.task==task].pivot(index='variant_id',columns='lr',values='mean_val_accuracy')
        im=axis.imshow(p,aspect='auto',vmin=.5,vmax=1.)
        axis.set_xticks(range(3),p.columns);axis.set_yticks(range(len(p)),p.index,fontsize=7)
        axis.set_xlabel('learning rate');axis.set_title(task.upper()+' tuning validation')
        fig.colorbar(im,ax=axis)
    save(fig,plots,'validation_grid.png')
    conclusions=('Partial widening outperformed full dense release in the independent primary comparison.' if primary['mean']>0 and not primary['spans_zero'] else
        'Full dense release outperformed partial widening in the independent primary comparison.' if primary['mean']<0 and not primary['spans_zero'] else
        'The primary comparison did not resolve an advantage of partial widening over full dense release; its interval includes zero.')
    lines=['# CoNN Experiment 1C — Locality Versus Collapse','',conclusions,'',
        f"Measured sole primary result (AND, n=20 new paired seeds): P − D = {primary['mean']:+.8f} ({100*primary['mean']:+.4f} percentage points); sample SD {primary['sd']:.8f}; two-sided 95% paired Student-t CI [{primary['ci_low']:.8f}, {primary['ci_high']:.8f}], df=19; paired Cohen dz {fmt(primary['cohen_dz'])}. Positive/tied/negative seeds: {primary['positive_seeds']}/{primary['tied_seeds']}/{primary['negative_seeds']}.",'',
        '## Phase 1: audit completed before new experimental training','',
        'The frozen 1B protocol, diagnostics, checkpoint ablations and full training curves were read before fitting the new experiment. The audit reloaded D and S weights and measured native/dense gate interventions on train and validation only, without weight updates. The full phase-1 diagnostic report and source measurement labels are in audit/diagnostic_report.md and audit/*.csv.',
        'In the existing 1B validation measurements, D moved 0.916016 → 0.902344 at the same-weight exact gate switch, then to 0.909375 after 960 updates. S moved 0.872266 → 0.944141 over the same final optimization interval with unchanged local gates. Forcing dense inference on final S weights scored 0.770703 validation accuracy instead of native 0.944141. These are exploratory audit observations, not independent 1C evidence.',
        'Missing historical test measurements before release and under counterfactual gates, earlier unsaved weights, and a trained wide-Gaussian-hold branch were identified rather than reconstructed. The historical 1B D − B′ +0.04843750 result remains unchanged and is not pooled with 1C.', '',
        '## Frozen controlled design','',
        'All new fits use the original 100-feature synthetic generators, split sizes 512/128/128, two 32-unit hidden layers, 4321 parameters, paired initialization and minibatch order, and plain SGD without momentum or weight decay. Every model receives exactly 3200 updates. AND data difficulty was not changed. XOR uses the existing balanced four-cell generator, without task-specific redesign.',
        'S: fixed d50=.08. F: fixed d50=.25. P: smooth d50=.08→.25 by update 2240, then hold. W: smooth d50=.08→1 by update 2240, then retain the wide Gaussian. H: fixed d50=1 Gaussian. D: original smooth .08→1 path followed by exact dense release at 2240. All six policies have a frozen within-row shuffled gate control matching each row’s gate multiset and energy. A is exact dense throughout.',
        'W and D share bitwise-identical pre-release weights within each seed, task, LR and shuffle condition; only their terminal policy differs. P versus F and W versus H compare schedules at matched terminal Gaussian widths. S versus its shuffle tests locality alignment without expansion. The alignment-by-expansion interaction is (P−P_shuffle)−(S−S_shuffle).',
        'The AND alignment study permutes 0%, 50% or 100% of input coordinates for S, P and D. Feature values, labels, hidden geometry and parameter count remain unchanged. The affected subsets are nested; permutation mappings can change with dose. Per-row gate energy remains matched. This is input alignment degradation, not a full disruption of all hidden-layer geometry.',
        'Every one of the 32 task/variant combinations received LR [.01,.05,.1] × five new tuning seeds 10–14: 480 validation-only fits, 48,000 optimizer updates per variant. A single LR per task maximized macro final validation accuracy across the 13 uncorrupted core variants; ties used lower validation loss then lower LR. Degradation cells received the same grid but did not drive selection. No tuning test scores were measured.',
        f"Frozen shared learning rates: AND {frozen['learning_rate_by_task']['and']:g}; XOR {frozen['learning_rate_by_task']['xor']:g}. The protocol declared contrasts before tuning; the sealed selection preceded any replication data/model generation. Replication used seeds 200–219: 640 models with one native final test evaluation each, no test-based adaptation or replacement seeds.", '',
        '## Independent replication means','',
        '| Task | Variant | Mean test accuracy | Sample SD | Mean test loss | Sample SD loss |','|---|---|---|---|---|---|']
    for r in summary.itertuples():
        lines.append(f'| {r.task} | {r.variant_id} | {r.mean_accuracy:.6f} | {r.sd_accuracy:.6f} | {r.mean_loss:.6f} | {r.sd_loss:.6f} |')
    lines += ['', '## Predeclared contrast statistics', '',
        'Only AND P − D is primary. All other intervals below are unadjusted secondary descriptions; they do not support a collection of confirmatory claims. Effect sizes standardize paired differences by their sample SD; undefined dz means zero paired SD.', '',
        '| Task | Contrast | Mean difference | Paired SD | 95% CI | Paired dz |','|---|---|---|---|---|---|']
    focus=['P_minus_D','W_minus_D','S_minus_D','P_minus_F_same_terminal_width','W_minus_H_same_terminal_width',
        'S_aligned_minus_shuffle','P_aligned_minus_shuffle','D_aligned_minus_shuffle','alignment_by_expansion_interaction']
    for r in contrasts:
        if r['contrast'] in focus:
            lines.append(f"| {r['task']} | {r['contrast']} | {r['mean']:+.6f} | {r['sd']:.6f} | [{r['ci_low']:+.6f}, {r['ci_high']:+.6f}] | {fmt(r['cohen_dz'])} |")
    lines += ['', 'Every individual seed accuracy/loss and paired contrast is retained in replication/*individual_seed*.csv. Full secondary statistics, including degradation and dense-baseline comparisons, are in paired_statistical_summary.csv.', '',
        '## Gate intervention versus adaptation', '',
        '| Task | Policy | Immediate gate-switch validation accuracy change | Subsequent 960-update accuracy change |', '|---|---|---|---|']
    for task in ['and','xor']:
        for policy in ['S','P','W','D']:
            f=changes[(changes.task==task)&(changes.variant_id==policy+'_aligned_q0')&(changes.metric=='accuracy')].groupby('effect').delta.mean()
            lines.append(f"| {task} | {policy} | {f['gate_only']:+.6f} | {f['optimization_after_release']:+.6f} |")
    lines += ['', 'Checkpoint comparisons reload the serialized native state, then evaluate native and exact dense gates without weight changes. These are validation interventions. Only D’s native gate jumps to exact dense at the boundary; retained-Gaussian policies have zero same-weight switch change. The release pair is not confused with later optimization.', '',
        '## Measured alignment degradation', '',
        '| Policy | Corrupted input-coordinate fraction | Mean AND test accuracy | Mean AND test loss |', '|---|---|---|---|']
    for r in summary[(summary.task=='and')&~summary.shuffled&summary.policy.isin(['S','P','D'])].sort_values(['policy','alignment_fraction']).itertuples():
        lines.append(f'| {r.policy} | {r.alignment_fraction:g} | {r.mean_accuracy:.6f} | {r.mean_loss:.6f} |')
    lines += ['', '## Convergence, gradients and distant connections', '',
        'Raw curves measure full training and validation losses/accuracies every 160 updates, with explicit same-weight release snapshots and the final checkpoint. convergence.csv records the first sampled crossing of validation accuracy .8/.9/.95 and explicitly leaves unreached thresholds missing. This measures sampled threshold crossings, not an interpolated convergence step or sustained convergence. See convergence_summary.csv for reached counts and means among reached seeds.',
        'Per-job gradients distinguish gate-attenuated dL/dW from dL/dW_eff on the same fixed paired minibatch, binned by original task-coordinate distance. Effective weight norms, activation variances and each destination unit’s sum G² are recorded. Gradient plots separate policies, layers and distance bins. Unit-energy audits cover shuffled controls and coordinate degradation; changing a policy can itself change gate magnitude and is not an energy-normalized policy contrast.',
        'Distant ablations suppress the top base-distance quartile, in input/hidden layers separately and jointly, using validation only. final_ablation_summary.csv and checkpoint_summary.csv contain means and SDs; every raw intervention remains in its job directory. An ablation loss increase measures dependence at those weights, not a necessary causal contribution during training. Near-zero far gates can make a static-local ablation null by construction.', '',
        '## What the results do and do not determine', '',
        conclusions,
        'The W − D branch comparison tests exact dense release followed by adaptation with a common prehistory. It is distinct from the S − D comparison, which changes the entire training path. Aligned-minus-shuffled contrasts measure source-alignment/topology effects under matched row energy at each policy; the predeclared interaction asks whether that effect changes with partial expansion. Matched-terminal fixed-versus-scheduled controls test whether early local history matters beyond the endpoint. The tables report these effects separately rather than assigning an unsupported percentage of the benefit to “geometry” or “schedule.”',
        'Static-local superiority, if present, does not prove that widening is harmful. Final S weights can be co-adapted to small gates; forcibly removing that gating is not equivalent to training under a wide policy. Width changes also alter effective weight magnitude. Row shuffling changes alignment and higher-order mask topology together, so it does not prove a pure geometry hypothesis.',
        'Limitations: two synthetic tasks, a narrow three-value SGD LR grid, one .25 partial terminal width and .7 release time, one 3200-step budget, five tuning seeds, 20 replication seeds and 128 test examples per seed. Task-specific shared LRs preserve mechanism pairing but differ from condition-specific 1B tuning. Coordinate degradation is one prescribed family, does not change hidden alignment, and is not a universal corruption benchmark. Secondary contrasts are numerous and unadjusted. Gate-switch and optimization effects are conditional on this exact protocol; they are not architecture-level claims.', '',
        '## Reproducibility and preservation', '',
        'protocol.frozen.json predates all experimental fits and declares the primary/secondary contrasts. selection.frozen.json and replication.started.json record the configuration freeze ordering. Configurations, SHA-256 hashes, full source snapshot, package lock, environment/runtime flags, independent stream labels, data, paired initial states, masks, alignment permutations, model/optimizer checkpoints, raw measurements and figures are versioned here. No Git repository was present; the content-addressed source snapshot identifies the code revision.',
        'Prior 1A/pilot/smoke and 1B directories are checked byte-for-byte by before/after manifests. The original 1B result remains historical independent evidence. This study adds evidence about locality, expansion and their interaction under controlled settings; it neither claims a new architecture nor proves the geometry hypothesis.', '']
    (root/'research_report.md').write_text('\n'.join(lines),encoding='utf-8')
    dump(root/'report.generated.json',dict(primary=primary,figures=len(list(plots.glob('*.png'))),
        replication_test_evaluations=int(scores.test_evaluations.sum()),tuning_test_evaluations=0))


if __name__=='__main__':
    import sys
    build_report(Path(sys.argv[1]))
