"""Configurable terminal gates and paired training for Experiment 1C.

The original task, MLP, initialization, SGD update, batching and RNG streams are reused.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from conn_exp.diagnostics import bin_mean_abs, far_suppression, prestep_gradients
from conn_exp.gates import apply_row_permutation, gaussian_gate, summarize_gate
from conn_exp.models import CoNNMLP, build_optimizer, fingerprint_state
from conn_exp.schedules import EXACT_FACTOR, _tau_between
from conn_exp.train import batch_indices, evaluate_split


@dataclass(frozen=True)
class TerminalPolicy:
    name: str
    path: str
    terminal_d50: float
    dense_release: bool = False

    def __post_init__(self):
        if self.path not in ('fixed', 'smooth') or self.terminal_d50 <= 0:
            raise ValueError('Invalid terminal policy')
        if self.dense_release and self.path != 'smooth':
            raise ValueError('Dense release follows the smooth pre-release path')

    def at(self, step, config):
        if not 0 <= step < config.total_steps:
            raise ValueError('Step out of range')
        if self.dense_release and step >= config.release_step:
            return None, True
        end = self.terminal_d50 / EXACT_FACTOR
        if self.path == 'fixed' or step >= config.release_step:
            return end, False
        return _tau_between(step, 0, config.release_step, config.tau_start, end), False


POLICIES = {
    'S': TerminalPolicy('S', 'fixed', .08),
    'F': TerminalPolicy('F', 'fixed', .25),
    'P': TerminalPolicy('P', 'smooth', .25),
    'W': TerminalPolicy('W', 'smooth', 1.),
    'H': TerminalPolicy('H', 'fixed', 1.),
    'D': TerminalPolicy('D', 'smooth', 1., True),
}


@dataclass(frozen=True)
class Variant:
    policy: str
    shuffled: bool = False
    alignment_fraction: float = 0.

    @property
    def id(self):
        if self.policy == 'A':
            return 'A'
        return f'{self.policy}_{"shuffle" if self.shuffled else "aligned"}_q{self.alignment_fraction:g}'


def variants(task):
    core = [Variant('A')] + [Variant(p, shuffled) for p in POLICIES for shuffled in [False, True]]
    if task == 'and':
        core += [Variant(p, False, q) for q in [.5, 1.] for p in ['S', 'P', 'D']]
    return core


def alignment_permutation(n, seed, fraction):
    """Label-independent, nested affected sets, with exactly fraction*n coordinates moved.

    An independent stream (seed + 4,000,003) orders coordinates. Cyclically rotate
    the selected prefix. The mapping of a boundary coordinate can change with dose.
    """
    if not 0 <= fraction <= 1:
        raise ValueError('Alignment fraction outside [0,1]')
    perm = torch.arange(n)
    k = int(round(fraction*n))
    if k == 1:
        raise ValueError('A permutation cannot move exactly one coordinate')
    if k:
        order = np.random.default_rng(int(seed)+4_000_003).permutation(n)
        chosen = torch.as_tensor(order[:k].copy())
        perm[chosen] = chosen.roll(1)
    return perm


class PolicyGates:
    def __init__(self, geometry, masks, variant, seed):
        self.geometry, self.masks, self.variant = geometry, masks, variant
        self.perm = alignment_permutation(geometry.dist_ih.shape[1], seed, variant.alignment_fraction)
        self.dist_ih = geometry.dist_ih[:, self.perm]
        self.dist_hh = geometry.dist_hh
        self.cached = {}

    def build(self, step, config):
        if self.variant.policy == 'A':
            tau, exact = None, True
        else:
            tau, exact = POLICIES[self.variant.policy].at(step, config)
        static = exact or self.variant.policy == 'A' or POLICIES[self.variant.policy].path == 'fixed' or step >= config.release_step
        key = (tau, exact)
        if static and key in self.cached:
            return self.cached[key]
        if exact:
            ih, hh = torch.ones_like(self.dist_ih), torch.ones_like(self.dist_hh)
        else:
            ih, _ = gaussian_gate(self.dist_ih, tau, config.tau_floor)
            hh, _ = gaussian_gate(self.dist_hh, tau, config.tau_floor)
        if self.variant.shuffled:
            ih = apply_row_permutation(ih, self.masks.row_perm_ih)
            hh = apply_row_permutation(hh, self.masks.row_perm_hh)
        result = ih, hh
        if static:
            self.cached[key] = result
        return result


def train_variant(config, variant, seed, init, masks, geometry, order, dataset, job,
                  replication=False, diagnostics=True):
    """Exactly total_steps SGD updates; test data accessed only at the native final gate."""
    job = Path(job)
    job.mkdir(parents=True, exist_ok=False)
    payload = config.to_resolved_dict()
    payload.update(variant=asdict(variant), variant_id=variant.id,
        terminal_policy=asdict(POLICIES[variant.policy]) if variant.policy != 'A' else 'exact_dense_every_step',
        lr_selection='shared_task_validation_grid_1c', lr_grid_executed=True,
        test_protocol='native_final_once' if replication else 'forbidden',
        alignment_stream_seed=int(seed)+4_000_003)
    body = json.dumps(payload, indent=2, sort_keys=True)+'\n'
    config_hash = hashlib.sha256(body.encode()).hexdigest()
    (job/'config.resolved.json').write_text(body, encoding='utf-8', newline='\n')
    (job/'config_sha256.txt').write_text(config_hash+'\n')
    model = CoNNMLP(config.n_features, config.hidden_sizes)
    model.load_state_dict(init)
    optimizer, hparams = build_optimizer(model, config.lr)
    gates = PolicyGates(geometry, masks, variant, seed)
    xtr, ytr = torch.from_numpy(dataset.train.x), torch.from_numpy(dataset.train.y)
    xv, yv = torch.from_numpy(dataset.val.x), torch.from_numpy(dataset.val.y)
    index_diag = torch.as_tensor(batch_indices(order, config.batch_size, 0))
    xd, yd = xtr[index_diag], ytr[index_diag]
    base = dict(seed=seed, task=config.task, variant_id=variant.id, policy=variant.policy,
        shuffled=variant.shuffled, alignment_fraction=variant.alignment_fraction, lr=config.lr)
    records = {k: [] for k in ['metrics', 'gradients', 'row_energy', 'ablation', 'checkpoint_comparisons']}
    checkpoints = {}
    fingerprints = dict(data_fingerprint=dataset.fingerprint(), init_fingerprint=fingerprint_state(init),
        order_sha256=hashlib.sha256(order.tobytes()).hexdigest())
    test_evaluations = 0

    def measure(step, completed, kind, save=False, final=False):
        nonlocal test_evaluations
        ih, hh = gates.build(step, config)
        weight_stats = {}
        for layer, weight, gate in [('ih', model.fc1.weight, ih), ('hh', model.fc2.weight, hh)]:
            eff = weight.detach()*gate
            stats = summarize_gate(gate)
            weight_stats.update({f'mean_abs_W_eff_{layer}': float(eff.abs().mean()),
                f'norm_W_eff_{layer}': float(torch.linalg.vector_norm(eff)),
                f'norm_W_{layer}': float(torch.linalg.vector_norm(weight.detach())),
                f'gate_mean_square_{layer}': stats.energy, f'row_energy_std_{layer}': stats.row_energy_std})
            for unit, energy in enumerate(stats.row_energy.tolist()):
                records['row_energy'].append(dict(**base, optimizer_steps_completed=completed, kind=kind,
                    layer=layer, unit=unit, energy=energy))
        with torch.no_grad():
            _, cache = model(xd, ih, hh)
            weight_stats.update(act_var_h1=float(cache.h1.var(correction=0)), act_var_h2=float(cache.h2.var(correction=0)))
        common = dict(**base, step_index=step, optimizer_steps_completed=completed, kind=kind, **weight_stats)
        val_loss, val_accuracy = evaluate_split(model, xv, yv, ih, hh)
        train_loss, train_accuracy = evaluate_split(model, xtr, ytr, ih, hh)
        for split, loss, accuracy in [('train', train_loss, train_accuracy), ('val', val_loss, val_accuracy)]:
            if not np.isfinite([loss, accuracy]).all():
                raise RuntimeError(f'Nonfinite diagnostic {job}')
            records['metrics'].append(dict(**common, split=split, loss=loss, accuracy=accuracy))
        if diagnostics:
            grads = prestep_gradients(model, xd, yd, ih, hh)
            for layer, distance in [('ih', geometry.dist_ih), ('hh', geometry.dist_hh)]:
                for quantity in ['dL_dW', 'dL_dW_eff']:
                    binned = bin_mean_abs(grads[layer][quantity], distance, config.distance_bin_edges)
                    for b in binned:
                        records['gradients'].append(dict(**base, optimizer_steps_completed=completed, kind=kind,
                            layer=layer, quantity=quantity, **b, gradient_batch='fixed_first_paired_minibatch',
                            gradient_timing='fixed_weights_before_any_next_update', distance='base_task_geometry'))
            suppressed = far_suppression(ih, hh, geometry.dist_ih, geometry.dist_hh, config.far_quantile)
            for intervention, gi, gh in [('both', suppressed['gate_ih'], suppressed['gate_hh']),
                    ('ih', suppressed['gate_ih'], hh), ('hh', ih, suppressed['gate_hh'])]:
                loss, acc = evaluate_split(model, xv, yv, gi, gh)
                records['ablation'].append(dict(**base, optimizer_steps_completed=completed, kind=kind,
                    intervention=intervention, split='val', far_quantile=config.far_quantile,
                    threshold_ih=suppressed['threshold_ih'], threshold_hh=suppressed['threshold_hh'],
                    val_loss=val_loss, val_accuracy=val_accuracy, suppressed_val_loss=loss,
                    suppressed_val_accuracy=acc, val_loss_delta=loss-val_loss, val_accuracy_delta=acc-val_accuracy,
                    distance='base_task_geometry', threshold_method='linear_quantile'))
        if save:
            checkpoint = dict(model={k: v.detach().clone() for k, v in model.state_dict().items()},
                optimizer=optimizer.state_dict(), gate_ih=ih.clone(), gate_hh=hh.clone(),
                **base, optimizer_steps_completed=completed, step_index=step, kind=kind, config_sha256=config_hash)
            path = job / f'{kind}.pt'
            torch.save(checkpoint, path)
            checkpoints[kind] = fingerprint_state(checkpoint['model'])
            # Reload the serialized checkpoint for counterfactual gate evaluations.
            reloaded = torch.load(path, map_location='cpu', weights_only=False)
            clone = CoNNMLP(config.n_features, config.hidden_sizes)
            clone.load_state_dict(reloaded['model'])
            before = fingerprint_state(clone.state_dict())
            for mode, gi, gh in [('native', reloaded['gate_ih'], reloaded['gate_hh']),
                    ('exact_dense', torch.ones_like(ih), torch.ones_like(hh))]:
                loss, acc = evaluate_split(clone, xv, yv, gi, gh)
                records['checkpoint_comparisons'].append(dict(**base, optimizer_steps_completed=completed,
                    kind=kind, gate_mode=mode, split='val', loss=loss, accuracy=acc,
                    weight_fingerprint=before, checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            assert fingerprint_state(clone.state_dict()) == before
        if final and replication:
            # The only use of these tensors in the training procedure.
            test_loss, test_accuracy = evaluate_split(model, torch.from_numpy(dataset.test.x),
                torch.from_numpy(dataset.test.y), ih, hh)
            test_evaluations += 1
            records['metrics'].append(dict(**common, split='test', loss=test_loss, accuracy=test_accuracy))

    if diagnostics:
        measure(0, 0, 'initial')
    for step in range(config.total_steps):
        if step == config.release_step:
            measure(step, step, 'post_release', save=True)
        elif diagnostics and step > 0 and step % config.eval_interval == 0:
            measure(step, step, 'curve')
        ih, hh = gates.build(step, config)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        index = torch.as_tensor(batch_indices(order, config.batch_size, step))
        logits, _ = model(xtr[index], ih, hh)
        loss = F.binary_cross_entropy_with_logits(logits, ytr[index])
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite training loss {job} step {step}')
        loss.backward()
        optimizer.step()
        if step+1 == config.release_step:
            measure(step, step+1, 'pre_release', save=True)
    measure(config.total_steps-1, config.total_steps, 'final', save=True, final=True)
    assert checkpoints['pre_release'] == checkpoints['post_release']
    for name, rows in records.items():
        pd.DataFrame(rows).to_csv(job/f'{name}.csv', index=False)
    final = [r for r in records['metrics'] if r['kind'] == 'final']
    score = dict(**base, **fingerprints, total_steps=config.total_steps,
        parameter_count=sum(p.numel() for p in model.parameters()), test_evaluations=test_evaluations,
        config_sha256=config_hash, pre_release_weight_fingerprint=checkpoints['pre_release'])
    for r in final:
        score[r['split']+'_loss'], score[r['split']+'_accuracy'] = r['loss'], r['accuracy']
    assert test_evaluations == int(replication)
    return score, model.state_dict()
