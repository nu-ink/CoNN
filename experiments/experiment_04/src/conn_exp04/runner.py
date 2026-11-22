"""Incremental, fail-recording paired training; a separate evaluation command reads tests."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time
import traceback
import uuid

import numpy as np
import psutil
import torch
from torch.nn import functional as F

from . import ROOT
from .core import (Config, CONDITIONS, TASKS, Gates, generate_split, arrays_hash, stream,
                   digest_json, CoNNMLP, initial_state_dict, fingerprint_state, build_optimizer,
                   trainable_count, evaluate, weight_distance)

BASE = ROOT / 'experiments' / 'experiment_04'


def write_json(path, value):
    # atomic replacement is used only for the status of this newly created run.
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def configure(cfg):
    torch.set_num_threads(cfg.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False


def source_manifest():
    paths = list((BASE / 'src').rglob('*.py')) + list((BASE / 'scripts').glob('*.*'))
    paths += [ROOT / 'conn_experiment_01' / 'src' / 'conn_exp' / (n + '.py')
              for n in ('gates', 'geometry', 'models', 'schedules', 'seeding', 'schema')]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def environment():
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    return {'python': platform.python_version(), 'executable': os.sys.executable,
            'torch': torch.__version__, 'numpy': np.__version__, 'psutil': psutil.__version__,
            'platform': platform.platform(), 'processor': platform.processor(),
            'logical_cpus': psutil.cpu_count(), 'device': 'cpu', 'cuda_available': torch.cuda.is_available(),
            'deterministic_algorithms': True, 'mkldnn': False, 'git_commit': git.stdout.strip() or None,
            'git_note': 'No repository; user authorized skipping Git checkout.',
            'source_hashes': source_manifest()}


def create_run(cfg, parent=None):
    cfg.validate()
    directory = Path(parent or BASE / 'results') / (datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ_') + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'config.json', asdict(cfg))
    write_json(directory / 'environment.json', environment())
    write_json(directory / 'status.json', {'state': 'created', 'stage': cfg.stage, 'config_sha256': digest_json(asdict(cfg))})
    return directory


def save_checkpoint(path, model, optimizer, gates, completed, gate_step, seed, condition, task, cfg, init_hash):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'gates': gates, 'completed_steps': completed, 'gate_step': gate_step,
                'seed': seed, 'condition': condition, 'task': task,
                'config_sha256': digest_json(asdict(cfg)), 'initialization_sha256': init_hash,
                'torch_rng': torch.get_rng_state()}, path)


def fit_condition(cfg, seed, task, condition, data, batch_indices, gates, initial, directory):
    directory.mkdir(exist_ok=False)
    status = {'state': 'running', 'seed': seed, 'task': task, 'condition': condition,
              'completed_steps': 0}
    write_json(directory / 'status.json', status)
    begun = time.perf_counter()
    try:
        init_seed = stream(seed, 'init')
        training_seed = stream(seed, 'training')
        random.seed(training_seed)
        np.random.seed(training_seed)
        torch.manual_seed(training_seed)
        model = CoNNMLP(cfg.n_features, cfg.hidden)
        model.load_state_dict(initial)
        optimizer, hparams = build_optimizer(model, cfg.lr)
        tx, ty = (torch.from_numpy(data['train'][k]) for k in ('x', 'y'))
        vx, vy = (torch.from_numpy(data['val'][k]) for k in ('x', 'y'))
        init_hash = fingerprint_state(model.state_dict())
        provenance = {'seed': seed, 'model_init_stream_seed': init_seed,
                      'actual_initial_state_seed': init_seed + 1000003,
                      'training_seed': training_seed, 'order_seed': stream(seed, 'order'),
                      'mask_stream_seed': stream(seed, 'mask'),
                      'actual_mask_seed': stream(seed, 'mask') + 3000003,
                      'dataset_seeds': {s: int(data[s]['data_seed']) for s in ('train', 'val')},
                      'data_hashes': {s: arrays_hash(data[s]['x'], data[s]['y']) for s in ('train', 'val')},
                      'initialization_sha256': init_hash, 'batch_sha256': arrays_hash(batch_indices),
                      'parameters': trainable_count(model), 'optimizer': hparams}
        write_json(directory / 'provenance.json', provenance)
        write_json(directory / 'initial_metrics.json', {
            s: evaluate(model, torch.from_numpy(data[s]['x']), torch.from_numpy(data[s]['y']), gates.at(condition, 0))[0]
            for s in ('train', 'val')})
        curves = []
        step_seconds = 0.0
        rss = psutil.Process().memory_info().rss
        for t in range(cfg.total_steps):
            stamp = time.perf_counter()
            model.train()
            g = gates.at(condition, t)
            ix = batch_indices[t]
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(tx[ix], *g)
            loss = F.binary_cross_entropy_with_logits(logits, ty[ix])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite loss at {t}')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f'nonfinite gradient at {t}')
            optimizer.step()
            step_seconds += time.perf_counter() - stamp
            status['completed_steps'] = t + 1
            if (t + 1) % cfg.eval_interval == 0 or t + 1 in (cfg.release_step, cfg.total_steps):
                curves.append({'completed_steps': t + 1, 'gate_step': t,
                               'train': evaluate(model, tx, ty, g)[0], 'val': evaluate(model, vx, vy, g)[0]})
                rss = max(rss, psutil.Process().memory_info().rss)
                write_json(directory / 'curves.json', curves)
                write_json(directory / 'status.json', status)
            if t + 1 == cfg.release_step:
                save_checkpoint(directory / 'pre_release.pt', model, optimizer, g, t + 1, t, seed, condition, task, cfg, init_hash)
        g = gates.at(condition, cfg.total_steps - 1)
        save_checkpoint(directory / 'final.pt', model, optimizer, g, cfg.total_steps, cfg.total_steps - 1, seed, condition, task, cfg, init_hash)
        write_json(directory / 'weight_distance.json', weight_distance(model, g, gates.distances))
        status.update(state='trained', wall_seconds=time.perf_counter()-begun,
                      optimization_seconds=step_seconds, peak_sampled_process_rss_bytes=rss,
                      memory_note='Process RSS sampled at logging steps; includes interpreter/data, not isolated model peak.')
        write_json(directory / 'status.json', status)
        return status
    except BaseException as exc:
        status.update(state='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      wall_seconds=time.perf_counter()-begun, error=repr(exc), traceback=traceback.format_exc())
        write_json(directory / 'status.json', status)
        raise


def train_run(cfg, parent=None):
    configure(cfg)
    run = create_run(cfg, parent)
    print(f'RUN {run}', flush=True)
    write_json(run / 'status.json', {'state': 'training', 'config_sha256': digest_json(asdict(cfg))})
    begun = time.perf_counter()
    try:
        for seed in cfg.seeds:
            gates = Gates(cfg, seed)
            seed_dir = run / f'seed_{seed}'
            seed_dir.mkdir()
            write_json(seed_dir / 'gate_audit.json', gates.audit())
            torch.save({'perms': gates.perms, 'distances': gates.distances}, seed_dir / 'geometry.pt')
            initial = initial_state_dict(cfg.n_features, cfg.hidden, stream(seed, 'init'))
            torch.save(initial, seed_dir / 'initial_state.pt')
            # One fixed order, cycling as in original train.batch_indices.
            order = np.random.default_rng(stream(seed, 'order')).permutation(cfg.n_train)
            batches = order[(np.arange(cfg.total_steps)[:, None] * cfg.batch_size + np.arange(cfg.batch_size)) % cfg.n_train]
            np.save(seed_dir / 'batches.npy', batches)
            for task in TASKS:
                task_dir = seed_dir / task
                task_dir.mkdir()
                data = {s: generate_split(cfg, seed, s, task) for s in ('train', 'val')}
                for s, values in data.items():
                    np.savez_compressed(task_dir / f'{s}.npz', **values)
                # Deterministic rotation reduces condition-order timing confounding.
                rotated = CONDITIONS[seed % 4:] + CONDITIONS[:seed % 4]
                for condition in rotated:
                    result = fit_condition(cfg, seed, task, condition, data, batches, gates, initial, task_dir / condition)
                    print(f'{cfg.stage} seed={seed} {task}/{condition}: trained {result["wall_seconds"]:.2f}s', flush=True)
        write_json(run / 'status.json', {'state': 'trained', 'wall_seconds': time.perf_counter()-begun,
                                         'config_sha256': digest_json(asdict(cfg))})
        return run
    except BaseException as exc:
        write_json(run / 'status.json', {'state': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                                         'error': repr(exc), 'traceback': traceback.format_exc(),
                                         'wall_seconds': time.perf_counter()-begun})
        raise


def evaluate_run(run):
    run = Path(run)
    cfg = Config.load(run / 'config.json')
    configure(cfg)
    state = json.loads((run / 'status.json').read_text())
    if state['state'] != 'trained':
        raise ValueError('evaluation requires a fully trained, unevaluated run')
    output = run / 'evaluation'
    output.mkdir(exist_ok=False)
    write_json(output / 'status.json', {'state': 'evaluating'})
    rows = []
    try:
        for seed in cfg.seeds:
            gates = Gates(cfg, seed)
            for task in TASKS:
                data = generate_split(cfg, seed, 'test', task)
                x, y = torch.from_numpy(data['x']), torch.from_numpy(data['y'])
                np.savez_compressed(output / f'{seed}_{task}_test.npz', **data)
                for condition in CONDITIONS:
                    directory = run / f'seed_{seed}' / task / condition
                    provenance = json.loads((directory / 'provenance.json').read_text())
                    predictions = {}
                    for endpoint, checkpoint in (('pre_release', 'pre_release.pt'), ('immediate_release', 'pre_release.pt'), ('final', 'final.pt')):
                        # These are checkpoints generated by this run; never arbitrary external pickle files.
                        ck = torch.load(directory / checkpoint, map_location='cpu', weights_only=True)
                        if ck['config_sha256'] != digest_json(asdict(cfg)):
                            raise ValueError('checkpoint configuration mismatch')
                        model = CoNNMLP(cfg.n_features, cfg.hidden)
                        model.load_state_dict(ck['model'])
                        g = ck['gates'] if endpoint != 'immediate_release' else gates.ones
                        result, logits = evaluate(model, x, y, g)
                        predictions[endpoint] = logits
                        rows.append({'seed': seed, 'task': task, 'condition': condition, 'endpoint': endpoint,
                                     'test_data_seed': int(data['data_seed']), 'test_sha256': arrays_hash(data['x'], data['y']),
                                     'initialization_sha256': provenance['initialization_sha256'],
                                     'completed_steps': ck['completed_steps'], **result})
                    np.savez_compressed(output / f'{seed}_{task}_{condition}_predictions.npz', y=data['y'], **predictions)
        write_json(output / 'metrics.json', rows)
        write_json(output / 'status.json', {'state': 'evaluated', 'rows': len(rows),
                                           'test_endpoint_evaluations_per_fit': 3, 'selection': 'none; predefined checkpoints'})
        write_json(run / 'status.json', {**state, 'state': 'evaluated'})
        print(f'EVALUATED {run}: {len(rows)} endpoint metrics', flush=True)
        return rows
    except BaseException as exc:
        write_json(output / 'status.json', {'state': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                                           'error': repr(exc), 'traceback': traceback.format_exc()})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--config', type=Path, required=True)
    ev = sub.add_parser('evaluate')
    ev.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'train':
        train_run(Config.load(a.config))
    else:
        evaluate_run(a.run)


if __name__ == '__main__':
    main()
