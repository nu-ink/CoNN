"""Specified DGP, controls and metrics. No fitting or filesystem side effects."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import numpy as np
import torch
from torch.nn import functional as F

from conn_exp.gates import gaussian_gate, apply_row_permutation, scalar_gate, draw_masks
from conn_exp.geometry import make_task_geometry
from conn_exp.models import CoNNMLP, initial_state_dict, fingerprint_state, build_optimizer, trainable_count
from conn_exp.schedules import schedule_at, EXACT_FACTOR, release_step_index

CONDITIONS = ('mlp', 'aligned', 'scrambled', 'uniform')
TASKS = ('structured', 'negative')


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def stream(seed, name):
    """Independent stable streams; does not use process-randomized Python hash()."""
    return int.from_bytes(hashlib.sha256(f'exp04:{seed}:{name}'.encode()).digest()[:4], 'little')


def arrays_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class Config:
    stage: str = 'confirmatory'
    seeds: tuple = tuple(range(4000, 4010))
    n_features: int = 100
    groups: int = 5
    hidden: tuple = (32, 32)
    n_train: int = 512
    n_val: int = 256
    n_test: int = 2048
    feature_noise: float = 0.3
    label_flip: float = 0.05
    total_steps: int = 1600
    batch_size: int = 32
    eval_interval: int = 100
    lr: float = 0.1
    d50_start: float = 0.08
    d50_end: float = 1.0
    release_fraction: float = 0.7
    practical_effect: float = 0.02
    overhead_limit: float = 1.5
    threads: int = 1

    def validate(self):
        for k in ('n_features', 'groups', 'n_train', 'n_val', 'n_test', 'total_steps', 'batch_size', 'eval_interval', 'threads'):
            v = getattr(self, k)
            if type(v) is not int or v <= 0:
                raise ValueError(f'{k} must be a positive integer')
        if self.n_features % (2 * self.groups) or len(self.hidden) != 2 or any(type(v) is not int or v <= 0 for v in self.hidden):
            raise ValueError('features must divide into equal paired groups; two positive hidden widths required')
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(type(s) is not int or s < 0 for s in self.seeds):
            raise ValueError('seeds must be unique nonnegative integers')
        for k in ('lr', 'd50_start', 'd50_end', 'practical_effect', 'overhead_limit'):
            if not math.isfinite(getattr(self, k)) or getattr(self, k) <= 0:
                raise ValueError(f'{k} must be finite and positive')
        if not math.isfinite(self.feature_noise) or self.feature_noise < 0 or not 0 <= self.label_flip < 0.5:
            raise ValueError('invalid noise')
        if not 0 < self.release_fraction < 1 or self.d50_start > self.d50_end:
            raise ValueError('invalid schedule')
        if not 6 <= self.release_step < self.total_steps:
            raise ValueError('schedule requires >=6 gated steps and a dense phase')
        if self.stage not in ('smoke', 'pilot', 'confirmatory'):
            raise ValueError('unknown stage')
        return self

    @property
    def release_step(self):
        return release_step_index(self.total_steps, self.release_fraction)

    @classmethod
    def load(cls, path):
        v = json.loads(path.read_text())
        for k in ('seeds', 'hidden'):
            if k in v:
                v[k] = tuple(v[k])
        return cls(**v).validate()


def target_score(z):
    """z[n, group, (a,b)]; no coordinates or gate values enter the target."""
    return (np.abs(z[..., 0] + z[..., 1]) - np.abs(z[..., 0] - z[..., 1])).sum(axis=1)


def generate_split(cfg, seed, split, task):
    if task not in TASKS or split not in ('train', 'val', 'test'):
        raise ValueError('invalid split/task')
    rng = np.random.default_rng(stream(seed, f'data:{split}'))
    n = getattr(cfg, f'n_{split}')
    z = rng.normal(size=(n, cfg.groups, 2))
    repeats = cfg.n_features // (2 * cfg.groups)
    # Each group contains repeated noisy measurements of a, then of b.
    base = np.repeat(z, repeats, axis=2).reshape(n, cfg.n_features)
    eps = rng.normal(size=base.shape)
    x = (base + cfg.feature_noise * eps).astype('float32')
    clean = (target_score(z) > 0).astype('float32')
    flipped = rng.random(n) < cfg.label_flip
    y = np.logical_xor(clean.astype(bool), flipped).astype('float32')
    perm = np.arange(cfg.n_features)
    if task == 'negative':
        # Same permutation for all splits; it is independent of labels and z.
        perm = np.random.default_rng(stream(seed, 'negative:columns')).permutation(cfg.n_features)
        x = x[:, perm].copy()
    return dict(x=x, y=y, z=z, clean=clean, flipped=flipped, permutation=perm,
                data_seed=np.array(stream(seed, f'data:{split}'), dtype=np.uint64))


class Gates:
    def __init__(self, cfg, seed):
        self.cfg = cfg
        geom = make_task_geometry(cfg.n_features, cfg.hidden)
        self.distances = (geom.dist_ih, geom.dist_hh)
        self.masks = draw_masks(cfg.n_features, *cfg.hidden, stream(seed, 'mask'))
        self.perms = (self.masks.row_perm_ih, self.masks.row_perm_hh)
        self.ones = tuple(torch.ones_like(d) for d in self.distances)

    def at(self, condition, step):
        if condition not in CONDITIONS:
            raise ValueError('unknown condition')
        c = self.cfg
        point = schedule_at('D', step, c.total_steps, c.d50_start / EXACT_FACTOR,
                            c.d50_end / EXACT_FACTOR, c.release_fraction)
        if condition == 'mlp' or point.exact_dense:
            return self.ones
        gaussian = tuple(gaussian_gate(d, point.tau_used)[0] for d in self.distances)
        if condition == 'aligned':
            return gaussian
        if condition == 'scrambled':
            return tuple(apply_row_permutation(g, p) for g, p in zip(gaussian, self.perms))
        return tuple(scalar_gate(g) for g in gaussian)

    def audit(self):
        rows = []
        # Check every scheduled step, not only initial energy.
        for t in range(self.cfg.total_steps):
            a, b, u = (self.at(c, t) for c in ('aligned', 'scrambled', 'uniform'))
            for layer, (ga, gb, gu) in enumerate(zip(a, b, u)):
                exact = torch.equal(ga.sort(1).values, gb.sort(1).values)
                # Accumulate diagnostics in float64: float32 reductions can differ
                # by a few ulps under permutations despite identical row multisets.
                row_error = float((ga.double().square().sum(1) - gb.double().square().sum(1)).abs().max())
                uniform_error = float((ga.double().square().mean() - gu.double().square().mean()).abs())
                if not exact or row_error > 1e-12 or uniform_error > 2e-7:
                    raise AssertionError('gate invariant failed')
                rows.append((t, layer, row_error, uniform_error))
        return {'checked_steps': self.cfg.total_steps, 'row_multisets_exact': True,
                'max_row_energy_abs_error': max(r[2] for r in rows),
                'max_uniform_energy_abs_error': max(r[3] for r in rows)}


def metrics(logits, labels):
    pred = logits > 0
    truth = labels.bool()
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    return {'accuracy': float((pred == truth).float().mean()),
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            'loss': float(F.binary_cross_entropy_with_logits(logits, labels)),
            'n': len(labels), 'positive_fraction': float(truth.float().mean())}


def evaluate(model, x, y, gates):
    model.eval()
    with torch.inference_mode():
        logits, _ = model(x, *gates)
        return metrics(logits, y), logits.cpu().numpy()


def weight_distance(model, gates, distances):
    result = []
    for layer, (w, g, d) in enumerate(zip((model.fc1.weight, model.fc2.weight), gates, distances)):
        for lo, hi in zip((0, .2, .4, .6, .8), (.2, .4, .6, .8, 1.00001)):
            mask = (d >= lo) & (d < hi)
            if mask.any():
                result.append({'layer': layer, 'distance_lo': lo, 'distance_hi': hi,
                               'n': int(mask.sum()), 'mean_abs_weight': float(w.detach()[mask].abs().mean()),
                               'mean_abs_effective': float((w.detach() * g)[mask].abs().mean())})
    return result
