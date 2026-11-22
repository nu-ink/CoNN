"""Synthetic motif task. One stream produces train, then validation, then test."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from conn_exp.config import ExperimentConfig
from conn_exp.seeding import paired_seed

CELLS = ("both", "a_only", "b_only", "neither")


@dataclass
class SplitData:
    x: np.ndarray
    y: np.ndarray
    motif_a: np.ndarray
    motif_b: np.ndarray
    motif_a_pos: np.ndarray
    motif_b_pos: np.ndarray
    distractor_pos: np.ndarray
    cell: np.ndarray


@dataclass
class ExperimentData:
    train: SplitData
    val: SplitData
    test: SplitData
    seed: int

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for split in (self.train, self.val, self.test):
            digest.update(np.ascontiguousarray(split.x).tobytes())
            digest.update(np.ascontiguousarray(split.y).tobytes())
        return digest.hexdigest()


def cell_counts(n: int, task: str) -> dict[str, int]:
    if task == "and":
        n_pos = n // 2
        n_neg = n - n_pos
        base, remainder = divmod(n_neg, 3)
        counts = {"both": n_pos, "a_only": base, "b_only": base, "neither": base}
        for name in ("a_only", "b_only", "neither")[:remainder]:
            counts[name] += 1
        return counts
    if task == "xor":
        base, remainder = divmod(n, 4)
        counts = {name: base for name in CELLS}
        for name in CELLS[:remainder]:
            counts[name] += 1
        return counts
    raise ValueError(task)


def label_from_cell(cell: str, task: str) -> float:
    present_a = cell in {"both", "a_only"}
    present_b = cell in {"both", "b_only"}
    if task == "and":
        return 1.0 if present_a and present_b else 0.0
    if task == "xor":
        return 1.0 if present_a != present_b else 0.0
    raise ValueError(task)


def _overlaps(start: int, length: int, spans: list[tuple[int, int]]) -> bool:
    end = start + length
    return any(start < span_end and end > span_start for span_start, span_end in spans)


def _sample_distractor(
    rng: np.random.Generator,
    length: int,
    forbidden: list[np.ndarray],
) -> np.ndarray:
    for _ in range(100):
        pattern = rng.choice(np.array([-1.0, 1.0]), size=length)
        if not any(np.array_equal(pattern, item) for item in forbidden):
            return pattern.astype(np.float64)
    raise RuntimeError("could not sample a distractor distinct from the motifs")


def _place_start(rng: np.random.Generator, n_features: int, length: int, region: tuple[int, int]) -> int:
    low, high = region
    return int(rng.integers(low, high - length + 1))


def _generate_split(config: ExperimentConfig, n: int, rng: np.random.Generator) -> SplitData:
    counts = cell_counts(n, config.task)
    cells = np.array(sum(([name] * count for name, count in counts.items()), []), dtype=object)
    rng.shuffle(cells)
    length = config.motif_length
    template_a = np.asarray(config.template_a, dtype=np.float64)
    template_b = np.asarray(config.template_b, dtype=np.float64)
    xs = np.zeros((n, config.n_features), dtype=np.float64)
    y = np.zeros(n, dtype=np.float32)
    motif_a = np.zeros(n, dtype=np.int8)
    motif_b = np.zeros(n, dtype=np.int8)
    pos_a = np.full(n, -1, dtype=np.int16)
    pos_b = np.full(n, -1, dtype=np.int16)
    distractors = np.full((n, max(config.n_distractors, 1)), -1, dtype=np.int16)
    if config.n_distractors == 0:
        distractors = np.zeros((n, 0), dtype=np.int16)
    for index, cell in enumerate(cells.tolist()):
        spans: list[tuple[int, int]] = []
        if cell in {"both", "a_only"}:
            start = _place_start(rng, config.n_features, length, config.region_a)
            xs[index, start : start + length] += config.motif_amplitude * template_a
            motif_a[index] = 1
            pos_a[index] = start
            spans.append((start, start + length))
        if cell in {"both", "b_only"}:
            start = _place_start(rng, config.n_features, length, config.region_b)
            xs[index, start : start + length] += config.motif_amplitude * template_b
            motif_b[index] = 1
            pos_b[index] = start
            spans.append((start, start + length))
        for distractor_index in range(config.n_distractors):
            pattern = _sample_distractor(rng, length, [template_a, template_b])
            valid = [
                start
                for start in range(0, config.n_features - length + 1)
                if not _overlaps(start, length, spans)
            ]
            if not valid:
                break
            start = int(rng.choice(np.asarray(valid, dtype=np.int64)))
            xs[index, start : start + length] += config.distractor_amplitude * pattern
            distractors[index, distractor_index] = start
            spans.append((start, start + length))
        if config.noise_std > 0:
            xs[index] += rng.normal(0.0, config.noise_std, size=config.n_features)
        y[index] = label_from_cell(cell, config.task)
    return SplitData(
        x=xs.astype(np.float32),
        y=y,
        motif_a=motif_a,
        motif_b=motif_b,
        motif_a_pos=pos_a,
        motif_b_pos=pos_b,
        distractor_pos=distractors,
        cell=cells,
    )


def generate_dataset(config: ExperimentConfig, seed: int) -> ExperimentData:
    rng = np.random.default_rng(paired_seed(seed, "data"))
    return ExperimentData(
        train=_generate_split(config, config.n_train, rng),
        val=_generate_split(config, config.n_val, rng),
        test=_generate_split(config, config.n_test, rng),
        seed=int(seed),
    )


def best_single_feature_accuracy(x: np.ndarray, y: np.ndarray) -> float:
    labels = np.asarray(y, dtype=np.float64)
    features = np.asarray(x, dtype=np.float64)
    n = labels.shape[0]
    best = max(float(labels.mean()), float(1.0 - labels.mean()))
    for column in range(features.shape[1]):
        order = np.argsort(features[:, column], kind="mergesort")
        ordered = labels[order]
        cumulative = np.cumsum(ordered)
        total_positive = cumulative[-1]
        prefix = np.arange(1, n, dtype=np.float64)
        left_positive = cumulative[:-1]
        left_negative = prefix - left_positive
        right_positive = total_positive - left_positive
        right_negative = (n - prefix) - right_positive
        best = max(
            best,
            float(((left_negative + right_positive) / n).max()),
            float(((left_positive + right_negative) / n).max()),
        )
    return best


def binary_predictor_accuracy(predictor: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.asarray(predictor).astype(np.float64) == np.asarray(y).astype(np.float64)))


def _row_set(x: np.ndarray) -> set[bytes]:
    return {np.ascontiguousarray(row).tobytes() for row in x}


def assert_dataset_invariants(data: ExperimentData, config: ExperimentConfig) -> None:
    parts = {"train": data.train, "val": data.val, "test": data.test}
    expected = {"train": config.n_train, "val": config.n_val, "test": config.n_test}
    for name, split in parts.items():
        if split.x.shape != (expected[name], config.n_features):
            raise AssertionError(f"{name} feature shape is wrong")
        if split.y.shape[0] != expected[name]:
            raise AssertionError(f"{name} label length is wrong")
        if not np.isfinite(split.x).all() or not np.isin(split.y, [0.0, 1.0]).all():
            raise AssertionError(f"{name} has non-finite features or unexpected labels")
        if set(split.cell.tolist()) != set(CELLS):
            raise AssertionError(f"{name} is missing a motif cell")
        reconstructed = np.array([label_from_cell(cell, config.task) for cell in split.cell.tolist()])
        if not np.array_equal(reconstructed, split.y):
            raise AssertionError(f"{name} labels do not match the motif cells")
        present = (split.cell == "both") | (split.cell == "a_only")
        if not np.array_equal(present, split.motif_a.astype(bool)):
            raise AssertionError("motif A metadata does not match the cell")
        present_b = (split.cell == "both") | (split.cell == "b_only")
        if not np.array_equal(present_b, split.motif_b.astype(bool)):
            raise AssertionError("motif B metadata does not match the cell")
        for position in split.motif_a_pos[split.motif_a == 1]:
            if not (config.region_a[0] <= int(position) <= config.region_a[1] - config.motif_length):
                raise AssertionError("motif A jitter left its region")
        for position in split.motif_b_pos[split.motif_b == 1]:
            if not (config.region_b[0] <= int(position) <= config.region_b[1] - config.motif_length):
                raise AssertionError("motif B jitter left its region")
        if abs(float(split.y.mean()) - 0.5) > 1e-8:
            raise AssertionError(f"{name} labels are not balanced")
        if config.task == "and":
            if float(split.y[split.motif_a == 1].mean()) in {0.0, 1.0}:
                raise AssertionError("motif A presence trivially equals the AND label")
            if float(split.y[split.motif_b == 1].mean()) in {0.0, 1.0}:
                raise AssertionError("motif B presence trivially equals the AND label")
    if config.noise_std > 0:
        train_rows = _row_set(data.train.x)
        val_rows = _row_set(data.val.x)
        test_rows = _row_set(data.test.x)
        if train_rows & val_rows or train_rows & test_rows or val_rows & test_rows:
            raise AssertionError("exact feature rows are shared across splits")
