"""Typed pilot configuration. Tau is derived from d50 targets."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from conn_exp.schedules import EXACT_FACTOR, release_step_index
from conn_exp.schema import (
    PRIMARY_CONTROL,
    PRIMARY_CONTROL_NAME,
    PRIMARY_METRIC,
    PRIMARY_TASK,
    PRIMARY_TREATMENT,
)

YAML_KEYS = {
    "experiment_name",
    "task",
    "seed_list",
    "n_features",
    "motif_length",
    "motif_amplitude",
    "region_a",
    "region_b",
    "template_a",
    "template_b",
    "noise_std",
    "n_distractors",
    "distractor_amplitude",
    "n_train",
    "n_val",
    "n_test",
    "batch_size",
    "hidden_sizes",
    "total_steps",
    "eval_interval",
    "lr",
    "momentum",
    "weight_decay",
    "optimizer",
    "d50_start",
    "d50_end",
    "release_fraction",
    "tau_floor",
    "far_quantile",
    "distance_bin_edges",
    "probe_l2",
    "device",
    "primary_treatment",
    "primary_control",
    "primary_control_name",
    "primary_metric",
    "primary_task",
    "lr_grid",
    "lr_selection",
}

SMOKE_KEYS = {
    "seed_list",
    "n_train",
    "n_val",
    "n_test",
    "total_steps",
    "eval_interval",
    "hidden_sizes",
    "batch_size",
}


@dataclass
class ExperimentConfig:
    experiment_name: str
    task: str
    seed_list: tuple[int, ...]
    n_features: int
    motif_length: int
    motif_amplitude: float
    region_a: tuple[int, int]
    region_b: tuple[int, int]
    template_a: tuple[float, ...]
    template_b: tuple[float, ...]
    noise_std: float
    n_distractors: int
    distractor_amplitude: float
    n_train: int
    n_val: int
    n_test: int
    batch_size: int
    hidden_sizes: tuple[int, int]
    total_steps: int
    eval_interval: int
    lr: float
    momentum: float
    weight_decay: float
    optimizer: str
    d50_start: float
    d50_end: float
    release_fraction: float
    tau_floor: float
    far_quantile: float
    distance_bin_edges: tuple[float, ...]
    probe_l2: float
    device: str
    primary_treatment: str
    primary_control: str
    primary_control_name: str
    primary_metric: str
    primary_task: str
    lr_grid: tuple[float, ...]
    lr_selection: str
    tau_start: float
    tau_end: float
    release_step: int

    def override(self, **kwargs: object) -> ExperimentConfig:
        raw = asdict(self)
        raw.pop("tau_start", None)
        raw.pop("tau_end", None)
        raw.pop("release_step", None)
        raw.update(kwargs)
        return _build(raw)

    def to_resolved_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, tuple):
                payload[key] = list(value)
        payload["permutation_rule"] = "within_row_column_permutation_frozen"
        payload["distance"] = "absolute_difference_on_unit_interval"
        payload["optimizer_foreach"] = False
        payload["optimizer_fused"] = False
        payload["test_protocol"] = "final_step_only"
        payload["lr_grid_executed"] = False
        return payload


def config_sha256(config: ExperimentConfig) -> str:
    body = json.dumps(config.to_resolved_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _as_int_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return tuple(int(item) for item in value)


def _as_float_tuple(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return tuple(float(item) for item in value)


def _pair_int(value: object, name: str) -> tuple[int, int]:
    items = _as_int_tuple(value, name)
    if len(items) != 2:
        raise ValueError(f"{name} must have length 2")
    return items[0], items[1]


def _build(raw: dict[str, object]) -> ExperimentConfig:
    if "tau_start" in raw or "tau_end" in raw:
        raise ValueError("tau_start and tau_end are derived from d50_start and d50_end")
    hidden = _pair_int(raw["hidden_sizes"], "hidden_sizes")
    edges = _as_float_tuple(raw["distance_bin_edges"], "distance_bin_edges")
    tau_start = float(raw["d50_start"]) / EXACT_FACTOR
    tau_end = float(raw["d50_end"]) / EXACT_FACTOR
    release_step = release_step_index(int(raw["total_steps"]), float(raw["release_fraction"]))
    config = ExperimentConfig(
        experiment_name=str(raw["experiment_name"]),
        task=str(raw["task"]),
        seed_list=_as_int_tuple(raw["seed_list"], "seed_list"),
        n_features=int(raw["n_features"]),
        motif_length=int(raw["motif_length"]),
        motif_amplitude=float(raw["motif_amplitude"]),
        region_a=_pair_int(raw["region_a"], "region_a"),
        region_b=_pair_int(raw["region_b"], "region_b"),
        template_a=_as_float_tuple(raw["template_a"], "template_a"),
        template_b=_as_float_tuple(raw["template_b"], "template_b"),
        noise_std=float(raw["noise_std"]),
        n_distractors=int(raw["n_distractors"]),
        distractor_amplitude=float(raw["distractor_amplitude"]),
        n_train=int(raw["n_train"]),
        n_val=int(raw["n_val"]),
        n_test=int(raw["n_test"]),
        batch_size=int(raw["batch_size"]),
        hidden_sizes=hidden,
        total_steps=int(raw["total_steps"]),
        eval_interval=int(raw["eval_interval"]),
        lr=float(raw["lr"]),
        momentum=float(raw["momentum"]),
        weight_decay=float(raw["weight_decay"]),
        optimizer=str(raw["optimizer"]),
        d50_start=float(raw["d50_start"]),
        d50_end=float(raw["d50_end"]),
        release_fraction=float(raw["release_fraction"]),
        tau_floor=float(raw["tau_floor"]),
        far_quantile=float(raw["far_quantile"]),
        distance_bin_edges=edges,
        probe_l2=float(raw["probe_l2"]),
        device=str(raw["device"]),
        primary_treatment=str(raw["primary_treatment"]),
        primary_control=str(raw["primary_control"]),
        primary_control_name=str(raw["primary_control_name"]),
        primary_metric=str(raw["primary_metric"]),
        primary_task=str(raw["primary_task"]),
        lr_grid=_as_float_tuple(raw["lr_grid"], "lr_grid"),
        lr_selection=str(raw["lr_selection"]),
        tau_start=tau_start,
        tau_end=tau_end,
        release_step=release_step,
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    if config.optimizer != "sgd" or config.momentum != 0.0 or config.weight_decay != 0.0:
        raise ValueError(
            "Optimizer is plain SGD with momentum=0 and weight_decay=0. "
            "Refusing to change it in this runner. Record a separate experiment for any other optimizer."
        )
    if (
        config.primary_treatment != PRIMARY_TREATMENT
        or config.primary_control != PRIMARY_CONTROL
        or config.primary_control_name != PRIMARY_CONTROL_NAME
        or config.primary_metric != PRIMARY_METRIC
        or config.primary_task != PRIMARY_TASK
    ):
        raise ValueError(
            "Primary contrast is frozen: final test accuracy of D minus B_prime (row_shuffle) on AND."
        )
    if config.lr_selection != "shared_initial":
        raise ValueError(
            "This runner executes the shared initial learning rate only. "
            "A validation grid must be a separate recorded experiment."
        )
    if config.task not in {"and", "xor"}:
        raise ValueError("task must be 'and' or 'xor'")
    if config.n_features != 100:
        raise ValueError("n_features is frozen at 100 for this experiment")
    if config.release_fraction != 0.7:
        raise ValueError("release_fraction is frozen at 0.7")
    if config.device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if config.lr <= 0 or config.eval_interval <= 0 or config.batch_size <= 0:
        raise ValueError("lr, eval_interval, and batch_size must be positive")
    if config.batch_size > config.n_train:
        raise ValueError("batch_size cannot exceed n_train")
    if min(config.hidden_sizes) < 2:
        raise ValueError("each hidden layer needs at least 2 units")
    if len(config.template_a) != config.motif_length or len(config.template_b) != config.motif_length:
        raise ValueError("template length must equal motif_length")
    if config.template_a == config.template_b:
        raise ValueError("motif templates must differ")
    for name, region in (("region_a", config.region_a), ("region_b", config.region_b)):
        start, end = region
        if not (0 <= start < end <= config.n_features):
            raise ValueError(f"{name} must lie inside the feature axis")
        if end - start < config.motif_length:
            raise ValueError(f"{name} is shorter than the motif")
    if config.region_b[0] - config.region_a[1] < 20:
        raise ValueError("motif regions must be separated by at least 20 features")
    if config.noise_std < 0 or config.n_distractors < 0:
        raise ValueError("noise and distractor count cannot be negative")
    if not 0 < config.far_quantile < 1:
        raise ValueError("far_quantile must lie in (0, 1)")
    edges = config.distance_bin_edges
    if list(edges) != sorted(edges) or edges[0] != 0.0 or edges[-1] < 1.0:
        raise ValueError("distance bins must start at 0, increase, and reach 1")
    if config.release_step < 6 or config.release_step >= config.total_steps:
        raise ValueError("total_steps does not leave both a curriculum window and a final dense phase")
    divisor = 2 if config.task == "and" else 4
    minimum = 6 if config.task == "and" else 4
    for count in (config.n_train, config.n_val, config.n_test):
        if count < minimum or count % divisor != 0:
            raise ValueError(f"split size {count} is not compatible with a balanced {config.task} task")
    if not config.seed_list:
        raise ValueError("seed_list is empty")


def load_config(path: str | Path, smoke: bool = False) -> ExperimentConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config root must be a mapping")
    smoke_payload = payload.pop("smoke", None)
    unknown = set(payload) - YAML_KEYS
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    missing = YAML_KEYS - set(payload)
    if missing:
        raise ValueError(f"Missing config keys: {sorted(missing)}")
    if smoke:
        if not isinstance(smoke_payload, dict):
            raise ValueError("smoke overrides are missing")
        extra = set(smoke_payload) - SMOKE_KEYS
        if extra:
            raise ValueError(f"Smoke cannot override: {sorted(extra)}")
        payload.update(smoke_payload)
    return _build(payload)


def field_names() -> list[str]:
    return [item.name for item in fields(ExperimentConfig)]
