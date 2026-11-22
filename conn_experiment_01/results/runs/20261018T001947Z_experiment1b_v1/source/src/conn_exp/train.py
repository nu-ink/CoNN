"""Paired SGD loop. The test split is read only after the last update."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from conn_exp.config import ExperimentConfig
from conn_exp.diagnostics import bin_mean_abs, far_suppression, fit_and_score_probe, prestep_gradients
from conn_exp.gates import GateBuilder
from conn_exp.models import CoNNMLP
from conn_exp.schedules import SchedulePoint, e_phase_bounds, schedule_at
from conn_exp.schema import (
    ABLATION_COLUMNS,
    ACTIVATION_BATCH,
    DELTA_DEFINITION,
    DISTANCE_DEFINITION,
    GRADIENT_COLUMNS,
    GRADIENT_DEFINITION,
    METRIC_COLUMNS,
    PROBE_COLUMNS,
    QUANTILE_METHOD,
    ROW_ENERGY_COLUMNS,
    condition_by_id,
    make_row,
)
from conn_exp.seeding import paired_seed


def minibatch_order(n_train: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(paired_seed(seed, "order"))
    return rng.permutation(n_train)


def batch_indices(order: np.ndarray, batch_size: int, step: int) -> np.ndarray:
    n = len(order)
    start = (step * batch_size) % n
    return np.asarray([order[(start + offset) % n] for offset in range(batch_size)], dtype=np.int64)


def diagnostic_step_indices(total_steps: int, eval_interval: int, release_step: int) -> set[int]:
    steps = set(range(0, total_steps, eval_interval))
    p1, p2 = e_phase_bounds(release_step)
    steps.update({0, total_steps - 1, release_step - 1, release_step, p1 - 1, p1, p2 - 1, p2})
    return {step for step in steps if 0 <= step < total_steps}


def evaluate_split(
    model: CoNNMLP,
    features: torch.Tensor,
    labels: torch.Tensor,
    gate_ih: torch.Tensor,
    gate_hh: torch.Tensor,
    batch_size: int = 256,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch_x = features[start : start + batch_size]
            batch_y = labels[start : start + batch_size]
            logits, _ = model(batch_x, gate_ih, gate_hh)
            total_loss += float(F.binary_cross_entropy_with_logits(logits, batch_y, reduction="sum"))
            correct += float(((logits > 0).to(dtype=batch_y.dtype) == batch_y).sum())
            count += int(batch_y.shape[0])
    return total_loss / count, correct / count


def _hidden_matrix(
    model: CoNNMLP,
    features: torch.Tensor,
    gate_ih: torch.Tensor,
    gate_hh: torch.Tensor,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    hidden_1: list[torch.Tensor] = []
    hidden_2: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            _, cache = model(features[start : start + batch_size], gate_ih, gate_hh)
            hidden_1.append(cache.h1.detach().cpu())
            hidden_2.append(cache.h2.detach().cpu())
    return torch.cat(hidden_1).numpy(), torch.cat(hidden_2).numpy()


def _moved_gates(pack_ih: torch.Tensor, pack_hh: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return pack_ih.to(device), pack_hh.to(device)


def _norms(weight: torch.Tensor, gate: torch.Tensor) -> tuple[float, float, float, float]:
    applied = gate.detach().to(dtype=weight.dtype, device=weight.device)
    effective = weight.detach() * applied
    return (
        float(weight.detach().abs().mean()),
        float(effective.abs().mean()),
        float(torch.linalg.vector_norm(weight.detach())),
        float(torch.linalg.vector_norm(effective)),
    )


class HeldOutSplit:
    """Counts every read of the test tensors."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels
        self.evaluations = 0

    def evaluate(
        self,
        model: CoNNMLP,
        gate_ih: torch.Tensor,
        gate_hh: torch.Tensor,
        batch_size: int,
    ) -> tuple[float, float]:
        self.evaluations += 1
        return evaluate_split(model, self.features, self.labels, gate_ih, gate_hh, batch_size)


@dataclass
class TrainResult:
    metrics: list[dict[str, object]] = field(default_factory=list)
    gradients: list[dict[str, object]] = field(default_factory=list)
    ablation: list[dict[str, object]] = field(default_factory=list)
    probes: list[dict[str, object]] = field(default_factory=list)
    row_energy: list[dict[str, object]] = field(default_factory=list)
    non_finite: bool = False
    test_evaluations: int = 0
    optimizer_hparams: dict[str, object] = field(default_factory=dict)


def _save_checkpoint(
    path: Path,
    model: CoNNMLP,
    optimizer: torch.optim.Optimizer,
    *,
    gate_ih: torch.Tensor,
    gate_hh: torch.Tensor,
    point: SchedulePoint,
    measure_timing: str,
    completed: int,
    seed: int,
    condition_id: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "gate_ih": gate_ih.detach().cpu().clone(),
            "gate_hh": gate_hh.detach().cpu().clone(),
            "step_index": point.step_index,
            "measure_timing": measure_timing,
            "optimizer_steps_completed": completed,
            "exact_dense": point.exact_dense,
            "phase": point.phase,
            "tau_requested": point.tau_requested,
            "boundary": point.boundary,
            "condition_id": condition_id,
            "seed": seed,
            "permutation_rule": "within_row_column_permutation_frozen",
        },
        path,
    )


def train_condition(
    *,
    model: CoNNMLP,
    optimizer: torch.optim.Optimizer,
    optimizer_hparams: dict[str, object],
    config: ExperimentConfig,
    gate_builder: GateBuilder,
    order: np.ndarray,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    motif_a_train: np.ndarray,
    motif_b_train: np.ndarray,
    motif_a_val: np.ndarray,
    motif_b_val: np.ndarray,
    test_access: HeldOutSplit,
    base_dist_ih: torch.Tensor,
    base_dist_hh: torch.Tensor,
    seed: int,
    condition_id: str,
    run_id: str,
    data_fingerprint: str,
    init_fingerprint: str,
    artifact_dir: Path | None = None,
    start_step: int = 0,
    stop_step: int | None = None,
    diagnostics: bool = True,
    evaluate_test: bool = True,
) -> TrainResult:
    if optimizer_hparams.get("name") != "sgd":
        raise RuntimeError("Refusing an optimizer other than plain SGD.")
    if float(optimizer_hparams["momentum"]) != 0.0 or float(optimizer_hparams["weight_decay"]) != 0.0:
        raise RuntimeError("Refusing nonzero momentum or weight decay.")
    device = x_train.device
    base_dist_ih = base_dist_ih.to(device)
    base_dist_hh = base_dist_hh.to(device)
    total_steps = config.total_steps
    end_step = total_steps if stop_step is None else int(stop_step)
    condition = condition_by_id(condition_id)
    diag_steps = diagnostic_step_indices(total_steps, config.eval_interval, config.release_step)
    result = TrainResult(optimizer_hparams=dict(optimizer_hparams))
    index_diag = torch.as_tensor(batch_indices(order, config.batch_size, 0), dtype=torch.long, device=device)
    x_diag = x_train.index_select(0, index_diag)
    y_diag = y_train.index_select(0, index_diag)

    def point_at(step_index: int) -> SchedulePoint:
        return schedule_at(
            condition_id,
            step_index,
            total_steps,
            config.tau_start,
            config.tau_end,
            config.release_fraction,
            config.tau_floor,
        )

    def log_measurement(
        timing: str,
        step_index: int,
        completed: int,
        point: SchedulePoint,
        gate_ih: torch.Tensor,
        gate_hh: torch.Tensor,
        pack_stats_ih: object,
        pack_stats_hh: object,
        tau_floored: bool,
        batch_loss: float | None,
        include_test: bool,
    ) -> None:
        ih_stats = pack_stats_ih
        hh_stats = pack_stats_hh
        mean_w_ih, mean_we_ih, norm_w_ih, norm_we_ih = _norms(model.fc1.weight, gate_ih)
        mean_w_hh, mean_we_hh, norm_w_hh, norm_we_hh = _norms(model.fc2.weight, gate_hh)
        model.eval()
        with torch.no_grad():
            _, cache = model(x_diag, gate_ih, gate_hh)
            act_var_h1 = float(cache.h1.var(correction=0))
            act_var_h2 = float(cache.h2.var(correction=0))
        train_loss, train_acc = evaluate_split(model, x_train, y_train, gate_ih, gate_hh)
        val_loss, val_acc = evaluate_split(model, x_val, y_val, gate_ih, gate_hh)
        test_loss = None
        test_acc = None
        if include_test:
            test_loss, test_acc = test_access.evaluate(model, gate_ih, gate_hh, 256)
        shared = {
            "run_id": run_id,
            "seed": seed,
            "condition_id": condition.id,
            "condition_name": condition.name,
            "task": config.task,
            "step_index": step_index,
            "measure_timing": timing,
            "optimizer_steps_completed": completed,
            "phase": point.phase,
            "exact_dense": point.exact_dense,
            "tau_requested": point.tau_requested,
            "tau_used": point.tau_used,
            "tau_floored": tau_floored,
            "d50_exact": point.d50_exact,
            "d50_brief": point.d50_brief,
            "boundary": point.boundary,
            "C_ih": ih_stats.energy,
            "C_hh": hh_stats.energy,
            "row_energy_mean_ih": ih_stats.row_energy_mean,
            "row_energy_std_ih": ih_stats.row_energy_std,
            "row_energy_mean_hh": hh_stats.row_energy_mean,
            "row_energy_std_hh": hh_stats.row_energy_std,
            "mean_abs_W_ih": mean_w_ih,
            "mean_abs_W_eff_ih": mean_we_ih,
            "norm_W_ih": norm_w_ih,
            "norm_W_eff_ih": norm_we_ih,
            "mean_abs_W_hh": mean_w_hh,
            "mean_abs_W_eff_hh": mean_we_hh,
            "norm_W_hh": norm_w_hh,
            "norm_W_eff_hh": norm_we_hh,
            "frac_gate_lt_1e-6_ih": ih_stats.frac_lt_1e_6,
            "frac_gate_eq_0_ih": ih_stats.frac_eq_0,
            "frac_gate_lt_1e-6_hh": hh_stats.frac_lt_1e_6,
            "frac_gate_eq_0_hh": hh_stats.frac_eq_0,
            "act_var_h1": act_var_h1,
            "act_var_h2": act_var_h2,
            "activation_batch": ACTIVATION_BATCH,
            "lr": config.lr,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "optimizer": "sgd",
            "distance_definition": DISTANCE_DEFINITION,
            "non_finite": False,
            "data_fingerprint": data_fingerprint,
            "init_fingerprint": init_fingerprint,
        }
        for split, loss, accuracy, batch in (
            ("train", train_loss, train_acc, batch_loss if timing == "post_step" else None),
            ("val", val_loss, val_acc, None),
        ):
            result.metrics.append(make_row(METRIC_COLUMNS, **shared, split=split, loss=loss, accuracy=accuracy, batch_loss=batch))
        if include_test:
            result.metrics.append(
                make_row(METRIC_COLUMNS, **shared, split="test", loss=test_loss, accuracy=test_acc, batch_loss=None)
            )
        for layer, stats in (("ih", ih_stats), ("hh", hh_stats)):
            for row_index, energy in enumerate(stats.row_energy.tolist()):
                result.row_energy.append(
                    make_row(
                        ROW_ENERGY_COLUMNS,
                        run_id=run_id,
                        seed=seed,
                        condition_id=condition.id,
                        step_index=step_index,
                        measure_timing=timing,
                        optimizer_steps_completed=completed,
                        layer=layer,
                        row=row_index,
                        energy=float(energy),
                    )
                )
        if not diagnostics:
            return
        hidden_train = _hidden_matrix(model, x_train, gate_ih, gate_hh, 256)
        hidden_val = _hidden_matrix(model, x_val, gate_ih, gate_hh, 256)
        for representation, train_h, val_h in (
            ("h1", hidden_train[0], hidden_val[0]),
            ("h2", hidden_train[1], hidden_val[1]),
        ):
            for target, y_tr, y_va in (
                ("motif_a", motif_a_train, motif_a_val),
                ("motif_b", motif_b_train, motif_b_val),
            ):
                train_acc_probe, val_acc_probe, status = fit_and_score_probe(
                    train_h, y_tr, val_h, y_va, config.probe_l2
                )
                result.probes.append(
                    make_row(
                        PROBE_COLUMNS,
                        run_id=run_id,
                        seed=seed,
                        condition_id=condition.id,
                        step_index=step_index,
                        measure_timing=timing,
                        optimizer_steps_completed=completed,
                        representation=representation,
                        target=target,
                        train_accuracy=train_acc_probe,
                        val_accuracy=val_acc_probe,
                        n_train=int(len(y_tr)),
                        n_val=int(len(y_va)),
                        fit_split="train",
                        eval_split="val",
                        probe_status=status,
                    )
                )
        if timing != "pre_step":
            return
        grads = prestep_gradients(model, x_diag, y_diag, gate_ih, gate_hh)
        for layer, distance in (("ih", base_dist_ih), ("hh", base_dist_hh)):
            binned_w = bin_mean_abs(grads[layer]["dL_dW"], distance, config.distance_bin_edges)
            binned_we = bin_mean_abs(grads[layer]["dL_dW_eff"], distance, config.distance_bin_edges)
            for left, right in zip(binned_w, binned_we):
                result.gradients.append(
                    make_row(
                        GRADIENT_COLUMNS,
                        run_id=run_id,
                        seed=seed,
                        condition_id=condition.id,
                        condition_name=condition.name,
                        step_index=step_index,
                        measure_timing="pre_step",
                        optimizer_steps_completed=completed,
                        layer=layer,
                        bin_id=int(left["bin_id"]),
                        bin_lo=left["bin_lo"],
                        bin_hi=left["bin_hi"],
                        distance_definition=DISTANCE_DEFINITION,
                        gradient_definition=GRADIENT_DEFINITION,
                        mean_abs_dL_dW=left["mean_abs"],
                        mean_abs_dL_dW_eff=right["mean_abs"],
                        n_weights=int(left["n_weights"]),
                        gradient_batch=ACTIVATION_BATCH,
                    )
                )
        suppressed = far_suppression(gate_ih, gate_hh, base_dist_ih, base_dist_hh, config.far_quantile)
        suppressed_loss, _ = evaluate_split(
            model, x_val, y_val, suppressed["gate_ih"], suppressed["gate_hh"]
        )
        result.ablation.append(
            make_row(
                ABLATION_COLUMNS,
                run_id=run_id,
                seed=seed,
                condition_id=condition.id,
                step_index=step_index,
                measure_timing="pre_step",
                optimizer_steps_completed=completed,
                far_quantile=config.far_quantile,
                quantile_method=QUANTILE_METHOD,
                far_threshold_ih=suppressed["threshold_ih"],
                far_threshold_hh=suppressed["threshold_hh"],
                val_loss=val_loss,
                val_loss_far_suppressed=suppressed_loss,
                val_loss_delta=suppressed_loss - val_loss,
                delta_definition=DELTA_DEFINITION,
                distance_definition=DISTANCE_DEFINITION,
                n_suppressed_ih=suppressed["n_suppressed_ih"],
                n_suppressed_hh=suppressed["n_suppressed_hh"],
                fraction_suppressed_ih=suppressed["fraction_suppressed_ih"],
                fraction_suppressed_hh=suppressed["fraction_suppressed_hh"],
            )
        )

    def measure(timing: str, step_index: int, completed: int, batch_loss: float | None, include_test: bool) -> None:
        point = point_at(step_index)
        pack = gate_builder.build(point.tau_requested, point.exact_dense)
        gate_ih, gate_hh = _moved_gates(pack.ih, pack.hh, device)
        log_measurement(
            timing,
            step_index,
            completed,
            point,
            gate_ih,
            gate_hh,
            pack.stats_ih,
            pack.stats_hh,
            pack.tau_floored,
            batch_loss,
            include_test,
        )
        return point, pack

    def maybe_save(name: str, timing: str, completed: int, point: SchedulePoint, pack: object) -> None:
        if artifact_dir is None:
            return
        path = artifact_dir / "checkpoints" / f"seed_{seed}" / condition_id / name
        _save_checkpoint(
            path,
            model,
            optimizer,
            gate_ih=pack.ih,
            gate_hh=pack.hh,
            point=point,
            measure_timing=timing,
            completed=completed,
            seed=seed,
            condition_id=condition_id,
        )

    if start_step == 0 and diagnostics:
        point, pack = measure("pre_step", 0, 0, None, False)

    for step in range(start_step, end_step):
        point = point_at(step)
        pack = gate_builder.build(point.tau_requested, point.exact_dense)
        gate_ih, gate_hh = _moved_gates(pack.ih, pack.hh, device)
        if diagnostics and step != 0 and step in diag_steps:
            log_measurement(
                "pre_step",
                step,
                step,
                point,
                gate_ih,
                gate_hh,
                pack.stats_ih,
                pack.stats_hh,
                pack.tau_floored,
                None,
                False,
            )
            if step == config.release_step:
                maybe_save("gate_post_release.pt", "pre_step", step, point, pack)
        optimizer.zero_grad(set_to_none=True)
        index = torch.as_tensor(batch_indices(order, config.batch_size, step), dtype=torch.long, device=device)
        model.train()
        logits, _ = model(x_train.index_select(0, index), gate_ih, gate_hh)
        labels = y_train.index_select(0, index)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        batch_loss = float(loss.detach())
        if not np.isfinite(batch_loss):
            result.non_finite = True
            result.metrics.append(
                make_row(
                    METRIC_COLUMNS,
                    run_id=run_id,
                    seed=seed,
                    condition_id=condition.id,
                    condition_name=condition.name,
                    task=config.task,
                    step_index=step,
                    measure_timing="pre_step",
                    optimizer_steps_completed=step,
                    split="train",
                    loss=batch_loss,
                    batch_loss=batch_loss,
                    non_finite=True,
                    optimizer="sgd",
                    momentum=0.0,
                    weight_decay=0.0,
                    lr=config.lr,
                    data_fingerprint=data_fingerprint,
                    init_fingerprint=init_fingerprint,
                    distance_definition=DISTANCE_DEFINITION,
                )
            )
            break
        loss.backward()
        optimizer.step()
        completed = step + 1
        if step in diag_steps:
            log_measurement(
                "post_step",
                step,
                completed,
                point,
                gate_ih,
                gate_hh,
                pack.stats_ih,
                pack.stats_hh,
                pack.tau_floored,
                batch_loss,
                include_test=evaluate_test and completed == total_steps,
            )
            if step == config.release_step - 1:
                maybe_save("gate_pre_release.pt", "post_step", completed, point, pack)
            if completed == total_steps:
                maybe_save("final.pt", "post_step", completed, point, pack)

    result.test_evaluations = test_access.evaluations
    return result
