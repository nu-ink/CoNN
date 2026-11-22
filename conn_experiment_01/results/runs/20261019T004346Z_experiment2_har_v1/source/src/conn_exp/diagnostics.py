"""Pre-step gradients, base-geometry bins, far ablation, and train-only probes."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from conn_exp.models import CoNNMLP


def prestep_gradients(
    model: CoNNMLP,
    features: torch.Tensor,
    labels: torch.Tensor,
    gate_ih: torch.Tensor,
    gate_hh: torch.Tensor,
) -> dict[str, dict[str, torch.Tensor]]:
    """Return dL/dW and dL/dW_eff without writing optimizer state.

    dL/dW is gate-attenuated. dL/dW_eff is the data gradient on the effective weight.
    """
    model.zero_grad(set_to_none=True)
    logits, cache = model(features, gate_ih, gate_hh)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    g_w_ih, g_we_ih, g_w_hh, g_we_hh = torch.autograd.grad(
        loss,
        [model.fc1.weight, cache.w_eff_ih, model.fc2.weight, cache.w_eff_hh],
    )
    model.zero_grad(set_to_none=True)
    return {
        "ih": {"dL_dW": g_w_ih.detach(), "dL_dW_eff": g_we_ih.detach()},
        "hh": {"dL_dW": g_w_hh.detach(), "dL_dW_eff": g_we_hh.detach()},
    }


def bin_mean_abs(
    values: torch.Tensor, distance: torch.Tensor, edges: tuple[float, ...] | list[float]
) -> list[dict[str, float]]:
    """Partition weights by base-geometry distance. The last bin is right-inclusive."""
    flat_values = values.detach().abs().reshape(-1).cpu()
    flat_distance = distance.detach().reshape(-1).cpu()
    if flat_values.numel() != flat_distance.numel():
        raise ValueError("values and distance must have the same number of entries")
    upper = float(edges[-1])
    if float(flat_distance.max()) > upper + 1e-4:
        raise RuntimeError("a task-coordinate distance falls outside the declared bins")
    rows: list[dict[str, float]] = []
    accounted = 0
    last = len(edges) - 2
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        if index == last:
            mask = flat_distance >= low
        else:
            mask = (flat_distance >= low) & (flat_distance < high)
        count = int(mask.sum())
        accounted += count
        mean = float(flat_values[mask].mean()) if count else float("nan")
        rows.append(
            {
                "bin_id": index,
                "bin_lo": float(low),
                "bin_hi": float(high),
                "mean_abs": mean,
                "n_weights": count,
            }
        )
    if accounted != int(flat_values.numel()):
        raise RuntimeError("distance bins do not partition the weight matrix")
    return rows


def far_suppression(
    gate_ih: torch.Tensor,
    gate_hh: torch.Tensor,
    dist_ih: torch.Tensor,
    dist_hh: torch.Tensor,
    quantile: float,
) -> dict[str, Any]:
    """Zero gates whose base-geometry distance is at or above the linear quantile."""

    def threshold(distance: torch.Tensor) -> float:
        return float(np.quantile(distance.detach().cpu().numpy(), quantile, method="linear"))

    threshold_ih = threshold(dist_ih)
    threshold_hh = threshold(dist_hh)
    keep_ih = dist_ih < threshold_ih
    keep_hh = dist_hh < threshold_hh
    return {
        "gate_ih": gate_ih * keep_ih.to(dtype=gate_ih.dtype),
        "gate_hh": gate_hh * keep_hh.to(dtype=gate_hh.dtype),
        "threshold_ih": threshold_ih,
        "threshold_hh": threshold_hh,
        "n_suppressed_ih": int((~keep_ih).sum()),
        "n_suppressed_hh": int((~keep_hh).sum()),
        "fraction_suppressed_ih": float((~keep_ih).float().mean()),
        "fraction_suppressed_hh": float((~keep_hh).float().mean()),
    }


def _design(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    standardized = (features - mean) / scale
    bias = np.ones((standardized.shape[0], 1), dtype=np.float64)
    return np.concatenate([standardized, bias], axis=1)


def fit_and_score_probe(
    hidden_train: np.ndarray,
    labels_train: np.ndarray,
    hidden_val: np.ndarray,
    labels_val: np.ndarray,
    l2: float,
) -> tuple[float, float, str]:
    """Ridge probe fit on train rows only. Validation is scored and never fit."""
    train_x = np.asarray(hidden_train, dtype=np.float64)
    train_y = np.asarray(labels_train, dtype=np.float64)
    val_x = np.asarray(hidden_val, dtype=np.float64)
    val_y = np.asarray(labels_val, dtype=np.float64)
    if train_x.shape[0] < 2 or val_x.shape[0] < 1:
        return float("nan"), float("nan"), "failed"
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    design = _design(train_x, mean, scale)
    gram = design.T @ design
    penalty = float(l2) * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    try:
        beta = np.linalg.solve(gram + penalty, design.T @ train_y)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), "failed"

    def accuracy(features: np.ndarray, labels: np.ndarray) -> float:
        prediction = _design(features, mean, scale) @ beta
        return float(np.mean((prediction >= 0.5) == (labels >= 0.5)))

    return accuracy(train_x, train_y), accuracy(val_x, val_y), "ok"
