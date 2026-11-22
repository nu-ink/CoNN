"""Stable condition identifiers, column names, and quantity definitions.

Plot code and log contracts read these names. Renaming a quantity here is a
visible change; do not alias dL/dW to the data gradient or B_prime to a
global shuffle.
"""

from __future__ import annotations

from dataclasses import dataclass

DISTANCE_DEFINITION = "task_coordinate_absolute_difference_base_geometry"
GRADIENT_DEFINITION = (
    "dL/dW is the gate-attenuated parameter gradient; "
    "dL/dW_eff is the data gradient with respect to W_eff; "
    "dL/dW = dL/dW_eff * G when the gate does not depend on W. "
    "Bins use task-coordinate distance, not shuffled-mask distance."
)
DELTA_DEFINITION = (
    "val_loss_far_suppressed minus val_loss. "
    "Positive means suppressing far connections increased validation loss. "
    "Far means base-geometry distance >= the linear quantile of that distance matrix, "
    "not a gate-value threshold and not a shuffled-mask distance."
)
ACTIVATION_BATCH = "fixed_first_minibatch_of_paired_order"
QUANTILE_METHOD = "linear"
D50_RECORD_FACTOR = 117741 / 100000


@dataclass(frozen=True)
class Condition:
    id: str
    name: str


CONDITIONS: tuple[Condition, ...] = (
    Condition("A", "dense"),
    Condition("A_prime", "scalar"),
    Condition("B_prime", "row_shuffle"),
    Condition("C", "permuted_input"),
    Condition("D", "structured"),
    Condition("E", "reverse"),
    Condition("S", "static_local"),
)

PRIMARY_TREATMENT = "D"
PRIMARY_CONTROL = "B_prime"
PRIMARY_CONTROL_NAME = "row_shuffle"
PRIMARY_METRIC = "final_test_accuracy"
PRIMARY_TASK = "and"

CURRICULUM_IDS = ("A_prime", "B_prime", "C", "D")


def condition_by_id(condition_id: str) -> Condition:
    for condition in CONDITIONS:
        if condition.id == condition_id:
            return condition
    raise KeyError(condition_id)


def condition_role(condition_id: str) -> str:
    if condition_id == PRIMARY_TREATMENT:
        return "primary_treatment"
    if condition_id == PRIMARY_CONTROL:
        return "primary_control"
    return "secondary"


def make_row(columns: tuple[str, ...], **kwargs: object) -> dict[str, object]:
    unknown = set(kwargs) - set(columns)
    if unknown:
        raise KeyError(f"Unknown columns: {sorted(unknown)}")
    row = {column: None for column in columns}
    row.update(kwargs)
    return row


METRIC_COLUMNS = (
    "run_id",
    "seed",
    "condition_id",
    "condition_name",
    "task",
    "step_index",
    "measure_timing",
    "optimizer_steps_completed",
    "split",
    "loss",
    "accuracy",
    "batch_loss",
    "phase",
    "exact_dense",
    "tau_requested",
    "tau_used",
    "tau_floored",
    "d50_exact",
    "d50_brief",
    "boundary",
    "C_ih",
    "C_hh",
    "row_energy_mean_ih",
    "row_energy_std_ih",
    "row_energy_mean_hh",
    "row_energy_std_hh",
    "mean_abs_W_ih",
    "mean_abs_W_eff_ih",
    "norm_W_ih",
    "norm_W_eff_ih",
    "mean_abs_W_hh",
    "mean_abs_W_eff_hh",
    "norm_W_hh",
    "norm_W_eff_hh",
    "frac_gate_lt_1e-6_ih",
    "frac_gate_eq_0_ih",
    "frac_gate_lt_1e-6_hh",
    "frac_gate_eq_0_hh",
    "act_var_h1",
    "act_var_h2",
    "activation_batch",
    "lr",
    "momentum",
    "weight_decay",
    "optimizer",
    "distance_definition",
    "non_finite",
    "data_fingerprint",
    "init_fingerprint",
)

GRADIENT_COLUMNS = (
    "run_id",
    "seed",
    "condition_id",
    "condition_name",
    "step_index",
    "measure_timing",
    "optimizer_steps_completed",
    "layer",
    "bin_id",
    "bin_lo",
    "bin_hi",
    "distance_definition",
    "gradient_definition",
    "mean_abs_dL_dW",
    "mean_abs_dL_dW_eff",
    "n_weights",
    "gradient_batch",
)

ABLATION_COLUMNS = (
    "run_id",
    "seed",
    "condition_id",
    "step_index",
    "measure_timing",
    "optimizer_steps_completed",
    "far_quantile",
    "quantile_method",
    "far_threshold_ih",
    "far_threshold_hh",
    "val_loss",
    "val_loss_far_suppressed",
    "val_loss_delta",
    "delta_definition",
    "distance_definition",
    "n_suppressed_ih",
    "n_suppressed_hh",
    "fraction_suppressed_ih",
    "fraction_suppressed_hh",
)

CHECKPOINT_ABLATION_COLUMNS = (
    "run_id",
    "seed",
    "condition_id",
    "checkpoint",
    "step_index",
    "optimizer_steps_completed",
    "far_quantile",
    "quantile_method",
    "far_threshold_ih",
    "far_threshold_hh",
    "val_loss",
    "val_loss_far_suppressed",
    "val_loss_delta",
    "delta_definition",
    "distance_definition",
    "n_suppressed_ih",
    "n_suppressed_hh",
    "fraction_suppressed_ih",
    "fraction_suppressed_hh",
)

PROBE_COLUMNS = (
    "run_id",
    "seed",
    "condition_id",
    "step_index",
    "measure_timing",
    "optimizer_steps_completed",
    "representation",
    "target",
    "train_accuracy",
    "val_accuracy",
    "n_train",
    "n_val",
    "fit_split",
    "eval_split",
    "probe_status",
)

ROW_ENERGY_COLUMNS = (
    "run_id",
    "seed",
    "condition_id",
    "step_index",
    "measure_timing",
    "optimizer_steps_completed",
    "layer",
    "row",
    "energy",
)

SCORE_COLUMNS = (
    "run_id",
    "seed",
    "condition_id",
    "condition_name",
    "task",
    "role",
    "optimizer_steps",
    "final_train_loss",
    "final_train_accuracy",
    "final_val_loss",
    "final_val_accuracy",
    "final_test_loss",
    "final_test_accuracy",
    "lr",
    "momentum",
    "weight_decay",
    "optimizer",
    "non_finite",
    "data_fingerprint",
    "init_fingerprint",
    "test_evaluations",
)

CONTRAST_COLUMNS = (
    "run_id",
    "task",
    "metric",
    "treatment",
    "control",
    "control_name",
    "seed",
    "treatment_value",
    "control_value",
    "delta",
    "used_in_aggregate",
)
