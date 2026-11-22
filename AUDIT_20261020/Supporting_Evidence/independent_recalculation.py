"""Read-only recalculation of published CoNN summaries from saved CSV tables.

This script does not import repository packages, load checkpoints, or train models.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"E:\CoNN")
OUT = ROOT / "AUDIT_20261020" / "Supporting_Evidence"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sample_sd(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return float("nan")
    return float(values.std(ddof=1))


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Continued-fraction regularized incomplete beta, Numerical Recipes form."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - ln_beta) / a
    # Use the symmetry identity when x is above the mean, for convergence.
    if x > (a + 1) / (a + b + 2):
        return 1.0 - regularized_incomplete_beta(1 - x, b, a)
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d
    for m in range(1, 300):
        # even step
        numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        f *= d * c
        # odd step
        numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        f *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return front * f


def student_t_cdf(t: float, df: float) -> float:
    x = df / (df + t * t)
    ib = regularized_incomplete_beta(x, 0.5 * df, 0.5)
    if t >= 0:
        return 1.0 - 0.5 * ib
    return 0.5 * ib


def student_t_ppf(p: float, df: float) -> float:
    lo, hi = 0.0, 1.0
    while student_t_cdf(hi, df) < p:
        hi *= 2
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def paired_summary(deltas: np.ndarray) -> dict[str, float | int]:
    deltas = np.asarray(deltas, dtype=np.float64)
    n = int(deltas.size)
    mean = float(deltas.mean())
    sd = sample_sd(deltas)
    crit = student_t_ppf(0.975, n - 1)
    se = sd / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "sample_sd": sd,
        "t_crit_975": crit,
        "ci95_low": mean - crit * se,
        "ci95_high": mean + crit * se,
        "cohens_dz": mean / sd if sd else float("nan"),
        "n_positive": int(np.sum(deltas > 0)),
        "n_zero": int(np.sum(deltas == 0)),
        "n_negative": int(np.sum(deltas < 0)),
        "min": float(deltas.min()),
        "max": float(deltas.max()),
    }


def pivot_metric(rows: list[dict[str, str]], key_fields: tuple[str, ...], metric: str, condition_field: str) -> dict[tuple, dict[str, float]]:
    out: dict[tuple, dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        out[key][row[condition_field]] = float(row[metric])
    return out


def contrast_from_rows(rows, key_fields, metric, condition_field, treatment, control) -> dict:
    table = pivot_metric(rows, key_fields, metric, condition_field)
    deltas = []
    pairs = []
    for key in sorted(table):
        if treatment in table[key] and control in table[key]:
            delta = table[key][treatment] - table[key][control]
            deltas.append(delta)
            pairs.append({"key": key, "treatment": table[key][treatment], "control": table[key][control], "delta": delta})
    summary = paired_summary(np.asarray(deltas, dtype=np.float64))
    summary["pairs"] = pairs
    return summary


def condition_means(rows, condition_field, metric) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row[condition_field]].append(float(row[metric]))
    return {
        name: {"n": len(vals), "mean": float(np.mean(vals)), "sample_sd": sample_sd(np.asarray(vals))}
        for name, vals in sorted(grouped.items())
    }


def fingerprint_agreement(rows, seed_field, condition_field, fingerprint_field) -> dict:
    by_seed: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_seed[row[seed_field]].add(row[fingerprint_field])
    mismatches = {seed: sorted(values) for seed, values in by_seed.items() if len(values) != 1}
    return {"n_seeds": len(by_seed), "n_seeds_with_disagreement": len(mismatches), "examples": dict(list(mismatches.items())[:3])}


def gate_profile(n_dest: int, n_src: int, d50: float) -> dict[str, float]:
    factor = math.sqrt(2.0 * math.log(2.0))
    tau = d50 / factor
    dest = np.linspace(0, 1, n_dest)
    src = np.linspace(0, 1, n_src)
    distance = np.abs(dest[:, None] - src[None, :])
    exponent = -(distance * distance) / (2.0 * tau * tau)
    gate = np.exp(exponent)
    gate = np.clip(gate, 0.0, 1.0)
    # float32 exponential underflow, matching the training dtype
    gate32 = np.exp(exponent.astype(np.float32)).astype(np.float32)
    gate32 = np.clip(gate32, 0.0, 1.0)
    return {
        "n_dest": n_dest,
        "n_src": n_src,
        "d50": d50,
        "tau": tau,
        "energy_mean_g2_float64": float(np.mean(gate * gate)),
        "rms": float(np.sqrt(np.mean(gate * gate))),
        "frac_lt_1e-6_float64": float(np.mean(gate < 1e-6)),
        "frac_eq_0_float32": float(np.mean(gate32 == 0)),
        "frac_lt_1e-6_float32": float(np.mean(gate32 < np.float32(1e-6))),
        "frac_gt_0_5_float32": float(np.mean(gate32 > np.float32(0.5))),
        "min_float32": float(gate32.min()),
        "median_float32": float(np.median(gate32)),
    }


def bootstrap_mean_ci(values: np.ndarray, n_resamples: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_resamples, values.size), replace=True)
    means = draws.mean(axis=1)
    return {
        "mean": float(values.mean()),
        "sample_sd": sample_sd(values),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n": int(values.size),
        "n_resamples": n_resamples,
        "rng": f"numpy.default_rng({seed})",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory() -> dict:
    skip_parts = {".venv", ".venv_cuda23", ".pip-cache23", ".pytest_cache", "__pycache__", "site-packages", "AUDIT_20261020"}
    counts = defaultdict(int)
    extensions = defaultdict(int)
    inaccessible = []
    largest = []
    for path in ROOT.rglob("*"):
        try:
            relative = path.relative_to(ROOT)
        except Exception as exc:  # noqa: BLE001
            inaccessible.append({"path": str(path), "reason": str(exc)})
            continue
        if any(part in skip_parts for part in relative.parts):
            continue
        if path.is_file():
            top = relative.parts[0] if len(relative.parts) > 1 else "(root)"
            counts[top] += 1
            extensions[path.suffix.lower() or "(none)"] += 1
            try:
                size = path.stat().st_size
            except OSError as exc:
                inaccessible.append({"path": str(relative), "reason": str(exc)})
                continue
            largest.append((size, str(relative)))
    largest.sort(reverse=True)
    return {
        "files_excluding_venvs_caches_and_this_audit": dict(counts),
        "extensions": dict(extensions),
        "largest_20": [{"bytes": size, "path": name} for size, name in largest[:20]],
        "inaccessible": inaccessible,
        "git_repository_present": (ROOT / ".git").exists(),
        "checkpoint_policy": "No .pt/.pth checkpoint was deserialized. Counts below are names only.",
    }


def main() -> None:
    t_check = {
        "df4": student_t_ppf(0.975, 4),
        "df8": student_t_ppf(0.975, 8),
        "df19": student_t_ppf(0.975, 19),
        "reference_df4": 2.7764451051977987,
        "reference_df19": 2.093024054408263,
    }
    pilot = read_csv(ROOT / "conn_experiment_01/results/runs/20261017T235839Z_pilot/scores.csv")
    rep_1b = read_csv(ROOT / "conn_experiment_01/results/runs/20261018T001947Z_experiment1b_v1/replication/scores.csv")
    rep_1c = read_csv(ROOT / "conn_experiment_01/results/runs/20261018T024146Z_experiment1c_v1/replication/scores.csv")
    rep_2 = read_csv(ROOT / "conn_experiment_01/results/runs/20261019T004346Z_experiment2_har_v1/replication/scores.csv")
    tune_2 = read_csv(ROOT / "conn_experiment_01/results/runs/20261019T004346Z_experiment2_har_v1/tuning/scores.csv")
    phase_seeds = read_csv(ROOT / "CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/analysis/selected_seed_scores.csv")
    phase_seed_deltas = read_csv(ROOT / "CoNN_Associative_Relationships/results/runs/20261019T030442340025Z_phase23_v1/analysis/individual_seed_differences.csv")
    p3_subjects = read_csv(ROOT / "CoNN_Associative_Relationships/phase3b_context_diagnostics/results/20261020_phase3b_v3_run01/analysis/subject_differences.csv")
    p3_seeds = read_csv(ROOT / "CoNN_Associative_Relationships/phase3b_context_diagnostics/results/20261020_phase3b_v3_run01/analysis/seed_differences.csv")

    pilot_contrast = contrast_from_rows(pilot, ("seed",), "final_test_accuracy", "condition_id", "D", "B_prime")
    one_b = contrast_from_rows(rep_1b, ("seed",), "test_accuracy", "condition_id", "D", "B_prime")
    one_b_ds = contrast_from_rows(rep_1b, ("seed",), "test_accuracy", "condition_id", "D", "S")
    one_b_means = condition_means(rep_1b, "condition_id", "test_accuracy")

    and_1c = [row for row in rep_1c if row["task"] == "and"]
    xor_1c = [row for row in rep_1c if row["task"] == "xor"]
    one_c = {
        "and_P_minus_D": contrast_from_rows(and_1c, ("seed",), "test_accuracy", "variant_id", "P_aligned_q0", "D_aligned_q0"),
        "and_D_minus_shuffle": contrast_from_rows(and_1c, ("seed",), "test_accuracy", "variant_id", "D_aligned_q0", "D_shuffle_q0"),
        "and_S_minus_D": contrast_from_rows(and_1c, ("seed",), "test_accuracy", "variant_id", "S_aligned_q0", "D_aligned_q0"),
        "and_F_mean": condition_means([row for row in and_1c if row["variant_id"] in {"F_aligned_q0", "P_aligned_q0", "D_aligned_q0", "S_aligned_q0", "A"}], "variant_id", "test_accuracy"),
        "xor_D_minus_shuffle": contrast_from_rows(xor_1c, ("seed",), "test_accuracy", "variant_id", "D_aligned_q0", "D_shuffle_q0"),
        "test_evaluations_max": max(int(row["test_evaluations"]) for row in rep_1c),
        "n_rows": len(rep_1c),
    }
    for item in one_c.values():
        if isinstance(item, dict) and "pairs" in item:
            item.pop("pairs")
    pilot_contrast.pop("pairs")
    one_b.pop("pairs")
    one_b_ds.pop("pairs")

    exp2 = {}
    for representation in ("engineered", "temporal"):
        subset = [row for row in rep_2 if row["representation"] == representation]
        exp2[representation] = {
            "D_minus_B_macro_f1": contrast_from_rows(subset, ("seed",), "test_macro_f1", "condition", "D", "B"),
            "D_minus_A_macro_f1": contrast_from_rows(subset, ("seed",), "test_macro_f1", "condition", "D", "A"),
            "means_macro_f1": condition_means(subset, "condition", "test_macro_f1"),
            "selected_lr": sorted({(row["condition"], row["lr"]) for row in subset}),
            "test_evaluations_max": max(int(row["test_evaluations"]) for row in subset),
            "init_fingerprint_agreement": fingerprint_agreement(subset, "seed", "condition", "init_fingerprint"),
        }
        exp2[representation]["D_minus_B_macro_f1"].pop("pairs")
        exp2[representation]["D_minus_A_macro_f1"].pop("pairs")

    matched = {}
    for representation in ("engineered", "temporal"):
        subset = [row for row in tune_2 if row["representation"] == representation and row["condition"] in {"D", "B", "A"}]
        by_lr = {}
        for lr in sorted({row["lr"] for row in subset}):
            rows = [row for row in subset if row["lr"] == lr]
            by_lr[lr] = {
                "D_minus_B_val_macro_f1": contrast_from_rows(rows, ("seed",), "val_macro_f1", "condition", "D", "B"),
                "means": condition_means(rows, "condition", "val_macro_f1"),
            }
            by_lr[lr]["D_minus_B_val_macro_f1"].pop("pairs")
        matched[representation] = by_lr
        assert max(int(row["test_evaluations"]) for row in subset) == 0

    def seed_delta(contrast: str, metric: str) -> dict:
        values = np.asarray(
            [float(row["difference"]) for row in phase_seed_deltas if row["contrast"] == contrast and row["metric"] == metric],
            dtype=np.float64,
        )
        return paired_summary(values)

    phase2 = {
        "gaussian_minus_dense_macro_f1_seed_mean": seed_delta("P2_gaussian minus P2_dense", "macro_f1"),
        "gaussian_minus_rowshuffle_macro_f1_seed_mean": seed_delta("P2_gaussian minus P2_rowshuffle", "macro_f1"),
        "fixed25_minus_gaussian_macro_f1_seed_mean": seed_delta("P2_fixed25 minus P2_gaussian", "macro_f1"),
        "context_minus_masked_macro_f1_seed_mean": seed_delta("P3_gaussian_context minus P3_gaussian_masked", "macro_f1"),
        "selected_seed_means": condition_means(phase_seeds, "condition", "macro_f1"),
    }

    primary_subjects = np.asarray(
        [float(row["difference"]) for row in p3_subjects if row["contrast"] == "gaussian_C4-gaussian_masked"],
        dtype=np.float64,
    )
    primary_seeds = np.asarray(
        [float(row["difference"]) for row in p3_seeds if row["contrast"] == "gaussian_C4-gaussian_masked"],
        dtype=np.float64,
    )
    phase3b = {
        "subject_mean_t_interval": paired_summary(primary_subjects),
        "subject_bootstrap_rng_230319": bootstrap_mean_ci(primary_subjects, 10000, 230319),
        "seed_mean_t_interval": paired_summary(primary_seeds),
        "published_subject_bootstrap_ci": [0.00011565544771546819, 0.004446269568044705],
        "published_mean": 0.0021339832267019214,
    }

    profiles = [
        gate_profile(32, 100, 0.08),
        gate_profile(32, 32, 0.08),
        gate_profile(32, 100, 1.0),
        gate_profile(32, 32, 1.0),
        gate_profile(32, 100, 0.25),
        gate_profile(64, 1152, 0.08),
        gate_profile(64, 64, 0.08),
        gate_profile(64, 1152, 0.25),
        gate_profile(64, 64, 1.0),
    ]

    duplicates = [
        ROOT / "CoNN_Associative_Relationships/phase3b_context_diagnostics/reports/PHASE3B_FINAL_REPORT.md",
        ROOT / "CoNN_Associative_Relationships/phase3b_context_diagnostics/results/20261020_phase3b_v3_run01/reports/PHASE3B_FINAL_REPORT.md",
        ROOT / "CoNN_Associative_Relationships/phase3b_context_diagnostics/results/20261020_phase3b_v3_run01/research_report.md",
    ]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in duplicates}

    # 1B duration macro from the published tuning table, recomputed here as a check of the selection coupling.
    duration_table = {
        400: [0.832812, 0.759375, 0.767188, 0.753125, 0.753125, 0.795312, 0.603125],
        800: [0.853125, 0.850000, 0.850000, 0.846875, 0.875000, 0.885938, 0.654687],
        1600: [0.856250, 0.848437, 0.876563, 0.868750, 0.904687, 0.884375, 0.756250],
        3200: [0.857812, 0.865625, 0.876563, 0.865625, 0.925000, 0.862500, 0.948438],
    }
    duration = {}
    for steps, vals in duration_table.items():
        arr = np.asarray(vals, dtype=np.float64)
        without_s = arr[:-1]
        duration[steps] = {"macro_7": float(arr.mean()), "macro_without_S": float(without_s.mean())}

    payload = {
        "t_critical_check": t_check,
        "pilot_D_minus_Bprime": pilot_contrast,
        "experiment_1b": {"D_minus_Bprime": one_b, "D_minus_S": one_b_ds, "condition_means": one_b_means,
                          "init_fingerprint_agreement": fingerprint_agreement(rep_1b, "seed", "condition_id", "init_fingerprint"),
                          "data_fingerprint_agreement": fingerprint_agreement(rep_1b, "seed", "condition_id", "data_fingerprint"),
                          "test_evaluations_max": max(int(row["test_evaluations"]) for row in rep_1b),
                          "n_rows": len(rep_1b)},
        "experiment_1c": one_c,
        "experiment_2_replication": exp2,
        "experiment_2_tuning_matched_lr_validation_only": matched,
        "phase23_seed_level": phase2,
        "phase3b_primary": phase3b,
        "analytical_gate_profiles": profiles,
        "duplicate_report_sha256": hashes,
        "experiment_1b_published_duration_table_macro": duration,
        "inventory": inventory(),
        "equations_absent": [
            "tau(q)=sigma/r(q) was not found in Python, YAML, Markdown, or JSON sources",
        ],
    }
    destination = OUT / "recalculated_statistics.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(destination)
    print("1B mean", one_b["mean"], "ci", one_b["ci95_low"], one_b["ci95_high"], "dz", one_b["cohens_dz"])
    print("pilot", pilot_contrast["mean"], pilot_contrast["ci95_low"], pilot_contrast["ci95_high"])
    print("3b subject mean", phase3b["subject_mean_t_interval"]["mean"], "boot", phase3b["subject_bootstrap_rng_230319"])
    print("t19", t_check["df19"])


if __name__ == "__main__":
    main()
