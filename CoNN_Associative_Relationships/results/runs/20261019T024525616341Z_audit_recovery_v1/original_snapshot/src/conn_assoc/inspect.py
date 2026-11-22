"""Phase 0/1 inspection. Writes only under this experiment directory."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from conn_assoc.data import (
    assert_no_subject_leakage,
    inertial_windows,
    overlap_exclusion_available,
    scaler,
    inverse_vendor,
    stitch_official_train,
    transform_vendor,
    vendor_matrix,
    verify_archive,
    verify_prepared_manifest,
)
from conn_assoc.paths import EXPERIMENT_ROOT, archive_path, experiment2_run
from conn_assoc.protocol import frozen_constants, protocol_sha256

INSPECTION_PATH = EXPERIMENT_ROOT / "audit" / "inspection.json"


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def run_inspection() -> dict:
    constants = frozen_constants()
    verify_archive()
    verify_prepared_manifest()
    leakage = assert_no_subject_leakage("audit")
    report = {
        "inspected_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_sha256(),
        "archive": str(archive_path()),
        "experiment2_run": str(experiment2_run()),
        "overlap_exclusion_available": overlap_exclusion_available(),
        "subject_sets": leakage,
        "representations": {},
    }
    vendor_subjects = vendor_matrix("train/subject_train.txt").astype(np.int64)
    if sorted(set(vendor_subjects.tolist())) != constants["official_train_subjects"]:
        raise RuntimeError("Vendor training subject IDs differ from the protocol")
    test_subjects = vendor_matrix("test/subject_test.txt").astype(np.int64)
    if sorted(set(test_subjects.tolist())) != constants["official_test_subjects"]:
        raise RuntimeError("Vendor test subject IDs differ from the protocol")
    fit_ids = set(constants["experiment2_fit_subjects"])
    for representation in ("temporal", "engineered"):
        stitched = stitch_official_train(representation)
        stats = scaler(representation)
        fit = np.isin(vendor_subjects, list(fit_ids))
        if representation == "temporal":
            vendor = inertial_windows("train").reshape(len(vendor_subjects), -1)
        else:
            vendor = vendor_matrix("train/X_train.txt")
        rebuilt = transform_vendor(vendor[fit], representation, stats["mean"], stats["scale"])
        prepared_fit_x = stitched["x"][fit]
        gap = _max_abs(rebuilt, prepared_fit_x)
        if gap > 1e-5:
            raise RuntimeError(f"{representation} fit rows do not match the vendor transform ({gap})")
        inverse = inverse_vendor(prepared_fit_x, representation, stats["mean"], stats["scale"])
        reference = vendor[fit].reshape(-1, 128, 9) if representation == "temporal" else vendor[fit]
        inverse_gap = _max_abs(inverse, reference)
        if inverse_gap > 1e-5:
            raise RuntimeError(f"{representation} inverse normalization failed ({inverse_gap})")
        report["representations"][representation] = {
            "official_train_shape": list(stitched["x"].shape),
            "fit_transform_max_abs": gap,
            "inverse_to_vendor_max_abs": inverse_gap,
            "scaler_fit_subjects": [int(v) for v in stats["fit_subjects"].tolist()],
            "constant_dimensions": int(np.sum(stats["constant"])),
        }
    windows = inertial_windows("test")
    report["raw_test_window_shape"] = list(windows.shape)
    report["raw_windows_available"] = True
    report["models_trained"] = 0
    report["final_test_evaluation"] = "not_started"
    return report


def main() -> int:
    report = run_inspection()
    INSPECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    INSPECTION_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PHASE 0/1 INSPECTION PASSED: {INSPECTION_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
