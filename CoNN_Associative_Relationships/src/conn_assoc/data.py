"""Read-only access to the Experiment 2 UCI HAR artifacts.

Loaders never write into ``conn_experiment_01``. Official test features stay
embargoed unless the caller passes purpose ``audit``. Purpose
``final_evaluation`` is rejected while this protocol is in phase 0/1.
"""

from __future__ import annotations

import hashlib
from functools import cache
from io import BytesIO
from zipfile import ZipFile

import numpy as np

from conn_assoc.paths import archive_path, experiment2_run
from conn_assoc.protocol import frozen_constants

_PREFIX = "UCI HAR Dataset/"


class EmbargoError(PermissionError):
    """Raised when official test features are requested outside an audit."""


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive() -> str:
    path = archive_path()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = _sha256(path)
    expected = frozen_constants()["experiment2"]["archive_sha256"]
    if digest != expected:
        raise RuntimeError(f"Archive SHA-256 changed: {digest}")
    return digest


@cache
def _nested_zip_bytes() -> bytes:
    verify_archive()
    with ZipFile(archive_path()) as outer:
        return outer.read("UCI HAR Dataset.zip")


def _read_member(name: str) -> bytes:
    with ZipFile(BytesIO(_nested_zip_bytes())) as inner:
        return inner.read(_PREFIX + name)


def vendor_matrix(name: str) -> np.ndarray:
    """Load one dataset text matrix. The archive is not extracted."""
    return np.loadtxt(BytesIO(_read_member(name)))


def inertial_windows(split: str) -> np.ndarray:
    """Vendor 128-step windows, shape (n, 128, 9), channel order from the protocol."""
    if split not in {"train", "test"}:
        raise ValueError("Vendor files exist only for the official train and test splits")
    channels = frozen_constants()["channels"]
    planes = [vendor_matrix(f"{split}/Inertial Signals/{channel}_{split}.txt") for channel in channels]
    stacked = np.stack(planes, axis=2)
    if stacked.shape[1:] != (128, len(channels)):
        raise RuntimeError(f"Unexpected inertial shape {stacked.shape}")
    return stacked


def prepared_dir(representation: str):
    if representation not in {"engineered", "temporal"}:
        raise ValueError(representation)
    return experiment2_run() / "prepared" / representation


def verify_prepared_manifest() -> None:
    import json

    manifest_path = experiment2_run() / "prepared_manifest.json"
    digest = _sha256(manifest_path)
    expected = frozen_constants()["experiment2"]["prepared_manifest_sha256"]
    if digest != expected:
        raise RuntimeError("prepared_manifest.json hash does not match the protocol")
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, file_hash in recorded.items():
        actual = _sha256(experiment2_run() / "prepared" / relative)
        if actual != file_hash:
            raise RuntimeError(f"Prepared file hash mismatch: {relative}")


def _check_purpose(split: str, purpose: str) -> None:
    if purpose == "final_evaluation":
        raise EmbargoError("Final test evaluation is not authorized in phase 0/1")
    if purpose not in {"development", "audit"}:
        raise ValueError("purpose must be 'development' or 'audit'")
    if split == "test" and purpose != "audit":
        raise EmbargoError("Official test features are embargoed until a later authorized evaluation")


def load_prepared(representation: str, split: str, kind: str, purpose: str) -> np.ndarray:
    """Load one prepared array.

    ``kind`` is ``x``, ``y``, or ``subject``. ``split`` is the Experiment 2
    prepared split: ``train`` (16 fit subjects), ``val`` (5 validation subjects),
    or ``test`` (9 official test subjects).
    """
    if kind not in {"x", "y", "subject"}:
        raise ValueError(kind)
    if split not in {"train", "val", "test"}:
        raise ValueError(split)
    _check_purpose(split, purpose)
    path = prepared_dir(representation) / f"{kind}_{split}.npy"
    # np.load is a read. The memmap is copied so the file handle is not retained.
    return np.array(np.load(path, mmap_mode="r"))


def scaler(representation: str) -> dict[str, np.ndarray]:
    path = prepared_dir(representation) / "scaler.npz"
    with np.load(path) as payload:
        return {key: np.array(payload[key]) for key in payload.files}


def official_train_subjects_in_file_order() -> np.ndarray:
    return vendor_matrix("train/subject_train.txt").astype(np.int64)


def stitch_official_train(representation: str) -> dict[str, np.ndarray]:
    """Rebuild the 7352 official training rows in released file order.

    Experiment 2 stored the 16 fit subjects and 5 validation subjects as two
    arrays, each preserving the original relative order. This interleaves those
    arrays with the vendor subject vector. It does not treat adjacent rows as
    overlapping recordings.
    """
    constants = frozen_constants()
    fit_ids = set(constants["experiment2_fit_subjects"])
    val_ids = set(constants["experiment2_validation_subjects"])
    subjects = official_train_subjects_in_file_order()
    x_fit = load_prepared(representation, "train", "x", "development")
    y_fit = load_prepared(representation, "train", "y", "development")
    s_fit = load_prepared(representation, "train", "subject", "development")
    x_val = load_prepared(representation, "val", "x", "development")
    y_val = load_prepared(representation, "val", "y", "development")
    s_val = load_prepared(representation, "val", "subject", "development")
    x_out = np.empty((len(subjects), x_fit.shape[1]), dtype=x_fit.dtype)
    y_out = np.empty(len(subjects), dtype=y_fit.dtype)
    s_out = np.empty(len(subjects), dtype=np.int64)
    i_fit = i_val = 0
    for row, subject in enumerate(subjects.tolist()):
        if subject in fit_ids:
            x_out[row] = x_fit[i_fit]
            y_out[row] = y_fit[i_fit]
            s_out[row] = s_fit[i_fit]
            i_fit += 1
        elif subject in val_ids:
            x_out[row] = x_val[i_val]
            y_out[row] = y_val[i_val]
            s_out[row] = s_val[i_val]
            i_val += 1
        else:
            raise RuntimeError(f"Official training row {row} has subject {subject} outside the frozen partition")
    if i_fit != len(s_fit) or i_val != len(s_val):
        raise RuntimeError("Prepared train/val arrays were not consumed exactly once")
    return {"x": x_out, "y": y_out, "subject": s_out}


def inverse_vendor(values: np.ndarray, representation: str, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Undo the Experiment 2 z-score. This does not undo vendor filtering or [-1, 1] scaling."""
    values = np.asarray(values, dtype=np.float64)
    if representation == "temporal":
        return values.reshape(-1, 128, 9) * scale + mean
    if representation == "engineered":
        return values * scale + mean
    raise ValueError(representation)


def transform_vendor(values: np.ndarray, representation: str, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if representation == "temporal":
        transformed = ((values.reshape(-1, 128, 9) - mean) / scale).reshape(values.shape)
    elif representation == "engineered":
        transformed = (values - mean) / scale
    else:
        raise ValueError(representation)
    return np.ascontiguousarray(transformed, dtype=np.float32)


def assert_no_subject_leakage(purpose: str = "audit") -> dict[str, list[int]]:
    """Check the prepared subject IDs. Test features are read only for this audit."""
    found = {}
    for representation in ("engineered", "temporal"):
        sets = {}
        for split, key in (("train", "fit"), ("val", "validation"), ("test", "test")):
            subjects = load_prepared(representation, split, "subject", purpose)
            sets[key] = set(int(v) for v in np.unique(subjects))
        if sets["fit"] & sets["validation"] or sets["fit"] & sets["test"] or sets["validation"] & sets["test"]:
            raise RuntimeError(f"Subject leakage in {representation}: {sets}")
        constants = frozen_constants()
        if sets["fit"] != set(constants["experiment2_fit_subjects"]):
            raise RuntimeError("Fit subjects differ from the frozen protocol")
        if sets["validation"] != set(constants["experiment2_validation_subjects"]):
            raise RuntimeError("Validation subjects differ from the frozen protocol")
        if sets["test"] != set(constants["official_test_subjects"]):
            raise RuntimeError("Test subjects differ from the frozen protocol")
        found[representation] = {name: sorted(values) for name, values in sets.items()}
    return found


def overlap_exclusion_available() -> bool:
    """Released files do not identify which distinct rows share a recording interval."""
    return False


def neighbor_window_ids(row: int) -> list[int]:
    """Refuse to invent neighbors from row order."""
    if overlap_exclusion_available():
        raise RuntimeError("Overlap metadata appeared; this function must be revised before use")
    raise RuntimeError(
        "Window-overlap neighbors cannot be reconstructed. Row order is not recording continuity."
    )
