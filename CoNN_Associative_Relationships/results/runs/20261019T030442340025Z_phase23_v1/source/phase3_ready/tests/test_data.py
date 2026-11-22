"""Read-only HAR integrity, subject separation, and the embargo on test features."""

import numpy as np
import pytest

from conn_assoc.data import (
    EmbargoError,
    assert_no_subject_leakage,
    inertial_windows,
    inverse_vendor,
    load_prepared,
    neighbor_window_ids,
    overlap_exclusion_available,
    scaler,
    stitch_official_train,
    transform_vendor,
    verify_archive,
    verify_prepared_manifest,
)
from conn_assoc.paths import archive_path, experiment2_run
from conn_assoc.protocol import frozen_constants


def test_archive_and_prepared_hashes_match_the_protocol():
    assert verify_archive() == frozen_constants()["experiment2"]["archive_sha256"]
    verify_prepared_manifest()
    assert archive_path().is_file()
    assert experiment2_run().is_dir()


def test_raw_128_step_windows_exist_for_both_official_splits():
    train = inertial_windows("train")
    test = inertial_windows("test")
    assert train.shape == (7352, 128, 9)
    assert test.shape == (2947, 128, 9)
    assert np.isfinite(train).all() and np.isfinite(test).all()


def test_no_training_test_subject_leakage():
    found = assert_no_subject_leakage("audit")
    constants = frozen_constants()
    train_ids = set(constants["official_train_subjects"])
    test_ids = set(constants["official_test_subjects"])
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids) == 21 and len(test_ids) == 9
    for representation in ("engineered", "temporal"):
        fit = set(found[representation]["fit"])
        validation = set(found[representation]["validation"])
        test = set(found[representation]["test"])
        assert fit | validation == train_ids
        assert fit.isdisjoint(validation)
        assert test == test_ids
        y_train = load_prepared(representation, "train", "y", "development")
        assert set(int(v) for v in np.unique(y_train)) == set(range(6))


def test_development_loader_refuses_the_official_test_split():
    with pytest.raises(EmbargoError):
        load_prepared("temporal", "test", "x", "development")
    with pytest.raises(EmbargoError):
        load_prepared("engineered", "test", "y", "final_evaluation")
    audited = load_prepared("temporal", "test", "subject", "audit")
    assert set(int(v) for v in np.unique(audited)) == set(frozen_constants()["official_test_subjects"])


def test_experiment2_normalization_inverts_on_fit_subjects_only():
    constants = frozen_constants()
    fit_ids = set(constants["experiment2_fit_subjects"])
    for representation in ("temporal", "engineered"):
        stitched = stitch_official_train(representation)
        stats = scaler(representation)
        assert set(int(v) for v in stats["fit_subjects"]) == fit_ids
        assert int(np.sum(stats["constant"])) == 0
        fit = np.isin(stitched["subject"], list(fit_ids))
        if representation == "temporal":
            vendor = inertial_windows("train").reshape(7352, -1)
        else:
            from conn_assoc.data import vendor_matrix

            vendor = vendor_matrix("train/X_train.txt")
            assert vendor.min() >= -1 and vendor.max() <= 1
        rebuilt = transform_vendor(vendor[fit], representation, stats["mean"], stats["scale"])
        assert np.allclose(rebuilt, stitched["x"][fit], atol=1e-5, rtol=1e-5)
        recovered = inverse_vendor(stitched["x"][fit], representation, stats["mean"], stats["scale"])
        reference = vendor[fit].reshape(recovered.shape)
        assert np.allclose(recovered, reference, atol=1e-5, rtol=1e-5)
        # Validation rows were not part of the scaler fit, so their mean is not forced to zero.
        val = ~np.isin(stitched["subject"], list(fit_ids))
        assert set(int(v) for v in np.unique(stitched["subject"][val])).isdisjoint(fit_ids)


def test_grouped_development_folds_cover_training_subjects_once():
    constants = frozen_constants()
    folds = constants["development_cv"]["folds"]
    assert len(folds) == 7
    flat = [subject for fold in folds for subject in fold]
    assert sorted(flat) == constants["official_train_subjects"]
    assert len(flat) == len(set(flat)) == 21
    assert set(flat).isdisjoint(constants["official_test_subjects"])
    for fold in folds:
        assert len(fold) == 3


def test_row_order_is_not_treated_as_overlap_provenance():
    assert overlap_exclusion_available() is False
    with pytest.raises(RuntimeError, match="Row order is not recording continuity"):
        neighbor_window_ids(0)
    stitched = stitch_official_train("temporal")
    # Adjacent file rows can share a subject. That fact is not an overlap label.
    same_subject_neighbors = int(np.sum(stitched["subject"][1:] == stitched["subject"][:-1]))
    assert same_subject_neighbors > 0


def test_loaders_do_not_write_into_experiment2():
    before = {
        path: path.stat().st_mtime_ns
        for path in (experiment2_run() / "prepared").rglob("*")
        if path.is_file()
    }
    stitch_official_train("engineered")
    after = {
        path: path.stat().st_mtime_ns
        for path in (experiment2_run() / "prepared").rglob("*")
        if path.is_file()
    }
    assert before == after
