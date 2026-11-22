"""The frozen protocol, its hash, and the phase 0/1 scope."""

import math

import numpy as np

from conn_assoc.protocol import frozen_constants, manifest, protocol_sha256, protocol_text


def test_manifest_hash_matches_protocol_bytes():
    record = manifest()
    assert record["protocol_sha256"] == protocol_sha256()
    assert record["hash_algorithm"] == "SHA-256"
    assert record["final_test_evaluation"] == "not_started"
    assert record["git"]["repository"] is False
    assert record["git"]["commit"] is None
    assert record["phase_status"] == "phase_0_and_1_only"


def test_protocol_states_the_research_question_and_success_rules():
    text = protocol_text()
    for phrase in (
        "Can Gaussian-gated connectivity improve",
        "Track A",
        "Track B",
        "subject-clustered",
        "Recall@1",
            "Five training seeds",
        "Row order is not recording continuity",
        "final test evaluation",
    ):
        assert phrase in text


def test_development_folds_match_the_declared_generator():
    constants = frozen_constants()
    ids = np.array(constants["official_train_subjects"])
    order = np.random.default_rng(constants["development_cv"]["seed"]).permutation(ids)
    folds = [sorted(order[i * 3:(i + 1) * 3].tolist()) for i in range(7)]
    assert folds == constants["development_cv"]["folds"]


def test_tau_schedule_constants_match_the_recorded_formula():
    schedule = frozen_constants()["schedule"]
    exact = math.sqrt(2.0 * math.log(2.0))
    assert schedule["release_step"] == int(math.floor(schedule["release_fraction"] * schedule["total_steps"] + 0.5))
    assert math.isclose(schedule["tau_start"], schedule["d50_start"] / exact)
    assert math.isclose(schedule["tau_end"], schedule["d50_end"] / exact)
    assert math.isclose(schedule["s08_tau"], schedule["s08_d50"] / exact)
    assert math.isclose(schedule["s25_tau"], schedule["s25_d50"] / exact)
    assert schedule["engineering_seeds"] == [400, 401, 402, 403, 404]
    assert set(schedule["engineering_seeds"]).isdisjoint(range(30, 35))
    assert set(schedule["engineering_seeds"]).isdisjoint(range(300, 320))
