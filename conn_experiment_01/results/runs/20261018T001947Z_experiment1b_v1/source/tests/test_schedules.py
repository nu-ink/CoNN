from __future__ import annotations

import math

import torch

from conn_exp.gates import GateBuilder, draw_masks
from conn_exp.geometry import make_task_geometry
from conn_exp.schedules import EXACT_FACTOR, release_step_index, schedule_at, smoothstep


def _tau_pair():
    tau_start = 0.08 / EXACT_FACTOR
    tau_end = 1.0 / EXACT_FACTOR
    return tau_start, tau_end


def test_smoothstep_endpoints_and_monotonicity():
    assert smoothstep(0.0) == 0.0
    assert smoothstep(1.0) == 1.0
    values = [smoothstep(index / 20) for index in range(21)]
    assert values == sorted(values)


def test_d_curriculum_endpoints_and_exact_final_phase():
    tau_start, tau_end = _tau_pair()
    total = 400
    release = release_step_index(total, 0.7)
    assert release == 280
    taus = []
    for step in range(release):
        point = schedule_at("D", step, total, tau_start, tau_end)
        assert point.phase == "local_to_global"
        assert point.exact_dense is False
        taus.append(point.tau_requested)
    assert taus[0] == tau_start
    assert taus[-1] == tau_end
    assert all(taus[index] <= taus[index + 1] + 1e-12 for index in range(len(taus) - 1))
    pre = schedule_at("D", release - 1, total, tau_start, tau_end)
    post = schedule_at("D", release, total, tau_start, tau_end)
    assert "pre_release" in pre.boundary and pre.exact_dense is False
    assert "post_release" in post.boundary and post.exact_dense is True
    for step in range(release, total):
        point = schedule_at("D", step, total, tau_start, tau_end)
        assert point.exact_dense and point.phase == "dense_final"
    for condition in ("A_prime", "B_prime", "C"):
        matched = schedule_at(condition, 10, total, tau_start, tau_end)
        reference = schedule_at("D", 10, total, tau_start, tau_end)
        assert matched.tau_requested == reference.tau_requested
        assert matched.exact_dense is False


def test_e_has_an_observable_local_phase_and_a_dense_endpoint():
    tau_start, tau_end = _tau_pair()
    total = 400
    release = release_step_index(total, 0.7)
    start = schedule_at("E", 0, total, tau_start, tau_end)
    assert start.exact_dense and start.phase == "E_global"
    local = None
    for step in range(total):
        point = schedule_at("E", step, total, tau_start, tau_end)
        if "E_local_extreme" in point.boundary:
            local = point
    assert local is not None
    assert local.exact_dense is False
    assert local.tau_requested == tau_start
    assert "E_leave_exact" in schedule_at("E", release // 3, total, tau_start, tau_end).boundary
    end = schedule_at("E", total - 1, total, tau_start, tau_end)
    assert end.exact_dense and end.phase == "dense_final"
    geometry = make_task_geometry(100, (32, 32))
    builder = GateBuilder(geometry, draw_masks(100, 32, 32, 0), "E", 1e-6)
    local_gate = builder.build(local.tau_requested, local.exact_dense)
    wide_gate = builder.build(tau_end, False)
    assert local_gate.stats_ih.energy < wide_gate.stats_ih.energy
    assert local_gate.stats_ih.energy < 0.5
    dense = builder.build(None, True)
    assert torch.equal(dense.ih, torch.ones_like(dense.ih))


def test_s_never_releases_and_a_is_always_dense():
    tau_start, tau_end = _tau_pair()
    total = 30
    release = release_step_index(total, 0.7)
    for step in range(total):
        static = schedule_at("S", step, total, tau_start, tau_end)
        dense = schedule_at("A", step, total, tau_start, tau_end)
        assert static.exact_dense is False
        assert static.tau_requested == tau_start
        assert static.boundary == ""
        assert dense.exact_dense is True
        assert dense.tau_requested is None
    assert schedule_at("S", release, total, tau_start, tau_end).exact_dense is False
    assert math.isclose(schedule_at("D", 0, total, tau_start, tau_end).d50_exact, 0.08, rel_tol=0, abs_tol=1e-9)
