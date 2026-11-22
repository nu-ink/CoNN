"""Fixed step schedules. The index is the optimizer step about to be taken."""

from __future__ import annotations

import math
from dataclasses import dataclass

from conn_exp.gates import resolve_tau
from conn_exp.schema import CURRICULUM_IDS, D50_RECORD_FACTOR

EXACT_FACTOR = math.sqrt(2.0 * math.log(2.0))


def smoothstep(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


def e_phase_bounds(release_step: int) -> tuple[int, int]:
    """Return the starts of E's second and third phases inside the pre-release window."""
    return release_step // 3, (2 * release_step) // 3


def release_step_index(total_steps: int, release_fraction: float = 0.7) -> int:
    """First step index that uses exact dense gates for D and for E's final phase."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    return int(math.floor(release_fraction * total_steps + 0.5))


def _tau_between(step: int, start: int, end: int, tau_a: float, tau_b: float) -> float:
    """Interpolate from tau_a at ``start`` to tau_b at ``end - 1``."""
    span = end - start
    if span < 2:
        raise ValueError("schedule phase must contain at least two steps")
    if step == start:
        return float(tau_a)
    if step == end - 1:
        return float(tau_b)
    if step < start or step >= end:
        raise ValueError("step outside phase")
    u = (step - start) / (span - 1)
    return tau_a + (tau_b - tau_a) * smoothstep(u)


@dataclass(frozen=True)
class SchedulePoint:
    condition_id: str
    step_index: int
    total_steps: int
    release_step: int
    phase: str
    exact_dense: bool
    tau_requested: float | None
    tau_used: float | None
    tau_floored: bool
    d50_exact: float | None
    d50_brief: float | None
    boundary: str


def _point_from_tau(
    *,
    condition_id: str,
    step_index: int,
    total_steps: int,
    release_step: int,
    phase: str,
    tau: float | None,
    exact_dense: bool,
    tau_floor: float,
    boundary: str,
) -> SchedulePoint:
    if exact_dense:
        return SchedulePoint(
            condition_id=condition_id,
            step_index=step_index,
            total_steps=total_steps,
            release_step=release_step,
            phase=phase,
            exact_dense=True,
            tau_requested=None,
            tau_used=None,
            tau_floored=False,
            d50_exact=None,
            d50_brief=None,
            boundary=boundary,
        )
    if tau is None:
        raise ValueError("non-dense phase requires tau")
    tau_used, floored = resolve_tau(tau, tau_floor)
    return SchedulePoint(
        condition_id=condition_id,
        step_index=step_index,
        total_steps=total_steps,
        release_step=release_step,
        phase=phase,
        exact_dense=False,
        tau_requested=float(tau),
        tau_used=tau_used,
        tau_floored=floored,
        d50_exact=tau_used * EXACT_FACTOR,
        d50_brief=tau_used * D50_RECORD_FACTOR,
        boundary=boundary,
    )


def _boundary(flags: list[str]) -> str:
    return "|".join(flags)


def _curriculum_flags(step: int, release_step: int) -> list[str]:
    flags: list[str] = []
    if step == release_step - 1:
        flags.append("pre_release")
    if step == release_step:
        flags.append("post_release")
    return flags


def schedule_at(
    condition_id: str,
    step_index: int,
    total_steps: int,
    tau_start: float,
    tau_end: float,
    release_fraction: float = 0.7,
    tau_floor: float = 1e-6,
) -> SchedulePoint:
    if not 0 <= step_index < total_steps:
        raise ValueError("step_index out of range")
    release_step = release_step_index(total_steps, release_fraction)
    if release_step < 6 or release_step >= total_steps:
        raise ValueError(
            f"release_step={release_step} does not leave a three-phase window and a final dense phase"
        )
    if condition_id == "A":
        return _point_from_tau(
            condition_id=condition_id,
            step_index=step_index,
            total_steps=total_steps,
            release_step=release_step,
            phase="dense",
            tau=None,
            exact_dense=True,
            tau_floor=tau_floor,
            boundary=_boundary(_curriculum_flags(step_index, release_step)),
        )
    if condition_id == "S":
        return _point_from_tau(
            condition_id=condition_id,
            step_index=step_index,
            total_steps=total_steps,
            release_step=release_step,
            phase="static_local",
            tau=tau_start,
            exact_dense=False,
            tau_floor=tau_floor,
            boundary="",
        )
    if condition_id in CURRICULUM_IDS:
        flags = _curriculum_flags(step_index, release_step)
        if step_index >= release_step:
            return _point_from_tau(
                condition_id=condition_id,
                step_index=step_index,
                total_steps=total_steps,
                release_step=release_step,
                phase="dense_final",
                tau=None,
                exact_dense=True,
                tau_floor=tau_floor,
                boundary=_boundary(flags),
            )
        tau = _tau_between(step_index, 0, release_step, tau_start, tau_end)
        return _point_from_tau(
            condition_id=condition_id,
            step_index=step_index,
            total_steps=total_steps,
            release_step=release_step,
            phase="local_to_global",
            tau=tau,
            exact_dense=False,
            tau_floor=tau_floor,
            boundary=_boundary(flags),
        )
    if condition_id == "E":
        p1, p2 = e_phase_bounds(release_step)
        if min(p1, p2 - p1, release_step - p2) < 2:
            raise ValueError("E phases collapsed; increase total_steps")
        flags = _curriculum_flags(step_index, release_step)
        if step_index == p1 - 1:
            flags.append("E_end_global")
        if step_index == p1:
            flags.append("E_leave_exact")
        if step_index == p2 - 1:
            flags.append("E_local_extreme")
        if step_index == p2:
            flags.append("E_start_return")
        boundary = _boundary(flags)
        if step_index >= release_step:
            phase, tau, exact = "dense_final", None, True
        elif step_index < p1:
            phase, tau, exact = "E_global", None, True
        elif step_index < p2:
            phase = "E_to_local"
            tau = _tau_between(step_index, p1, p2, tau_end, tau_start)
            exact = False
        else:
            phase = "E_return"
            tau = _tau_between(step_index, p2, release_step, tau_start, tau_end)
            exact = False
        return _point_from_tau(
            condition_id=condition_id,
            step_index=step_index,
            total_steps=total_steps,
            release_step=release_step,
            phase=phase,
            tau=tau,
            exact_dense=exact,
            tau_floor=tau_floor,
            boundary=boundary,
        )
    raise KeyError(condition_id)
