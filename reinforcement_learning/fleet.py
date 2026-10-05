"""F450s flown by nobody, spawned where asked (or on a grid), in step mode."""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from pterosim import PteroSim
from pterosim.aircraft import Aircraft
from pterosim.step import StepMode

AIRCRAFT_CLASS = "F450"
# Step mode's default physics rate; gym-pybullet-drones simulates at 240 Hz too.
PHYSICS_HZ = 240.0
# Chosen: ten F450 motor diagonals (0.47 m, Propulsion.xml), well clear of each other's rotors.
GRID_SPACING_M = 5.0
# UE world z of every spawn point. An SDK spawn starts the aircraft on the ground beneath the point, so this only has
# to be above the ground; chosen.
SPAWN_HEIGHT_M = 10.0
CM_PER_M = 100.0
# What an F450 motor channel takes (ActuatorMapping input_min/input_max).
THROTTLE_MIN = 0.0
THROTTLE_MAX = 1.0
# Float rounding only: the simulator's step length against 1 / PHYSICS_HZ.
DT_RELATIVE_TOLERANCE = 1e-9
NO_FLIGHT_STACK = ""

Placement = tuple[float, float, float, float]  # UE x, y, z (cm) of the spawn point, yaw (deg)


@contextmanager
def timed(timings: dict[str, float], phase: str) -> Iterator[None]:
    """Record the wall seconds of the block under timings[phase]."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[phase] = time.perf_counter() - t0


def grid_placements(count: int, spacing_m: float, height_m: float) -> list[Placement]:
    """`count` spawn points on a square grid, all facing yaw 0."""
    columns = math.ceil(math.sqrt(count))
    return [
        ((i % columns) * spacing_m * CM_PER_M, (i // columns) * spacing_m * CM_PER_M, height_m * CM_PER_M, 0.0)
        for i in range(count)
    ]


def _check_mode(mode: StepMode, count: int) -> None:
    if mode.num_envs != count:
        raise RuntimeError(f"step mode has {mode.num_envs} environments, {count} aircraft were spawned")
    if abs(mode.dt * PHYSICS_HZ - 1.0) > DT_RELATIVE_TOLERANCE:
        raise RuntimeError(f"physics step is {mode.dt} s, expected 1/{PHYSICS_HZ:g} s")
    odd = [
        c
        for c in mode.action_channels
        if c.type != "motor" or c.input_min != THROTTLE_MIN or c.input_max != THROTTLE_MAX
    ]
    if odd:
        raise RuntimeError(f"expected only motor channels taking {THROTTLE_MIN}..{THROTTLE_MAX}, got {odd}")


@contextmanager
def existing_fleet_in_step_mode(sim: PteroSim, timings: dict[str, float]) -> Iterator[StepMode]:
    """Every aircraft already in the running world, in step mode; on exit they fly free again, still there.

    They must be F450s with no flight stack, at PHYSICS_HZ.
    """
    if not sim.status().is_running:
        raise RuntimeError('the simulation is not running: spawn the aircraft (flight stack "") and start it first')
    count = len(sim.aircraft_status())
    if count == 0:
        raise RuntimeError("the world has no aircraft")
    with timed(timings, "enter_step_mode_s"):
        mode = sim.enter_step_mode()
    try:
        _check_mode(mode, count)
        yield mode
    finally:
        with timed(timings, "exit_step_mode_s"):
            mode.reset()  # ExitStepMode refuses while an env is crashed: every env back on its start first
            mode.close()


@contextmanager
def spawned_fleet(
    sim: PteroSim, placements: Sequence[Placement], timings: dict[str, float]
) -> Iterator[list[Aircraft]]:
    """Spawn an F450 at each placement with no flight stack and start; stop and remove them all on exit.

    Needs a stopped simulator with no aircraft, because every aircraft in the world becomes an environment.
    """
    if sim.status().is_running:
        raise RuntimeError("the simulation is running; stop it first (flight stack and physics rate need it stopped)")
    present = sim.aircraft_status()
    if present:
        names = ", ".join(f"{a.aircraft_name}#{a.instance_id}" for a in present)
        raise RuntimeError(f"the world already has aircraft ({names}); each would become an environment")
    sim.set_physics_frequency(PHYSICS_HZ)
    aircraft = []
    try:
        with timed(timings, "spawn_s"):
            for x_cm, y_cm, z_cm, yaw_deg in placements:
                aircraft.append(sim.spawn(AIRCRAFT_CLASS, x=x_cm, y=y_cm, z=z_cm, yaw=yaw_deg))
        with timed(timings, "flight_stack_s"):
            for a in aircraft:
                a.set_flight_stack(NO_FLIGHT_STACK)
        with timed(timings, "start_s"):
            sim.start()
        try:
            yield aircraft
        finally:
            with timed(timings, "stop_s"):
                sim.stop()
    finally:
        with timed(timings, "remove_s"):
            for a in aircraft:
                a.remove()


@contextmanager
def fleet_in_step_mode(
    sim: PteroSim,
    count: int,
    timings: dict[str, float],
    *,
    spacing_m: float = GRID_SPACING_M,
    height_m: float = SPAWN_HEIGHT_M,
) -> Iterator[StepMode]:
    """Spawn `count` F450s on a grid (spawned_fleet), enter step mode; undo all of it on exit."""
    with (
        spawned_fleet(sim, grid_placements(count, spacing_m, height_m), timings),
        existing_fleet_in_step_mode(sim, timings) as mode,
    ):
        yield mode
