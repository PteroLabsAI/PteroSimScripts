"""Fly training generations one after another with the whole fleet, in the simulator, in real time.

Spawns --count F450s on a square grid centred at --centre-cm and, after the startup measurements, for each model
given in order: every aircraft restarts on the ground and the whole fleet flies that model for one 10 s episode, all
given the same target height and heading (the first EVAL_SEED draws), paced to real time, then rests --hold-s.
"untrained" flies a fresh policy. An env that ends is not restarted within its generation.

    python show_hover.py untrained runs/hover_s0/checkpoints/hover_1000000_steps.zip runs/hover_s0/model.zip --count 100
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fleet import SPAWN_HEIGHT_M, existing_fleet_in_step_mode, grid_placements, spawned_fleet
from frames import DOWN
from hover import (
    DEFAULT_ADDRESS,
    EVAL_SEED,
    FINAL_MODEL_STEM,
    MAX_EPISODE_STEPS,
    TARGET_HEIGHT_MAX_M,
    TARGET_HEIGHT_MIN_M,
    HoverVecEnv,
    load_trained,
    make_ppo,
)
from pterosim import PteroSim
from stable_baselines3 import PPO
from step_env import ACTION_SIZE, AGENT_DT, POS, Actuation, measure_actuation, measure_sensor_model

UNTRAINED = "untrained"
DEFAULT_COUNT = 100  # as many as trained together
DEFAULT_SPACING_M = 1.2  # chosen: a 10 x 10 grid fits one camera's frame; step mode has no contact between aircraft
DEFAULT_HOLD_S = 2.0  # chosen: the fleet at rest after each episode
START_SIGNAL_POLL_S = 0.05  # chosen
STEP_SCALES = ((1_000_000, "M"), (1_000, "k"))


@dataclass(frozen=True)
class Generation:
    """A policy to fly: a trained model with its actuation, or the untrained one (model None)."""

    label: str
    model: PPO | None
    actuation: Actuation | None


def steps_label(steps: int) -> str:
    """'250k', '1M', '10M'."""
    for scale, suffix in STEP_SCALES:
        if steps >= scale:
            return f"{steps / scale:g}{suffix}"
    return str(steps)


def load_generation(spec: str) -> Generation:
    """'untrained', or a model .zip."""
    if spec == UNTRAINED:
        return Generation(UNTRAINED, None, None)
    path = Path(spec)
    model, actuation = load_trained(path)
    label = steps_label(model.num_timesteps) + (" (final)" if path.stem == FINAL_MODEL_STEM else "")
    return Generation(label, model, actuation)


def main() -> None:
    """Spawn, measure, and fly every generation."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+", help=f'"{UNTRAINED}" or model .zip files, flown in this order')
    p.add_argument("--count", type=int, default=DEFAULT_COUNT)
    p.add_argument("--centre-cm", type=float, nargs=2, default=(0.0, 0.0), help="UE x y of the grid's centre")
    p.add_argument("--spacing-m", type=float, default=DEFAULT_SPACING_M)
    p.add_argument("--hold-s", type=float, default=DEFAULT_HOLD_S, help="wall seconds at rest after each episode")
    p.add_argument("--start-signal", default=None, help="wait for this file before the first generation flies")
    p.add_argument("--seed", type=int, default=0, help="the untrained policy's")
    p.add_argument("--address", default=DEFAULT_ADDRESS)
    args = p.parse_args()

    generations = [load_generation(spec) for spec in args.models]
    trained = [g.actuation for g in generations if g.actuation is not None]
    if not trained:
        raise SystemExit("give at least one trained model: its run sets the hover throttle and the mixer to check")
    grid = grid_placements(args.count, args.spacing_m, SPAWN_HEIGHT_M)
    middle = np.mean([g[:2] for g in grid], axis=0)
    placements = [
        (x - middle[0] + args.centre_cm[0], y - middle[1] + args.centre_cm[1], z, yaw) for x, y, z, yaw in grid
    ]
    rng = np.random.default_rng(EVAL_SEED)
    height, heading = rng.uniform(TARGET_HEIGHT_MIN_M, TARGET_HEIGHT_MAX_M), rng.uniform(-math.pi, math.pi)
    sim = PteroSim(args.address)
    timings: dict[str, float] = {}
    results = []
    try:
        with spawned_fleet(sim, placements, timings), existing_fleet_in_step_mode(sim, timings) as mode:
            sensor_model = measure_sensor_model(mode)
            measured = measure_actuation(mode, trained[0].hover_throttle)
            for actuation in trained:
                if not np.array_equal(actuation.mixer, measured.mixer):
                    raise RuntimeError(
                        f"a model was trained on mixer {actuation.mixer.tolist()}, the motors pulse as "
                        f"{measured.mixer.tolist()}"
                    )
            n = mode.num_envs
            target = np.zeros((n, 3))
            target[:, DOWN] = -height
            for k, g in enumerate(generations):
                env = HoverVecEnv(mode, g.actuation or measured, EVAL_SEED, sensor_model, autoreset=False)
                model = g.model or make_ppo(env, n, args.seed)
                env.seed(EVAL_SEED)
                env.reset()
                obs = env.retarget(target, np.full(n, heading))
                # After the first reset, so a recorder started on the signal does not see the measurements' last pose
                # jump back to the starts.
                if k == 0 and args.start_signal:
                    print(f"READY: waiting for {args.start_signal}", flush=True)
                    while not os.path.exists(args.start_signal):
                        time.sleep(START_SIGNAL_POLL_S)
                print(f"generation {g.label}: flying {n} in real time", flush=True)
                flying = np.ones(n, bool)
                ends: Counter[str] = Counter()
                late, start = 0, time.perf_counter()
                for step in range(1, MAX_EPISODE_STEPS + 1):
                    actions = np.zeros((n, ACTION_SIZE), np.float32)
                    actions[flying] = model.predict(obs[flying], deterministic=True)[0]
                    obs, _, dones, infos = env.step(actions)
                    for i in np.flatnonzero(flying & dones):
                        ends[infos[i]["end_reason"]] += 1
                    flying &= ~dones
                    lag = time.perf_counter() - (start + step * AGENT_DT)
                    if lag > 0.0:
                        late += 1
                    else:
                        time.sleep(-lag)
                distance = np.linalg.norm(env.last_truth[:, POS] - env.targets, axis=1)
                results.append(
                    f"{g.label:>12}: ends {dict(ends)}; distance to target at the end median "
                    f"{np.median(distance):.2f} m;"
                    f" steps late {late} of {MAX_EPISODE_STEPS}"
                )
                print(results[-1], flush=True)
                time.sleep(args.hold_s)
    finally:
        sim.close()
    print(f"target {height:.2f} m above the starts, heading {math.degrees(heading):.0f} deg")
    print("\n".join(results))


if __name__ == "__main__":
    main()
