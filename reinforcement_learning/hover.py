"""F450 take-off and hold: from the ground straight up to a height above the start and a heading, PPO on step mode.

Every aircraft is one environment of one SB3 VecEnv, and every episode starts on the ground. The reward is Isaac
Lab's quadcopter task (isaaclab_tasks/direct/quadcopter/quadcopter_env.py: 10 s episodes, 15(1 - tanh(d/0.8))
- 0.05|v|^2 - 0.01|w|^2 times the agent step, no crash penalty) plus a heading term of the same shape; the target is
0.5..1.5 m straight above the start, Isaac Lab's height range. The action is [collective, roll, pitch, yaw] in
[-1, 1] with full authority (step_env.Actuation). The policy observes the client-side estimator's state, from the
simulator's IMU, magnetometer and baro and a UWB positioning model; reward, termination and evaluation use the truth.

Run against a stopped simulator with no aircraft (training on 100 envs takes ~10 min on 24 cores):
    python hover.py train --n-envs 100 --total-steps 10000000 --seed 0 --out runs/hover_s0
    python hover.py eval --n-envs 100 --model runs/hover_s0/model.zip
"""

from __future__ import annotations

import os

# The client's numpy and torch threads, set before they load: unset, they took 17 of 24 cores and crowded step mode's
# physics pool (6-13k agent steps/s against 19-25k with 4 threads, measured).
CLIENT_THREADS = 4
os.environ["OMP_NUM_THREADS"] = str(CLIENT_THREADS)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from estimator import SensorModel, UwbPositioning  # noqa: E402
from fleet import GRID_SPACING_M, SPAWN_HEIGHT_M, fleet_in_step_mode  # noqa: E402
from frames import DOWN, EAST, NORTH, body_to_ned, gravity_body, heading, to_body, wrap_angle, yaw_rate  # noqa: E402
from pterosim import PteroSim  # noqa: E402
from pterosim.step import StepMode  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.vec_env import VecEnv, VecMonitor  # noqa: E402
from step_env import (  # noqa: E402
    ACTION_SIZE,
    AGENT_DT,
    POS,
    QUAT,
    RATES,
    STEPS_PER_ACTION,
    VEL,
    Actuation,
    StepModeVecEnv,
    measure_startup,
)
from torch import nn  # noqa: E402

torch.set_num_threads(CLIENT_THREADS)

# --- Task ---
EPISODE_SECONDS = 10.0  # Isaac Lab quadcopter episode_length_s
MAX_EPISODE_STEPS = round(EPISODE_SECONDS / AGENT_DT)
TARGET_HEIGHT_MIN_M = 0.5  # Isaac Lab quadcopter: goal z uniform in 0.5..1.5 m above the ground start
TARGET_HEIGHT_MAX_M = 1.5
ERROR_CLIP_M = TARGET_HEIGHT_MAX_M  # the farthest a target is from the start
RATE_SCALE_RADPS = 4.0  # chosen: rates of a few rad/s map to order one
# error_b, v_b, rates, gravity_b, sin and cos of the heading error, previous action
OBSERVATION_SIZE = 3 + 3 + 3 + 3 + 2 + ACTION_SIZE

DISTANCE_REWARD_SCALE = 15.0  # Isaac Lab quadcopter distance_to_goal_reward_scale
DISTANCE_TANH_SCALE_M = 0.8  # Isaac Lab quadcopter: 1 - tanh(d / 0.8)
LIN_VEL_REWARD_SCALE = -0.05  # Isaac Lab quadcopter lin_vel_reward_scale
ANG_VEL_REWARD_SCALE = -0.01  # Isaac Lab quadcopter ang_vel_reward_scale
# Heading term, the position term's tanh shape. Scale chosen: a third of the position's, so holding the point stays
# first, yet spinning through every heading earns only ~22% of holding one. Tanh scale chosen: 1 rad keeps a slope
# out to ~120 deg off, while 5 deg off still pays 9% less than 0.
HEADING_REWARD_SCALE = 5.0
HEADING_TANH_SCALE_RAD = 1.0

# Sanity guards only, the ground stops a fall: chosen far beyond the gear's compression (centimetres).
BELOW_START_MARGIN_M = 0.5
CEILING_ABOVE_START_M = 4.0  # chosen: 2.5 m above the highest target
MAX_HORIZONTAL_M = 5.0  # chosen: a hover this far from above its start is lost
# Isaac Lab's quadcopter ends an episode below 0.1 m (quadcopter_env.py _get_dones, root_pos_w z < 0.1); without it
# the ground was a haven the policy never left.
GROUNDED_HEIGHT_M = 0.1
# Chosen: time allowed to climb past GROUNDED_HEIGHT_M from the gear; full collective takes ~0.3 s.
TAKEOFF_WINDOW_S = 1.0
TAKEOFF_WINDOW_STEPS = round(TAKEOFF_WINDOW_S / AGENT_DT)
TERMINATIONS = ("crashed", "grounded", "below_start", "ceiling", "horizontal", "tilt")  # tilt: past 90 deg

# --- PPO (SB3). Isaac Lab quadcopter rsl_rl: 5 epochs, 4 minibatches, gamma .99, lambda .95, entropy 0,
# [64, 64] ELU, max_grad_norm 1. ---
# Initial policy std per action axis, learned freely after. Derived from the F450's response (+0.1 throttle on one
# motor -> ~2.0 rad/s after 0.2 s): a moment std of 0.02 walks the tilt 19 deg in 1 s, so a lifting-off aircraft
# flies a second or two before noise can flip it. Collective noise cannot tilt it: 0.1 is ~+-30% thrust, a height
# walk of ~0.7 m in 2 s, enough to find the targets from the ground.
INITIAL_ACTION_STD = (0.1, 0.02, 0.02, 0.02)
SAMPLES_PER_UPDATE = 9216  # chosen: 18 envs x 512 steps; 100 envs -> 128 steps, 1000 -> MIN_ROLLOUT_STEPS
MIN_ROLLOUT_STEPS = 32  # chosen, near Isaac Lab's 24 steps per env
MINIBATCHES = 4
N_EPOCHS = 5
LEARNING_RATE = 3e-4  # SB3 default; Isaac Lab uses 5e-4 with an adaptive-KL schedule SB3 lacks
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.0
VF_COEF = 0.5  # SB3 default; Isaac Lab value_loss_coef 1.0
MAX_GRAD_NORM = 1.0
NET_ARCH = [64, 64]
DEFAULT_CHECKPOINT_EVERY = 1_000_000  # chosen, agent env-steps
DEFAULT_TOTAL_STEPS = 10_000_000  # the pass criterion's budget
RUN_FILE = "run.json"
FINAL_MODEL_STEM = "model"

# --- Evaluation: the pass criterion, over t in [5, 10] s of each env's first episode ---
EVAL_SEED = 20261004  # chosen, fixed
WINDOW_START_S = 5.0
WINDOW_START_STEP = round(WINDOW_START_S / AGENT_DT)
SURVIVAL_MIN = 0.95
RMS_MEDIAN_MAX_M = 0.10
RMS_P95_MAX_M = 0.25
HEADING_ERROR_MEDIAN_MAX_DEG = 5.0  # median over envs of each env's mean |heading error|
YAW_RATE_MEDIAN_MAX_DEGPS = 10.0  # median over envs of each env's mean |yaw rate|: no spinning
P95 = 95
ARRIVE_M = 0.1  # chosen: an aircraft has arrived within 10 cm of its target
CLIMB_MARGIN_M = 0.1  # chosen: the climb ends 10 cm under the target's height
DEFAULT_ADDRESS = "localhost:10010"


def observation(
    r: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    rates: np.ndarray,
    target: np.ndarray,
    target_heading: np.ndarray,
    prev_action: np.ndarray,
) -> np.ndarray:
    """The policy's observation from an attitude, position, velocity and body rates."""
    error_b = to_body(r, target - pos)
    norm = np.linalg.norm(error_b, axis=1, keepdims=True)
    error_b *= np.minimum(1.0, ERROR_CLIP_M / np.maximum(norm, np.finfo(float).tiny))
    heading_error = wrap_angle(target_heading - heading(r))
    parts = [
        error_b,
        to_body(r, vel),
        rates / RATE_SCALE_RADPS,
        gravity_body(r),
        np.sin(heading_error)[:, None],
        np.cos(heading_error)[:, None],
        prev_action,
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def hover_reward(truth: np.ndarray, target: np.ndarray, target_heading: np.ndarray) -> tuple[np.ndarray, ...]:
    """Isaac Lab's quadcopter reward plus the heading term per agent step; each env's distance and heading error."""
    distance = np.linalg.norm(target - truth[:, POS], axis=1)
    heading_error = wrap_angle(target_heading - heading(body_to_ned(truth[:, QUAT])))
    speed_sq = np.sum(truth[:, VEL].astype(np.float64) ** 2, axis=1)  # |v_b| = |v_ned|
    rate_sq = np.sum(truth[:, RATES].astype(np.float64) ** 2, axis=1)
    reward = (
        DISTANCE_REWARD_SCALE * (1.0 - np.tanh(distance / DISTANCE_TANH_SCALE_M))
        + HEADING_REWARD_SCALE * (1.0 - np.tanh(np.abs(heading_error) / HEADING_TANH_SCALE_RAD))
        + LIN_VEL_REWARD_SCALE * speed_sq
        + ANG_VEL_REWARD_SCALE * rate_sq
    ) * AGENT_DT
    return reward, distance, heading_error


def termination_reasons(truth: np.ndarray, crashed: np.ndarray, t: np.ndarray) -> dict[str, np.ndarray]:
    """Which envs each termination ends; t: agent steps since each env's reset, after this step."""
    return {
        "crashed": crashed,
        "grounded": (t >= TAKEOFF_WINDOW_STEPS) & (-truth[:, DOWN] < GROUNDED_HEIGHT_M),
        "below_start": truth[:, DOWN] > BELOW_START_MARGIN_M,
        "ceiling": truth[:, DOWN] < -CEILING_ABOVE_START_M,
        "horizontal": np.hypot(truth[:, NORTH], truth[:, EAST]) > MAX_HORIZONTAL_M,
        "tilt": gravity_body(body_to_ned(truth[:, QUAT]))[:, 2] < 0.0,
    }


class HoverVecEnv(StepModeVecEnv):
    """Take off from the ground start straight up to a random height, and hold it and a random heading.

    The last_* fields hold this step's per-env distance, heading error and yaw rate (truth).
    """

    def __init__(
        self,
        mode: StepMode,
        actuation: Actuation,
        seed: int,
        sensor_model: SensorModel,
        autoreset: bool = True,
    ) -> None:
        """Every env of mode, its episodes drawn from seed."""
        super().__init__(
            mode, actuation, seed, sensor_model, OBSERVATION_SIZE, TERMINATIONS, MAX_EPISODE_STEPS, autoreset
        )
        n = self.num_envs
        self.targets = np.zeros((n, 3))  # NED m from each env's start
        self.target_headings = np.zeros(n)  # rad from north toward east
        self.last_distance_m = np.zeros(n)
        self.last_heading_error_rad = np.zeros(n)
        self.last_yaw_rate_radps = np.zeros(n)

    def retarget(self, targets: np.ndarray, headings: np.ndarray) -> np.ndarray:
        """Give every env this target and heading instead of its drawn ones (right after reset); the observation."""
        self.targets[:] = targets
        self.target_headings[:] = headings
        return self._observe()

    def _begin(self, mask: np.ndarray, truth: np.ndarray) -> None:
        count = int(mask.sum())
        self.targets[mask] = 0.0
        self.targets[mask, DOWN] = -self._rng.uniform(TARGET_HEIGHT_MIN_M, TARGET_HEIGHT_MAX_M, count)
        self.target_headings[mask] = self._rng.uniform(-math.pi, math.pi, count)

    def _observe_state(self, r: np.ndarray, pos: np.ndarray, vel: np.ndarray, rates: np.ndarray) -> np.ndarray:
        return observation(r, pos, vel, rates, self.targets, self.target_headings, self._prev_action)

    def _score(self, truth: np.ndarray, crashed: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        rewards, self.last_distance_m, self.last_heading_error_rad = hover_reward(
            truth, self.targets, self.target_headings
        )
        self.last_yaw_rate_radps = yaw_rate(body_to_ned(truth[:, QUAT]), truth[:, RATES])
        return rewards, termination_reasons(truth, crashed, self._t)


# --- Evaluation ---


def median_p95(values: np.ndarray) -> tuple[float, float] | None:
    """Median and 95th percentile of the finite values; None when there are none."""
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    return (float(np.median(values)), float(np.percentile(values, P95))) if values.size else None


def evaluate(env: HoverVecEnv, policy: Callable[[np.ndarray], np.ndarray]) -> dict[str, Any]:
    """One episode per env from EVAL_SEED: the pass criterion over it, the take-off and the estimator's errors.

    The take-off ends when the aircraft first comes within CLIMB_MARGIN_M of its target's height; climb drift is
    the horizontal path flown until then, overshoot the highest it got above the target's height.
    """
    env.seed(EVAL_SEED)
    obs = env.reset()
    n = env.num_envs
    first = np.ones(n, bool)
    survived = np.zeros(n, bool)
    window = {name: np.zeros(n) for name in ("error_sq", "heading_deg", "yaw_rate_degps")}
    samples = np.zeros(n, np.int64)
    climbed = np.zeros(n, bool)
    climb_drift = np.zeros(n)
    overshoot = np.zeros(n)
    arrive_s = np.full(n, np.nan)
    previous_ne = np.zeros((n, 2))  # every env starts at its frame's origin
    estimate_errors: dict[str, list[np.ndarray]] = {}
    for k in range(1, MAX_EPISODE_STEPS + 1):
        obs, _, dones, infos = env.step(policy(obs))
        truth = env.last_truth
        ne = truth[:, POS][:, :2].astype(np.float64)
        height_above_target = -truth[:, DOWN] + env.targets[:, DOWN]
        climbing = first & ~climbed
        climb_drift[climbing] += np.linalg.norm(ne - previous_ne, axis=1)[climbing]
        climbed |= height_above_target > -CLIMB_MARGIN_M
        previous_ne = ne
        overshoot[first] = np.maximum(overshoot, height_above_target)[first]
        arrive_s[first & np.isnan(arrive_s) & (env.last_distance_m < ARRIVE_M)] = k * AGENT_DT
        in_window = first & (k >= WINDOW_START_STEP)
        for name, value in (
            ("error_sq", env.last_distance_m**2),
            ("heading_deg", np.degrees(np.abs(env.last_heading_error_rad))),
            ("yaw_rate_degps", np.degrees(np.abs(env.last_yaw_rate_radps))),
        ):
            window[name][in_window] += value[in_window]
        samples[in_window] += 1
        for name, value in env.last_estimate_error.items():
            estimate_errors.setdefault(name, []).append(value[first])
        for i in np.flatnonzero(first & dones):
            survived[i] = infos[i]["TimeLimit.truncated"]
        first &= ~dones
    if first.any():
        raise RuntimeError(f"env(s) {np.flatnonzero(first).tolist()} still flying after {MAX_EPISODE_STEPS} steps")
    mean = {name: np.where(samples > 0, total / np.maximum(samples, 1), math.nan) for name, total in window.items()}
    rms_median, rms_p95 = median_p95(np.sqrt(mean["error_sq"][survived])) or (math.inf, math.inf)
    heading_median, heading_p95 = median_p95(mean["heading_deg"][survived]) or (math.inf, math.inf)
    yaw_median, yaw_p95 = median_p95(mean["yaw_rate_degps"][survived]) or (math.inf, math.inf)
    survival = float(survived.mean())
    passed = (
        survival >= SURVIVAL_MIN
        and rms_median <= RMS_MEDIAN_MAX_M
        and rms_p95 <= RMS_P95_MAX_M
        and heading_median <= HEADING_ERROR_MEDIAN_MAX_DEG
        and yaw_median <= YAW_RATE_MEDIAN_MAX_DEGPS
    )
    return {
        "n_envs": n,
        "survival": survival,
        "rms_median_m": rms_median,
        "rms_p95_m": rms_p95,
        "heading_error_median_deg": heading_median,
        "heading_error_p95_deg": heading_p95,
        "yaw_rate_median_degps": yaw_median,
        "yaw_rate_p95_degps": yaw_p95,
        "passed": passed,
        "climb_drift_m": median_p95(climb_drift[survived]),
        "overshoot_m": median_p95(overshoot[survived]),
        "arrive_s": median_p95(arrive_s[survived]),
        "estimator_error": {name: median_p95(np.concatenate(values)) for name, values in estimate_errors.items()},
        "episode_ends": env.pop_episode_ends(),
    }


def print_evaluation(what: str, result: dict[str, Any]) -> None:
    """The evaluation, readable."""
    window = f"{WINDOW_START_S:g}-{EPISODE_SECONDS:g} s"
    print(
        f"\n{what}: survival {result['survival']:.1%} (need {SURVIVAL_MIN:.0%}), RMS error over {window} median "
        f"{result['rms_median_m']:.3f} m, p95 {result['rms_p95_m']:.3f} m (need {RMS_MEDIAN_MAX_M}, {RMS_P95_MAX_M}), "
        f"heading error median {result['heading_error_median_deg']:.1f} deg (need {HEADING_ERROR_MEDIAN_MAX_DEG}), "
        f"|yaw rate| median {result['yaw_rate_median_degps']:.1f} deg/s (need {YAW_RATE_MEDIAN_MAX_DEGPS}) "
        f"-> {'PASS' if result['passed'] else 'FAIL'}"
    )
    print(
        f"take-off, (median, p95) over the survivors: climb drift {result['climb_drift_m']} m, overshoot "
        f"{result['overshoot_m']} m, within {ARRIVE_M} m after {result['arrive_s']} s"
    )
    print(f"estimator error, (median, p95) over flying env-steps: {result['estimator_error']}")
    print(f"episode ends: {result['episode_ends']}")


# --- Training ---


def rollout_steps(num_envs: int) -> int:
    """Steps per env per rollout: SAMPLES_PER_UPDATE in all, a power of two, at least MIN_ROLLOUT_STEPS."""
    return max(MIN_ROLLOUT_STEPS, int(2 ** round(math.log2(SAMPLES_PER_UPDATE / num_envs))))


class StepRateCallback(BaseCallback):  # type: ignore[misc]
    """Per rollout: env-steps/s, the share of wall time inside step-mode calls, and the env's rollout_stats()."""

    def __init__(self, env: StepModeVecEnv) -> None:
        """Report on env."""
        super().__init__()
        self._env = env
        self._t0 = 0.0
        self._steps0 = 0
        self._sim0 = 0.0

    def _on_rollout_start(self) -> None:
        self._t0 = time.perf_counter()
        self._steps0 = self.num_timesteps
        self._sim0 = self._env.sim_wall_s

    def _on_rollout_end(self) -> None:
        wall = time.perf_counter() - self._t0
        steps = self.num_timesteps - self._steps0
        self.logger.record("sim/rollout_agent_env_steps_per_s", steps / wall)
        self.logger.record("sim/rollout_physics_env_steps_per_s", steps * STEPS_PER_ACTION / wall)
        self.logger.record("sim/step_call_share_of_rollout", (self._env.sim_wall_s - self._sim0) / wall)
        for key, value in self._env.rollout_stats().items():
            self.logger.record(key, value)

    def _on_step(self) -> bool:
        return True


def make_ppo(env: VecEnv, num_envs: int, seed: int) -> PPO:
    """A fresh PPO with the rollout length for num_envs and INITIAL_ACTION_STD."""
    n_steps = rollout_steps(num_envs)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        batch_size=n_steps * num_envs // MINIBATCHES,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs={"net_arch": {"pi": NET_ARCH, "vf": NET_ARCH}, "activation_fn": nn.ELU},
        device="cpu",
        seed=seed,
        verbose=1,
    )
    with torch.no_grad():  # SB3's log_std_init is one value for every axis; set the parameter per axis
        model.policy.log_std.copy_(torch.log(torch.tensor(INITIAL_ACTION_STD)))
    return model


def run_file(model_path: Path) -> Path:
    """The run.json of a final model (beside it) or of a checkpoint (one folder up)."""
    for folder in (model_path.parent, model_path.parent.parent):
        if (folder / RUN_FILE).exists():
            return folder / RUN_FILE
    raise SystemExit(f"no {RUN_FILE} beside {model_path} or one folder up")


def load_trained(model_path: Path) -> tuple[PPO, Actuation]:
    """A trained policy with the actuation its run.json records."""
    run = json.loads(run_file(model_path).read_text())
    if run["steps_per_action"] != STEPS_PER_ACTION:
        raise SystemExit(
            f"{model_path} acts every {run['steps_per_action']} physics steps, this script every {STEPS_PER_ACTION}"
        )
    model = PPO.load(model_path, device="cpu")
    if model.observation_space.shape != (OBSERVATION_SIZE,):
        raise SystemExit(f"{model_path} observes {model.observation_space.shape} floats, this task {OBSERVATION_SIZE}")
    return model, Actuation.from_json(run["actuation"])


def train(args: argparse.Namespace) -> None:
    """Train from scratch, saving run.json, checkpoints and the final model.zip under args.out."""
    out = Path(args.out)
    if (out / RUN_FILE).exists():
        raise SystemExit(f"{out} already holds a run; pick another --out")
    out.mkdir(parents=True, exist_ok=True)
    sim = PteroSim(args.address)
    timings: dict[str, float] = {}
    try:
        with fleet_in_step_mode(sim, args.n_envs, timings, spacing_m=args.spacing_m, height_m=args.height_m) as mode:
            actuation, sensor_model = measure_startup(mode)
            env = HoverVecEnv(mode, actuation, args.seed, sensor_model)
            model = make_ppo(VecMonitor(env), env.num_envs, args.seed)
            run = {
                "n_envs": env.num_envs,
                "seed": args.seed,
                "total_steps": args.total_steps,
                "actuation": actuation.to_json(),
                "sensor_model": sensor_model.to_json(),
                "position_source": UwbPositioning.to_json(),
                "initial_action_std": INITIAL_ACTION_STD,
                "n_steps": model.n_steps,
                "batch_size": model.batch_size,
                "steps_per_action": STEPS_PER_ACTION,
                "checkpoint_every": args.checkpoint_every,
                "timings": timings,
            }
            (out / RUN_FILE).write_text(json.dumps(run, indent=2))
            model.set_logger(configure(str(out), ["stdout", "csv", "tensorboard"]))
            checkpoints = CheckpointCallback(
                max(1, args.checkpoint_every // env.num_envs), str(out / "checkpoints"), name_prefix="hover"
            )
            try:
                model.learn(args.total_steps, callback=CallbackList([checkpoints, StepRateCallback(env)]))
            finally:
                model.save(out / FINAL_MODEL_STEM)
                print(f"saved {out / FINAL_MODEL_STEM}.zip")
    finally:
        sim.close()
    print(f"timings: {timings}")


def run_evaluation(args: argparse.Namespace) -> None:
    """Evaluate args.model on a fresh fleet; print and write the result beside the model."""
    model_path = Path(args.model)
    model, trained = load_trained(model_path)
    sim = PteroSim(args.address)
    timings: dict[str, float] = {}
    try:
        with fleet_in_step_mode(sim, args.n_envs, timings, spacing_m=args.spacing_m, height_m=args.height_m) as mode:
            actuation, sensor_model = measure_startup(mode, trained)
            env = HoverVecEnv(mode, actuation, EVAL_SEED, sensor_model)
            result = evaluate(env, lambda obs: model.predict(obs, deterministic=True)[0])
    finally:
        sim.close()
    print_evaluation(str(model_path), result)
    path = Path(args.report) if args.report else model_path.with_name(f"{model_path.stem}_eval_n{args.n_envs}.json")
    path.write_text(json.dumps(result, indent=2))
    print(f"wrote {path}")


def main() -> None:
    """Train or evaluate."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["train", "eval"])
    p.add_argument("--n-envs", type=int, required=True)
    p.add_argument("--address", default=DEFAULT_ADDRESS)
    p.add_argument("--spacing-m", type=float, default=GRID_SPACING_M)
    p.add_argument("--height-m", type=float, default=SPAWN_HEIGHT_M, help="UE world z of the spawns, m")
    p.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS, help="agent env-steps (train)")
    p.add_argument("--seed", type=int, default=0, help="train")
    p.add_argument(
        "--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY, help="agent env-steps between checkpoints"
    )
    p.add_argument("--out", default=None, help="run directory (train; default runs/hover_n<N>_s<seed>)")
    p.add_argument("--model", default=None, help="model .zip, its run.json beside it or one folder up (eval)")
    p.add_argument("--report", default=None, help="evaluation JSON (eval; default beside the model)")
    args = p.parse_args()
    if args.command == "train":
        args.out = args.out or f"runs/hover_n{args.n_envs}_s{args.seed}"
        train(args)
    elif not args.model:
        raise SystemExit("eval needs --model")
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()
