"""Step mode as a Stable-Baselines3 VecEnv, and what an aircraft must be measured for before it can be flown.

Every aircraft in the world is one environment. Before training, the startup measurements find the throttle that
lifts it off the ground, how each motor turns it (the mixer, which must be an X quad) and its sensors' noise at rest.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from estimator import ACC, BARO, GYRO, MAG, SENSOR_FIELDS, SensorModel, StateEstimator, UwbPositioning
from fleet import PHYSICS_HZ, THROTTLE_MAX, THROTTLE_MIN
from frames import DOWN, DOWN_AXIS, STANDARD_GRAVITY, body_to_ned, tilt_deg, to_ned
from gymnasium import spaces
from pterosim.step import StepMode
from pterosim.types import StepResult
from stable_baselines3.common.vec_env import VecEnv

# The flight model's truth (PteroSimCore StepSession.h OBSERVATION_FIELDS), looked up by name; the slices index this
# order. Reward, termination and evaluation use it; the policy sees the estimator's state.
TRUTH_FIELDS = (
    "north_m",
    "east_m",
    "down_m",
    "qw",
    "qx",
    "qy",
    "qz",
    "v_north_mps",
    "v_east_mps",
    "v_down_mps",
    "p_radps",
    "q_radps",
    "r_radps",
    "f_x_mps2",
    "f_y_mps2",
    "f_z_mps2",
)
POS, QUAT, VEL, RATES, FORCE = slice(0, 3), slice(3, 7), slice(7, 10), slice(10, 13), slice(13, 16)
MOTORS = 4
ACTION_SIZE = 1 + 3  # collective, roll, pitch, yaw (Isaac Lab quadcopter: thrust + three moments)
INT32_SEED_LIMIT = 2**31  # StepService.proto: each env's seed is an int32
# 240 / 5 = 48 Hz agent: gym-pybullet-drones' default ctrl_freq at its 240 Hz pyb_freq.
STEPS_PER_ACTION = 5
AGENT_DT = STEPS_PER_ACTION / PHYSICS_HZ

# Startup measurements; every env starts on the ground.
MIN_SWEEP_ENVS = 2  # the hover sweep brackets the lift-off throttle between envs
SWEEP_LOW_THROTTLE = 0.3  # chosen around the F450 hover estimates in earlier scripts (0.40..0.43)
SWEEP_HIGH_THROTTLE = 0.6
SWEEP_ROUNDS = 3  # with 3 envs the bracket ends 0.3 / 2 / 4 / 4 ~ 0.009 wide; wider fleets far narrower
SWEEP_S = 3.0  # each throttle held 3 s from the ground
CLIMB_SPEED_MPS = 0.02  # chosen: a resting aircraft reads 0; 0.5% over hover climbs >0.1 m/s by 3 s
HOVER_THROTTLE_RESOLUTION = 1e-3  # chosen: half of it is ~0.25% thrust (~0.02 m/s^2)
LEVEL_MAX_TILT_DEG = 10.0  # chosen: four equal throttles must keep the aircraft about level
CLIMB_THROTTLE_MARGIN = 0.05  # chosen: ~25% more thrust than hover, ~2 m/s^2 upward
CHECK_HEIGHT_M = 1.0  # chosen: the motor checks run here, clear of the gear
CLIMB_TIMEOUT_S = 5.0  # chosen: the climb to CHECK_HEIGHT_M takes ~1 s at that margin
CHANNEL_PROBE_S = 0.2  # chosen
CHANNEL_PROBE_DELTA = 0.1  # chosen: throttle added to the probed motor
RATE_RESPONSE_MIN_RADPS = 0.05  # chosen: well clear of zero, far below the ~1 rad/s a push gives
FRAME_CHECK_MIN_TILT_DEG = 5.0  # chosen: a transposed attitude then errs by >= 2 g sin(5 deg) ~ 1.7 m/s^2
FRAME_CHECK_TOLERANCE_MPS2 = 1.0  # chosen: under that error, above drag and gravity-model differences
SENSOR_CALIBRATION_S = 10.0  # chosen: motors off on the ground; 2400 IMU and magnetometer, ~500 baro readings per env
GRAVITY_TOLERANCE = 0.05  # chosen: at rest the accelerometer must read standard gravity within 5%
NOISE_FLOOR = 1e-6  # chosen: below it a measured std or field counts as zero


def field_columns(fields: Sequence[str], names: Sequence[str]) -> np.ndarray:
    """Each named field's column in step mode's observations; fails naming every missing one."""
    fields = list(fields)
    missing = [name for name in names if name not in fields]
    if missing:
        raise RuntimeError(f"step mode reports no {missing}; its observation fields are {fields}")
    return np.array([fields.index(name) for name in names])


@dataclass(frozen=True)
class Actuation:
    """Four throttles from a [collective, roll, pitch, yaw] action in [-1, 1], with full authority.

    Each motor's demand d = collective + its roll, pitch and yaw signs times those commands, clipped to [-1, 1],
    then d = 0 is hover, d = 1 full throttle, d = -1 zero throttle (linear on each side). A static remap, no feedback.
    """

    hover_throttle: float
    mixer: np.ndarray  # (MOTORS, 3): each channel's sign of p, q, r when it alone is pushed

    def motors(self, actions: np.ndarray) -> np.ndarray:
        """Throttles (N, MOTORS) for actions (N, ACTION_SIZE)."""
        demand = np.clip(actions[:, :1] + actions[:, 1:] @ self.mixer.T, -1.0, 1.0)
        up, down = THROTTLE_MAX - self.hover_throttle, self.hover_throttle - THROTTLE_MIN
        return (self.hover_throttle + np.where(demand >= 0.0, demand * up, demand * down)).astype(np.float32)

    def to_json(self) -> dict[str, Any]:
        """The actuation for run.json."""
        return {"hover_throttle": self.hover_throttle, "mixer": self.mixer.tolist()}

    @staticmethod
    def from_json(d: dict[str, Any]) -> Actuation:
        """The actuation run.json recorded."""
        return Actuation(float(d["hover_throttle"]), np.array(d["mixer"], int))


# --- Startup measurements ---


def step_checked(
    mode: StepMode, motors: np.ndarray, steps: int, elapsed_s: float, names: Sequence[str] = TRUTH_FIELDS
) -> np.ndarray:
    """One step() call that must not crash or diverge; the named observation columns, in that order."""
    result = mode.step(motors, steps=steps)
    bad = np.flatnonzero(np.asarray(result.crashed, bool) | ~np.isfinite(result.observations).all(axis=1))
    if bad.size:
        raise RuntimeError(
            f"env(s) {bad.tolist()} crashed {elapsed_s + steps * mode.dt:.2f} s into holding motors "
            f"{motors[bad[0]].tolist()}"
        )
    return result.observations[:, field_columns(mode.observation_fields, names)]


def fly(
    mode: StepMode, motors: np.ndarray, seconds: float, steps_per_call: int, names: Sequence[str] = TRUTH_FIELDS
) -> np.ndarray:
    """Hold the motor commands for `seconds` of sim time; the named columns after every call, (calls, N, fields)."""
    calls = round(seconds / (steps_per_call * mode.dt))
    return np.stack(
        [step_checked(mode, motors, steps_per_call, call * steps_per_call * mode.dt, names) for call in range(calls)]
    )


def require_level(trace: np.ndarray, what: str) -> None:
    """Fail if any env of a (calls, N, truth) trace tilted past LEVEL_MAX_TILT_DEG."""
    tilt = np.stack([tilt_deg(body_to_ned(obs[:, QUAT])) for obs in trace]).max(axis=0)
    if (tilt > LEVEL_MAX_TILT_DEG).any():
        raise RuntimeError(
            f"{what} tilted env(s) {np.flatnonzero(tilt > LEVEL_MAX_TILT_DEG).tolist()} up to {tilt.max():.1f} deg "
            f"(limit {LEVEL_MAX_TILT_DEG}): the motors are not balanced (a channel dead or misrouted?)"
        )


def measure_hover_throttle(mode: StepMode) -> float:
    """Throttle on all four motors that just lifts the aircraft off its ground start, by repeated sweeps.

    Each round restarts every env on the ground and holds one throttle per env for SWEEP_S: an env rising faster
    than CLIMB_SPEED_MPS at the end climbs, any other sinks. Round 1 spreads SWEEP_LOW..SWEEP_HIGH over the envs;
    later rounds split the bracket between the highest sinking and the lowest climbing throttle.
    """
    n = mode.num_envs
    if n < MIN_SWEEP_ENVS:
        raise RuntimeError(f"the hover sweep needs at least {MIN_SWEEP_ENVS} envs")
    lo, hi = SWEEP_LOW_THROTTLE, SWEEP_HIGH_THROTTLE
    throttles = np.linspace(lo, hi, n)
    for sweep in range(SWEEP_ROUNDS):
        if hi - lo <= HOVER_THROTTLE_RESOLUTION:
            break
        mode.reset()
        trace = fly(
            mode, np.repeat(throttles[:, None], mode.action_size, axis=1).astype(np.float32), SWEEP_S, STEPS_PER_ACTION
        )
        require_level(trace, f"equal throttles in sweep {sweep + 1}")
        v_down = trace[-1, :, VEL][:, DOWN]
        height = -trace[-1, :, DOWN]
        sinking = v_down > -CLIMB_SPEED_MPS
        table = ", ".join(f"{t:.4f}:{v:+.3f}m/s@{h:.2f}m" for t, v, h in zip(throttles, v_down, height, strict=True))
        print(f"hover sweep {sweep + 1}: throttle:v_down@height {table}")
        if sweep == 0 and (sinking.all() or not sinking.any()):
            raise RuntimeError(f"lift-off throttle outside {SWEEP_LOW_THROTTLE}..{SWEEP_HIGH_THROTTLE}: {table}")
        if sinking.any():
            lo = max(lo, float(throttles[sinking].max()))
        if (~sinking).any():
            hi = min(hi, float(throttles[~sinking].min()))
        if lo >= hi:
            raise RuntimeError(f"grounded above climbing throttles, not monotonic: {table}")
        throttles = np.linspace(lo, hi, n + 2)[1:-1]
    hover = (lo + hi) / 2
    print(f"hover throttle {hover:.4f} (bracket {lo:.4f}..{hi:.4f})")
    return hover


def climb_to_check_height(mode: StepMode, hover_throttle: float) -> None:
    """Restart every env and climb on equal throttles until all are CHECK_HEIGHT_M above their start."""
    mode.reset()
    motors = np.full((mode.num_envs, mode.action_size), hover_throttle + CLIMB_THROTTLE_MARGIN, np.float32)
    trace = []
    for call in range(round(CLIMB_TIMEOUT_S / AGENT_DT)):
        trace.append(step_checked(mode, motors, STEPS_PER_ACTION, call * AGENT_DT))
        if (-trace[-1][:, DOWN] >= CHECK_HEIGHT_M).all():
            break
    else:
        raise RuntimeError(
            f"{CLIMB_TIMEOUT_S} s at throttle {motors[0, 0]:.4f} lifted the envs only "
            f"{-trace[-1][:, DOWN].min():.2f}..{-trace[-1][:, DOWN].max():.2f} m (need {CHECK_HEIGHT_M} m): "
            "a motor dead, or the hover throttle wrong"
        )
    require_level(np.stack(trace), f"climbing at {motors[0, 0]:.4f} on all motors")


def measure_actuation(mode: StepMode, hover_throttle: float) -> Actuation:
    """In the air, push each motor alone; its signs of p, q, r form the mixer, which must be an X quad.

    Every channel gets its own climb to CHECK_HEIGHT_M first, clear of the gear. Each push must turn the aircraft
    about every axis, and while it tilts, the specific force rotated to NED plus gravity must match the measured
    acceleration (the attitude is body-to-NED). With the collective, the mixer's columns must be orthogonal: four
    corners, diagonal motors yawing the same way.
    """
    if mode.action_size != MOTORS:
        raise RuntimeError(f"expected {MOTORS} motor channels, got {mode.action_size}")
    responses = []
    for channel in range(MOTORS):
        climb_to_check_height(mode, hover_throttle)
        motors = np.full((mode.num_envs, MOTORS), hover_throttle, np.float32)
        motors[:, channel] += CHANNEL_PROBE_DELTA
        trace = fly(mode, motors, CHANNEL_PROBE_S, STEPS_PER_ACTION)
        rates = trace[-1, :, RATES].mean(axis=0)
        print(f"channel {channel} pushed: p, q, r = {np.array2string(rates, precision=2)} rad/s")
        if (np.abs(rates) < RATE_RESPONSE_MIN_RADPS).any():
            raise RuntimeError(
                f"channel {channel} turns the aircraft less than {RATE_RESPONSE_MIN_RADPS} rad/s "
                f"about some axis: {rates}"
            )
        _check_frame(trace[-2], trace[-1], channel)
        responses.append(rates)
    mixer = np.sign(np.array(responses)).astype(int)
    basis = np.column_stack([np.ones(MOTORS, int), mixer])
    if not np.array_equal(basis.T @ basis, MOTORS * np.eye(MOTORS, dtype=int)):
        raise RuntimeError(
            f"the pulse signs are not an X quad (collective, roll, pitch and yaw must be orthogonal): {mixer.tolist()}"
        )
    print(f"mixer (p, q, r sign per channel): {mixer.tolist()}")
    return Actuation(hover_throttle, mixer)


def _check_frame(before: np.ndarray, after: np.ndarray, channel: int) -> None:
    r_before, r_after = body_to_ned(before[:, QUAT]), body_to_ned(after[:, QUAT])
    tilt = tilt_deg(r_after)
    if tilt.min() < FRAME_CHECK_MIN_TILT_DEG:
        raise RuntimeError(f"channel {channel} probe tilted only {tilt.min():.1f} deg, too little to check the frame")
    measured = (after[:, VEL] - before[:, VEL]) / AGENT_DT
    force_ned = 0.5 * (to_ned(r_before, before[:, FORCE]) + to_ned(r_after, after[:, FORCE]))
    error = np.linalg.norm(measured - (force_ned + STANDARD_GRAVITY * DOWN_AXIS), axis=1)
    if error.max() > FRAME_CHECK_TOLERANCE_MPS2:
        raise RuntimeError(
            f"R f + g differs from the measured acceleration by up to {error.max():.2f} m/s^2 at {tilt.min():.0f}+ deg "
            "tilt: the quaternion or the specific force is not in the assumed frame"
        )


def measure_sensor_model(mode: StepMode) -> SensorModel:
    """Motors off on the ground for SENSOR_CALIBRATION_S: each reading against the truth."""
    mode.reset()
    motors_off = np.zeros((mode.num_envs, mode.action_size), np.float32)
    trace = fly(mode, motors_off, SENSOR_CALIBRATION_S, STEPS_PER_ACTION, (*TRUTH_FIELDS, *SENSOR_FIELDS))
    trace = trace.astype(np.float64)
    truth, sensors = trace[..., : len(TRUTH_FIELDS)], trace[..., len(TRUTH_FIELDS) :]
    calls, n = trace.shape[:2]
    f = sensors[..., ACC]
    gravity = float(np.linalg.norm(f, axis=-1).mean())
    if abs(gravity / STANDARD_GRAVITY - 1.0) > GRAVITY_TOLERANCE:
        raise RuntimeError(f"at rest the accelerometer reads {gravity:.3f} m/s^2, not ~{STANDARD_GRAVITY}")
    rot = body_to_ned(truth[..., QUAT].reshape(-1, 4)).reshape(calls, n, 3, 3)
    field_ned = np.einsum("cnij,cnj->cni", rot, sensors[..., MAG]).mean(axis=0)
    if (np.linalg.norm(field_ned, axis=1) < NOISE_FLOOR).any():
        raise RuntimeError("the magnetometer reads no field")

    def spread(residual: np.ndarray) -> float:  # per axis, each env's own mean (tilt, bias) removed
        return float(np.maximum(NOISE_FLOOR, (residual - residual.mean(axis=0)).std(axis=(0, 1))).mean())

    baro_error = sensors[..., BARO] + truth[..., DOWN]  # baro_up against -down, both from the start
    model = SensorModel(
        gravity=gravity,
        accel_std=spread(f),
        gyro_std=spread(sensors[..., GYRO] - truth[..., RATES]),
        baro_std=float(max(NOISE_FLOOR, np.sqrt((baro_error**2).mean()))),
        field_ned=field_ned,
    )
    print(f"sensor model: {json.dumps(model.to_json())}")
    return model


def measure_startup(mode: StepMode, trained: Actuation | None = None) -> tuple[Actuation, SensorModel]:
    """The sensor model (on the ground, first), then the hover throttle and the mixer.

    With `trained`, its hover throttle is kept and the mixer measured now must equal its mixer.
    """
    sensor_model = measure_sensor_model(mode)
    if trained is None:
        return measure_actuation(mode, measure_hover_throttle(mode)), sensor_model
    measured = measure_actuation(mode, trained.hover_throttle)
    if not np.array_equal(measured.mixer, trained.mixer):
        raise RuntimeError(
            f"the motors pulse as {measured.mixer.tolist()}, the policy was trained on {trained.mixer.tolist()}"
        )
    return trained, sensor_model


# --- The environments ---


class StepModeVecEnv(VecEnv):  # type: ignore[misc]
    """Every step-mode env as one SB3 VecEnv env; a subclass sets the task (_begin, _observe_state, _score).

    The policy observes the StateEstimator's estimate; reward and termination use the truth. Same-step reset as SB3's
    DummyVecEnv: an env that ends returns its first observation of the next episode, with the last one in
    info["terminal_observation"], info["TimeLimit.truncated"] and info["end_reason"] (a termination or "truncated").
    With autoreset=False an ended env is not restarted: it stays in `ended` with its motors off until the next
    reset(). last_truth holds every env's truth before any restart.
    """

    def __init__(
        self,
        mode: StepMode,
        actuation: Actuation,
        seed: int,
        sensor_model: SensorModel,
        observation_size: int,
        terminations: Sequence[str],
        max_episode_steps: int,
        autoreset: bool,
    ) -> None:
        """Wrap mode's envs: the subclass's observation size, terminations and episode length."""
        self.mode = mode
        self.actuation = actuation
        n = mode.num_envs
        self._truth_cols = field_columns(mode.observation_fields, TRUTH_FIELDS)
        self._sensor_cols = field_columns(mode.observation_fields, SENSOR_FIELDS)
        self.estimator = StateEstimator(sensor_model, n, AGENT_DT)
        self.uwb = UwbPositioning(n, AGENT_DT)
        self.render_mode = None
        super().__init__(
            n,
            spaces.Box(-np.inf, np.inf, (observation_size,), np.float32),
            spaces.Box(-1.0, 1.0, (ACTION_SIZE,), np.float32),
        )
        self._rng = np.random.default_rng(seed)
        self.episode_ends = (*terminations, "truncated")
        self._max_episode_steps = max_episode_steps
        self.autoreset = autoreset
        self.ended = np.zeros(n, bool)
        self._prev_action = np.zeros((n, ACTION_SIZE), np.float32)
        self._t = np.zeros(n, np.int64)
        self._actions = np.zeros((n, ACTION_SIZE), np.float32)
        self.last_estimate_error: dict[str, np.ndarray] = {}
        self.last_truth = np.zeros((n, len(TRUTH_FIELDS)), np.float32)
        self._last_rows = np.zeros((n, mode.observation_size), np.float32)
        self.sim_wall_s = 0.0
        self._ends: Counter[str] = Counter()

    # The task: new episodes' state, the observation from a state, and each step's reward and terminations.
    def _begin(self, mask: np.ndarray, truth: np.ndarray) -> None:
        raise NotImplementedError

    def _observe_state(self, r: np.ndarray, pos: np.ndarray, vel: np.ndarray, rates: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _score(self, truth: np.ndarray, crashed: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Each env's reward and, per termination in order, which envs it ends."""
        raise NotImplementedError

    def _episode_ended(self, i: int, info: dict[str, Any]) -> None:
        """Bookkeeping when env i's episode ends, before any restart; may add to its info."""

    def seed(self, seed: int | None = None) -> list[int | None]:
        """Reseed the targets and the episode seeds."""
        self._rng = np.random.default_rng(seed)
        return [seed] * self.num_envs

    def _call(self, fn: Callable[[], StepResult]) -> tuple[np.ndarray, np.ndarray]:
        """One step-mode call: every env's observation row, and which envs crashed.

        A crashed env's row can be non-finite (a diverged model publishes what it holds): it keeps its last finite row,
        so the estimator and the reward stay finite through its terminal step. A non-finite row of an env that did not
        crash is the simulator's fault and fails.
        """
        t0 = time.perf_counter()
        result = fn()
        self.sim_wall_s += time.perf_counter() - t0
        crashed = np.asarray(result.crashed, bool)
        rows = np.array(result.observations, np.float32)
        bad = ~np.isfinite(rows).all(axis=1)
        if (bad & ~crashed).any():
            raise RuntimeError(
                f"non-finite observation in env(s) {np.flatnonzero(bad & ~crashed).tolist()}, not crashed"
            )
        rows[bad] = self._last_rows[bad]
        self._last_rows = rows
        return rows, crashed

    def _restart(self, mask: np.ndarray) -> np.ndarray:
        """Send the masked envs back to their start for a new episode; the truth of all envs."""
        seeds = self._rng.integers(0, INT32_SEED_LIMIT, self.num_envs)
        rows, crashed = self._call(lambda: self.mode.reset(mask=mask, seeds=seeds))
        if (mask & crashed).any():
            raise RuntimeError(f"env(s) {np.flatnonzero(mask & crashed).tolist()} crashed right after a reset")
        truth = rows[:, self._truth_cols]
        self._begin(mask, truth)
        self._prev_action[mask] = 0.0
        self._t[mask] = 0
        fix = self.uwb.reset(mask, seeds, truth[:, POS])
        self.estimator.reset(mask, rows[:, self._sensor_cols], fix)
        return truth

    def _observe(self) -> np.ndarray:
        e = self.estimator
        return self._observe_state(e.r, e.pos, e.vel, e.rates)

    def reset(self) -> np.ndarray:
        """Restart every env; their first observations."""
        self.ended[:] = False
        self.last_truth = self._restart(np.ones(self.num_envs, bool))
        return self._observe()

    def step_async(self, actions: np.ndarray) -> None:
        """Take the actions for the next step_wait()."""
        self._actions = np.clip(np.asarray(actions, np.float32), -1.0, 1.0)

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Step every env STEPS_PER_ACTION physics steps; observations, rewards, dones and infos."""
        actions = self._actions
        motors = self.actuation.motors(actions)
        motors[self.ended] = THROTTLE_MIN
        rows, crashed = self._call(lambda: self.mode.step(motors, steps=STEPS_PER_ACTION))
        truth = self.last_truth = rows[:, self._truth_cols]
        self.estimator.step(rows[:, self._sensor_cols], *self.uwb.measure(truth[:, POS]))
        self.last_estimate_error = self.estimator.errors(body_to_ned(truth[:, QUAT]), truth[:, POS], truth[:, VEL])
        self._prev_action = actions
        self._t += 1
        rewards, reasons = self._score(truth, crashed)
        terminated = np.logical_or.reduce(list(reasons.values()))
        truncated = (self._t >= self._max_episode_steps) & ~terminated
        dones = (terminated | truncated) & ~self.ended
        obs = self._observe()
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        if dones.any():
            for i in np.flatnonzero(dones):
                infos[i]["terminal_observation"] = obs[i].copy()
                infos[i]["TimeLimit.truncated"] = bool(truncated[i])
                infos[i]["end_reason"] = next((k for k, hit in reasons.items() if hit[i]), "truncated")
                self._ends[infos[i]["end_reason"]] += 1
                self._episode_ended(i, infos[i])
            if self.autoreset:
                self._restart(dones)
                obs[dones] = self._observe()[dones]
            else:
                self.ended |= dones
        return obs, rewards.astype(np.float32), dones, infos

    def pop_episode_ends(self) -> dict[str, int]:
        """How episodes ended since the last call, by reason (every reason present, zero or not)."""
        ends = {k: self._ends[k] for k in self.episode_ends}
        self._ends.clear()
        return ends

    def rollout_stats(self) -> dict[str, float]:
        """What to log after a rollout, accumulated since the last call."""
        return {f"episodes/{end}": count for end, count in self.pop_episode_ends().items()}

    def close(self) -> None:
        """Nothing to release: the fleet context owns step mode."""

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        """The attribute, once per env asked for (one batched env holds it once)."""
        return [getattr(self, attr_name)] * len(list(self._get_indices(indices)))

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        """Set the attribute for every env at once."""
        if len(list(self._get_indices(indices))) != self.num_envs:
            raise NotImplementedError(f"{type(self).__name__} sets attributes for all envs at once")
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args: Any, indices: Any = None, **method_kwargs: Any) -> list[Any]:
        """Not supported: there are no per-env objects."""
        raise NotImplementedError(f"{type(self).__name__} is one batched environment: there are no per-env methods")

    def env_is_wrapped(self, wrapper_class: type, indices: Any = None) -> list[bool]:
        """No env is wrapped."""
        return [False] * len(list(self._get_indices(indices)))
