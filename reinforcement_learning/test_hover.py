"""hover.py without a simulator: a fake StepMode of rigid X quads stands in for PteroSim.

python -m pytest test_hover.py -q
"""

from __future__ import annotations

import math
from typing import Any

import estimator
import hover
import numpy as np
import pytest
from estimator import SENSOR_FIELDS, SensorModel, StateEstimator, UwbPositioning
from fleet import PHYSICS_HZ
from frames import (
    DOWN,
    DOWN_AXIS,
    NORTH,
    STANDARD_GRAVITY,
    body_to_ned,
    gravity_body,
    tilt_deg,
    to_body,
    to_ned,
    wrap_angle,
)
from pterosim.types import ActuatorMapping, StepResult
from stable_baselines3.common.vec_env import VecMonitor
from step_env import (
    ACTION_SIZE,
    AGENT_DT,
    CHANNEL_PROBE_DELTA,
    CLIMB_THROTTLE_MARGIN,
    INT32_SEED_LIMIT,
    MOTORS,
    POS,
    QUAT,
    STEPS_PER_ACTION,
    TRUTH_FIELDS,
    VEL,
    Actuation,
    field_columns,
    measure_actuation,
    measure_hover_throttle,
    measure_sensor_model,
)

FAKE_DT = 1.0 / PHYSICS_HZ
FAKE_MAX_STEPS_PER_CALL = 100  # SimulationConstants::MAX_STEPS_PER_CALL
# F450 Mass.xml and Propulsion.xml, rounded; the fake's own hover throttle is chosen.
FAKE_MASS_KG = 1.4
FAKE_ARM_M = 0.1651
FAKE_INERTIA = np.array([0.019, 0.019, 0.0252])
FAKE_YAW_M = 0.016  # chosen: prop drag torque per newton of thrust
FAKE_HOVER = 0.43
FAKE_GROUND_DOWN_M = 0.0  # every env starts on the floor, as an SDK spawn does
# p, q, r sign per channel: the fake's torque law below, and what the live F450 pulses give.
FAKE_MIXER = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1], [-1, -1, -1]])
# The fake's sensors take the simulator's noise model, held between updates like the simulator's.
FAKE_ACCEL_STD = 7.7e-3  # m/s^2, white
FAKE_GYRO_STD = 5.5e-4  # rad/s, white
FAKE_MAG_STD = 0.3  # uT, white
FAKE_BARO_STD = 0.11  # m, white
FAKE_FIELD_NED = np.array([21.0, 2.0, 43.0])  # uT, a mid-latitude field with ~5 deg declination
# The simulator's sensor block, in its order; the GPS fields ride along unread (truth, no noise).
FAKE_SENSOR_FIELDS = (
    *SENSOR_FIELDS[:9],
    "gps_north_m",
    "gps_east_m",
    "gps_down_m",
    "gps_vn_mps",
    "gps_ve_mps",
    "gps_vd_mps",
    SENSOR_FIELDS[9],
)
FAKE_NOISE_SEED = 11
LPS_SIGMA = estimator.LPS_POSITION_SIGMA_M  # read before any monkeypatch
FAKE_START_YAW_DEG = 70.0  # a start heading away from north, for the TRIAD start
SEED = 3
# Scripted test policy, chosen: climb to the target height and turn to the target heading, no horizontal control.
SCRIPT_CLIMB_GAIN = 0.3  # collective per metre of height error
SCRIPT_CLIMB_DAMPING = 0.3  # collective per m/s of vertical speed
SCRIPT_YAW_GAIN = 0.3  # yaw command per unit sin(heading error)
SCRIPT_YAW_DAMPING = 0.5  # yaw command per unit scaled yaw rate
# Observation columns (hover.observation): error_b 0-2, v_b 3-5, rates/4 6-8, gravity_b 9-11, sin/cos 12-13
OBS_ERROR, OBS_GRAVITY = slice(0, 3), slice(9, 12)
OBS_ERROR_DOWN, OBS_VEL_DOWN, OBS_YAW_RATE, OBS_SIN_HEADING, OBS_COS_HEADING = 2, 5, 8, 12, 13
ESTIMATE_TOLERANCE_M = 4 * LPS_SIGMA  # chosen: the first estimate is one UWB fix and one baro reading
ESTIMATE_TOLERANCE_RAD = 0.05  # chosen: TRIAD from one noisy reading errs under a degree


class FakeQuadMode:
    """N rigid X quads (PX4 order: front right, aft left, front left, aft right) resting on a flat floor, in vacuum.

    Validates requests like StepMode and records every reset mask. Reports the truth and (emit_sensors) a sensor
    block with white noise, held between updates. Test hooks: crash_next, dead_channel, conjugate_quaternion
    (publish the attitude transposed), hover (throttle that balances weight), one_spin_direction (all props turn
    the same way: no yaw pairs), start_yaw_deg.
    """

    def __init__(
        self,
        n: int,
        *,
        hover: float = FAKE_HOVER,
        dead_channel: int | None = None,
        conjugate_quaternion: bool = False,
        one_spin_direction: bool = False,
        emit_sensors: bool = True,
        start_yaw_deg: float = 0.0,
    ) -> None:
        """N quads at rest on the floor."""
        self.num_envs = n
        self.action_size = MOTORS
        self.observation_fields = TRUTH_FIELDS + (FAKE_SENSOR_FIELDS if emit_sensors else ())
        self.observation_size = len(self.observation_fields)
        self.action_channels = tuple(
            ActuatorMapping(channel=i, type="motor", component_index=i, input_min=0.0, input_max=1.0, name=f"motor{i}")
            for i in range(MOTORS)
        )
        self.dt = FAKE_DT
        self.max_steps_per_call = FAKE_MAX_STEPS_PER_CALL
        self.thrust_max_n = FAKE_MASS_KG * STANDARD_GRAVITY / (MOTORS * hover)
        self.dead_channel = dead_channel
        self.conjugate_quaternion = conjugate_quaternion
        self.emit_sensors = emit_sensors
        half_yaw = math.radians(start_yaw_deg) / 2.0
        self.start_quat = np.array([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)])
        self.yaw_signs = np.ones(MOTORS) if one_spin_direction else np.array([1.0, 1.0, -1.0, -1.0])
        self.crash_next = np.zeros(n, bool)
        self.reset_masks: list[np.ndarray] = []
        self.step_count = 0
        self._was_reset = np.zeros(n, bool)
        self._pos = np.zeros((n, 3))
        self._vel = np.zeros((n, 3))
        self._quat = np.tile(self.start_quat, (n, 1))
        self._rates = np.zeros((n, 3))
        self._force = np.tile([0.0, 0.0, -STANDARD_GRAVITY], (n, 1))
        self._noise = np.random.default_rng(FAKE_NOISE_SEED)
        self._sensors = np.zeros((n, len(FAKE_SENSOR_FIELDS)))

    def step(self, actions: Any, steps: int = 1, reset: Any = None, seeds: Any = None) -> StepResult:
        """Validate like StepMode.step, restart the reset envs, then integrate `steps` physics steps."""
        actions = np.asarray(actions, np.float32)
        if actions.shape != (self.num_envs, self.action_size):
            raise ValueError(f"actions shape {actions.shape}")
        if not 0 <= steps <= self.max_steps_per_call:
            raise ValueError(f"steps {steps}")
        mask = np.zeros(self.num_envs, bool) if reset is None else np.asarray(reset, bool)
        if seeds is not None:
            seeds = list(seeds)
            if len(seeds) != self.num_envs or not all(0 <= s < INT32_SEED_LIMIT for s in seeds):
                raise ValueError(f"seeds {seeds}")
        live = ~mask
        if ((actions < 0.0) | (actions > 1.0))[live].any():
            raise ValueError("action out of range")
        if steps and (live & ~self._was_reset).any():
            raise ValueError("env never reset")
        if mask.any():
            self.reset_masks.append(mask.copy())
            self._restart(mask)
        for _ in range(steps):
            self._physics(actions.astype(np.float64), live)
        self.step_count += steps
        crashed, self.crash_next = self.crash_next, np.zeros(self.num_envs, bool)
        return StepResult(
            observations=self._observations(),
            crashed=crashed,
            step_count=self.step_count,
            sim_time=self.step_count * self.dt,
        )

    def reset(self, mask: Any = None, seeds: Any = None) -> StepResult:
        """Restart the masked envs (all by default)."""
        mask = np.ones(self.num_envs, bool) if mask is None else mask
        return self.step(np.zeros((self.num_envs, self.action_size), np.float32), steps=0, reset=mask, seeds=seeds)

    def _restart(self, mask: np.ndarray) -> None:
        self._was_reset |= mask
        self._pos[mask] = 0.0
        self._vel[mask] = 0.0
        self._quat[mask] = self.start_quat
        self._rates[mask] = 0.0
        self._force[mask] = [0.0, 0.0, -STANDARD_GRAVITY]
        self._sense(mask)

    def _physics(self, motors: np.ndarray, live: np.ndarray) -> None:
        if self.dead_channel is not None:
            motors = motors.copy()
            motors[:, self.dead_channel] = 0.0
        t0, t1, t2, t3 = (self.thrust_max_n * motors).T
        torque = np.stack(
            [
                FAKE_ARM_M * (-t0 + t1 + t2 - t3),
                FAKE_ARM_M * (t0 - t1 + t2 - t3),
                FAKE_YAW_M * (self.thrust_max_n * motors) @ self.yaw_signs,
            ],
            1,
        )
        airborne = live & (self._pos[:, 2] < FAKE_GROUND_DOWN_M)  # the gear holds a grounded quad level
        self._rates[airborne] += (torque / FAKE_INERTIA * self.dt)[airborne]
        w, x, y, z = self._quat.T
        p, q, r = self._rates.T
        q_dot = 0.5 * np.stack(
            [-x * p - y * q - z * r, w * p + y * r - z * q, w * q + z * p - x * r, w * r + x * q - y * p], 1
        )
        self._quat[airborne] += (q_dot * self.dt)[airborne]
        self._quat /= np.linalg.norm(self._quat, axis=1, keepdims=True)
        force = np.zeros_like(self._vel)
        force[:, 2] = -(t0 + t1 + t2 + t3) / FAKE_MASS_KG
        rot = body_to_ned(self._quat)
        accel = to_ned(rot, force) + STANDARD_GRAVITY * DOWN_AXIS
        self._vel[live] += (accel * self.dt)[live]
        self._pos[live] += (self._vel * self.dt)[live]
        self._force[live] = force[live]
        grounded = self._pos[:, 2] >= FAKE_GROUND_DOWN_M
        self._pos[grounded, 2] = FAKE_GROUND_DOWN_M
        self._vel[grounded] = 0.0
        self._rates[grounded] = 0.0
        self._force[grounded] = -STANDARD_GRAVITY * gravity_body(rot)[grounded]
        self._sense(live)

    def _sense(self, envs: np.ndarray) -> None:
        """Every sensor every physics step, with white noise (GPS without)."""
        n = self.num_envs
        rot = body_to_ned(self._quat)
        fresh = np.concatenate(
            [
                self._force + self._noise.normal(0.0, FAKE_ACCEL_STD, (n, 3)),
                self._rates + self._noise.normal(0.0, FAKE_GYRO_STD, (n, 3)),
                to_body(rot, np.tile(FAKE_FIELD_NED, (n, 1))) + self._noise.normal(0.0, FAKE_MAG_STD, (n, 3)),
                self._pos,
                self._vel,
                (-self._pos[:, 2] + self._noise.normal(0.0, FAKE_BARO_STD, n))[:, None],
            ],
            axis=1,
        )
        self._sensors[envs] = fresh[envs]

    def _observations(self) -> np.ndarray:
        quat = self._quat * ([1, -1, -1, -1] if self.conjugate_quaternion else 1)
        truth = [self._pos, quat, self._vel, self._rates, self._force]
        return np.concatenate(truth + ([self._sensors] if self.emit_sensors else []), 1).astype(np.float32)


def fake_sensor_model(n: int) -> SensorModel:
    """The fake's own noise and field, as a measurement at rest would find them."""
    return SensorModel(STANDARD_GRAVITY, FAKE_ACCEL_STD, FAKE_GYRO_STD, FAKE_BARO_STD, np.tile(FAKE_FIELD_NED, (n, 1)))


def fake_env(n: int = 6, seed: int = SEED, mode: FakeQuadMode | None = None) -> tuple[FakeQuadMode, hover.HoverVecEnv]:
    """A hover env on a fake fleet."""
    mode = mode or FakeQuadMode(n)
    return mode, hover.HoverVecEnv(mode, Actuation(FAKE_HOVER, FAKE_MIXER), seed, fake_sensor_model(mode.num_envs))


def hover_actions(env: hover.HoverVecEnv) -> np.ndarray:
    """Hover throttle on every motor."""
    return np.zeros((env.num_envs, ACTION_SIZE), np.float32)


def climb_and_turn(obs: np.ndarray) -> np.ndarray:
    """Scripted: climb to each target's height and turn to its heading; no horizontal control."""
    actions = np.zeros((len(obs), ACTION_SIZE), np.float32)
    actions[:, 0] = np.clip(
        -SCRIPT_CLIMB_GAIN * obs[:, OBS_ERROR_DOWN] + SCRIPT_CLIMB_DAMPING * obs[:, OBS_VEL_DOWN], -1.0, 1.0
    )
    actions[:, 3] = np.clip(
        SCRIPT_YAW_GAIN * obs[:, OBS_SIN_HEADING] - SCRIPT_YAW_DAMPING * obs[:, OBS_YAW_RATE], -1.0, 1.0
    )
    return actions


def test_startup_measures_lift_off_and_checks_channels_in_the_air() -> None:
    mode = FakeQuadMode(3)
    hover_throttle = measure_hover_throttle(mode)
    assert abs(hover_throttle - FAKE_HOVER) < 0.005, hover_throttle
    resets = len(mode.reset_masks)
    actuation = measure_actuation(mode, hover_throttle)
    assert len(mode.reset_masks) == resets + MOTORS  # one climb from the ground per channel
    np.testing.assert_array_equal(actuation.mixer, FAKE_MIXER)
    assert actuation.hover_throttle == hover_throttle


def test_hover_outside_the_sweep_fails_loud() -> None:
    with pytest.raises(RuntimeError, match="outside"):
        measure_hover_throttle(FakeQuadMode(3, hover=0.7))


def test_a_dead_channel_fails_loud() -> None:
    with pytest.raises(RuntimeError, match="not balanced"):
        measure_hover_throttle(FakeQuadMode(3, dead_channel=2))  # the top throttle lifts on three motors and tips
    with pytest.raises(RuntimeError, match="lifted the envs only"):
        measure_actuation(FakeQuadMode(3, dead_channel=2), FAKE_HOVER)


def test_a_transposed_attitude_fails_the_frame_check() -> None:
    with pytest.raises(RuntimeError, match="assumed frame"):
        measure_actuation(FakeQuadMode(3, conjugate_quaternion=True), FAKE_HOVER)


def test_props_turning_one_way_are_not_an_x_quad() -> None:
    with pytest.raises(RuntimeError, match="not an X quad"):
        measure_actuation(FakeQuadMode(3, one_spin_direction=True), FAKE_HOVER)


def test_every_motor_reaches_zero_and_full_throttle() -> None:
    act = Actuation(FAKE_HOVER, FAKE_MIXER)
    commands = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],  # hover
            [1.0, 0.0, 0.0, 0.0],  # full collective
            [-1.0, 0.0, 0.0, 0.0],  # no collective
            [0.0, 1.0, 0.0, 0.0],  # full roll: the motors with p sign +1 full, the others off
            [0.5, 0.0, 0.0, 0.0],  # half way from hover to full
            [1.0, 1.0, 0.0, 0.0],  # saturates: the roll-up motors stay full, the others drop to hover
        ],
        np.float32,
    )
    up = 1.0 - FAKE_HOVER
    roll_up = FAKE_MIXER[:, 0] > 0
    expected = np.array(
        [
            [FAKE_HOVER] * 4,
            [1.0] * 4,
            [0.0] * 4,
            np.where(roll_up, 1.0, 0.0),
            [FAKE_HOVER + 0.5 * up] * 4,
            np.where(roll_up, 1.0, FAKE_HOVER),
        ]
    )
    np.testing.assert_allclose(act.motors(commands), expected, atol=1e-6)


def test_reset_draws_seeded_targets_straight_above_the_starts() -> None:
    _, env = fake_env()
    obs = env.reset()
    assert obs.shape == (env.num_envs, hover.OBSERVATION_SIZE) and obs.dtype == np.float32
    np.testing.assert_array_equal(env.targets[:, :DOWN], 0.0)
    height = -env.targets[:, DOWN]
    assert ((height >= hover.TARGET_HEIGHT_MIN_M) & (height <= hover.TARGET_HEIGHT_MAX_M)).all()
    assert (np.abs(env.target_headings) <= math.pi).all()
    assert len(set(height)) == env.num_envs
    np.testing.assert_allclose(obs[:, OBS_ERROR], env.targets, atol=ESTIMATE_TOLERANCE_M)  # level: error_b ~ target
    np.testing.assert_allclose(obs[:, OBS_GRAVITY], np.tile(DOWN_AXIS, (env.num_envs, 1)), atol=ESTIMATE_TOLERANCE_RAD)
    _, again = fake_env()
    np.testing.assert_array_equal(again.reset(), obs)


def test_hovering_truncates_every_env_at_the_episode_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hover, "TAKEOFF_WINDOW_STEPS", hover.MAX_EPISODE_STEPS + 1)  # truncation, not the grounded rule
    mode, env = fake_env()
    env.reset()
    for _ in range(hover.MAX_EPISODE_STEPS - 1):
        _, _, dones, infos = env.step(hover_actions(env))
        assert not dones.any() and infos == [{}] * env.num_envs
    resets_before = len(mode.reset_masks)
    _, _, dones, infos = env.step(hover_actions(env))
    assert dones.all()
    assert len(mode.reset_masks) == resets_before + 1 and mode.reset_masks[-1].all()
    for info in infos:
        assert info["TimeLimit.truncated"] is True
        assert info["terminal_observation"].shape == (hover.OBSERVATION_SIZE,)
    assert (env._t == 0).all()
    assert env.pop_episode_ends()["truncated"] == env.num_envs


def test_a_climbing_env_terminates_alone_and_restarts() -> None:
    mode, env = fake_env()
    env.reset()
    actions = hover_actions(env)
    actions[0, 0] = 1.0  # full collective
    steps, dones = 0, np.zeros(env.num_envs, bool)
    while not dones.any() and steps < hover.MAX_EPISODE_STEPS - 1:  # before the time limit
        obs, _, dones, infos = env.step(actions)
        steps += 1
    assert dones.tolist() == [True] + [False] * (env.num_envs - 1), steps
    assert infos[0]["TimeLimit.truncated"] is False
    assert infos[0]["terminal_observation"][-ACTION_SIZE:].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert mode.reset_masks[-1].tolist() == dones.tolist()
    assert obs[0, -ACTION_SIZE:].tolist() == [0.0] * ACTION_SIZE
    assert env._t[0] == 0 and (env._t[1:] == steps).all()
    assert env.pop_episode_ends()["ceiling"] == 1


def test_each_termination_reason() -> None:
    rows = np.zeros((7, len(TRUTH_FIELDS)), np.float32)
    rows[:, QUAT] = [1.0, 0.0, 0.0, 0.0]
    t = np.zeros(7, np.int64)
    t[6] = hover.TAKEOFF_WINDOW_STEPS  # still on the ground when the take-off window closes
    rows[1, DOWN] = hover.BELOW_START_MARGIN_M + 0.1
    rows[2, DOWN] = -hover.CEILING_ABOVE_START_M - 0.1
    rows[3, NORTH] = hover.MAX_HORIZONTAL_M + 0.1
    rows[4, QUAT] = [0.0, 1.0, 0.0, 0.0]  # upside down
    crashed = np.array([False, False, False, False, False, True, False])
    reasons = hover.termination_reasons(rows, crashed, t)
    hit = {k: np.flatnonzero(v).tolist() for k, v in reasons.items()}
    assert hit == {"crashed": [5], "grounded": [6], "below_start": [1], "ceiling": [2], "horizontal": [3], "tilt": [4]}


def test_the_crashed_flag_terminates() -> None:
    mode, env = fake_env()
    env.reset()
    mode.crash_next[1] = True
    _, _, dones, infos = env.step(hover_actions(env))
    assert dones.tolist() == [False, True] + [False] * (env.num_envs - 2)
    assert infos[1]["TimeLimit.truncated"] is False
    assert env.pop_episode_ends()["crashed"] == 1


def test_reward_peaks_at_the_target_and_actions_are_clipped() -> None:
    _, env = fake_env()
    env.reset()
    env.targets[:] = 0.0
    env.target_headings[:] = 0.0
    _, rewards, _, _ = env.step(hover_actions(env))  # hover at the start, facing north
    np.testing.assert_allclose(
        rewards, (hover.DISTANCE_REWARD_SCALE + hover.HEADING_REWARD_SCALE) * AGENT_DT, rtol=1e-4
    )
    env.step(np.full((env.num_envs, ACTION_SIZE), 5.0, np.float32))  # clipped to 1: the fake refuses > 1


def test_the_heading_term_and_its_observation() -> None:
    _, env = fake_env(mode=FakeQuadMode(6, start_yaw_deg=FAKE_START_YAW_DEG))
    env.reset()
    env.targets[:] = 0.0
    env.target_headings[:] = np.radians([0.0, 90.0, -90.0, 180.0, 45.0, FAKE_START_YAW_DEG])
    obs, rewards, _, _ = env.step(hover_actions(env))
    error = wrap_angle(env.target_headings - math.radians(FAKE_START_YAW_DEG))
    expected = (
        hover.DISTANCE_REWARD_SCALE
        + hover.HEADING_REWARD_SCALE * (1.0 - np.tanh(np.abs(error) / hover.HEADING_TANH_SCALE_RAD))
    ) * AGENT_DT
    np.testing.assert_allclose(rewards, expected, rtol=1e-4)
    np.testing.assert_allclose(env.last_heading_error_rad, error, atol=1e-5)
    np.testing.assert_allclose(obs[:, OBS_SIN_HEADING], np.sin(error), atol=ESTIMATE_TOLERANCE_RAD)
    np.testing.assert_allclose(obs[:, OBS_COS_HEADING], np.cos(error), atol=ESTIMATE_TOLERANCE_RAD)
    assert rewards[5] == rewards.max()  # facing its target heading


def test_zero_actions_stay_on_the_ground_and_fail() -> None:
    _, env = fake_env(n=8)
    result = hover.evaluate(env, lambda obs: np.zeros((len(obs), ACTION_SIZE), np.float32))
    assert result["survival"] == 0.0  # every env still on the ground when the take-off window closes
    assert result["rms_median_m"] > hover.RMS_MEDIAN_MAX_M and not result["passed"]
    assert {end for end, count in result["episode_ends"].items() if count} == {"grounded"}  # restarted, again


def test_a_vertical_climb_reports_the_take_off() -> None:
    _, env = fake_env(n=8, mode=FakeQuadMode(8, start_yaw_deg=FAKE_START_YAW_DEG))
    result = hover.evaluate(env, climb_and_turn)
    assert result["survival"] == 1.0
    assert result["heading_error_median_deg"] < hover.HEADING_ERROR_MEDIAN_MAX_DEG
    assert result["yaw_rate_median_degps"] < hover.YAW_RATE_MEDIAN_MAX_DEGPS
    assert result["climb_drift_m"][1] < 0.01  # level the whole climb: nothing pushes it sideways
    assert result["overshoot_m"] is not None and result["arrive_s"] is not None
    assert result["estimator_error"]["attitude_deg"][0] < 1.0 and result["estimator_error"]["position_m"][0] < LPS_SIGMA


def test_missing_sensor_fields_fail_loud() -> None:
    with pytest.raises(RuntimeError, match="imu_ax_mps2"):
        fake_env(mode=FakeQuadMode(2, emit_sensors=False))


def test_the_sensor_model_is_measured_at_rest() -> None:
    model = measure_sensor_model(FakeQuadMode(4, start_yaw_deg=FAKE_START_YAW_DEG))
    assert abs(model.gravity - STANDARD_GRAVITY) < FAKE_ACCEL_STD
    np.testing.assert_allclose(model.accel_std, FAKE_ACCEL_STD, rtol=0.1)
    np.testing.assert_allclose(model.gyro_std, FAKE_GYRO_STD, rtol=0.1)
    np.testing.assert_allclose(model.baro_std, FAKE_BARO_STD, rtol=0.1)
    np.testing.assert_allclose(model.field_ned, np.tile(FAKE_FIELD_NED, (4, 1)), atol=5 * FAKE_MAG_STD)


def run_estimator(
    mode: FakeQuadMode, phases: list[tuple[np.ndarray, float]]
) -> tuple[list[dict[str, np.ndarray]], list[np.ndarray]]:
    """Reset every env, then hold each (motors, seconds) phase; estimator errors and truth after every step."""
    est = StateEstimator(measure_sensor_model(mode), mode.num_envs, AGENT_DT)
    truth_cols = field_columns(mode.observation_fields, TRUTH_FIELDS)
    sensor_cols = field_columns(mode.observation_fields, SENSOR_FIELDS)
    uwb = UwbPositioning(mode.num_envs, AGENT_DT)
    every = np.ones(mode.num_envs, bool)

    def errors(truth: np.ndarray) -> dict[str, np.ndarray]:
        return est.errors(body_to_ned(truth[:, QUAT]), truth[:, POS], truth[:, VEL])

    raw = mode.reset().observations
    est.reset(every, raw[:, sensor_cols], uwb.reset(every, np.arange(mode.num_envs), raw[:, truth_cols][:, POS]))
    found, truths = [errors(raw[:, truth_cols])], [raw[:, truth_cols]]
    for motors, seconds in phases:
        for _ in range(round(seconds / AGENT_DT)):
            raw = mode.step(motors, steps=STEPS_PER_ACTION).observations
            est.step(raw[:, sensor_cols], *uwb.measure(raw[:, truth_cols][:, POS]))
            found.append(errors(raw[:, truth_cols]))
            truths.append(raw[:, truth_cols])
    return found, truths


def test_the_estimator_starts_from_a_yawed_rest_and_converges() -> None:
    mode = FakeQuadMode(4, start_yaw_deg=FAKE_START_YAW_DEG)
    errors, _ = run_estimator(mode, [(np.zeros((4, MOTORS), np.float32), 2 / estimator.ATT_W_MAG)])  # 2 time constants
    assert errors[0]["attitude_deg"].max() < 2.0  # TRIAD from one noisy reading
    assert errors[-1]["attitude_deg"].max() < 1.0
    assert errors[-1]["position_m"].max() < 0.1  # UWB 0.10 m per fix, filtered
    assert errors[-1]["velocity_mps"].max() < 0.1


def test_uwb_fixes_are_seeded_noisy_and_taken_at_the_ticks(monkeypatch: pytest.MonkeyPatch) -> None:
    n, steps = 50, 200
    seeds = np.arange(n) + SEED
    speed = np.array([1.0, -2.0, 0.5])  # m/s, a straight line from the start

    def fly(uwb: UwbPositioning) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        first = uwb.reset(np.ones(n, bool), seeds, np.zeros((n, 3)))
        return first, [uwb.measure(np.tile(speed * k * AGENT_DT, (n, 1))) for k in range(1, steps + 1)]

    first, rest = fly(UwbPositioning(n, AGENT_DT))
    first_again, rest_again = fly(UwbPositioning(n, AGENT_DT))
    np.testing.assert_array_equal(first, first_again)  # the episode seeds decide the noise
    np.testing.assert_array_equal(np.stack([f for f, _ in rest]), np.stack([f for f, _ in rest_again]))
    assert all(new.all() for _, new in rest)  # 50 Hz ticks outpace the 48 Hz agent: a new fix every step
    monkeypatch.setattr(estimator, "LPS_POSITION_SIGMA_M", 0.0)
    _, exact = fly(UwbPositioning(n, AGENT_DT))
    noise = np.stack([f for f, _ in rest]) - np.stack([f for f, _ in exact])
    np.testing.assert_allclose(noise.std(), LPS_SIGMA, rtol=0.05)
    for k, (fix, _) in enumerate(exact, start=1):
        tick = math.floor(k * AGENT_DT * estimator.LPS_RATE_HZ + estimator.TICK_ROUNDING) / estimator.LPS_RATE_HZ
        np.testing.assert_allclose(fix, np.tile(speed * tick, (n, 1)), atol=1e-9)  # the truth at the last tick


def test_the_estimator_tracks_a_climb_and_a_tilt() -> None:
    n = 4
    mode = FakeQuadMode(n, start_yaw_deg=FAKE_START_YAW_DEG)
    climb = np.full((n, MOTORS), FAKE_HOVER + 2 * CLIMB_THROTTLE_MARGIN, np.float32)
    push = np.full((n, MOTORS), FAKE_HOVER, np.float32)
    push[:, 0] += CHANNEL_PROBE_DELTA
    rest_s = 1.0
    errors, truths = run_estimator(mode, [(np.zeros((n, MOTORS), np.float32), rest_s), (climb, 1.5), (push, 0.3)])
    tilt = tilt_deg(body_to_ned(truths[-1][:, QUAT]))
    assert tilt.min() > 10.0 and (-truths[-1][:, DOWN]).min() > 0.5  # it did climb and tilt
    assert max(e["attitude_deg"].max() for e in errors) < 2.5  # the start's TRIAD error, then no worse
    flown = errors[round(rest_s / AGENT_DT) :]  # past the first fixes' noise, averaged on the ground
    assert max(e["position_m"].max() for e in flown) < LPS_SIGMA
    assert max(e["velocity_mps"].max() for e in flown) < 0.2


def test_rollout_steps_follow_the_table() -> None:
    assert [hover.rollout_steps(n) for n in (18, 100, 1000)] == [512, 128, 32]


def test_ppo_trains_two_updates_on_the_fake() -> None:
    mode = FakeQuadMode(hover.SAMPLES_PER_UPDATE // hover.MIN_ROLLOUT_STEPS)  # the shortest real rollout
    _, env = fake_env(mode=mode)
    model = hover.make_ppo(VecMonitor(env), env.num_envs, SEED)
    assert model.n_steps == hover.MIN_ROLLOUT_STEPS
    np.testing.assert_allclose(model.policy.log_std.exp().detach().numpy(), hover.INITIAL_ACTION_STD, rtol=1e-6)
    model.learn(2 * hover.SAMPLES_PER_UPDATE, callback=hover.StepRateCallback(env))
    assert model.num_timesteps == 2 * hover.SAMPLES_PER_UPDATE
    assert env.sim_wall_s > 0.0
