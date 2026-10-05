"""What a real autopilot knows of its state: attitude, position and velocity estimated from sensors.

The simulator's IMU, magnetometer and barometer arrive in step mode's sensor block. Horizontal position comes from a
client-side UWB local positioning model (UwbPositioning: the truth plus noise, not a simulator sensor); GPS is not used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from frames import (
    DOWN,
    DOWN_AXIS,
    EAST,
    NORTH,
    STANDARD_GRAVITY,
    orthonormalize,
    rotation_from_vector,
    to_body,
    to_ned,
    triad,
    wrap_angle,
)

# The sensor block the estimator reads, in this order (IMU every physics step, baro 50 Hz, held between updates).
SENSOR_FIELDS = (
    "imu_ax_mps2",
    "imu_ay_mps2",
    "imu_az_mps2",
    "imu_p_radps",
    "imu_q_radps",
    "imu_r_radps",
    "mag_x_ut",
    "mag_y_ut",
    "mag_z_ut",
    "baro_up_m",
)
ACC, GYRO, MAG = slice(0, 3), slice(3, 6), slice(6, 9)
BARO = 9

# Attitude: PX4's attitude_estimator_q, a Mahony complementary filter; gains are its parameter defaults
# (PX4-Autopilot src/modules/attitude_estimator_q/attitude_estimator_q_params.yaml).
ATT_W_ACC = 0.2  # ATT_W_ACC, rad/s per unit tilt error: the accelerometer pulls tilt in with a 5 s time constant
ATT_W_MAG = 0.1  # ATT_W_MAG, yaw only: 10 s
ATT_W_GYRO_BIAS = 0.1  # ATT_W_GYRO_BIAS, integral gain of the gyro bias
ATT_BIAS_MAX_RADPS = 0.05  # ATT_BIAS_MAX
BIAS_LEARN_MAX_RATE_RADPS = 0.175  # attitude_estimator_q's source learns the bias only below this spin rate
# Chosen: the accelerometer corrects tilt only while |f| is within 10% of g. A multirotor's accelerometer reads
# thrust, so this gate and the 5 s time constant keep manoeuvres from dragging the tilt estimate.
ACC_GATE = 0.1

# Position: a UWB local positioning system, a CLIENT-SIDE measurement model: the truth NED position plus white noise
# on every axis, sampled at LPS_RATE_HZ and held. Decawave DW1000 datasheet: +-30 cm in x and y, taken as 3 sigma.
LPS_POSITION_SIGMA_M = 0.10
LPS_RATE_HZ = 50.0  # chosen: a typical UWB TDOA/TWR positioning rate
TICK_ROUNDING = 1e-9  # float rounding when a fix's time falls exactly on an LPS tick
# Process noise of the position filter: the accelerometer's, plus g x 1 deg (chosen: the tilt error the attitude
# filter leaves, which rotates gravity into the predicted acceleration).
PREDICTION_ACCEL_FLOOR_MPS2 = STANDARD_GRAVITY * math.radians(1.0)
# The filter's trust in a UWB fix: horizontal as the measurement model. Vertical, chosen ten times less: UWB height is
# geometry-limited, so the baro (~0.1 m, measured) dominates height.
LPS_VERTICAL_DISTRUST = 10.0
LPS_FILTER_SIGMA_M = np.array(
    [LPS_POSITION_SIGMA_M, LPS_POSITION_SIGMA_M, LPS_VERTICAL_DISTRUST * LPS_POSITION_SIGMA_M]
)
RESET_VELOCITY_STD_MPS = 0.1  # chosen: every reset puts the aircraft at rest on the ground; UWB has no velocity


@dataclass(frozen=True)
class SensorModel:
    """The IMU's and baro's noise and the magnetic reference, measured at rest on the ground.

    What a datasheet and a world magnetic model lookup give a real autopilot.
    """

    gravity: float  # m/s^2, mean |f| at rest
    accel_std: float  # m/s^2, per axis
    gyro_std: float  # rad/s, per axis
    baro_std: float  # m
    field_ned: np.ndarray  # (N, 3) uT, each env's magnetic field in NED

    def to_json(self) -> dict[str, Any]:
        """The model for run.json (the field averaged over the envs)."""
        return {
            "gravity": self.gravity,
            "accel_std": self.accel_std,
            "gyro_std": self.gyro_std,
            "baro_std": self.baro_std,
            "field_ned_mean": self.field_ned.mean(axis=0).tolist(),
        }


class UwbPositioning:
    """UWB fixes made on the client from the truth: a measurement model, NOT a simulator sensor.

    NED position plus white noise (LPS_POSITION_SIGMA_M on every axis), sampled at LPS_RATE_HZ and held. A fix is
    taken at each LPS tick: the truth interpolated back to it between the last two agent steps. Each env's noise
    comes from its own generator, seeded with the episode's seed at each reset.
    """

    def __init__(self, n: int, agent_dt: float) -> None:
        """Positioning for n envs stepped every agent_dt seconds."""
        self._agent_dt = agent_dt
        self._rngs = [np.random.default_rng(i) for i in range(n)]  # replaced at each env's first reset
        self._prev = np.zeros((n, 3))
        self._fix = np.zeros((n, 3))
        self._t = np.zeros(n)

    def reset(self, mask: np.ndarray, seeds: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Restart the masked envs at time 0 (an LPS tick) with noise from their episode seeds; every env's fix."""
        for i in np.flatnonzero(mask):
            self._rngs[i] = np.random.default_rng(int(seeds[i]))
        fresh = pos + self._noise(mask)
        self._fix[mask] = fresh[mask]
        self._prev[mask] = pos[mask]
        self._t[mask] = 0.0
        return self._fix.copy()

    def measure(self, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Every env's fix after one agent step, given its truth position now, and which fixes are new."""
        t = self._t + self._agent_dt
        tick = np.floor(t * LPS_RATE_HZ + TICK_ROUNDING) / LPS_RATE_HZ
        new = tick > self._t
        fraction = ((tick - self._t) / self._agent_dt)[:, None]
        fresh = self._prev + fraction * (pos - self._prev) + self._noise(new)
        self._fix[new] = fresh[new]
        self._prev, self._t = pos.copy(), t
        return self._fix.copy(), new

    def _noise(self, envs: np.ndarray) -> np.ndarray:
        noise = np.zeros((len(self._rngs), 3))
        for i in np.flatnonzero(envs):
            noise[i] = self._rngs[i].normal(0.0, LPS_POSITION_SIGMA_M, 3)
        return noise

    @staticmethod
    def to_json() -> dict[str, Any]:
        """The model for run.json."""
        return {
            "kind": "lps-uwb",
            "model": "client-side measurement model (truth position + white noise), not a simulator sensor",
            "sigma_m": LPS_POSITION_SIGMA_M,
            "rate_hz": LPS_RATE_HZ,
            "filter_sigma_m": LPS_FILTER_SIGMA_M.tolist(),
        }


class StateEstimator:
    """Per-env attitude, position and velocity from the IMU, magnetometer, baro and UWB fixes, at the agent rate.

    Attitude: PX4 attitude_estimator_q's Mahony filter on a rotation matrix. The gyro (trapezoid of the last two
    readings, plus the learned bias) propagates it; the accelerometer, while |f| ~ g, pulls the estimated down toward
    the measured one; the magnetometer corrects yaw only, by the heading of the measured field in NED against each
    env's declination; the bias integrates the corrections while the aircraft barely turns.
    Position and velocity: per axis a Kalman filter [p, v], predicted with R f + g, updated by each new UWB fix and
    each new baro reading (height). reset() starts an env from its readings at rest: TRIAD attitude from gravity and
    the field, position from the UWB fix and then the baro, velocity 0.
    """

    def __init__(self, model: SensorModel, n: int, agent_dt: float) -> None:
        """An estimator for n envs, stepped every agent_dt seconds."""
        self.model = model
        self.r = np.tile(np.eye(3), (n, 1, 1))
        self.bias = np.zeros((n, 3))
        self.pos = np.zeros((n, 3))
        self.vel = np.zeros((n, 3))
        self._dt = agent_dt
        self._p00, self._p01, self._p11 = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
        self._gyro = np.zeros((n, 3))
        self._baro = np.zeros(n)
        self._declination = np.arctan2(model.field_ned[:, EAST], model.field_ned[:, NORTH])
        accel_var = model.accel_std**2 + PREDICTION_ACCEL_FLOOR_MPS2**2
        self._q00, self._q01, self._q11 = (
            accel_var * agent_dt**4 / 4.0,
            accel_var * agent_dt**3 / 2.0,
            accel_var * agent_dt**2,
        )
        self._r_fix = LPS_FILTER_SIGMA_M**2
        self._r_baro = model.baro_std**2

    @property
    def rates(self) -> np.ndarray:
        """The last gyro reading with the learned bias, rad/s."""
        return self._gyro + self.bias

    def reset(self, mask: np.ndarray, sensors: np.ndarray, fix: np.ndarray) -> None:
        """Start the masked envs from their readings at rest."""
        s = sensors[mask].astype(np.float64)
        self.r[mask] = triad(-s[:, ACC], s[:, MAG], self.model.field_ned[mask])
        self.bias[mask] = 0.0
        self._gyro[mask] = s[:, GYRO]
        self.pos[mask] = fix[mask]
        self.vel[mask] = 0.0
        self._p00[mask], self._p01[mask], self._p11[mask] = self._r_fix, 0.0, RESET_VELOCITY_STD_MPS**2
        self._baro[mask] = sensors[mask, BARO]
        self._baro_update(sensors, mask)

    def step(self, sensors: np.ndarray, fix: np.ndarray, new_fix: np.ndarray) -> None:
        """Advance every env by one agent step with its new readings."""
        sensors = sensors.astype(np.float64)
        self._attitude(sensors)
        self._predict(sensors[:, ACC])
        self._update(fix, self._r_fix, np.broadcast_to(new_fix[:, None], fix.shape))
        self._baro_update(sensors, sensors[:, BARO] != self._baro)
        self._baro = sensors[:, BARO].copy()

    def errors(self, r_true: np.ndarray, pos_true: np.ndarray, vel_true: np.ndarray) -> dict[str, np.ndarray]:
        """Each env's attitude error (deg), position error (m) and velocity error (m/s) against the truth."""
        cos = (np.einsum("nij,nij->n", self.r, r_true) - 1.0) / 2.0  # trace(R_est^T R_true) = 1 + 2 cos(angle)
        return {
            "attitude_deg": np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))),
            "position_m": np.linalg.norm(self.pos - pos_true, axis=1),
            "velocity_mps": np.linalg.norm(self.vel - vel_true, axis=1),
        }

    def _attitude(self, sensors: np.ndarray) -> None:
        gyro, f = sensors[:, GYRO], sensors[:, ACC]
        f_norm = np.linalg.norm(f, axis=1, keepdims=True)
        down = self.r[:, 2, :]
        accel_ok = np.abs(f_norm - self.model.gravity) < ACC_GATE * self.model.gravity
        correction = np.where(accel_ok, ATT_W_ACC * np.cross(-f / f_norm, down), 0.0)
        field = to_ned(self.r, sensors[:, MAG])  # as attitude_estimator_q: the field's heading in NED vs declination
        yaw_error = wrap_angle(np.arctan2(field[:, EAST], field[:, NORTH]) - self._declination)
        correction += ATT_W_MAG * to_body(self.r, -yaw_error[:, None] * DOWN_AXIS)
        learn = np.linalg.norm(gyro, axis=1, keepdims=True) < BIAS_LEARN_MAX_RATE_RADPS
        learned = np.clip(self.bias + correction * ATT_W_GYRO_BIAS * self._dt, -ATT_BIAS_MAX_RADPS, ATT_BIAS_MAX_RADPS)
        self.bias = np.where(learn, learned, self.bias)
        rate = 0.5 * (gyro + self._gyro) + self.bias + correction
        self.r = orthonormalize(self.r @ rotation_from_vector(rate * self._dt))
        self._gyro = gyro

    def _predict(self, f: np.ndarray) -> None:
        dt = self._dt
        accel = to_ned(self.r, f) + self.model.gravity * DOWN_AXIS
        self.pos += self.vel * dt + 0.5 * accel * dt**2
        self.vel += accel * dt
        p00 = self._p00 + 2.0 * dt * self._p01 + dt**2 * self._p11 + self._q00
        p01 = self._p01 + dt * self._p11 + self._q01
        self._p00, self._p01, self._p11 = p00, p01, self._p11 + self._q11

    def _update(self, z: np.ndarray, noise_var: np.ndarray | float, mask: np.ndarray) -> None:
        """Kalman update by a position measurement z, per env and axis where mask is set."""
        p00, p01, p11 = self._p00, self._p01, self._p11
        s = p00 + noise_var
        k0, k1 = p00 / s, p01 / s
        innovation = z - self.pos
        self.pos = np.where(mask, self.pos + k0 * innovation, self.pos)
        self.vel = np.where(mask, self.vel + k1 * innovation, self.vel)
        self._p00 = np.where(mask, p00 - k0 * p00, p00)
        self._p01 = np.where(mask, p01 - k0 * p01, p01)
        self._p11 = np.where(mask, p11 - k1 * p01, p11)

    def _baro_update(self, sensors: np.ndarray, envs: np.ndarray) -> None:
        z = self.pos.copy()
        z[:, DOWN] = -sensors[:, BARO]
        axes = np.zeros(self.pos.shape, bool)
        axes[:, DOWN] = envs
        self._update(z, self._r_baro, axes)
