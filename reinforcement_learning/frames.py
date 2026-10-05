"""Rotations between the body frame (FRD) and NED, batched over environments."""

from __future__ import annotations

import math

import numpy as np

NORTH, EAST, DOWN = 0, 1, 2
DOWN_AXIS = np.array([0.0, 0.0, 1.0])
STANDARD_GRAVITY = 9.80665  # m/s^2, ISO 80000-3
SMALL_ANGLE_RAD = 1e-4  # below it the Taylor series replace sin(x)/x and (1 - cos x)/x^2 (float64 rounding)


def body_to_ned(q: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotations taking body FRD vectors to NED, from (N, 4) w, x, y, z quaternions.

    JSBSim's local quaternion gives Tl2b = this transposed (FGQuaternion.cpp, Stevens & Lewis eq. 1.3-32).
    """
    w, x, y, z = q.T.astype(np.float64)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def to_body(r: np.ndarray, v_ned: np.ndarray) -> np.ndarray:
    """NED vectors in the body frame."""
    return np.einsum("nji,nj->ni", r, v_ned)


def to_ned(r: np.ndarray, v_body: np.ndarray) -> np.ndarray:
    """Body vectors in NED."""
    return np.einsum("nij,nj->ni", r, v_body)


def gravity_body(r: np.ndarray) -> np.ndarray:
    """Unit gravity direction in the body frame, (0, 0, 1) when level."""
    return r[:, 2, :]


def tilt_deg(r: np.ndarray) -> np.ndarray:
    """Angle between the body z axis and down."""
    return np.degrees(np.arccos(np.clip(gravity_body(r)[:, 2], -1.0, 1.0)))


def heading(r: np.ndarray) -> np.ndarray:
    """Yaw (rad, from north toward east) of the body x axis."""
    return np.arctan2(r[:, 1, 0], r[:, 0, 0])


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Angles in [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_rate(r: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """d(heading)/dt from body rates: (q sin roll + r cos roll) / cos pitch."""
    roll = np.arctan2(r[:, 2, 1], r[:, 2, 2])
    cos_pitch = np.sqrt(np.clip(1.0 - r[:, 2, 0] ** 2, 0.0, 1.0))
    return (rates[:, 1] * np.sin(roll) + rates[:, 2] * np.cos(roll)) / np.maximum(cos_pitch, np.finfo(float).tiny)


def skew(v: np.ndarray) -> np.ndarray:
    """(N, 3, 3) cross-product matrices of (N, 3) vectors."""
    zero = np.zeros(len(v))
    x, y, z = v.T
    return np.stack([np.stack([zero, -z, y], -1), np.stack([z, zero, -x], -1), np.stack([-y, x, zero], -1)], -2)


def rotation_from_vector(phi: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotations by the rotation vectors phi (N, 3) (Rodrigues)."""
    theta = np.linalg.norm(phi, axis=1)[:, None, None]
    small = theta < SMALL_ANGLE_RAD
    safe = np.where(small, 1.0, theta)
    a = np.where(small, 1.0 - theta**2 / 6.0, np.sin(safe) / safe)
    b = np.where(small, 0.5 - theta**2 / 24.0, (1.0 - np.cos(safe)) / safe**2)
    k = skew(phi)
    return np.eye(3) + a * k + b * (k @ k)


def orthonormalize(r: np.ndarray) -> np.ndarray:
    """The nearest rotation matrices (SVD)."""
    u, _, vt = np.linalg.svd(r)
    return u @ vt


def triad(down_b: np.ndarray, field_b: np.ndarray, field_ned: np.ndarray) -> np.ndarray:
    """(N, 3, 3) body-to-NED rotations from gravity (down) and the magnetic field seen in the body (TRIAD)."""

    def frame(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        c = np.cross(a, b)
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        return np.stack([a, c, np.cross(a, c)], axis=-1)

    return frame(np.broadcast_to(DOWN_AXIS, down_b.shape), field_ned) @ frame(down_b, field_b).transpose(0, 2, 1)
