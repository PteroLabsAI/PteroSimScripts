"""Minimal PX4 Offboard adapter for the drone detection recording demo.

The NED position and streamed-setpoint approach follows XTDrone and the local
formation examples. This module uses pymavlink directly, without ROS or MAVROS.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

POSITION_AND_YAW_MASK = 0x09F8


def _mavutil() -> Any:
    """Load MAVLink only when a flight operation needs it."""
    from pymavlink import mavutil

    return mavutil


@dataclass(frozen=True)
class LocalPoseNed:
    """Local position in metres (north, east, down) and yaw in radians."""

    north_m: float
    east_m: float
    down_m: float
    yaw_rad: float


class Px4Vehicle:
    """One connection to a PX4 API endpoint, separate from its simulator HIL link."""

    def __init__(self, connection: Any, *, name: str) -> None:
        """Keep the heartbeat's target identity and latest local pose."""
        self.connection = connection
        self.name = name
        self.target_system = int(connection.target_system)
        self.target_component = int(connection.target_component or 1)
        self._position: tuple[float, float, float] | None = None
        self._yaw: float | None = None

    @classmethod
    def connect(cls, endpoint: str, *, name: str, heartbeat_timeout_s: float = 20.0) -> Px4Vehicle:
        """Wait for PX4 on an API UDP endpoint, never on HIL TCP port 4560."""
        connection = _mavutil().mavlink_connection(endpoint, source_system=250, autoreconnect=True)
        try:
            if connection.wait_heartbeat(timeout=heartbeat_timeout_s) is None:
                raise TimeoutError(f"{name}: no PX4 heartbeat on {endpoint}")
            return cls(connection, name=name)
        except Exception:
            connection.close()
            raise

    def close(self) -> None:
        """Release the API socket."""
        self.connection.close()

    def wait_pose(self, timeout_s: float = 15.0) -> LocalPoseNed:
        """Read initial estimator position and attitude, with a bounded wait."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            while True:
                message = self.connection.recv_match(type=["LOCAL_POSITION_NED", "ATTITUDE"], blocking=False)
                if message is None:
                    break
                if message.get_srcSystem() != self.target_system:
                    continue
                if message.get_type() == "LOCAL_POSITION_NED":
                    self._position = (float(message.x), float(message.y), float(message.z))
                elif message.get_type() == "ATTITUDE":
                    self._yaw = float(message.yaw)
            if self._position is not None and self._yaw is not None:
                return LocalPoseNed(*self._position, self._yaw)
            time.sleep(0.01)
        raise TimeoutError(f"{self.name}: no LOCAL_POSITION_NED/ATTITUDE telemetry")

    def send_position_target(self, north_m: float, east_m: float, down_m: float, yaw_rad: float) -> None:
        """Stream a position and yaw target in PX4's own local NED frame."""
        self.connection.mav.set_position_target_local_ned_send(
            int(time.monotonic() * 1000.0) & 0xFFFFFFFF,
            self.target_system,
            self.target_component,
            _mavutil().mavlink.MAV_FRAME_LOCAL_NED,
            POSITION_AND_YAW_MASK,
            float(north_m),
            float(east_m),
            float(down_m),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            float(yaw_rad),
            0.0,
        )

    def set_mode(self, mode: str) -> None:
        """Select a PX4 mode using pymavlink's PX4-aware command helper."""
        name = mode.upper()
        if name not in (self.connection.mode_mapping() or {}):
            raise RuntimeError(f"{self.name}: PX4 mode {mode!r} is unavailable")
        self.connection.set_mode(name)

    def arm(self) -> None:
        """Request arming while retaining PX4's preflight checks."""
        self.connection.mav.command_long_send(
            self.target_system,
            self.target_component,
            _mavutil().mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
