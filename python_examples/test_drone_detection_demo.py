"""Exercise geometry, telemetry failure, overlays, and MP4 without a simulator."""

from __future__ import annotations

import json
import math
import time
from types import SimpleNamespace
from typing import Any, cast

import cv2
import numpy as np
import pytest
from drone_detection_demo import Flight, Route, Sitl, annotate, parser, record
from drone_detection_px4 import POSITION_AND_YAW_MASK, LocalPoseNed, Px4Vehicle

CONTINUITY_TOLERANCE = 1e-5


def detection_result(*, tracked: bool = True) -> Any:
    """Build a synthetic inference result, explicitly unrelated to accuracy."""
    box = SimpleNamespace(
        cls=np.array(1),
        conf=np.array(0.9),
        id=np.array(7) if tracked else None,
        xyxy=np.array([[40.0, 50.0, 140.0, 110.0]]),
    )
    return SimpleNamespace(boxes=[box], names={1: "fixed-wing"})


def test_routes_preserve_independent_origins_and_start_continuously() -> None:
    """Each local origin survives calibration; altitude uses NED down sign."""
    anchor = LocalPoseNed(150.0, -250.0, 17.0, 0.7)
    for phase in (0.0, 1.0, 2.0):
        route = Route(8.0, phase)
        assert route.target(anchor, 0.0) == anchor
        assert route.target(anchor, 10.0) == LocalPoseNed(150.0, -250.0, 9.0, 0.7)
        before = route.target(anchor, 10.0 - 1e-6)
        after = route.target(anchor, 10.0 + 1e-6)
        assert (
            math.dist((before.north_m, before.east_m, before.down_m), (after.north_m, after.east_m, after.down_m))
            < CONTINUITY_TOLERANCE
        )


def test_route_envelopes_stay_in_front_of_camera() -> None:
    """Bound all horizontal excursions, including arbitrary NED/world rotation."""
    anchor = LocalPoseNed(0.0, 0.0, 0.0, 0.0)
    for index in range(3):
        for elapsed in np.linspace(0.0, 100.0, 300):
            target = Route(7.0 + index, float(index)).target(anchor, float(elapsed))
            assert math.hypot(target.north_m, target.east_m) < 3.2
            assert -(7.0 + index) <= target.down_m <= 0.0


def test_overlay_draws_real_pixels_without_mutating_source() -> None:
    """Boxes and labels correspond to inference values, including missing IDs."""
    for tracked in (False, True):
        image = np.zeros((180, 240, 3), dtype=np.uint8)
        canvas, detections = annotate(cv2, image, detection_result(tracked=tracked))
        assert not image.any()
        assert canvas[50, 40].any()
        assert detections == [
            {
                "class": "fixed-wing",
                "confidence": 0.9,
                "track_id": 7 if tracked else None,
                "xyxy": [40.0, 50.0, 140.0, 110.0],
            }
        ]


def test_no_detections_does_not_invent_boxes() -> None:
    """An empty result leaves all pixels untouched."""
    image = np.zeros((180, 240, 3), dtype=np.uint8)
    canvas, detections = annotate(cv2, image, SimpleNamespace(boxes=None))
    assert not detections
    assert not canvas.any()


class FakeVehicle:
    """Stream synthetic PX4 telemetry without sockets or an autopilot."""

    def __init__(self, *, silent: bool = False) -> None:
        """Provide the surface consumed by Flight, with an optional link outage."""
        self.name = "test"
        self.target_system = 1
        self.targets: list[Any] = []
        self.modes: list[str] = []
        self.pending = (
            []
            if silent
            else [
                SimpleNamespace(
                    get_srcSystem=lambda: 1, get_type=lambda: "HEARTBEAT", base_mode=128, custom_mode=6 << 16
                ),
                SimpleNamespace(get_srcSystem=lambda: 1, get_type=lambda: "LOCAL_POSITION_NED"),
            ]
        )
        self.connection = SimpleNamespace(recv_match=self.recv, mav=SimpleNamespace(heartbeat_send=lambda *_: None))

    def recv(self, **_kwargs: Any) -> Any:
        """Return queued telemetry, then no messages."""
        return self.pending.pop(0) if self.pending else None

    def wait_pose(self, **_kwargs: Any) -> LocalPoseNed:
        """Use an arbitrary local origin."""
        return LocalPoseNed(25.0, 50.0, 0.0, 0.0)

    def send_position_target(self, *target: Any) -> None:
        """Record a setpoint delivered by the actual background thread."""
        self.targets.append(target)

    def set_mode(self, mode: str) -> None:
        """Record mode changes, including cleanup's LAND command."""
        self.modes.append(mode)


def test_flight_stream_continues_while_recorder_is_busy() -> None:
    """Slow inference on the main thread cannot starve the flight thread."""
    vehicle = FakeVehicle()
    flight = Flight([cast(Px4Vehicle, vehicle)], [Route(8.0, 0.0)])
    try:
        flight.thread.start()
        assert flight.ready.wait(1.0)
        before = len(vehicle.targets)
        time.sleep(0.25)
        flight.check()
        assert len(vehicle.targets) - before >= 4
    finally:
        flight.stop()
    assert not flight.thread.is_alive()
    assert vehicle.modes[-1] == "LAND"


def test_telemetry_outage_propagates_and_requests_land(monkeypatch: Any) -> None:
    """A lost local-position stream ends flight instead of using stale data."""
    monkeypatch.setattr("drone_detection_demo.TELEMETRY_MAX_AGE", 0.05)
    vehicle = FakeVehicle(silent=True)
    flight = Flight([cast(Px4Vehicle, vehicle)], [Route(8.0, 0.0)])
    flight.thread.start()
    flight.thread.join(timeout=2.0)
    assert not flight.thread.is_alive()
    assert "LOCAL_POSITION_NED lost" in str(flight.error)
    with pytest.raises(RuntimeError, match="route control failed"):
        flight.check()
    assert vehicle.modes[-1] == "LAND"


def test_recorder_encodes_annotated_video_and_capture_log(tmp_path: Any) -> None:
    """Decode actual output and check its overlay and logged frame mapping."""
    args = parser().parse_args(["--no-preview", "--duration", "0.25", "--width", "240", "--height", "180"])
    image = np.zeros((180, 240, 3), dtype=np.uint8)
    camera = SimpleNamespace(camera=lambda *_a, **_k: SimpleNamespace(image=image, timestamp=2.5, sequence_number=3))
    model = SimpleNamespace(track=lambda *_a, **_k: [detection_result()])
    controller = SimpleNamespace(check=lambda: None)
    count = record(args, camera, model, cast(Flight, controller), cast(Sitl, controller), tmp_path)
    capture = cv2.VideoCapture(str(tmp_path / "detections.mp4"))
    try:
        ok, decoded = capture.read()
        assert ok
        assert decoded.shape == image.shape
        assert decoded[48:53, 38:43].any()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == count
    finally:
        capture.release()
    rows = [json.loads(line) for line in (tmp_path / "detections.jsonl").read_text().splitlines()]
    assert rows[-1]["video_frame_end"] == count
    assert rows[0]["detections"][0]["track_id"] == 7
    assert rows[0]["sim_seconds"] == 2.5


def test_px4_position_message_uses_target_identity_and_ned_mask() -> None:
    """Check actual MAVLink fields rather than only the route math."""
    sent: list[tuple[Any, ...]] = []
    connection = SimpleNamespace(
        target_system=3,
        target_component=1,
        mav=SimpleNamespace(set_position_target_local_ned_send=lambda *values: sent.append(values)),
    )
    vehicle = Px4Vehicle(connection, name="test")
    vehicle.send_position_target(12.0, -4.0, -8.0, 0.3)
    assert len(sent) == 1
    assert sent[0][1:5] == (3, 1, 1, POSITION_AND_YAW_MASK)
    assert sent[0][5:] == (12.0, -4.0, -8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0)


def test_px4_heartbeat_timeout_closes_socket(monkeypatch: Any) -> None:
    """A failed connection must not leave a UDP port bound."""
    closed: list[bool] = []
    connection = SimpleNamespace(wait_heartbeat=lambda **_kwargs: None, close=lambda: closed.append(True))
    monkeypatch.setattr(
        "drone_detection_px4._mavutil", lambda: SimpleNamespace(mavlink_connection=lambda *_args, **_kwargs: connection)
    )
    with pytest.raises(TimeoutError, match="no PX4 heartbeat"):
        Px4Vehicle.connect("udpin:0.0.0.0:15540", name="test", heartbeat_timeout_s=0.01)
    assert closed == [True]
