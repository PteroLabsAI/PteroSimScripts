"""Fly PX4 drones past a camera and record YOLO tracking rectangles.

Prerequisites: stopped, empty PteroSim scene; installed aircraft; Linux PX4 SITL
build (Windows: WSL Ubuntu-24.04 with python3). Choose an unobstructed spawn area.
A licence that allows the default four aircraft (three flyers and the camera stand),
and a simulator newer than v0.5.0, whose cameras serve raw_stream frames through shared memory.

Install into the active virtual environment:
    python -m pip install pterosim numpy opencv-python pymavlink ultralytics huggingface-hub lap
If pterosim is not published on your package index, install the simulator SDK:
    python -m pip install -e <simulator>/Plugins/PteroSimScripting/SDK/python

Windows/WSL NAT, from PowerShell:
    $simHostIp = (wsl -d Ubuntu-24.04 --exec bash -lc "ip route show default").Trim().Split()[2]
    python python_examples/drone_detection_demo.py --sim-host $simHostIp --api-host $simHostIp --y 8000

Use --aircraft-root <simulator>/PteroSim/PteroSimAircrafts for downloaded builds.
Linux on the same host: python python_examples/drone_detection_demo.py
With an NVIDIA GPU add --device 0: on the CPU the detector runs a few frames a second.
Press Q/Escape in the preview to finish. Output: runs/drone-detection/<timestamp>/detections.mp4.
PX4 runtime copies, detection timestamps and logs stay alongside the recording.

By default standard_vtol hovers between two x500; each rises, sinks and slowly turns in its lane.
standard_vtol stays in multicopter mode, so its wing is in the shot next to the quads.
The detector is TomSmail/drone-yolo-v1 (CC-BY-4.0): quadcopter and fixed-wing classes.
It calls standard_vtol fixed-wing; x500 quadcopter or fixed-wing by the angle, F450 nearly always
fixed-wing. --weights takes your own instead.
Flight follows XTDrone's streamed Offboard setpoints, using pymavlink without ROS.
"""

from __future__ import annotations

# Optional runtime dependencies are loaded after --help, and by the flight worker.
# ruff: noqa: PLC0415
import argparse
import contextlib
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
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


HERE = Path(__file__).resolve().parent
VEHICLES = ("F450", "x500", "standard_vtol")
# The plane between two drones; TomSmail called F450 fixed-wing in 98% of frames here, x500 sometimes a quadcopter.
FLEET = ("x500", "standard_vtol", "x500")
COLORS = ((60, 230, 60), (30, 160, 255), (255, 160, 60), (220, 70, 220))
SETPOINT_HZ = 30.0
TELEMETRY_MAX_AGE = 3.0
PX4_MAIN_MODE_OFFBOARD = 6
ARMING_TIMEOUT = 40.0
COMMAND_PERIOD = 2.0
CM_PER_M = 100.0
# Chosen: at CAMERA_FOV_DEG a 0.45 m F450 spans ~40 px of 1280 here, standard_vtol's 2.14 m wing ~200 px.
TARGET_DISTANCE_M = 12.0
CAMERA_FOV_DEG = 60.0
# Chosen: standard_vtol's 2.14 m wing beside a quad still leaves ~2 m between them.
LANE_SPACING_M = 3.5
CAMERA_TIMEOUT_S = 10.0  # chosen: the camera has stopped if no frame came for this long
RAW_STREAM_HOST = "127.0.0.1"  # the camera's raw_stream_host default; shared memory needs the simulator's machine
RAW_STREAM_PORTS_PER_INSTANCE = 2  # BaseRenderSensor.h SensorOutputPortsPerInstance: raw_stream_port + 2*instance_id
# Chosen: lifts the lens above the stand and tilts it up so the hovering aircraft sit against the sky.
CAMERA_HEIGHT_M = 3.0
CAMERA_PITCH_DEG = 20.0
# Chosen: lane altitudes above each take-off point, one metre apart from left to right.
BASE_ALTITUDE_M = 6.0
ALTITUDE_STEP_M = 1.0
CLIMB_S = 10.0
# Chosen: each aircraft rises and sinks in its lane and turns on the spot, so the camera sees it from every side.
BOB_M = 1.0
BOB_PERIOD_S = 12.0
TURN_PERIOD_S = 30.0


@dataclass(frozen=True)
class Route:
    """Displacement from each vehicle's independently calibrated PX4 origin."""

    altitude_m: float
    phase_rad: float

    def target(self, anchor: LocalPoseNed, elapsed_s: float) -> LocalPoseNed:
        """Climb smoothly, then bob and turn over the take-off point: no move across the lanes."""
        climb = min(max(elapsed_s / CLIMB_S, 0.0), 1.0)
        climb = climb * climb * (3.0 - 2.0 * climb)
        hovered_s = max(0.0, elapsed_s - CLIMB_S)
        angle = 2.0 * math.pi * hovered_s / BOB_PERIOD_S
        # Zero displacement at route start, with no discontinuity after climb.
        bob = BOB_M * (math.sin(angle + self.phase_rad) - math.sin(self.phase_rad))
        turn = 2.0 * math.pi * hovered_s / TURN_PERIOD_S
        return LocalPoseNed(
            anchor.north_m,
            anchor.east_m,
            anchor.down_m - self.altitude_m * climb - bob,
            math.remainder(anchor.yaw_rad + turn, 2.0 * math.pi),
        )


class Flight:
    """Own MAVLink in one thread so slow GPU inference cannot starve Offboard."""

    def __init__(self, vehicles: list[Px4Vehicle], routes: list[Route]) -> None:
        """Allocate control state; start only after every endpoint is connected."""
        self.vehicles = vehicles
        self.routes = routes
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, name="px4-routes", daemon=True)

    def _run(self) -> None:
        try:
            self._fly()
        except Exception as exc:
            self.error = exc
        finally:
            # LAND is autonomous; PX4 handles descent after the stream stops.
            for vehicle in self.vehicles:
                with contextlib.suppress(Exception):
                    vehicle.set_mode("LAND")
            self.ready.set()

    def _fly(self) -> None:
        from pymavlink import mavutil

        anchors = []
        for vehicle in self.vehicles:
            if self.stop_event.is_set():
                return
            anchors.append(vehicle.wait_pose(timeout_s=15.0))
        last_pose = [time.monotonic()] * len(anchors)
        last_heartbeat = last_pose.copy()
        armed = [False] * len(anchors)
        offboard = [False] * len(anchors)
        start = time.monotonic()
        last_command = 0.0
        route_start: float | None = None
        heartbeat_sent = 0.0
        while not self.stop_event.is_set():
            now = time.monotonic()
            for i, (vehicle, route, anchor) in enumerate(zip(self.vehicles, self.routes, anchors, strict=True)):
                while True:
                    message = vehicle.connection.recv_match(blocking=False)
                    if message is None:
                        break
                    if message.get_srcSystem() != vehicle.target_system:
                        continue
                    kind = message.get_type()
                    if kind == "LOCAL_POSITION_NED":
                        last_pose[i] = now
                    elif kind == "HEARTBEAT":
                        last_heartbeat[i] = now
                        armed[i] = bool(message.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                        offboard[i] = (message.custom_mode >> 16) & 0xFF == PX4_MAIN_MODE_OFFBOARD
                    elif kind == "STATUSTEXT":
                        print(f"[{vehicle.name}] {message.text}", flush=True)
                if now - max(start, last_pose[i]) > TELEMETRY_MAX_AGE:
                    raise RuntimeError(f"{vehicle.name}: LOCAL_POSITION_NED lost")
                if now - last_heartbeat[i] > TELEMETRY_MAX_AGE:
                    raise RuntimeError(f"{vehicle.name}: heartbeat lost")
                elapsed = 0.0 if route_start is None else now - route_start
                target = route.target(anchor, elapsed)
                vehicle.send_position_target(target.north_m, target.east_m, target.down_m, target.yaw_rad)
                if now - heartbeat_sent >= 1.0:
                    vehicle.connection.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
                    )
            if now - heartbeat_sent >= 1.0:
                heartbeat_sent = now
            if route_start is None:
                if all(armed) and all(offboard):
                    route_start = now
                    self.ready.set()
                elif now - start > ARMING_TIMEOUT:
                    raise RuntimeError(f"PX4 refused arming/OFFBOARD: armed={armed}, offboard={offboard}; see PX4 logs")
                elif now - start > COMMAND_PERIOD and now - last_command > COMMAND_PERIOD:
                    for vehicle in self.vehicles:
                        vehicle.set_mode("OFFBOARD")
                        vehicle.arm()
                    last_command = now
            elif not all(armed) or not all(offboard):
                raise RuntimeError("PX4 left armed OFFBOARD flight; stopping demo")
            self.stop_event.wait(1.0 / SETPOINT_HZ)

    def check(self) -> None:
        """Propagate flight failures to the recorder."""
        if self.error is not None:
            raise RuntimeError("PX4 route control failed") from self.error

    def stop(self) -> None:
        """Stop the route stream and request autonomous landing."""
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=20.0)


def px4_worker(arguments: list[str]) -> None:
    """Prepare private PX4 resources and replace this worker with the firmware."""
    build_arg, firmware_arg, runtime_arg, instance_arg, sim_host, api_host, api_port_arg = arguments
    build = Path(build_arg).expanduser().resolve()
    binary = build / "bin/px4"
    if not binary.is_file() or not (build / "etc").is_dir():
        raise FileNotFoundError(f"Build PX4 SITL first: {build}")
    runtime = Path(runtime_arg).resolve()
    resources = runtime / "etc"
    runtime.mkdir(parents=True, exist_ok=True)
    shutil.copytree(build / "etc", resources)
    airframe = resources / "init.d-posix/airframes/31000_pterosim_detection"
    airframe.write_text(
        Path(firmware_arg).read_text().replace("\r", "")
        + "\nparam set-default COM_RC_IN_MODE 4\nparam set-default COM_DISARM_PRFLT 30\n"
    )
    instance = int(instance_arg)
    api_port = int(api_port_arg)
    startup = resources / "init.d-posix/px4-rc.mavlink"
    lines = startup.read_text().splitlines()
    matches = [
        i for i, line in enumerate(lines) if line.startswith("mavlink start ") and "udp_offboard_port_local" in line
    ]
    if len(matches) != 1:
        raise RuntimeError("Cannot locate PX4's API link in px4-rc.mavlink")
    lines[matches[0]] = (
        f"mavlink start -x -u {14600 + instance} -o {api_port} -t {shlex.quote(api_host)} -r 1000000 -m onboard"
    )
    startup.write_text("\n".join(lines) + "\n")
    (runtime / "px4.pid").write_text(str(os.getpid()))
    environment = {
        **os.environ,
        "PX4_SYS_AUTOSTART": "31000",
        "PX4_SIM_MODEL": "none_iris",
        "PX4_SIM_HOSTNAME": sim_host,
        "PX4_LOCKSTEP": "1",
        "PX4_SIM_SPEED_FACTOR": "1",
    }
    os.chdir(runtime)
    os.execve(str(binary), [str(binary), "-i", str(instance), "-d", str(resources)], environment)


def stop_px4_worker(pid_file: str) -> None:
    """Signal only the firmware process recorded by this run's worker."""
    with contextlib.suppress(FileNotFoundError, ProcessLookupError):
        pid = int(Path(pid_file).read_text().strip())
        if pid <= 1:
            raise ValueError("Invalid PX4 process ID")
        os.kill(pid, signal.SIGTERM)


class Sitl:
    """Launch private Linux PX4 resources, optionally through Windows WSL."""

    def __init__(self, args: argparse.Namespace, output: Path) -> None:
        """Remember processes and their log files for bounded cleanup."""
        self.args = args
        self.output = output
        self.items: list[tuple[Any, Any, str]] = []
        self.prefix: list[str] = []
        if os.name == "nt":
            wsl = shutil.which("wsl.exe") or str(Path(os.environ["WINDIR"]) / "System32/wsl.exe")
            self.prefix = [wsl, "-d", args.wsl_distro, "--exec"]

    def linux_path(self, path: Path) -> str:
        """Translate host paths using WSL, without interpolating shell code."""
        if self.prefix:
            return subprocess.check_output([*self.prefix, "wslpath", "-a", str(path)], text=True).strip()
        return str(path)

    def start(self, aircraft: Any, model: str, index: int) -> None:
        """Start the airframe matching this aircraft's actual instance ID."""
        firmware = self.args.aircraft_root / model / "firmwares" / f"px4_{model}"
        if not firmware.is_file():
            raise FileNotFoundError(firmware)
        runtime = self.output / f"px4-{aircraft.instance_id}"
        log = (self.output / f"px4-{aircraft.instance_id}.log").open("w", encoding="utf-8")
        runtime_linux = self.linux_path(runtime)
        command = [
            *self.prefix,
            "python3" if self.prefix else sys.executable,
            self.linux_path(Path(__file__).resolve()),
            "--px4-worker",
            self.args.px4_build,
            self.linux_path(firmware),
            runtime_linux,
            str(aircraft.instance_id),
            self.args.sim_host,
            self.args.api_host,
            str(self.args.api_port + index),
        ]
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        except Exception:
            log.close()
            raise
        self.items.append((process, log, runtime_linux))

    def check(self) -> None:
        """Report an exited firmware process before waiting on MAVLink."""
        if any(process.poll() is not None for process, _, _ in self.items):
            raise RuntimeError(f"PX4 exited; inspect {self.output}/px4-*.log")

    def stop(self) -> None:
        """Terminate only firmware processes created by this run."""
        for process, log, runtime in self.items:
            if self.prefix and process.poll() is None:
                with contextlib.suppress(Exception):
                    subprocess.run(
                        [
                            *self.prefix,
                            "python3",
                            self.linux_path(Path(__file__).resolve()),
                            "--px4-stop",
                            f"{runtime}/px4.pid",
                        ],
                        timeout=5,
                        check=False,
                    )
            elif process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            log.close()


def annotate(cv2: Any, image: Any, result: Any) -> tuple[Any, list[dict[str, Any]]]:
    """Draw only model detections; never substitute simulator boxes for YOLO."""
    canvas = image.copy()
    detections: list[dict[str, Any]] = []
    if result.boxes is None:
        return canvas, detections
    for box in result.boxes:
        class_id = int(box.cls.item())
        label = result.names[class_id]
        confidence = float(box.conf.item())
        track_id = int(box.id.item()) if box.id is not None else None
        coords = [float(value) for value in box.xyxy[0].tolist()]
        x1, y1, x2, y2 = [round(value) for value in coords]
        color = COLORS[class_id % len(COLORS)]
        text = f"{label} {confidence:.0%}" + (f" ID {track_id}" if track_id is not None else "")
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, text, (max(0, x1), max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        detections.append({"class": label, "confidence": confidence, "track_id": track_id, "xyxy": coords})
    return canvas, detections


def record(args: argparse.Namespace, stream: Any, model: Any, flight: Flight, sitl: Sitl, output: Path) -> int:
    """Write annotated MP4 in sim time, every captured frame once, with capture times in JSONL."""
    import cv2

    writer = cv2.VideoWriter(
        str(output / "detections.mp4"), cv2.VideoWriter.fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("Cannot open MP4 encoder; use an OpenCV build with FFmpeg")
    start: float | None = None
    written = 0
    detected = 0
    try:
        with (output / "detections.jsonl").open("w", encoding="utf-8") as log:
            while True:
                flight.check()
                sitl.check()
                frame = stream.read()
                start = frame.sim_time if start is None else start
                elapsed = frame.sim_time - start
                if elapsed >= args.duration:
                    break
                if frame.image.shape[:2] != (args.height, args.width):
                    raise RuntimeError(f"Unexpected camera size: {frame.image.shape}")
                image = cv2.cvtColor(frame.image, cv2.COLOR_BGRA2BGR)
                result = model.track(
                    image,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=args.conf,
                    imgsz=args.imgsz,
                    device=args.device,
                    agnostic_nms=True,  # one box per aircraft, under the class it scores higher
                    verbose=False,
                )[0]
                canvas, detections = annotate(cv2, image, result)
                detected += len(detections)
                cv2.putText(
                    canvas,
                    f"PX4 + YOLO | {elapsed:.1f}s | detections {len(detections)}",
                    (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
                if not detections:
                    cv2.putText(
                        canvas, "No YOLO detections", (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 160, 255), 2
                    )
                # A frame holds the video until the next one the detector takes, so the video keeps sim time.
                target_count = max(written + 1, math.ceil(elapsed * args.fps))
                for _ in range(target_count - written):
                    writer.write(canvas)
                log.write(
                    json.dumps(
                        {
                            "sim_seconds": frame.sim_time,
                            "sequence": frame.frame,
                            "video_frame_start": written,
                            "video_frame_end": target_count,
                            "detections": detections,
                        }
                    )
                    + "\n"
                )
                written = target_count
                if not args.no_preview:
                    cv2.imshow("PteroSim - PX4 drone detection", canvas)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
    finally:
        writer.release()
        if not args.no_preview:
            cv2.destroyAllWindows()
    if not written:
        raise RuntimeError("Camera returned no frames")
    if stream.dropped:
        print(f"{stream.dropped} camera frames were overwritten before the detector took them")
    if not detected:
        raise RuntimeError(
            f"Saved video but YOLO detected no drones. Inspect {output}; try --conf 0.15 or custom weights"
        )
    return written


def parser() -> argparse.ArgumentParser:
    """Expose scene, firmware and detector settings without editing source."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--address", default="localhost:10010")
    ap.add_argument("--sim-host", default="127.0.0.1", help="Simulator address as seen by Linux PX4")
    ap.add_argument("--api-host", default="127.0.0.1", help="Python address as seen by Linux PX4")
    ap.add_argument("--api-port", type=int, default=15540)
    ap.add_argument("--raw-stream-port", type=int, default=5710, help="Camera frames' port before the instance offset")
    ap.add_argument("--wsl-distro", default="Ubuntu-24.04")
    ap.add_argument("--px4-build", default="~/Documents/PX4-Autopilot/build/px4_sitl_default")
    ap.add_argument("--aircraft-root", type=Path, default=HERE.parents[1] / "PteroSimAircrafts")
    ap.add_argument(
        "--models",
        nargs="+",
        default=list(FLEET),
        choices=VEHICLES,
        help="Hover-capable aircraft to fly, in left-to-right lane order",
    )
    ap.add_argument("--weights", type=Path, help="Custom Ultralytics detection weights; default TomSmail best.pt")
    ap.add_argument("--device", default="cpu", help="cpu or CUDA device index, e.g. 0")
    ap.add_argument("--inference-threads", type=int, default=4, help="CPU inference threads; leave capacity for PX4")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--landing-timeout", type=float, default=45.0)
    ap.add_argument("--x", type=float, default=0.0, help="Camera stand position, UE cm")
    ap.add_argument("--y", type=float, default=0.0)
    ap.add_argument("--z", type=float, default=100.0)
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--output", type=Path, default=HERE.parent / "runs" / "drone-detection")
    return ap


def main() -> None:
    """Prepare the detector, spawn the scene, fly, record, land, and clean up."""
    args = parser().parse_args()
    if not 0 < args.conf <= 1 or any(
        not math.isfinite(v) or v <= 0
        for v in (
            args.duration,
            args.fps,
            args.landing_timeout,
            args.width,
            args.height,
            args.imgsz,
            args.inference_threads,
        )
    ):
        raise ValueError("Confidence must be in (0,1]; duration, rates, timeout and sizes must be positive")
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from pterosim import PteroSim, RawStream
    from ultralytics import YOLO

    torch.set_num_threads(args.inference_threads)
    weights = args.weights or hf_hub_download("TomSmail/drone-yolo-v1", "best.pt")
    model = YOLO(str(weights))
    # Warm up here: the first inference loads the device for seconds, which would hold the video's first frame.
    model.predict(np.zeros((args.height, args.width, 3), np.uint8), imgsz=args.imgsz, device=args.device, verbose=False)
    print(f"Detector classes: {model.names}; these are visual predictions, not aircraft model names")
    output = args.output.resolve() / time.strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    sitl = Sitl(args, output)
    vehicles: list[Px4Vehicle] = []
    spawned = []
    flight = None
    started = False
    with PteroSim(args.address) as sim:
        status = sim.status()
        if status.is_running or status.aircraft_count:
            raise RuntimeError("Open a stopped, empty PteroSim scene before running this demo")
        available = {item.name for item in sim.list_aircraft_classes()}
        if not set(args.models) <= available:
            raise RuntimeError(f"Install missing aircraft: {set(args.models) - available}")
        previous_bind = sim.get_mavlink_bind_address()
        try:
            sim.set_mavlink_bind_address("0.0.0.0")
            # Spawn flight vehicles first: HIL system IDs must match PX4 instance + 1.
            for index, name in enumerate(args.models):
                lane_m = (index - (len(args.models) - 1) / 2) * LANE_SPACING_M
                aircraft = sim.spawn(
                    name, x=args.x + TARGET_DISTANCE_M * CM_PER_M, y=args.y + lane_m * CM_PER_M, z=args.z, yaw=0.0
                )
                spawned.append(aircraft)
                aircraft.set_flight_stack("PX4")
                print(f"{name}: instance={aircraft.instance_id}, HIL={aircraft.link_port}, API={args.api_port + index}")
                for sensor in aircraft.list_sensors():
                    if sensor.type == "camera":
                        aircraft.set_sensor_param(sensor.name, enabled=False, stream=False)
            camera = sim.spawn("x500", x=args.x, y=args.y, z=args.z, yaw=0.0)
            spawned.append(camera)
            camera.set_flight_stack("")
            for sensor in camera.list_sensors():
                if sensor.type == "camera":
                    camera.set_sensor_param(sensor.name, enabled=sensor.name == "camera", stream=False)
            camera.set_sensor_pose(
                "camera", position=(0.0, 0.0, CAMERA_HEIGHT_M), orientation=(0.0, CAMERA_PITCH_DEG, 0.0)
            )
            camera.set_sensor_param(
                "camera",
                image_width=args.width,
                image_height=args.height,
                field_of_view=CAMERA_FOV_DEG,
                update_hz=args.fps,
                raw_stream=True,
                raw_stream_port=args.raw_stream_port,
            )
            started = True
            sim.start()
            for index, aircraft in enumerate(spawned[:-1]):
                sitl.start(aircraft, args.models[index], index)
            manifest = {
                "models": args.models,
                "detector_classes": model.names,
                "weights": str(weights),
                "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "instances": [a.instance_id for a in spawned],
            }
            (output / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            for index, name in enumerate(args.models):
                sitl.check()
                vehicles.append(
                    Px4Vehicle.connect(f"udpin:0.0.0.0:{args.api_port + index}", name=name, heartbeat_timeout_s=60.0)
                )
                if vehicles[-1].target_system != spawned[index].instance_id + 1:
                    raise RuntimeError("Unexpected MAVLink system ID: another PX4 may own this port")
            routes = [
                Route(BASE_ALTITUDE_M + index * ALTITUDE_STEP_M, 2.0 * math.pi * index / len(vehicles))
                for index in range(len(vehicles))
            ]
            flight = Flight(vehicles, routes)
            flight.thread.start()
            if not flight.ready.wait(90.0):
                raise TimeoutError("PX4 flight startup timed out")
            flight.check()
            raw_port = args.raw_stream_port + RAW_STREAM_PORTS_PER_INSTANCE * camera.instance_id
            with RawStream(RAW_STREAM_HOST, raw_port, timeout=CAMERA_TIMEOUT_S, shared_memory=True) as stream:
                count = record(args, stream, model, flight, sitl, output)
            print(f"Saved {count} frames: {output / 'detections.mp4'}")
        finally:
            try:
                if flight:
                    flight.stop()
                    # Keep physics and PX4 running until autonomous landing disarms.
                    deadline = time.monotonic() + args.landing_timeout
                    while time.monotonic() < deadline:
                        for vehicle in vehicles:
                            vehicle.connection.recv_match(type="HEARTBEAT", blocking=False)
                        if all(not v.connection.motors_armed() for v in vehicles):
                            break
                        time.sleep(0.1)
            finally:
                # A disconnected simulator must not prevent firmware cleanup.
                actions = ([sim.stop] if started else []) + [sitl.stop]
                actions += [v.close for v in vehicles] + [a.remove for a in reversed(spawned)]
                actions.append(lambda: sim.set_mavlink_bind_address(previous_bind))
                for action in actions:
                    try:
                        action()
                    except Exception as exc:
                        print(f"Cleanup failed ({action}): {exc}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--px4-worker"]:
        px4_worker(sys.argv[2:])
    elif sys.argv[1:2] == ["--px4-stop"]:
        stop_px4_worker(sys.argv[2])
    else:
        main()
