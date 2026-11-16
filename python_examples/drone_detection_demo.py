"""Spawn a camera stand and a mixed PX4 fleet; record tracked YOLO boxes.

See drone_detection_demo.md for Windows/WSL and Linux commands and limitations.
Flight uses real PX4 Offboard MAVLink, following XTDrone's streamed-setpoint design.
"""

from __future__ import annotations

# Optional runtime dependencies are loaded after --help, and by the flight worker.
# ruff: noqa: PLC0415
import argparse
import contextlib
import json
import math
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drone_detection_px4 import LocalPoseNed, Px4Vehicle

HERE = Path(__file__).resolve().parent
VEHICLES = ("F450", "x500")
COLORS = ((60, 230, 60), (30, 160, 255), (255, 160, 60), (220, 70, 220))
SETPOINT_HZ = 30.0
TELEMETRY_MAX_AGE = 3.0
PX4_MAIN_MODE_OFFBOARD = 6
ARMING_TIMEOUT = 40.0
COMMAND_PERIOD = 2.0


@dataclass(frozen=True)
class Route:
    """Displacement from each vehicle's independently calibrated PX4 origin."""

    altitude: float
    phase: float

    def target(self, anchor: LocalPoseNed, elapsed: float) -> LocalPoseNed:
        """Climb smoothly, then fly a small ellipse inside a separate viewing lane."""
        climb = min(max(elapsed / 10.0, 0.0), 1.0)
        climb = climb * climb * (3.0 - 2.0 * climb)
        angle = 2.0 * math.pi * max(0.0, elapsed - 10.0) / 24.0
        # Zero displacement at route start, with no discontinuity after climb.
        north = 1.0 * (math.cos(angle + self.phase) - math.cos(self.phase))
        east = 1.2 * (math.sin(angle + self.phase) - math.sin(self.phase))
        return LocalPoseNed(
            anchor.north_m + north, anchor.east_m + east, anchor.down_m - self.altitude * climb, anchor.yaw_rad
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
            "bash",
            self.linux_path(HERE / "drone_detection_px4.sh"),
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
                            "bash",
                            self.linux_path(HERE / "drone_detection_px4.sh"),
                            "--stop",
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


def record(args: argparse.Namespace, camera: Any, model: Any, flight: Flight, sitl: Sitl, output: Path) -> int:
    """Write annotated MP4 at wall time speed, with capture times in JSONL."""
    import cv2

    writer = cv2.VideoWriter(
        str(output / "detections.mp4"), cv2.VideoWriter.fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("Cannot open MP4 encoder; use an OpenCV build with FFmpeg")
    start = time.monotonic()
    written = 0
    detected = 0
    try:
        with (output / "detections.jsonl").open("w", encoding="utf-8") as log:
            while time.monotonic() - start < args.duration:
                flight.check()
                sitl.check()
                frame = camera.camera("camera", timeout=10.0)
                image = frame.image
                if image.shape[:2] != (args.height, args.width):
                    raise RuntimeError(f"Unexpected camera size: {image.shape}")
                result = model.track(
                    image,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=args.conf,
                    imgsz=args.imgsz,
                    device=args.device,
                    verbose=False,
                )[0]
                canvas, detections = annotate(cv2, image, result)
                detected += len(detections)
                elapsed = time.monotonic() - start
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
                # Duplicate frames if inference is slower than playback FPS.
                target_count = max(written + 1, math.ceil(min(elapsed, args.duration) * args.fps))
                for _ in range(target_count - written):
                    writer.write(canvas)
                log.write(
                    json.dumps(
                        {
                            "wall_seconds": elapsed,
                            "sim_seconds": frame.timestamp,
                            "sequence": frame.sequence_number,
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
                time.sleep(max(0.0, written / args.fps - (time.monotonic() - start)))
    finally:
        writer.release()
        if not args.no_preview:
            cv2.destroyAllWindows()
    if not written:
        raise RuntimeError("Camera returned no frames")
    if not detected:
        raise RuntimeError(
            f"Saved video but YOLO detected no drones. Inspect {output}; try --conf 0.15 or custom weights"
        )
    return written


def parser() -> argparse.ArgumentParser:
    """Expose scene, firmware and detector settings without editing source."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--address", default="localhost:10010")
    ap.add_argument("--sim-host", default="127.0.0.1", help="Simulator address as seen by Linux PX4")
    ap.add_argument("--api-host", default="127.0.0.1", help="Python address as seen by Linux PX4")
    ap.add_argument("--api-port", type=int, default=15540)
    ap.add_argument("--wsl-distro", default="Ubuntu-24.04")
    ap.add_argument("--px4-build", default="~/Documents/PX4-Autopilot/build/px4_sitl_default")
    ap.add_argument("--aircraft-root", type=Path, default=HERE.parents[1] / "PteroSimAircrafts")
    ap.add_argument(
        "--models",
        nargs="+",
        default=list(VEHICLES),
        choices=("F450", "x500", "standard_vtol", "tiltrotor", "crazyflie_bl"),
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
    ap.add_argument("--output", type=Path, default=HERE.parent / "recordings")
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
    import torch
    from huggingface_hub import hf_hub_download
    from pterosim import PteroSim
    from ultralytics import YOLO

    torch.set_num_threads(args.inference_threads)
    weights = args.weights or hf_hub_download("TomSmail/drone-yolo-v1", "best.pt")
    model = YOLO(str(weights))
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
                aircraft = sim.spawn(
                    name, x=args.x + 1800.0, y=args.y + (index - (len(args.models) - 1) / 2) * 650.0, z=args.z, yaw=0.0
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
            camera.set_sensor_pose("camera", position=(0.0, 0.0, 3.0), orientation=(0.0, 12.0, 0.0))
            camera.set_sensor_param(
                "camera", image_width=args.width, image_height=args.height, field_of_view=80.0, update_hz=args.fps
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
            flight = Flight(vehicles, [Route(7.0 + index, index * 1.0) for index in range(len(vehicles))])
            flight.thread.start()
            if not flight.ready.wait(90.0):
                raise TimeoutError("PX4 flight startup timed out")
            flight.check()
            count = record(args, camera, model, flight, sitl, output)
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
    main()
