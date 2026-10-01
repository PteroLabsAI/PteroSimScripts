#!/usr/bin/env python3
"""PteroSim drone cameras -> ROS 2 -> Foxglove, in one file.

Flies a ring of aircraft around its own centre with every nose on that centre, and publishes each
aircraft's gimbal camera to ROS 2 as foxglove_msgs/msg/CompressedVideo, with PX4's estimate of
every aircraft's pose on /tf. Viewing is left to Foxglove, so there is no viewer here. The steps:

    pterosim_cameras.py setup        spawn the ring, point the cameras at this machine, start
    MicroXRCEAgent udp4 -p 8888      PX4's link to ROS 2; every PX4 instance shares this one Agent
    pterosim_cameras.py fly          one PX4 SITL per aircraft: take off, orbit; Ctrl+C lands
    pterosim_cameras.py bridge       publish the streams and PX4's poses to ROS 2
    pterosim_cameras.py foxglove     serve those topics to Foxglove over a WebSocket
    pterosim_cameras.py status       what is publishing, and at what rate

`setup` puts four x500 on a circle, each facing its centre, and streams every gimbal_camera over
RTP/H.264 to this machine; the simulator only accepts sensor settings while it is stopped, which
is why this is a step of its own. `fly` starts one PX4 SITL per aircraft, takes the fleet off and
orbits the circle's centre with every nose held on it: aircraft on one circle at one speed keep
their spacing, so each camera keeps the others in view. `bridge` forwards the H.264 frames
undecoded and turns each PX4's vehicle_odometry into map -> x500_<id>/base_link.

WHICH MACHINE RUNS WHICH STEP

`setup` talks to the simulator over gRPC, and the simulator binds that server to 127.0.0.1, so
`setup` runs where PteroSim runs. The other steps run in WSL next to ROS 2 and PX4; with mirrored
networking PX4 reaches the simulator's HIL ports on that loopback too. The steps share a
manifest: `setup` writes it next to the user's home, and WSL reaches it as
/mnt/c/Users/<you>/camera_fleet.json, so the steps there need --manifest pointed at that path.

Mirrored WSL drops UDP over 1472 bytes sent to 127.0.0.1, which breaks Fast DDS discovery between
local nodes, so the ROS steps load fastdds_wsl.xml; any other ROS node needs it too, via
FASTRTPS_DEFAULT_PROFILES_FILE. From Windows, connect Foxglove to ws://127.0.0.1:8765:
localhost resolves to ::1 first, which mirrored WSL does not forward.
python_examples/pterosim_cameras_foxglove.json is a ready layout -- the four cameras and a 3D view
of the fleet -- import it in Foxglove instead of arranging the panels by hand.

WHY IT LOOKS THE WAY IT DOES

The simulator already has a camera pipeline: its own encoder sends RTP/H.264 to
udp://<host>:<stream_port + instance_id>, the same stream a ground station displays. This
consumes that stream instead of pulling frames over gRPC, which is what
python_examples/drone_camera_display.py does. That matters: a gRPC pull is a synchronous GPU
readback on the sim's render path, paid for every frame pulled, while reading a push stream
touches nothing in the simulator. Nothing is decoded here either: Foxglove decodes
H.264 itself, so the bridge only moves each access unit into a message.

Two details that are easy to get wrong:

  * Ports step by two, not one. ffmpeg binds port+1 alongside every RTP port for RTCP, so a
    decoder on 5600 also holds 5601 -- which would be the next aircraft's own stream. The SDK
    derives the port as stream_port + instance_id, so giving each aircraft
    stream_port = BASE + instance_id lands the real ports BASE + 2*instance_id and the RTCP
    ports fall on free ones.

  * The SDP file needs format="sdp" and protocol_whitelist. Payload type 96 says nothing on
    its own, ffmpeg takes such a description only from a file, and without the whitelist it
    cannot open the nested rtp/udp stream at all.

The cameras are configured for 768x480; --tile changes that.

Requires: ROS 2 with foxglove_msgs, PyAV, the PteroSim SDK for `setup`
($PTEROSIM_SDK), and for `fly` pymavlink plus a PX4-Autopilot checkout built for px4_sitl_default
with the vehicle's airframe installed. The Micro XRCE-DDS Agent and px4_msgs as in PX4's ROS 2 User
Guide, px4_msgs sourced for `bridge` and `foxglove`. `foxglove` needs the foxglove_bridge package:

    sudo apt install ros-$ROS_DISTRO-foxglove-bridge ros-$ROS_DISTRO-foxglove-msgs
"""

import argparse
import array
import functools
import json
import math
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

DEFAULT_TILE = "768x480"
# A hand-placed formation is given as x,y,yaw triples.
LAYOUT_FIELDS = 3
# spawn() takes Unreal units, which are centimetres; every distance on this command line is metres.
CM_PER_M = 100.0
MANIFEST = str(Path.home() / "camera_fleet.json")
# The simulator binds its scripting server to loopback only -- PteroSimScripting's
# GrpcServer.cpp hardcodes 127.0.0.1, and the port defaults to 10010. So `setup` has to run on
# the same machine as PteroSim, which for this setup is Windows.
GRPC_ADDRESS = "127.0.0.1:10010"
# 127.0.0.1, not localhost: from Windows, localhost resolves to ::1 first, which mirrored WSL
# does not forward, and the WebSocket handshake hangs.
FOXGLOVE_URL = "ws://127.0.0.1:8765"
VIDEO_TYPE = "foxglove_msgs/msg/CompressedVideo"
# Measured on Humble's Fast DDS 2.6.12 under mirrored WSL; see the module docstring.
FASTDDS_PROFILE = str(Path(__file__).resolve().parent / "fastdds_wsl.xml")

# PX4's onboard link for an API listens on this + instance id (px4-rc.mavlink).
PX4_API_PORT = 14580
HIL_BASE_PORT = 4560  # PteroSim's PX4 HIL server listens on this + instance id
PX4_AIRFRAME = 22100  # the id x500's firmwares/px4_x500 is installed under in the PX4 tree
GCS_SYSTEM_ID = 245  # a ground station's id, clear of the vehicles' 1..N
HEARTBEAT_PERIOD_S = 1.0  # PX4 declares a GCS lost after a few silent seconds and will not arm
POLL_PERIOD_S = 0.01  # chosen, no evidence: fast enough that no link's socket fills
PX4_BOOT_TIMEOUT_S = 60.0  # boot is a few seconds; chosen, generous
ARM_TIMEOUT_S = 120.0  # the estimator settles on GPS in ~20 s at 1x; chosen, generous
ARM_RETRY_S = 2.0  # chosen, no evidence
TAKEOFF_TIMEOUT_S = 60.0  # a 30 m climb takes ~15 s; chosen, generous
ACK_TIMEOUT_S = 5.0  # chosen, no evidence
LAND_TIMEOUT_S = 120.0  # descent plus PX4's auto-disarm on the ground; chosen, generous
STATUS_PERIOD_S = 10.0  # chosen, no evidence
ALT_TOLERANCE_M = 1.0  # PX4's takeoff stops a little short of the target; chosen
MM_PER_M = 1000.0
DEG_E7 = 1e7  # MAVLink carries latitude and longitude as degrees * 1e7
EARTH_RADIUS_M = 6_371_000.0  # mean radius (IUGG); metres per degree near the ground is all it is for
MAV_CMD_DO_ORBIT = 34  # MAVLink common.xml
ORBIT_YAW_FRONT_TO_CENTRE = 0  # PX4 msg/OrbitStatus.msg HOLD_FRONT_TO_CIRCLE_CENTER
WORLD_FRAME = "map"
TF_RATE_HZ = 20.0  # chosen: smooth in a 3D view; PX4 sends odometry at 100 Hz (dds_topics.yaml)
TF_QUEUE_DEPTH = 10  # rclpy's customary history depth
SQRT_HALF = math.sqrt(0.5)  # cos and sin of 45 degrees: NED->ENU and FRD->FLU are both half-turns


# --------------------------------------------------------------------------- shared


def local_address() -> str:
    """Return this machine's own IPv4 address, the one the simulator must aim a stream at.

    A connected UDP socket picks the source address the kernel would use to leave, without
    sending anything. Under WSL this changes across restarts, so it is read at run time.
    """
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return str(s.getsockname()[0])
    finally:
        s.close()


def write_sdp(host: str, port: int) -> Path:
    """Write the session description a decoder needs to open a bare RTP/H.264 stream.

    Payload type 96 says nothing on its own and ffmpeg takes such a description only from a
    file, so the stream cannot be opened without one.

    Args:
    ----
        host: Address the stream is sent to, which the description also names.
        port: RTP port to receive on.

    Returns:
    -------
        Path of the SDP file written into the temporary directory.

    """
    path = Path(tempfile.gettempdir()) / f"pterosim_stream_{port}.sdp"
    path.write_text(
        "\n".join(
            [
                "v=0",
                f"o=- 0 0 IN IP4 {host}",
                "s=PteroSim camera",
                f"c=IN IP4 {host}",
                "t=0 0",
                f"m=video {port} RTP/AVP 96",
                "a=rtpmap:96 H264/90000",
            ]
        )
        + "\n"
    )
    return path


def open_stream(host: str, port: int) -> Any:
    """Open one aircraft's RTP/H.264 stream as an ffmpeg container.

    format="sdp" is required -- by extension alone ffmpeg does not pick the SDP demuxer --
    and protocol_whitelist is what lets that demuxer open the nested rtp/udp stream, the same
    option drone_camera_stream.py sets for OpenCV.

    Args:
    ----
        host: Address the stream is sent to.
        port: RTP port to receive on.

    Returns:
    -------
        An av container holding the video stream.

    """
    import av

    # Nothing is decoded here, so no decoder options -- only ask the demuxer not to buffer or
    # reorder packets: a live preview is worth more than a packet-perfect order.
    return av.open(
        str(write_sdp(host, port)),
        format="sdp",
        options={
            "protocol_whitelist": "file,rtp,udp",
            "fflags": "nobuffer",
            "reorder_queue_size": "0",
        },
    )


def ring(count: int, radius: float) -> list[tuple[float, float, float]]:
    """Space aircraft evenly on a circle, every one facing its centre.

    That is the formation that shows the most: each camera looks across the centre at all the
    others, instead of down a line where the trailing ones would only ever see a tail.

    Args:
    ----
        count: How many aircraft.
        radius: Distance from the centre, in metres.

    Returns:
    -------
        One (x, y, yaw) per aircraft, yaw in Unreal's sense: degrees clockwise from +X.

    """
    out = []
    for n in range(count):
        a = math.tau * n / count
        x, y = radius * math.cos(a), radius * math.sin(a)
        # atan2 of the vector to the centre is the heading that faces it.
        out.append((x, y, math.degrees(math.atan2(-y, -x))))
    return out


def parse_tile(text: str) -> tuple[int, int]:
    """Split a WIDTHxHEIGHT tile description.

    Args:
    ----
        text: Tile size such as "768x480".

    Returns:
    -------
        The width and the height.

    """
    w, h = text.lower().split("x")
    return int(w), int(h)


def parse_layout(text: str) -> list[tuple[float, float, float]]:
    """Parse a hand-placed formation: semicolon-separated x,y,yaw triples in degrees.

    This is the escape hatch from the generated formations -- whatever arrangement was set up
    by hand can be written down once and replayed exactly.

    Args:
    ----
        text: Triples such as "2.5,0,180; -2.5,0,0".

    Returns:
    -------
        One (x, y, yaw) per aircraft.

    """
    out: list[tuple[float, float, float]] = []
    for raw in text.split(";"):
        entry = raw.strip()
        if not entry:
            continue
        fields = entry.split(",")
        if len(fields) != LAYOUT_FIELDS:
            raise ValueError(f"layout entry {entry!r} needs x,y,yaw")
        out.append((float(fields[0]), float(fields[1]), float(fields[2])))
    if not out:
        raise ValueError("layout is empty")
    return out


def import_pterosim() -> Any:
    """Import the PteroSim SDK's client class: from $PTEROSIM_SDK if set, else the installed package.

    Returns
    -------
        The PteroSim class.

    """
    sdk = os.environ.get("PTEROSIM_SDK")
    if sdk:
        sys.path.insert(0, sdk)
    from pterosim import PteroSim

    return PteroSim


def vehicle_ns(stream: dict[str, Any]) -> str:
    """ROS namespace of one aircraft in the manifest, shared by its camera, its PX4 and its tf frame."""
    return f"{stream['aircraft']}_{stream['instance_id']}"


def metres_per_degree(lat: float) -> tuple[float, float]:
    """Metres per degree of latitude and of longitude at a latitude, on a spherical Earth."""
    north = math.radians(1.0) * EARTH_RADIUS_M
    return north, north * math.cos(math.radians(lat))


def enu_flu_pose(ned: Any, q: Any) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Turn a PX4 pose into ROS conventions.

    Args:
    ----
        ned: Position, north-east-down, in metres.
        q: PX4's quaternion (w, x, y, z) from the forward-right-down body to NED.

    Returns:
    -------
        Position east-north-up, and the quaternion (x, y, z, w) from the forward-left-up body to ENU.

    """
    n, e, d = (float(v) for v in ned)
    w, x, y, z = (float(v) for v in q)
    # q_ENU<-NED * q * q_FRD<-FLU, with both half-turns multiplied out.
    return (e, n, -d), (SQRT_HALF * (x + y), SQRT_HALF * (x - y), SQRT_HALF * (w - z), SQRT_HALF * (w + z))


def ensure_ros() -> None:
    """Re-exec under a sourced ROS 2 environment if rclpy is not importable yet.

    rclpy lives in the ROS install's site-packages, which only its setup script puts on
    PYTHONPATH -- a --system-site-packages virtualenv does not see it. Asking the reader to
    remember a source line is the one thing a file meant to be handed to someone else
    should not depend on, so do it here. The guard variable keeps a failing source from
    turning into a re-exec loop.
    """
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pass
    else:
        return

    if os.environ.get("PTEROSIM_CAMERAS_REEXEC"):
        raise SystemExit("rclpy still not importable after sourcing ROS 2")

    setups = sorted(Path("/opt/ros").glob("*/setup.bash"))
    if not setups:
        raise SystemExit(
            "rclpy not found and no /opt/ros/*/setup.bash to source; " "source your ROS 2 install and retry"
        )
    # sys.argv[1:], not sys.argv: argv[0] is this script, which is already named by __file__.
    argv = " ".join(f'"{a}"' for a in sys.argv[1:])
    script = f'source "{setups[-1]}" && exec "{sys.executable}" "{os.path.abspath(__file__)}" {argv}'
    os.environ["PTEROSIM_CAMERAS_REEXEC"] = "1"
    os.execv("/bin/bash", ["/bin/bash", "-c", script])


# --------------------------------------------------------------------------- setup


def cmd_setup(args: argparse.Namespace) -> int:
    """Spawn the fleet, point every camera at this machine, and start the simulation.

    Args:
    ----
        args: Parsed command line, holding the fleet size, tile, fps and addresses.

    Returns:
    -------
        Zero on success.

    """
    PteroSim = import_pterosim()

    width, height = parse_tile(args.tile)
    host = args.host or local_address()
    print(f"streaming to {host}, cameras at {width}x{height}@{args.fps:g}")

    sim = PteroSim(args.address)
    try:
        if sim.status().is_running:
            # Sensor params and add_sensor are only accepted before start().
            print("stopping the simulation...")
            sim.stop()

        have = sorted(a.instance_id for a in sim.aircraft_status())
        print(f"aircraft in the scene: {have or 'none'}")

        if args.respawn and have:
            # The SDK only takes a pose at spawn time, so a new formation means new aircraft.
            for i in have:
                print(f"removing instance {i} for a new formation")
                sim.get_aircraft(i).remove()
            have = []

        poses = parse_layout(args.layout) if args.layout else ring(args.count, args.spacing)
        while len(have) < args.count:
            n = len(have)
            px, py, yaw = poses[n]
            # z is only where the ground trace starts looking from: a spawn always sits on the ground.
            drone = sim.spawn(args.aircraft, x=px * CM_PER_M, y=py * CM_PER_M, z=0.0, yaw=yaw)
            print(
                f"spawned {args.aircraft} instance_id={drone.instance_id} at ({px:+.1f}, {py:+.1f}) yaw {yaw:+.0f} deg"
            )
            have = sorted(set(have) | {drone.instance_id})
        have = have[: args.count]

        streams = []
        for i in have:
            drone = sim.get_aircraft(i)
            cameras = [s for s in drone.list_sensors() if s.type == "camera"]
            if not cameras:
                # A freshly spawned aircraft carries whatever its Sensors.xml defines, which
                # is not necessarily a camera, so this cannot assume a name is present.
                name = drone.add_sensor("camera")
                print(f"  instance {i}: no camera aboard, added {name!r}")
                cameras = [s for s in drone.list_sensors() if s.type == "camera"]
            cam = next((c for c in cameras if c.name == args.camera), cameras[0])
            # The mount's zero is level. Taken from the autopilot, it stays there, so the cameras
            # keep looking at each other rather than wherever the flight controller parks them.
            drone.set_gimbal_source("api")

            # stream_port is per aircraft, so the real port steps by two.
            stream_port = args.base_port + i
            drone.set_sensor_param(
                cam.name,
                update_hz=args.fps,
                image_width=width,
                image_height=height,
                stream=True,
                stream_port=stream_port,
                stream_host=host,
            )
            stream = {
                "instance_id": i,
                "aircraft": getattr(drone, "aircraft_name", None) or args.aircraft,
                "camera": cam.name,
                "port": stream_port + i,
            }
            topic = stream["topic"] = f"/{vehicle_ns(stream)}/{cam.name}/compressed_video"
            streams.append(stream)
            print(f"  instance {i}: {cam.name} -> udp://{host}:{stream_port + i}  {topic}")

        sim.start()
    finally:
        # Channel close only. shutdown() would take the whole simulator down.
        sim.close()

    Path(MANIFEST).write_text(json.dumps({"host": host, "tile": [width, height], "streams": streams}, indent=2))
    print(f"\nsimulation started; wrote {MANIFEST}")
    return 0


# ----------------------------------------------------------------------------- fly


def launch_px4(px4_root: Path, instance: int, airframe: int, namespace: str) -> Any:
    """Start one PX4 SITL instance that flies the simulator's aircraft of that instance id.

    Args:
    ----
        px4_root: The PX4-Autopilot checkout, built with `make px4_sitl_default`.
        instance: The aircraft's instance id; PX4 then dials HIL on 4560 + instance.
        airframe: SYS_AUTOSTART id of the vehicle's PX4 airframe.
        namespace: ROS namespace of the aircraft's uXRCE-DDS topics.

    Returns:
    -------
        The running process; its console goes to rootfs/<instance>/px4.log.

    """
    import subprocess

    build = px4_root / "build" / "px4_sitl_default"
    rootfs = build / "rootfs" / str(instance)
    rootfs.mkdir(parents=True, exist_ok=True)
    # A parameter store left by another airframe would leak its gains into this one.
    for store in ("parameters.bson", "parameters_backup.bson"):
        (rootfs / store).unlink(missing_ok=True)
    # No PX4_SIM_MODEL: that selects PX4's own simulators. The simulator is the Windows process,
    # which mirrored WSL reaches on loopback.
    env = {k: v for k, v in os.environ.items() if k != "PX4_SIM_MODEL"}
    # rcS would name the topics px4_<i>, and leave instance 0 with none at all.
    env.update(PX4_SIM_HOSTNAME="127.0.0.1", PX4_SYS_AUTOSTART=str(airframe), PX4_UXRCE_DDS_NS=namespace)
    with (rootfs / "px4.log").open("w") as log:
        # A session of its own: Ctrl+C in this terminal must reach this script, which lands the
        # fleet, and not the autopilots, which would quit in the air.
        return subprocess.Popen(
            [str(build / "bin" / "px4"), "-i", str(instance), "-d", str(build / "etc")],
            cwd=rootfs,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def cmd_fly(args: argparse.Namespace) -> int:
    """Fly the fleet: take off, then orbit the fleet's centre with every nose on that centre.

    Aircraft on one circle at one speed keep their spacing, so with every nose held on the centre
    each camera keeps the others in view for as long as the orbit runs. Ctrl+C lands.

    Args:
    ----
        args: Parsed command line, holding the manifest, the PX4 checkout, altitude and speed.

    Returns:
    -------
        Zero once the fleet has landed.

    """
    from pymavlink import mavutil

    px4_root = Path(args.px4).expanduser()
    if not list((px4_root / "build/px4_sitl_default/etc/init.d-posix/airframes").glob(f"{args.airframe}_*")):
        raise SystemExit(
            f"airframe {args.airframe} is not in {px4_root}/build/px4_sitl_default/etc -- copy the vehicle's "
            f"firmwares/px4_* there as {args.airframe}_<name>, through `tr -d '\\r'` if it comes from a "
            "Windows checkout"
        )
    streams = json.loads(Path(args.manifest).read_text())["streams"]
    ids = [s["instance_id"] for s in streams]
    procs: dict[int, Any] = {}
    links: dict[int, Any] = {}
    next_beat = 0.0

    def pump(seconds: float, done: Any) -> bool:
        """Keep every link alive and read, until done() holds or the time is up.

        One thread for all links: pymavlink keeps the latest message of each type in
        link.messages, so reading is all the bookkeeping there is.

        Args:
        ----
            seconds: How long to wait at most.
            done: Predicate checked between reads.

        Returns:
        -------
            Whether done() came true in time.

        """
        nonlocal next_beat
        end = time.time() + seconds
        while not done():
            now = time.time()
            if now > end:
                return False
            for i, proc in procs.items():
                if proc.poll() is not None:
                    raise SystemExit(f"PX4 instance {i} exited with {proc.returncode}; see rootfs/{i}/px4.log")
            if now >= next_beat:
                # PX4 answers the first address that talks to it, and refuses to arm without a GCS.
                for link in links.values():
                    link.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
                    )
                next_beat = now + HEARTBEAT_PERIOD_S
            for i, link in links.items():
                while (msg := link.recv_msg()) is not None:
                    if msg.get_type() == "STATUSTEXT" and msg.severity <= mavutil.mavlink.MAV_SEVERITY_WARNING:
                        print(f"  px4 {i}: {msg.text}")
            time.sleep(POLL_PERIOD_S)
        return True

    def must(seconds: float, done: Any, what: str) -> None:
        """Pump until done() holds, or stop the run naming what never happened."""
        if not pump(seconds, done):
            raise SystemExit(f"timed out after {seconds:g}s waiting for {what}")

    def position(link: Any) -> Any:
        """Latest GLOBAL_POSITION_INT from a link, or None before the first."""
        return link.messages.get("GLOBAL_POSITION_INT")

    try:
        for s in streams:
            i = s["instance_id"]
            procs[i] = launch_px4(px4_root, i, args.airframe, vehicle_ns(s))
            links[i] = mavutil.mavlink_connection(f"udpout:127.0.0.1:{PX4_API_PORT + i}", source_system=GCS_SYSTEM_ID)
        print(f"PX4 x{len(ids)} starting; each dials HIL on 127.0.0.1:{HIL_BASE_PORT + ids[0]}..")
        # sysid locks onto the first autopilot heartbeat; the simulator's camera heartbeats, which PX4
        # forwards on this link too, do not count.
        must(PX4_BOOT_TIMEOUT_S, lambda: all(link.sysid for link in links.values()), "PX4 heartbeats")

        for link in links.values():
            link.param_set_send("MIS_TAKEOFF_ALT", args.alt, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
            # set_mode() would guess the autopilot from the last HEARTBEAT, which may be that camera's.
            link.set_mode_px4(*mavutil.px4_map["TAKEOFF"])
        # Pre-arm checks refuse until the estimator has settled on GPS, so ask again until it takes.
        print(f"arming, then taking off to {args.alt:g} m")

        def armed() -> bool:
            return all(link.motors_armed() for link in links.values())

        # From the first arm on, Ctrl+C means land, not kill the autopilots in the air.
        try:
            end = time.time() + ARM_TIMEOUT_S
            while not armed():
                if time.time() > end:
                    raise SystemExit(f"not armed after {ARM_TIMEOUT_S:g}s; the px4 lines above say why")
                for link in links.values():
                    if not link.motors_armed():
                        link.arducopter_arm()
                pump(ARM_RETRY_S, armed)

            top = (args.alt - ALT_TOLERANCE_M) * MM_PER_M
            must(
                TAKEOFF_TIMEOUT_S,
                lambda: all((p := position(link)) and p.relative_alt >= top for link in links.values()),
                "the climb",
            )

            # The centre is where the fleet is, not where it was asked to be: spawns snap to the ground.
            fixes = [position(link) for link in links.values()]
            lat0 = sum(p.lat for p in fixes) / len(fixes) / DEG_E7
            lon0 = sum(p.lon for p in fixes) / len(fixes) / DEG_E7
            amsl = sum(p.alt for p in fixes) / len(fixes) / MM_PER_M
            north_m, east_m = metres_per_degree(lat0)
            offsets = [math.hypot((p.lat / DEG_E7 - lat0) * north_m, (p.lon / DEG_E7 - lon0) * east_m) for p in fixes]
            radius = sum(offsets) / len(offsets)
            print(f"orbiting {lat0:.7f},{lon0:.7f} at {amsl:.1f} m AMSL, radius {radius:.1f} m, {args.speed:g} m/s")
            for link in links.values():
                link.messages.pop("COMMAND_ACK", None)
            # All at once and alike, so the ring they started in is the ring they keep.
            for link in links.values():
                link.mav.command_int_send(
                    link.target_system,
                    link.target_component,
                    mavutil.mavlink.MAV_FRAME_GLOBAL,
                    MAV_CMD_DO_ORBIT,
                    0,
                    0,
                    radius,
                    args.speed,
                    ORBIT_YAW_FRONT_TO_CENTRE,
                    0,
                    round(lat0 * DEG_E7),
                    round(lon0 * DEG_E7),
                    amsl,
                )
            acked = lambda: all(  # noqa: E731
                (a := link.messages.get("COMMAND_ACK")) and a.command == MAV_CMD_DO_ORBIT for link in links.values()
            )
            must(ACK_TIMEOUT_S, acked, "the orbit to be acknowledged")
            refused = [
                i
                for i, link in links.items()
                if link.messages["COMMAND_ACK"].result != mavutil.mavlink.MAV_RESULT_ACCEPTED
            ]
            if refused:
                raise SystemExit(f"orbit refused by instance(s) {refused}")

            print("orbiting; Ctrl+C lands")
            while True:
                pump(STATUS_PERIOD_S, lambda: False)
                line = "  ".join(
                    f"{i}: {p.relative_alt / MM_PER_M:4.1f} m {math.hypot(p.vx, p.vy) / CM_PER_M:3.1f} m/s"
                    for i, link in links.items()
                    if (p := position(link))
                )
                print(line)
        except KeyboardInterrupt:
            pass

        print("landing")
        for link in links.values():
            link.set_mode_px4(*mavutil.px4_map["LAND"])
        must(LAND_TIMEOUT_S, lambda: not any(link.motors_armed() for link in links.values()), "the landing")
    finally:
        for proc in procs.values():
            proc.terminate()
        for proc in procs.values():
            proc.wait()
    return 0


# -------------------------------------------------------------------------- bridge


def cmd_bridge(args: argparse.Namespace) -> int:
    """Publish each stream in the manifest as CompressedVideo, and each PX4's pose on /tf.

    Args:
    ----
        args: Parsed command line, holding the manifest path and report period.

    Returns:
    -------
        Zero on success.

    """
    import rclpy
    from foxglove_msgs.msg import CompressedVideo
    from geometry_msgs.msg import TransformStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from tf2_msgs.msg import TFMessage

    try:
        from px4_msgs.msg import VehicleLocalPosition, VehicleOdometry
    except ImportError:
        raise SystemExit(
            "px4_msgs not found: set it up as PX4's ROS 2 User Guide says, and source its workspace"
        ) from None

    m = json.loads(Path(args.manifest).read_text())
    host, pairs = m["host"], [(s["port"], s["topic"]) for s in m["streams"]]
    names = [vehicle_ns(s) for s in m["streams"]]

    class Stream(threading.Thread):
        """Forwards one aircraft's H.264 access units, undecoded, one message per frame."""

        def __init__(self, node: Any, host: str, port: int, topic: str) -> None:
            super().__init__(daemon=True)
            self.node = node
            self.port = port
            self.topic = topic
            self.frames = 0
            self.error = ""
            self.pub = node.create_publisher(CompressedVideo, topic, qos_profile_sensor_data)
            # open_stream blocks until the stream's first keyframe, with no timeout of its own.
            node.get_logger().info(f"{topic}: waiting for the stream on udp port {port}")
            self.container = open_stream(host, port)
            self.vstream = self.container.streams.video[0]
            self.size = f"{self.vstream.width}x{self.vstream.height}"

        def run(self) -> None:
            try:
                # The RTP demuxer hands over Annex B access units, and the simulator puts SPS/PPS
                # ahead of every keyframe with no B-frames -- exactly what Foxglove decodes.
                for packet in self.container.demux(self.vstream):
                    if packet.size == 0:
                        continue
                    msg = CompressedVideo()
                    msg.timestamp = self.node.get_clock().now().to_msg()
                    msg.frame_id = self.topic.strip("/").replace("/", "_")
                    msg.format = "h264"
                    # array.array, not bytes: Humble's uint8[] setter checks a plain sequence
                    # element by element, and rejects an ndarray outright.
                    msg.data = array.array("B", bytes(packet))
                    self.pub.publish(msg)
                    self.frames += 1
            except Exception as exc:
                # One lost stream must not take the other aircraft down with it. Deliberately
                # broad: anything ffmpeg or the transport raises ends the stream, and the
                # other aircraft's threads have to survive it.
                self.error = f"{type(exc).__name__}: {exc}"
                self.node.get_logger().error(f"{self.topic}: {self.error}")

        def close(self) -> None:
            """Close the underlying container."""
            self.container.close()

    rclpy.init()
    # A plain node rather than a subclass: rclpy ships no type stubs, so a base class of
    # type Any is exactly what mypy --strict refuses to subclass. The report state belongs
    # to this function anyway.
    node = Node("pterosim_camera")
    streams = [Stream(node, host, port, topic) for port, topic in pairs]
    for s in streams:
        s.start()
        node.get_logger().info(f"publishing {s.topic}  {s.size} from udp port {s.port}")

    # Rate over the last report period: an average since start would carry the stream's
    # start-up wait forever and read low.
    last = {s.topic: (s.frames, time.time()) for s in streams}

    # PX4's own estimate of every aircraft, latest of each, and where each estimator started.
    odometry: dict[str, Any] = {}
    origins: dict[str, tuple[float, float, float]] = {}
    heard: set[str] = set()  # aircraft whose PX4 sent odometry since the last report

    def on_odometry(ns: str, msg: Any) -> None:
        """Keep an aircraft's latest PX4 odometry, and when it arrived, until the next tf tick sends it."""
        odometry[ns] = (msg, node.get_clock().now().to_msg())
        heard.add(ns)

    def px4_topic(ns: str, name: str, msg_type: Any) -> str:
        """Name PX4's uXRCE-DDS client publishes a topic under: versioned messages carry _v<N>."""
        # px4_msgs before PX4 1.16 has no MESSAGE_VERSION; PX4 reads a missing one as 0 too.
        version = getattr(msg_type, "MESSAGE_VERSION", 0)
        return f"/{ns}/fmu/out/{name}" + (f"_v{version}" if version else "")

    def on_origin(ns: str, msg: Any) -> None:
        """Keep the global position of an aircraft's local origin, once its estimator has one."""
        if msg.xy_global and msg.z_global:
            origins[ns] = (msg.ref_lat, msg.ref_lon, msg.ref_alt)

    for ns in names:
        node.create_subscription(
            VehicleOdometry,
            px4_topic(ns, "vehicle_odometry", VehicleOdometry),
            functools.partial(on_odometry, ns),
            qos_profile_sensor_data,
        )
        node.create_subscription(
            VehicleLocalPosition,
            px4_topic(ns, "vehicle_local_position", VehicleLocalPosition),
            functools.partial(on_origin, ns),
            qos_profile_sensor_data,
        )

    def report() -> None:
        """Log one line per stream with its rate over the last period, and what PX4 is missing."""
        for s, ns in zip(streams, names, strict=True):
            frames, since = last[s.topic]
            now = time.time()
            hz = (s.frames - frames) / (now - since)
            last[s.topic] = (s.frames, now)
            line = f"{s.topic}  {s.frames} frames  {hz:5.1f} Hz  {s.size}"
            if ns not in heard:
                line += "  no PX4 odometry this period"
            elif ns not in origins:
                line += "  no PX4 origin yet"
            node.get_logger().info(line + (f"  ERROR {s.error}" if s.error else ""))
        heard.clear()

    node.create_timer(args.report, report)
    tf_pub = node.create_publisher(TFMessage, "/tf", TF_QUEUE_DEPTH)

    def publish_poses() -> None:
        """Publish map -> <aircraft>_<id>/base_link for every aircraft PX4 has a valid pose for."""
        # map's origin is the fleet's mean start, so it waits for every aircraft's origin.
        if len(origins) < len(names):
            return
        lat0, lon0, alt0 = (sum(o[k] for o in origins.values()) / len(origins) for k in range(3))
        north_m, east_m = metres_per_degree(lat0)
        out = TFMessage()
        # Each sample goes out once, so a PX4 that goes quiet leaves its tf stale instead of re-sent.
        for ns in list(odometry):
            odom, arrived = odometry.pop(ns)
            # NaN marks an estimate PX4 does not have yet.
            if math.isnan(odom.position[0]) or math.isnan(odom.q[0]):
                continue
            (x, y, z), (qx, qy, qz, qw) = enu_flu_pose(odom.position, odom.q)
            lat, lon, alt = origins[ns]
            t = TransformStamped()
            # Arrival, like the video: under lockstep PX4's synced stamp swung -240..+575 ms here.
            t.header.stamp = arrived
            t.header.frame_id = WORLD_FRAME
            t.child_frame_id = f"{ns}/base_link"
            t.transform.translation.x = (lon - lon0) * east_m + x
            t.transform.translation.y = (lat - lat0) * north_m + y
            t.transform.translation.z = alt - alt0 + z
            r = t.transform.rotation
            r.x, r.y, r.z, r.w = qx, qy, qz, qw
            out.transforms.append(t)
        if out.transforms:
            tf_pub.publish(out)

    node.create_timer(1.0 / TF_RATE_HZ, publish_poses)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for s in streams:
            s.close()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


# --------------------------------------------------------------------------- foxglove


def cmd_foxglove(args: argparse.Namespace) -> int:
    """Serve the camera topics to a Foxglove client over a WebSocket.

    Runs the foxglove_bridge package the same way the ROSCon PX4 workshop does: the client
    connects as a Foxglove WebSocket, not over DDS, which is what makes it reachable from a
    host that is not on the same ROS 2 network.

    Args:
    ----
        args: Parsed command line, holding the port and the topic whitelist.

    Returns:
    -------
        The bridge's exit status.

    """
    import shutil
    import subprocess

    if shutil.which("ros2") is None:
        print(
            "ros2 is not on PATH; source /opt/ros/<distro>/setup.bash first",
            file=sys.stderr,
        )
        return 1

    # ament installs the binary into lib/<package>/, which it does not add to PATH, so
    # `ros2 run` is the only portable way to reach it.
    probe = subprocess.run(
        ["ros2", "pkg", "executables", "foxglove_bridge"],
        capture_output=True,
        text=True,
        check=False,
    )
    if "foxglove_bridge" not in probe.stdout:
        print(
            "the foxglove_bridge package is not installed.\n"
            "Install it with:\n"
            f"  sudo apt install ros-{os.environ['ROS_DISTRO']}-foxglove-bridge",
            file=sys.stderr,
        )
        return 1

    # The bridge reads its settings only as ROS parameters -- plain --port/--topics arguments are
    # silently ignored. The whitelist keeps whatever else is on the bus out of the client.
    cmd = [
        "ros2",
        "run",
        "foxglove_bridge",
        "foxglove_bridge",
        "--ros-args",
        "-p",
        f"port:={args.port}",
        "-p",
        f"topic_whitelist:=['{args.topics}']",
    ]
    print(f"serving {args.topics} on {FOXGLOVE_URL}")
    print(
        "in Foxglove: connect with 'Foxglove WebSocket' and that URL, then open "
        "python_examples/pterosim_cameras_foxglove.json"
    )
    try:
        return subprocess.run(cmd, check=False).returncode
    except KeyboardInterrupt:
        return 0


# -------------------------------------------------------------------------- status


def cmd_status(args: argparse.Namespace) -> int:
    """Report what is publishing right now, and at what rate.

    Args:
    ----
        args: Parsed command line, holding the measurement window in seconds.

    Returns:
    -------
        Zero if any frames arrived, one otherwise.

    """
    import rclpy
    from foxglove_msgs.msg import CompressedVideo
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data

    rclpy.init()
    node = Node("pterosim_camera_status")
    # topic -> [(arrival time, payload bytes), ...]
    seen: dict[str, list[tuple[float, int]]] = {}
    subs: list[Any] = []

    def on_frame(topic: str, msg: Any) -> None:
        """Record a frame's arrival time and size.

        Args:
        ----
            topic: Topic the frame arrived on.
            msg: The video message.

        """
        seen.setdefault(topic, []).append((time.time(), len(msg.data)))

    end = time.time() + args.seconds
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.05)

    for name, types in node.get_topic_names_and_types():
        if VIDEO_TYPE in types:
            subs.append(
                node.create_subscription(
                    CompressedVideo, name, functools.partial(on_frame, name), qos_profile_sensor_data
                )
            )

    if not subs:
        print(f"no {VIDEO_TYPE} topics on the bus -- is the bridge running?")
        rclpy.shutdown()
        return 1

    # Measure over a clean window, after discovery has settled.
    seen.clear()
    end = time.time() + args.seconds
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.02)

    total = 0
    for name in sorted(seen):
        fr = seen[name]
        total += len(fr)
        span = max(1e-6, fr[-1][0] - fr[0][0])
        kbps = sum(n for _, n in fr) * 8 / 1000 / span
        print(f"{name}  {len(fr):>4} frames  {len(fr)/span:5.1f} Hz  {kbps:6.0f} kbit/s")
    print(f"\ntotal {total} frames in {args.seconds:g}s across {len(seen)} topic(s)")

    node.destroy_node()
    rclpy.shutdown()
    return 0 if total else 1


# ---------------------------------------------------------------------------- main


def main() -> int:
    """Parse the command line and run the requested subcommand.

    Returns
    -------
        The subcommand's exit status.

    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="spawn the fleet and point the cameras here")
    s.add_argument("--count", type=int, default=4)
    s.add_argument("--aircraft", default="x500")
    s.add_argument("--camera", default="gimbal_camera", help="preferred camera; falls back to any camera aboard")
    s.add_argument("--tile", default=DEFAULT_TILE, help="camera resolution, e.g. 768x480")
    s.add_argument("--fps", type=float, default=30.0)
    s.add_argument("--base-port", type=int, default=5600)
    s.add_argument("--host", default=None, help="address the cameras stream to; auto-detected by default")
    s.add_argument("--address", default=GRPC_ADDRESS, help="PteroSim gRPC host:port")
    s.add_argument(
        "--spacing", type=float, default=2.5, help="distance from the ring's centre to each aircraft, in metres"
    )
    s.add_argument(
        "--layout",
        default=None,
        help="hand-placed formation as x,y,yaw triples in metres and degrees, e.g. '-10,-10,45; 10,-10,135'",
    )
    s.add_argument("--respawn", action="store_true", help="remove existing aircraft first, to pick up a new formation")
    s.set_defaults(func=cmd_setup)

    s = sub.add_parser("fly", help="start PX4 for each aircraft, take off and orbit, nose to the centre")
    s.add_argument("--manifest", default=MANIFEST)
    s.add_argument("--px4", required=True, help="PX4-Autopilot checkout with a built px4_sitl_default")
    s.add_argument("--airframe", type=int, default=PX4_AIRFRAME, help="SYS_AUTOSTART of the vehicle's PX4 airframe")
    s.add_argument("--alt", type=float, default=30.0, help="takeoff altitude above the ground, in metres")
    s.add_argument("--speed", type=float, default=1.0, help="orbit speed in m/s; PX4 caps it at sqrt(2 * radius)")
    s.set_defaults(func=cmd_fly)

    s = sub.add_parser("bridge", help="publish the streams as CompressedVideo topics, and PX4's poses as /tf")
    s.add_argument("--manifest", default=MANIFEST)
    s.add_argument("--report", type=float, default=10.0)
    s.set_defaults(func=cmd_bridge)

    s = sub.add_parser("foxglove", help="serve the camera topics to Foxglove")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--topics", default="/x500_[0-9]+/.*compressed_video|/tf", help="topic whitelist for the bridge")
    s.set_defaults(func=cmd_foxglove)

    s = sub.add_parser("status", help="what is publishing, and at what rate")
    s.add_argument("--seconds", type=float, default=10.0)
    s.set_defaults(func=cmd_status)

    args = p.parse_args()
    if args.cmd in ("bridge", "foxglove", "status"):
        # Set before the re-exec, so the sourced environment inherits it.
        os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = FASTDDS_PROFILE
        ensure_ros()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
