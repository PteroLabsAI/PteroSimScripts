#!/usr/bin/env python3
"""PteroSim drone cameras -> ROS 2 -> Foxglove, in one file.

Publishes each aircraft's camera to ROS 2 as sensor_msgs/msg/Image. Viewing is left to
Foxglove, so there is no viewer here. Four steps, because the simulator only accepts sensor
settings while it is stopped:

    pterosim_cameras.py setup     spawn the fleet, point the cameras at this machine
    pterosim_cameras.py bridge    decode the streams and publish them as Image topics
    pterosim_cameras.py foxglove  serve those topics to Foxglove over a WebSocket
    pterosim_cameras.py status    what is publishing, and at what rate

`setup` puts four x500 in the scene and streams each gimbal_camera over RTP/H.264 to this
machine. `bridge` decodes those streams and publishes one Image topic per aircraft.
`foxglove` runs foxglove_bridge so the Foxglove client can connect to ws://localhost:8765.
`status` reports the frame rate on each topic.

Foxglove needs no image transcoding: its Image panel subscribes to a
sensor_msgs/msg/Image topic directly, and the rgb8 frames published here are exactly that.
python_examples/pterosim_cameras_foxglove.json is a ready 2x2 layout for four aircraft --
import it in Foxglove instead of arranging the panels by hand.

WHY IT LOOKS THE WAY IT DOES

The simulator already has a camera pipeline: its own encoder sends RTP/H.264 to
udp://<host>:<stream_port + instance_id>, the same stream a ground station displays. This
consumes that stream instead of pulling frames over gRPC, which is what
PteroSimScripting/python_examples/drone_camera_display.py does. That matters: a gRPC pull is
a synchronous GPU readback on the sim's render path, and one at four aircraft and ~10 Hz each
-- roughly 37 MB/s -- is enough to take PteroSim down. Decoding a push stream touches nothing
in the simulator, so there is no such ceiling.

Two details that are easy to get wrong:

  * Ports step by two, not one. ffmpeg binds port+1 alongside every RTP port for RTCP, so a
    decoder on 5600 also holds 5601 -- which would be the next aircraft's own stream. The SDK
    derives the port as stream_port + instance_id, so giving each aircraft
    stream_port = BASE + instance_id lands the real ports BASE + 2*instance_id and the RTCP
    ports fall on free ones.

  * The SDP file needs format="sdp" and protocol_whitelist. Payload type 96 says nothing on
    its own, ffmpeg takes such a description only from a file, and without the whitelist it
    cannot open the nested rtp/udp stream at all.

The cameras are configured for 768x480, which is half a 1536x960 screen in each direction, so
four of them tile it in the 2x2 Foxglove layout with no scaling. --tile changes the camera
resolution, and the layout's panels to match.

    4 x 768x480 at 30 fps source, ~20 Hz per stream delivered on one decode process.

Requires: ROS 2 (rclpy, sensor_msgs), PyAV, and the PteroSim SDK for `setup`. `bridge` and
`status` need PyAV; `foxglove` needs the foxglove_bridge package:

    sudo apt install ros-jazzy-foxglove-bridge
"""

import argparse
import functools
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

DEFAULT_TILE = "768x480"
MANIFEST = str(Path.home() / "camera_fleet.json")
GRPC_ADDRESS = "172.26.48.1:10011"
FOXGLOVE_URL = "ws://localhost:8765"


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

    return av.open(str(write_sdp(host, port)), format="sdp", options={"protocol_whitelist": "file,rtp,udp"})


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
    sdk = os.environ.get("PTEROSIM_SDK", "")
    if not sdk:
        default = "/mnt/c/Users/Yollnahkriin/Documents/Unreal_Projects/PteroSim/" "Plugins/PteroSimScripting/SDK/python"
        sdk = default if Path(default).exists() else ""
    if sdk:
        sys.path.insert(0, sdk)
    from pterosim import PteroSim

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
        while len(have) < args.count:
            n = len(have)
            drone = sim.spawn(args.aircraft, x=-492.0 + 8.0 * n, y=-199.0, z=30.0, yaw=0.0)
            print(f"spawned {args.aircraft} instance_id={drone.instance_id}")
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
            aircraft = getattr(drone, "aircraft_name", None) or args.aircraft
            topic = f"/{aircraft}_{i}/{cam.name}/image_raw"
            streams.append(
                {"instance_id": i, "aircraft": aircraft, "camera": cam.name, "port": stream_port + i, "topic": topic}
            )
            print(f"  instance {i}: {cam.name} -> udp://{host}:{stream_port + i}  {topic}")

        sim.start()
    finally:
        # Channel close only. shutdown() would take the whole simulator down.
        sim.close()

    Path(MANIFEST).write_text(json.dumps({"host": host, "tile": [width, height], "streams": streams}, indent=2))
    print(f"\nsimulation started; wrote {MANIFEST}")
    return 0


# -------------------------------------------------------------------------- bridge


def cmd_bridge(args: argparse.Namespace) -> int:
    """Decode each stream in the manifest and publish it as a sensor_msgs/msg/Image topic.

    Args:
    ----
        args: Parsed command line, holding the manifest path and report period.

    Returns:
    -------
        Zero on success.

    """
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    m = json.loads(Path(args.manifest).read_text())
    host, pairs = m["host"], [(s["port"], s["topic"]) for s in m["streams"]]

    class Stream(threading.Thread):
        """Decodes one aircraft's stream and publishes every frame it decodes."""

        def __init__(self, node: Any, host: str, port: int, topic: str) -> None:
            super().__init__(daemon=True)
            self.node = node
            self.port = port
            self.topic = topic
            self.frames = 0
            self.started = time.time()
            self.size = "?"
            self.error = ""
            self.pub = node.create_publisher(Image, topic, qos_profile_sensor_data)
            self.container = open_stream(host, port)
            self.vstream = self.container.streams.video[0]
            self.size = f"{self.vstream.width}x{self.vstream.height}"

        def run(self) -> None:
            try:
                for frame in self.container.decode(self.vstream):
                    msg = Image()
                    msg.header.stamp = self.node.get_clock().now().to_msg()
                    msg.header.frame_id = self.topic.strip("/").replace("/", "_")
                    msg.height = int(frame.height)
                    msg.width = int(frame.width)
                    msg.encoding = "rgb8"
                    msg.is_bigendian = 0
                    msg.step = int(frame.width) * 3
                    msg.data = frame.to_ndarray(format="rgb24").tobytes()
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

    def report() -> None:
        """Log one line per stream with its running average."""
        for s in streams:
            hz = s.frames / max(1e-6, time.time() - s.started)
            line = f"{s.topic}  {s.frames} frames  {hz:5.1f} Hz  {s.size}"
            node.get_logger().info(line + (f"  ERROR {s.error}" if s.error else ""))

    node.create_timer(args.report, report)
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

    if shutil.which("foxglove_bridge") is None:
        print(
            "foxglove_bridge is not on PATH.\n" "Install it with:\n" "  sudo apt install ros-jazzy-foxglove-bridge",
            file=sys.stderr,
        )
        return 1

    # Restricting the whitelist keeps four 768x480 image topics from being joined by whatever
    # else is on the bus, which is what makes the bridge fall over on a busy graph.
    cmd = ["foxglove_bridge", "--port", str(args.port), "--topics", args.topics]
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
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    rclpy.init()
    node = Node("pterosim_camera_status")
    # topic -> [(arrival time, width, height), ...]
    seen: dict[str, list[tuple[float, int, int]]] = {}
    subs: list[Any] = []

    def on_image(topic: str, msg: Any) -> None:
        """Record a frame's arrival time and size.

        Args:
        ----
            topic: Topic the frame arrived on.
            msg: The image message.

        """
        seen.setdefault(topic, []).append((time.time(), msg.width, msg.height))

    end = time.time() + args.seconds
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.05)

    for name, types in node.get_topic_names_and_types():
        if "sensor_msgs/msg/Image" in types:
            subs.append(
                node.create_subscription(Image, name, functools.partial(on_image, name), qos_profile_sensor_data)
            )

    if not subs:
        print("no sensor_msgs/msg/Image topics on the bus -- is the bridge running?")
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
        w, h = fr[-1][1], fr[-1][2]
        print(f"{name}  {len(fr):>4} frames  {len(fr)/span:5.1f} Hz  {w}x{h}")
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
    s.add_argument("--tile", default=DEFAULT_TILE, help="camera resolution and window tile, e.g. 768x480")
    s.add_argument("--fps", type=float, default=30.0)
    s.add_argument("--base-port", type=int, default=5600)
    s.add_argument("--host", default=None, help="address the cameras stream to; auto-detected by default")
    s.add_argument("--address", default=GRPC_ADDRESS, help="PteroSim gRPC host:port")
    s.set_defaults(func=cmd_setup)

    s = sub.add_parser("bridge", help="publish the streams as Image topics")
    s.add_argument("--manifest", default=MANIFEST)
    s.add_argument("--report", type=float, default=10.0)
    s.set_defaults(func=cmd_bridge)

    s = sub.add_parser("foxglove", help="serve the camera topics to Foxglove")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--topics", default="/x500_[0-9]+/.*image_raw", help="topic whitelist for the bridge")
    s.set_defaults(func=cmd_foxglove)

    s = sub.add_parser("status", help="what is publishing, and at what rate")
    s.add_argument("--seconds", type=float, default=10.0)
    s.set_defaults(func=cmd_status)

    args = p.parse_args()
    if args.cmd in ("bridge", "view", "status"):
        ensure_ros()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
