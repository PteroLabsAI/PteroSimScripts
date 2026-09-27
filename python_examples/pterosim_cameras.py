#!/usr/bin/env python3
"""PteroSim drone cameras -> ROS 2, in one file.

Publishes each aircraft's camera to ROS 2 as sensor_msgs/msg/Image and shows them tiled
fullscreen. Three steps, because the simulator only accepts sensor settings while it is
stopped:

    pterosim_cameras.py setup     spawn the fleet, point the cameras at this machine
    pterosim_cameras.py bridge    decode the streams and publish them as Image topics
    pterosim_cameras.py view      fullscreen 2x2 window
    pterosim_cameras.py all       all three, in that order
    pterosim_cameras.py status    what is publishing, and at what rate

`setup` puts four x500 in the scene and streams each gimbal_camera over RTP/H.264 to this
machine. `bridge` decodes those streams and publishes one topic per aircraft. `view` shows
them. `all` is the whole thing.

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

The cameras are configured for 768x480 because four of them tile a 1536x960 screen exactly.
--tile changes both the camera resolution and the window layout together.

    4 x 768x480 at 30 fps source, ~20 Hz per stream delivered on one decode process.

Requires: ROS 2 (rclpy, sensor_msgs), PyAV, and the PteroSim SDK for `setup`. Run `bridge`
under a virtualenv that has PyAV; run `view` under one that has PySide6.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

DEFAULT_TILE = "768x480"
MANIFEST = str(Path.home() / "camera_fleet.json")
GRPC_ADDRESS = "172.26.48.1:10011"
COLS = 2


# --------------------------------------------------------------------------- shared

def local_address():
    """This machine's IPv4 address, which is what the simulator must aim the stream at.

    A connected UDP socket picks the source address the kernel would use to leave, without
    sending anything. Under WSL this changes across restarts, so it is read at run time.
    """
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def write_sdp(host, port):
    """The session description a decoder needs to open a bare RTP/H.264 stream."""
    path = Path(tempfile.gettempdir()) / f"pterosim_stream_{port}.sdp"
    path.write_text("\n".join([
        "v=0",
        f"o=- 0 0 IN IP4 {host}",
        "s=PteroSim camera",
        f"c=IN IP4 {host}",
        "t=0 0",
        f"m=video {port} RTP/AVP 96",
        "a=rtpmap:96 H264/90000",
    ]) + "\n")
    return path


def open_stream(host, port):
    import av
    sdp = write_sdp(host, port)
    return av.open(str(sdp), format="sdp",
                   options={"protocol_whitelist": "file,rtp,udp"})


def parse_tile(text):
    w, h = text.lower().split("x")
    return int(w), int(h)


def ensure_ros():
    """Re-exec under a sourced ROS 2 environment if rclpy is not importable yet.

    rclpy lives in the ROS install's site-packages, which only its setup script puts on
    PYTHONPATH -- a --system-site-packages virtualenv does not see it. Asking the reader to
    remember a source line is the one thing a file meant to be handed to someone else
    should not depend on, so do it here. The guard variable keeps a failing source from
    turning into a re-exec loop.
    """
    try:
        import rclpy  # noqa: F401
        return
    except ImportError:
        pass

    if os.environ.get("PTEROSIM_CAMERAS_REEXEC"):
        raise SystemExit("rclpy still not importable after sourcing ROS 2")

    setups = sorted(Path("/opt/ros").glob("*/setup.bash"))
    if not setups:
        raise SystemExit("rclpy not found and no /opt/ros/*/setup.bash to source; "
                         "source your ROS 2 install and retry")
    # sys.argv[1:], not sys.argv: argv[0] is this script, which is already named by __file__.
    argv = " ".join(f'"{a}"' for a in sys.argv[1:])
    script = f'source "{setups[-1]}" && exec "{sys.executable}" "{os.path.abspath(__file__)}" {argv}'
    os.environ["PTEROSIM_CAMERAS_REEXEC"] = "1"
    os.execv("/bin/bash", ["/bin/bash", "-c", script])


# --------------------------------------------------------------------------- setup

def cmd_setup(args):
    """Spawn the fleet and point every camera at this machine, then start the simulation."""
    sdk = os.environ.get("PTEROSIM_SDK", "")
    if not sdk:
        default = ("/mnt/c/Users/Yollnahkriin/Documents/Unreal_Projects/PteroSim/"
                   "Plugins/PteroSimScripting/SDK/python")
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
        have = have[:args.count]

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
                cam.name, update_hz=args.fps, image_width=width, image_height=height,
                stream=True, stream_port=stream_port, stream_host=host)
            aircraft = getattr(drone, "aircraft_name", None) or args.aircraft
            topic = f"/{aircraft}_{i}/{cam.name}/image_raw"
            streams.append({"instance_id": i, "aircraft": aircraft, "camera": cam.name,
                            "port": stream_port + i, "topic": topic})
            print(f"  instance {i}: {cam.name} -> udp://{host}:{stream_port + i}  {topic}")

        sim.start()
    finally:
        # Channel close only. shutdown() would take the whole simulator down.
        sim.close()

    Path(MANIFEST).write_text(json.dumps(
        {"host": host, "tile": [width, height], "streams": streams}, indent=2))
    print(f"\nsimulation started; wrote {MANIFEST}")
    return 0


# -------------------------------------------------------------------------- bridge

def cmd_bridge(args):
    """Decode each stream and publish it as sensor_msgs/msg/Image."""
    import av
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    m = json.loads(Path(args.manifest).read_text())
    host, pairs = m["host"], [(s["port"], s["topic"]) for s in m["streams"]]

    class Stream(threading.Thread):
        """Decodes one aircraft's stream and publishes every frame it decodes."""

        def __init__(self, node, host, port, topic):
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

        def run(self):
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
            except Exception as exc:  # noqa: BLE001
                # One lost stream must not take the other aircraft down with it.
                self.error = f"{type(exc).__name__}: {exc}"
                self.node.get_logger().error(f"{self.topic}: {self.error}")

        def close(self):
            self.container.close()

    class Bridge(Node):
        def __init__(self):
            super().__init__("pterosim_camera")
            # Built after super().__init__ because each stream needs the node to publish on.
            self.streams = [Stream(self, host, p_, t) for p_, t in pairs]
            for s in self.streams:
                s.start()
                self.get_logger().info(
                    f"publishing {s.topic}  {s.size} from udp port {s.port}")
            self.create_timer(args.report, self.report)

        def report(self):
            for s in self.streams:
                hz = s.frames / max(1e-6, time.time() - s.started)
                line = f"{s.topic}  {s.frames} frames  {hz:5.1f} Hz  {s.size}"
                self.get_logger().info(line + (f"  ERROR {s.error}" if s.error else ""))

        def close(self):
            for s in self.streams:
                s.close()

    rclpy.init()
    node = None
    try:
        node = Bridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


# ----------------------------------------------------------------------------- view

def cmd_view(args):
    """Fullscreen tiled view of every camera topic on the bus."""
    import numpy as np
    import rclpy
    from PySide6.QtCore import Qt, QRect, QTimer
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPixmap
    from PySide6.QtWidgets import QApplication, QGridLayout, QLabel, QWidget
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    width, height = parse_tile(args.tile)
    rows = max(1, args.rows)

    def to_qimage(msg, caption=""):
        """rgb8 Image -> QImage, copied, with the caption painted into the pixels.

        The numpy array is a view onto msg.data, which the next message replaces, so handing
        it to QImage uncopied would show torn frames. The caption goes into the image rather
        than the layout: a layout label would shrink the tile, and the tiles are sized to
        cover the screen exactly.
        """
        h, w = int(msg.height), int(msg.width)
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, msg.step // 3, 3)[:, :w, :3]
        img = QImage(arr.tobytes(), w, h, msg.step, QImage.Format.Format_RGB888).copy()
        if caption:
            p = QPainter(img)
            p.setFont(QFont("", 13, QFont.Weight.Bold))
            wpx = 12 + 8 * len(caption)
            p.fillRect(0, 0, wpx, 22, QColor(0, 0, 0, 130))
            p.setPen(QColor(255, 255, 255))
            p.drawText(QRect(6, 3, wpx, 18), Qt.AlignmentFlag.AlignLeft, caption)
            p.end()
        return img

    rclpy.init()
    node = Node("pterosim_camera_view")
    latest, counts = {}, {}
    subs = []

    def on_image(topic, msg):
        latest[topic] = msg
        counts[topic] += 1

    # Reception on its own thread: sensor-data QoS is best effort with a shallow queue, so a
    # consumer sharing the Qt event loop would simply lose frames.
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()

    app = QApplication(sys.argv)
    win = QWidget()
    win.setWindowTitle("PteroSim cameras (ROS 2)")
    grid = QGridLayout(win)
    # Nothing between the tiles: the grid is the screen, so any margin or caption row would
    # leave the fleet not quite filling it.
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setSpacing(0)

    tiles = {}
    state = {"fps_t": time.time(), "fps": {}, "logged": time.time()}
    known = set()

    def add_tile(topic):
        label = QLabel("waiting...")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("background:#111; color:#888;")
        label.setFixedSize(width, height)
        n = len(tiles)
        grid.addWidget(label, n // COLS, n % COLS)
        tiles[topic] = [label, -1]
        print(f"discovered {topic}", flush=True)

    def tick():
        # Discovery runs continuously: the fleet may be respawned under us.
        for name, types in node.get_topic_names_and_types():
            if "sensor_msgs/msg/Image" in types and name not in known:
                known.add(name)
                counts[name] = 0
                subs.append(node.create_subscription(
                    Image, name, (lambda t: (lambda m: on_image(t, m)))(name),
                    qos_profile_sensor_data))
                add_tile(name)

        now = time.time()
        if now - state["fps_t"] >= 1.0:
            # Rate from frame deltas, not timer ticks: the timer runs at 10 Hz here, so
            # counting it would misreport anything faster. What the next pass subtracts is
            # the stored count, so the count is what has to be stored.
            dt = now - state["fps_t"]
            for t, c in counts.items():
                prev = state["fps"].get(t, (c, 0.0))[0]
                state["fps"][t] = (c, (c - prev) / dt)
            state["fps_t"] = now

        for topic, tile in list(tiles.items()):
            msg = latest.get(topic)
            if msg is None:
                continue
            if counts[topic] != tile[1]:
                tile[1] = counts[topic]
                fps = state["fps"].get(topic, (0, 0.0))[1]
                short = topic.strip("/").split("/")[0]
                # Tile is already the frame's own size, so this is 1:1 on the screen grid.
                tile[0].setPixmap(QPixmap.fromImage(to_qimage(
                    msg, f"{short}  {fps:.1f} fps  {msg.width}x{msg.height}")))

        if time.time() - state["logged"] >= 1.0:
            state["logged"] = time.time()
            if tiles:
                print("  " + " | ".join(
                    f"{t.strip('/').split('/')[0]}:{state['fps'].get(t, (0, 0.0))[1]:.1f}Hz"
                    for t in tiles), flush=True)

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(100)

    def on_key(event):
        if event.key() in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            app.quit()
    win.keyPressEvent = on_key

    win.showFullScreen()
    app.exec()
    rclpy.shutdown()
    return 0


# -------------------------------------------------------------------------- status

def cmd_status(args):
    """What is publishing right now, and at what rate."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    rclpy.init()
    node = Node("pterosim_camera_status")
    seen = {}
    subs = []

    end = time.time() + args.seconds
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.05)

    for name, types in node.get_topic_names_and_types():
        if "sensor_msgs/msg/Image" in types:
            subs.append(node.create_subscription(
                Image, name, (lambda t: (lambda m: seen.setdefault(
                    t, []).append((time.time(), m.width, m.height))))(name),
                qos_profile_sensor_data))

    if not subs:
        print("no sensor_msgs/msg/Image topics on the bus -- is the bridge running?")
        rclpy.shutdown()
        return 1

    seen.clear()
    t0 = time.time()
    end = t0 + args.seconds
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

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="spawn the fleet and point the cameras here")
    s.add_argument("--count", type=int, default=4)
    s.add_argument("--aircraft", default="x500")
    s.add_argument("--camera", default="gimbal_camera",
                   help="preferred camera; falls back to any camera aboard")
    s.add_argument("--tile", default=DEFAULT_TILE,
                   help="camera resolution and window tile, e.g. 768x480")
    s.add_argument("--fps", type=float, default=30.0)
    s.add_argument("--base-port", type=int, default=5600)
    s.add_argument("--host", default=None,
                   help="address the cameras stream to; auto-detected by default")
    s.add_argument("--address", default=GRPC_ADDRESS, help="PteroSim gRPC host:port")
    s.set_defaults(func=cmd_setup)

    s = sub.add_parser("bridge", help="publish the streams as Image topics")
    s.add_argument("--manifest", default=MANIFEST)
    s.add_argument("--report", type=float, default=10.0)
    s.set_defaults(func=cmd_bridge)

    s = sub.add_parser("view", help="fullscreen tiled window")
    s.add_argument("--tile", default=DEFAULT_TILE)
    s.add_argument("--rows", type=int, default=2)
    s.set_defaults(func=cmd_view)

    s = sub.add_parser("status", help="what is publishing, and at what rate")
    s.add_argument("--seconds", type=float, default=10.0)
    s.set_defaults(func=cmd_status)

    args = p.parse_args()
    if args.cmd in ("bridge", "view", "status"):
        ensure_ros()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
