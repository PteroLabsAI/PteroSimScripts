"""Turn on a drone camera's video stream from Python and watch it.

A camera sensor with ``stream`` on sends RTP/H.264 to udp://<host>:<stream_port + instance id> for as
long as the simulation runs -- the same stream a ground station shows. With PX4 or ArduPilot SITL
connected, the simulator announces the camera on the autopilot's link (MAVLink camera protocol) and
QGroundControl / Mission Planner pick the video up on their own; without an autopilot, set QGC's
Application Settings > Video > "UDP h.264 Video Stream" to the port printed below. Nothing to install
on either side: PteroSim ships its own encoder.

The other camera example, drone_camera_display.py, pulls raw frames over gRPC one by one -- right for
vision code that wants every pixel. This one is a push stream for people to look at.

Usage:
    # 1. Start PteroSim (PteroSim.exe, or Play in Editor from UE5)
    # 2. python drone_camera_stream.py [--aircraft x500] [--camera camera] [--port 5600] [--view]

Options:
    --aircraft   Aircraft to spawn when the sim is empty (default x500)
    --instance   Use the aircraft with this instance_id when one is already there (default 0)
    --camera     Camera sensor name (default: the first camera the aircraft carries)
    --port       Base UDP port; the aircraft's instance id is added (default 5600)
    --host       Where to send it -- the ground station's address (default 127.0.0.1)
    --fps        Stream frame rate (default 30; the engine's own frame rate is the ceiling)
    --width/--height  Frame size (default 1280x720)
    --view       Also decode and show the stream here with OpenCV (needs opencv-python)
    --seconds    Stop after N seconds (0 = until Ctrl-C / q)
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

from pterosim import PteroSim

try:
    import cv2
except ImportError:
    cv2 = None

GRPC_ADDRESS = "localhost:10010"
ESC_KEY = 27

# OpenCV's ffmpeg reads its options from the environment once, when the DLL loads -- before this script can
# set them -- so a viewer run re-launches itself with them in place (os.execve is not a real exec on Windows).
_VIEWER_ENV = {
    "OPENCV_FFMPEG_CAPTURE_OPTIONS": "protocol_whitelist;file,rtp,udp|fflags;nobuffer|flags;low_delay",
    "OPENCV_FFMPEG_LOGLEVEL": "8",  # fatal only: joining mid-GOP spams "non-existing PPS" until the next keyframe
}


def write_sdp(host: str, port: int) -> Path:
    """The session description a decoder needs to open a bare RTP/H.264 stream.

    Payload type 96 means nothing on its own, and ffmpeg takes the description only from a file (a data: URL
    is not probed as SDP).
    """
    path = Path(tempfile.gettempdir()) / f"pterosim_stream_{port}.sdp"
    lines = [
        "v=0",
        f"o=- 0 0 IN IP4 {host}",
        "s=PteroSim camera",
        f"c=IN IP4 {host}",
        "t=0 0",
        f"m=video {port} RTP/AVP 96",
        "a=rtpmap:96 H264/90000",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def view(host: str, port: int, seconds: float) -> None:
    """Decode the stream with OpenCV and show it with a frame counter and fps."""
    sdp = write_sdp(host, port)
    deadline = time.perf_counter() + 30.0
    cap = cv2.VideoCapture(str(sdp), cv2.CAP_FFMPEG)
    while not cap.isOpened() and time.perf_counter() < deadline:
        time.sleep(0.5)
        cap = cv2.VideoCapture(str(sdp), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"No stream on udp://{host}:{port} within 30 s (is the camera's stream on, and the sim started?)")
        return

    window = f"PteroSim stream :{port}"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    frames, t0 = 0, time.perf_counter()
    stop_at = t0 + seconds if seconds else float("inf")
    arrivals: deque[float] = deque(
        maxlen=30
    )  # fps over the last 30 frames: since-start would carry the decoder's start-up wait forever
    try:
        while time.perf_counter() < stop_at:
            ok, img = cap.read()
            if not ok:
                print("Stream ended.")
                break
            frames += 1
            arrivals.append(time.perf_counter())
            fps = (
                (len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
                if len(arrivals) > 1 and arrivals[-1] > arrivals[0]
                else 0.0
            )
            cv2.putText(
                img,
                f"frame={frames} fps={fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow(window, img)
            # The window's own close button is not a key: without asking whether it is still there, closing it
            # leaves this loop decoding a stream nobody watches and the simulation running behind it.
            if cv2.waitKey(1) & 0xFF in (ord("q"), ESC_KEY) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"{frames} frames shown")


def main() -> None:
    """Point a camera at a UDP port, start the sim, then wait or watch."""
    parser = argparse.ArgumentParser(description="Stream a PteroSim drone camera as RTP/H.264 over UDP")
    parser.add_argument("--aircraft", default="x500")
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--camera", default="")
    parser.add_argument("--port", type=int, default=5600)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--seconds", type=float, default=0.0)
    args = parser.parse_args()

    if args.view and cv2 is None:
        print("--view needs OpenCV: pip install opencv-python")
        sys.exit(1)
    if args.view and any(os.environ.get(k) != v for k, v in _VIEWER_ENV.items()):
        sys.exit(subprocess.run([sys.executable, *sys.argv], env={**os.environ, **_VIEWER_ENV}, check=False).returncode)

    try:
        sim = PteroSim(GRPC_ADDRESS)
        status = sim.status()
    except Exception as e:  # grpc's error text is a paragraph; the reason is enough
        raise SystemExit(f"PteroSim is not reachable on {GRPC_ADDRESS} (start it first): {type(e).__name__}") from None
    if status.is_running:
        print("The simulation is already running; a stream can only be configured before start(). Stop it first.")
        sys.exit(1)

    if status.aircraft_count == 0:
        drone = sim.spawn(args.aircraft, x=0, y=0, z=100, yaw=0)
        print(f"Spawned {args.aircraft}: instance_id={drone.instance_id}")
    else:
        drone = sim.get_aircraft(args.instance)
        print(f"Using existing aircraft instance_id={args.instance}")

    cameras = [s for s in drone.list_sensors() if s.type == "camera" and (not args.camera or s.name == args.camera)]
    if not cameras:
        print(f"No camera sensor {args.camera!r} on this aircraft. Add one with drone.add_sensor('camera') first.")
        sys.exit(1)
    camera = cameras[0]

    # Everything the stream needs is a sensor attribute -- the same keys as in the vehicle's Sensors.xml.
    drone.set_sensor_param(
        camera.name,
        update_hz=args.fps,
        image_width=args.width,
        image_height=args.height,
        stream=True,
        stream_port=args.port,
        stream_host=args.host,
    )
    port = args.port + drone.instance_id
    sim.start()
    print(f"Streaming '{camera.name}' {args.width}x{args.height}@{args.fps} -> udp://{args.host}:{port}")
    print(f"QGroundControl: Video Source = UDP h.264 Video Stream, port {port}. Ctrl-C stops the simulation.")

    try:
        if args.view:
            view(args.host, port, args.seconds)
        elif args.seconds:
            time.sleep(args.seconds)
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        sim.stop()
        print("Simulation stopped; stream ended.")


if __name__ == "__main__":
    main()
