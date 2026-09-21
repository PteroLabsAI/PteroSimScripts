"""PteroSim drone camera stream to screen, with per-frame latency.

Connects to an already-running PteroSim (started manually), spawns a drone if needed,
starts the simulation, and continuously pulls camera frames, displaying them with OpenCV.
Every frame's round-trip (request -> frame in hand) is measured; a summary is printed
every --report-every frames and, with --latency-log, every frame goes to a CSV.

Usage:
    # 1. Start PteroSim (PteroSim.exe, or Play in Editor from UE5)
    # 2. python drone_camera_display.py [--aircraft x500] [--width 1280] [--height 720]

Options:
    --aircraft      Aircraft class to spawn (default F450)
    --camera        Camera sensor name (default: the first camera on the aircraft)
    --width         Camera frame width (default 1280)
    --height        Camera frame height (default 720)
    --instance      Use existing aircraft with this instance_id (default 0)
    --max-frames    Auto-exit after N frames (0 = run indefinitely)
    --report-every  Print latency stats every N frames (default 30)
    --latency-log   CSV file: seq, wall_s, sim_t, rtt_ms
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

from pterosim import PteroSim

GRPC_ADDRESS = "localhost:10010"
ESC_KEY = 27
MAX_CONSECUTIVE_ERRORS = 5

try:
    import cv2
except ImportError:
    cv2 = None


def connect(address: str) -> PteroSim:
    """Connect to the running PteroSim; it is not started from here."""
    try:
        sim = PteroSim(address)
        status = sim.status()
    except Exception as e:  # grpc's error text is a paragraph; the reason is enough
        raise SystemExit(f"PteroSim is not reachable on {address} (start it first): {type(e).__name__}") from None
    print(f"Connected to {address}: running={status.is_running}, aircraft={status.aircraft_count}")
    return sim


def report(rtts_ms: list[float], frames: int, elapsed_s: float) -> None:
    """One line of latency stats for the last --report-every frames.

    pull fps is what the camera delivers when asked back to back (frames per second of
    round-trip time); loop fps also counts the time this script spends drawing.
    """
    print(
        f"frames={frames}  pull fps={frames / max(sum(rtts_ms) / 1000.0, 1e-6):.1f}  "
        f"loop fps={frames / max(elapsed_s, 1e-6):.1f}  "
        f"rtt ms: min={min(rtts_ms):.0f} avg={statistics.fmean(rtts_ms):.0f} "
        f"p50={statistics.median(rtts_ms):.0f} max={max(rtts_ms):.0f}"
    )


def main() -> None:
    """Display the selected drone's live camera feed and log its latency."""
    parser = argparse.ArgumentParser(description="Display PteroSim drone camera feed")
    parser.add_argument("--aircraft", default="F450", help="Aircraft class to spawn (default F450)")
    parser.add_argument("--camera", default="", help="Camera sensor name (default: first camera found)")
    parser.add_argument("--width", type=int, default=1280, help="Camera frame width")
    parser.add_argument("--height", type=int, default=720, help="Camera frame height")
    parser.add_argument("--instance", type=int, default=0, help="Use existing aircraft with this instance_id")
    parser.add_argument("--max-frames", type=int, default=0, help="Auto-exit after N frames (0 = run indefinitely)")
    parser.add_argument("--report-every", type=int, default=30, help="Print latency stats every N frames")
    parser.add_argument("--latency-log", type=Path, help="CSV file for per-frame latency")
    args = parser.parse_args()

    if cv2 is None:
        print("OpenCV (cv2) is required. Install with: pip install opencv-python")
        sys.exit(1)

    sim = None
    we_started = False
    csv = args.latency_log.open("w", encoding="utf-8") if args.latency_log else None
    if csv:
        csv.write("seq,wall_s,sim_t,rtt_ms\n")
    try:
        sim = connect(GRPC_ADDRESS)

        # Spawn a drone if the scene is empty
        status = sim.status()
        if status.aircraft_count == 0:
            print(f"Spawning {args.aircraft} ...")
            drone = sim.spawn(args.aircraft, x=0, y=0, z=100, yaw=0)
            print(f"  instance_id={drone.instance_id}, mavlink_port={drone.mavlink_port}")
        else:
            drone = sim.get_aircraft(args.instance)
            print(f"Using existing aircraft instance_id={args.instance}")

        # Start the simulation unless it already runs
        if not status.is_running:
            print("Starting simulation ...")
            sim.set_time_scale(1.0)
            sim.start()
            we_started = True
            print("  Simulation started.")

        # Check camera sensor exists
        cameras = [s for s in drone.list_sensors() if s.type == "camera"]
        if args.camera:
            cameras = [s for s in cameras if s.name == args.camera]
        if not cameras:
            print(f"ERROR: No camera sensor {args.camera!r} on this aircraft. Use add_sensor('camera') before start().")
            sys.exit(1)
        camera = cameras[0]
        print(f"Camera sensor: {camera.name} ({camera.update_hz} Hz)")

        # Pull frames to the screen
        print(f"\nDisplaying camera feed {args.width}x{args.height} ...")
        print("Press 'q' or Esc to quit.\n")

        window = "PteroSim Drone Camera"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, args.width, args.height)

        frame_count = 0
        errors = 0
        rtts_ms: list[float] = []
        t0 = time.perf_counter()
        t_report = t0
        while True:
            try:
                t_req = time.perf_counter()
                frame = drone.camera(camera.name, width=args.width, height=args.height, timeout=2.0)
                rtt_ms = (time.perf_counter() - t_req) * 1000.0
                errors = 0
            except Exception as e:
                errors += 1
                print(f"Frame error ({errors}/{MAX_CONSECUTIVE_ERRORS}): {e}")
                if errors >= MAX_CONSECUTIVE_ERRORS:
                    break
                time.sleep(0.1)
                continue

            frame_count += 1
            rtts_ms.append(rtt_ms)
            if csv:
                csv.write(f"{frame.sequence_number},{t_req - t0:.4f},{frame.timestamp:.4f},{rtt_ms:.1f}\n")

            # SDK returns a read-only numpy array; copy for OpenCV drawing
            img = frame.image.copy()
            fps = frame_count / max(time.perf_counter() - t0, 1e-6)
            hud = f"seq={frame.sequence_number}  loop fps={fps:.1f}  rtt={rtt_ms:.0f}ms  t={frame.timestamp:.1f}s"
            cv2.putText(img, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window, img)

            if frame_count % args.report_every == 0:
                now = time.perf_counter()
                report(rtts_ms[-args.report_every :], args.report_every, now - t_report)
                t_report = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ESC_KEY:
                break
            if args.max_frames and frame_count >= args.max_frames:
                print(f"Reached {args.max_frames} frames, exiting.")
                break

        if rtts_ms:
            print("\nTotal:")
            report(rtts_ms, frame_count, time.perf_counter() - t0)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if csv:
            csv.close()
        if sim is not None and we_started:
            try:
                sim.stop()
                print("Simulation stopped.")
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
