"""PteroSim drone camera stream to screen.

Connects to an already-running PteroSim (started manually), spawns a drone if needed,
starts the simulation, and continuously pulls camera frames, displaying them with OpenCV.

Usage:
    # 1. Start PteroSim manually:
    #    PteroSim.exe   (or Play in Editor from UE5)

    # 2. Then run this script:
    python drone_camera_display.py [--aircraft F450] [--width 1280] [--height 720]

Options:
    --launch        Also launch PteroSim.exe if not already running
    --aircraft      Aircraft class to spawn (default F450)
    --width         Camera frame width (default 1280)
    --height        Camera frame height (default 720)
    --instance      Use existing aircraft with this instance_id (default 0)
    --max-frames    Auto-exit after N frames (0 = run indefinitely)
"""

import argparse
import subprocess
import sys
import time

from pterosim import PteroSim

PTEROSIM_EXE = r"C:\Users\Yollnahkriin\Downloads\PteroSim_Release\PteroSim.exe"
GRPC_ADDRESS = "localhost:10010"
ESC_KEY = 27

try:
    import cv2
except ImportError:
    cv2 = None


def launch_pterosim() -> subprocess.Popen[bytes]:
    """Start PteroSim.exe as a background process."""
    print(f"Launching PteroSim: {PTEROSIM_EXE}")
    proc = subprocess.Popen(
        [PTEROSIM_EXE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"  PID: {proc.pid}")
    return proc


def wait_for_grpc(address: str, timeout: float = 90.0) -> PteroSim:
    """Poll gRPC until the server is up."""
    print(f"Waiting for gRPC server on {address} ...")
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            sim = PteroSim(address)
            status = sim.status()
            print(f"  Connected! running={status.is_running}, aircraft={status.aircraft_count}")
            return sim
        except Exception as e:
            last_error = e
            time.sleep(2)
    raise TimeoutError(f"gRPC server not reachable after {timeout}s: {last_error}")


def main() -> None:
    """Display the selected drone's live camera feed."""
    parser = argparse.ArgumentParser(description="Display PteroSim drone camera feed")
    parser.add_argument("--launch", action="store_true", help="Launch PteroSim.exe if not already running")
    parser.add_argument("--aircraft", default="F450", help="Aircraft class to spawn (default F450)")
    parser.add_argument("--width", type=int, default=1280, help="Camera frame width")
    parser.add_argument("--height", type=int, default=720, help="Camera frame height")
    parser.add_argument("--instance", type=int, default=0, help="Use existing aircraft with this instance_id")
    parser.add_argument("--max-frames", type=int, default=0, help="Auto-exit after N frames (0 = run indefinitely)")
    args = parser.parse_args()

    if cv2 is None:
        print("OpenCV (cv2) is required. Install with: pip install opencv-python")
        sys.exit(1)

    sim_process = None
    sim = None
    try:
        # Step 1: Optionally launch PteroSim
        if args.launch:
            sim_process = launch_pterosim()

        # Step 2: Connect gRPC
        sim = wait_for_grpc(GRPC_ADDRESS)

        # Step 3: Spawn drone if not already present
        status = sim.status()
        if status.aircraft_count == 0:
            print(f"Spawning {args.aircraft} ...")
            drone_info = sim.spawn(args.aircraft, x=0, y=0, z=100, yaw=0)
            print(f"  instance_id={drone_info.instance_id}, mavlink_port={drone_info.mavlink_port}")
            instance_id = drone_info.instance_id
        else:
            instance_id = args.instance
            print(f"Using existing aircraft instance_id={instance_id}")

        drone = sim.get_aircraft(instance_id)

        # Step 4: Start simulation
        if not status.is_running:
            print("Starting simulation ...")
            sim.set_time_scale(1.0)
            sim.start()
            print("  Simulation started.")

        # Check camera sensor exists
        sensors = drone.list_sensors()
        cam_sensors = [s for s in sensors if s.type == "camera"]
        if not cam_sensors:
            print("ERROR: No camera sensor found on this aircraft. Use add_sensor('camera') before start().")
            sys.exit(1)
        print(f"Camera sensor: {cam_sensors[0].name} ({cam_sensors[0].update_hz} Hz)")

        # Step 5: Stream frames to screen
        print(f"\nDisplaying camera feed {args.width}x{args.height} ...")
        print("Press 'q' or Esc to quit.\n")

        window = "PteroSim Drone Camera"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        frame_count = 0
        t0 = time.perf_counter()
        while True:
            try:
                frame = drone.camera(width=args.width, height=args.height, timeout=2.0)
                # SDK returns a read-only numpy array; copy for OpenCV drawing
                img = frame.image.copy()
                cv2.imshow(window, img)

                # HUD overlay
                fps = frame_count / max(time.perf_counter() - t0, 1e-6)
                hud = f"seq={frame.sequence_number}  fps={fps:.1f}  t={frame.timestamp:.1f}s"
                cv2.putText(img, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ESC_KEY:  # 'q' or Esc
                    break
                frame_count += 1
                if args.max_frames and frame_count >= args.max_frames:
                    print(f"Reached {args.max_frames} frames, exiting.")
                    break
            except Exception as e:
                print(f"Frame error: {e}")
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if sim is not None:
            try:
                sim.stop()
                print("Simulation stopped.")
            except Exception:
                pass
        if sim_process is not None:
            sim_process.terminate()
            sim_process.wait(timeout=5)
            print("PteroSim terminated.")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
