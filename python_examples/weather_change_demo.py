"""Launch PteroSim and cycle through weather and wind for a short demo.

The script uses the SDK shipped with this checkout. Wind control requires an
Edu or Pro license; weather presets are available with any license.

    python weather_change_demo.py
    python weather_change_demo.py --no-launch  # connect to a running simulator
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SDK_DIR = REPO_ROOT / "Plugins" / "PteroSimScripting" / "SDK" / "python"
DEFAULT_SIMULATOR = REPO_ROOT / "Build" / "Windows" / "PteroSim.exe"
WEATHER_AND_WIND = (
    ("clear_skies", 0.0, 0.0),
    ("partly_cloudy", 3.0, 90.0),
    ("rain_light", 7.0, 180.0),
    ("overcast", 10.0, 270.0),
    ("foggy", 2.0, 45.0),
)

sys.path.insert(0, str(SDK_DIR))

try:
    import grpc
    from pterosim import PteroSim
except ImportError as exc:
    raise SystemExit(
        f'Could not import the PteroSim SDK from {SDK_DIR}. Install it with: python -m pip install -e "{SDK_DIR}"'
    ) from exc


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="localhost:10010", help="PteroSim gRPC address")
    parser.add_argument(
        "--simulator",
        type=Path,
        default=DEFAULT_SIMULATOR,
        help=f"PteroSim executable to launch (default: {DEFAULT_SIMULATOR})",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=120.0,
        help="seconds to wait for the gRPC server (default: 120)",
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="do not launch PteroSim when the server is unavailable",
    )
    return parser.parse_args()


def connect_when_ready(address: str, timeout_s: float) -> PteroSim:
    """Wait for the simulator's gRPC server and return a connected client."""
    sim = PteroSim(address)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            sim.status()
            return sim
        except grpc.RpcError as exc:
            if exc.code() != grpc.StatusCode.UNAVAILABLE:
                sim.close()
                raise
            if time.monotonic() >= deadline:
                sim.close()
                raise TimeoutError(f"PteroSim did not become available at {address} within {timeout_s:g} s") from exc
            time.sleep(1.0)


def main() -> int:
    """Connect to PteroSim and run the weather and wind sequence."""
    args = parse_args()

    # Check whether a server is already listening before launching another process.
    try:
        sim = connect_when_ready(args.address, timeout_s=1.0)
        print(f"Connected to the running simulator at {args.address}.")
    except TimeoutError:
        if args.no_launch:
            print(f"No simulator is listening at {args.address}; start it and retry.", file=sys.stderr)
            return 1
        if not args.simulator.is_file():
            print(f"PteroSim.exe was not found: {args.simulator}", file=sys.stderr)
            print("Pass its location with --simulator, or start it manually.", file=sys.stderr)
            return 1
        print(f"Launching {args.simulator} ...")
        subprocess.Popen([str(args.simulator)], cwd=str(args.simulator.parent))
        try:
            sim = connect_when_ready(args.address, timeout_s=args.startup_timeout)
        except (TimeoutError, grpc.RpcError) as exc:
            print(f"Could not connect to PteroSim: {exc}", file=sys.stderr)
            print("If the app is open, start a Play session and retry.", file=sys.stderr)
            return 1

    try:
        status = sim.status()
        print(f"Simulator ready: running={status.is_running}, aircraft={status.aircraft_count}.")
        if not status.is_running:
            print("Starting simulation physics ...")
            sim.start()

        wind_available = True
        for index, (preset, speed_ms, from_deg) in enumerate(WEATHER_AND_WIND):
            try:
                sim.set_weather(preset)
            except grpc.RpcError as exc:
                if exc.code() == grpc.StatusCode.UNIMPLEMENTED:
                    print(
                        "This PteroSim build does not expose the weather API. "
                        "Rebuild/relaunch it with the current PteroSimScripting plugin.",
                        file=sys.stderr,
                    )
                    return 1
                raise
            print(f"Weather: {preset}")
            if wind_available:
                try:
                    sim.set_wind(speed_ms=speed_ms, from_deg=from_deg)
                    print(f"Wind: {speed_ms:g} m/s from {from_deg:g} degrees")
                except grpc.RpcError as exc:
                    if exc.code() != grpc.StatusCode.PERMISSION_DENIED:
                        raise
                    wind_available = False
                    print("Wind changes require an Edu or Pro license; continuing with weather only.")
            if index < len(WEATHER_AND_WIND) - 1:
                time.sleep(5.0)

        print("Weather and wind demo complete.")
        return 0
    finally:
        sim.close()


if __name__ == "__main__":
    raise SystemExit(main())
