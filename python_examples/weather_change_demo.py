"""Cycle through weather and wind settings in a running PteroSim instance.

Install the ``pterosim`` SDK shipped with PteroSim before running this example.
Wind control requires an Edu or Pro license; weather presets are available with
any license.

    python weather_change_demo.py
    python weather_change_demo.py --address localhost:10010
"""

from __future__ import annotations

import argparse
import time

import grpc
from pterosim import PteroSim

DEMO_INTERVAL_S = 5.0
SERVER_POLL_INTERVAL_S = 1.0
DEFAULT_CONNECTION_TIMEOUT_S = 120.0

WEATHER_AND_WIND = (
    ("clear_skies", 0.0, 0.0),
    ("partly_cloudy", 3.0, 90.0),
    ("rain_light", 7.0, 180.0),
    ("overcast", 10.0, 270.0),
    ("foggy", 2.0, 45.0),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="localhost:10010", help="PteroSim gRPC address")
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=DEFAULT_CONNECTION_TIMEOUT_S,
        help=f"seconds to wait for PteroSim (default: {DEFAULT_CONNECTION_TIMEOUT_S:g})",
    )
    return parser.parse_args()


def connect_when_ready(address: str, timeout_s: float) -> PteroSim:
    """Wait for the running simulator's gRPC server and return a client."""
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
                raise TimeoutError(f"PteroSim did not respond at {address} within {timeout_s:g} seconds") from exc
            time.sleep(SERVER_POLL_INTERVAL_S)


def ensure_simulation_running(sim: PteroSim) -> None:
    """Resume a held simulation, or start it if it is idle."""
    if sim.status().is_running:
        return

    # Resume is a no-op while idle; checking status afterward distinguishes idle
    # from Hold without relying on the clock's Holding state, which both share.
    sim.resume()
    if sim.status().is_running:
        print("Resumed the held simulation.")
        return

    print("Starting simulation physics ...")
    sim.start()


def main() -> int:
    """Connect to PteroSim and run the weather and wind sequence."""
    args = parse_args()
    try:
        sim = connect_when_ready(args.address, timeout_s=args.connect_timeout)
    except TimeoutError as exc:
        print(f"{exc}. Open PteroSim and start a simulation, then retry.")
        return 1
    except grpc.RpcError as exc:
        print(f"Could not connect to PteroSim at {args.address}: {exc.details()}")
        return 1

    try:
        status = sim.status()
        print(f"Connected: running={status.is_running}, aircraft={status.aircraft_count}.")
        try:
            ensure_simulation_running(sim)
        except grpc.RpcError as exc:
            print(f"Could not start or resume the simulation: {exc.details()}")
            return 1

        wind_available = True
        for index, (preset, speed_ms, from_deg) in enumerate(WEATHER_AND_WIND):
            try:
                sim.set_weather(preset)
            except grpc.RpcError as exc:
                if exc.code() == grpc.StatusCode.UNIMPLEMENTED:
                    print("This weather API requires PteroSim 0.5.0 or newer. Update PteroSim and try again.")
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
                time.sleep(DEMO_INTERVAL_S)

        print("Weather and wind demo complete.")
        return 0
    finally:
        sim.close()


if __name__ == "__main__":
    raise SystemExit(main())
