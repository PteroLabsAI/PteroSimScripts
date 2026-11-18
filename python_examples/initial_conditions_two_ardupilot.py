"""Configure two ArduPilot drones on the currently open PteroSim map.

The map must already contain the desired Cesium tileset. Geographic spawn and
origin calls wait for Cesium terrain beneath the requested coordinates.

Example (run from the PteroSimScripts directory):
    python python_examples/initial_conditions_two_ardupilot.py
    python python_examples/initial_conditions_two_ardupilot.py --start-time 2026-06-21T22:00:00

Pass --drone-1 LAT LON ALT and --drone-2 LAT LON ALT to change locations.

The altitude is WGS84 metres. Ground spawns stand on loaded terrain; the
altitude does not set height above ground. ArduPilot SITL must be started
separately for each JSON/UDP link port printed by this script.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime

from pterosim import PteroSim

DEFAULT_DRONE_1 = (36.146, -113.8900, 1425.6)
DEFAULT_DRONE_2 = (36.146, -113.8898, 1425.6)
LAT_LIMIT = 90.0
LON_LIMIT = 180.0
MIN_PHYSICS_HZ = 10.0
MAX_PHYSICS_HZ = 1000.0
DRONE_COUNT = 2
WEATHER_PRESETS = (
    "clear_skies",
    "partly_cloudy",
    "cloudy",
    "overcast",
    "foggy",
    "rain_light",
    "rain",
    "rain_thunderstorm",
    "snow_light",
    "snow",
    "snow_blizzard",
    "dust",
)


def coordinates(values: list[str]) -> tuple[float, float, float]:
    """Parse and validate latitude, longitude, and WGS84 altitude."""
    try:
        lat, lon, alt = (float(value) for value in values)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("coordinates must be numbers: LAT LON ALT") from exc
    if not (-LAT_LIMIT <= lat <= LAT_LIMIT and -LON_LIMIT <= lon <= LON_LIMIT):
        raise argparse.ArgumentTypeError("latitude must be within +/-90 and longitude within +/-180")
    if not all(math.isfinite(value) for value in (lat, lon, alt)):
        raise argparse.ArgumentTypeError("coordinates must be finite")
    return lat, lon, alt


def parse_args() -> argparse.Namespace:
    """Read initial conditions from the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--address", default="localhost:10010", help="PteroSim gRPC endpoint")
    parser.add_argument("--aircraft", default="x500", help="installed drone class (default: x500)")
    parser.add_argument("--drone-1", nargs=3, metavar=("LAT", "LON", "ALT"), default=DEFAULT_DRONE_1)
    parser.add_argument("--drone-2", nargs=3, metavar=("LAT", "LON", "ALT"), default=DEFAULT_DRONE_2)
    parser.add_argument(
        "--start-time",
        default="2026-06-21T12:00:00",
        help="local date and time at the map origin, ISO format; use 22:00 for night",
    )
    parser.add_argument("--weather", choices=WEATHER_PRESETS, default="clear_skies")
    parser.add_argument("--physics-hz", type=float, default=1000.0, help="physics frequency, 10..1000 Hz")
    parser.add_argument("--replace-existing", action="store_true", help="remove existing aircraft before spawning two")
    parser.add_argument("--no-start", action="store_true", help="leave simulation stopped after setup")
    args = parser.parse_args()
    try:
        args.drone_1 = coordinates(args.drone_1)
        args.drone_2 = coordinates(args.drone_2)
        args.start_time = datetime.fromisoformat(args.start_time)
        if args.start_time.tzinfo is not None:
            raise ValueError("time zone is not accepted")
    except (ValueError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    if not MIN_PHYSICS_HZ <= args.physics_hz <= MAX_PHYSICS_HZ:
        parser.error("--physics-hz must be between 10 and 1000")
    if args.drone_1[:2] == args.drone_2[:2]:
        parser.error("drones need distinct latitude/longitude positions")
    return args


def main() -> int:
    """Apply world conditions, spawn two aircraft, and fit their flight stacks."""
    args = parse_args()
    with PteroSim(args.address) as sim:
        existing = sim.aircraft_status()
        if existing and not args.replace_existing:
            raise SystemExit("Scene has aircraft; use --replace-existing to remove them explicitly.")

        if sim.status().is_running:
            sim.stop()
        if existing:
            for aircraft in existing:
                sim.get_aircraft(aircraft.instance_id).remove()
            if sim.aircraft_status():
                raise RuntimeError("Could not clear existing aircraft; setup stopped.")

        # A Cesium map already open in PteroSim supplies terrain and imagery here.
        # On a tiled map set_geo_origin and spawn wait for terrain to load.
        sim.set_geo_origin(lat=args.drone_1[0], lon=args.drone_1[1], alt=args.drone_1[2])
        sim.set_start_time(args.start_time)
        sim.set_weather(args.weather)
        sim.set_physics_frequency(args.physics_hz)

        for number, (lat, lon, alt) in enumerate((args.drone_1, args.drone_2), start=1):
            aircraft = sim.spawn(args.aircraft, lat=lat, lon=lon, alt=alt)
            stack = aircraft.set_flight_stack("ArduPilot")
            print(
                f"Drone {number}: instance={aircraft.instance_id}, ArduPilot JSON/UDP port={aircraft.link_port}, "
                f"position=({lat:.6f}, {lon:.6f}, {alt:.1f}), stack={stack}"
            )

        if not args.no_start:
            sim.start()
        status = sim.status()
        if (
            status.aircraft_count != DRONE_COUNT
            or status.physics_frequency_hz != args.physics_hz
            or status.is_running == args.no_start
        ):
            raise RuntimeError(f"Unexpected simulator state: {status}")
        print(
            f"Simulation: running={status.is_running}, physics={status.physics_frequency_hz:g} Hz, "
            f"local start={args.start_time.isoformat()}, weather={sim.get_weather().preset}"
        )
        print("Start one ArduPilot SITL instance per drone; fitting the stack does not launch SITL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
