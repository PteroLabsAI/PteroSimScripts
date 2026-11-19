"""Configure two ArduPilot x500 drones on the currently open PteroSim map.

The map must already contain the desired Cesium tileset. Geographic spawn and
origin calls wait for Cesium terrain beneath the requested coordinates.

Example (run from the PteroSimScripts directory):
    python python_examples/initial_conditions_two_ardupilot.py
    python python_examples/initial_conditions_two_ardupilot.py --start-time 2026-06-21T22:00:00

Pass --drone-1 LAT LON ALT and --drone-2 LAT LON ALT to change locations.

The altitude is WGS84 metres. Ground spawns stand on loaded terrain; the
altitude does not set height above ground. Pick flat ground: a drone spawned on
a slope tips over and slides down it once the simulation starts.

The script ends by printing the ArduPilot SITL command that flies both drones;
run it on the simulator's machine.
"""

from __future__ import annotations

import argparse
from datetime import datetime

from pterosim import PteroSim

AIRCRAFT = "x500"
# Grand Canyon airport (KGCN): x500s stood level there (0.2 deg) on 2026-11-19; alt is the ground measured then.
DEFAULT_DRONE_1 = (35.9524, -112.1470, 1979.0)
DEFAULT_DRONE_2 = (35.9524, -112.1468, 1979.0)


def parse_args() -> argparse.Namespace:
    """Read initial conditions from the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--address", default="localhost:10010", help="PteroSim gRPC endpoint")
    parser.add_argument("--drone-1", nargs=3, type=float, metavar=("LAT", "LON", "ALT"), default=DEFAULT_DRONE_1)
    parser.add_argument("--drone-2", nargs=3, type=float, metavar=("LAT", "LON", "ALT"), default=DEFAULT_DRONE_2)
    parser.add_argument(
        "--start-time",
        type=datetime.fromisoformat,
        default="2026-06-21T12:00:00",
        help="local date and time at the map origin, ISO format; use 22:00 for night",
    )
    parser.add_argument("--weather", default="clear_skies", help="weather preset, e.g. partly_cloudy, rain, snow")
    parser.add_argument("--physics-hz", type=float, default=1000.0, help="physics frequency in Hz")
    parser.add_argument("--replace-existing", action="store_true", help="remove existing aircraft before spawning two")
    parser.add_argument("--no-start", action="store_true", help="leave simulation stopped after setup")
    return parser.parse_args()


def main() -> int:
    """Apply world conditions, spawn two aircraft, and fit their flight stacks."""
    args = parse_args()
    drones = (args.drone_1, args.drone_2)
    with PteroSim(args.address) as sim:
        existing = sim.aircraft_status()
        if existing and not args.replace_existing:
            raise SystemExit("Scene has aircraft; use --replace-existing to remove them explicitly.")

        # Refusable calls go first, so a bad value leaves the aircraft; the origin moves only in an empty scene.
        sim.set_weather(args.weather)
        sim.stop()  # a running or paused simulation goes Idle; an Idle one stays as it is
        sim.set_physics_frequency(args.physics_hz)
        for aircraft in existing:
            sim.get_aircraft(aircraft.instance_id).remove()
        sim.set_geo_origin(lat=args.drone_1[0], lon=args.drone_1[1], alt=args.drone_1[2])
        sim.set_start_time(args.start_time)

        for number, (lat, lon, alt) in enumerate(drones, start=1):
            aircraft = sim.spawn(AIRCRAFT, lat=lat, lon=lon, alt=alt)
            stack = aircraft.set_flight_stack("ArduPilot")
            print(
                f"Drone {number}: instance={aircraft.instance_id}, ArduPilot JSON/UDP port={aircraft.link_port}, "
                f"position=({lat:.6f}, {lon:.6f}, {alt:.1f}), stack={stack}"
            )

        if not args.no_start:
            sim.start()
        status = sim.status()
        print(
            f"Simulation: running={status.is_running}, physics={status.physics_frequency_hz:g} Hz, "
            f"local start={args.start_time.isoformat()}, weather={sim.get_weather().preset}"
        )
        print(
            f"Fly them with ArduPilot SITL, run from the PteroSimAircrafts/{AIRCRAFT} folder:\n"
            f"  sim_vehicle.py -v ArduCopter -f quad --model JSON:127.0.0.1 -n {len(drones)} --auto-sysid "
            f"--add-param-file=firmwares/ardupilot_{AIRCRAFT}.param"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
