"""Configure two ArduPilot x500 drones on the currently open PteroSim map.

The map must already contain the desired Cesium tileset. Geographic spawn and
origin calls wait for Cesium terrain beneath the requested coordinates. Two
drones need an Edu or Pro license: a Free license spawns one aircraft.

Example (run from the PteroSimScripts directory):
    python python_examples/initial_conditions_two_ardupilot.py
    python python_examples/initial_conditions_two_ardupilot.py --start-time 2026-06-21T22:00:00

Pass --drone-1 LAT LON ALT and --drone-2 LAT LON ALT to change locations.

The altitude is WGS84 metres. Ground spawns stand on loaded terrain; the
altitude does not set height above ground. Pick flat ground: a drone spawned on
a slope tips over and slides down it once the simulation starts.

The script ends by printing the ArduPilot SITL command that flies both drones.
Its JSON:127.0.0.1 reaches the simulator from SITL on the same machine, or from
WSL with networkingMode=mirrored; otherwise put the Windows host's address there.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime

from pterosim import PteroSim

AIRCRAFT = "x500"
# Grand Canyon airport (KGCN): x500s stood level there (0.2 deg) on 2026-11-19; alt is the ground measured then.
DEFAULT_DRONE_1 = (35.9524, -112.1470, 1979.0)
DEFAULT_DRONE_2 = (35.9524, -112.1468, 1979.0)
# The rate the ArduPilot SITL page of the docs recommends; the printed command flew both drones at it.
ARDUPILOT_PHYSICS_HZ = 1000.0
# WGS84 bounds: the poles and the antimeridian.
LAT_LIMIT_DEG = 90.0
LON_LIMIT_DEG = 180.0


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
        help="local date and time at the map origin, ISO format without a time zone; use 22:00 for night",
    )
    parser.add_argument("--weather", default="clear_skies", help="weather preset, e.g. partly_cloudy, rain, snow")
    parser.add_argument("--physics-hz", type=float, default=ARDUPILOT_PHYSICS_HZ, help="physics frequency in Hz")
    parser.add_argument("--replace-existing", action="store_true", help="remove existing aircraft before spawning two")
    parser.add_argument("--no-start", action="store_true", help="leave simulation stopped after setup")
    args = parser.parse_args()
    # Checked here: the spawn takes any finite point, and the origin refuses one only after the removal.
    for lat, lon, alt in (args.drone_1, args.drone_2):
        if not (abs(lat) <= LAT_LIMIT_DEG and abs(lon) <= LON_LIMIT_DEG and math.isfinite(alt)):
            parser.error(f"{lat} {lon} {alt}: need |lat| <= {LAT_LIMIT_DEG:g}, |lon| <= {LON_LIMIT_DEG:g}, finite alt")
    if tuple(args.drone_1[:2]) == tuple(args.drone_2[:2]):
        parser.error("drones need distinct latitude/longitude positions")
    return args


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
        sim.set_start_time(args.start_time)  # kept as local time, so it holds at the origin set below
        sim.stop()  # a running or paused simulation goes Idle; an Idle one stays as it is
        sim.set_physics_frequency(args.physics_hz)
        for aircraft in existing:
            sim.get_aircraft(aircraft.instance_id).remove()
        sim.set_geo_origin(lat=args.drone_1[0], lon=args.drone_1[1], alt=args.drone_1[2])

        for number, (lat, lon, alt) in enumerate(drones, start=1):
            aircraft = sim.spawn(AIRCRAFT, lat=lat, lon=lon, alt=alt)
            aircraft.set_flight_stack("ArduPilot")
            print(
                f"Drone {number}: instance={aircraft.instance_id}, ArduPilot JSON/UDP port={aircraft.link_port}, "
                f"position=({lat:.6f}, {lon:.6f}, {alt:.1f})"
            )

        if not args.no_start:
            sim.start()
        status = sim.status()
        print(f"Simulation: running={status.is_running}, physics={status.physics_frequency_hz:g} Hz")
        print(
            f"Fly them with ArduPilot SITL, run from the PteroSimAircrafts/{AIRCRAFT} folder:\n"
            f"  sim_vehicle.py -v ArduCopter -f quad --model JSON:127.0.0.1 -n {len(drones)} --auto-sysid "
            f"--add-param-file=firmwares/ardupilot_{AIRCRAFT}.param"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
