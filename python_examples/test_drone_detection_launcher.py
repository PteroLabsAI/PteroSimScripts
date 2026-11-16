"""Verify isolated PX4 resources using a fake executable; runs in Linux/WSL."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


@unittest.skipIf(os.name == "nt", "Run this test using WSL python3")
class LauncherTest(unittest.TestCase):
    """Use the actual Bash launcher without starting a real autopilot."""

    def test_private_resources_and_endpoint(self) -> None:
        """Airframe and endpoint edits must affect only this run's private copy."""
        launcher = Path(__file__).with_name("drone_detection_px4.sh")
        with tempfile.TemporaryDirectory(prefix="pterosim-launch-test-") as directory:
            root = Path(directory)
            build = root / "build with spaces"
            (build / "bin").mkdir(parents=True)
            init = build / "etc/init.d-posix"
            (init / "airframes").mkdir(parents=True)
            mavlink = init / "px4-rc.mavlink"
            original = "mavlink start -x -u $udp_offboard_port_local -o $udp_offboard_port_remote\n"
            mavlink.write_text(original)
            binary = build / "bin/px4"
            binary.write_text('#!/bin/sh\nprintf "%s\\n" "$@" "$PX4_SYS_AUTOSTART" "$PX4_SIM_HOSTNAME" > launch.txt\n')
            binary.chmod(0o755)
            firmware = root / "firmware"
            firmware.write_bytes(b"param set-default MAV_TYPE 22\r\n")
            runtime = root / "runtime"
            subprocess.run(
                [
                    "bash",
                    str(launcher),
                    str(build),
                    str(firmware),
                    str(runtime),
                    "2",
                    "192.0.2.1",
                    "192.0.2.2",
                    "15542",
                ],
                check=True,
                timeout=10,
            )
            self.assertEqual(mavlink.read_text(), original)
            private_init = runtime / "etc/init.d-posix"
            private_link = (private_init / "px4-rc.mavlink").read_text()
            self.assertIn("-u 14602 -o 15542 -t 192.0.2.2", private_link)
            airframe = (private_init / "airframes/31000_pterosim_detection").read_bytes()
            self.assertNotIn(b"\r", airframe)
            self.assertIn(b"param set-default MAV_TYPE 22", airframe)
            self.assertEqual(
                (runtime / "launch.txt").read_text().splitlines(),
                ["-i", "2", "-d", str(runtime / "etc"), "31000", "192.0.2.1"],
            )
            self.assertTrue((runtime / "px4.pid").read_text().strip().isdigit())


if __name__ == "__main__":
    unittest.main()
