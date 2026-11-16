# PX4 drone detection recording

The demo spawns an F450 and X500 18 m in front of a camera by default.
Each vehicle gets its own real PX4 SITL process, the airframe configuration from
`PteroSimAircrafts/<model>/firmwares/px4_<model>`, and a MAVLink connection.
After confirmed arming and Offboard entry, they climb to 7 and 8 m and fly
small elliptical routes in separate lanes. Additional hover-capable models can
be selected with `--models`; VTOL models stay in multicopter mode. The standard
VTOL reported rotor strikes and the tiltrotor reported a hard landing during
live testing with the installed v0.5.0 aircraft collection, so neither is a default.
The controller captures each vehicle's local NED origin independently and sends
position targets at 30 Hz in a background thread, independent of inference speed.

The camera is an unpowered, grounded X500 with its `camera` sensor on a 3 m
virtual mast, pitched up 12 degrees. This uses the existing SDK camera API;
it does not require a new Unreal actor or changes to aircraft files. Ground under
the camera and drone spawn positions should be flat and unobstructed. Choose a different
location with `--x`, `--y`, `--z` (Unreal cm) if necessary.

## Setup on Windows

Open PteroSim and enter a play session so gRPC is listening. Leave the simulator
stopped with an empty aircraft outliner. The script refuses occupied or running
scenes, and removes only the vehicles it creates when finished.

From `PteroSimScripts`, use a virtual environment with the current PteroSim SDK:

```powershell
python -m venv pterosim-venv
.\pterosim-venv\Scripts\Activate.ps1
python -m pip install -e ..\Plugins\PteroSimScripting\SDK\python
python -m pip install -r python_examples\drone_detection_requirements.txt
```

PX4 SITL must already be built inside WSL Ubuntu-24.04, by default at
`~/Documents/PX4-Autopilot/build/px4_sitl_default`. Use `--px4-build` and
`--wsl-distro` for different locations. The aircraft collection must be installed
in the simulator; `--aircraft-root` must point to the matching on-disk collection.
The default is the sibling `PteroSimAircrafts` directory.

For WSL NAT, obtain the Windows address visible from Linux:

```powershell
$simHostIp = (wsl -d Ubuntu-24.04 --exec bash -lc "ip route show default").Trim().Split()[2]
python python_examples\drone_detection_demo.py --sim-host $simHostIp --api-host $simHostIp
```

For WSL mirrored networking, both host arguments can usually remain `127.0.0.1`.
The gRPC address defaults to `localhost:10010` because Python runs on Windows.
Windows must accept PX4 TCP connections to HIL ports `4560 + instance_id` and
UDP telemetry on 15540–15541 by default (one additional port per added drone).
Do not run other SITL processes with the same IDs
or applications bound to these API ports. The demo temporarily sets the
simulator's MAVLink bind address to `0.0.0.0` and restores it on exit.

The launcher copies PX4's `etc` into the recording directory, installs a private
demo airframe there, and changes that copy's API destination. It never changes
the PX4 build's shared resources or the aircraft collection. Parameter files,
logs, and instance state are isolated in the recording directory.

## Linux

Install the same requirements and SDK in a virtual environment, then run:

```bash
python python_examples/drone_detection_demo.py --px4-build "$HOME/Documents/PX4-Autopilot/build/px4_sitl_default"
```

If Python and PX4 run together in WSL while the simulator runs on Windows, pass
`--address WINDOWS_IP:10010 --sim-host WINDOWS_IP`; leave `--api-host 127.0.0.1`.

## Video and detection classes

A preview window shows the annotated camera. Press **Q** or **Escape** to finish
early; `--no-preview` records without a window. The default duration is 60 seconds.
PX4 receives LAND on exit; the script waits up to 45 seconds for disarming,
then stops its simulation and firmware processes. Ctrl+C also enters cleanup.

Each run creates `recordings/<timestamp>/` containing:

- `detections.mp4`: 1280×720 video with colored rectangles, predicted class,
  confidence and ByteTrack ID. Slow inference duplicates frames to preserve
  approximate wall time playback speed; it does not increase inference rate.
- `detections.jsonl`: model boxes, confidence, track IDs, capture sequence,
  simulation and wall times, and corresponding video frame intervals.
- `run.json`: scene IDs, detector class names, weights, and arguments.
- `px4-*.log` and `px4-*/`: firmware output and private runtime resources.

The first run downloads `best.pt` from
[TomSmail/drone-yolo-v1](https://huggingface.co/TomSmail/drone-yolo-v1).
This YOLOv8n model has **quadcopter** and **fixed-wing** classes. It does not
recognize F450 versus X500, and its fixed-wing training examples differ from
the simulated VTOL in hover. In the live F450/X500 test, both quadcopters were
incorrectly labeled fixed-wing; the initial scene also produced a false detection
on the moon. Boxes appear only for actual model detections. Missing or mistaken detections
are possible, especially for the VTOL. The script reports an error if the entire
recording has no detections, retaining the video and logs for inspection.

To distinguish exact aircraft types visually, train Ultralytics detection weights
with those classes and pass `--weights path/to/best.pt`; labels and colors follow
the weights' class map automatically. No ground-truth simulator labels are
presented as visual predictions. ByteTrack IDs distinguish detected individuals,
but are not guaranteed to persist through occlusion.

Other useful settings:

```powershell
python python_examples\drone_detection_demo.py --device 0 --imgsz 1280 --conf 0.2 --duration 90
python python_examples\drone_detection_demo.py --weights .\models\drone-types.pt --no-preview
python python_examples\drone_detection_demo.py --models F450 x500 --y 8000 --inference-threads 4
```

`--models` selects the hover-capable fleet. If a VTOL in your installed aircraft
collection fails takeoff, use `--models F450 x500` to test the camera and recording
with the two quadcopter models. CPU inference defaults to four threads to leave
capacity for simulation and MAVLink control. On the military airport map tested
with the v0.5.0 Windows build, the default origin is among pipes and containers;
`--y 8000` moves the scene toward the open area beside the road. Check clearance
on your map before flying.

For a downloaded simulator build, pass `--aircraft-root` pointing to its own
`PteroSim/PteroSimAircrafts` folder. Its aircraft files can differ from this
workspace. F450's channel 4 gripper binding must name the gripper actually in
Visual.xml (`gripper="gripper_1"` when the visual part is named `gripper_1`).

CPU is the default. CUDA requires a compatible PyTorch installation.
The TomSmail weights are attributed to FRAID Labs and distributed under CC-BY-4.0;
see the model card and the licenses of Ultralytics and its dependencies.
The control approach follows [PteroLabsAI/XTDrone](https://github.com/PteroLabsAI/XTDrone)
using MAVLink directly without ROS. The small `drone_detection_px4.py` adapter
contains the MAVLink connection and setpoint operations needed by this demo.

## Local verification

```powershell
python -m pytest python_examples/test_drone_detection_demo.py
python -m ruff check python_examples/drone_detection_demo.py python_examples/test_drone_detection_demo.py
wsl -d Ubuntu-24.04 --exec python3 /mnt/c/path/to/PteroSimScripts/python_examples/test_drone_detection_launcher.py
```

These tests exercise route geometry, real OpenCV overlays/MP4 encoding, and flight
failure handling with synthetic inputs. They do not validate PX4 flight dynamics,
the rendered scene, or model accuracy. A running simulator and installed detector
dependencies are required for the actual recording.
