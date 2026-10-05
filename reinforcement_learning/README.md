# Reinforcement learning on PteroSim step mode

An F450 learns to take off from the ground, climb straight up to a height above where it stands and hold it and a
heading, from its sensors. Every aircraft in the world is one environment: step mode advances all of them together,
as fast as the CPU allows, and PPO (Stable-Baselines3) trains one policy on the whole fleet.

| File | What it holds |
|---|---|
| `hover.py` | The task (Isaac Lab's quadcopter reward, targets 0.5-1.5 m above the start), the evaluation, PPO, `train` and `eval` |
| `show_hover.py` | Flies saved policies one after another with the whole fleet, in real time, to watch them learn |
| `step_env.py` | Step mode as an SB3 `VecEnv`; the startup measurements (lift-off throttle, motor mixer, sensor noise) |
| `estimator.py` | What the policy sees: attitude, position and velocity estimated from the IMU, magnetometer and baro |
| `fleet.py` | Spawning F450s on a grid and entering step mode |
| `frames.py` | Rotations between the body frame and NED |
| `test_hover.py` | All of the above against a fake step mode, no simulator needed |

## Setup

Install the `pterosim` package from the simulator (see the top README), then:

```bash
python -m pip install -r reinforcement_learning/requirements.txt
```

Start the simulator on a map with flat ground around the world origin and leave it stopped with no aircraft: the
scripts spawn their own. Training needs as many aircraft as `--n-envs`, so your licence must allow that many.

## Train, evaluate, watch

```bash
cd reinforcement_learning
python hover.py train --n-envs 100 --total-steps 10000000 --checkpoint-every 1000000 --out runs/hover_s0
python hover.py eval --n-envs 100 --model runs/hover_s0/model.zip
python show_hover.py untrained runs/hover_s0/checkpoints/hover_1000000_steps.zip runs/hover_s0/model.zip --count 100
```

A run writes `run.json` (the measured hover throttle, mixer and sensor model), checkpoints, CSV and TensorBoard logs.
The evaluation flies one 10 s episode per aircraft and passes when, over its last 5 s, 95% survive with a position
error under 10 cm (median) and 25 cm (p95), a heading error under 5 deg and no spinning.

## Tests

```bash
cd reinforcement_learning
python -m pytest test_hover.py -q
```

## Limits

Step mode has no world collision, the ground under each aircraft is a plane at its start, and the wind is fixed for
the session. The UWB position fixes are a client-side measurement model (truth plus noise), not a simulator sensor.
