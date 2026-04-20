# Architecture

Drone AI is built from **four independent AI modules** that collaborate through a thin orchestration layer (`DroneAI`).

## Layer diagram

```
       ┌──────────────────────────────────────────┐
       │              DroneAI (orchestrator)      │
       │                                          │
       │   ┌──────────────┐    ┌──────────────┐   │
       │   │   Manager    │──▶│   Pathfinder  │   │
       │   │  (Layer 1)   │    │   (Layer 2)  │   │
       │   └──────────────┘    └──────┬───────┘   │
       │         ▲                    │           │
       │         │                    ▼           │
       │   ┌──────────────┐    ┌──────────────┐   │
       │   │  Perception  │◀──│   FlyControl  │   │
       │   │  (Layer 3)   │    │   (Layer 4)  │   │
       │   └──────────────┘    └──────────────┘   │
       └──────────────────────────────────────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │   QuadrotorPhysics  │
            │   (simulation)      │
            └─────────────────────┘
```

## Data flow per tick

1. **Perception** observes the world, returns noisy detections based on grade
2. **Pathfinder** updates its internal world model from confirmed detections
3. **Manager** picks the next delivery (or decides to return to base)
4. **Pathfinder** plans a path to the manager's chosen destination
5. **FlyControl** converts the next waypoint into motor commands via PPO policy
6. **QuadrotorPhysics** integrates the motor commands forward by `DT`

## Why separate AIs?

Each module has different ML requirements:

| Module | Best-suited approach | Why |
|--------|---------------------|-----|
| flycontrol | PPO RL | continuous control, sparse rewards |
| pathfinder | A\* + RRT | deterministic, optimal on grid |
| perception | CNN + Kalman | pattern recognition + tracking |
| manager | heuristic or RL | discrete scheduling |

Training them separately means each can use the right algorithm, have its own curriculum, and be graded independently.

## Physics simulation

`QuadrotorPhysics` is a self-contained quadrotor simulator (no external deps):

- **State:** `position(3), velocity(3), orientation(3), angular_velocity(3)` = 12D
- **Action:** 4 motor commands in `[0, 1]`
- **X-configuration** motor layout
- **Euler integration** at 50 Hz (`DT = 0.02s`)
- **Crash conditions:** hard ground impact, tilt > 60°

## Observation space (FlyControl)

19-dimensional vector, all normalized to `[-1, 1]`:

```
[0:3]   position / 50m
[3:6]   velocity / 15 m/s
[6:9]   orientation (roll, pitch, yaw) / π
[9:12]  angular_velocity / 6 rad/s
[12:15] (target - position) / 50m
[15]    distance_to_target / 100m
[16]    battery ∈ [0, 1]
[17]    nearest_obstacle_distance / 20m
[18]    carrying_package ∈ {0, 1}
```

## File layout

```
src/drone_ai/
├── grading.py              # Unified tier system + scoring
├── simulation/             # Shared physics and world
│   ├── physics.py
│   └── world.py
├── modules/
│   ├── flycontrol/         # PPO motor control
│   ├── pathfinder/         # A*/RRT planning
│   ├── perception/         # Detection + tracking
│   └── manager/            # Mission planning
├── drone.py                # DroneAI orchestrator
├── curriculum.py           # Full training pipeline
├── experiment.py           # Grade-mixing experiments
└── cli.py                  # Entry point
```
