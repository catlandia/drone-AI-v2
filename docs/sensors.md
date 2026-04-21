# Sensors

The drone must localise and act from onboard sensors alone. No GPS.
No satellites. No cloud. No absolute world coordinates.

This is a hard design constraint driven by the operating environment:
the drone may be flying in an area where internet is jammed or
deliberately cut off. Any sensor that requires a remote service is
forbidden.

## Sensor suite

| Sensor | Provides | Notes |
|--------|----------|-------|
| IMU | Orientation, angular velocity, (derived) linear velocity, acceleration | Primary attitude reference |
| Barometer | Altitude above takeoff | Drifts with weather; fused with VIO z |
| VIO (camera + IMU) | Displacement from takeoff | Drift-accumulating; relative only |
| Camera + Perception-Obstacles | Nearest-obstacle distance | Plus Perception-Hazards/Targets/Agents for classes |
| ToF | Close-range obstacle distance | Complements camera near ground / in dust |
| Battery monitor | State-of-charge, temperature | Temperature feeds sag model |
| Internal state | Carrying flag | Set by Manager at pickup/drop |

## What is NOT available

- **GPS / GNSS.** Explicitly ruled out. Any attempt to wire it in is a
  design bug.
- **Cloud maps, weather APIs, traffic feeds, any remote data source.**
- **Ground-truth absolute world position.** The drone only knows its
  displacement from takeoff.
- **Radio comms mid-mission.** See [`comms.md`](comms.md).

## Noise model (planned Phase 1)

Realistic training requires realistic noise. Each sensor gets a
grade-parameterized noise profile in the same spirit as Perception's
detection noise:

- **IMU:** gyro bias drift, accelerometer white noise.
- **VIO:** integrated drift (growing with distance travelled).
- **Barometer:** slow drift + pressure-front shock.
- **Camera / ToF:** range-dependent variance, occasional dropouts.
- **Battery monitor:** sag that depends on temperature + cell age.

Grade-linked profiles let us train policies against a wide spread of
hardware conditions so Phase 1.5 doesn't face a sim-to-real cliff.

## Why this matters for the architecture

- FlyControl's observation vector can **only** contain data these
  sensors can produce. No "target in world coords" — only
  displacement from takeoff, which is what VIO measures.
- Pathfinder plans in the drone's own relative frame, not a global
  one.
- Manager's feasibility checks use battery-sag + VIO-drift margins,
  not map-lookups.
- Perception-Targets does the landing-pad re-acquisition on return —
  because the VIO drift by then makes the takeoff origin unreliable.
