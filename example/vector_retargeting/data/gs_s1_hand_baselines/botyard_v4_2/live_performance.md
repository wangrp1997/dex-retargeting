# Botyard v4.2 live scheduling and geometry reuse

The webcam path separates camera acquisition, vision/MANO display and robot
optimization. This follows the practical separation of tracking, retargeting and
control in [DexPilot](https://arxiv.org/abs/1910.03135), while retaining Botyard's
existing objective and physical constraints. It introduces no gesture rules or
contact switches. Video replay and synchronized recording retain serial execution.

## Exact geometry reuse

Repeated identical actuator states reuse MuJoCo kinematics and contacts. Distal
and palm distance queries reuse exact STL distances and nearest points only when
both queried body transforms match exactly. Value-only calls can subsequently
supply analytic Jacobians at the current MuJoCo state. Moving one finger leaves
the other unchanged thumb/finger pairs eligible for reuse. Meshes, margins,
source protections, objective weights and optimizer budgets are unchanged.

On 2026-11-28, the same right-hand example observations were replayed from source
indices 300–331: eight warmup frames and 24 measured frames (308–331). Each run
started a fresh causal fitter and solver, with one BLAS thread and process niceness
10. Baseline geometry modules came from commit `c5b983d`. This is a small CPU
microbenchmark, excluding camera, MediaPipe, drawing, ROS transport and hardware.
The measurements are in [live_performance.json](live_performance.json).

| Stage | Before mean, ms | Cached mean, ms |
| --- | ---: | ---: |
| MANO fit | 7.38 | 7.54 |
| Finger shape | 19.94 | 19.47 |
| Thumb base | 7.52 | 7.45 |
| Surface closure | 111.55 | 100.87 |
| Total fit and retarget | 151.45 | 140.47 |

Total mean time fell about 7.3%; p95 fell from 206.87 to 188.11 ms. Full mesh
closure remains the largest solver cost. This does not imply a 30 Hz robot solver.

A further local change on 2026-11-28 retains exact distal gaps and analytic
Jacobians for up to 32 previously evaluated actuator states. This also reuses
nonconsecutive trial states revisited by line search. MuJoCo kinematics still
describe the requested state, and returned arrays cannot modify cached results.
Against the existing single-pose cache, another paired run of the same 24 inputs
measured 139.09 to 133.62 ms total mean (3.9% reduction) and 185.39 to 178.86 ms p95.
This is a small cached-input benchmark, not a measured hardware latency gain.
Full example/custom/sweep replay produced exactly identical commanded joints
(621/395/149 frames), with the same source/physical protections and numerical
fallbacks. The 25 geometry, source-priority and analytic-gradient tests passed.
Optimizer tolerances, evaluation budgets and objective weights were not changed.

To reproduce on an existing cached observation file, in the sourced ROS workspace:

```zsh
PYTHONPATH="$PWD/src:$PWD/example/vector_retargeting:$PYTHONPATH" \
OPENBLAS_NUM_THREADS=1 nice -n 10 \
  ~/miniconda3/envs/dex_retargeting/bin/python \
  example/vector_retargeting/data/gs_s1_hand_baselines/benchmark_botyard_live.py \
  --source /path/to/source.npz --start 300 --warmup 8 --frames 24
```

The benchmark creates no ROS node, camera capture or motion publisher. Its optional
`--out` writes a timing JSON, not a video.

## Independent visual updates

A production spawned solver was exercised with 90 cached observations offered at
30 Hz, source indices 290–379, and causal MANO fitting plus mesh drawing in the main
process. A tracking outage was injected at inputs 40–43. The loop processed 90
inputs in 3.106 s (28.98 Hz), including 86 fitted/drawn MANO frames. Fit and draw
averaged 16.02 ms (p95 16.98 ms). The worker produced 24 accepted robot commands
(7.73 Hz), with one old result discarded. Mean accepted solve time was 97.60 ms;
mean source-to-result age was 136.17 ms (p95 182.07 ms). All accepted commands
passed independent exact geometry and velocity checks.

These are cached-input scheduling measurements. They exclude camera acquisition,
MediaPipe, window presentation, ROS transport and real hardware. The worker sees
a subset of observations, so its mean solve time is not directly comparable to
the same-frame serial microbenchmark. Actual camera refresh and hand response must
be measured in the operator's live session.

The live path keeps at most one camera frame and one in-flight solve. It dispatches
the current MANO pose when idle. It rejects results older than 0.5 s or from an
earlier tracking period. Every task carries the last published actuator command,
so a discarded result cannot advance the physical warm start. Source timestamps
survive solving and transport, including in the serial comparison path. The
hardware relay counts this age against its target timeout. The UI exposes vision
time, solve time, frame age and command age separately.

Use the existing launch command with `objective:=botyard`; `live_pipeline:=latest`
is now the webcam default. Select `live_pipeline:=serial` for comparison. Hand
selection and backend selection remain launch arguments.

## Regression checks

Full deterministic serial replay uses the existing cached MANO fits: official
example (621 tracked frames), custom recording (395 total, 330 tracked), and sweep
(149 tracked). Before/after commanded joints match exactly. Physical clearance,
joint limits, velocity and source-shape protections pass. Existing numerical
fallbacks and v4.2 gesture limitations are retained; this change improves runtime
scheduling rather than changing the contact policy.

Automated tests cover cached values and derivatives on both hands, camera draining
and failure, bounded dispatch, tracking loss/reacquisition, capture gaps, invalid
and expired replies, preservation of input timestamps, hardware expiration that
includes solving time, process errors and cleanup during an unfinished solve.
The default controller relay is also exercised with GenericSystem mock hardware
in a separate ROS domain, without opening RS485 or sending real-hand commands.
