## Retarget Robot Motion from Human Hand Video

![teaser](teaser.webp)

### Botyard GS S1 V3.2 webcam preview

From the repository root on the Botyard laptop, source ROS 2 and the workspace, then choose a backend, hand and objective.

```zsh
source /opt/ros/humble/setup.zsh
source ~/botyard_ws/install/setup.zsh
ros2 launch "$PWD/example/vector_retargeting/gs_s1_preview.launch.py" \
  backend:=mujoco hand:=right objective:=botyard camera:=0
```

Use `backend:=rviz|mujoco|hardware`, `hand:=right|left` and `objective:=botyard|dexpilot|wuji|vector|tsinghua` (one value per argument).
The defaults are `rviz`, `right` and `dexpilot`; the hardware backend defaults to `left` and requires `objective:=botyard`. Use `video:=/path/to/video.mp4` to replay the same input.
The four landmark objectives apply One Euro smoothing to the wrist-local 3D landmarks before optimization.
Use `smoothing:=one_euro` (default) or `smoothing:=off` to compare them on the same video.
The camera overlay shows the original 2D detections. Press **q** in that window to stop.

For the real GS S1 over RS485, use the same entry point with the hardware backend:

```zsh
ros2 launch "$PWD/example/vector_retargeting/gs_s1_preview.launch.py" \
  backend:=hardware hand:=left objective:=botyard camera:=0 \
  botyard_shape:=v4_2 botyard_gradient:=analytic
```

This starts the existing `ros2_botyard` default `dexhand_controller`, the Botyard tracker and a position-message relay. The serial device defaults to `/dev/dexhand`; override it with `hardware_port:=/dev/ttyACM0` when needed. `start_hardware:=false` reuses an already running default controller, and `controller_namespace:=lh` selects a namespaced controller. The hardware description uses the same GS S1 sub-version 2 and handedness as the tracker. RS485 runs as the current user without the EtherCAT capability wrapper.

The relay reads the controller's actual `joints` parameter and maps targets by name to its 18-position `Float64MultiArray` protocol. It holds the measured wrist pose, starts finger motion from joint feedback, and limits command changes to the lower of the URDF velocity and the default controller's 1.5708 rad/s. Only `/gs_s1_preview/tracked_joint_targets` drives the relay, so waiting for a hand does not send zero targets. Missing tracked input or hardware feedback stops publication; resumption starts from fresh feedback. Controller speed, effort and tactile defaults remain those of `dexhand_controller`. No recording or video export is started.

The default controller chain was verified with mock hardware in an isolated ROS domain, including nonzero wrist feedback and stale-input stopping. This is software integration verification; real RS485 motion still needs the operator's test.

To save diagnostics for later analysis, add `hardware_log:=true` to the hardware launch command:

```zsh
ros2 launch "$PWD/example/vector_retargeting/gs_s1_preview.launch.py" \
  backend:=hardware hand:=left objective:=botyard camera:=0 hardware_log:=true
```

The relay prints the absolute session directory under `logs/hardware/YYYYMMDD-HHMMSS-xxxxxxxx/`.
Use `hardware_log_dir:=/path/to/logs` to choose another parent directory. The default is
`hardware_log:=false`; this diagnostic mode saves no images or videos. Normal ROS launch
console logs are independent of this switch. Press **Q** or **Ctrl+C** to end the session
and finish the diagnostic files:

- `events.jsonl`: source-stamped optimization targets, named controller commands, joint feedback,
  controller/relay state, optional stamped hardware acquisitions and vision/solver timings.
  Each command links to the accepted target sequence and its original source timestamp.
- `session.json`: hand/controller settings and recording coverage, including queue overflow,
  unfinished writes or disk errors. A bounded background writer keeps disk I/O out of relay callbacks.
- `summary.json`: observed update rates, input age, stage processing times and position errors.
  Feedback is compared with the most recent command published at its source timestamp.

The logger subscribes to `dexhand_interfaces/HandSample` when available. Its valid acquisition
sequences distinguish fresh hardware samples from `/joint_states` repeatedly publishing old
readings. If that driver telemetry is absent, the hardware acquisition rate remains unknown.
Receipt timestamps do not measure sensor exposure, RS485 application time or motor response onset;
the summary does not report an inferred motor delay. The default latest pipeline reports
detection, MANO fit and robot solve separately; serial comparison reports MANO plus solve together.
Logging retains the existing control settings and sends no extra movement commands.

For serial send/response/ACK timing, restart the hardware session with the additional
`hardware_io_trace:=true` switch:

```zsh
ros2 launch "$PWD/example/vector_retargeting/gs_s1_preview.launch.py" \
  backend:=hardware hand:=left objective:=botyard camera:=0 \
  hardware_log:=true hardware_io_trace:=true
```

This adds `io_trace.jsonl` inside the same session. It records SDK serial send entry/return,
response wait entry/return and validated ACK success, linked by request IDs and host wall/steady
timestamps. Control requests retain their encoded parameter bytes. Summary statistics separate
command IDs so startup version-probe failures do not look like motion-time failures. These are
host SDK boundaries: ACK acceptance is not firmware application time or motion completion.

The opt-in Linux tracer builds locally with `g++` against the installed SDK headers, checks its
exported RS485 functions, and loads only in the hardware bringup's process environment. Installed
libraries are retained. A bounded background writer saves the trace, including a dropped-event
count at clean shutdown. The cache is under the ignored log directory. This requires
`start_hardware:=true`, since an existing controller process cannot pick up the tracer on restart
of the camera alone. Both logging switches default to false. Virtual-serial tests compare the
unmodified SDK's outgoing packets and return values with tracing off/on, including NACK and timeout.

Sessions are ignored by Git. To regenerate a summary after an interrupted session, or inspect an
existing session, run the read-only command:

```zsh
~/miniconda3/envs/dex_retargeting/bin/python \
  example/vector_retargeting/gs_s1_hardware_log.py /path/to/session
```

The **Botyard** objective fits the selected MANO hand to each frame's 3D detections and 2D image points using the previous pose. It uses the template shape and estimates palm scale online. The offline RGB benchmark fits shape and pose jointly across the full recording. The live optimizer uses joint limits, passive couplings and mesh checks. The camera window overlays the translucent cyan MANO surface from the current fit under the MediaPipe skeleton. Press **M** in the camera window to hide or show the surface while retargeting continues.

Botyard webcam preview defaults to `live_pipeline:=latest`. A capture thread retains one latest frame, the main process detects/fits/displays the current hand, and a spawned process owns the complete robot solver. Only one solve is in flight; its next input is the latest available MANO pose. The window reports vision processing time, solver processing time, camera-frame age and command-source age separately. Optimization still runs more slowly than visualization. Both simulation and hardware use the same scheduling.

Robot solutions retain the camera receipt timestamp and publish once on `tracked_joint_targets`. Results older than 0.5 s, from a previous tracking period, or spanning a long capture gap are discarded. The next solve starts from the last published actuator command, including after a discarded result. The hardware relay counts solve/transport age toward its existing target timeout. Stale input stops new targets. Camera or worker failure exits the stream and cleans up the worker. Use `live_pipeline:=serial` to compare with the previous single-process webcam pipeline. Video replay and synchronized recording always use the serial path.

Geometry acceleration caches only identical actuator states or identical pairs of body transforms. It reuses exact STL nearest points for subsequent value/Jacobian calls; no mesh simplification, iteration reduction or objective change is used. Deterministic full-input replay is compared with the v4.2 reference. See [performance measurements and limits](data/gs_s1_hand_baselines/botyard_v4_2/live_performance.md).

The default `botyard_shape:=v4_2` selects **Botyard v4.2**, using analytic gradients by default. It builds the direction/bend and custom thumb-base reference described below, then minimizes gaps between the actual distal surfaces. Source-referenced per-digit bend and proximal-direction bounds prevent the contact task from repeatedly undoing a fist. Four-finger J4 remains free; CAD limits, passive coupling, velocity and collision checks remain active. During closure the custom thumb-facing field is a weak preference, allowing adjustment for index opposition. No gesture classifier or index-crossing switch is used. See [v4.2 parameters, replay and limitations](data/gs_s1_hand_baselines/botyard_v4_2/README.md). Use `botyard_shape:=sweep` for the reference without closure or `botyard_shape:=baseline` for historical v3.3. The upstream Python package version and GS S1 V3.2 CAD designation are independent of the Botyard algorithm version.

The comparison mode `botyard_shape:=sweep` uses two continuous kinematic fits. First it fits the four long fingers' segment directions and bends, plus the thumb's proximal direction and intrinsic bend angles. It then holds all four fingers and thumb J1/J2 fixed while optimizing only THJ3/THJ4 to balance the complete thumb's root-to-tip direction with its pad facing toward the other fingers. The direction is reconstructed from observed MANO bones using the robot's own segment lengths. The facing field combines vectors to the four fixed robot pad centres with smooth inverse-square weights; the 10 mm length unit regularizes coincident points. Distances weight orientation and are not position targets. Both fits use weak temporal smoothing and the joint, velocity and collision constraints. There is no index-axis crossing condition, inside/outside rotation curve, pinch detector, cross-digit distance target or surface-gap refinement. The recorded `thumb_target_direction` exposes the continuous palm-frame direction target. The thumb bend solve starts on the flexion branch beyond the CAD's small geometric offset at zero, avoiding a local minimum at its joint stop.

For live `sweep` and `v4_2`, the causal 3D/2D MANO keypoint fit keeps its own history. A following step refines only the final thumb IP rotation using 3D and projected distal directions with a low-curvature prior. It preserves the proximal chain and other fingers. Direction residuals reduce template-length mismatch without requiring a reference gesture. This does not guarantee a completely straight fitted thumb when its proximal chain is inaccurate, and can underestimate ambiguous curls. The recording field `mano_thumb_fit` identifies the source objective. Saved MANO fits retain their original bending; use `--refit-mano` with `compare_recorded_shapes.py` to evaluate the current fit from saved raw observations.

Use `botyard_shape:=direction` to compare with the earlier single-stage objective that fits all three unit bone directions and both bend angles of every digit. Both modes use the same palm rotation as the MuJoCo MANO overlay and One Euro landmark filtering. Four-finger J4 remains free to follow observed spread. `botyard_shape:=baseline` selects the original objective with distance refinement for comparison.

In the **MuJoCo window**, press **N** to toggle the translucent fitted MANO hand on the simulated GS S1. Drag to rotate the view. The cyan dot marks the MANO thumb tip, the orange dot marks the simulated thumb tip, and a line connects them in 3D. The two hands are registered at the index knuckle with the palm axes aligned and MANO scaled uniformly to the neutral robot palm width (index to little knuckle). The overlay starts off and disappears on tracking loss. This control is independent of **M** in the camera window. During recording, `mujoco.mp4` and the MuJoCo panel of `comparison.mp4` reproduce the overlay visible at capture time.

Use `botyard_gradient:=analytic` to test the analytical first-stage gradient with either backend. Use `botyard_gradient:=finite_difference` to compare the reference gradient with the default `analytic` gradient.

Add `botyard_shape:=phalanx` to try the earlier shape objective with thumb-to-finger position vectors. It fits the directions of all three segments of each long finger, aligns each finger's flexion plane with the observed proximal segment, and gives the thumb pad normal extra weight. The same residuals and weights apply throughout the sequence. Both gradient modes support it.

Use `botyard_shape:=thumb_span` to also track how far the thumb tip moves sideways across the palm. The target is the signed projection from the index knuckle toward the thumb side, expressed in palm widths. Both optimization stages track this span using the fitted MANO landmarks and the neutral GS S1 palm anchors. The distal-gap refinement also tracks the thumb pad normal. Compare it with `phalanx` on an existing recording using `--shapes phalanx thumb_span` in the replay command below.

#### Record the camera and simulated hand together

Close a running preview, then run `./example/vector_retargeting/record_gs_s1.zsh hand:=right camera:=0` from the repository root. Use `hand:=left` for the left hand. Focus the camera window and press **Q** to finish. The script exports H.264 videos to a new timestamped directory under `~/Videos/botyard/` and prints its full path.

`comparison.mp4` shows the clean camera image, MediaPipe skeleton, fitted MANO surface and simulated GS S1 in a 2×2 grid. `human.mp4`, `skeleton.mp4`, `mano.mp4` and `mujoco.mp4` contain the individual views. The three human views use the same processed camera frame. Each frame is paired at camera read time with the latest received MuJoCo joint positions and viewer camera. Missing detections are marked. Camera capture intervals are preserved on export. The simulator is sampled asynchronously and `frames.csv` reports its age in milliseconds for each frame. `sim_states.jsonl` preserves the sampled joint positions, including the passive joints, and the interactive viewing angle.

For a pinch diagnosis, `retargeting.jsonl` stores the published joint names, angles in radians and ROS command timestamp for every camera frame. Tracked frames also contain the 2D MediaPipe landmarks in pixels, wrist-relative MediaPipe and MANO data in camera axes (metres), and the first-stage joint solution. The `baseline`, `phalanx` and `thumb_span` modes additionally record measured MANO and target distal-surface gaps in thumb-to-index, middle, ring and little finger order; `direction` and `sweep` record a null gap target. `v4_2` also records `algorithm_version`, `closure_weights`, `closure_primary_position_rad`, protection margin and numerical fallback information; its gap target is the collision clearance. `tracking=false` marks a held command. In `sim_states.jsonl`, match `applied_command_stamp_ns` to `command_stamp_ns` to find which earlier camera command the simulator used for that state. The simulator logs its accepted actuator targets in `ctrl_rad`, joint positions in `qpos` (scene order), and its cumulative count of rejected physics steps. A null applied timestamp marks the period before the simulator received its first command.

The physics hand is rendered after capture to keep the live simulator responsive. If export is interrupted, run `MUJOCO_GL=osmesa ~/miniconda3/envs/dex_retargeting/bin/python example/vector_retargeting/gs_s1_recording.py /path/to/session` in the ROS environment to finish the saved session.

To compare the two objectives on the exact MANO observations and time steps saved by the recorder, run the following in the sourced ROS environment from the repository root. The CSV contains paired kinematic scores and commanded angles for every tracked frame.

```zsh
session=/path/to/recording
PYTHONPATH="$PWD/src:$PWD/example/vector_retargeting:$PYTHONPATH" OPENBLAS_NUM_THREADS=1 \
  ~/miniconda3/envs/dex_retargeting/bin/python example/vector_retargeting/data/gs_s1_hand_baselines/compare_recorded_shapes.py \
  "$session" --out "$session/diagnostics/phalanx_ab/command_metrics.csv"
```

The tracker loads a prepared model from `~/.cache/botyard-retargeting/mano/right.npz` or `left.npz` according to `hand`. Both are prepared on the development laptop. On a new machine, run the following once with licensed MANO data and a Python environment that can read the original chumpy pickle. Use `mano_model:=/path/to/model.npz` to override the selected path.

```zsh
mkdir -p -m 700 ~/.cache/botyard-retargeting/mano
for side in right left; do
  ~/miniconda3/envs/grabnet/bin/python example/vector_retargeting/data/gs_s1_hand_baselines/fit_example_mano.py prepare \
    "$HOME/Downloads/mano_v1_2/models/MANO_${(U)side}.pkl" "$HOME/.cache/botyard-retargeting/mano/$side.npz"
done
```

For the four landmark objectives, to calibrate the finger chains, hold the fingers straight and together,
focus the camera window and press **c**. Hold the pose until the window reads
**Finger calibration ON** (20 consecutive detections). Press **b** to compare with the
original mapping, or **c** to repeat. Calibration applies to the selected hand for this
session. The capture registers each finger root and neutral phalanx direction in the palm
frame. The tracker reconstructs the four finger chains using the GS S1 segment lengths
before computing palm-relative targets. The camera shows the observed pinky PIP and DIP
bend, the optimizer's LFJ2 target in degrees and the estimated thumb-to-finger gaps in mm
(index, middle, ring, little). DexPilot also displays the finger pairs in pinch projection.
The `vector` option uses the upstream VectorOptimizer with GS S1 palm-to-tip, phalanx
and thumb-to-tip vectors. It tracks the observed thumb-to-tip displacements continuously.

The `tsinghua` option ports the hand terms of `VectorWristJointOptimizerV2` to the shared
GS S1 solver. It reads the authors' five-finger Panda/Shadow weights and thresholds from a
sibling `tsinghua-retargeting` checkout (tested at `3846d3f`). Set `tsinghua_root:=...` to
select another checkout. With the wrist fixed, the world-to-thumb vector is measured from
the palm. Joint penalties use matching actuator names and the GS S1 neutral pose.
The calibrated palm-to-tip and distal vectors track finger position and direction. The
filtered, uncalibrated thumb-to-tip distances determine the pinch weights and targets.
In the supplied profile a gap of 10 mm or less gives a zero target distance. The loss uses
fingertip frame positions and distal vectors, so a small mesh gap can occur along the side
of a finger. [Authors' implementation](https://github.com/Mingrui-Yu/retargeting)

All four objectives use the same 16 actuator coordinates, the URDF velocity and angle bounds,
and MuJoCo mesh distance inequalities. The four passive distal joints follow their respective J2 in the model.
The upright MuJoCo hand publishes its 20 simulated finger angles on `/gs_s1_preview/sim_joint_states`.
The retargeting targets, including the two fixed wrist angles, are on `/gs_s1_preview/joint_states`.
MuJoCo reads the joint and servo position ranges from the same side-specific ROS URDF.
At each integrated substep it enforces the stops on all 20 finger hinges, removes outward
velocity at a stop, and checks the resulting pose for intersections between enabled collision geometries.
An additional triangle mesh check covers the palm cutouts where overlapping convex hulls would give
false contacts in the open hand. Adjacent joint housings retain their assembly exclusions.
The passive joint coupling is a kinematic approximation.

The example reads GS S1 CAD from the installed `dexhand_description` package and loads the side-specific
V3.2 MJCF and five-finger servo settings from a sibling `tendon-spin/assets` checkout.
The Wuji option uses the upstream adaptive analytical objective from a sibling `wuji-retargeting`
checkout (tested at `531f6ed`) with GS S1 link names and unit segment scales.
Override these locations with `mujoco_assets:=...` and `wuji_root:=...`.
The tracker environment also needs MuJoCo (tested with 3.3.2).

The following command compares DexPilot and Wuji on the same detected keypoints.
Use `--objectives dexpilot wuji vector tsinghua` to compare all four on identical inputs.
Use `--smoothing off` to reproduce the input settings of the saved objective comparison.

```zsh
PYTHONPATH="$PWD/src:$PYTHONPATH" ~/miniconda3/envs/dex_retargeting/bin/python \
  example/vector_retargeting/gs_s1_compare.py --hand right
```

### Score a GS S1 grasp on the DexMV mustard bottle

From the repository root with the ROS and Botyard setup sourced, run

```zsh
PYTHONPATH="$PWD/src:$PYTHONPATH" ~/miniconda3/envs/dex_retargeting/bin/python \
  example/vector_retargeting/gs_s1_dexmv.py --with-control --output /tmp/gs_s1_dexmv.json
```

This replays frames 11–122 at 25 Hz from the trusted sibling `dexmv-sim` example.
The two source cameras are registered using the example's extrinsics. All four
objectives receive the same wrist-local 21 points and One Euro filter. The
mount follows the recorded wrist and the four passive J1 joints follow J2.
The bottle has a free joint, DexMV collision meshes and density. The 1.2 m
tabletop pose is estimated from the demonstration. Use `--tracks /tmp/dexmv.npz`
to save frame IDs, hinge and bottle poses, the human reference, clearance and contacts.

The report gives palm-relative and thumb-relative tip errors, distal-direction
errors, per-finger contact and tracking, self-collision, and hand–object
penetration at the frame and physics rates. Object translation and rotation
errors use the complete trajectory, including failed attempts. The separate
mean-pose tracking threshold is 100 mm and 0.5 rad.
For this trial a stable grasp requires positive hand contact, no tabletop support,
and 50 mm of clearance under the lowest bottle vertex continuously for at least
0.5 s. Relocation also requires returning the bottle to the table within 50 mm
of the demonstrated final position without an early drop. The negative control
disables hand–bottle contacts with all other physics and inputs held fixed.
The bundled sample contains one demonstration.

### Generate the robot joint pose trajectory from our pre-recorded video

```shell
cd example/vector_retargeting
python3 detect_from_video.py \
  --robot-name allegro \
  --video-path data/human_hand_video.mp4 \
  --retargeting-type dexpilot \
  --hand-type right \
  --output-path data/allegro_joints.pkl 
```

This command will output the joint trajectory as a pickle file at the `output_path`.

The pickle file is a python dictionary with two keys: `meta_data` and `data`. `meta_data`, a dictionary, includes
details about the robot, while `data`, a list, contains the robotic joint positions for each frame. For additional
options, refer to the help information. Note that the time cost here includes both the hand pose detection from video,
and the hand pose retargeting in single process mode.

```shell
python3 detect_from_video.py --help
```

### Utilize the pickle file to produce a video of the robot

```shell
python3 render_robot_hand.py \
  --pickle-path data/allegro_joints.pkl \
  --output-video-path data/allegro.mp4 \
  --headless
```

This command uses the data saved from the previous step to create a rendered video.

### Capture a Video Using Your Webcam

*The following instructions assume that your computer has a webcam connected.*

```bash
python3 capture_webcam.py --video-path data/my_human_hand_video.mp4
```

This command enables you to use your webcam to record a video saved in MP4 format. To stop recording, press `Esc` on your
keyboard.

### Real-time Visualization of Hand Retargeting via Webcam

```bash
pip install loguru
python3 show_realtime_retargeting.py \
  --robot-name allegro \
  --retargeting-type dexpilot \
  --hand-type right 
```

This process integrates the tasks described above. It involves capturing your hand movements through the webcam and
instantaneously displaying the retargeting outcomes in the SAPIEN viewer. Special thanks
to [@xbkaishui](https://github.com/xbkaishui) for contributing the initial pull request.

![realtime_example](data/realtime_example.webp)





