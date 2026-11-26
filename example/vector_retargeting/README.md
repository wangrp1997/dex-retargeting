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

Use `backend:=rviz|mujoco`, `hand:=right|left` and `objective:=botyard|dexpilot|wuji|vector|tsinghua` (one value per argument).
The defaults are `rviz`, `right` and `dexpilot`. Use `video:=/path/to/video.mp4` to replay the same input.
The four landmark objectives apply One Euro smoothing to the wrist-local 3D landmarks before optimization.
Use `smoothing:=one_euro` (default) or `smoothing:=off` to compare them on the same video.
The camera overlay shows the original 2D detections. Press **q** in that window to stop.

The **Botyard** objective fits the selected MANO hand to each frame's 3D detections and 2D image points using the previous pose. It uses the template shape and estimates palm scale online. The offline RGB benchmark fits shape and pose jointly across the full recording. The live optimizer uses the same two robot objectives, joint limits, passive couplings and mesh checks as the offline benchmark. One Euro filtering covers the fitted landmarks, pad centres, pad normals and distal surface gaps. The camera window overlays the translucent cyan MANO surface from the current fit under the MediaPipe skeleton. Press **M** in the camera window to hide or show the surface while retargeting continues. The bottom line shows mesh visibility and mean MANO-plus-solver time. Both backends receive the same joint commands.

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





