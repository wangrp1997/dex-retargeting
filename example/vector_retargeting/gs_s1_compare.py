"""Replay identical detected keypoints through the selected GS S1 objectives."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import mujoco
import numpy as np

from gs_s1_filter import LandmarkFilter

from gs_s1_rviz import (MANO_TO_GS_S1, OBJECTIVES, SingleHandDetector, build_constrained,
                        build_solver, get_package_share_directory, load_hand)
from gs_s1_mujoco import ContactStepper


def evaluate(method, poses, fps, args):
    solver, cad = build_solver(show_collisions=True, hand=args.hand, objective=method)
    mesh = Path(get_package_share_directory("dexhand_description")) / "meshes/gs_s1"
    retarget = build_constrained(solver, load_hand(args.assets, mesh, args.hand), method, args)
    landmark_filter = LandmarkFilter(args.smoothing)
    model = load_hand(args.assets, mesh, args.hand)
    physics = ContactStepper(model)
    data = physics.data
    drives = [solver.joint_names.index(model.joint(j).name) for j in model.actuator_trnid[:, 0]]
    tips = [model.site(f + "tip_frame").id for f in ("th", "ff", "mf", "rf", "lf")]
    palm = model.site("palm_frame").id
    little_distal = model.body("lfdistal").id
    pad_geoms = [model.body(name).geomadr.item() for name in ("thdistal", "ffdistal")]
    tip_errors, angles, jumps, solve_ms, pinch_gaps = [], [], [], [], []
    band_gaps, band_projected = [], 0
    max_depth = 0.0
    previous = data.qpos.copy()
    cad_counts = {}
    cad_pose = solver.optimizer.robot.q0.copy()
    cad_indices = [solver.joint_names.index(model.joint(j).name) for j in range(model.njnt)]
    for frame, points in enumerate(poses):
        if points is not None:
            target = landmark_filter.update(points, frame / fps)
            start = perf_counter()
            q = retarget.solve(target, 1 / fps)
            solve_ms.append((perf_counter() - start) * 1000)
            data.ctrl[:] = q[drives]
        steps = round((frame + 1) / (fps * model.opt.timestep)) - round(frame / (fps * model.opt.timestep))
        physics.advance(steps)
        mujoco.mj_forward(model, data)
        max_depth = max(max_depth, -np.min(data.contact.dist, initial=0.0))
        cad_pose[cad_indices] = data.qpos[model.jnt_qposadr]
        pair = cad.first_intersection(cad_pose)
        if pair:
            label = " / ".join(pair)
            cad_counts[label] = cad_counts.get(label, 0) + 1
        jumps.append(np.max(np.abs(data.qpos - previous)))
        previous = data.qpos.copy()
        if points is None:
            continue
        rotation = data.site_xmat[palm].reshape(3, 3)
        robot_tips = (data.site_xpos[tips] - data.site_xpos[palm]) @ rotation
        tip_errors.append(np.linalg.norm(robot_tips - (points[4::4] - points[0]), axis=1) * 1000)
        expected = points[20] - points[19]
        actual = (data.site_xpos[tips[-1]] - data.xpos[little_distal]) @ rotation
        cosine = expected @ actual / max(np.linalg.norm(expected) * np.linalg.norm(actual), 1e-12)
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1., 1.))))
        near = np.linalg.norm(points[4] - points[8]) < .03
        in_band = .03 <= np.linalg.norm(target[4] - target[8]) < .05
        if near or in_band:
            gap_mm = mujoco.mj_geomDistance(model, data, *pad_geoms, .3, None) * 1000
            if near:
                pinch_gaps.append(gap_mm)
            if in_band:
                band_gaps.append(gap_mm)
                if method == "dexpilot":
                    band_projected += int(solver.optimizer.projected[0])
    return {
        "smoothing": args.smoothing,
        "tracked_frames": len(angles), "total_frames": len(poses),
        "median_five_tip_error_mm": float(np.median(tip_errors)),
        "median_pinky_direction_error_deg": float(np.median(angles)),
        "p95_largest_joint_step_deg": float(np.degrees(np.percentile(jumps, 95))),
        "thumb_index_near_frames": len(pinch_gaps),
        "median_thumb_index_mesh_gap_mm": float(np.median(pinch_gaps)) if pinch_gaps else None,
        "thumb_index_30_50mm_input_frames": len(band_gaps),
        "thumb_index_30_50mm_mesh_within_2mm_frames": sum(g <= 2 for g in band_gaps),
        "thumb_index_30_50mm_dexpilot_projection_frames": band_projected if method == "dexpilot" else None,
        "max_sampled_penetration_mm": max_depth * 1000,
        "nonadjacent_cad_intersection_frames": sum(cad_counts.values()),
        "first_cad_intersections": cad_counts,
        "rejected_physics_substeps": physics.rejected,
        "held_solver_frames": retarget.held,
        "median_solve_ms": float(np.median(solve_ms)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    example = Path(__file__).resolve().parent
    projects = example.parents[2]
    parser.add_argument("--video", type=Path, default=example / "data/human_hand_video.mp4")
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--smoothing", choices=("one_euro", "off"), default="one_euro")
    parser.add_argument("--objectives", nargs="+", choices=OBJECTIVES,
                        default=("dexpilot", "wuji"))
    parser.add_argument("--mirror", action="store_true", help="Reflect the source before detecting the opposite hand")
    parser.add_argument("--assets", type=Path, default=projects / "tendon-spin/assets")
    parser.add_argument("--wuji-root", type=Path, default=projects / "wuji-retargeting")
    parser.add_argument("--tsinghua-root", type=Path, default=projects / "tsinghua-retargeting")
    args = parser.parse_args()
    capture = cv2.VideoCapture(str(args.video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
        parser.error(f"Could not read the video frame rate: {args.video}")
    detector = SingleHandDetector(args.hand.title(), selfie=False)
    poses = []
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if args.mirror:
                image = cv2.flip(image, 1)
            _, points, _, _ = detector.detect(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            poses.append(points @ MANO_TO_GS_S1 if points is not None and np.isfinite(points).all() else None)
    finally:
        capture.release()
        detector.hand_detector.close()
    if all(p is None for p in poses):
        parser.error(f"No {args.hand} hand detected")
    for method in args.objectives:
        print(json.dumps({method: evaluate(method, poses, fps, args)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
