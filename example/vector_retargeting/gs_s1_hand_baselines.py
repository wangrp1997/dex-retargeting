"""Compare published hand objectives on fixed-wrist GS S1 kinematics and DexMV landmarks."""

import argparse
import csv
import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml

from gs_s1_dexmv import DIGITS, FPS, load_sequence
from gs_s1_filter import LandmarkFilter
from gs_s1_mujoco import hand_urdf, load_hand
from gs_s1_rviz import FINGER_JOINTS, build_configured_solver, build_constrained, get_package_share_directory

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
METHODS = ("position", "vector", "dexpilot", "wuji", "tsinghua")
PROFILE = {"position": "offline/shadow_hand_right.yml", "vector": "teleop/shadow_hand_right.yml",
           "dexpilot": "teleop/shadow_hand_right_dexpilot.yml"}
THUMB_JOINTS = ("THJ4", "THJ3", "THJ2", "THJ1")


def angle_deg(a, b):
    lengths = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    if np.any(lengths < 1e-12):
        raise ValueError("Cannot score the direction of a zero-length bone")
    return np.degrees(np.arccos(np.clip((a * b).sum(axis=-1) / lengths, -1, 1)))


def library_config(method):
    name = method if method in PROFILE else "dexpilot"
    path = REPO / "src/dex_retargeting/configs" / PROFILE[name]
    cfg = yaml.safe_load(path.read_text())["retargeting"]
    # Keep the published five-finger link pairs and landmark indices. Override
    # the body, wrist, input/output filtering and scale for the common GS S1 task.
    for override in ("urdf_path", "target_joint_names", "low_pass_alpha", "add_dummy_free_joint"):
        cfg.pop(override, None)
    if "scaling_factor" in cfg:
        cfg["scaling_factor"] = 1.0
    if "target_link_human_indices" in cfg:
        cfg["target_link_human_indices"] = np.asarray(cfg["target_link_human_indices"])
    if name == "dexpilot":
        cfg["wrist_link_name"] = "palm"
    return cfg


def compare(method, args, mesh_dir, human, filtered, ids):
    seq, _ = build_configured_solver(library_config(method), "right")
    solver = build_constrained(seq, load_hand(args.assets, mesh_dir, "right"), method, args)
    robot = seq.optimizer.robot
    get_id = robot.get_link_index
    palm_id = get_id("palm")
    robot.compute_forward_kinematics(robot.q0)
    palm_pose = robot.get_link_pose(palm_id).copy()
    rot, wrist = palm_pose[:3, :3], palm_pose[:3, 3]
    links = [d + "tip" for d in DIGITS] + [d + "distal" for d in DIGITS] + ["thproximal", "thmiddle"]
    frames = [get_id(name) for name in links]
    if method == "wuji":
        native_robot = solver.wuji.robot
        native_frames = [native_robot.get_link_index(name) for name in links]
    controls = [seq.joint_names.index(n) for n in FINGER_JOINTS]
    thumb_idx = [seq.joint_names.index(n) for n in THUMB_JOINTS]
    fixed = robot.q0[seq.optimizer.idx_pin2fixed].copy()
    xyz, joints = [], []
    for pose in filtered:
        if method in PROFILE:
            # Library objectives evaluate frames in the root coordinate system.
            indices = seq.optimizer.target_link_human_indices
            if method == "position":
                reference = pose[indices] @ rot.T + wrist
            else:
                origin, task = indices
                reference = (pose[task] - pose[origin]) @ rot.T
            objective = seq.optimizer.get_objective_function(reference.astype(np.float32), fixed, solver.last)
            q = solver.solve_objective(objective, 1 / FPS)
        else:
            q = solver.solve(pose, 1 / FPS)
        robot.compute_forward_kinematics(q)
        xyz.append([(robot.get_link_pose(f)[:3, 3] - wrist) @ rot for f in frames])
        if method == "wuji":
            native_robot.compute_forward_kinematics(q)
            native_xyz = [(native_robot.get_link_pose(f)[:3, 3] - wrist) @ rot for f in native_frames]
            np.testing.assert_allclose(native_xyz, xyz[-1], atol=1e-5, rtol=0)
        joints.append(q)
    xyz, joints = np.asarray(xyz), np.asarray(joints)
    if not (np.isfinite(xyz).all() and np.isfinite(joints).all()):
        raise ValueError(f"Non-finite output from {method}")
    if np.any(joints < robot.joint_limits[:, 0] - 1e-7) or np.any(joints > robot.joint_limits[:, 1] + 1e-7):
        raise ValueError(f"Joint limit violation in {method}")
    for digit in ("FF", "MF", "RF", "LF"):
        child, parent = [seq.joint_names.index(digit + j) for j in ("J1", "J2")]
        np.testing.assert_allclose(joints[:, child], joints[:, parent], atol=1e-9, rtol=0)
    tips, distal = xyz[:, :5], xyz[:, 5:10]
    tip_error = 1000 * np.linalg.norm(tips - human[:, 4::4], axis=-1)
    axes_error = angle_deg(tips - distal, human[:, 4::4] - human[:, 3::4])
    human_gap = 1000 * np.linalg.norm(human[:, 4] - human[:, 8], axis=1)
    robot_gap = 1000 * np.linalg.norm(tips[:, 0] - tips[:, 1], axis=1)
    # Compare physical bone chords, not a motor angle to a human anatomical angle.
    robot_chain = xyz[:, [10, 11, 5, 0]]  # proximal -> middle -> distal -> tip
    human_chain = human[:, 1:5]  # CMC -> MCP -> IP -> tip
    rb, hb = np.diff(robot_chain, axis=1), np.diff(human_chain, axis=1)
    robot_bend = angle_deg(rb[:, :-1], rb[:, 1:])
    human_bend = angle_deg(hb[:, :-1], hb[:, 1:])
    angles = np.degrees(joints[:, thumb_idx])
    def stats(mask):
        return dict(five_tip_mean_mm=float(tip_error[mask].mean()),
                    thumb_index_gap_mae_mm=float(np.abs(robot_gap[mask] - human_gap[mask]).mean()),
                    five_distal_direction_mean_deg=float(axes_error[mask].mean()),
                    thumb_tip_mean_mm=float(tip_error[mask, 0].mean()),
                    thumb_distal_direction_mean_deg=float(axes_error[mask, 0].mean()),
                    human_thumb_ip_bend_mean_deg=float(human_bend[mask, 1].mean()),
                    robot_thumb_distal_bend_mean_deg=float(robot_bend[mask, 1].mean()),
                    thumb_distal_bend_mae_deg=float(abs(robot_bend[mask, 1] - human_bend[mask, 1]).mean()))
    result = stats(np.ones(len(ids), dtype=bool))
    result["frames_53_to_70"] = stats((ids >= 53) & (ids <= 70))
    result["p95_max_active_joint_step_deg"] = float(np.percentile(np.degrees(abs(np.diff(joints[:, controls],axis=0))).max(axis=1), 95))
    result["solver_held_frames"] = solver.held
    result["thumb_j1_within_1deg_upper_frames"] = int(np.sum(
        angles[:, -1] >= np.degrees(robot.joint_limits[thumb_idx[-1], 1]) - 1))
    rows = []
    for i, number in enumerate(ids):
        row = dict(frame=int(number), method=method, thumb_tip_mm=tip_error[i, 0],
                   thumb_distal_direction_deg=axes_error[i, 0], human_gap_mm=human_gap[i], robot_gap_mm=robot_gap[i],
                   human_thumb_mcp_bend_deg=human_bend[i, 0], human_thumb_ip_bend_deg=human_bend[i, 1],
                   robot_thumb_proximal_bend_deg=robot_bend[i, 0], robot_thumb_distal_bend_deg=robot_bend[i, 1])
        row.update(zip(THUMB_JOINTS, angles[i]))
        rows.append(row)
    return result, rows, joints[:, controls]


def plot_thumb(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    names = ["Position", "Vector", "DexPilot", "Wuji Adaptive", "Xin et al."]
    fig, axes = plt.subplots(5, 1, figsize=(9.2, 7.8), sharex=True, sharey=True)
    fig.subplots_adjust(top=.82, left=.10, right=.96, bottom=.09, hspace=.36)
    fig.suptitle("Thumb bend during the DexMV mustard-bottle motion", x=.10, y=.975,
                 ha="left", fontsize=17, fontweight="bold")
    fig.text(.10, .91, "Angle between the last two bone segments in 3D (degrees). Straight = 0°.")
    for ax, method, label in zip(axes, METHODS, names):
        data = [r for r in rows if r["method"] == method]
        x = [r["frame"] for r in data]
        ax.axvline(88, ls=":", lw=1, color="#6b7280")
        robot_line, = ax.plot(x, [r["robot_thumb_distal_bend_deg"] for r in data],
                              color="#b54e09", lw=2, label="GS S1 target")
        human_line, = ax.plot(x, [r["human_thumb_ip_bend_deg"] for r in data],
                              ls="--", color="#064871", lw=1.7, label="Human hand")
        ax.set_title(label, loc="left", fontsize=10, pad=1, fontweight="bold")
        ax.set(ylim=(-3, 113), xlim=(11, 122), yticks=[0, 50, 100])
        ax.grid(axis="y", color="#e0e3e6", lw=.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=[human_line, robot_line], loc="center left", bbox_to_anchor=(.09,.87),
               frameon=False, ncol=2)
    axes[-1].set_xlabel("Source frame  ·  25 Hz  ·  dotted line marks frame 88")
    axes[-1].set_xticks([11, 30, 50, 70, 88, 105, 122])
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def main():
    projects = REPO.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dexmv-root", type=Path, default=projects / "dexmv-sim")
    parser.add_argument("--wuji-root", type=Path, default=projects / "wuji-retargeting")
    parser.add_argument("--tsinghua-root", type=Path, default=projects / "tsinghua-retargeting")
    parser.add_argument("--assets", type=Path, default=projects / "tendon-spin/assets")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.hand = "right"
    ids, _, human, _, _, _ = load_sequence(args.dexmv_root)
    ids = np.array(ids)
    if not np.array_equal(ids, np.arange(11, 123)) or human.shape != (112, 21, 3):
        raise ValueError("Expected the DexMV mustard sequence with source frames 11 through 122")
    filt = LandmarkFilter("one_euro")
    filtered = [filt.update(p, i / FPS).copy() for i, p in enumerate(human)]
    mesh_dir = Path(get_package_share_directory("dexhand_description")) / "meshes/gs_s1"
    sources = dict(dex_retargeting=REPO, dexmv=args.dexmv_root, wuji=args.wuji_root,
                   tsinghua=args.tsinghua_root, tendon_spin=args.assets)
    scripts = [Path(__file__), *(HERE / f"gs_s1_{name}.py" for name in
               ("dexmv", "filter", "mujoco", "optimization", "palm", "rviz", "tsinghua")),
               HERE / "single_hand_detector.py"]
    report = dict(sequence="DexMV relocate_mustard_example_seq", frames=len(ids), fps=FPS,
                  first_frame=int(ids[0]), last_frame=int(ids[-1]), experimental_unit="one clip",
                  control="right GS S1 v3.2, fixed wrist; kinematic targets before object or servo physics",
                  input="wrist-local metres; shared causal One Euro input; raw landmarks scored; neutral start",
                  constraints="16 actuators, 4 non-thumb J1 mimic J2; same SLSQP, speed/range and self-collision constraints",
                  profile_sources=PROFILE, output_joint_unit="degrees in CSV",
                  git={k:subprocess.check_output(["git","-C",str(v),"rev-parse","HEAD"],text=True).strip() for k,v in sources.items()},
                  versions={k:version(k) for k in ("numpy","scipy","nlopt","torch","mujoco","pin")},
                  hashes_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},
                  gs_s1_urdf_sha256=hashlib.sha256(hand_urdf("right")).hexdigest(),
                  raw_palm_track_sha256=hashlib.sha256(np.asarray(human,dtype="<f8").tobytes()).hexdigest(), methods={})
    rows = []
    tracks = dict(frame=ids, joint_names=np.array(FINGER_JOINTS), raw_human_palm_m=human,
                  filtered_human_palm_m=np.asarray(filtered))
    for method in METHODS:
        report["methods"][method], detail, tracks[method + "_active_qpos_rad"] = compare(
            method, args, mesh_dir, human, filtered, ids)
        rows.extend(detail)
        print(f"{method} finished {len(detail)} frames", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "scores.json").write_text(json.dumps(report, indent=2,allow_nan=False) + "\n")
    with (args.out_dir / "thumb.csv").open("w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.out_dir / "trajectories.npz", **tracks)
    plot_thumb(rows, args.out_dir / "thumb_bend.png")


if __name__ == "__main__":
    main()
