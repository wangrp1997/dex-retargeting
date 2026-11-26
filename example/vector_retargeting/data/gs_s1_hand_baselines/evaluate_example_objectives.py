"""Compare fixed-wrist GS S1 command kinematics on the same fitted right-hand RGB clip."""
import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

from botyard_live import BotyardRetargeter
from distal_closure import DistalClosure, mano_surface_gaps
from gs_s1_filter import LandmarkFilter
from gs_s1_rviz import (MANO_TO_GS_S1, build_configured_solver, build_constrained, build_solver,
                        get_package_share_directory, load_hand)
from palm_anchor_experiment import Hand, angle_deg, finger_shape, surface_targets
from single_hand_detector import OPERATOR2MANO_RIGHT, SingleHandDetector

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
METHODS = ('dexpilot', 'anyteleop_vector', 'tsinghua', 'wuji', 'botyard')
DIGITS = ('thumb', 'index', 'middle', 'ring', 'little')


def local_joints(joints):
    frame = SingleHandDetector.estimate_frame_from_hand_points(joints) @ OPERATOR2MANO_RIGHT @ MANO_TO_GS_S1
    return (joints - joints[0]) @ frame


def windows_for(frames, fps, example):
    windows = np.zeros((len(frames), 4), dtype=bool)
    with (example / 'v3/opposition_comparison.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    expected_digits = ('食指', '中指', '无名指', '小指')
    for i, (row, expected) in enumerate(zip(rows, expected_digits, strict=True)):
        assert row['指对'] == expected
        windows[:, i] = (frames / fps >= float(row['观察窗起始秒'])) & (frames / fps <= float(row['观察窗末秒']))
    assert np.all(windows.any(axis=0))
    return windows


def vector_solver():
    profile = ROOT / 'src/dex_retargeting/configs/teleop/shadow_hand_right.yml'
    config = yaml.safe_load(profile.read_text())['retargeting']
    # The adapted GS S1 uses the ten upstream palm-to-tip/middle vectors. Its palm
    # is close in width to this fit. Do not apply Shadow's 1.2x length multiplier.
    keys = ('type', 'target_origin_link_names', 'target_task_link_names')
    return build_configured_solver(dict(**{k: config[k] for k in keys}, scaling_factor=1.,
                                        target_link_human_indices=np.array(config['target_link_human_indices'])),
                                   'right', show_collisions=True)


def replay(method, data, args, meshes, *, gradient="finite_difference", trace=None):
    if method == 'anyteleop_vector':
        seq, cad = vector_solver()
    else:
        seq, cad = build_solver(show_collisions=True, hand='right', objective=method)
    model = load_hand(args.assets, meshes, 'right')
    filt = LandmarkFilter('one_euro')
    if method == 'botyard':
        retargeter = BotyardRetargeter(seq, model, meshes, 'right', None, 'one_euro', gradient=gradient)
        np.testing.assert_array_equal(retargeter.mano.faces, data['faces'])
        solver = retargeter.output
    else:
        solver = build_constrained(seq, model, method, args)
    commands, elapsed, held_frames = [], [], []
    dt = 1. / data['fps']
    for i, number in enumerate(data['frame']):
        previous_held = solver.held
        if method == 'botyard':
            previous_reference_held = retargeter.reference.held
        joints = data['joints'][i]
        timestamp = number * dt
        start = perf_counter()
        if method == 'botyard':
            retargeter.solve_mano(joints, data['vertices'][i], timestamp, dt)
        else:
            solver.solve(filt.update(local_joints(joints), timestamp), dt)
        elapsed.append((perf_counter() - start) * 1000)
        held = solver.held > previous_held
        if method == 'botyard':
            held |= retargeter.reference.held > previous_reference_held
        held_frames.append(held)
        commands.append(solver.last.copy())
    commands = np.stack(commands)
    assert np.isfinite(commands).all()
    assert np.all(commands >= solver.limits[:, 0] - 1e-7) and np.all(commands <= solver.limits[:, 1] + 1e-7)
    assert np.all(np.abs(np.diff(commands, axis=0)) <= solver.speed * dt + 1e-7)
    collisions = sum(cad.first_intersection(solver.full_pose(q)) is not None for q in commands)
    hand = Hand(solver, meshes)
    chain, _, normals = hand.states(commands)
    closure = DistalClosure(solver)
    gaps = 1000 * np.stack([closure.distances(q)[0] for q in commands])
    max_step = np.max(np.abs(np.rad2deg(np.diff(commands, axis=0))), axis=1)
    diagnostics = dict(held_frames=sum(held_frames), cad_collision_frames=collisions,
                       max_joint_step_p95_deg=np.percentile(max_step, 95),
                       retarget_p50_ms=np.median(elapsed), retarget_p95_ms=np.percentile(elapsed, 95))
    if trace is not None:
        trace.update(commands=commands, joint_names=solver.opt.target_joint_names, elapsed=np.array(elapsed))
    return chain, normals, gaps, diagnostics


def scores(method, robot, normals, gaps, diagnostics, data, windows):
    human = data['local'][:, 1:].reshape(-1, 5, 4, 3)
    point_error = 1000 * np.linalg.norm(robot - human, axis=-1)
    tip = point_error[:, :, -1]
    delta = robot[:, :, -1] - human[:, :, -1]
    lag = max(1, round(.1 * data['fps']))
    # Change in tip error is the error in the tip's displacement over the same interval.
    displacement_error = 1000 * np.linalg.norm(delta[lag:] - delta[:-lag], axis=-1)
    relative = 1000 * np.linalg.norm(delta[:, 1:] - delta[:, :1], axis=-1)
    direction = angle_deg(robot[:, :, -1] - robot[:, :, -2], human[:, :, -1] - human[:, :, -2])
    bend = np.rad2deg(abs(finger_shape(robot)[1] - finger_shape(human)[1])).mean(axis=-1)
    surface_angle = angle_deg(normals, data['normals'])
    gap_error = abs(gaps - data['gaps_mm'])
    near, far = data['gaps_mm'] <= 2, data['gaps_mm'] >= 5
    row = dict(method=method, frames=len(robot), wrist_tip_mm=tip.mean(), thumb_relative_all_mm=relative.mean(),
               thumb_relative_opposition_mm=relative[windows].mean(), distal_direction_deg=direction.mean(),
               tip_displacement_100ms_mm=displacement_error.mean(),
               mpjpe21_wrist_mm=point_error.sum() / (len(robot) * 21), bend_deg=bend.mean(),
               thumb_pad_normal_deg=surface_angle[:, 0].mean(), distal_gap_mae_mm=gap_error.mean(),
               opposition_gap_mae_mm=gap_error[windows].mean(), near_pairs=near.sum(),
               matched_near_pairs=(gaps[near] <= 2).sum(), separated_pairs=far.sum(),
               false_near_pairs=(gaps[far] <= 2).sum(), **diagnostics)
    digits = []
    for i, name in enumerate(DIGITS):
        entry = dict(method=method, digit=name, wrist_tip_mm=tip[:, i].mean(), distal_direction_deg=direction[:, i].mean(),
                     tip_displacement_100ms_mm=displacement_error[:, i].mean(),
                     bend_deg=bend[:, i].mean(), pad_normal_deg=surface_angle[:, i].mean(),
                     opposition_frames=None, opposition_thumb_relative_mm=None, opposition_gap_mae_mm=None)
        if i:
            mask = windows[:, i-1]
            entry.update(opposition_frames=mask.sum(), opposition_thumb_relative_mm=relative[mask, i-1].mean(),
                         opposition_gap_mae_mm=gap_error[mask, i-1].mean())
        digits.append(entry)
    return row, digits


def write_rows(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def compare_botyard_gradients(data, args, meshes, windows):
    """Replay from neutral in forward and reverse order, pool per-frame latencies."""
    traces, timings, results = {}, {}, {}
    for mode in ('finite_difference', 'analytic', 'analytic', 'finite_difference'):
        trace = {}
        chain, normals, gaps, diagnostics = replay('botyard', data, args, meshes, gradient=mode, trace=trace)
        timings.setdefault(mode, []).append(trace['elapsed'])
        if mode in traces:
            np.testing.assert_array_equal(trace['commands'], traces[mode]['commands'])
            continue
        traces[mode] = trace
        row, digits = scores(mode, chain, normals, gaps, diagnostics, data, windows)
        row['little_bend_deg'] = digits[-1]['bend_deg']
        row['little_opposition_gap_mae_mm'] = digits[-1]['opposition_gap_mae_mm']
        results[mode] = row

    baseline = traces['finite_difference']['commands']
    for mode, row in results.items():
        elapsed = np.concatenate(timings[mode])
        delta = np.rad2deg(abs(traces[mode]['commands'] - baseline))
        i, j = np.unravel_index(delta.argmax(), delta.shape)
        row.update(timing_passes=len(timings[mode]), retarget_mean_ms=elapsed.mean(),
                   retarget_p50_ms=np.median(elapsed), retarget_p95_ms=np.percentile(elapsed, 95),
                   joint_delta_p95_deg=np.percentile(delta.max(axis=1), 95),
                   joint_delta_max_deg=delta[i, j], frames_over_5deg=int((delta.max(axis=1) > 5).sum()),
                   worst_frame=int(data['frame'][i]) if delta[i, j] else None,
                   worst_joint=traces[mode]['joint_names'][j] if delta[i, j] else None)
    write_rows(args.out / 'botyard_gradients.csv', list(results.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example', type=Path, default=HERE / 'example_rgb_mano')
    parser.add_argument('--out', type=Path, default=HERE / 'example_paper_metrics')
    parser.add_argument('--assets', type=Path, default=ROOT.parent / 'tendon-spin/assets')
    parser.add_argument('--wuji-root', type=Path, default=ROOT.parent / 'wuji-retargeting')
    parser.add_argument('--tsinghua-root', type=Path, default=ROOT.parent / 'tsinghua-retargeting')
    parser.add_argument('--compare-botyard-gradients', action='store_true',
                        help='Compare the two Botyard shape gradients on the same clip, twice each')
    args = parser.parse_args()
    args.hand = 'right'
    with np.load(args.example / 'mano.npz') as m:
        frames, fps = m['frame'], float(m['fps'])
        np.testing.assert_array_equal(frames, np.arange(int(m['total_frames'])))
        if fps != 30.:
            raise ValueError("This example benchmark uses consecutive frames at 30 Hz.")
        data = dict(frame=frames, fps=fps, joints=m['joints_camera_wrist_m'].astype(float),
                    vertices=m['vertices_camera_wrist_m'].astype(float), faces=m['faces'])
        data['gaps_mm'] = 1000 * mano_surface_gaps(m, args.example / 'mano.distal.npz')
    data['local'] = np.stack([local_joints(p) for p in data['joints']])
    _, data['normals'], _ = surface_targets(data, dict(frame=frames, raw_human_palm_m=data['local']))
    windows = windows_for(frames, fps, args.example)
    meshes = Path(get_package_share_directory('dexhand_description')) / 'meshes/gs_s1'
    args.out.mkdir(exist_ok=True, parents=True)
    if args.compare_botyard_gradients:
        compare_botyard_gradients(data, args, meshes, windows)
        return
    rows, digits = [], []
    for method in METHODS:
        row, by_digit = scores(method, *replay(method, data, args, meshes), data, windows)
        print(row, flush=True)
        rows.append(row)
        digits.extend(by_digit)
    write_rows(args.out / 'summary.csv', rows)
    write_rows(args.out / 'per_digit.csv', digits)


if __name__ == '__main__':
    main()
