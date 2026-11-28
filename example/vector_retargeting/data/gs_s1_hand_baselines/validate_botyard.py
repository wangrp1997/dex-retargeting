"""Replay cached MANO inputs through the public Botyard v4.2 controller.

Required NPZ arrays: joints, vertices, time_s, dt. Optional: frame and valid.
Inputs and generated trajectories stay local; only small summaries need tracking.
"""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
from ament_index_python.packages import get_package_share_directory

from botyard_live import BotyardRetargeter
from distal_closure import DistalClosure
from gs_s1_mujoco import load_hand
from gs_s1_rviz import build_solver
from palm_anchor_experiment import bone_geometry, unit


WINDOWS = [('thumb_bend', .35, 1.3), ('index_bend', 1.7, 2.7),
           ('middle_bend', 3.4, 4.25), ('ring_bend', 5.15, 6.),
           ('little_bend', 6.8, 7.43), ('fist', 8.5, 9.3),
           ('index_pinch', 10.3, 11.1), ('middle_pinch', 11.85, 12.54),
           ('ring_pinch', 13.67, 14.21), ('little_pinch', 15.12, 15.65),
           ('palm_rotation', 16.7, 17.57), ('second_fist', 20.2, 20.67)]


def direction_error(a, b):
    return np.rad2deg(np.arccos(np.clip(np.sum(unit(a) * unit(b), axis=-1), -1., 1.)))


def main(*, retargeter_factory=BotyardRetargeter, extra_metrics=None, metadata=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--assets', type=Path, default=Path(__file__).resolve().parents[5] / 'tendon-spin/assets')
    parser.add_argument('--hand', choices=('right', 'left'), default='right')
    parser.add_argument('--mano-model', type=Path)
    parser.add_argument('--baseline-controls', type=Path,
                        help='Optional paired controls NPZ with baseline/frame/time_s arrays')
    parser.add_argument('--baseline-key', choices=('baseline', 'validation'), default='baseline',
                        help='Select validation when comparing against a saved v4.2 output')
    parser.add_argument('--label', default='replay')
    args = parser.parse_args()
    source = np.load(args.source, allow_pickle=False)
    times, dt = source['time_s'], source['dt']
    n = len(times)
    frames = source['frame'] if 'frame' in source else np.arange(n)
    valid = source['valid'] if 'valid' in source else np.ones(n, bool)
    if (not np.isfinite(times).all() or (np.diff(times) < 0).any()
            or not np.isfinite(dt).all() or (dt <= 0).any()):
        parser.error('Expected finite ordered timestamps and positive frame dt')
    meshes = Path(get_package_share_directory('dexhand_description')) / 'meshes/gs_s1'
    model = (mujoco.MjModel.from_xml_path(str(args.scene)) if args.scene
             else load_hand(args.assets, meshes, args.hand))
    ret = retargeter_factory(build_solver(hand=args.hand, objective='botyard')[0], model, meshes,
                             args.hand, args.mano_model, 'one_euro', gradient='analytic', shape_model='v4_2')
    solver = ret.output
    closure = ret.protected_closure
    distance = DistalClosure(solver)
    baseline = None
    if args.baseline_controls:
        cached = np.load(args.baseline_controls, allow_pickle=False)
        np.testing.assert_array_equal(cached['frame'], frames)
        np.testing.assert_array_equal(cached['time_s'], times)
        np.testing.assert_array_equal(cached['joint_names'], solver.opt.target_joint_names)
        baseline = cached[args.baseline_key]
    previous = solver.last.copy()
    commands, references, weights, human_gaps, rows = [], [], [], [], []
    start = perf_counter()
    for k, t in enumerate(times):
        trace = {}
        row = dict(frame=int(frames[k]), time_s=float(t), tracking=bool(valid[k]),
                   baseline_exception=False, reference_exception=False, correction_exception=False)
        if valid[k]:
            ret.solve_mano(source['joints'][k], source['vertices'][k], float(t), float(dt[k]), trace=trace)
            chain = ret.filters[0].position[1:].reshape(5, 4, 3)
            source_bones = bone_geometry(chain)[0][1:]
            command = solver.last.copy()
            qref = trace['closure_primary_position_rad']
            row['reference_exception'] = trace['primary_optimizer_exception'] is not None
            row['correction_exception'] = trace['closure_optimizer_exception'] is not None
            margin = trace['closure_min_protection_margin']
            row['protected'] = margin is None or margin >= -1e-5
            row['min_protection_margin'] = float('nan') if margin is None else margin
            robot = bone_geometry(ret.hand.states(command)[0][0])[0][1:]
            row['validation_direction_error_deg'] = float(direction_error(robot, source_bones).mean())
            if baseline is not None:
                robot = bone_geometry(ret.hand.states(baseline[k])[0][0])[0][1:]
                row['baseline_direction_error_deg'] = float(direction_error(robot, source_bones).mean())
            else:
                row['baseline_direction_error_deg'] = float('nan')
            w = trace['closure_weights']
            gap = 1000 * trace['mano_distal_gaps_m']
        else:
            command, qref = previous.copy(), previous.copy()
            w, gap = np.zeros(4), np.full(4, np.nan)
            row.update(protected=True, min_protection_margin=float('nan'),
                       validation_direction_error_deg=float('nan'), baseline_direction_error_deg=float('nan'))
        row['validation_geometry_ok'] = bool(solver.geom.feasible(command)
                                             and solver.palm.clear(solver.geom.data, .0005 - 1e-7))
        row['validation_velocity_violation_rad'] = float(max(0., np.max(abs(command - previous) - solver.speed * dt[k])))
        row['joint_limits_ok'] = bool((command >= solver.limits[:, 0] - 1e-7).all()
                                      and (command <= solver.limits[:, 1] + 1e-7).all())
        diff = np.rad2deg(abs(command - (baseline[k] if baseline is not None else qref)))
        row['max_joint_difference_deg'] = float(diff.max())
        row['largest_difference_joint'] = solver.opt.target_joint_names[int(diff.argmax())]
        row['max_j4_difference_deg'] = float(diff[closure.spread].max())
        row['max_contact_weight'] = float(w.max())
        for label, q in [('baseline', baseline[k] if baseline is not None else qref), ('validation', command)]:
            for d, value in enumerate(1000 * distance.distances(q)[0]):
                row[f'{label}_gap_{d}_mm'] = float(value)
        if extra_metrics is not None:
            row.update(extra_metrics(ret, dict(baseline=baseline[k] if baseline is not None else qref,
                                               validation=command)))
        commands.append(command)
        references.append(qref)
        weights.append(w)
        human_gaps.append(gap)
        rows.append(row)
        previous = command
        if k % 150 == 0:
            print(args.label, k, '/', n, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'frame_metrics.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.out / 'controls.npz', frame=frames, time_s=times,
                        joint_names=solver.opt.target_joint_names, validation=commands, reference=references,
                        baseline=baseline if baseline is not None else references,
                        weights=weights, human_gap_mm=human_gaps)
    tracked = [row for row in rows if row['tracking']]
    report = dict(algorithm='botyard', algorithm_version=ret.algorithm_version, dataset=args.label,
                  frames=n, tracked_frames=int(valid.sum()), elapsed_s=perf_counter() - start,
                  baseline_mean_direction_error_deg=float(np.mean([r['baseline_direction_error_deg'] for r in tracked])),
                  validation_mean_direction_error_deg=float(np.mean([r['validation_direction_error_deg'] for r in tracked])),
                  validation_geometry_failures=[r['frame'] for r in rows if not r['validation_geometry_ok']],
                  protection_failures=[r['frame'] for r in rows if not r['protected']],
                  joint_limit_failures=[r['frame'] for r in rows if not r['joint_limits_ok']],
                  max_velocity_violation_rad=max(r['validation_velocity_violation_rad'] for r in rows),
                  reference_exceptions=[r['frame'] for r in rows if r['reference_exception']],
                  correction_exceptions=[r['frame'] for r in rows if r['correction_exception']],
                  experiment=metadata,
                  worst_departures=sorted(tracked, key=lambda r: r['max_joint_difference_deg'], reverse=True)[:8],
                  note='All source frames replayed through BotyardRetargeter, shape_model=v4_2. Direction errors score four long fingers only. Thumb facing is an intentional custom soft preference; MANO proximity is not tactile contact truth.')
    if args.label == 'example':
        report['gesture_windows'] = []
        for label, lo, hi in WINDOWS:
            selected = [r for r in tracked if lo <= r['time_s'] <= hi]
            report['gesture_windows'].append(dict(gesture=label, start_s=lo, end_s=hi, n=len(selected),
                baseline_direction_error_deg=float(np.mean([r['baseline_direction_error_deg'] for r in selected])),
                validation_direction_error_deg=float(np.mean([r['validation_direction_error_deg'] for r in selected]))))
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('worst_departures', 'gesture_windows')}), flush=True)
    if (report['validation_geometry_failures'] or report['protection_failures']
            or report['joint_limit_failures'] or report['max_velocity_violation_rad'] > 1e-7):
        raise SystemExit('Physical or source-priority validation failed')


if __name__ == '__main__':
    main()
