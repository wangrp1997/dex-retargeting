"""Replay selected Botyard shape objectives on the same recorded, timestamped MANO fits."""
import argparse
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from ament_index_python.packages import get_package_share_directory

from botyard_live import BotyardRetargeter
from distal_closure import DistalClosure
from gs_s1_hand_baselines import angle_deg
from gs_s1_rviz import build_solver
from palm_anchor_experiment import bone_geometry, surface_targets, thumb_span_in_palm_widths


def score_frame(retargeter, distances, row, joints, vertices, *, refit_mano=False):
    t = row['time_s']
    trace = {}
    if refit_mano:
        retargeter.solve_camera(np.asarray(row['skeleton_wrist_m']), np.asarray(row['skeleton_px']),
                                row['image_size_px'], t, row['solve_dt_s'], trace=trace)
        joints, vertices = trace['mano_joints_wrist_m'], trace['mano_vertices_wrist_m']
    else:
        retargeter.solve_mano(joints, vertices, t, row['solve_dt_s'], trace=trace)
    names = retargeter.output.opt.target_joint_names
    q = retargeter.output.last
    frame = retargeter.target_frame(joints)
    local = (joints - joints[0]) @ frame
    human_bones, _, _, _, human_bends = bone_geometry(local[1:].reshape(5, 4, 3))
    robot, pads, normals = retargeter.hand.states(q)
    robot_bones, _, _, _, robot_bends = bone_geometry(robot[0])
    surface = dict(frame=[row['frame']], vertices=vertices[None], joints=joints[None], faces=retargeter.mano.faces)
    palm_frame = dict(frame=surface['frame'], raw_human_palm_m=local[None])
    human_pads, human_normals, _ = surface_targets(surface, palm_frame)
    relative = (pads[0, 1:] - pads[0, :1]) - (human_pads[0, 1:] - human_pads[0, :1])
    recorded_q = dict(zip(row['command_joint_names'], row['command_position_rad']))
    max_command_delta = max(abs(value - recorded_q[name]) for name, value in zip(names, q))
    gaps = 1000 * distances.distances(q)[0]
    target_gaps = trace['target_distal_gaps_m']
    desired = None if target_gaps is None else 1000 * target_gaps
    result = dict(frame=row['frame'], time_s=t, shape_model=retargeter.shape_model,
                  max_delta_from_recorded_deg=np.rad2deg(max_command_delta),
                  long_finger_bone_error_deg=angle_deg(robot_bones[1:], human_bones[1:]).mean(),
                  long_finger_bend_error_deg=np.rad2deg(abs(robot_bends[1:] - human_bends[1:])).mean(),
                  thumb_pad_normal_error_deg=angle_deg(normals[0, 0], human_normals[0, 0]),
                  thumb_relative_pad_error_mm=1000 * np.linalg.norm(relative, axis=-1).mean(),
                  gap_mae_mm=None if desired is None else abs(gaps - desired).mean(), reference_holds_total=retargeter.reference.held,
                  output_holds_total=retargeter.output.held)
    result['thumb_distal_axis_error_deg'] = angle_deg(robot_bones[0, -1], human_bones[0, -1])
    # A matched surface normal in the palm frame can still miss the opposing pad.
    # The bearing is undefined when pad centers coincide; leave that cell empty.
    for source, centers, thumb_normal in (('mano', human_pads[0], human_normals[0, 0]),
                                           ('robot', pads[0], normals[0, 0])):
        for digit, center in zip(('index', 'middle', 'ring', 'little'), centers[1:]):
            ray = center - centers[0]
            result[f'{source}_thumb_facing_{digit}_deg'] = (
                angle_deg(thumb_normal, ray) if np.linalg.norm(ray) > 1e-10 else None)
    result['mano_thumb_span_widths'] = thumb_span_in_palm_widths(*joints[[4, 5, 17]])
    result['robot_thumb_span_widths'] = thumb_span_in_palm_widths(robot[0, 0, -1], robot[0, 1, 0], robot[0, 4, 0])
    result['skeleton_thumb_span_widths'] = None
    if row.get('skeleton_wrist_m') is not None:
        skeleton = np.asarray(row['skeleton_wrist_m'])
        result['skeleton_thumb_span_widths'] = thumb_span_in_palm_widths(*skeleton[[4, 5, 17]])
    for digit, gap, target in zip(('index', 'middle', 'ring', 'little'), gaps,
                                 [None] * 4 if desired is None else desired):
        result[f'{digit}_gap_mm'] = gap
        result[f'{digit}_target_mm'] = target
    result.update({f'{name}_deg': np.rad2deg(value) for name, value in zip(names, q)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoothing', choices=('one_euro', 'off'), default='one_euro')
    parser.add_argument('--gradient', choices=('finite_difference', 'analytic'), default='finite_difference')
    parser.add_argument('--mano-model', type=Path)
    parser.add_argument('--refit-mano', action='store_true',
                        help='Refit saved raw camera observations with the current MANO objective')
    parser.add_argument('--shapes', nargs='+', choices=('v4_2', 'sweep', 'direction', 'baseline', 'phalanx', 'thumb_span'),
                        default=('baseline', 'phalanx'))
    args = parser.parse_args()
    scene = args.session / 'scene.xml'
    side = ET.parse(scene).getroot().get('model', '').rsplit('_', 1)[-1]
    if side not in ('left', 'right'):
        parser.error(f'Cannot determine the GS S1 hand from {scene}')
    meshes = Path(get_package_share_directory('dexhand_description')) / 'meshes/gs_s1'
    variants = []
    for shape in args.shapes:
        solver, _ = build_solver(hand=side, objective='botyard')
        retargeter = BotyardRetargeter(solver, mujoco.MjModel.from_xml_path(str(scene)), meshes, side,
                                       args.mano_model, args.smoothing, gradient=args.gradient, shape_model=shape)
        variants.append((retargeter, DistalClosure(retargeter.output)))
    results = []
    with (args.session / 'retargeting.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['tracking']:
                joints = np.asarray(row['mano_joints_wrist_m'])
                vertices = np.asarray(row['mano_vertices_wrist_m'])
                for retargeter, distances in variants:
                    results.append(score_frame(retargeter, distances, row, joints, vertices,
                                               refit_mano=args.refit_mano))
    if not results:
        parser.error('No tracked MANO frames in this session')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)
    print(f'{len(results)//len(variants)} MANO frames, {len(variants)} objectives saved to {args.out}')


if __name__ == '__main__':
    main()
