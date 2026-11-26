"""Replay fitted RGB-example MANO poses through the fixed-wrist GS S1 objective."""
import argparse
import json
from pathlib import Path

import numpy as np

from gs_s1_filter import LandmarkFilter
from gs_s1_rviz import MANO_TO_GS_S1, get_package_share_directory, build_solver
from single_hand_detector import SingleHandDetector, OPERATOR2MANO_RIGHT
from gs_s1_optimization import ConstrainedRetargeter
from gs_s1_mujoco import load_hand
from palm_anchor_experiment import Hand, angle_deg, finger_shape, objective, surface_targets, unit
from distal_closure import DistalClosure, closure_weights, mano_surface_gaps


def run(mano, out, assets, opposition=False, v3=False, closure_weight=None, mano_model=None, v31=False):
    assisted_mode = v3 or v31
    if closure_weight is not None and (not assisted_mode or not np.isfinite(closure_weight) or closure_weight < 0):
        raise ValueError('--closure-weight requires --v3 or --v3.1 and a finite, nonnegative value')
    strength = .35 if closure_weight is None else closure_weight
    m=np.load(mano)
    frames=m['frame']
    fps=float(m['fps'])
    joints=m['joints_camera_wrist_m']
    wrist_to_palm=np.stack([SingleHandDetector.estimate_frame_from_hand_points(p) @ OPERATOR2MANO_RIGHT @ MANO_TO_GS_S1 for p in joints])
    local=(joints-joints[:,:1]) @ wrist_to_palm
    np.testing.assert_allclose(np.linalg.det(wrist_to_palm),1,atol=1e-5)
    surface=dict(frame=frames,joints=joints,vertices=m['vertices_camera_wrist_m'],faces=m['faces'])
    pads,normals,_=surface_targets(surface,dict(frame=frames,raw_human_palm_m=local))
    mesh_dir=Path(get_package_share_directory('dexhand_description'))/'meshes/gs_s1'
    seq,_=build_solver()
    solver=ConstrainedRetargeter(seq,load_hand(assets,mesh_dir,'right'))
    hand=Hand(solver,mesh_dir)
    chain_filter,pad_filter,normal_filter=(LandmarkFilter('one_euro') for _ in range(3))
    controls=[]
    held=[]
    reference_controls = []
    weights = []
    extra_tracks = {}
    if assisted_mode:
        topology = mano.with_suffix('.distal.npz')
        if mano_model is None and topology.is_file():
            mano_model = topology
        human_surfaces = mano_surface_gaps(m, mano_model)
        gap_filter = LandmarkFilter('one_euro')
        correction = ConstrainedRetargeter(seq, solver.geom.model)
        closure = DistalClosure(correction)
    if v31:
        # Compare index and little finger MCP centers in each hand.
        human_width = np.median(np.linalg.norm(local[:,5] - local[:,17], axis=1))
        neutral_chain = hand.states(solver.last)[0][0]
        robot_width = np.linalg.norm(neutral_chain[1,0] - neutral_chain[4,0])
        if not np.isfinite([human_width, robot_width]).all() or min(human_width, robot_width) <= 0:
            raise ValueError('Expected two finite, positive palm widths')
        gap_scale = robot_width / human_width
        targets = []
    for i,frame in enumerate(frames):
        timestamp=frame/fps
        chain=chain_filter.update(local[i],timestamp)[1:].reshape(5,4,3)
        pad=pad_filter.update(pads[i],timestamp)
        norm=unit(normal_filter.update(normals[i],timestamp))
        fn=objective(hand,chain,pad,norm,solver.last.copy(),weight=0.,opposition=opposition or assisted_mode)
        previous_held=solver.held
        solver.solve_objective(fn,1/fps)
        reference = solver.last.copy()
        output = reference
        frame_held = solver.held > previous_held
        if assisted_mode:
            # Stage one advances an independent v2 reference. Stage two gets
            # one frame of motion from its own previous executed command.
            reference_controls.append(reference)
            gaps = gap_filter.update(human_surfaces[i, :, None], timestamp).ravel()
            weights.append(closure_weights(gaps, strength))
            target = np.hypot(closure.clearance, gap_scale * gaps) if v31 else None
            if v31:
                targets.append(target)
            if strength > 0:
                assisted = closure.objective(reference, weights[-1], target)
                previous_held = correction.held
                correction.solve_objective(assisted, 1/fps, initial=reference)
                frame_held = correction.held > previous_held
                output = correction.last.copy()
        held.append(frame_held)
        controls.append(output)
    q=np.stack(controls)
    assert np.isfinite(q).all() and np.all(q>=solver.limits[:,0]-1e-7) and np.all(q<=solver.limits[:,1]+1e-7)
    if assisted_mode:
        steps = np.diff(q, axis=0, prepend=solver.robot.q0[solver.drive][None, :])
        assert np.all(np.abs(steps) <= solver.speed/fps + 1e-7)
    for cmd in q:
        assert solver.geom.feasible(cmd) and solver.palm.clear(solver.geom.data,solver.geom.clearance-1e-7)
        for digit in ('FF','MF','RF','LF'):
            j1,j2=[solver.geom.model.joint(digit+k).qposadr.item() for k in ('J1','J2')]
            np.testing.assert_equal(solver.geom.data.qpos[j1], solver.geom.data.qpos[j2])
    xyz,robot_pad,robot_normal=hand.states(q)
    _,rb=finger_shape(xyz)
    _,hb=finger_shape(local[:,1:].reshape(xyz.shape))
    theta=angle_deg(robot_normal[:,0],normals[:,0])
    human_vec,robot_vec=[p[:,1:]-p[:,:1] for p in (pads,robot_pad)]
    human_gap, robot_gap=[1000*np.linalg.norm(v,axis=-1) for v in (human_vec,robot_vec)]
    th=list(seq.optimizer.target_joint_names).index('THJ1')
    metrics=dict(total_frames=int(m['total_frames']),fitted_frames=len(frames),fps=fps,
                 opposition_objective=opposition or assisted_mode,
                 mean_reprojection_px=float(m['reprojection_to_detector_px'].mean()),
                 median_reprojection_px=float(np.median(m['reprojection_to_detector_px'])),
                 p95_reprojection_px=float(np.percentile(m['reprojection_to_detector_px'],95)),
                 mean_fit_to_detector_3d_mm=float(m['landmark_to_detector_mm'].mean()), hand_scale=float(m['hand_scale']),
                 mean_bend_abs_error_deg=float(np.degrees(abs(rb-hb)).mean()),thumb_normal_error_mean_deg=float(theta.mean()),
                 mean_relative_pad_vector_error_mm=float(1000*np.linalg.norm(human_vec-robot_vec,axis=-1).mean()),
                 thumb_index_pad_center_distance_mae_mm=float(abs(human_gap[:,0]-robot_gap[:,0]).mean()),
                 thumb_j1_within_one_degree_of_lower=int(np.sum(q[:,th] <= solver.limits[th,0]+np.deg2rad(1))),
                 collision_checked_frames=len(q),solver_held_frames=int(np.count_nonzero(held)),
                 note='All 2D/3D fit scores refer to MediaPipe estimates. Hand-only robot command kinematics, no contact ground truth.')
    k=int(np.argmin(human_gap[:,0]))
    metrics['closest_fitted_thumb_index'] = dict(frame_zero_based=int(frames[k]), seconds=float(frames[k]/fps),
                  mano_center_distance_mm=float(human_gap[k,0]), gs_center_distance_mm=float(robot_gap[k,0]))
    k=int(np.argmin(human_gap[:,-1]))
    metrics['closest_fitted_thumb_little'] = dict(frame_zero_based=int(frames[k]), seconds=float(frames[k]/fps),
                  mano_center_distance_mm=float(human_gap[k,-1]), gs_center_distance_mm=float(robot_gap[k,-1]))
    if assisted_mode:
        ref_q = np.stack(reference_controls)
        ref_shape = finger_shape(hand.states(ref_q)[0])[1]
        bend_change = np.degrees(np.abs(rb - ref_shape))
        extra_tracks.update(v2_active_qpos_rad=ref_q, closure_weights=np.stack(weights),
                            human_distal_gaps_mm=1000*human_surfaces,
                            v2_distal_gaps_mm=1000*np.stack([closure.distances(x)[0] for x in ref_q]),
                            robot_distal_gaps_mm=1000*np.stack([closure.distances(x)[0] for x in q]),
                            bend_change_from_v2_deg=bend_change)
        metrics.update(version='v3.1' if v31 else 'v3', closure_weight=strength, v2_reference_held_frames=solver.held,
                       bend_change_from_v2_mean_deg=float(bend_change.mean()),
                       bend_change_from_v2_max_deg=float(bend_change.max()),
                       thumb_ip_change_from_v2_max_deg=float(np.rad2deg(abs(q[:,th]-ref_q[:,th])).max()))
    if v31:
        extra_tracks.update(target_distal_gaps_mm=1000*np.stack(targets), gap_scale=gap_scale)
        metrics.update(human_palm_width_mm=1000*human_width, robot_palm_width_mm=1000*robot_width,
                       gap_scale=gap_scale)
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'gs_s1.npz',frame=frames,fps=fps,joint_names=seq.optimizer.target_joint_names,
                  active_qpos_rad=q, solver_held=held, camera_to_palm_row=wrist_to_palm, thumb_normal_deg=theta,
                  human_pad_center_gaps_mm=human_gap, robot_pad_center_gaps_mm=robot_gap, **extra_tracks)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mano',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--assets',type=Path,default=Path.home()/'Projects/tendon-spin/assets')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--opposition',action='store_true',help='Use the v2 opposition objective.')
    modes.add_argument('--v3', action='store_true', help='Refine the v2 reference with continuous, shape-preserving distal closure.')
    modes.add_argument('--v3.1', dest='v31', action='store_true', help='Track palm-scaled MANO distal gaps with the same v3 weights and shape cost.')
    parser.add_argument('--closure-weight', type=float, help='Maximum v3 weight per thumb pair (default 0.35, 0 recovers v2).')
    parser.add_argument('--mano-model', type=Path, help='Prepared MANO topology for older fit files.')
    run(**vars(parser.parse_args()))
