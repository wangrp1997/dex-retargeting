"""Replay just the constrained v3 refinement on cached, identical first-stage poses."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from distal_closure import DistalClosure
from gs_s1_mujoco import load_hand
from gs_s1_optimization import ConstrainedRetargeter
from gs_s1_rviz import build_solver, get_package_share_directory
from palm_anchor_experiment import Hand, LENGTH, angle_deg, finger_shape


def configure_closure(solver, base_cost, mcp_flex):
    closure = DistalClosure(solver)
    # Lower the cost of adjusting the thumb base and MCP orientation. Keep
    # the thumb IP and other distal flexion penalties exactly as in v3.
    axes = [k for k,name in enumerate(solver.opt.target_joint_names) if name.endswith('J4') or name == 'THJ3'
            or (mcp_flex and name.endswith('J3'))]
    closure.joint_penalties[axes] *= base_cost
    return closure


def refine_commands(solver, closure, reference, weights_by_frame, targets, frames, fps, carry_offset,
                    motion_weight=0., smooth_gap=None):
    controls, gaps, held_frames = [], [], []
    prev_ref = solver.last.copy()
    lag = max(1,round(.10 * fps))
    for i,(ref,weights,target) in enumerate(zip(reference,weights_by_frame,targets)):
        start = ref + (solver.last - prev_ref) if carry_offset else ref
        motion = None
        if motion_weight and i >= lag and frames[i]-frames[i-lag] == lag:
            # Match the human gap change since the earlier executed pose.
            desired = gaps[i-lag]*.001 + (smooth_gap[i]-smooth_gap[i-lag])
            importance = motion_weight * .5 * (weights+weights_by_frame[i-lag])
            motion = (desired, importance)
        previous_held = solver.held
        solver.solve_objective(closure.objective(ref,weights,target,motion=motion),1/fps,initial=start)
        held_frames.append(solver.held > previous_held)
        prev_ref = ref
        cmd = solver.last.copy()
        assert solver.geom.feasible(cmd) and solver.palm.clear(solver.geom.data,solver.geom.clearance-1e-7)
        physical = solver.geom.data.qpos
        for digit in ('FF','MF','RF','LF'):
            a,b = [solver.geom.model.joint(digit+suffix).qposadr.item() for suffix in ('J1','J2')]
            assert physical[a] == physical[b]
        controls.append(cmd)
        gaps.append(1000*closure.distances(cmd)[0])
    q, gaps = np.stack(controls),np.stack(gaps)
    assert np.isfinite(q).all() and np.isfinite(gaps).all()
    assert (q>=solver.limits[:,0]-1e-7).all() and (q<=solver.limits[:,1]+1e-7).all()
    steps = np.diff(q,axis=0,prepend=solver.robot.q0[solver.drive][None,:])
    assert (np.abs(steps)<=solver.speed/fps+1e-7).all()
    return q, gaps, held_frames


def run(example, out, weight, base_cost, gap_target, carry_offset, mcp_flex, motion_weight=0.):
    baseline = np.load(example/'v3/gs_s1.npz')
    reference = baseline['v2_active_qpos_rad']
    human = baseline['human_distal_gaps_mm']
    frames, fps = baseline['frame'], float(baseline['fps'])
    seq,_ = build_solver()
    names = seq.optimizer.target_joint_names
    np.testing.assert_array_equal(names, baseline['joint_names'])
    meshes = Path(get_package_share_directory('dexhand_description'))/'meshes/gs_s1'
    solver = ConstrainedRetargeter(seq,load_hand(Path.home()/'Projects/tendon-spin/assets',meshes,'right'))
    hand = Hand(solver,meshes)
    closure = configure_closure(solver,base_cost,mcp_flex)
    saved_strength = json.loads((example/'v3/metrics.json').read_text())['closure_weight']
    weights_by_frame = baseline['closure_weights'] * (weight / saved_strength)
    smooth_gap = None
    targets = [None] * len(frames)
    if gap_target in ('scaled','compressed') or motion_weight:
        # Reuse the size-scaled, filtered MANO gaps from the exact same clip.
        scaled = np.load(example/'v3_1/gs_s1.npz')
        np.testing.assert_array_equal(frames,scaled['frame'])
        np.testing.assert_array_equal(reference,scaled['v2_active_qpos_rad'])
        scaled_targets = scaled['target_distal_gaps_mm']*.001
        smooth_gap = np.sqrt(np.maximum(0.,scaled_targets**2-closure.clearance**2))
        if gap_target == 'compressed':
            targets = np.hypot(closure.clearance, smooth_gap**3 / (smooth_gap**2 + LENGTH**2))
        elif gap_target == 'scaled':
            targets = scaled_targets
    q, gaps, held_frames = refine_commands(solver,closure,reference,weights_by_frame,targets,frames,fps,
                                          carry_offset,motion_weight,smooth_gap)
    lag = max(1,round(.10*fps))
    if weight == .35 and base_cost == 1. and gap_target == 'v3' and not carry_offset and not motion_weight:
        np.testing.assert_allclose(gaps,baseline['robot_distal_gaps_mm'],rtol=0,atol=.005)
    pos,_,normals = hand.states(q)
    ref_pos,_,ref_normals = hand.states(reference)
    bend_delta = np.degrees(abs(finger_shape(pos)[1]-finger_shape(ref_pos)[1]))
    normal_delta = angle_deg(normals,ref_normals)
    thumb_ip = np.degrees(abs(q[:,names.index('THJ1')]-reference[:,names.index('THJ1')]))
    report = dict(weight=weight,base_cost=base_cost,gap_target=gap_target,carry_offset=carry_offset,mcp_flex=mcp_flex,
                  motion_weight=motion_weight,motion_lag_frames=lag,checked_frames=len(frames),held=solver.held,
                  separated_pairs=int((human>=5).sum()),false_near_pairs=int(((human>=5)&(gaps<=2)).sum()),
                  max_thumb_ip_delta_deg=float(thumb_ip.max()),max_bend_delta_deg=float(bend_delta.max()),
                  max_thumb_normal_delta_deg=float(normal_delta[:,0].max()),
                  p95_thumb_normal_delta_deg=float(np.percentile(normal_delta[:,0],95)))
    with (example/'v3/opposition_comparison.csv').open(encoding='utf-8-sig') as f:
        for j,(digit,row) in enumerate(zip(('index','middle','ring','little'),csv.DictReader(f))):
            win=(frames/fps>=float(row['观察窗起始秒'])) & (frames/fps<=float(row['观察窗末秒']))
            report[digit+'_min_mm'] = float(gaps[win,j].min())
            report[digit+'_median_mm'] = float(np.median(gaps[win,j]))
            report[digit+'_within_2mm_frames'] = int((gaps[win,j]<=2).sum())
    report['little_frame462_mm'] = float(gaps[list(frames).index(462),-1])
    out.mkdir(parents=True,exist_ok=True)
    tag = f'w{weight:.2f}_base{base_cost:g}'
    if gap_target != 'v3':
        tag += '_'+gap_target
    if carry_offset:
        tag += '_carry'
    if mcp_flex:
        tag += '_mcp'
    if motion_weight:
        tag += f'_motion{motion_weight:g}'
    np.savez_compressed(out/(tag+'.npz'),frame=frames,active_qpos_rad=q,robot_distal_gaps_mm=gaps,
                        bend_delta_deg=bend_delta,normal_delta_deg=normal_delta,solver_held=held_frames)
    (out/(tag+'.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('example',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--weight',type=float,required=True)
    parser.add_argument('--base-cost',type=float,default=1.)
    parser.add_argument('--gap-target',choices=('v3','compressed','scaled'),default='v3')
    parser.add_argument('--carry-offset',action='store_true',help='Seed each iterate with the previous correction relative to v2.')
    parser.add_argument('--mcp-flex',action='store_true',help='Apply the base cost to the four non-thumb MCP flexion axes as well.')
    parser.add_argument('--motion-weight',type=float,default=0.,help='Relative weight of tracking the signed 100 ms gap change.')
    args = parser.parse_args()
    if not np.isfinite([args.weight,args.base_cost,args.motion_weight]).all() or min(args.weight,args.motion_weight)<0 or args.base_cost<=0:
        parser.error('Weights must be nonnegative and base cost positive; all must be finite.')
    run(**vars(args))
