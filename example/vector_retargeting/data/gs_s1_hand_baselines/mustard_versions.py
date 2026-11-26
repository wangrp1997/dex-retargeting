"""Replay the frozen v3.2/v3.3 objectives on the DexMV free mustard bottle."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from pinocchio import hppfcl

from distal_closure import closure_weights, mano_surface_gaps
from gs_s1_dexmv import DIGITS, FPS, LIFT_M, load_sequence, replay_commands
from gs_s1_mujoco import ContactStepper, load_hand
from gs_s1_optimization import ConstrainedRetargeter
from gs_s1_rviz import build_solver, get_package_share_directory
from palm_anchor_experiment import Hand, LENGTH, angle_deg, filtered, finger_shape, objective, surface_targets, unit
from sweep_closure import configure_closure, refine_commands


def distal_bottle_gaps(surface, vertices, hand, closure, commands, bottle_pose, bottle_meshes):
    """Five unsigned distances to the annotated bottle. Overlaps count as zero."""
    request = hppfcl.DistanceRequest()
    identity = hppfcl.Transform3f()
    regions = [surface['faces'][surface['distal_face_labels']==j] for j in range(5)]
    human,robot = [],[]
    for i,(angles,bottle) in enumerate(zip(commands,bottle_pose)):
        closure.distances(angles)
        h,r = [],[]
        def measure(mesh,pose):
            return 1000*max(0.,min(hppfcl.distance(mesh,pose,b,bottle,request,hppfcl.DistanceResult())
                                    for b in bottle_meshes))
        for faces,body,mesh in zip(regions,closure.ids,closure.meshes):
            patch = hppfcl.BVHModelOBBRSS()
            patch.beginModel(len(faces),3*len(faces))
            for a,b,c in vertices[i,faces]:
                patch.addTriangle(a,b,c)
            patch.endModel()
            h.append(measure(patch,identity))
            pos = hand.axes.T @ (closure.data.xpos[body]-hand.wrist)
            rot = hand.axes.T @ closure.data.xmat[body].reshape(3,3)
            r.append(measure(mesh,hppfcl.Transform3f(rot,pos)))
        human.append(h)
        robot.append(r)
    return np.asarray(human),np.asarray(robot)


def run(args):
    args.hand = 'right'
    seq = load_sequence(args.dexmv_root)
    ids,world,local,palm,bottle,orient = seq
    surface = np.load(args.surface)
    np.testing.assert_array_equal(ids,surface['frame'])
    np.testing.assert_array_equal(ids,np.arange(11,123))
    ref_track = dict(frame=ids,raw_human_palm_m=local)
    pads,normals,_ = surface_targets(surface,ref_track)
    raw = surface['joints']-surface['joints'][:,:1]
    u,_,vt = np.linalg.svd(raw.transpose(0,2,1)@local)
    vertices = (surface['vertices']-surface['joints'][:,:1])@(u@vt)
    np.testing.assert_allclose(raw@(u@vt),local,atol=1e-6)
    human_gaps = mano_surface_gaps(dict(vertices_camera_wrist_m=vertices,faces=surface['faces'],
                                       distal_face_labels=surface['distal_face_labels']),None)
    smooth = filtered(human_gaps[:,:,None])[:,:,0]
    meshes = Path(get_package_share_directory('dexhand_description'))/'meshes/gs_s1'
    solver,cad = build_solver(True,'right')
    base = ConstrainedRetargeter(solver,load_hand(args.assets,meshes,'right'))
    hand = Hand(base,meshes)
    neutral = hand.states(base.last)[0][0]
    ratio = np.linalg.norm(neutral[1,0]-neutral[4,0])/np.median(np.linalg.norm(local[:,5]-local[:,17],axis=1))
    refs = []
    for chain,pad,normal in zip(filtered(local)[:,1:].reshape(-1,5,4,3),filtered(pads),unit(filtered(normals))):
        base.solve_objective(objective(hand,chain,pad,normal,base.last.copy(),0.,opposition=True),1/FPS)
        refs.append(base.last.copy())
    refs=np.asarray(refs)
    weight = closure_weights(smooth,.70)
    scaled = smooth*ratio
    bottle_pose = [hppfcl.Transform3f(R.T@O,(p-w[0])@R) for R,O,p,w in zip(palm,orient,bottle,world)]
    bottle_files = sorted((args.dexmv_root/'hand_imitation/env/models/assets/ycb/collision/006_mustard_bottle').glob('*.stl'))
    assert len(bottle_files)==2
    loader = hppfcl.MeshLoader()
    bottle_meshes = [loader.load(str(file)) for file in bottle_files]
    human_bends = finger_shape(local[:,1:].reshape(-1,5,4,3))[1]
    args.out.mkdir(parents=True,exist_ok=True)
    for version in ('v3.2','v3.3'):
        correction = ConstrainedRetargeter(solver,load_hand(args.assets,meshes,'right'))
        closure = configure_closure(correction,.006,True)
        target = scaled**3/(scaled**2+LENGTH**2) if version=='v3.2' else scaled
        target = np.hypot(closure.clearance,target)
        q,gaps,held = refine_commands(correction,closure,refs,weight,target,ids,FPS,True)
        h_obj,r_obj = distal_bottle_gaps(surface,vertices,hand,closure,q,bottle_pose,bottle_meshes)
        fresh_hand = load_hand(args.assets,meshes,'right')
        full = np.stack([correction.full_pose(x) for x in q])
        physics,tracks = replay_commands(args,seq,meshes,fresh_hand,ContactStepper(fresh_hand),
                                         full,solver,cad,int(sum(held)),None)
        xyz,_,robot_normals = hand.states(q)
        bends = finger_shape(xyz)[1]
        lift = tracks['reference_clearance_m'] >= LIFT_M
        assert lift.any()
        i88 = ids.index(88)
        summary = dict(fps=FPS,n=len(ids),version=version,palm_scale=ratio,shared_reference_holds=base.held,
                       hand_gap_mae_mm=np.abs(gaps-1000*human_gaps).mean(),
                       bottle_gap_mae_mm=np.abs(r_obj-h_obj).mean(),
                       liftphase_bottle_gap_mae_mm=np.abs(r_obj[lift]-h_obj[lift]).mean(),
                       bend_mae_deg=np.degrees(np.abs(bends-human_bends)).mean(),
                       thumb_normal_deg=angle_deg(robot_normals[:,0],normals[:,0]).mean(),
                       source88_thumb_distal_deg=float(np.degrees(bends[i88,0,1])),
                       source88_mano_thumb_distal_deg=float(np.degrees(human_bends[i88,0,1])),
                       reference_lift_frames=int(lift.sum()), initial_bottle_settle_s=1. if getattr(args,'settle_bottle',False) else 0., physics=physics)
        (args.out/(version+'.json')).write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        np.savez_compressed(args.out/(version+'.npz'),active_qpos_rad=q,joint_names=solver.optimizer.target_joint_names,
                            solver_held=held, reference_active_qpos_rad=refs,human_gaps_mm=1000*human_gaps,
                            robot_gaps_mm=gaps,human_bottle_gaps_mm=h_obj,robot_bottle_gaps_mm=r_obj,
                            **tracks)
        print(version,summary['hand_gap_mae_mm'],physics['stable_grasp'],physics['peak_bottom_clearance_mm'],flush=True)
        if args.with_control and version=='v3.3':
            model=load_hand(args.assets,meshes,'right')
            result,control=replay_commands(args,seq,meshes,model,ContactStepper(model),full,solver,cad,
                                           int(sum(held)),None,control=True)
            xy=np.asarray(control['object_xyz'])[:,:2]
            axes=np.asarray(control['object_rotation'])[:,:,2]
            check=dict(hand_object_contact_frames=result['hand_object_contact_frames'],lift_success=result['lift_success'],
                       max_horizontal_drift_mm=1000*np.linalg.norm(xy-xy[0],axis=1).max(),
                       max_tilt_deg=np.degrees(np.arccos(np.clip(axes@axes[0],-1,1))).max())
            assert not check['hand_object_contact_frames'] and not check['lift_success']
            (args.out/'no_contact_control.json').write_text(json.dumps(check,indent=2)+'\n')
    write_table(args.out)


def write_table(out):
    tracks = [dict(np.load(out/(v+'.npz'))) for v in ('v3.2','v3.3')]
    scores = [json.loads((out/(v+'.json')).read_text()) for v in ('v3.2','v3.3')]
    for key in ('frame','human_palm','reference_active_qpos_rad','reference_xyz','reference_rotation',
                'wrist_xyz','wrist_rotation','human_gaps_mm','human_bottle_gaps_mm'):
        np.testing.assert_allclose(tracks[0][key],tracks[1][key],rtol=0,atol=1e-10)
    rows = []
    for key,label,unit in [('hand_gap_mae_mm','拇指到四指末节间隙平均绝对误差','mm'),
                           ('bottle_gap_mae_mm','五指末节到标注瓶面的间隙平均绝对误差','mm'),
                           ('liftphase_bottle_gap_mae_mm','演示抬升段五指到标注瓶面的间隙误差','mm'),
                           ('bend_mae_deg','五指弯曲平均绝对误差','°'),
                           ('thumb_normal_deg','拇指指腹朝向平均误差','°'),
                           ('source88_thumb_distal_deg','源帧88拇指末节弯曲角','°')]:
        rows.append(['指令姿态与MANO',label,unit,*[round(s[key],3) for s in scores]])
    for key,label,unit in [('hand_object_contact_frames','有手瓶受力接触的帧数','帧'),
                           ('peak_bottom_clearance_mm','瓶底最大离桌高度','mm'),
                           ('longest_supported_lift_s','连续持稳时间（离桌至少50 mm）','s'),
                           ('final_position_error_mm','末帧瓶中心位置误差','mm'),
                           ('physics_hand_object_peak_mm','手瓶最大瞬时穿透深度','mm'),
                           ('self_cad_intersection_frames','独立CAD检查报告手部相交','帧'),
                           ('passive_mimic_max_residual_deg','四指J1与J2最大联动误差','°'),
                           ('stable_grasp','稳定抓取成功','是/否'),('relocate_success','搬运成功','是/否')]:
        vals=[s['physics'][key] for s in scores]
        rows.append(['自由瓶仿真',label,unit,*[('是' if x else '否') if isinstance(x,bool) else round(x,3) for x in vals]])
    lift=tracks[0]['reference_clearance_m']>=LIFT_M
    rows.append(['自由瓶仿真','演示抬升段有手瓶受力接触的帧数',f'帧 / {lift.sum()}',
                 *[int(x['hand_contact'][lift].sum()) for x in tracks]])
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['测量条件','指标','单位','v3.2','v3.3'])
        writer.writerows(rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('surface',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--dexmv-root',type=Path,default=Path.home()/'Projects/dexmv-sim')
    parser.add_argument('--assets',type=Path,default=Path.home()/'Projects/tendon-spin/assets')
    parser.add_argument('--with-control',action='store_true',help='Repeat v3.3 with hand–bottle collision disabled.')
    parser.add_argument('--settle-bottle',action='store_true',help='Let the free bottle rest on the table for 1 s before replay.')
    run(parser.parse_args())
