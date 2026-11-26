"""Paired fixed-wrist experiment on the DexMV mustard clip. Run as a script."""
import argparse
import csv
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from gs_s1_dexmv import DIGITS, FPS
from gs_s1_filter import LandmarkFilter
from gs_s1_hand_baselines import angle_deg
from gs_s1_mujoco import load_hand
from gs_s1_optimization import ConstrainedRetargeter
from gs_s1_rviz import build_solver, get_package_share_directory
from measure_finger_shape import patch_normal, unit

HERE = Path(__file__).resolve().parent
# Central MANO volar vertices nearest the midpoint of the two distal flank
# contact anchors per digit in the neutral right-hand template (manotorch).
PAD_VERTEX = (763, 343, 455, 566, 683)
WEAK_WEIGHT = .1
LENGTH = .010
ANGLE = np.deg2rad(20.)


def surface_targets(surface, track):
    np.testing.assert_array_equal(surface['frame'], track['frame'])
    xyz = surface['joints'] - surface['joints'][:, :1]
    u, _, vt = np.linalg.svd(xyz.transpose(0, 2, 1) @ track['raw_human_palm_m'])
    rot = u @ vt
    np.testing.assert_allclose(np.linalg.det(rot), 1., atol=1e-6)
    np.testing.assert_allclose(xyz @ rot, track['raw_human_palm_m'], atol=1e-6)
    verts = (surface['vertices'] - surface['joints'][:, :1]) @ rot
    faces = surface['faces']
    patches = [faces[(faces == v).any(axis=1)] for v in PAD_VERTEX]
    normal = np.stack([patch_normal(verts, p) for p in patches], axis=1)
    # A conservative lower bound on the closest points of each pair of modeled
    # surface patches. Every triangle lies inside its vertex bounding sphere.
    center = verts[:, PAD_VERTEX]
    radii = np.stack([np.linalg.norm(verts[:, np.unique(p)] - center[:, i:i+1], axis=-1).max(axis=1)
                      for i, p in enumerate(patches)], axis=1)
    lower = np.linalg.norm(center[:, 1:]-center[:, :1], axis=2)-radii[:, 1:]-radii[:, :1]
    return center, normal, lower


def cad_patches(mesh_dir):
    dtype = [('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')]
    centers, normals = [], []
    for finger in DIGITS:
        sign = -1 if finger == 'th' else 1
        path = mesh_dir / ('rh' if finger == 'th' else '') / (finger+'distal.STL')
        faces = np.fromfile(path, dtype=dtype, offset=84)
        a, b, c = faces['vertices'].astype(float).transpose(1, 0, 2)
        ab, ac = b-a, c-a
        det = ab[:, 0]*ac[:, 2] - ab[:, 2]*ac[:, 0]
        hits, directions = [], []
        for x in np.arange(-2, 3)*.001:
            for z in np.arange(16, 23)*.001:
                with np.errstate(divide='ignore', invalid='ignore'):
                    u = ((x-a[:, 0])*ac[:, 2] - (z-a[:, 2])*ac[:, 0])/det
                    v = (ab[:, 0]*(z-a[:, 2]) - ab[:, 2]*(x-a[:, 0]))/det
                    ids = np.flatnonzero((u>=0) & (v>=0) & (u+v<=1))
                ys = a[ids,1]+u[ids]*ab[ids,1]+v[ids]*ac[ids,1]
                pick = np.argmax(sign*ys)
                n = faces['normal'][ids[pick]]
                if sign*n[1] <= 0:
                    raise ValueError(f'Inward normal at {finger} pad')
                hits.append([x, ys[pick], z])
                directions.append(n)
        centers.append(np.mean(hits, axis=0))
        normals.append(unit(np.mean(directions, axis=0)))
    return np.array(centers), np.array(normals)


def finger_shape(chain):
    bones = unit(np.diff(chain, axis=-2))
    a, b = bones[..., :-1, :], bones[..., 1:, :]
    bend = np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), (a*b).sum(axis=-1))
    return bones[..., 0, :], bend


class Hand:
    def __init__(self, solver, mesh_dir):
        model = solver.geom.model
        self.model = model
        self.data = mj.MjData(model)
        self.mapping = solver.geom.mapping
        self.ids = np.array([[model.body(d+s).id for s in ('proximal','middle','distal','tip')] for d in DIGITS])
        mj.mj_kinematics(model, self.data)
        palm = model.body('palm').id
        self.wrist = self.data.xpos[palm].copy()
        self.axes = self.data.xmat[palm].reshape(3,3).copy()
        self.centers, self.normals = cad_patches(mesh_dir)

    def states(self, controls):
        # Perturb the independent actuators only. The same compiled coupling
        # matrix used for collision checking expands the four passive joints.
        controls = np.atleast_2d(controls)
        pos = np.empty((len(controls),5,4,3))
        rot = np.empty((len(controls),5,3,3))
        for i, q in enumerate(controls @ self.mapping.T):
            self.data.qpos[:] = q
            mj.mj_kinematics(self.model, self.data)
            pos[i] = self.data.xpos[self.ids]
            rot[i] = self.data.xmat[self.ids[:,2]].reshape(5,3,3)
        pos = (pos-self.wrist) @ self.axes
        offset = np.einsum('bfij,fj->bfi', rot, self.centers) @ self.axes
        normal = np.einsum('bfij,fj->bfi', rot, self.normals) @ self.axes
        return pos, pos[:,:,2]+offset, normal


def filtered(series):
    filt = LandmarkFilter('one_euro')
    return np.stack([filt.update(v, i/FPS).copy() for i,v in enumerate(series)])


def objective(hand, chain, pad, normal, previous, weight, *, opposition=False):
    target_root, target_bend = finger_shape(chain)
    target_vectors = pad[1:] - pad[:1]
    # Nearer pads need finer relative accuracy. Use the same continuous rule
    # for all four pairs, bounded at zero gap by the length tolerance.
    if opposition:
        palm_width_sq = np.sum((chain[1,0] - chain[4,0])**2)
        pair_weights = 1 + palm_width_sq / (LENGTH**2 + np.sum(target_vectors**2, axis=-1))
    # Distances and angles are normalized by 10 mm and 20 degrees, then each
    # term is averaged across the corresponding digits. No pinch projection.
    def losses(x):
        pos, pads, normals = hand.states(x)
        root, bend = finger_shape(pos)
        pair_errors = np.sum(((pads[:,1:]-pads[:,:1]-target_vectors)/LENGTH)**2, axis=-1)
        terms = (pair_errors * pair_weights if opposition else pair_errors).mean(axis=1)
        terms += np.mean(np.sum(((root-target_root)/ANGLE)**2, axis=-1),axis=1)
        terms += np.mean(((bend-target_bend)/ANGLE)**2, axis=(1,2))
        terms += np.sum(((normals[:,0]-normal[0])/ANGLE)**2, axis=-1)
        if opposition:
            pad_angles = np.sum(((normals[:,1:]-normal[1:])/ANGLE)**2, axis=-1)
            terms += (np.sqrt(pair_weights) * pad_angles).mean(axis=1)
        centroid_delta = pads.mean(1)-pad.mean(0)
        terms += weight*np.sum((centroid_delta/LENGTH)**2, axis=-1)
        terms += .03*np.mean(((x-previous)/ANGLE)**2,axis=1)
        return terms

    n = len(previous)
    eps = 1e-5
    displacements = np.vstack([np.zeros((1,n)), eps*np.eye(n), -eps*np.eye(n)])
    def callback(x, grad):
        values = losses(x+displacements if grad.size else x[None,:])
        if grad.size:
            grad[:] = (values[1:n+1]-values[n+1:])/(2*eps)
        return float(values[0])
    return callback


def run(args):
    track = np.load(HERE/'trajectories.npz')
    surface = np.load(args.surface)
    mesh_dir = Path(get_package_share_directory('dexhand_description'))/'meshes/gs_s1'
    mesh_targets, normals, gap_bound = surface_targets(surface,track)
    hands = track['raw_human_palm_m'][:,1:].reshape(-1,5,4,3)
    input_chain = track['filtered_human_palm_m'][:,1:].reshape(hands.shape)
    input_pad = filtered(mesh_targets)
    input_normal = unit(filtered(normals))
    human_root, human_bend = finger_shape(hands)
    human_vec = mesh_targets[:,1:]-mesh_targets[:,:1]
    ids = track['frame']
    archive = dict(frame=ids, joint_names=track['joint_names'])
    report = dict(sequence='DexMV mustard, right hand, frames 11-122 at 25 Hz',
                  input='paired methods share filtered MANO pads, normals and landmarks; DexPilot uses filtered landmarks only; score raw MANO',
                  weak_weight=WEAK_WEIGHT, length_unit_m=LENGTH, angle_unit_deg=np.degrees(ANGLE),
                  temporal_weight=.03, method='soft least squares, 4 inter-pad vectors, 5 proximal directions, 10 bone angles, thumb pad normal, previous command; optional single five-pad centroid',
                  min_human_pad_patch_gap_lower_bound_mm=(1000*gap_bound.min(0)).tolist(),
                  methods={})
    rows = []
    for name, weight in [('dexpilot',None),('no_anchor',0.),('weak_anchor',WEAK_WEIGHT)]:
        seq, _ = build_solver()
        solver = ConstrainedRetargeter(seq,load_hand(args.assets,mesh_dir,'right'))
        np.testing.assert_array_equal(seq.optimizer.target_joint_names, track['joint_names'])
        hand = Hand(solver,mesh_dir)
        if weight is None:
            q = track['dexpilot_active_qpos_rad']
        else:
            output = []
            for i in range(len(ids)):
                fn = objective(hand,input_chain[i],input_pad[i],input_normal[i],solver.last.copy(),weight)
                solver.solve_objective(fn,1/FPS)
                output.append(solver.last.copy())
            q = np.stack(output)
        assert np.isfinite(q).all() and np.all(q >= solver.limits[:,0]-1e-7) and np.all(q <= solver.limits[:,1]+1e-7)
        for cmd in q:
            assert solver.geom.feasible(cmd) and solver.palm.clear(solver.geom.data,solver.geom.clearance-1e-7)
            for d in ('FF','MF','RF','LF'):
                joint = solver.geom.model.joint
                assert solver.geom.data.qpos[joint(d+'J1').qposadr] == solver.geom.data.qpos[joint(d+'J2').qposadr]
        archive[name+'_active_qpos_rad'] = q
        pos,pad,norm = hand.states(q)
        root,bend = finger_shape(pos)
        shape_error = np.degrees(abs(bend-human_bend))
        diff = pad[:,1:]-pad[:,:1]
        vec_err = 1000*np.linalg.norm(diff-human_vec,axis=-1)
        gap_err = 1000*abs(np.linalg.norm(diff,axis=-1)-np.linalg.norm(human_vec,axis=-1))
        normal_err = angle_deg(norm[:,0],normals[:,0])
        centroid_err = 1000*np.linalg.norm(pad.mean(1)-mesh_targets.mean(1),axis=-1)
        tip_abs = 1000*np.linalg.norm(pos[:,:,-1]-hands[:,:,-1],axis=-1)
        if name=='dexpilot':
            with (HERE/'dexpilot_finger_shape.csv').open() as f:
                prior=[r for r in csv.DictReader(f) if r['digit']=='th']
            np.testing.assert_allclose(tip_abs[:,0], [float(r['tip_error_mm']) for r in prior],atol=.025)
            np.testing.assert_allclose(normal_err,[float(r['thumb_pad_angle_deg']) for r in prior],atol=.05)
        focus = list(ids).index(88)
        thumb_j1 = list(track['joint_names']).index('THJ1')
        report['methods'][name] = dict(bend_mae_deg=shape_error.mean(), thumb_distal_bend_mae_deg=shape_error[:,0,1].mean(),
             thumb_pad_normal_mean_deg=normal_err.mean(), thumb_pad_normal_median_deg=np.median(normal_err),
             relative_pad_vector_mean_mm=vec_err.mean(), relative_pad_distance_mae_mm=gap_err.mean(),
             thumb_index_pad_distance_mae_mm=gap_err[:,0].mean(), proximal_direction_mean_deg=angle_deg(root,human_root).mean(),
             pad_centroid_mean_mm=centroid_err.mean(), absolute_tip_mean_mm=tip_abs.mean(),
             held_frames=solver.held if weight is not None else json.loads((HERE/'scores.json').read_text())['methods']['dexpilot']['solver_held_frames'],
             checked_feasible_frames=len(q),
             thumb_j1_within_one_degree_of_lower=int(np.sum(q[:,thumb_j1] <= solver.limits[thumb_j1,0]+np.deg2rad(1))),
             thumb_j1_within_one_degree_of_upper=int(np.sum(q[:,thumb_j1] >= solver.limits[thumb_j1,1]-np.deg2rad(1))),
             p95_max_joint_step_deg=np.percentile(np.degrees(abs(np.diff(q,axis=0))).max(1),95),
             frame88=dict(thumb_distal_bend_deg=np.degrees(bend[focus,0,1]),thumb_pad_normal_deg=normal_err[focus],
                          ring_distal_bend_deg=np.degrees(bend[focus,3,1]),relative_pad_vector_mean_mm=vec_err[focus].mean()))
        for i,frame in enumerate(ids):
            for j,d in enumerate(DIGITS):
                rows.append(dict(method=name,frame=int(frame),digit=d,mano_mid_deg=np.degrees(human_bend[i,j,0]),
                          robot_mid_deg=np.degrees(bend[i,j,0]),mano_distal_deg=np.degrees(human_bend[i,j,1]),
                          robot_distal_deg=np.degrees(bend[i,j,1]),thumb_pad_normal_deg=normal_err[i] if not j else '',
                          thumb_to_digit_vec_err_mm=vec_err[i,j-1] if j else '',thumb_to_digit_gap_err_mm=gap_err[i,j-1] if j else '',
                          absolute_tip_mm=tip_abs[i,j]))
        print(name, json.dumps(report['methods'][name]), flush=True)
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'per_frame.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(args.out/'trajectories.npz',**archive)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,default=Path.home()/'Projects/tendon-spin/assets')
    parser.add_argument('--surface',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=HERE/'palm_anchor_ablation')
    run(parser.parse_args())
