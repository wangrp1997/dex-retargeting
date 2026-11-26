"""Static thumb–little reachability around the v2 pose at a chosen RGB frame."""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from distal_closure import DistalClosure
from gs_s1_mujoco import load_hand
from gs_s1_optimization import ConstrainedRetargeter
from gs_s1_rviz import build_solver, get_package_share_directory
from palm_anchor_experiment import Hand, angle_deg, finger_shape


def run(example, frame):
    dataset = np.load(example/'v3_1/gs_s1.npz')
    i = list(dataset['frame']).index(frame)
    ref = dataset['v2_active_qpos_rad'][i]
    seq,_ = build_solver()
    mesh_dir = Path(get_package_share_directory('dexhand_description'))/'meshes/gs_s1'
    solver = ConstrainedRetargeter(seq,load_hand(Path.home()/'Projects/tendon-spin/assets',mesh_dir,'right'))
    hand = Hand(solver, mesh_dir)
    closure = DistalClosure(solver)
    active = [k for k,name in enumerate(seq.optimizer.target_joint_names) if name.startswith(('TH','LF'))]
    np.testing.assert_array_equal(seq.optimizer.target_joint_names,dataset['joint_names'])
    def expand(x):
        full = ref.copy()
        full[active] = x
        return full

    def shape(x):
        xyz, _, normals = hand.states(expand(x))
        dirs,bends = finger_shape(xyz[0,[0,4]])
        return dirs,np.degrees(bends),normals[0,[0,4]]

    start = ref[active]
    base_dirs,base_bend,base_normals = shape(start)
    def changes(x):
        dirs,bends,normals = shape(x)
        return (bends-base_bend,
                np.concatenate([angle_deg(dirs,base_dirs),angle_deg(normals,base_normals)]))

    def geometry_margin(x):
        full = expand(x)
        g = solver.geom
        result = np.empty(g.count)
        g.inequalities(result,full,np.empty(0))
        palm = [solver.palm.measure(g.data,j)[0]-g.clearance for j in range(len(solver.palm.pairs))]
        return 1000*np.r_[-result,palm]

    def gap(x):
        values,jac = closure.distances(expand(x),with_gradient=True)
        return 1000*values[-1], 1000*jac[-1,active]

    limits = solver.limits[active]
    baseline_gap = gap(start)[0]
    rows = [[frame,'固定v2',0,0,baseline_gap,0.,0.,0.]]
    seeds = [start, dataset['active_qpos_rad'][i,active]]
    for bend_cap,dir_cap in ((2,2),(5,2),(2,5),(5,5),(10,10)):
        def shape_margin(x):
            bends,dirs = changes(x)
            return np.r_[(bend_cap-bends).ravel(),(bend_cap+bends).ravel(),dir_cap-dirs]
        best = start.copy()
        best_gap = baseline_gap
        def consider(x):
            nonlocal best,best_gap
            if (not np.isfinite(x).all() or (x<limits[:,0]-1e-9).any() or (x>limits[:,1]+1e-9).any()
                    or shape_margin(x).min() < -1e-5 or geometry_margin(x).min() < -1e-4):
                return
            value,_ = gap(x)
            if value < best_gap:
                best_gap, best = value,x.copy()
        for initial in seeds:
            consider(initial)
            result = minimize(gap, initial, jac=True, bounds=limits, method='SLSQP', callback=consider,
                              constraints=[dict(type='ineq',fun=shape_margin),
                                           dict(type='ineq',fun=geometry_margin)],
                              options=dict(ftol=1e-9,eps=1e-6,maxiter=120))
            consider(result.x)
        assert solver.geom.feasible(expand(best)) and solver.palm.clear(solver.geom.data,solver.geom.clearance-1e-7)
        bends,dirs = changes(best)
        rows.append([frame,'静态局部搜索',bend_cap,dir_cap,best_gap,abs(bends).max(),dirs.max(),
                     np.degrees(abs(best-start)).max()])
        seeds.append(best)
    out = example/'v3_1'/f'pinky_budget_frame_{frame}.csv'
    with out.open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['帧_从0','姿态','弯曲预算_deg','朝向预算_deg','可行间隙_mm','最大弯曲变化_deg',
                         '最大朝向变化_deg','最大主动关节变化_deg'])
        writer.writerows(rows)
    for row in rows:
        print(row,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('example',type=Path)
    parser.add_argument('--frame',type=int,default=462)
    run(**vars(parser.parse_args()))
