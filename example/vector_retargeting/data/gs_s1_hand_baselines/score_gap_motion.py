"""Score the thumb gap and motion ablation against unfiltered, same-frame MANO."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from gs_s1_mujoco import load_hand
from gs_s1_optimization import ConstrainedRetargeter
from gs_s1_rviz import build_solver, get_package_share_directory
from palm_anchor_experiment import Hand, angle_deg, finger_shape, surface_targets


def score(example, experiments):
    v3 = np.load(example/'v3/gs_s1.npz')
    saved = np.load(example/'v3_weight_sweep/w0.70_base0.006_compressed_carry_mcp.npz')
    human = v3['human_distal_gaps_mm']
    frames, fps = v3['frame'],float(v3['fps'])
    time = frames/fps
    lag = max(1,round(.10*fps))
    human_step = human[lag:] - human[:-lag]
    complete = frames[lag:]-frames[:-lag] == lag
    oppose = np.zeros(human.shape,dtype=bool)
    release = np.zeros(human_step.shape,dtype=bool)
    windows = ('食指','中指','无名指','小指')
    with (example/'v3/opposition_comparison.csv').open(encoding='utf-8-sig') as f:
        for j,row in enumerate(csv.DictReader(f)):
            assert row['指对'] == windows[j]
            beg,end = float(row['观察窗起始秒']),float(row['观察窗末秒'])
            oppose[:,j] = (time>=beg)&(time<=end)
            idx = np.flatnonzero(oppose[:,j])
            t_min = time[idx[np.argmin(human[idx,j])]]
            release[:,j] = complete & (time[:-lag]>=t_min) & (time[lag:]<=end+.50) & (human_step[:,j]>0)

    seq,_ = build_solver()
    meshes = Path(get_package_share_directory('dexhand_description'))/'meshes/gs_s1'
    hand = Hand(ConstrainedRetargeter(seq,load_hand(Path.home()/'Projects/tendon-spin/assets',meshes,'right')),meshes)
    np.testing.assert_array_equal(seq.optimizer.target_joint_names,v3['joint_names'])
    mano = np.load(example/'mano.npz')
    np.testing.assert_array_equal(frames,mano['frame'])
    joints = mano['joints_camera_wrist_m']
    local = (joints-joints[:,:1]) @ v3['camera_to_palm_row']
    human_bend = finger_shape(local[:,1:].reshape(-1,5,4,3))[1]
    _,human_normals,_ = surface_targets(dict(frame=frames,joints=joints,vertices=mano['vertices_camera_wrist_m'],
                                             faces=mano['faces']),dict(frame=frames,raw_human_palm_m=local))
    human_near = human<=2
    separated = human>=5
    rows = []
    for params_path in sorted(experiments.glob('*.json')):
        p = json.loads(params_path.read_text())
        data = np.load(params_path.with_suffix('.npz'))
        np.testing.assert_array_equal(frames,data['frame'])
        gaps,q = data['robot_distal_gaps_mm'], data['active_qpos_rad']
        assert gaps.shape == human.shape and np.isfinite(gaps).all() and gaps.min()>=.5-1e-4
        assert p['weight']==.70 and p['carry_offset'] and p['mcp_flex'] and p['motion_lag_frames']==lag
        if (p['gap_target'],p['base_cost'],p['motion_weight']) == ('compressed',.006,0):
            np.testing.assert_allclose(gaps,saved['robot_distal_gaps_mm'],rtol=0,atol=1e-6)
            np.testing.assert_allclose(q,saved['active_qpos_rad'],rtol=0,atol=1e-8)
        robot_step = gaps[lag:]-gaps[:-lag]
        release_error = (robot_step-human_step)[release]
        err = gaps-human
        xyz,_,normals = hand.states(q)
        row = dict(间隙目标=p['gap_target'],指根代价倍率=p['base_cost'],变化项权重=p['motion_weight'],
                   全片间隙MAE_mm=np.abs(err).mean(),对指窗MAE_mm=np.abs(err[oppose]).mean())
        for j,digit in enumerate(windows):
            e = err[oppose[:,j],j]
            row[digit+'窗MAE_mm'] = abs(e).mean()
            row[digit+'窗平均有符号误差_mm'] = e.mean()
        row.update(松开指对帧数=release.sum(),松开时100ms变化MAE_mm=abs(release_error).mean(),
                   松开时100ms变化平均偏差_mm=release_error.mean(),松开方向一致帧数=(robot_step[release]>0).sum(),
                   MANO近接指对帧数=human_near.sum(),机器人同帧近接数=(gaps[human_near]<=2).sum(),
                   MANO分开指对帧数=separated.sum(),疑似误合数=(gaps[separated]<=2).sum(),求解回退帧=p['held'],
                   对MANO平均弯曲误差_deg=np.degrees(abs(finger_shape(xyz)[1]-human_bend)).mean(),
                   对MANO拇指指腹平均朝向误差_deg=angle_deg(normals[:,0],human_normals[:,0]).mean())
        rows.append(row)
    expected={(t,c,m) for t in ('compressed','scaled') for c,m in ((1,0),(.006,0),(.006,.35),(.006,1))}
    assert {(r['间隙目标'],r['指根代价倍率'],r['变化项权重']) for r in rows} == expected
    with (experiments/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(r['间隙目标'],r['指根代价倍率'],r['变化项权重'],
              *[round(r[key],3) for key in ('对指窗MAE_mm','松开时100ms变化MAE_mm','机器人同帧近接数','疑似误合数','求解回退帧')])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('example',type=Path)
    parser.add_argument('experiments',type=Path)
    score(**vars(parser.parse_args()))
