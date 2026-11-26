"""Measure hand shape in the palm frame of the saved GS S1 DexPilot replay."""
import argparse
import csv
from pathlib import Path

import numpy as np

from gs_s1_hand_baselines import angle_deg
from gs_s1_rviz import build_solver

DIGITS = ("th", "ff", "mf", "rf", "lf")


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def cad_pad_normal(stl):
    # Uniform samples on the outer volar surface of thdistal, ray-cast from -Y.
    dtype = [('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')]
    faces = np.fromfile(stl, dtype=dtype, offset=84)
    a, b, c = (faces['vertices'] * 1000).transpose(1, 0, 2)
    ab, ac = b - a, c - a
    det = ab[:, 0] * ac[:, 2] - ac[:, 0] * ab[:, 2]
    hits = []
    for x in (-2, -1, 0, 1, 2):
        for z in (16, 17, 18, 19, 20, 21, 22):
            with np.errstate(divide='ignore', invalid='ignore'):
                u = ((x-a[:, 0])*ac[:, 2] - (z-a[:, 2])*ac[:, 0])/det
                v = (ab[:, 0]*(z-a[:, 2]) - ab[:, 2]*(x-a[:, 0]))/det
            ids = np.flatnonzero((u >= 0) & (v >= 0) & (u+v <= 1))
            y = a[ids, 1] + u[ids]*ab[ids, 1] + v[ids]*ac[ids, 1]
            hit = faces['normal'][ids[np.argmin(y)]]
            assert hit[1] < 0
            hits.append(hit)
    return unit(np.mean(hits, axis=0))


def patch_normal(vertices, faces):
    a, b, c = vertices[:, faces].transpose(2, 0, 1, 3)
    return unit(np.cross(b-a, c-a).sum(axis=1))


def draw_frame(path, frame, human, robot, human_bend, robot_bend, tips, separation):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Arc
    from matplotlib.lines import Line2D

    plt.rcParams['font.family'] = 'Noto Sans CJK JP'
    blue, orange = '#006da1', '#a94700'
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.55))
    fig.subplots_adjust(left=.045, right=.98, bottom=.23, top=.73, wspace=.27)
    fig.text(.035, .935, f'DexMV 芥末瓶抓取  /  源帧 {frame}', fontsize=18, weight='bold')
    legend = [Line2D([], [], color=blue, marker='o', label='MANO 人手'),
              Line2D([], [], color=orange, marker='s', ls='--', label='GS S1 · DexPilot')]
    fig.legend(handles=legend, loc='upper right', bbox_to_anchor=(.99,.985), frameon=False, ncol=2)
    for ax, j, label in zip(axes[:2], (0, 3), ('拇指', '无名指')):
        h, r = 1000*human[j], 1000*robot[j]
        all_points = np.concatenate((h, r))
        center = all_points.mean(axis=0)
        _, _, basis = np.linalg.svd(all_points-center, full_matrices=False)
        # Both hands use the same orthographic projection and a common origin.
        plane = basis[:2].copy()
        if (h[-1]-h[0]) @ plane[0] < 0:
            plane[0] *= -1
        if (r[1]-h[1]) @ plane[1] < 0:
            plane[1] *= -1
        for xyz, c, mark, ls in ((h,blue,'o','-'),(r,orange,'s','--')):
            xy=(xyz-center) @ plane.T
            ax.plot(*xy.T, color=c, marker=mark, ls=ls, lw=2.2, ms=5)
        ax.set(aspect='equal', xlim=(-68,68), ylim=(-39,39))
        ax.plot([-60,-40],[-33,-33],c='#444',lw=1.8)
        ax.text(-50,-29,'20 mm',ha='center',fontsize=9)
        ax.set_title(f'{label}   ·   指尖相距 {tips[j]:.1f} mm',fontsize=13,loc='left')
        ax.text(.50,-.06,f'末节弯曲   人手 {human_bend[j,1]:.1f}°   /   GS S1 {robot_bend[j,1]:.1f}°',
                transform=ax.transAxes,ha='center',fontsize=10.5)
    ax=axes[2]
    angle=np.radians(separation)
    ax.set(xlim=(-.5,1.42),ylim=(-.25,1.25),aspect='equal')
    ax.set_title('拇指指腹   ·   表面朝向',fontsize=13,loc='left')
    for a, c, style in ((0,blue,'-'),(angle,orange,'--')):
        end=(np.cos(a),np.sin(a))
        ax.annotate('',xy=end,xytext=(0,0),arrowprops=dict(arrowstyle='-|>',lw=2.5,color=c,ls=style,mutation_scale=18))
    ax.add_patch(Arc((0,0),.9,.9,theta1=0,theta2=separation,color='#444',lw=1.2))
    ax.text(.42,.38,f'{separation:.1f}°',fontsize=14)
    ax.text(.5,-.06,'两支箭头平移到同一起点',transform=ax.transAxes,ha='center',fontsize=10.5)
    for ax in axes:
        ax.axis('off')
    fig.text(.035,.072,'前两图分别将两只手的同一根手指投影到共同平面。标注的距离和夹角均由三维数据计算。',fontsize=10,color='#3c4853')
    fig.savefig(path,dpi=165,facecolor='white')
    plt.close(fig)


def main(args):
    track = np.load(args.data_dir / 'trajectories.npz')
    src = np.load(args.surface)
    frames = track['frame']
    np.testing.assert_array_equal(frames, src['frame'])
    points = src['joints'] - src['joints'][:, :1]
    human = track['raw_human_palm_m']
    u, _, vt = np.linalg.svd(points.transpose(0, 2, 1) @ human)
    rotation = u @ vt
    np.testing.assert_allclose(np.linalg.det(rotation), 1., atol=1e-5)
    np.testing.assert_allclose(points @ rotation, human, atol=1e-6)
    vertices = (src['vertices']-src['joints'][:, :1]) @ rotation
    faces = src['faces']
    # Vertex 763 is central along the surface between the two flank anchors of
    # the distal thumb in manotorch. Hold the patch indices fixed across time.
    center_patch = np.any(faces == 763, axis=1)
    hnormal = patch_normal(vertices, faces[center_patch])
    local_normal = cad_pad_normal(args.stl)
    print('robot normal, human patch size', local_normal, center_patch.sum())

    seq, _ = build_solver()
    robot = seq.optimizer.robot
    idx = [seq.joint_names.index(n) for n in track['joint_names']]
    names = [d+s for d in DIGITS for s in ('proximal', 'middle', 'distal', 'tip')]
    links = [robot.get_link_index(n) for n in names]
    robot.compute_forward_kinematics(robot.q0)
    palm_inv = np.linalg.inv(robot.get_link_pose(robot.get_link_index('palm')))
    xyz, rnormal, nominal = [], [], []
    for controls in track['dexpilot_active_qpos_rad']:
        q = robot.q0.copy()
        q[idx] = controls
        q = seq.optimizer.adaptor.forward_qpos(q)
        for d in ('FF', 'MF', 'RF', 'LF'):
            assert q[seq.joint_names.index(d+'J1')] == q[seq.joint_names.index(d+'J2')]
        robot.compute_forward_kinematics(q)
        pose = np.stack([palm_inv @ robot.get_link_pose(link) for link in links])
        xyz.append(pose[:, :3, 3].reshape(5, 4, 3))
        rnormal.append(pose[2, :3, :3] @ local_normal)
        nominal.append(-pose[2, :3, 1])
    robot_xyz = np.array(xyz)
    human_xyz = human[:, 1:].reshape(robot_xyz.shape)
    rb, hb = (np.diff(a, axis=2) for a in (robot_xyz, human_xyz))
    robot_bend, human_bend = (angle_deg(a[:, :, :-1], a[:, :, 1:]) for a in (rb, hb))
    tip_error = 1000*np.linalg.norm(robot_xyz[:, :, -1]-human_xyz[:, :, -1], axis=-1)
    separation = angle_deg(hnormal, np.array(rnormal))
    focus = list(frames).index(88)
    for name, n in [('center', hnormal),('CPF distal',patch_normal(vertices, np.array([[766,764,768]]))),('flank A',patch_normal(vertices, np.array([[739,736,740]]))),('flank B',patch_normal(vertices, np.array([[761,759,762]])))]:
        val = angle_deg(n, np.array(rnormal))
        print(name,'frame 88/median/min/max/obtuse',val[focus],np.median(val),val.min(),val.max(),sum(val>90))
    print('nominal direction at 88',angle_deg(hnormal,np.array(nominal))[focus])
    print('frame 88 human/robot, tip mm',human_bend[focus].round(2),robot_bend[focus].round(2),tip_error[focus].round(2))
    print('mean absolute discrepancies',np.abs(robot_bend-human_bend).mean(0).round(2))
    with (args.data_dir/'thumb.csv').open() as f:
        old = [row for row in csv.DictReader(f) if row['method']=='dexpilot']
    for col, actual in [('thumb_tip_mm', tip_error[:,0]),('robot_thumb_distal_bend_deg', robot_bend[:,0,1]),('human_thumb_ip_bend_deg',human_bend[:,0,1])]:
        np.testing.assert_allclose([float(row[col]) for row in old],actual,atol=1e-5)
    if args.figure:
        draw_frame(args.figure, frames[focus], human_xyz[focus], robot_xyz[focus], human_bend[focus], robot_bend[focus], tip_error[focus], separation[focus])
    rows = []
    for i, frame in enumerate(frames):
        for j, digit in enumerate(DIGITS):
            rows.append(dict(frame=int(frame), digit=digit, tip_error_mm=tip_error[i,j],
                             mano_mid_bend_deg=human_bend[i,j,0], gs_mid_bend_deg=robot_bend[i,j,0],
                             mano_distal_bend_deg=human_bend[i,j,1], gs_distal_bend_deg=robot_bend[i,j,1],
                             thumb_pad_angle_deg=separation[i] if j==0 else ''))
    with args.output.open('w',newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('surface', 'stl', 'data-dir', 'output'):
        parser.add_argument('--'+key, required=True, type=Path)
    parser.add_argument('--figure', type=Path)
    main(parser.parse_args())
