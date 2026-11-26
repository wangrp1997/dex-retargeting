"""Compare v3 and v3.1 on the example's indexed actions and MANO near misses."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def compare(example):
    baseline = np.load(example/'v3/gs_s1.npz')
    current = np.load(example/'v3_1/gs_s1.npz')
    for key in ('frame', 'fps', 'joint_names', 'human_distal_gaps_mm',
                'v2_active_qpos_rad', 'closure_weights'):
        np.testing.assert_array_equal(current[key], baseline[key])
    frames = current['frame']
    time = frames / float(current['fps'])
    human = current['human_distal_gaps_mm']
    scale = float(current['gap_scale'])
    target = np.hypot(.5, scale * human)  # use unfiltered measurements for evaluation
    tracks = [baseline['robot_distal_gaps_mm'], current['robot_distal_gaps_mm']]
    digits = ['食指','中指','无名指','小指']
    rows = []
    def record(kind, digit, idx):
        gaps = [g[idx, digit] for g in [human, *tracks]]
        rows.append([kind, digits[digit], int(frames[idx[0]]), int(frames[idx[-1]]),
                     *[round(float(g.min()), 3) for g in gaps],
                     *[int((g <= 2).sum()) for g in gaps[1:]]])

    with (example/'contact_review/原视频动作索引.csv').open(encoding='utf-8-sig') as f:
        actions = {r['画面动作_遮挡接触未确认']:r for r in csv.DictReader(f)}
    for j,digit in enumerate(digits):
        r = actions['拇指与'+digit+'对指']
        idx = np.flatnonzero((time >= float(r['观察窗起始秒'])) & (time <= float(r['观察窗末秒'])))
        record('原视频对指窗', j, idx)

    # Select complete approach-and-withdrawal events using only the MANO track.
    # The flanking frames must be observed, contiguous, and outside 20 mm.
    near = np.zeros(human.shape, dtype=bool)
    for j in range(4):
        inside = np.flatnonzero(human[:,j] < 20)
        for idx in np.split(inside, np.flatnonzero(np.diff(inside) != 1) + 1):
            if len(idx) < 3 or idx[0] == 0 or idx[-1] == len(frames)-1:
                continue
            if np.any(np.diff(frames[idx[0]-1:idx[-1]+2]) != 1):
                continue
            if 3 <= human[idx,j].min() <= 15:
                near[idx,j] = True
                record('MANO靠近后分开', j, idx)
    out = example/'v3_1'
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['片段','拇指对指','起始帧_从0','末帧_从0','MANO最小_mm','v3最小_mm',
                         'v3.1最小_mm','v3间隙不大于2mm帧数','v3.1间隙不大于2mm帧数'])
        writer.writerows(rows)
    separated = human >= 5
    report = dict(near_pass_events=len(rows)-4, near_pass_pair_frames=int(near.sum()),
                  separated_pair_frames=int(separated.sum()), human_scale=scale)
    for name,gap in zip(('v3','v3.1'),tracks):
        too_small = np.maximum(target-gap, 0)
        report[name] = dict(near_passes_reaching_2mm=sum(r[-2 if name=='v3' else -1]>0 for r in rows[4:]),
                            near_pass_pair_frames_within_2mm=int((gap[near]<=2).sum()),
                            separated_pair_frames_within_2mm=int((gap[separated]<=2).sum()),
                            near_pass_mean_gap_shortfall_mm=float(too_small[near].mean()) if near.any() else None,
                            near_pass_mean_abs_scaled_gap_error_mm=float(abs(target-gap)[near].mean()) if near.any() else None,
                            clip_mean_abs_scaled_gap_error_mm=float(abs(target-gap).mean()))
    (out/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))
    for row in rows[:4]:
        print(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('example', type=Path)
    compare(parser.parse_args().example)
