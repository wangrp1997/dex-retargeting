"""Time the causal Botyard pipeline on cached observations without publishing motion."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from ament_index_python.packages import get_package_share_directory

from botyard_live import BotyardRetargeter
from gs_s1_mujoco import load_hand
from gs_s1_rviz import build_solver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Cached world, pixels, size, time_s and dt arrays')
    parser.add_argument('--start', type=int, default=300)
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--frames', type=int, default=24)
    parser.add_argument('--hand', choices=('right', 'left'), default='right')
    parser.add_argument('--assets', type=Path, default=Path(__file__).resolve().parents[5] / 'tendon-spin/assets')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    source = np.load(args.source, allow_pickle=False)
    stop = args.start + args.warmup + args.frames
    if args.start < 0 or args.warmup < 0 or args.frames < 1 or stop > len(source['time_s']):
        parser.error('Requested frame range is outside the source')
    if 'valid' in source and not source['valid'][args.start:stop].all():
        parser.error('Select a fully tracked frame range')
    meshes = Path(get_package_share_directory('dexhand_description')) / 'meshes/gs_s1'
    retargeter = BotyardRetargeter(build_solver(hand=args.hand, objective='botyard')[0],
                                  load_hand(args.assets, meshes, args.hand), meshes, args.hand,
                                  None, 'one_euro', gradient='analytic', shape_model='v4_2')
    timings, current = {}, {}

    def instrument(obj, method, label):
        original = getattr(obj, method)

        def measured(*values, **kwargs):
            start = perf_counter()
            try:
                return original(*values, **kwargs)
            finally:
                current[label] = perf_counter() - start
        setattr(obj, method, measured)

    instrument(retargeter.mano, 'fit', 'mano_fit')
    instrument(retargeter.reference, 'solve_objective', 'finger_shape')
    instrument(retargeter.output, 'solve_objective', 'thumb_base')
    instrument(retargeter.protected_closure, 'refine', 'surface_closure')
    for k in range(args.start, stop):
        current.clear()
        begin = perf_counter()
        retargeter.solve_camera(source['world'][k], source['pixels'][k], source['size'],
                                float(source['time_s'][k]), float(source['dt'][k]))
        current['total_fit_and_retarget'] = perf_counter() - begin
        if k >= args.start + args.warmup:
            for name, seconds in current.items():
                timings.setdefault(name, []).append(1000 * seconds)
    report = dict(hand=args.hand, algorithm='botyard', algorithm_version='4.2', start=args.start,
                  warmup=args.warmup, frames=args.frames,
                  stages={name: dict(mean_ms=float(np.mean(values)), median_ms=float(np.median(values)),
                                     p95_ms=float(np.percentile(values, 95)))
                          for name, values in timings.items()},
                  note='Cached observations only; excludes camera, MediaPipe, drawing, ROS transport and hardware. '
                       'Run with OPENBLAS_NUM_THREADS=1. No ROS node or motion publisher is created.')
    encoded = json.dumps(report, indent=2) + '\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
    print(encoded, end='')


if __name__ == '__main__':
    main()
