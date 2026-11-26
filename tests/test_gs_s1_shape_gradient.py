"""Gradient checks for the optional GS S1 ROS/MuJoCo example."""
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def shape(monkeypatch):
    pytest.importorskip('rclpy')
    pytest.importorskip('mujoco')
    pytest.importorskip('mediapipe')
    example = Path(__file__).resolve().parents[1] / 'example/vector_retargeting'
    monkeypatch.syspath_prepend(str(example))
    monkeypatch.syspath_prepend(str(example / 'data/gs_s1_hand_baselines'))
    import palm_anchor_experiment
    return palm_anchor_experiment


def test_straight_finger_symmetric_derivative(shape):
    chain = np.zeros((5, 4, 3))
    chain[:, :, 2] = [.0, .033, .055, .071]
    bones, lengths, axis, sine, bend = shape.bone_geometry(chain)
    # Give each finger a nonzero target bend. The two directions away from
    # an exactly straight finger cancel in a centered derivative.
    gradient = shape.shape_point_gradient(bones, lengths, axis, sine, np.zeros((5, 3)), bend - .5)
    np.testing.assert_array_equal(gradient, np.zeros_like(chain))


@pytest.mark.parametrize('side', ['left', 'right'])
def test_coupled_hand_gradient(shape, side):
    from ament_index_python.packages import PackageNotFoundError
    try:
        share = shape.get_package_share_directory('dexhand_description')
    except PackageNotFoundError:
        pytest.skip('GS S1 description is not installed')
    assets = Path(__file__).resolve().parents[2] / 'tendon-spin/assets'
    if not assets.is_dir():
        pytest.skip('GS S1 MuJoCo assets are not installed')
    meshes = Path(share) / 'meshes/gs_s1'
    seq, _ = shape.build_solver(hand=side, objective='botyard')
    solver = shape.ConstrainedRetargeter(seq, shape.load_hand(assets, meshes, side))
    hand = shape.Hand(solver, meshes, side)
    rng = np.random.default_rng(42)
    lo, hi = solver.limits.T
    poses = np.vstack((solver.last, lo, hi, rng.uniform(lo, hi, (8, len(lo)))))
    for i, q in enumerate(poses):
        chain, pads, normals = (v[0] for v in hand.states(rng.uniform(lo, hi)))
        args = hand, chain, pads, normals, solver.last, .1
        analytic, finite = [shape.objective(*args, opposition=bool(i % 2), gradient=mode)
                            for mode in ('analytic', 'finite_difference')]
        actual, expected = np.empty_like(q), np.empty_like(q)
        np.testing.assert_allclose(analytic(q, actual), finite(q, expected), rtol=1e-13)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-6)
