"""Exact geometry reuse must preserve witnesses, derivatives and MuJoCo state."""
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(params=['right', 'left'])
def geometry(request, monkeypatch):
    pytest.importorskip('mujoco')
    pytest.importorskip('rclpy')
    example = Path(__file__).resolve().parents[1] / 'example/vector_retargeting'
    monkeypatch.syspath_prepend(str(example))
    monkeypatch.syspath_prepend(str(example / 'data/gs_s1_hand_baselines'))
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
    from gs_s1_mujoco import load_hand
    from gs_s1_optimization import ConstrainedRetargeter
    from gs_s1_rviz import build_solver
    assets = example.parents[2] / 'tendon-spin/assets'
    if not assets.is_dir():
        pytest.skip('GS S1 MuJoCo assets are not installed')
    try:
        meshes = Path(get_package_share_directory('dexhand_description')) / 'meshes/gs_s1'
    except PackageNotFoundError:
        pytest.skip('GS S1 description is not installed')
    return ConstrainedRetargeter(build_solver(hand=request.param, objective='botyard')[0],
                                load_hand(assets, meshes, request.param))


def test_distal_cache_value_then_gradient_and_independent_finger(geometry, monkeypatch):
    import mujoco
    from distal_closure import DistalClosure, hppfcl
    distance = DistalClosure(geometry)
    original = hppfcl.distance
    calls = []

    def query(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(hppfcl, 'distance', query)
    q = geometry.last.copy()
    expected, _ = distance.distances(q)
    assert len(calls) == 4
    actual, analytic = distance.distances(q, with_gradient=True)
    np.testing.assert_array_equal(actual, expected)
    assert len(calls) == 4
    # Returned arrays are caller-owned; mutating them cannot poison the cache.
    actual[:] = -1
    analytic[:] = -1
    actual, analytic = distance.distances(q, with_gradient=True)
    np.testing.assert_array_equal(actual, expected)
    assert (analytic != -1).any()
    q[list(geometry.opt.target_joint_names).index('FFJ2')] += .06
    distance.distances(q, with_gradient=True)
    assert len(calls) == 5  # Three unchanged finger/thumb pairs reuse witnesses.
    np.testing.assert_array_equal(distance.data.qpos, distance.mapping @ q)
    fresh = mujoco.MjData(distance.model)
    fresh.qpos[:] = distance.mapping @ q
    mujoco.mj_kinematics(distance.model, fresh)
    np.testing.assert_array_equal(distance.data.xpos, fresh.xpos)
    expected, jac = DistalClosure(geometry).distances(q, with_gradient=True)
    actual, cached_jac = distance.distances(q, with_gradient=True)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    np.testing.assert_allclose(cached_jac, jac, atol=1e-8)
    # A line search may return to an older trial, with another pose in between.
    # Its Jacobian is reusable, but MuJoCo must still describe the requested q.
    calls_before = len(calls)
    earlier = geometry.last.copy()
    restored, restored_jac = distance.distances(earlier, with_gradient=True)
    assert len(calls) == calls_before
    np.testing.assert_array_equal(distance.data.qpos, distance.mapping @ earlier)
    reference, reference_jac = DistalClosure(geometry).distances(earlier, with_gradient=True)
    np.testing.assert_array_equal(restored, reference)
    np.testing.assert_array_equal(restored_jac, reference_jac)
    restored[:] = -1
    restored_jac[:] = -1
    restored, restored_jac = distance.distances(earlier, with_gradient=True)
    np.testing.assert_array_equal(restored, reference)
    np.testing.assert_array_equal(restored_jac, reference_jac)
    eps = 1e-6
    numeric = np.column_stack([(distance.distances(q + d)[0] - distance.distances(q - d)[0]) / (2 * eps)
                               for d in eps * np.eye(len(q))])
    np.testing.assert_allclose(cached_jac, numeric, atol=1e-6, rtol=1e-4)
    assert len(distance.state_cache) <= 32


def test_collision_cache_preserves_contacts_and_constraint_slopes(geometry, monkeypatch):
    import mujoco
    geom = geometry.geom
    original, calls = mujoco.mj_collision, []

    def collide(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(mujoco, 'mj_collision', collide)
    geom.have_state = False
    q = geometry.last.copy()
    feasible = geom.feasible(q)
    value, jac = np.empty(geom.count), np.empty((geom.count, len(q)))
    geom.inequalities(value, q, jac)
    assert len(calls) == 1
    assert feasible == (value.max() <= 1e-7)
    geom.have_state = False
    expected, expected_jac = np.empty_like(value), np.empty_like(jac)
    geom.inequalities(expected, q, expected_jac)
    np.testing.assert_array_equal(value, expected)
    np.testing.assert_array_equal(jac, expected_jac)
    q[0] += .03
    geom.feasible(q)
    assert len(calls) == 3


def test_palm_cache_witness_and_both_body_poses(geometry, monkeypatch):
    from gs_s1_palm import PalmDistances, hppfcl
    palm = PalmDistances(geometry.geom.model)
    data = geometry.geom.data
    original, calls = hppfcl.distance, []

    def query(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(hppfcl, 'distance', query)
    expected, jac = palm.linearize(data, geometry.geom.mapping)
    count = len(calls)
    actual, cached_jac = palm.linearize(data, geometry.geom.mapping)
    assert len(calls) == count == len(palm.pairs)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(cached_jac, jac)
    distance, result = palm.measure(data, 0)
    assert np.isfinite(result.getNearestPoint1()).all()
    a, b = palm.pairs[0]
    data.xpos[a, 0] += .001
    palm.measure(data, 0)
    assert len(calls) == count + 1
    data.xpos[b, 0] += .001
    palm.measure(data, 0)
    assert len(calls) == count + 2
