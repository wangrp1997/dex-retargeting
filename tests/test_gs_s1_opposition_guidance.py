"""Offline guidance must have correct derivatives and retain v4.2 protections."""
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(params=[('left', 'volar'), ('right', 'volar'),
                        ('left', 'distance_priority'), ('right', 'distance_priority')])
def guided(request, monkeypatch):
    pytest.importorskip('rclpy')
    pytest.importorskip('mujoco')
    example = Path(__file__).resolve().parents[1] / 'example/vector_retargeting'
    monkeypatch.syspath_prepend(str(example))
    monkeypatch.syspath_prepend(str(example / 'data/gs_s1_hand_baselines'))
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
    from gs_s1_mujoco import load_hand
    from gs_s1_optimization import ConstrainedRetargeter
    from gs_s1_rviz import build_solver
    from opposition_guidance_experiment import DistancePriorityClosure, GuidedClosure
    from palm_anchor_experiment import Hand
    assets = example.parents[2] / 'tendon-spin/assets'
    if not assets.is_dir():
        pytest.skip('GS S1 MuJoCo assets are not installed')
    try:
        meshes = Path(get_package_share_directory('dexhand_description')) / 'meshes/gs_s1'
    except PackageNotFoundError:
        pytest.skip('GS S1 description is not installed')
    side, mode = request.param
    solver = ConstrainedRetargeter(build_solver(hand=side, objective='botyard')[0],
                                  load_hand(assets, meshes, side))
    closure = DistancePriorityClosure if mode == 'distance_priority' else GuidedClosure
    return closure(solver, Hand(solver, meshes, side))


def test_guidance_gradient_and_zero_proximity(guided):
    q = guided.solver.last.copy()
    q += .1 * (guided.solver.limits[:, 1] - q)
    chain = guided.hand.states(q)[0][0]
    guided.setup(q, chain, q, .1)
    weights = np.array([1., .7, .3, .1])
    _, analytic = guided.objective(q, weights)
    eps = 1e-6
    numeric = np.array([(guided.objective(q + d, weights)[0]
                         - guided.objective(q - d, weights)[0]) / (2 * eps)
                        for d in eps * np.eye(len(q))])
    np.testing.assert_allclose(analytic, numeric, atol=1e-4, rtol=1e-5)
    value, slope = guided.objective(q, np.zeros(4))
    assert value == 0.
    np.testing.assert_array_equal(slope, 0.)
    assert 10 < guided.pad_facets < guided.distance.meshes[0].num_tris
    # A subset of the same surface cannot have a smaller unsigned separation.
    full = guided.distance.distances(q)[0]
    patch = guided.pad_distance.distances(q)[0]
    assert (patch >= full - 1e-9).all()


def test_restarts_retain_source_bounds_and_one_velocity_allowance(guided, monkeypatch):
    previous = guided.solver.last.copy()
    chain = guided.hand.states(previous)[0][0]
    original = guided.solver.solve_objective
    starts = []

    def record(*args, **kwargs):
        starts.append(guided.solver.last.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(guided.solver, 'solve_objective', record)
    dt = .03
    q, info = guided.refine(previous, chain, np.ones(4), previous, dt)
    assert len(starts) >= 2
    for start in starts:
        np.testing.assert_array_equal(start, previous)
    assert np.max(abs(q - previous) - guided.solver.speed * dt) < 1e-7
    assert guided.feasible(q)
    assert info['min_protection_margin'] >= -1e-5
    if hasattr(guided, 'gap_reference'):
        # The refinement cannot buy orientation by worsening the active link gaps.
        margins = np.empty(4)
        guided.distance_inequalities(margins, q, np.empty(0))
        assert margins.max() <= 1e-7
        _, analytic = guided.objective(q, np.ones(4))
        eps = 1e-6
        numeric = np.array([(guided.objective(q + d, np.ones(4))[0]
                             - guided.objective(q - d, np.ones(4))[0]) / (2 * eps)
                            for d in eps * np.eye(len(q))])
        np.testing.assert_allclose(analytic, numeric, atol=1e-5, rtol=1e-5)


def test_failed_starts_retain_feasible_primary(guided, monkeypatch):
    import nlopt
    q = guided.solver.last.copy()
    chain = guided.hand.states(q)[0][0]

    def fail(*args, **kwargs):
        raise nlopt.invalid_argument('Exercise failed experimental starts')

    monkeypatch.setattr(guided.solver, 'solve_objective', fail)
    result, info = guided.refine(q, chain, np.zeros(4), q, .1)
    np.testing.assert_array_equal(result, q)
    assert info['optimizer_exception'] is not None
    assert guided.feasible(result)


def test_negligible_proximity_cannot_select_visible_inward_motion(guided):
    if not hasattr(guided, 'gap_reference'):
        pytest.skip('Applies to the corrected distance-priority experiment')
    q = guided.solver.last.copy()
    q[guided.base] = np.deg2rad([10., 10.])
    assert guided.geom.feasible(q)
    chain = guided.hand.states(q)[0][0]
    result, _ = guided.refine(q, chain, np.full(4, 1e-12), q, .1)
    np.testing.assert_allclose(result, q, atol=1e-7)
    # A nearby alternate seed still pays a finite posture cost as confidence
    # tends to zero, so it cannot win merely by having a better pad normal.
    displaced = guided.closure_reference.copy()
    displaced[guided.base[0]] += .01
    assert guided.objective(displaced, np.full(4, 1e-12))[0] > 1e-6
