"""Regression coverage for source priority in optional GS S1 closure."""
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def controller(monkeypatch):
    pytest.importorskip('rclpy')
    pytest.importorskip('mujoco')
    example = Path(__file__).resolve().parents[1] / 'example/vector_retargeting'
    monkeypatch.syspath_prepend(str(example))
    monkeypatch.syspath_prepend(str(example / 'data/gs_s1_hand_baselines'))
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
    from gs_s1_mujoco import load_hand
    from gs_s1_optimization import ConstrainedRetargeter
    from gs_s1_rviz import build_solver
    from palm_anchor_experiment import Hand
    try:
        meshes = Path(get_package_share_directory('dexhand_description')) / 'meshes/gs_s1'
    except PackageNotFoundError:
        pytest.skip('GS S1 description is not installed')
    assets = example.parents[2] / 'tendon-spin/assets'
    if not assets.is_dir():
        pytest.skip('GS S1 MuJoCo assets are not installed')
    solver = ConstrainedRetargeter(build_solver(hand='right', objective='botyard')[0],
                                  load_hand(assets, meshes, 'right'))
    return solver, Hand(solver, meshes)


def test_source_priority_gradient_and_bone_length_invariance(controller):
    from shape_priority_closure import SourceShapePriority
    solver, hand = controller
    rng = np.random.default_rng(77)
    target, current = rng.uniform(*solver.limits.T, (2, len(solver.last)))
    chain = hand.states(target)[0][0]
    priority = SourceShapePriority(hand).bind(chain, target)
    values, analytic = priority.losses(current)
    eps = 1e-6
    numeric = np.column_stack([
        (priority.losses(current + d)[0] - priority.losses(current - d)[0]) / (2 * eps)
        for d in eps * np.eye(len(current))])
    np.testing.assert_allclose(analytic, numeric, atol=1e-5, rtol=1e-5)
    # Human palm width, finger lengths and root offsets are not robot targets.
    bones = np.diff(chain, axis=1) * rng.uniform(.5, 2., (5, 3, 1))
    roots = chain[:, :1] + rng.uniform(-.04, .04, (5, 1, 3))
    resized = np.concatenate((roots, roots + np.cumsum(bones, axis=1)), axis=1)
    priority.bind(resized, target)
    actual, slope = priority.losses(current)
    np.testing.assert_allclose(actual, values, atol=1e-10)
    np.testing.assert_allclose(slope, analytic, atol=1e-9)


def test_local_corrections_cannot_gradually_erase_a_fist(controller):
    from shape_priority_closure import SourceShapePriority
    solver, hand = controller
    names = list(solver.opt.target_joint_names)
    pip = names.index('FFJ2')
    target = solver.last.copy()
    target[pip] = np.deg2rad(75.)
    chain = hand.states(target)[0][0]
    priority = SourceShapePriority(hand)
    previous = target.copy()
    rejected = 0
    for _ in range(100):
        # Primary makes progress toward the current source from actual state;
        # a secondary closure task tries to undo slightly more on every frame.
        primary = previous.copy()
        primary[pip] = min(target[pip], primary[pip] + np.deg2rad(.3))
        priority.bind(chain, primary)
        proposed = primary.copy()
        proposed[pip] -= np.deg2rad(.8)
        if priority.margins(proposed)[0].min() >= -1e-8:
            previous = proposed
        else:
            previous = primary
            rejected += 1
    assert rejected > 0
    assert np.rad2deg(previous[pip]) >= 65. - 1e-5
    # A 10 degree bound around a moving joint reference would accept every
    # proposed 0.8 degree step and lose about 50 degrees of this fist.


def test_j4_can_follow_nonzero_source_spread(controller):
    from shape_priority_closure import SourceShapePriority
    solver, hand = controller
    names = list(solver.opt.target_joint_names)
    spread = names.index('FFJ4')
    source_pose = solver.last.copy()
    source_pose[spread] = solver.limits[spread, 1]
    assert abs(source_pose[spread]) > np.deg2rad(5.)
    chain = hand.states(source_pose)[0][0]
    priority = SourceShapePriority(hand).bind(chain, source_pose)
    assert priority.margins(source_pose)[0].min() >= 0.
    zeroed = source_pose.copy()
    zeroed[spread] = 0.
    assert priority.margins(zeroed)[0][5] < 0.


def test_surface_gap_cost_gradient_and_zero_weight(controller):
    from distal_closure import DistalClosure
    from shape_priority_closure import distal_gap_cost
    solver, hand = controller
    distance = DistalClosure(solver)
    q = solver.last.copy()
    q += .3 * (solver.limits[:, 1] - q)
    weights = np.array([1., .7, .3, .1])
    _, analytic = distal_gap_cost(distance, q, q, weights)
    eps = 1e-6
    numeric = np.array([
        (distal_gap_cost(distance, q + d, q, weights)[0]
         - distal_gap_cost(distance, q - d, q, weights)[0]) / (2 * eps)
        for d in eps * np.eye(len(q))])
    np.testing.assert_allclose(analytic, numeric, atol=1e-4, rtol=1e-5)
    value, slope = distal_gap_cost(distance, q, q, np.zeros(4))
    assert value == 0.
    np.testing.assert_array_equal(slope, 0.)


def test_complete_closure_jacobians_and_soft_facing(controller):
    from shape_priority_closure import ProtectedSurfaceClosure
    solver, hand = controller
    closure = ProtectedSurfaceClosure(solver, hand)
    q = solver.last + .3 * (solver.limits[:, 1] - solver.last)
    chain = hand.states(q)[0][0]
    closure.setup(q, chain, q, .1)
    weights = np.array([1., .7, .3, .1])
    value, analytic = closure.objective(q, weights)
    _, jac = closure.protection(q)
    eps = 1e-6
    numeric = np.array([(closure.objective(q + d, weights)[0]
                         - closure.objective(q - d, weights)[0]) / (2 * eps)
                        for d in eps * np.eye(len(q))])
    numeric_jac = np.column_stack([(closure.protection(q + d)[0]
                                   - closure.protection(q - d)[0]) / (2 * eps)
                                  for d in eps * np.eye(len(q))])
    np.testing.assert_allclose(analytic, numeric, atol=1e-4, rtol=1e-5)
    np.testing.assert_allclose(jac, numeric_jac, atol=1e-4, rtol=1e-5)
    # With no proximity evidence the secondary task has its unique minimum at
    # the primary pose, even when its custom facing field is imperfect.
    value, slope = closure.objective(q, np.zeros(4))
    assert value == 0.
    np.testing.assert_array_equal(slope, 0.)


def test_closure_retains_feasible_primary_on_native_nlopt_error(controller, monkeypatch):
    import nlopt
    from shape_priority_closure import ProtectedSurfaceClosure
    solver, hand = controller
    closure = ProtectedSurfaceClosure(solver, hand)
    primary = solver.last.copy()
    chain = hand.states(primary)[0][0]
    original_limits = solver.limits.copy()

    def fail(*args, **kwargs):
        raise nlopt.runtime_error('injected numerical failure')

    monkeypatch.setattr(solver, 'solve_objective', fail)
    command, info = closure.refine(primary, chain, np.ones(4), primary, .1)
    np.testing.assert_array_equal(command, primary)
    np.testing.assert_array_equal(solver.limits, original_limits)
    assert info['optimizer_exception'] == 'runtime_error'
    assert closure.feasible(command)


def test_warm_start_cannot_undo_primary_bending_progress(controller, monkeypatch):
    import nlopt
    from shape_priority_closure import ProtectedSurfaceClosure
    solver, hand = controller
    closure = ProtectedSurfaceClosure(solver, hand)
    pip = list(solver.opt.target_joint_names).index('FFJ2')
    previous = solver.last.copy()
    primary = previous.copy()
    primary[pip] = np.deg2rad(40.)
    target = primary.copy()
    target[pip] = np.deg2rad(75.)
    chain = hand.states(target)[0][0]

    def cheaper_when_straight(q, weights):
        grad = np.zeros_like(q)
        grad[pip] = 2 * q[pip]
        return float(q[pip]**2), grad

    def inspect_initial(*args, initial, **kwargs):
        np.testing.assert_array_equal(initial, primary)
        raise nlopt.runtime_error('injected numerical failure')

    monkeypatch.setattr(closure, 'objective', cheaper_when_straight)
    monkeypatch.setattr(solver, 'solve_objective', inspect_initial)
    result, _ = closure.refine(primary, chain, np.ones(4), previous, 1.)
    np.testing.assert_array_equal(result, primary)
