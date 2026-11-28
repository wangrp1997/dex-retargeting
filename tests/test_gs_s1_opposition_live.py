"""Explicit live selection must run the reviewed solver on both hands, without devices."""
from pathlib import Path
from time import monotonic, sleep

import numpy as np
import pytest


@pytest.fixture(params=['left', 'right'])
def environment(request, monkeypatch):
    pytest.importorskip('rclpy')
    pytest.importorskip('mujoco')
    example = Path(__file__).resolve().parents[1] / 'example/vector_retargeting'
    monkeypatch.syspath_prepend(str(example))
    monkeypatch.syspath_prepend(str(example / 'data/gs_s1_hand_baselines'))
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
    assets = example.parents[2] / 'tendon-spin/assets'
    if not assets.is_dir():
        pytest.skip('GS S1 MuJoCo assets are not installed')
    try:
        meshes = Path(get_package_share_directory('dexhand_description')) / 'meshes/gs_s1'
    except PackageNotFoundError:
        pytest.skip('GS S1 description is not installed')
    if not (Path.home() / '.cache/botyard-retargeting/mano' / f'{request.param}.npz').is_file():
        pytest.skip('Licensed MANO template is not prepared')
    return assets, meshes, request.param


def construct(environment, factory, **kwargs):
    from gs_s1_mujoco import load_hand
    from gs_s1_rviz import build_solver
    assets, meshes, side = environment
    return factory(build_solver(hand=side, objective='botyard')[0], load_hand(assets, meshes, side),
                   meshes, side, None, 'one_euro', gradient='analytic', shape_model='v4_2', **kwargs)


def test_default_stays_v4_2_and_explicit_mode_uses_reviewed_settings(environment):
    from botyard_live import make_botyard_retargeter
    from opposition_guidance_experiment import DistancePriorityClosure
    from shape_priority_closure import ProtectedSurfaceClosure
    default = construct(environment, make_botyard_retargeter)
    assert type(default.protected_closure) is ProtectedSurfaceClosure
    assert default.opposition == 'v4_2'
    assert default.algorithm_version == '4.2'
    selected = construct(environment, make_botyard_retargeter, opposition='inward')
    assert isinstance(selected.protected_closure, DistancePriorityClosure)
    assert selected.opposition == 'inward'
    assert selected.algorithm_version == '4.3'
    assert selected.protected_closure.guide_weight == .2
    assert selected.protected_closure.starts == 3
    assert selected.protected_closure.facing_target == 'inward'
    assert selected.protected_closure.gap_slack_m == .0001


def test_invalid_mode_is_rejected_before_constructing_robot(environment):
    from botyard_live import make_botyard_retargeter
    from gs_s1_live_pipeline import SolverConfig
    with pytest.raises(ValueError, match='Unknown Botyard opposition mode'):
        make_botyard_retargeter(opposition='volar')
    with pytest.raises(ValueError, match='requires the v4_2 shape objective'):
        make_botyard_retargeter(opposition='inward', shape_model='sweep')
    with pytest.raises(ValueError, match='requires the v4_2 shape objective'):
        SolverConfig('', environment[2], None, 'one_euro', 'analytic', 'sweep', 'inward')


def wait_for(worker, *, ready=False):
    deadline = monotonic() + 30.
    while monotonic() < deadline:
        result = worker.poll()
        if (ready and worker.ready) or result is not None:
            return result
        sleep(.01)
    raise AssertionError('Production solver worker timed out')


def test_spawned_live_solver_matches_reviewed_serial_controller(environment):
    from gs_s1_live_pipeline import PoseTask, SolverConfig, SolverProcess
    from opposition_guidance_experiment import GuidedRetargeter
    source_path = Path('/tmp/gs-s1-botyard-v4.2-20261128/example/source.npz')
    if not source_path.is_file():
        pytest.skip('Cached licensed MANO observations are not available')
    with np.load(source_path) as cached:
        joints = cached['joints'][321:324].copy()
        vertices = cached['vertices'][321:324].copy()
    assets, _, side = environment
    if side == 'left':
        # Reflect the cached right-hand observation to exercise the left CAD.
        joints[..., 0] *= -1
        vertices[..., 0] *= -1
    serial = construct(environment, GuidedRetargeter, guide_weight=.2, starts=3,
                       mode='distance_priority', facing_target='inward')
    config = SolverConfig(str(assets), side, None, 'one_euro', 'analytic', 'v4_2', 'inward')
    worker = SolverProcess(config)
    try:
        wait_for(worker, ready=True)
        previous = serial.output.last.copy()
        for sequence, (pose, mesh) in enumerate(zip(joints, vertices)):
            stamp = 10.7 + sequence * .1
            serial.reference.last = previous.copy()
            serial.output.last = previous.copy()
            serial.protected_closure.solver.last = previous.copy()
            trace = {}
            expected = serial.solve_mano(pose, mesh, stamp, .1, trace=trace)
            worker.send(PoseTask(sequence, 0, stamp, pose, mesh, previous.copy(), .1))
            actual = wait_for(worker)
            np.testing.assert_allclose(actual.q, expected, atol=1e-10, rtol=0)
            np.testing.assert_allclose(actual.actuators, serial.output.last, atol=1e-10, rtol=0)
            assert trace['botyard_opposition'] == 'inward'
            assert trace['algorithm_version'] == '4.3'
            assert np.max(abs(actual.actuators - previous) - serial.output.speed * .1) < 1e-7
            assert serial.output.geom.feasible(actual.actuators)
            assert serial.output.palm.clear(serial.output.geom.data, .0005 - 1e-7)
            previous = actual.actuators.copy()
    finally:
        worker.close()
