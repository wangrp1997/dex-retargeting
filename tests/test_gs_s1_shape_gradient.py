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
        target_q = rng.uniform(lo, hi)
        chain, pads, normals = (v[0] for v in hand.states(target_q))
        spread = target_q[hand.spread_indices, None]
        parallel, cosine, sine = hand.flexion_basis.transpose(1, 0, 2)
        actual_axes = [hand.data.xaxis[hand.model.joint(d + 'J3').id] for d in ('FF', 'MF', 'RF', 'LF')]
        np.testing.assert_allclose(parallel + cosine * np.cos(spread) + sine * np.sin(spread),
                                   np.asarray(actual_axes) @ hand.axes, atol=1e-12)
        args = hand, chain, pads, normals, solver.last, .1
        for phalanx, thumb_span in ((False, False), (True, False), (True, True)):
            analytic, finite = [shape.objective(*args, opposition=bool(i % 2), gradient=mode,
                                                phalanx=phalanx, thumb_span=thumb_span)
                                for mode in ('analytic', 'finite_difference')]
            actual, expected = np.empty_like(q), np.empty_like(q)
            np.testing.assert_allclose(analytic(q, actual), finite(q, expected), rtol=1e-13)
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-6)

        analytic, finite = [shape.direction_objective(hand, chain, solver.last, gradient=mode)
                            for mode in ('analytic', 'finite_difference')]
        actual, expected = np.empty_like(q), np.empty_like(q)
        np.testing.assert_allclose(analytic(q, actual), finite(q, expected), rtol=1e-13)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-6)

        for factory in (lambda mode: shape.direction_objective(
                hand, chain, solver.last, gradient=mode, thumb_base_only=True),
                lambda mode: shape.thumb_base_objective(hand, chain, solver.last, gradient=mode),
                lambda mode: shape.thumb_base_objective(hand, chain, solver.last, gradient=mode, opponents=pads[1:])):
            analytic_base, finite_base = [factory(mode) for mode in ('analytic', 'finite_difference')]
            np.testing.assert_allclose(analytic_base(q, actual), finite_base(q, expected), rtol=1e-13)
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-6)

        # Moving roots and changing individual bone lengths must not change a
        # direction objective. This includes thumb-to-finger distances.
        bones = np.diff(chain, axis=1)
        roots = chain[:, :1] + rng.uniform(-.04, .04, (5, 1, 3))
        resized = np.concatenate((roots, roots + np.cumsum(
            bones * rng.uniform(.5, 1.8, (5, 3, 1)), axis=1)), axis=1)
        invariant = shape.direction_objective(hand, resized, solver.last, gradient='analytic')
        shifted_grad = np.empty_like(q)
        np.testing.assert_allclose(invariant(q, shifted_grad), analytic(q, actual), atol=1e-10)
        np.testing.assert_allclose(shifted_grad, actual, atol=1e-9)

        loss = hand.thumb_refinement_loss(chain, normals[0])
        hand.states(q)
        shape.mj.mj_comPos(hand.model, hand.data)
        actual = np.zeros_like(q)
        loss(hand.data, actual)
        slopes = []
        for step in 1e-5 * np.eye(len(q)):
            hand.states(q + step)
            plus = loss(hand.data, np.empty(0))
            hand.states(q - step)
            minus = loss(hand.data, np.empty(0))
            slopes.append((plus - minus) / 2e-5)
        np.testing.assert_allclose(actual, slopes, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('gesture', ['straight', 'curled', 'sideways'])
def test_direction_mode_tracks_fingers_without_distance_targets(shape, monkeypatch, side, gesture):
    from botyard_live import BotyardRetargeter
    from ament_index_python.packages import PackageNotFoundError
    try:
        share = Path(shape.get_package_share_directory('dexhand_description'))
    except PackageNotFoundError:
        pytest.skip('GS S1 description is not installed')
    assets = Path(__file__).resolve().parents[2] / 'tendon-spin/assets'
    if not assets.is_dir():
        pytest.skip('GS S1 MuJoCo assets are not installed')
    # A direction solve uses joint observations and needs no MANO mesh model.
    monkeypatch.setattr('botyard_live.OnlineMano', lambda path, **kwargs: None)
    meshes = share / 'meshes/gs_s1'
    seq, _ = shape.build_solver(hand=side, objective='botyard')
    retargeter = BotyardRetargeter(seq, shape.load_hand(assets, meshes, side), meshes, side,
                                   None, 'off', shape_model='direction', gradient='analytic')
    target = retargeter.reference.last.copy()
    names = list(retargeter.reference.opt.target_joint_names)
    spread = retargeter.hand.spread_indices
    if gesture == 'curled':
        for finger in ('FF', 'MF', 'RF', 'LF'):
            target[names.index(finger + 'J3')] = np.deg2rad(30)
            target[names.index(finger + 'J2')] = np.deg2rad(20)
    elif gesture == 'sideways':
        target[spread] = np.deg2rad(5)
    neutral = retargeter.hand.states(np.zeros_like(target))[0][0]
    source = retargeter.hand.states(target)[0][0]
    # Preserve the observed angles while independently changing roots/bone
    # lengths, so copying absolute inter-finger vectors would be incorrect.
    lengths = np.array([[.7, 1.3, .8], [.65, .8, .6], [.9, .7, 1.1],
                        [.75, 1.2, .7], [.5, .6, .7]])[..., None]
    bones = np.diff(source, axis=1) * lengths
    chains = np.concatenate((neutral[:, :1], neutral[:, :1] + np.cumsum(bones, axis=1)), axis=1)
    joints = np.vstack((np.zeros(3), chains.reshape(-1, 3)))
    camera = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    observed = 1.4 * joints @ camera + [.3, -.2, 1.1]
    # Start with a visible lateral error to test convergence back to straight.
    retargeter.reference.last[spread] = np.deg2rad(-3)
    assert retargeter.reference.geom.feasible(retargeter.reference.last)
    for i in range(12):
        trace = {}
        retargeter.solve_mano(observed, None, .1 * i, .1, trace=trace)
    np.testing.assert_allclose(retargeter.output.last[spread], target[spread], atol=np.deg2rad(.3))
    if gesture == 'curled':
        flexion = [names.index(f + suffix) for f in ('FF', 'MF', 'RF', 'LF') for suffix in ('J3', 'J2')]
        np.testing.assert_allclose(retargeter.output.last[flexion], target[flexion], atol=np.deg2rad(1.))
    assert trace['target_distal_gaps_m'] is None


@pytest.mark.parametrize('side', ['left', 'right'])
def test_thumb_base_tracks_heading_with_fixed_bends_and_no_crossing_rule(shape, monkeypatch, side):
    from botyard_live import BotyardRetargeter
    from ament_index_python.packages import PackageNotFoundError
    try:
        share = Path(shape.get_package_share_directory('dexhand_description'))
    except PackageNotFoundError:
        pytest.skip('GS S1 description is not installed')
    assets = Path(__file__).resolve().parents[2] / 'tendon-spin/assets'
    if not assets.is_dir():
        pytest.skip('GS S1 MuJoCo assets are not installed')
    monkeypatch.setattr('botyard_live.OnlineMano', lambda path, **kwargs: None)
    monkeypatch.setattr(shape, 'THUMB_PAD_WEIGHT', 0.)  # isolate the exact inverse-kinematics round trip
    meshes = share / 'meshes/gs_s1'
    seq, _ = shape.build_solver(hand=side, objective='botyard')
    ret = BotyardRetargeter(seq, shape.load_hand(assets, meshes, side), meshes, side,
                            None, 'off', shape_model='sweep', gradient='analytic')
    names = list(ret.output.opt.target_joint_names)
    base = [names.index(n) for n in ('THJ4', 'THJ3')]
    flexion = [names.index(n) for n in ('THJ2', 'THJ1')]
    fixed = [i for i in range(len(names)) if i not in base]
    clock = 0.
    for angle in (-20., 0., 20.):
        target = np.zeros(len(names))
        target[base] = np.deg2rad([angle, 25.])
        target[flexion] = np.deg2rad([20., 15.])
        source = ret.hand.states(target)[0][0]
        # Human and robot lengths can differ independently along the chain.
        lengths = np.array([.7, 1.3, .8])[None, :, None]
        chains = np.concatenate((source[:, :1], source[:, :1] + np.cumsum(
            np.diff(source, axis=1) * lengths, axis=1)), axis=1)
        observed = np.vstack((np.zeros(3), chains.reshape(-1, 3)))
        for _ in range(15):
            ret.solve_mano(observed, None, clock, .1)
            clock += .1
        np.testing.assert_allclose(ret.output.last[base], target[base], atol=np.deg2rad(2.))
        np.testing.assert_allclose(ret.output.last[flexion], target[flexion], atol=np.deg2rad(.7))
        np.testing.assert_allclose(ret.output.last[fixed], ret.reference.last[fixed], atol=1e-12)
        np.testing.assert_allclose(ret.output.last[ret.hand.spread_indices], 0, atol=np.deg2rad(.3))
    previous = ret.output.last.copy()
    # Moving the entire thumb across the index axis leaves its intrinsic pose
    # unchanged and must not trigger an extra rotation or a different branch.
    for sign in (-1., 1.):
        moved = observed.copy()
        moved[1:5] += sign * .15 * ret.hand.radial_axis
        ret.solve_mano(moved, None, clock, .1)
        clock += .1
        np.testing.assert_allclose(ret.output.last, previous, atol=np.deg2rad(.5))

    def facing_hand():
        _, pads, normals = ret.hand.states(ret.output.last)
        toward = pads[0, 1:].mean(axis=0) - pads[0, 0]
        toward /= np.linalg.norm(toward)
        return np.arccos(np.clip(np.dot(normals[0, 0], toward), -1., 1.))

    before = facing_hand()
    monkeypatch.setattr(shape, 'THUMB_PAD_WEIGHT', .2)
    for _ in range(15):
        ret.solve_mano(observed, None, clock, .1)
        clock += .1
    assert facing_hand() < before - np.deg2rad(.1)
    np.testing.assert_allclose(ret.output.last[flexion], target[flexion], atol=np.deg2rad(.7))
    np.testing.assert_allclose(ret.output.last[fixed], ret.reference.last[fixed], atol=1e-12)

    fallback = ret.output.last.copy()
    fallback[flexion[0]] += np.deg2rad(1.)
    assert ret.output.geom.feasible(fallback)

    def failed_refinement(x, grad):
        raise RuntimeError('Exercise a failed base-only optimization')

    count = ret.output.fallbacks
    ret.output.solve_objective(failed_refinement, .1, initial=fallback,
                                fixed={i: fallback[i] for i in fixed})
    np.testing.assert_allclose(ret.output.last, fallback, atol=1e-12)
    assert ret.output.fallbacks == count + 1


def test_thumb_fit_handles_shorter_observed_bones_without_losing_visible_curl(shape):
    from botyard_live import OnlineMano
    from fit_example_mano import ORDER
    from gs_s1_mano_overlay import palm_frame
    template = Path.home() / '.cache/botyard-retargeting/mano/right.npz'
    if not template.is_file():
        pytest.skip('Licensed MANO template is not prepared')
    model = OnlineMano(template, thumb_direction_fit=True)
    keypoint_model = OnlineMano(template)
    joints, tips = model.skin(np.zeros((1, 45)), np.eye(3), model.tips)
    neutral = np.concatenate((joints, tips), axis=1)[0, ORDER]
    rotation = palm_frame(*neutral[[0, 5, 9, 17]])[1]
    observed = neutral @ rotation
    lengths = np.linalg.norm(np.diff(observed[1:5], axis=0), axis=-1)
    direction = observed[2] - observed[1]
    direction /= np.linalg.norm(direction)
    # The measured thumb is straight and its distal phalanx is shorter than
    # the fixed template. An absolute endpoint fit can hide the excess in depth.
    observed[2:5] = observed[1] + np.cumsum(lengths * [1., 1., .65])[:, None] * direction
    camera = observed + [0., 0., .6]
    pixels = 640 * camera[:, :2] / camera[:, 2:] + [320., 240.]
    for i in range(12):
        fitted, _ = model.fit(observed, pixels, [640, 480], i * .1)
        keypoint_fitted, _ = keypoint_model.fit(observed, pixels, [640, 480], i * .1)
    straight_bend = np.rad2deg(shape.bone_geometry(fitted[1:5])[4])
    # The final implementation refines only the IP rotation, preserving the
    # keypoint fit's proximal chain. A bound on ALL intrinsic thumb bends was
    # left over from the rejected full-thumb fit and conflicts with that scope.
    # Verify the visible distal direction and preservation of other landmarks.
    def distal_error(points):
        ray = points[4] - points[3]
        ray /= np.linalg.norm(ray)
        return np.rad2deg(np.arccos(np.clip(np.dot(ray, direction), -1., 1.)))
    assert distal_error(fitted) < 5.
    assert distal_error(fitted) < .5 * distal_error(keypoint_fitted)
    np.testing.assert_allclose(fitted[np.arange(21) != 4],
                               keypoint_fitted[np.arange(21) != 4], atol=1e-6, rtol=0.)

    pose = np.zeros((1, 45))
    pose[0, -3] = 1.5
    joints, tips = model.skin(pose, np.eye(3), model.tips)
    observed = np.concatenate((joints, tips), axis=1)[0, ORDER]
    bones = shape.bone_geometry(observed[1:5])[0]
    normal = np.cross(bones[1], bones[2])
    normal /= np.linalg.norm(normal)
    # Turn the flexion plane toward the camera: this bend has image evidence.
    rotation = np.stack((bones[1], np.cross(normal, bones[1]), normal))
    observed = observed @ rotation.T
    camera = observed + [0., 0., .6]
    pixels = 640 * camera[:, :2] / camera[:, 2:] + [320., 240.]
    for i in range(12, 24):
        fitted, _ = model.fit(observed, pixels, [640, 480], i * .1)
    curled_bend = np.rad2deg(shape.bone_geometry(fitted[1:5])[4])
    assert curled_bend[1] > 30.
    assert curled_bend[1] > straight_bend[1] + 20.
