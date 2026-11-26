from unittest.mock import Mock

import numpy as np

from example.vector_retargeting.gs_s1_calibration import FingerRootCalibration


def test_short_human_pinky_maps_to_extended_robot_then_curls():
    # Model FK deliberately uses longer phalanges than the observed hand.
    links = {}
    for x, digit in zip((-.035, -.012, .012, .035), ("ff", "mf", "rf", "lf")):
        for z, link in zip((.10, .145, .170, .197), ("proximal", "middle", "distal", "tip")):
            pose = np.eye(4)
            pose[:3, 3] = [x, .003 if link == "tip" else 0., z]
            links[digit + link] = pose
    robot = Mock()
    robot.get_link_index.side_effect = lambda name: name
    robot.get_link_pose_inv.return_value = np.eye(4)
    robot.get_link_pose.side_effect = links.__getitem__
    calibration = FingerRootCalibration(robot)
    human = np.zeros((21, 3))
    for k in calibration.knuckles:
        human[k:k+4, 0] = .001 * k
        human[k:k+4, 2] = [.08, .106, .125, .144]

    calibration.start()
    for _ in range(calibration.sample_count):
        calibration.observe(human)
    extended = calibration.apply(human)
    for j, link in enumerate(("lfproximal", "lfmiddle", "lfdistal", "lftip"), 17):
        np.testing.assert_allclose(extended[j], links[link][:3, 3], atol=1e-12)

    bent = human.copy()
    bent[19] = bent[18] + [0., .019, 0.]
    bent[20] = bent[19] + [0., 0., -.019]
    mapped = calibration.apply(bent)
    np.testing.assert_allclose(mapped[:19], extended[:19])
    np.testing.assert_allclose(mapped[19] - mapped[18], [0., .025, 0.], atol=1e-12)
    np.testing.assert_allclose(np.diff(mapped[19:], axis=0), -np.diff(extended[19:], axis=0))
    calibration.toggle()
    np.testing.assert_array_equal(calibration.apply(bent), bent)
