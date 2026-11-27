"""Palm registration must work for both handednesses and arbitrary camera axes."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest


@pytest.mark.parametrize('sign', [-1, 1])
def test_mano_overlay_registration(monkeypatch, tmp_path, sign):
    mj = pytest.importorskip('mujoco')
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'example/vector_retargeting'))
    from gs_s1_mano_overlay import ManoOverlay, add_overlay_asset, encode_pose

    scene = ET.fromstring('<mujoco><asset/><worldbody><body name="palm"/></worldbody></mujoco>')
    palm = scene.find('.//body')
    joints = np.zeros((21, 3))
    joints[[4, 5, 9, 17]] = [[sign * .03, .07, .09], [sign * .035, 0, .08],
                              [0, 0, .09], [-sign * .035, 0, .07]]
    for name, point in zip(('thtip', 'ffproximal', 'mfproximal', 'lfproximal'), joints[[4, 5, 9, 17]]):
        ET.SubElement(palm, 'body', name=name, pos=' '.join(map(str, point)))
    vertices = joints[[0, 4, 5, 17]]
    template = tmp_path / 'test.npz'
    np.savez(template, v_template=vertices, f=[[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    add_overlay_asset(scene, template)
    model = mj.MjModel.from_xml_string(ET.tostring(scene, encoding='unicode'))
    assert model.ngeom == 0
    overlay = ManoOverlay(model)
    # Human observation in camera coordinates, at a different hand size.
    camera_rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    observed = 1.4 * joints @ camera_rotation + [.3, -.2, 1.1]
    overlay.update_vertices(encode_pose(observed, observed[[0, 4, 5, 17]]))
    np.testing.assert_allclose(overlay.vertices, vertices, atol=1e-7)
    np.testing.assert_allclose(overlay.mano_thumb, joints[4], atol=1e-7)
