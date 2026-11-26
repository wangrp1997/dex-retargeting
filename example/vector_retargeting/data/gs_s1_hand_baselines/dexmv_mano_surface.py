"""Rebuild the private local MANO surface from a DexMV sequence; do not redistribute MANO weights."""
import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from fit_example_mano import distal_face_labels


def reconstruct(model_path, sequence, output):
    with model_path.open("rb") as f:
        model = pickle.load(f, encoding="latin1")
    ids = np.arange(11, 123)
    params = np.stack([np.load(sequence / f"mano_{i}.npy") for i in ids])
    joints = np.stack([np.load(sequence / f"joints_{i}.npy") for i in ids])
    frames = np.stack([np.load(sequence / f"results_global_{i}.npy") for i in ids])
    # Convert DexMV thumb-first global frames to MANO's native joint order.
    frames = frames[:, [0,4,5,6,7,8,9,13,14,15,10,11,12,1,2,3]]
    absolute, centers = frames[:, :, :3, :3], frames[:, :, :3, 3]
    parent = model["kintree_table"][0, 1:].astype(int)
    local = absolute[:, parent].transpose(0,1,3,2) @ absolute[:, 1:]
    expected = Rotation.from_rotvec(params[:,3:48].reshape(-1,3)).as_matrix()
    np.testing.assert_allclose(expected, local.reshape(-1,3,3), atol=1e-5)
    np.testing.assert_allclose(params[:,58:61], joints[:,0], atol=1e-6)
    shaped = model["v_template"] + np.einsum("vck,bk->bvc", np.asarray(model["shapedirs"]), params[:,48:58])
    rest_joints = np.stack([model["J_regressor"] @ x for x in shaped])
    posed = shaped + np.einsum("vck,bk->bvc", model["posedirs"], (local-np.eye(3)).reshape(len(ids),135))
    translation = centers - np.einsum("bjxy,bjy->bjx",absolute,rest_joints)
    rotation = np.einsum("vj,bjxy->bvxy",model["weights"],absolute)
    vertices = np.einsum("bvxy,bvy->bvx",rotation,posed) + np.einsum("vj,bjx->bvx",model["weights"],translation)
    vertex_ids = [728, 353, 442, 576, 694]  # DexMV tip vertices, checked against all source frames
    tip_err = np.linalg.norm(vertices[:,vertex_ids] - joints[:,4::4],axis=-1)
    assert np.max(tip_err) < 0.0001
    thumb = np.argmax(model["weights"], axis=1) >= 13
    faces = model["f"].astype(int)
    np.savez_compressed(output,frame=ids,vertices=vertices.astype("f4"),faces=faces,
                        thumb_faces=faces[thumb[faces].all(axis=1)],joints=joints,
                        distal_face_labels=distal_face_labels(faces,np.asarray(model["weights"])))
    print(f"DexMV fingertip agreement, maximum over all frames: {1000*tip_err.max():.4f} mm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path",type=Path)
    parser.add_argument("sequence",type=Path)
    parser.add_argument("output",type=Path)
    reconstruct(**vars(parser.parse_args()))
