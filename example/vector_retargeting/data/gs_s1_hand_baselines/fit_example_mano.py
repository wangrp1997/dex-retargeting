"""Fit right MANO to the official RGB hand example. Intermediate arrays stay local."""
import argparse
from pathlib import Path

import numpy as np

TIPS = [728, 353, 442, 576, 694]
ORDER = [0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20]

def distal_face_labels(faces, weights):
    """Assign MANO distal triangles by majority of dominant skinning joints."""
    joints = np.argmax(weights, axis=1)[faces]
    labels = np.full(len(faces), -1, dtype=np.int8)
    for digit, distal_joint in enumerate((15, 3, 6, 12, 9)):
        labels[np.sum(joints == distal_joint, axis=1) >= 2] = digit
    return labels


def prepare(model, output):
    # Use an environment with chumpy for the original, user-owned MANO pickle.
    import pickle
    with model.open('rb') as f:
        m = pickle.load(f, encoding='latin1')
    keys = ('v_template','f','hands_components','hands_mean','weights','posedirs','shapedirs','kintree_table')
    arrays = {k:np.asarray(m[k]) for k in keys}
    np.savez(output, J_regressor=m['J_regressor'].toarray(), **arrays)


def detect(video, output):
    import cv2
    from single_hand_detector import SingleHandDetector, OPERATOR2MANO_RIGHT
    reader = cv2.VideoCapture(str(video))
    detector = SingleHandDetector('Right')
    world, pixels = [], []
    fps = reader.get(cv2.CAP_PROP_FPS)
    size = [reader.get(cv2.CAP_PROP_FRAME_WIDTH), reader.get(cv2.CAP_PROP_FRAME_HEIGHT)]
    try:
        while True:
            ok, img = reader.read()
            if not ok:
                break
            count, p, image_points, r = detector.detect(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            if count:
                world.append(p @ OPERATOR2MANO_RIGHT.T @ r.T)
                pixels.append(detector.parse_keypoint_2d(image_points,img.shape))
            else:
                world.append(np.full((21,3), np.nan))
                pixels.append(np.full((21,2), np.nan))
    finally:
        reader.release()
        detector.hand_detector.close()
    if not world:
        raise ValueError('Empty or unreadable video')
    np.savez_compressed(output,world=world,pixels=pixels,fps=fps,size=size)


def fit(model, observations, output, smplx_root=None):
    import sys
    if smplx_root:
        sys.path.append(str(smplx_root))
    import torch
    from smplx.lbs import batch_rigid_transform, batch_rodrigues
    from scipy.spatial.transform import Rotation
    torch.set_num_threads(2)
    data, obs = np.load(model), np.load(observations)
    valid = np.flatnonzero(np.isfinite(obs['pixels']).all((1,2)) & np.isfinite(obs['world']).all((1,2)))
    if not len(valid):
        raise ValueError('No detected right hand')
    def tensor(a):
        return torch.as_tensor(a,dtype=torch.float32)
    observed = tensor(obs['world'][valid])
    pixels = tensor(obs['pixels'][valid])
    size = tensor(obs['size'])
    focal = float(size.max())
    center = size/2
    B = len(valid)
    template = tensor(data['v_template'])
    shape_dirs = tensor(data['shapedirs'])
    pose_dirs = tensor(data['posedirs'])
    weights = tensor(data['weights'])
    regressor = tensor(data['J_regressor'])
    parents = torch.as_tensor(data['kintree_table'][0].copy(),dtype=torch.long)
    parents[0] = -1
    components = tensor(data['hands_components'])
    mean_pose = tensor(data['hands_mean'])
    eye = torch.eye(3)
    base_joints = regressor @ template
    # Initial proper rotation from four knuckles to the observed camera axes.
    base = (base_joints-base_joints[:1])[[1,4,10,7]].numpy()
    obs_palm = observed[:,[5,9,13,17]].numpy()
    u, _, vt = np.linalg.svd(base.T @ obs_palm)
    correction = np.broadcast_to(np.eye(3),(B,3,3)).copy()
    correction[:,-1,-1] = np.linalg.det(u @ vt)
    root0 = tensor((u @ correction @ vt).transpose(0,2,1).copy())
    image_delta = pixels - pixels[:,:1]
    slope = (image_delta*observed[:,:,:2]).sum((1,2))/observed[:,:,:2].square().sum((1,2))
    initial_depth = (focal/slope).clamp(.15,3.)
    root_delta = torch.nn.Parameter(torch.zeros(B,3))
    coeff = torch.nn.Parameter(torch.zeros(B,45))
    beta = torch.nn.Parameter(torch.zeros(10))
    log_scale = torch.nn.Parameter(torch.zeros(()))
    depth_delta = torch.nn.Parameter(torch.zeros(B))
    wrist_offset = torch.nn.Parameter(torch.zeros(B,2))
    adjacent = tensor(np.diff(valid)==1).bool()
    wrist_px = pixels[:,0]

    def forward(frames, indices):
        bs = len(frames)
        shaped = template + shape_dirs @ beta
        rest = (regressor @ shaped).expand(bs,-1,-1)
        local = (mean_pose + coeff[frames] @ components).reshape(bs,15,3)
        local_rot = batch_rodrigues(local.reshape(-1,3)).reshape(bs,15,3,3)
        root = batch_rodrigues(root_delta[frames]) @ root0[frames]
        rotations = torch.cat([root[:,None],local_rot], dim=1)
        j, transforms = batch_rigid_transform(rotations,rest,parents)
        posed = shaped[indices] + ((local_rot-eye).reshape(bs,135) @ pose_dirs[indices].reshape(-1,135).T).reshape(bs,-1,3)
        transforms = (weights[indices] @ transforms.reshape(bs,16,16)).reshape(bs,-1,4,4)
        v = (transforms[:,:,:3,:3] @ posed.unsqueeze(-1)).squeeze(-1) + transforms[:,:,:3,3]
        root_joint = j[:,:1]
        scale = torch.exp(log_scale)
        return (v-root_joint)*scale,(j-root_joint)*scale,local,root

    def camera(frames):
        depth = initial_depth[frames] * torch.exp(depth_delta[frames])
        xy = (wrist_px[frames] + 40*wrist_offset[frames] - center) * (depth/focal)[:,None]
        return torch.cat([xy,depth[:,None]],axis=1)

    def project(x, t):
        p=x+t[:,None]
        return focal*p[:,:,:2]/p[:,:,2:].clamp_min(.05)+center

    def robust(x):
        return (torch.sqrt(1+x.square())-1).mean()

    all_frames = torch.arange(B)
    opt = torch.optim.Adam([dict(params=[coeff],lr=.09),dict(params=[root_delta,depth_delta,wrist_offset],lr=.016),
                             dict(params=[beta,log_scale],lr=.008)])
    for step in range(240):
        if step in (120,195):
            for group in opt.param_groups:
                group['lr'] *= .45
        opt.zero_grad()
        tips, joints, local, root = forward(all_frames,TIPS)
        pred = torch.cat([joints,tips],axis=1)[:,ORDER]
        reproj = project(pred,camera(all_frames))
        loss = robust((pred-observed)/.006) + 1.4*robust((reproj-pixels)/4.)
        loss += .018*coeff.square().mean() + .025*beta.square().mean() + 1.5*log_scale.square()
        if adjacent.any():
            loss += .12*(coeff[1:]-coeff[:-1])[adjacent].square().mean()
            loss += .07*(depth_delta[1:]-depth_delta[:-1])[adjacent].square().mean()
        loss.backward()
        opt.step()
        if step in (0,60,120,239):
            print(f'fit iteration {step}, loss {loss.detach().item():.3f}',flush=True)
    vertices=[]; landmarks=[]; poses=[]; rotations=[]; translations=[]; projections=[]
    with torch.no_grad():
        for subset in all_frames.split(40):
            v,j,local,r = forward(subset,slice(None))
            joints = torch.cat([j,v[:,TIPS]],dim=1)[:,ORDER]
            t = camera(subset)
            for dst,value in zip((vertices,landmarks,poses,rotations,translations,projections),
                                 (v,joints,local,r,t,project(joints,t))):
                dst.append(value.numpy())
        scale=float(torch.exp(log_scale))
    vv,jj,aa,rr,tt,uv=map(np.concatenate,(vertices,landmarks,poses,rotations,translations,projections))
    e2=np.linalg.norm(uv-pixels.numpy(),axis=2)
    e3=1000*np.linalg.norm(jj-observed.numpy(),axis=2)
    assert np.isfinite(vv).all() and np.all(vv[:,:,2]+tt[:,None,2] > 0)
    np.savez_compressed(output,frame=valid,fps=obs['fps'],total_frames=len(obs['world']),size=obs['size'],
                        vertices_camera_wrist_m=vv,joints_camera_wrist_m=jj,faces=data['f'],image_joints_px=uv,
                        distal_face_labels=distal_face_labels(data['f'],data['weights']),
                        hand_pose_rotvec=aa.reshape(-1,45),global_rotvec=Rotation.from_matrix(rr).as_rotvec(),
                        betas=beta.detach().numpy(),hand_scale=scale,camera_wrist_translation_m=tt,focal_px=focal,
                        reprojection_to_detector_px=e2,landmark_to_detector_mm=e3)
    print('frames, reproj median/mean/p95 px, mean 3D mm, scale',len(valid),np.median(e2),e2.mean(),np.percentile(e2,95),e3.mean(),scale,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    for name,keys in [('prepare',('model','output')),('detect',('video','output')),('fit',('model','observations','output'))]:
        p=sub.add_parser(name)
        for key in keys:
            p.add_argument(key,type=Path)
        if name=='fit':
            p.add_argument('--smplx-root',type=Path)
    a=vars(parser.parse_args())
    {'prepare':prepare,'detect':detect,'fit':fit}[a.pop('action')](**a)
