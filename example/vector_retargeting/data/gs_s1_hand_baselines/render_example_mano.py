"""Side-by-side RGB, projected MANO mesh and same-view GS S1 in MuJoCo."""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from gs_s1_mujoco import load_hand
from gs_s1_rviz import get_package_share_directory

SIZE = 400


def load_sweep(path, gap_target, frames, controls_shape, gaps_shape):
    report = json.loads(path.with_suffix('.json').read_text())
    if (report['weight'], report['base_cost'], report['gap_target'], report['carry_offset'], report['mcp_flex'],
        report.get('motion_weight', 0.)) != (.70, .006, gap_target, True, True, 0.):
        raise ValueError(f'Expected the {gap_target} sweep configuration for {path}')
    with np.load(path) as candidate:
        np.testing.assert_array_equal(frames, candidate['frame'])
        controls = candidate['active_qpos_rad']
        gaps = candidate['robot_distal_gaps_mm']
        if 'solver_held' in candidate:
            held = candidate['solver_held']
        elif report['held'] == 0:
            held = np.zeros(len(frames), dtype=bool)
        else:
            raise ValueError('The candidate has solver holds but lacks per-frame hold flags')
    if controls.shape != controls_shape or gaps.shape != gaps_shape or held.shape != frames.shape:
        raise ValueError('Sweep and reference trajectory dimensions differ')
    return controls, gaps, held


def render(video, mano, robot, output, observations, preview=None, show_pinky_gap=False, compare_v2=False, compare_v3=None, compare_v32=None, compare_v33=None):
    m, q, obs = np.load(mano), np.load(robot), np.load(observations)
    frames=m['frame']; fps=float(m['fps'])
    total_frames = int(m['total_frames'])
    image_points = obs['pixels']
    held = q['solver_held']
    active_track = q['active_qpos_rad']
    np.testing.assert_array_equal(frames,q['frame'])
    comparing = compare_v2 or compare_v3 is not None or compare_v32 is not None or compare_v33 is not None
    if comparing:
        required = ('v2_active_qpos_rad','v2_distal_gaps_mm','robot_distal_gaps_mm','human_distal_gaps_mm')
        if any(key not in q for key in required):
            raise ValueError('Comparison requires trajectories with distal gap measurements')
        reference_track = q['v2_active_qpos_rad']
        reference_gaps = q['v2_distal_gaps_mm']
        selected_gaps = q['robot_distal_gaps_mm']
        reference_label = 'GS S1 · 第二版'
        selected_label = 'GS S1 · v3.1（跟踪人手间隙）' if 'target_distal_gaps_mm' in q else 'GS S1 · v3（连续闭合辅助）'
        if compare_v3 is not None:
            if 'target_distal_gaps_mm' not in q:
                raise ValueError('--compare-v3 expects the selected trajectory to be v3.1')
            with np.load(compare_v3) as baseline:
                if 'target_distal_gaps_mm' in baseline:
                    raise ValueError('Provide a v3 baseline with --compare-v3')
                for key in ('frame','fps','joint_names','camera_to_palm_row','v2_active_qpos_rad','human_distal_gaps_mm'):
                    np.testing.assert_array_equal(q[key],baseline[key])
                reference_track = baseline['active_qpos_rad']
                reference_gaps = baseline['robot_distal_gaps_mm']
            reference_label = 'GS S1 · v3（连续闭合辅助）'
        elif compare_v32 is not None or compare_v33 is not None:
            if 'target_distal_gaps_mm' in q:
                raise ValueError('Provide the original v3 track as the reference')
            reference_track, reference_gaps = active_track, selected_gaps
            reference_label = 'GS S1 · v3（连续闭合辅助）'
            selected_label = 'GS S1 · v3.2（指根调整与间隙压缩）'
            v32_path = compare_v32 if compare_v33 is None else compare_v33[0]
            active_track, selected_gaps, held = load_sweep(v32_path,'compressed',frames,
                                                           reference_track.shape,reference_gaps.shape)
            if compare_v33 is not None:
                reference_track, reference_gaps, reference_label = active_track, selected_gaps, selected_label
                selected_label = 'GS S1 · v3.3（按掌宽跟踪间隙）'
                active_track, selected_gaps, held = load_sweep(compare_v33[1],'scaled',frames,
                                                               reference_track.shape,reference_gaps.shape)
    v=m['vertices_camera_wrist_m'] + m['camera_wrist_translation_m'][:,None]
    px=v[:,:,:2]/v[:,:,2:]*m['focal_px'] + m['size']/2
    lo=px.min(axis=(0,1)); hi=px.max(axis=(0,1))
    side=int(np.ceil((hi-lo).max()*1.21))
    x0,y0=np.clip(np.rint((lo+hi-side)*.5),0,m['size']-side).astype(int)
    def crop_uv(uv):
        return np.rint((uv-[x0,y0])*SIZE/side).astype(np.int32)
    faces=m['faces'].astype(int)
    mesh_dir=Path(get_package_share_directory('dexhand_description'))/'meshes/gs_s1'
    model=load_hand(Path.home()/'Projects/tendon-spin/assets',mesh_dir,'right')
    data=mj.MjData(model)
    model.vis.global_.offwidth=SIZE
    model.vis.global_.offheight=SIZE
    model.vis.quality.offsamples=2
    model.vis.headlight.ambient[:]=.45
    for g, body in enumerate(model.geom_bodyid):
        name=model.body(body).name
        model.geom_group[g]=1 if name=='palm' or name[:2] in ('th','ff','mf','rf','lf') else 5
        model.geom_rgba[g]=[.95,.52,.17,1] if name.startswith('th') else [.84,.72,.51,1]
    renderer=mj.Renderer(model,height=SIZE,width=SIZE)
    opt=mj.MjvOption()
    opt.geomgroup[:]=0; opt.geomgroup[1]=1; opt.sitegroup[:]=0
    mj.mj_forward(model,data)
    pid=model.body('palm').id
    axes=data.xmat[pid].reshape(3,3).copy()
    focus=data.xpos[pid]+axes@np.array([0.,.018,.081])
    adr=np.array([model.joint(n).qposadr.item() for n in q['joint_names']])
    passive=np.array([model.joint(d+'J1').qposadr.item() for d in ('FF','MF','RF','LF')])
    active=np.array([model.joint(d+'J2').qposadr.item() for d in ('FF','MF','RF','LF')])
    camera_track=q['camera_to_palm_row']
    # A common distance frames both complete GS S1 hands through this clip.
    camera_distance = .38 if comparing else .315
    def sim(i, controls):
        data.qpos[adr]=controls[i]
        data.qpos[passive]=data.qpos[active]
        mj.mj_forward(model,data)
        renderer.update_scene(data,scene_option=opt)
        # CV camera +X right, +Y down, +Z into image. MuJoCo free camera
        # orientation is overridden after update, using the same palm rotation.
        camera_axes=axes @ camera_track[i].T
        forward=camera_axes[:,2]
        up=-camera_axes[:,1]
        for cam in renderer.scene.camera:
            cam.pos[:]=focus-forward*camera_distance
            cam.forward[:]=forward
            cam.up[:]=up
        return renderer.render().copy()
    def overlay(i,background):
        a,b,c=v[i,faces].transpose(1,0,2)
        n=np.cross(b-a,c-a)
        n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-10)
        normals_light=n@np.array([-.30,-.43,-.85])
        shades=.56+.42*np.maximum(normals_light,0)
        colors=np.clip(shades[:,None]*[66,194,226],0,255).astype(np.uint8)
        points=crop_uv(px[i])[faces]
        result=background.copy()
        depth=(a[:,2]+b[:,2]+c[:,2])
        for k in np.argsort(-depth):
            cv2.fillConvexPoly(result,points[k],tuple(map(int,colors[k])),lineType=cv2.LINE_AA)
        return cv2.addWeighted(result,.75,background,.25,0)
    def skeleton(i,raw):
        panel=raw.copy()
        uv=crop_uv(image_points[int(frames[i])])
        for first in [1,5,9,13,17]:
            cv2.polylines(panel,[uv[[0,first,first+1,first+2,first+3]]],False,(62,222,177),1,cv2.LINE_AA)
        for pt in uv: cv2.circle(panel,pt,2,(50,238,181),-1,cv2.LINE_AA)
        return cv2.addWeighted(panel,.8,raw,.2,0)
    gap_tracks = [q['human_distal_gaps_mm'],reference_gaps,selected_gaps] if comparing else []
    basename=Image.new('RGB',(1630,540) if comparing else (1224,478),(23,30,37))
    fontpath='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    big=ImageFont.truetype(fontpath,23,index=2); small=ImageFont.truetype(fontpath,18,index=2)
    ui=ImageDraw.Draw(basename)
    ui.text((8,3),'示例视频右手  /  MANO 拟合与 GS S1 重定向',font=big,fill='white')
    offsets=(6,412,818,1224) if comparing else (6,412,818)
    robot_labels = (reference_label,selected_label) if comparing else ('MuJoCo · GS S1 右手（同向视角）',)
    for x,label in zip(offsets,('原视频 · 检测关键点','同一画面 · MANO 表面投影',*robot_labels)):
        ui.text((x,46),label,font=small,fill='#dcecf0')
    if comparing:
        ui.text((8,488),'拇指到其他指的末节表面间隙（mm）',font=small,fill='#dcecf0')
    lookup={int(f):i for i,f in enumerate(frames)}
    def panel(frame,rgb):
        img=basename.copy()
        raw=cv2.resize(rgb[y0:y0+side,x0:x0+side],(SIZE,SIZE),interpolation=cv2.INTER_AREA)
        if frame in lookup:
            i=lookup[frame]
            panels=[skeleton(i,raw),overlay(i,raw)]
            if comparing:
                panels.append(sim(i,reference_track))
            panels.append(sim(i,active_track))
            for x,p in zip(offsets,panels):
                img.paste(Image.fromarray(p),(x,74))
            if comparing:
                draw = ImageDraw.Draw(img)
                for x,gaps in zip(offsets[1:],gap_tracks):
                    line = '   '.join(f'{name} {gap:.1f}' for name,gap in zip(('食','中','无名','小'),gaps[i]))
                    draw.text((x,488),line,font=small,fill='#c1eff0')
            if show_pinky_gap:
                draw=ImageDraw.Draw(img)
                for x,key in zip(offsets[1:],('human_pad_center_gaps_mm','robot_pad_center_gaps_mm')):
                    gap=q[key][i,-1]
                    draw.rectangle((x+4,78,x+246,108),fill='#26333e')
                    draw.text((x+9,80),f'拇指–小指指腹中心距  {gap:.1f} mm',font=small,fill='white')
            if held[i]:
                draw=ImageDraw.Draw(img)
                x=offsets[-1]
                draw.rectangle((x+7,434,x+202,468), fill='#26333e')
                draw.text((x+13,438),'关节目标沿用上一帧',font=small,fill='#ffdf98')
        else:
            img.paste(Image.fromarray(raw),(offsets[0],74))
            ImageDraw.Draw(img).text((440,180),'本帧未检测到右手',font=big,fill='white')
        ImageDraw.Draw(img).text((965,6),f'{frame/fps:05.2f} s    帧 {frame+1}/{total_frames}',font=small,fill='white')
        return img
    cap=cv2.VideoCapture(str(video))
    proc=None
    try:
        assert round(cap.get(cv2.CAP_PROP_FPS),4)==round(fps,4)
        if preview is None:
            proc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24',
              '-s',f'{basename.width}x{basename.height}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264',
              '-threads','3','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
        for frame in range(total_frames):
            if preview is not None and frame!=preview: continue
            if preview is not None: cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
            ok,bgr=cap.read()
            if not ok: raise ValueError(f'Missing RGB frame {frame}')
            composed=panel(frame,cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            if preview is not None:
                composed.save(output)
                break
            proc.stdin.write(composed.tobytes())
    finally:
        renderer.close()
        cap.release()
        if proc is not None:
            proc.stdin.close()
            if proc.wait(): raise RuntimeError('Video encoding failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('video','mano','robot','output','observations'):
        parser.add_argument(name,type=Path)
    parser.add_argument('--preview',type=int)
    layout=parser.add_mutually_exclusive_group()
    layout.add_argument('--show-pinky-gap',action='store_true')
    layout.add_argument('--compare-v2',action='store_true',help='Show v2 and v3 from the same frame with complete distal surface gaps.')
    layout.add_argument('--compare-v3',type=Path,metavar='V3_NPZ',help='Show v3 and v3.1 from the same frame with complete distal surface gaps.')
    layout.add_argument('--compare-v3.2',dest='compare_v32',type=Path,metavar='SWEEP_NPZ',help='Show the v3 reference and selected v3.2 sweep in the same view.')
    layout.add_argument('--compare-v3.3',dest='compare_v33',type=Path,nargs=2,metavar=('V32_NPZ','V33_NPZ'),help='Compare the recorded v3.2 and v3.3 poses on the same frames.')
    render(**vars(parser.parse_args()))
